from __future__ import annotations

import argparse
from pathlib import Path
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import json
import os
import re
import socket
import sys
import tempfile
import time
from typing import Any
import uuid
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen


JsonMap = dict[str, Any]
TypeKind = str


class LeaseUnavailable(RuntimeError):
    pass


class MissingLanceColumns(RuntimeError):
    def __init__(self, table_name: str, target_uri: str, columns: list[str]) -> None:
        self.table_name = table_name
        self.target_uri = target_uri
        self.columns = columns
        super().__init__(f"Lance table {table_name} at {target_uri} is missing incoming row columns: {columns}")


def log_event(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, "ts": int(time.time()), **fields}, sort_keys=True), flush=True)


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def lance_index_column_reference(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Unsafe Lance index column reference: {value!r}")
    return value


def quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def env_or_arg(args: argparse.Namespace, attr: str, env_name: str) -> str:
    value = getattr(args, attr) or os.environ.get(env_name)
    if not value:
        raise SystemExit(f"--{attr.replace('_', '-')} or {env_name} is required")
    return value


class ConvexClient:
    def __init__(self, convex_url: str, deploy_key: str, timeout_seconds: float = 60, max_retries: int = 3) -> None:
        self.convex_url = convex_url.rstrip("/") + "/"
        self.deploy_key = deploy_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def _request(self, path: str, params: dict[str, str] | None = None) -> JsonMap:
        query = f"?{urlencode(params)}" if params else ""
        url = urljoin(self.convex_url, path.lstrip("/")) + query
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                req = Request(url, headers={"Accept": "application/json", "Authorization": f"Convex {self.deploy_key}"})
                with urlopen(req, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError(f"Convex {path} returned non-object payload")
                return payload
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep((0.5 * (2**attempt)) + 0.1)
        raise RuntimeError(f"Convex request failed: {path}") from last_error

    def iter_document_deltas(self, cursor: str | None) -> Iterable[tuple[list[JsonMap], str | None, bool]]:
        current_cursor = cursor
        while True:
            params = {"cursor": current_cursor} if current_cursor is not None else None
            payload = self._request("/api/document_deltas", params)
            values = payload.get("values")
            rows = [v for v in values if isinstance(v, dict)] if isinstance(values, list) else []
            next_cursor_raw = payload.get("cursor", current_cursor)
            next_cursor = str(next_cursor_raw) if next_cursor_raw is not None else current_cursor
            has_more = bool(payload.get("hasMore"))
            if has_more and (not next_cursor or next_cursor == current_cursor):
                raise ValueError("document_deltas cursor did not advance while hasMore=true")
            yield rows, next_cursor, has_more
            current_cursor = next_cursor
            if not has_more:
                break

    def json_schemas(self, delta_schema: bool = True) -> JsonMap:
        return self._request("/api/json_schemas", {"delta_schema": "true" if delta_schema else "false"})


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    kind: TypeKind
    required: bool = False
    element_kind: TypeKind | None = None


@dataclass(frozen=True)
class IndexSpec:
    column: str
    index_type: str
    name: str | None = None


@dataclass(frozen=True)
class TableConfig:
    table_name: str
    version: str
    indexes: list[IndexSpec]


class S3JsonState:
    def __init__(self, bucket: str, prefix: str) -> None:
        import boto3

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client = boto3.client("s3")

    def key(self, name: str) -> str:
        name = name.strip("/")
        return f"{self.prefix}/{name}" if self.prefix else name

    def read(self, name: str) -> JsonMap | None:
        payload, _etag = self.read_with_etag(name)
        return payload

    def read_with_etag(self, name: str) -> tuple[JsonMap | None, str | None]:
        from botocore.exceptions import ClientError

        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=self.key(name))
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
                return None, None
            raise
        payload = json.loads(obj["Body"].read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"S3 state object {name} must contain a JSON object")
        return payload, obj.get("ETag")

    def write(self, name: str, value: JsonMap) -> str | None:
        response = self.client.put_object(
            Bucket=self.bucket,
            Key=self.key(name),
            Body=json.dumps(value, indent=2, sort_keys=True).encode("utf-8"),
            ContentType="application/json",
        )
        return response.get("ETag")

    def write_if_match(self, name: str, value: JsonMap, etag: str) -> str | None:
        from botocore.exceptions import ClientError

        try:
            response = self.client.put_object(
                Bucket=self.bucket,
                Key=self.key(name),
                Body=json.dumps(value, indent=2, sort_keys=True).encode("utf-8"),
                ContentType="application/json",
                IfMatch=etag,
            )
            return response.get("ETag")
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"PreconditionFailed", "ConditionalRequestConflict"}:
                raise RuntimeError(f"S3 state object changed before conditional write: {name}") from exc
            raise

    def delete(self, name: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=self.key(name))

    def try_create(self, name: str, value: JsonMap) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=self.key(name),
                Body=json.dumps(value, indent=2, sort_keys=True).encode("utf-8"),
                ContentType="application/json",
                IfNoneMatch="*",
            )
            return True
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"PreconditionFailed", "ConditionalRequestConflict"}:
                return False
            raise


@dataclass(frozen=True)
class TableLease:
    owner: str
    table_name: str


def _owner_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{int(time.time())}"


def acquire_lease(state: S3JsonState, name: str, ttl_seconds: int, force: bool) -> TableLease:
    now = int(time.time())
    owner = _owner_id()
    key = f"locks/{name}.json"
    payload = {"table": name, "owner": owner, "created_at": now, "updated_at": now, "expires_at": now + ttl_seconds}
    if state.try_create(key, payload):
        return TableLease(owner=owner, table_name=name)
    existing = state.read(key)
    expired = existing is not None and int(existing.get("expires_at", 0)) < now
    if not force and not expired:
        raise LeaseUnavailable(f"Incremental lock already exists for {name}: {existing}")
    state.write(key, payload)
    return TableLease(owner=owner, table_name=name)


def heartbeat_lease(state: S3JsonState, lease: TableLease, ttl_seconds: int) -> None:
    now = int(time.time())
    verify_lease_owner(state, lease, now)
    state.write("locks/%s.json" % lease.table_name, {"table": lease.table_name, "owner": lease.owner, "updated_at": now, "expires_at": now + ttl_seconds})


def release_lease(state: S3JsonState, lease: TableLease) -> None:
    existing = state.read("locks/%s.json" % lease.table_name)
    if existing and existing.get("owner") == lease.owner:
        state.delete("locks/%s.json" % lease.table_name)


def verify_lease_owner(state: S3JsonState, lease: TableLease, now: int | None = None) -> None:
    now = int(time.time()) if now is None else now
    existing = state.read("locks/%s.json" % lease.table_name)
    if not existing:
        raise RuntimeError(f"Incremental lock disappeared for {lease.table_name}")
    if existing.get("owner") != lease.owner:
        raise RuntimeError(f"Incremental lock owner changed for {lease.table_name}: {existing}")
    if int(existing.get("expires_at", 0)) < now:
        raise RuntimeError(f"Incremental lock expired for {lease.table_name}: {existing}")


def source_to_physical_table(table_name: str) -> str:
    out = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in table_name).strip("_")
    if not out:
        out = "table"
    if out[0].isdigit():
        out = f"t_{out}"
    return out.lower()


def normalize_prefix(prefix: str) -> str:
    return prefix.strip("/")


def lance_uri(bucket: str, prefix: str, table_name: str) -> str:
    clean_prefix = normalize_prefix(prefix)
    suffix = f"{source_to_physical_table(table_name)}.lance"
    return f"s3://{bucket}/{clean_prefix}/{suffix}" if clean_prefix else f"s3://{bucket}/{suffix}"


def create_lance_secret(conn: Any, scope: str, region: str | None) -> None:
    parts = ["TYPE lance", "PROVIDER credential_chain", f"SCOPE {quote_literal(scope)}"]
    if region:
        parts.append(f"REGION {quote_literal(region)}")
    conn.execute(f"CREATE OR REPLACE SECRET lance_object_store ({', '.join(parts)})")


def create_s3_secret(conn: Any, region: str | None) -> None:
    parts = ["TYPE s3", "PROVIDER credential_chain"]
    if region:
        parts.append(f"REGION {quote_literal(region)}")
    conn.execute(f"CREATE OR REPLACE SECRET s3tables_object_store ({', '.join(parts)})")


def install_and_load(conn: Any, extension: str) -> None:
    conn.execute(f"INSTALL {extension}")
    conn.execute(f"LOAD {extension}")


def open_duckdb(args: argparse.Namespace) -> Any:
    import duckdb

    conn = duckdb.connect(args.duckdb_path or ":memory:")
    if args.threads:
        conn.execute(f"SET threads = {int(args.threads)}")
    if args.memory_limit:
        conn.execute(f"SET memory_limit = {quote_literal(args.memory_limit)}")
    if args.temp_directory:
        conn.execute(f"SET temp_directory = {quote_literal(args.temp_directory)}")
    for extension in ("aws", "httpfs", "iceberg", "lance"):
        install_and_load(conn, extension)
    create_s3_secret(conn, args.aws_region)
    create_lance_secret(conn, args.lance_scope, args.aws_region)
    conn.execute(
        f"ATTACH {quote_literal(args.s3tables_warehouse_arn)} AS {quote_identifier(args.catalog_alias)} "
        "(TYPE iceberg, ENDPOINT_TYPE s3_tables)"
    )
    return conn


def open_lance_duckdb(args: argparse.Namespace) -> Any:
    import duckdb

    conn = duckdb.connect(args.duckdb_path or ":memory:")
    if args.threads:
        conn.execute(f"SET threads = {int(args.threads)}")
    if args.memory_limit:
        conn.execute(f"SET memory_limit = {quote_literal(args.memory_limit)}")
    if args.temp_directory:
        conn.execute(f"SET temp_directory = {quote_literal(args.temp_directory)}")
    for extension in ("httpfs", "lance"):
        install_and_load(conn, extension)
    create_lance_secret(conn, args.lance_scope, args.aws_region)
    return conn


GENERATED_COLUMNS = {"__status", "__status_id", "__id_ts"}
DROPPABLE_DELTA_METADATA_COLUMNS = {"_component"}


def status_expr() -> str:
    return (
        "CAST("
        "(CASE WHEN COALESCE(_current, FALSE) THEN 1 ELSE 0 END) + "
        "(CASE WHEN COALESCE(_deleted, FALSE) THEN 2 ELSE 0 END) "
        "AS TINYINT)"
    )


def status_value(current: bool, deleted: bool) -> int:
    return (1 if current else 0) + (2 if deleted else 0)


def id_ts_value(row: JsonMap) -> str:
    return f"{row['_id']}#{row['_ts']}"


def status_id_value(status: int, row_id: Any) -> str:
    return f"{status}#{row_id}"


def status_id_expr() -> str:
    return f"CAST(({status_expr()}) AS VARCHAR) || '#' || CAST(_id AS VARCHAR)"


def id_ts_expr() -> str:
    return "CAST(_id AS VARCHAR) || '#' || CAST(_ts AS VARCHAR)"


def required_generated_source_columns(columns: set[str]) -> set[str]:
    return {"_id", "_ts", "_current", "_deleted"} - columns


def _as_object(value: Any) -> JsonMap | None:
    return value if isinstance(value, dict) else None


def _is_null_schema(schema: JsonMap) -> bool:
    t = schema.get("type")
    return t == "null" or (isinstance(t, list) and t and all(v == "null" for v in t))


def expand_validator_json(schema: JsonMap) -> JsonMap:
    t = schema.get("type")
    if t == "union" and isinstance(schema.get("value"), list):
        return {"anyOf": [expand_validator_json(v) for v in schema["value"] if isinstance(v, dict)]}
    if t == "array":
        items = _as_object(schema.get("items")) or _as_object(schema.get("value"))
        return {"type": "array", "items": expand_validator_json(items)} if items else {**schema}
    if t == "object":
        props = _as_object(schema.get("properties"))
        if props:
            return {**schema, "type": "object", "properties": {k: expand_validator_json(v) for k, v in sorted(props.items()) if isinstance(v, dict)}}
        value = _as_object(schema.get("value"))
        if value:
            properties: JsonMap = {}
            required: list[str] = []
            for key, wrapper in sorted(value.items()):
                if not isinstance(wrapper, dict):
                    continue
                field_type = _as_object(wrapper.get("fieldType"))
                if not field_type:
                    continue
                inner = expand_validator_json(field_type)
                if wrapper.get("optional") is True:
                    inner = {"anyOf": [{"type": "null"}, inner]}
                else:
                    required.append(key)
                properties[key] = inner
            return {"type": "object", "properties": properties, "required": required}
    if t == "id":
        return {"type": "string", "x-convex": "id"}
    if t == "literal":
        return {"const": schema.get("value")}
    if t == "bigint":
        return {"type": "integer", "x-convex": "int64"}
    if t == "number":
        return {"type": "number"}
    if t == "bytes":
        return {"type": "string", "x-convex": "bytes"}
    if t == "boolean":
        return {"type": "boolean"}
    if t == "string":
        return {"type": "string"}
    if t == "null":
        return {"type": "null"}
    if t == "any":
        return {"x-convex": "any"}
    if t == "record":
        return {"x-convex": "record"}
    return schema


def _collect_union_members(schema: JsonMap) -> list[JsonMap]:
    expanded = expand_validator_json(schema)
    branches = expanded.get("anyOf") or expanded.get("oneOf")
    if not isinstance(branches, list):
        return [expanded]
    out: list[JsonMap] = []
    for branch in branches:
        if isinstance(branch, dict):
            out.extend(_collect_union_members(branch))
    return out


def _peel_nullable(schema: JsonMap) -> tuple[bool, JsonMap]:
    expanded = expand_validator_json(schema)
    t = expanded.get("type")
    if isinstance(t, list):
        non_null = [v for v in t if v != "null"]
        if len(non_null) == 1:
            return len(non_null) != len(t), expand_validator_json({**expanded, "type": non_null[0]})
    members = _collect_union_members(expanded)
    if len(members) > 1:
        nullable = any(_is_null_schema(m) for m in members)
        rest = [m for m in members if not _is_null_schema(m)]
        if len(rest) == 1:
            return nullable, expand_validator_json(rest[0])
        return nullable, {"anyOf": rest}
    return False, expanded


def infer_kind(schema: JsonMap) -> tuple[TypeKind, bool, TypeKind | None]:
    nullable, core = _peel_nullable(schema)
    members = [m for m in _collect_union_members(core) if not _is_null_schema(m)]
    if len(members) > 1:
        inferred = [infer_kind(m)[0] for m in members]
        if "json" in inferred or "array" in inferred:
            return "json", nullable, None
        if "string" in inferred:
            return "string", nullable, None
        if all(k in {"bool", "int64", "float64"} for k in inferred):
            if "float64" in inferred:
                return "float64", nullable, None
            if "int64" in inferred:
                return "int64", nullable, None
            return "bool", nullable, None
        return "string", nullable, None
    c = expand_validator_json(members[0] if members else core)
    if c.get("x-convex") in {"any", "record"}:
        return "json", nullable, None
    if "const" in c:
        value = c["const"]
        if isinstance(value, bool):
            return "bool", nullable, None
        if isinstance(value, int) and not isinstance(value, bool):
            return "int64", nullable, None
        if isinstance(value, float):
            return "float64", nullable, None
        return "string", nullable, None
    t = c.get("type")
    if t is None:
        return "json", nullable, None
    if t == "string":
        return "string", nullable, None
    if t == "boolean":
        return "bool", nullable, None
    if t == "integer":
        return "int64", nullable, None
    if t == "number":
        return "float64", nullable, None
    if t == "null":
        # Convex delta schemas can contain fields whose current observed type is
        # only null. The incremental stream still includes those keys with null
        # values, so omitting the column from Lance makes later merges fail with
        # MissingLanceColumns before the field ever has a concrete type. Create
        # a nullable string placeholder instead; schema reconciliation can widen
        # compatible string-like placeholders later, and incompatible concrete
        # changes still route through the controlled drift path.
        return "string", True, None
    if t == "array":
        item = _as_object(c.get("items"))
        if not item:
            return "json", nullable, None
        item_kind, item_nullable, _ = infer_kind(item)
        if item_nullable or item_kind in {"json", "array"}:
            return "json", nullable, None
        return "array", nullable, item_kind
    if t == "object" or c.get("x-convex") in {"any", "record"}:
        return "json", nullable, None
    return "string", nullable, None


def schema_column_specs(table_schema: JsonMap) -> list[ColumnSpec]:
    columns = [
        ColumnSpec("_id", "string", True),
        ColumnSpec("_creationTime", "float64"),
        ColumnSpec("_table", "string"),
        ColumnSpec("_ts", "int64", True),
        ColumnSpec("_deleted", "bool", True),
        ColumnSpec("_convex_cursor", "string"),
        ColumnSpec("__status", "int8", True),
        ColumnSpec("__status_id", "string", True),
        ColumnSpec("__id_ts", "string", True),
    ]
    expanded = expand_validator_json(table_schema)
    props = _as_object(expanded.get("properties"))
    if not props:
        return columns
    reserved = {c.name for c in columns} | {"_current"}
    required = set(expanded.get("required") if isinstance(expanded.get("required"), list) else [])
    for key in sorted(props):
        if key in reserved or not isinstance(props[key], dict):
            continue
        kind, nullable, element_kind = infer_kind(props[key])
        # Convex object schemas can mark fields as required while individual
        # deltas still carry null values for those fields. Lance enforces
        # non-nullability on append, so schema-derived application fields must
        # stay nullable. Keep required constraints only on our generated
        # metadata columns above.
        columns.append(ColumnSpec(key, kind, False, element_kind))
    return columns


def _ts_value(row: JsonMap) -> int | float | str:
    value = row.get("_ts")
    if value is None:
        raise ValueError(f"row missing _ts: {row}")
    return value  # Convex _ts values are comparable within a table.


def prepare_incremental_merge_rows(incoming_rows: list[JsonMap], existing_current_rows: list[JsonMap]) -> list[JsonMap]:
    latest_incoming_by_id: dict[str, JsonMap] = {}
    for row in incoming_rows:
        row_id = row.get("_id")
        if row_id is None:
            raise ValueError(f"row missing _id: {row}")
        key = str(row_id)
        existing = latest_incoming_by_id.get(key)
        if existing is None or _ts_value(row) > _ts_value(existing):
            latest_incoming_by_id[key] = row

    existing_by_id = {str(row["_id"]): row for row in existing_current_rows if row.get("_id") is not None}
    merge_rows: list[JsonMap] = []
    demoted_versions: set[str] = set()
    for raw in incoming_rows:
        row = dict(raw)
        row_id = str(row["_id"])
        is_latest_in_page = id_ts_value(row) == id_ts_value(latest_incoming_by_id[row_id])
        existing_current = existing_by_id.get(row_id)
        becomes_current = is_latest_in_page and (existing_current is None or _ts_value(row) >= _ts_value(existing_current))
        deleted = bool(row.get("_deleted", False))
        status = status_value(becomes_current, deleted)
        row.pop("_current", None)
        row["__status"] = status
        row["__status_id"] = status_id_value(status, row["_id"])
        row["__id_ts"] = id_ts_value(row)
        merge_rows.append(row)

        if becomes_current and existing_current is not None and id_ts_value(existing_current) != row["__id_ts"]:
            existing_version = id_ts_value(existing_current)
            if existing_version not in demoted_versions:
                demoted = {
                    "_id": existing_current["_id"],
                    "_ts": existing_current["_ts"],
                    "__id_ts": existing_version,
                }
                demoted_status = status_value(False, bool(existing_current.get("_deleted", False)))
                demoted["__status"] = demoted_status
                demoted["__status_id"] = status_id_value(demoted_status, existing_current["_id"])
                merge_rows.append(demoted)
                demoted_versions.add(existing_version)
    return merge_rows


def build_select_sql(conn: Any, args: argparse.Namespace, source_table: str) -> str:
    source_ref = ".".join(
        [
            quote_identifier(args.catalog_alias),
            quote_identifier(args.namespace),
            quote_identifier(source_table),
        ]
    )
    column_names = table_column_names(conn, source_ref)
    column_set = set(column_names)
    missing_required = required_generated_source_columns(column_set)
    if missing_required:
        raise RuntimeError(f"source table {source_table} missing required generated-column inputs: {sorted(missing_required)}")
    base_columns = [column for column in column_names if column not in GENERATED_COLUMNS and column != "_current"]
    select_list = ", ".join(quote_identifier(column) for column in base_columns)
    sql = (
        f"SELECT {select_list}, "
        f"{status_expr()} AS __status, "
        f"{status_id_expr()} AS __status_id, "
        f"{id_ts_expr()} AS __id_ts "
        f"FROM {source_ref}"
    )
    if args.where:
        sql += f" WHERE {args.where}"
    if args.order_by:
        sql += f" ORDER BY {args.order_by}"
    if args.max_rows is not None:
        sql += f" LIMIT {int(args.max_rows)}"
    return sql


def run_migrate_table(args: argparse.Namespace) -> None:
    table_name = env_or_arg(args, "table", "TABLE_NAME")
    source_table = args.source_table or os.environ.get("ICEBERG_TABLE_NAME") or table_name
    output_table = args.output_table or os.environ.get("LANCE_TABLE_NAME") or source_to_physical_table(table_name)
    bucket = env_or_arg(args, "lance_bucket", "LANCE_BUCKET")
    target_uri = args.lance_uri or lance_uri(bucket, args.lance_prefix, output_table)
    args.lance_scope = args.lance_scope or f"s3://{bucket}/"
    args.s3tables_warehouse_arn = env_or_arg(args, "s3tables_warehouse_arn", "S3TABLES_CATALOG_WAREHOUSE_ARN")
    args.namespace = args.namespace or os.environ.get("S3TABLES_NAMESPACE", "convex")
    args.catalog_alias = args.catalog_alias or os.environ.get("ICEBERG_CATALOG_ALIAS", "s3tables")
    args.aws_region = args.aws_region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")

    log_event(
        "lance_migration_started",
        source_table=source_table,
        namespace=args.namespace,
        target_uri=target_uri,
        mode=args.mode,
    )
    started = time.monotonic()
    conn = open_duckdb(args)
    try:
        select_sql = build_select_sql(conn, args, source_table)
        conn.execute(f"COPY ({select_sql}) TO {quote_literal(target_uri)} (FORMAT lance, MODE {quote_literal(args.mode)})")
        source_count = conn.execute(f"SELECT count(*) FROM ({select_sql})").fetchone()[0] if args.verify_count else None
        lance_count = conn.execute(f"SELECT count(*) FROM {quote_literal(target_uri)}").fetchone()[0] if args.verify_count else None
        if args.verify_count and source_count != lance_count:
            raise RuntimeError(f"count mismatch source={source_count} lance={lance_count}")
        generated_indexes = 0
        index_skipped: list[JsonMap] = []
        config: TableConfig | None = None
        if args.create_generated_indexes or args.table_config_indexes:
            requested, config = requested_index_specs(argparse.Namespace(**vars(args), generated_indexes=args.create_generated_indexes), output_table)
            generated_indexes, index_skipped = create_indexes_for_columns(conn, output_table, quote_literal(target_uri), requested)
            write_applied_table_config_state(table_config_state(args), output_table, config, generated_indexes, index_skipped)
        log_event(
            "lance_migration_completed",
            source_table=source_table,
            target_uri=target_uri,
            source_count=source_count,
            lance_count=lance_count,
            generated_indexes=generated_indexes,
            index_skipped=index_skipped,
            table_config_version=config.version if config else None,
            elapsed_seconds=round(time.monotonic() - started, 3),
        )
    finally:
        conn.close()


def _convex_client(args: argparse.Namespace) -> ConvexClient:
    convex_url = args.convex_url or os.environ.get("CONVEX_URL")
    deploy_key = args.convex_deploy_key or os.environ.get("CONVEX_DEPLOY_KEY")
    if not convex_url or not deploy_key:
        raise SystemExit("CONVEX_URL and CONVEX_DEPLOY_KEY are required")
    return ConvexClient(convex_url, deploy_key)


def _incremental_state(args: argparse.Namespace) -> S3JsonState:
    bucket = args.state_bucket or os.environ.get("LANCE_INCREMENTAL_STATE_BUCKET") or os.environ.get("S3TABLES_BACKFILL_STATE_BUCKET")
    prefix = args.state_prefix or os.environ.get("LANCE_INCREMENTAL_STATE_PREFIX") or "lance-incremental"
    if not bucket:
        raise SystemExit("--state-bucket, LANCE_INCREMENTAL_STATE_BUCKET, or S3TABLES_BACKFILL_STATE_BUCKET is required")
    return S3JsonState(bucket, prefix)


def _read_incremental_cursor(state: S3JsonState, args: argparse.Namespace) -> tuple[str, str]:
    payload, etag = state.read_with_etag(args.cursor_key)
    if not payload or payload.get("cursor") is None:
        raise SystemExit(f"No Lance incremental cursor found at {args.cursor_key}")
    if etag is None:
        raise SystemExit(f"No ETag returned for Lance incremental cursor at {args.cursor_key}")
    return str(payload["cursor"]), etag


def _write_incremental_cursor(state: S3JsonState, args: argparse.Namespace, cursor: str, etag: str, extra: JsonMap | None = None) -> str:
    new_etag = state.write_if_match(args.cursor_key, {"cursor": cursor, "updated_at": int(time.time()), **(extra or {})}, etag)
    if new_etag is None:
        raise RuntimeError(f"No ETag returned after writing Lance incremental cursor at {args.cursor_key}")
    return new_etag


def _audit_cursor_page(state: S3JsonState, lease: TableLease, page: int, start_cursor: str, end_cursor: str | None, table_rows: dict[str, int], rows_seen: int, rows_accepted: int, rows_merged: int) -> None:
    now = int(time.time())
    safe_start = "".join(ch if ch.isalnum() else "_" for ch in start_cursor)[:80]
    safe_end = "".join(ch if ch.isalnum() else "_" for ch in str(end_cursor))[:80]
    state.write(
        f"incremental/page_audit/{now}-{page}-{safe_start}-{safe_end}.json",
        {
            "owner": lease.owner,
            "page": page,
            "start_cursor": start_cursor,
            "end_cursor": end_cursor,
            "rows_seen": rows_seen,
            "rows_accepted": rows_accepted,
            "rows_merged": rows_merged,
            "tables": table_rows,
            "updated_at": now,
        },
    )


def _table_column_types(conn: Any, table_ref: str) -> dict[str, str]:
    return {str(row[0]): str(row[1]).upper() for row in conn.execute(f"DESCRIBE {table_ref}").fetchall()}


def _coerce_merge_rows_for_schema(rows: list[JsonMap], column_types: dict[str, str]) -> list[JsonMap]:
    coerced_rows: list[JsonMap] = []
    for raw in rows:
        row = dict(raw)
        for column, column_type in column_types.items():
            if column_type == "JSON" and column in row and row[column] is not None:
                value = row[column]
                if not isinstance(value, str):
                    row[column] = json.dumps(value, separators=(",", ":"))
                else:
                    try:
                        json.loads(value)
                    except json.JSONDecodeError:
                        row[column] = json.dumps(value)
        coerced_rows.append(row)
    return coerced_rows


def normalize_value_for_column(value: Any, column: ColumnSpec) -> Any:
    if value is None:
        return None
    if column.kind == "json":
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list)):
            return json.dumps(value, separators=(",", ":"))
        return json.dumps(value, separators=(",", ":"))
    if column.kind == "string":
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list)):
            return json.dumps(value, separators=(",", ":"))
        return str(value)
    if column.kind in {"int64", "int8"}:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return None
    if column.kind == "float64":
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return None
    if column.kind == "bool":
        return value if isinstance(value, bool) else None
    return value


def normalize_rows_for_specs(rows: list[JsonMap], columns: list[ColumnSpec]) -> list[JsonMap]:
    specs = {column.name: column for column in columns}
    normalized: list[JsonMap] = []
    for raw in rows:
        row = dict(raw)
        for name, spec in specs.items():
            if name in row:
                row[name] = normalize_value_for_column(row[name], spec)
        normalized.append(row)
    return normalized


def _coerce_rows_for_arrow_schema(rows: list[JsonMap], schema: Any) -> list[JsonMap]:
    import pyarrow.types as pat

    coerced_rows: list[JsonMap] = []
    for raw in rows:
        row = dict(raw)
        for field in schema:
            if field.name not in row or row[field.name] is None:
                continue
            value = row[field.name]
            if pat.is_string(field.type) or pat.is_large_string(field.type):
                row[field.name] = value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))
        coerced_rows.append(row)
    return coerced_rows


def missing_lance_columns_for_rows(rows: list[JsonMap], schema: Any) -> list[str]:
    field_names = {field.name for field in schema}
    return sorted({column for row in rows for column in row if column not in field_names and column not in DROPPABLE_DELTA_METADATA_COLUMNS})


def ensure_rows_fit_lance_schema(table_name: str, target_uri: str, rows: list[JsonMap], schema: Any) -> None:
    missing = missing_lance_columns_for_rows(rows, schema)
    if not missing:
        return
    log_event("lance_incremental_merge_missing_lance_columns", table=table_name, target_uri=target_uri, columns=missing)
    raise MissingLanceColumns(table_name, target_uri, missing)


def _pa_type_for_column(column: ColumnSpec) -> Any:
    import pyarrow as pa

    if column.kind in {"string", "json"}:
        return pa.string()
    if column.kind == "bool":
        return pa.bool_()
    if column.kind == "int64":
        return pa.int64()
    if column.kind == "int8":
        return pa.int8()
    if column.kind == "float64":
        return pa.float64()
    if column.kind == "array":
        return pa.list_(_pa_type_for_column(ColumnSpec("item", column.element_kind or "string")))
    raise ValueError(f"Unsupported schema column kind: {column.kind}")


def _arrow_schema_for_specs(columns: list[ColumnSpec]) -> Any:
    import pyarrow as pa

    return pa.schema([pa.field(column.name, _pa_type_for_column(column), nullable=not column.required) for column in columns])


def _schema_types_compatible(existing_type: Any, desired_type: Any) -> bool:
    import pyarrow.types as pat

    if existing_type.equals(desired_type):
        return True
    if pat.is_integer(existing_type) and pat.is_integer(desired_type):
        return True
    if pat.is_floating(existing_type) and pat.is_floating(desired_type):
        return True
    if (pat.is_string(existing_type) or pat.is_large_string(existing_type)) and (pat.is_string(desired_type) or pat.is_large_string(desired_type)):
        return True
    return False


def _schema_type_alteration(existing_type: Any, desired_type: Any) -> JsonMap | None:
    import pyarrow.types as pat

    if pat.is_integer(existing_type) and pat.is_integer(desired_type) and desired_type.bit_width > existing_type.bit_width:
        return {"data_type": desired_type}
    if pat.is_floating(existing_type) and pat.is_floating(desired_type) and desired_type.bit_width > existing_type.bit_width:
        return {"data_type": desired_type}
    if _schema_types_compatible(existing_type, desired_type):
        return None
    return None


def _lance_sql_type(column: ColumnSpec) -> str:
    if column.kind in {"string", "json"}:
        return "VARCHAR"
    if column.kind == "bool":
        return "BOOLEAN"
    if column.kind == "int64":
        return "BIGINT"
    if column.kind == "int8":
        return "TINYINT"
    if column.kind == "float64":
        return "DOUBLE"
    if column.kind == "array":
        item = _lance_sql_type(ColumnSpec("item", column.element_kind or "string"))
        return f"{item}[]"
    raise ValueError(f"Unsupported schema column kind: {column.kind}")


def _schema_cache_stale(payload: JsonMap | None, ttl_seconds: int) -> bool:
    if payload is None:
        return True
    return time.time() - int(payload.get("fetched_at", 0)) >= ttl_seconds


def incremental_schema_payload_with_status(state: S3JsonState, args: argparse.Namespace, client: ConvexClient, force_refresh: bool = False) -> tuple[JsonMap, bool]:
    payload = None if force_refresh else state.read(args.schema_key)
    if force_refresh or _schema_cache_stale(payload, args.schema_refresh_seconds):
        schemas = client.json_schemas(delta_schema=True)
        payload = {"fetched_at": int(time.time()), "schemas": schemas}
        state.write(args.schema_key, payload)
        log_event("lance_incremental_schema_refreshed", tables=len([k for k, v in schemas.items() if isinstance(v, dict)]))
        return payload, True
    return payload or {"schemas": {}}, False


def incremental_schema_payload(state: S3JsonState, args: argparse.Namespace, client: ConvexClient) -> JsonMap:
    payload, _refreshed = incremental_schema_payload_with_status(state, args, client)
    return payload


def table_schema_from_payload(schema_payload: JsonMap, source_table: str) -> JsonMap | None:
    schemas = schema_payload.get("schemas")
    if not isinstance(schemas, dict):
        return None
    table_schema = schemas.get(source_table)
    return table_schema if isinstance(table_schema, dict) else None


def reconcile_lance_append_only_columns(args: argparse.Namespace, table_name: str, target_uri: str, table_schema: JsonMap) -> list[str]:
    import lance

    dataset = lance.dataset(target_uri)
    existing = _dataset_column_types(dataset)
    added = [spec for spec in schema_column_specs(table_schema) if spec.name not in existing]
    if not added:
        return []
    columns = [spec.name for spec in added]
    log_event("lance_incremental_schema_columns_add_started", table=table_name, target_uri=target_uri, columns=columns)
    try:
        if not _add_lance_columns_native(dataset, added):
            log_event("lance_incremental_schema_add_columns_duckdb_fallback", table=table_name, target_uri=target_uri, columns=columns)
            _add_lance_columns_duckdb(args, target_uri, added)
    except Exception as exc:  # noqa: BLE001
        log_event("lance_incremental_schema_add_columns_failed", table=table_name, target_uri=target_uri, columns=columns, error=repr(exc))
        raise
    log_event("lance_incremental_schema_columns_added", table=table_name, target_uri=target_uri, columns=columns)
    return columns


def reconcile_schema_payload_existing_lance_tables(args: argparse.Namespace, bucket: str, schema_payload: JsonMap) -> JsonMap:
    schema_by_physical = physical_to_source_schema_map(schema_payload)
    reconcile_args = argparse.Namespace(**vars(args))
    reconcile_args.unknown_table_policy = "skip"
    try:
        existing_tables = list_lance_tables(
            argparse.Namespace(
                tables=None,
                tables_file=None,
                lance_bucket=bucket,
                lance_prefix=args.lance_prefix,
                aws_region=args.aws_region,
            )
        )
    except Exception as exc:  # noqa: BLE001
        failure = {"error": repr(exc)}
        log_event("lance_incremental_schema_refresh_reconcile_list_failed", **failure)
        return {"checked": 0, "reconciled": 0, "skipped": 0, "failed": 1, "failures": [failure]}
    checked = reconciled = skipped = columns_added = failed = 0
    failures: list[JsonMap] = []
    log_event("lance_incremental_schema_refresh_reconcile_started", tables=len(existing_tables))
    for physical in existing_tables:
        schema_entry = schema_by_physical.get(physical)
        if schema_entry is None:
            skipped += 1
            continue
        source_table, table_schema = schema_entry
        target_uri = lance_uri(bucket, args.lance_prefix, physical)
        checked += 1
        try:
            columns_added += len(reconcile_lance_append_only_columns(reconcile_args, physical, target_uri, table_schema))
            reconciled += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            failure = {"table": source_table, "physical_table": physical, "target_uri": target_uri, "error": repr(exc)}
            failures.append(failure)
            log_event("lance_incremental_schema_refresh_reconcile_failed", **failure)
    result = {"checked": checked, "reconciled": reconciled, "skipped": skipped, "columns_added": columns_added, "failed": failed}
    log_event("lance_incremental_schema_refresh_reconcile_completed", **result)
    return {**result, "failures": failures}


def _dataset_column_types(dataset: Any) -> dict[str, Any]:
    return {field.name: field.type for field in dataset.schema}


def _add_lance_columns_native(dataset: Any, specs: list[ColumnSpec]) -> bool:
    if not specs:
        return True
    add_columns = getattr(dataset, "add_columns", None)
    if add_columns is None:
        return False
    import pyarrow as pa

    fields = [pa.field(spec.name, _pa_type_for_column(spec), nullable=True) for spec in specs]
    try:
        add_columns(pa.schema(fields))
    except TypeError:
        add_columns(fields)
    return True


def _add_lance_columns_duckdb(args: argparse.Namespace, target_uri: str, specs: list[ColumnSpec]) -> None:
    conn = open_lance_duckdb(args)
    try:
        for spec in specs:
            conn.execute(f"ALTER TABLE {quote_literal(target_uri)} ADD COLUMN {quote_identifier(spec.name)} {_lance_sql_type(spec)}")
    finally:
        conn.close()


def create_indexes_for_dataset(dataset: Any, table_name: str, requested_columns: Iterable[IndexSpec | tuple[str, str] | tuple[str, str, str]]) -> tuple[int, list[JsonMap]]:
    columns = set(_dataset_column_types(dataset))
    skipped: list[JsonMap] = []
    created = 0
    for spec in _index_specs(requested_columns):
        column = spec.column
        index_type = _validate_index_type(spec.index_type)
        if column not in columns:
            skipped.append({"table": table_name, "column": column, "reason": "missing_column"})
            continue
        try:
            dataset.create_scalar_index(column, index_type, name=spec.name or index_name(table_name, column), replace=False)
            created += 1
        except TypeError:
            try:
                dataset.create_scalar_index(column, index_type=index_type, name=spec.name or index_name(table_name, column), replace=False)
                created += 1
            except TypeError:
                dataset.create_scalar_index(column, index_type=index_type, name=spec.name or index_name(table_name, column))
                created += 1
        except Exception as exc:  # noqa: BLE001
            if "already" in str(exc).lower():
                skipped.append({"table": table_name, "column": column, "error": repr(exc), "reason": "already_exists"})
                continue
            raise
    return created, skipped


def _column_has_data(dataset: Any, column_name: str) -> bool:
    """Check if a column has any non-null values in the dataset."""
    try:
        table = dataset.to_table(columns=[column_name])
        if table.num_rows == 0:
            return False
        # Check for non-null values
        column = table.column(column_name)
        for i in range(column.num_rows):
            if not column.is_null(i):
                return True
        return False
    except Exception:  # noqa: BLE001
        # If we can't check, assume it has data to be safe
        return True


def _table_is_empty(dataset: Any) -> bool:
    """Check if the entire dataset table is empty."""
    try:
        return dataset.to_table().num_rows == 0
    except Exception:  # noqa: BLE001
        return False


def _recreate_dataset_with_new_schema(args: argparse.Namespace, table_name: str, target_uri: str, specs: list[ColumnSpec]) -> None:
    """Recreate a dataset with a new schema when the existing one is empty or has empty columns."""
    import lance
    import pyarrow as pa
    import boto3

    # Delete the existing dataset
    try:
        # Lance datasets are directories, so we need to remove them
        # Parse S3 URI
        if target_uri.startswith('s3://'):
            uri_parts = target_uri[5:].split('/', 1)
            bucket = uri_parts[0]
            prefix = uri_parts[1] if len(uri_parts) > 1 else ''
            # List and delete all objects in the dataset directory
            s3 = boto3.client('s3')
            paginator = s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                if 'Contents' in page:
                    objects = [{'Key': obj['Key']} for obj in page['Contents']]
                    if objects:
                        s3.delete_objects(Bucket=bucket, Delete={'Objects': objects})
    except Exception as exc:  # noqa: BLE001
        log_event("lance_incremental_schema_recreate_delete_failed", table=table_name, target_uri=target_uri, error=repr(exc))

    # Create new dataset with correct schema
    empty = pa.Table.from_pylist([], schema=_arrow_schema_for_specs(specs))
    lance.write_dataset(empty, target_uri, mode='create')
    log_event("lance_incremental_schema_recreated", table=table_name, target_uri=target_uri, columns=len(specs))


def reconcile_lance_schema(args: argparse.Namespace, table_name: str, target_uri: str, table_schema: JsonMap | None) -> list[ColumnSpec] | None:
    if table_schema is None:
        if args.unknown_table_policy == "fail":
            raise RuntimeError(f"Schema missing for incremental table {table_name}; refusing to merge without Convex schema")
        return
    import lance

    specs = schema_column_specs(table_schema)
    try:
        dataset = lance.dataset(target_uri)
    except Exception:  # noqa: BLE001
        if args.unknown_table_policy == "fail":
            raise RuntimeError(f"Lance table missing for {table_name} at {target_uri}; run full backfill before allowing incremental deltas")
        if args.unknown_table_policy == "skip":
            log_event("lance_incremental_table_skipped_missing_dataset", table=table_name, target_uri=target_uri)
            return []
        import pyarrow as pa

        empty = pa.Table.from_pylist([], schema=_arrow_schema_for_specs(specs))
        lance.write_dataset(empty, target_uri, mode="create")
        dataset = lance.dataset(target_uri)
        # Do not create generated scalar indexes before the first incremental
        # upsert into a newly discovered table. Lance 4.0.x can fail
        # merge_insert on S3-backed tables that have just been created with
        # empty scalar indexes ("Spill has sent an error"). Let the row merge
        # establish the table contents first; normal table-config/index
        # reconciliation can add configured indexes later.
        requested, config = requested_index_specs(argparse.Namespace(**vars(args), generated_indexes=False), table_name)
        generated_indexes, index_skipped = create_indexes_for_dataset(dataset, table_name, requested)
        write_applied_table_config_state(table_config_state(args), table_name, config, generated_indexes, index_skipped)
        log_event(
            "lance_incremental_table_created_from_schema",
            table=table_name,
            target_uri=target_uri,
            columns=len(specs),
            generated_indexes=generated_indexes,
            index_skipped=index_skipped,
            table_config_version=config.version if config else None,
        )
        return specs
    existing = _dataset_column_types(dataset)
    added: list[ColumnSpec] = []
    alterations: list[JsonMap] = []
    drift: list[JsonMap] = []
    for spec in specs:
        desired_type = _pa_type_for_column(spec)
        existing_type = existing.get(spec.name)
        if existing_type is not None:
            alteration = _schema_type_alteration(existing_type, desired_type)
            if alteration is None:
                if not _schema_types_compatible(existing_type, desired_type):
                    drift.append({"column": spec.name, "existing": str(existing_type), "desired": str(desired_type), "reason": "duckdb_rewrite_required"})
                continue
            alterations.append({"path": spec.name, **alteration})
            continue
        added.append(spec)
    if drift:
        # Check if we can safely handle the drift by checking if affected columns are empty
        if args.auto_recreate_empty_schema_drift:
            table_empty = _table_is_empty(dataset)
            columns_with_data = [d["column"] for d in drift if _column_has_data(dataset, d["column"])]
            
            if table_empty:
                # Entire table is empty, safe to recreate with new schema
                log_event("lance_incremental_schema_drift_table_empty", table=table_name, target_uri=target_uri, drift=drift)
                _recreate_dataset_with_new_schema(args, table_name, target_uri, specs)
                return specs
            elif not columns_with_data:
                # Only affected columns are empty, safe to recreate with new schema
                log_event("lance_incremental_schema_drift_columns_empty", table=table_name, target_uri=target_uri, drift=drift)
                _recreate_dataset_with_new_schema(args, table_name, target_uri, specs)
                return specs
            else:
                # Some affected columns have data, need manual intervention
                drift_with_data = [d for d in drift if d["column"] in columns_with_data]
                log_event("lance_incremental_schema_type_drift", table=table_name, target_uri=target_uri, drift=drift_with_data)
                raise RuntimeError(f"Lance schema type drift for {table_name}; run a controlled DuckDB rewrite before incremental merge: {drift_with_data}")
        else:
            # Auto-recreate disabled, fail as before
            log_event("lance_incremental_schema_type_drift", table=table_name, target_uri=target_uri, drift=drift)
            raise RuntimeError(f"Lance schema type drift for {table_name}; run a controlled DuckDB rewrite before incremental merge: {drift}")
    if alterations:
        try:
            dataset.alter_columns(*alterations)
        except Exception as exc:  # noqa: BLE001
            logged = [{"path": alteration["path"], "data_type": str(alteration.get("data_type"))} for alteration in alterations]
            log_event("lance_incremental_schema_alter_failed", table=table_name, target_uri=target_uri, alterations=logged, error=repr(exc))
            raise
        log_event("lance_incremental_schema_columns_altered", table=table_name, target_uri=target_uri, columns=[str(alteration["path"]) for alteration in alterations])
    if added:
        columns = [spec.name for spec in added]
        log_event("lance_incremental_schema_columns_add_started", table=table_name, target_uri=target_uri, columns=columns)
        try:
            if not _add_lance_columns_native(dataset, added):
                log_event("lance_incremental_schema_add_columns_duckdb_fallback", table=table_name, target_uri=target_uri, columns=columns)
                _add_lance_columns_duckdb(args, target_uri, added)
        except Exception as exc:  # noqa: BLE001
            log_event("lance_incremental_schema_add_columns_failed", table=table_name, target_uri=target_uri, columns=columns, error=repr(exc))
            raise
        log_event("lance_incremental_schema_columns_added", table=table_name, target_uri=target_uri, columns=[spec.name for spec in added])
    return specs


def _status_id_filter_values(ids: Iterable[str]) -> list[str]:
    return sorted({status_id_value(status, row_id) for row_id in ids for status in (1, 3)})


def _lance_in_filter(column: str, values: list[str]) -> str:
    return f"{column} IN (" + ", ".join(quote_literal(value) for value in values) + ")"


def _existing_current_rows_native(dataset: Any, ids: Iterable[str]) -> list[JsonMap]:
    wanted = _status_id_filter_values(ids)
    if not wanted:
        return []
    table = dataset.to_table(filter=_lance_in_filter("__status_id", wanted))
    return table.to_pylist()


def _existing_current_rows(conn: Any, target_ref: str, ids: Iterable[str]) -> list[JsonMap]:
    wanted = _status_id_filter_values(ids)
    if not wanted:
        return []
    values = ", ".join(f"({quote_literal(value)})" for value in wanted)
    result = conn.execute(f"SELECT * FROM {target_ref} WHERE __status_id IN (SELECT * FROM (VALUES {values}))")
    column_names = [column[0] for column in result.description]
    return [dict(zip(column_names, row, strict=False)) for row in result.fetchall()]


def merge_incremental_rows(table_name: str, target_uri: str, rows: list[JsonMap], specs: list[ColumnSpec] | None = None) -> int:
    if not rows:
        return 0
    ids = [str(row["_id"]) for row in rows if row.get("_id") is not None]
    import lance
    import pyarrow as pa

    dataset = lance.dataset(target_uri)
    existing_current = _existing_current_rows_native(dataset, ids)
    merge_rows = prepare_incremental_merge_rows(rows, existing_current)
    if specs is not None:
        merge_rows = normalize_rows_for_specs(merge_rows, specs)
    ensure_rows_fit_lance_schema(table_name, target_uri, merge_rows, dataset.schema)
    merge_rows = _coerce_rows_for_arrow_schema(merge_rows, dataset.schema)
    merge_table = pa.Table.from_pylist(merge_rows, schema=dataset.schema)
    try:
        row_count = dataset.count_rows()
    except Exception:  # noqa: BLE001
        row_count = None
    if row_count == 0:
        # For an empty target, append is equivalent to merge_insert (there are
        # no matched rows). It also avoids a Lance/DataFusion spill failure seen
        # when merge_insert is the first write into a newly created S3-backed
        # table.
        lance.write_dataset(merge_table, target_uri, mode="append")
        log_event("lance_incremental_empty_table_appended", table=table_name, target_uri=target_uri, rows=len(merge_rows))
        return len(merge_rows)
    dataset.merge_insert("__id_ts").when_matched_update_all().when_not_matched_insert_all().execute(merge_table)
    return len(merge_rows)


def merge_incremental_rows_with_schema_refresh(
    args: argparse.Namespace,
    state: S3JsonState,
    client: ConvexClient,
    schema_payload: JsonMap,
    source_table: str,
    physical: str,
    target_uri: str,
    rows: list[JsonMap],
    specs: list[ColumnSpec] | None,
) -> tuple[int, JsonMap]:
    try:
        return merge_incremental_rows(physical, target_uri, rows, specs), schema_payload
    except MissingLanceColumns as exc:
        if not args.reconcile_schema:
            raise
        log_event(
            "lance_incremental_missing_columns_schema_refresh_started",
            table=source_table,
            physical_table=physical,
            target_uri=target_uri,
            columns=exc.columns,
        )
        refreshed_payload, _refreshed = incremental_schema_payload_with_status(state, args, client, force_refresh=True)
        refreshed_specs = reconcile_lance_schema(args, physical, target_uri, table_schema_from_payload(refreshed_payload, source_table))
        if refreshed_specs == []:
            return 0, refreshed_payload
        merged = merge_incremental_rows(physical, target_uri, rows, refreshed_specs)
        log_event(
            "lance_incremental_missing_columns_schema_refresh_completed",
            table=source_table,
            physical_table=physical,
            target_uri=target_uri,
            columns=exc.columns,
        )
        return merged, refreshed_payload


def run_incremental_once(args: argparse.Namespace) -> JsonMap:
    bucket = env_or_arg(args, "lance_bucket", "LANCE_BUCKET")
    args.lance_scope = args.lance_scope or f"s3://{bucket}/"
    args.lance_root_uri = args.lance_root_uri or f"s3://{bucket}/{normalize_prefix(args.lance_prefix)}"
    args.catalog_alias = args.catalog_alias or os.environ.get("LANCE_CATALOG_ALIAS", "lance")
    args.aws_region = args.aws_region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    args.cursor_key = args.cursor_key or os.environ.get("LANCE_INCREMENTAL_CURSOR_KEY", "incremental/cursor.json")
    state = _incremental_state(args)
    lease = acquire_lease(state, "incremental", args.lock_ttl_seconds, args.force)
    client = _convex_client(args)
    cursor, cursor_etag = _read_incremental_cursor(state, args)
    if args.reconcile_schema:
        schema_payload, schema_refreshed = incremental_schema_payload_with_status(state, args, client)
        if schema_refreshed and getattr(args, "reconcile_existing_tables_on_schema_refresh", True):
            reconcile_schema_payload_existing_lance_tables(args, bucket, schema_payload)
    else:
        schema_payload = {"schemas": {}}
    start_cursor = cursor
    pages = rows_seen = rows_accepted = rows_merged = 0
    table_rows: dict[str, int] = {}
    try:
        for raw_rows, page_cursor, has_more in client.iter_document_deltas(cursor):
            pages += 1
            page_start_cursor = cursor
            page_rows_seen = len(raw_rows)
            page_rows_accepted = 0
            page_rows_merged = 0
            page_table_rows: dict[str, int] = {}
            rows_seen += len(raw_rows)
            by_table: dict[str, list[JsonMap]] = {}
            for raw in raw_rows:
                table_name = raw.get("_table")
                if not isinstance(table_name, str) or raw.get("_id") is None or raw.get("_ts") is None:
                    continue
                row = dict(raw)
                row["_convex_cursor"] = page_cursor
                by_table.setdefault(table_name, []).append(row)
                rows_accepted += 1
                page_rows_accepted += 1
            for source_table, table_page_rows in by_table.items():
                physical = source_to_physical_table(source_table)
                target_uri = lance_uri(bucket, args.lance_prefix, physical)
                specs = reconcile_lance_schema(args, physical, target_uri, table_schema_from_payload(schema_payload, source_table))
                if specs == []:
                    continue
                config_result = reconcile_table_config_for_dataset(args, physical, target_uri) if args.table_config_indexes else None
                try:
                    merged, schema_payload = merge_incremental_rows_with_schema_refresh(
                        args,
                        state,
                        client,
                        schema_payload,
                        source_table,
                        physical,
                        target_uri,
                        table_page_rows,
                        specs,
                    )
                except Exception as exc:  # noqa: BLE001
                    log_event(
                        "lance_incremental_table_merge_failed",
                        table=source_table,
                        physical_table=physical,
                        target_uri=target_uri,
                        rows=len(table_page_rows),
                        cursor=page_cursor,
                        page=pages,
                        error=repr(exc),
                    )
                    raise
                rows_merged += merged
                page_rows_merged += merged
                table_rows[source_table] = table_rows.get(source_table, 0) + merged
                page_table_rows[source_table] = page_table_rows.get(source_table, 0) + merged
                log_event("lance_incremental_table_merged", table=source_table, physical_table=physical, rows=len(table_page_rows), merge_rows=merged, cursor=page_cursor, table_config=config_result)
            if page_cursor is not None:
                verify_lease_owner(state, lease)
                current_payload, current_etag = state.read_with_etag(args.cursor_key)
                if current_payload is None or current_etag is None or str(current_payload.get("cursor")) != page_start_cursor or current_etag != cursor_etag:
                    raise RuntimeError(
                        f"Incremental cursor changed before page commit: expected cursor={page_start_cursor} etag={cursor_etag}, got payload={current_payload} etag={current_etag}"
                    )
                _audit_cursor_page(state, lease, pages, page_start_cursor, page_cursor, page_table_rows, page_rows_seen, page_rows_accepted, page_rows_merged)
                cursor = page_cursor
                cursor_etag = _write_incremental_cursor(state, args, cursor, cursor_etag, {"owner": lease.owner, "pages_processed": pages, "previous_cursor": page_start_cursor})
            heartbeat_lease(state, lease, args.lock_ttl_seconds)
            log_event("lance_incremental_page", page=pages, values=len(raw_rows), accepted=sum(len(v) for v in by_table.values()), cursor=cursor, has_more=has_more)
            if pages >= args.max_pages_per_sync or not has_more:
                break
        state.delete("incremental/last_error.json")
        log_event("lance_incremental_completed", pages=pages, rows_seen=rows_seen, rows_accepted=rows_accepted, rows_merged=rows_merged, tables=len(table_rows), start_cursor=start_cursor, end_cursor=cursor)
        return {"pages": pages, "rows_seen": rows_seen, "rows_accepted": rows_accepted, "rows_merged": rows_merged, "tables": sorted(source_to_physical_table(table) for table in table_rows), "start_cursor": start_cursor, "end_cursor": cursor}
    except BaseException as exc:
        state.write("incremental/last_error.json", {"owner": lease.owner, "error": repr(exc), "start_cursor": start_cursor, "last_cursor": cursor, "updated_at": int(time.time())})
        raise
    finally:
        release_lease(state, lease)


def run_incremental_loop(args: argparse.Namespace) -> None:
    touched_tables: set[str] = set()
    pages_since_optimize = 0
    rows_since_optimize = 0
    last_optimized_at = time.time()
    while True:
        optimized = False
        try:
            stats = run_incremental_once(args)
            touched_tables.update(str(table) for table in stats.get("tables", []))
            pages_since_optimize += int(stats.get("pages", 0))
            rows_since_optimize += int(stats.get("rows_merged", 0))
            optimized = maybe_optimize_incremental_indices(args, touched_tables, pages_since_optimize, rows_since_optimize, last_optimized_at)
            if optimized:
                touched_tables.clear()
                pages_since_optimize = 0
                rows_since_optimize = 0
                last_optimized_at = time.time()
            if not optimized and should_run_idle_maintenance(args, stats):
                optimized = run_idle_maintenance_once(args, _incremental_state(args))
        except LeaseUnavailable as exc:
            log_event("lance_incremental_lock_unavailable", error=str(exc))
        if not optimized:
            time.sleep(args.sleep_seconds)


def maybe_optimize_incremental_indices(args: argparse.Namespace, touched_tables: set[str], pages: int, rows: int, last_optimized_at: float) -> bool:
    if not args.optimize_touched_indices or not touched_tables:
        return False
    elapsed = time.time() - last_optimized_at
    if pages < args.optimize_indices_pages and rows < args.optimize_indices_rows and elapsed < args.optimize_indices_seconds:
        return False
    bucket = env_or_arg(args, "lance_bucket", "LANCE_BUCKET")
    optimized = 0
    failed: list[JsonMap] = []
    import lance

    for table in sorted(touched_tables):
        target_uri = lance_uri(bucket, args.lance_prefix, table)
        try:
            lance.dataset(target_uri).optimize.optimize_indices()
            optimized += 1
            log_event("lance_incremental_indices_optimized", table=table, target_uri=target_uri, pages=pages, rows=rows, elapsed_seconds=round(elapsed, 3))
        except Exception as exc:  # noqa: BLE001
            failed.append({"table": table, "error": repr(exc)})
            log_event("lance_incremental_indices_optimize_failed", table=table, target_uri=target_uri, error=repr(exc))
    log_event("lance_incremental_indices_optimize_completed", tables=len(touched_tables), optimized=optimized, failed=len(failed), pages=pages, rows=rows, elapsed_seconds=round(elapsed, 3))
    return True


def should_run_idle_maintenance(args: argparse.Namespace, stats: JsonMap) -> bool:
    return bool(args.idle_maintenance and int(stats.get("rows_accepted", 0)) <= args.idle_maintenance_max_rows and int(stats.get("pages", 0)) <= args.idle_maintenance_max_pages)


def maintenance_actions(args: argparse.Namespace) -> list[tuple[str, int]]:
    return [
        ("optimize_indices", args.maintenance_optimize_indices_seconds),
        ("compact_files", args.maintenance_compact_files_seconds),
        ("cleanup_old_versions", args.maintenance_cleanup_old_versions_seconds),
    ]


def in_utc_hour_window(start_hour: int, end_hour: int) -> bool:
    hour = time.gmtime().tm_hour
    if start_hour == end_hour:
        return True
    if start_hour < end_hour:
        return start_hour <= hour < end_hour
    return hour >= start_hour or hour < end_hour


def maintenance_action_allowed(args: argparse.Namespace, action: str) -> bool:
    if action == "optimize_indices":
        return True
    return in_utc_hour_window(args.maintenance_heavy_window_start_utc, args.maintenance_heavy_window_end_utc)


def default_maintenance_state() -> JsonMap:
    return {"next_action_index": 0, "next_table_index_by_action": {}, "last_run_by_action": {}, "failures": []}


def read_maintenance_state(state: S3JsonState, args: argparse.Namespace) -> JsonMap:
    payload = state.read(args.maintenance_state_key)
    return payload if payload else default_maintenance_state()


def write_maintenance_state(state: S3JsonState, args: argparse.Namespace, payload: JsonMap) -> None:
    payload["updated_at"] = int(time.time())
    state.write(args.maintenance_state_key, payload)


def write_maintenance_audit(state: S3JsonState, args: argparse.Namespace, event: JsonMap) -> None:
    if not args.maintenance_audit_key_prefix:
        return
    now = time.time()
    parts = time.gmtime(now)
    key = f"{args.maintenance_audit_key_prefix.strip('/')}/{parts.tm_year:04d}/{parts.tm_mon:02d}/{parts.tm_mday:02d}/{int(now * 1000)}-{event['action']}-{event['table']}-{uuid.uuid4().hex}.jsonl"
    payload = {"ts": int(now), **event}
    state.client.put_object(Bucket=state.bucket, Key=state.key(key), Body=(json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"), ContentType="application/x-ndjson")


def choose_maintenance_work(args: argparse.Namespace, maintenance_state: JsonMap, tables: list[str]) -> tuple[str, str] | None:
    if not tables:
        return None
    now = int(time.time())
    actions = maintenance_actions(args)
    next_action_index = int(maintenance_state.get("next_action_index", 0))
    last_run_by_action = maintenance_state.setdefault("last_run_by_action", {})
    next_table_index_by_action = maintenance_state.setdefault("next_table_index_by_action", {})
    for action_offset in range(len(actions)):
        action_index = (next_action_index + action_offset) % len(actions)
        action, cadence_seconds = actions[action_index]
        if cadence_seconds <= 0 or not maintenance_action_allowed(args, action):
            continue
        table_index = int(next_table_index_by_action.get(action, 0))
        action_last_runs = last_run_by_action.setdefault(action, {})
        for table_offset in range(len(tables)):
            candidate_index = (table_index + table_offset) % len(tables)
            table = tables[candidate_index]
            if now - int(action_last_runs.get(table, 0)) >= cadence_seconds:
                maintenance_state["next_action_index"] = (action_index + 1) % len(actions)
                next_table_index_by_action[action] = (candidate_index + 1) % len(tables)
                return action, table
    return None


def run_idle_maintenance_once(args: argparse.Namespace, state: S3JsonState) -> bool:
    bucket = env_or_arg(args, "lance_bucket", "LANCE_BUCKET")
    tables = list_lance_tables(argparse.Namespace(tables=args.maintenance_tables, tables_file=args.maintenance_tables_file, lance_bucket=bucket, lance_prefix=args.lance_prefix, aws_region=args.aws_region))
    maintenance_state = read_maintenance_state(state, args)
    work = choose_maintenance_work(args, maintenance_state, tables)
    if work is None:
        return False
    action, table = work
    target_uri = lance_uri(bucket, args.lance_prefix, table)
    started = time.monotonic()
    import lance

    try:
        dataset = lance.dataset(target_uri)
        if action == "optimize_indices":
            result = dataset.optimize.optimize_indices()
        elif action == "compact_files":
            result = dataset.optimize.compact_files(materialize_deletions=True, num_threads=args.maintenance_compact_threads)
        elif action == "cleanup_old_versions":
            from datetime import timedelta

            result = dataset.cleanup_old_versions(older_than=timedelta(seconds=args.maintenance_cleanup_older_than_seconds), retain_versions=args.maintenance_retain_versions)
        else:
            raise ValueError(f"unknown maintenance action: {action}")
        elapsed = round(time.monotonic() - started, 3)
        maintenance_state.setdefault("last_run_by_action", {}).setdefault(action, {})[table] = int(time.time())
        audit_event = {"status": "completed", "action": action, "table": table, "target_uri": target_uri, "result": repr(result), "elapsed_seconds": elapsed}
        log_event("lance_incremental_idle_maintenance_completed", **audit_event)
        write_maintenance_audit(state, args, audit_event)
    except Exception as exc:  # noqa: BLE001
        elapsed = round(time.monotonic() - started, 3)
        failures = maintenance_state.setdefault("failures", [])
        failures.append({"action": action, "table": table, "error": repr(exc), "updated_at": int(time.time())})
        del failures[:-20]
        audit_event = {"status": "failed", "action": action, "table": table, "target_uri": target_uri, "error": repr(exc), "elapsed_seconds": elapsed}
        log_event("lance_incremental_idle_maintenance_failed", **audit_event)
        write_maintenance_audit(state, args, audit_event)
    write_maintenance_state(state, args, maintenance_state)
    return True


def list_lance_tables(args: argparse.Namespace) -> list[str]:
    if args.tables:
        return [source_to_physical_table(part.strip()) for part in args.tables.split(",") if part.strip()]
    if args.tables_file:
        with open(args.tables_file, encoding="utf-8") as handle:
            return [source_to_physical_table(line.strip()) for line in handle if line.strip() and not line.strip().startswith("#")]
    if not args.lance_bucket:
        raise SystemExit("--tables, --tables-file, or --lance-bucket is required")
    import boto3

    prefix = normalize_prefix(args.lance_prefix)
    prefix = f"{prefix}/" if prefix else ""
    paginator = boto3.client("s3", region_name=args.aws_region).get_paginator("list_objects_v2")
    tables: set[str] = set()
    for page in paginator.paginate(Bucket=args.lance_bucket, Prefix=prefix, Delimiter="/"):
        for item in page.get("CommonPrefixes", []):
            raw = str(item.get("Prefix") or "")
            if not raw.endswith(".lance/"):
                continue
            name = raw.removeprefix(prefix).removesuffix("/").removesuffix(".lance")
            if name:
                tables.add(name)
    return sorted(tables)


def table_column_names(conn: Any, table_ref: str) -> list[str]:
    rows = conn.execute(f"DESCRIBE {table_ref}").fetchall()
    return [str(row[0]) for row in rows]


def table_columns(conn: Any, table_ref: str) -> set[str]:
    return set(table_column_names(conn, table_ref))


LANCE_SCALAR_INDEX_ENV = {
    "LANCE_BYPASS_SPILLING": "true",
}

DEFAULT_LANCE_TABLE_CONFIG_STATE_PREFIX = "lance-incremental/table-config-state"


def index_name(table_name: str, column: str) -> str:
    return source_to_physical_table(f"{table_name}_{column}_idx")


def generated_index_columns() -> list[IndexSpec]:
    return [IndexSpec("__id_ts", "BTREE"), IndexSpec("__status_id", "BTREE"), IndexSpec("__status", "BITMAP")]


def _index_specs(values: Iterable[IndexSpec | tuple[str, str] | tuple[str, str, str]]) -> list[IndexSpec]:
    specs: list[IndexSpec] = []
    for value in values:
        if isinstance(value, IndexSpec):
            specs.append(value)
        elif len(value) == 2:
            column, index_type = value
            specs.append(IndexSpec(str(column), str(index_type)))
        else:
            column, index_type, name = value
            specs.append(IndexSpec(str(column), str(index_type), str(name)))
    return specs


def _validate_index_type(index_type: str) -> str:
    normalized = index_type.upper()
    if normalized not in {"BTREE", "BITMAP", "LABEL_LIST"}:
        raise ValueError(f"Unsupported Lance scalar index type: {index_type}")
    return normalized


def _table_config_name(table_name: str) -> str:
    return f"{source_to_physical_table(table_name)}.toml"


def _parse_table_config_toml(table_name: str, payload: bytes, version: str) -> TableConfig:
    import tomllib

    parsed = tomllib.loads(payload.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"Lance table config for {table_name} must be a TOML object")
    raw_indexes = parsed.get("indexes", [])
    if not isinstance(raw_indexes, list):
        raise ValueError(f"Lance table config for {table_name} indexes must be an array of tables")
    indexes: list[IndexSpec] = []
    for raw in raw_indexes:
        if not isinstance(raw, dict):
            raise ValueError(f"Lance table config for {table_name} index entries must be tables")
        column = raw.get("column")
        if not isinstance(column, str) or not column:
            raise ValueError(f"Lance table config for {table_name} index entries require column")
        raw_type = raw.get("type", raw.get("index_type", "BTREE"))
        if not isinstance(raw_type, str) or not raw_type:
            raise ValueError(f"Lance table config for {table_name} index {column} requires string type")
        raw_name = raw.get("name")
        if raw_name is not None and (not isinstance(raw_name, str) or not raw_name):
            raise ValueError(f"Lance table config for {table_name} index {column} name must be a non-empty string")
        indexes.append(IndexSpec(column, _validate_index_type(raw_type), raw_name))
    return TableConfig(table_name=source_to_physical_table(table_name), version=version, indexes=indexes)


def _read_local_table_config(table_name: str, config_dir: str) -> TableConfig | None:
    path = Path(config_dir) / _table_config_name(table_name)
    if not path.exists():
        return None
    payload = path.read_bytes()
    return _parse_table_config_toml(table_name, payload, f"sha256:{hashlib.sha256(payload).hexdigest()}")


def _read_s3_table_config(table_name: str, bucket: str, prefix: str, region: str | None) -> TableConfig | None:
    from botocore.exceptions import ClientError
    import boto3

    key_prefix = prefix.strip("/")
    key = f"{key_prefix}/{_table_config_name(table_name)}" if key_prefix else _table_config_name(table_name)
    try:
        obj = boto3.client("s3", region_name=region).get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
            return None
        raise
    payload = obj["Body"].read()
    etag = str(obj.get("ETag") or "").strip('"')
    version = f"etag:{etag}" if etag else f"sha256:{hashlib.sha256(payload).hexdigest()}"
    return _parse_table_config_toml(table_name, payload, version)


def load_table_config(args: argparse.Namespace, table_name: str) -> TableConfig | None:
    local_dir = getattr(args, "table_config_dir", None) or os.environ.get("LANCE_TABLE_CONFIG_DIR")
    if local_dir:
        config = _read_local_table_config(table_name, local_dir)
        if config is not None:
            return config
    bucket = getattr(args, "table_config_bucket", None) or os.environ.get("LANCE_TABLE_CONFIG_BUCKET")
    prefix = getattr(args, "table_config_prefix", None) or os.environ.get("LANCE_TABLE_CONFIG_PREFIX", "config/lance/tables")
    if bucket:
        return _read_s3_table_config(table_name, bucket, prefix, getattr(args, "aws_region", None))
    return None


def requested_index_specs(args: argparse.Namespace, table_name: str) -> tuple[list[IndexSpec], TableConfig | None]:
    requested: list[IndexSpec] = []
    if getattr(args, "generated_indexes", True):
        requested.extend(generated_index_columns())
    config = load_table_config(args, table_name) if getattr(args, "table_config_indexes", True) else None
    if config is not None:
        requested.extend(config.indexes)
    deduped: list[IndexSpec] = []
    seen: set[tuple[str, str, str | None]] = set()
    for spec in requested:
        key = (spec.column, _validate_index_type(spec.index_type), spec.name)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(IndexSpec(spec.column, key[1], spec.name))
    return deduped, config


def desired_index_names(table_name: str, specs: Iterable[IndexSpec]) -> set[str]:
    return {spec.name or index_name(table_name, spec.column) for spec in specs}


def dataset_index_names(dataset: Any) -> set[str]:
    list_indices = getattr(dataset, "list_indices", None)
    if list_indices is None:
        return set()
    names: set[str] = set()
    for item in list_indices() or []:
        if isinstance(item, str):
            names.add(item)
            continue
        if isinstance(item, dict):
            raw_name = item.get("name")
        else:
            raw_name = getattr(item, "name", None)
        if raw_name:
            names.add(str(raw_name))
    return names


def log_extra_lance_indexes(table_name: str, target_uri: str, dataset: Any, desired_names: set[str]) -> list[str]:
    extra = sorted(dataset_index_names(dataset) - desired_names)
    if extra:
        log_event("lance_table_config_extra_indexes_detected", table=table_name, target_uri=target_uri, extra_indexes=extra, action="would_drop_if_subtractive")
    return extra


def table_config_state(args: argparse.Namespace) -> S3JsonState | None:
    bucket = (
        getattr(args, "table_config_state_bucket", None)
        or os.environ.get("LANCE_TABLE_CONFIG_STATE_BUCKET")
        or getattr(args, "state_bucket", None)
        or os.environ.get("LANCE_INCREMENTAL_STATE_BUCKET")
        or os.environ.get("S3TABLES_BACKFILL_STATE_BUCKET")
    )
    if not bucket:
        return None
    prefix = getattr(args, "table_config_state_prefix", None) or os.environ.get("LANCE_TABLE_CONFIG_STATE_PREFIX") or DEFAULT_LANCE_TABLE_CONFIG_STATE_PREFIX
    return S3JsonState(bucket, prefix)


def table_config_state_key(table_name: str) -> str:
    return f"{source_to_physical_table(table_name)}.json"


def read_applied_table_config_version(state: S3JsonState | None, table_name: str) -> str | None:
    if state is None:
        return None
    payload = state.read(table_config_state_key(table_name))
    return str(payload.get("config_version")) if payload and payload.get("config_version") is not None else None


def write_applied_table_config_state(state: S3JsonState | None, table_name: str, config: TableConfig | None, created: int, skipped: list[JsonMap]) -> None:
    if state is None or config is None:
        return
    if any(item.get("reason") != "already_exists" for item in skipped):
        return
    state.write(
        table_config_state_key(table_name),
        {
            "table": source_to_physical_table(table_name),
            "config_version": config.version,
            "applied_at": int(time.time()),
            "indexes": [{"column": index.column, "type": index.index_type, **({"name": index.name} if index.name else {})} for index in config.indexes],
            "created_indexes": created,
            "skipped_indexes": skipped,
        },
    )


def reconcile_table_config_for_dataset(args: argparse.Namespace, table_name: str, target_uri: str, force: bool = False) -> JsonMap:
    config = load_table_config(args, table_name)
    if config is None:
        return {"configured": False, "created": 0, "skipped": [], "version": None}
    requested, _config = requested_index_specs(args, table_name)
    desired_names = desired_index_names(table_name, requested)
    state = table_config_state(args)
    import lance

    dataset = lance.dataset(target_uri)
    extra_indexes = log_extra_lance_indexes(table_name, target_uri, dataset, desired_names)
    if not force and read_applied_table_config_version(state, table_name) == config.version:
        return {"configured": True, "created": 0, "skipped": [], "version": config.version, "already_applied": True, "extra_indexes": extra_indexes}

    created, skipped = create_indexes_for_dataset(dataset, table_name, config.indexes)
    write_applied_table_config_state(state, table_name, config, created, skipped)
    return {"configured": True, "created": created, "skipped": skipped, "version": config.version, "already_applied": False, "extra_indexes": extra_indexes}


def create_indexes_for_columns(conn: Any, table_name: str, table_ref: str, requested_columns: Iterable[IndexSpec | tuple[str, str] | tuple[str, str, str]]) -> tuple[int, list[JsonMap]]:
    columns = table_columns(conn, table_ref)
    skipped: list[JsonMap] = []
    created = 0
    for spec in _index_specs(requested_columns):
        column = spec.column
        index_type = _validate_index_type(spec.index_type)
        if column not in columns:
            skipped.append({"table": table_name, "column": column, "reason": "missing_column"})
            continue
        statement = f"CREATE INDEX {spec.name or index_name(table_name, column)} ON {table_ref} ({lance_index_column_reference(column)}) USING {index_type}"
        try:
            conn.execute(statement)
            created += 1
        except Exception as exc:  # noqa: BLE001
            if "already exists" in str(exc).lower():
                skipped.append({"table": table_name, "column": column, "error": repr(exc), "reason": "already_exists"})
                continue
            raise
    return created, skipped


def run_create_lance_indexes(args: argparse.Namespace) -> None:
    args.aws_region = args.aws_region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    bucket = args.lance_bucket or os.environ.get("LANCE_BUCKET")
    if not args.lance_root_uri:
        if not bucket:
            raise SystemExit("--lance-root-uri, --lance-bucket, or LANCE_BUCKET is required")
        args.lance_root_uri = f"s3://{bucket}/{normalize_prefix(args.lance_prefix)}"
    args.lance_scope = args.lance_scope or (f"s3://{bucket}/" if bucket else args.lance_root_uri)
    tables = list_lance_tables(args)
    if args.max_tables is not None:
        tables = tables[: args.max_tables]
    if not tables:
        raise SystemExit("No Lance tables found")

    log_event("lance_index_creation_started", tables=len(tables), lance_root_uri=args.lance_root_uri, dry_run=args.dry_run)
    conn = open_lance_duckdb(args)
    completed = 0
    failures: list[JsonMap] = []
    skipped: list[JsonMap] = []
    try:
        for table_name in tables:
            dataset_uri = f"{args.lance_root_uri.rstrip('/')}/{table_name}.lance"
            table_ref = quote_literal(dataset_uri)
            try:
                requested_columns, config = requested_index_specs(args, table_name)
                if args.dry_run:
                    for spec in requested_columns:
                        print(f"CREATE INDEX {spec.name or index_name(table_name, spec.column)} ON {table_ref} ({lance_index_column_reference(spec.column)}) USING {_validate_index_type(spec.index_type)};")
                    indexes = len(requested_columns)
                else:
                    indexes, table_skipped = create_indexes_for_columns(conn, table_name, table_ref, requested_columns)
                    skipped.extend(table_skipped)
                    write_applied_table_config_state(table_config_state(args), table_name, config, indexes, table_skipped)
                completed += 1
                log_event("lance_index_creation_table_completed", table=table_name, indexes=indexes, table_config_version=config.version if config else None)
            except Exception as exc:  # noqa: BLE001
                failure = {"table": table_name, "error": repr(exc)}
                failures.append(failure)
                log_event("lance_index_creation_table_failed", **failure)
                if args.stop_on_failure:
                    break
    finally:
        conn.close()
    log_event("lance_index_creation_finished", completed=completed, failed=len(failures), skipped=len(skipped))
    if skipped:
        print(json.dumps({"skipped": skipped}, indent=2, sort_keys=True), file=sys.stderr)
    if failures:
        print(json.dumps({"failures": failures}, indent=2, sort_keys=True), file=sys.stderr)
        raise SystemExit(1)


def run_backfill_generated_columns(args: argparse.Namespace) -> None:
    args.aws_region = args.aws_region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    bucket = args.lance_bucket or os.environ.get("LANCE_BUCKET")
    if not args.lance_root_uri:
        if not bucket:
            raise SystemExit("--lance-root-uri, --lance-bucket, or LANCE_BUCKET is required")
        args.lance_root_uri = f"s3://{bucket}/{normalize_prefix(args.lance_prefix)}"
    args.lance_scope = args.lance_scope or (f"s3://{bucket}/" if bucket else args.lance_root_uri)
    args.catalog_alias = args.catalog_alias or os.environ.get("LANCE_CATALOG_ALIAS", "lance_ns")
    tables = list_lance_tables(args)
    if args.max_tables is not None:
        tables = tables[: args.max_tables]
    if not tables:
        raise SystemExit("No Lance tables found")

    log_event("lance_generated_backfill_started", tables=len(tables), lance_root_uri=args.lance_root_uri, dry_run=args.dry_run)
    conn = open_lance_duckdb(args)
    completed = 0
    failures: list[JsonMap] = []
    try:
        for table_name in tables:
            dataset_uri = f"{args.lance_root_uri.rstrip('/')}/{table_name}.lance"
            table_ref = quote_literal(dataset_uri)
            try:
                columns = table_columns(conn, table_ref)
                missing_required = {"_id", "_ts", "_current"} - columns
                if missing_required:
                    log_event("lance_generated_backfill_skipped", table=table_name, reason="missing_required_columns", columns=sorted(missing_required))
                    continue
                statements: list[str] = []
                base_columns = [column for column in columns if column not in GENERATED_COLUMNS]
                select_list = ", ".join(quote_identifier(column) for column in base_columns)
                temp_table = quote_identifier(f"generated_backfill_{table_name}")
                statements.append(
                    f"CREATE OR REPLACE TEMP TABLE {temp_table} AS SELECT "
                    f"{select_list}, "
                    f"{status_expr()} AS __status, "
                    f"{status_id_expr()} AS __status_id, "
                    f"{id_ts_expr()} AS __id_ts "
                    f"FROM {table_ref}"
                )
                statements.append(f"COPY {temp_table} TO {table_ref} (FORMAT lance, mode 'overwrite')")
                if args.create_indexes:
                    statements.extend(
                        [
                            f"CREATE INDEX {source_to_physical_table(f'{table_name}__id_ts_idx')} ON {quote_literal(dataset_uri)} (__id_ts) USING BTREE",
                            f"CREATE INDEX {source_to_physical_table(f'{table_name}__status_id_idx')} ON {quote_literal(dataset_uri)} (__status_id) USING BTREE",
                            f"CREATE INDEX {source_to_physical_table(f'{table_name}__status_idx')} ON {quote_literal(dataset_uri)} (__status) USING BITMAP",
                        ]
                    )
                if args.dry_run:
                    for statement in statements:
                        print(statement + ";")
                else:
                    for statement in statements:
                        conn.execute(statement)
                completed += 1
                log_event("lance_generated_backfill_table_completed", table=table_name, statements=len(statements))
            except Exception as exc:  # noqa: BLE001
                failure = {"table": table_name, "error": repr(exc)}
                failures.append(failure)
                log_event("lance_generated_backfill_table_failed", **failure)
                if args.stop_on_failure:
                    break
    finally:
        conn.close()
    log_event("lance_generated_backfill_finished", completed=completed, failed=len(failures))
    if failures:
        print(json.dumps({"failures": failures}, indent=2, sort_keys=True), file=sys.stderr)
        raise SystemExit(1)


def run_drop_legacy_current(args: argparse.Namespace) -> None:
    args.aws_region = args.aws_region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    bucket = args.lance_bucket or os.environ.get("LANCE_BUCKET")
    if not args.lance_root_uri:
        if not bucket:
            raise SystemExit("--lance-root-uri, --lance-bucket, or LANCE_BUCKET is required")
        args.lance_root_uri = f"s3://{bucket}/{normalize_prefix(args.lance_prefix)}"
    args.lance_scope = args.lance_scope or (f"s3://{bucket}/" if bucket else args.lance_root_uri)
    args.catalog_alias = args.catalog_alias or os.environ.get("LANCE_CATALOG_ALIAS", "lance_ns")
    tables = list_lance_tables(args)
    if args.max_tables is not None:
        tables = tables[: args.max_tables]
    if not tables:
        raise SystemExit("No Lance tables found")

    log_event("lance_drop_legacy_current_started", tables=len(tables), lance_root_uri=args.lance_root_uri, dry_run=args.dry_run)
    conn = open_lance_duckdb(args)
    completed = 0
    failures: list[JsonMap] = []
    try:
        for table_name in tables:
            dataset_uri = f"{args.lance_root_uri.rstrip('/')}/{table_name}.lance"
            table_ref = quote_literal(dataset_uri)
            try:
                columns = table_columns(conn, table_ref)
                if "_current" not in columns:
                    log_event("lance_drop_legacy_current_skipped", table=table_name, reason="missing_column")
                    continue
                if {"__status", "__status_id", "__id_ts"} - columns:
                    missing = sorted({"__status", "__status_id", "__id_ts"} - columns)
                    raise RuntimeError(f"refusing to drop _current before generated columns exist: {missing}")
                kept_columns = [column for column in columns if column != "_current"]
                select_list = ", ".join(quote_identifier(column) for column in kept_columns)
                temp_table = quote_identifier(f"drop_current_{table_name}")
                statements = [
                    f"CREATE OR REPLACE TEMP TABLE {temp_table} AS SELECT {select_list} FROM {table_ref}",
                    f"COPY {temp_table} TO {table_ref} (FORMAT lance, mode 'overwrite')",
                ]
                if args.dry_run:
                    for statement in statements:
                        print(statement + ";")
                else:
                    for statement in statements:
                        conn.execute(statement)
                completed += 1
                log_event("lance_drop_legacy_current_table_completed", table=table_name)
            except Exception as exc:  # noqa: BLE001
                failure = {"table": table_name, "error": repr(exc)}
                failures.append(failure)
                log_event("lance_drop_legacy_current_table_failed", **failure)
                if args.stop_on_failure:
                    break
    finally:
        conn.close()
    log_event("lance_drop_legacy_current_finished", completed=completed, failed=len(failures))
    if failures:
        print(json.dumps({"failures": failures}, indent=2, sort_keys=True), file=sys.stderr)
        raise SystemExit(1)


def decode_json_string_literal(value: Any) -> Any:
    if not isinstance(value, str) or len(value) < 2 or value[0] != '"' or value[-1] != '"':
        return value
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return value
    return decoded if isinstance(decoded, str) else value


def quoted_json_string_condition(column: str) -> str:
    ident = quote_identifier(column)
    value = f"CAST({ident} AS VARCHAR)"
    parsed = f"CAST({value} AS JSON)"
    return f"json_valid({value}) AND json_type({parsed}) = 'VARCHAR'"


def repair_quoted_json_string_expr(column: str) -> str:
    ident = quote_identifier(column)
    value = f"CAST({ident} AS VARCHAR)"
    parsed = f"CAST({value} AS JSON)"
    return f"CASE WHEN {quoted_json_string_condition(column)} THEN json_extract_string({parsed}, '$') ELSE {ident} END"


def build_repair_quoted_json_strings_select_sql(conn: Any, table_ref: str, repair_columns: set[str]) -> str:
    columns = table_column_names(conn, table_ref)
    select_list = []
    for column in columns:
        if column in repair_columns:
            select_list.append(f"{repair_quoted_json_string_expr(column)} AS {quote_identifier(column)}")
        else:
            select_list.append(quote_identifier(column))
    return f"SELECT {', '.join(select_list)} FROM {table_ref}"


def count_quoted_json_string_values(conn: Any, table_ref: str, columns: list[str], limit: int | None = None) -> tuple[int, int]:
    predicates = [quoted_json_string_condition(column) for column in columns]
    select_list = ", ".join(quote_identifier(column) for column in columns)
    source = f"SELECT {select_list} FROM {table_ref}"
    if limit is not None:
        source += f" LIMIT {int(limit)}"
    checked = conn.execute(f"SELECT count(*) FROM ({source})").fetchone()[0] * len(columns)
    repaired = conn.execute(f"SELECT {sum_expr(predicates)} FROM ({source})").fetchone()[0] or 0
    return int(checked), int(repaired)


def sum_expr(predicates: list[str]) -> str:
    return " + ".join(f"sum(CASE WHEN {predicate} THEN 1 ELSE 0 END)" for predicate in predicates) if predicates else "0"


def _bytes_map_to_strings(value: dict[bytes, bytes] | None) -> dict[str, str]:
    if not value:
        return {}
    return {key.decode("utf-8", errors="replace"): item.decode("utf-8", errors="replace") for key, item in sorted(value.items())}


def arrow_schema_signature(schema: Any) -> JsonMap:
    return {
        "metadata": _bytes_map_to_strings(schema.metadata),
        "fields": [
            {
                "name": field.name,
                "type": str(field.type),
                "nullable": bool(field.nullable),
                "metadata": _bytes_map_to_strings(field.metadata),
            }
            for field in schema
        ],
    }


def validate_repair_columns(dataset: Any, requested_columns: list[str], fail_on_missing: bool) -> tuple[list[str], list[str]]:
    import pyarrow.types as pat

    fields = {field.name: field for field in dataset.schema}
    missing = sorted(set(requested_columns) - set(fields))
    if missing and fail_on_missing:
        raise RuntimeError(f"repair columns missing from Lance dataset: {missing}")
    columns = [column for column in requested_columns if column in fields]
    non_string = [column for column in columns if not (pat.is_string(fields[column].type) or pat.is_large_string(fields[column].type))]
    if non_string:
        raise RuntimeError(f"repair-quoted-json-strings only supports string columns, got non-string columns: {non_string}")
    return columns, missing


def restore_lance_version(dataset_uri: str, version: int) -> None:
    import lance

    lance.dataset(dataset_uri).checkout_version(version).restore()


def repair_index_columns(args: argparse.Namespace, table_name: str) -> list[IndexSpec]:
    requested, _config = requested_index_specs(args, table_name)
    return requested


REPAIR_EXCLUDED_SCHEMA_COLUMNS = {
    "_id",
    "_creationTime",
    "_table",
    "_ts",
    "_deleted",
    "_convex_cursor",
    "__status",
    "__status_id",
    "__id_ts",
}


def desired_repair_columns_from_schema(table_schema: JsonMap) -> list[str]:
    return [spec.name for spec in schema_column_specs(table_schema) if spec.kind == "string" and spec.name not in REPAIR_EXCLUDED_SCHEMA_COLUMNS]


def physical_to_source_schema_map(schema_payload: JsonMap) -> dict[str, tuple[str, JsonMap]]:
    schemas = schema_payload.get("schemas", schema_payload)
    if not isinstance(schemas, dict):
        return {}
    return {
        source_to_physical_table(source_table): (source_table, table_schema)
        for source_table, table_schema in schemas.items()
        if isinstance(source_table, str) and isinstance(table_schema, dict)
    }


def load_schema_payload(args: argparse.Namespace) -> JsonMap:
    if args.schema_json:
        with open(args.schema_json, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise SystemExit("--schema-json must contain a JSON object")
        return payload
    return _convex_client(args).json_schemas(delta_schema=True)


def _duckdb_string_columns(conn: Any, table_ref: str) -> set[str]:
    rows = conn.execute(f"DESCRIBE {table_ref}").fetchall()
    return {str(row[0]) for row in rows if "VARCHAR" in str(row[1]).upper() or str(row[1]).upper() == "STRING"}


def run_discover_quoted_json_string_repairs(args: argparse.Namespace) -> None:
    args.aws_region = args.aws_region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    bucket = args.lance_bucket or os.environ.get("LANCE_BUCKET")
    if not args.lance_root_uri:
        if not bucket:
            raise SystemExit("--lance-root-uri, --lance-bucket, or LANCE_BUCKET is required")
        args.lance_root_uri = f"s3://{bucket}/{normalize_prefix(args.lance_prefix)}"
    args.lance_scope = args.lance_scope or (f"s3://{bucket}/" if bucket else args.lance_root_uri)
    schema_by_physical = physical_to_source_schema_map(load_schema_payload(args))
    tables = list_lance_tables(args)
    if args.max_tables is not None:
        tables = tables[: args.max_tables]

    log_event("lance_discover_quoted_json_string_repairs_started", tables=len(tables), lance_root_uri=args.lance_root_uri)
    candidates: list[JsonMap] = []
    skipped: list[JsonMap] = []
    failures: list[JsonMap] = []
    conn = open_lance_duckdb(args)
    try:
        for table_name in tables:
            schema_entry = schema_by_physical.get(table_name)
            if schema_entry is None:
                skipped.append({"table": table_name, "reason": "missing_convex_schema"})
                continue
            source_table, table_schema = schema_entry
            desired_columns = desired_repair_columns_from_schema(table_schema)
            if not desired_columns:
                skipped.append({"table": table_name, "source_table": source_table, "reason": "no_desired_string_columns"})
                continue
            table_ref = quote_literal(f"{args.lance_root_uri.rstrip('/')}/{table_name}.lance")
            try:
                actual_string_columns = _duckdb_string_columns(conn, table_ref)
                columns = [column for column in desired_columns if column in actual_string_columns]
                missing = sorted(set(desired_columns) - actual_string_columns)
                if not columns:
                    skipped.append({"table": table_name, "source_table": source_table, "reason": "no_matching_string_columns", "missing_or_non_string": missing})
                    continue
                for column in columns:
                    checked, repaired = count_quoted_json_string_values(conn, table_ref, [column], args.limit)
                    if repaired:
                        candidates.append(
                            {
                                "table": table_name,
                                "source_table": source_table,
                                "column": column,
                                "checked": checked,
                                "would_repair": repaired,
                                "sampled": args.limit is not None,
                                "sample_limit": args.limit,
                            }
                        )
            except Exception as exc:  # noqa: BLE001
                failures.append({"table": table_name, "source_table": source_table, "error": repr(exc)})
                if args.stop_on_failure:
                    break
    finally:
        conn.close()
    log_event("lance_discover_quoted_json_string_repairs_finished", candidates=len(candidates), skipped=len(skipped), failed=len(failures))
    print(json.dumps({"candidates": candidates, "skipped": skipped if args.include_skipped else [], "failures": failures}, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


def run_repair_quoted_json_strings(args: argparse.Namespace) -> None:
    args.aws_region = args.aws_region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    bucket = args.lance_bucket or os.environ.get("LANCE_BUCKET")
    if not args.lance_root_uri:
        if not bucket:
            raise SystemExit("--lance-root-uri, --lance-bucket, or LANCE_BUCKET is required")
        args.lance_root_uri = f"s3://{bucket}/{normalize_prefix(args.lance_prefix)}"
    args.lance_scope = args.lance_scope or (f"s3://{bucket}/" if bucket else args.lance_root_uri)
    tables = list_lance_tables(args)
    if args.max_tables is not None:
        tables = tables[: args.max_tables]

    lease: TableLease | None = None
    state: S3JsonState | None = None
    if args.apply and args.use_lock:
        state = _incremental_state(args)
        lease = acquire_lease(state, "incremental", args.lock_ttl_seconds, args.force)

    import lance

    log_event("lance_repair_quoted_json_strings_started", tables=len(tables), lance_root_uri=args.lance_root_uri, apply=args.apply)
    report: list[JsonMap] = []
    failures: list[JsonMap] = []
    completed = 0
    conn = None
    try:
        conn = open_lance_duckdb(args)
        for table_name in tables:
            dataset_uri = f"{args.lance_root_uri.rstrip('/')}/{table_name}.lance"
            table_ref = quote_literal(dataset_uri)
            requested_columns = [column.strip() for column in args.columns.split(",") if column.strip()]
            try:
                if state is not None and lease is not None:
                    heartbeat_lease(state, lease, args.lock_ttl_seconds)
                dataset = lance.dataset(dataset_uri)
                old_version = int(dataset.version)
                before_schema = arrow_schema_signature(dataset.schema)
                columns, missing = validate_repair_columns(dataset, requested_columns, args.fail_on_missing_columns)
                if not columns:
                    result = {"table": table_name, "checked": 0, "would_repair": 0, "missing_columns": missing}
                    report.append(result)
                    log_event("lance_repair_quoted_json_strings_table_skipped", **result)
                    continue
                checked, repaired = count_quoted_json_string_values(conn, table_ref, columns, args.limit if not args.apply else None)
                result: JsonMap = {
                    "table": table_name,
                    "checked": checked,
                    "would_repair": repaired,
                    "missing_columns": missing,
                    "sampled": bool(not args.apply and args.limit is not None),
                    "sample_limit": args.limit if not args.apply else None,
                    "old_version": old_version,
                }
                if args.apply:
                    source_count = conn.execute(f"SELECT count(*) FROM {table_ref}").fetchone()[0]
                    select_sql = build_repair_quoted_json_strings_select_sql(conn, table_ref, set(columns))
                    temp_table = quote_identifier(f"repair_json_strings_{table_name}")
                    overwritten = False
                    try:
                        conn.execute(f"CREATE OR REPLACE TEMP TABLE {temp_table} AS {select_sql}")
                        temp_count = conn.execute(f"SELECT count(*) FROM {temp_table}").fetchone()[0]
                        if source_count != temp_count:
                            raise RuntimeError(f"repair count mismatch before overwrite: source={source_count} temp={temp_count}")
                        conn.execute(f"COPY {temp_table} TO {table_ref} (FORMAT lance, MODE 'overwrite')")
                        overwritten = True
                        target_count = conn.execute(f"SELECT count(*) FROM {table_ref}").fetchone()[0]
                        after_dataset = lance.dataset(dataset_uri)
                        after_schema = arrow_schema_signature(after_dataset.schema)
                        if source_count != target_count:
                            raise RuntimeError(f"repair count mismatch after overwrite: source={source_count} target={target_count}")
                        if before_schema != after_schema:
                            raise RuntimeError(f"repair schema changed after overwrite: before={before_schema} after={after_schema}")
                    except Exception:
                        if overwritten:
                            restore_lance_version(dataset_uri, old_version)
                        raise
                    result["source_count"] = source_count
                    result["target_count"] = target_count
                    result["new_version"] = int(lance.dataset(dataset_uri).version)
                    if args.create_indexes:
                        created, skipped = create_indexes_for_columns(conn, table_name, table_ref, repair_index_columns(args, table_name))
                        result["generated_indexes"] = created
                        result["index_skipped"] = skipped
                report.append(result)
                completed += 1
                log_event("lance_repair_quoted_json_strings_table_completed", **result)
            except Exception as exc:  # noqa: BLE001
                failure = {"table": table_name, "error": repr(exc)}
                failures.append(failure)
                log_event("lance_repair_quoted_json_strings_table_failed", **failure)
                if args.stop_on_failure:
                    break
    finally:
        if conn is not None:
            conn.close()
        if state is not None and lease is not None:
            release_lease(state, lease)
    log_event("lance_repair_quoted_json_strings_finished", completed=completed, failed=len(failures), apply=args.apply)
    print(json.dumps({"dry_run": not args.apply, "tables": report}, indent=2, sort_keys=True))
    if failures:
        print(json.dumps({"failures": failures}, indent=2, sort_keys=True), file=sys.stderr)
        raise SystemExit(1)


def parse_env_pairs(values: Iterable[str]) -> list[dict[str, str]]:
    env: list[dict[str, str]] = []
    for raw in values:
        if "=" not in raw:
            raise SystemExit(f"--env must be KEY=VALUE, got {raw!r}")
        key, value = raw.split("=", 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise SystemExit(f"Invalid env var name: {key}")
        env.append({"name": key, "value": value})
    return env


def append_missing_env(env: list[dict[str, str]], defaults: dict[str, str]) -> None:
    existing = {item["name"] for item in env}
    env.extend({"name": key, "value": value} for key, value in defaults.items() if key not in existing)


def parse_table_specs(args: argparse.Namespace) -> deque[tuple[str, str]]:
    raw_tables: list[str] = []
    if args.tables:
        raw_tables.extend(part.strip() for part in args.tables.split(",") if part.strip())
    if args.tables_file:
        with open(args.tables_file, encoding="utf-8") as handle:
            raw_tables.extend(line.strip() for line in handle if line.strip() and not line.strip().startswith("#"))
    if not raw_tables:
        raise SystemExit("--tables or --tables-file is required")

    specs: deque[tuple[str, str]] = deque()
    for raw in raw_tables:
        source, _, target = raw.partition("=")
        source = source.strip()
        target = target.strip() or source_to_physical_table(source)
        specs.append((source, target))
    return specs


def network_configuration(args: argparse.Namespace) -> JsonMap:
    subnets = [item.strip() for item in args.subnets.split(",") if item.strip()]
    security_groups = [item.strip() for item in args.security_groups.split(",") if item.strip()]
    if not subnets or not security_groups:
        raise SystemExit("--subnets and --security-groups are required")
    return {
        "awsvpcConfiguration": {
            "subnets": subnets,
            "securityGroups": security_groups,
            "assignPublicIp": "ENABLED" if args.assign_public_ip else "DISABLED",
        }
    }


def run_task(
    ecs: Any,
    args: argparse.Namespace,
    source: str,
    target: str,
    command: list[str] | None = None,
) -> str:
    env = parse_env_pairs(args.env)
    if command and command[0] == "create-lance-indexes":
        append_missing_env(env, LANCE_SCALAR_INDEX_ENV)
    env.extend(
        [
            {"name": "TABLE_NAME", "value": source},
            {"name": "LANCE_TABLE_NAME", "value": target},
        ]
    )
    if command is None or command[0] == "migrate-table":
        env.append({"name": "ICEBERG_TABLE_NAME", "value": source})
    kwargs: JsonMap = {
        "cluster": args.cluster,
        "taskDefinition": args.task_definition,
        "networkConfiguration": network_configuration(args),
        "overrides": {
            "containerOverrides": [
                {
                    "name": args.container_name,
                    "environment": env,
                    **({"command": command} if command else {}),
                }
            ]
        },
        "startedBy": args.started_by,
    }
    if args.capacity_provider:
        kwargs["capacityProviderStrategy"] = [{"capacityProvider": args.capacity_provider, "weight": 1}]
    else:
        kwargs["launchType"] = "FARGATE"
    response = ecs.run_task(**kwargs)
    if response.get("failures"):
        raise RuntimeError(f"run_task failed for {source}: {response['failures']}")
    tasks = response.get("tasks") or []
    if not tasks:
        raise RuntimeError(f"run_task returned no task for {source}")
    return tasks[0]["taskArn"]


def task_exit(task: JsonMap, container_name: str) -> tuple[bool, int | None, str | None]:
    containers = task.get("containers") or []
    selected = next((container for container in containers if container.get("name") == container_name), containers[0] if containers else {})
    exit_code = selected.get("exitCode")
    reason = selected.get("reason") or task.get("stoppedReason")
    ok = task.get("lastStatus") == "STOPPED" and exit_code == 0
    return ok, exit_code, reason


def run_parallel(args: argparse.Namespace) -> None:
    import boto3

    args.cluster = env_or_arg(args, "cluster", "ECS_CLUSTER")
    args.task_definition = env_or_arg(args, "task_definition", "LANCE_MIGRATION_TASK_DEFINITION")
    pending = parse_table_specs(args)
    ecs = boto3.client("ecs", region_name=args.aws_region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"))
    running: dict[str, tuple[str, str]] = {}
    failures: list[JsonMap] = []
    completed = 0

    while pending or running:
        while pending and len(running) < args.concurrency:
            source, target = pending.popleft()
            task_arn = run_task(ecs, args, source, target)
            running[task_arn] = (source, target)
            log_event("lance_migration_task_started", table=source, target_table=target, task_arn=task_arn)

        if not running:
            continue

        time.sleep(args.poll_seconds)
        task_arns = list(running)
        response = ecs.describe_tasks(cluster=args.cluster, tasks=task_arns)
        if response.get("failures"):
            for failure in response["failures"]:
                arn = failure.get("arn")
                source, target = running.pop(arn, ("unknown", "unknown"))
                failures.append({"table": source, "target_table": target, "task_arn": arn, "failure": failure})
                log_event("lance_migration_task_describe_failed", table=source, task_arn=arn, failure=failure)
        for task in response.get("tasks", []):
            if task.get("lastStatus") != "STOPPED":
                continue
            arn = task["taskArn"]
            source, target = running.pop(arn)
            ok, exit_code, reason = task_exit(task, args.container_name)
            if ok:
                completed += 1
                log_event("lance_migration_task_completed", table=source, target_table=target, task_arn=arn)
            else:
                failure = {"table": source, "target_table": target, "task_arn": arn, "exit_code": exit_code, "reason": reason}
                failures.append(failure)
                log_event("lance_migration_task_failed", **failure)
                if args.stop_on_failure:
                    pending.clear()

    log_event("lance_migration_parallel_finished", completed=completed, failed=len(failures))
    if failures:
        print(json.dumps({"failures": failures}, indent=2, sort_keys=True), file=sys.stderr)
        raise SystemExit(1)


def run_generated_parallel(args: argparse.Namespace) -> None:
    import boto3

    args.cluster = env_or_arg(args, "cluster", "ECS_CLUSTER")
    args.task_definition = env_or_arg(args, "task_definition", "LANCE_MIGRATION_TASK_DEFINITION")
    args.lance_bucket = args.lance_bucket or os.environ.get("LANCE_BUCKET")
    args.aws_region = args.aws_region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    pending = deque((table, table) for table in list_lance_tables(args))
    ecs = boto3.client("ecs", region_name=args.aws_region)
    running: dict[str, tuple[str, str]] = {}
    failures: list[JsonMap] = []
    completed = 0

    while pending or running:
        while pending and len(running) < args.concurrency:
            source, target = pending.popleft()
            command = [
                "backfill-generated-columns",
                "--tables",
                target,
                "--lance-prefix",
                args.lance_prefix,
                "--duckdb-path",
                args.duckdb_path,
            ]
            if args.create_indexes:
                command.append("--create-indexes")
            if args.lance_bucket:
                command.extend(["--lance-bucket", args.lance_bucket])
            if args.lance_root_uri:
                command.extend(["--lance-root-uri", args.lance_root_uri])
            task_arn = run_task(ecs, args, source, target, command=command)
            running[task_arn] = (source, target)
            log_event("lance_generated_backfill_task_started", table=source, target_table=target, task_arn=task_arn)

        if not running:
            continue

        time.sleep(args.poll_seconds)
        task_arns = list(running)
        response = ecs.describe_tasks(cluster=args.cluster, tasks=task_arns)
        if response.get("failures"):
            for failure in response["failures"]:
                arn = failure.get("arn")
                source, target = running.pop(arn, ("unknown", "unknown"))
                failures.append({"table": source, "target_table": target, "task_arn": arn, "failure": failure})
                log_event("lance_generated_backfill_task_describe_failed", table=source, task_arn=arn, failure=failure)
        for task in response.get("tasks", []):
            if task.get("lastStatus") != "STOPPED":
                continue
            arn = task["taskArn"]
            source, target = running.pop(arn)
            ok, exit_code, reason = task_exit(task, args.container_name)
            if ok:
                completed += 1
                log_event("lance_generated_backfill_task_completed", table=source, target_table=target, task_arn=arn)
            else:
                failure = {"table": source, "target_table": target, "task_arn": arn, "exit_code": exit_code, "reason": reason}
                failures.append(failure)
                log_event("lance_generated_backfill_task_failed", **failure)
                if args.stop_on_failure:
                    pending.clear()

    log_event("lance_generated_backfill_parallel_finished", completed=completed, failed=len(failures))
    if failures:
        print(json.dumps({"failures": failures}, indent=2, sort_keys=True), file=sys.stderr)
        raise SystemExit(1)


def run_drop_current_parallel(args: argparse.Namespace) -> None:
    import boto3

    args.cluster = env_or_arg(args, "cluster", "ECS_CLUSTER")
    args.task_definition = env_or_arg(args, "task_definition", "LANCE_MIGRATION_TASK_DEFINITION")
    args.lance_bucket = args.lance_bucket or os.environ.get("LANCE_BUCKET")
    args.aws_region = args.aws_region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    pending = deque((table, table) for table in list_lance_tables(args))
    ecs = boto3.client("ecs", region_name=args.aws_region)
    running: dict[str, tuple[str, str]] = {}
    failures: list[JsonMap] = []
    completed = 0

    while pending or running:
        while pending and len(running) < args.concurrency:
            source, target = pending.popleft()
            command = [
                "drop-legacy-current",
                "--tables",
                target,
                "--lance-prefix",
                args.lance_prefix,
                "--duckdb-path",
                args.duckdb_path,
            ]
            if args.lance_bucket:
                command.extend(["--lance-bucket", args.lance_bucket])
            if args.lance_root_uri:
                command.extend(["--lance-root-uri", args.lance_root_uri])
            task_arn = run_task(ecs, args, source, target, command=command)
            running[task_arn] = (source, target)
            log_event("lance_drop_legacy_current_task_started", table=source, target_table=target, task_arn=task_arn)

        if not running:
            continue

        time.sleep(args.poll_seconds)
        response = ecs.describe_tasks(cluster=args.cluster, tasks=list(running))
        if response.get("failures"):
            for failure in response["failures"]:
                arn = failure.get("arn")
                source, target = running.pop(arn, ("unknown", "unknown"))
                failures.append({"table": source, "target_table": target, "task_arn": arn, "failure": failure})
                log_event("lance_drop_legacy_current_task_describe_failed", table=source, task_arn=arn, failure=failure)
        for task in response.get("tasks", []):
            if task.get("lastStatus") != "STOPPED":
                continue
            arn = task["taskArn"]
            source, target = running.pop(arn)
            ok, exit_code, reason = task_exit(task, args.container_name)
            if ok:
                completed += 1
                log_event("lance_drop_legacy_current_task_completed", table=source, target_table=target, task_arn=arn)
            else:
                failure = {"table": source, "target_table": target, "task_arn": arn, "exit_code": exit_code, "reason": reason}
                failures.append(failure)
                log_event("lance_drop_legacy_current_task_failed", **failure)
                if args.stop_on_failure:
                    pending.clear()

    log_event("lance_drop_legacy_current_parallel_finished", completed=completed, failed=len(failures))
    if failures:
        print(json.dumps({"failures": failures}, indent=2, sort_keys=True), file=sys.stderr)
        raise SystemExit(1)


def run_create_indexes_parallel(args: argparse.Namespace) -> None:
    import boto3

    args.cluster = env_or_arg(args, "cluster", "ECS_CLUSTER")
    args.task_definition = env_or_arg(args, "task_definition", "LANCE_MIGRATION_TASK_DEFINITION")
    args.lance_bucket = args.lance_bucket or os.environ.get("LANCE_BUCKET")
    args.aws_region = args.aws_region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    pending = deque((table, table) for table in list_lance_tables(args))
    ecs = boto3.client("ecs", region_name=args.aws_region)
    running: dict[str, tuple[str, str]] = {}
    failures: list[JsonMap] = []
    completed = 0

    while pending or running:
        while pending and len(running) < args.concurrency:
            source, target = pending.popleft()
            command = [
                "create-lance-indexes",
                "--tables",
                target,
                "--lance-prefix",
                args.lance_prefix,
                "--duckdb-path",
                args.duckdb_path,
            ]
            command.append("--generated-indexes" if args.generated_indexes else "--no-generated-indexes")
            command.append("--table-config-indexes" if args.table_config_indexes else "--no-table-config-indexes")
            if args.lance_bucket:
                command.extend(["--lance-bucket", args.lance_bucket])
            if args.lance_root_uri:
                command.extend(["--lance-root-uri", args.lance_root_uri])
            if args.table_config_bucket:
                command.extend(["--table-config-bucket", args.table_config_bucket])
            if args.table_config_prefix:
                command.extend(["--table-config-prefix", args.table_config_prefix])
            if args.table_config_state_bucket:
                command.extend(["--table-config-state-bucket", args.table_config_state_bucket])
            if args.table_config_state_prefix:
                command.extend(["--table-config-state-prefix", args.table_config_state_prefix])
            task_arn = run_task(ecs, args, source, target, command=command)
            running[task_arn] = (source, target)
            log_event("lance_index_creation_task_started", table=source, target_table=target, task_arn=task_arn)

        if not running:
            continue

        time.sleep(args.poll_seconds)
        response = ecs.describe_tasks(cluster=args.cluster, tasks=list(running))
        if response.get("failures"):
            for failure in response["failures"]:
                arn = failure.get("arn")
                source, target = running.pop(arn, ("unknown", "unknown"))
                failures.append({"table": source, "target_table": target, "task_arn": arn, "failure": failure})
                log_event("lance_index_creation_task_describe_failed", table=source, task_arn=arn, failure=failure)
        for task in response.get("tasks", []):
            if task.get("lastStatus") != "STOPPED":
                continue
            arn = task["taskArn"]
            source, target = running.pop(arn)
            ok, exit_code, reason = task_exit(task, args.container_name)
            if ok:
                completed += 1
                log_event("lance_index_creation_task_completed", table=source, target_table=target, task_arn=arn)
            else:
                failure = {"table": source, "target_table": target, "task_arn": arn, "exit_code": exit_code, "reason": reason}
                failures.append(failure)
                log_event("lance_index_creation_task_failed", **failure)
                if args.stop_on_failure:
                    pending.clear()

    log_event("lance_index_creation_parallel_finished", completed=completed, failed=len(failures))
    if failures:
        print(json.dumps({"failures": failures}, indent=2, sort_keys=True), file=sys.stderr)
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sirlance", description="container loader")
    sub = parser.add_subparsers(dest="command", required=True)

    migrate = sub.add_parser("migrate-table")
    migrate.add_argument("--table")
    migrate.add_argument("--source-table")
    migrate.add_argument("--output-table")
    migrate.add_argument("--namespace")
    migrate.add_argument("--catalog-alias")
    migrate.add_argument("--s3tables-warehouse-arn")
    migrate.add_argument("--lance-bucket")
    migrate.add_argument("--lance-prefix", default=os.environ.get("LANCE_PREFIX", "tables"))
    migrate.add_argument("--lance-uri")
    migrate.add_argument("--lance-scope")
    migrate.add_argument("--mode", choices=["overwrite", "append"], default=os.environ.get("LANCE_WRITE_MODE", "overwrite"))
    migrate.add_argument("--where")
    migrate.add_argument("--order-by")
    migrate.add_argument("--max-rows", type=int)
    migrate.add_argument("--verify-count", action=argparse.BooleanOptionalAction, default=os.environ.get("LANCE_VERIFY_COUNT", "true").lower() == "true")
    migrate.add_argument(
        "--create-generated-indexes",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("LANCE_CREATE_GENERATED_INDEXES", "true").lower() == "true",
    )
    migrate.add_argument("--table-config-indexes", action=argparse.BooleanOptionalAction, default=os.environ.get("LANCE_TABLE_CONFIG_INDEXES", "true").lower() not in {"0", "false", "no"})
    migrate.add_argument("--table-config-dir")
    migrate.add_argument("--table-config-bucket")
    migrate.add_argument("--table-config-prefix", default=os.environ.get("LANCE_TABLE_CONFIG_PREFIX", "config/lance/tables"))
    migrate.add_argument("--table-config-state-bucket")
    migrate.add_argument("--table-config-state-prefix", default=os.environ.get("LANCE_TABLE_CONFIG_STATE_PREFIX", DEFAULT_LANCE_TABLE_CONFIG_STATE_PREFIX))
    migrate.add_argument("--duckdb-path")
    migrate.add_argument("--threads", type=int, default=int(os.environ.get("DUCKDB_THREADS", "0")) or None)
    migrate.add_argument("--memory-limit", default=os.environ.get("DUCKDB_MEMORY_LIMIT"))
    migrate.add_argument("--temp-directory", default=os.environ.get("DUCKDB_TEMP_DIRECTORY"))
    migrate.add_argument("--aws-region")
    migrate.set_defaults(func=run_migrate_table)

    incremental = sub.add_parser("incremental-once")
    incremental.add_argument("--convex-url")
    incremental.add_argument("--convex-deploy-key")
    incremental.add_argument("--state-bucket")
    incremental.add_argument("--state-prefix")
    incremental.add_argument("--cursor-key", default=os.environ.get("LANCE_INCREMENTAL_CURSOR_KEY", "incremental/cursor.json"))
    incremental.add_argument("--schema-key", default=os.environ.get("LANCE_INCREMENTAL_SCHEMA_KEY", "incremental/schema.json"))
    incremental.add_argument("--schema-refresh-seconds", type=int, default=int(os.environ.get("LANCE_INCREMENTAL_SCHEMA_REFRESH_SECONDS", "600")))
    incremental.add_argument("--reconcile-schema", action=argparse.BooleanOptionalAction, default=os.environ.get("LANCE_INCREMENTAL_RECONCILE_SCHEMA", "true").lower() not in {"0", "false", "no"})
    incremental.add_argument(
        "--reconcile-existing-tables-on-schema-refresh",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("LANCE_INCREMENTAL_RECONCILE_EXISTING_TABLES_ON_SCHEMA_REFRESH", "true").lower() not in {"0", "false", "no"},
    )
    incremental.add_argument("--unknown-table-policy", choices=["fail", "create", "skip"], default=os.environ.get("LANCE_INCREMENTAL_UNKNOWN_TABLE_POLICY", "fail"))
    incremental.add_argument("--auto-recreate-empty-schema-drift", action=argparse.BooleanOptionalAction, default=os.environ.get("LANCE_INCREMENTAL_AUTO_RECREATE_EMPTY_SCHEMA_DRIFT", "true").lower() not in {"0", "false", "no"})
    incremental.add_argument("--table-config-indexes", action=argparse.BooleanOptionalAction, default=os.environ.get("LANCE_TABLE_CONFIG_INDEXES", "true").lower() not in {"0", "false", "no"})
    incremental.add_argument("--table-config-dir")
    incremental.add_argument("--table-config-bucket")
    incremental.add_argument("--table-config-prefix", default=os.environ.get("LANCE_TABLE_CONFIG_PREFIX", "config/lance/tables"))
    incremental.add_argument("--table-config-state-bucket")
    incremental.add_argument("--table-config-state-prefix", default=os.environ.get("LANCE_TABLE_CONFIG_STATE_PREFIX", DEFAULT_LANCE_TABLE_CONFIG_STATE_PREFIX))
    incremental.add_argument("--lance-bucket")
    incremental.add_argument("--lance-prefix", default=os.environ.get("LANCE_PREFIX", "tables"))
    incremental.add_argument("--lance-root-uri")
    incremental.add_argument("--lance-scope")
    incremental.add_argument("--catalog-alias")
    incremental.add_argument("--max-pages-per-sync", type=int, default=int(os.environ.get("LANCE_INCREMENTAL_MAX_PAGES_PER_SYNC", "100")))
    incremental.add_argument("--lock-ttl-seconds", type=int, default=int(os.environ.get("LANCE_INCREMENTAL_LOCK_TTL_SECONDS", "300")))
    incremental.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    incremental.add_argument("--duckdb-path", default=os.environ.get("LANCE_INCREMENTAL_DUCKDB_PATH", "/tmp/lance-incremental.duckdb"))
    incremental.add_argument("--threads", type=int, default=int(os.environ.get("DUCKDB_THREADS", "0")) or None)
    incremental.add_argument("--memory-limit", default=os.environ.get("DUCKDB_MEMORY_LIMIT"))
    incremental.add_argument("--temp-directory", default=os.environ.get("DUCKDB_TEMP_DIRECTORY"))
    incremental.add_argument("--aws-region")
    incremental.set_defaults(func=run_incremental_once)

    incremental_loop = sub.add_parser("incremental-loop")
    incremental_loop.add_argument("--convex-url")
    incremental_loop.add_argument("--convex-deploy-key")
    incremental_loop.add_argument("--state-bucket")
    incremental_loop.add_argument("--state-prefix")
    incremental_loop.add_argument("--cursor-key", default=os.environ.get("LANCE_INCREMENTAL_CURSOR_KEY", "incremental/cursor.json"))
    incremental_loop.add_argument("--schema-key", default=os.environ.get("LANCE_INCREMENTAL_SCHEMA_KEY", "incremental/schema.json"))
    incremental_loop.add_argument("--schema-refresh-seconds", type=int, default=int(os.environ.get("LANCE_INCREMENTAL_SCHEMA_REFRESH_SECONDS", "600")))
    incremental_loop.add_argument("--reconcile-schema", action=argparse.BooleanOptionalAction, default=os.environ.get("LANCE_INCREMENTAL_RECONCILE_SCHEMA", "true").lower() not in {"0", "false", "no"})
    incremental_loop.add_argument(
        "--reconcile-existing-tables-on-schema-refresh",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("LANCE_INCREMENTAL_RECONCILE_EXISTING_TABLES_ON_SCHEMA_REFRESH", "true").lower() not in {"0", "false", "no"},
    )
    incremental_loop.add_argument("--unknown-table-policy", choices=["fail", "create", "skip"], default=os.environ.get("LANCE_INCREMENTAL_UNKNOWN_TABLE_POLICY", "fail"))
    incremental_loop.add_argument("--auto-recreate-empty-schema-drift", action=argparse.BooleanOptionalAction, default=os.environ.get("LANCE_INCREMENTAL_AUTO_RECREATE_EMPTY_SCHEMA_DRIFT", "true").lower() not in {"0", "false", "no"})
    incremental_loop.add_argument("--table-config-indexes", action=argparse.BooleanOptionalAction, default=os.environ.get("LANCE_TABLE_CONFIG_INDEXES", "true").lower() not in {"0", "false", "no"})
    incremental_loop.add_argument("--table-config-dir")
    incremental_loop.add_argument("--table-config-bucket")
    incremental_loop.add_argument("--table-config-prefix", default=os.environ.get("LANCE_TABLE_CONFIG_PREFIX", "config/lance/tables"))
    incremental_loop.add_argument("--table-config-state-bucket")
    incremental_loop.add_argument("--table-config-state-prefix", default=os.environ.get("LANCE_TABLE_CONFIG_STATE_PREFIX", DEFAULT_LANCE_TABLE_CONFIG_STATE_PREFIX))
    incremental_loop.add_argument("--lance-bucket")
    incremental_loop.add_argument("--lance-prefix", default=os.environ.get("LANCE_PREFIX", "tables"))
    incremental_loop.add_argument("--lance-root-uri")
    incremental_loop.add_argument("--lance-scope")
    incremental_loop.add_argument("--catalog-alias")
    incremental_loop.add_argument("--max-pages-per-sync", type=int, default=int(os.environ.get("LANCE_INCREMENTAL_MAX_PAGES_PER_SYNC", "100")))
    incremental_loop.add_argument("--sleep-seconds", type=int, default=int(os.environ.get("LANCE_INCREMENTAL_SLEEP_SECONDS", "20")))
    incremental_loop.add_argument("--optimize-touched-indices", action=argparse.BooleanOptionalAction, default=os.environ.get("LANCE_INCREMENTAL_OPTIMIZE_TOUCHED_INDICES", "true").lower() not in {"0", "false", "no"})
    incremental_loop.add_argument("--optimize-indices-pages", type=int, default=int(os.environ.get("LANCE_INCREMENTAL_OPTIMIZE_INDICES_PAGES", "50")))
    incremental_loop.add_argument("--optimize-indices-rows", type=int, default=int(os.environ.get("LANCE_INCREMENTAL_OPTIMIZE_INDICES_ROWS", "5000")))
    incremental_loop.add_argument("--optimize-indices-seconds", type=int, default=int(os.environ.get("LANCE_INCREMENTAL_OPTIMIZE_INDICES_SECONDS", "900")))
    incremental_loop.add_argument("--idle-maintenance", action=argparse.BooleanOptionalAction, default=os.environ.get("LANCE_INCREMENTAL_IDLE_MAINTENANCE", "true").lower() not in {"0", "false", "no"})
    incremental_loop.add_argument("--idle-maintenance-max-rows", type=int, default=int(os.environ.get("LANCE_INCREMENTAL_IDLE_MAINTENANCE_MAX_ROWS", "0")))
    incremental_loop.add_argument("--idle-maintenance-max-pages", type=int, default=int(os.environ.get("LANCE_INCREMENTAL_IDLE_MAINTENANCE_MAX_PAGES", "1")))
    incremental_loop.add_argument("--maintenance-state-key", default=os.environ.get("LANCE_INCREMENTAL_MAINTENANCE_STATE_KEY", "incremental/maintenance_state.json"))
    incremental_loop.add_argument("--maintenance-audit-key-prefix", default=os.environ.get("LANCE_INCREMENTAL_MAINTENANCE_AUDIT_KEY_PREFIX", "incremental/maintenance-audit"))
    incremental_loop.add_argument("--maintenance-tables")
    incremental_loop.add_argument("--maintenance-tables-file")
    incremental_loop.add_argument("--maintenance-optimize-indices-seconds", type=int, default=int(os.environ.get("LANCE_MAINTENANCE_OPTIMIZE_INDICES_SECONDS", "3600")))
    incremental_loop.add_argument("--maintenance-compact-files-seconds", type=int, default=int(os.environ.get("LANCE_MAINTENANCE_COMPACT_FILES_SECONDS", "86400")))
    incremental_loop.add_argument("--maintenance-cleanup-old-versions-seconds", type=int, default=int(os.environ.get("LANCE_MAINTENANCE_CLEANUP_OLD_VERSIONS_SECONDS", "604800")))
    incremental_loop.add_argument("--maintenance-cleanup-older-than-seconds", type=int, default=int(os.environ.get("LANCE_MAINTENANCE_CLEANUP_OLDER_THAN_SECONDS", "259200")))
    incremental_loop.add_argument("--maintenance-retain-versions", type=int, default=int(os.environ.get("LANCE_MAINTENANCE_RETAIN_VERSIONS", "500")))
    incremental_loop.add_argument("--maintenance-compact-threads", type=int, default=int(os.environ.get("LANCE_MAINTENANCE_COMPACT_THREADS", "1")))
    incremental_loop.add_argument("--maintenance-heavy-window-start-utc", type=int, default=int(os.environ.get("LANCE_MAINTENANCE_HEAVY_WINDOW_START_UTC", "7")))
    incremental_loop.add_argument("--maintenance-heavy-window-end-utc", type=int, default=int(os.environ.get("LANCE_MAINTENANCE_HEAVY_WINDOW_END_UTC", "12")))
    incremental_loop.add_argument("--lock-ttl-seconds", type=int, default=int(os.environ.get("LANCE_INCREMENTAL_LOCK_TTL_SECONDS", "300")))
    incremental_loop.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    incremental_loop.add_argument("--duckdb-path", default=os.environ.get("LANCE_INCREMENTAL_DUCKDB_PATH", "/tmp/lance-incremental.duckdb"))
    incremental_loop.add_argument("--threads", type=int, default=int(os.environ.get("DUCKDB_THREADS", "0")) or None)
    incremental_loop.add_argument("--memory-limit", default=os.environ.get("DUCKDB_MEMORY_LIMIT"))
    incremental_loop.add_argument("--temp-directory", default=os.environ.get("DUCKDB_TEMP_DIRECTORY"))
    incremental_loop.add_argument("--aws-region")
    incremental_loop.set_defaults(func=run_incremental_loop)

    generated = sub.add_parser("backfill-generated-columns")
    generated.add_argument("--tables")
    generated.add_argument("--tables-file")
    generated.add_argument("--max-tables", type=int)
    generated.add_argument("--lance-bucket")
    generated.add_argument("--lance-prefix", default=os.environ.get("LANCE_PREFIX", "tables"))
    generated.add_argument("--lance-root-uri")
    generated.add_argument("--lance-scope")
    generated.add_argument("--catalog-alias")
    generated.add_argument("--create-indexes", action=argparse.BooleanOptionalAction, default=False)
    generated.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    generated.add_argument("--stop-on-failure", action=argparse.BooleanOptionalAction, default=True)
    generated.add_argument("--duckdb-path")
    generated.add_argument("--threads", type=int, default=int(os.environ.get("DUCKDB_THREADS", "0")) or None)
    generated.add_argument("--memory-limit", default=os.environ.get("DUCKDB_MEMORY_LIMIT"))
    generated.add_argument("--temp-directory", default=os.environ.get("DUCKDB_TEMP_DIRECTORY"))
    generated.add_argument("--aws-region")
    generated.set_defaults(func=run_backfill_generated_columns)

    drop_current = sub.add_parser("drop-legacy-current")
    drop_current.add_argument("--tables")
    drop_current.add_argument("--tables-file")
    drop_current.add_argument("--max-tables", type=int)
    drop_current.add_argument("--lance-bucket")
    drop_current.add_argument("--lance-prefix", default=os.environ.get("LANCE_PREFIX", "tables"))
    drop_current.add_argument("--lance-root-uri")
    drop_current.add_argument("--lance-scope")
    drop_current.add_argument("--catalog-alias")
    drop_current.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    drop_current.add_argument("--stop-on-failure", action=argparse.BooleanOptionalAction, default=True)
    drop_current.add_argument("--duckdb-path", default=os.environ.get("LANCE_DROP_CURRENT_DUCKDB_PATH", "/tmp/lance-drop-current.duckdb"))
    drop_current.add_argument("--threads", type=int, default=int(os.environ.get("DUCKDB_THREADS", "0")) or None)
    drop_current.add_argument("--memory-limit", default=os.environ.get("DUCKDB_MEMORY_LIMIT"))
    drop_current.add_argument("--temp-directory", default=os.environ.get("DUCKDB_TEMP_DIRECTORY"))
    drop_current.add_argument("--aws-region")
    drop_current.set_defaults(func=run_drop_legacy_current)

    discover_repairs = sub.add_parser("discover-quoted-json-string-repairs")
    discover_repairs.add_argument("--convex-url")
    discover_repairs.add_argument("--convex-deploy-key")
    discover_repairs.add_argument("--schema-json")
    discover_repairs.add_argument("--tables")
    discover_repairs.add_argument("--tables-file")
    discover_repairs.add_argument("--max-tables", type=int)
    discover_repairs.add_argument("--limit", type=int, default=10000)
    discover_repairs.add_argument("--include-skipped", action=argparse.BooleanOptionalAction, default=False)
    discover_repairs.add_argument("--lance-bucket")
    discover_repairs.add_argument("--lance-prefix", default=os.environ.get("LANCE_PREFIX", "tables"))
    discover_repairs.add_argument("--lance-root-uri")
    discover_repairs.add_argument("--lance-scope")
    discover_repairs.add_argument("--stop-on-failure", action=argparse.BooleanOptionalAction, default=True)
    discover_repairs.add_argument("--duckdb-path", default=os.environ.get("LANCE_REPAIR_DISCOVERY_DUCKDB_PATH", ":memory:"))
    discover_repairs.add_argument("--threads", type=int, default=int(os.environ.get("DUCKDB_THREADS", "0")) or None)
    discover_repairs.add_argument("--memory-limit", default=os.environ.get("DUCKDB_MEMORY_LIMIT"))
    discover_repairs.add_argument("--temp-directory", default=os.environ.get("DUCKDB_TEMP_DIRECTORY"))
    discover_repairs.add_argument("--aws-region")
    discover_repairs.set_defaults(func=run_discover_quoted_json_string_repairs)

    repair_json_strings = sub.add_parser("repair-quoted-json-strings")
    repair_json_strings.add_argument("--tables")
    repair_json_strings.add_argument("--tables-file")
    repair_json_strings.add_argument("--max-tables", type=int)
    repair_json_strings.add_argument("--columns", required=True)
    repair_json_strings.add_argument("--limit", type=int, default=1000)
    repair_json_strings.add_argument("--lance-bucket")
    repair_json_strings.add_argument("--lance-prefix", default=os.environ.get("LANCE_PREFIX", "tables"))
    repair_json_strings.add_argument("--lance-root-uri")
    repair_json_strings.add_argument("--lance-scope")
    repair_json_strings.add_argument("--apply", action=argparse.BooleanOptionalAction, default=False)
    repair_json_strings.add_argument("--create-indexes", action=argparse.BooleanOptionalAction, default=True)
    repair_json_strings.add_argument("--generated-indexes", action=argparse.BooleanOptionalAction, default=True)
    repair_json_strings.add_argument("--hot-indexes", action=argparse.BooleanOptionalAction, default=True)
    repair_json_strings.add_argument("--table-config-indexes", action=argparse.BooleanOptionalAction, default=os.environ.get("LANCE_TABLE_CONFIG_INDEXES", "true").lower() not in {"0", "false", "no"})
    repair_json_strings.add_argument("--table-config-dir")
    repair_json_strings.add_argument("--table-config-bucket")
    repair_json_strings.add_argument("--table-config-prefix", default=os.environ.get("LANCE_TABLE_CONFIG_PREFIX", "config/lance/tables"))
    repair_json_strings.add_argument("--table-config-state-bucket")
    repair_json_strings.add_argument("--table-config-state-prefix", default=os.environ.get("LANCE_TABLE_CONFIG_STATE_PREFIX", DEFAULT_LANCE_TABLE_CONFIG_STATE_PREFIX))
    repair_json_strings.add_argument("--fail-on-missing-columns", action=argparse.BooleanOptionalAction, default=True)
    repair_json_strings.add_argument("--use-lock", action=argparse.BooleanOptionalAction, default=True)
    repair_json_strings.add_argument("--state-bucket")
    repair_json_strings.add_argument("--state-prefix")
    repair_json_strings.add_argument("--lock-ttl-seconds", type=int, default=int(os.environ.get("LANCE_REPAIR_LOCK_TTL_SECONDS", "3600")))
    repair_json_strings.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    repair_json_strings.add_argument("--stop-on-failure", action=argparse.BooleanOptionalAction, default=True)
    repair_json_strings.add_argument("--duckdb-path", default=os.environ.get("LANCE_REPAIR_DUCKDB_PATH", "/tmp/lance-repair.duckdb"))
    repair_json_strings.add_argument("--threads", type=int, default=int(os.environ.get("DUCKDB_THREADS", "0")) or None)
    repair_json_strings.add_argument("--memory-limit", default=os.environ.get("DUCKDB_MEMORY_LIMIT"))
    repair_json_strings.add_argument("--temp-directory", default=os.environ.get("DUCKDB_TEMP_DIRECTORY"))
    repair_json_strings.add_argument("--aws-region")
    repair_json_strings.set_defaults(func=run_repair_quoted_json_strings)

    create_indexes = sub.add_parser("create-lance-indexes")
    create_indexes.add_argument("--tables")
    create_indexes.add_argument("--tables-file")
    create_indexes.add_argument("--max-tables", type=int)
    create_indexes.add_argument("--lance-bucket")
    create_indexes.add_argument("--lance-prefix", default=os.environ.get("LANCE_PREFIX", "tables"))
    create_indexes.add_argument("--lance-root-uri")
    create_indexes.add_argument("--lance-scope")
    create_indexes.add_argument("--generated-indexes", action=argparse.BooleanOptionalAction, default=True)
    create_indexes.add_argument("--hot-indexes", action=argparse.BooleanOptionalAction, default=True)
    create_indexes.add_argument("--table-config-indexes", action=argparse.BooleanOptionalAction, default=os.environ.get("LANCE_TABLE_CONFIG_INDEXES", "true").lower() not in {"0", "false", "no"})
    create_indexes.add_argument("--table-config-dir")
    create_indexes.add_argument("--table-config-bucket")
    create_indexes.add_argument("--table-config-prefix", default=os.environ.get("LANCE_TABLE_CONFIG_PREFIX", "config/lance/tables"))
    create_indexes.add_argument("--table-config-state-bucket")
    create_indexes.add_argument("--table-config-state-prefix", default=os.environ.get("LANCE_TABLE_CONFIG_STATE_PREFIX", DEFAULT_LANCE_TABLE_CONFIG_STATE_PREFIX))
    create_indexes.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    create_indexes.add_argument("--stop-on-failure", action=argparse.BooleanOptionalAction, default=True)
    create_indexes.add_argument("--duckdb-path", default=os.environ.get("LANCE_INDEX_DUCKDB_PATH", "/tmp/lance-indexes.duckdb"))
    create_indexes.add_argument("--threads", type=int, default=int(os.environ.get("DUCKDB_THREADS", "0")) or None)
    create_indexes.add_argument("--memory-limit", default=os.environ.get("DUCKDB_MEMORY_LIMIT"))
    create_indexes.add_argument("--temp-directory", default=os.environ.get("DUCKDB_TEMP_DIRECTORY"))
    create_indexes.add_argument("--aws-region")
    create_indexes.set_defaults(func=run_create_lance_indexes)

    parallel = sub.add_parser("run-parallel")
    parallel.add_argument("--tables")
    parallel.add_argument("--tables-file")
    parallel.add_argument("--concurrency", type=int, default=int(os.environ.get("LANCE_MIGRATION_CONCURRENCY", "4")))
    parallel.add_argument("--cluster")
    parallel.add_argument("--task-definition")
    parallel.add_argument("--container-name", default=os.environ.get("LANCE_MIGRATION_CONTAINER_NAME", "lance-migration"))
    parallel.add_argument("--subnets", default=os.environ.get("ECS_SUBNETS", ""))
    parallel.add_argument("--security-groups", default=os.environ.get("ECS_SECURITY_GROUPS", ""))
    parallel.add_argument("--assign-public-ip", action=argparse.BooleanOptionalAction, default=os.environ.get("ECS_ASSIGN_PUBLIC_IP", "true").lower() == "true")
    parallel.add_argument("--capacity-provider", default=os.environ.get("ECS_CAPACITY_PROVIDER"))
    parallel.add_argument("--started-by", default=os.environ.get("ECS_STARTED_BY", "lance-migration-runner"))
    parallel.add_argument("--poll-seconds", type=int, default=int(os.environ.get("LANCE_MIGRATION_POLL_SECONDS", "20")))
    parallel.add_argument("--stop-on-failure", action=argparse.BooleanOptionalAction, default=True)
    parallel.add_argument("--env", action="append", default=[])
    parallel.add_argument("--aws-region")
    parallel.set_defaults(func=run_parallel)

    generated_parallel = sub.add_parser("run-generated-parallel")
    generated_parallel.add_argument("--tables")
    generated_parallel.add_argument("--tables-file")
    generated_parallel.add_argument("--max-tables", type=int)
    generated_parallel.add_argument("--concurrency", type=int, default=int(os.environ.get("LANCE_GENERATED_BACKFILL_CONCURRENCY", "4")))
    generated_parallel.add_argument("--cluster")
    generated_parallel.add_argument("--task-definition")
    generated_parallel.add_argument("--container-name", default=os.environ.get("LANCE_MIGRATION_CONTAINER_NAME", "lance-migration"))
    generated_parallel.add_argument("--subnets", default=os.environ.get("ECS_SUBNETS", ""))
    generated_parallel.add_argument("--security-groups", default=os.environ.get("ECS_SECURITY_GROUPS", ""))
    generated_parallel.add_argument("--assign-public-ip", action=argparse.BooleanOptionalAction, default=os.environ.get("ECS_ASSIGN_PUBLIC_IP", "true").lower() == "true")
    generated_parallel.add_argument("--capacity-provider", default=os.environ.get("ECS_CAPACITY_PROVIDER"))
    generated_parallel.add_argument("--started-by", default=os.environ.get("ECS_STARTED_BY", "lance-generated-backfill-runner"))
    generated_parallel.add_argument("--poll-seconds", type=int, default=int(os.environ.get("LANCE_MIGRATION_POLL_SECONDS", "20")))
    generated_parallel.add_argument("--stop-on-failure", action=argparse.BooleanOptionalAction, default=True)
    generated_parallel.add_argument("--env", action="append", default=[])
    generated_parallel.add_argument("--lance-bucket")
    generated_parallel.add_argument("--lance-prefix", default=os.environ.get("LANCE_PREFIX", "tables"))
    generated_parallel.add_argument("--lance-root-uri")
    generated_parallel.add_argument("--create-indexes", action=argparse.BooleanOptionalAction, default=False)
    generated_parallel.add_argument("--duckdb-path", default=os.environ.get("LANCE_GENERATED_BACKFILL_DUCKDB_PATH", "/tmp/lance-generated-backfill.duckdb"))
    generated_parallel.add_argument("--aws-region")
    generated_parallel.set_defaults(func=run_generated_parallel)

    drop_current_parallel = sub.add_parser("run-drop-current-parallel")
    drop_current_parallel.add_argument("--tables")
    drop_current_parallel.add_argument("--tables-file")
    drop_current_parallel.add_argument("--max-tables", type=int)
    drop_current_parallel.add_argument("--concurrency", type=int, default=int(os.environ.get("LANCE_DROP_CURRENT_CONCURRENCY", "4")))
    drop_current_parallel.add_argument("--cluster")
    drop_current_parallel.add_argument("--task-definition")
    drop_current_parallel.add_argument("--container-name", default=os.environ.get("LANCE_MIGRATION_CONTAINER_NAME", "lance-migration"))
    drop_current_parallel.add_argument("--subnets", default=os.environ.get("ECS_SUBNETS", ""))
    drop_current_parallel.add_argument("--security-groups", default=os.environ.get("ECS_SECURITY_GROUPS", ""))
    drop_current_parallel.add_argument("--assign-public-ip", action=argparse.BooleanOptionalAction, default=os.environ.get("ECS_ASSIGN_PUBLIC_IP", "true").lower() == "true")
    drop_current_parallel.add_argument("--capacity-provider", default=os.environ.get("ECS_CAPACITY_PROVIDER"))
    drop_current_parallel.add_argument("--started-by", default=os.environ.get("ECS_STARTED_BY", "lance-drop-current-runner"))
    drop_current_parallel.add_argument("--poll-seconds", type=int, default=int(os.environ.get("LANCE_MIGRATION_POLL_SECONDS", "20")))
    drop_current_parallel.add_argument("--stop-on-failure", action=argparse.BooleanOptionalAction, default=True)
    drop_current_parallel.add_argument("--env", action="append", default=[])
    drop_current_parallel.add_argument("--lance-bucket")
    drop_current_parallel.add_argument("--lance-prefix", default=os.environ.get("LANCE_PREFIX", "tables"))
    drop_current_parallel.add_argument("--lance-root-uri")
    drop_current_parallel.add_argument("--duckdb-path", default=os.environ.get("LANCE_DROP_CURRENT_DUCKDB_PATH", "/tmp/lance-drop-current.duckdb"))
    drop_current_parallel.add_argument("--aws-region")
    drop_current_parallel.set_defaults(func=run_drop_current_parallel)

    indexes_parallel = sub.add_parser("run-create-indexes-parallel")
    indexes_parallel.add_argument("--tables")
    indexes_parallel.add_argument("--tables-file")
    indexes_parallel.add_argument("--max-tables", type=int)
    indexes_parallel.add_argument("--concurrency", type=int, default=int(os.environ.get("LANCE_INDEX_CONCURRENCY", "4")))
    indexes_parallel.add_argument("--cluster")
    indexes_parallel.add_argument("--task-definition")
    indexes_parallel.add_argument("--container-name", default=os.environ.get("LANCE_MIGRATION_CONTAINER_NAME", "lance-migration"))
    indexes_parallel.add_argument("--subnets", default=os.environ.get("ECS_SUBNETS", ""))
    indexes_parallel.add_argument("--security-groups", default=os.environ.get("ECS_SECURITY_GROUPS", ""))
    indexes_parallel.add_argument("--assign-public-ip", action=argparse.BooleanOptionalAction, default=os.environ.get("ECS_ASSIGN_PUBLIC_IP", "true").lower() == "true")
    indexes_parallel.add_argument("--capacity-provider", default=os.environ.get("ECS_CAPACITY_PROVIDER"))
    indexes_parallel.add_argument("--started-by", default=os.environ.get("ECS_STARTED_BY", "lance-index-creation-runner"))
    indexes_parallel.add_argument("--poll-seconds", type=int, default=int(os.environ.get("LANCE_MIGRATION_POLL_SECONDS", "20")))
    indexes_parallel.add_argument("--stop-on-failure", action=argparse.BooleanOptionalAction, default=True)
    indexes_parallel.add_argument("--env", action="append", default=[])
    indexes_parallel.add_argument("--lance-bucket")
    indexes_parallel.add_argument("--lance-prefix", default=os.environ.get("LANCE_PREFIX", "tables"))
    indexes_parallel.add_argument("--lance-root-uri")
    indexes_parallel.add_argument("--generated-indexes", action=argparse.BooleanOptionalAction, default=True)
    indexes_parallel.add_argument("--hot-indexes", action=argparse.BooleanOptionalAction, default=True)
    indexes_parallel.add_argument("--table-config-indexes", action=argparse.BooleanOptionalAction, default=os.environ.get("LANCE_TABLE_CONFIG_INDEXES", "true").lower() not in {"0", "false", "no"})
    indexes_parallel.add_argument("--table-config-bucket")
    indexes_parallel.add_argument("--table-config-prefix", default=os.environ.get("LANCE_TABLE_CONFIG_PREFIX", "config/lance/tables"))
    indexes_parallel.add_argument("--table-config-state-bucket")
    indexes_parallel.add_argument("--table-config-state-prefix", default=os.environ.get("LANCE_TABLE_CONFIG_STATE_PREFIX", DEFAULT_LANCE_TABLE_CONFIG_STATE_PREFIX))
    indexes_parallel.add_argument("--duckdb-path", default=os.environ.get("LANCE_INDEX_DUCKDB_PATH", "/tmp/lance-indexes.duckdb"))
    indexes_parallel.add_argument("--aws-region")
    indexes_parallel.set_defaults(func=run_create_indexes_parallel)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
