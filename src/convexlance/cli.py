from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
import json
import os
import signal
import socket
import threading
import time
from typing import Any
import uuid
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen


JsonMap = dict[str, Any]
TypeKind = str


class LeaseUnavailable(RuntimeError):
    pass


_shutdown_event = threading.Event()
_shutdown_signal = ""


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return str(signum)


def request_shutdown(signum: int, _frame: Any) -> None:
    global _shutdown_signal
    _shutdown_signal = _signal_name(signum)
    _shutdown_event.set()


def install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)


def shutdown_requested() -> bool:
    return _shutdown_event.is_set()


def shutdown_signal() -> str:
    return _shutdown_signal


def log_event(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, "ts": int(time.time()), **fields}, sort_keys=True), flush=True)


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


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


def install_and_load(conn: Any, extension: str) -> None:
    conn.execute(f"INSTALL {extension}")
    conn.execute(f"LOAD {extension}")



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
        if all(k in {"int64", "float64"} for k in inferred):
            return ("float64" if "float64" in inferred else "int64", nullable, None)
        if len(set(inferred)) == 1 and inferred[0] != "array":
            return inferred[0], nullable, None
        return "json", nullable, None
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
    if t == "string":
        return "string", nullable, None
    if t == "boolean":
        return "bool", nullable, None
    if t == "integer":
        return "int64", nullable, None
    if t == "number":
        return "float64", nullable, None
    if t == "array":
        item = _as_object(c.get("items"))
        if not item:
            return "json", nullable, None
        item_kind, item_nullable, _ = infer_kind(item)
        if item_nullable or item_kind in {"json", "array"}:
            return "json", nullable, None
        return "array", nullable, item_kind
    return "json", nullable, None


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
        columns.append(ColumnSpec(key, kind, key in required and not nullable, element_kind))
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



def _convex_client(args: argparse.Namespace) -> ConvexClient:
    convex_url = args.convex_url or os.environ.get("CONVEX_URL")
    deploy_key = args.convex_deploy_key or os.environ.get("CONVEX_DEPLOY_KEY")
    if not convex_url or not deploy_key:
        raise SystemExit("CONVEX_URL and CONVEX_DEPLOY_KEY are required")
    return ConvexClient(convex_url, deploy_key)


def _incremental_state(args: argparse.Namespace) -> S3JsonState:
    bucket = args.state_bucket or os.environ.get("LANCE_INCREMENTAL_STATE_BUCKET")
    prefix = args.state_prefix or os.environ.get("LANCE_INCREMENTAL_STATE_PREFIX") or "lance-incremental"
    if not bucket:
        raise SystemExit("--state-bucket or LANCE_INCREMENTAL_STATE_BUCKET is required")
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
    if pat.is_integer(existing_type) and pat.is_floating(desired_type):
        return False
    if pat.is_string(existing_type) and pat.is_string(desired_type):
        return True
    if pat.is_large_string(existing_type) and pat.is_string(desired_type):
        return True
    return False


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


def incremental_schema_payload(state: S3JsonState, args: argparse.Namespace, client: ConvexClient) -> JsonMap:
    payload = state.read(args.schema_key)
    if _schema_cache_stale(payload, args.schema_refresh_seconds):
        schemas = client.json_schemas(delta_schema=True)
        payload = {"fetched_at": int(time.time()), "schemas": schemas}
        state.write(args.schema_key, payload)
        log_event("lance_incremental_schema_refreshed", tables=len([k for k, v in schemas.items() if isinstance(v, dict)]))
    return payload or {"schemas": {}}


def table_schema_from_payload(schema_payload: JsonMap, source_table: str) -> JsonMap | None:
    schemas = schema_payload.get("schemas")
    if not isinstance(schemas, dict):
        return None
    table_schema = schemas.get(source_table)
    return table_schema if isinstance(table_schema, dict) else None


def reconcile_lance_schema(conn: Any, args: argparse.Namespace, table_name: str, target_uri: str, table_schema: JsonMap | None) -> None:
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
            raise RuntimeError(f"Lance table missing for {table_name} at {target_uri}; run an initial load before allowing incremental deltas")
        if args.unknown_table_policy == "skip":
            log_event("lance_incremental_table_skipped_missing_dataset", table=table_name, target_uri=target_uri)
            return
        import pyarrow as pa

        empty = pa.Table.from_pylist([], schema=_arrow_schema_for_specs(specs))
        lance.write_dataset(empty, target_uri, mode="create")
        generated_indexes, index_skipped = create_indexes_for_columns(conn, table_name, quote_literal(target_uri), generated_index_columns())
        log_event(
            "lance_incremental_table_created_from_schema",
            table=table_name,
            target_uri=target_uri,
            columns=len(specs),
            generated_indexes=generated_indexes,
            index_skipped=index_skipped,
        )
        return
    existing = {field.name: field.type for field in dataset.schema}
    added: list[str] = []
    drift: list[JsonMap] = []
    for spec in specs:
        desired_type = _pa_type_for_column(spec)
        existing_type = existing.get(spec.name)
        if existing_type is not None:
            if not _schema_types_compatible(existing_type, desired_type):
                drift.append({"column": spec.name, "existing": str(existing_type), "desired": str(desired_type)})
            continue
        statement = f"ALTER TABLE {quote_literal(target_uri)} ADD COLUMN {quote_identifier(spec.name)} {_lance_sql_type(spec)}"
        conn.execute(statement)
        added.append(spec.name)
    if drift:
        log_event("lance_incremental_schema_type_drift", table=table_name, target_uri=target_uri, drift=drift)
        raise RuntimeError(f"Lance schema type drift for {table_name}: {drift}")
    if added:
        log_event("lance_incremental_schema_columns_added", table=table_name, target_uri=target_uri, columns=added)


def _existing_current_rows(conn: Any, target_ref: str, ids: Iterable[str]) -> list[JsonMap]:
    wanted = sorted({status_id_value(status, row_id) for row_id in ids for status in (1, 3)})
    if not wanted:
        return []
    values = ", ".join(f"({quote_literal(value)})" for value in wanted)
    result = conn.execute(f"SELECT * FROM {target_ref} WHERE __status_id IN (SELECT * FROM (VALUES {values}))")
    column_names = [column[0] for column in result.description]
    return [dict(zip(column_names, row, strict=False)) for row in result.fetchall()]


def merge_incremental_rows(conn: Any, table_name: str, target_uri: str, rows: list[JsonMap]) -> int:
    if not rows:
        return 0
    target_ref = quote_literal(target_uri)
    ids = [str(row["_id"]) for row in rows if row.get("_id") is not None]
    column_types = _table_column_types(conn, target_ref)
    existing_current = _existing_current_rows(conn, target_ref, ids)
    merge_rows = prepare_incremental_merge_rows(rows, existing_current)
    merge_rows = _coerce_merge_rows_for_schema(merge_rows, column_types)
    import lance
    import pyarrow as pa

    dataset = lance.dataset(target_uri)
    merge_rows = _coerce_rows_for_arrow_schema(merge_rows, dataset.schema)
    merge_table = pa.Table.from_pylist(merge_rows, schema=dataset.schema)
    dataset.merge_insert("__id_ts").when_matched_update_all().when_not_matched_insert_all().execute(merge_table)
    return len(merge_rows)


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
    schema_payload = incremental_schema_payload(state, args, client) if args.reconcile_schema else {"schemas": {}}
    start_cursor = cursor
    conn = open_lance_duckdb(args)
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
                reconcile_lance_schema(conn, args, physical, target_uri, table_schema_from_payload(schema_payload, source_table))
                merged = merge_incremental_rows(conn, physical, target_uri, table_page_rows)
                rows_merged += merged
                page_rows_merged += merged
                table_rows[source_table] = table_rows.get(source_table, 0) + merged
                page_table_rows[source_table] = page_table_rows.get(source_table, 0) + merged
                log_event("lance_incremental_table_merged", table=source_table, physical_table=physical, rows=len(table_page_rows), merge_rows=merged, cursor=page_cursor)
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
            if shutdown_requested():
                log_event("lance_incremental_shutdown_requested", signal=shutdown_signal(), start_cursor=start_cursor, last_cursor=cursor, pages_completed=pages)
                break
            if pages >= args.max_pages_per_sync or not has_more:
                break
        state.delete("incremental/last_error.json")
        log_event("lance_incremental_completed", pages=pages, rows_seen=rows_seen, rows_accepted=rows_accepted, rows_merged=rows_merged, tables=len(table_rows), start_cursor=start_cursor, end_cursor=cursor)
        return {"pages": pages, "rows_seen": rows_seen, "rows_accepted": rows_accepted, "rows_merged": rows_merged, "tables": sorted(source_to_physical_table(table) for table in table_rows), "start_cursor": start_cursor, "end_cursor": cursor}
    except BaseException as exc:
        state.write("incremental/last_error.json", {"owner": lease.owner, "error": repr(exc), "start_cursor": start_cursor, "last_cursor": cursor, "updated_at": int(time.time())})
        raise
    finally:
        conn.close()
        release_lease(state, lease)


def run_incremental_loop(args: argparse.Namespace) -> None:
    touched_tables: set[str] = set()
    pages_since_optimize = 0
    rows_since_optimize = 0
    last_optimized_at = time.time()
    while not shutdown_requested():
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
        if shutdown_requested():
            log_event("lance_incremental_loop_stopped", signal=shutdown_signal())
            break
        if not optimized:
            _shutdown_event.wait(args.sleep_seconds)


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


def index_name(table_name: str, column: str) -> str:
    return source_to_physical_table(f"{table_name}_{column}_idx")


def generated_index_columns() -> list[tuple[str, str]]:
    return [("__id_ts", "BTREE"), ("__status_id", "BTREE"), ("__status", "BITMAP")]


def create_indexes_for_columns(conn: Any, table_name: str, table_ref: str, requested_columns: list[tuple[str, str]]) -> tuple[int, list[JsonMap]]:
    columns = table_columns(conn, table_ref)
    skipped: list[JsonMap] = []
    created = 0
    for column, index_type in requested_columns:
        if column not in columns:
            skipped.append({"table": table_name, "column": column, "reason": "missing_column"})
            continue
        statement = f"CREATE INDEX {index_name(table_name, column)} ON {table_ref} ({column}) USING {index_type}"
        try:
            conn.execute(statement)
            created += 1
        except Exception as exc:  # noqa: BLE001
            if "already exists" in str(exc).lower():
                skipped.append({"table": table_name, "column": column, "error": repr(exc), "reason": "already_exists"})
                continue
            raise
    return created, skipped

def add_incremental_arguments(parser: argparse.ArgumentParser, *, loop: bool) -> None:
    parser.add_argument("--convex-url")
    parser.add_argument("--convex-deploy-key")
    parser.add_argument("--state-bucket")
    parser.add_argument("--state-prefix")
    parser.add_argument("--cursor-key", default=os.environ.get("LANCE_INCREMENTAL_CURSOR_KEY", "incremental/cursor.json"))
    parser.add_argument("--schema-key", default=os.environ.get("LANCE_INCREMENTAL_SCHEMA_KEY", "incremental/schema.json"))
    parser.add_argument("--schema-refresh-seconds", type=int, default=int(os.environ.get("LANCE_INCREMENTAL_SCHEMA_REFRESH_SECONDS", "600")))
    parser.add_argument("--reconcile-schema", action=argparse.BooleanOptionalAction, default=os.environ.get("LANCE_INCREMENTAL_RECONCILE_SCHEMA", "true").lower() not in {"0", "false", "no"})
    parser.add_argument("--unknown-table-policy", choices=["fail", "create", "skip"], default=os.environ.get("LANCE_INCREMENTAL_UNKNOWN_TABLE_POLICY", "fail"))
    parser.add_argument("--lance-bucket")
    parser.add_argument("--lance-prefix", default=os.environ.get("LANCE_PREFIX", "tables"))
    parser.add_argument("--lance-root-uri")
    parser.add_argument("--lance-scope")
    parser.add_argument("--catalog-alias")
    parser.add_argument("--max-pages-per-sync", type=int, default=int(os.environ.get("LANCE_INCREMENTAL_MAX_PAGES_PER_SYNC", "100")))
    if loop:
        parser.add_argument("--sleep-seconds", type=int, default=int(os.environ.get("LANCE_INCREMENTAL_SLEEP_SECONDS", "20")))
        parser.add_argument("--optimize-touched-indices", action=argparse.BooleanOptionalAction, default=os.environ.get("LANCE_INCREMENTAL_OPTIMIZE_TOUCHED_INDICES", "true").lower() not in {"0", "false", "no"})
        parser.add_argument("--optimize-indices-pages", type=int, default=int(os.environ.get("LANCE_INCREMENTAL_OPTIMIZE_INDICES_PAGES", "50")))
        parser.add_argument("--optimize-indices-rows", type=int, default=int(os.environ.get("LANCE_INCREMENTAL_OPTIMIZE_INDICES_ROWS", "5000")))
        parser.add_argument("--optimize-indices-seconds", type=int, default=int(os.environ.get("LANCE_INCREMENTAL_OPTIMIZE_INDICES_SECONDS", "900")))
        parser.add_argument("--idle-maintenance", action=argparse.BooleanOptionalAction, default=os.environ.get("LANCE_INCREMENTAL_IDLE_MAINTENANCE", "true").lower() not in {"0", "false", "no"})
        parser.add_argument("--idle-maintenance-max-rows", type=int, default=int(os.environ.get("LANCE_INCREMENTAL_IDLE_MAINTENANCE_MAX_ROWS", "0")))
        parser.add_argument("--idle-maintenance-max-pages", type=int, default=int(os.environ.get("LANCE_INCREMENTAL_IDLE_MAINTENANCE_MAX_PAGES", "1")))
        parser.add_argument("--maintenance-state-key", default=os.environ.get("LANCE_INCREMENTAL_MAINTENANCE_STATE_KEY", "incremental/maintenance_state.json"))
        parser.add_argument("--maintenance-audit-key-prefix", default=os.environ.get("LANCE_INCREMENTAL_MAINTENANCE_AUDIT_KEY_PREFIX", "incremental/maintenance-audit"))
        parser.add_argument("--maintenance-tables")
        parser.add_argument("--maintenance-tables-file")
        parser.add_argument("--maintenance-optimize-indices-seconds", type=int, default=int(os.environ.get("LANCE_MAINTENANCE_OPTIMIZE_INDICES_SECONDS", "3600")))
        parser.add_argument("--maintenance-compact-files-seconds", type=int, default=int(os.environ.get("LANCE_MAINTENANCE_COMPACT_FILES_SECONDS", "86400")))
        parser.add_argument("--maintenance-cleanup-old-versions-seconds", type=int, default=int(os.environ.get("LANCE_MAINTENANCE_CLEANUP_OLD_VERSIONS_SECONDS", "604800")))
        parser.add_argument("--maintenance-cleanup-older-than-seconds", type=int, default=int(os.environ.get("LANCE_MAINTENANCE_CLEANUP_OLDER_THAN_SECONDS", "259200")))
        parser.add_argument("--maintenance-retain-versions", type=int, default=int(os.environ.get("LANCE_MAINTENANCE_RETAIN_VERSIONS", "500")))
        parser.add_argument("--maintenance-compact-threads", type=int, default=int(os.environ.get("LANCE_MAINTENANCE_COMPACT_THREADS", "1")))
        parser.add_argument("--maintenance-heavy-window-start-utc", type=int, default=int(os.environ.get("LANCE_MAINTENANCE_HEAVY_WINDOW_START_UTC", "7")))
        parser.add_argument("--maintenance-heavy-window-end-utc", type=int, default=int(os.environ.get("LANCE_MAINTENANCE_HEAVY_WINDOW_END_UTC", "12")))
    parser.add_argument("--lock-ttl-seconds", type=int, default=int(os.environ.get("LANCE_INCREMENTAL_LOCK_TTL_SECONDS", "300")))
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--duckdb-path", default=os.environ.get("LANCE_INCREMENTAL_DUCKDB_PATH", "/tmp/lance-incremental.duckdb"))
    parser.add_argument("--threads", type=int, default=int(os.environ.get("DUCKDB_THREADS", "0")) or None)
    parser.add_argument("--memory-limit", default=os.environ.get("DUCKDB_MEMORY_LIMIT"))
    parser.add_argument("--temp-directory", default=os.environ.get("DUCKDB_TEMP_DIRECTORY"))
    parser.add_argument("--aws-region")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="convexlance")
    sub = parser.add_subparsers(dest="command", required=True)

    incremental = sub.add_parser("incremental-once")
    add_incremental_arguments(incremental, loop=False)
    incremental.set_defaults(func=run_incremental_once)

    incremental_loop = sub.add_parser("incremental-loop")
    add_incremental_arguments(incremental_loop, loop=True)
    incremental_loop.set_defaults(func=run_incremental_loop)
    return parser


def main() -> None:
    install_signal_handlers()
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
