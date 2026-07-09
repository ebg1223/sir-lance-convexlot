from argparse import Namespace
from contextlib import contextmanager
import tempfile
import threading
import time
import unittest
from unittest.mock import patch, MagicMock

from convexlance.cli import (
    BufferedDeltaPage,
    ColumnSpec,
    MissingLanceColumns,
    arrow_schema_signature,
    build_repair_quoted_json_strings_select_sql,
    build_select_sql,
    create_indexes_for_columns,
    dataset_index_names,
    decode_json_string_literal,
    desired_repair_columns_from_schema,
    desired_index_names,
    flush_incremental_buffer,
    generated_index_columns,
    infer_kind,
    load_table_config,
    merge_incremental_rows_with_schema_refresh,
    merge_incremental_rows,
    missing_lance_columns_for_rows,
    normalize_rows_for_specs,
    physical_to_source_schema_map,
    prepare_incremental_merge_rows,
    reconcile_lance_schema,
    reconcile_schema_payload_existing_lance_tables,
    read_applied_table_config_version,
    reconcile_table_config_for_dataset,
    requested_index_specs,
    run_incremental_once,
    schema_column_specs,
    TableLease,
    table_config_state_key,
    validate_repair_columns,
    write_applied_table_config_state,
    _add_lance_columns_native,
    _coerce_drift_columns,
    _coerce_scalar_to_kind,
    _incremental_pass_hit_page_cap,
    _lance_in_filter,
    _schema_type_alteration,
    _schema_types_compatible,
    _status_id_filter_values,
)


class FakeConn:
    def execute(self, sql: str):
        assert sql == 'DESCRIBE "s3tables"."convex"."Records"'
        return self

    def fetchall(self):
        return [
            ("_id",),
            ("_ts",),
            ("_current",),
            ("_deleted",),
            ("recordId",),
            ("__status",),
            ("__status_id",),
            ("__id_ts",),
        ]


class FakeIndexConn:
    def __init__(self):
        self.statements: list[str] = []

    def execute(self, sql: str):
        self.statements.append(sql)
        return self

    def fetchall(self):
        return [("__id_ts",), ("__status_id",), ("__status",)]


class FakeRepairConn:
    def execute(self, sql: str):
        assert sql == "DESCRIBE 's3://bucket/tables/records.lance'"
        return self

    def fetchall(self):
        return [("_id",), ("recordId",), ("payload",)]


class FakeDataset:
    def __init__(self, schema):
        self.schema = schema


class FakeAddColumnsDataset:
    def __init__(self):
        self.added_schema = None

    def add_columns(self, schema):
        self.added_schema = schema


class FakeIndex:
    def __init__(self, name):
        self.name = name


class FakeIndexDataset:
    def list_indices(self):
        return [FakeIndex("records___id_ts_idx"), {"name": "records_id_idx"}, "records_old_idx"]


class FakeEmptyMergeDataset:
    def __init__(self, schema):
        self.schema = schema
        self.merge_called = False

    def count_rows(self):
        return 0

    def merge_insert(self, key):
        self.merge_called = True
        return self

    def when_matched_update_all(self):
        return self

    def when_not_matched_insert_all(self):
        return self

    def execute(self, table):
        return None


class FakeSpillMergeDataset(FakeEmptyMergeDataset):
    def __init__(self, schema):
        super().__init__(schema)
        self.deleted_predicate = None

    def count_rows(self):
        return 1

    def execute(self, table):
        raise OSError("LanceError(IO): Execution error: Spill has sent an error")

    def delete(self, predicate):
        self.deleted_predicate = predicate


class FakeState:
    def __init__(self):
        self.values = {}

    def read(self, name: str):
        return self.values.get(name)

    def write(self, name: str, value):
        self.values[name] = value
        return '"etag"'


class BuildSelectSqlTest(unittest.TestCase):
    def test_schema_types_accept_legacy_lance_columns(self):
        import pyarrow as pa

        self.assertTrue(_schema_types_compatible(pa.int64(), pa.int8()))
        self.assertTrue(_schema_types_compatible(pa.large_string(), pa.string()))
        self.assertFalse(_schema_types_compatible(pa.string(), pa.float64()))
        self.assertFalse(_schema_types_compatible(pa.int64(), pa.float64()))

    def test_schema_type_alteration_allows_safe_widening(self):
        import pyarrow as pa

        self.assertEqual(_schema_type_alteration(pa.int32(), pa.int64()), {"data_type": pa.int64()})
        self.assertIsNone(_schema_type_alteration(pa.int64(), pa.float64()))
        self.assertIsNone(_schema_type_alteration(pa.float64(), pa.string()))
        self.assertIsNone(_schema_type_alteration(pa.string(), pa.float64()))

    def test_add_lance_columns_uses_nullable_fields(self):
        dataset = FakeAddColumnsDataset()

        self.assertTrue(_add_lance_columns_native(dataset, [ColumnSpec("recordId", "string", required=True)]))

        self.assertIsNotNone(dataset.added_schema)
        self.assertTrue(dataset.added_schema.field("recordId").nullable)

    def test_build_select_sql_generates_selected_columns(self):
        args = Namespace(
            catalog_alias="s3tables",
            namespace="convex",
            where=None,
            order_by=None,
            max_rows=None,
        )

        query = build_select_sql(FakeConn(), args, "Records")

        self.assertNotIn('"_current"', query)
        self.assertNotIn('"__status"', query)
        self.assertIn('"_id", "_ts", "_deleted", "recordId"', query)
        self.assertIn("AS __status", query)
        self.assertIn("AS __status_id", query)
        self.assertIn("AS __id_ts", query)
        self.assertIn("COALESCE(_current, FALSE)", query)
        self.assertIn("COALESCE(_deleted, FALSE)", query)
        self.assertIn("CAST(_id AS VARCHAR) || '#' || CAST(_ts AS VARCHAR)", query)

    def test_create_indexes_for_generated_columns(self):
        conn = FakeIndexConn()

        created, skipped = create_indexes_for_columns(conn, "records", "'s3://bucket/tables/records.lance'", generated_index_columns())

        self.assertEqual(created, 3)
        self.assertEqual(skipped, [])
        self.assertIn("CREATE INDEX records___id_ts_idx ON 's3://bucket/tables/records.lance' (__id_ts) USING BTREE", conn.statements)
        self.assertIn("CREATE INDEX records___status_id_idx ON 's3://bucket/tables/records.lance' (__status_id) USING BTREE", conn.statements)
        self.assertIn("CREATE INDEX records___status_idx ON 's3://bucket/tables/records.lance' (__status) USING BITMAP", conn.statements)

    def test_table_config_filename_identifies_table_and_hash_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(f"{tmp}/recordworkitems.toml", "w", encoding="utf-8") as handle:
                handle.write('[[indexes]]\ncolumn = "recordPointer"\ntype = "btree"\n')

            config = load_table_config(Namespace(table_config_dir=tmp, table_config_bucket=None, table_config_prefix=None), "recordworkitems")

        self.assertIsNotNone(config)
        assert config is not None
        self.assertEqual(config.table_name, "recordworkitems")
        self.assertTrue(config.version.startswith("sha256:"))
        self.assertEqual([(idx.column, idx.index_type, idx.name) for idx in config.indexes], [("recordPointer", "BTREE", None)])

    def test_requested_index_specs_combines_generated_and_table_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(f"{tmp}/records.toml", "w", encoding="utf-8") as handle:
                handle.write('[[indexes]]\ncolumn = "__id_ts"\ntype = "BTREE"\n\n[[indexes]]\ncolumn = "id"\ntype = "BTREE"\n')

            requested, config = requested_index_specs(
                Namespace(generated_indexes=True, table_config_indexes=True, table_config_dir=tmp, table_config_bucket=None, table_config_prefix=None),
                "records",
            )

        self.assertIsNotNone(config)
        self.assertEqual([(idx.column, idx.index_type) for idx in requested], [("__id_ts", "BTREE"), ("__status_id", "BTREE"), ("__status", "BITMAP"), ("id", "BTREE")])

    def test_incremental_unknown_table_create_disables_generated_indexes(self):
        args = Namespace(
            unknown_table_policy="create",
            table_config_indexes=True,
            table_config_dir=None,
            table_config_bucket=None,
            table_config_prefix=None,
            table_config_state_bucket=None,
            table_config_state_prefix="state",
            aws_region="us-west-2",
        )
        table_schema = {"type": "object", "properties": {"value": {"type": "string"}}}
        requested_args = []

        def fake_requested_index_specs(index_args, table_name):
            requested_args.append((index_args, table_name))
            return [], None

        with (
            patch("lance.dataset", side_effect=[RuntimeError("missing"), object()]),
            patch("lance.write_dataset"),
            patch("convexlance.cli.requested_index_specs", side_effect=fake_requested_index_specs),
            patch("convexlance.cli.create_indexes_for_dataset", return_value=(0, [])),
            patch("convexlance.cli.write_applied_table_config_state"),
        ):
            specs = reconcile_lance_schema(args, "records", "s3://bucket/tables/records.lance", table_schema)

        self.assertIn("value", [spec.name for spec in specs])
        self.assertEqual(requested_args[0][1], "records")
        self.assertFalse(requested_args[0][0].generated_indexes)

    def test_dataset_index_names_detects_extra_indexes(self):
        desired = desired_index_names("records", generated_index_columns())

        extra = dataset_index_names(FakeIndexDataset()) - desired

        self.assertEqual(extra, {"records_id_idx", "records_old_idx"})

    def test_table_config_applied_state_uses_external_json_state(self):
        state = FakeState()
        with tempfile.TemporaryDirectory() as tmp:
            with open(f"{tmp}/recordworkitems.toml", "w", encoding="utf-8") as handle:
                handle.write('[[indexes]]\ncolumn = "recordPointer"\ntype = "BTREE"\n')
            config = load_table_config(Namespace(table_config_dir=tmp, table_config_bucket=None, table_config_prefix=None), "recordworkitems")

        write_applied_table_config_state(state, "Record Work Items", config, 1, [])

        self.assertEqual(table_config_state_key("Record Work Items"), "record_work_items.json")
        self.assertEqual(read_applied_table_config_version(state, "Record Work Items"), config.version)
        self.assertEqual(state.values["record_work_items.json"]["table"], "record_work_items")
        self.assertEqual(state.values["record_work_items.json"]["created_indexes"], 1)

    def test_prepare_incremental_merge_rows_derives_current(self):
        incoming = [
            {"_id": "a", "_ts": 100, "_deleted": False, "_current": False, "value": "old"},
            {"_id": "a", "_ts": 200, "_deleted": True, "value": "new"},
            {"_id": "b", "_ts": 50, "_deleted": False, "value": "first"},
        ]
        existing = [{"_id": "a", "_ts": 150, "_deleted": False, "__id_ts": "a#150", "__status": 1, "__status_id": "1#a"}]

        rows = prepare_incremental_merge_rows(incoming, existing)
        by_version = {row["__id_ts"]: row for row in rows}

        self.assertEqual(by_version["a#100"]["__status"], 0)
        self.assertEqual(by_version["a#100"]["__status_id"], "0#a")
        self.assertEqual(by_version["a#200"]["__status"], 3)
        self.assertEqual(by_version["a#200"]["__status_id"], "3#a")
        self.assertEqual(by_version["a#150"]["__status"], 0)
        self.assertEqual(by_version["a#150"]["__status_id"], "0#a")
        self.assertEqual(by_version["b#50"]["__status"], 1)
        self.assertEqual(by_version["b#50"]["__status_id"], "1#b")
        self.assertNotIn("_current", by_version["a#200"])

    def test_prepare_incremental_merge_rows_materializes_deleted(self):
        incoming = [
            {"_id": "a", "_ts": 100, "value": "live"},
            {"_id": "b", "_ts": 100, "_deleted": True},
        ]

        rows = prepare_incremental_merge_rows(incoming, [])
        by_version = {row["__id_ts"]: row for row in rows}

        self.assertIs(by_version["a#100"]["_deleted"], False)
        self.assertIs(by_version["b#100"]["_deleted"], True)

    def test_prepare_incremental_merge_rows_demotion_preserves_existing_payload(self):
        incoming = [{"_id": "a", "_ts": 200, "value": "new"}]
        existing = [
            {
                "_id": "a",
                "_ts": 150,
                "_deleted": None,
                "value": "old",
                "extra": 7,
                "__id_ts": "a#150",
                "__status": 1,
                "__status_id": "1#a",
            }
        ]

        rows = prepare_incremental_merge_rows(incoming, existing)
        by_version = {row["__id_ts"]: row for row in rows}

        demoted = by_version["a#150"]
        self.assertEqual(demoted["value"], "old")
        self.assertEqual(demoted["extra"], 7)
        self.assertIs(demoted["_deleted"], False)
        self.assertEqual(demoted["__status"], 0)
        self.assertEqual(demoted["__status_id"], "0#a")

    def test_prepare_incremental_merge_rows_preserves_existing_current_for_older_delta(self):
        incoming = [{"_id": "a", "_ts": 100, "_deleted": False, "value": "old"}]
        existing = [{"_id": "a", "_ts": 150, "_deleted": False, "__id_ts": "a#150", "__status": 1, "__status_id": "1#a"}]

        rows = prepare_incremental_merge_rows(incoming, existing)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["__id_ts"], "a#100")
        self.assertEqual(rows[0]["__status"], 0)
        self.assertEqual(rows[0]["__status_id"], "0#a")

    def test_infer_kind_mixed_string_number_prefers_string(self):
        schema = {"anyOf": [{"type": "string"}, {"type": "number"}]}

        self.assertEqual(infer_kind(schema), ("string", False, None))

    def test_infer_kind_complex_values_are_json(self):
        self.assertEqual(infer_kind({"type": "array", "items": {"type": "object", "properties": {"a": {"type": "string"}}}}), ("json", False, None))
        self.assertEqual(infer_kind({"type": "object", "properties": {"a": {"type": "string"}}}), ("json", False, None))
        self.assertEqual(infer_kind({"x-convex": "record"}), ("json", False, None))
        self.assertEqual(infer_kind({}), ("json", False, None))
        self.assertEqual(infer_kind({"anyOf": [{"type": "null"}, {}]}), ("json", True, None))
        self.assertEqual(infer_kind({"anyOf": [{"type": "null"}, {"type": "string"}, {}]}), ("json", True, None))

    def test_infer_kind_scalar_arrays_stay_arrays(self):
        self.assertEqual(infer_kind({"type": "array", "items": {"type": "string"}}), ("array", False, "string"))
        self.assertEqual(infer_kind({"type": "array", "items": {"type": "number"}}), ("array", False, "float64"))

    def test_infer_kind_numeric_and_boolean(self):
        self.assertEqual(infer_kind({"anyOf": [{"type": "integer"}, {"type": "number"}]}), ("float64", False, None))
        self.assertEqual(infer_kind({"anyOf": [{"type": "boolean"}, {"type": "integer"}]}), ("int64", False, None))
        self.assertEqual(infer_kind({"type": "integer"}), ("int64", False, None))
        self.assertEqual(infer_kind({"type": "boolean"}), ("bool", False, None))

    def test_infer_kind_bare_null_creates_nullable_string_placeholder(self):
        self.assertEqual(infer_kind({"type": "null"}), ("string", True, None))

    def test_schema_column_specs_materialize_null_fields_as_nullable_placeholders(self):
        specs = schema_column_specs(
            {
                "type": "object",
                "properties": {
                    "leaseExpiresAt": {"type": "null"},
                    "processedCount": {"type": "number"},
                },
            },
        )
        by_name = {spec.name: spec for spec in specs}
        self.assertEqual(by_name["leaseExpiresAt"].kind, "string")
        self.assertFalse(by_name["leaseExpiresAt"].required)
        self.assertEqual(by_name["processedCount"].kind, "float64")

    def test_schema_column_specs_keep_application_fields_nullable_even_when_required(self):
        specs = schema_column_specs(
            {
                "type": "object",
                "required": ["completedAt"],
                "properties": {"completedAt": {"type": "string"}},
            },
        )

        completed_at = {spec.name: spec for spec in specs}["completedAt"]
        self.assertEqual(completed_at.kind, "string")
        self.assertFalse(completed_at.required)

    def test_normalize_rows_for_specs_preserves_scalar_strings(self):
        rows = normalize_rows_for_specs(
            [{"json_col": "abc", "string_col": {"a": 1}, "int_col": "nope", "bool_col": "true"}],
            [
                ColumnSpec("json_col", "json"),
                ColumnSpec("string_col", "string"),
                ColumnSpec("int_col", "int64"),
                ColumnSpec("bool_col", "bool"),
            ],
        )

        self.assertEqual(rows[0]["json_col"], "abc")
        self.assertEqual(rows[0]["string_col"], '{"a":1}')
        self.assertIsNone(rows[0]["int_col"])
        self.assertIsNone(rows[0]["bool_col"])

    def test_normalize_rows_for_specs_json_encodes_complex_values(self):
        rows = normalize_rows_for_specs([{"json_col": {"a": 1}, "list_col": [1, 2]}], [ColumnSpec("json_col", "json"), ColumnSpec("list_col", "json")])

        self.assertEqual(rows[0]["json_col"], '{"a":1}')
        self.assertEqual(rows[0]["list_col"], "[1,2]")

    def test_status_id_filter_helpers(self):
        values = _status_id_filter_values(["b", "a"])

        self.assertEqual(values, ["1#a", "1#b", "3#a", "3#b"])
        self.assertEqual(_lance_in_filter("__status_id", values), "__status_id IN ('1#a', '1#b', '3#a', '3#b')")

    def test_decode_json_string_literal_only_decodes_string_literals(self):
        self.assertEqual(decode_json_string_literal('"abc"'), "abc")
        self.assertEqual(decode_json_string_literal('{"a":1}'), '{"a":1}')
        self.assertEqual(decode_json_string_literal("abc"), "abc")
        self.assertEqual(decode_json_string_literal('"unterminated'), '"unterminated')

    def test_build_repair_quoted_json_strings_select_sql(self):
        sql = build_repair_quoted_json_strings_select_sql(FakeRepairConn(), "'s3://bucket/tables/records.lance'", {"recordId"})

        self.assertIn('"_id"', sql)
        self.assertIn('CASE WHEN json_valid(CAST("recordId" AS VARCHAR))', sql)
        self.assertIn("json_type(CAST(CAST(\"recordId\" AS VARCHAR) AS JSON)) = 'VARCHAR'", sql)
        self.assertIn("json_extract_string(CAST(CAST(\"recordId\" AS VARCHAR) AS JSON), '$')", sql)
        self.assertIn('AS "recordId"', sql)
        self.assertIn('"payload"', sql)
        self.assertTrue(sql.endswith("FROM 's3://bucket/tables/records.lance'"))

    def test_validate_repair_columns_requires_string_columns(self):
        import pyarrow as pa

        dataset = FakeDataset(pa.schema([pa.field("recordId", pa.string()), pa.field("amount", pa.float64())]))

        self.assertEqual(validate_repair_columns(dataset, ["recordId"], True), (["recordId"], []))
        with self.assertRaisesRegex(RuntimeError, "missing"):
            validate_repair_columns(dataset, ["missing"], True)
        with self.assertRaisesRegex(RuntimeError, "non-string"):
            validate_repair_columns(dataset, ["amount"], True)

    def test_arrow_schema_signature_tracks_schema_shape(self):
        import pyarrow as pa

        signature = arrow_schema_signature(pa.schema([pa.field("recordId", pa.string(), nullable=True, metadata={b"k": b"v"})], metadata={b"schema": b"meta"}))

        self.assertEqual(
            signature,
            {
                "metadata": {"schema": "meta"},
                "fields": [{"name": "recordId", "type": "string", "nullable": True, "metadata": {"k": "v"}}],
            },
        )

    def test_schema_driven_repair_candidates_only_include_desired_strings(self):
        table_schema = {
            "type": "object",
            "properties": {
                "recordId": {"type": "string"},
                "externalJson": {"type": "object", "properties": {"a": {"type": "string"}}},
                "validatedResponse": {"anyOf": [{"type": "null"}, {"type": "string"}, {}]},
                "amount": {"type": "number"},
            },
        }

        self.assertEqual(desired_repair_columns_from_schema(table_schema), ["recordId"])

    def test_physical_to_source_schema_map(self):
        payload = {"schemas": {"Records": {"type": "object"}, "Record Work Items": {"type": "object"}}}

        mapped = physical_to_source_schema_map(payload)

        self.assertEqual(mapped["records"][0], "Records")
        self.assertEqual(mapped["record_work_items"][0], "Record Work Items")

    def test_missing_lance_columns_for_rows_detects_projection_drops(self):
        import pyarrow as pa

        schema = pa.schema([pa.field("_id", pa.string()), pa.field("_ts", pa.int64())])

        self.assertEqual(missing_lance_columns_for_rows([{"_id": "a", "_ts": 1, "newField": "x"}], schema), ["newField"])
        self.assertEqual(missing_lance_columns_for_rows([{"_id": "a", "_ts": 1, "_component": "system"}], schema), [])

    def test_merge_guard_raises_before_unknown_columns_can_drop(self):
        import pyarrow as pa
        from convexlance.cli import ensure_rows_fit_lance_schema

        schema = pa.schema([pa.field("_id", pa.string()), pa.field("_ts", pa.int64())])

        with self.assertRaisesRegex(MissingLanceColumns, "newField"):
            ensure_rows_fit_lance_schema("records", "s3://bucket/tables/records.lance", [{"_id": "a", "_ts": 1, "newField": "x"}], schema)

    def test_schema_refresh_reconciles_existing_lance_tables_only(self):
        payload = {"schemas": {"Records": {"type": "object"}, "Missing": {"type": "object"}}}
        args = Namespace(lance_prefix="tables", aws_region="us-west-2")
        calls = []

        def fake_reconcile(args, physical, target_uri, table_schema):
            calls.append((physical, target_uri, table_schema))
            return ["newField"]

        with (
            patch("convexlance.cli.list_lance_tables", return_value=["orphan", "records"]),
            patch("convexlance.cli.reconcile_lance_append_only_columns", side_effect=fake_reconcile),
        ):
            result = reconcile_schema_payload_existing_lance_tables(args, "bucket", payload)

        self.assertEqual(calls, [("records", "s3://bucket/tables/records.lance", {"type": "object"})])
        self.assertEqual(result["checked"], 1)
        self.assertEqual(result["reconciled"], 1)
        self.assertEqual(result["skipped"], 1)
        self.assertEqual(result["columns_added"], 1)
        self.assertEqual(result["failed"], 0)

    def test_missing_columns_force_schema_refresh_before_retry(self):
        args = Namespace(reconcile_schema=True)
        stale_payload = {"schemas": {"Records": {"type": "object", "properties": {}}}}
        fresh_schema = {"type": "object", "properties": {"newField": {"type": "string"}}}
        fresh_payload = {"schemas": {"Records": fresh_schema}}
        missing = MissingLanceColumns("records", "s3://bucket/tables/records.lance", ["newField"])

        with (
            patch("convexlance.cli.merge_incremental_rows", side_effect=[missing, 2]) as merge_rows,
            patch("convexlance.cli.incremental_schema_payload_with_status", return_value=(fresh_payload, True)) as refresh,
            patch("convexlance.cli.reconcile_lance_schema", return_value=[ColumnSpec("newField", "string")]) as reconcile,
        ):
            merged, payload = merge_incremental_rows_with_schema_refresh(
                args,
                FakeState(),
                object(),
                stale_payload,
                "Records",
                "records",
                "s3://bucket/tables/records.lance",
                [{"_id": "a", "_ts": 1, "newField": "x"}],
                [],
            )

        self.assertEqual(merged, 2)
        self.assertEqual(payload, fresh_payload)
        self.assertEqual(merge_rows.call_count, 2)
        refresh.assert_called_once()
        reconcile.assert_called_once_with(args, "records", "s3://bucket/tables/records.lance", fresh_schema)

    def test_merge_incremental_rows_appends_when_target_is_empty(self):
        import pyarrow as pa

        schema = pa.schema(
            [
                pa.field("_id", pa.string()),
                pa.field("_ts", pa.int64()),
                pa.field("_deleted", pa.bool_()),
                pa.field("__status", pa.int8()),
                pa.field("__status_id", pa.string()),
                pa.field("__id_ts", pa.string()),
                pa.field("value", pa.string()),
            ],
        )
        dataset = FakeEmptyMergeDataset(schema)

        with (
            patch("lance.dataset", return_value=dataset),
            patch("lance.write_dataset") as write_dataset,
            patch("convexlance.cli._existing_current_rows_native", return_value=[]),
        ):
            merged = merge_incremental_rows("records", "s3://bucket/tables/records.lance", [{"_id": "a", "_ts": 1, "_deleted": False, "value": "x"}])

        self.assertEqual(merged, 1)
        write_dataset.assert_called_once()
        self.assertEqual(write_dataset.call_args.kwargs["mode"], "append")
        self.assertFalse(dataset.merge_called)

    def test_merge_incremental_rows_falls_back_to_delete_append_on_spill(self):
        import pyarrow as pa

        schema = pa.schema(
            [
                pa.field("_id", pa.string()),
                pa.field("_ts", pa.int64()),
                pa.field("_deleted", pa.bool_()),
                pa.field("__status", pa.int8()),
                pa.field("__status_id", pa.string()),
                pa.field("__id_ts", pa.string()),
                pa.field("value", pa.string()),
            ],
        )
        dataset = FakeSpillMergeDataset(schema)

        with (
            patch("lance.dataset", return_value=dataset),
            patch("lance.write_dataset") as write_dataset,
            patch("convexlance.cli._existing_current_rows_native", return_value=[]),
        ):
            merged = merge_incremental_rows("records", "s3://bucket/tables/records.lance", [{"_id": "a", "_ts": 1, "_deleted": False, "value": "x"}])

        self.assertEqual(merged, 1)
        self.assertEqual(dataset.deleted_predicate, "__id_ts IN ('a#1')")
        write_dataset.assert_called_once()
        self.assertEqual(write_dataset.call_args.kwargs["mode"], "append")

    def test_merge_incremental_rows_rejects_null_in_non_nullable_column(self):
        import pyarrow as pa

        schema = pa.schema(
            [
                pa.field("_id", pa.string(), nullable=False),
                pa.field("_ts", pa.int64(), nullable=False),
                pa.field("_deleted", pa.bool_(), nullable=False),
                pa.field("__status", pa.int8(), nullable=False),
                pa.field("__status_id", pa.string(), nullable=False),
                pa.field("__id_ts", pa.string(), nullable=False),
                pa.field("value", pa.string()),
            ],
        )
        dataset = FakeSpillMergeDataset(schema)

        with (
            patch("lance.dataset", return_value=dataset),
            patch("lance.write_dataset") as write_dataset,
            patch("convexlance.cli._existing_current_rows_native", return_value=[]),
            patch("convexlance.cli.prepare_incremental_merge_rows", side_effect=lambda rows, existing: [dict(row, _deleted=None, __status=1, __status_id="1#a", __id_ts="a#1") for row in rows]),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                merge_incremental_rows("records", "s3://bucket/tables/records.lance", [{"_id": "a", "_ts": 1, "value": "x"}])

        self.assertIn("_deleted", str(ctx.exception))
        write_dataset.assert_not_called()
        self.assertIsNone(dataset.deleted_predicate)


class SchemaDriftCoercionTest(unittest.TestCase):
    def test_coerce_scalar_to_kind_parses_or_nulls(self):
        # float64: parse numeric strings, null the unparseable, reject bools
        self.assertEqual(_coerce_scalar_to_kind("123456789", "float64"), 123456789.0)
        self.assertEqual(_coerce_scalar_to_kind("1.5", "float64"), 1.5)
        self.assertEqual(_coerce_scalar_to_kind(7, "float64"), 7.0)
        self.assertIsNone(_coerce_scalar_to_kind("N/A", "float64"))
        self.assertIsNone(_coerce_scalar_to_kind(True, "float64"))
        self.assertIsNone(_coerce_scalar_to_kind(None, "float64"))
        # int kinds: truncate float strings, enforce int8 range
        self.assertEqual(_coerce_scalar_to_kind("42", "int64"), 42)
        self.assertEqual(_coerce_scalar_to_kind("42.9", "int64"), 42)
        self.assertIsNone(_coerce_scalar_to_kind("200", "int8"))
        self.assertEqual(_coerce_scalar_to_kind("100", "int8"), 100)
        self.assertIsNone(_coerce_scalar_to_kind("99999999999999999999999", "int64"))
        # bool: parse common truthy/falsey strings
        self.assertIs(_coerce_scalar_to_kind("true", "bool"), True)
        self.assertIs(_coerce_scalar_to_kind("0", "bool"), False)
        self.assertIsNone(_coerce_scalar_to_kind("maybe", "bool"))
        # string/json: stringify non-strings, JSON-encode containers
        self.assertEqual(_coerce_scalar_to_kind(123, "string"), "123")
        self.assertEqual(_coerce_scalar_to_kind({"a": 1}, "json"), '{"a":1}')

    def _write_table(self, uri, rows, schema):
        import lance
        import pyarrow as pa

        lance.write_dataset(pa.Table.from_pylist(rows, schema=schema), uri, mode="create")

    def test_coerce_drift_columns_rewrites_string_to_double(self):
        import lance
        import pyarrow as pa

        schema = pa.schema(
            [
                pa.field("_id", pa.string(), nullable=False),
                pa.field("claimMdId", pa.string()),
                pa.field("note", pa.string()),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            uri = f"{tmp}/claimmdsubmissionclaims.lance"
            self._write_table(
                uri,
                [
                    {"_id": "a", "claimMdId": "123456789", "note": "keep-a"},
                    {"_id": "b", "claimMdId": "not-a-number", "note": "keep-b"},
                    {"_id": "c", "claimMdId": None, "note": "keep-c"},
                ],
                schema,
            )
            specs = [
                ColumnSpec("_id", "string", required=True),
                ColumnSpec("claimMdId", "float64"),
                ColumnSpec("note", "string"),
            ]
            drift = [{"column": "claimMdId", "existing": "string", "desired": "double", "reason": "duckdb_rewrite_required"}]

            _coerce_drift_columns("claimmdsubmissionclaims", uri, specs, drift)

            ds = lance.dataset(uri)
            self.assertEqual(ds.count_rows(), 3)
            self.assertTrue(pa.types.is_floating(ds.schema.field("claimMdId").type))
            rows = {r["_id"]: r for r in ds.to_table().to_pylist()}
            self.assertEqual(rows["a"]["claimMdId"], 123456789.0)
            self.assertIsNone(rows["b"]["claimMdId"])  # unparseable -> null
            self.assertIsNone(rows["c"]["claimMdId"])
            # non-drift column preserved exactly
            self.assertEqual(rows["a"]["note"], "keep-a")
            self.assertEqual(rows["b"]["note"], "keep-b")

    def test_coerce_drift_columns_refuses_total_destruction(self):
        import pyarrow as pa

        schema = pa.schema([pa.field("_id", pa.string(), nullable=False), pa.field("code", pa.string())])
        with tempfile.TemporaryDirectory() as tmp:
            uri = f"{tmp}/records.lance"
            self._write_table(uri, [{"_id": "a", "code": "xyz"}, {"_id": "b", "code": "pqr"}], schema)
            specs = [ColumnSpec("_id", "string", required=True), ColumnSpec("code", "float64")]
            drift = [{"column": "code", "existing": "string", "desired": "double", "reason": "duckdb_rewrite_required"}]

            with self.assertRaises(RuntimeError) as ctx:
                _coerce_drift_columns("records", uri, specs, drift)

            self.assertIn("code", str(ctx.exception))

    def test_reconcile_lance_schema_coerces_string_to_double_end_to_end(self):
        import lance
        import pyarrow as pa

        schema = pa.schema(
            [
                pa.field("_id", pa.string(), nullable=False),
                pa.field("claimMdId", pa.string()),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            uri = f"{tmp}/claimmdsubmissionclaims.lance"
            self._write_table(uri, [{"_id": "a", "claimMdId": "42"}, {"_id": "b", "claimMdId": "bad"}], schema)
            args = Namespace(
                unknown_table_policy="fail",
                auto_recreate_empty_schema_drift=True,
                coerce_schema_type_drift=True,
            )
            table_schema = {
                "type": "object",
                "properties": {"claimMdId": {"type": "number"}},
            }

            specs = reconcile_lance_schema(args, "claimmdsubmissionclaims", uri, table_schema)

            self.assertIn("claimMdId", [spec.name for spec in specs])
            ds = lance.dataset(uri)
            self.assertTrue(pa.types.is_floating(ds.schema.field("claimMdId").type))
            rows = {r["_id"]: r for r in ds.to_table().to_pylist()}
            self.assertEqual(rows["a"]["claimMdId"], 42.0)
            self.assertIsNone(rows["b"]["claimMdId"])

    def test_reconcile_lance_schema_can_disable_coercion(self):
        import pyarrow as pa

        schema = pa.schema(
            [
                pa.field("_id", pa.string(), nullable=False),
                pa.field("claimMdId", pa.string()),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            uri = f"{tmp}/claimmdsubmissionclaims.lance"
            self._write_table(uri, [{"_id": "a", "claimMdId": "42"}], schema)
            args = Namespace(
                unknown_table_policy="fail",
                auto_recreate_empty_schema_drift=True,
                coerce_schema_type_drift=False,
            )
            table_schema = {"type": "object", "properties": {"claimMdId": {"type": "number"}}}

            with self.assertRaises(RuntimeError) as ctx:
                reconcile_lance_schema(args, "claimmdsubmissionclaims", uri, table_schema)

            self.assertIn("claimMdId", str(ctx.exception))


class IncrementalLoopThrottleTest(unittest.TestCase):
    def test_hit_page_cap_signals_backlog(self):
        args = Namespace(max_pages_per_sync=100)
        # Full batch -> backlog remains -> skip the idle sleep.
        self.assertTrue(_incremental_pass_hit_page_cap(args, {"pages": 100}))
        self.assertTrue(_incremental_pass_hit_page_cap(args, {"pages": 137}))
        # Partial batch -> caught up -> sleep.
        self.assertFalse(_incremental_pass_hit_page_cap(args, {"pages": 42}))
        self.assertFalse(_incremental_pass_hit_page_cap(args, {"pages": 0}))
        # No stats (e.g. lease unavailable) -> sleep/back off.
        self.assertFalse(_incremental_pass_hit_page_cap(args, {}))
        self.assertFalse(_incremental_pass_hit_page_cap(args, None))

    def test_no_page_cap_configured_always_sleeps(self):
        self.assertFalse(_incremental_pass_hit_page_cap(Namespace(max_pages_per_sync=0), {"pages": 500}))
        self.assertFalse(_incremental_pass_hit_page_cap(Namespace(max_pages_per_sync=None), {"pages": 500}))



class FakeCursorState:
    """In-memory S3JsonState stand-in that enforces conditional writes."""

    def __init__(self):
        self.values = {}
        self.etags = {}
        self._etag_seq = 0
        self.write_calls = []
        # Spy on the conditional write so tests can use call_count.
        self.write_if_match = MagicMock(wraps=self._write_if_match_impl)

    def _next_etag(self):
        etag = f"etag-{self._etag_seq}"
        self._etag_seq += 1
        return etag

    def seed(self, key, payload):
        self.values[key] = payload
        self.etags[key] = self._next_etag()

    def read(self, name):
        return self.values.get(name)

    def read_with_etag(self, name):
        return self.values.get(name), self.etags.get(name)

    def write(self, name, value):
        self.values[name] = value
        new_etag = self._next_etag()
        self.etags[name] = new_etag
        return new_etag

    def _write_if_match_impl(self, name, value, etag):
        if self.etags.get(name) != etag:
            raise RuntimeError(f"FakeCursorState precondition failed for {name}")
        self.values[name] = value
        self.write_calls.append((name, value, etag))
        new_etag = self._next_etag()
        self.etags[name] = new_etag
        return new_etag

    def delete(self, name):
        self.values.pop(name, None)
        self.etags.pop(name, None)

    def try_create(self, name, value):
        if name in self.values:
            return False
        self.values[name] = value
        self.etags[name] = self._next_etag()
        return True


class FakeConvexClient:
    def __init__(self, pages):
        self.pages = pages

    def iter_document_deltas(self, cursor):
        for page in self.pages:
            yield page


class BufferedIncrementalMergeTest(unittest.TestCase):
    def setUp(self):
        self.events = []

    def _record_event(self, event, **fields):
        self.events.append((event, fields))

    @contextmanager
    def _patched_run(self, state, client, merge_side_effect=None, plain_merge_side_effect=None):
        merge_mock = (
            MagicMock(side_effect=merge_side_effect)
            if merge_side_effect is not None
            else MagicMock(return_value=(1, {"schemas": {}}))
        )
        plain_merge_mock = (
            MagicMock(side_effect=plain_merge_side_effect)
            if plain_merge_side_effect is not None
            else MagicMock(return_value=1)
        )
        with (
            patch("convexlance.cli._incremental_state", return_value=state),
            patch("convexlance.cli._convex_client", return_value=client),
            patch("convexlance.cli.acquire_lease", return_value=TableLease(owner="test-owner", table_name="incremental")),
            patch("convexlance.cli.release_lease"),
            patch("convexlance.cli.verify_lease_owner"),
            patch("convexlance.cli.heartbeat_lease"),
            patch("convexlance.cli.reconcile_lance_schema", return_value=[ColumnSpec("value", "string")]),
            patch("convexlance.cli.merge_incremental_rows_with_schema_refresh", merge_mock),
            patch("convexlance.cli.merge_incremental_rows", plain_merge_mock),
            patch("convexlance.cli.log_event", side_effect=self._record_event),
        ):
            yield merge_mock, plain_merge_mock

    def _base_args(self, **overrides):
        defaults = {
            "lance_bucket": "b",
            "lance_scope": None,
            "lance_root_uri": None,
            "catalog_alias": None,
            "aws_region": None,
            "cursor_key": "incremental/cursor.json",
            "lance_prefix": "tables",
            "reconcile_schema": False,
            "reconcile_existing_tables_on_schema_refresh": False,
            "max_pages_per_sync": 100,
            "lock_ttl_seconds": 300,
            "force": False,
            "table_config_indexes": False,
            "merge_pages": 1,
            "merge_max_rows": 5000,
            "merge_table_concurrency": 1,
        }
        defaults.update(overrides)
        return Namespace(**defaults)

    def test_batching_reduces_merge_calls(self):
        pages = []
        for i in range(1, 11):
            rows = [
                {"_table": "A", "_id": f"a{i}", "_ts": i, "_deleted": False},
                {"_table": "B", "_id": f"b{i}", "_ts": i, "_deleted": False},
            ]
            pages.append((rows, f"cursor-{i}", i < 10))
        client = FakeConvexClient(pages)
        state = FakeCursorState()
        state.seed("incremental/cursor.json", {"cursor": "cursor-0"})
        args = self._base_args(merge_pages=10, merge_max_rows=10_000_000, merge_table_concurrency=1)

        with self._patched_run(state, client) as (merge_mock, _plain):
            result = run_incremental_once(args)

        self.assertEqual(merge_mock.call_count, 2)
        self.assertEqual(state.write_if_match.call_count, 1)
        self.assertEqual(state.values["incremental/cursor.json"]["cursor"], "cursor-10")
        self.assertEqual(result["pages"], 10)

    def test_row_cap_flushes_early(self):
        pages = []
        for i in range(1, 4):
            rows = [{"_table": "T", "_id": f"r{i}-{j}", "_ts": i, "_deleted": False} for j in range(6)]
            pages.append((rows, f"cursor-{i}", i < 3))
        client = FakeConvexClient(pages)
        state = FakeCursorState()
        state.seed("incremental/cursor.json", {"cursor": "cursor-0"})
        args = self._base_args(merge_pages=100, merge_max_rows=5, merge_table_concurrency=1)

        with self._patched_run(state, client) as (merge_mock, _plain):
            result = run_incremental_once(args)

        self.assertEqual(result["pages"], 3)
        self.assertLess(result["pages"], 100)
        self.assertGreaterEqual(state.write_if_match.call_count, 1)
        self.assertEqual(state.write_if_match.call_count, result["pages"])
        self.assertEqual(merge_mock.call_count, result["pages"])

    def test_failed_merge_does_not_commit_cursor(self):
        pages = [
            ([{"_table": "T", "_id": "r1", "_ts": 1, "_deleted": False}], "cursor-1", True),
            ([{"_table": "T", "_id": "r2", "_ts": 2, "_deleted": False}], "cursor-2", False),
        ]
        client = FakeConvexClient(pages)
        state = FakeCursorState()
        state.seed("incremental/cursor.json", {"cursor": "cursor-0"})
        args = self._base_args(merge_pages=100, merge_max_rows=10000, merge_table_concurrency=1)

        def boom(*args, **kwargs):
            raise RuntimeError("boom")

        with self._patched_run(state, client, merge_side_effect=boom) as (merge_mock, _plain):
            with self.assertRaises(RuntimeError) as ctx:
                run_incremental_once(args)
            self.assertIn("boom", str(ctx.exception))

        self.assertEqual(state.write_if_match.call_count, 0)
        self.assertIsNotNone(state.read("incremental/last_error.json"))

    def test_parallel_table_merge_waits_for_all(self):
        completed = []
        lock = threading.Lock()

        def fake_plain_merge(physical, target_uri, table_rows, specs):
            time.sleep(0.05)
            with lock:
                completed.append(physical)
            return 1

        pages = [
            (
                [
                    {"_table": "A", "_id": "a1", "_ts": 1, "_deleted": False},
                    {"_table": "B", "_id": "b1", "_ts": 1, "_deleted": False},
                ],
                "cursor-1",
                False,
            ),
        ]
        client = FakeConvexClient(pages)
        state = FakeCursorState()
        state.seed("incremental/cursor.json", {"cursor": "cursor-0"})
        args = self._base_args(merge_pages=10, merge_max_rows=10000, merge_table_concurrency=2)

        with self._patched_run(state, client, plain_merge_side_effect=fake_plain_merge) as (_refresh, plain_mock):
            result = run_incremental_once(args)

        # Cursor commits only after both parallel table merges finished.
        self.assertEqual(sorted(completed), ["a", "b"])
        self.assertEqual(state.write_if_match.call_count, 1)
        self.assertEqual(plain_mock.call_count, 2)
        self.assertEqual(result["pages"], 1)

    def test_parallel_table_merge_failure_no_commit(self):
        completed = []
        lock = threading.Lock()

        def fake_plain_merge(physical, target_uri, table_rows, specs):
            with lock:
                completed.append(physical)
            if physical == "b":
                raise RuntimeError("boom")
            return 1

        pages = [
            (
                [
                    {"_table": "A", "_id": "a1", "_ts": 1, "_deleted": False},
                    {"_table": "B", "_id": "b1", "_ts": 1, "_deleted": False},
                ],
                "cursor-1",
                False,
            ),
        ]
        client = FakeConvexClient(pages)
        state = FakeCursorState()
        state.seed("incremental/cursor.json", {"cursor": "cursor-0"})
        args = self._base_args(merge_pages=10, merge_max_rows=10000, merge_table_concurrency=2)

        with self._patched_run(state, client, plain_merge_side_effect=fake_plain_merge) as (_refresh, _plain):
            with self.assertRaises(RuntimeError) as ctx:
                run_incremental_once(args)
            self.assertIn("boom", str(ctx.exception))

        self.assertEqual(state.write_if_match.call_count, 0)

    def test_parallel_missing_columns_refreshes_then_retries_sequentially(self):
        # A parallel plain merge that reports missing columns must abort the
        # chunk, refresh schema once in the main thread, and retry via the
        # refreshing (sequential) path, committing the cursor exactly once.
        def fake_plain_merge(physical, target_uri, table_rows, specs):
            raise MissingLanceColumns(physical, target_uri, ["newField"])

        pages = [
            (
                [
                    {"_table": "A", "_id": "a1", "_ts": 1, "_deleted": False},
                    {"_table": "B", "_id": "b1", "_ts": 1, "_deleted": False},
                ],
                "cursor-1",
                False,
            ),
        ]
        client = FakeConvexClient(pages)
        state = FakeCursorState()
        state.seed("incremental/cursor.json", {"cursor": "cursor-0"})
        args = self._base_args(merge_pages=10, merge_max_rows=10000, merge_table_concurrency=2, reconcile_schema=True)

        with (
            self._patched_run(state, client, plain_merge_side_effect=fake_plain_merge) as (refresh_mock, plain_mock),
            patch("convexlance.cli.incremental_schema_payload_with_status", return_value=({"schemas": {}}, True)) as schema_refresh,
        ):
            result = run_incremental_once(args)

        # Sequential retry ran both tables through the refreshing merge.
        self.assertEqual(refresh_mock.call_count, 2)
        # Startup reconcile + one forced refresh during the fallback.
        self.assertTrue(any(c.kwargs.get("force_refresh") for c in schema_refresh.call_args_list))
        self.assertEqual(state.write_if_match.call_count, 1)
        self.assertEqual(state.values["incremental/cursor.json"]["cursor"], "cursor-1")
        self.assertEqual(result["pages"], 1)

    def test_backward_compatibility_per_page(self):
        pages = []
        for i in range(1, 4):
            pages.append(
                (
                    [{"_table": "T", "_id": f"r{i}", "_ts": i, "_deleted": False}],
                    f"cursor-{i}",
                    i < 3,
                )
            )
        client = FakeConvexClient(pages)
        state = FakeCursorState()
        state.seed("incremental/cursor.json", {"cursor": "cursor-0"})
        args = self._base_args(merge_pages=1, merge_max_rows=10000, merge_table_concurrency=1)

        with self._patched_run(state, client) as (merge_mock, _plain):
            result = run_incremental_once(args)

        self.assertEqual(result["pages"], 3)
        self.assertEqual(merge_mock.call_count, 3)
        self.assertEqual(state.write_if_match.call_count, 3)
        audit_keys = list(state.values.keys())
        self.assertTrue(any("page_audit" in k for k in audit_keys))
        self.assertFalse(any("chunk_audit" in k for k in audit_keys))

    def _page_events(self):
        return [fields for event, fields in self.events if event == "lance_incremental_page"]

    def test_single_page_event_omits_buffered_field(self):
        pages = [([{"_table": "T", "_id": "r1", "_ts": 1, "_deleted": False}], "cursor-1", False)]
        client = FakeConvexClient(pages)
        state = FakeCursorState()
        state.seed("incremental/cursor.json", {"cursor": "cursor-0"})
        args = self._base_args(merge_pages=1, merge_max_rows=10000, merge_table_concurrency=1)

        with self._patched_run(state, client):
            run_incremental_once(args)

        page_events = self._page_events()
        self.assertEqual(len(page_events), 1)
        self.assertNotIn("buffered", page_events[0])

    def test_multi_page_chunk_events_flag_buffered(self):
        pages = [
            ([{"_table": "T", "_id": "r1", "_ts": 1, "_deleted": False}], "cursor-1", True),
            ([{"_table": "T", "_id": "r2", "_ts": 2, "_deleted": False}], "cursor-2", False),
        ]
        client = FakeConvexClient(pages)
        state = FakeCursorState()
        state.seed("incremental/cursor.json", {"cursor": "cursor-0"})
        args = self._base_args(merge_pages=10, merge_max_rows=10000, merge_table_concurrency=1)

        with self._patched_run(state, client):
            run_incremental_once(args)

        page_events = self._page_events()
        self.assertEqual(len(page_events), 2)
        self.assertTrue(all(fields.get("buffered") is True for fields in page_events))

    def test_parallel_missing_columns_reraises_original_when_reconcile_disabled(self):
        def fake_plain_merge(physical, target_uri, table_rows, specs):
            raise MissingLanceColumns(physical, target_uri, ["newField"])

        pages = [
            (
                [
                    {"_table": "A", "_id": "a1", "_ts": 1, "_deleted": False},
                    {"_table": "B", "_id": "b1", "_ts": 1, "_deleted": False},
                ],
                "cursor-1",
                False,
            ),
        ]
        client = FakeConvexClient(pages)
        state = FakeCursorState()
        state.seed("incremental/cursor.json", {"cursor": "cursor-0"})
        args = self._base_args(merge_pages=10, merge_max_rows=10000, merge_table_concurrency=2, reconcile_schema=False)

        with self._patched_run(state, client, plain_merge_side_effect=fake_plain_merge):
            with self.assertRaises(MissingLanceColumns) as ctx:
                run_incremental_once(args)

        self.assertEqual(ctx.exception.columns, ["newField"])
        self.assertEqual(state.write_if_match.call_count, 0)


class PrepareIncrementalMergeRowsTest(unittest.TestCase):
    def test_same_id_across_pages(self):
        incoming = [
            {"_id": "a", "_ts": 100, "_deleted": False},
            {"_id": "a", "_ts": 200, "_deleted": False},
        ]
        existing = [{"_id": "a", "_ts": 50, "_deleted": False}]
        rows = prepare_incremental_merge_rows(incoming, existing)
        by_id_ts = {row["__id_ts"]: row for row in rows}

        self.assertEqual(len(by_id_ts), 3)
        self.assertEqual(by_id_ts["a#200"]["__status"] & 1, 1)
        self.assertEqual(by_id_ts["a#100"]["__status"] & 1, 0)
        self.assertEqual(by_id_ts["a#50"]["__status"] & 1, 0)

if __name__ == "__main__":
    unittest.main()
