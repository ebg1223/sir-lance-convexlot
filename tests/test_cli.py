from argparse import Namespace
import tempfile
import unittest
from unittest.mock import patch

from convexlance.cli import (
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
    generated_index_columns,
    infer_kind,
    load_table_config,
    merge_incremental_rows_with_schema_refresh,
    missing_lance_columns_for_rows,
    normalize_rows_for_specs,
    physical_to_source_schema_map,
    prepare_incremental_merge_rows,
    reconcile_schema_payload_existing_lance_tables,
    read_applied_table_config_version,
    reconcile_table_config_for_dataset,
    requested_index_specs,
    schema_column_specs,
    table_config_state_key,
    validate_repair_columns,
    write_applied_table_config_state,
    _add_lance_columns_native,
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


if __name__ == "__main__":
    unittest.main()
