import contextlib
import io
import unittest

from convexlance.cli import (
    build_parser,
    create_indexes_for_columns,
    generated_index_columns,
    install_signal_handlers,
    prepare_incremental_merge_rows,
    request_shutdown,
    run_idle_maintenance_once,
    schema_column_specs,
    should_run_idle_maintenance,
    shutdown_requested,
    shutdown_signal,
    _shutdown_event,
)


class FakeIndexConn:
    def __init__(self):
        self.statements: list[str] = []

    def execute(self, sql: str):
        self.statements.append(sql)
        return self

    def fetchall(self):
        return [("__id_ts",), ("__status_id",), ("__status",)]


class FakeState:
    pass


class ConvexLanceCliTest(unittest.TestCase):
    def test_parser_exposes_only_incremental_commands(self):
        parser = build_parser()

        self.assertEqual(parser.parse_args(["incremental-once"]).__dict__["func"].__name__, "run_incremental_once")
        self.assertEqual(parser.parse_args(["incremental-loop"]).__dict__["func"].__name__, "run_incremental_loop")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["migrate-table"])

    def test_create_indexes_for_generated_columns(self):
        conn = FakeIndexConn()

        created, skipped = create_indexes_for_columns(conn, "records", "'s3://bucket/tables/records.lance'", generated_index_columns())

        self.assertEqual(created, 3)
        self.assertEqual(skipped, [])
        self.assertIn("CREATE INDEX records___id_ts_idx ON 's3://bucket/tables/records.lance' (__id_ts) USING BTREE", conn.statements)
        self.assertIn("CREATE INDEX records___status_id_idx ON 's3://bucket/tables/records.lance' (__status_id) USING BTREE", conn.statements)
        self.assertIn("CREATE INDEX records___status_idx ON 's3://bucket/tables/records.lance' (__status) USING BITMAP", conn.statements)

    def test_schema_column_specs_uses_int8_status(self):
        specs = {column.name: column for column in schema_column_specs({"type": "object", "properties": {}})}

        self.assertEqual(specs["__status"].kind, "int8")
        self.assertTrue(specs["__status"].required)

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
        self.assertEqual(by_version["a#200"]["__status"], 3)
        self.assertEqual(by_version["a#150"]["__status"], 0)
        self.assertEqual(by_version["b#50"]["__status"], 1)
        self.assertNotIn("_current", by_version["a#200"])

    def test_prepare_incremental_merge_rows_preserves_existing_current_for_older_delta(self):
        incoming = [{"_id": "a", "_ts": 100, "_deleted": False, "value": "old"}]
        existing = [{"_id": "a", "_ts": 150, "_deleted": False, "__id_ts": "a#150", "__status": 1, "__status_id": "1#a"}]

        rows = prepare_incremental_merge_rows(incoming, existing)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["__id_ts"], "a#100")
        self.assertEqual(rows[0]["__status"], 0)
        self.assertEqual(rows[0]["__status_id"], "0#a")

    def setUp(self):
        _shutdown_event.clear()

    def tearDown(self):
        _shutdown_event.clear()

    def test_signal_handler_requests_deferred_shutdown(self):
        request_shutdown(15, None)

        self.assertTrue(shutdown_requested())
        self.assertEqual(shutdown_signal(), "SIGTERM")

    def test_install_signal_handlers(self):
        install_signal_handlers()

    def test_idle_maintenance_skips_when_shutdown_requested(self):
        request_shutdown(15, None)
        args = build_parser().parse_args(["incremental-loop"])

        self.assertFalse(run_idle_maintenance_once(args, FakeState()))

    def test_idle_maintenance_not_selected_when_shutdown_requested(self):
        request_shutdown(15, None)
        args = build_parser().parse_args(["incremental-loop"])

        self.assertFalse(should_run_idle_maintenance(args, {"rows_accepted": 0, "pages": 1}))


if __name__ == "__main__":
    unittest.main()
