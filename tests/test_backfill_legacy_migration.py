"""回填迁移回归测试：旧 Schema 库必须能直接补上 provenance 三列。

对应 indicators PR #19 的 P2 评审意见 "Migrate build-identity columns
before inserting provenance"：当 --apply 跑在早于这三列的
daily_ts_zscore 上时，仅靠 CREATE TABLE IF NOT EXISTS 不会补列，
新 metadata 插入会因缺列失败。

本测试断言 prepare_target_table(apply=True) 走的 ensure_table() 确实
包含幂等的 ALTER ... ADD COLUMN IF NOT EXISTS，因此旧库无需先跑
init-db 也能继续回填。
"""
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_backfill_module():
    path = REPO_ROOT / "scripts" / "backfill_zscore_2015.py"
    spec = importlib.util.spec_from_file_location("backfill_zscore_2015", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PROVENANCE_COLUMNS = ("factor_set_id", "algorithm_version", "build_id")


class TestLegacySchemaMigration:
    def setup_method(self):
        self.mod = _load_backfill_module()

    def test_migration_sql_covers_all_provenance_columns(self):
        """ENSURE_MIGRATION_SQL 必须 ADD COLUMN IF NOT EXISTS 三列。"""
        sql = self.mod.ENSURE_MIGRATION_SQL
        assert "ALTER TABLE" in sql
        assert "IF NOT EXISTS" in sql
        for col in PROVENANCE_COLUMNS:
            assert col in sql, f"迁移缺少 {col}"

    def test_ensure_table_runs_migration(self):
        """ensure_table() 必须真正执行迁移，而不只是 CREATE IF NOT EXISTS。"""
        calls = []

        class _FakeDb:
            @staticmethod
            def execute_sql(sql, *a, **kw):
                calls.append(sql)

        original = self.mod.db
        self.mod.db = _FakeDb
        try:
            self.mod.ensure_table()
        finally:
            self.mod.db = original

        joined = "\n".join(calls)
        assert any("CREATE SCHEMA" in c for c in calls)
        assert any("daily_ts_zscore" in c and "CREATE TABLE" in c.replace("\n", " ")
                   or "CREATE TABLE IF NOT EXISTS indicators.daily_ts_zscore" in c
                   for c in calls), "未创建表"
        assert self.mod.ENSURE_MIGRATION_SQL in calls, \
            "ensure_table 未执行 provenance 三列的幂等迁移"

    def test_apply_mode_calls_ensure_table(self):
        """--apply 路径必须走 ensure_table（迁移与建表一起）。"""
        called = []

        original = self.mod.ensure_table
        self.mod.ensure_table = lambda: called.append(True)
        try:
            self.mod.prepare_target_table(apply=True)
        finally:
            self.mod.ensure_table = original
        assert called == [True]

    def test_dry_run_is_read_only(self):
        """dry-run 不得建表/迁移。"""
        def _boom():
            raise AssertionError("dry-run 不得调用 ensure_table")

        original_ensure = self.mod.ensure_table
        original_exists = self.mod.target_table_exists
        self.mod.ensure_table = _boom
        self.mod.target_table_exists = lambda: True
        try:
            self.mod.prepare_target_table(apply=False)  # 表存在 → 直接返回
        finally:
            self.mod.ensure_table = original_ensure
            self.mod.target_table_exists = original_exists