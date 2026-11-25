"""Database connection and operations."""
import psycopg2
from psycopg2.extras import execute_values
from contextlib import contextmanager
from typing import Optional
import pandas as pd
import numpy as np
import time
from io import StringIO

from .. import config

# Chunked-COPY retry policy for transient connection/DNS blips.
_COPY_MAX_RETRY = 6
_COPY_BACKOFF = (5, 15, 30, 60, 120, 180)  # seconds between attempts


@contextmanager
def get_connection():
    """Get database connection context manager."""
    conn = psycopg2.connect(
        host=config.PG_HOST,
        port=config.PG_PORT,
        user=config.PG_USER,
        password=config.PG_PASSWORD,
        database=config.PG_DATABASE,
        sslmode=config.PG_SSLMODE,
    )
    try:
        yield conn
    finally:
        conn.close()


def execute_sql(sql: str, params: Optional[tuple] = None, fetch: bool = False):
    """Execute SQL statement."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            if fetch:
                return cur.fetchall()
            conn.commit()


def read_sql(query: str, params: Optional[tuple] = None) -> pd.DataFrame:
    """Read SQL query into DataFrame."""
    with get_connection() as conn:
        return pd.read_sql(query, conn, params=params)


def copy_from_dataframe(
    df: pd.DataFrame,
    table: str,
    schema: str = "indicators",
    if_exists: str = "append",
    chunk_size: int = 100_000,
):
    """
    Bulk insert a DataFrame via chunked COPY + temp-table upsert.

    The frame is split into chunks of at most ``chunk_size`` rows; each chunk
    uses its own short-lived connection and commits independently. This keeps
    every single DB operation small, which avoids the
    ``server closed the connection unexpectedly`` errors that occur when a
    single huge COPY overwhelms the front-end proxy (e.g. 3.7M rows in one
    statement), and makes a failed batch safely resumable: upserts are
    idempotent, so a re-run simply overwrites already-written chunks.

    Args:
        df: rows to write
        table: target table name
        schema: target schema
        if_exists: ignored (always upsert/append)
        chunk_size: max rows per COPY statement
    """
    if df.empty:
        return

    temp_table = f"_temp_{table}"

    # Resolve target table column order once and reconcile the frame to it.
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = '{schema}' AND table_name = '{table}'
                ORDER BY ordinal_position
            """)
            target_cols = [r[0] for r in cur.fetchall()]
    for col in target_cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[target_cols]

    columns = list(df.columns)
    col_str = ", ".join(columns)
    n = len(df)
    n_chunks = (n + chunk_size - 1) // chunk_size

    if n_chunks > 1:
        print(f"    {table}: {n} rows in {n_chunks} chunks (<= {chunk_size})")

    for ci in range(n_chunks):
        start = ci * chunk_size
        end = min(start + chunk_size, n)
        _copy_chunk(
            df.iloc[start:end], table, schema, temp_table,
            col_str, columns, ci + 1, n_chunks,
        )


def _copy_chunk(sub, table, schema, temp_table, col_str, columns, ci, n_chunks):
    """COPY + upsert a single chunk, retrying transient connection errors.

    Retries up to ``_COPY_MAX_RETRY`` times with exponential back-off so a
    brief DNS/network blip (the front-end proxy host becoming temporarily
    unresolvable) is ridden out instead of aborting the whole batch.
    """
    last_err = None
    for attempt in range(_COPY_MAX_RETRY):
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"DROP TABLE IF EXISTS {temp_table}")
                    cur.execute(
                        f"CREATE TEMP TABLE {temp_table} "
                        f"(LIKE {schema}.{table} INCLUDING ALL)"
                    )

                    output = StringIO()
                    sub.to_csv(
                        output, sep="\t", header=False,
                        index=False, na_rep="\\N",
                    )
                    output.seek(0)
                    cur.copy_from(output, temp_table, null="\\N")

                    # Resolve primary key for the upsert conflict target.
                    cur.execute(f"""
                        SELECT a.attname
                        FROM pg_index i
                        JOIN pg_attribute a
                            ON a.attrelid = i.indrelid
                            AND a.attnum = ANY(i.indkey)
                        WHERE i.indrelid = '{schema}.{table}'::regclass
                            AND i.indisprimary
                    """)
                    pk_cols = [row[0] for row in cur.fetchall()]
                    pk_str = ", ".join(pk_cols)

                    update_cols = [c for c in columns if c not in pk_cols]
                    if update_cols:
                        update_str = ", ".join(
                            f"{c} = EXCLUDED.{c}" for c in update_cols
                        )
                        upsert_sql = f"""
                            INSERT INTO {schema}.{table} ({col_str})
                            SELECT {col_str} FROM {temp_table}
                            ON CONFLICT ({pk_str}) DO UPDATE SET {update_str}
                        """
                    else:
                        upsert_sql = f"""
                            INSERT INTO {schema}.{table} ({col_str})
                            SELECT {col_str} FROM {temp_table}
                            ON CONFLICT ({pk_str}) DO NOTHING
                        """
                    cur.execute(upsert_sql)
                    cur.execute(f"DROP TABLE IF EXISTS {temp_table}")
                conn.commit()
            return
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            last_err = e
            if attempt < _COPY_MAX_RETRY - 1:
                wait = _COPY_BACKOFF[min(attempt, len(_COPY_BACKOFF) - 1)]
                print(
                    f"    [retry {attempt + 1}/{_COPY_MAX_RETRY}] chunk "
                    f"{ci}/{n_chunks} COPY failed: {e}; retry in {wait}s"
                )
                time.sleep(wait)
            else:
                print(
                    f"    [retry {attempt + 1}/{_COPY_MAX_RETRY}] chunk "
                    f"{ci}/{n_chunks} COPY failed: {e}; giving up"
                )
    raise last_err


def init_schema():
    """Initialize indicators schema and tables."""
    schema_sql = """
    CREATE SCHEMA IF NOT EXISTS indicators;
    """
    execute_sql(schema_sql)

    # Create tables
    tables_sql = """
    -- Meta table: indicator registry
    CREATE TABLE IF NOT EXISTS indicators.meta (
        indicator_name TEXT PRIMARY KEY,
        category TEXT NOT NULL,
        display_name TEXT,
        formula TEXT,
        warmup_days INTEGER,
        talib_function TEXT,
        parameters JSONB,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    -- Base aligned daily data
    CREATE TABLE IF NOT EXISTS indicators.daily_base_aligned (
        code TEXT NOT NULL,
        date INTEGER NOT NULL,
        open NUMERIC(12,2),
        high NUMERIC(12,2),
        low NUMERIC(12,2),
        close NUMERIC(12,2),
        volume NUMERIC(20,2),
        amount NUMERIC(20,2),
        pre_close NUMERIC(12,2),
        simple_return DOUBLE PRECISION,
        log_return DOUBLE PRECISION,
        is_st BOOLEAN,
        is_limit_up BOOLEAN,
        is_limit_down BOOLEAN,
        is_censored BOOLEAN,
        PRIMARY KEY (code, date)
    );

    -- Volatility indicators
    CREATE TABLE IF NOT EXISTS indicators.daily_volatility (
        code TEXT NOT NULL,
        date INTEGER NOT NULL,
        period INTEGER NOT NULL,
        vol_ccc DOUBLE PRECISION,
        vol_parkinson DOUBLE PRECISION,
        vol_gk DOUBLE PRECISION,
        vol_rs DOUBLE PRECISION,
        vol_ccc_rank DOUBLE PRECISION,
        vol_parkinson_rank DOUBLE PRECISION,
        vol_gk_rank DOUBLE PRECISION,
        vol_rs_rank DOUBLE PRECISION,
        PRIMARY KEY (code, date, period)
    );

    -- Momentum indicators
    CREATE TABLE IF NOT EXISTS indicators.daily_momentum (
        code TEXT NOT NULL,
        date INTEGER NOT NULL,
        indicator_name TEXT NOT NULL,
        value DOUBLE PRECISION,
        cross_section_rank DOUBLE PRECISION,
        cross_section_zscore DOUBLE PRECISION,
        PRIMARY KEY (code, date, indicator_name)
    );

    -- Volume indicators
    CREATE TABLE IF NOT EXISTS indicators.daily_volume_indicators (
        code TEXT NOT NULL,
        date INTEGER NOT NULL,
        indicator_name TEXT NOT NULL,
        value DOUBLE PRECISION,
        cross_section_rank DOUBLE PRECISION,
        cross_section_zscore DOUBLE PRECISION,
        PRIMARY KEY (code, date, indicator_name)
    );

    -- Price position indicators
    CREATE TABLE IF NOT EXISTS indicators.daily_price_position (
        code TEXT NOT NULL,
        date INTEGER NOT NULL,
        indicator_name TEXT NOT NULL,
        value DOUBLE PRECISION,
        cross_section_rank DOUBLE PRECISION,
        cross_section_zscore DOUBLE PRECISION,
        PRIMARY KEY (code, date, indicator_name)
    );

    -- Cross-section statistics
    CREATE TABLE IF NOT EXISTS indicators.cross_section_stats (
        date INTEGER NOT NULL,
        indicator_name TEXT NOT NULL,
        count INTEGER,
        mean DOUBLE PRECISION,
        std DOUBLE PRECISION,
        min DOUBLE PRECISION,
        q25 DOUBLE PRECISION,
        median DOUBLE PRECISION,
        q75 DOUBLE PRECISION,
        max DOUBLE PRECISION,
        PRIMARY KEY (date, indicator_name)
    );

    -- Batch checkpoint: marks whole batches fully written so a resumed run
    -- can skip completed batches (and re-run only unfinished ones).
    CREATE TABLE IF NOT EXISTS indicators._batch_checkpoint (
        batch_id    INTEGER PRIMARY KEY,
        n_stocks    INTEGER,
        status      TEXT,
        updated_at  TIMESTAMPTZ DEFAULT NOW()
    );
    """

    for stmt in tables_sql.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            execute_sql(stmt + ";")

    # Create indexes
    indexes_sql = """
    CREATE INDEX IF NOT EXISTS idx_daily_base_aligned_date ON indicators.daily_base_aligned (date);
    CREATE INDEX IF NOT EXISTS idx_daily_volatility_date ON indicators.daily_volatility (date);
    CREATE INDEX IF NOT EXISTS idx_daily_momentum_date ON indicators.daily_momentum (date);
    CREATE INDEX IF NOT EXISTS idx_daily_volume_date ON indicators.daily_volume_indicators (date);
    CREATE INDEX IF NOT EXISTS idx_daily_price_position_date ON indicators.daily_price_position (date);
    """

    for stmt in indexes_sql.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            execute_sql(stmt + ";")


def ensure_checkpoint_table():
    """Create the batch checkpoint table if it does not exist.

    batch_id 编码含计算范围（start_date*10000 + batch_seq, BIGINT），
    避免"换 start_date 重跑被旧 checkpoint 错误跳过"。旧 INTEGER 表自动升级。
    """
    execute_sql(
        """
        CREATE TABLE IF NOT EXISTS indicators._batch_checkpoint (
            batch_id    BIGINT PRIMARY KEY,
            n_stocks    INTEGER,
            status      TEXT,
            updated_at  TIMESTAMPTZ DEFAULT NOW()
        )
        """
    )
    execute_sql(
        "ALTER TABLE indicators._batch_checkpoint ALTER COLUMN batch_id TYPE BIGINT"
    )


def is_batch_done(batch_id: int) -> bool:
    """Return True if a batch was fully written (checkpoint status = 'done')."""
    try:
        rows = execute_sql(
            "SELECT status FROM indicators._batch_checkpoint WHERE batch_id=%s",
            (batch_id,),
            fetch=True,
        )
        return bool(rows) and rows[0][0] == "done"
    except Exception:
        return False


def mark_batch_done(batch_id: int, n_stocks: int):
    """Record a batch as fully written."""
    execute_sql(
        """
        INSERT INTO indicators._batch_checkpoint (batch_id, n_stocks, status, updated_at)
        VALUES (%s, %s, 'done', NOW())
        ON CONFLICT (batch_id) DO UPDATE
            SET n_stocks = EXCLUDED.n_stocks, status = 'done', updated_at = NOW()
        """,
        (batch_id, n_stocks),
    )


def has_range_checkpoints(range_key: int) -> bool:
    """Return True if any checkpoint already uses the range-encoded key space."""
    try:
        rows = execute_sql(
            "SELECT 1 FROM indicators._batch_checkpoint WHERE batch_id >= %s LIMIT 1",
            (range_key,),
            fetch=True,
        )
        return bool(rows)
    except Exception:
        return False


def clear_checkpoints():
    """Drop all batch checkpoints（键语义变更/全量重算前使用）."""
    execute_sql("DELETE FROM indicators._batch_checkpoint")


def update_cross_section_stats(df: pd.DataFrame, table: str) -> None:
    """Update cross-section rank and zscore for existing indicator rows.
    
    This is used in Phase B to update cross-section statistics after all raw
    indicators have been computed. The update is done via a temporary table
    and an UPDATE JOIN to ensure atomicity.
    
    Args:
        df: DataFrame with columns: code, date, indicator_name, cross_section_rank, cross_section_zscore
        table: Target table name (e.g., 'daily_momentum')
    """
    if df.empty:
        return
    
    temp_table = f"_temp_cs_{table}"
    
    # Process in chunks to avoid memory issues
    chunk_size = 100_000
    n_chunks = (len(df) + chunk_size - 1) // chunk_size
    
    if n_chunks > 1:
        print(f"    Updating {table} cross-section stats in {n_chunks} chunks")
    
    for ci in range(n_chunks):
        start_idx = ci * chunk_size
        end_idx = min(start_idx + chunk_size, len(df))
        chunk = df.iloc[start_idx:end_idx]
        
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    # Create temp table
                    cur.execute(f"DROP TABLE IF EXISTS {temp_table}")
                    cur.execute(f"""
                        CREATE TEMP TABLE {temp_table} (
                            code TEXT,
                            date INTEGER,
                            indicator_name TEXT,
                            cross_section_rank DOUBLE PRECISION,
                            cross_section_zscore DOUBLE PRECISION
                        )
                    """)
                    
                    # Copy data to temp table
                    output = StringIO()
                    chunk.to_csv(output, sep="\t", header=False, index=False, na_rep="\\N")
                    output.seek(0)
                    cur.copy_from(output, temp_table, null="\\N")
                    
                    # Update main table from temp table
                    cur.execute(f"""
                        UPDATE indicators.{table} AS t
                        SET cross_section_rank = tmp.cross_section_rank,
                            cross_section_zscore = tmp.cross_section_zscore
                        FROM {temp_table} AS tmp
                        WHERE t.code = tmp.code
                          AND t.date = tmp.date
                          AND t.indicator_name = tmp.indicator_name
                    """)
                    
                    # Drop temp table
                    cur.execute(f"DROP TABLE IF EXISTS {temp_table}")
                    
                conn.commit()
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            print(f"    [error] chunk {ci+1}/{n_chunks} failed: {e}")
            raise
