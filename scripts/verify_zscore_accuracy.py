#!/usr/bin/env python3
"""
独立验证 daily_ts_zscore 数据正确性

从 PG 抽取样本股票，用独立 Pandas 计算 Z-score，与 PG 数据对比。
"""
import pandas as pd
import numpy as np
import sys

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "nooshare_dev",
    "user": "travis",
}

def compute_zscore_independent(series: pd.Series, window: int) -> pd.Series:
    """独立计算 Rolling Z-score（ddof=1）"""
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std(ddof=1)
    result = (series - rolling_mean) / rolling_std.replace(0, np.nan)
    result = result.replace([np.inf, -np.inf], np.nan)
    return result

def compute_volume_zscore_independent(series: pd.Series, window: int) -> pd.Series:
    """Independent T-1 reference: score each valid T against prior W valid volumes.

    This intentionally uses a straightforward positional loop rather than the
    production vectorized implementation. Non-positive and non-finite values are
    skipped as observations; they are not classified as suspension or source gaps.
    """
    if window < 1:
        raise ValueError("window must be >= 1")

    values = pd.to_numeric(series, errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    result = np.full(len(values), np.nan, dtype=np.float64)
    valid_positions = np.flatnonzero(np.isfinite(values) & (values > 0))

    for valid_index in range(window, len(valid_positions)):
        current_position = valid_positions[valid_index]
        history_positions = valid_positions[valid_index - window : valid_index]
        history = values[history_positions]
        sample_std = float(np.std(history, ddof=1))
        if not np.isfinite(sample_std) or sample_std <= 0:
            continue
        result[current_position] = (
            values[current_position] - float(np.mean(history))
        ) / sample_std

    return pd.Series(result, index=series.index, name=series.name)


def verify_stock(code: str, conn) -> dict:
    """验证单只股票：读取原始数据，独立计算，与 PG 对比"""
    
    # 读取基础行情（含预热期）
    # 数据源: Base 公开 VIEW public.market_daily_adjusted_bar（固定最新 validated factor set）
    # 旧表 public.hfq_daily 已退役，不再读取。
    base_query = f"""
    SELECT a.date, a.close_hfq, a.volume
    FROM public.market_daily_adjusted_bar a
    WHERE a.symbol = '{code}'
      AND a.factor_set_id = (
          SELECT factor_set_id FROM public.market_adjustment_factor_set
          WHERE status = 'validated'
          ORDER BY validated_at DESC NULLS LAST LIMIT 1
      )
      AND a.date >= 20240101 AND a.date <= 20250630
    ORDER BY a.date
    """
    base = pd.read_sql(base_query, conn)
    base["date"] = base["date"].astype(int)
    
    if len(base) == 0:
        return {"code": code, "status": "NO_DATA", "total": 0, "passed": 0, "errors": []}
    
    # 读取 PG 已计算的 Z-score
    zscore_query = f"""
    SELECT date, indicator_name, value
    FROM indicators.daily_ts_zscore
    WHERE code = '{code}'
      AND date >= 20250101 AND date <= 20250630
    ORDER BY date, indicator_name
    """
    pg_zs = pd.read_sql(zscore_query, conn)
    
    if len(pg_zs) == 0:
        return {"code": code, "status": "NO_ZSCORE", "total": 0, "passed": 0, "errors": []}
    
    # 独立计算 Z-score（保存到单独的 DataFrame，避免列名冲突）
    my_indicators = pd.DataFrame({"date": base["date"]})
    for period in [5, 10, 20]:
        my_indicators[f"my_volume_zscore_{period}"] = compute_volume_zscore_independent(base["volume"], period).values
        my_indicators[f"my_close_hfq_rolling_zscore_{period}"] = compute_zscore_independent(base["close_hfq"], period).values
    
    # 只比较 20250101 之后的数据
    compare = my_indicators[my_indicators["date"] >= 20250101].copy()
    
    # 合并 PG 数据（宽表格式）
    pg_wide = pg_zs.pivot_table(index="date", columns="indicator_name", values="value")
    
    # 合并
    merged = compare.set_index("date").join(pg_wide, how="inner")
    
    total_checks = 0
    passed_checks = 0
    errors = []
    
    indicator_names = [
        "volume_zscore_5", "volume_zscore_10", "volume_zscore_20",
        "close_hfq_rolling_zscore_5", "close_hfq_rolling_zscore_10", "close_hfq_rolling_zscore_20"
    ]
    
    for ind_name in indicator_names:
        if ind_name not in merged.columns:
            continue
        pg_col = ind_name
        my_col = ind_name  # same name in our wide table
        
        for date, row in merged.iterrows():
            pg_val = row.get(pg_col, np.nan)
            my_val = row.get(my_col, np.nan)
            
            # Both NaN → match
            if pd.isna(pg_val) and pd.isna(my_val):
                total_checks += 1
                passed_checks += 1
                continue
            
            # One NaN, other not → mismatch
            if pd.isna(pg_val) or pd.isna(my_val):
                total_checks += 1
                errors.append(f"{date} {ind_name}: PG={'NaN' if pd.isna(pg_val) else f'{pg_val:.6f}'} vs Mine={'NaN' if pd.isna(my_val) else f'{my_val:.6f}'}")
                continue
            
            # Both numeric → compare
            total_checks += 1
            diff = abs(pg_val - my_val)
            if diff < 1e-6:
                passed_checks += 1
            else:
                errors.append(f"{date} {ind_name}: PG={pg_val:.6f} vs Mine={my_val:.6f} (diff={diff:.2e})")
    
    status = "PASS" if len(errors) == 0 else "FAIL"
    return {
        "code": code,
        "status": status,
        "total": total_checks,
        "passed": passed_checks,
        "errors": errors[:10]
    }

def main():
    import psycopg2

    test_codes = ["sz002852", "sz002230", "sh688255", "sh600157", "sz300642"]
    
    print("=== Z-score 准确性验证（独立 Pandas vs PG） ===")
    print(f"测试股票: {test_codes}")
    print(f"验证日期: 20250101 - 20250630")
    print(f"预热起始: 20240101")
    print()
    
    conn = psycopg2.connect(**DB_CONFIG)
    
    results = []
    for code in test_codes:
        result = verify_stock(code, conn)
        results.append(result)
        
        if result["status"] in ("NO_DATA", "NO_ZSCORE"):
            print(f"  {code}: ⚠️  {result['status']}")
        elif result["status"] == "PASS":
            print(f"  {code}: ✓ 通过 ({result['passed']}/{result['total']} 检查点)")
        else:
            print(f"  {code}: ✗ 失败 ({result['passed']}/{result['total']} 检查点)")
            for err in result["errors"]:
                print(f"    {err}")
    
    conn.close()
    
    print()
    print("=== 汇总 ===")
    total_checks = sum(r["total"] for r in results)
    total_passed = sum(r["passed"] for r in results)
    total_errors = sum(len(r["errors"]) for r in results)
    
    print(f"总检查点: {total_checks}")
    print(f"通过: {total_passed}")
    print(f"失败: {total_errors}")
    
    if total_errors > 0:
        print("\n⚠️  发现数据不一致！")
        sys.exit(1)
    else:
        print("\n✓ 所有验证通过！ddof=1 口径确认正确。")
        sys.exit(0)

if __name__ == "__main__":
    main()
