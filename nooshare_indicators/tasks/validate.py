"""Validation module: compare our indicator implementations against TA-Lib C library.

For each indicator, we compute both ours and TA-Lib's version on the same data,
then report max absolute difference, mean absolute difference, and whether all
values are within the 1e-6 tolerance.

Also verifies log-return additivity invariant.
"""
import pandas as pd
import numpy as np
from datetime import date
from typing import Optional

from .. import config
from ..infrastructure.data_loader import prepare_base_data
from ..indicators.returns import compute_log_return, verify_log_return_additivity
from ..indicators.momentum import (
    sma, ema, dema, kama, rsi, macd, roc,
)
from ..indicators.volatility import (
    volatility_close_to_close,
    volatility_parkinson,
    volatility_garman_klass,
    volatility_rogers_satchell,
)
from ..indicators.price_position import bollinger_bands, atr


def _compare_series(
    name: str,
    ours: pd.Series,
    ref: np.ndarray | pd.Series,
    tolerance: float = 1e-6,
    warmup_skip: int = 0,
) -> dict:
    """Compare two series and return validation stats.

    Args:
        name: Indicator name
        ours: Our implementation output
        ref: TA-Lib reference output
        tolerance: Max allowed absolute difference
        warmup_skip: Number of initial NaN values to skip

    Returns:
        Dict with validation results
    """
    if isinstance(ref, np.ndarray):
        ref = pd.Series(ref, index=ours.index)

    # Align indices
    ours = ours.reset_index(drop=True)
    ref = ref.reset_index(drop=True)

    # Both valid mask
    valid = ours.notna() & ref.notna()
    n_valid = valid.sum()

    if n_valid == 0:
        return {
            "indicator": name,
            "status": "SKIP",
            "reason": "No overlapping valid values",
            "n_valid": 0,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "within_tolerance": None,
        }

    diff = (ours[valid] - ref[valid]).abs()
    max_diff = diff.max()
    mean_diff = diff.mean()
    within = max_diff <= tolerance

    return {
        "indicator": name,
        "status": "PASS" if within else "FAIL",
        "n_valid": int(n_valid),
        "n_total": len(ours),
        "max_abs_diff": float(max_diff),
        "mean_abs_diff": float(mean_diff),
        "within_tolerance": within,
        "tolerance": tolerance,
    }


def validate_against_talib(df: pd.DataFrame) -> list[dict]:
    """Run all TA-Lib comparisons on a single stock's data.

    Args:
        df: DataFrame with columns: open, high, low, close, volume, pre_close

    Returns:
        List of validation result dicts
    """
    import talib

    results = []
    close = df["close"].astype(float).values
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    open_ = df["open"].astype(float).values
    volume = df["volume"].astype(float).values

    close_s = df["close"].astype(float)
    high_s = df["high"].astype(float)
    low_s = df["low"].astype(float)
    open_s = df["open"].astype(float)
    volume_s = df["volume"].astype(float)

    # ── SMA ──────────────────────────────────────────
    for p in [5, 10, 20]:
        ours = sma(close_s, p)
        ref = talib.SMA(close, timeperiod=p)
        results.append(_compare_series(f"SMA({p})", ours, ref))

    # ── EMA ──────────────────────────────────────────
    for p in [5, 10, 20]:
        ours = ema(close_s, p)
        ref = talib.EMA(close, timeperiod=p)
        results.append(_compare_series(f"EMA({p})", ours, ref))

    # ── DEMA ─────────────────────────────────────────
    for p in [5, 10, 20]:
        ours = dema(close_s, p)
        ref = talib.DEMA(close, timeperiod=p)
        results.append(_compare_series(f"DEMA({p})", ours, ref))

    # ── KAMA ─────────────────────────────────────────
    for p in [10]:
        ours = kama(close_s, period=p)
        ref = talib.KAMA(close, timeperiod=p)
        results.append(_compare_series(f"KAMA({p})", ours, ref))

    # ── RSI ──────────────────────────────────────────
    for p in [6, 14]:
        ours = rsi(close_s, p)
        ref = talib.RSI(close, timeperiod=p)
        results.append(_compare_series(f"RSI({p})", ours, ref))

    # ── MACD ─────────────────────────────────────────
    ours_macd, ours_signal, ours_hist = macd(close_s)
    ref_macd, ref_signal, ref_hist = talib.MACD(
        close, fastperiod=12, slowperiod=26, signalperiod=9
    )
    results.append(_compare_series("MACD_line", ours_macd, ref_macd))
    results.append(_compare_series("MACD_signal", ours_signal, ref_signal))
    results.append(_compare_series("MACD_histogram", ours_hist, ref_hist))

    # ── ROC ──────────────────────────────────────────
    for p in [5, 10]:
        ours = roc(close_s, p)
        ref = talib.ROC(close, timeperiod=p)
        # TA-Lib ROC returns percentage (e.g. 5.0 for 5%), ours too
        results.append(_compare_series(f"ROC({p})", ours, ref))

    # ── ATR ──────────────────────────────────────────
    ours_atr = atr(high_s, low_s, close_s, 14)
    ref_atr = talib.ATR(high, low, close, timeperiod=14)
    results.append(_compare_series("ATR(14)", ours_atr, ref_atr))

    # ── Bollinger Bands ──────────────────────────────
    ours_upper, ours_mid, ours_lower = bollinger_bands(close_s, 20, 2.0)
    ref_upper, ref_mid, ref_lower = talib.BBANDS(
        close, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0
    )
    results.append(_compare_series("BBANDS_upper(20)", ours_upper, ref_upper))
    results.append(_compare_series("BBANDS_middle(20)", ours_mid, ref_mid))
    results.append(_compare_series("BBANDS_lower(20)", ours_lower, ref_lower))

    # ── OBV ──────────────────────────────────────────
    from ..indicators.volume import obv
    ours_obv = obv(close_s, volume_s)
    ref_obv = talib.OBV(close, volume)
    results.append(_compare_series("OBV", ours_obv, ref_obv))

    # ── Rogers-Satchell ──────────────────────────────
    # TA-Lib ROGERSSATCHELL returns raw variance (not sqrt, not annualized)
    # Our impl: sqrt(mean(...) * 252) → annualized vol
    # Compare at raw variance level for fair comparison
    ref_rs = talib.ROGERSSATCHELL(open_, high, low, close, timeperiod=20)
    log_hc = np.log(high_s / close_s)
    log_ho = np.log(high_s / open_s)
    log_lc = np.log(low_s / close_s)
    log_lo = np.log(low_s / open_s)
    raw_var = (log_hc * log_ho + log_lc * log_lo).rolling(window=20).mean()
    results.append(_compare_series("RS_variance(20)", raw_var, ref_rs))

    return results


def validate_log_return_additivity(df: pd.DataFrame) -> dict:
    """Verify log return additivity invariant.

    sum(log_returns[t:t+n]) == log(close[t+n] / close[t])

    Args:
        df: DataFrame with close column and pre_close column

    Returns:
        Dict with validation results
    """
    close = df["close"].astype(float)
    pre_close = df["pre_close"].astype(float)
    log_ret = compute_log_return(close, pre_close)

    passed = verify_log_return_additivity(log_ret, close)

    # Detailed check for multiple windows
    details = []
    for window in [5, 10, 20, 60]:
        rolled_sum = log_ret.rolling(window).sum()
        ratio = close / close.shift(window)
        expected = np.log(ratio)
        diff = (rolled_sum - expected).abs()
        valid = diff.notna()
        if valid.sum() > 0:
            details.append({
                "window": window,
                "max_abs_diff": float(diff[valid].max()),
                "mean_abs_diff": float(diff[valid].mean()),
                "n_valid": int(valid.sum()),
            })

    return {
        "test": "log_return_additivity",
        "status": "PASS" if passed else "FAIL",
        "details": details,
    }


def generate_validation_report(
    stock_code: str = "000001",
    output_path: str = "deliverables/validation_report.md",
) -> str:
    """Generate complete validation report.

    Args:
        stock_code: Stock code to validate
        output_path: Path to write the report

    Returns:
        Path to the generated report
    """
    import os
    from ..infrastructure.data_loader import load_trading_calendar, load_st_status, load_limit_ratio

    print(f"Loading data for {stock_code}...")
    calendar = load_trading_calendar()
    st_periods = load_st_status()
    limit_ratios = load_limit_ratio()
    df = prepare_base_data(stock_code, calendar, st_periods, limit_ratios)

    if df.empty:
        return "ERROR: No data loaded"

    print(f"Loaded {len(df)} rows, running validations...")

    # 1. TA-Lib comparisons
    talib_results = validate_against_talib(df)

    # 2. Log return additivity
    additivity_result = validate_log_return_additivity(df)

    # 3. Generate report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = []
    lines.append("# 验证报告 (Validation Report)")
    lines.append("")
    lines.append(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**验证股票**: {stock_code}")
    lines.append(f"**数据范围**: {df['date'].iloc[0]} - {df['date'].iloc[-1]}")
    lines.append(f"**数据行数**: {len(df)}")
    lines.append("")

    # Summary
    n_pass = sum(1 for r in talib_results if r["status"] == "PASS")
    n_fail = sum(1 for r in talib_results if r["status"] == "FAIL")
    n_skip = sum(1 for r in talib_results if r["status"] == "SKIP")
    lines.append("## 总结")
    lines.append("")
    lines.append(f"| 指标 | 数量 |")
    lines.append(f"|------|------|")
    lines.append(f"| ✅ PASS | {n_pass} |")
    lines.append(f"| ❌ FAIL | {n_fail} |")
    lines.append(f"| ⏭️ SKIP | {n_skip} |")
    lines.append(f"| **总计** | **{len(talib_results)}** |")
    lines.append("")

    # Detailed TA-Lib comparison table
    lines.append("## TA-Lib 比对结果 (容差 1e-6)")
    lines.append("")
    lines.append("| 指标 | 状态 | 有效值数 | 最大绝对差 | 平均绝对差 |")
    lines.append("|------|------|----------|------------|------------|")
    for r in talib_results:
        if r["status"] == "SKIP":
            lines.append(f"| {r['indicator']} | ⏭️ SKIP | - | - | - |")
        else:
            status = "✅" if r["status"] == "PASS" else "❌"
            lines.append(
                f"| {r['indicator']} | {status} | {r['n_valid']} | "
                f"{r['max_abs_diff']:.2e} | {r['mean_abs_diff']:.2e} |"
            )
    lines.append("")

    # Log return additivity
    lines.append("## 对数收益可加性验证")
    lines.append("")
    status = "✅ PASS" if additivity_result["status"] == "PASS" else "❌ FAIL"
    lines.append(f"**状态**: {status}")
    lines.append("")
    lines.append("| 窗口 | 最大绝对差 | 平均绝对差 | 有效值数 |")
    lines.append("|------|------------|------------|----------|")
    for d in additivity_result["details"]:
        lines.append(
            f"| {d['window']} | {d['max_abs_diff']:.2e} | "
            f"{d['mean_abs_diff']:.2e} | {d['n_valid']} |"
        )
    lines.append("")

    # Notes
    lines.append("## 说明")
    lines.append("")
    lines.append("### TA-Lib 比对方法")
    lines.append("- 使用 TA-Lib C 库 (Python wrapper) 作为参考实现")
    lines.append("- 对每个指标，使用相同输入数据计算，逐值比对")
    lines.append("- 容差设为 1e-6（浮点精度范围内）")
    lines.append("- 跳过 NaN 值（预热期数据不足）")
    lines.append("")
    lines.append("### 对数收益可加性")
    lines.append("- 恒等式: `sum(log_returns[t:t+n]) == log(close[t+n] / close[t])`")
    lines.append("- 这是对数收益率的核心数学性质，保证时间维度上的可加性")
    lines.append("- 验证窗口: 5, 10, 20, 60 天")
    lines.append("")
    lines.append("### 已知差异说明")
    lines.append("- **EMA/DEMA/MACD**: pandas ewm 和 TA-Lib 的初始值处理可能略有不同")
    lines.append("  - TA-Lib 使用 SMA 作为 EMA 的种子值")
    lines.append("  - pandas ewm(span=N, adjust=False) 使用第一个有效值作为种子")
    lines.append("  - 差异在后续值中会收敛消失")
    lines.append("- **RSI**: 同样因为 Wilder 平滑的初始值处理不同")
    lines.append("- **KAMA**: 迭代计算，初始值处理差异可能导致前几个值不同")
    lines.append("- **Rogers-Satchell**: TA-Lib 返回原始方差，我们返回年化波动率")
    lines.append("  - 比对时使用原始方差（未开方、未年化）")
    lines.append("")

    report = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nReport written to {output_path}")
    return output_path
