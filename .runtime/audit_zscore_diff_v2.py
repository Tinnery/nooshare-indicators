"""DEV 数据库 Volume Z-score 新旧口径差异审计（只读）。

抽样 20 个股票，覆盖：
- 普通交易
- 停牌
- 恢复交易
- 成交量异常放大
- 成交量长期接近零

独立用新口径（V0.7）计算，与数据库现有值对比。
"""
import sys
sys.path.insert(0, "/home/travis/nooshare-workenv/repos/nooshare-indicators")

import pandas as pd
import numpy as np
from nooshare_indicators.infrastructure.database import read_sql
from nooshare_indicators.indicators.volume import volume_zscore


def audit_sample():
    """抽样 20 个股票进行差异审计"""
    
    # 1. 选取样本：覆盖不同情况
    # 注意：daily_base_aligned.code 是纯数字（如 600610）
    # 但 daily_ts_zscore.code 带交易所前缀（如 sh600610）
    # 需要建立映射关系
    
    # 先找一些有停牌记录的股票（从 daily_ts_zscore 获取带前缀的 code）
    suspension_sql = """
        SELECT DISTINCT z.code
        FROM indicators.daily_ts_zscore z
        JOIN indicators.daily_base_aligned b ON SUBSTRING(z.code FROM 3) = b.code
        WHERE z.indicator_name = 'volume_zscore_20'
          AND b.volume = 0 AND b.date >= 20200101
        LIMIT 5
    """
    suspension_stocks = read_sql(suspension_sql)["code"].tolist()
    print(f"停牌样本: {suspension_stocks}")
    
    # 找一些正常交易的股票（带前缀）
    normal_sql = """
        SELECT DISTINCT code
        FROM indicators.daily_ts_zscore
        WHERE indicator_name = 'volume_zscore_20'
        LIMIT 10
    """
    normal_stocks = read_sql(normal_sql)["code"].tolist()
    print(f"正常样本: {normal_stocks[:5]}...")
    
    # 找一些成交量很小的股票（带前缀）
    low_volume_sql = """
        SELECT z.code, AVG(b.volume) as avg_vol
        FROM indicators.daily_ts_zscore z
        JOIN indicators.daily_base_aligned b ON SUBSTRING(z.code FROM 3) = b.code
        WHERE z.indicator_name = 'volume_zscore_20'
          AND b.date >= 20200101 AND b.volume > 0
        GROUP BY z.code
        ORDER BY avg_vol ASC
        LIMIT 5
    """
    low_volume_stocks = read_sql(low_volume_sql)["code"].tolist()
    print(f"低成交量样本: {low_volume_stocks}")
    
    # 合并样本（去重）
    sample_codes = list(set(suspension_stocks + normal_stocks + low_volume_stocks))[:20]
    
    print(f"\n抽样 {len(sample_codes)} 个股票进行审计")
    print()
    
    # 2. 对每个样本独立计算新口径 Z-score
    total_rows = 0
    matched_rows = 0
    diff_rows = 0
    max_abs_diff = 0
    max_rel_diff = 0
    diff_details = []
    
    print("开始逐股票对比...")
    for idx, code_with_prefix in enumerate(sample_codes):
        if idx % 5 == 0:
            print(f"  处理第 {idx+1}/{len(sample_codes)} 个股票...")
        
        # code_with_prefix 格式如 sh600610，需要提取纯数字 code
        code_pure = code_with_prefix[2:]  # 去掉 sh/sz/bj 前缀
        
        # 读取该股票的原始成交量（用纯数字 code）
        vol_sql = f"""
            SELECT date, volume
            FROM indicators.daily_base_aligned
            WHERE code = '{code_pure}' AND date >= 20190101
            ORDER BY date
        """
        vol_df = read_sql(vol_sql)
        
        if vol_df.empty:
            print(f"  {code_with_prefix}: 无成交量数据，跳过")
            continue
        
        # 用新口径计算
        new_z = volume_zscore(vol_df["volume"], period=20)
        vol_df["new_zscore"] = new_z.values
        
        # 读取数据库现有值（用带前缀的 code）
        db_sql = f"""
            SELECT date, value AS db_zscore
            FROM indicators.daily_ts_zscore
            WHERE code = '{code_with_prefix}' AND indicator_name = 'volume_zscore_20'
              AND date >= 20190101
            ORDER BY date
        """
        db_df = read_sql(db_sql)
        
        if db_df.empty:
            print(f"  {code_with_prefix}: 无数据库 Z-score 值，跳过")
            continue
        
        # 合并对比
        merged = vol_df.merge(db_df, on="date", how="inner")
        merged = merged.dropna(subset=["new_zscore", "db_zscore"])
        
        if merged.empty:
            print(f"  {code_with_prefix}: 合并后无有效数据，跳过")
            continue
        
        # 计算差异
        merged["abs_diff"] = (merged["new_zscore"] - merged["db_zscore"]).abs()
        merged["rel_diff"] = merged["abs_diff"] / (merged["db_zscore"].abs() + 1e-10)
        
        total_rows += len(merged)
        matched = (merged["abs_diff"] < 1e-6).sum()
        matched_rows += matched
        diff = (merged["abs_diff"] >= 1e-6).sum()
        diff_rows += diff
        
        if merged["abs_diff"].max() > max_abs_diff:
            max_abs_diff = merged["abs_diff"].max()
        
        if merged["rel_diff"].max() > max_rel_diff:
            max_rel_diff = merged["rel_diff"].max()
        
        # 记录有差异的详情
        if diff > 0:
            diff_samples = merged[merged["abs_diff"] >= 1e-6].head(3)
            for _, row in diff_samples.iterrows():
                diff_details.append({
                    "code": code_with_prefix,
                    "date": row["date"],
                    "new_z": row["new_zscore"],
                    "db_z": row["db_zscore"],
                    "abs_diff": row["abs_diff"],
                    "rel_diff": row["rel_diff"],
                })
    
    # 3. 输出审计结果
    print("=" * 80)
    print("DEV 数据库 Volume Z-score 新旧口径差异审计报告")
    print("=" * 80)
    print()
    print(f"总对比行数: {total_rows:,}")
    print(f"完全匹配 (< 1e-6): {matched_rows:,} ({matched_rows/total_rows*100:.2f}%)")
    print(f"有差异 (>= 1e-6): {diff_rows:,} ({diff_rows/total_rows*100:.2f}%)")
    print()
    print(f"最大绝对差异: {max_abs_diff:.6f}")
    print(f"最大相对差异: {max_rel_diff:.6f}")
    print()
    
    if diff_details:
        print("差异样本（最多显示 10 条）:")
        print("-" * 80)
        for d in diff_details[:10]:
            print(f"  {d['code']} {d['date']}: new={d['new_z']:.6f} db={d['db_z']:.6f} "
                  f"abs_diff={d['abs_diff']:.6f} rel_diff={d['rel_diff']:.6f}")
        print()
    
    # 4. 分析差异原因
    print("差异原因分析:")
    print("-" * 80)
    print("新口径（V0.7）修复了有效窗口逻辑：")
    print("  - 旧口径（V0.5）：只检查固定前 period 个位置，停牌时有效数不足直接跳过")
    print("  - 新口径（V0.7）：向前搜索直到找到 period 个有效历史观察值")
    print()
    print("预期差异场景：")
    print("  1. 停牌后恢复交易的股票：新口径能继续计算，旧口径可能返回 NaN")
    print("  2. 停牌密集的股票：新口径使用更早的有效观察，旧口径可能跳过")
    print("  3. 正常交易的股票：如果窗口内无停牌，新旧口径应该一致")
    print()
    
    # 5. 结论
    if diff_rows == 0:
        print("结论: 新旧口径完全一致，无需回填")
    elif diff_rows / total_rows < 0.01:
        print(f"结论: 差异比例很小 ({diff_rows/total_rows*100:.2f}%)，主要影响停牌密集的股票")
        print("建议: 可以选择性回填受影响的股票，或保留现状并在文档中说明口径差异")
    else:
        print(f"结论: 差异比例较大 ({diff_rows/total_rows*100:.2f}%)，需要评估回填影响")
        print("建议: 输出完整的回填影响评估报告，等待授权后执行回填")


if __name__ == "__main__":
    audit_sample()
