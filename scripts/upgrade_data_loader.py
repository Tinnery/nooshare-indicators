#!/usr/bin/env python3
"""
T2.5: 改造 indicators 的 data_loader.py，只读 public.* 产品表

修改：
1. load_trading_calendar() - ju.trading_day -> public.trading_calendar
2. load_limit_ratio() - ju.limit_ratio -> public.limit_rule_pit
3. load_stock_daily() - tong.stock_daily -> public.hfq_daily
"""

import re

DATA_LOADER_FILE = "/home/travis/progspace/nooshare-indicators/nooshare_indicators/infrastructure/data_loader.py"


def upgrade_load_trading_calendar():
    """改造 load_trading_calendar 从 ju.trading_day 改为 public.trading_calendar"""
    
    with open(DATA_LOADER_FILE, "r") as f:
        content = f.read()
    
    # 找到 load_trading_calendar 函数
    pattern = r'(def load_trading_calendar\(\) -> pd\.DataFrame:\n    """Load trading calendar from ju\.trading_day\."""\n    sql = """\n    SELECT trade_date\n    FROM ju\.trading_day\n    ORDER BY trade_date\n    """)'
    
    replacement = '''def load_trading_calendar() -> pd.DataFrame:
    """Load trading calendar from public.trading_calendar (PIT product)."""
    sql = """
    SELECT cal_date AS trade_date
    FROM public.trading_calendar
    WHERE is_open = true
    ORDER BY cal_date
    """'''
    
    new_content = re.sub(pattern, replacement, content)
    
    if new_content == content:
        print("✗ load_trading_calendar 未找到匹配")
        return False
    
    with open(DATA_LOADER_FILE, "w") as f:
        f.write(new_content)
    
    print("✓ load_trading_calendar 已改造")
    return True


def upgrade_load_limit_ratio():
    """改造 load_limit_ratio 从 ju.limit_ratio 改为 public.limit_rule_pit"""
    
    with open(DATA_LOADER_FILE, "r") as f:
        content = f.read()
    
    # 找到 load_limit_ratio 函数
    pattern = r'(def load_limit_ratio\(\) -> pd\.DataFrame:\n    """Load limit ratio from ju\.limit_ratio\."""\n    sql = """\n    SELECT stock_code, start_date, end_date, limit_ratio\n    FROM ju\.limit_ratio\n    ORDER BY stock_code, start_date\n    """)'
    
    replacement = '''def load_limit_ratio() -> pd.DataFrame:
    """Load limit ratio from public.limit_rule_pit (PIT product).
    
    注意：limit_rule_pit 按 (security_type, exchange) 聚合，不含 stock_code。
    需要根据股票代码推断 security_type 和 exchange，然后查询对应的 limit_ratio。
    """
    # 先加载所有规则
    sql = """
    SELECT security_type, exchange, limit_ratio, effective_from, effective_to
    FROM public.limit_rule_pit
    ORDER BY security_type, exchange, effective_from
    """
    return db.read_sql(sql)'''
    
    new_content = re.sub(pattern, replacement, content)
    
    if new_content == content:
        print("✗ load_limit_ratio 未找到匹配")
        return False
    
    with open(DATA_LOADER_FILE, "w") as f:
        f.write(new_content)
    
    print("✓ load_limit_ratio 已改造")
    return True


def upgrade_load_stock_daily():
    """改造 load_stock_daily 从 tong.stock_daily 改为 public.hfq_daily"""
    
    with open(DATA_LOADER_FILE, "r") as f:
        content = f.read()
    
    # 找到 load_stock_daily 函数
    pattern = r'(def load_stock_daily\(symbol: str, start_date: Optional\[int\] = None\) -> pd\.DataFrame:\n    """Load daily OHLCV for a single stock from tong\.stock_daily\."""\n    sql = """\n    SELECT symbol, date, open, high, low, close, volume, amount\n    FROM tong\.stock_daily\n    WHERE symbol = %s AND date >= %s\n    ORDER BY date\n    """)'
    
    replacement = '''def load_stock_daily(symbol: str, start_date: Optional[int] = None) -> pd.DataFrame:
    """Load daily OHLCV for a single stock from public.hfq_daily (前复权价格产品)."""
    sql = """
    SELECT symbol, date, open, high, low, close, volume, amount
    FROM public.hfq_daily
    WHERE symbol = %s AND date >= %s
    ORDER BY date
    """'''
    
    new_content = re.sub(pattern, replacement, content)
    
    if new_content == content:
        print("✗ load_stock_daily 未找到匹配")
        return False
    
    with open(DATA_LOADER_FILE, "w") as f:
        f.write(new_content)
    
    print("✓ load_stock_daily 已改造")
    return True


def upgrade_load_all_stocks_daily_batch():
    """改造 load_all_stocks_daily_batch 从 tong.stock_daily 改为 public.hfq_daily"""
    
    with open(DATA_LOADER_FILE, "r") as f:
        content = f.read()
    
    # 找到 FROM tong.stock_daily 在 load_all_stocks_daily_batch 中
    pattern = r'(    sql = f"""\n    SELECT symbol, date, open, high, low, close, volume, amount\n    FROM tong\.stock_daily\n    WHERE symbol IN \(\{symbol_list\}\) AND date >= \{start\}\n    ORDER BY symbol, date\n    """)'
    
    replacement = '''    sql = f"""
    SELECT symbol, date, open, high, low, close, volume, amount
    FROM public.hfq_daily
    WHERE symbol IN ({symbol_list}) AND date >= {start}
    ORDER BY symbol, date
    """'''
    
    new_content = re.sub(pattern, replacement, content)
    
    if new_content == content:
        print("✗ load_all_stocks_daily_batch 未找到匹配")
        return False
    
    with open(DATA_LOADER_FILE, "w") as f:
        f.write(new_content)
    
    print("✓ load_all_stocks_daily_batch 已改造")
    return True


def main():
    print("Upgrading data_loader.py (T2.5)...")
    
    success = True
    success &= upgrade_load_trading_calendar()
    success &= upgrade_load_limit_ratio()
    success &= upgrade_load_stock_daily()
    success &= upgrade_load_all_stocks_daily_batch()
    
    if success:
        print("✓ T2.5 upgrade complete")
    else:
        print("✗ T2.5 upgrade failed")


if __name__ == "__main__":
    main()
