"""Test mark_limit_status PIT query logic."""
import pandas as pd
import pytest
from nooshare_indicators.infrastructure.data_loader import (
    mark_limit_status,
    _code_to_security_type,
)


class TestCodeToSecurityType:
    """Test _code_to_security_type mapping."""

    def test_sh_main_board(self):
        assert _code_to_security_type("600000") == ("main_board", "sh")
        assert _code_to_security_type("601398") == ("main_board", "sh")

    def test_sz_main_board(self):
        assert _code_to_security_type("000001") == ("main_board", "sz")
        assert _code_to_security_type("002230") == ("main_board", "sz")

    def test_chi_next(self):
        assert _code_to_security_type("300001") == ("chi_next", "sz")
        assert _code_to_security_type("301001") == ("chi_next", "sz")

    def test_star_market(self):
        assert _code_to_security_type("688001") == ("star_market", "sh")
        assert _code_to_security_type("689001") == ("star_market", "sh")

    def test_bse(self):
        assert _code_to_security_type("830799") == ("bse", "bj")
        assert _code_to_security_type("430047") == ("bse", "bj")


class TestMarkLimitStatus:
    """Test mark_limit_status with PIT semantics."""

    @pytest.fixture
    def sample_limit_rules(self):
        """Sample limit_rule_pit data (4 rows after rebuild)."""
        return pd.DataFrame({
            "security_type": ["chi_next", "main_board", "main_board", "star_market"],
            "exchange": ["sz", "sh", "sz", "sh"],
            "limit_ratio": [0.2000, 0.1000, 0.1000, 0.2000],
            "effective_from": [
                pd.Timestamp("2020-08-24"),
                pd.Timestamp("2008-02-28"),
                pd.Timestamp("2008-02-28"),
                pd.Timestamp("2019-07-29"),
            ],
        })

    @pytest.fixture
    def sample_stock_data(self):
        """Sample stock daily data."""
        return pd.DataFrame({
            "date": [20200824, 20200825, 20200826],
            "pre_close": [10.00, 11.00, 12.10],
            "close": [11.00, 12.10, 13.31],
        })

    def test_chi_next_uses_20pct_after_2020_08_24(
        self, sample_limit_rules, sample_stock_data
    ):
        """ChiNext stocks should use 20% limit after 2020-08-24."""
        result = mark_limit_status(
            sample_stock_data.copy(),
            sample_limit_rules,
            "300001",  # ChiNext stock
        )

        # 2020-08-24: pre_close=10.00, limit_up = 10.00 * 1.20 = 12.00
        assert result.iloc[0]["limit_up_price"] == 12.00
        assert result.iloc[0]["limit_down_price"] == 8.00

    def test_sh_main_board_uses_10pct(self, sample_limit_rules, sample_stock_data):
        """SH main board stocks should use 10% limit."""
        result = mark_limit_status(
            sample_stock_data.copy(),
            sample_limit_rules,
            "600000",  # SH main board
        )

        # 2020-08-24: pre_close=10.00, limit_up = 10.00 * 1.10 = 11.00
        assert result.iloc[0]["limit_up_price"] == 11.00
        assert result.iloc[0]["limit_down_price"] == 9.00

    def test_star_market_uses_20pct(self, sample_limit_rules, sample_stock_data):
        """STAR Market stocks should use 20% limit."""
        result = mark_limit_status(
            sample_stock_data.copy(),
            sample_limit_rules,
            "688001",  # STAR Market
        )

        # 2020-08-24: pre_close=10.00, limit_up = 10.00 * 1.20 = 12.00
        assert result.iloc[0]["limit_up_price"] == 12.00

    def test_limit_up_detection(self, sample_limit_rules, sample_stock_data):
        """Should detect when close hits limit up price."""
        data = sample_stock_data.copy()
        data.loc[0, "close"] = 12.00  # Exactly at limit up for ChiNext

        result = mark_limit_status(data, sample_limit_rules, "300001")

        assert result.iloc[0]["is_limit_up"] == True
        assert result.iloc[0]["is_censored"] == True

    def test_limit_down_detection(self, sample_limit_rules, sample_stock_data):
        """Should detect when close hits limit down price."""
        data = sample_stock_data.copy()
        data.loc[0, "close"] = 8.00  # Exactly at limit down for ChiNext

        result = mark_limit_status(data, sample_limit_rules, "300001")

        assert result.iloc[0]["is_limit_down"] == True
        assert result.iloc[0]["is_censored"] == True

    def test_no_limit_hit(self, sample_limit_rules, sample_stock_data):
        """Should not flag when close is within limits."""
        data = sample_stock_data.copy()
        data.loc[0, "close"] = 10.50  # Within limits for ChiNext

        result = mark_limit_status(data, sample_limit_rules, "300001")

        assert result.iloc[0]["is_limit_up"] == False
        assert result.iloc[0]["is_limit_down"] == False
        assert result.iloc[0]["is_censored"] == False

    def test_empty_dataframe(self, sample_limit_rules):
        """Should handle empty DataFrame gracefully."""
        empty_df = pd.DataFrame(columns=["date", "pre_close", "close"])
        result = mark_limit_status(empty_df, sample_limit_rules, "300001")
        assert len(result) == 0

    def test_missing_pre_close(self, sample_limit_rules):
        """Should handle missing pre_close column."""
        df = pd.DataFrame({"date": [20200824], "close": [10.00]})
        result = mark_limit_status(df, sample_limit_rules, "300001")
        assert "limit_up_price" not in result.columns
