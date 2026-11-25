"""CLI entry point."""
import typer
from typing import Optional
from .infrastructure import database as db
from .infrastructure.data_loader import load_stock_list
from .tasks.compute import compute_all_indicators, save_results_to_db

app = typer.Typer(help="A-share technical indicator computation engine")

# Batch size for Phase A: process N stocks per batch, compute raw indicators + save.
# Phase B computes cross-section statistics on the full universe after all batches complete.
BATCH_SIZE = 200


@app.command()
def init_db():
    """Initialize indicators schema and tables."""
    print("Initializing indicators schema...")
    db.init_schema()
    print("Done!")


@app.command()
def compute(
    start_date: Optional[int] = typer.Option(None, help="Start date (YYYYMMDD)"),
    max_stocks: Optional[int] = typer.Option(None, help="Maximum stocks to process"),
    cross_section: bool = typer.Option(True, help="Compute cross-section metrics (Phase B)"),
    save: bool = typer.Option(True, help="Save to database"),
    batch_size: int = typer.Option(BATCH_SIZE, help="Stocks per batch (Phase A)"),
):
    """Compute all indicators (two-phase: raw indicators + cross-section).
    
    Phase A: Compute raw indicators for all stocks in batches, save to DB.
    Phase B: Compute cross-section statistics on the full universe, update DB.
    
    This ensures cross-section statistics are computed on the complete universe,
    not within each batch.
    """
    print("Starting indicator computation...")

    # Determine the full stock list (or a capped subset for testing).
    all_codes = load_stock_list()["stock_code"].tolist()
    if max_stocks:
        all_codes = all_codes[:max_stocks]

    # Resume mode is driven by the batch checkpoint table.
    db.ensure_checkpoint_table()
    range_key = (start_date or 0) * 10000
    if start_date and not db.has_range_checkpoints(range_key):
        db.clear_checkpoints()

    n_batches = (len(all_codes) + batch_size - 1) // batch_size
    print(f"Total stocks: {len(all_codes)} | batches: {n_batches} (batch_size={batch_size})")

    # ========== Phase A: Compute raw indicators (batched) ==========
    print("\n" + "=" * 60)
    print("Phase A: Computing raw indicators (batched)")
    print("=" * 60)
    
    total = {k: 0 for k in ("base", "volatility", "momentum", "volume", "price_position")}
    skipped = 0

    for b, i in enumerate(range(0, len(all_codes), batch_size)):
        chunk = all_codes[i : i + batch_size]

        if save and db.is_batch_done(range_key + b):
            print(f"\n--- Batch {b + 1}/{n_batches}: SKIP (checkpoint done) ---")
            skipped += 1
            continue

        print(f"\n--- Batch {b + 1}/{n_batches}: {len(chunk)} stocks ---")

        # Compute raw indicators (no cross-section yet)
        results = compute_all_indicators(codes=chunk, start_date=start_date)

        if save:
            save_results_to_db(results)
            db.mark_batch_done(range_key + b, len(chunk))

        for k in total:
            total[k] += len(results[k])

    print("\nPhase A Summary:")
    for key, n in total.items():
        print(f"  {key}: {n} records")
    if skipped:
        print(f"  (skipped {skipped} already-completed batches)")

    # ========== Phase B: Compute cross-section statistics (full universe) ==========
    if cross_section and save:
        print("\n" + "=" * 60)
        print("Phase B: Computing cross-section statistics (full universe)")
        print("=" * 60)
        
        from .tasks.compute import compute_cross_section_for_all_dates
        compute_cross_section_for_all_dates(start_date=start_date)
        
        print("\nPhase B Complete: Cross-section statistics updated in DB")


@app.command()
def validate(
    stock_code: str = typer.Option("000001", help="Stock code to validate"),
    output_path: str = typer.Option(
        "deliverables/validation_report.md", help="Path to write the validation report"
    ),
):
    """Validate indicators against TA-Lib (requires TA-Lib installation).

    Delegates to tasks.validate.generate_validation_report: TA-Lib 12-indicator
    comparison + log-return additivity, written as a Markdown report.
    """
    try:
        import talib  # noqa: F41
    except ImportError:
        print("TA-Lib not installed. Install with: pip install TA-Lib")
        raise typer.Exit(code=1)

    from .tasks.validate import generate_validation_report

    print(f"Validating stock {stock_code} against TA-Lib...")
    result = generate_validation_report(stock_code, output_path)
    print(f"Validation report written to: {result}")


@app.command()
def test_stock(
    code: str = typer.Argument(..., help="Stock code to test"),
    start_date: Optional[int] = typer.Option(None, help="Start date (YYYYMMDD)"),
):
    """Test indicator computation for a single stock."""
    from .tasks.compute import compute_indicators_for_stock
    from .infrastructure.data_loader import load_trading_calendar, load_st_status, load_limit_ratio

    print(f"Testing indicators for {code}...")

    calendar = load_trading_calendar()
    st_periods = load_st_status()
    limit_ratios = load_limit_ratio()

    results = compute_indicators_for_stock(
        code, calendar, st_periods, limit_ratios, start_date
    )

    if not results:
        print("No data computed.")
        return

    print("\nResults:")
    for key, df in results.items():
        print(f"  {key}: {len(df)} records")
        if not df.empty:
            print(f"    Columns: {list(df.columns)}")
            print(f"    Date range: {df['date'].min()} - {df['date'].max()}")


if __name__ == "__main__":
    app()
