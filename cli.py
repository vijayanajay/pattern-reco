#!/usr/bin/env python3
"""
CLI entry point for the stock pattern detection system.
"""
import argparse
import logging
import sys
from datetime import date

from rich.logging import RichHandler

# Set up logging
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("main")

# Import refactored components
from src.adapters.yfinance_api import fetch_and_snapshot, load_snapshots
from src.config import load_config
from src.core.universe import get_nse_symbols, select_universe


def setup_parser() -> argparse.ArgumentParser:
    """Set up the argument parser."""
    parser = argparse.ArgumentParser(
        description="A stock pattern detection system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    run_parser = subparsers.add_parser("run", help="Execute the backtest pipeline")
    run_parser.add_argument(
        "--config", required=True, type=str, help="Path to YAML configuration file"
    )

    refresh_parser = subparsers.add_parser(
        "refresh-data", help="Refresh data snapshots"
    )
    refresh_parser.add_argument(
        "--config", required=True, type=str, help="Path to YAML configuration file"
    )
    return parser


def main() -> None:
    """Main CLI entry point."""
    parser = setup_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        log.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)

        if args.command == "refresh-data":
            handle_refresh_data(config)
        elif args.command == "run":
            handle_run_pipeline(config)

    except (FileNotFoundError, ValueError) as e:
        log.error(f"Error: {e}", extra={"markup": True})
        sys.exit(1)
    except Exception:
        log.exception("An unexpected error occurred:")
        sys.exit(1)


from typing import Any, Dict

def handle_refresh_data(config: Dict[str, Any]) -> None:
    """Handle the 'refresh-data' command."""
    log.info("Starting data refresh...")
    symbols = get_nse_symbols()
    fetch_and_snapshot(symbols, config)
    log.info("[green]Data refresh completed successfully.[/green]", extra={"markup": True})


def handle_run_pipeline(config: Dict[str, Any]) -> None:
    """Handle the 'run' command."""
    log.info("Starting backtest pipeline...")

    # 1. Load all available data
    log.info("Loading data snapshots...")
    all_symbols = get_nse_symbols()
    all_data = load_snapshots(all_symbols, config)

    # 2. Select universe at t0 (for now, t0 is hardcoded, should be from config)
    # The PRD/design implies a single t0 for the whole backtest.
    t0 = date(2020, 1, 1) # Placeholder
    log.info(f"Selecting universe at t0 = {t0}...")
    universe = select_universe(all_data, config.get("universe", {}), t0)

    log.info(f"Selected universe: {universe}")

    # TODO: Implement walk-forward backtesting logic here.
    # This involves:
    # - Creating walk-forward splits (IS/OOS).
    # - For each split and each stock in the universe:
    #   - Fit detector on IS data.
    #   - Run backtest on OOS data.
    #   - Collect and aggregate results.

    log.warning("Pipeline execution logic is not fully implemented yet.")
    log.info("[green]Pipeline run finished (placeholder).[/green]", extra={"markup": True})


if __name__ == "__main__":
    main()
