#!/usr/bin/env python3
"""
CLI entry point for anomaly pattern detection system.
Only two commands: run and refresh-data.
"""

import argparse
import sys

from src.config import load_config
from src.data import fetch_and_snapshot


def main():
    """Main CLI entry point with minimal interface."""
    parser = argparse.ArgumentParser(
        description="Anomaly Pattern Detection for Indian Stocks",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Execute the backtest pipeline')
    run_parser.add_argument('--config', required=True, type=str,
                           help='Path to YAML configuration file')
    
    # Refresh data command
    refresh_parser = subparsers.add_parser('refresh-data', help='Refresh data snapshots')
    refresh_parser.add_argument('--config', required=True, type=str,
                               help='Path to YAML configuration file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        # Load and validate configuration
        config = load_config(args.config)
        print(f"Loaded configuration from: {args.config}")
        
        if args.command == 'run':
            print("Starting backtest pipeline...")
            # TODO: Implement pipeline execution
            print("Pipeline execution not yet implemented")
            
        elif args.command == 'refresh-data':
            print("Starting data refresh...")
            
            # Get symbols from universe selection
            from src.universe import select_universe
            symbols, _ = select_universe(config)
            
            fetch_and_snapshot(symbols, config)
            print("Data refresh completed successfully")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()