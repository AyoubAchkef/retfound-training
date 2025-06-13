"""
Main CLI Entry Point
===================

Main command-line interface for RETFound.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List

from .commands import train, evaluate, predict
from .utils import setup_cli_logging, print_banner

logger = logging.getLogger(__name__)

# Import export with error handling
try:
    from .commands import export
    EXPORT_AVAILABLE = True
except (ImportError, AttributeError) as e:
    logger.warning(f"Export command not available: {e}")
    EXPORT_AVAILABLE = False
    export = None


def create_parser() -> argparse.ArgumentParser:
    """Create main argument parser"""
    
    parser = argparse.ArgumentParser(
        prog='retfound',
        description='RETFound - Medical AI for Ophthalmology',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  retfound train --config config.yaml --weights cfp --epochs 100
  
  # Evaluate a trained model
  retfound evaluate --checkpoint model.pth --dataset /path/to/test
  
  # Export model to different formats
  retfound export --checkpoint model.pth --formats onnx tensorrt
  
  # Run inference
  retfound predict --model model.pth --image eye.jpg

For more help on a specific command:
  retfound <command> --help
        """
    )
    
    # Global arguments
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 2.0.0'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='count',
        default=0,
        help='Increase verbosity (can be repeated: -v, -vv, -vvv)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress all output except errors'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log to file'
    )
    
    parser.add_argument(
        '--no-color',
        action='store_true',
        help='Disable colored output'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(
        title='Commands',
        dest='command',
        required=True,
        help='Available commands'
    )
    
    # Add subcommands
    train.add_subparser(subparsers)
    evaluate.add_subparser(subparsers)
    if EXPORT_AVAILABLE and export is not None:
        export.add_subparser(subparsers)
    predict.add_subparser(subparsers)
    
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main CLI entry point
    
    Args:
        argv: Command line arguments (defaults to sys.argv)
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # Setup logging
    log_level = logging.WARNING
    if args.quiet:
        log_level = logging.ERROR
    elif args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose >= 2:
        log_level = logging.DEBUG
    
    setup_cli_logging(
        verbose=log_level <= logging.INFO,
        log_file=Path(args.log_file) if args.log_file else None
    )
    
    # Print banner for interactive sessions
    if sys.stdout.isatty() and not args.quiet:
        print_banner()
    
    try:
        # Dispatch to command
        if args.command == 'train':
            return train.run_train(args)
        elif args.command == 'evaluate':
            return evaluate.run_evaluate(args)
        elif args.command == 'export':
            if EXPORT_AVAILABLE and export is not None:
                return export.run_export(args)
            else:
                logger.error("Export command is not available due to missing dependencies or configuration issues")
                return 1
        elif args.command == 'predict':
            return predict.run_predict(args)
        else:
            parser.error(f"Unknown command: {args.command}")
            
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130  # Standard exit code for SIGINT
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose >= 2)
        return 1


if __name__ == '__main__':
    sys.exit(main())
