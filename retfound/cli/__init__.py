"""
Command Line Interface
=====================

CLI for RETFound training, evaluation, and inference.
"""

from .main import main, create_parser
from .utils import (
    setup_cli_logging,
    print_banner,
    print_config,
    confirm_action,
    format_table
)

__all__ = [
    'main',
    'create_parser',
    'setup_cli_logging',
    'print_banner',
    'print_config',
    'confirm_action',
    'format_table'
]