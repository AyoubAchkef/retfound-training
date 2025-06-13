"""
CLI Main Entry Point
===================

Entry point for running RETFound CLI as a module with python -m retfound.cli
"""

import sys
from .main import main

if __name__ == '__main__':
    sys.exit(main())
