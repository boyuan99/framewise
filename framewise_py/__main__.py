"""Enables `python -m framewise_py` invocation."""

import sys

from .main import main

if __name__ == "__main__":
    sys.exit(main())
