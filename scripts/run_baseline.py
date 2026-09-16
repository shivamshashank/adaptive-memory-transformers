#!/usr/bin/env python3
# Direct command-line entry point for the full-cache baseline.

import sys
from pathlib import Path

# Make the repository package importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline import main

if __name__ == "__main__":
    main()
