"""
Entrypoint script for Vision AI Tier autonomous oil spill detection.
"""
import sys
from pathlib import Path

# Ensure root directory is on Python path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from vision_ai.cli import main

if __name__ == "__main__":
    main()
