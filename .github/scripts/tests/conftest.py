"""Pytest configuration: ensure pr_merge_sync_patches and siblings are importable."""
import sys
from pathlib import Path

scripts_dir = Path(__file__).resolve().parent.parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))
