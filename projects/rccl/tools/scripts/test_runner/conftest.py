import sys
from pathlib import Path


TEST_RUNNER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_RUNNER_ROOT))
