from contextlib import contextmanager
from pathlib import Path
import tempfile


@contextmanager
def temporary_root():
    with tempfile.TemporaryDirectory() as temporary:
        yield Path(temporary)
