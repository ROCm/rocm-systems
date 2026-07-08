"""User-facing CuTeDSL bindings for the NCCL device API.

See ``examples/cute/main.py`` for a complete, runnable example.
"""

try:
    import cutlass.cute  # noqa: F401
except ImportError as e:
    raise ImportError(
        "nccl.core.device.cute requires the nvidia-cutlass-dsl package. "
        "Install it via the matching extra:\n"
        "    pip install 'nccl4py[cu12]'   # for CUDA 12\n"
        "    pip install 'nccl4py[cu13]'   # for CUDA 13"
    ) from e

from . import types, coop, comm, window, gin, barrier
from .types import *    # MemoryOrder, GinFenceLevel, GinBackendMask
from .coop import *     # Coop, cta, warp, thread, lanes, warp_span
from .comm import *     # Team, DevComm
from .window import *   # Window
from .gin import *      # Gin
from .barrier import *  # session classes + factories

__all__ = [
    "types",
    "coop",
    "comm",
    "window",
    "gin",
    "barrier",
    *types.__all__,
    *coop.__all__,
    *comm.__all__,
    *window.__all__,
    *gin.__all__,
    *barrier.__all__,
]
