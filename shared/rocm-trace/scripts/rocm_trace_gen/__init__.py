# Copyright Advanced Micro Devices, Inc. SPDX-License-Identifier: MIT

"""Generate TraceLogging probes for ROCm runtime dispatch tables.

Each non-null entry of the configured tables gets a probe that emits an
``<api>_enter`` event with the arguments, calls the saved entry, and emits an
``<api>_exit`` event with the result and the arguments the call wrote.
See ``README.md`` next to this package for the configuration format.
"""

__version__ = "1.0.0"
