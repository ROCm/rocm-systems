#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. SPDX-License-Identifier: MIT

"""Generate TraceLogging probes for a ROCm runtime's dispatch tables.

    rocm_trace_gen.py --config <runtime>.yaml --output-dir <dir> \
        -I <include dir> ... [-D NAME ...] [--resource-dir DIR] [--check]

See README.md in this directory.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rocm_trace_gen.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
