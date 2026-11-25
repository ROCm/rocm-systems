# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import subprocess
import sys
import tempfile

if __name__ == "__main__":
    with tempfile.TemporaryDirectory(
        prefix="rj-sg-", dir=os.environ.get("XDG_RUNTIME_DIR") or None
    ) as directory:
        sys.exit(
            subprocess.call(
                sys.argv[1:], env={**os.environ, "ROCJITSU_RUNTIME_DIR": directory}
            )
        )
