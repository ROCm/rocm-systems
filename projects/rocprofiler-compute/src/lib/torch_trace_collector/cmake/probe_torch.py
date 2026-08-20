# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Build probe for the torch_trace_collector CMake build.

On success, prints one value per line to stdout:

    1. Python major version
    2. Python minor version
    3. torch version
    4. source fingerprint
    5. libstdc++ ABI torch was built with (0 or 1)

Exit codes:
    0  success
    3  torch is not importable
    1  probe failure
"""

import pathlib
import sys


def main() -> int:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"torch_trace_collector probe: torch not importable: {exc}\n")
        return 3

    # parents[3] is the repo `src` dir (dev) or `libexec/<project>` (installed);
    # both host the utils package.
    src_root = pathlib.Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(src_root))
    from utils.inject_roctx._backends.torch_trace_fingerprint import source_fingerprint

    lines = [
        str(sys.version_info.major),
        str(sys.version_info.minor),
        torch.__version__,
        source_fingerprint(),
        str(int(torch.compiled_with_cxx11_abi())),
    ]
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
