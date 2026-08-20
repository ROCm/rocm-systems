# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Print the torch_trace_collector source fingerprint."""

import pathlib
import sys


def main() -> int:
    # parents[3] is the repo `src` dir (dev) or `libexec/<project>` (installed);
    # both host the utils package.
    src_root = pathlib.Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(src_root))
    from utils.inject_roctx._backends.torch_trace_fingerprint import source_fingerprint

    sys.stdout.write(source_fingerprint() + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
