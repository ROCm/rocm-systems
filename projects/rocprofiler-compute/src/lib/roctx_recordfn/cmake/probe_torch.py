# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Build probe for the roctx_recordfn CMake build.

Prints the Python/torch metadata and source fingerprint that CMake needs to
tag and configure the extension.

On success, prints one value per line to stdout:

    1. Python major version
    2. Python minor version
    3. torch version
    4. torch include dirs  (';'-separated)
    5. torch library dirs  (';'-separated)
    6. torch wheel lib dir
    7. source fingerprint

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
        import torch.utils.cpp_extension as ext
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"roctx_recordfn probe: torch not importable: {exc}\n")
        return 3

    # Reuse the loader's fingerprint helper. parents[3] is the repo `src` dir
    # (dev) or `libexec/<project>` (installed); both host the utils package.
    src_root = pathlib.Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(src_root))
    from utils.inject_roctx._backends.torch_cpp_loader import _source_fingerprint

    wheel_lib_dir = str(pathlib.Path(torch.__file__).parent / "lib")

    lines = [
        str(sys.version_info.major),
        str(sys.version_info.minor),
        torch.__version__,
        ";".join(ext.include_paths()),
        ";".join(ext.library_paths()),
        wheel_lib_dir,
        _source_fingerprint(),
    ]
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
