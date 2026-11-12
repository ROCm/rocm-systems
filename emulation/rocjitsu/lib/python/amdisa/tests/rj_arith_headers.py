# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Public rj-arith declarations used to check generated instruction bindings."""

from pathlib import Path


def public_headers() -> str:
    directory = Path(__file__).resolve().parents[3] / 'arith/include/rj_arith/detail'
    return '\n'.join(
        path.read_text() for path in sorted(directory.glob('rj_arith_*.h'))
    )
