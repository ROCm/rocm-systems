# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""rj-arith wave suffixes follow Clang BuiltinsAMDGPU: _w32/_w64 before _gfx12."""


def wave_api(stem: str, wave_size: int) -> str:
    """Add a wave suffix to an API stem, before its optional gfx suffix."""
    base, separator, arch = stem.partition('_gfx')
    return f'{base}_w{wave_size}{separator}{arch}'
