#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generate non-executable firmware-format fixtures for the M2 guest.

The shipping amdgpu driver parses firmware headers during GFX, SDMA, and MES
early initialization even in emulation mode. These files carry only the
minimum header metadata and conspicuous rocjitsu sentinel payloads needed to
reach the device model. They are not copies of AMD firmware.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import struct
import sys
import zlib

FIXED_HEADER_BYTES = 0x100
SENTINEL = 0x524A4657  # ASCII "RJFW" in big-endian display order.


class FixtureError(ValueError):
    """A requested fixture output is unsafe or invalid."""


def common_header(
    *,
    total_size: int,
    header_size: int,
    header_major: int,
    header_minor: int,
    ip_major: int,
    ip_minor: int,
    ucode_size: int,
    ucode_offset: int,
    payload: bytes,
) -> bytes:
    return struct.pack(
        "<IIHHHHIIII",
        total_size,
        header_size,
        header_major,
        header_minor,
        ip_major,
        ip_minor,
        1,  # Synthetic fixture revision.
        ucode_size,
        ucode_offset,
        zlib.crc32(payload) & 0xFFFFFFFF,
    )


def padded_header(common: bytes, extension: bytes) -> bytearray:
    header = bytearray(FIXED_HEADER_BYTES)
    header[: len(common)] = common
    header[len(common) : len(common) + len(extension)] = extension
    return header


def rlc_fixture() -> bytes:
    payload = struct.pack("<I", SENTINEL)
    total_size = FIXED_HEADER_BYTES + len(payload)
    common = common_header(
        total_size=total_size,
        header_size=104,
        header_major=2,
        header_minor=0,
        ip_major=12,
        ip_minor=1,
        ucode_size=len(payload),
        ucode_offset=FIXED_HEADER_BYTES,
        payload=payload,
    )
    fields = [0] * 18
    fields[11] = FIXED_HEADER_BYTES
    fields[13] = FIXED_HEADER_BYTES
    fields[15] = FIXED_HEADER_BYTES
    fields[17] = FIXED_HEADER_BYTES
    return bytes(padded_header(common, struct.pack("<18I", *fields))) + payload


def mec_fixture() -> bytes:
    ucode = struct.pack("<I", SENTINEL)
    data = struct.pack("<I", SENTINEL)
    ucode_offset = FIXED_HEADER_BYTES
    data_offset = ucode_offset + len(ucode)
    total_size = data_offset + len(data)
    common = common_header(
        total_size=total_size,
        header_size=60,
        header_major=2,
        header_minor=0,
        ip_major=12,
        ip_minor=1,
        ucode_size=len(ucode),
        ucode_offset=ucode_offset,
        payload=ucode,
    )
    extension = struct.pack(
        "<7I",
        0,  # Feature version.
        len(ucode),
        ucode_offset,
        len(data),
        data_offset,
        0x3000,
        0,
    )
    return bytes(padded_header(common, extension)) + ucode + data


def sdma_fixture() -> bytes:
    payload = struct.pack("<I", SENTINEL)
    total_size = FIXED_HEADER_BYTES + len(payload)
    common = common_header(
        total_size=total_size,
        header_size=44,
        header_major=3,
        header_minor=0,
        ip_major=7,
        ip_minor=1,
        ucode_size=len(payload),
        ucode_offset=FIXED_HEADER_BYTES,
        payload=payload,
    )
    extension = struct.pack("<3I", 0, FIXED_HEADER_BYTES, len(payload))
    return bytes(padded_header(common, extension)) + payload


def mes_fixture() -> bytes:
    ucode_words = [SENTINEL] * 32
    ucode_words[24] = 1  # Version field read by amdgpu_mes_init_microcode().
    ucode = struct.pack("<32I", *ucode_words)
    data = struct.pack("<I", SENTINEL)
    ucode_offset = FIXED_HEADER_BYTES
    data_offset = ucode_offset + len(ucode)
    total_size = data_offset + len(data)
    common = common_header(
        total_size=total_size,
        header_size=72,
        header_major=1,
        header_minor=0,
        ip_major=12,
        ip_minor=1,
        ucode_size=len(ucode),
        ucode_offset=ucode_offset,
        payload=ucode,
    )
    extension = struct.pack(
        "<10I",
        1,
        len(ucode),
        ucode_offset,
        1,
        len(data),
        data_offset,
        0x3000,
        0,
        0,
        0,
    )
    return bytes(padded_header(common, extension)) + ucode + data


def imu_fixture() -> bytes:
    iram = struct.pack("<I", SENTINEL)
    dram = struct.pack("<I", SENTINEL)
    iram_offset = FIXED_HEADER_BYTES
    dram_offset = iram_offset + len(iram)
    total_size = dram_offset + len(dram)
    common = common_header(
        total_size=total_size,
        header_size=48,
        header_major=1,
        header_minor=0,
        ip_major=12,
        ip_minor=1,
        ucode_size=len(iram) + len(dram),
        ucode_offset=iram_offset,
        payload=iram + dram,
    )
    extension = struct.pack("<4I", len(iram), iram_offset, len(dram), dram_offset)
    return bytes(padded_header(common, extension)) + iram + dram


FIXTURES = {
    "gc_12_1_0_imu.bin": imu_fixture,
    "gc_12_1_0_mec.bin": mec_fixture,
    "gc_12_1_0_rlc_1.bin": rlc_fixture,
    "gc_12_1_0_uni_mes.bin": mes_fixture,
    "sdma_7_1_0.bin": sdma_fixture,
}


def generate(output: Path) -> None:
    if output.is_symlink():
        raise FixtureError(f"output is a symlink: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for name, builder in FIXTURES.items():
        path = output / name
        if path.exists():
            raise FixtureError(f"refusing to replace existing fixture: {path}")
        path.write_bytes(builder())


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(arguments)
    try:
        generate(args.output)
    except (FixtureError, OSError) as error:
        print(
            f"vfio guest firmware fixture generation failed: {error}", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
