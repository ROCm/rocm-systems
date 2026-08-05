#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# Sanitizes real GFX950 CPER captures into the committed mock fixtures in this
# folder. The raw captures are NOT committed; this script documents exactly what
# is stripped so the fixtures are reviewable and reproducible from a fresh
# capture.
#
# What is kept (non-identifying, structural / public-spec data):
#   - CPER header: signature, revision, signature_end, sec_cnt, error_severity,
#     valid_bits, record_length, platform_id (PCI vendor:device), creator_id
#     ("amdgpu"), notify_type GUID (CMC/MCE spec GUID), record_id index
#   - the section descriptor (section type GUID, fru text, section severity)
#
# What is scrubbed:
#   - header timestamp -> fixed synthetic 2026-01-01T00:00:00 (removes the
#     capture-time fingerprint and makes fixtures deterministic)
#   - the section data body [section_offset, record_length) -> zeroed. This is
#     the injected RAS error payload (physical addresses, syndrome, register
#     dumps, AFIDs) and is the only machine-specific content in the record.
#
# The read path under test only counts records and filters by severity, so a
# zeroed body does not affect the tests. Zeroing also guarantees no stray "CPER"
# signature survives in the body to spoof a phantom record when records are
# concatenated.
#
# Usage:  python3 sanitize_cper.py <raw_capture_dir>
# where <raw_capture_dir> holds corrected-1..5.cper, uncorrected-6.cper,
# fatal-7.cper.

import struct
import sys
from pathlib import Path

# Header field offsets (amdsmi_cper_hdr_t, include/amd_smi/amdsmi.h, pack(1)).
TIMESTAMP_OFF = 24  # amdsmi_cper_timestamp_t, 8 bytes
SIG_OFF, SIG_END_OFF, SEC_CNT_OFF, RECLEN_OFF = 0, 6, 10, 20
HDR_SIZE = 128
# Section descriptor (immediately after the header): u32 section_offset, then
# u32 section_length.
SEC_OFFSET_OFF = HDR_SIZE
SEC_LENGTH_OFF = HDR_SIZE + 4

# Fixed synthetic timestamp: seconds, minutes, hours, flag, day, month, year,
# century -> 2026-01-01 00:00:00.
SYNTHETIC_TIMESTAMP = bytes([0, 0, 0, 0, 1, 1, 26, 20])


def sanitize(raw: bytes) -> bytes:
    buf = bytearray(raw)
    assert buf[SIG_OFF : SIG_OFF + 4] == b"CPER", "not a CPER record"
    assert struct.unpack_from("<I", buf, SIG_END_OFF)[0] == 0xFFFFFFFF, "bad signature_end"

    record_length = struct.unpack_from("<I", buf, RECLEN_OFF)[0]
    assert record_length == len(buf), f"record_length {record_length} != file size {len(buf)}"

    buf[TIMESTAMP_OFF : TIMESTAMP_OFF + 8] = SYNTHETIC_TIMESTAMP

    sec_cnt = struct.unpack_from("<H", buf, SEC_CNT_OFF)[0]
    assert sec_cnt == 1, f"expected a single-section record, got sec_cnt {sec_cnt}"
    sec_offset = struct.unpack_from("<I", buf, SEC_OFFSET_OFF)[0]
    sec_length = struct.unpack_from("<I", buf, SEC_LENGTH_OFF)[0]
    assert sec_offset + sec_length == record_length, "section does not span to record end"
    for i in range(sec_offset, record_length):
        buf[i] = 0

    return bytes(buf)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <raw_capture_dir>")
    src = Path(sys.argv[1])
    out = Path(__file__).parent

    def load(name: str) -> bytes:
        return sanitize((src / name).read_bytes())

    corrected = b"".join(load(f"corrected-{i}.cper") for i in range(1, 6))
    uncorrected = load("uncorrected-6.cper")
    fatal = load("fatal-7.cper")

    fixtures = {
        "cper_corrected.cper": corrected,
        "cper_uncorrected.cper": uncorrected,
        "cper_fatal.cper": fatal,
        "cper_mixed.cper": corrected + uncorrected + fatal,
    }
    for name, data in fixtures.items():
        (out / name).write_bytes(data)
        print(f"wrote {name} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
