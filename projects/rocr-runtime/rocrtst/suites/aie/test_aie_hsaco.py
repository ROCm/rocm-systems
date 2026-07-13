# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
import struct

import pytest

import aie_hsaco
import aie_hsaco_dump


def _parse(section: bytes):
    magic, vmaj, vmin, hdr_size, kcount, kentry, st_off, st_size, pool_off = \
        struct.unpack_from("<IHHIIIIII", section, 0)
    assert magic == 0x4B454941
    assert (vmaj, vmin) == (1, 0)
    kernels = []
    for i in range(kcount):
        base = hdr_size + i * kentry
        (name_off, insts_off, insts_size, pdi_off, pdi_size,
         kernarg_size, num_cols) = struct.unpack_from("<IIIIIII", section, base)
        name_abs = st_off + name_off
        end = section.index(b"\x00", name_abs)
        name = section[name_abs:end].decode()
        insts = section[insts_off:insts_off + insts_size]
        pdi = section[pdi_off:pdi_off + pdi_size] if pdi_size else None
        kernels.append(dict(name=name, insts=insts, pdi=pdi,
                            kernarg_size=kernarg_size, num_cols=num_cols))
    return kernels


def test_build_section_round_trip():
    kernels = [
        dict(name="add_one", insts=b"\x01\x02\x03\x04", pdi=b"\xaa\xbb",
             kernarg_size=32, num_cols=1),
        dict(name="no_pdi", insts=b"\x05\x06\x07\x08", pdi=None,
             kernarg_size=16, num_cols=2),
    ]
    section = aie_hsaco.build_section("aie2p", kernels)
    out = _parse(section)
    assert out == kernels


def test_shared_blob_deduplicated():
    shared = b"\x09\x09\x09\x09"
    kernels = [
        dict(name="a", insts=shared, pdi=None, kernarg_size=0, num_cols=1),
        dict(name="b", insts=shared, pdi=None, kernarg_size=0, num_cols=1),
    ]
    section = aie_hsaco.build_section("aie2p", kernels)
    out = _parse(section)
    # Both kernels resolve to the same bytes and the blob appears once in the pool.
    assert out[0]["insts"] == out[1]["insts"] == shared
    assert section.count(shared) == 1


def test_dump_round_trip():
    section = aie_hsaco.build_section("aie2", [
        dict(name="k", insts=b"\x01\x02\x03\x04", pdi=b"\xaa", kernarg_size=8, num_cols=1)])
    info = aie_hsaco_dump.parse_section(section)
    assert info["arch_version"] == (1, 0)
    assert info["kernels"][0]["name"] == "k"
    assert info["kernels"][0]["has_pdi"] is True


def test_dump_rejects_bad_magic():
    section = bytearray(aie_hsaco.build_section("aie2", [
        dict(name="k", insts=b"\x01\x02\x03\x04", pdi=None, kernarg_size=0, num_cols=1)]))
    section[0] ^= 0xFF
    with pytest.raises(ValueError, match="magic"):
        aie_hsaco_dump.parse_section(bytes(section))


def test_dump_rejects_blob_overrun():
    section = bytearray(aie_hsaco.build_section("aie2", [
        dict(name="k", insts=b"\x01\x02\x03\x04", pdi=None, kernarg_size=0, num_cols=1)]))
    # Corrupt insts_size in the first entry to overrun the section.
    entry_base = struct.unpack_from("<I", section, 8)[0]  # header_size
    struct.pack_into("<I", section, entry_base + 8, 0xFFFFFFFF)  # insts_size field
    with pytest.raises(ValueError, match="overrun|bounds"):
        aie_hsaco_dump.parse_section(bytes(section))
