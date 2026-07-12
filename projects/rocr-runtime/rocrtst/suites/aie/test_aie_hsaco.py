# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
import struct

import aie_hsaco


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
