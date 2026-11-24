#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for scripts/bench-vfio-user.py."""

from __future__ import annotations

import argparse
import array
import contextlib
import errno
from fractions import Fraction
import importlib.util
import io
import json
import math
import mmap
import os
from pathlib import Path
import random
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
import types
from typing import Callable, NamedTuple
import unittest
from unittest import mock

SCRIPT = Path(__file__).parents[2] / "scripts" / "bench-vfio-user.py"
SPEC = importlib.util.spec_from_file_location("bench_vfio_user", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BENCH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCH
SPEC.loader.exec_module(BENCH)

PAGE = 4096
U32 = 0xFFFF_FFFF

# One function without a loop, and one whose loop holds a two-dword instruction
# and is preceded by a forward branch that must not be taken for a loop.
LISTING = """
kernels.so:\tfile format elf64-amdgpu

Disassembly of section .text:

0000000000001000 <noop>:
\ts_endpgm                                 // 000000001000: BFB00000

0000000000001100 <int_loop>:
\ts_load_dwordx2 s[0:1], s[4:5], 0x0       // 000000001100: F4040002 F8000000
\tv_mov_b32_e32 v1, 0                      // 000000001108: 7E020280
\ts_cbranch_scc1 8 <int_loop+0x2c>         // 00000000110C: BFA20008
\tv_lshlrev_b32_e32 v2, 13, v1             // 000000001110: 3404038D
\tv_xor_b32_e32 v1, v2, v1                 // 000000001114: 3A020302
\tv_add_co_u32 v0, vcc, 0x61, v0           // 000000001118: 400000FF 00000061
\ts_add_u32 s2, s2, -1                     // 000000001120: 8002C102
\ts_cmp_lg_u32 s2, 0                       // 000000001124: BF078002
\ts_cbranch_scc1 65529 <int_loop+0x10>     // 000000001128: BFA2FFF9
\ts_endpgm                                 // 00000000112C: BFB00000

0000000000001200 <two_loops>:
\tv_mov_b32_e32 v1, 0                      // 000000001200: 7E020280
\ts_cbranch_scc1 65535 <two_loops+0x0>     // 000000001204: BFA2FFFF
\tv_mov_b32_e32 v2, 0                      // 000000001208: 7E040280
\tv_mov_b32_e32 v3, 0                      // 00000000120C: 7E060280
\ts_branch 65533 <two_loops+0x8>           // 000000001210: BFA0FFFD
\ts_endpgm                                 // 000000001214: BFB00000
"""

KERNEL_NAMES = ("noop", *BENCH.LOOP_KERNELS)
DESCRIPTOR_BYTES = 64
DESCRIPTOR_VADDR = 0x200
CODE_VADDR = 0x1000
CODE_BYTES = bytes(range(1, 17))
CODE_MEMORY_BYTES = 0x40

ELF_HEADER_BYTES = 64
PROGRAM_HEADER = struct.Struct("<IIQQQQQQ")
SECTION_HEADER = struct.Struct("<IIQQQQIIQQ")
SYMBOL = struct.Struct("<IBBHQQ")
PT_LOAD, PT_NOTE = 1, 4
SHT_STRTAB, SHT_NOTE, SHT_DYNSYM = 3, 7, 11

# AMDGPU page-table entry bits the tests rely on (amdgpu_vm.h).
PTE_VALID = 1 << 0
PTE_SYSTEM = 1 << 1
PTE_ADDRESS_MASK = 0x0000_FFFF_FFFF_F000

DISPATCH_PACKET = struct.Struct("<HHHHHHIIIIIQQQQ")
HSA_PACKET_TYPE_KERNEL_DISPATCH = 2

# kernel_descriptor_t field the fake device uses to tell kernels apart.
KD_CODE_ENTRY_OFFSET = 16
KERNARG = struct.Struct("<QIIQI")

# Dword offsets the device reads in an MES ADD_QUEUE frame and in the MQD the
# frame points at.
FRAME_PAGE_TABLE_ROOT, FRAME_DOORBELL, FRAME_MQD_ADDRESS = 2, 18, 20
FRAME_WRITE_POINTER_ADDRESS, FRAME_QUEUE_TYPE, FRAME_STATUS_ADDRESS = 22, 28, 38
MQD_RING_BASE_LO, MQD_RING_BASE_HI, MQD_DOORBELL = 136, 137, 143

PROCESS_BYTES = 3 << 20
VRAM_BYTES = BENCH.VRAM_DATA + BENCH.MEM_WORDS * 4


def elf_note(owner: bytes, kind: int, description: bytes) -> bytes:
    def padded(raw: bytes) -> bytes:
        return raw.ljust((len(raw) + 3) & ~3, b"\0")

    header = struct.pack("<III", len(owner), len(description), kind)
    return header + padded(owner) + padded(description)


def synthetic_elf(
    kernel_names: tuple[str, ...] = KERNEL_NAMES,
    private_segment_bytes: int = 0,
    wave32_kernels: tuple[str, ...] = (),
    load_segments: bool = True,
) -> bytes:
    """An ELF64 code object with two PT_LOAD segments, a dynsym and a note section.

    Both segments are loaded at addresses that differ from their file offsets,
    and the second is longer in memory than in the file.
    """
    descriptors = bytearray()
    strings = bytearray(b"\0")
    symbols = bytearray(SYMBOL.size)
    for index, name in enumerate(kernel_names):
        descriptor = bytearray(DESCRIPTOR_BYTES)
        struct.pack_into(
            "<I", descriptor, BENCH.KD_PRIVATE_SEGMENT, private_segment_bytes
        )
        properties = BENCH.KD_ENABLE_WAVEFRONT_SIZE32 if name in wave32_kernels else 0
        struct.pack_into("<H", descriptor, BENCH.KD_CODE_PROPERTIES, properties)
        descriptors += descriptor
        address = DESCRIPTOR_VADDR + index * DESCRIPTOR_BYTES
        symbols += SYMBOL.pack(len(strings), 0x11, 0, 1, address, DESCRIPTOR_BYTES)
        strings += f"{name}.kd".encode() + b"\0"

    notes = elf_note(b"GNU\0", 3, b"\xaa\xbb\xcc\xdd")

    program_headers_at = ELF_HEADER_BYTES
    descriptors_at = 0x100
    code_at = descriptors_at + len(descriptors)
    symbols_at = code_at + len(CODE_BYTES)
    strings_at = symbols_at + len(symbols)
    notes_at = strings_at + len(strings)
    section_headers_at = notes_at + len(notes)

    program_headers = [
        # Not loadable, and placed high so that honouring it would grow the image.
        PROGRAM_HEADER.pack(PT_NOTE, 4, notes_at, 0x9000, 0, len(notes), len(notes), 4),
        PROGRAM_HEADER.pack(
            PT_LOAD,
            4,
            descriptors_at,
            DESCRIPTOR_VADDR,
            0,
            len(descriptors),
            len(descriptors),
            PAGE,
        ),
        PROGRAM_HEADER.pack(
            PT_LOAD,
            5,
            code_at,
            CODE_VADDR,
            0,
            len(CODE_BYTES),
            CODE_MEMORY_BYTES,
            PAGE,
        ),
    ]
    if not load_segments:
        del program_headers[1:]
    section_headers = [
        bytes(SECTION_HEADER.size),
        SECTION_HEADER.pack(
            0, SHT_DYNSYM, 2, 0, symbols_at, len(symbols), 2, 1, 8, SYMBOL.size
        ),
        SECTION_HEADER.pack(0, SHT_STRTAB, 2, 0, strings_at, len(strings), 0, 0, 1, 0),
        SECTION_HEADER.pack(0, SHT_NOTE, 2, 0, notes_at, len(notes), 0, 0, 4, 0),
    ]

    header = bytearray(ELF_HEADER_BYTES)
    header[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<QQ", header, 0x20, program_headers_at, section_headers_at)
    struct.pack_into(
        "<HHHHHH",
        header,
        0x34,
        ELF_HEADER_BYTES,
        PROGRAM_HEADER.size,
        len(program_headers),
        SECTION_HEADER.size,
        len(section_headers),
        0,
    )
    image = header + b"".join(program_headers)
    image = image.ljust(descriptors_at, b"\0")
    image += descriptors + CODE_BYTES + symbols + strings + notes
    return bytes(image + b"".join(section_headers))


def read_u32(memory: mmap.mmap | bytes, offset: int) -> int:
    return struct.unpack_from("<I", memory, offset)[0]


def read_u64(memory: mmap.mmap | bytes, offset: int) -> int:
    return struct.unpack_from("<Q", memory, offset)[0]


def joined_address(high: int, low: int, shift: int = 0) -> int:
    """Address held by a register pair whose low half stores `address >> shift`."""
    return (high << (shift + 32)) | (low << shift)


class DispatchPacket(NamedTuple):
    """hsa_kernel_dispatch_packet_t."""

    header: int
    setup: int
    workgroup_size_x: int
    workgroup_size_y: int
    workgroup_size_z: int
    reserved0: int
    grid_size_x: int
    grid_size_y: int
    grid_size_z: int
    private_segment_size: int
    group_segment_size: int
    kernel_object: int
    kernarg_address: int
    reserved2: int
    completion_signal: int


class TranslationFault(Exception):
    """An address the fake device cannot translate with what it was given."""


class QueueRegistration(NamedTuple):
    """What an MES ADD_QUEUE frame tells the device about a queue."""

    opcode: int
    page_table_root: int
    doorbell_dword: int
    mqd_address: int
    write_pointer_address: int
    queue_type: int


class FakeDevice:
    """Stands in for VfioUserClient and for the device model behind it.

    It keeps what a server keeps (its own mapping of guest RAM, VRAM, the
    registers) and answers the MES doorbell by following the registers and the
    GART table the client programmed. A layout mistake therefore shows up as a
    failed translation or an unanswered command, not as a mismatched constant.
    """

    def __init__(self, mappable_vram: bool):
        self.vram = mmap.mmap(-1, VRAM_BYTES)
        self.mappable_vram = mappable_vram
        self.guest: mmap.mmap | None = None
        self.dma_window = (0, 0)
        self.registers: dict[int, int] = {}
        self.irqs: list[tuple[int, int]] = []
        self.doorbells: list[tuple[int, int, bool]] = []
        self.mes_frames: list[bytes] = []
        self.largest_trapped_vram_write = 0

    def close(self) -> None:
        if self.guest is not None:
            self.guest.close()
        self.vram.close()

    def dma_map(self, iova: int, size: int, fd: int) -> None:
        self.dma_window = (iova, size)
        self.guest = mmap.mmap(fd, size)

    def region_mmap(self, region: int) -> mmap.mmap | None:
        if self.mappable_vram and region == BENCH.BAR_VRAM:
            return self.vram
        return None

    def arm_irq(self, index: int, fd: int) -> None:
        self.irqs.append((index, fd))

    def region_write(
        self, region: int, offset: int, data: bytes, posted: bool = False
    ) -> None:
        if region == BENCH.BAR_REGISTERS:
            self.registers[offset] = struct.unpack("<I", data)[0]
        elif region == BENCH.BAR_VRAM:
            self.vram[offset : offset + len(data)] = data
            self.largest_trapped_vram_write = max(
                self.largest_trapped_vram_write, len(data)
            )
        elif region == BENCH.BAR_DOORBELL:
            value = struct.unpack("<Q", data)[0]
            self.doorbells.append((offset, value, posted))
            if offset == BENCH.MES_DOORBELL:
                self._complete_mes_command(value)
        else:
            raise AssertionError(f"write to unexpected region {region}")

    def gc_address(self, high: int, low: int, shift: int = 0) -> int:
        """Address programmed into a pair of GC registers."""
        return joined_address(
            self.registers[(BENCH.GC_BASE + high) * 4],
            self.registers[(BENCH.GC_BASE + low) * 4],
            shift,
        )

    def gart_to_guest_offset(self, address: int) -> int:
        """Translate a GART address the way VMID 0 does; returns a guest RAM offset."""
        start = self.gc_address(BENCH.REG_GART_START_HI, BENCH.REG_GART_START_LO, 12)
        last_page = self.gc_address(BENCH.REG_GART_END_HI, BENCH.REG_GART_END_LO, 12)
        if not start <= address < last_page + PAGE:
            raise TranslationFault(f"{address:#x} is outside the programmed GART")
        table = self.gc_address(BENCH.REG_GART_PT_HI, BENCH.REG_GART_PT_LO)
        entry = read_u64(self.vram, table + ((address - start) // PAGE) * 8)
        if not entry & PTE_VALID or not entry & PTE_SYSTEM:
            raise TranslationFault(f"GART entry for {address:#x} is not system memory")
        return self.bus_to_guest_offset((entry & PTE_ADDRESS_MASK) | (address % PAGE))

    def bus_to_guest_offset(self, bus_address: int) -> int:
        iova, size = self.dma_window
        if not iova <= bus_address < iova + size:
            raise TranslationFault(f"{bus_address:#x} is outside the shared guest RAM")
        return bus_address - iova

    def walk(self, root: int, address: int) -> int | None:
        """Leaf entry for a process address, or None where nothing is mapped."""
        table = root
        for shift in (48, 39, 30, 21):
            entry = read_u64(self.vram, table + ((address >> shift) & 0x1FF) * 8)
            if not entry & PTE_VALID:
                return None
            table = entry & PTE_ADDRESS_MASK
        entry = read_u64(self.vram, table + ((address >> 12) & 0x1FF) * 8)
        return entry if entry & PTE_VALID else None

    def process_to_guest_offset(self, root: int, address: int) -> int:
        entry = self.walk(root, address)
        if entry is None or not entry & PTE_SYSTEM:
            raise TranslationFault(f"{address:#x} does not map guest RAM")
        return self.bus_to_guest_offset((entry & PTE_ADDRESS_MASK) | (address % PAGE))

    def queue_registration(self) -> QueueRegistration:
        frame = self.mes_frames[-1]
        return QueueRegistration(
            opcode=read_u32(frame, 0),
            page_table_root=read_u64(frame, FRAME_PAGE_TABLE_ROOT * 4),
            doorbell_dword=read_u32(frame, FRAME_DOORBELL * 4),
            mqd_address=read_u64(frame, FRAME_MQD_ADDRESS * 4),
            write_pointer_address=read_u64(frame, FRAME_WRITE_POINTER_ADDRESS * 4),
            queue_type=read_u32(frame, FRAME_QUEUE_TYPE * 4),
        )

    def _complete_mes_command(self, write_pointer: int) -> None:
        assert self.guest is not None
        ring = self.gc_address(BENCH.REG_MES_RING_HI, BENCH.REG_MES_RING_LO, 8)
        first_dword = (write_pointer - BENCH.MES_FRAME_DWORDS) % BENCH.MES_RING_DWORDS
        frame_offset = self.gart_to_guest_offset(ring + first_dword * 4)
        frame_bytes = BENCH.MES_FRAME_DWORDS * 4
        frame = bytes(self.guest[frame_offset : frame_offset + frame_bytes])
        self.mes_frames.append(frame)
        status_address, status_value = struct.unpack_from(
            "<QQ", frame, FRAME_STATUS_ADDRESS * 4
        )
        struct.pack_into(
            "<Q", self.guest, self.gart_to_guest_offset(status_address), status_value
        )


class Dispatch(NamedTuple):
    """One dispatch the fake device ran."""

    benchmark: str
    grid: int
    iterations: int


class ExecutingDevice(FakeDevice):
    """A FakeDevice that also runs the dispatches an AQL doorbell publishes.

    It reaches each packet through the MQD and the process page tables, reads
    the kernarg the client wrote and writes each work item's result through the
    output pointer, then clears the completion signal. A kernel is recognised
    by the code entry offset in the descriptor the packet points at. `spoil`
    lets a test change a result, or withhold it by returning None.

    The results are those of the script's own kernel models, so tests built on
    this check how the script drives and reads a device, not the models.
    """

    REGION_BYTES = {
        BENCH.BAR_VRAM: VRAM_BYTES,
        BENCH.BAR_DOORBELL: 0x2000,
        BENCH.BAR_REGISTERS: 0x8_0000,
    }

    def __init__(self, code_entries: dict[int, str], mappable_vram: bool = True):
        super().__init__(mappable_vram)
        self.code_entries = code_entries
        self.completes_dispatches = True
        self.completed = 0
        self.dispatches: list[Dispatch] = []
        self.spoil: Callable[[str, int, float], float | None] | None = None
        self.fused_multiply_add = True
        self.spin = False
        self.register_reads = 0
        self.closed = False

    def close(self) -> None:
        self.closed = True
        super().close()

    def region_size(self, region: int) -> int:
        return self.REGION_BYTES[region]

    def region_read(self, region: int, offset: int, length: int) -> bytes:
        self.register_reads += 1
        return bytes(length)

    def region_write(
        self, region: int, offset: int, data: bytes, posted: bool = False
    ) -> None:
        super().region_write(region, offset, data, posted)
        if region == BENCH.BAR_DOORBELL and offset == BENCH.AQL_DOORBELL:
            if self.completes_dispatches:
                self._run_published_packets(struct.unpack("<Q", data)[0])

    def read_process(self, root: int, address: int, length: int) -> bytes:
        """Process memory, whichever of guest RAM and VRAM each page maps."""
        assert self.guest is not None
        data = bytearray()
        while len(data) < length:
            at = address + len(data)
            entry = self.walk(root, at)
            if entry is None:
                raise TranslationFault(f"{at:#x} is not mapped")
            count = min(length - len(data), PAGE - at % PAGE)
            physical = (entry & PTE_ADDRESS_MASK) | (at % PAGE)
            if entry & PTE_SYSTEM:
                offset = self.bus_to_guest_offset(physical)
                data += self.guest[offset : offset + count]
            else:
                data += self.vram[physical : physical + count]
        return bytes(data)

    def _run_published_packets(self, last_index: int) -> None:
        assert self.guest is not None
        registration = self.queue_registration()
        root = registration.page_table_root
        descriptor = self.gart_to_guest_offset(registration.mqd_address)
        ring = joined_address(
            read_u32(self.guest, descriptor + MQD_RING_BASE_HI * 4),
            read_u32(self.guest, descriptor + MQD_RING_BASE_LO * 4),
            8,
        )
        while self.completed <= last_index:
            slot = self.completed % BENCH.AQL_PACKETS
            offset = self.process_to_guest_offset(
                root, ring + slot * DISPATCH_PACKET.size
            )
            packet = DispatchPacket._make(
                DISPATCH_PACKET.unpack_from(self.guest, offset)
            )
            self._execute(root, packet)
            signal_offset = self.process_to_guest_offset(root, packet.completion_signal)
            struct.pack_into("<q", self.guest, signal_offset + BENCH.SIGNAL_VALUE, 0)
            self.completed += 1
        for _, trigger in self.irqs:
            os.eventfd_write(trigger, 1)

    def _execute(self, root: int, packet: DispatchPacket) -> None:
        assert self.guest is not None
        descriptor = self.read_process(root, packet.kernel_object, DESCRIPTOR_BYTES)
        kernel = self.code_entries[
            struct.unpack_from("<q", descriptor, KD_CODE_ENTRY_OFFSET)[0]
        ]
        kernarg = self.read_process(root, packet.kernarg_address, KERNARG.size)
        out_pointer, iterations, _, buffer_pointer, mask = KERNARG.unpack(kernarg)
        benchmark, words = kernel, []
        if kernel == "mem_loop":
            raw = self.read_process(root, buffer_pointer, (mask + 1) * 4)
            words = array.array("I", raw).tolist()
            entry = self.walk(root, buffer_pointer)
            if entry is not None and not entry & PTE_SYSTEM:
                benchmark = "mem_loop_vram"
        self.dispatches.append(Dispatch(benchmark, packet.grid_size_x, iterations))
        if kernel == "noop":
            return
        for gid in range(packet.grid_size_x):
            candidates = BENCH.expected(kernel, gid, iterations, words)
            value = candidates[-1 if self.fused_multiply_add else 0]
            if self.spoil is not None:
                value = self.spoil(benchmark, gid, value)
            if value is None:
                continue
            target = self.process_to_guest_offset(root, out_pointer + gid * 4)
            struct.pack_into(
                "<f" if kernel == "fp_loop" else "<I", self.guest, target, value
            )


def full_code_object() -> tuple[BENCH.CodeObject, dict[int, str]]:
    """A loaded image with a descriptor per kernel, and each code entry offset."""
    image = bytearray(len(KERNEL_NAMES) * DESCRIPTOR_BYTES)
    kernels, entries = {}, {}
    for index, name in enumerate(KERNEL_NAMES):
        descriptor = index * DESCRIPTOR_BYTES
        entry = 0x1000 + index * 0x100
        struct.pack_into("<q", image, descriptor + KD_CODE_ENTRY_OFFSET, entry)
        kernels[name] = BENCH.Kernel(name, descriptor, 4, 32)
        entries[entry] = name
    return BENCH.CodeObject(bytes(image), kernels, "gfx1250"), entries


def open_descriptors() -> int:
    return len(os.listdir("/proc/self/fd"))


def is_running(pid: int) -> bool:
    """Whether a process is alive; one that exited but was not reaped is not."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    # The state follows the parenthesised name, which may hold spaces itself.
    return stat[stat.rindex(")") + 2] != "Z"


def loaded_code_object() -> BENCH.CodeObject:
    """A loaded image with one descriptor that asks for 256 bytes of LDS."""
    image = bytearray(0x100)
    struct.pack_into("<I", image, 0x40 + BENCH.KD_GROUP_SEGMENT, 256)
    kernels = {
        "noop": BENCH.Kernel("noop", 0x80, 0, 32),
        "int_loop": BENCH.Kernel("int_loop", 0x40, 5, 32),
        "mem_loop": BENCH.Kernel("mem_loop", 0xC0, 4, 32),
    }
    return BENCH.CodeObject(bytes(image), kernels, "gfx1250")


class DeviceTestCase(unittest.TestCase):
    def make_device(
        self, mappable_vram: bool = True
    ) -> tuple[BENCH.Device, FakeDevice]:
        fake = FakeDevice(mappable_vram)
        self.addCleanup(fake.close)
        device = BENCH.Device(fake, PROCESS_BYTES)
        self.addCleanup(device.close)
        return device, fake


class ParseVariantTest(unittest.TestCase):
    def parse(self, text: str) -> BENCH.Variant:
        return BENCH.parse_variant(text, "build/rocjitsu", "base.json", "gfx1250")

    def test_name_alone_takes_every_default(self) -> None:
        for text in ("baseline", "baseline:", "baseline:,,"):
            with self.subTest(text=text):
                variant = self.parse(text)

                self.assertEqual(variant.name, "baseline")
                self.assertEqual(variant.binary, "build/rocjitsu")
                self.assertEqual(variant.config, "base.json")
                self.assertEqual(variant.target, "gfx1250")
                self.assertEqual(variant.env, {})
                self.assertEqual(variant.args, [])
                self.assertEqual(variant.cpus, "")
                self.assertEqual(variant.cfg, {})

    def test_every_key_overrides_its_field(self) -> None:
        variant = self.parse(
            "tuned:bin=/opt/rj/rocjitsu,config=other.json,target=gfx1151,"
            "cpus=0-3,arg=--first,arg=--second,env=RJ_A=1,env=RJ_B=x=y,"
            "cfg=vm.gpu.count=4"
        )

        self.assertEqual(variant.name, "tuned")
        self.assertEqual(variant.binary, "/opt/rj/rocjitsu")
        self.assertEqual(variant.config, "other.json")
        self.assertEqual(variant.target, "gfx1151")
        self.assertEqual(variant.cpus, "0-3")
        self.assertEqual(variant.args, ["--first", "--second"])
        # Only the first '=' of an env item separates the name from the value.
        self.assertEqual(variant.env, {"RJ_A": "1", "RJ_B": "x=y"})
        self.assertEqual(variant.cfg, {"vm.gpu.count": 4})

    def test_cfg_value_is_json_when_it_parses_and_raw_text_otherwise(self) -> None:
        variant = self.parse(
            'sized:cfg=threads=8,cfg=enabled=true,cfg=ratio=0.5,cfg=quoted="text",'
            "cfg=name=mi455x,cfg=empty="
        )

        self.assertEqual(
            variant.cfg,
            {
                "threads": 8,
                "enabled": True,
                "ratio": 0.5,
                "quoted": "text",
                "name": "mi455x",
                "empty": "",
            },
        )

    def test_commas_inside_a_cfg_json_value_do_not_end_the_item(self) -> None:
        variant = self.parse(
            "wide:cfg=execution_threads.dispatch=[8,8],cpus=0-3,"
            'cfg=plugins={"throughput":{},"names":["a,b","c]"]},arg=--last'
        )

        self.assertEqual(
            variant.cfg,
            {
                "execution_threads.dispatch": [8, 8],
                "plugins": {"throughput": {}, "names": ["a,b", "c]"]},
            },
        )
        self.assertEqual(variant.cpus, "0-3")
        self.assertEqual(variant.args, ["--last"])

    def test_brackets_outside_a_cfg_value_are_plain_text(self) -> None:
        variant = self.parse("odd:env=RJ_LIST=[1,arg=--flag]")

        self.assertEqual(variant.env, {"RJ_LIST": "[1"})
        self.assertEqual(variant.args, ["--flag]"])

    def test_cfg_value_left_open_is_an_error(self) -> None:
        for text in ("open:cfg=list=[1,2,arg=--flag", 'open:cfg=name="a,arg=--flag'):
            with self.subTest(text=text):
                with self.assertRaisesRegex(BENCH.HarnessError, "unbalanced") as raised:
                    self.parse(text)

                self.assertIn(text, str(raised.exception))

    def test_unknown_key_is_reported_with_the_variant_text(self) -> None:
        with self.assertRaises(BENCH.HarnessError) as raised:
            self.parse("typo:binary=/opt/rj/rocjitsu")

        self.assertIn("'binary'", str(raised.exception))
        self.assertIn("typo:binary=/opt/rj/rocjitsu", str(raised.exception))


class CountLoopInstsTest(unittest.TestCase):
    def test_counts_from_the_backward_branch_target_through_the_branch(self) -> None:
        # 0x1110 through 0x1128; the two-dword instruction at 0x1118 counts once.
        self.assertEqual(BENCH.count_loop_insts(LISTING, "int_loop"), 6)

    def test_function_without_a_backward_branch_has_no_loop(self) -> None:
        self.assertEqual(BENCH.count_loop_insts(LISTING, "noop"), 0)

    def test_function_missing_from_the_listing_has_no_loop(self) -> None:
        self.assertEqual(BENCH.count_loop_insts(LISTING, "fp_loop"), 0)
        self.assertEqual(BENCH.count_loop_insts("", "int_loop"), 0)

    def test_forward_branch_alone_is_not_a_loop(self) -> None:
        listing = (
            "0000000000002000 <skip>:\n"
            "\ts_cbranch_scc1 1 <skip+0x8>             // 000000002000: BFA20001\n"
            "\tv_mov_b32_e32 v1, 0                     // 000000002004: 7E020280\n"
            "\ts_endpgm                                // 000000002008: BFB00000\n"
        )

        self.assertEqual(BENCH.count_loop_insts(listing, "skip"), 0)

    def test_forward_branch_after_the_loop_does_not_replace_it(self) -> None:
        listing = (
            "0000000000003000 <exit>:\n"
            "\tv_mov_b32_e32 v1, 0                     // 000000003000: 7E020280\n"
            "\ts_cbranch_scc1 65535 <exit+0x0>         // 000000003004: BFA2FFFF\n"
            "\ts_cbranch_scc0 1 <exit+0x10>            // 000000003008: BFA10001\n"
            "\tv_mov_b32_e32 v2, 0                     // 00000000300C: 7E040280\n"
            "\ts_endpgm                                // 000000003010: BFB00000\n"
        )

        self.assertEqual(BENCH.count_loop_insts(listing, "exit"), 2)

    def test_last_backward_branch_decides_the_loop(self) -> None:
        # 0x1208 through 0x1210, not the two instructions of the first loop.
        self.assertEqual(BENCH.count_loop_insts(LISTING, "two_loops"), 3)


class ExpectedTest(unittest.TestCase):
    def test_int_loop_is_xorshift32_seeded_from_the_work_item(self) -> None:
        self.assertEqual(BENCH.expected("int_loop", 0, 0, []), [1])
        self.assertEqual(BENCH.expected("int_loop", 1, 0, []), [2654435762])
        # 2 * 2654435761 + 1 wraps at 32 bits.
        self.assertEqual(BENCH.expected("int_loop", 2, 0, []), [1013904227])
        # 1 -> 0x2001 -> 0x2001 -> 0x2001 ^ 0x40020.
        self.assertEqual(BENCH.expected("int_loop", 0, 1, []), [0x42021])
        # 0x42021 -> 0x84000021 -> 0x84004221 -> 0x84004221 ^ 0x80084420.
        self.assertEqual(BENCH.expected("int_loop", 0, 2, []), [0x04080601])

    def test_i64_loop_folds_the_64_bit_state_to_32_bits(self) -> None:
        self.assertEqual(BENCH.expected("i64_loop", 0, 0, []), [1])
        # 0x9E3779B97F4A7C16 folds to 0x7F4A7C16 ^ 0x9E3779B9.
        self.assertEqual(BENCH.expected("i64_loop", 1, 0, []), [0xE17D05AF])
        # 1 -> 0x2001 -> 0x2041 -> 0x2041 ^ 0x40820000.
        self.assertEqual(BENCH.expected("i64_loop", 0, 1, []), [0x40822041])

    def test_fp_loop_accepts_separate_and_fused_multiply_add(self) -> None:
        self.assertEqual(BENCH.expected("fp_loop", 0, 0, []), [0.5, 0.5])
        # Only the low ten bits of the work item seed the value.
        self.assertEqual(
            BENCH.expected("fp_loop", 1024 + 7, 2, []),
            BENCH.expected("fp_loop", 7, 2, []),
        )

        # 1.5 -> 1.4995 -> 1.4990005 -> 1.4985014995
        candidates = BENCH.expected("fp_loop", 1000, 3, [])

        self.assertEqual(len(candidates), 2)
        for candidate in candidates:
            self.assertAlmostEqual(candidate, 1.4985014995, delta=1e-6)
            self.assertEqual(candidate, BENCH.f32(candidate))

    def test_fp_loop_candidates_are_the_two_ways_to_round_each_step(self) -> None:
        scale, offset = Fraction(BENCH.f32(0.999)), Fraction(BENCH.f32(0.001))
        for gid, iterations in ((5, 2), (7, 37), (1000, 3), (1023, 60)):
            with self.subTest(gid=gid, iterations=iterations):
                separate = fused = BENCH.expected("fp_loop", gid, 0, [])[0]
                for _ in range(iterations):
                    # A separate multiply rounds its product before the add.
                    product = nearest_binary32(Fraction(separate) * scale)
                    separate = nearest_binary32(Fraction(product) + offset)
                    fused = nearest_binary32(Fraction(fused) * scale + offset)

                self.assertEqual(
                    BENCH.expected("fp_loop", gid, iterations, []), [separate, fused]
                )
                # These work items are ones where the two roundings part ways.
                self.assertNotEqual(separate, fused)

    def test_fp_loop_seed_rounds_the_multiply_and_the_add_separately(self) -> None:
        for gid in (1, 7, 1000, 1023):
            with self.subTest(gid=gid):
                scaled = nearest_binary32(Fraction(gid) * Fraction(BENCH.f32(0.001)))
                seed = nearest_binary32(Fraction(scaled) + Fraction(1, 2))

                self.assertEqual(BENCH.expected("fp_loop", gid, 0, []), [seed, seed])

    def test_mem_loop_sums_buffer_words_at_a_stride_of_97(self) -> None:
        identity = list(range(BENCH.MEM_WORDS))

        for name in ("mem_loop", "mem_loop_vram"):
            with self.subTest(kernel=name):
                self.assertEqual(BENCH.expected(name, 5, 0, identity), [5])
                # 5 + buf[5] + buf[102] + buf[199]
                self.assertEqual(BENCH.expected(name, 5, 3, identity), [311])
                # The index wraps at the buffer size: 65535 + buf[65535] + buf[96].
                self.assertEqual(BENCH.expected(name, 65535, 2, identity), [131166])

    def test_mem_loop_sum_wraps_at_32_bits(self) -> None:
        saturated = [0xFFFF_FFFF] * BENCH.MEM_WORDS

        self.assertEqual(BENCH.expected("mem_loop", 1, 1, saturated), [0])
        self.assertEqual(BENCH.expected("mem_loop", 1, 2, saturated), [0xFFFF_FFFF])

    def test_lds_loop_reads_its_own_workgroup(self) -> None:
        self.assertEqual(BENCH.expected("lds_loop", 70, 0, []), [70])
        # Work item 70 is local 6 of the group at 64: 70 + (70*3+1) + (71*3+1).
        self.assertEqual(BENCH.expected("lds_loop", 70, 2, []), [495])
        # Local 63 wraps to local 0 of the same group: 127 + (127*3+1) + (64*3+1).
        self.assertEqual(BENCH.expected("lds_loop", 127, 2, []), [702])

    def test_kernel_without_a_model_has_no_expected_output(self) -> None:
        self.assertEqual(BENCH.expected("noop", 3, 9, []), [])


def mixed_words() -> list[int]:
    """A buffer whose words differ, as the one the script loads does."""
    return [(index * 2246822519 + 374761393) & U32 for index in range(BENCH.MEM_WORDS)]


class ExpectedGridTest(unittest.TestCase):
    """expected_grid() is the model the timed runs are checked against."""

    WORDS = mixed_words()

    def assert_matches_reference(self, name: str, grid: int, iterations: int) -> None:
        fast = BENCH.expected_grid(name, grid, iterations, self.WORDS)
        for gid in range(grid):
            self.assertEqual(
                [values[gid] for values in fast],
                BENCH.expected(name, gid, iterations, self.WORDS),
                (name, gid, iterations),
            )

    def test_every_work_item_matches_the_stepwise_model(self) -> None:
        for name in BENCH.BENCHMARKS:
            for iterations in (0, 1, 2, 37, 63, 64, 65, 200):
                with self.subTest(kernel=name, iterations=iterations):
                    self.assert_matches_reference(name, 192, iterations)

    def test_work_items_past_the_first_1024_and_the_last_of_the_largest_grid(
        self,
    ) -> None:
        grid = BENCH.Bench.max_grid
        for name in BENCH.BENCHMARKS:
            with self.subTest(kernel=name):
                fast = BENCH.expected_grid(name, grid, 5, self.WORDS)
                self.assertEqual({len(values) for values in fast}, {grid})
                for gid in (1023, 1024, 1025, 4095, 40000, grid - 1):
                    self.assertEqual(
                        [values[gid] for values in fast],
                        BENCH.expected(name, gid, 5, self.WORDS),
                    )

    def test_buffer_and_lds_sums_match_across_a_whole_lap(self) -> None:
        # The buffer walk repeats every MEM_WORDS steps and the LDS walk every
        # workgroup; the remainder is taken from a lap that wraps.
        for name, iterations in (
            ("mem_loop", BENCH.MEM_WORDS - 1),
            ("mem_loop", BENCH.MEM_WORDS),
            ("mem_loop_vram", BENCH.MEM_WORDS + 1),
            ("mem_loop", 3 * BENCH.MEM_WORDS + 4097),
            ("lds_loop", 64 * 1000 + 63),
        ):
            with self.subTest(kernel=name, iterations=iterations):
                fast = BENCH.expected_grid(name, 4096, iterations, self.WORDS)[0]
                for gid in (0, 1, 63, 64, 97, 4095):
                    self.assertEqual(
                        [fast[gid]], BENCH.expected(name, gid, iterations, self.WORDS)
                    )

    def test_integer_loops_return_to_their_start_after_a_full_period(self) -> None:
        # Both xorshifts visit every nonzero state before repeating, which no
        # stepwise model could confirm; the power of the step has to.
        for name, bits in (("int_loop", 32), ("i64_loop", 64)):
            with self.subTest(kernel=name):
                start = BENCH.expected_grid(name, 128, 0, [])
                self.assertEqual(
                    BENCH.expected_grid(name, 128, (1 << bits) - 1, []), start
                )
                self.assertEqual(
                    BENCH.expected_grid(name, 128, (1 << bits) + 36, []),
                    BENCH.expected_grid(name, 128, 37, []),
                )
                self.assertNotEqual(
                    BENCH.expected_grid(name, 128, (1 << bits) - 2, []), start
                )

    def test_iteration_count_of_the_calibration_limit_is_modelled(self) -> None:
        limit = 1 << 28
        laps = limit // BENCH.MEM_WORDS
        total = sum(self.WORDS)
        self.assertEqual(
            BENCH.expected_grid("mem_loop", 128, limit, self.WORDS)[0][5],
            (5 + laps * total) & U32,
        )
        # Every workgroup-sized lap adds each of the group's 64 values once.
        group = sum((64 + lid) * 3 + 1 for lid in range(64))
        self.assertEqual(
            BENCH.expected_grid("lds_loop", 128, limit, [])[0][70],
            (70 + (limit // 64) * group) & U32,
        )

    def test_fp_loop_stops_changing_and_is_then_the_stepwise_value(self) -> None:
        settled = 12000
        for gid in (0, 499, 500, 501, 1023):
            with self.subTest(gid=gid):
                reference = BENCH.expected("fp_loop", gid, settled, [])
                self.assertEqual(
                    reference, BENCH.expected("fp_loop", gid, settled + 1, [])
                )
                self.assertEqual(
                    [BENCH.fp_loop_value(gid, 1 << 28, fused) for fused in (0, 1)],
                    reference,
                )
        fast = BENCH.expected_grid("fp_loop", 64, 1 << 28, [])
        self.assertEqual(
            [values[60] for values in fast],
            BENCH.expected("fp_loop", 60, settled, []),
        )

    def test_fp_loop_before_it_settles_is_not_answered_from_a_settled_value(
        self,
    ) -> None:
        BENCH.expected_grid("fp_loop", 64, 1 << 20, [])

        self.assert_matches_reference("fp_loop", 64, 300)

    def test_random_work_items_and_iteration_counts_match_the_stepwise_model(
        self,
    ) -> None:
        # Iteration counts are drawn evenly over their magnitudes, up to what
        # the stepwise model can still walk, and one count per integer kernel
        # is as long as a short timed dispatch. fp_loop has stopped changing
        # well before its limit here.
        grid = BENCH.Bench.max_grid
        for name in BENCH.BENCHMARKS:
            rng = random.Random(f"expected_grid {name}")
            limit, longest = (
                (20_000, []) if name == "fp_loop" else (200_000, [1_000_003])
            )
            drawn = [int(math.exp(rng.uniform(0, math.log(limit)))) for _ in range(6)]
            for iterations in drawn + longest:
                fast = BENCH.expected_grid(name, grid, iterations, self.WORDS)
                gids = [
                    rng.randrange(grid) for _ in range(1 if iterations > limit else 4)
                ]
                for gid in gids:
                    with self.subTest(kernel=name, iterations=iterations, gid=gid):
                        self.assertEqual(
                            [values[gid] for values in fast],
                            BENCH.expected(name, gid, iterations, self.WORDS),
                        )

    def test_kernel_without_a_model_has_no_expected_output(self) -> None:
        self.assertEqual(BENCH.expected_grid("noop", 64, 9, []), [])


class CountWrongTest(unittest.TestCase):
    def test_integer_outputs_must_be_equal(self) -> None:
        want = [[7, 0xFFFF_FFFF, 0, 9]]
        raw = array.array("I", want[0])

        self.assertEqual(BENCH.count_wrong("int_loop", raw.tobytes(), want), 0)
        raw[1] ^= 1 << 31
        raw[3] += 1
        self.assertEqual(BENCH.count_wrong("int_loop", raw.tobytes(), want), 2)

    def test_fp_output_may_match_either_device_within_the_last_bits(self) -> None:
        want = [[1.0, 2.0, 3.0, 4.0], [1.5, 2.0, 3.0, 4.0]]
        last_bit = binary32_from_bits(binary32_bits(3.0) + 1)
        raw = array.array("f", [1.0, 1.5, last_bit, 4.0])

        self.assertEqual(BENCH.count_wrong("fp_loop", raw.tobytes(), want), 1)
        raw[3] = 4.0 * (1 + 1e-4)
        self.assertEqual(BENCH.count_wrong("fp_loop", raw.tobytes(), want), 2)

    def test_output_no_dispatch_wrote_is_wrong_for_every_kernel(self) -> None:
        cleared = b"\xff" * 16
        self.assertEqual(
            BENCH.count_wrong("fp_loop", cleared, [[1.0] * 4, [1.0] * 4]), 4
        )
        self.assertEqual(BENCH.count_wrong("lds_loop", cleared, [[1, 2, 3, 4]]), 4)


def binary32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def binary32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def nearest_binary32(exact: Fraction) -> float:
    """The binary32 nearest a positive rational, found by comparing distances.

    It rounds a first guess and then measures that value and both neighbours
    against the exact input, so it shares no arithmetic with the script.
    """
    guess = binary32_bits(struct.unpack("<f", struct.pack("<f", float(exact)))[0])
    candidates = [binary32_from_bits(bits) for bits in (guess - 1, guess, guess + 1)]
    return min(
        candidates,
        key=lambda value: (abs(Fraction(value) - exact), binary32_bits(value) & 1),
    )


class BinaryRoundingTest(unittest.TestCase):
    def assertSameFloat(self, actual: float, expected_hex: str) -> None:
        self.assertEqual(actual.hex(), float.fromhex(expected_hex).hex())

    def test_exactly_representable_results_are_returned_unchanged(self) -> None:
        self.assertEqual(BENCH.round_to_f32(1, 0), 1.0)
        self.assertEqual(BENCH.round_to_f32(3, 1), 1.5)
        self.assertEqual(BENCH.round_to_f32(-5, 2), -1.25)
        self.assertEqual(BENCH.round_to_f32(0, 7), 0.0)
        self.assertEqual(BENCH.fma32(1.5, 2.0, 0.25), 3.25)
        self.assertEqual(BENCH.fma32(-1.5, 2.0, 0.25), -2.75)
        self.assertEqual(BENCH.fma32(3.0, 0.0, -0.5), -0.5)

    def test_a_tie_goes_to_the_even_significand(self) -> None:
        # 2**24 + 1 lies midway between two binary32 values; so does 2**24 + 3.
        self.assertEqual(BENCH.round_to_f32(2**24 + 1, 0), 16777216.0)
        self.assertEqual(BENCH.round_to_f32(2**24 + 3, 0), 16777220.0)
        self.assertEqual(BENCH.round_to_f32(-(2**24 + 3), 0), -16777220.0)
        # 1 + 2**-24 is a tie that falls back to 1; one step up, it rises.
        self.assertSameFloat(BENCH.fma32(2.0**-24, 1.0, 1.0), "0x1.000000p+0")
        self.assertSameFloat(
            BENCH.fma32(2.0**-24, 1.0, float.fromhex("0x1.000002p+0")), "0x1.000004p+0"
        )

    def test_anything_past_a_tie_rounds_away_from_it(self) -> None:
        self.assertEqual(BENCH.round_to_f32(2**25 + 3, 1), 16777218.0)
        self.assertEqual(BENCH.round_to_f32(2**25 + 1, 1), 16777216.0)

    def test_the_result_is_rounded_once_not_through_binary64(self) -> None:
        # The exact value is 1 + 2**-23 + 2**-24 - 2**-70: just under the tie
        # between 1 + 2**-23 and 1 + 2**-22. A binary64 cannot hold the 2**-70,
        # so rounding to binary64 first lands on the tie and then goes up.
        multiplicand = float.fromhex("0x1.000002p-24")
        multiplier = float.fromhex("0x1.fffffcp-1")
        addend = float.fromhex("0x1.000002p+0")

        self.assertSameFloat(
            BENCH.fma32(multiplicand, multiplier, addend), "0x1.000002p+0"
        )
        self.assertSameFloat(
            BENCH.f32(multiplicand * multiplier + addend), "0x1.000004p+0"
        )

    def test_subnormal_results_keep_the_bits_binary32_has_left(self) -> None:
        smallest = 2.0**-149
        self.assertEqual(BENCH.fma32(smallest, 1.0, 0.0), smallest)
        # Half of the smallest subnormal is a tie with zero, which is even.
        self.assertEqual(BENCH.fma32(smallest, 0.5, 0.0), 0.0)
        self.assertEqual(BENCH.fma32(smallest, 0.75, 0.0), smallest)
        self.assertEqual(BENCH.fma32(smallest, 1.5, 0.0), 2 * smallest)
        self.assertEqual(BENCH.fma32(smallest, -0.75, 0.0), -smallest)
        self.assertEqual(BENCH.round_to_f32(1, 200), 0.0)

    def test_results_past_the_largest_binary32_overflow_to_infinity(self) -> None:
        largest = float.fromhex("0x1.fffffep+127")
        self.assertEqual(BENCH.fma32(largest, 1.0, 0.0), largest)
        self.assertEqual(BENCH.fma32(largest, 2.0, 0.0), math.inf)
        self.assertEqual(BENCH.fma32(largest, -2.0, 0.0), -math.inf)
        # Half a unit in the last place above the largest value is a tie that rounds up.
        self.assertEqual(BENCH.fma32(largest, 1.0, 2.0**103), math.inf)
        self.assertEqual(BENCH.fma32(largest, 1.0, 2.0**102), largest)
        self.assertEqual(BENCH.round_to_f32(1 << 2000, 0), math.inf)

    def test_rounding_a_binary64_agrees_with_the_platform_conversion(self) -> None:
        generator = random.Random(20261123)
        for _ in range(4000):
            # Exponents from far below the subnormals to just short of overflow.
            sign = generator.choice((-1.0, 1.0))
            exponent = generator.randrange(-1000, 127)
            value = math.ldexp(sign * generator.uniform(1.0, 2.0), exponent)
            numerator, denominator = value.as_integer_ratio()

            self.assertEqual(
                BENCH.round_to_f32(numerator, denominator.bit_length() - 1).hex(),
                BENCH.f32(value).hex(),
            )

    def test_fused_multiply_add_agrees_with_exact_rational_arithmetic(self) -> None:
        generator = random.Random(20261123)

        def operand() -> float:
            return binary32_from_bits(
                generator.randrange(1 << 23) | (generator.randrange(100, 150) << 23)
            )

        for _ in range(2000):
            multiplicand, multiplier, addend = operand(), operand(), operand()
            exact = Fraction(multiplicand) * Fraction(multiplier) + Fraction(addend)

            self.assertEqual(
                BENCH.fma32(multiplicand, multiplier, addend), nearest_binary32(exact)
            )


class ProcessAccountingTest(unittest.TestCase):
    PID = 4242

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.proc = Path(directory.name)
        self.ticks = os.sysconf("SC_CLK_TCK")

    def thread(
        self,
        tid: int,
        stat: str | None = None,
        schedstat: str | None = None,
        status: str | None = None,
    ) -> None:
        directory = self.proc / str(self.PID) / "task" / str(tid)
        directory.mkdir(parents=True)
        for name, text in (
            ("stat", stat),
            ("schedstat", schedstat),
            ("status", status),
        ):
            if text is not None:
                (directory / name).write_text(text)

    @staticmethod
    def stat(tid: int, comm: str, user_ticks: int, system_ticks: int) -> str:
        # /proc/<pid>/task/<tid>/stat as Linux prints it; utime and stime are
        # the 14th and 15th fields.
        return (
            f"{tid} ({comm}) S 1 4242 4242 0 -1 4194560 2731 0 12 0 "
            f"{user_ticks} {system_ticks} 0 0 20 0 9 0 8112345 1234567168 5321 "
            "18446744073709551615 1 1 0 0 0 0 0 4096 81923 0 0 0 -1 7 0 0 0 0 0 "
            "0 0 0 0 0 0 0 0\n"
        )

    @staticmethod
    def status(voluntary: int, involuntary: int) -> str:
        return (
            "Name:\trocjitsu\n"
            "State:\tS (sleeping)\n"
            "Threads:\t9\n"
            "Cpus_allowed_list:\t0-31\n"
            f"voluntary_ctxt_switches:\t{voluntary}\n"
            f"nonvoluntary_ctxt_switches:\t{involuntary}\n"
        )

    def test_cpu_seconds_are_user_and_system_ticks_per_thread(self) -> None:
        self.thread(4242, stat=self.stat(4242, "rocjitsu", 250, 50))
        self.thread(4250, stat=self.stat(4250, "rj-dispatch-3", 1200, 0))

        usage = BENCH.process_cpu_seconds(self.PID, self.proc)

        self.assertEqual(
            usage,
            {
                "4242 rocjitsu": (250 / self.ticks, 50 / self.ticks),
                "4250 rj-dispatch-3": (1200 / self.ticks, 0.0),
            },
        )

    def test_thread_name_may_hold_spaces_and_parentheses(self) -> None:
        self.thread(4251, stat=self.stat(4251, "rj (vfio) S 9 9", 7, 3))

        usage = BENCH.process_cpu_seconds(self.PID, self.proc)

        self.assertEqual(
            usage, {"4251 rj (vfio) S 9 9": (7 / self.ticks, 3 / self.ticks)}
        )

    def test_threads_caught_while_exiting_are_left_out_of_cpu_seconds(self) -> None:
        self.thread(4242, stat=self.stat(4242, "rocjitsu", 250, 50))
        self.thread(4260)  # the directory is still listed, its files are gone
        self.thread(4261, stat="")
        self.thread(4262, stat="4262 (rj-dispatch-1) Z 1 4242\n")
        self.thread(4263, stat="4263 rj-dispatch-2\n")

        usage = BENCH.process_cpu_seconds(self.PID, self.proc)

        self.assertEqual(list(usage), ["4242 rocjitsu"])

    def test_a_process_that_is_gone_has_no_threads(self) -> None:
        self.assertEqual(BENCH.process_cpu_seconds(self.PID, self.proc), {})
        self.assertEqual(BENCH.process_sched(self.PID, self.proc), (0, 0, 0, 0))

    def test_scheduling_counters_are_summed_over_threads(self) -> None:
        # schedstat: nanoseconds on a CPU, nanoseconds waiting for one, timeslices.
        self.thread(4242, schedstat="5000000 250000 40\n", status=self.status(30, 2))
        self.thread(4250, schedstat="900000000 1750000 12\n", status=self.status(5, 11))

        self.assertEqual(
            BENCH.process_sched(self.PID, self.proc), (905000000, 2000000, 35, 13)
        )

    def test_threads_without_readable_scheduling_files_are_left_out(self) -> None:
        self.thread(4242, schedstat="5000000 250000 40\n", status=self.status(30, 2))
        self.thread(4260, schedstat="1 1 1\n")
        self.thread(4261, status=self.status(100, 100))
        self.thread(4262, schedstat="", status=self.status(100, 100))
        self.thread(4263, schedstat="7 seven 7\n", status=self.status(100, 100))

        self.assertEqual(
            BENCH.process_sched(self.PID, self.proc), (5000000, 250000, 30, 2)
        )

    def test_status_lines_without_a_count_are_ignored(self) -> None:
        status = (
            "voluntary_ctxt_switches:\n"
            "nonvoluntary_ctxt_switches:\tmany\n"
            "voluntary_ctxt_switches_total:\t99\n"
        )
        self.thread(4242, schedstat="10 20 1\n", status=status)

        self.assertEqual(BENCH.process_sched(self.PID, self.proc), (10, 20, 0, 0))

    def test_reads_the_live_proc_filesystem_by_default(self) -> None:
        usage = BENCH.process_cpu_seconds(os.getpid())
        on_cpu, waited, voluntary, involuntary = BENCH.process_sched(os.getpid())

        self.assertIn(os.getpid(), [int(thread.split()[0]) for thread in usage])
        for user, system in usage.values():
            self.assertGreaterEqual(user, 0.0)
            self.assertGreaterEqual(system, 0.0)
        self.assertGreaterEqual(min(on_cpu, waited, voluntary, involuntary), 0)


class SummarizeTest(unittest.TestCase):
    def test_reports_median_p99_and_minimum_of_unordered_samples(self) -> None:
        samples = [float(value) for value in range(200, 0, -1)]

        self.assertEqual(
            BENCH.summarize(samples),
            {"median_us": 100.5, "p99_us": 199.0, "min_us": 1.0},
        )

    def test_p99_of_a_short_run_is_its_largest_sample(self) -> None:
        self.assertEqual(
            BENCH.summarize([5.0, 1.0, 3.0, 2.0, 4.0]),
            {"median_us": 3.0, "p99_us": 5.0, "min_us": 1.0},
        )
        self.assertEqual(
            BENCH.summarize([7.5]),
            {"median_us": 7.5, "p99_us": 7.5, "min_us": 7.5},
        )


def make_tool(path: Path) -> str:
    """An executable file standing in for a toolchain binary that is never run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 1\n")
    path.chmod(0o755)
    return str(path)


class CodeObjectTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.toolchain = Path(directory.name)
        self.clang = make_tool(self.toolchain / "bin/clang")
        self.objdump = make_tool(self.toolchain / "bin/llvm-objdump")

    def build(
        self, elf: bytes, target: str = "gfx1151"
    ) -> tuple[BENCH.CodeObject, list[list[str]]]:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        commands: list[list[str]] = []

        def run(command: list[str], **options: object) -> subprocess.CompletedProcess:
            commands.append(command)
            if "-o" in command:
                Path(command[command.index("-o") + 1]).write_bytes(elf)
                return subprocess.CompletedProcess(command, 0)
            return subprocess.CompletedProcess(command, 0, stdout=LISTING)

        with mock.patch.object(BENCH.subprocess, "run", side_effect=run):
            code = BENCH.build_code_object(self.clang, target, Path(directory.name))
        return code, commands

    def test_image_places_each_load_segment_at_its_virtual_address(self) -> None:
        code, _ = self.build(synthetic_elf())

        descriptors_end = DESCRIPTOR_VADDR + len(KERNEL_NAMES) * DESCRIPTOR_BYTES
        self.assertEqual(len(code.image), CODE_VADDR + CODE_MEMORY_BYTES)
        self.assertEqual(code.image[:DESCRIPTOR_VADDR], bytes(DESCRIPTOR_VADDR))
        self.assertEqual(
            code.image[CODE_VADDR : CODE_VADDR + len(CODE_BYTES)], CODE_BYTES
        )
        # Memory past the bytes a segment has in the file reads as zero.
        self.assertEqual(
            code.image[CODE_VADDR + len(CODE_BYTES) :],
            bytes(CODE_MEMORY_BYTES - len(CODE_BYTES)),
        )
        self.assertEqual(
            code.image[descriptors_end:CODE_VADDR], bytes(CODE_VADDR - descriptors_end)
        )

    def test_kernels_are_found_through_their_descriptor_symbols(self) -> None:
        code, _ = self.build(synthetic_elf(wave32_kernels=("fp_loop",)))

        self.assertEqual(set(code.kernels), set(KERNEL_NAMES))
        for index, name in enumerate(KERNEL_NAMES):
            with self.subTest(kernel=name):
                kernel = code.kernels[name]
                self.assertEqual(kernel.name, name)
                self.assertEqual(
                    kernel.kd_offset, DESCRIPTOR_VADDR + index * DESCRIPTOR_BYTES
                )
        properties_at = code.kernels["fp_loop"].kd_offset + BENCH.KD_CODE_PROPERTIES
        self.assertEqual(
            struct.unpack_from("<H", code.image, properties_at)[0],
            BENCH.KD_ENABLE_WAVEFRONT_SIZE32,
        )

    def test_loop_lengths_come_from_the_disassembly(self) -> None:
        code, _ = self.build(synthetic_elf())

        self.assertEqual(code.kernels["noop"].loop_insts, 0)
        self.assertEqual(code.kernels["int_loop"].loop_insts, 6)

    def test_wave_width_follows_the_target_and_the_descriptor(self) -> None:
        elf = synthetic_elf(wave32_kernels=("fp_loop",))
        expectations = {
            # RDNA reads the descriptor; CDNA4 is wave64 and CDNA5 wave32 regardless.
            "gfx1151": {"int_loop": 64, "fp_loop": 32},
            "gfx950": {"int_loop": 64, "fp_loop": 64},
            "gfx1250": {"int_loop": 32, "fp_loop": 32},
        }
        for target, lanes in expectations.items():
            with self.subTest(target=target):
                code, _ = self.build(elf, target)

                self.assertEqual(code.target, target)
                for name, expected_lanes in lanes.items():
                    self.assertEqual(code.kernels[name].wave_lanes, expected_lanes)

    def test_compiles_for_the_target_and_disassembles_with_the_sibling_objdump(
        self,
    ) -> None:
        code, commands = self.build(synthetic_elf(), "gfx1250")

        compile_command, disassemble_command = commands
        self.assertEqual(compile_command[0], self.clang)
        self.assertIn("-mcpu=gfx1250", compile_command)
        self.assertEqual(compile_command[compile_command.index("-x") + 1], "cl")
        self.assertEqual(disassemble_command[0], self.objdump)
        self.assertIn("--mcpu=gfx1250", disassemble_command)
        # The file that is compiled is the file that is disassembled.
        self.assertEqual(disassemble_command[-1], compile_command[-1])
        self.assertEqual(code.target, "gfx1250")

    def test_rejects_compiler_output_that_is_not_elf(self) -> None:
        with self.assertRaisesRegex(BENCH.HarnessError, "ELF"):
            self.build(b"not an object file")

    def test_rejects_a_code_object_missing_a_kernel_descriptor(self) -> None:
        without_lds = tuple(name for name in KERNEL_NAMES if name != "lds_loop")

        with self.assertRaisesRegex(BENCH.HarnessError, "lds_loop"):
            self.build(synthetic_elf(kernel_names=without_lds))

    def test_rejects_a_kernel_that_needs_private_segment_memory(self) -> None:
        with self.assertRaisesRegex(BENCH.HarnessError, "128-byte private segment"):
            self.build(synthetic_elf(private_segment_bytes=128))

    def test_rejects_a_code_object_without_a_loadable_segment(self) -> None:
        with self.assertRaisesRegex(BENCH.HarnessError, "no loadable segment"):
            self.build(synthetic_elf(load_segments=False))

    def test_compiler_that_fails_is_reported_with_its_exit_status(self) -> None:
        failure = subprocess.CalledProcessError(1, [self.clang])

        with mock.patch.object(BENCH.subprocess, "run", side_effect=failure):
            with self.assertRaisesRegex(
                BENCH.HarnessError,
                "clang could not compile the kernels for gfx1250.*exit status 1",
            ):
                BENCH.build_code_object(self.clang, "gfx1250", self.toolchain)

    def test_compiler_that_cannot_be_started_is_reported(self) -> None:
        # The stand-in is a file the kernel refuses to execute.
        Path(self.clang).write_bytes(b"\0not a program")

        with self.assertRaisesRegex(
            BENCH.HarnessError, "cannot run .*clang to compile the kernels"
        ):
            BENCH.build_code_object(self.clang, "gfx1250", self.toolchain)


class ToolchainDiscoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path_directory = self.root / "path"
        self.path_directory.mkdir()

    def environment(self, **variables: str) -> contextlib.AbstractContextManager:
        """The process environment with only PATH and the given variables set."""
        values = {"PATH": str(self.path_directory), **variables}
        return mock.patch.dict(os.environ, values, clear=True)

    def test_an_explicit_clang_is_used_as_given(self) -> None:
        make_tool(self.root / "rocm/lib/llvm/bin/clang")
        chosen = make_tool(self.root / "chosen/clang")

        with self.environment(ROCM_PATH=str(self.root / "rocm")):
            self.assertEqual(BENCH.find_clang(chosen), chosen)

    def test_an_explicit_clang_that_is_not_there_is_an_error(self) -> None:
        make_tool(self.root / "rocm/lib/llvm/bin/clang")
        plain = self.root / "plain-file"
        plain.write_text("not a program")

        with self.environment(ROCM_PATH=str(self.root / "rocm")):
            for requested in ("/no/such/clang", str(plain)):
                with self.subTest(requested=requested):
                    with self.assertRaisesRegex(
                        BENCH.HarnessError, f"--clang {requested} is not an executable"
                    ):
                        BENCH.find_clang(requested)

    def test_clang_comes_from_the_llvm_of_rocm_path(self) -> None:
        make_tool(self.path_directory / "clang")
        for layout in ("lib/llvm/bin", "llvm/bin"):
            with self.subTest(layout=layout):
                rocm = self.root / layout.replace("/", "-")
                clang = make_tool(rocm / layout / "clang")

                with self.environment(ROCM_PATH=str(rocm)):
                    self.assertEqual(BENCH.find_clang(None), clang)

    def test_the_packaged_layout_is_preferred_within_one_rocm_tree(self) -> None:
        rocm = self.root / "rocm"
        packaged = make_tool(rocm / "lib/llvm/bin/clang")
        make_tool(rocm / "llvm/bin/clang")

        with self.environment(ROCM_PATH=str(rocm)):
            self.assertEqual(BENCH.find_clang(None), packaged)

    def test_clang_falls_back_to_path_preferring_amdclang(self) -> None:
        clang = make_tool(self.path_directory / "clang")

        with self.environment(ROCM_PATH=str(self.root / "empty")):
            self.assertEqual(BENCH.find_clang(None), clang)
            amdclang = make_tool(self.path_directory / "amdclang")
            self.assertEqual(BENCH.find_clang(None), amdclang)

    def test_no_clang_anywhere_is_an_error_that_names_the_remedies(self) -> None:
        with self.environment(ROCM_PATH=str(self.root / "empty")):
            with self.assertRaisesRegex(BENCH.HarnessError, "--clang or set ROCM_PATH"):
                BENCH.find_clang(None)

    def test_objdump_is_taken_from_beside_clang(self) -> None:
        make_tool(self.path_directory / "llvm-objdump")
        clang = make_tool(self.root / "llvm/bin/clang")
        objdump = make_tool(self.root / "llvm/bin/llvm-objdump")

        with self.environment():
            self.assertEqual(BENCH.find_objdump(clang), objdump)

    def test_objdump_is_found_through_a_symbolic_link_to_clang(self) -> None:
        clang = make_tool(self.root / "llvm/bin/clang")
        objdump = make_tool(self.root / "llvm/bin/llvm-objdump")
        (self.root / "bin").mkdir()
        (self.root / "bin/amdclang").symlink_to(clang)

        with self.environment():
            self.assertEqual(
                BENCH.find_objdump(str(self.root / "bin/amdclang")),
                str(Path(objdump).resolve()),
            )

    def test_objdump_of_a_clang_named_without_a_directory_is_looked_up_on_path(
        self,
    ) -> None:
        make_tool(self.path_directory / "clang")
        objdump = make_tool(self.path_directory / "llvm-objdump")

        with self.environment():
            self.assertEqual(BENCH.find_objdump("clang"), objdump)

    def test_objdump_falls_back_to_path(self) -> None:
        clang = make_tool(self.root / "llvm/bin/clang")
        objdump = make_tool(self.path_directory / "llvm-objdump")

        with self.environment():
            self.assertEqual(BENCH.find_objdump(clang), objdump)

    def test_missing_objdump_is_an_error(self) -> None:
        clang = make_tool(self.root / "llvm/bin/clang")

        with self.environment():
            with self.assertRaisesRegex(BENCH.HarnessError, "no llvm-objdump"):
                BENCH.find_objdump(clang)


class PreconnectedSocket(socket.socket):
    """One end of a socketpair that accepts the connect() the client makes."""

    requested_address: str | None = None

    def connect(self, address: str) -> None:
        self.requested_address = address


class Request(NamedTuple):
    msg_id: int
    command: int
    flags: int
    error: int
    payload: bytes
    fds: list[int]


class FakeVfioUserServer:
    """The server end of a socketpair, driven by the test between client calls.

    The client is synchronous, so a reply is queued before the call that will
    read it and the request is inspected afterwards; the socket buffers both.
    """

    REPLY = 0x1

    def __init__(self) -> None:
        client_end, self.peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        self.client_end = PreconnectedSocket(fileno=client_end.detach())
        self.peer.settimeout(5.0)
        self.received_fds: list[int] = []

    def close(self) -> None:
        for fd in self.received_fds:
            os.close(fd)
        self.peer.close()
        self.client_end.close()

    def connect(self) -> BENCH.VfioUserClient:
        """Run the client's constructor against this server; returns the client."""
        self.queue_reply(struct.pack("<HH", 0, 2) + b'{"capabilities":{}}\0')
        with mock.patch.object(
            BENCH.socket, "socket", return_value=self.client_end
        ) as factory:
            client = BENCH.VfioUserClient("/run/rocjitsu/fake.sock", timeout=2.0)
        factory.assert_called_once_with(socket.AF_UNIX, socket.SOCK_STREAM)
        return client

    def queue_reply(
        self,
        body: bytes = b"",
        flags: int = REPLY,
        error: int = 0,
        fds: tuple[int, ...] = (),
    ) -> None:
        size = BENCH.HEADER.size + len(body)
        message = BENCH.HEADER.pack(0, 0, size, flags, error) + body
        if fds:
            socket.send_fds(self.peer, [message], list(fds))
        else:
            self.peer.sendall(message)

    def take_request(self) -> Request:
        first, fds, _, _ = socket.recv_fds(self.peer, BENCH.HEADER.size, 8)
        self.received_fds += fds
        header = first + self._receive(BENCH.HEADER.size - len(first))
        msg_id, command, size, flags, error = BENCH.HEADER.unpack(header)
        payload = self._receive(size - BENCH.HEADER.size)
        return Request(msg_id, command, flags, error, payload, fds)

    def _receive(self, length: int) -> bytes:
        received = bytearray()
        while len(received) < length:
            chunk = self.peer.recv(length - len(received))
            if not chunk:
                raise AssertionError("the client closed the connection mid-request")
            received += chunk
        return bytes(received)


def region_read_reply(offset: int, region: int, data: bytes) -> bytes:
    return struct.pack("<QII", offset, region, len(data)) + data


class VfioUserClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server = FakeVfioUserServer()
        self.addCleanup(self.server.close)
        self.client = self.server.connect()
        self.handshake = self.server.take_request()

    def test_connects_to_the_socket_path_and_negotiates_the_version(self) -> None:
        self.assertEqual(
            self.server.client_end.requested_address, "/run/rocjitsu/fake.sock"
        )
        self.assertEqual(self.handshake.command, BENCH.CMD_VERSION)
        self.assertEqual(self.handshake.msg_id, 1)
        self.assertEqual(self.handshake.flags, 0)
        self.assertEqual(self.handshake.error, 0)
        self.assertEqual(self.handshake.fds, [])
        # Major 0, minor 2, then NUL-terminated capability JSON.
        self.assertEqual(self.handshake.payload[:4], b"\x00\x00\x02\x00")
        self.assertEqual(self.handshake.payload[-1:], b"\0")
        self.assertEqual(
            json.loads(self.handshake.payload[4:-1]),
            {"capabilities": {"max_msg_fds": 8, "max_data_xfer_size": 1 << 20}},
        )

    def test_region_write_carries_offset_region_count_and_data(self) -> None:
        self.server.queue_reply()

        self.client.region_write(BENCH.BAR_REGISTERS, 0x1234, b"\xaa\xbb\xcc\xdd")

        request = self.server.take_request()
        self.assertEqual(request.command, BENCH.CMD_REGION_WRITE)
        self.assertEqual(request.flags, 0)
        self.assertEqual(
            request.payload,
            bytes.fromhex("3412000000000000 05000000 04000000 aabbccdd"),
        )

    def test_region_read_asks_for_a_range_and_returns_its_bytes(self) -> None:
        self.server.queue_reply(
            region_read_reply(0x40, BENCH.BAR_DOORBELL, b"\x11\x22\x33\x44")
        )

        data = self.client.region_read(BENCH.BAR_DOORBELL, 0x40, 4)

        request = self.server.take_request()
        self.assertEqual(data, b"\x11\x22\x33\x44")
        self.assertEqual(request.command, BENCH.CMD_REGION_READ)
        self.assertEqual(request.flags, 0)
        self.assertEqual(
            request.payload, bytes.fromhex("4000000000000000 02000000 04000000")
        )

    def test_each_request_takes_the_next_message_id(self) -> None:
        for _ in range(3):
            self.server.queue_reply()
            self.client.region_write(BENCH.BAR_REGISTERS, 0, b"\0\0\0\0")

        ids = [self.server.take_request().msg_id for _ in range(3)]

        self.assertEqual(ids, [2, 3, 4])

    def test_error_reply_raises_and_leaves_the_stream_in_step(self) -> None:
        self.server.queue_reply(
            b"\xee" * 16,
            flags=FakeVfioUserServer.REPLY | BENCH.F_ERROR,
            error=errno.ENOENT,
        )
        self.server.queue_reply(
            region_read_reply(0, BENCH.BAR_REGISTERS, b"\x01\0\0\0")
        )

        with self.assertRaises(BENCH.HarnessError) as raised:
            self.client.region_read(BENCH.BAR_REGISTERS, 0, 4)

        self.assertIn(f"command {BENCH.CMD_REGION_READ}", str(raised.exception))
        self.assertIn(os.strerror(errno.ENOENT), str(raised.exception))
        # The failed reply's body was consumed, so the next reply is read whole.
        self.assertEqual(
            self.client.region_read(BENCH.BAR_REGISTERS, 0, 4), b"\x01\0\0\0"
        )

    def test_posted_write_asks_for_no_reply_and_reads_none(self) -> None:
        value = struct.pack("<Q", 7)

        # Nothing is queued: returning at all shows no reply was awaited.
        self.client.region_write(
            BENCH.BAR_DOORBELL, BENCH.AQL_DOORBELL, value, posted=True
        )

        request = self.server.take_request()
        self.assertEqual(request.command, BENCH.CMD_REGION_WRITE)
        self.assertEqual(request.flags, BENCH.F_NO_REPLY)
        self.assertEqual(
            request.payload,
            struct.pack("<QII", BENCH.AQL_DOORBELL, BENCH.BAR_DOORBELL, 8) + value,
        )

        # A reply already waiting belongs to the next command, not to a posted write.
        self.server.queue_reply(
            region_read_reply(0, BENCH.BAR_REGISTERS, b"\x55\x66\x77\x88")
        )
        self.client.region_write(
            BENCH.BAR_DOORBELL, BENCH.AQL_DOORBELL, value, posted=True
        )
        self.assertEqual(
            self.client.region_read(BENCH.BAR_REGISTERS, 0, 4), b"\x55\x66\x77\x88"
        )

    def test_end_of_stream_instead_of_a_reply_raises(self) -> None:
        self.server.peer.shutdown(socket.SHUT_WR)

        with self.assertRaisesRegex(BENCH.HarnessError, "closed the connection"):
            self.client.region_read(BENCH.BAR_REGISTERS, 0, 4)

    def test_reply_cut_short_raises(self) -> None:
        self.server.peer.sendall(BENCH.HEADER.pack(0, 0, 32, 1, 0)[:8])
        self.server.peer.shutdown(socket.SHUT_WR)

        with self.assertRaisesRegex(BENCH.HarnessError, "closed the connection"):
            self.client.region_read(BENCH.BAR_REGISTERS, 0, 4)

    def test_dma_map_shares_the_memory_descriptor_with_the_window(self) -> None:
        memory = os.memfd_create("bench-test-ram", os.MFD_CLOEXEC)
        self.addCleanup(os.close, memory)
        self.server.queue_reply()

        self.client.dma_map(0x1000_0000, 0x20_0000, memory)

        request = self.server.take_request()
        self.assertEqual(request.command, BENCH.CMD_DMA_MAP)
        argsz, flags, fd_offset, iova, size = struct.unpack("<IIQQQ", request.payload)
        self.assertEqual(argsz, len(request.payload))
        self.assertEqual(flags, BENCH.DMA_READ | BENCH.DMA_WRITE | BENCH.DMA_MMAP)
        self.assertEqual((fd_offset, iova, size), (0, 0x1000_0000, 0x20_0000))
        self.assertEqual(len(request.fds), 1)
        # The descriptor that arrived is the caller's memory, not a copy of it.
        os.pwrite(request.fds[0], b"guest", 0)
        self.assertEqual(os.pread(memory, 5, 0), b"guest")

    def test_arm_irq_hands_over_one_eventfd_trigger(self) -> None:
        trigger = os.eventfd(0, os.EFD_CLOEXEC)
        self.addCleanup(os.close, trigger)
        self.server.queue_reply()

        self.client.arm_irq(BENCH.IRQ_MSIX, trigger)

        request = self.server.take_request()
        self.assertEqual(request.command, BENCH.CMD_SET_IRQS)
        argsz, flags, index, start, count = struct.unpack("<IIIII", request.payload)
        self.assertEqual(argsz, len(request.payload))
        self.assertEqual(flags, BENCH.IRQ_SET_EVENTFD_TRIGGER)
        self.assertEqual((index, start, count), (BENCH.IRQ_MSIX, 0, 1))
        self.assertEqual(len(request.fds), 1)
        os.eventfd_write(request.fds[0], 5)
        self.assertEqual(os.eventfd_read(trigger), 5)

    def test_region_size_reads_the_size_field_of_the_region_info(self) -> None:
        self.server.queue_reply(struct.pack("<IIIIQQ", 32, 7, 5, 0, 0x8_0000, 0))

        size = self.client.region_size(BENCH.BAR_REGISTERS)

        request = self.server.take_request()
        self.assertEqual(size, 0x8_0000)
        self.assertEqual(request.command, BENCH.CMD_GET_REGION_INFO)
        argsz, _, index, _, _, _ = struct.unpack("<IIIIQQ", request.payload)
        self.assertEqual((argsz, index), (len(request.payload), BENCH.BAR_REGISTERS))

    def test_region_mmap_maps_the_descriptor_at_the_offset_the_server_names(
        self,
    ) -> None:
        backing = os.memfd_create("bench-test-vram", os.MFD_CLOEXEC)
        self.addCleanup(os.close, backing)
        os.ftruncate(backing, 2 * PAGE)
        os.pwrite(backing, b"second page", PAGE)
        self.server.queue_reply(
            struct.pack("<IIIIQQ", 32, 7, BENCH.BAR_VRAM, 0, PAGE, PAGE), fds=(backing,)
        )

        mapping = self.client.region_mmap(BENCH.BAR_VRAM)

        self.assertIsNotNone(mapping)
        self.addCleanup(mapping.close)
        self.assertEqual(len(mapping), PAGE)
        self.assertEqual(mapping[:11], b"second page")
        # The mapping is shared with the server's memory, not a private copy.
        mapping[0:5] = b"SECON"
        self.assertEqual(os.pread(backing, 5, PAGE), b"SECON")
        request = self.server.take_request()
        self.assertEqual(
            struct.unpack_from("<I", request.payload, 8)[0], BENCH.BAR_VRAM
        )

    def test_region_mmap_returns_none_for_a_region_the_device_traps(self) -> None:
        self.server.queue_reply(
            struct.pack("<IIIIQQ", 32, 3, BENCH.BAR_REGISTERS, 0, 0x8_0000, 0)
        )

        self.assertIsNone(self.client.region_mmap(BENCH.BAR_REGISTERS))

    def test_region_info_cut_short_is_an_error_and_keeps_no_descriptor(self) -> None:
        backing = os.memfd_create("bench-test-vram", os.MFD_CLOEXEC)
        self.addCleanup(os.close, backing)
        open_before = len(os.listdir("/proc/self/fd"))
        self.server.queue_reply(struct.pack("<IIII", 32, 7, 0, 0), fds=(backing,))

        with self.assertRaisesRegex(BENCH.HarnessError, "region 0 info .* truncated"):
            self.client.region_mmap(BENCH.BAR_VRAM)

        self.assertEqual(len(os.listdir("/proc/self/fd")), open_before)

    def test_region_that_cannot_be_mapped_is_an_error(self) -> None:
        backing = os.memfd_create("bench-test-vram", os.MFD_CLOEXEC)
        self.addCleanup(os.close, backing)
        os.ftruncate(backing, PAGE)
        # The server names more than its descriptor holds.
        self.server.queue_reply(
            struct.pack("<IIIIQQ", 32, 7, BENCH.BAR_VRAM, 0, 2 * PAGE, 0),
            fds=(backing,),
        )

        with self.assertRaisesRegex(BENCH.HarnessError, "cannot map vfio-user region"):
            self.client.region_mmap(BENCH.BAR_VRAM)


class DeviceLayoutTest(DeviceTestCase):
    def test_system_objects_fit_the_gart_window_below_process_memory(self) -> None:
        system_objects = (
            BENCH.SYS_MES_RING,
            BENCH.SYS_MES_RPTR,
            BENCH.SYS_MES_STATUS,
            BENCH.SYS_MES_FENCE,
            BENCH.SYS_MQD,
            BENCH.SYS_IH_RING,
            BENCH.SYS_IH_WPTR,
        )

        self.assertEqual(len(set(system_objects)), len(system_objects))
        self.assertLessEqual(max(system_objects) + PAGE, BENCH.GART_PAGES * PAGE)
        self.assertLessEqual(BENCH.GART_PAGES * PAGE, BENCH.PROC_GUEST_OFFSET)

    def test_guest_ram_is_shared_as_one_window_at_the_guest_base(self) -> None:
        device, fake = self.make_device()

        total = BENCH.PROC_GUEST_OFFSET + PROCESS_BYTES
        self.assertEqual(fake.dma_window, (BENCH.GUEST_BASE, total))
        self.assertEqual(len(fake.guest), total)
        # Both sides see the same pages.
        device.put(0x7000, "<Q", 0x1122_3344_5566_7788)
        self.assertEqual(read_u64(fake.guest, 0x7000), 0x1122_3344_5566_7788)

    def test_gart_maps_every_system_page_onto_guest_ram_in_order(self) -> None:
        for mappable_vram in (True, False):
            with self.subTest(mappable_vram=mappable_vram):
                _, fake = self.make_device(mappable_vram)

                for page in range(BENCH.GART_PAGES):
                    address = BENCH.GART_START + page * PAGE + 0x123
                    self.assertEqual(
                        fake.gart_to_guest_offset(address), page * PAGE + 0x123
                    )
                for outside in (
                    BENCH.GART_START - 1,
                    BENCH.GART_START + BENCH.GART_PAGES * PAGE,
                ):
                    with self.assertRaises(TranslationFault):
                        fake.gart_to_guest_offset(outside)

    def test_trapped_vram_is_written_in_register_sized_pieces(self) -> None:
        _, trapped = self.make_device(mappable_vram=False)
        _, mapped = self.make_device(mappable_vram=True)

        self.assertEqual(trapped.largest_trapped_vram_write, 8)
        self.assertEqual(mapped.largest_trapped_vram_write, 0)
        # Either way the device ends up with the same tables.
        tables_end = BENCH.VRAM_PTB_BASE + 3 * PAGE
        self.assertEqual(trapped.vram[:tables_end], mapped.vram[:tables_end])

    def test_interrupt_ring_and_vector_are_programmed_at_bus_addresses(self) -> None:
        device, fake = self.make_device()

        registers = fake.registers
        ring = joined_address(registers[BENCH.IH_BASE_HI], registers[BENCH.IH_BASE], 8)
        write_pointer = joined_address(
            registers[BENCH.IH_WPTR_ADDR_HI], registers[BENCH.IH_WPTR_ADDR_LO]
        )
        control = registers[BENCH.IH_CONTROL]
        self.assertEqual(fake.bus_to_guest_offset(ring), BENCH.SYS_IH_RING)
        self.assertEqual(fake.bus_to_guest_offset(write_pointer), BENCH.SYS_IH_WPTR)
        for bit in (
            BENCH.IH_RING_ENABLE,
            BENCH.IH_INTERRUPT_ENABLE,
            BENCH.IH_BUS_ADDRESS,
        ):
            self.assertEqual(control & bit, bit)
        self.assertEqual(fake.irqs, [(BENCH.IRQ_MSIX, device.irq)])

    def test_mes_ring_registers_lead_to_the_add_queue_frame(self) -> None:
        _, fake = self.make_device()

        # The fake found this frame by following the ring registers through the GART.
        self.assertEqual(len(fake.mes_frames), 1)
        registration = fake.queue_registration()
        self.assertEqual(registration.opcode, BENCH.MES_ADD_QUEUE)
        self.assertEqual(registration.queue_type, BENCH.MES_QUEUE_COMPUTE)
        self.assertEqual(
            fake.doorbells, [(BENCH.MES_DOORBELL, BENCH.MES_FRAME_DWORDS, False)]
        )
        read_pointer = fake.gc_address(BENCH.REG_MES_RPTR_HI, BENCH.REG_MES_RPTR_LO)
        self.assertEqual(fake.gart_to_guest_offset(read_pointer), BENCH.SYS_MES_RPTR)

    def test_process_page_tables_map_all_of_process_memory_onto_guest_ram(
        self,
    ) -> None:
        for mappable_vram in (True, False):
            with self.subTest(mappable_vram=mappable_vram):
                _, fake = self.make_device(mappable_vram)
                root = fake.queue_registration().page_table_root

                # 3 MiB spans two page tables; probe both and their boundary.
                for offset in (
                    0x123,
                    (2 << 20) - 1,
                    (2 << 20),
                    PROCESS_BYTES - 1,
                ):
                    self.assertEqual(
                        fake.process_to_guest_offset(root, BENCH.PROC_VA_BASE + offset),
                        BENCH.PROC_GUEST_OFFSET + offset,
                    )
                self.assertIsNone(fake.walk(root, BENCH.PROC_VA_BASE - 1))
                self.assertIsNone(fake.walk(root, BENCH.PROC_VA_BASE + PROCESS_BYTES))

    def test_vram_window_follows_process_memory_and_maps_vram_not_guest_ram(
        self,
    ) -> None:
        device, fake = self.make_device()
        root = fake.queue_registration().page_table_root

        # Process memory ends inside its second 2 MiB table; the window takes the next.
        self.assertEqual(device.vram_va, BENCH.PROC_VA_BASE + (4 << 20))
        for offset in (0, PAGE, BENCH.VRAM_WINDOW_BYTES - PAGE):
            entry = fake.walk(root, device.vram_va + offset)
            self.assertIsNotNone(entry)
            self.assertEqual(entry & PTE_ADDRESS_MASK, BENCH.VRAM_DATA + offset)
            self.assertFalse(entry & PTE_SYSTEM)
        self.assertIsNone(fake.walk(root, device.vram_va + BENCH.VRAM_WINDOW_BYTES))

    def test_alloc_hands_out_page_aligned_ranges_that_do_not_overlap(self) -> None:
        device, fake = self.make_device()
        root = fake.queue_registration().page_table_root
        used = device.proc_cursor

        first_va, first_offset = device.alloc(1)
        second_va, second_offset = device.alloc(PAGE + 1)
        third_va, _ = device.alloc(PAGE)

        self.assertEqual(first_va, BENCH.PROC_VA_BASE + used)
        self.assertEqual(second_va - first_va, PAGE)
        self.assertEqual(third_va - second_va, 2 * PAGE)
        self.assertEqual(fake.process_to_guest_offset(root, first_va), first_offset)
        self.assertEqual(fake.process_to_guest_offset(root, second_va), second_offset)

    def test_alloc_refuses_to_run_past_process_memory(self) -> None:
        device, _ = self.make_device()
        remaining = PROCESS_BYTES - device.proc_cursor

        device.alloc(remaining)

        with self.assertRaisesRegex(BENCH.HarnessError, "--memory-mib"):
            device.alloc(1)


class DeviceTeardownTest(unittest.TestCase):
    def test_close_gives_back_every_descriptor_the_device_opened(self) -> None:
        class DescriptorBackedVram(FakeDevice):
            """Maps the VRAM BAR from a descriptor, as the real client does."""

            def region_mmap(self, region: int) -> mmap.mmap | None:
                backing = os.memfd_create("bench-test-vram", os.MFD_CLOEXEC)
                try:
                    os.ftruncate(backing, VRAM_BYTES)
                    self.vram = mmap.mmap(backing, VRAM_BYTES)
                finally:
                    os.close(backing)
                return self.vram

        fake = DescriptorBackedVram(mappable_vram=True)
        self.addCleanup(fake.close)
        open_before = open_descriptors()

        device = BENCH.Device(fake, PROCESS_BYTES)
        # Guest RAM and its mapping, the VRAM mapping, the interrupt eventfd,
        # and the fake's own mapping of the guest RAM it was handed.
        self.assertEqual(open_descriptors(), open_before + 5)
        device.close()

        self.assertEqual(open_descriptors(), open_before + 1)
        self.assertTrue(device.mem.closed)
        self.assertTrue(fake.vram.closed)

    def test_bring_up_that_fails_leaves_nothing_open(self) -> None:
        class RefusingDevice(FakeDevice):
            def arm_irq(self, index: int, fd: int) -> None:
                raise BENCH.HarnessError("vfio-user command 8 failed")

        fake = RefusingDevice(mappable_vram=True)
        self.addCleanup(fake.close)
        open_before = open_descriptors()

        with self.assertRaisesRegex(BENCH.HarnessError, "command 8 failed"):
            BENCH.Device(fake, PROCESS_BYTES)
        fake.close()

        self.assertEqual(open_descriptors(), open_before)


class DeviceDispatchTest(DeviceTestCase):
    def setUp(self) -> None:
        self.device, self.fake = self.make_device()
        self.code = loaded_code_object()
        self.device.load(self.code)
        self.registration = self.fake.queue_registration()
        self.root = self.registration.page_table_root

    def guest_offset(self, address: int) -> int:
        return self.fake.process_to_guest_offset(self.root, address)

    def queue_descriptor(self) -> int:
        """Guest RAM offset of the MQD the ADD_QUEUE frame points at."""
        return self.fake.gart_to_guest_offset(self.registration.mqd_address)

    def packet(self, slot: int) -> DispatchPacket:
        """The packet in a ring slot, found through the MQD as the device finds it."""
        descriptor = self.queue_descriptor()
        ring = joined_address(
            read_u32(self.fake.guest, descriptor + MQD_RING_BASE_HI * 4),
            read_u32(self.fake.guest, descriptor + MQD_RING_BASE_LO * 4),
            8,
        )
        offset = self.guest_offset(ring + slot * DISPATCH_PACKET.size)
        return DispatchPacket._make(
            DISPATCH_PACKET.unpack_from(self.fake.guest, offset)
        )

    def write_pointer(self) -> int:
        return read_u64(
            self.fake.guest, self.guest_offset(self.registration.write_pointer_address)
        )

    def test_registered_queue_leads_the_device_to_the_dispatch_packet(self) -> None:
        kernel = self.code.kernels["int_loop"]
        arguments = bytes(range(1, 21))

        slot = self.device.enqueue(kernel, 128, arguments)

        packet = self.packet(slot)
        self.assertEqual(slot, 0)
        self.assertEqual(packet.header & 0xFF, HSA_PACKET_TYPE_KERNEL_DISPATCH)
        self.assertEqual(packet.setup, 1)
        self.assertEqual(
            (packet.workgroup_size_x, packet.workgroup_size_y, packet.workgroup_size_z),
            (BENCH.WORKGROUP, 1, 1),
        )
        self.assertEqual(
            (packet.grid_size_x, packet.grid_size_y, packet.grid_size_z), (128, 1, 1)
        )
        # Segment sizes are taken from the kernel descriptor.
        self.assertEqual(packet.private_segment_size, 0)
        self.assertEqual(packet.group_segment_size, 256)
        descriptor = self.guest_offset(packet.kernel_object)
        self.assertEqual(
            self.fake.guest[descriptor : descriptor + DESCRIPTOR_BYTES],
            self.code.image[kernel.kd_offset : kernel.kd_offset + DESCRIPTOR_BYTES],
        )
        kernarg = self.guest_offset(packet.kernarg_address)
        self.assertEqual(
            self.fake.guest[kernarg : kernarg + PAGE], arguments.ljust(PAGE, b"\0")
        )
        # A non-zero signal value marks the dispatch as still running.
        signal_value = self.guest_offset(packet.completion_signal) + BENCH.SIGNAL_VALUE
        self.assertEqual(struct.unpack_from("<q", self.fake.guest, signal_value)[0], 1)

    def test_enqueue_alone_publishes_nothing(self) -> None:
        self.device.enqueue(self.code.kernels["noop"], BENCH.WORKGROUP, b"")

        self.assertEqual(self.write_pointer(), 0)
        self.assertEqual(len(self.fake.doorbells), 1)

    def test_kick_publishes_the_write_index_and_rings_the_registered_doorbell(
        self,
    ) -> None:
        self.device.enqueue(self.code.kernels["noop"], BENCH.WORKGROUP, b"")

        self.device.kick()

        offset, value, posted = self.fake.doorbells[-1]
        self.assertEqual(self.write_pointer(), 1)
        self.assertEqual(offset, self.registration.doorbell_dword * 4)
        self.assertEqual(
            offset,
            read_u32(self.fake.guest, self.queue_descriptor() + MQD_DOORBELL * 4),
        )
        # The doorbell carries the index of the last published packet.
        self.assertEqual(value, 0)
        self.assertFalse(posted)

    def test_one_kick_covers_every_packet_enqueued_since_the_last(self) -> None:
        kernel = self.code.kernels["noop"]

        slots = [self.device.enqueue(kernel, BENCH.WORKGROUP, b"") for _ in range(3)]
        self.device.kick()

        self.assertEqual(slots, [0, 1, 2])
        self.assertEqual(self.write_pointer(), 3)
        self.assertEqual(self.fake.doorbells[-1][1], 2)
        self.assertEqual(len(self.fake.doorbells), 2)
        signals = {self.packet(slot).completion_signal for slot in slots}
        self.assertEqual(len(signals), 3)

    def test_posted_doorbells_are_sent_without_waiting_for_a_reply(self) -> None:
        self.device.posted_doorbells = True
        self.device.enqueue(self.code.kernels["noop"], BENCH.WORKGROUP, b"")

        self.device.kick()

        self.assertTrue(self.fake.doorbells[-1][2])

    def test_wait_returns_once_the_device_clears_the_signal(self) -> None:
        slot = self.device.enqueue(self.code.kernels["noop"], BENCH.WORKGROUP, b"")
        completion_signal = self.packet(slot).completion_signal
        signal_value = self.guest_offset(completion_signal) + BENCH.SIGNAL_VALUE
        server = types.SimpleNamespace(
            process=types.SimpleNamespace(poll=lambda: None, returncode=None),
            log="server.log",
        )
        struct.pack_into("<q", self.fake.guest, signal_value, 0)

        self.device.wait(slot, 1.0, server)

    def test_wait_reports_a_server_that_exited_before_completing(self) -> None:
        slot = self.device.enqueue(self.code.kernels["noop"], BENCH.WORKGROUP, b"")
        server = types.SimpleNamespace(
            process=types.SimpleNamespace(poll=lambda: 3, returncode=3),
            log="server.log",
        )

        with self.assertRaisesRegex(
            BENCH.HarnessError, "server exited with 3; see server.log"
        ):
            self.device.wait(slot, 1.0, server)


class BenchTest(DeviceTestCase):
    def setUp(self) -> None:
        self.device, self.fake = self.make_device()
        self.code = loaded_code_object()
        self.device.load(self.code)
        self.server = types.SimpleNamespace(
            process=types.SimpleNamespace(pid=4242), io_syscalls=mock.Mock()
        )
        self.bench = BENCH.Bench(self.device, self.server, self.code, 1.0)
        self.root = self.fake.queue_registration().page_table_root
        self.known: dict[tuple[str, int, int], list[int]] = {}

    def results(self, name: str, grid: int, iterations: int) -> list[int]:
        """What a correct device writes, from the stepwise model."""
        key = (name, grid, iterations)
        if key not in self.known:
            self.known[key] = [
                BENCH.expected(name, gid, iterations, self.bench.buf)[0]
                for gid in range(grid)
            ]
        return list(self.known[key])

    def measure(
        self,
        seconds: list[float],
        spoil: Callable[[int, list[int]], list[int] | None] | None = None,
        name: str = "int_loop",
        results: Callable[[str, int, int], list[int]] | None = None,
    ) -> dict[str, object]:
        """Run throughput() with stubbed dispatch times, CPU accounting and clock.

        Each stubbed dispatch writes a correct device's output. `spoil` gets
        the index of the dispatch and that output, and returns what the device
        writes instead, or None for a dispatch that writes nothing.
        """
        cpu_seconds = [
            {"11 rocjitsu": (1.0, 0.5)},
            {"11 rocjitsu": (2.0, 1.0), "12 dispatch": (0.2, 0.0)},
        ]
        scheduling = [(100, 50, 10, 2), (300, 150, 30, 6)]
        self.server.io_syscalls.side_effect = [100, 100 + 15360]
        self.seen_before: list[bytes] = []
        times = iter(seconds)

        def run(kernel: str, grid: int, iterations: int) -> float:
            start = self.bench.out_off
            self.seen_before.append(bytes(self.device.mem[start : start + grid * 4]))
            values = (results or self.results)(kernel, grid, iterations)
            if spoil is not None:
                values = spoil(len(self.seen_before) - 1, values)
            if values is not None:
                raw = array.array("I", values).tobytes()
                self.device.mem[start : start + len(raw)] = raw
            return next(times)

        with contextlib.ExitStack() as stack:
            self.run_calls = stack.enter_context(
                mock.patch.object(self.bench, "run", side_effect=run)
            )
            self.cpu_calls = stack.enter_context(
                mock.patch.object(BENCH, "process_cpu_seconds", side_effect=cpu_seconds)
            )
            stack.enter_context(
                mock.patch.object(BENCH, "process_sched", side_effect=scheduling)
            )
            stack.enter_context(
                mock.patch.object(BENCH.time, "perf_counter", side_effect=[10.0, 12.0])
            )
            return self.bench.throughput(name, 128, 1.0, 3)

    def test_throughput_calibrates_then_reports_rates_from_the_best_run(self) -> None:
        # 0.1 s at 64 iterations is under half the target, so the run is scaled 10x.
        result = self.measure([0.1, 0.6, 0.5, 0.25, 0.5])

        self.assertEqual(
            self.run_calls.call_args_list,
            [mock.call("int_loop", 128, 64)] + [mock.call("int_loop", 128, 640)] * 4,
        )
        self.assertEqual(result["iters"], 640)
        self.assertEqual(result["best_s"], 0.25)
        self.assertEqual(result["median_s"], 0.5)
        # 128 lanes are 4 waves of 32; the loop is 5 instructions long.
        self.assertEqual(result["lane_iters_per_s"], 128 * 640 / 0.25)
        self.assertEqual(result["wave_insts_per_s"], 4 * 640 * 5 / 0.25)
        self.assertEqual((result["wave_lanes"], result["loop_insts"]), (32, 5))
        # 15360 calls over 4 waves * 640 iterations * 3 timed runs.
        self.assertEqual(result["io_syscalls_per_wave_iter"], 2.0)
        self.assertEqual(result["wrong_items"], [0, 0, 0])
        self.assertEqual(result["output"], "ok")

    def test_throughput_accounts_cpu_per_thread_over_the_timed_runs(self) -> None:
        result = self.measure([0.6, 0.5, 0.25, 0.5])

        # 1.5 s on the main thread and 0.2 s on a new one, over 2 s of wall time.
        self.assertAlmostEqual(result["server_cores"], 0.85)
        self.assertAlmostEqual(result["top_thread_share"], 0.75)
        self.assertEqual(result["busy_threads"], 1)
        self.assertAlmostEqual(result["server_sys_share"], 0.5 / 1.7)
        self.assertAlmostEqual(result["runq_wait_share"], 100 / 300)
        self.assertEqual(result["vol_switches_per_s"], 10.0)
        self.assertEqual(result["invol_switches_per_s"], 2.0)

    def test_throughput_counts_wrong_work_items_of_every_timed_run(self) -> None:
        def spoil(dispatch: int, values: list[int]) -> list[int]:
            # Dispatch 0 calibrates. The first timed run gets one item wrong
            # and the second three; none is the last item or in the last run.
            for gid in {1: [0], 2: [5, 64, 126]}.get(dispatch, []):
                values[gid] ^= 1
            return values

        result = self.measure([0.6, 0.5, 0.25, 0.5], spoil)

        self.assertEqual(result["iters"], 64)
        self.assertEqual(result["wrong_items"], [1, 3, 0])
        self.assertEqual(result["output"], "3/128 wrong")

    def test_throughput_does_not_check_a_calibration_run(self) -> None:
        result = self.measure(
            [0.1, 0.6, 0.5, 0.25, 0.5],
            lambda dispatch, values: [0] * 128 if dispatch < 2 else values,
        )

        self.assertEqual(result["output"], "ok")

    def test_timed_run_that_writes_nothing_is_wrong_in_every_work_item(self) -> None:
        # The calibration run and the first timed run leave correct results
        # for the same iteration count behind them.
        result = self.measure(
            [0.6, 0.5, 0.25, 0.5],
            lambda dispatch, values: None if dispatch >= 2 else values,
        )

        self.assertEqual(result["wrong_items"], [0, 128, 128])
        self.assertEqual(result["output"], "128/128 wrong")

    def test_output_is_cleared_before_each_timed_run_only(self) -> None:
        self.measure([0.1, 0.6, 0.5, 0.25, 0.5])

        cleared = b"\xff" * (128 * 4)
        calibration = array.array("I", self.results("int_loop", 128, 64)).tobytes()
        self.assertEqual(self.seen_before[1], calibration)
        self.assertEqual(self.seen_before[2:], [cleared] * 3)
        # A cleared fp_loop item is a NaN, which no tolerance accepts.
        self.assertTrue(math.isnan(struct.unpack("<f", cleared[:4])[0]))

    def test_output_is_compared_after_the_accounting_is_closed(self) -> None:
        model = BENCH.expected_grid
        closed = []

        def expected_grid(*arguments: object) -> list[list[float]]:
            closed.append(
                (self.cpu_calls.call_count, self.server.io_syscalls.call_count)
            )
            return model(*arguments)

        with mock.patch.object(BENCH, "expected_grid", expected_grid):
            self.measure([0.6, 0.5, 0.25, 0.5])

        # Once for all three runs, after the second reading of each counter.
        self.assertEqual(closed, [(2, 2)])

    def test_run_calibrated_to_the_iteration_limit_is_still_checked(self) -> None:
        limit = 1 << 28

        def device(name: str, grid: int, iterations: int) -> list[int]:
            return BENCH.expected_grid(name, grid, iterations, self.bench.buf)[0]

        def spoil(dispatch: int, values: list[int]) -> list[int]:
            if dispatch == self.dispatches - 1:
                values[77] += 1
            return values

        # A server this fast never reaches half the target.
        seconds = [1e-9] * 40
        for name in ("int_loop", "mem_loop_vram"):
            with self.subTest(kernel=name):
                result = self.measure(seconds, name=name, results=device)
                self.dispatches = self.run_calls.call_count
                self.assertEqual(result["iters"], limit)
                self.assertEqual(result["output"], "ok")

                result = self.measure(seconds, spoil, name=name, results=device)
                self.assertEqual(result["wrong_items"], [0, 0, 1])

    def test_buffer_is_reachable_in_guest_ram_and_in_vram_with_the_same_words(
        self,
    ) -> None:
        words = struct.pack("<4I", *self.bench.buf[:4])
        ram_arguments = struct.unpack("<QIIQI", self.bench.args(7, "mem_loop"))
        vram_arguments = struct.unpack("<QIIQI", self.bench.args(7, "mem_loop_vram"))

        out_pointer, iterations, _, ram_pointer, mask = ram_arguments
        vram_pointer = vram_arguments[3]
        self.assertEqual(
            self.fake.process_to_guest_offset(self.root, out_pointer),
            self.bench.out_off,
        )
        self.assertEqual((iterations, mask), (7, BENCH.MEM_WORDS - 1))
        self.assertEqual(ram_arguments[:3], vram_arguments[:3])
        ram_offset = self.fake.process_to_guest_offset(self.root, ram_pointer)
        self.assertEqual(self.fake.guest[ram_offset : ram_offset + 16], words)
        vram_entry = self.fake.walk(self.root, vram_pointer)
        self.assertIsNotNone(vram_entry)
        self.assertFalse(vram_entry & PTE_SYSTEM)
        vram_offset = vram_entry & PTE_ADDRESS_MASK
        self.assertEqual(self.fake.vram[vram_offset : vram_offset + 16], words)
        # Both placements run the one mem_loop kernel.
        self.assertIs(self.bench.kernel("mem_loop_vram"), self.bench.kernel("mem_loop"))


class ExecutingDeviceTestCase(unittest.TestCase):
    """A Bench on a device that runs what is dispatched to it."""

    GRID = 128
    ITERATIONS = 5

    def setUp(self) -> None:
        self.code, entries = full_code_object()
        self.fake = ExecutingDevice(entries)
        self.addCleanup(self.fake.close)
        self.device = BENCH.Device(self.fake, PROCESS_BYTES)
        self.addCleanup(self.device.close)
        self.device.load(self.code)
        self.process = types.SimpleNamespace(pid=4242, returncode=None)
        self.process.poll = lambda: self.process.returncode
        self.server = types.SimpleNamespace(process=self.process, log="server.log")
        self.bench = BENCH.Bench(self.device, self.server, self.code, 1.0)

    def verify(self) -> dict[str, str]:
        return self.bench.verify(self.GRID, self.ITERATIONS)


class VerifyTest(ExecutingDeviceTestCase):
    def test_every_kernel_of_a_correct_device_is_ok(self) -> None:
        verdicts = self.verify()

        self.assertEqual(verdicts, {name: "ok" for name in BENCH.BENCHMARKS})
        self.assertEqual(list(verdicts), list(BENCH.BENCHMARKS))
        # One dispatch per benchmark, the buffer kernel once from each placement.
        self.assertEqual(
            self.fake.dispatches,
            [Dispatch(name, self.GRID, self.ITERATIONS) for name in BENCH.BENCHMARKS],
        )

    def test_default_run_checks_4096_work_items_over_37_iterations(self) -> None:
        self.assertEqual(self.bench.verify()["int_loop"], "ok")

        self.assertEqual(self.fake.dispatches[0], Dispatch("int_loop", 4096, 37))

    def test_wrong_work_items_are_counted_per_kernel(self) -> None:
        def spoil(benchmark: str, gid: int, value: float) -> float | None:
            if benchmark == "int_loop" and gid in (0, 64, 127):
                return value ^ 1
            if benchmark == "lds_loop" and gid >= 64:
                return value + 1
            return value

        self.fake.spoil = spoil

        verdicts = self.verify()

        wrong = {"int_loop": "3/128 wrong", "lds_loop": "64/128 wrong"}
        self.assertEqual(
            verdicts, {name: wrong.get(name, "ok") for name in BENCH.BENCHMARKS}
        )

    def test_results_a_dispatch_did_not_write_are_wrong(self) -> None:
        # mem_loop_vram follows mem_loop into the same output buffer with the
        # same expected values, so stale results there would look correct.
        self.fake.spoil = lambda benchmark, gid, value: (
            None if benchmark == "mem_loop_vram" else value
        )

        verdicts = self.verify()

        self.assertEqual(verdicts["mem_loop"], "ok")
        self.assertEqual(verdicts["mem_loop_vram"], "128/128 wrong")

    def test_results_left_by_an_earlier_check_are_not_accepted(self) -> None:
        self.assertEqual(self.verify()["int_loop"], "ok")
        self.fake.spoil = lambda benchmark, gid, value: None

        verdicts = self.verify()

        self.assertEqual(verdicts, {name: "128/128 wrong" for name in BENCH.BENCHMARKS})

    def test_fp_loop_accepts_a_device_that_fuses_and_one_that_does_not(self) -> None:
        # After 200 iterations the two roundings are further apart than the
        # tolerance, so each device is only right against its own candidate.
        separate, fused = BENCH.expected("fp_loop", 3, 200, [])
        self.assertFalse(math.isclose(separate, fused, rel_tol=1e-6))
        for fuses in (True, False):
            with self.subTest(fuses=fuses):
                self.fake.fused_multiply_add = fuses

                self.assertEqual(self.bench.verify(64, 200)["fp_loop"], "ok")

    def test_fp_loop_allows_the_last_bit_but_not_a_visible_error(self) -> None:
        def spoil(benchmark: str, gid: int, value: float) -> float | None:
            if benchmark != "fp_loop":
                return value
            if gid == 3:
                return binary32_from_bits(binary32_bits(value) + 1)
            if gid == 9:
                return value * (1 + 1e-4)
            return value

        self.fake.spoil = spoil

        self.assertEqual(self.verify()["fp_loop"], "1/128 wrong")

    def test_a_server_that_exits_mid_check_ends_it(self) -> None:
        self.fake.completes_dispatches = False
        self.process.returncode = 3

        with self.assertRaisesRegex(BENCH.HarnessError, "server exited with 3"):
            self.verify()


class TimedOutputTest(ExecutingDeviceTestCase):
    """throughput() on a device that runs the stepwise model for each dispatch."""

    def setUp(self) -> None:
        super().setUp()
        self.server.io_syscalls = lambda: None
        for name, value in (
            ("process_cpu_seconds", {"11 rocjitsu": (1.0, 0.5)}),
            ("process_sched", (100, 50, 10, 2)),
        ):
            patcher = mock.patch.object(BENCH, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def timed(self, name: str, repeats: int = 2) -> dict[str, object]:
        # Any dispatch is longer than half of this target, so 64 iterations stay.
        return self.bench.throughput(name, self.GRID, 1e-9, repeats)

    def test_every_kernel_of_a_correct_device_is_ok(self) -> None:
        for name in BENCH.BENCHMARKS:
            with self.subTest(kernel=name):
                result = self.timed(name)

                self.assertEqual(result["iters"], 64)
                self.assertEqual(result["wrong_items"], [0, 0])
                self.assertEqual(result["output"], "ok")
        # One calibration and two timed dispatches per benchmark.
        self.assertEqual(
            self.fake.dispatches,
            [Dispatch(name, self.GRID, 64) for name in BENCH.BENCHMARKS for _ in "123"],
        )

    def test_fp_loop_is_ok_whether_or_not_the_device_fuses(self) -> None:
        for fuses in (True, False):
            with self.subTest(fuses=fuses):
                self.fake.fused_multiply_add = fuses

                self.assertEqual(self.timed("fp_loop")["output"], "ok")

    def test_wrong_items_anywhere_in_any_timed_run_are_counted(self) -> None:
        def spoil(benchmark: str, gid: int, value: float) -> float | None:
            timed_run = len(self.fake.dispatches) - 1
            if benchmark == "fp_loop" and timed_run == 1 and gid == 9:
                return value * (1 + 1e-4)
            if benchmark == "i64_loop" and timed_run == 2 and gid in (0, 100):
                return value ^ 0x8000_0000
            return value

        self.fake.spoil = spoil

        for name, wrong in (("fp_loop", [1, 0]), ("i64_loop", [0, 2])):
            with self.subTest(kernel=name):
                self.fake.dispatches.clear()
                result = self.timed(name)

                self.assertEqual(result["wrong_items"], wrong)
                self.assertEqual(result["output"], f"{max(wrong)}/128 wrong")

    def test_timed_run_that_completes_without_writing_is_wrong(self) -> None:
        self.fake.spoil = lambda benchmark, gid, value: (
            None if len(self.fake.dispatches) == 3 else value
        )

        for name in ("fp_loop", "lds_loop"):
            with self.subTest(kernel=name):
                self.fake.dispatches.clear()

                self.assertEqual(self.timed(name)["wrong_items"], [0, 128])


FAKE_SERVER = """#!/usr/bin/env python3
import json
import os
import sys

arguments = sys.argv[1:]
config = arguments[arguments.index("--config") + 1]
with open(config) as file:
    document = json.load(file)
with open(os.environ["RJ_BENCH_TEST_RECORD"], "w") as file:
    json.dump({"argv": arguments, "config": document}, file)
print("fake server started", flush=True)
if os.environ.get("RJ_BENCH_TEST_READY") == "1":
    os.write(int(arguments[arguments.index("--vfio-ready-fd") + 1]), b"\\x01")
if os.environ.get("RJ_BENCH_TEST_READY") == "never":
    import time

    time.sleep(60)
if os.environ.get("RJ_BENCH_TEST_LINGER") == "1":
    import time

    time.sleep(60)
"""

# Starts a server as the harness does, reports its process id, and waits to be killed.
SERVER_OWNER = """
import importlib.util
import sys
import time
from pathlib import Path

script, binary, config, workdir = sys.argv[1:]
spec = importlib.util.spec_from_file_location("bench_vfio_user", script)
bench = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bench
spec.loader.exec_module(bench)
variant = bench.parse_variant(
    "owned:env=RJ_BENCH_TEST_READY=1,env=RJ_BENCH_TEST_LINGER=1,"
    f"env=RJ_BENCH_TEST_RECORD={workdir}/record.json",
    binary,
    config,
    "",
)
server = bench.Server(variant, Path(workdir))
print(server.process.pid, flush=True)
time.sleep(60)
"""


class ServerTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.workdir = Path(directory.name)
        self.binary = self.workdir / "fake-rocjitsu"
        self.binary.write_text(FAKE_SERVER)
        self.binary.chmod(0o755)
        self.config = self.workdir / "base.json"
        self.config.write_text(json.dumps({"vm": {"gpu": {"count": 1, "name": "x"}}}))
        self.record = self.workdir / "record.json"

    def variant(self, text: str, ready: str = "1") -> BENCH.Variant:
        variant = BENCH.parse_variant(text, str(self.binary), str(self.config), "")
        variant.env["RJ_BENCH_TEST_RECORD"] = str(self.record)
        variant.env["RJ_BENCH_TEST_READY"] = ready
        return variant

    def start(self, text: str) -> tuple[BENCH.Server, dict]:
        server = BENCH.Server(self.variant(text), self.workdir)
        self.addCleanup(server.stop)
        return server, json.loads(self.record.read_text())

    def test_starts_the_binary_on_its_own_socket_and_waits_for_readiness(
        self,
    ) -> None:
        server, record = self.start("plain:arg=--extra")

        self.assertEqual(server.socket_path, str(self.workdir / "plain.sock"))
        self.assertEqual(
            record["argv"][:4],
            ["--config", str(self.config), "--vfio-socket", server.socket_path],
        )
        self.assertEqual(record["argv"][4], "--vfio-ready-fd")
        self.assertEqual(record["argv"][6:], ["--extra"])
        server.process.wait(timeout=10)
        self.assertIn("fake server started", server.log.read_text())
        self.assertIsNone(server.io_syscalls())

    def test_cfg_overrides_are_applied_to_a_copy_of_the_config(self) -> None:
        original = self.config.read_text()

        server, record = self.start(
            "sized:cfg=vm.gpu.count=4,cfg=execution_threads.dispatch=[8]"
        )

        # An existing leaf is replaced, its siblings kept, a missing branch created.
        self.assertEqual(
            record["config"],
            {
                "vm": {"gpu": {"count": 4, "name": "x"}},
                "execution_threads": {"dispatch": [8]},
            },
        )
        self.assertNotEqual(record["argv"][1], str(self.config))
        self.assertEqual(Path(record["argv"][1]).parent, self.workdir)
        self.assertEqual(self.config.read_text(), original)
        self.assertEqual(server.log, self.workdir / "sized.log")

    def test_server_that_exits_without_signalling_readiness_is_an_error(self) -> None:
        with self.assertRaisesRegex(BENCH.HarnessError, "did not become ready"):
            BENCH.Server(self.variant("silent", ready="0"), self.workdir)

        self.assertIn("fake server started", (self.workdir / "silent.log").read_text())

    def test_server_that_never_signals_readiness_is_stopped_after_the_timeout(
        self,
    ) -> None:
        started = []
        popen = subprocess.Popen

        def recording_popen(*arguments: object, **options: object) -> subprocess.Popen:
            started.append(popen(*arguments, **options))
            return started[-1]

        with mock.patch.object(BENCH.subprocess, "Popen", side_effect=recording_popen):
            with self.assertRaisesRegex(
                BENCH.HarnessError, "not ready within 0.5s; see .*hung.log"
            ):
                BENCH.Server(
                    self.variant("hung", ready="never"), self.workdir, ready_timeout=0.5
                )

        (process,) = started
        # The server was terminated and reaped, not left behind.
        self.assertEqual(process.returncode, -signal.SIGTERM)

    def test_server_does_not_outlive_a_harness_that_is_killed(self) -> None:
        owner = subprocess.Popen(
            [
                sys.executable,
                "-B",
                "-c",
                SERVER_OWNER,
                str(SCRIPT),
                str(self.binary),
                str(self.config),
                str(self.workdir),
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        self.addCleanup(owner.stdout.close)
        server = int(owner.stdout.readline())
        self.assertTrue(is_running(server))

        owner.kill()
        owner.wait()

        deadline = time.monotonic() + 10
        while is_running(server) and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(is_running(server))

    def test_binary_that_cannot_be_started_leaves_nothing_open(self) -> None:
        variant = self.variant("absent:bin=" + str(self.workdir / "no-such-binary"))
        open_before = open_descriptors()

        with self.assertRaises(FileNotFoundError):
            BENCH.Server(variant, self.workdir)

        self.assertEqual(open_descriptors(), open_before)

    def test_config_that_is_not_json_is_reported_by_name(self) -> None:
        self.config.write_text("{ not json")

        with self.assertRaisesRegex(BENCH.HarnessError, "base.json is not valid JSON"):
            BENCH.Server(self.variant("broken:cfg=vm.gpu.count=4"), self.workdir)

        self.assertFalse(self.record.exists())

    def test_cfg_path_through_a_value_that_is_not_an_object_is_reported(self) -> None:
        with self.assertRaisesRegex(
            BENCH.HarnessError, "cfg=vm.gpu.count.limit .* 'count' .* not an object"
        ):
            BENCH.Server(self.variant("deep:cfg=vm.gpu.count.limit=4"), self.workdir)

        self.assertFalse(self.record.exists())

    def test_config_that_is_not_an_object_is_reported(self) -> None:
        self.config.write_text("[1, 2]")

        with self.assertRaisesRegex(BENCH.HarnessError, "does not hold a JSON object"):
            BENCH.Server(self.variant("list:cfg=vm.gpu.count=4"), self.workdir)


class LogTailTest(unittest.TestCase):
    def test_returns_the_last_lines_of_a_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            log.write_bytes(b"".join(b"line %d\n" % index for index in range(20)))

            self.assertEqual(
                BENCH.log_tail(log), [f"line {index}" for index in range(12, 20)]
            )
            self.assertEqual(BENCH.log_tail(log, 2), ["line 18", "line 19"])

    def test_bytes_that_are_not_text_do_not_stop_the_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            log.write_bytes(b"ready\n\xff\xfe crash\n")

            self.assertEqual(BENCH.log_tail(log), ["ready", "\ufffd\ufffd crash"])

    def test_a_log_that_was_never_written_has_no_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(BENCH.log_tail(Path(directory) / "absent.log"), [])


def markdown_tables(report: str) -> dict[str, list[list[str]]]:
    """Rows of each '## Title' table in a report, header first, separator dropped."""
    tables: dict[str, list[list[str]]] = {}
    title = ""
    for line in report.splitlines():
        if line.startswith("## "):
            title = line[3:]
        elif line.startswith("|") and not line.startswith("|---"):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            tables.setdefault(title, []).append(cells)
    return tables


def variant_result(name: str, lane_iters_per_s: float, **extra: object) -> dict:
    latency = {"median_us": 12.5, "p99_us": 40.0, "min_us": 9.0}
    timing = {
        "kernel": "int_loop",
        "grid": 64,
        "iters": 640,
        "best_s": 0.5,
        "lane_iters_per_s": lane_iters_per_s,
        "loop_insts": 5,
        "wave_insts_per_s": lane_iters_per_s * 5 / 32,
        "server_cores": 1.5,
        "server_sys_share": 0.25,
        "runq_wait_share": 0.1,
        "vol_switches_per_s": 1234.0,
        "io_syscalls_per_wave_iter": None,
        "wrong_items": [0, 0, 0],
        "output": "ok",
    }
    result = {
        "variant": name,
        "target": "gfx1250",
        "wave_lanes": 32,
        "verify": {kernel: "ok" for kernel in BENCH.BENCHMARKS},
        "mmio_read": latency,
        "mmio_read_spin": latency,
        "noop_dispatch": latency,
        "noop_dispatch_irq": latency,
        "noop_dispatch_posted": latency,
        "batched_dispatch_per_s": {"1": 1000.0, "16": 20000.0},
        "throughput": [timing],
    }
    result.update(extra)
    return result


class PrintReportTest(unittest.TestCase):
    def report(self, results: list[dict]) -> dict[str, list[list[str]]]:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            BENCH.print_report(results)
        tables = markdown_tables(output.getvalue())
        for title, rows in tables.items():
            with self.subTest(table=title):
                self.assertEqual({len(row) for row in rows}, {len(rows[0])})
        return tables

    def test_throughput_is_compared_against_the_first_variant(self) -> None:
        tables = self.report(
            [variant_result("base", 1e6), variant_result("fast", 2.5e6)]
        )

        header, base, fast = tables["Throughput (best of repeats)"]
        ratio = header.index("vs first")
        self.assertEqual((base[0], base[ratio]), ("base", "1.00x"))
        self.assertEqual((fast[0], fast[ratio]), ("fast", "2.50x"))
        self.assertEqual(fast[header.index("M lane-iters/s")], "2.50")
        self.assertEqual(fast[header.index("kernel")], "int_loop")
        self.assertEqual(fast[header.index("sys time")], "25%")
        self.assertEqual(fast[header.index("result")], "ok")

    def test_transport_table_reports_latencies_and_batched_rates(self) -> None:
        tables = self.report([variant_result("base", 1e6)])

        header, row = tables["Transport"]
        self.assertEqual(header[-2:], ["batch 1 disp/s", "batch 16 disp/s"])
        self.assertEqual(row[:3], ["base", "ok", "12.5"])
        self.assertEqual(row[header.index("p99 us")], "40.0")
        self.assertEqual(row[-2:], ["1,000", "20,000"])

    def test_result_column_is_the_verdict_on_the_timed_output(self) -> None:
        uncounted = variant_result("base", 1e6)
        wrong = variant_result("wrong", 1e6)
        wrong["throughput"][0]["output"] = "3/64 wrong"
        wrong["throughput"][0]["io_syscalls_per_wave_iter"] = 2.0

        tables = self.report([uncounted, wrong])

        header, first, second = tables["Throughput (best of repeats)"]
        syscalls = header.index("pread+pwrite per wave-iter")
        self.assertEqual((first[syscalls], first[-1]), ("-", "ok"))
        self.assertEqual((second[syscalls], second[-1]), ("2.00", "3/64 wrong"))

    def test_failed_variant_is_listed_with_its_error_and_left_out_of_timings(
        self,
    ) -> None:
        failed = {
            "variant": "broken",
            "target": "gfx1151",
            "wave_lanes": 64,
            "error": "server did not become ready",
        }

        tables = self.report([variant_result("base", 1e6), failed])

        self.assertEqual(
            tables["Verify"][2],
            ["broken", "gfx1151", "64"]
            + ["-"] * len(BENCH.BENCHMARKS)
            + ["server did not become ready"],
        )
        self.assertEqual(len(tables["Transport"]), 2)
        self.assertEqual(len(tables["Throughput (best of repeats)"]), 2)

    def test_verify_only_run_prints_just_the_verify_table(self) -> None:
        verified = {
            "variant": "base",
            "target": "gfx1250",
            "wave_lanes": 32,
            "verify": {"int_loop": "ok", "fp_loop": "3/4096 wrong"},
        }

        tables = self.report([verified])

        self.assertEqual(list(tables), ["Verify"])
        header, row = tables["Verify"]
        self.assertEqual(row[header.index("int_loop")], "ok")
        self.assertEqual(row[header.index("fp_loop")], "3/4096 wrong")
        self.assertEqual(row[header.index("lds_loop")], "-")


class StubServer:
    """Stands in for Server: the same surface and a log on disk, but no process."""

    def __init__(
        self,
        variant: BENCH.Variant,
        workdir: Path,
        shim: str = "",
        ready_timeout: float = 120.0,
    ):
        self.variant, self.shim, self.ready_timeout = variant, shim, ready_timeout
        self.socket_path = str(workdir / f"{variant.name}.sock")
        self.log = workdir / f"{variant.name}.log"
        self.process = types.SimpleNamespace(pid=4242, returncode=None)
        self.process.poll = lambda: self.process.returncode
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True
        if self.process.returncode is None:
            self.process.returncode = -signal.SIGTERM

    def cpu_seconds(self) -> dict[str, tuple[float, float]]:
        return {"4242 rocjitsu": (0.5, 0.1), "4243 rj-dispatch-0": (0.2, 0.0)}

    def io_syscalls(self) -> int | None:
        return None


class RunVariantTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.workdir = Path(directory.name)
        self.code, entries = full_code_object()
        self.fake = ExecutingDevice(entries)
        self.addCleanup(self.fake.close)
        self.servers: list[StubServer] = []
        self.connections: list[str] = []
        self.variant = BENCH.Variant("candidate", "/opt/rj/rocjitsu", "base.json")
        self.variant.cfg["cpu_dispatch_threads"] = 4
        self.options = argparse.Namespace(
            memory_mib=3,
            timeout=1.0,
            ready_timeout=7.0,
            verify_only=True,
            latency_samples=4,
            batches=[1, 2],
            kernels=[],
            grids=[64],
            target_seconds=1.5,
            repeats=2,
        )
        verify = BENCH.Bench.verify
        # The default check models 4096 work items per kernel; 128 take the same
        # path.
        self.patch(BENCH.Bench, "verify", lambda bench: verify(bench, 128, 5))
        self.patch(BENCH, "Server", self.start_server)
        self.patch(BENCH, "VfioUserClient", self.connect)

    def patch(self, target: object, name: str, replacement: object) -> None:
        patcher = mock.patch.object(target, name, replacement)
        patcher.start()
        self.addCleanup(patcher.stop)

    def start_server(self, *arguments: object) -> StubServer:
        self.servers.append(StubServer(*arguments))
        return self.servers[-1]

    def connect(self, path: str) -> ExecutingDevice:
        self.connections.append(path)
        return self.fake

    def run_variant(self) -> dict:
        return BENCH.run_variant(
            self.variant, self.code, self.options, self.workdir, "shim.so"
        )

    def test_verify_only_run_reports_the_variant_the_bars_and_the_verdicts(
        self,
    ) -> None:
        result = self.run_variant()

        (server,) = self.servers
        self.assertEqual(
            result,
            {
                "variant": "candidate",
                "binary": "/opt/rj/rocjitsu",
                "config": "base.json",
                "env": {},
                "args": [],
                "cpus": "",
                "cfg": {"cpu_dispatch_threads": 4},
                "target": "gfx1250",
                "wave_lanes": 32,
                "bar_bytes": ExecutingDevice.REGION_BYTES,
                "verify": {name: "ok" for name in BENCH.BENCHMARKS},
                "server_exit": -signal.SIGTERM,
            },
        )
        self.assertEqual(self.connections, [server.socket_path])
        self.assertEqual(
            (server.variant, server.shim, server.ready_timeout),
            (self.variant, "shim.so", 7.0),
        )
        # --memory-mib is the process memory, on top of the system pages.
        self.assertEqual(
            self.fake.dma_window,
            (BENCH.GUEST_BASE, BENCH.PROC_GUEST_OFFSET + (3 << 20)),
        )
        self.assertEqual(len(self.fake.dispatches), len(BENCH.BENCHMARKS))

    def test_everything_is_released_after_a_run(self) -> None:
        open_before = open_descriptors()

        self.run_variant()
        self.fake.close()

        self.assertTrue(self.servers[0].stopped)
        self.assertTrue(self.fake.closed)
        self.assertEqual(open_descriptors(), open_before)

    def test_wrong_results_are_a_verdict_not_an_error(self) -> None:
        self.fake.spoil = lambda benchmark, gid, value: (
            value + 1 if benchmark == "i64_loop" and gid < 2 else value
        )

        result = self.run_variant()

        self.assertEqual(result["verify"]["i64_loop"], "2/128 wrong")
        self.assertEqual(result["verify"]["int_loop"], "ok")
        self.assertNotIn("error", result)

    def test_full_run_adds_latencies_batched_rates_and_throughput(self) -> None:
        self.options.verify_only = False
        self.options.kernels = ["int_loop", "fp_loop"]
        self.options.grids = [64, 128]
        timed: list[tuple] = []

        def throughput(bench: BENCH.Bench, *arguments: object) -> dict:
            timed.append(arguments)
            return {"kernel": arguments[0], "grid": arguments[1]}

        self.patch(BENCH.Bench, "throughput", throughput)

        result = self.run_variant()

        for name in ("mmio_read", "mmio_read_spin", "noop_dispatch"):
            with self.subTest(measurement=name):
                self.assertEqual(set(result[name]), {"median_us", "p99_us", "min_us"})
                self.assertGreater(result[name]["min_us"], 0.0)
        self.assertEqual(set(result["noop_dispatch_irq"]), set(result["mmio_read"]))
        self.assertEqual(set(result["noop_dispatch_posted"]), set(result["mmio_read"]))
        # --latency-samples register reads for the sleeping client, and again
        # for the spinning one.
        self.assertEqual(self.fake.register_reads, 8)
        self.assertFalse(self.fake.spin)
        self.assertEqual(list(result["batched_dispatch_per_s"]), ["1", "2"])
        for rate in result["batched_dispatch_per_s"].values():
            self.assertGreater(rate, 0.0)
        noops = [dispatch for dispatch in self.fake.dispatches if dispatch[0] == "noop"]
        # Three latency runs of 4, then 4 rounds of 1 and 2 rounds of 2.
        self.assertEqual(len(noops), 3 * 4 + 4 * 1 + 2 * 2)
        # Only the posted latency run sends doorbells that expect no reply.
        posted = [doorbell for doorbell in self.fake.doorbells if doorbell[2]]
        self.assertEqual(len(posted), 4)
        self.assertEqual(result["threads"], 2)
        self.assertEqual(
            timed,
            [
                ("int_loop", 64, 1.5, 2),
                ("int_loop", 128, 1.5, 2),
                ("fp_loop", 64, 1.5, 2),
                ("fp_loop", 128, 1.5, 2),
            ],
        )
        self.assertEqual(result["throughput"][3], {"kernel": "fp_loop", "grid": 128})
        self.assertNotIn("error", result)

    def test_server_that_does_not_start_is_recorded_with_the_end_of_its_log(
        self,
    ) -> None:
        log = self.workdir / "candidate.log"
        log.write_text("".join(f"startup line {index}\n" for index in range(12)))

        def refuse(*arguments: object) -> StubServer:
            raise BENCH.HarnessError(f"server did not become ready; see {log}")

        self.patch(BENCH, "Server", refuse)

        result = self.run_variant()

        self.assertEqual(result["error"], f"server did not become ready; see {log}")
        self.assertEqual(
            result["server_log_tail"],
            [f"startup line {index}" for index in range(4, 12)],
        )
        self.assertEqual(result["variant"], "candidate")
        self.assertNotIn("verify", result)
        self.assertNotIn("server_exit", result)
        self.assertEqual(self.connections, [])

    def test_binary_that_cannot_be_started_is_recorded_without_a_log(self) -> None:
        def absent(*arguments: object) -> StubServer:
            raise FileNotFoundError(2, "No such file or directory", "/opt/rj/rocjitsu")

        self.patch(BENCH, "Server", absent)

        result = self.run_variant()

        self.assertIn("No such file or directory: '/opt/rj/rocjitsu'", result["error"])
        self.assertEqual(result["server_log_tail"], [])

    def test_client_that_cannot_connect_stops_the_server(self) -> None:
        def refused(path: str) -> ExecutingDevice:
            raise ConnectionRefusedError(errno.ECONNREFUSED, "Connection refused")

        self.patch(BENCH, "VfioUserClient", refused)

        result = self.run_variant()

        self.assertIn("Connection refused", result["error"])
        self.assertTrue(self.servers[0].stopped)
        self.assertEqual(result["server_exit"], -signal.SIGTERM)
        self.assertNotIn("verify", result)

    def test_device_that_cannot_be_brought_up_is_recorded_and_released(self) -> None:
        # 1 GiB of process memory needs more page tables than one directory holds.
        self.options.memory_mib = 1024
        open_before = open_descriptors()

        result = self.run_variant()
        self.fake.close()

        self.assertIn("does not fit one page directory", result["error"])
        self.assertTrue(self.fake.closed)
        self.assertTrue(self.servers[0].stopped)
        self.assertEqual(open_descriptors(), open_before)

    def test_server_that_exits_during_the_check_is_recorded_with_its_status(
        self,
    ) -> None:
        self.fake.completes_dispatches = False

        def crashed(*arguments: object) -> StubServer:
            server = self.start_server(*arguments)
            server.process.returncode = 3
            server.log.write_text("ready\nsegmentation fault\n")
            return server

        self.patch(BENCH, "Server", crashed)

        result = self.run_variant()

        self.assertIn("server exited with 3", result["error"])
        self.assertEqual(result["server_log_tail"], ["ready", "segmentation fault"])
        self.assertEqual(result["server_exit"], 3)
        self.assertEqual(result["bar_bytes"], ExecutingDevice.REGION_BYTES)
        self.assertNotIn("verify", result)
        self.assertTrue(self.fake.closed)

    def test_a_defect_propagates_but_still_stops_the_server(self) -> None:
        def defective(bench: BENCH.Bench) -> dict[str, str]:
            raise TypeError("unsupported operand")

        self.patch(BENCH.Bench, "verify", defective)

        with self.assertRaisesRegex(TypeError, "unsupported operand"):
            self.run_variant()

        self.assertTrue(self.servers[0].stopped)
        self.assertTrue(self.fake.closed)


class MainTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.binary = make_tool(self.root / "rocjitsu")
        self.clang = make_tool(self.root / "toolchain/bin/clang")
        self.config = self.root / "base.json"
        self.config.write_text("{}")
        self.code = full_code_object()[0]
        self.builds: list[tuple[str, str]] = []
        self.runs: list[BENCH.Variant] = []
        self.workdirs: list[Path] = []
        self.results: dict[str, dict] = {}
        self.build_error: Exception | None = None
        self.patch("build_code_object", self.build_code_object)
        self.patch("run_variant", self.run_variant)
        self.patch("find_objdump", lambda clang: "/toolchain/bin/llvm-objdump")

    def patch(self, name: str, replacement: object) -> None:
        patcher = mock.patch.object(BENCH, name, replacement)
        patcher.start()
        self.addCleanup(patcher.stop)

    def build_code_object(
        self, clang: str, target: str, workdir: Path
    ) -> BENCH.CodeObject:
        self.workdirs.append(workdir)
        self.builds.append((clang, target))
        if self.build_error is not None:
            raise self.build_error
        return BENCH.CodeObject(self.code.image, self.code.kernels, target)

    def run_variant(
        self,
        variant: BENCH.Variant,
        code: BENCH.CodeObject,
        options: argparse.Namespace,
        workdir: Path,
        shim: str,
    ) -> dict:
        self.runs.append(variant)
        self.assertEqual(code.target, variant.target)
        self.assertTrue(workdir.is_dir())
        result = {
            "variant": variant.name,
            "target": code.target,
            "wave_lanes": 32,
            "verify": {name: "ok" for name in BENCH.BENCHMARKS},
        }
        result.update(self.results.get(variant.name, {}))
        return result

    def main(self, *arguments: str) -> tuple[int, str, str]:
        """Run main() on a verify-only command line; returns status, stdout, stderr."""
        output, errors = io.StringIO(), io.StringIO()
        defaults = [
            "--rocjitsu",
            self.binary,
            "--config",
            str(self.config),
            "--clang",
            self.clang,
        ]
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            status = BENCH.main([*defaults, *arguments])
        for workdir in self.workdirs:
            self.addCleanup(BENCH.shutil.rmtree, workdir, ignore_errors=True)
        return status, output.getvalue(), errors.getvalue()

    def test_every_variant_ok_exits_zero_and_removes_the_work_directory(self) -> None:
        status, output, errors = self.main(
            "--variant", "base", "--variant", "wide:target=gfx1151", "--variant", "last"
        )

        self.assertEqual(status, 0)
        self.assertEqual(
            [variant.name for variant in self.runs], ["base", "wide", "last"]
        )
        # One code object per distinct target, all in the one work directory.
        self.assertEqual(
            self.builds, [(self.clang, "gfx1250"), (self.clang, "gfx1151")]
        )
        self.assertEqual(len(set(self.workdirs)), 1)
        self.assertFalse(self.workdirs[0].exists())
        rows = markdown_tables(output)["Verify"]
        self.assertEqual([row[0] for row in rows[1:]], ["base", "wide", "last"])
        self.assertNotIn("error:", errors)

    def test_a_wrong_result_fails_the_run_and_keeps_the_work_directory(self) -> None:
        self.results["wide"] = {"verify": {"int_loop": "ok", "fp_loop": "3/4096 wrong"}}

        status, output, errors = self.main("--variant", "base", "--variant", "wide")

        self.assertEqual(status, 1)
        self.assertIn("error: variants failed: wide\n", errors)
        self.assertIn(f"work directory kept: {self.workdirs[0]}\n", errors)
        self.assertTrue(self.workdirs[0].is_dir())
        # The report is still printed, so the failure can be read from it.
        self.assertIn("3/4096 wrong", output)

    def test_a_wrong_timed_output_fails_the_run_like_a_wrong_check(self) -> None:
        for name in ("base", "slow"):
            self.results[name] = variant_result(name, 1e6)
            first = self.results[name]["throughput"][0]
            self.results[name]["throughput"].append(dict(first, grid=4096))
        self.results["slow"]["throughput"][1].update(
            output="2/4096 wrong", wrong_items=[0, 2, 1]
        )
        results = self.root / "results.json"

        status, output, errors = self.main(
            "--variant", "base", "--variant", "slow", "--output", str(results)
        )

        self.assertEqual(status, 1)
        self.assertIn("error: variants failed: slow\n", errors)
        self.assertIn(
            "[slow] wrong output: int_loop timed at grid 4096, 640 iterations: "
            "2/4096 wrong (per run: [0, 2, 1])\n",
            errors,
        )
        self.assertNotIn("[base] wrong output", errors)
        self.assertIn(f"work directory kept: {self.workdirs[0]}\n", errors)
        self.assertTrue(self.workdirs[0].is_dir())
        # The pre-timing check passed, and the row still shows its rate.
        rows = markdown_tables(output)["Throughput (best of repeats)"]
        self.assertEqual(
            [row[-1] for row in rows[1:]], ["ok", "ok", "ok", "2/4096 wrong"]
        )
        written = json.loads(results.read_text())[1]["throughput"][1]
        self.assertEqual(written["wrong_items"], [0, 2, 1])

    def test_timed_runs_with_correct_output_exit_zero(self) -> None:
        self.results["default"] = variant_result("default", 1e6)

        status, output, errors = self.main()

        self.assertEqual(status, 0)
        self.assertNotIn("wrong output", errors)
        self.assertIn("## Throughput", output)

    def test_a_wrong_check_is_named_on_stderr(self) -> None:
        self.results["default"] = {
            "verify": {"int_loop": "ok", "fp_loop": "3/4096 wrong"}
        }

        status, _, errors = self.main()

        self.assertEqual(status, 1)
        self.assertIn("[default] wrong output: fp_loop: 3/4096 wrong\n", errors)

    def test_every_failed_variant_is_named_in_order(self) -> None:
        self.results["first"] = {"error": "server did not become ready"}
        self.results["third"] = {"error": "server exited with 3", "verify": {}}

        status, _, errors = self.main(
            "--variant", "first", "--variant", "second", "--variant", "third"
        )

        self.assertEqual(status, 1)
        self.assertIn("error: variants failed: first, third\n", errors)
        self.assertEqual(len(self.runs), 3)

    def test_the_end_of_a_failed_server_log_is_shown(self) -> None:
        self.results["crashed"] = {
            "error": "server exited with -11",
            "server_log_tail": ["ready", "Segmentation fault"],
        }

        status, _, errors = self.main("--variant", "fine", "--variant", "crashed")

        self.assertEqual(status, 1)
        self.assertIn("[crashed] server log: ready\n", errors)
        self.assertIn("[crashed] server log: Segmentation fault\n", errors)
        self.assertNotIn("[fine] server log", errors)

    def test_a_variant_that_failed_before_its_check_fails_the_run(self) -> None:
        def unstarted(variant: BENCH.Variant, *arguments: object) -> dict:
            return {
                "variant": variant.name,
                "target": "gfx1250",
                "wave_lanes": 32,
                "error": "server did not become ready",
            }

        self.patch("run_variant", unstarted)

        status, output, errors = self.main()

        self.assertEqual(status, 1)
        self.assertIn("error: variants failed: default\n", errors)
        self.assertIn("server did not become ready", output)

    def test_results_are_written_as_json_whether_or_not_the_run_passed(self) -> None:
        self.results["default"] = {"error": "dispatch did not complete in 600s"}
        results = self.root / "results.json"

        status, _, _ = self.main("--output", str(results))

        self.assertEqual(status, 1)
        (written,) = json.loads(results.read_text())
        self.assertEqual(written["variant"], "default")
        self.assertEqual(written["error"], "dispatch did not complete in 600s")

    def test_results_that_cannot_be_written_fail_without_a_traceback(self) -> None:
        status, _, errors = self.main(
            "--output", str(self.root / "absent/results.json")
        )

        self.assertEqual(status, 1)
        self.assertIn("error: ", errors)
        self.assertIn("absent/results.json", errors)

    def test_keep_preserves_the_work_directory_of_a_passing_run(self) -> None:
        status, _, _ = self.main("--keep")

        self.assertEqual(status, 0)
        self.assertTrue(self.workdirs[0].is_dir())

    def test_check_toolchain_builds_every_target_and_starts_no_server(self) -> None:
        status, output, errors = self.main(
            "--check-toolchain",
            "--rocjitsu",
            str(self.root / "not-built-yet"),
            "--variant",
            "base",
            "--variant",
            "other:target=gfx1151",
        )

        self.assertEqual(status, 0)
        self.assertEqual(
            output.splitlines(),
            [
                f"clang: {self.clang}",
                "llvm-objdump: /toolchain/bin/llvm-objdump",
                "gfx1250: 6 kernels, 32 lanes per wave",
                "gfx1151: 6 kernels, 32 lanes per wave",
            ],
        )
        self.assertEqual(self.runs, [])
        self.assertFalse(self.workdirs[0].exists())
        self.assertEqual(errors, "")

    def test_check_toolchain_failure_exits_nonzero_and_leaves_nothing(self) -> None:
        self.build_error = BENCH.HarnessError(
            "/toolchain/bin/clang could not compile the kernels for gfx1250 "
            "(exit status 1)"
        )

        status, output, errors = self.main("--check-toolchain")

        self.assertEqual(status, 1)
        self.assertEqual(output, "")
        self.assertEqual(
            errors,
            "error: /toolchain/bin/clang could not compile the kernels for gfx1250 "
            "(exit status 1)\n",
        )
        self.assertFalse(self.workdirs[0].exists())

    def test_clang_that_cannot_be_found_fails_before_any_work(self) -> None:
        def missing(requested: str | None) -> str:
            raise BENCH.HarnessError("no clang found under ROCM_PATH or on PATH")

        self.patch("find_clang", missing)

        status, _, errors = self.main()

        self.assertEqual(status, 1)
        self.assertEqual(errors, "error: no clang found under ROCM_PATH or on PATH\n")
        self.assertEqual((self.builds, self.runs), ([], []))

    def test_inputs_a_variant_cannot_start_with_fail_before_any_work(self) -> None:
        unreadable = self.root / "plain-file"
        unreadable.write_text("not a program")
        cases = {
            "rocjitsu binary is not executable: /no/such/rocjitsu": (
                "--variant",
                "moved:bin=/no/such/rocjitsu",
            ),
            f"rocjitsu binary is not executable: {unreadable}": (
                "--variant",
                f"plain:bin={unreadable}",
            ),
            "config not found: /no/such/config.json": (
                "--variant",
                "moved:config=/no/such/config.json",
            ),
            "variant name 'twin' is used more than once": (
                "--variant",
                "twin",
                "--variant",
                "twin:cpus=0",
            ),
            "'a/b' is not usable as a variant name": ("--variant", "a/b"),
            "unknown variant key 'binary'": ("--variant", "typo:binary=/x"),
        }
        for message, arguments in cases.items():
            with self.subTest(message=message):
                status, _, errors = self.main(*arguments)

                self.assertEqual(status, 1)
                self.assertIn("error: ", errors)
                self.assertIn(message, errors)
                self.assertEqual((self.builds, self.runs), ([], []))

    def test_pinning_needs_taskset(self) -> None:
        with mock.patch.dict(os.environ, {"PATH": str(self.root)}):
            status, _, errors = self.main("--variant", "pinned:cpus=0-3")

        self.assertEqual(status, 1)
        self.assertIn("[pinned] cpus= needs taskset on PATH", errors)

    def test_values_that_would_divide_by_zero_or_index_nothing_are_rejected(
        self,
    ) -> None:
        cases = (
            ("--grids", "0"),
            ("--grids", "96"),
            ("--grids", str(2 * BENCH.Bench.max_grid)),
            ("--batches", "0"),
            ("--batches", str(BENCH.AQL_PACKETS + 1)),
            ("--repeats", "0"),
            ("--latency-samples", "0"),
            ("--memory-mib", "0"),
            ("--target-seconds", "0"),
            ("--timeout", "-1"),
            ("--ready-timeout", "0"),
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaises(SystemExit) as raised:
                    self.main(*arguments)

                self.assertEqual(raised.exception.code, 2)
                self.assertEqual(self.runs, [])

    def test_pci_front_end_version_is_a_default_a_variant_can_override(self) -> None:
        status, _, _ = self.main(
            "--pci-gfx-target-version",
            "120500",
            "--variant",
            "plain",
            "--variant",
            "own:cfg=vm.gpu.device.gfx_target_version=110501",
        )

        self.assertEqual(status, 0)
        key = "vm.gpu.device.gfx_target_version"
        self.assertEqual([variant.cfg[key] for variant in self.runs], [120500, 110501])

    def test_default_binary_and_config_do_not_depend_on_the_working_directory(
        self,
    ) -> None:
        source_root = SCRIPT.parents[1]

        options = BENCH.parse_arguments([])

        self.assertEqual(
            options.rocjitsu, str(source_root / "build/tools/rocjitsu/rocjitsu")
        )
        self.assertEqual(
            options.config, str(source_root / "configs/gfx1250_mi455x.json")
        )
        self.assertTrue(Path(options.config).is_file())
        self.assertIsNone(options.clang)

    def test_termination_unwinds_through_the_cleanup(self) -> None:
        handler_before = signal.getsignal(signal.SIGTERM)

        def terminated(*arguments: object) -> dict:
            os.kill(os.getpid(), signal.SIGTERM)
            raise AssertionError("the signal handler did not interrupt the variant")

        self.patch("run_variant", terminated)

        with self.assertRaises(SystemExit) as raised:
            self.main()

        self.assertEqual(raised.exception.code, 128 + signal.SIGTERM)
        self.assertFalse(self.workdirs[0].exists())
        self.assertIs(signal.getsignal(signal.SIGTERM), handler_before)

    def test_a_python_older_than_the_script_supports_is_refused(self) -> None:
        with mock.patch.object(BENCH.sys, "version_info", (3, 9, 18)):
            status, _, errors = self.main()

        self.assertEqual(status, 1)
        self.assertIn("needs Python 3.10 or newer", errors)
        self.assertEqual(self.builds, [])


if __name__ == "__main__":
    unittest.main()
