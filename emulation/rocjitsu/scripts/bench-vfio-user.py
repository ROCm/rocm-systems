#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Benchmark the rocjitsu vfio-user GPU without a VM.

The script is the vfio-user client. It does what the guest amdgpu driver and
ROCr would do, and nothing more: share memory, program the VMID-0 GART and the
MES ring, register one AQL compute queue, then dispatch OpenCL kernels compiled
for the emulated target and wait on their completion signals.

Each --variant starts its own server, so builds, environments and configs are
compared under one client and one set of kernels.

The client only speaks the GC 12.1 / MES front end. --pci-gfx-target-version
120500 puts a config for another shader ISA behind it, and a variant's target=
compiles the kernels for that ISA, so one run can compare shader cores:

  --pci-gfx-target-version 120500
  --variant gfx1250:config=configs/gfx1250_mi455x.json,target=gfx1250
  --variant gfx1151:config=configs/gfx1151.json,target=gfx1151

No guest driver can use such a device: it would read GC 12.1 and load gfx1250
code. Throughput across ISAs is not like for like either (wave width, compute
units per XCD and loop bodies differ).
"""

from __future__ import annotations

import argparse
import array
import ctypes
import json
import math
import mmap
import os
import re
import select
import shutil
import signal
import socket
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

PAGE = 4096

# vfio-user wire protocol (libvfio-user include/vfio-user.h).
CMD_VERSION = 1
CMD_DMA_MAP = 2
CMD_GET_REGION_INFO = 5
CMD_SET_IRQS = 8
CMD_REGION_READ = 9
CMD_REGION_WRITE = 10
F_NO_REPLY = 0x10
F_ERROR = 0x20
DMA_READ, DMA_WRITE, DMA_MMAP = 1, 2, 4
HEADER = struct.Struct("<HHIII")

BAR_VRAM, BAR_DOORBELL, BAR_REGISTERS = 0, 2, 5
IRQ_MSIX = 2
IRQ_SET_EVENTFD_TRIGGER = (1 << 2) | (1 << 5)

# Device layout the client chooses. Guest RAM is one memfd shared at GUEST_BASE.
GUEST_BASE = 0x1000_0000
GART_START = 0x1_0000_0000
GART_PAGES = 16
SYS_MES_RING, SYS_MES_RPTR, SYS_MES_STATUS, SYS_MES_FENCE, SYS_MQD = (
    0x0000,
    0x1000,
    0x2000,
    0x3000,
    0x4000,
)
SYS_IH_RING, SYS_IH_WPTR = 0x8000, 0x9000
PROC_GUEST_OFFSET = 0x10_0000
PROC_VA_BASE = 0x20_0000
VRAM_GART_PT = 0x1_0000
VRAM_PROC_ROOT, VRAM_PDB2, VRAM_PDB1, VRAM_PDB0 = 0x3_0000, 0x3_1000, 0x3_2000, 0x3_3000
VRAM_PTB_BASE = 0x4_0000
VRAM_DATA = 0x100_0000
VRAM_WINDOW_BYTES = 512 * PAGE  # what _map_process() maps of VRAM_DATA
REGION_INFO = struct.Struct("<IIIIQQ")

# The same kernel can read its buffer from guest RAM or from VRAM; the two
# reach the device model through different paths.
BUFFER_PLACEMENT = {"mem_loop": "ram", "mem_loop_vram": "vram"}

# GFX12 PTE: valid | system | executable | readable | writeable, leaf marker.
PTE_LEAF = (1 << 63) | 0x73
PTE_LEAF_VRAM = (1 << 63) | 0x71
PDE_VALID = 1

# GC 12.1 registers, as dword offsets from the GC segment the driver uses.
GC_BASE = 0x1260
REG_GART_PT_LO, REG_GART_PT_HI = 0x169F, 0x16A0
REG_GART_START_LO, REG_GART_START_HI = 0x16BF, 0x16C0
REG_GART_END_LO, REG_GART_END_HI = 0x16DF, 0x16E0
REG_VM_INVALIDATE_ENG17 = 0x1657 + 17
REG_MES_RING_LO, REG_MES_RING_HI = 0x1FB1, 0x1FB2
REG_MES_RPTR_LO, REG_MES_RPTR_HI = 0x1FB4, 0x1FB5
REG_MES_DOORBELL, REG_MES_CONTROL, REG_MES_ACTIVE = 0x1FB8, 0x1FBA, 0x1FAB

# OSSSYS 7.1 interrupt ring registers, as byte offsets in the register BAR.
IH_CONTROL, IH_BASE, IH_BASE_HI, IH_WPTR_ADDR_HI, IH_WPTR_ADDR_LO = (
    0x4480,
    0x448C,
    0x4490,
    0x4494,
    0x4498,
)
IH_RING_ENABLE, IH_INTERRUPT_ENABLE, IH_BUS_ADDRESS = 1 << 0, 1 << 17, 2 << 28

MES_DOORBELL = 0x60
MES_FRAME_DWORDS = 64
MES_RING_DWORDS = 1024
MES_ADD_QUEUE = 0x0004_0021
MES_QUEUE_COMPUTE = 1
PASID = 3

AQL_DOORBELL = 0x900
AQL_PACKETS = 1024
AQL_QUEUE_SIZE_FIELD = 13  # ring dwords = 1 << (field + 1) = AQL_PACKETS * 16
MQD_AQL = 0x0800_0000

# amd_queue_t / amd_signal_t / kernel_descriptor_t offsets (hsa headers).
Q_BASE_ADDRESS, Q_SIZE, Q_WRITE_ID, Q_READ_ID = 8, 24, 56, 128
Q_READ_ID_OFFSET_FIELD, Q_PROPERTIES = 136, 180
SIGNAL_BYTES, SIGNAL_VALUE = 64, 8
KD_GROUP_SEGMENT, KD_PRIVATE_SEGMENT, KD_KERNARG_SIZE = 0, 4, 8
KD_CODE_PROPERTIES, KD_ENABLE_WAVEFRONT_SIZE32 = 56, 1 << 10

WORKGROUP = 64

KERNELS_CL = r"""
#define WG 64
__attribute__((reqd_work_group_size(WG, 1, 1)))
kernel void noop(global uint *out, uint iters) {}

__attribute__((reqd_work_group_size(WG, 1, 1)))
kernel void int_loop(global uint *out, uint iters) {
  uint gid = (uint)get_global_id(0);
  uint x = gid * 2654435761u + 1u;
  for (uint i = 0; i < iters; ++i) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; }
  out[gid] = x;
}

__attribute__((reqd_work_group_size(WG, 1, 1)))
kernel void i64_loop(global uint *out, uint iters) {
  uint gid = (uint)get_global_id(0);
  ulong x = (ulong)gid * 0x9E3779B97F4A7C15ul + 1ul;
  for (uint i = 0; i < iters; ++i) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; }
  out[gid] = (uint)(x ^ (x >> 32));
}

__attribute__((reqd_work_group_size(WG, 1, 1)))
kernel void fp_loop(global float *out, uint iters) {
  uint gid = (uint)get_global_id(0);
  float x = (float)(gid & 1023u) * 0.001f + 0.5f;
  for (uint i = 0; i < iters; ++i) x = x * 0.999f + 0.001f;
  out[gid] = x;
}

__attribute__((reqd_work_group_size(WG, 1, 1)))
kernel void mem_loop(global uint *out, uint iters, global const uint *buf, uint mask) {
  uint gid = (uint)get_global_id(0);
  uint acc = gid;
  for (uint i = 0; i < iters; ++i) acc += buf[(gid + i * 97u) & mask];
  out[gid] = acc;
}

__attribute__((reqd_work_group_size(WG, 1, 1)))
kernel void lds_loop(global uint *out, uint iters) {
  local uint tmp[WG];
  uint gid = (uint)get_global_id(0);
  uint lid = (uint)get_local_id(0);
  uint acc = gid;
  tmp[lid] = gid * 3u + 1u;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint i = 0; i < iters; ++i) acc += tmp[(lid + i) & (WG - 1)];
  out[gid] = acc;
}
"""

# Preloaded into the server by --count-syscalls. The VRAM store falls back to
# pread/pwrite outside its mapped aperture, and not every kernel (WSL, for one)
# provides /proc/<pid>/io to count them.
SYSCALL_SHIM_C = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>
static _Atomic uint64_t local[2];
static _Atomic uint64_t *counters = local;
static ssize_t (*real_pread)(int, void *, size_t, off_t);
static ssize_t (*real_pwrite)(int, const void *, size_t, off_t);
__attribute__((constructor)) static void init(void) {
  real_pread = dlsym(RTLD_NEXT, "pread64");
  real_pwrite = dlsym(RTLD_NEXT, "pwrite64");
  const char *path = getenv("RJ_BENCH_SYSCALL_COUNTERS");
  int fd = path ? open(path, O_RDWR) : -1;
  if (fd < 0) return;
  void *shared = mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (shared != MAP_FAILED) counters = shared;
  close(fd);
}
ssize_t pread64(int fd, void *buf, size_t n, off_t at) {
  if (!real_pread) init();
  atomic_fetch_add_explicit(&counters[0], 1, memory_order_relaxed);
  return real_pread(fd, buf, n, at);
}
ssize_t pwrite64(int fd, const void *buf, size_t n, off_t at) {
  if (!real_pwrite) init();
  atomic_fetch_add_explicit(&counters[1], 1, memory_order_relaxed);
  return real_pwrite(fd, buf, n, at);
}
ssize_t pread(int fd, void *buf, size_t n, off_t at) { return pread64(fd, buf, n, at); }
ssize_t pwrite(int fd, const void *buf, size_t n, off_t at) { return pwrite64(fd, buf, n, at); }
"""

LOOP_KERNELS = ("int_loop", "i64_loop", "fp_loop", "mem_loop", "lds_loop")
BENCHMARKS = (*LOOP_KERNELS, "mem_loop_vram")
MEM_WORDS = 1 << 16
U32 = 0xFFFF_FFFF
U64 = 0xFFFF_FFFF_FFFF_FFFF

SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROC = Path("/proc")
LIBC = ctypes.CDLL(None)
PR_SET_PDEATHSIG = 1


class HarnessError(RuntimeError):
    pass


# vfio-user client


class VfioUserClient:
    def __init__(self, path: str, timeout: float = 30.0):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.timeout = timeout
        self.sock.connect(path)
        self.msg_id = 0
        self.spin = False
        self.reply_fds: list[int] = []
        caps = b'{"capabilities":{"max_msg_fds":8,"max_data_xfer_size":1048576}}\0'
        self.request(CMD_VERSION, struct.pack("<HH", 0, 2) + caps)

    def close(self) -> None:
        self.sock.close()

    def request(
        self, cmd: int, payload: bytes = b"", fd: int = -1, reply: bool = True
    ) -> bytes:
        try:
            return self._exchange(cmd, payload, fd, reply)
        except TimeoutError:
            raise HarnessError(
                f"vfio-user command {cmd} got no reply within {self.timeout:g}s"
            ) from None

    def _exchange(self, cmd: int, payload: bytes, fd: int, reply: bool) -> bytes:
        self.msg_id = (self.msg_id + 1) & 0xFFFF
        flags = 0 if reply else F_NO_REPLY
        message = (
            HEADER.pack(self.msg_id, cmd, HEADER.size + len(payload), flags, 0)
            + payload
        )
        if fd >= 0:
            sent = socket.send_fds(self.sock, [message], [fd])
            if sent < len(message):
                self.sock.sendall(message[sent:])
        else:
            self.sock.sendall(message)
        if not reply:
            return b""
        if self.spin:
            # Burn a core instead of sleeping, to take this side's wake-up out
            # of the round trip. A zero-timeout select never reaches the socket
            # timeout, so the same limit is kept here, read between batches of
            # polls to leave the clock out of the interval being measured.
            deadline = time.monotonic() + self.timeout
            while not self._poll_reply():
                if time.monotonic() > deadline:
                    raise TimeoutError
        # Descriptors ride with the first byte of a reply.
        first, self.reply_fds, _, _ = socket.recv_fds(self.sock, HEADER.size, 8)
        if not first:
            raise HarnessError("vfio-user server closed the connection")
        header = first + self._recv(HEADER.size - len(first))
        _, _, size, flags, error = HEADER.unpack(header)
        body = self._recv(size - HEADER.size)
        if flags & F_ERROR:
            raise HarnessError(f"vfio-user command {cmd} failed: {os.strerror(error)}")
        return body

    def _poll_reply(self) -> bool:
        """Whether a reply arrived within one batch of non-blocking polls."""
        poll, sockets = select.select, [self.sock]
        for _ in range(1024):
            if poll(sockets, (), (), 0)[0]:
                return True
        return False

    def _recv(self, length: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < length:
            chunk = self.sock.recv(length - len(chunks))
            if not chunk:
                raise HarnessError("vfio-user server closed the connection")
            chunks += chunk
        return bytes(chunks)

    def _region_info(self, region: int, argsz: int) -> tuple[int, int, list[int]]:
        """(size, mmap offset, descriptors) the server reports for a region."""
        reply = self.request(
            CMD_GET_REGION_INFO, REGION_INFO.pack(argsz, 0, region, 0, 0, 0)
        )
        fds, self.reply_fds = self.reply_fds, []
        if len(reply) < REGION_INFO.size:
            for fd in fds:
                os.close(fd)
            raise HarnessError(f"vfio-user region {region} info reply is truncated")
        _, _, _, _, size, offset = REGION_INFO.unpack_from(reply)
        return size, offset, fds

    def region_size(self, region: int) -> int:
        size, _, fds = self._region_info(region, REGION_INFO.size)
        for fd in fds:
            os.close(fd)
        return size

    def region_mmap(self, region: int) -> mmap.mmap | None:
        """Map a BAR the way a VMM does, or None when the device traps it."""
        size, offset, fds = self._region_info(region, 256)
        if not fds:
            return None
        try:
            return mmap.mmap(fds[0], size, offset=offset)
        except ValueError as error:
            # mmap reports a region its descriptor does not cover this way.
            raise HarnessError(
                f"cannot map vfio-user region {region}: {error}"
            ) from error
        finally:
            for fd in fds:
                os.close(fd)

    def region_write(
        self, region: int, offset: int, data: bytes, posted: bool = False
    ) -> None:
        payload = struct.pack("<QII", offset, region, len(data)) + data
        self.request(CMD_REGION_WRITE, payload, reply=not posted)

    def region_read(self, region: int, offset: int, length: int) -> bytes:
        reply = self.request(
            CMD_REGION_READ, struct.pack("<QII", offset, region, length)
        )
        return reply[16 : 16 + length]

    def arm_irq(self, index: int, fd: int) -> None:
        self.request(
            CMD_SET_IRQS,
            struct.pack("<IIIII", 20, IRQ_SET_EVENTFD_TRIGGER, index, 0, 1),
            fd,
        )

    def dma_map(self, iova: int, size: int, fd: int) -> None:
        flags = DMA_READ | DMA_WRITE | DMA_MMAP
        self.request(CMD_DMA_MAP, struct.pack("<IIQQQ", 32, flags, 0, iova, size), fd)


# Kernels


@dataclass
class Kernel:
    name: str
    kd_offset: int
    loop_insts: int
    wave_lanes: int


@dataclass
class CodeObject:
    image: bytes
    kernels: dict[str, Kernel]
    target: str = ""


def wave_lanes(target: str, code_properties: int) -> int:
    """Lanes per wave the simulator launches, following its kernel_wavefront_size().

    It reads the descriptor's wave32 bit only on RDNA; CDNA4 and older are
    always wave64 and CDNA5 is always wave32.
    """
    if target.startswith("gfx9"):
        return 64
    if target.startswith("gfx125"):
        return 32
    return 32 if code_properties & KD_ENABLE_WAVEFRONT_SIZE32 else 64


def find_clang(requested: str | None) -> str:
    """The amdgcn-capable clang: --clang, else the ROCm LLVM, else one on PATH.

    The ROCm tree is $ROCM_PATH, or /opt/rocm when that is unset, as in
    cmake/rj_rocm_path.cmake.
    """
    if requested:
        if shutil.which(requested) is None:
            raise HarnessError(f"--clang {requested} is not an executable")
        return requested
    root = Path(os.environ.get("ROCM_PATH") or "/opt/rocm")
    for candidate in (root / "lib/llvm/bin/clang", root / "llvm/bin/clang"):
        if os.access(candidate, os.X_OK):
            return str(candidate)
    for name in ("amdclang", "clang"):
        found = shutil.which(name)
        if found:
            return found
    raise HarnessError(
        "no clang found under ROCM_PATH or on PATH; pass --clang or set ROCM_PATH"
    )


def find_objdump(clang: str) -> str:
    """llvm-objdump from the LLVM that clang belongs to, else one on PATH."""
    located = Path(shutil.which(clang) or clang)
    for directory in dict.fromkeys((located.parent, located.resolve().parent)):
        candidate = directory / "llvm-objdump"
        if os.access(candidate, os.X_OK):
            return str(candidate)
    found = shutil.which("llvm-objdump")
    if found:
        return found
    raise HarnessError(f"no llvm-objdump found beside {clang} or on PATH")


TOOL_TIMEOUT = 300.0


def run_tool(command: list[str], purpose: str, capture: bool = False) -> str:
    """Run a toolchain command and return what it printed when `capture` is set."""
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=capture,
            text=True,
            timeout=TOOL_TIMEOUT,
        )
    except OSError as error:
        raise HarnessError(
            f"cannot run {command[0]} to {purpose}: {error.strerror}"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise HarnessError(
            f"{command[0]} did not {purpose} within {TOOL_TIMEOUT:g}s"
        ) from error
    except subprocess.CalledProcessError as error:
        raise HarnessError(
            f"{command[0]} could not {purpose} (exit status {error.returncode})"
        ) from error
    return completed.stdout or ""


def build_code_object(clang: str, target: str, workdir: Path) -> CodeObject:
    source = workdir / "kernels.cl"
    binary = workdir / f"kernels-{target}.so"
    source.write_text(KERNELS_CL)
    objdump = find_objdump(clang)
    run_tool(
        [clang, "-x", "cl", "-Xclang", "-finclude-default-header", "-target",
         "amdgcn-amd-amdhsa", f"-mcpu={target}", "-O3", "-fno-unroll-loops", "-fno-vectorize", "-w",
         str(source), "-o", str(binary)],
        f"compile the kernels for {target}",
    )  # fmt: skip
    elf = binary.read_bytes()
    if len(elf) < 64 or elf[:4] != b"\x7fELF":
        raise HarnessError("the kernel compiler did not produce an ELF code object")

    phoff, shoff = struct.unpack_from("<QQ", elf, 0x20)
    phentsize, phnum, shentsize, shnum = struct.unpack_from("<HHHH", elf, 0x36)
    loads = []
    for index in range(phnum):
        p_type, _, p_offset, p_vaddr, _, p_filesz, p_memsz, _ = struct.unpack_from(
            "<IIQQQQQQ", elf, phoff + index * phentsize
        )
        if p_type == 1:
            loads.append((p_vaddr, p_offset, p_filesz, p_memsz))
    if not loads:
        raise HarnessError("the code object has no loadable segment")
    image = bytearray(max(vaddr + memsz for vaddr, _, _, memsz in loads))
    for vaddr, offset, filesz, _ in loads:
        image[vaddr : vaddr + filesz] = elf[offset : offset + filesz]

    symbols: dict[str, int] = {}
    for index in range(shnum):
        fields = struct.unpack_from("<IIQQQQIIQQ", elf, shoff + index * shentsize)
        if fields[1] != 11:  # SHT_DYNSYM
            continue
        sym_offset, sym_size, strtab_index = fields[4], fields[5], fields[6]
        strtab = struct.unpack_from(
            "<IIQQQQIIQQ", elf, shoff + strtab_index * shentsize
        )[4]
        for entry in range(sym_offset, sym_offset + sym_size, 24):
            st_name, _, _, _, st_value, _ = struct.unpack_from("<IBBHQQ", elf, entry)
            end = elf.index(b"\0", strtab + st_name)
            symbols[elf[strtab + st_name : end].decode()] = st_value

    listing = run_tool(
        [objdump, "-d", f"--mcpu={target}", str(binary)],
        f"disassemble the kernels for {target}",
        capture=True,
    )
    kernels = {}
    for name in ("noop", *LOOP_KERNELS):
        if f"{name}.kd" not in symbols:
            raise HarnessError(
                f"kernel descriptor for {name} is missing from the code object"
            )
        kd = symbols[f"{name}.kd"]
        private = struct.unpack_from("<I", image, kd + KD_PRIVATE_SEGMENT)[0]
        properties = struct.unpack_from("<H", image, kd + KD_CODE_PROPERTIES)[0]
        if private:
            # Nothing here provisions scratch, so the device would wait for a runtime to supply it.
            raise HarnessError(
                f"{name} compiled for {target} needs a {private}-byte private segment"
            )
        kernels[name] = Kernel(
            name, kd, count_loop_insts(listing, name), wave_lanes(target, properties)
        )
    return CodeObject(bytes(image), kernels, target)


def count_loop_insts(listing: str, name: str) -> int:
    """Instructions in the innermost loop: target of the last backward branch through the branch."""
    block = re.search(
        rf"^[0-9a-f]+ <{re.escape(name)}>:\n(.*?)(?=^[0-9a-f]+ <|\Z)",
        listing,
        re.S | re.M,
    )
    if not block:
        return 0
    addresses, loop = [], None
    for line in block.group(1).splitlines():
        match = re.search(r"//\s*([0-9A-Fa-f]+):", line)
        if not match:
            continue
        address = int(match.group(1), 16)
        addresses.append(address)
        target = re.search(rf"<{re.escape(name)}\+0x([0-9a-f]+)>", line)
        if "branch" in line and target:
            start = addresses[0] + int(target.group(1), 16)
            if start <= address:
                loop = (start, address)
    if loop is None:
        return 0
    return sum(1 for address in addresses if loop[0] <= address <= loop[1])


# Device bring-up and dispatch


class Device:
    def __init__(self, client: VfioUserClient, proc_bytes: int):
        self.client = client
        self.sys_bytes = PROC_GUEST_OFFSET
        self.proc_bytes = proc_bytes
        self.proc_cursor = 0
        self.posted_doorbells = False
        self.mes_wptr = 0
        self.write_id = 0
        self.irq = -1
        self.vram: mmap.mmap | None = None
        self.fd = os.memfd_create("rj-bench-guest-ram", os.MFD_CLOEXEC)
        try:
            os.ftruncate(self.fd, self.sys_bytes + proc_bytes)
            self.mem = mmap.mmap(self.fd, self.sys_bytes + proc_bytes)
        except BaseException:
            os.close(self.fd)
            raise
        self.view = memoryview(self.mem)
        try:
            client.dma_map(GUEST_BASE, self.sys_bytes + proc_bytes, self.fd)
            self.vram = client.region_mmap(BAR_VRAM)
            self._program_interrupts()
            self._program_gart_and_mes()
            self._map_process()
            self._create_queue()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Give back guest RAM, the VRAM mapping and the interrupt eventfd."""
        self.view.release()
        self.mem.close()
        os.close(self.fd)
        if self.vram is not None:
            self.vram.close()
        if self.irq >= 0:
            os.close(self.irq)

    # Raw access

    def reg_write_at(self, byte_offset: int, value: int) -> None:
        self.client.region_write(
            BAR_REGISTERS, byte_offset, struct.pack("<I", value & U32)
        )

    def reg_write(self, reg: int, value: int) -> None:
        self.reg_write_at((GC_BASE + reg) * 4, value)

    def vram_write(self, offset: int, data: bytes) -> None:
        if self.vram is not None:
            self.vram[offset : offset + len(data)] = data
            return
        # A trapped BAR only takes register-sized accesses.
        for start in range(0, len(data), 8):
            self.client.region_write(BAR_VRAM, offset + start, data[start : start + 8])

    def put(self, guest_offset: int, fmt: str, *values: int) -> None:
        struct.pack_into(fmt, self.mem, guest_offset, *values)

    def get(self, guest_offset: int, fmt: str) -> int:
        return struct.unpack_from(fmt, self.mem, guest_offset)[0]

    def alloc(self, size: int) -> tuple[int, int]:
        """Allocate process memory; returns (GPU virtual address, guest RAM offset)."""
        size = (size + PAGE - 1) & ~(PAGE - 1)
        if self.proc_cursor + size > self.proc_bytes:
            raise HarnessError("out of process memory; raise --memory-mib")
        offset = self.proc_cursor
        self.proc_cursor += size
        return PROC_VA_BASE + offset, PROC_GUEST_OFFSET + offset

    # Bring-up

    def _program_interrupts(self) -> None:
        """A 4 KiB interrupt ring at a bus address and one armed MSI-X vector."""
        ring, wptr = GUEST_BASE + SYS_IH_RING, GUEST_BASE + SYS_IH_WPTR
        self.irq = os.eventfd(0, os.EFD_CLOEXEC | os.EFD_NONBLOCK)
        self.client.arm_irq(IRQ_MSIX, self.irq)
        self.reg_write_at(IH_BASE, ring >> 8)
        self.reg_write_at(IH_BASE_HI, (ring >> 40) & 0xFF)
        self.reg_write_at(IH_WPTR_ADDR_LO, wptr)
        self.reg_write_at(IH_WPTR_ADDR_HI, (wptr >> 32) & 0xFFFF)
        self.reg_write_at(
            IH_CONTROL,
            (10 << 1) | IH_RING_ENABLE | IH_INTERRUPT_ENABLE | IH_BUS_ADDRESS,
        )

    def _program_gart_and_mes(self) -> None:
        gart_end = GART_START + GART_PAGES * PAGE - 1
        ptes = b"".join(
            struct.pack("<Q", (GUEST_BASE + page * PAGE) | PTE_LEAF)
            for page in range(GART_PAGES)
        )
        self.vram_write(VRAM_GART_PT, ptes)
        self.reg_write(REG_GART_PT_LO, VRAM_GART_PT)
        self.reg_write(REG_GART_PT_HI, VRAM_GART_PT >> 32)
        self.reg_write(REG_GART_START_LO, GART_START >> 12)
        self.reg_write(REG_GART_START_HI, GART_START >> 44)
        self.reg_write(REG_GART_END_LO, gart_end >> 12)
        self.reg_write(REG_GART_END_HI, gart_end >> 44)
        # VMID-0 only reaches the translator when the hub invalidation publishes it.
        self.reg_write(REG_VM_INVALIDATE_ENG17, 1)

        ring, rptr = GART_START + SYS_MES_RING, GART_START + SYS_MES_RPTR
        self.reg_write(REG_MES_RING_LO, ring >> 8)
        self.reg_write(REG_MES_RING_HI, ring >> 40)
        self.reg_write(REG_MES_RPTR_LO, rptr)
        self.reg_write(REG_MES_RPTR_HI, rptr >> 32)
        self.reg_write(REG_MES_DOORBELL, 0x4000_0000 | MES_DOORBELL)
        self.reg_write(REG_MES_CONTROL, 9)
        self.reg_write(REG_MES_ACTIVE, 1)

    def _map_process(self) -> None:
        pages = self.proc_bytes // PAGE
        tables = (pages + 511) // 512
        first = PROC_VA_BASE >> 21
        if first + tables + 1 > 512:
            raise HarnessError("process memory does not fit one page directory")
        self.vram_write(VRAM_PROC_ROOT, struct.pack("<Q", VRAM_PDB2 | PDE_VALID))
        self.vram_write(VRAM_PDB2, struct.pack("<Q", VRAM_PDB1 | PDE_VALID))
        self.vram_write(VRAM_PDB1, struct.pack("<Q", VRAM_PDB0 | PDE_VALID))
        pdes = b"".join(
            struct.pack("<Q", (VRAM_PTB_BASE + table * PAGE) | PDE_VALID)
            for table in range(tables)
        )
        self.vram_write(VRAM_PDB0 + first * 8, pdes)
        base = GUEST_BASE + PROC_GUEST_OFFSET
        ptes = array.array(
            "Q", ((base + page * PAGE) | PTE_LEAF for page in range(pages))
        )
        self.vram_write(VRAM_PTB_BASE, ptes.tobytes())
        # One more table maps a window of VRAM right after the process memory.
        self.vram_va = PROC_VA_BASE + tables * 512 * PAGE
        vram_table = VRAM_PTB_BASE + tables * PAGE
        self.vram_write(
            VRAM_PDB0 + (first + tables) * 8, struct.pack("<Q", vram_table | PDE_VALID)
        )
        ptes = array.array(
            "Q",
            (
                (VRAM_DATA + page) | PTE_LEAF_VRAM
                for page in range(0, VRAM_WINDOW_BYTES, PAGE)
            ),
        )
        self.vram_write(vram_table, ptes.tobytes())

    def _mes_submit(self, frame: bytes) -> None:
        slot = (self.mes_wptr % MES_RING_DWORDS) * 4
        self.mem[SYS_MES_RING + slot : SYS_MES_RING + slot + len(frame)] = frame
        self.put(SYS_MES_STATUS, "<I", 0)
        self.mes_wptr += MES_FRAME_DWORDS
        self.client.region_write(
            BAR_DOORBELL, MES_DOORBELL, struct.pack("<Q", self.mes_wptr)
        )
        deadline = time.monotonic() + 10
        while self.get(SYS_MES_STATUS, "<I") != 1:
            if time.monotonic() > deadline:
                raise HarnessError("MES did not complete a command; see the server log")
            time.sleep(0.001)

    def _create_queue(self) -> None:
        self.queue_va, self.queue_off = self.alloc(PAGE)
        self.ring_va, self.ring_off = self.alloc(AQL_PACKETS * 64)
        self.signal_va, self.signal_off = self.alloc(AQL_PACKETS * SIGNAL_BYTES)
        self.put(self.queue_off + Q_BASE_ADDRESS, "<Q", self.ring_va)
        self.put(self.queue_off + Q_SIZE, "<I", AQL_PACKETS)
        self.put(self.queue_off + Q_READ_ID_OFFSET_FIELD, "<I", Q_READ_ID)
        self.put(
            self.queue_off + Q_PROPERTIES, "<I", 2
        )  # AMD_QUEUE_PROPERTIES_IS_PTR64

        read_ptr, write_ptr = self.queue_va + Q_READ_ID, self.queue_va + Q_WRITE_ID
        mqd = SYS_MQD
        self.put(mqd + 136 * 4, "<I", (self.ring_va >> 8) & U32)
        self.put(mqd + 137 * 4, "<I", self.ring_va >> 40)
        self.put(mqd + 139 * 4, "<Q", read_ptr)
        self.put(mqd + 141 * 4, "<Q", write_ptr)
        self.put(mqd + 143 * 4, "<I", AQL_DOORBELL)
        self.put(mqd + 145 * 4, "<I", AQL_QUEUE_SIZE_FIELD | MQD_AQL)
        self.put(mqd + 181 * 4, "<I", 1)

        frame = bytearray(MES_FRAME_DWORDS * 4)
        struct.pack_into("<II", frame, 0, MES_ADD_QUEUE, PASID)
        struct.pack_into("<Q", frame, 2 * 4, VRAM_PROC_ROOT)
        struct.pack_into("<I", frame, 18 * 4, AQL_DOORBELL // 4)
        struct.pack_into("<Q", frame, 20 * 4, GART_START + SYS_MQD)
        struct.pack_into("<Q", frame, 22 * 4, write_ptr)
        struct.pack_into("<I", frame, 28 * 4, MES_QUEUE_COMPUTE)
        struct.pack_into("<Q", frame, 38 * 4, GART_START + SYS_MES_STATUS)
        struct.pack_into("<Q", frame, 40 * 4, 1)
        self._mes_submit(bytes(frame))

    # Dispatch

    def load(self, code: CodeObject) -> None:
        self.code = code
        self.code_va, code_off = self.alloc(len(code.image))
        self.mem[code_off : code_off + len(code.image)] = code.image
        self.code_off = code_off
        self.kernarg_va, self.kernarg_off = self.alloc(PAGE)

    def enqueue(self, kernel: Kernel, grid: int, args: bytes) -> int:
        """Write one dispatch packet; returns its signal slot. Call kick() to submit."""
        slot = self.write_id % AQL_PACKETS
        self.mem[self.kernarg_off : self.kernarg_off + PAGE] = args.ljust(PAGE, b"\0")
        signal_off = self.signal_off + slot * SIGNAL_BYTES
        self.mem[signal_off : signal_off + SIGNAL_BYTES] = bytes(SIGNAL_BYTES)
        self.put(signal_off + SIGNAL_VALUE, "<q", 1)
        kd = self.code_off + kernel.kd_offset
        header = 2 | (1 << 8) | (2 << 9) | (2 << 11)
        packet = struct.pack(
            "<HHHHHHIIIIIQQQQ",
            0, 1, WORKGROUP, 1, 1, 0, grid, 1, 1,
            self.get(kd + KD_PRIVATE_SEGMENT, "<I"), self.get(kd + KD_GROUP_SEGMENT, "<I"),
            self.code_va + kernel.kd_offset, self.kernarg_va, 0,
            self.signal_va + slot * SIGNAL_BYTES,
        )  # fmt: skip
        packet_off = self.ring_off + slot * 64
        self.mem[packet_off : packet_off + 64] = packet
        self.put(packet_off, "<H", header)
        self.write_id += 1
        return slot

    def kick(self) -> None:
        self.put(self.queue_off + Q_WRITE_ID, "<Q", self.write_id)
        # The doorbell carries the index of the last published packet.
        value = struct.pack("<Q", self.write_id - 1)
        self.client.region_write(
            BAR_DOORBELL, AQL_DOORBELL, value, posted=self.posted_doorbells
        )

    def wait(
        self, slot: int, timeout: float, server: "Server", irq: bool = False
    ) -> None:
        value = self.signal_off + slot * SIGNAL_BYTES + SIGNAL_VALUE
        unpack, mem = struct.unpack_from, self.mem
        deadline = time.monotonic() + timeout
        while irq and unpack("<q", mem, value)[0] != 0:
            # Interrupts that complete nothing must not extend the wait.
            if time.monotonic() > deadline:
                raise HarnessError(
                    f"dispatch did not complete in {timeout:g}s; see {server.log}"
                )
            # Sleep until the device raises its vector, as a driver would.
            if select.select([self.irq], [], [], min(1.0, timeout))[0]:
                os.eventfd_read(self.irq)
            elif server.process.poll() is not None:
                raise HarnessError(f"no completion interrupt arrived; see {server.log}")
        spins = 0
        while unpack("<q", mem, value)[0] != 0:
            spins += 1
            if spins & 0xFFFF == 0:
                if server.process.poll() is not None:
                    raise HarnessError(
                        f"server exited with {server.process.returncode}; see {server.log}"
                    )
                if time.monotonic() > deadline:
                    raise HarnessError(
                        f"dispatch did not complete in {timeout:g}s; see {server.log}"
                    )

    def run(
        self,
        kernel: Kernel,
        grid: int,
        args: bytes,
        timeout: float,
        server: "Server",
        irq: bool = False,
    ) -> float:
        slot = self.enqueue(kernel, grid, args)
        start = time.perf_counter_ns()
        self.kick()
        self.wait(slot, timeout, server, irq)
        return (time.perf_counter_ns() - start) / 1e9


# Server


@dataclass
class Variant:
    name: str
    binary: str
    config: str
    env: dict[str, str] = field(default_factory=dict)
    args: list[str] = field(default_factory=list)
    cpus: str = ""
    cfg: dict[str, object] = field(default_factory=dict)
    target: str = ""


def split_variant_items(rest: str, text: str) -> list[str]:
    """Split a variant's items at commas, except those inside a cfg JSON value."""
    items, start, depth, quoted = [], 0, 0, False
    structured = rest.startswith("cfg=")
    for index, char in enumerate(rest):
        if structured and char == '"' and rest[index - 1] != "\\":
            quoted = not quoted
        elif structured and not quoted and char in "[{":
            depth += 1
        elif structured and not quoted and char in "]}":
            depth -= 1
        elif char == "," and depth <= 0 and not quoted:
            items.append(rest[start:index])
            start, depth = index + 1, 0
            structured = rest.startswith("cfg=", start)
    if depth > 0 or quoted:
        raise HarnessError(f"unbalanced brackets or quotes in '{text}'")
    items.append(rest[start:])
    return items


def parse_variant(
    text: str, default_binary: str, default_config: str, default_target: str
) -> Variant:
    name, _, rest = text.partition(":")
    variant = Variant(name, default_binary, default_config, target=default_target)
    for item in filter(None, split_variant_items(rest, text)):
        key, _, value = item.partition("=")
        if key == "bin":
            variant.binary = value
        elif key == "config":
            variant.config = value
        elif key == "target":
            variant.target = value
        elif key == "cpus":
            variant.cpus = value
        elif key == "arg":
            variant.args.append(value)
        elif key == "env":
            env_key, _, env_value = value.partition("=")
            variant.env[env_key] = env_value
        elif key == "cfg":
            path, _, raw = value.partition("=")
            try:
                variant.cfg[path] = json.loads(raw)
            except json.JSONDecodeError:
                variant.cfg[path] = raw
        else:
            raise HarnessError(f"unknown variant key '{key}' in '{text}'")
    return variant


def build_syscall_shim(compiler: str, workdir: Path) -> str:
    source, library = workdir / "syscall_shim.c", workdir / "syscall_shim.so"
    source.write_text(SYSCALL_SHIM_C)
    run_tool(
        [compiler, "-O2", "-shared", "-fPIC", str(source), "-o", str(library), "-ldl"],
        "build the syscall-counting shim",
    )
    return str(library)


def process_cpu_seconds(pid: int, proc: Path = PROC) -> dict[str, tuple[float, float]]:
    """(user, system) CPU seconds per live thread, keyed by '<tid> <comm>'."""
    ticks = os.sysconf("SC_CLK_TCK")
    usage: dict[str, tuple[float, float]] = {}
    try:
        tasks = list((proc / str(pid) / "task").iterdir())
    except OSError:
        return usage
    for task in tasks:
        try:
            stat = (task / "stat").read_text()
        except OSError:
            continue
        # comm fills the outermost parentheses and may itself hold spaces and
        # parentheses.
        opened, closed = stat.find("("), stat.rfind(")")
        fields = stat[closed + 1 :].split()
        if opened < 0 or closed < opened or len(fields) < 13:
            continue  # a thread caught while it exits
        user, system = fields[11], fields[12]
        if user.isdigit() and system.isdigit():
            usage[f"{task.name} {stat[opened + 1 : closed]}"] = (
                int(user) / ticks,
                int(system) / ticks,
            )
    return usage


def process_sched(pid: int, proc: Path = PROC) -> tuple[int, int, int, int]:
    """(on-CPU ns, runqueue-wait ns, voluntary switches, involuntary switches), summed over live threads.

    A thread that exits between samples takes its counts with it, so deltas can undercount.
    """
    run = wait = voluntary = involuntary = 0
    try:
        tasks = list((proc / str(pid) / "task").iterdir())
    except OSError:
        return 0, 0, 0, 0
    for task in tasks:
        try:
            schedstat = (task / "schedstat").read_text().split()
            status = (task / "status").read_text()
        except OSError:
            continue
        if len(schedstat) != 3 or not all(value.isdigit() for value in schedstat):
            continue
        run += int(schedstat[0])
        wait += int(schedstat[1])
        for line in status.splitlines():
            key, _, value = line.partition(":")
            if not value.strip().isdigit():
                continue
            if key == "voluntary_ctxt_switches":
                voluntary += int(value)
            elif key == "nonvoluntary_ctxt_switches":
                involuntary += int(value)
    return run, wait, voluntary, involuntary


def override_config(path: str, overrides: dict[str, object]) -> dict:
    """The JSON config at `path` with each dotted-path override applied."""
    try:
        document = json.loads(Path(path).read_text())
    except json.JSONDecodeError as error:
        raise HarnessError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(document, dict):
        raise HarnessError(f"{path} does not hold a JSON object")
    for dotted, value in overrides.items():
        node = document
        *parents, leaf = dotted.split(".")
        for parent in parents:
            node = node.setdefault(parent, {})
            if not isinstance(node, dict):
                raise HarnessError(
                    f"cfg={dotted} cannot be applied: "
                    f"'{parent}' in {path} is not an object"
                )
        node[leaf] = value
    return document


def log_tail(log: Path, lines: int = 8) -> list[str]:
    try:
        return log.read_text(errors="replace").splitlines()[-lines:]
    except OSError:
        return []


def die_with_parent() -> None:
    """Runs in the forked server: the kernel ends it if this process is killed.

    The server keeps serving after its client is gone, so without this a
    harness ended by SIGKILL (a test timeout, say) would leave it running.
    """
    LIBC.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)


class Server:
    def __init__(
        self,
        variant: Variant,
        workdir: Path,
        shim: str = "",
        ready_timeout: float = 120.0,
    ):
        self.socket_path = str(workdir / f"{variant.name}.sock")
        self.log = workdir / f"{variant.name}.log"
        self.counters = None
        env = {**os.environ, **variant.env}
        if shim:
            counters = workdir / f"{variant.name}.counters"
            counters.write_bytes(bytes(PAGE))
            with open(counters, "r+b") as file:
                self.counters = mmap.mmap(file.fileno(), PAGE)
            env |= {"LD_PRELOAD": shim, "RJ_BENCH_SYSCALL_COUNTERS": str(counters)}
        config = variant.config
        if variant.cfg:
            document = override_config(config, variant.cfg)
            config = str(workdir / f"{variant.name}.json")
            Path(config).write_text(json.dumps(document))
        ready_read, ready_write = os.pipe()
        command = [variant.binary, "--config", config, "--vfio-socket", self.socket_path,
                   "--vfio-ready-fd", str(ready_write), *variant.args]  # fmt: skip
        if variant.cpus:
            command = ["taskset", "-c", variant.cpus, *command]
        try:
            with open(self.log, "wb") as log:
                self.process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    pass_fds=[ready_write],
                    env=env,
                    preexec_fn=die_with_parent,
                )
        except OSError:
            os.close(ready_read)
            raise
        finally:
            os.close(ready_write)
        with os.fdopen(ready_read, "rb") as ready:
            # One byte arrives once the server listens. End of file means it
            # exited first.
            if not select.select([ready], [], [], ready_timeout)[0]:
                self.stop()
                raise HarnessError(
                    f"server was not ready within {ready_timeout:g}s; see {self.log}"
                )
            if ready.read(1) != b"\x01":
                try:
                    # Let a server that is exiting report how, before stop()
                    # signals it.
                    self.process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass
                fault = self.stop()
                raise HarnessError(
                    "server did not become ready"
                    + (f" ({fault})" if fault else "")
                    + f"; see {self.log}"
                )

    def cpu_seconds(self) -> dict[str, tuple[float, float]]:
        return process_cpu_seconds(self.process.pid)

    def io_syscalls(self) -> int | None:
        """pread + pwrite calls the server has made, when --count-syscalls is on."""
        return (
            sum(struct.unpack_from("<QQ", self.counters))
            if self.counters is not None
            else None
        )

    stop_timeout = 20.0

    def stop(self) -> str:
        """Stop the server; returns what was wrong with how it ended, or "".

        The server ends its session and exits with status 0 when it is sent
        SIGTERM, and with another status when the simulation or the serving
        thread failed. It blocks SIGTERM until its status is final, and stays
        up after its client leaves, so one that dies of the SIGTERM sent here
        was already leaving because it had failed. Only status 0 is a clean
        end: a non-zero status, death by any signal, and a server that had to
        be killed are faults.
        """
        process = self.process
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=self.stop_timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                return f"did not stop within {self.stop_timeout:g}s of SIGTERM and was killed"
        status = process.returncode
        if status == 0:
            return ""
        if status > 0:
            return f"exited with status {status}"
        try:
            return f"was killed by {signal.Signals(-status).name}"
        except ValueError:
            return f"was killed by signal {-status}"


# Expected results


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def round_to_f32(numerator: int, shift: int) -> float:
    """numerator / 2**shift rounded once to the nearest binary32, ties to even."""
    magnitude = abs(numerator)
    if magnitude == 0:
        return 0.0
    # The leading bit weighs 2**exponent. A binary32 keeps 23 bits below it, and
    # nothing finer than 2**-149 once the value is subnormal.
    exponent = magnitude.bit_length() - 1 - shift
    ulp = max(exponent - 23, -149)
    drop = shift + ulp
    if drop <= 0:
        quotient = magnitude << -drop
    else:
        quotient, remainder = magnitude >> drop, magnitude & ((1 << drop) - 1)
        half = 1 << (drop - 1)
        if remainder > half or (remainder == half and quotient & 1):
            quotient += 1
    # At most 25 bits are left, so a double holds quotient * 2**ulp exactly.
    rounded = math.ldexp(quotient, ulp) if exponent <= 127 else math.inf
    if rounded >= 2.0**128:
        rounded = math.inf
    return -rounded if numerator < 0 else rounded


def fma32(multiplicand: float, multiplier: float, addend: float) -> float:
    """multiplicand * multiplier + addend on finite binary32 values, rounded once.

    The exact result is formed over the integers. math.fma needs Python 3.13 and
    rounds to binary64, and rounding that again to binary32 can land on the
    wrong neighbour. An exact zero comes back as +0.0.
    """
    left, left_scale = multiplicand.as_integer_ratio()
    right, right_scale = multiplier.as_integer_ratio()
    added, added_scale = addend.as_integer_ratio()
    # Every scale is a power of two, so the common denominator is one as well.
    numerator = left * right * added_scale + added * left_scale * right_scale
    denominator = left_scale * right_scale * added_scale
    return round_to_f32(numerator, denominator.bit_length() - 1)


def expected(name: str, gid: int, iters: int, buf: list[int]) -> list[float]:
    """Acceptable outputs for one work item."""
    if name == "int_loop":
        state = (gid * 2654435761 + 1) & U32
        for _ in range(iters):
            state ^= (state << 13) & U32
            state ^= state >> 17
            state ^= (state << 5) & U32
        return [state]
    if name == "i64_loop":
        state = (gid * 0x9E3779B97F4A7C15 + 1) & U64
        for _ in range(iters):
            state ^= (state << 13) & U64
            state ^= state >> 7
            state ^= (state << 17) & U64
        return [(state ^ (state >> 32)) & U32]
    if name == "fp_loop":
        start = f32(f32(f32(float(gid & 1023)) * f32(0.001)) + f32(0.5))
        scale, offset = f32(0.999), f32(0.001)
        separate = fused = start
        for _ in range(iters):
            separate = f32(f32(separate * scale) + offset)
            fused = fma32(fused, scale, offset)
        return [separate, fused]
    if name in BUFFER_PLACEMENT:
        acc = gid
        for step in range(iters):
            acc = (acc + buf[(gid + step * 97) & (MEM_WORDS - 1)]) & U32
        return [acc]
    if name == "lds_loop":
        group, lid = gid - gid % WORKGROUP, gid % WORKGROUP
        acc = gid
        for step in range(iters):
            acc = (acc + (group + ((lid + step) % WORKGROUP)) * 3 + 1) & U32
        return [acc]
    return []


# The same results without stepping through every iteration, so that a timed
# dispatch of any length can be checked in full. expected() above stays the
# reference; the unit tests hold the two to each other.


def linear_power(step: Callable[[int], int], bits: int, count: int) -> list[list[int]]:
    """Byte lookup tables for `count` applications of a GF(2)-linear step.

    A map that only shifts and XORs is linear over GF(2), so it is fixed by the
    images of the single-bit states, and composing it with itself squares it.
    """

    def apply(images: list[int], state: int) -> int:
        result = 0
        while state:
            low = state & -state
            result ^= images[low.bit_length() - 1]
            state ^= low
        return result

    power = [1 << bit for bit in range(bits)]
    square = [step(1 << bit) for bit in range(bits)]
    while count:
        if count & 1:
            power = [apply(square, image) for image in power]
        square = [apply(square, image) for image in square]
        count >>= 1
    tables = []
    for base in range(0, bits, 8):
        table = [0] * 256
        for value in range(1, 256):
            low = value & -value
            table[value] = table[value ^ low] ^ power[base + low.bit_length() - 1]
        tables.append(table)
    return tables


def int_loop_step(state: int) -> int:
    state ^= (state << 13) & U32
    state ^= state >> 17
    return state ^ ((state << 5) & U32)


def i64_loop_step(state: int) -> int:
    state ^= (state << 13) & U64
    state ^= state >> 7
    return state ^ ((state << 17) & U64)


# (start value index, fused) -> (iterations after which fp_loop stops changing, value).
FP_SETTLED: dict[tuple[int, bool], tuple[int, float]] = {}


def fp_loop_value(index: int, iters: int, fused: bool) -> float:
    """fp_loop's result for a work item whose gid & 1023 is `index`.

    Rounding keeps x * 0.999 + 0.001 non-decreasing in x, so the iterates move
    one way and stop at a value the step maps to itself. Nothing changes after
    that, however many iterations follow.
    """
    settled = FP_SETTLED.get((index, fused))
    if settled is not None and iters >= settled[0]:
        return settled[1]
    pack, unpack = struct.Struct("<f").pack, struct.Struct("<f").unpack
    scale, offset = f32(0.999), f32(0.001)
    value = f32(f32(f32(float(index)) * f32(0.001)) + f32(0.5))
    for step in range(iters):
        # Two binary32 values multiply exactly in binary64.
        product = value * scale
        if fused:
            total = product + offset
            tail = total - product
            if (product - (total - tail)) + (offset - tail) == 0.0:
                # The sum was exact as well, so this is the one rounding of a fused step.
                after = unpack(pack(total))[0]
            else:
                after = fma32(value, scale, offset)
        else:
            after = unpack(pack(unpack(pack(product))[0] + offset))[0]
        if after == value:
            FP_SETTLED[(index, fused)] = (step, value)
            return value
        value = after
    return value


def expected_grid(
    name: str, grid: int, iters: int, buf: list[int]
) -> list[list[float]]:
    """Acceptable outputs of work items 0..grid-1, one list per acceptable device.

    Agrees with expected() for every work item and takes time that does not
    grow with `iters` (fp_loop: not beyond the point where it stops changing).
    """
    if name == "int_loop":
        t0, t1, t2, t3 = linear_power(int_loop_step, 32, iters)
        starts = ((gid * 2654435761 + 1) & U32 for gid in range(grid))
        return [
            [t0[x & 255] ^ t1[x >> 8 & 255] ^ t2[x >> 16 & 255] ^ t3[x >> 24] for x in starts]
        ]  # fmt: skip
    if name == "i64_loop":
        tables = linear_power(i64_loop_step, 64, iters)
        values = []
        for gid in range(grid):
            start, state = (gid * 0x9E3779B97F4A7C15 + 1) & U64, 0
            for table in tables:
                state ^= table[start & 255]
                start >>= 8
            values.append((state ^ (state >> 32)) & U32)
        return [values]
    if name == "fp_loop":
        distinct = min(grid, 1024)
        return [
            [values[gid & 1023] for gid in range(grid)]
            for values in (
                [fp_loop_value(index, iters, fused) for index in range(distinct)]
                for fused in (False, True)
            )
        ]
    if name in BUFFER_PLACEMENT:
        # 97 is odd, so gid + 97 * step visits every word once per MEM_WORDS
        # steps. Sum along that order once and every item is two lookups.
        mask = MEM_WORDS - 1
        prefix, total = [0], 0
        for position in range(MEM_WORDS):
            total += buf[(position * 97) & mask]
            prefix.append(total)
        laps, rest = divmod(iters, MEM_WORDS)
        inverse = pow(97, -1, MEM_WORDS)
        values = []
        for gid in range(grid):
            first = (gid * inverse) & mask
            last = first + rest
            part = prefix[last] - prefix[first] if last <= MEM_WORDS else (
                total - prefix[first] + prefix[last - MEM_WORDS])  # fmt: skip
            values.append((gid + laps * total + part) & U32)
        return [values]
    if name == "lds_loop":
        # Sum of (lid + step) % WORKGROUP over the steps left after whole laps.
        prefix = [0]
        for position in range(2 * WORKGROUP):
            prefix.append(prefix[-1] + position % WORKGROUP)
        laps, rest = divmod(iters, WORKGROUP)
        values = []
        for gid in range(grid):
            lid = gid % WORKGROUP
            lids = laps * prefix[WORKGROUP] + prefix[lid + rest] - prefix[lid]
            values.append((gid + iters * ((gid - lid) * 3 + 1) + lids * 3) & U32)
        return [values]
    return []


def count_wrong(name: str, raw: bytes, candidates: list[list[float]]) -> int:
    """How many work items in `raw`, a dispatch's output, match no candidate."""
    if name == "fp_loop":
        return sum(
            not any(math.isclose(got, value, rel_tol=1e-6) for value in want)
            for got, *want in zip(array.array("f", raw), *candidates)
        )
    (want,) = candidates
    if raw == array.array("I", want).tobytes():
        return 0
    return sum(got != value for got, value in zip(array.array("I", raw), want))


# Benchmarks


@dataclass
class Bench:
    device: Device
    server: Server
    code: CodeObject
    timeout: float

    def __post_init__(self) -> None:
        device = self.device
        self.out_va, self.out_off = device.alloc(self.max_grid * 4)
        self.buf_va, self.buf_off = device.alloc(MEM_WORDS * 4)
        self.buf = [
            (index * 2246822519 + 374761393) & U32 for index in range(MEM_WORDS)
        ]
        raw = array.array("I", self.buf).tobytes()
        device.mem[self.buf_off : self.buf_off + MEM_WORDS * 4] = raw
        device.vram_write(VRAM_DATA, raw)

    max_grid: int = 1 << 16

    def args(self, iters: int, name: str = "") -> bytes:
        buf = (
            self.device.vram_va if BUFFER_PLACEMENT.get(name) == "vram" else self.buf_va
        )
        return struct.pack("<QIIQI", self.out_va, iters, 0, buf, MEM_WORDS - 1)

    def kernel(self, name: str) -> Kernel:
        return self.code.kernels["mem_loop" if name in BUFFER_PLACEMENT else name]

    def run(self, name: str, grid: int, iters: int) -> float:
        return self.device.run(
            self.kernel(name), grid, self.args(iters, name), self.timeout, self.server
        )

    def clear_output(self, grid: int) -> None:
        """Overwrite the output with all-ones words.

        That is a NaN for fp_loop, which never matches. An integer kernel
        computes it for a work item only by coincidence, so an item left
        unwritten is counted wrong unless its expected value is 0xFFFFFFFF.
        """
        self.device.mem[self.out_off : self.out_off + grid * 4] = b"\xff" * (grid * 4)

    def read_output(self, grid: int) -> bytes:
        return bytes(self.device.mem[self.out_off : self.out_off + grid * 4])

    def verify(self, grid: int = 4096, iters: int = 37) -> dict[str, str]:
        """Check every work item of every kernel, across enough workgroups to span compute units."""
        verdicts = {}
        for name in BENCHMARKS:
            self.clear_output(grid)
            self.run(name, grid, iters)
            items = [expected(name, gid, iters, self.buf) for gid in range(grid)]
            bad = count_wrong(
                name, self.read_output(grid), [list(c) for c in zip(*items)]
            )
            verdicts[name] = "ok" if bad == 0 else f"{bad}/{grid} wrong"
        return verdicts

    def mmio_latency(self, count: int, spin: bool = False) -> dict[str, float]:
        client, samples = self.device.client, []
        client.spin = spin
        for _ in range(count):
            start = time.perf_counter_ns()
            client.region_read(BAR_REGISTERS, 0, 4)
            samples.append((time.perf_counter_ns() - start) / 1e3)
        client.spin = False
        return summarize(samples)

    def dispatch_latency(
        self, count: int, irq: bool = False, posted: bool = False
    ) -> dict[str, float]:
        kernel, args = self.code.kernels["noop"], self.args(0)
        run = self.device.run
        self.device.posted_doorbells = posted
        samples = [
            run(kernel, WORKGROUP, args, self.timeout, self.server, irq) * 1e6
            for _ in range(count)
        ]
        self.device.posted_doorbells = False
        return summarize(samples)

    def batched_dispatch(self, batch: int, rounds: int) -> float:
        """Dispatches per second when one doorbell covers `batch` packets."""
        kernel, args = self.code.kernels["noop"], self.args(0)
        start = time.perf_counter_ns()
        for _ in range(rounds):
            slots = [self.device.enqueue(kernel, WORKGROUP, args) for _ in range(batch)]
            self.device.kick()
            self.device.wait(slots[-1], self.timeout, self.server)
        return batch * rounds / ((time.perf_counter_ns() - start) / 1e9)

    def throughput(
        self, name: str, grid: int, target_seconds: float, repeats: int
    ) -> dict[str, object]:
        """Time `repeats` dispatches and check every work item each one wrote.

        "wrong_items" counts the wrong work items of each timed dispatch and
        "output" is "ok" or the worst of them. The calibration dispatches are
        neither timed nor checked.
        """
        pid = self.server.process.pid
        iters = 64
        while True:
            seconds = self.run(name, grid, iters)
            if seconds >= target_seconds / 2 or iters >= 1 << 28:
                break
            growth = max(2.0, target_seconds / max(seconds, 1e-6))
            iters = min(1 << 28, int(iters * growth))
        before, wall_start = process_cpu_seconds(pid), time.perf_counter()
        sched_before = process_sched(pid)
        io_before = self.server.io_syscalls()
        times, outputs = [], []
        for _ in range(repeats):
            # Each timed dispatch has to produce its own output, and all of it
            # is kept. Both copies are outside the interval run() times; the
            # comparison waits until the accounting below is closed.
            self.clear_output(grid)
            times.append(self.run(name, grid, iters))
            outputs.append(self.read_output(grid))
        wall = time.perf_counter() - wall_start
        after = process_cpu_seconds(pid)
        run_ns, wait_ns, voluntary, involuntary = (
            now - then for now, then in zip(process_sched(pid), sched_before)
        )
        executed = self.kernel(name)
        lanes = executed.wave_lanes
        wave_iters = (grid / lanes) * iters * repeats
        io_after = self.server.io_syscalls()
        io_per_wave_iter = (
            None if io_after is None else (io_after - io_before) / wave_iters
        )
        zero = (0.0, 0.0)
        user = sum(after[thread][0] - before.get(thread, zero)[0] for thread in after)
        system = sum(after[thread][1] - before.get(thread, zero)[1] for thread in after)
        threads = sorted(
            (sum(after[thread]) - sum(before.get(thread, zero))) / wall
            for thread in after
        )[::-1]
        best = min(times)
        loop_insts = executed.loop_insts
        candidates = expected_grid(name, grid, iters, self.buf)
        wrong = [count_wrong(name, raw, candidates) for raw in outputs]
        return {
            "kernel": name,
            "grid": grid,
            "iters": iters,
            "best_s": best,
            "median_s": statistics.median(times),
            "lane_iters_per_s": grid * iters / best,
            "wave_insts_per_s": (grid / lanes) * iters * loop_insts / best,
            "loop_insts": loop_insts,
            "wave_lanes": lanes,
            "server_cores": sum(threads),
            "server_sys_share": system / (user + system) if user + system else 0.0,
            "runq_wait_share": (
                wait_ns / (run_ns + wait_ns) if run_ns + wait_ns > 0 else 0.0
            ),
            "vol_switches_per_s": voluntary / wall,
            "invol_switches_per_s": involuntary / wall,
            "io_syscalls_per_wave_iter": io_per_wave_iter,
            "busy_threads": sum(1 for share in threads if share > 0.5),
            "top_thread_share": threads[0] if threads else 0.0,
            "wrong_items": wrong,
            "output": "ok" if not any(wrong) else f"{max(wrong)}/{grid} wrong",
        }


def summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "median_us": statistics.median(ordered),
        "p99_us": ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))],
        "min_us": ordered[0],
    }


def run_variant(
    variant: Variant,
    code: CodeObject,
    options: argparse.Namespace,
    workdir: Path,
    shim: str,
) -> dict:
    result: dict[str, object] = {"variant": variant.name, "binary": variant.binary,
                                 "config": variant.config, "env": variant.env, "args": variant.args,
                                 "cpus": variant.cpus, "cfg": variant.cfg, "target": code.target,
                                 "wave_lanes": code.kernels["noop"].wave_lanes}  # fmt: skip
    server = client = device = None
    try:
        server = Server(variant, workdir, shim, options.ready_timeout)
        client = VfioUserClient(server.socket_path)
        bars = {
            bar: client.region_size(bar)
            for bar in (BAR_VRAM, BAR_DOORBELL, BAR_REGISTERS)
        }
        device = Device(client, options.memory_mib << 20)
        device.load(code)
        bench = Bench(device, server, code, options.timeout)
        result["bar_bytes"] = bars
        result["verify"] = bench.verify()
        if not options.verify_only:
            result["mmio_read"] = bench.mmio_latency(options.latency_samples)
            result["mmio_read_spin"] = bench.mmio_latency(
                options.latency_samples, spin=True
            )
            result["noop_dispatch"] = bench.dispatch_latency(options.latency_samples)
            result["noop_dispatch_irq"] = bench.dispatch_latency(
                options.latency_samples, irq=True
            )
            result["noop_dispatch_posted"] = bench.dispatch_latency(
                options.latency_samples, posted=True
            )
            result["batched_dispatch_per_s"] = {
                str(batch): bench.batched_dispatch(
                    batch, max(1, options.latency_samples // batch)
                )
                for batch in options.batches
            }
            result["threads"] = len(server.cpu_seconds())
            result["throughput"] = [
                bench.throughput(name, grid, options.target_seconds, options.repeats)
                for name in options.kernels
                for grid in options.grids
            ]
    except (HarnessError, OSError) as error:
        # A shader ISA the front end cannot serve must not discard the other variants' results.
        result["error"] = str(error)
        result["server_log_tail"] = log_tail(workdir / f"{variant.name}.log")
    finally:
        # A defect in this script unwinds through here as well, and must not
        # leave the server running.
        if device is not None:
            device.close()
        if client is not None:
            client.close()
        if server is not None:
            fault = server.stop()
            result["server_exit"] = server.process.returncode
            if fault:
                # Results from a server that ended this way are not to be used,
                # even when every check before it passed.
                result["server_fault"] = f"server {fault}"
                result.setdefault("server_log_tail", log_tail(server.log))
    return result


def print_report(results: list[dict]) -> None:
    print("\n## Verify\n")
    print("| variant | target | wave lanes | " + " | ".join(BENCHMARKS) + " | error |")
    print("|---|---|---|" + "---|" * len(BENCHMARKS) + "---|")
    for result in results:
        verify = result.get("verify", {})
        print(
            f"| {result['variant']} | {result['target']} | {result['wave_lanes']} | "
            + " | ".join(verify.get(name, "-") for name in BENCHMARKS)
            + f" | {result.get('error') or result.get('server_fault', '')} |"
        )
    results = [result for result in results if "throughput" in result]
    if not results:
        return
    print("\n## Transport\n")
    print(
        "| variant | verify | MMIO read us | MMIO read, client spinning us | noop dispatch us | p99 us | "
        "via IRQ us | posted doorbell us | "
        + " | ".join(
            f"batch {batch} disp/s" for batch in results[0]["batched_dispatch_per_s"]
        )
        + " |"
    )
    print(
        "|---|---|---|---|---|---|---|---|"
        + "---|" * len(results[0]["batched_dispatch_per_s"])
    )
    for result in results:
        verify = (
            "ok"
            if all(verdict == "ok" for verdict in result["verify"].values())
            else json.dumps(result["verify"])
        )
        print(
            f"| {result['variant']} | {verify} | "
            f"{result['mmio_read']['median_us']:.1f} | "
            f"{result['mmio_read_spin']['median_us']:.1f} | "
            f"{result['noop_dispatch']['median_us']:.1f} | "
            f"{result['noop_dispatch']['p99_us']:.1f} | "
            f"{result['noop_dispatch_irq']['median_us']:.1f} | "
            f"{result['noop_dispatch_posted']['median_us']:.1f} | "
            + " | ".join(
                f"{rate:,.0f}" for rate in result["batched_dispatch_per_s"].values()
            )
            + " |"
        )
    print("\n## Throughput (best of repeats)\n")
    print(
        "| variant | target | kernel | grid | iters | seconds | M lane-iters/s | loop insts | "
        "M wave-insts/s | vs first | server cores | sys time | runq wait | vol cs/s | "
        "pread+pwrite per wave-iter | result |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    baseline = {
        (timing["kernel"], timing["grid"]): timing["lane_iters_per_s"]
        for timing in results[0]["throughput"]
    }
    for result in results:
        for timing in result["throughput"]:
            ratio = (
                timing["lane_iters_per_s"]
                / baseline[(timing["kernel"], timing["grid"])]
            )
            print(
                f"| {result['variant']} | {result['target']} | "
                f"{timing['kernel']} | {timing['grid']} | {timing['iters']} | "
                f"{timing['best_s']:.3f} | "
                f"{timing['lane_iters_per_s'] / 1e6:.2f} | "
                f"{timing['loop_insts']} | "
                f"{timing['wave_insts_per_s'] / 1e6:.2f} | {ratio:.2f}x | "
                f"{timing['server_cores']:.2f} | "
                f"{timing['server_sys_share']:.0%} | "
                f"{timing['runq_wait_share']:.0%} | "
                f"{timing['vol_switches_per_s']:,.0f} | "
                + (
                    "-"
                    if timing["io_syscalls_per_wave_iter"] is None
                    else f"{timing['io_syscalls_per_wave_iter']:.2f}"
                )
                + f" | {timing['output']} |"
            )


def wrong_outputs(result: dict) -> list[str]:
    """Each check of a variant's output that failed, before or after timing."""
    wrong = [
        f"{name}: {verdict}"
        for name, verdict in result.get("verify", {}).items()
        if verdict != "ok"
    ]
    wrong += [
        f"{timing['kernel']} timed at grid {timing['grid']}, {timing['iters']} "
        f"iterations: {timing['output']} (per run: {timing['wrong_items']})"
        for timing in result.get("throughput", [])
        if timing["output"] != "ok"
    ]
    return wrong


def require_inputs(variants: list[Variant]) -> None:
    """Report a variant that cannot start before anything is compiled or run."""
    names = [variant.name for variant in variants]
    for name in names:
        if names.count(name) > 1:
            # Each variant owns <name>.sock and <name>.log in the work directory.
            raise HarnessError(f"variant name '{name}' is used more than once")
    for variant in variants:
        if "/" in variant.name or not variant.name:
            raise HarnessError(f"'{variant.name}' is not usable as a variant name")
        if shutil.which(variant.binary) is None:
            raise HarnessError(
                f"[{variant.name}] rocjitsu binary is not executable: {variant.binary}"
            )
        if not Path(variant.config).is_file():
            raise HarnessError(f"[{variant.name}] config not found: {variant.config}")
        if variant.cpus and shutil.which("taskset") is None:
            raise HarnessError(f"[{variant.name}] cpus= needs taskset on PATH")


def parse_arguments(arguments: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--rocjitsu",
        default=str(SOURCE_ROOT / "build/tools/rocjitsu/rocjitsu"),
        help="default server binary",
    )
    parser.add_argument(
        "--config",
        default=str(SOURCE_ROOT / "configs/gfx1250_mi455x.json"),
        help="default config",
    )
    parser.add_argument("--variant", action="append", default=[],
                        help="NAME[:bin=PATH,config=PATH,target=GPU,env=K=V,arg=FLAG,cpus=LIST,"
                             "cfg=dotted.path=JSON]; repeatable, the first is the baseline")  # fmt: skip
    parser.add_argument(
        "--clang",
        help="amdgcn-capable clang; by default the one under $ROCM_PATH "
        "(/opt/rocm when unset), else amdclang or clang on PATH",
    )
    parser.add_argument("--target", default="gfx1250",
                        help="GPU the kernels are compiled for; a variant's target= overrides it")  # fmt: skip
    parser.add_argument("--pci-gfx-target-version", type=int,
                        help="set vm.gpu.device.gfx_target_version in every variant's config, so a config "
                             "for another shader ISA is served behind a PCI front end the server models "
                             "(120500 is the GC 12.1 / MES one this client drives)")  # fmt: skip
    parser.add_argument(
        "--check-toolchain",
        action="store_true",
        help="compile and disassemble the kernels for every target, report the "
        "tools used, and exit without starting a server",
    )
    parser.add_argument("--verify-only", action="store_true",
                        help="check every kernel's outputs and skip the timed runs")  # fmt: skip
    parser.add_argument(
        "--kernels", nargs="+", default=list(BENCHMARKS), choices=BENCHMARKS
    )
    parser.add_argument("--grids", nargs="+", type=int, default=[64, 4096])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 16, 256])
    parser.add_argument(
        "--target-seconds", type=float, default=2.0, help="length of a timed dispatch"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--latency-samples", type=int, default=2000)
    parser.add_argument(
        "--memory-mib",
        type=int,
        default=32,
        help="process memory shared with the device",
    )
    parser.add_argument(
        "--timeout", type=float, default=600.0, help="seconds to wait for one dispatch"
    )
    parser.add_argument(
        "--ready-timeout",
        type=float,
        default=120.0,
        help="seconds to wait for a server to start listening",
    )
    parser.add_argument("--count-syscalls", action="store_true",
                        help="preload a shim that counts the server's pread/pwrite calls")  # fmt: skip
    parser.add_argument(
        "--cc", default="cc", help="host C compiler for --count-syscalls"
    )
    parser.add_argument("--output", help="write results as JSON")
    parser.add_argument(
        "--keep", action="store_true", help="keep the work directory and server logs"
    )
    options = parser.parse_args(arguments)
    if any(
        grid < WORKGROUP or grid % WORKGROUP or grid > Bench.max_grid
        for grid in options.grids
    ):
        parser.error(f"grids must be multiples of {WORKGROUP} up to {Bench.max_grid}")
    if any(not 1 <= batch <= AQL_PACKETS for batch in options.batches):
        parser.error(f"batches must be between 1 and {AQL_PACKETS}")
    for name in ("repeats", "latency_samples", "memory_mib"):
        if getattr(options, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    for name in ("target_seconds", "timeout", "ready_timeout"):
        if not getattr(options, name) > 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return options


def main(arguments: list[str] | None = None) -> int:
    if sys.version_info < (3, 10):
        print("error: this script needs Python 3.10 or newer", file=sys.stderr)
        return 1
    options = parse_arguments(arguments)

    def terminate(signum: int, frame: object) -> None:
        sys.exit(128 + signum)

    # A cancelled CI job or test ends with SIGTERM. Unwind instead of dying in
    # place, so that the server this process started is stopped as well.
    previous_handler = signal.signal(signal.SIGTERM, terminate)
    workdir = None
    try:
        variants = [parse_variant(text, options.rocjitsu, options.config, options.target)
                    for text in options.variant or ["default"]]  # fmt: skip
        if options.pci_gfx_target_version is not None:
            for variant in variants:
                variant.cfg.setdefault(
                    "vm.gpu.device.gfx_target_version", options.pci_gfx_target_version
                )
        if not options.check_toolchain:
            require_inputs(variants)
        clang = find_clang(options.clang)
        workdir = Path(tempfile.mkdtemp(prefix="rj-vfio-bench-"))
        codes = {target: build_code_object(clang, target, workdir)
                 for target in dict.fromkeys(variant.target for variant in variants)}  # fmt: skip
        if options.check_toolchain:
            print(f"clang: {clang}")
            print(f"llvm-objdump: {find_objdump(clang)}")
            for target, code in codes.items():
                lanes = code.kernels["noop"].wave_lanes
                print(f"{target}: {len(code.kernels)} kernels, {lanes} lanes per wave")
            return 0
        shim = build_syscall_shim(options.cc, workdir) if options.count_syscalls else ""
        results = []
        for variant in variants:
            print(
                f"[{variant.name}] {variant.binary} ({variant.target})", file=sys.stderr
            )
            results.append(
                run_variant(variant, codes[variant.target], options, workdir, shim)
            )
        if options.output:
            Path(options.output).write_text(json.dumps(results, indent=2))
        print_report(results)
        for result in results:
            # The work directory may be out of reach (a CI runner), so show
            # why the server stopped here as well.
            for line in result.get("server_log_tail", []):
                print(f"[{result['variant']}] server log: {line}", file=sys.stderr)
            for line in wrong_outputs(result):
                print(f"[{result['variant']}] wrong output: {line}", file=sys.stderr)
            if "server_fault" in result:
                print(
                    f"[{result['variant']}] {result['server_fault']}", file=sys.stderr
                )
        failed = [
            result["variant"]
            for result in results
            if "error" in result or "server_fault" in result or wrong_outputs(result)
        ]
        if failed:
            raise HarnessError(f"variants failed: {', '.join(failed)}")
    except (HarnessError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        if workdir is not None and not options.check_toolchain:
            options.keep = True
            print(f"work directory kept: {workdir}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        if workdir is not None and not options.keep:
            shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
