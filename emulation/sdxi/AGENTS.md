# AGENTS.md — Context for AI assistants working on rocm-sdxi

This file records project context, external references, and conventions
that are useful across sessions but are not derivable from the code alone.

---

## Purpose

rocm-sdxi is a vfio-user emulator of an AMD SDXI (SNIA Smart Data
Accelerator Interface) PCIe copy-engine device.  It is intended to let
software stacks targeting SDXI be developed and tested before silicon or
FPGA targets are available, using QEMU and the vfio-user protocol.

---

## Linux kernel driver

The upstream Linux driver for SDXI is being developed out-of-tree and
submitted to the dmaengine subsystem.  The primary authors are **Nathan
Lynch** and **Wei Huang** (AMD).

| Version | List | Thread URL |
|---------|------|------------|
| RFC (Sep 2025) | lkml | https://www.spinics.net/lists/kernel/msg5839841.html |
| v1 (Apr 2026) | lkml | https://ratatoskr.run/lkml/2026/04/3508169/t |
| v2 (May 2026) | linux-pci | https://ratatoskr.run/linux-pci/2026/05/8996752/t |
| v3 (Jun 2026) | linux-pci | https://ratatoskr.run/linux-pci/2026/06/17096410/t |
| v4 | linux-pci | **not yet posted** — URL unknown as of 2026-11-04; update `scripts/fetch-kernel-patches.sh` and this table when it appears |

LWN coverage: https://lwn.net/Articles/1036975/
Phoronix: https://www.phoronix.com/news/Initial-Linux-SDXI-Driver

The driver lives under `drivers/dma/sdxi/` and registers as a standard
dmaengine provider.  Its register header is the ground truth for the
hardware ABI that this emulator must implement.

### Fetching patches locally

```sh
./scripts/fetch-kernel-patches.sh            # fetches v3 (latest)
./scripts/fetch-kernel-patches.sh --all      # fetches rfc, v1, v2, v3
./scripts/fetch-kernel-patches.sh --version v2
./scripts/fetch-kernel-patches.sh --list     # print thread URLs
```

Patches are saved under `patches/kernel/<version>/`.

---

## SNIA specification

SDXI 1.0 is a SNIA standard.  Draft and released versions are available
at https://www.snia.org/sdxi.

---

## Emulator ABI notes

These values are fixed by the hardware spec and must match the kernel
driver exactly.  Update this table whenever the driver or emulator
changes them.

| Item | Value | Source |
|------|-------|--------|
| DEVID magic | `0x53445849` ("SDXI") | `src/sdxi_regs.h` |
| VERSION register | `0x0100` | `src/sdxi_regs.h` |
| SDXI_OP_MEMCPY | `0x01` | `src/sdxi_regs.h` |
| SDXI_OP_MEMSET | `0x02` | `src/sdxi_regs.h` |
| SDXI_OP_FENCE | `0x03` | `src/sdxi_regs.h` |
| CQE status OK | `0x00` | `src/sdxi_regs.h` |
| CQE status INVAL | `0x01` | `src/sdxi_regs.h` |
| CQE status FAULT | `0x02` | `src/sdxi_regs.h` |
| SQE size | 64 bytes | `src/sdxi_regs.h` |
| CQE size | 16 bytes | `src/sdxi_regs.h` |
| BAR0 size | 128 KiB | `src/rocm_sdxi_server.c` |
| BAR2 size | 4 MiB (doorbell aperture) | `src/rocm_sdxi_server.c` |
| Doorbell stride | 4096 bytes/queue | `src/sdxi_engine.c` |
| Max queues | 16 | `src/sdxi_regs.h` |

---

## Emulator vs. kernel driver alignment (reviewed 2026-11-03, v3 patches)

The emulator **cannot be driven by the v3 kernel driver** in its current
form.  Gaps span every layer:

| Area | Emulator | Kernel driver v3 |
|------|----------|------------------|
| PCI class | `0x08/0x80/0x00` (Base peripheral) | `0x120100` (Processing Accelerator / SDXI) — driver won't claim device |
| `VERSION` reg | offset 0x004 | `SDXI_MMIO_VERSION` at 0x210 |
| `CAP` reg | offset 0x008 | `SDXI_MMIO_CAP0/1` at 0x200/0x208 |
| `CTRL` reg | offset 0x014 | `SDXI_MMIO_CTL0` at 0x000 |
| `STATUS` reg | offset 0x010 | `SDXI_MMIO_STS0` at 0x100 |
| Queue access | Indirect via Q_SELECT (0x020–0x054) | Context table in guest memory at 0x10000 |
| Descriptor | 64 bytes; opcode MEMCPY=1/MEMSET=2/FENCE=3 | 64 bytes; type+subtype (DMAB/ADMIN; COPY/NOP/SYNC) — incompatible encoding |
| Completion | `struct sdxi_cqe` — 16 bytes | `struct sdxi_cst_blk` — 32 bytes, different layout |
| Architecture | Flat register-visible queue model | Two-level context table (L1: 128 ctx, L2: 512 entries each) |
| Context lifecycle | Queue enable via Q_CTRL register write | CXT_START / CXT_STOP admin descriptors; admin context required first |
| Access keys | None | Per-context `struct sdxi_akey_ent` (16 bytes) |

To achieve kernel driver interop the emulator needs: new PCI class code,
new register map, 32-byte CSBs, L1/L2 context table support, and an
admin context boot sequence.  This is effectively a rewrite of
`src/sdxi_regs.h`, `src/rocm_sdxi_server.c`, and `src/sdxi_engine.c`.

### In-VM probe experiment (2026-11-03)

A minimal out-of-tree probe module (`sdxi_probe.ko`) was built against the
Ubuntu 24.04 6.8 kernel and loaded into a QEMU VM running the emulator.
After updating the emulator PCI class from `0x088000` to `0x120100`
(`SDXI_PCI_CLASS_BASE=0x12`, `SDXI_PCI_CLASS_SUB=0x01`), the kernel
correctly claimed the device (confirmed via lspci: "SNIA Smart Data
Accelerator Interface (SDXI) controller").

The probe module read the five registers the v3 driver interrogates at
probe time.  All returned `0xffffffffffffffff` — the vfio-user default
for unmapped BAR offsets — confirming the register-map gap:

| Register | Expected offset | Emulator response |
|----------|-----------------|-------------------|
| CTL0     | 0x000           | 0xffffffffffffffff |
| STS0     | 0x100           | 0xffffffffffffffff |
| CAP0     | 0x200           | 0xffffffffffffffff |
| CAP1     | 0x208           | 0xffffffffffffffff |
| VERSION  | 0x210           | 0xffffffffffffffff |

Implementing those five offsets (and the context-table plumbing they gate)
is the first concrete step toward driver interop.

---

## Test suite

Python tests under `tests/py/` are wired into CTest:

```sh
cd build && ctest --output-on-failure
```

Two suites:
- `sdxi-protocol` — protocol-level BAR/register/DMA-map checks
- `sdxi-dma` — DMA engine functional tests (memcpy, memset, fence,
  zero-length, boundary-crossing, multi-queue)

The Python driver (`tests/py/sdxi/`) is modelled on rocm-ernic's
`tests/py/ernic/` and uses only the standard library.

---

## Compression algorithms

The emulator supports three lossless compression algorithms selectable
per-operation.  Lossy algorithms are explicitly out of scope — SDXI is a
data-movement/storage engine where data integrity is mandatory.

| Constant (UAPI) | Value | Library | Notes |
|-----------------|-------|---------|-------|
| `SDXI_ALGO_ZLIB` | 0 | zlib (`-lz`) | Default; deflate stream |
| `SDXI_ALGO_LZ4`  | 1 | liblz4 (`-llz4`) | Fast compression; level ignored (`LZ4_compress_default`) |
| `SDXI_ALGO_ZSTD` | 2 | libzstd (`-lzstd`) | High ratio; level 0 = `ZSTD_CLEVEL_DEFAULT` |

### Descriptor encoding

The 32-bit flags field at descriptor offset 12 (`SDXI_COMP_FLAGS_OFF`) encodes:

- bits[7:0] — compression level (0 = algorithm default, 1–9 = explicit)
- bits[15:8] — algorithm (`SDXI_COMP_ALGO_*` from `src/sdxi_regs.h`)

Zero in bits[15:8] selects ZLIB for backward compatibility.

### CLI usage

```sh
sdxi-cli -a zlib  compress in.txt out.zlib
sdxi-cli -a lz4   roundtrip in.txt
sdxi-cli -a zstd  bench compress 1048576 100
sdxi-cli          stress         # iterates all three algos internally
```

### Kernel crypto subsystem

The SDXI kernel driver (`drivers/dma/sdxi/`) registers as a **dmaengine**
provider, NOT a `crypto_engine` provider.  These are entirely separate
kernel frameworks.

Integrating with the kernel crypto subsystem (so user code can open
`/dev/crypto` or use `AF_ALG` with an SDXI-backed compressor) requires a
*separate* driver, e.g. `drivers/crypto/sdxi/sdxi_acomp.c`, that calls
`crypto_register_acomp()` / `crypto_register_scomp()`.  This would be a
distinct patch series from the dmaengine patches and cannot be part of
`drivers/dma/sdxi/`.
