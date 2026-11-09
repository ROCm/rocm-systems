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
| v4 | linux-pci | https://github.com/AMDESE/linux-sdxi/tree/sent/20250813-sdxi-base-73d7c9fdce57-v4 |

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

## Emulator vs. kernel driver alignment (updated 2026-11-05, v3 patches)

The emulator is now aligned with the v3 kernel driver register map and
architecture.  All areas that were previously mismatched have been
implemented:

| Area | Status | Notes |
| ---- | ------ | ----- |
| PCI class | aligned | `0x120100` (Processing Accelerator / SDXI) in `src/rocm_sdxi_server.c` |
| Register map | aligned | CTL0=0x000, CTL2=0x010, STS0=0x100, CAP0=0x200, CAP1=0x208, VERSION=0x210 |
| Context table | aligned | CXT_L2 pointer at 0x10000; L1 capped at 128 contexts (`SDXI_MAX_CXT`) |
| Descriptor | aligned | 64-byte `sdxi_dsc`; DMAB/ADMIN type+subtype encoding |
| Completion | aligned | 32-byte `sdxi_cst_blk` (`SDXI_CST_BLK_SIZE`) |
| Context lifecycle | aligned | CXT_START / CXT_STOP admin descriptors; context 0 is the admin context |
| CTL0 FN_GSR | aligned | FN_GSR bits[1:0] mirrored into STS0 FN_GSV bits[2:0] at probe time |

The emulator does not implement per-context access keys (`sdxi_akey_ent`);
the kernel driver does not enforce them in the current v3 patches so this
is not a blocker for driver interop.

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

---

## Compute / elementwise arithmetic

The emulator supports a **COMPUTE descriptor type** (0x008) with one subtype:
**ELEMWISE** (0x01).  This performs element-wise or reductive arithmetic on a
guest buffer without moving data through any host-side copy engine.

### Descriptor layout (64 bytes)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 4 | opcode | `VL(0) \| (SUBTYPE_ELEMWISE << 8) \| (TYPE_COMPUTE << 16)` |
| 4 | 4 | src_len | Source byte count (LE32) |
| 8 | 4 | dtype | `SDXI_DTYPE_*` (LE32) |
| 12 | 4 | op | `SDXI_ELEMOP_*` (LE32) |
| 16 | 8 | src_addr | Source GPA (LE64) |
| 24 | 8 | dst_addr | Destination GPA (LE64) |
| 32 | 8 | scalar | Scalar as dtype-native bits (LE64) |
| 56 | 8 | csb_ptr | NP \| (csb_gpa >> 5) (LE64) |

### Element data types (`SDXI_DTYPE_*`)

| Constant | Value | Elem size | Notes |
|----------|-------|-----------|-------|
| `SDXI_DTYPE_FP32` | 0 | 4 bytes | IEEE 754 single |
| `SDXI_DTYPE_FP16` | 1 | 2 bytes | IEEE 754 half; bias=15 |
| `SDXI_DTYPE_BF16` | 2 | 2 bytes | Top 16 bits of FP32 |
| `SDXI_DTYPE_INT8` | 3 | 1 byte | Signed two's complement |
| `SDXI_DTYPE_INT32` | 4 | 4 bytes | Signed two's complement |
| `SDXI_DTYPE_FP8_E4M3` | 5 | 1 byte | OCP E4M3: bias=7, max=448, NaN=0x7F/0xFF, no Inf |
| `SDXI_DTYPE_FP8_E5M2` | 6 | 1 byte | OCP E5M2: bias=15, has Inf, NaN when e=11111 m≠0 |
| `SDXI_DTYPE_INT4` | 7 | ½ byte | Nibble-packed, low nibble first; n_elems = src_len × 2 |

### Element operations (`SDXI_ELEMOP_*`)

| Constant | Value | Semantics |
|----------|-------|-----------|
| `SDXI_ELEMOP_ADD` | 0 | `dst[i] = src[i] + scalar` |
| `SDXI_ELEMOP_MIN` | 1 | `dst[i] = min(src[i], scalar)` |
| `SDXI_ELEMOP_MAX` | 2 | `dst[i] = max(src[i], scalar)` |
| `SDXI_ELEMOP_REDUCE_MIN` | 3 | `dst[0] = min over all src elements` |
| `SDXI_ELEMOP_REDUCE_MAX` | 4 | `dst[0] = max over all src elements` |

For reduce ops dst is one element (elem_sz bytes); for elementwise ops dst is
src_len bytes (or src_len/2 for INT4 since they pack 2 per byte).

### FP8 OCP conventions

Both E4M3 and E5M2 follow the OCP MX specification:

- **E4M3**: bias=7, max representable=448.  `0x7F` and `0xFF` are NaN.
  No infinities; overflow saturates to max normal (0x7E / 0xFE).
- **E5M2**: bias=15, has ±Inf (`0x7C`/`0xFC`).  NaN when exponent=11111 and
  mantissa≠0.  Overflow to ±Inf.

### INT4 packing

Two signed nibbles per byte, low nibble first.  Range −8..7 (two's complement
in 4 bits).  Element index `i` lives in byte `i/2`; low nibble when i is even,
high nibble when i is odd.

### Design principle (rocm-ernic P4 engine)

All type conversions use `memcpy` for type-punning (never pointer casts to avoid
UB).  Float is the universal intermediate arithmetic type — decode to float, add/
min/max in float, re-encode.  This matches the pattern in
`rocm-ernic/src/p4_fp8.c`.

### CLI usage

```sh
# elementwise: add scalar 5 to each int8 element
sdxi-cli elemwise add int8 5 src.bin dst.bin

# elementwise: clamp fp32 values to max 1.0
sdxi-cli elemwise min fp32 1.0 src.bin dst.bin

# reduce: find minimum fp16 value across the buffer
sdxi-cli elemwise reduce_min fp16 0 src.bin result.bin

# reduce: find maximum fp8 (E4M3) value
sdxi-cli elemwise reduce_max fp8e4m3 0 src.bin result.bin

# INT4 add nibble-packed buffer
sdxi-cli elemwise add int4 2 src_packed.bin dst_packed.bin
```

Scalar is a decimal integer or float for FP types, or a hex value (`0x...`)
for raw bit-pattern overrides (useful for FP16/BF16/FP8 when you know the
encoding).

---

## Performance counters

Each context tracks 14 64-bit counters, indexed by `SDXI_CTR_STAT_*` selector.
The `CTR_SELECT` register (BAR0 0x300) takes `cid | (stat_sel << 8)`; reading
`CTR_VAL` (BAR0 0x308) returns the counter atomically.

The `SDXI_IOC_STATS` ioctl (`_IOWR('X', 6, struct sdxi_ioc_stats)`) returns all
14 counters for a given context in one call.

| Selector | Constant | Description |
|----------|----------|-------------|
| `0x00` | `SDXI_CTR_STAT_BYTES_MOVED` | Total bytes DMA-copied |
| `0x01` | `SDXI_CTR_STAT_NUM_COPY` | COPY descriptors completed |
| `0x02` | `SDXI_CTR_STAT_NUM_ERRORS` | Descriptors completing with error |
| `0x03` | `SDXI_CTR_STAT_NUM_INTR` | Interrupts delivered |
| `0x04` | `SDXI_CTR_STAT_NUM_CXT_START` | CXT_START admin descriptors |
| `0x05` | `SDXI_CTR_STAT_NUM_CXT_STOP` | CXT_STOP admin descriptors |
| `0x06` | `SDXI_CTR_STAT_NUM_NOP` | NOP descriptors completed |
| `0x07` | `SDXI_CTR_STAT_NUM_SYNC` | SYNC descriptors completed |
| `0x08` | `SDXI_CTR_STAT_BYTES_COMPRESSED` | Input bytes to COMPRESS |
| `0x09` | `SDXI_CTR_STAT_BYTES_DECOMPRESSED` | Input bytes to DECOMPRESS |
| `0x0A` | `SDXI_CTR_STAT_NUM_COMPRESS` | COMPRESS descriptors completed |
| `0x0B` | `SDXI_CTR_STAT_NUM_DECOMPRESS` | DECOMPRESS descriptors completed |
| `0x0C` | `SDXI_CTR_STAT_NUM_ELEMWISE` | ELEMWISE descriptors completed |
| `0x0D` | `SDXI_CTR_STAT_BYTES_ELEMWISE` | Source bytes processed by ELEMWISE |

Constants are defined in `src/sdxi_regs.h`.  Python bindings
(`CTR_NUM_ELEMWISE`, `CTR_BYTES_ELEMWISE`, etc.) are in
`tests/py/sdxi/cxt_driver.py`.

### CLI usage

```sh
sdxi-cli info              # print VERSION, PCI IDs, max contexts, CAP0/CAP1
sdxi-cli stats             # print all 14 counters for context 0
sdxi-cli stats 2           # print all 14 counters for context 2
```

---

## dma-buf importer

The v4 kernel driver and the out-of-tree char driver both support importing
dma-bufs so the SDXI device can DMA-copy between GPU memory, host
CMA/system memory, or any other dma-buf exporter (amdgpu, nvidia, udmabuf,
dma-heap) without bouncing through a userspace-mapped buffer.

### Constraint

SDXI copy descriptors carry a single (src, dst) address pair.  Both
dma-bufs must map to a **single sg entry** (`sg_table.nents == 1`) from the
SDXI device's perspective.  Allocations from `/dev/dma_heap/system` and
`/dev/udmabuf` always satisfy this; fragmented GPU BO memory may not.

### v4 upstream driver API

`sdxi_dma_prep_dma_buf_memcpy()` in `drivers/dma/sdxi/dma.c` extends the
standard dmaengine interface:

```c
struct dma_async_tx_descriptor *
sdxi_dma_prep_dma_buf_memcpy(struct dma_chan *chan,
                              struct dma_buf *dst_buf, struct dma_buf *src_buf,
                              size_t len, unsigned long flags);
```

Attachment lifecycle is owned by the `struct sdxi_dma_txd`; both
`src_attach` and `dst_attach` are released in `sdxi_dma_txd_free()` when
the transaction completes or is freed.

New files: `drivers/dma/sdxi/dmabuf.h`, `drivers/dma/sdxi/dmabuf.c`.
Kconfig selects `DMA_SHARED_BUFFER`; Makefile adds `dmabuf.o` to `sdxi-core-y`.

### Out-of-tree char driver ioctl

`SDXI_IOC_DMABUF_COPY` (`_IOWR('X', 7, struct sdxi_ioc_dmabuf_copy)`)
in `emulation/sdxi/drivers/sdxi_char/`:

```c
struct sdxi_ioc_dmabuf_copy {
    __s32 src_fd;       /* open dma-buf fd for source */
    __s32 dst_fd;       /* open dma-buf fd for destination */
    __u64 src_offset;   /* byte offset into source dma-buf */
    __u64 dst_offset;   /* byte offset into destination dma-buf */
    __u64 len;          /* bytes to copy */
};
```

### Python CLI

```sh
# src_fd and dst_fd are open dma-buf file descriptor numbers
sdxi-cli dmabuf-copy <SRC_FD> <DST_FD> <LEN>
sdxi-cli dmabuf-copy 5 6 1048576 --src-offset 4096 --dst-offset 0
```

Python API (`sdxi_cli.ioctl`):

```python
from sdxi_cli.ioctl import sdxi_dmabuf_copy

sdxi_dmabuf_copy(dev_fd, src_dmabuf_fd, dst_dmabuf_fd, length,
                 src_offset=0, dst_offset=0)
```
