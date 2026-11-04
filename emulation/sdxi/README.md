# SDXI Emulator

A vfio-user based emulator of the AMD SDXI (SNIA Data Exchange Interface)
PCIe copy-engine device.  It presents a SDXI device to QEMU over a Unix
socket using the vfio-user protocol, without requiring any in-kernel device
model.

This component lives under `emulation/sdxi/` in the
[rocm-systems](https://github.com/ROCm/rocm-systems) monorepo.

## Overview

SDXI is a SNIA-standardised data-mover that will be integrated into AMD CPU
SoCs.  This emulator lets software stacks targeting SDXI be developed and
tested inside a VM before silicon or FPGA targets are available.

The emulator supports:

- SQ/CQ descriptor rings (NVMe-style)
- `MEMCPY` and `MEMSET` DMA operations
- `FENCE` ordering descriptors
- Up to 16 independent queues
- MSI-X interrupts per queue
- BAR0 (64 KiB MMIO registers) and BAR2 (4 MiB doorbell aperture)

## Dependencies

- [libvfio-user](https://github.com/nutanix/libvfio-user) ≥ 0.9
- CMake ≥ 3.25.2
- C11 compiler (GCC or Clang)

Build libvfio-user first:

```sh
git clone https://github.com/nutanix/libvfio-user.git
cd libvfio-user
mkdir build && cd build
cmake ..
make
sudo make install
```

## Build

```sh
mkdir build && cd build
cmake ..
make
```

## Usage

Start the emulator, then attach QEMU:

```sh
# Terminal 1 — start the emulator
./build/rocm-sdxi -s /tmp/sdxi.sock -v

# Terminal 2 — launch QEMU with the emulated device
qemu-system-x86_64 \
    -M pc,accel=kvm \
    -drive file=disk.qcow2 \
    -device vfio-user-pci,socket=/tmp/sdxi.sock
```

Options:

| Flag | Description |
|------|-------------|
| `-s PATH` / `--socket PATH` | Unix socket path (default: `/tmp/vfio-user-rocm-sdxi.sock`) |
| `-v` / `--verbose` | Increase verbosity; repeat for debug (`-vv`) |
| `-V` / `--version` | Print version and exit |
| `-h` / `--help` | Print help |

## Testing

The test suite uses Python and CTest.  Build first, then:

```sh
cd build
ctest --output-on-failure
```

To run a single suite with verbose output:

```sh
ctest -R sdxi-protocol -V
ctest -R sdxi-dma -V
```

Python 3.8+ is required.  No extra packages are needed; the tests use
only the standard library.

## Device registers (BAR0)

| Offset | Name | Access | Description |
|--------|------|--------|-------------|
| 0x000 | DEVID | RO | `0x53445849` ("SDXI") |
| 0x004 | VERSION | RO | `0x00010000` (1.0) |
| 0x008 | CAP | RO | Capability flags |
| 0x00C | NUM\_QUEUES | RO | Number of queues |
| 0x010 | STATUS | RO | Device status |
| 0x014 | CTRL | RW | Device control (bit 0 = enable) |
| 0x020 | Q\_SELECT | RW | Queue index for indirect registers |
| 0x024 | Q\_CTRL | RW | Queue control (bit 0 = enable) |
| 0x028 | Q\_STATUS | RO | Queue status |
| 0x02C | Q\_SIZE | RW | SQ depth |
| 0x030 | Q\_HEAD | RO | SQ head pointer |
| 0x034 | Q\_TAIL | RW | SQ tail pointer |
| 0x038 | Q\_BASE\_LO | RW | SQ GPA low 32 bits |
| 0x03C | Q\_BASE\_HI | RW | SQ GPA high 32 bits |
| 0x040 | CQ\_SIZE | RW | CQ depth |
| 0x044 | CQ\_HEAD | RW | CQ head (software advances) |
| 0x048 | CQ\_TAIL | RO | CQ tail (hardware advances) |
| 0x04C | CQ\_BASE\_LO | RW | CQ GPA low 32 bits |
| 0x050 | CQ\_BASE\_HI | RW | CQ GPA high 32 bits |
| 0x054 | Q\_INTR\_VEC | RW | MSI-X vector for this queue |

Doorbell writes (BAR2) use `queue_id * 4096` byte stride.

## SQ descriptor format (64 bytes)

| Field | Bytes | Description |
|-------|-------|-------------|
| opcode | 4 | 1 = MEMCPY, 2 = MEMSET, 3 = FENCE |
| flags | 4 | Reserved |
| sq\_id | 4 | Source queue ID |
| cid | 4 | Command identifier (echoed in CQE) |
| src\_addr | 8 | Source GPA (MEMCPY) or fill pattern byte (MEMSET) |
| dst\_addr | 8 | Destination GPA |
| length | 8 | Transfer length in bytes |
| reserved | 24 | Must be zero |

## License

MIT — see SPDX headers in source files.
