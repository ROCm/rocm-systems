<!-- Copyright (c) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->

# Runtime API Headers

This directory contains static, dependency-free API headers intended for both
C++ consumers and Rust API bindings.

The headers are organized by API family under `include/`:

- `abce`: Accelerated Blit Copy Engine, the header-only SDMA copy library; see
  `include/abce/README.md`.
- `amdf`: AMD Fabric API headers.
- `hsa`: Heterogeneous System Architecture API headers.
- `uapi`: Linux userspace API headers used by ROCm runtimes.

## Source authority

These files are mirrors for runtime consumers; their primary sources remain:

- `amdf`: based on `hrx-system/libamdf/include/amdf` at
  `hrx-system@8cd8e8b913c3997f6c63515876edf265dbd4a854`;
- `hsa`: `projects/rocr-runtime/runtime/hsa-runtime/inc` in this repository;
- DRM: `projects/rocr-runtime/libhsakmt/include/hsakmt/drm` in this repository;
- KFD ioctl and UDMABUF: `projects/rocr-runtime/libhsakmt/include/hsakmt/linux`
  in this repository.
- KFD sysfs: Linux `include/uapi/linux/kfd_sysfs.h` at
  [`e83f63da2ac776fbc30861e4ce8b798df6ee8a7a`](https://github.com/torvalds/linux/blob/e83f63da2ac776fbc30861e4ce8b798df6ee8a7a/include/uapi/linux/kfd_sysfs.h).

The AMDF headers carry local pre-beta API changes, including removal of the
instance host allocator callback, in addition to their AMD copyright and MIT
SPDX preamble. The Rust `ddi/libamdf` provider implements ABI v11, GPU
extension v3, and observation extension v1. It does not advertise the XDNA
extension. The HSA, DRM, and UDMABUF mirrors are byte-identical to their
primary files in this checkout. `kfd_ioctl.h` differs only in its DRM include
path, which points to the shared `uapi/linux/drm` location.

`kfd_sysfs.h` contains KFD topology capability definitions used by rocjitsu.
It is copied verbatim from the Linux revision above; refresh it from that
upstream file so the per-queue reset and CAP2 definitions remain in sync.

The HSA headers and Rust `ddi/libhsa` provider expose AMD extension revision
1.36. Signal batches are supported in fine-grained system pools and in
qualified host-visible GPU pools. PCIe-backed GPU signal stores use a
readback to flush posted writes. Async copy and SVM completion signals require
host atomic support, so PCIe-backed batch members cannot serve as completions.
Query `HSA_AMD_MEMORY_POOL_INFO_SIGNAL_CAPABLE` for each pool and
`HSA_AMD_AGENT_INFO_SIGNAL_HINTS_SUPPORTED` before using completion hints.
The provider currently reports signal hints unsupported because it does not
probe the CP firmware's queue capability bit.

`abce` is not a mirror: `include/abce` is its primary location.
