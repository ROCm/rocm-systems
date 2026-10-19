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
- KFD and UDMABUF: `projects/rocr-runtime/libhsakmt/include/hsakmt/linux` in
  this repository.

The AMDF headers carry local pre-beta API changes, including removal of the
instance host allocator callback, in addition to their AMD copyright and MIT
SPDX preamble. The Rust `ddi/libamdf` provider implements ABI v11, GPU
extension v3, and observation extension v1. It does not advertise the XDNA
extension. The HSA, DRM, KFD, and UDMABUF mirrors are byte-identical to their
primary files in this checkout.

`abce` is not a mirror: `include/abce` is its primary location.
