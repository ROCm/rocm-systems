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

- `amdf`: `hrx-system/libamdf/include/amdf`, synchronized from
  `hrx-system@bd24215e6a5d1e12570c892356fd5b343ac38a6d`;
- `hsa`: `projects/rocr-runtime/runtime/hsa-runtime/inc` in this repository;
- DRM: `projects/rocr-runtime/libhsakmt/include/hsakmt/drm` in this repository;
- KFD and UDMABUF: `projects/rocr-runtime/libhsakmt/include/hsakmt/linux` in
  this repository.

The AMDF headers preserve upstream contents except that their copyright line
names Advanced Micro Devices, Inc. and their SPDX identifier is MIT. The HSA, DRM,
KFD, and UDMABUF mirrors are byte-identical to their primary files in this
checkout.

`abce` is not a mirror: `include/abce` is its primary location.
