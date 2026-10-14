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
  `hrx-system@bd24215e6a5d1e12570c892356fd5b343ac38a6d`;
- `hsa`: `projects/rocr-runtime/runtime/hsa-runtime/inc` in this repository;
- DRM: `projects/rocr-runtime/libhsakmt/include/hsakmt/drm` in this repository;
- KFD and UDMABUF: `projects/rocr-runtime/libhsakmt/include/hsakmt/linux` in
  this repository.

The AMDF headers carry local pre-beta API changes, including removal of the
instance host allocator callback, in addition to their AMD copyright and MIT
SPDX preamble. The HSA, DRM, KFD, and UDMABUF mirrors are byte-identical to
their primary files in this checkout.

`abce` is not a mirror: `include/abce` is its primary location.

## Standalone public AMDF package

`cmake -S runtimes/api-headers -B build/runtime-api-headers` configures a header-only installation without runtime binaries, Rust or a native compiler. Install it with `cmake --install build/runtime-api-headers --prefix <absolute-prefix>`.

Native consumers use `find_package(ROCmRuntimeAPI CONFIG REQUIRED COMPONENTS AMDF)` and `ROCmRuntimeAPI::AMDF`. This target exposes only the installed AMDF headers and stays valid after relocation. The package includes this source-provenance document. HSA and private/UAPI targets are not published by this initial package; the existing `rocm_runtime::runtime_headers` source-tree umbrella is retained for implementation consumers.

The KFD test snapshot is independently owned under [`testing/drivers/kfd/uapi`](../testing/drivers/kfd/uapi/README.md). Updating these runtime mirrors never changes that test contract automatically.
