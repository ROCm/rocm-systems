# Runtime API Headers

This directory contains static, dependency-free API headers intended for both
C++ consumers and Rust API bindings.

The headers are organized by API family under `include/`:

- `amdf`: AMD Framework API headers.
- `hsa`: Heterogeneous System Architecture API headers.
- `uapi`: Linux userspace API headers used by ROCm runtimes.

## Source authority

These files are mirrors for runtime consumers; their primary sources remain:

- `amdf`: `hrx-system/libamdf/include/amdf`, synchronized from
  `hrx-system@4aa34130de44c45d68a48575cebfd0ff0610c461`;
- `hsa`: `projects/rocr-runtime/runtime/hsa-runtime/inc` in this repository;
- DRM: `projects/rocr-runtime/libhsakmt/include/hsakmt/drm` in this repository;
- KFD and UDMABUF: `projects/rocr-runtime/libhsakmt/include/hsakmt/linux` in
  this repository.

Changes belong in the primary location first and are then copied here without
local edits. The HSA, DRM, KFD, and UDMABUF mirrors are byte-identical to their
primary files in this checkout.

These headers are not yet exposed through a CMake target. As runtime components
begin consuming them from this repository, build integration will be added to
export the target and install the headers.
