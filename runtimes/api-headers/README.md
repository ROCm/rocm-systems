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
  `hrx-system@4aa34130de44c45d68a48575cebfd0ff0610c461`;
- `hsa`: `projects/rocr-runtime/runtime/hsa-runtime/inc` in this repository;
- DRM: `projects/rocr-runtime/libhsakmt/include/hsakmt/drm` in this repository;
- KFD ioctl and UDMABUF: `projects/rocr-runtime/libhsakmt/include/hsakmt/linux`
  in this repository.
- KFD sysfs: Linux `include/uapi/linux/kfd_sysfs.h` at
  [`e83f63da2ac776fbc30861e4ce8b798df6ee8a7a`](https://github.com/torvalds/linux/blob/e83f63da2ac776fbc30861e4ce8b798df6ee8a7a/include/uapi/linux/kfd_sysfs.h).

The AMDF headers preserve upstream contents except that their copyright line
names Advanced Micro Devices, Inc. and their SPDX identifier is MIT. The HSA,
DRM, KFD ioctl, and UDMABUF mirrors are byte-identical to their primary files
in this checkout, except that `kfd_ioctl.h` includes DRM from its shared
`uapi/linux/drm` location rather than the libhsakmt source-tree path.

`kfd_sysfs.h` contains KFD topology capability definitions used by rocjitsu.
It is copied verbatim from the Linux revision above; refresh it from that
upstream file so the per-queue reset and CAP2 definitions remain in sync.

`abce` is not a mirror: `include/abce` is its primary location.
