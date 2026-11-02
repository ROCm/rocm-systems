<!-- Copyright (c) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->

# rocaddr3

`rocaddr3` computes GFX12 image layouts from the device's GB_ADDR_CONFIG,
surface dimensions, format size, and mip count. It wraps AMD AddrLib3 behind a
small Rust interface. Its row-mapping operation returns checked byte offsets
for individual mip levels and array or depth slices. The included AddrLib
source is copied from
`projects/rocr-runtime/runtime/hsa-runtime/image/addrlib` in this repository;
its source files retain their AMD MIT license notices.

The crate owns no GPU allocation and needs no KFD or HSA handles. Callers must
obtain the address configuration from the GPU that will use the surface and
retain the returned layout when constructing resource descriptors and image
transfer addresses.
