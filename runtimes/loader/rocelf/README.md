<!-- Copyright (c) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->

# rocelf

`rocelf` is a standalone Rust library for checked ELF64 little-endian parsing
and AMDGPU load-image construction. It has no dependency on rocddi, libhsa, an
operating system driver, or a C ABI. Its inputs are caller-owned ELF bytes and a
caller-owned destination image.

The library validates the ELF header and table bounds, extracts AMDGPU symbol
records, plans the aligned load extent, clears and copies load segments, and
applies supported AMDGPU RELA relocations. It reports malformed objects and
unresolved external symbols as Rust errors. The caller
chooses the accepted machine, OS ABI, and target; allocates the destination;
resolves external symbols; and publishes the completed image.

For gfx1250 code objects, it also plans and writes kernel entry trampolines
when the object lacks the required prefetch prologue. The caller supplies the
device addresses and executable backing for those stubs.

`rocloader` uses rocelf to interpret AMD HSA code objects and construct
device-resident images. `libhsa` retains the HSA C ABI and opaque handle
registry.
