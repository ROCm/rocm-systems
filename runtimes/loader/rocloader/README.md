<!-- Copyright (c) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->

# rocloader

`rocloader` is the Rust AMD HSA code-object and executable loading library. It
uses `rocelf` for bounded ELF parsing and image construction and `rocddi` for
device allocations. It has no dependency on the HSA C frontend or Linux KFD
implementation details.

The library validates gfx12 target compatibility, interprets HSA code-object
symbols, relocates an image into device-visible storage, and retains the image
and any gfx1250 entry trampolines. Its executable owns loaded objects, scoped
symbols and their resolved addresses, publication order, and freeze state.
Loading stages allocations and symbols before committing them to an executable;
a failed commit leaves the existing executable unchanged.

The frontend chooses the target device and memory kind, supplies external
symbol addresses and opaque IDs, and retains its own ABI metadata such as HSA
handles, source URIs, profiles, and status codes. `libhsa` is currently that
frontend. Additional device families can use separate code-object rules under
`loader/` while sharing `rocelf` where the ELF format applies.
