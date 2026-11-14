<!-- Copyright (c) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->

# rocloader

`rocloader` is the Rust AMD HSA code-object and executable loading library. It
uses `rocelf` for bounded ELF parsing and image construction and `rocddi` for
device allocations. It has no dependency on the HSA C frontend or Linux KFD
implementation details.

The library validates explicit GFX9, GFX10, GFX11, and GFX12 target rows,
including their generic ISA and XNACK compatibility. It interprets modern HSA
code-object symbols, relocates images into owned device-visible storage or
caller-owned destinations, and retains staging and any gfx1250 entry
trampolines through checked publication. Legacy V1/V2 objects are rejected. Its executable owns loaded objects, scoped
symbols and their resolved addresses, publication order, and freeze state.
Loading stages allocations and symbols before committing them to an executable;
a failed commit leaves the existing executable unchanged.

The frontend chooses the target device and memory kind, supplies external
symbol addresses and opaque IDs, and retains its own ABI metadata such as HSA
handles, source URIs, profiles, and status codes. `libhsa` is currently that
frontend. Additional device families can use separate code-object rules under
`loader/` while sharing `rocelf` where the ELF format applies.

Caller-owned destinations need not have a writable host mapping. Freeze
copies relocated staging through the GPU and publishes instruction-cache
invalidation before marking the executable frozen. Destination lifetime stays
with the caller. Program storage can be allocated with all consuming device
mappings retained by the loaded image.
