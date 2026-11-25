<!-- Copyright (c) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->

# rocloader

`rocloader` is the Rust AMD HSA code-object and executable loading library. It
uses a private, safe `elf` module for bounded byte parsing and construction,
and `rocddi` for device allocations. It has no dependency on the HSA C frontend
or Linux KFD implementation details. The former `rocelf` package is consolidated
here; its byte operations remain independent of device allocation and prohibit
unsafe code.

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
this crate while sharing its ELF byte operations where that format applies.

Caller-owned destinations need not have a writable host mapping. Freeze
copies relocated staging through the GPU and publishes instruction-cache
invalidation before marking the executable frozen. Destination lifetime stays
with the caller. Program storage can be allocated with all consuming device
mappings retained by the loaded image.

The implementation is organized by ownership:

- `elf.rs` validates tables, plans load spans and applies relocations to
  caller-owned bytes. It never allocates device storage or publishes code.
- `code_object.rs` interprets immutable HSA code objects and checks target and
  executable compatibility. One input can produce several loaded instances.
- `target/gfx1250.rs` plans and writes the existing kernel-entry fixup without
  owning native memory. Its applicability remains a target/compiler policy;
  the refactor does not establish a new hardware requirement.
- `loaded_image.rs` owns relocated storage, retained source bytes and checked
  publication work. Caller-provided destinations remain borrowed.
- `executable.rs` owns symbol scopes and the mutable, publishing and frozen
  phases. The HSA frontend separately owns public handles and debugger records.

Run `cargo test -p rocloader --frozen` for host-only ELF corruption/relocation,
code-object compatibility, fixup bounds and publication-state tests. These tests
do not qualify native cache publication or execution on every recognized ISA.
