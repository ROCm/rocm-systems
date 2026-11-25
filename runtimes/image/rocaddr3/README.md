<!-- Copyright (c) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->

# rocaddr3

`rocaddr3` computes GFX9–GFX12 image layouts from the device's GB_ADDR_CONFIG,
family/revision, surface dimensions, format size, and mip count. It wraps AMD
AddrLib2 and AddrLib3 behind a small Rust interface. Its row-mapping operation
returns checked byte offsets for individual mip levels and array or depth slices.
The default source is a pinned copy of
`projects/rocr-runtime/runtime/hsa-runtime/image/addrlib` in this repository;
its source files retain their AMD MIT license notices. The
[source record](vendor/README.md) identifies its revision and checksums.

The crate owns no GPU allocation and needs no KFD or HSA handles. Callers must
obtain the address configuration from the GPU that will use the surface and
retain the returned layout when constructing resource descriptors and image
transfer addresses.

The crate name is retained for existing workspace paths; both AddrLib2 and
AddrLib3 calculations are available. `SurfaceMapper` currently supports GFX12
coordinate mapping only. The wrapper owns the native layout handle, validates
requests, and translates errors; the selected source tree supplies the hardware
layout algorithms.

## Source selection

Standalone Cargo builds use `vendor/addrlib` by default. To share an explicitly
selected source tree, set `ROCADDR_SOURCE_DIR` to its directory before running
Cargo. Relative paths are anchored to this crate's directory. For example, from
the repository root:

```sh
ROCADDR_SOURCE_DIR="$PWD/projects/rocr-runtime/runtime/hsa-runtime/image/addrlib" \
  cargo test --manifest-path runtimes/Cargo.toml -p rocaddr3 --frozen
```

CMake accepts `ROCM_RUNTIMES_ADDRLIB_SOURCE_DIR`. An empty value selects the
pinned copy; a nonempty value is resolved once during configuration and passed
explicitly to every Cargo command. The CMake option controls source selection
even when an unrelated `ROCADDR_SOURCE_DIR` is set in the invoking environment.

An explicit tree must contain the complete expected AddrLib source and headers.
The build compiles that tree and this crate's bridge together; it neither links
an installed HSA runtime nor modifies the selected source. Missing files fail
the build without falling back to another revision. Changes to the selection
or the source tree trigger Cargo's native rebuild.

Selecting another revision is a native dependency update. Compare layout sizes,
pitches, mip offsets, and coordinate maps, then qualify affected device transfers
before adopting it as the default. The pinned copy includes the GFX12 linear
row/slice pitch correction from AIRUNTIME-2652 (#15261).
