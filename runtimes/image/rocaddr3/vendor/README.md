<!-- Copyright (c) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->

# AddrLib source record

`addrlib/` contains the 35 files copied from
`projects/rocr-runtime/runtime/hsa-runtime/image/addrlib/` at rocm-systems
revision `c01a4d15d500a045adbb1af5d70c575203250162`. Every file matches that
revision byte for byte except for the local patch recorded below. Individual
source files retain their license notices. `SHA256SUMS` records the vendored
bytes, including that patch.

From this directory, verify the copy with:

```sh
sha256sum -c SHA256SUMS
```

Update the entire source set from a selected revision, record the revision and
any intentional patches here, and regenerate the checksum file. Run the layout
and mapping tests and compare affected outputs before changing the default
source. Keep dependency updates separate from wrapper and module refactors.

This revision includes AIRUNTIME-2652 (#15261), which corrects GFX12 linear
row and slice pitches. It changes `inc/addrinterface.h`,
`src/core/addrlib3.cpp`, and `src/gfx12/gfx12addrlib.cpp` relative to the
previous pin `cec96ccea9d97113c542f4fd33f89c00a3a11d9e`; the other 32 files
are unchanged. Linear rows permit 128-byte alignment, while slice allocation
still respects 256-byte blocks. The layout and mapping tests cover these
distinct pitches, including explicit 128-byte and 384-byte rows. Host tests
establish calculated layouts and offsets, not GPU execution.

Local patch: `src/gfx12/gfx12addrlib.cpp` accumulates the sanity check's
mip-chain block count in 64 bits so its conversion to bytes does not wrap at
4 GiB. Both the trimmed-linear and exact-size assertions remain enabled.
CPU layout tests cover linear and tiled 2-GiB controls, 4-GiB slices, and
5-GiB mip chains, each with one and two array slices. They allocate no image
storage and do not establish native GPU behavior.
