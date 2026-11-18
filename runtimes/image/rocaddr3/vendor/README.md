<!-- Copyright (c) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->

# AddrLib source record

`addrlib/` contains the 35 files copied from
`projects/rocr-runtime/runtime/hsa-runtime/image/addrlib/` at rocm-systems
revision `cec96ccea9d97113c542f4fd33f89c00a3a11d9e`. Every file matches that
revision byte for byte; there are no local patches. Individual source files
retain their license notices. `SHA256SUMS` records the vendored bytes.

From this directory, verify the copy with:

```sh
sha256sum -c SHA256SUMS
```

Update the entire source set from a selected revision, record the revision and
any intentional patches here, and regenerate the checksum file. Run the layout
and mapping tests and compare affected outputs before changing the default
source. Keep dependency updates separate from wrapper and module refactors.

At the implementation baseline `3003d5f1a39d343ee950e29e99a0f2b7090c2fc8`,
the current ROCr tree differs in `inc/addrinterface.h`, `src/core/addrlib3.cpp`,
and `src/gfx12/gfx12addrlib.cpp`. These differences include `pitchForSlice`
fields and linear pitch/slice calculations. They are not applied by this
refactor. An explicitly selected source tree compiles its own headers and
implementation together with the wrapper.

The host comparison against that baseline's ROCr source builds successfully but
fails `gfx12_linear_pitches_respect_row_and_slice_alignment`: a 96-element,
32-bit linear row reports a 512-byte slice where the pinned implementation
reports 384 bytes. Four other generation/layout tests pass. The newer source
has therefore not been adopted or qualified as an equivalent default. This
record describes host layout results, not GPU execution.
