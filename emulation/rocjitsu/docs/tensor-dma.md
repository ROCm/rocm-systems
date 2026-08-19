# CDNA5 Tensor DMA model

rocJITsu models the Tensor Data Mover (TDM) used by gfx1250/CDNA5
`tensor_load_to_lds` and `tensor_store_from_lds` instructions. TDM moves a
rectangular tile between global memory and LDS without routing element data
through VGPRs.

The primary architectural reference is the
[AMD CDNA5 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna5-instruction-set-architecture.pdf),
especially sections 10.11.1 through 10.11.6. The descriptor field layout is
also exposed by the ROCm HIP header
`hip/amd_detail/amd_gfx1250_TDM.h`. LLVM's AMDGPU
`make_dma_descriptor` lowering is the reference for descriptors produced from
MLIR.

This document covers descriptor coordinates, bounds, iteration, gather, LDS
padding, and completion. Group 0 transfer enablement, process/VMID address
translation, and cache visibility are separate runtime contracts.

## Descriptor coordinate system

Tensor dimensions are numbered from the contiguous dimension outward.
Dimension zero has implicit element stride one. For an element coordinate
`c`, the global element offset is:

```text
offset(c) =
    c[0]
  + c[1] * global_stride[0]
  + c[2] * global_stride[1]
  + c[3] * global_stride[2]
  + c[4] * global_stride[3]
```

The byte address is:

```text
global_base + element_size * offset(c)
```

The stride fields are consumed literally. A zero stride aliases coordinates;
it is not interpreted as an omitted dense stride. Logical dimension order and
increasing memory-stride order may differ, so a descriptor can represent
permuted or padded layouts.

Normal descriptor rank is inferred from the highest active tile dimension.
Gather mode is architecturally always rank two: `tile_dim0` is the contiguous
row width, the descriptor's repurposed `tile_dim1` field is the number of
gather indices, and `tensor_dim1` is the row bound. A null descriptor group is
equivalent to a group whose SGPR values are zero; it does not remove an active
dimension established by an earlier group.

## Tile traversal and bounds

For a non-gather tile, rocJITsu enumerates tile coordinates with dimension zero
varying fastest. LDS receives or supplies one dense element stream. The global
address uses the descriptor strides while the LDS stream uses the tile
extents.

The active tensor domain is:

```text
0 <= c[dimension] < tensor_extent[dimension]
```

If any active tensor extent is zero, the domain is empty. This follows directly
from the ISA's positive-end bounds rule; zero is not an "unbounded" sentinel.

For elements outside the active tensor domain:

- a tensor load writes zero to the corresponding LDS element; and
- a tensor store suppresses the corresponding global-memory write.

The transfer still completes normally. In particular, it retires TENSORcnt and
performs an enabled completion-barrier arrival even if every element is
masked.

## Descriptor iteration

Iteration is available for rank-two and rank-three normal tensors. For
iteration `i`, the descriptor advances the starting addresses by:

```text
global element offset = i * global_increment
LDS element offset    = i * lds_increment
```

The ISA describes iteration as repeated row-skipping transfers but does not
provide pseudocode for converting an arbitrary linear global increment back
into tensor coordinates for bounds checking. rocJITsu makes that conversion
explicit so that address generation and bounds use the same logical origin.

For every active dimension whose extent is greater than one, rocJITsu
constructs an address-varying axis:

```text
(logical dimension, tensor extent, element stride)
```

Extent-one dimensions remain active for rank and bounds but have only
coordinate zero, so their stride cannot create overlap and does not consume a
mixed-radix digit.

The axes are sorted by increasing stride. A layout supports advancing
iteration when every axis starts at or beyond the complete occupied span of
all faster axes. Starting with an occupied span of one element:

```text
stride[axis] >= occupied_span
occupied_span += stride[axis] * (extent[axis] - 1)
```

Saturating arithmetic is used for validation. This mixed-radix condition
supports ordinary dense layouts, padding between rows or planes, and axis
permutations while guaranteeing a unique greedy inverse.

The iteration origin is decoded in decreasing-stride order:

```text
origin[axis] = remaining / stride[axis]
remaining    = remaining % stride[axis]
```

Bounds are then checked using:

```text
origin[dimension] + tile_coordinate[dimension]
    < tensor_extent[dimension]
```

rocJITsu rejects an advancing, non-empty iterating descriptor when its stride
layout does not satisfy the invertibility condition. No inverse is required
for:

- a single iteration;
- repeated iterations with `global_increment == 0`; or
- an empty tensor domain.

This rejection is a simulator support boundary, not an additional ISA
restriction. Non-iterating descriptors may use overlapping or zero strides
because their global addresses do not need to be inverted.

## Gather and scatter

Gather mode is a two-dimensional row operation. Each index selects a global
row and transfers `tile_dim0` contiguous elements:

```text
global element offset =
    contiguous_coordinate + gather_index * global_stride[0]
```

Rows are packed densely into LDS in index-list order. The same index list
performs the reverse scatter for `tensor_store_from_lds`.

The ISA allows repeated and non-monotonic indices, but only guarantees correct
out-of-bounds behavior when the index list is nondecreasing. rocJITsu currently
applies deterministic per-index bounds checks even for other index orders.

## LDS padding

Padding applies only to memory-to-LDS loads. It inserts periodic gaps in the
dense LDS destination stream; skipped locations retain their previous values.

For LDS-to-memory stores, the ISA specifies no de-padding operation. rocJITsu
therefore ignores `pad_enable`, `pad_interval`, and `pad_amount` and reads the
ordinary dense LDS source stream.

## Completion and ordering

Each tensor instruction increments TENSORcnt once for the complete descriptor,
not once per element or iteration. Tensor operations from one wave complete in
instruction order but are unordered with other memory-instruction classes.

When `atomic_barrier_enable` is set, the LDS barrier arrival occurs after the
complete tensor operation. It is a completion effect and is not conditional on
the number of in-bounds elements.

## Focused validation

The `Gfx1250ExecutionTest.TensorDma*` tests in
`tests/cdna5_tensor_dma_test.cpp` cover:

- dense, padded, iterating, and gather load/store addressing;
- sub-row, row, plane, padded, and permuted iteration origins;
- truly overlapping versus merely permuted stride layouts;
- aliased strides on extent-one dimensions;
- zero extents in every modeled transfer form;
- literal zero gather stride;
- null descriptor groups;
- out-of-bounds load zero-fill and store suppression; and
- TENSORcnt and completion-barrier retirement.
