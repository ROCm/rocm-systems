// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

/// Private-memory probe: a kernel whose result can only be right if its scratch
/// slot was really backed.
///
/// The local array is indexed dynamically, so the compiler cannot promote it to
/// registers and has to place it in private (scratch) memory. That gives the
/// kernel descriptor a nonzero private_segment_fixed_size, which is what drives
/// the command processor down the scratch-allocation path. A wave's slot within
/// the pool is indexed by its grid-wide workgroup id, so a pool sized or mapped
/// wrongly shows up as a wrong result here rather than as a silent no-op.
///
/// The stride and workgroup size are passed as ordinary kernargs: these tests
/// dispatch the code object through raw AQL rather than the HIP runtime, and a
/// raw dispatch does not materialize every HIP implicit field.
extern "C" __global__ void scratch_spill_probe(unsigned *__restrict__ out, unsigned n,
                                               unsigned workgroup_size) {
  constexpr unsigned kSlots = 64;
  unsigned scratch[kSlots];

  const unsigned gid = blockIdx.x * workgroup_size + threadIdx.x;

  for (unsigned i = 0; i < kSlots; ++i)
    scratch[i] = gid * kSlots + i;

  // Dynamic index, so the array cannot live in registers. Reading back in a
  // rotated order also means a slot that was clobbered by a neighbouring wave
  // changes the sum rather than cancelling out.
  unsigned acc = 0;
  for (unsigned i = 0; i < kSlots; ++i)
    acc += scratch[(gid + i) % kSlots];

  if (gid < n)
    out[gid] = acc;
}
