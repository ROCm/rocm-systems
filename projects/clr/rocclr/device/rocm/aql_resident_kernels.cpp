// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "aql_resident_kernels.hpp"

namespace amd::roc::aql_resident {
const char* kernelSource() {
  // Apply the 16-byte BindingFixup relocation contract. Bounds, overlap and
  // arithmetic validation happen before submission, not independently per lane.
  // Ordering before this kernel and dirty-target publication after it are
  // submission-layer responsibilities; a workgroup barrier cannot replace them.
  return R"CLC(
typedef struct {
  uint target_offset;
  uint binding_slot;
  ulong binding_offset;
} AqlResidentBindingFixup;

__kernel void __amd_rocclr_resident_binding_fixup(
    __global const ulong* restrict bindings,
    __global const AqlResidentBindingFixup* restrict entries,
    __global uchar* restrict target_base,
    uint entry_count) {
  size_t index = get_global_id(0);
  if (index >= entry_count) return;
  AqlResidentBindingFixup entry = entries[index];
  *(__global ulong*)(target_base + entry.target_offset) =
      bindings[entry.binding_slot] + entry.binding_offset;
}
)CLC";
}
}  // namespace amd::roc::aql_resident
