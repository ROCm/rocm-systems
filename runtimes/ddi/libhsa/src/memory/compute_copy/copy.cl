// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Byte accesses preserve arbitrary pointer alignment. A bounded grid strides
// across large transfers without truncating the public size_t byte extent.
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void copy_bytes(__global uchar* destination,
                         __global const uchar* source, ulong size, uint stride) {
  ulong first = (ulong)__builtin_amdgcn_workgroup_id_x() * 64
      + __builtin_amdgcn_workitem_id_x();
  for (ulong index = first; index < size; index += stride)
    destination[index] = source[index];
}

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void fill_words(__global uint* destination, uint value, ulong count, uint stride) {
  ulong first = (ulong)__builtin_amdgcn_workgroup_id_x() * 64
      + __builtin_amdgcn_workitem_id_x();
  for (ulong index = first; index < count; index += stride)
    destination[index] = value;
}
