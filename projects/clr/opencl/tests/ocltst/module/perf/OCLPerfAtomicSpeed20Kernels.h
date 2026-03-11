/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

static const char* global_atomics_sum_reduction_all_to_zero =
    "#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable\n"
    " __kernel void global_atomics_sum_reduction_all_to_zero(uint "
    "ItemsPerThread, __global uint *Input, __global atomic_int *Output )\n"
    "{\n"
    "    uint sum = 0;\n"
    "    const uint msk =  (uint)3;\n"
    "    const uint shft = (uint)8;\n"
    "    \n"
    "    uint tid = get_global_id(0);\n"
    "    uint Stride  = get_global_size(0);\n"
    "    for( int i = 0; i < ItemsPerThread; i++)\n"
    "    {\n"
    "       uint data = Input[tid];\n"
    "       sum += data & msk;\n"
    "       data = data >> shft;"
    "       sum += data & msk;\n"
    "       data = data >> shft;"
    "       sum += data & msk;\n"
    "       data = data >> shft;"
    "       sum += data & msk;\n"
    "       tid += Stride;\n"
    "    }\n"
    "    atomic_fetch_add_explicit( &(Output[0]), sum, memory_order_relaxed, "
    "memory_scope_device);\n"
    "}\n";

static const char* global_atomics_sum_reduction_workgroup =
    "#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable\n"
    " __kernel void global_atomics_sum_reduction_workgroup(uint "
    "ItemsPerThread, __global uint *Input, __global atomic_int *Output )\n"
    "{\n"
    "    uint sum = 0;\n"
    "    const uint msk =  (uint)3;\n"
    "    const uint shft = (uint)8;\n"
    "    \n"
    "    uint tid = get_global_id(0);\n"
    "    uint Stride  = get_global_size(0);\n"
    "    for( int i = 0; i < ItemsPerThread; i++)\n"
    "    {\n"
    "       uint data = Input[tid];\n"
    "       sum += data & msk;\n"
    "       data = data >> shft;"
    "       sum += data & msk;\n"
    "       data = data >> shft;"
    "       sum += data & msk;\n"
    "       data = data >> shft;"
    "       sum += data & msk;\n"
    "       tid += Stride;\n"
    "    }\n"
    "    atomic_fetch_add_explicit( &(Output[get_group_id(0)]), sum, "
    "memory_order_relaxed, memory_scope_device);\n"
    "}\n";
