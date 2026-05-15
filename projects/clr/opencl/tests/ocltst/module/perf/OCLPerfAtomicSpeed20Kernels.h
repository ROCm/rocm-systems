/* Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

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
