/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2018, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
__kernel void test_atomic_add(volatile __global int *sysMemory,
                              volatile __global int *gpuMemory,
                             __global int *oldValues, int value) {
    int  tid = get_global_id(0);
    oldValues[tid] = atomic_add(&sysMemory[tid], value);
    atomic_add(&gpuMemory[tid], value);
}

__kernel void test_atomic_sub(volatile __global int *sysMemory,
                              volatile __global int *gpuMemory,
                             __global int *oldValues, int value) {
    int  tid = get_global_id(0);
    oldValues[tid] = atomic_sub(&sysMemory[tid], value);
    atomic_sub(&gpuMemory[tid], value);
}

__kernel void test_atomic_and(volatile __global int *sysMemory,
                              volatile __global int *gpuMemory,
                             __global int *oldValues, int value) {
    int  tid = get_global_id(0);
    oldValues[tid] = atomic_and(&sysMemory[tid], value);
    atomic_and(&gpuMemory[tid], value);
}

__kernel void test_atomic_or(volatile __global int *sysMemory,
                              volatile __global int *gpuMemory,
                             __global int *oldValues, int value) {
    int  tid = get_global_id(0);
    oldValues[tid] = atomic_or(&sysMemory[tid], value);
    atomic_or(&gpuMemory[tid], value);
}

__kernel void test_atomic_xor(volatile __global int *sysMemory,
                              volatile __global int *gpuMemory,
                             __global int *oldValues, int value) {
    int  tid = get_global_id(0);
    oldValues[tid] = atomic_xor(&sysMemory[tid], value);
    atomic_xor(&gpuMemory[tid], value);
}

__kernel void test_atomic_xchg(volatile __global int *sysMemory,
                              volatile __global int *gpuMemory,
                             __global int *oldValues, int value) {
    int  tid = get_global_id(0);
    oldValues[tid] = atomic_xchg(&sysMemory[tid], value);
    atomic_xchg(&gpuMemory[tid], value);
}

__kernel void test_atomic_inc(volatile __global int *sysMemory,
                              volatile __global int *gpuMemory,
                             __global int *oldValues) {
    int  tid = get_global_id(0);
    oldValues[tid] = atomic_inc(&sysMemory[tid]);
    atomic_inc(&sysMemory[tid]);
    atomic_inc(&sysMemory[tid]);
    atomic_inc(&sysMemory[tid]);

    atomic_inc(&gpuMemory[tid]);
    atomic_inc(&gpuMemory[tid]);
    atomic_inc(&gpuMemory[tid]);
    atomic_inc(&gpuMemory[tid]);
}

__kernel void test_atomic_dec(volatile __global int *sysMemory,
                              volatile __global int *gpuMemory,
                             __global int *oldValues) {
    int  tid = get_global_id(0);
    oldValues[tid] = atomic_dec(&sysMemory[tid]);
    atomic_dec(&sysMemory[tid]);
    atomic_dec(&sysMemory[tid]);
    atomic_dec(&sysMemory[tid]);

    atomic_dec(&gpuMemory[tid]);
    atomic_dec(&gpuMemory[tid]);
    atomic_dec(&gpuMemory[tid]);
    atomic_dec(&gpuMemory[tid]);
}

__kernel void test_atomic_max(volatile __global int *sysMemory,
                              volatile __global int *gpuMemory,
                             __global int *oldValues, int value) {
    int  tid = get_global_id(0);
    oldValues[tid] = atomic_max(&sysMemory[tid], value);
    atomic_max(&gpuMemory[tid], value);
}

__kernel void test_atomic_min(volatile __global int *sysMemory,
                              volatile __global int *gpuMemory,
                             __global int *oldValues, int value) {
    int  tid = get_global_id(0);
    oldValues[tid] = atomic_min(&sysMemory[tid], value);
    atomic_min(&gpuMemory[tid], value);
}


