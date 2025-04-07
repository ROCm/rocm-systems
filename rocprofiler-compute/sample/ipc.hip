/*
##############################################################################bl
# MIT License
#
# Copyright (c) 2021 - 2023 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################el



An example to explore IPC and divergence, written by Nicholas Curtis [AMD].
This example may not work on all CDNA accelerators, but has been verified on MI2XX.
*/

#include "common.h"

template<int N=1000>
__device__ void vrcp_op() {
    int dummy;
    if constexpr (N >= 1) {
        asm volatile("v_rcp_f64 v[0:1], v[0:1]\n" : : "{v31}"(dummy));
        vrcp_op<N - 1>();
    }
}

template<int N=1000>
__global__ void vrcp() {
    vrcp_op<N>();
}

template<int N=1000>
__device__ void vmov_op() {
    int dummy;
    if constexpr (N >= 1) {
        asm volatile("v_mov_b32 v0, v1\n" : : "{v31}"(dummy));
        vmov_op<N - 1>();
    }
}

template<int N=1000>
__global__ void vmov() {
    vmov_op<N>();
}

template<int N=1000>
__device__ void mfma_op() {
    int dummy;
    if constexpr (N >= 1) {
        asm volatile("v_mfma_f32_32x32x8bf16_1k v[0:15], v[16:17], v[18:19], v[0:15]\n" : : "{v31}"(dummy));
        mfma_op<N - 1>();
    }
}

template<int N=1000>
__global__ void mfma() {
    mfma_op<N>();
}

template<int N=1000>
__device__ void snop_op() {
    int dummy;
    if constexpr (N >= 1) {
        asm volatile("s_nop 0x0\n" : : "{v31}"(dummy));
        snop_op<N - 1>();
    }
}


template<int N=1000>
__global__ void snop() {
    snop_op<N>();
}

template<int N=1000>
__device__ void smov_op() {
    int dummy;
    if constexpr (N >= 1) {
        asm volatile("s_mov_b32 s0, s1\n" : : "{s31}"(dummy));
        smov_op<N - 1>();
    }
}

template<int N=1000>
__global__ void smov() {
    smov_op<N>();
}

template<int N=1000>
__global__ void vmov_with_divergence() {
    if (threadIdx.x % 64 == 0)
        vmov_op<N>();
}

int main() {
    // warmups, spam to all CUs
    vrcp<<<1024 * 1024, 1024>>>();
    vmov<<<1024 * 1024, 1024>>>();
    mfma<<<1024 * 1024, 1024>>>();
    snop<<<1024 * 1024, 1024>>>();
    smov<<<1024 * 1024, 1024>>>();
    vmov_with_divergence<<<1024 * 1024, 1024>>>();
    hipCheck(hipDeviceSynchronize());
    vrcp<<<1024 * 1024, 1024>>>();
    vmov<<<1024 * 1024, 1024>>>();
    mfma<<<1024 * 1024, 1024>>>();
    snop<<<1024 * 1024, 1024>>>();
    smov<<<1024 * 1024, 1024>>>();
    vmov_with_divergence<<<1024 * 1024, 1024>>>();
    hipCheck(hipDeviceSynchronize());
}
