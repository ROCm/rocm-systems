// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

/*
An example to explore IPC and divergence, written by Nicholas Curtis [AMD].
This example may not work on all CDNA accelerators, but has been verified on MI2XX.
*/

#include "common.h"

// Define MFMA support for CDNA2+ architectures (MI200, MI300, MI350 series)
#if defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__)
#define HAS_MFMA_SUPPORT 1
#endif

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

#ifdef HAS_MFMA_SUPPORT
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
#endif

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
#ifdef HAS_MFMA_SUPPORT
    mfma<<<1024 * 1024, 1024>>>();
#endif
    snop<<<1024 * 1024, 1024>>>();
    smov<<<1024 * 1024, 1024>>>();
    vmov_with_divergence<<<1024 * 1024, 1024>>>();
    hipCheck(hipDeviceSynchronize());
    vrcp<<<1024 * 1024, 1024>>>();
    vmov<<<1024 * 1024, 1024>>>();
#ifdef HAS_MFMA_SUPPORT
    mfma<<<1024 * 1024, 1024>>>();
#endif
    snop<<<1024 * 1024, 1024>>>();
    smov<<<1024 * 1024, 1024>>>();
    vmov_with_divergence<<<1024 * 1024, 1024>>>();
    hipCheck(hipDeviceSynchronize());
}
