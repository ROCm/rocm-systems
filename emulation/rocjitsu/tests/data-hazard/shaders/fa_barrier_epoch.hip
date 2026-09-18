// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "utils.hpp"

// Minimal reproducer for the false-positive LDS race warnings seen in a
// pipelined flash-attention kernel.
//
// In the FA kernel the instruction sequence is:
//
//   [all waves]  tensor_load_to_lds  (write LDS via TDM)
//                s_wait_tensorcnt 0  (drain intra-wave tensor counter)
//                s_wait_dscnt 0      (drain intra-wave DS counter)
//                s_barrier_signal -1 (announce "my writes are done")
//                ... VALU work ...
//                s_barrier_wait -1   (stall until all waves have signaled)
//                ds_load_u16         (read from LDS — data from other waves)
//
// A wave first issues s_barrier_signal to indicate that other waves may
// proceed. Later it issues s_barrier_wait, which stalls until every wave in
// the workgroup has also signalled.
//
// So after s_barrier_wait returns, all waves' prior LDS writes are visible.
// The ds_load_u16 reads are safe — no race.
//
// The plugin was not recognising s_barrier_wait as an epoch boundary, so the
// tensor writes (before the barrier) and ds_load_u16 reads (after it) ended
// up in the same epoch, producing false-positive cross-wave race warnings.
//
// This reproducer uses ds_store_b16 instead of tensor_load_to_lds (avoiding
// the tensor descriptor setup) but exercises the same synchronisation
// pattern and the same ds_load_u16 consumer instruction.
//
// Mutation ground truth:
//   - Baseline (unmodified): 0 hazards
//   - Remove s_wait_dscnt before barrier: intra-wave RAW — wave's own
//     ds_load_u16 may see stale data
//   - Remove __syncthreads (barrier): true cross-wave race — other waves
//     read LDS before the writer has finished
//
// The inline asm below is written in the gfx12 spellings — ds_store_b16,
// ds_load_u16 and s_wait_dscnt — which earlier targets do not assemble, so the
// harness is told to skip this kernel everywhere else:
// requires: gfx1250

static constexpr int WAVE_SIZE    = 32;
static constexpr int WAVES_PER_WG = 4;
static constexpr int BLOCK_SIZE   = WAVE_SIZE * WAVES_PER_WG;  // 128 threads

// Each wave writes WAVE_SIZE fp16 values (64 bytes) to its own LDS slice.
// After the barrier, all waves read from all slices via ds_load_u16.
static constexpr int LDS_ELEMS = BLOCK_SIZE;  // 128 fp16 values

__global__ void fa_barrier_epoch(const uint16_t* __restrict__ d_in,
                                 uint32_t* __restrict__       d_out,
                                 unsigned int                 n) {
    __shared__ uint16_t sdata[LDS_ELEMS];

    const uint32_t tid      = threadIdx.x;
    const uint32_t lane     = tid % WAVE_SIZE;
    const uint32_t wave_idx = tid / WAVE_SIZE;

    // LDS byte base address of sdata[0].
    uint32_t sdata_base;
    asm("v_mov_b32 %0, %1"
        : "=v"(sdata_base)
        : "v"(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&sdata[0]))));

    // -------------------------------------------------------------------------
    // Phase 1: Each wave writes its slice of LDS via ds_store_b16.
    // In the real FA kernel this is tensor_load_to_lds; we use ds_store_b16
    // to avoid tensor descriptor setup while exercising the same LDS write +
    // barrier + ds_load_u16 read pattern.
    // -------------------------------------------------------------------------
    uint32_t global_idx = blockIdx.x * BLOCK_SIZE + tid;
    uint16_t val        = (global_idx < n) ? d_in[global_idx] : static_cast<uint16_t>(1);
    uint32_t write_addr = sdata_base + tid * sizeof(uint16_t);
    asm volatile("ds_store_b16 %0, %1" :: "v"(write_addr), "v"(static_cast<uint32_t>(val)) : "memory");
    asm volatile("s_wait_dscnt 0" ::: "memory");

    // -------------------------------------------------------------------------
    // Phase 2: Workgroup barrier — s_barrier_signal / s_barrier_wait.
    // After this point, all waves' LDS writes are visible to all waves.
    // This is the epoch boundary that the plugin must recognise.
    // -------------------------------------------------------------------------
    __syncthreads();

    // -------------------------------------------------------------------------
    // Phase 3: All waves read from LDS via ds_load_u16 — the exact instruction
    // that appeared in the flash attention false-positive report.
    // Each thread reads 4 fp16 values from across the full tile (not just its
    // own wave's slice), proving cross-wave visibility.
    // -------------------------------------------------------------------------
    uint32_t sum = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t read_idx  = (tid * 4 + i) % LDS_ELEMS;
        uint32_t read_addr = sdata_base + read_idx * sizeof(uint16_t);
        uint32_t loaded;
        asm volatile("ds_load_u16 %0, %2\n\ts_wait_dscnt 0\n\tv_add_nc_u32 %1, %0, %1"
                     : "=&v"(loaded), "+v"(sum)
                     : "v"(read_addr)
                     : "memory");
    }

    uint32_t out_idx = blockIdx.x * BLOCK_SIZE + tid;
    if (out_idx < n)
        d_out[out_idx] = sum;
}

int main() {
    constexpr int NUM_BLOCKS = 4;
    constexpr int N          = NUM_BLOCKS * BLOCK_SIZE;

    std::vector<uint16_t> h_in(N);
    for (int i = 0; i < N; ++i)
        h_in[i] = static_cast<uint16_t>((i & 0xFF) + 1);

    std::vector<uint32_t> h_out(N, 0);

    uint16_t* d_in;
    uint32_t* d_out;
    HIP_CHECK(hipMalloc(&d_in, N * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&d_out, N * sizeof(uint32_t)));
    HIP_CHECK(hipMemcpy(d_in, h_in.data(), N * sizeof(uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_out, 0, N * sizeof(uint32_t)));

    fa_barrier_epoch<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, N);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_out.data(), d_out, N * sizeof(uint32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));

    std::cout << "[";
    for (int i = 0; i < N; ++i) {
        if (i > 0) std::cout << ",";
        std::cout << h_out[i];
    }
    std::cout << "]" << std::endl;

    bool pass = true;
    for (int i = 0; i < N; ++i) {
        if (h_out[i] == 0) {
            std::cerr << "Mismatch at " << i << ": got 0 (expected non-zero)\n";
            pass = false;
            break;
        }
    }
    std::cerr << (pass ? "PASS" : "FAIL") << std::endl;
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
