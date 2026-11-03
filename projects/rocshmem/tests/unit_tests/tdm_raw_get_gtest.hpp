/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

// Raw byte-array get tests exercising memcpy_wg_tdm via ipcCopy_wg. Mirrors
// tdm_raw_put_gtest.hpp with the direction and threshold reversed.
//
// Build: cmake -DUSE_TDM=ON -DUSE_IPC=ON -DAMDGPU_TARGETS=gfx1250 ...
// Run:   mpirun -n 2 ./rocshmem_unit_tests --gtest_filter="TdmRawGet*"

#ifndef ROCSHMEM_TDM_RAW_GET_GTEST_HPP
#define ROCSHMEM_TDM_RAW_GET_GTEST_HPP

#include "gtest/gtest.h"

#include <hip/hip_runtime.h>
#include <mpi.h>
#include <cassert>
#include <cstddef>
#include <vector>

#include "rocshmem/rocshmem.hpp"
#include "../src/atomic.hpp"
#include "../src/constmem.hpp"
#include "../src/ipc_policy.hpp"
#include "../src/memory/symmetric_heap.hpp"
#include "../src/util.hpp"
#include "ipc_test_config.hpp"

namespace rocshmem {

// Fixed TDM tile size for these tests
static constexpr uint32_t TEST_TDM_GET_TILE_BYTES = 16384;
// Distinct from TEST_TDM_PUT_THRESHOLD so a swapped put/get field shows up
// as a wrong-path boundary instead of silently passing.
static constexpr size_t TEST_TDM_GET_THRESHOLD = 1024;

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

// PE 1 puller: pulls bytes from PE 0's heap via the real library path.
template <typename IpcImplT>
__global__
void tdm_raw_get_kernel(IpcImplT* ipc_impl, uint8_t* local_dst,
                        const uint8_t* remote_src, size_t bytes) {
  extern __shared__ uint8_t tdm_lds[];
  rocshmem_set_tdm_lds(tdm_lds, rocshmem_query_tdm_lds_bytes());

  ipc_impl->template ipcCopy_wg<MemcpyKind::GetBlocking>(
      local_dst, const_cast<uint8_t*>(remote_src), bytes, /*local_pe=*/0);
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class TdmRawGetFixture : public ::testing::TestWithParam<size_t> {
  using IpcImplT = IpcImpl;   // whatever IpcImpl is (IpcOnImpl when USE_IPC)
  using MPI_T    = RemoteHeapInfo<CommunicatorMPI>;

 public:
  TdmRawGetFixture() {
    MPIInstance::mpilib_dl_init();
    hip_allocator_ = get_default_allocator();
    heap_mem_ = new HeapMemoryType(*hip_allocator_, envvar::heap_size.get_value());
    assert(heap_mem_);
    mpi_ = new MPI_T(heap_mem_->get_ptr(), heap_mem_->get_size(), MPI_COMM_WORLD);
    ipc_impl_.ipcHostInit(mpi_->my_pe(), mpi_->get_heap_bases(), MPI_COMM_WORLD);
    // Force a known tile size so the parameterized sizes land where intended.
    uint32_t tile_bytes = TEST_TDM_GET_TILE_BYTES;
    CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(constmem), &tile_bytes, sizeof(tile_bytes),
                                offsetof(constmem_t, tdm_tile_bytes)));
    // Force SDMA off: this fixture never sets up an SDMA channel.
    size_t sdma_threshold = SDMA_THRESHOLD_DISABLED;
    CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(constmem), &sdma_threshold, sizeof(sdma_threshold),
                                offsetof(constmem_t, ipc_sdma_threshold)));
    // Force a known get threshold so the parameterized sizes land where intended.
    size_t get_threshold = TEST_TDM_GET_THRESHOLD;
    CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(constmem), &get_threshold, sizeof(get_threshold),
                                offsetof(constmem_t, tdm_get_threshold)));
    hip_allocator_->allocate((void**)&ipc_impl_dptr_, sizeof(IpcImplT));
    CHECK_HIP(hipMemcpy(ipc_impl_dptr_, &ipc_impl_, sizeof(IpcImplT),
                        hipMemcpyHostToDevice));
  }

  ~TdmRawGetFixture() override {
    if (ipc_impl_dptr_) hip_allocator_->deallocate(ipc_impl_dptr_);
    ipc_impl_.ipcHostStop();
    delete mpi_;
    delete heap_mem_;
    MPIInstance::mpilib_dl_close();
  }

  void run_get(size_t bytes) {
    assert(bytes <= heap_mem_->get_size());

    // Varies per tile, unlike plain iota, so reordered/duplicated tiles
    // are detected.
    std::vector<uint8_t> host_gold(bytes);
    for (size_t i = 0; i < bytes; i++)
      host_gold[i] = static_cast<uint8_t>((i ^ (i >> 8) ^ (i >> 16)) & 0xFF);

    const int my_pe = mpi_->my_pe();

    if (my_pe == 0) {
      auto* local_heap = reinterpret_cast<uint8_t*>(ipc_impl_.ipc_bases[0]);
      CHECK_HIP(hipMemcpy(local_heap, host_gold.data(), bytes, hipMemcpyHostToDevice));
    }

    mpi_->barrier();

    if (my_pe == 1) {
      auto* local_dst  = reinterpret_cast<uint8_t*>(ipc_impl_.ipc_bases[1]);
      auto* remote_src = reinterpret_cast<const uint8_t*>(ipc_impl_.ipc_bases[0]);

      // Block size: full wavefront for TDM to have at least 1 wave.
      const dim3 block(WF_SIZE);
      // Backend-free: compute LDS size from the forced tile size directly.
      const size_t smem = tdm::lds_bytes_for_tile(TEST_TDM_GET_TILE_BYTES);
      tdm_raw_get_kernel<IpcImplT>
          <<<dim3(1), block, smem>>>(ipc_impl_dptr_, local_dst, remote_src, bytes);
      CHECK_HIP(hipStreamSynchronize(nullptr));

      std::vector<uint8_t> host_dst(bytes);
      CHECK_HIP(hipMemcpy(host_dst.data(), local_dst, bytes, hipMemcpyDeviceToHost));
      ASSERT_EQ(host_dst, host_gold);
    }

    mpi_->barrier();
  }

 private:
  HIPAllocator*   hip_allocator_{nullptr};
  HeapMemoryType* heap_mem_{nullptr};
  MPI_T*          mpi_{nullptr};
  IpcImplT        ipc_impl_{};
  IpcImplT*       ipc_impl_dptr_{nullptr};
};

}  // namespace rocshmem

#endif  // ROCSHMEM_TDM_RAW_GET_GTEST_HPP
