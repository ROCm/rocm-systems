/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

// Wave-granularity get tests exercising memcpy_wave_tdm (copy_pipelined_deep)
// via ipcCopy_wave.
//
// Requires 2 MPI ranks. PE 0 fills a symmetric buffer with a golden pattern
// that varies both per-byte and per-tile. PE 1 launches one block whose
// waves each call ipcCopy_wave<GetBlocking> concurrently on disjoint slices
// pulled from PE 0, exercising the per-wave LDS slicing and the
// copy_pipelined_deep pipeline memcpy_wave_tdm selects for gets. Because the
// golden pattern differs from tile to tile, a bug that duplicates or
// reorders tiles within the pipeline -- which a constant fill pattern (e.g.
// all 'a') could not detect -- shows up as a mismatch. Mirrors
// tdm_wave_put_gtest.hpp with the direction (and the get threshold)
// reversed.
//
// The kernel registers its dynamic LDS buffer via rocshmem_set_tdm_lds(),
// sized at launch via rocshmem_query_tdm_lds_bytes() -- the same public API
// any user kernel would use to opt into the TDM fast path.
//
// Build: cmake -DUSE_TDM=ON -DUSE_IPC=ON -DAMDGPU_TARGETS=gfx1250 ...
// Run:   mpirun -n 2 ./tests/unit_tests/rocshmem_unit_tests \
//               --gtest_filter="TdmWaveGet*"

#ifndef ROCSHMEM_TDM_WAVE_GET_GTEST_HPP
#define ROCSHMEM_TDM_WAVE_GET_GTEST_HPP

#include "gtest/gtest.h"

#include <hip/hip_runtime.h>
#include <mpi.h>
#include <cassert>
#include <cstddef>
#include <tuple>
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
static constexpr uint32_t TEST_TDM_WAVE_GET_TILE_BYTES = 16384;
// Fixed get threshold: per-wave sizes below this must take the memcpy_wave
// fallback, sizes at/above it are eligible for the TDM path. Deliberately
// distinct from TEST_TDM_PUT_THRESHOLD (tdm_wave_put_gtest.hpp) so a bug
// that swaps the put/get threshold fields shows up as a wrong-path
// boundary instead of accidentally still passing.
static constexpr size_t TEST_TDM_WAVE_GET_THRESHOLD = 1024;

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

// PE 1 puller: every wave in the block pulls its own disjoint
// [wave_id * bytes_per_wave, (wave_id+1) * bytes_per_wave) slice from PE 0,
// concurrently with every other wave.
template <typename IpcImplT>
__global__
void tdm_wave_get_kernel(IpcImplT* ipc_impl, uint8_t* local_dst,
                         const uint8_t* remote_src, size_t bytes_per_wave) {
  extern __shared__ uint8_t tdm_lds[];
  rocshmem_set_tdm_lds(tdm_lds, rocshmem_query_tdm_lds_bytes());

  const int wave_id = get_flat_block_id() / WF_SIZE;
  const size_t offset = static_cast<size_t>(wave_id) * bytes_per_wave;

  ipc_impl->template ipcCopy_wave<MemcpyKind::GetBlocking>(
      local_dst + offset, const_cast<uint8_t*>(remote_src) + offset,
      bytes_per_wave, /*local_pe=*/0);
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

// Params: (bytes_per_wave, num_waves).
class TdmWaveGetFixture
    : public ::testing::TestWithParam<std::tuple<size_t, int>> {
  using IpcImplT = IpcImpl;   // whatever IpcImpl is (IpcOnImpl when USE_IPC)
  using MPI_T    = RemoteHeapInfo<CommunicatorMPI>;

 public:
  TdmWaveGetFixture() {
    MPIInstance::mpilib_dl_init();
    hip_allocator_ = get_default_allocator();
    heap_mem_ = new HeapMemoryType(*hip_allocator_, envvar::heap_size.get_value());
    assert(heap_mem_);
    mpi_ = new MPI_T(heap_mem_->get_ptr(), heap_mem_->get_size(), MPI_COMM_WORLD);
    ipc_impl_.ipcHostInit(mpi_->my_pe(), mpi_->get_heap_bases(), MPI_COMM_WORLD);
    // Force a known tile size so the parameterized byte counts below
    // actually land on tile/slice boundaries their comments claim.
    uint32_t tile_bytes = TEST_TDM_WAVE_GET_TILE_BYTES;
    CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(constmem), &tile_bytes, sizeof(tile_bytes),
                                offsetof(constmem_t, tdm_tile_bytes)));
    // Force a known get threshold so the parameterized byte counts below
    // actually land on either side of it, as their comments claim.
    size_t get_threshold = TEST_TDM_WAVE_GET_THRESHOLD;
    CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(constmem), &get_threshold, sizeof(get_threshold),
                                offsetof(constmem_t, tdm_get_threshold)));
    hip_allocator_->allocate((void**)&ipc_impl_dptr_, sizeof(IpcImplT));
    CHECK_HIP(hipMemcpy(ipc_impl_dptr_, &ipc_impl_, sizeof(IpcImplT),
                        hipMemcpyHostToDevice));
  }

  ~TdmWaveGetFixture() override {
    if (ipc_impl_dptr_) hip_allocator_->deallocate(ipc_impl_dptr_);
    ipc_impl_.ipcHostStop();
    delete mpi_;
    delete heap_mem_;
    MPIInstance::mpilib_dl_close();
  }

  void run_get(size_t bytes_per_wave, int num_waves) {
    const size_t bytes = bytes_per_wave * static_cast<size_t>(num_waves);
    assert(bytes <= heap_mem_->get_size());

    // Build a per-byte pattern that varies across tiles (plain i & 0xFF
    // would repeat every 256 bytes -- a multiple of
    // TEST_TDM_WAVE_GET_TILE_BYTES -- so every tile would be bit-identical
    // and a reordered/duplicated tile would go undetected).
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

      const dim3 block(static_cast<unsigned>(WF_SIZE * num_waves));
      const size_t smem = rocshmem_query_tdm_lds_bytes();
      tdm_wave_get_kernel<IpcImplT>
          <<<dim3(1), block, smem>>>(ipc_impl_dptr_, local_dst, remote_src,
                                     bytes_per_wave);
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

#endif  // ROCSHMEM_TDM_WAVE_GET_GTEST_HPP
