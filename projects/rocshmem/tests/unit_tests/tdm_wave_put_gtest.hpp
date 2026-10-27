/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

// Wave-granularity put tests exercising memcpy_wave_tdm via ipcCopy_wave.
//
// Requires 2 MPI ranks. PE 0 launches one block whose waves each call
// ipcCopy_wave concurrently on disjoint slices of a shared symmetric
// buffer, exercising the per-wave LDS slicing that keeps concurrent waves
// from tiling through each other's bytes. The kernel opts in the same way
// any user kernel would: rocshmem_set_tdm_lds() sized via
// rocshmem_query_tdm_lds_bytes().
//
// Build: cmake -DUSE_TDM=ON -DUSE_IPC=ON -DAMDGPU_TARGETS=gfx1250 ...
// Run:   mpirun -n 2 ./tests/unit_tests/rocshmem_unit_tests \
//               --gtest_filter="TdmWave*"

#ifndef ROCSHMEM_TDM_WAVE_PUT_GTEST_HPP
#define ROCSHMEM_TDM_WAVE_PUT_GTEST_HPP

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
#include "../src/memory/notifier.hpp"
#include "../src/memory/symmetric_heap.hpp"
#include "../src/util.hpp"
#include "ipc_test_config.hpp"

namespace rocshmem {

static constexpr uint32_t TDM_WAVE_SIGNAL_OFFSET = 67108864u;  // 64 MB

// Fixed TDM tile size for these tests
static constexpr uint32_t TEST_TDM_TILE_BYTES = 16384;

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// PE 1 validator: spins on a signal word, then checks data.
template <typename NotifierT>
__global__
void tdm_wave_validator(bool* error, const uint8_t* golden, const uint8_t* dest,
                        size_t bytes, NotifierT* notifier) {
  if (!get_flat_id()) {
    uint32_t* sig = reinterpret_cast<uint32_t*>(
        const_cast<uint8_t*>(dest) + TDM_WAVE_SIGNAL_OFFSET);
    while (detail::atomic::load<detail::atomic::memory_scope::system,
                                detail::atomic::memory_order::acquire>(sig) == 0) {}
  }
  notifier->sync();
  for (size_t i = get_flat_id(); i < bytes; i += get_flat_grid_size()) {
    if (golden[i] != dest[i]) {
      printf("mismatch@%zu exp=%u got=%u\n", i, (unsigned)golden[i],
             (unsigned)dest[i]);
      *error = true;
    }
  }
}

// PE 0 sender: every wave in the block puts its own disjoint
// [wave_id * bytes_per_wave, (wave_id+1) * bytes_per_wave) slice,
// concurrently with every other wave.
template <typename IpcImplT>
__global__
void tdm_wave_put_kernel(IpcImplT* ipc_impl, const uint8_t* src,
                         uint8_t* remote_dst, size_t bytes_per_wave) {
  extern __shared__ uint8_t tdm_lds[];
  rocshmem_set_tdm_lds(tdm_lds, rocshmem_query_tdm_lds_bytes());

  const int wave_id = get_flat_block_id() / WF_SIZE;
  const size_t offset = static_cast<size_t>(wave_id) * bytes_per_wave;

  ipc_impl->template ipcCopy_wave<MemcpyKind::PutBlocking>(
      remote_dst + offset, const_cast<uint8_t*>(src) + offset, bytes_per_wave,
      /*local_pe=*/0);

  // Waves finish their (independently timed) TDM copies at different
  // points; every wave must land before any lane signals PE 1.
  __builtin_amdgcn_s_barrier();

  if (is_thread_zero_in_block()) {
    ipc_impl->ipcFence();
    uint32_t* sig = reinterpret_cast<uint32_t*>(remote_dst + TDM_WAVE_SIGNAL_OFFSET);
    ipc_impl->ipcAMOFetchAdd(sig, uint32_t{1});
  }
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

// Params: (bytes_per_wave, num_waves).
class TdmWavePutFixture
    : public ::testing::TestWithParam<std::tuple<size_t, int>> {
  using IpcImplT       = IpcImpl;   // whatever IpcImpl is (IpcOnImpl when USE_IPC)
  using MPI_T          = RemoteHeapInfo<CommunicatorMPI>;
  using NotifierT      = Notifier<detail::atomic::memory_scope::device>;
  using NotifierProxyT = NotifierProxy<detail::atomic::memory_scope::device>;

 public:
  TdmWavePutFixture() {
    MPIInstance::mpilib_dl_init();
    hip_allocator_ = get_default_allocator();
    heap_mem_ = new HeapMemoryType(*hip_allocator_, envvar::heap_size.get_value());
    assert(heap_mem_);
    mpi_ = new MPI_T(heap_mem_->get_ptr(), heap_mem_->get_size(), MPI_COMM_WORLD);
    ipc_impl_.ipcHostInit(mpi_->my_pe(), mpi_->get_heap_bases(), MPI_COMM_WORLD);
    // Force a known tile size so the parameterized byte counts below
    // actually land on tile/slice boundaries their comments claim.
    uint32_t tile_bytes = TEST_TDM_TILE_BYTES;
    CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(constmem), &tile_bytes, sizeof(tile_bytes),
                                offsetof(constmem_t, tdm_tile_bytes)));
    hip_allocator_->allocate((void**)&ipc_impl_dptr_, sizeof(IpcImplT));
    CHECK_HIP(hipMemcpy(ipc_impl_dptr_, &ipc_impl_, sizeof(IpcImplT),
                        hipMemcpyHostToDevice));
    hip_allocator_->allocate((void**)&error_dptr_, sizeof(bool));
    CHECK_HIP(hipMemset(error_dptr_, 0, sizeof(bool)));
  }

  ~TdmWavePutFixture() override {
    if (ipc_impl_dptr_) hip_allocator_->deallocate(ipc_impl_dptr_);
    if (error_dptr_)    hip_allocator_->deallocate(error_dptr_);
    if (golden_dptr_)   hip_allocator_->deallocate(golden_dptr_);
    ipc_impl_.ipcHostStop();
    delete mpi_;
    delete heap_mem_;
    MPIInstance::mpilib_dl_close();
  }

  void run_put(size_t bytes_per_wave, int num_waves) {
    const size_t bytes = bytes_per_wave * static_cast<size_t>(num_waves);
    assert(bytes + TDM_WAVE_SIGNAL_OFFSET + sizeof(uint32_t) <=
           heap_mem_->get_size());

    // Build iota pattern on host, upload to a device buffer.
    std::vector<uint8_t> host_gold(bytes);
    for (size_t i = 0; i < bytes; i++) host_gold[i] = static_cast<uint8_t>(i & 0xFF);
    hip_allocator_->allocate((void**)&golden_dptr_, bytes);
    CHECK_HIP(hipMemcpy(golden_dptr_, host_gold.data(), bytes, hipMemcpyHostToDevice));

    const int my_pe = mpi_->my_pe();

    if (my_pe == 0) {
      auto* local_heap = reinterpret_cast<uint8_t*>(ipc_impl_.ipc_bases[0]);
      CHECK_HIP(hipMemcpy(local_heap, golden_dptr_, bytes, hipMemcpyDeviceToDevice));
      auto* remote_heap = reinterpret_cast<uint8_t*>(ipc_impl_.ipc_bases[1]);
      CHECK_HIP(hipMemset(remote_heap + TDM_WAVE_SIGNAL_OFFSET, 0, sizeof(uint32_t)));
    }

    mpi_->barrier();

    const dim3 block(static_cast<unsigned>(WF_SIZE * num_waves));
    const size_t smem = rocshmem_query_tdm_lds_bytes();

    if (my_pe == 0) {
      auto* src  = reinterpret_cast<const uint8_t*>(ipc_impl_.ipc_bases[0]);
      auto* rdst = reinterpret_cast<uint8_t*>(ipc_impl_.ipc_bases[1]);
      tdm_wave_put_kernel<IpcImplT>
          <<<dim3(1), block, smem>>>(ipc_impl_dptr_, src, rdst, bytes_per_wave);
      CHECK_HIP(hipStreamSynchronize(nullptr));
    } else {
      auto* dest = reinterpret_cast<const uint8_t*>(ipc_impl_.ipc_bases[1]);
      tdm_wave_validator<NotifierT>
          <<<dim3(1), dim3(64)>>>(error_dptr_, golden_dptr_, dest, bytes,
                                  notifier_.get());
      CHECK_HIP(hipStreamSynchronize(nullptr));
    }

    mpi_->barrier();

    if (my_pe == 1) {
      bool host_error = false;
      CHECK_HIP(hipMemcpy(&host_error, error_dptr_, sizeof(bool),
                          hipMemcpyDeviceToHost));
      ASSERT_EQ(host_error, false);
    }
  }

 private:
  HIPAllocator*    hip_allocator_{nullptr};
  NotifierProxyT   notifier_{};
  HeapMemoryType*  heap_mem_{nullptr};
  MPI_T*           mpi_{nullptr};
  IpcImplT         ipc_impl_{};
  IpcImplT*        ipc_impl_dptr_{nullptr};
  bool*            error_dptr_{nullptr};
  uint8_t*         golden_dptr_{nullptr};
};

}  // namespace rocshmem

#endif  // ROCSHMEM_TDM_WAVE_PUT_GTEST_HPP
