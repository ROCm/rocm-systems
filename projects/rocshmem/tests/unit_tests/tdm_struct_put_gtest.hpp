/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

// Phase 2: Structured-data put via ipcCopy_wg / memcpy_wg_tdm.
// Transfers arrays of TdmTestRecord (32 bytes each) and validates field-by-field.
// The kernel registers its dynamic LDS buffer via rocshmem_set_tdm_lds(),
// sized at launch via rocshmem_query_tdm_lds_bytes() -- the same public API
// any user kernel would use to opt into the TDM fast path.

#ifndef ROCSHMEM_TDM_STRUCT_PUT_GTEST_HPP
#define ROCSHMEM_TDM_STRUCT_PUT_GTEST_HPP

#include "gtest/gtest.h"

#include <hip/hip_runtime.h>
#include <mpi.h>
#include <cassert>
#include <vector>

#include "rocshmem/rocshmem.hpp"
#include "../src/atomic.hpp"
#include "../src/ipc_policy.hpp"
#include "../src/memory/notifier.hpp"
#include "../src/memory/symmetric_heap.hpp"
#include "../src/util.hpp"
#include "ipc_test_config.hpp"

namespace rocshmem {

struct __attribute__((packed)) TdmTestRecord {
  int   id;
  float val[7];
};
static_assert(sizeof(TdmTestRecord) == 32, "TdmTestRecord must be 32 bytes");

static constexpr uint32_t TDM_STRUCT_SIGNAL_OFFSET = 67108864u;

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

template <typename NotifierT>
__global__
void tdm_struct_validator(bool* error, const TdmTestRecord* golden,
                          const TdmTestRecord* dest, size_t count,
                          NotifierT* notifier) {
  if (!get_flat_id()) {
    uint32_t* sig = reinterpret_cast<uint32_t*>(
        const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(dest)) +
        TDM_STRUCT_SIGNAL_OFFSET);
    while (detail::atomic::load<detail::atomic::memory_scope::system,
                                detail::atomic::memory_order::acquire>(sig) == 0) {}
  }
  notifier->sync();
  for (size_t i = get_flat_id(); i < count; i += get_flat_grid_size()) {
    if (dest[i].id != golden[i].id) {
      printf("record[%zu].id exp=%d got=%d\n", i, golden[i].id, dest[i].id);
      *error = true;
    }
    for (int f = 0; f < 7; f++) {
      if (dest[i].val[f] != golden[i].val[f]) {
        printf("record[%zu].val[%d] exp=%f got=%f\n",
               i, f, (double)golden[i].val[f], (double)dest[i].val[f]);
        *error = true;
      }
    }
  }
}

template <typename IpcImplT>
__global__
void tdm_struct_put_kernel(IpcImplT* ipc_impl, const TdmTestRecord* src,
                           TdmTestRecord* remote_dst, size_t count) {
  const size_t bytes = count * sizeof(TdmTestRecord);

  extern __shared__ uint8_t tdm_lds[];
  rocshmem_set_tdm_lds(tdm_lds, rocshmem_query_tdm_lds_bytes());

  ipc_impl->template ipcCopy_wg<MemcpyKind::PutBlocking>(
      remote_dst, const_cast<TdmTestRecord*>(src), bytes, /*local_pe=*/0);

  if (is_thread_zero_in_block()) {
    ipc_impl->ipcFence();
    uint32_t* sig = reinterpret_cast<uint32_t*>(
        reinterpret_cast<uint8_t*>(remote_dst) + TDM_STRUCT_SIGNAL_OFFSET);
    ipc_impl->ipcAMOFetchAdd(sig, uint32_t{1});
  }
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class TdmStructPutFixture : public ::testing::TestWithParam<size_t> {
  using IpcImplT       = IpcImpl;
  using MPI_T          = RemoteHeapInfo<CommunicatorMPI>;
  using NotifierT      = Notifier<detail::atomic::memory_scope::device>;
  using NotifierProxyT = NotifierProxy<detail::atomic::memory_scope::device>;

 public:
  TdmStructPutFixture() {
    MPIInstance::mpilib_dl_init();
    hip_allocator_ = get_default_allocator();
    heap_mem_ = new HeapMemoryType(*hip_allocator_, envvar::heap_size.get_value());
    assert(heap_mem_);
    mpi_ = new MPI_T(heap_mem_->get_ptr(), heap_mem_->get_size(), MPI_COMM_WORLD);
    ipc_impl_.ipcHostInit(mpi_->my_pe(), mpi_->get_heap_bases(), MPI_COMM_WORLD);
    hip_allocator_->allocate((void**)&ipc_impl_dptr_, sizeof(IpcImplT));
    CHECK_HIP(hipMemcpy(ipc_impl_dptr_, &ipc_impl_, sizeof(IpcImplT),
                        hipMemcpyHostToDevice));
    hip_allocator_->allocate((void**)&error_dptr_, sizeof(bool));
    CHECK_HIP(hipMemset(error_dptr_, 0, sizeof(bool)));
  }

  ~TdmStructPutFixture() override {
    if (ipc_impl_dptr_) hip_allocator_->deallocate(ipc_impl_dptr_);
    if (error_dptr_)    hip_allocator_->deallocate(error_dptr_);
    if (golden_dptr_)   hip_allocator_->deallocate(golden_dptr_);
    ipc_impl_.ipcHostStop();
    delete mpi_;
    delete heap_mem_;
    MPIInstance::mpilib_dl_close();
  }

  void run_put(size_t record_count) {
    const size_t bytes = record_count * sizeof(TdmTestRecord);
    assert(bytes + TDM_STRUCT_SIGNAL_OFFSET + sizeof(uint32_t) <=
           heap_mem_->get_size());

    std::vector<TdmTestRecord> host_gold(record_count);
    for (size_t i = 0; i < record_count; i++) {
      host_gold[i].id = static_cast<int>(i);
      for (int f = 0; f < 7; f++)
        host_gold[i].val[f] = static_cast<float>(i * 7 + f) * 0.1f;
    }
    hip_allocator_->allocate((void**)&golden_dptr_,
                             record_count * sizeof(TdmTestRecord));
    CHECK_HIP(hipMemcpy(golden_dptr_, host_gold.data(),
                        record_count * sizeof(TdmTestRecord),
                        hipMemcpyHostToDevice));

    const int my_pe = mpi_->my_pe();

    if (my_pe == 0) {
      auto* local_heap = reinterpret_cast<TdmTestRecord*>(ipc_impl_.ipc_bases[0]);
      CHECK_HIP(hipMemcpy(local_heap, golden_dptr_,
                          record_count * sizeof(TdmTestRecord),
                          hipMemcpyDeviceToDevice));
      auto* remote_heap = reinterpret_cast<uint8_t*>(ipc_impl_.ipc_bases[1]);
      CHECK_HIP(hipMemset(remote_heap + TDM_STRUCT_SIGNAL_OFFSET, 0, sizeof(uint32_t)));
    }

    mpi_->barrier();

    const dim3 block(WF_SIZE);
    const size_t smem = rocshmem_query_tdm_lds_bytes();

    if (my_pe == 0) {
      auto* src  = reinterpret_cast<const TdmTestRecord*>(ipc_impl_.ipc_bases[0]);
      auto* rdst = reinterpret_cast<TdmTestRecord*>(ipc_impl_.ipc_bases[1]);
      tdm_struct_put_kernel<IpcImplT>
          <<<dim3(1), block, smem>>>(ipc_impl_dptr_, src, rdst, record_count);
      CHECK_HIP(hipStreamSynchronize(nullptr));
    } else {
      auto* dest = reinterpret_cast<const TdmTestRecord*>(ipc_impl_.ipc_bases[1]);
      tdm_struct_validator<NotifierT>
          <<<dim3(1), dim3(64)>>>(error_dptr_, golden_dptr_, dest,
                                  record_count, notifier_.get());
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
  TdmTestRecord*   golden_dptr_{nullptr};
};

}  // namespace rocshmem

#endif  // ROCSHMEM_TDM_STRUCT_PUT_GTEST_HPP
