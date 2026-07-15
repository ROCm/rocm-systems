/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "queue_pair.hpp"

#include <hip/hip_runtime.h>

#include "backend_gda.hpp"
#include "constmem.hpp"
#include "constants.hpp"
#include "util.hpp"

namespace rocshmem {

QueuePair::QueuePair(struct ibv_pd* pd, int gda_provider) {
  int access = IBV_ACCESS_LOCAL_WRITE
             | IBV_ACCESS_REMOTE_WRITE
             | IBV_ACCESS_REMOTE_READ
             | IBV_ACCESS_REMOTE_ATOMIC;

  if (envvar::gda::pcie_relaxed_ordering) {
    access |= IBV_ACCESS_RELAXED_ORDERING;
  }
  allocator.allocate((void**)&nonfetching_atomic, 8);
  allocator.allocate((void**)&fetching_atomic, 8 * FETCHING_ATOMIC_CNT);
  allocator.allocate((void**)&fetching_atomic_freelist, sizeof(FreeListT*));
  new (fetching_atomic_freelist) FreeListT(HIPAllocator());

  CHECK_HIP(hipMemset(nonfetching_atomic, 0, 8));
  CHECK_HIP(hipMemset(fetching_atomic, 0, 8 * FETCHING_ATOMIC_CNT));

  mr_nonfetching_atomic = ibv.reg_mr(pd, nonfetching_atomic, 8, access, &allocator);
  CHECK_NNULL(mr_nonfetching_atomic, "ibv_reg_mr");

  mr_fetching_atomic = ibv.reg_mr(pd, fetching_atomic, 8 * FETCHING_ATOMIC_CNT, access, &allocator);
  CHECK_NNULL(mr_fetching_atomic, "ibv_reg_mr");

  nonfetching_atomic_lkey = mr_nonfetching_atomic->lkey;
  fetching_atomic_lkey = mr_fetching_atomic->lkey;

  static int wf_size = 0;
  if (wf_size == 0) {
    int deviceId;
    CHECK_HIP(hipGetDevice(&deviceId));
    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, deviceId));
    wf_size = prop.warpSize;
  }
  for(uint32_t i{0}; i < FETCHING_ATOMIC_CNT; i+=wf_size) {
    fetching_atomic_freelist->push_back(fetching_atomic + i);
  }

  /* Set Correct opcodes for each NIC */
  switch (gda_provider) {
#if defined(GDA_IONIC)
  case GDAProvider::IONIC:
    gda_op_rdma_write = IONIC_V2_OP_RDMA_WRITE;
    gda_op_rdma_read  = IONIC_V2_OP_RDMA_READ;
    gda_op_atomic_fa  = IONIC_V2_OP_ATOMIC_FA;
    gda_op_atomic_cs  = IONIC_V2_OP_ATOMIC_CS;
    break;
#endif //defined(GDA_IONIC)
#if defined(GDA_BNXT)
  case GDAProvider::BNXT:
    gda_op_rdma_write = BNXT_RE_WR_OPCD_RDMA_WRITE;
    gda_op_rdma_read  = BNXT_RE_WR_OPCD_RDMA_READ;
    gda_op_atomic_fa  = BNXT_RE_WR_OPCD_ATOMIC_FA;
    gda_op_atomic_cs  = BNXT_RE_WR_OPCD_ATOMIC_CS;
    break;
#endif //defined(GDA_BNXT)
#if defined(GDA_MLX5)
  case GDAProvider::MLX5:
    gda_op_rdma_write = MLX5_OPCODE_RDMA_WRITE;
    gda_op_rdma_read  = MLX5_OPCODE_RDMA_READ;
    gda_op_atomic_fa  = MLX5_OPCODE_ATOMIC_FA;
    gda_op_atomic_cs  = MLX5_OPCODE_ATOMIC_CS;
    break;
#endif //defined(GDA_MLX5)
  default:
    assert(false /* invalid nic provider */);
  }
  gda_provider_ = gda_provider;
  /* Setup User Buffer Registration Mechanism */
  pd_ = pd;
  num_user_buffers = envvar::gda::num_user_buffers;

  CHECK_HIP(hipMalloc(&user_buf_info, sizeof(struct user_buf_info_t) *  num_user_buffers));
  CHECK_HIP(hipMemset(user_buf_info, 0, sizeof(struct user_buf_info_t) *  num_user_buffers));
}

QueuePair::~QueuePair() {
  int err;

  err = ibv.dereg_mr(mr_nonfetching_atomic);
  CHECK_ZERO(err, "ibv_dereg_mr (nonfetching_atomic)");

  err = ibv.dereg_mr(mr_fetching_atomic);
  CHECK_ZERO(err, "ibv_dereg_mr (fetching_atomic)");

  allocator.deallocate((void*)nonfetching_atomic);
  allocator.deallocate((void*)fetching_atomic);

  fetching_atomic_freelist->~FreeListT();
  allocator.deallocate((void*)fetching_atomic_freelist);

  if (user_buf_info) {
    CHECK_HIP(hipFree(user_buf_info));
    user_buf_info = nullptr;
  }
}

__device__ uint64_t QueuePair::get_same_qp_lane_mask() {
  uint64_t active = get_active_lane_mask();
  uintptr_t this_qp = reinterpret_cast<uintptr_t>(this);
  // Bitmask of lanes in this warp whose value == this_qp
  uint64_t same_qp_mask = __match_any_sync(active, this_qp);
  return same_qp_mask;
}

/******************************************************************************
 ************************ PROVIDER-SPECIFIC HELPERS ***************************
 *****************************************************************************/
__device__ void QueuePair::post_wqe_rma(
    int32_t length, uintptr_t raddr, uint32_t rkey,
    uintptr_t laddr, uint32_t lkey,
    uint8_t opcode, ActiveWFInfo &wf_info, bool ring_db) {
  switch (constmem.gda_provider) {
#if defined(GDA_IONIC)
  case GDAProvider::IONIC:
    ionic_post_wqe_rma(length, raddr, rkey, laddr, lkey, opcode, wf_info, ring_db);
    return;
#endif
#if defined(GDA_BNXT)
  case GDAProvider::BNXT:
    bnxt_post_wqe_rma(length, raddr, rkey, laddr, lkey, opcode, wf_info, ring_db);
    return;
#endif
#if defined(GDA_MLX5)
  case GDAProvider::MLX5:
    mlx5_post_wqe_rma(length, raddr, rkey, laddr, lkey, opcode, wf_info, ring_db);
    return;
#endif
  default:
    assert(false /* invalid nic provider */);
    __builtin_unreachable();
  }
}

__device__ void QueuePair::post_wqe_rma_single(int32_t length,
    uintptr_t laddr, uint32_t lkey, uintptr_t raddr, uint32_t rkey,
    uint8_t opcode, bool ring_db) {
  switch (constmem.gda_provider) {
#if defined(GDA_IONIC)
  case GDAProvider::IONIC:
    ionic_post_wqe_rma_single(length, laddr, lkey, raddr, rkey, opcode, ring_db);
    return;
#endif
#if defined(GDA_BNXT)
  case GDAProvider::BNXT:
    bnxt_post_wqe_rma_single(length, laddr, lkey, raddr, rkey, opcode, ring_db);
    return;
#endif
#if defined(GDA_MLX5)
  case GDAProvider::MLX5:
    mlx5_post_wqe_rma_single(length, laddr, lkey, raddr, rkey, opcode, ring_db);
    return;
#endif
  default:
    assert(false /* invalid nic provider */);
    __builtin_unreachable();
  }
}

__device__ uint64_t QueuePair::post_wqe_amo(uintptr_t raddr, uint32_t rkey,
    uint8_t opcode, int64_t atomic_data, int64_t atomic_cmp,
    ActiveWFInfo &wf_info, bool fetching, bool fence) {
  switch (constmem.gda_provider) {
#if defined(GDA_IONIC)
  case GDAProvider::IONIC:
    return ionic_post_wqe_amo(raddr, rkey, opcode, atomic_data, atomic_cmp,
           wf_info, fetching, fence);
#endif
#if defined(GDA_BNXT)
  case GDAProvider::BNXT:
    return bnxt_post_wqe_amo(raddr, rkey, opcode, atomic_data, atomic_cmp,
           wf_info, fetching, fence);
#endif
#if defined(GDA_MLX5)
  case GDAProvider::MLX5:
    return mlx5_post_wqe_amo(raddr, rkey, opcode, atomic_data, atomic_cmp,
           wf_info, fetching, fence);
#endif
  default:
    assert(false /* invalid nic provider */);
    __builtin_unreachable();
  }
}

__device__ uint64_t QueuePair::post_wqe_amo_single(uintptr_t raddr,
    uint32_t rkey, uint8_t opcode, int64_t atomic_data, int64_t atomic_cmp,
    bool fetching, bool fence) {
  switch (constmem.gda_provider) {
#if defined(GDA_IONIC)
  case GDAProvider::IONIC:
    return ionic_post_wqe_amo_single(raddr, rkey, opcode, atomic_data, atomic_cmp, fetching, fence);
#endif
#if defined(GDA_BNXT)
  case GDAProvider::BNXT:
    return bnxt_post_wqe_amo_single(raddr, rkey, opcode, atomic_data, atomic_cmp,
           fetching, fence);
#endif
#if defined(GDA_MLX5)
  case GDAProvider::MLX5:
    return mlx5_post_wqe_amo_single(raddr, rkey, opcode, atomic_data, atomic_cmp, fetching, fence);
#endif
  default:
    assert(false /* invalid nic provider */);
    __builtin_unreachable();
  }
}

__device__ void QueuePair::quiet(ActiveWFInfo &wf_info) {
  if(wf_info.is_pe_group_first) {
      switch (constmem.gda_provider) {
    #if defined(GDA_IONIC)
      case GDAProvider::IONIC:
        ionic_quiet(wf_info);
        return;
    #endif
    #if defined(GDA_BNXT)
      case GDAProvider::BNXT:
          bnxt_quiet();
        return;
    #endif
    #if defined(GDA_MLX5)
      case GDAProvider::MLX5:
          mlx5_quiet();
        return;
    #endif
      default:
        assert(false /* invalid nic provider */);
        __builtin_unreachable();
      }
  }
}

__device__ void QueuePair::quiet_single() {
  switch (constmem.gda_provider) {
#if defined(GDA_IONIC)
  case GDAProvider::IONIC:
    ionic_quiet_single();
    return;
#endif
#if defined(GDA_BNXT)
  case GDAProvider::BNXT:
    bnxt_quiet_single();
    return;
#endif
#if defined(GDA_MLX5)
  case GDAProvider::MLX5:
    mlx5_quiet_single();
    return;
#endif
  default:
    assert(false /* invalid nic provider */);
    __builtin_unreachable();
  }
}

/******************************************************************************
 ****************************** SHMEM INTERFACE *******************************
 *****************************************************************************/
__device__ void QueuePair::put_nbi(void *dest, const void *source,
    size_t length, ActiveWFInfo &wf_info) {
  uint32_t dst_rkey = rkey;
  uint32_t src_lkey = (static_cast<int32_t>(length) <= static_cast<int32_t>(inline_threshold))
      ? 0 : get_lkey(reinterpret_cast<uintptr_t>(source));
  put_nbi(dest, dst_rkey, source, src_lkey, length, wf_info);
}

__device__ void QueuePair::put_nbi(void *raddr, uint32_t rkey,
    const void *laddr, uint32_t lkey,
    size_t length, ActiveWFInfo &wf_info, bool ring_db) {
  uintptr_t l = reinterpret_cast<uintptr_t>(laddr);
  uintptr_t r = reinterpret_cast<uintptr_t>(raddr);
  post_wqe_rma(static_cast<int32_t>(length), r, rkey, l, lkey,
               gda_op_rdma_write, wf_info, ring_db);
}

// Used in all to all
__device__ void QueuePair::put_nbi_single(void *dest, const void *source,
    size_t length, bool ring_db) {
  uintptr_t src = reinterpret_cast<uintptr_t>(source);
  uintptr_t dst = reinterpret_cast<uintptr_t>(dest);
  uint32_t src_lkey = (static_cast<int32_t>(length) <= static_cast<int32_t>(inline_threshold))
      ? 0 : get_lkey(src);
  post_wqe_rma_single(length, src, src_lkey, dst, rkey,
                       gda_op_rdma_write, ring_db);
}

__device__ void QueuePair::put_nbi_single(void *raddr, uint32_t rkey,
    const void *laddr, uint32_t lkey,
    size_t length, bool ring_db) {
  uintptr_t l = reinterpret_cast<uintptr_t>(laddr);
  uintptr_t r = reinterpret_cast<uintptr_t>(raddr);
  post_wqe_rma_single(length, l, lkey, r, rkey,
                       gda_op_rdma_write, ring_db);
}

__device__ void QueuePair::get_nbi_single(void *dest, const void *source, size_t length, bool ring_db) {
  uintptr_t src = reinterpret_cast<uintptr_t>(source);
  uintptr_t dst = reinterpret_cast<uintptr_t>(dest);
  uint32_t dst_lkey = get_lkey(dst);
  post_wqe_rma_single(length, dst, dst_lkey, src, rkey,
                       gda_op_rdma_read, ring_db);
}


__device__ void QueuePair::get_nbi(void *dest, const void *source,
    size_t length, ActiveWFInfo &wf_info) {
  uint32_t src_rkey = rkey;
  uint32_t dst_lkey = get_lkey(reinterpret_cast<uintptr_t>(dest));
  uintptr_t src = reinterpret_cast<uintptr_t>(source);
  uintptr_t dst = reinterpret_cast<uintptr_t>(dest);
  post_wqe_rma(static_cast<int32_t>(length), src, src_rkey, dst, dst_lkey,
               gda_op_rdma_read, wf_info, true);
}

__device__ int64_t QueuePair::atomic_cas(void *dest, int64_t atomic_data,
    int64_t atomic_cmp, ActiveWFInfo &wf_info) {
  uintptr_t dst = reinterpret_cast<uintptr_t>(dest);
  return post_wqe_amo(dst, rkey, gda_op_atomic_cs, atomic_data,
                      atomic_cmp, wf_info, true);
}

__device__ int64_t QueuePair::atomic_cas_nofetch(void *dest,
    int64_t atomic_data, int64_t atomic_cmp, ActiveWFInfo &wf_info) {
  uintptr_t dst = reinterpret_cast<uintptr_t>(dest);
  return post_wqe_amo(dst, rkey, gda_op_atomic_cs, atomic_data,
                      atomic_cmp, wf_info);
}

__device__ int64_t QueuePair::atomic_fetch(void *dest, int64_t atomic_data,
    int64_t atomic_cmp, ActiveWFInfo &wf_info) {
  uintptr_t dst = reinterpret_cast<uintptr_t>(dest);
  return post_wqe_amo(dst, rkey, gda_op_atomic_fa, atomic_data,
                      atomic_cmp, wf_info, true);
}

__device__ void QueuePair::atomic_nofetch(void *dest, int64_t atomic_data,
    int64_t atomic_cmp, ActiveWFInfo &wf_info) {
  uintptr_t dst = reinterpret_cast<uintptr_t>(dest);
  post_wqe_amo(dst, rkey, gda_op_atomic_fa, atomic_data,
               atomic_cmp, wf_info);
}

__device__ void QueuePair::atomic_nofetch_single(void *dest, int64_t value) {
  uintptr_t dst = reinterpret_cast<uintptr_t>(dest);
  post_wqe_amo_single(dst, rkey, gda_op_atomic_fa, value, 0);
}

__device__ void QueuePair::atomic_add_single(void *raddr, uint32_t rkey,
    int64_t value, bool fence) {
  uintptr_t r = reinterpret_cast<uintptr_t>(raddr);
  post_wqe_amo_single(r, rkey, gda_op_atomic_fa, value, 0, false, fence);
}

__device__ void QueuePair::atomic_add(void *raddr, uint32_t rkey,
    int64_t value, ActiveWFInfo &wf_info, bool fence) {
  uintptr_t r = reinterpret_cast<uintptr_t>(raddr);
  post_wqe_amo(r, rkey, gda_op_atomic_fa, value, 0, wf_info, false, fence);
}

int QueuePair::buffer_register(uintptr_t addr, size_t length) {
  struct ibv_mr *mr = nullptr;
  int access = 0;

  if (user_buffer_mrs.size() >= num_user_buffers) {
    LOG_WARN("Unable to register user buffer with QP. "
             "Please increase the value of ROCSHMEM_GDA_NUM_USER_BUFFERS");
    return ROCSHMEM_ERROR;
  }

  access = IBV_ACCESS_LOCAL_WRITE
         | IBV_ACCESS_REMOTE_WRITE
         | IBV_ACCESS_REMOTE_READ
         | IBV_ACCESS_REMOTE_ATOMIC;

  if (envvar::gda::pcie_relaxed_ordering) {
    access |= IBV_ACCESS_RELAXED_ORDERING;
  }

  mr = ibv.reg_mr(pd_, (void*)addr, length, access, &allocator);
  CHECK_NNULL(mr, "ibv_reg_mr (buffer_register)");

  user_buffer_mrs[addr] = mr;

  for (size_t i=0; i<num_user_buffers; i++) {
    if (user_buf_info[i].addr == 0) {
      user_buf_info[i].addr   = addr;
      user_buf_info[i].length = length;

      user_buf_info[i].lkey = mr->lkey;

      break;
    }
  }

  return ROCSHMEM_SUCCESS;
}

int QueuePair::buffer_unregister(uintptr_t addr) {
  int err;

  for (size_t i=0; i<num_user_buffers; i++) {
    if (is_ptr_in_range(user_buf_info[i].addr, user_buf_info[i].length, addr)) {
      CHECK_HIP(hipMemset(&user_buf_info[i], 0, sizeof(struct user_buf_info_t)));
      break;
    }
  }

  err = ibv.dereg_mr(user_buffer_mrs[addr]);
  CHECK_ZERO(err, "ibv_dereg_mr (buffer_unregister)");

  user_buffer_mrs.erase(addr);

  return ROCSHMEM_SUCCESS;
}

void QueuePair::buffer_unregister_all() {
  int err;

  /* Deregister every memory region registered with this QP */
  for (auto &entry : user_buffer_mrs) {
    err = ibv.dereg_mr(entry.second);
    CHECK_ZERO(err, "ibv_dereg_mr (buffer_unregister_all)");
  }

  user_buffer_mrs.clear();

  /* Clear all user buffer info slots */
  CHECK_HIP(hipMemset(user_buf_info, 0,
                      sizeof(struct user_buf_info_t) * num_user_buffers));
}

__device__ uint32_t QueuePair::get_lkey(uintptr_t addr) {
  /* Check if in heap */
  if (is_ptr_in_range(base_heap, base_heap_size, addr)) {
    return lkey;
  }

  /* Get the correct lkey for the user buffer */
  for (size_t i=0; i<num_user_buffers; i++) {
    uintptr_t uaddr = user_buf_info[i].addr;
    size_t uaddr_len = user_buf_info[i].length;

    if (is_ptr_in_range(uaddr, uaddr_len, addr)) {
      return user_buf_info[i].lkey;
    }
  }

  LOGD_ERROR_ABORT("Valid lkey buffer not found");
  return 0;
}

}  // namespace rocshmem

// Exported C function for GIN QP factory to initialize __constant__ constmem.
// Lives in librocshmem.a so HIP_SYMBOL(constmem) resolves via device linking.
// Callable from librccl.so via -rdynamic symbol export.
extern "C" void rocshmem_gin_init_constmem(int provider, int rank) {
  using namespace rocshmem;

  // Initialize constmem.gda_provider for QP device dispatch
  GDAProvider gda_prov = static_cast<GDAProvider>(provider);
  constmem_t* cm_addr{nullptr};
  if (hipGetSymbolAddress(reinterpret_cast<void**>(&cm_addr),
                          HIP_SYMBOL(constmem)) == hipSuccess) {
    CHECK_HIP(hipMemcpy(&cm_addr->gda_provider, &gda_prov, sizeof(gda_prov), hipMemcpyDefault));
  }

  // Initialize logd_constants for device-side error reporting
  log_pe_number = rank;
  uint32_t log_flags = 0;
  if (envvar::log_flags.show_error) log_flags |= logd_constants::SHOW_ERROR;
  if (envvar::log_flags.show_warn)  log_flags |= logd_constants::SHOW_WARN;
  if (envvar::log_flags.show_info)  log_flags |= logd_constants::SHOW_INFO;
  if (envvar::log_flags.show_trace) log_flags |= logd_constants::SHOW_TRACE;
  if (envvar::log_flags.show_color) log_flags |= logd_constants::SHOW_COLOR;
  struct logd_constants host_logd{rank, log_flags};
  struct logd_constants* logd_addr{nullptr};
  if (hipGetSymbolAddress(reinterpret_cast<void**>(&logd_addr),
                          HIP_SYMBOL(logd_constants)) == hipSuccess) {
    CHECK_HIP(hipMemcpy(logd_addr, &host_logd, sizeof(host_logd), hipMemcpyDefault));
  }
}
