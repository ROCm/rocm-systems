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

#include <cassert>

#include <tuple>
#include <utility>

#include <hip/hip_runtime.h>

#include "rocshmem/rocshmem_common.hpp"

#include "constants.hpp"
#include "containers/free_list_impl.hpp"
#include "log.hpp"
#include "util.hpp"

#include "queue_pair_common.hpp"
#include "queue_pair_host.hpp"
#include "gda/queue_pair_provider.hpp"

namespace rocshmem {

__host__ QueuePairHost::QueuePairHost(struct ibv_pd* pd) : pd{pd} {
  int access = IBV_ACCESS_LOCAL_WRITE
             | IBV_ACCESS_REMOTE_WRITE
             | IBV_ACCESS_REMOTE_READ
             | IBV_ACCESS_REMOTE_ATOMIC;

  if (envvar::gda::pcie_relaxed_ordering) {
    access |= IBV_ACCESS_RELAXED_ORDERING;
  }

  // Allocate and register the fetching and nonfetching atomics arrays
  std::tie(nonfetching_atomic, nonfetching_atomic_mr)
      = allocate_and_register<uint64_t>(1, access);
  std::tie(fetching_atomic, fetching_atomic_mr)
      = allocate_and_register<uint64_t>(FETCHING_ATOMIC_CNT, access);

  allocator.allocate(reinterpret_cast<void**>(&fetching_atomic_freelist), sizeof(FreeListT));
  new (fetching_atomic_freelist) FreeListT{allocator};

  int deviceId;
  CHECK_HIP(hipGetDevice(&deviceId));
  int wf_size = get_wf_size(deviceId);
  for (size_t i = 0; i < FETCHING_ATOMIC_CNT; i += wf_size) {
    fetching_atomic_freelist->push_back(&fetching_atomic[i]);
  }

  /* Setup User Buffer Registration Mechanism */
  num_user_buffers = envvar::gda::num_user_buffers;

  CHECK_HIP(hipMalloc(&buffer_info, sizeof(BufferInfo) * num_user_buffers));
  CHECK_HIP(hipMemset(buffer_info, 0, sizeof(BufferInfo) * num_user_buffers));

  /* Reserve memory to register up to num_user_buffers */
  buffer_info_map.reserve(num_user_buffers);

  /* Create buffer registration info freelist */
  buffer_info_freelist = iterator_freelist<BufferInfo*>{buffer_info, num_user_buffers};
}

__host__ QueuePairHost::QueuePairHost(QueuePairHost&& other) noexcept
  : buffer_info             {std::move(other.buffer_info)},
    num_user_buffers        {std::move(other.num_user_buffers)},
    fetching_atomic         {std::move(other.fetching_atomic)},
    nonfetching_atomic      {std::move(other.nonfetching_atomic)},
    fetching_atomic_freelist{std::move(other.fetching_atomic_freelist)},
    allocator               {std::move(other.allocator)},
    buffer_info_map         {std::move(other.buffer_info_map)},
    buffer_info_freelist    {std::move(other.buffer_info_freelist)},
    pd                      {std::move(other.pd)},
    fetching_atomic_mr      {std::move(other.fetching_atomic_mr)},
    nonfetching_atomic_mr   {std::move(other.nonfetching_atomic_mr)} {
  other.buffer_info              = nullptr;
  other.fetching_atomic          = nullptr;
  other.nonfetching_atomic       = nullptr;
  other.fetching_atomic_freelist = nullptr;
  other.fetching_atomic_mr       = nullptr;
  other.nonfetching_atomic_mr    = nullptr;
}

__host__ QueuePairHost& QueuePairHost::operator=(QueuePairHost&& other) {
  int err = 0;

  /* Step 1: ensure all resources in *this are deallocated */
  if (!buffer_info_map.empty()) {
    LOG_WARN("Unmatched buffer_register detected: "
             "move assignment operator %s called, but buffer registration map is not empty!",
             __PRETTY_FUNCTION__ );
    /* Deregister every memory region registered with this QP */
    for (auto&& [addr, host_info] : buffer_info_map) {
      err = ibv.dereg_mr(host_info.mr);
      CHECK_ZERO(err, "ibv_dereg_mr (QueuePairHost::operator=(QueuePairHost&&))");
    }
  }

  if (buffer_info) {
    CHECK_HIP(hipFree(buffer_info));
  }

  if (fetching_atomic_freelist) {
    fetching_atomic_freelist->~FreeListT();
    allocator.deallocate(static_cast<void*>(fetching_atomic_freelist));
  }

  if (fetching_atomic_mr) {
    err = ibv.dereg_mr(fetching_atomic_mr);
    CHECK_ZERO(err, "ibv_dereg_mr (fetching_atomic)");
  }

  if (fetching_atomic) {
    allocator.deallocate(static_cast<void*>(fetching_atomic));
  }

  if (nonfetching_atomic_mr) {
    err = ibv.dereg_mr(nonfetching_atomic_mr);
    CHECK_ZERO(err, "ibv_dereg_mr (nonfetching_atomic)");
  }

  if (nonfetching_atomic) {
    allocator.deallocate(static_cast<void*>(nonfetching_atomic));
  }

  /* Step 2: member-wise move of all data members from other to *this */
  buffer_info              = std::move(other.buffer_info);
  num_user_buffers         = std::move(other.num_user_buffers);
  fetching_atomic          = std::move(other.fetching_atomic);
  nonfetching_atomic       = std::move(other.nonfetching_atomic);
  fetching_atomic_freelist = std::move(other.fetching_atomic_freelist);
  allocator                = std::move(other.allocator);
  buffer_info_map          = std::move(other.buffer_info_map);
  buffer_info_freelist     = std::move(other.buffer_info_freelist);
  pd                       = std::move(other.pd);
  fetching_atomic_mr       = std::move(other.fetching_atomic_mr);
  nonfetching_atomic_mr    = std::move(other.nonfetching_atomic_mr);

  /* Step 3: reset allocated data members of other so that it can be safely reused or destroyed */
  other.buffer_info              = nullptr;
  other.fetching_atomic          = nullptr;
  other.nonfetching_atomic       = nullptr;
  other.fetching_atomic_freelist = nullptr;
  other.fetching_atomic_mr       = nullptr;
  other.nonfetching_atomic_mr    = nullptr;

  /* Step 4: return *this */
  return *this;
}

__host__ QueuePairHost::~QueuePairHost() {
  int err = 0;

  if (!buffer_info_map.empty()) {
    LOG_WARN("Unmatched buffer_register detected: "
             "destructor %s called, but buffer registration map is not empty!",
             __PRETTY_FUNCTION__ );
    /* Deregister every memory region registered with this QP */
    for (auto&& [addr, host_info] : buffer_info_map) {
      err = ibv.dereg_mr(host_info.mr);
      CHECK_ZERO(err, "ibv_dereg_mr (QueuePairHost::~QueuePairHost)");
    }
  }

  if (buffer_info) {
    CHECK_HIP(hipFree(buffer_info));
  }

  if (fetching_atomic_freelist) {
    fetching_atomic_freelist->~FreeListT();
    allocator.deallocate(static_cast<void*>(fetching_atomic_freelist));
  }

  if (fetching_atomic_mr) {
    err = ibv.dereg_mr(fetching_atomic_mr);
    CHECK_ZERO(err, "ibv_dereg_mr (fetching_atomic)");
  }

  if (fetching_atomic) {
    allocator.deallocate(static_cast<void*>(fetching_atomic));
  }

  if (nonfetching_atomic_mr) {
    err = ibv.dereg_mr(nonfetching_atomic_mr);
    CHECK_ZERO(err, "ibv_dereg_mr (nonfetching_atomic)");
  }

  if (nonfetching_atomic) {
    allocator.deallocate(static_cast<void*>(nonfetching_atomic));
  }
}

__host__ int QueuePairHost::buffer_register(void *addr, size_t length) {
  if (buffer_info_freelist.empty()) {
    assert(buffer_info_map.size() == num_user_buffers);
    LOG_WARN("Unable to register user buffer (%p, %zu) with QP. "
             "Please increase the value of %s.",
             addr, length, envvar::gda::num_user_buffers.get_name().c_str());
    return ROCSHMEM_ERROR;
  }

  /* Register addr */
  int access = IBV_ACCESS_LOCAL_WRITE
             | IBV_ACCESS_REMOTE_WRITE
             | IBV_ACCESS_REMOTE_READ
             | IBV_ACCESS_REMOTE_ATOMIC;

  if (envvar::gda::pcie_relaxed_ordering) {
    access |= IBV_ACCESS_RELAXED_ORDERING;
  }

  struct ibv_mr* mr = ibv.reg_mr(pd, addr, length, access, &allocator);
  CHECK_NNULL(mr, "ibv_reg_mr (buffer_register)");

  /* Get pointer to next free buffer_info entry */
  BufferInfo* info_ptr = buffer_info_freelist.top();

  /* Try inserting buffer host info into buffer_info_map; detects whether this is a duplicate */
  auto&& [it, inserted] = buffer_info_map.try_emplace(addr, mr, info_ptr);

  if (inserted) [[likely]] {
    /* Insertion succceeded, pop the buffer_info entry from the freelist */
    buffer_info_freelist.pop();
    /* Copy buffer info to device */
    BufferInfo info{reinterpret_cast<uintptr_t>(addr), length,
                    QueuePair::to_provider_endianness(mr->lkey)};
    CHECK_HIP(hipMemcpy(info_ptr, &info, sizeof(BufferInfo), hipMemcpyHostToDevice));
    return ROCSHMEM_SUCCESS;
  } else {
    auto&& [key, host_info] = *it;
    LOG_WARN("Unable to register user buffer (%p, %zu) with QP: "
             "already registered with length=%zu, lkey=%u, rkey=%u.",
             addr, length, host_info.mr->length, host_info.mr->lkey, host_info.mr->rkey);
    /* Deregister addr */
    int err = ibv.dereg_mr(mr);
    CHECK_ZERO(err, "ibv_dereg_mr (buffer_register)");
    return ROCSHMEM_ERROR;
  }
}

__host__ int QueuePairHost::buffer_unregister(void *addr) {
  /* Lookup buffer host info */
  auto it = buffer_info_map.find(addr);

  if (it != buffer_info_map.end()) [[likely]] {
    auto&& [key, host_info] = *it;
    /* Reset buffer_info entry and push to freelist */
    CHECK_HIP(hipMemset(host_info.info_ptr, 0, sizeof(BufferInfo)));
    buffer_info_freelist.push(host_info.info_ptr);
    /* Deregister addr */
    int err = ibv.dereg_mr(host_info.mr);
    CHECK_ZERO(err, "ibv_dereg_mr (buffer_unregister)");
    /* Remove from map */
    buffer_info_map.erase(it);
    return ROCSHMEM_SUCCESS;
  } else {
    LOG_WARN("Unable to unregister user buffer (%p) with this QP: "
             "user buffer not registered.", addr);
    return ROCSHMEM_ERROR;
  }
}

__host__ int QueuePairHost::buffer_unregister_all() {
  int err = 0;

  /* Deregister every memory region registered with this QP */
  for (auto&& [addr, host_info] : buffer_info_map) {
    err = ibv.dereg_mr(host_info.mr);
    CHECK_ZERO(err, "ibv_dereg_mr (QueuePairHost::buffer_unregister_all)");
  }
  buffer_info_map.clear();

  /* Clear all buffer_info slots */
  CHECK_HIP(hipMemset(buffer_info, 0, sizeof(BufferInfo) * num_user_buffers));

  /* Reset buffer_info_freelist */
  buffer_info_freelist.reset(buffer_info, num_user_buffers);
  return ROCSHMEM_SUCCESS;
}

}  // namespace rocshmem
