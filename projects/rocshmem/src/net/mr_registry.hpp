/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *****************************************************************************/

#ifndef ROCSHMEM_LIBRARY_SRC_NET_MR_REGISTRY_HPP_
#define ROCSHMEM_LIBRARY_SRC_NET_MR_REGISTRY_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "net/addr_exchange.hpp"
#include "net/ibv.hpp"
#include "net/remote_region.hpp"

namespace rocshmem {
namespace net {

/**
 * @brief Acquire a dma-buf fd+offset for a device pointer.
 *
 * Injected so mr_registry stays HIP-free and unit-testable with host memory.
 * The RO backend supplies a lambda over its heap allocator's GetDmabufHandle
 * (the same call GDA uses). Return false to fall back to plain ibv_reg_mr.
 */
using DmabufFn =
    std::function<bool(void *addr, size_t len, int *fd, uint64_t *offset)>;

/**
 * @brief Registers the symmetric heap and local return buffers, and holds the
 *        per-PE RemoteRegion table used to translate symmetric VAs to peers.
 *
 * - The heap is registered once (dma-buf for device pointers, else ibv_reg_mr)
 *   with LOCAL_WRITE|REMOTE_WRITE|REMOTE_READ|REMOTE_ATOMIC, then its {base,rkey}
 *   is all-gathered so put/get/amo can reach every peer.
 * - Local return buffers (g_ret_buffer_ / atomic_ret_buffer_, which are NOT in
 *   the symmetric heap) are registered for their lkey so RDMA_READ and fetching
 *   atomics can land into them.
 */
class MrRegistry {
 public:
  MrRegistry() = default;
  ~MrRegistry();
  MrRegistry(const MrRegistry &) = delete;
  MrRegistry &operator=(const MrRegistry &) = delete;

  /// Register the local symmetric heap. @p dmabuf may be empty for host memory.
  bool register_heap(Ibv *ibv, struct ibv_pd *pd, void *base, size_t size,
                     bool is_device, const DmabufFn &dmabuf);

  /// All-gather peer heap {base,rkey} into the RemoteRegion table.
  bool exchange_heap(int num_pes, int my_pe, const AllgatherFn &allgather);

  /// Register a local-only buffer (return buffers); returns its lkey (0 on fail).
  uint32_t register_local(Ibv *ibv, struct ibv_pd *pd, void *addr, size_t len,
                          bool is_device, const DmabufFn &dmabuf);

  const RemoteRegion &remote(int pe) const { return remote_[pe]; }
  uintptr_t local_heap_base() const { return local_base_; }
  uint32_t heap_lkey() const { return heap_lkey_; }
  uint32_t heap_rkey() const { return heap_rkey_; }

 private:
  struct ibv_mr *reg_region(Ibv *ibv, struct ibv_pd *pd, void *addr, size_t len,
                            bool is_device, const DmabufFn &dmabuf, int access);

  Ibv *ibv_{nullptr};
  struct ibv_mr *heap_mr_{nullptr};
  std::vector<struct ibv_mr *> local_mrs_;
  std::vector<RemoteRegion> remote_;
  uintptr_t local_base_{0};
  uint32_t heap_lkey_{0};
  uint32_t heap_rkey_{0};
};

}  // namespace net
}  // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_NET_MR_REGISTRY_HPP_
