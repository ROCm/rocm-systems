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

#ifndef ROCSHMEM_LIBRARY_SRC_NET_VERBS_HOST_HPP_
#define ROCSHMEM_LIBRARY_SRC_NET_VERBS_HOST_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "hdp_policy.hpp"
#include "net/addr_exchange.hpp"
#include "net/ib_ctrl.hpp"
#include "net/ibv.hpp"
#include "net/lane_map.hpp"
#include "net/mr_registry.hpp"
#include "net/rma_engine.hpp"
#include "net/window_info_verbs.hpp"

namespace rocshmem {
namespace net {

/**
 * @brief Self-contained host-side verbs stack for the host-initiated API.
 *
 * Owns everything the verbs host path needs -- its own device/PD (via IbCtrl,
 * independent of any GDA DevX QPs), heap + AMO-scratch MRs, RC mesh, a shared
 * RmaEngine, and a pool of WindowInfoVerbs (one per host context). HostInterface
 * constructs one of these when ROCSHMEM_HOST_TRANSPORT=verbs and routes host RMA
 * to the acquired WindowInfoVerbs; barrier/sync/collectives stay in HostInterface
 * over the bootstrap.
 *
 * init() is collective (all PEs) -- it all-gathers keys/QP dest-info and connects
 * the mesh -- so it must be called from the collective HostInterface constructor.
 */
class VerbsHost {
 public:
  VerbsHost() = default;
  ~VerbsHost();
  VerbsHost(const VerbsHost &) = delete;
  VerbsHost &operator=(const VerbsHost &) = delete;

  /**
   * @param hdp          HDP policy (for GPU<->NIC visibility flushes).
   * @param heap_base    Symmetric-heap base.
   * @param heap_size    Symmetric-heap size.
   * @param heap_device  True if the heap is device memory (dma-buf registration).
   * @param num_pes/my_pe  Rank info.
   * @param num_host_ctx Number of host contexts (pool size / LaneMap contexts).
   * @param allgather    Fixed-size per-PE all-gather (MPI or bootstrap).
   * @param dmabuf       dma-buf exporter for device pointers.
   * @param dev_name     HCA name filter, or nullptr for first ACTIVE port.
   * @return true on success.
   */
  bool init(HdpPolicy *hdp, void *heap_base, size_t heap_size, bool heap_device,
            int num_pes, int my_pe, int num_host_ctx, AllgatherFn allgather,
            DmabufFn dmabuf, const char *dev_name);

  /// Acquire a free WindowInfoVerbs (one per host context); nullptr if exhausted.
  WindowInfoVerbs *acquire();
  /// Return a window to the pool.
  void release(WindowInfoVerbs *w);

 private:
  Ibv ibv_{};
  IbCtrl ctrl_{};
  MrRegistry mr_{};
  RmaEngine rma_{};
  LaneMap lm_{};

  std::vector<std::unique_ptr<WindowInfoVerbs>> pool_;
  std::vector<bool> avail_;

  // One registered host buffer sliced into per-context 8-byte AMO/g result slots.
  void *amo_scratch_{nullptr};
  uint32_t amo_scratch_lkey_{0};

  bool initialized_{false};
};

}  // namespace net
}  // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_NET_VERBS_HOST_HPP_
