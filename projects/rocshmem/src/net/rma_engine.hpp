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

#ifndef ROCSHMEM_LIBRARY_SRC_NET_RMA_ENGINE_HPP_
#define ROCSHMEM_LIBRARY_SRC_NET_RMA_ENGINE_HPP_

#include <cstdint>
#include <vector>

#include "net/ib_ctrl.hpp"
#include "net/ibv.hpp"
#include "net/lane_map.hpp"
#include "net/mr_registry.hpp"

namespace rocshmem {
namespace net {

/**
 * @brief One registered local memory range and its lkey.
 *
 * lkey_for() scans these to resolve the lkey for a local buffer (the symmetric
 * heap and any return/scratch buffers registered outside it).
 */
struct LocalMr {
  uintptr_t base;
  uintptr_t end;
  uint32_t lkey;
};

/**
 * @brief Backend-neutral verbs RMA work-request builder.
 *
 * The single place work-requests are constructed and posted, shared by every
 * verbs consumer: the reverse-offload device proxy (asynchronous, GPU-notify)
 * and the host-initiated path (synchronous, host-return). It owns no threading
 * or completion policy -- it posts a WR and lets the caller decide how to reap
 * completions (poll() is provided). Addressing rides the symmetric-heap offset
 * model (see remote_region.hpp).
 *
 * Not owning: it holds non-owning pointers to the Ibv wrapper, IbCtrl (QPs/CQs),
 * and MrRegistry (peer rkeys), which must outlive it.
 */
class RmaEngine {
 public:
  RmaEngine() = default;
  RmaEngine(Ibv *ibv, IbCtrl *ctrl, MrRegistry *mr, LaneMap lm,
            uintptr_t heap_base, std::vector<LocalMr> local_mrs);

  // --- addressing ---------------------------------------------------------
  //! Peer VA for a local symmetric-heap VA (offset preserved).
  uintptr_t remote_va(int pe, const void *local_symmetric_va) const;
  //! Peer heap rkey.
  uint32_t rkey(int pe) const;
  //! lkey for a local buffer (heap or a registered scratch/return buffer); 0 if none.
  uint32_t lkey_for(const void *addr) const;

  // --- posting (return the wr_id used; 0 on post failure) -----------------
  uint64_t post_write(int lane, uintptr_t laddr, uint32_t lkey, uintptr_t raddr,
                      uint32_t rk, uint32_t len, bool inl, uint64_t wr_id);
  uint64_t post_read(int lane, uintptr_t laddr, uint32_t lkey, uintptr_t raddr,
                     uint32_t rk, uint32_t len, uint64_t wr_id);
  uint64_t post_fadd(int lane, uintptr_t result, uint32_t lkey, uintptr_t raddr,
                     uint32_t rk, uint64_t add, uint64_t wr_id);
  uint64_t post_cas(int lane, uintptr_t result, uint32_t lkey, uintptr_t raddr,
                    uint32_t rk, uint64_t cmp, uint64_t swp, uint64_t wr_id);

  // --- completion ---------------------------------------------------------
  //! Reap up to @p max completions from context @p cq_index's CQ.
  int poll(int cq_index, struct ibv_wc *wc, int max);

  const LaneMap &lanes() const { return lm_; }

 private:
  uint64_t post_atomic(int lane, int opcode, uintptr_t result, uint32_t lkey,
                       uintptr_t raddr, uint32_t rk, uint64_t compare_add,
                       uint64_t swap, uint64_t wr_id);

  Ibv *ibv_{nullptr};
  IbCtrl *ctrl_{nullptr};
  MrRegistry *mr_{nullptr};
  LaneMap lm_{};
  uintptr_t heap_base_{0};
  std::vector<LocalMr> local_mrs_{};
};

}  // namespace net
}  // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_NET_RMA_ENGINE_HPP_
