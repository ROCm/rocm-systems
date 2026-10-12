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

#ifndef ROCSHMEM_LIBRARY_SRC_NET_WINDOW_INFO_VERBS_HPP_
#define ROCSHMEM_LIBRARY_SRC_NET_WINDOW_INFO_VERBS_HPP_

#include <cstddef>
#include <cstdint>

#include "hdp_policy.hpp"
#include "memory/window_info.hpp"
#include "net/rma_engine.hpp"

namespace rocshmem {

/**
 * @brief A host-context window backed by verbs (a WindowInfo subtype).
 *
 * Mirrors the IPC precedent: the host path detects the transport by the window
 * subtype (dynamic_cast) and branches to a native RMA implementation, leaving
 * the MPI WindowInfoMPI path untouched. This one drives synchronous verbs RMA
 * through the shared net::RmaEngine on the *calling host thread* -- each op posts
 * and then spins the context's CQ to completion (contrast the RO device proxy,
 * which is asynchronous and notifies the GPU).
 *
 * One instance per host context: it owns a context index (its CQ) and its lanes
 * (one QP per peer) come from RmaEngine's LaneMap. Barrier/sync/collectives are
 * NOT handled here -- HostInterface keeps doing those over the bootstrap.
 */
class WindowInfoVerbs : public WindowInfo {
 public:
  /**
   * @param rma        Shared RMA engine (non-owning; must outlive this).
   * @param hdp        HDP policy for GPU<->NIC visibility flushes (non-owning).
   * @param ctx        This host context's index (selects its CQ + lane row).
   * @param heap_base  Symmetric-heap base (for the WindowInfo range).
   * @param heap_size  Symmetric-heap size.
   * @param amo_scratch      Registered 8-byte scratch for fetching-AMO results
   *                         (host-readable); may be null until AMO is wired.
   * @param amo_scratch_lkey lkey of @p amo_scratch.
   * @param sq_depth         Send-queue depth of this context's QPs. Used as the
   *                         ceiling on in-flight (unreaped) ops so a non-blocking
   *                         burst can never overflow the SQ (which would make
   *                         ibv_post_send return ENOMEM and later deadlock the
   *                         completion wait).
   */
  WindowInfoVerbs(net::RmaEngine *rma, HdpPolicy *hdp, int ctx, void *heap_base,
                  size_t heap_size, void *amo_scratch,
                  uint32_t amo_scratch_lkey, int sq_depth)
      : WindowInfo(heap_base, heap_size),
        rma_{rma},
        hdp_{hdp},
        ctx_{ctx},
        amo_scratch_{amo_scratch},
        amo_scratch_lkey_{amo_scratch_lkey},
        sq_depth_{sq_depth} {}

  // --- blocking RMA (local-complete on return) ----------------------------
  void put(void *dst, const void *src, size_t n, int pe);
  void get(void *dst, const void *src, size_t n, int pe);

  // --- non-blocking RMA (completed by quiet) ------------------------------
  void put_nbi(void *dst, const void *src, size_t n, int pe);
  void get_nbi(void *dst, const void *src, size_t n, int pe);

  // --- ordering / completion ----------------------------------------------
  void quiet();
  void fence() { quiet(); }

  // --- 8-byte fetching atomics (result returned) --------------------------
  // Wired into HostInterface in a later phase; present so the class is complete.
  uint64_t amo_fadd(void *dst, uint64_t add, int pe);
  uint64_t amo_cas(void *dst, uint64_t cmp, uint64_t swp, int pe);

  // --- put-with-signal + signal query -------------------------------------
  //! Blocking put-with-signal: deliver @p n bytes to @p dst at @p pe, then
  //! update the remote 8-byte signal word @p sig_addr with @p signal according
  //! to @p sig_op (ROCSHMEM_SIGNAL_SET => ordered inline write,
  //! ROCSHMEM_SIGNAL_ADD => fetch-and-add). Local-complete on return.
  void put_signal(void *dst, const void *src, size_t n, uint64_t *sig_addr,
                  uint64_t signal, int sig_op, int pe);
  //! Non-blocking put-with-signal; completed by a later quiet().
  void put_signal_nbi(void *dst, const void *src, size_t n, uint64_t *sig_addr,
                      uint64_t signal, int sig_op, int pe);
  //! Local read of the signal word (HDP-flushed direct load of fine-grained
  //! host-coherent heap). No fabric op.
  uint64_t signal_fetch(const uint64_t *sig_addr);
  //! Spin on the local signal word until (word @p cmp @p cmp_val) holds.
  void signal_wait_until(const uint64_t *sig_addr, int cmp, uint64_t cmp_val);

 private:
  //! Reap up to a batch of completions from the context CQ, decrementing
  //! outstanding_ for each (and logging any error status). Returns the count.
  int reap();
  //! Post the signal update (SET or ADD) on @p lane, ordered after the payload.
  void post_signal(int lane, uint64_t *sig_addr, uint64_t signal, int sig_op,
                   int pe);
  //! Spin the context CQ until all posted ops on it have completed.
  void wait_all();
  //! Block until there is room in the SQ for at least one more post, so the
  //! next ibv_post_send cannot fail with ENOMEM on a full queue.
  void make_room();
  //! Account for a just-attempted post: bump outstanding_ on success, or log a
  //! post failure without bumping it (so wait_all/quiet can't hang forever).
  void account_post(uint64_t wr_id);
  //! Lane (QP) for a given peer within this context.
  int lane_for(int pe) const { return rma_->lanes().lane(ctx_, pe); }

  net::RmaEngine *rma_{nullptr};
  HdpPolicy *hdp_{nullptr};
  int ctx_{0};
  void *amo_scratch_{nullptr};
  uint32_t amo_scratch_lkey_{0};
  int sq_depth_{0};
  uint64_t outstanding_{0};
  uint64_t next_id_{1};
};

}  // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_NET_WINDOW_INFO_VERBS_HPP_
