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

#include "net/rma_engine.hpp"

#include <cstdio>
#include <cstring>

#include "net/remote_region.hpp"

namespace rocshmem {
namespace net {

namespace {
constexpr uint32_t kInlineMax = 64;
}  // namespace

RmaEngine::RmaEngine(Ibv *ibv, IbCtrl *ctrl, MrRegistry *mr, LaneMap lm,
                     uintptr_t heap_base, std::vector<LocalMr> local_mrs)
    : ibv_{ibv},
      ctrl_{ctrl},
      mr_{mr},
      lm_{lm},
      heap_base_{heap_base},
      local_mrs_{std::move(local_mrs)} {}

uintptr_t RmaEngine::remote_va(int pe, const void *local_symmetric_va) const {
  return net::remote_addr(mr_->remote(pe),
                          reinterpret_cast<uintptr_t>(local_symmetric_va),
                          heap_base_);
}

uint32_t RmaEngine::rkey(int pe) const {
  return static_cast<uint32_t>(mr_->remote(pe).key);
}

uint32_t RmaEngine::lkey_for(const void *addr) const {
  auto a = reinterpret_cast<uintptr_t>(addr);
  for (const auto &m : local_mrs_) {
    if (a >= m.base && a < m.end) {
      return m.lkey;
    }
  }
  fprintf(stderr, "[rocSHMEM] net: no MR for local addr %p\n", addr);
  return 0;
}

uint64_t RmaEngine::post_write(int lane, uintptr_t laddr, uint32_t lkey,
                               uintptr_t raddr, uint32_t rk, uint32_t len,
                               bool inl, uint64_t wr_id) {
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));
  sge.addr = laddr;
  sge.length = len;
  sge.lkey = lkey;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = wr_id;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = IBV_SEND_SIGNALED;
  if (inl && len <= kInlineMax) {
    wr.send_flags |= IBV_SEND_INLINE;
  }
  wr.wr.rdma.remote_addr = raddr;
  wr.wr.rdma.rkey = rk;

  struct ibv_send_wr *bad = nullptr;
  int rc = ibv_->post_send(ctrl_->qp(lane), &wr, &bad);
  if (rc != 0) {
    fprintf(stderr, "[rocSHMEM] net: post_write rc=%d (lane %d)\n", rc, lane);
    return 0;
  }
  return wr_id;
}

uint64_t RmaEngine::post_read(int lane, uintptr_t laddr, uint32_t lkey,
                              uintptr_t raddr, uint32_t rk, uint32_t len,
                              uint64_t wr_id) {
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));
  sge.addr = laddr;
  sge.length = len;
  sge.lkey = lkey;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = wr_id;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = raddr;
  wr.wr.rdma.rkey = rk;

  struct ibv_send_wr *bad = nullptr;
  int rc = ibv_->post_send(ctrl_->qp(lane), &wr, &bad);
  if (rc != 0) {
    fprintf(stderr, "[rocSHMEM] net: post_read rc=%d (lane %d)\n", rc, lane);
    return 0;
  }
  return wr_id;
}

uint64_t RmaEngine::post_atomic(int lane, int opcode, uintptr_t result,
                                uint32_t lkey, uintptr_t raddr, uint32_t rk,
                                uint64_t compare_add, uint64_t swap,
                                uint64_t wr_id) {
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));
  sge.addr = result;  // fetched value lands here (8 bytes)
  sge.length = 8;
  sge.lkey = lkey;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = wr_id;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = static_cast<enum ibv_wr_opcode>(opcode);
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.atomic.remote_addr = raddr;
  wr.wr.atomic.rkey = rk;
  wr.wr.atomic.compare_add = compare_add;
  wr.wr.atomic.swap = swap;

  struct ibv_send_wr *bad = nullptr;
  int rc = ibv_->post_send(ctrl_->qp(lane), &wr, &bad);
  if (rc != 0) {
    fprintf(stderr, "[rocSHMEM] net: post_atomic rc=%d (lane %d)\n", rc, lane);
    return 0;
  }
  return wr_id;
}

uint64_t RmaEngine::post_fadd(int lane, uintptr_t result, uint32_t lkey,
                              uintptr_t raddr, uint32_t rk, uint64_t add,
                              uint64_t wr_id) {
  return post_atomic(lane, IBV_WR_ATOMIC_FETCH_AND_ADD, result, lkey, raddr, rk,
                     /*compare_add=*/add, /*swap=*/0, wr_id);
}

uint64_t RmaEngine::post_cas(int lane, uintptr_t result, uint32_t lkey,
                             uintptr_t raddr, uint32_t rk, uint64_t cmp,
                             uint64_t swp, uint64_t wr_id) {
  return post_atomic(lane, IBV_WR_ATOMIC_CMP_AND_SWP, result, lkey, raddr, rk,
                     /*compare_add=*/cmp, /*swap=*/swp, wr_id);
}

int RmaEngine::poll(int cq_index, struct ibv_wc *wc, int max) {
  return ibv_->poll_cq(ctrl_->cq(cq_index), max, wc);
}

}  // namespace net
}  // namespace rocshmem
