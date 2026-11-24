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

#include "net/window_info_verbs.hpp"

#include "log.hpp"

namespace rocshmem {

int WindowInfoVerbs::reap() {
  struct ibv_wc wc[16];
  int n = rma_->poll(ctx_, wc, 16);
  for (int i = 0; i < n; i++) {
    if (wc[i].status != IBV_WC_SUCCESS) {
      LOG_ERROR("host-verbs: WC error status=%d wr_id=%lu", wc[i].status,
                static_cast<unsigned long>(wc[i].wr_id));
    }
    if (outstanding_ > 0) {
      outstanding_--;
    }
  }
  return n;
}

void WindowInfoVerbs::wait_all() {
  while (outstanding_ > 0) {
    reap();
  }
}

void WindowInfoVerbs::make_room() {
  // Leave headroom for one more WR. Without this a non-blocking burst of more
  // than sq_depth_ posts would make ibv_post_send return ENOMEM; that post is
  // dropped but a naive caller would still count it outstanding and then hang in
  // wait_all()/quiet() waiting for a completion that never arrives.
  while (sq_depth_ > 0 &&
         outstanding_ >= static_cast<uint64_t>(sq_depth_)) {
    reap();
  }
}

void WindowInfoVerbs::account_post(uint64_t wr_id) {
  // post_* return 0 on ibv_post_send failure (wr_ids we hand out are always
  // non-zero). Only count a successful post as outstanding so a failed post
  // can't wedge wait_all() forever.
  if (wr_id != 0) {
    outstanding_++;
  } else {
    LOG_ERROR("host-verbs: post failed on ctx %d (dropped)", ctx_);
  }
}

void WindowInfoVerbs::put(void *dst, const void *src, size_t n, int pe) {
  // Ensure the NIC reads the GPU's latest write to the (symmetric-heap) source.
  hdp_->hdp_flush();
  make_room();
  int lane = lane_for(pe);
  account_post(rma_->post_write(lane, reinterpret_cast<uintptr_t>(src),
                                rma_->lkey_for(src), rma_->remote_va(pe, dst),
                                rma_->rkey(pe), static_cast<uint32_t>(n),
                                next_id_++));
  wait_all();  // blocking: local completion on return
}

void WindowInfoVerbs::get(void *dst, const void *src, size_t n, int pe) {
  make_room();
  int lane = lane_for(pe);
  account_post(rma_->post_read(lane, reinterpret_cast<uintptr_t>(dst),
                               rma_->lkey_for(dst), rma_->remote_va(pe, src),
                               rma_->rkey(pe), static_cast<uint32_t>(n),
                               next_id_++));
  wait_all();
  // Make the NIC-written data visible to the GPU that will consume it.
  hdp_->hdp_flush();
}

void WindowInfoVerbs::put_nbi(void *dst, const void *src, size_t n, int pe) {
  hdp_->hdp_flush();
  make_room();
  int lane = lane_for(pe);
  account_post(rma_->post_write(lane, reinterpret_cast<uintptr_t>(src),
                                rma_->lkey_for(src), rma_->remote_va(pe, dst),
                                rma_->rkey(pe), static_cast<uint32_t>(n),
                                next_id_++));
}

void WindowInfoVerbs::get_nbi(void *dst, const void *src, size_t n, int pe) {
  make_room();
  int lane = lane_for(pe);
  account_post(rma_->post_read(lane, reinterpret_cast<uintptr_t>(dst),
                               rma_->lkey_for(dst), rma_->remote_va(pe, src),
                               rma_->rkey(pe), static_cast<uint32_t>(n),
                               next_id_++));
}

void WindowInfoVerbs::quiet() {
  wait_all();
  hdp_->hdp_flush();  // any completed gets are now GPU-visible
}

// TODO: fetching AMOs land their result in this context's single 8-byte
// amo_scratch_ slot, so only one may be in flight per context -- these are
// blocking today, which makes that safe. A future non-blocking fetching-AMO
// API would need a ring of result slots per context to avoid serializing them.
uint64_t WindowInfoVerbs::amo_fadd(void *dst, uint64_t add, int pe) {
  make_room();
  int lane = lane_for(pe);
  account_post(rma_->post_fadd(lane, reinterpret_cast<uintptr_t>(amo_scratch_),
                               amo_scratch_lkey_, rma_->remote_va(pe, dst),
                               rma_->rkey(pe), add, next_id_++));
  wait_all();
  return *reinterpret_cast<volatile uint64_t *>(amo_scratch_);
}

uint64_t WindowInfoVerbs::amo_cas(void *dst, uint64_t cmp, uint64_t swp,
                                  int pe) {
  make_room();
  int lane = lane_for(pe);
  account_post(rma_->post_cas(lane, reinterpret_cast<uintptr_t>(amo_scratch_),
                              amo_scratch_lkey_, rma_->remote_va(pe, dst),
                              rma_->rkey(pe), cmp, swp, next_id_++));
  wait_all();
  return *reinterpret_cast<volatile uint64_t *>(amo_scratch_);
}

}  // namespace rocshmem
