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

#include "net/verbs_host.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace rocshmem {
namespace net {

namespace {
constexpr int kCqDepth = 256;
constexpr int kSqDepth = 256;
constexpr size_t kAmoSlot = 8;  // one 8-byte AMO/g result slot per host context
}  // namespace

VerbsHost::~VerbsHost() {
  // IbCtrl / MrRegistry / Ibv destructors release QPs/CQs/MRs/device.
  if (amo_scratch_) {
    free(amo_scratch_);
    amo_scratch_ = nullptr;
  }
}

bool VerbsHost::init(HdpPolicy *hdp, void *heap_base, size_t heap_size,
                     bool heap_device, int num_pes, int my_pe, int num_host_ctx,
                     AllgatherFn allgather, DmabufFn dmabuf,
                     const char *dev_name) {
  if (!ibv_.load()) {
    fprintf(stderr, "[rocSHMEM] host-verbs: libibverbs load failed\n");
    return false;
  }
  if (!ctrl_.open(&ibv_, dev_name, 1)) {
    fprintf(stderr, "[rocSHMEM] host-verbs: device bring-up failed\n");
    return false;
  }
  lm_ = LaneMap(num_host_ctx, num_pes);
  if (!ctrl_.create_queues(lm_, kCqDepth, kSqDepth)) {
    fprintf(stderr, "[rocSHMEM] host-verbs: queue creation failed\n");
    return false;
  }

  // Heap MR (dma-buf for device memory).
  if (!mr_.register_heap(&ibv_, const_cast<ibv_pd *>(ctrl_.dev().pd), heap_base,
                         heap_size, heap_device, dmabuf)) {
    fprintf(stderr, "[rocSHMEM] host-verbs: heap registration failed\n");
    return false;
  }

  // AMO/g result scratch: one host buffer, per-context 8-byte slots (host memory
  // so the calling thread can read the fetched value back).
  size_t scratch_bytes = static_cast<size_t>(num_host_ctx) * kAmoSlot;
  if (posix_memalign(&amo_scratch_, 64, scratch_bytes) != 0) {
    fprintf(stderr, "[rocSHMEM] host-verbs: amo scratch alloc failed\n");
    return false;
  }
  memset(amo_scratch_, 0, scratch_bytes);
  amo_scratch_lkey_ =
      mr_.register_local(&ibv_, const_cast<ibv_pd *>(ctrl_.dev().pd),
                         amo_scratch_, scratch_bytes, /*is_device=*/false, {});

  std::vector<LocalMr> local_mrs{
      {reinterpret_cast<uintptr_t>(heap_base),
       reinterpret_cast<uintptr_t>(heap_base) + heap_size, mr_.heap_lkey()},
      {reinterpret_cast<uintptr_t>(amo_scratch_),
       reinterpret_cast<uintptr_t>(amo_scratch_) + scratch_bytes,
       amo_scratch_lkey_},
  };

  // Exchange heap {base,rkey} and connect the RC mesh.
  if (!mr_.exchange_heap(num_pes, my_pe, allgather)) {
    fprintf(stderr, "[rocSHMEM] host-verbs: heap key exchange failed\n");
    return false;
  }
  std::vector<DestInfo> local, global;
  ctrl_.fill_local_dest(local);
  global = allgather_array<DestInfo>(local, num_pes, my_pe, lm_.num_lanes(),
                                     allgather);
  if (!ctrl_.connect(my_pe, global)) {
    fprintf(stderr, "[rocSHMEM] host-verbs: RC connect failed\n");
    return false;
  }

  rma_ = RmaEngine(&ibv_, &ctrl_, &mr_, lm_,
                   reinterpret_cast<uintptr_t>(heap_base), std::move(local_mrs));

  // One WindowInfoVerbs per host context (its own CQ/lane row + scratch slot).
  pool_.reserve(static_cast<size_t>(num_host_ctx));
  avail_.assign(static_cast<size_t>(num_host_ctx), true);
  for (int c = 0; c < num_host_ctx; c++) {
    void *slot = static_cast<char *>(amo_scratch_) + static_cast<size_t>(c) * kAmoSlot;
    pool_.push_back(std::make_unique<WindowInfoVerbs>(
        &rma_, hdp, c, heap_base, heap_size, slot, amo_scratch_lkey_));
  }

  initialized_ = true;
  fprintf(stderr,
          "[rocSHMEM] host-verbs: initialized (%d host contexts, %d PEs)\n",
          num_host_ctx, num_pes);
  return true;
}

WindowInfoVerbs *VerbsHost::acquire() {
  for (size_t i = 0; i < pool_.size(); i++) {
    if (avail_[i]) {
      avail_[i] = false;
      return pool_[i].get();
    }
  }
  return nullptr;
}

void VerbsHost::release(WindowInfoVerbs *w) {
  for (size_t i = 0; i < pool_.size(); i++) {
    if (pool_[i].get() == w) {
      avail_[i] = true;
      return;
    }
  }
}

}  // namespace net
}  // namespace rocshmem
