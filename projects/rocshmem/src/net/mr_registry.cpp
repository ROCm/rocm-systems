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

#include "net/mr_registry.hpp"

#include <cstdio>

namespace rocshmem {
namespace net {

namespace {
constexpr int kHeapAccess = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                            IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
}  // namespace

MrRegistry::~MrRegistry() {
  if (!ibv_) {
    return;
  }
  for (auto *mr : local_mrs_) {
    if (mr) {
      ibv_->dereg_mr(mr);
    }
  }
  local_mrs_.clear();
  if (heap_mr_) {
    ibv_->dereg_mr(heap_mr_);
    heap_mr_ = nullptr;
  }
}

struct ibv_mr *MrRegistry::reg_region(Ibv *ibv, struct ibv_pd *pd, void *addr,
                                      size_t len, bool is_device,
                                      const DmabufFn &dmabuf, int access) {
  if (is_device && dmabuf) {
    int fd = -1;
    uint64_t offset = 0;
    if (dmabuf(addr, len, &fd, &offset)) {
      struct ibv_mr *mr = ibv->reg_dmabuf_mr(
          pd, offset, len, reinterpret_cast<uint64_t>(addr), fd, access);
      if (mr) {
        return mr;
      }
      fprintf(stderr,
              "[rocSHMEM] verbs: reg_dmabuf_mr failed; trying ibv_reg_mr\n");
    }
  }
  return ibv->reg_mr(pd, addr, len, access);
}

bool MrRegistry::register_heap(Ibv *ibv, struct ibv_pd *pd, void *base,
                               size_t size, bool is_device,
                               const DmabufFn &dmabuf) {
  ibv_ = ibv;
  local_base_ = reinterpret_cast<uintptr_t>(base);
  heap_mr_ = reg_region(ibv, pd, base, size, is_device, dmabuf, kHeapAccess);
  if (!heap_mr_) {
    fprintf(stderr, "[rocSHMEM] verbs: heap MR registration failed\n");
    return false;
  }
  heap_lkey_ = heap_mr_->lkey;
  heap_rkey_ = heap_mr_->rkey;
  return true;
}

bool MrRegistry::exchange_heap(int num_pes, int my_pe,
                               const AllgatherFn &allgather) {
  struct HeapKey {
    uintptr_t base;
    uint64_t rkey;
  };
  HeapKey mine{local_base_, heap_rkey_};
  std::vector<HeapKey> all =
      allgather_value<HeapKey>(mine, num_pes, my_pe, allgather);

  remote_.assign(static_cast<size_t>(num_pes), RemoteRegion{});
  for (int pe = 0; pe < num_pes; pe++) {
    remote_[pe].remote_base = all[pe].base;
    remote_[pe].key = all[pe].rkey;
  }
  return true;
}

uint32_t MrRegistry::register_local(Ibv *ibv, struct ibv_pd *pd, void *addr,
                                    size_t len, bool is_device,
                                    const DmabufFn &dmabuf) {
  ibv_ = ibv;
  // Local return buffers only need LOCAL_WRITE (RDMA_READ / fetching-AMO land
  // here), but registering with the full access set is harmless and uniform.
  struct ibv_mr *mr =
      reg_region(ibv, pd, addr, len, is_device, dmabuf, kHeapAccess);
  if (!mr) {
    fprintf(stderr, "[rocSHMEM] verbs: local buffer MR registration failed\n");
    return 0;
  }
  local_mrs_.push_back(mr);
  return mr->lkey;
}

}  // namespace net
}  // namespace rocshmem
