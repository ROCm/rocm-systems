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

#ifndef ROCSHMEM_WINDOW_INFO_VERBS_GTEST_HPP
#define ROCSHMEM_WINDOW_INFO_VERBS_GTEST_HPP

#include <cstdlib>
#include <cstring>
#include <memory>

#include "gtest/gtest.h"

#include "hdp_policy.hpp"
#include "net/ib_ctrl.hpp"
#include "net/ibv.hpp"
#include "net/lane_map.hpp"
#include "net/mr_registry.hpp"
#include "net/rma_engine.hpp"
#include "net/window_info_verbs.hpp"

namespace rocshmem {

// Exercises net::WindowInfoVerbs (the synchronous host-thread RMA/AMO policy
// built on top of RmaEngine): put_bytes/get_bytes/amo_fadd/amo_cas/put_nbi+
// quiet, over a single-process 2-lane RC loopback on a real RoCE/IB device.
// Converted from net_loopback_tests/wiv_test.cpp.
//
// Uses NoHdpPolicy explicitly (not the build-config HdpPolicy typedef), same
// as the original standalone test: the loopback heap here is host memory, not
// GPU memory, so no real HDP flush is meaningful regardless of build flags.
//
// Skips (does not fail) when no active RDMA port is found -- see SetUp().
class WindowInfoVerbsTestFixture : public ::testing::Test {
 protected:
  static constexpr size_t kHeapSize = 65536;
  static constexpr size_t kScratchSize = 64;

  net::Ibv ibv_;
  net::IbCtrl ctrl_;
  net::MrRegistry mr_;
  net::LaneMap lm_{1 /*ctx*/, 2 /*pes*/};
  net::RmaEngine rma_;
  NoHdpPolicy hdp_;
  void *heap_{nullptr};
  void *scratch_{nullptr};
  uint32_t scratch_lkey_{0};
  // WindowInfoVerbs is neither default-constructible nor copy-assignable
  // (its WindowInfo base deletes copy), so it is built once via SetUp().
  std::unique_ptr<WindowInfoVerbs> win_;

  void SetUp() override {
    if (!ibv_.load()) {
      GTEST_SKIP() << "libibverbs not available on this system";
    }
    if (!ctrl_.open(&ibv_, nullptr, 1)) {
      GTEST_SKIP() << "no active RDMA port found";
    }
    ASSERT_TRUE(ctrl_.create_queues(lm_, 64, 64));

    // Cross-connect the 2 lanes for RC loopback (lane0 <-> lane1).
    std::vector<net::DestInfo> d;
    ctrl_.fill_local_dest(d);
    ASSERT_TRUE(ctrl_.connect_qp(0, d[1]));
    ASSERT_TRUE(ctrl_.connect_qp(1, d[0]));

    // Host "symmetric heap".
    ASSERT_EQ(posix_memalign(&heap_, 4096, kHeapSize), 0);
    memset(heap_, 0, kHeapSize);
    ASSERT_TRUE(mr_.register_heap(&ibv_, const_cast<ibv_pd *>(ctrl_.dev().pd),
                                  heap_, kHeapSize, /*is_device=*/false,
                                  /*dmabuf=*/{}));

    // Fake 2-PE allgather: both peers' region = our own heap (loopback).
    auto self_allgather = [this](void *inout, size_t bpp) {
      struct HeapKey {
        uintptr_t base;
        uint64_t rkey;
      };
      HeapKey me{reinterpret_cast<uintptr_t>(heap_), mr_.heap_rkey()};
      for (int pe = 0; pe < 2; ++pe) {
        memcpy(static_cast<char *>(inout) + pe * bpp, &me, sizeof(HeapKey));
      }
    };
    ASSERT_TRUE(mr_.exchange_heap(2, 0, self_allgather));

    // AMO result scratch (host, registered).
    ASSERT_EQ(posix_memalign(&scratch_, 64, kScratchSize), 0);
    *static_cast<uint64_t *>(scratch_) = 0;
    scratch_lkey_ =
        mr_.register_local(&ibv_, const_cast<ibv_pd *>(ctrl_.dev().pd),
                           scratch_, kScratchSize, /*is_device=*/false,
                           /*dmabuf=*/{});
    ASSERT_NE(scratch_lkey_, 0u);

    std::vector<net::LocalMr> local_mrs{
        {reinterpret_cast<uintptr_t>(heap_),
         reinterpret_cast<uintptr_t>(heap_) + kHeapSize, mr_.heap_lkey()},
        {reinterpret_cast<uintptr_t>(scratch_),
         reinterpret_cast<uintptr_t>(scratch_) + kScratchSize, scratch_lkey_},
    };
    rma_ = net::RmaEngine(&ibv_, &ctrl_, &mr_, lm_,
                          reinterpret_cast<uintptr_t>(heap_), local_mrs);
    win_ = std::make_unique<WindowInfoVerbs>(&rma_, &hdp_, /*ctx=*/0, heap_,
                                             kHeapSize, scratch_,
                                             scratch_lkey_);
  }

  void TearDown() override {
    if (heap_) {
      free(heap_);
    }
    if (scratch_) {
      free(scratch_);
    }
  }
};

}  // namespace rocshmem

#endif  // ROCSHMEM_WINDOW_INFO_VERBS_GTEST_HPP
