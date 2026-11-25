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

#ifndef ROCSHMEM_RDMA_LOOPBACK_GTEST_HPP
#define ROCSHMEM_RDMA_LOOPBACK_GTEST_HPP

#include <cstdlib>
#include <cstring>

#include "gtest/gtest.h"

#include "net/ib_ctrl.hpp"
#include "net/ibv.hpp"
#include "net/lane_map.hpp"
#include "net/mr_registry.hpp"

namespace rocshmem {

// End-to-end RDMA_WRITE loopback over net::IbCtrl + net::MrRegistry: connects
// two lanes, registers a host "heap" buffer, posts a real RDMA_WRITE from one
// lane to the other, and verifies the completion + data landed correctly.
// Converted from net_loopback_tests/rdma_loopback.cpp.
//
// Skips (does not fail) when no active RDMA port is found -- see SetUp().
class RdmaLoopbackTestFixture : public ::testing::Test {
 protected:
  static constexpr size_t kHeapSize = 4096;

  net::Ibv ibv_;
  net::IbCtrl ctrl_;
  net::MrRegistry mr_;
  net::LaneMap lm_{1, 2};
  void *heap_{nullptr};

  void SetUp() override {
    if (!ibv_.load()) {
      GTEST_SKIP() << "libibverbs not available on this system";
    }
    if (!ctrl_.open(&ibv_, nullptr, 1)) {
      GTEST_SKIP() << "no active RDMA port found";
    }
    ASSERT_TRUE(ctrl_.create_queues(lm_, 64, 64));

    std::vector<net::DestInfo> d;
    ctrl_.fill_local_dest(d);
    ASSERT_TRUE(ctrl_.connect_qp(0, d[1]));
    ASSERT_TRUE(ctrl_.connect_qp(1, d[0]));

    ASSERT_EQ(posix_memalign(&heap_, 4096, kHeapSize), 0);
    memset(heap_, 0, kHeapSize);
    ASSERT_TRUE(mr_.register_heap(&ibv_, const_cast<ibv_pd *>(ctrl_.dev().pd),
                                  heap_, kHeapSize, /*is_device=*/false,
                                  /*dmabuf=*/{}));
  }

  void TearDown() override {
    if (heap_) {
      free(heap_);
    }
  }
};

}  // namespace rocshmem

#endif  // ROCSHMEM_RDMA_LOOPBACK_GTEST_HPP
