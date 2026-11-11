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

#ifndef ROCSHMEM_IB_CTRL_GTEST_HPP
#define ROCSHMEM_IB_CTRL_GTEST_HPP

#include "gtest/gtest.h"

#include "net/ib_ctrl.hpp"
#include "net/ibv.hpp"
#include "net/lane_map.hpp"

namespace rocshmem {

// Exercises net::IbCtrl's RC control plane on a real RoCE/IB device: bring up
// one CQ per context plus one RC QP per lane, then cross-connect two lanes
// (a single-process RC loopback) and drive them to RTS. Converted from
// net_loopback_tests/ibctrl_smoke.cpp.
//
// Skips (does not fail) when IbCtrl::open() finds no active RDMA port -- see
// SetUp().
class IbCtrlTestFixture : public ::testing::Test {
 protected:
  net::Ibv ibv_;
  net::IbCtrl ctrl_;
  net::LaneMap lm_{1 /*ctx*/, 2 /*lanes as pseudo-pes*/};

  void SetUp() override {
    if (!ibv_.load()) {
      GTEST_SKIP() << "libibverbs not available on this system";
    }
    if (!ctrl_.open(&ibv_, nullptr, 1)) {
      GTEST_SKIP() << "no active RDMA port found";
    }
    ASSERT_TRUE(ctrl_.create_queues(lm_, 64, 64));
  }
};

}  // namespace rocshmem

#endif  // ROCSHMEM_IB_CTRL_GTEST_HPP
