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

#ifndef ROCSHMEM_IBV_SHIM_GTEST_HPP
#define ROCSHMEM_IBV_SHIM_GTEST_HPP

#include "gtest/gtest.h"

#include "net/ibv.hpp"

namespace rocshmem {

// Exercises net::Ibv (the dlopen'd libibverbs ABI shim) against a real
// RoCE/IB device: device enumeration, open, PD/CQ/QP creation, and the inline
// post_send/poll_cq op-table dispatch. Converted from
// net_loopback_tests/ibv_smoke.cpp.
//
// Skips (does not fail) when no RDMA-capable device is present, since that is
// an environment property rather than a bug -- see SetUp().
class IbvShimTestFixture : public ::testing::Test {
 protected:
  net::Ibv ibv_;
  struct ibv_device **device_list_{nullptr};
  int num_devices_{0};
  struct ibv_context *ctx_{nullptr};

  void SetUp() override {
    if (!ibv_.load()) {
      GTEST_SKIP() << "libibverbs not available on this system";
    }
    device_list_ = ibv_.get_device_list(&num_devices_);
    if (!device_list_ || num_devices_ == 0) {
      GTEST_SKIP() << "no RDMA devices enumerated";
    }
  }

  void TearDown() override {
    if (ctx_) {
      ibv_.close_device(ctx_);
    }
    if (device_list_) {
      ibv_.free_device_list(device_list_);
    }
  }
};

}  // namespace rocshmem

#endif  // ROCSHMEM_IBV_SHIM_GTEST_HPP
