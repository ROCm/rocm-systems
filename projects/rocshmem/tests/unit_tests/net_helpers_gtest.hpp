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

#ifndef ROCSHMEM_NET_HELPERS_GTEST_HPP
#define ROCSHMEM_NET_HELPERS_GTEST_HPP

#include "gtest/gtest.h"

#include "net/addr_exchange.hpp"
#include "net/completion.hpp"
#include "net/lane_map.hpp"
#include "net/remote_region.hpp"

namespace rocshmem {

// Pure header-only logic (LaneMap, RemoteRegion, CompletionCounters,
// AllgatherFn helpers) -- transport-neutral, no NIC headers, no hardware
// required. Converted from net_loopback_tests/net_smoke.cpp.
class NetHelpersTestFixture : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

}  // namespace rocshmem

#endif  // ROCSHMEM_NET_HELPERS_GTEST_HPP
