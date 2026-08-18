/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/**
 * NIC transport tests against the real backend on a live NIC. Self-skips when no
 * NIC is present, so the same binary runs everywhere. The hermetic parser and
 * consumer tests live in unit/nic/nic_transport_test.cc.
 *
 * This drives the internal transport layer directly (no amdsmi_init), so it uses
 * a plain TEST rather than the TestBase lifecycle.
 */

#include <gtest/gtest.h>

#include "smi_nic.h"
#include "smi_nic_system.h"

namespace {

TEST(NicFunctionalReadOnly, TestNicTransportRead) {
  SmiNicSystem system;
  system.discover_nics();  // ctor registers the default vendor plugins
  const auto& nics = system.get_nics();
  if (nics.empty()) {
    GTEST_SKIP() << "no NIC present";
  }

  const auto& ports = nics[0]->nic_ports();
  if (ports.empty()) {
    GTEST_SKIP() << "NIC has no ports";
  }
  const SmiNicPort& port = ports[0];

  /**
   * The point is exercising the real transport (and the C-ABI exception path it
   * feeds) without crashing; exact values are device-specific. A permanent
   * address, if present, must be a well-formed MAC rather than fabricated zeros.
   */
  auto addr = port.permanent_address();
  if (addr.has_value()) {
    EXPECT_EQ(addr->size(), 17u);
    EXPECT_NE(*addr, "00:00:00:00:00:00");
  }
  EXPECT_NO_THROW((void)port.autoneg());
  EXPECT_NO_THROW((void)port.pause_params());
}

}  // namespace
