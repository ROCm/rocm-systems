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
 * NIC transport unit tests: the pure perm-addr parser and the SmiNicPort
 * consumer logic over an injected fake transport. Both groups are hermetic
 * (no device, no syscall). The live-backend counterpart lives in
 * functional/nic/transport/.
 */

#include <gtest/gtest.h>

#include <cerrno>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "smi_nic.h"
#include "smi_nic_system.h"
#include "smi_nic_transport.h"

namespace {

using amd::smi::nic::transport::DriverInfo;
using amd::smi::nic::transport::LinkSettings;
using amd::smi::nic::transport::NicTransport;
using amd::smi::nic::transport::parse_perm_addr;
using amd::smi::nic::transport::PauseParams;
using amd::smi::nic::transport::PermanentAddress;
using amd::smi::nic::transport::Result;
using amd::smi::nic::transport::VendorStatistics;

// Pure perm-addr parser (guards the size-check regression)

TEST(NicUnit, ParseZeroSizeFails) {
  const uint8_t bytes[6] = {0, 0, 0, 0, 0, 0};
  auto r = parse_perm_addr(0, bytes);
  EXPECT_FALSE(r.success);
  EXPECT_EQ(r.error_code, ENODATA);
}

TEST(NicUnit, ParseShortSizeFails) {
  const uint8_t bytes[6] = {1, 2, 3, 4, 5, 6};
  auto r = parse_perm_addr(4, bytes);
  EXPECT_FALSE(r.success);
  EXPECT_EQ(r.error_code, ENODATA);
}

TEST(NicUnit, ParseOversizeFails) {
  // The guard is !=, not <; a kernel reporting more than 6 bytes is rejected too.
  const uint8_t bytes[6] = {1, 2, 3, 4, 5, 6};
  auto r = parse_perm_addr(8, bytes);
  EXPECT_FALSE(r.success);
  EXPECT_EQ(r.error_code, ENODATA);
}

TEST(NicUnit, ParseSixBytesSucceeds) {
  const uint8_t bytes[6] = {0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff};
  auto r = parse_perm_addr(6, bytes);
  ASSERT_TRUE(r.success);
  EXPECT_EQ(r.value.mac[0], 0xaa);
  EXPECT_EQ(r.value.mac[5], 0xff);
}

TEST(NicUnit, ParseSixZeroBytesSucceeds) {
  /**
   * The size==6 contract is size-only: an all-zero MAC is a valid parse. The
   * "fail rather than fabricate zeros" rule applies to wrong sizes, not values.
   */
  const uint8_t bytes[6] = {0, 0, 0, 0, 0, 0};
  EXPECT_TRUE(parse_perm_addr(6, bytes).success);
}

// SmiNicPort consumer logic over an injected fake transport

/**
 * Returns canned Results so the port's formatting / optional-mapping can be
 * exercised without a device. Fields default to failure; set the ones a test
 * needs.
 */
class FakeTransport : public NicTransport {
 public:
  Result<PauseParams> pause{false, {}, ENODATA};
  Result<LinkSettings> link{false, {}, ENODATA};
  Result<PermanentAddress> perm{false, {}, ENODATA};

  Result<PauseParams> get_pause_params(const std::string&) override { return pause; }
  Result<LinkSettings> get_link_settings(const std::string&) override { return link; }
  Result<DriverInfo> get_driver_info(const std::string&) override { return {false, {}, ENODATA}; }
  Result<VendorStatistics> get_statistics(const std::string&) override {
    return {false, {}, ENODATA};
  }
  Result<PermanentAddress> get_permanent_address(const std::string&) override { return perm; }
  std::string backend_name() const override { return "fake"; }
};

SmiNicPort make_fake_port(std::shared_ptr<NicTransport> transport) {
  /**
   * The sysfs paths are intentionally bogus; the ctor reads them via
   * get_sysfs_data, which returns nullopt on a missing path (no throw).
   */
  return SmiNicPort("eth-test", "0000:00:00.0", "/nonexistent/class", "/nonexistent/bus",
                    std::move(transport));
}

TEST(NicUnit, ConsumerPermanentAddressFormatsMac) {
  auto fake = std::make_shared<FakeTransport>();
  PermanentAddress pa;
  /**
   * Mix hex-letter nibbles and a byte needing a leading zero (0x01) to catch
   * width/case regressions in the formatter.
   */
  pa.mac = {0xab, 0xcd, 0xef, 0x01, 0x23, 0x45};
  fake->perm = {true, pa, 0};

  auto port = make_fake_port(fake);
  auto addr = port.permanent_address();
  ASSERT_TRUE(addr.has_value());
  EXPECT_EQ(*addr, "ab:cd:ef:01:23:45");
}

TEST(NicUnit, ConsumerPermanentAddressNulloptOnFailure) {
  auto fake = std::make_shared<FakeTransport>();  // perm defaults to failure
  auto port = make_fake_port(fake);
  EXPECT_FALSE(port.permanent_address().has_value());
}

TEST(NicUnit, ConsumerAutonegMapsFromLinkSettings) {
  auto fake = std::make_shared<FakeTransport>();
  fake->link = {true, {}, 0};
  fake->link.value.autoneg = 1;

  auto port = make_fake_port(fake);
  auto an = port.autoneg();
  ASSERT_TRUE(an.has_value());
  EXPECT_TRUE(*an);
}

TEST(NicUnit, ConsumerAutonegFalseFromLinkSettings) {
  auto fake = std::make_shared<FakeTransport>();
  fake->link = {true, {}, 0};
  fake->link.value.autoneg = 0;

  auto port = make_fake_port(fake);
  auto an = port.autoneg();
  ASSERT_TRUE(an.has_value());
  EXPECT_FALSE(*an);
}

TEST(NicUnit, ConsumerAutonegNulloptOnFailure) {
  auto fake = std::make_shared<FakeTransport>();  // link defaults to failure
  auto port = make_fake_port(fake);
  EXPECT_FALSE(port.autoneg().has_value());
}

TEST(NicUnit, ConsumerPauseParamsPassThrough) {
  auto fake = std::make_shared<FakeTransport>();
  fake->pause = {true, {}, 0};
  fake->pause.value.rx_pause = true;
  fake->pause.value.tx_pause = false;

  auto port = make_fake_port(fake);
  auto p = port.pause_params();
  ASSERT_TRUE(p.has_value());
  EXPECT_TRUE(p->rx_pause);
  EXPECT_FALSE(p->tx_pause);
}

}  // namespace
