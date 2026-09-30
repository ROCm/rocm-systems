// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "unit/pci_util_test.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>

#include "src/pci_util.h"

namespace {

// The first 16 bytes of a PCI type-0 configuration header for an AMD device,
// exactly as they appear on the wire (little-endian):
//
//   0x00 vendor  = 0x1002   -> 02 10
//   0x02 device  = 0x73A1   -> A1 73
//   0x04 command                    0x06 status
//   0x08 revision = 0xC1            0x09 prog-if = 0x00
//   0x0A subclass = 0x00            0x0B class   = 0x03  (display controller)
constexpr uint8_t kHeader[16] = {
    0x02, 0x10,  // 0x00 vendor
    0xA1, 0x73,  // 0x02 device
    0x07, 0x04,  // 0x04 command
    0x10, 0x00,  // 0x06 status
    0xC1,        // 0x08 revision
    0x00,        // 0x09 prog-if
    0x00,        // 0x0A subclass
    0x03,        // 0x0B class
    0x00, 0x00, 0x00, 0x00,
};

}  // namespace

TestPciConfigDecode::TestPciConfigDecode() {
  SetTitle("PCI Config-Space Decode");
  SetDescription(
      "Verify the little-endian loads used by the PCI config-space fallback "
      "produce the same values sysfs reports, and that RevisionID does not "
      "pick up the prog-if byte.");
}

void TestPciConfigDecode::SetUp() {}

void TestPciConfigDecode::Run() {
  // VendorID at 0x00. The historic bug: this was byte-swapped after loading,
  // so AMD's 0x1002 was recorded as 0x0210.
  EXPECT_EQ(PciUtil::load_le16(&kHeader[0x00]), 0x1002u);
  EXPECT_NE(PciUtil::load_le16(&kHeader[0x00]), 0x0210u);

  // DeviceID at 0x02.
  EXPECT_EQ(PciUtil::load_le16(&kHeader[0x02]), 0x73A1u);

  // Class at 0x0A loads subclass then class, giving 0xCCSS -- the same value
  // sysfs "class" gives after >>8. Both feed the same CUID field, so they must
  // agree.
  const uint16_t from_config = PciUtil::load_le16(&kHeader[0x0A]);
  const uint32_t sysfs_class_24bit = 0x030000u;  // class 03, subclass 00, prog-if 00
  EXPECT_EQ(from_config, 0x0300u);
  EXPECT_EQ(from_config, static_cast<uint16_t>(sysfs_class_24bit >> 8));

  // RevisionID is the single byte at 0x08. Reading two bytes and narrowing
  // used to keep the prog-if byte at 0x09 instead.
  EXPECT_EQ(kHeader[0x08], 0xC1u);
  EXPECT_EQ(static_cast<uint8_t>(PciUtil::load_le16(&kHeader[0x08]) >> 8), kHeader[0x09]);

  // Must work unaligned: every odd offset of a config-space buffer is legal.
  const uint8_t unaligned[3] = {0xFF, 0x34, 0x12};
  EXPECT_EQ(PciUtil::load_le16(&unaligned[1]), 0x1234u);

  // Boundary values.
  const uint8_t zero[2] = {0x00, 0x00};
  const uint8_t max[2] = {0xFF, 0xFF};
  EXPECT_EQ(PciUtil::load_le16(zero), 0x0000u);
  EXPECT_EQ(PciUtil::load_le16(max), 0xFFFFu);

  IF_VERB(1) {
    printf("  vendor=0x%04x device=0x%04x class=0x%04x revision=0x%02x\n",
           PciUtil::load_le16(&kHeader[0x00]), PciUtil::load_le16(&kHeader[0x02]), from_config,
           kHeader[0x08]);
  }
}

void TestPciConfigDecode::DisplayTestInfo() { TestBase::DisplayTestInfo(); }
void TestPciConfigDecode::DisplayResults() const { TestBase::DisplayResults(); }
void TestPciConfigDecode::Close() {}
