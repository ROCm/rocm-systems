// SPDX-License-Identifier: MIT
/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Unit tests for the PCI VPD parser (smi_nic_vpd). Hardware-independent: every
 * case feeds a synthetic VPD image built to the PCI resource-tag layout, so the
 * tests validate the TLV walk (keyword-length honouring, truncation safety),
 * not any real device.
 */

#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "smi_nic_vpd.h"

namespace vpd = amd::smi::nic::vpd;

// Test Infrastructure

static int tests_run = 0;
static int tests_failed = 0;

static void check(const std::string& name, bool passed, const std::string& detail = "") {
  tests_run++;
  if (!passed) {
    tests_failed++;
  }
  std::cout << (passed ? "  PASS: " : "  FAIL: ") << name;
  if (!detail.empty()) {
    std::cout << " - " << detail;
  }
  std::cout << "\n";
}

static bool eq(const std::optional<std::string>& got, const std::string& want) {
  return got.has_value() && got.value() == want;
}

// VPD image builders (PCI Local Bus spec resource tags).

static void put_large(std::vector<uint8_t>& v, uint8_t item, const std::vector<uint8_t>& data) {
  v.push_back(static_cast<uint8_t>(0x80 | item));
  v.push_back(static_cast<uint8_t>(data.size() & 0xff));
  v.push_back(static_cast<uint8_t>((data.size() >> 8) & 0xff));
  v.insert(v.end(), data.begin(), data.end());
}

static void put_keyword(std::vector<uint8_t>& v, const char* key, const std::string& val) {
  v.push_back(static_cast<uint8_t>(key[0]));
  v.push_back(static_cast<uint8_t>(key[1]));
  v.push_back(static_cast<uint8_t>(val.size()));
  v.insert(v.end(), val.begin(), val.end());
}

static std::vector<uint8_t> bytes(const std::string& s) { return {s.begin(), s.end()}; }

static const uint8_t kEndTag = 0x78;
static const uint8_t kIdentifier = 0x02;
static const uint8_t kVpdR = 0x10;

int main() {
  std::cout << "PCI VPD Parser Tests\n";
  std::cout << "====================\n";

  // A well-formed image: identifier string + VPD-R with PN/SN + end tag.
  {
    std::vector<uint8_t> vpd_r;
    put_keyword(vpd_r, "PN", "BCM957608-P1400GDF00");
    put_keyword(vpd_r, "SN", "P1400244200072FG");
    put_keyword(vpd_r, "RV", "\x01");  // checksum keyword the parser ignores

    std::vector<uint8_t> img;
    put_large(img, kIdentifier, bytes("Broadcom BCM57608 1x400G QSFP-DD PCIe Ethernet NIC"));
    put_large(img, kVpdR, vpd_r);
    img.push_back(kEndTag);

    auto f = vpd::parse_pci_vpd(img);
    check("product_name parsed",
          eq(f.product_name, "Broadcom BCM57608 1x400G QSFP-DD PCIe Ethernet NIC"));
    check("part_number parsed", eq(f.part_number, "BCM957608-P1400GDF00"));
    check("serial_number parsed", eq(f.serial_number, "P1400244200072FG"));
  }

  // Trailing space/NUL padding is trimmed.
  {
    std::vector<uint8_t> vpd_r;
    put_keyword(vpd_r, "PN", "AOC-S100G-b2C   ");
    put_keyword(vpd_r, "SN", std::string("OA24CS041598\0\0", 14));

    std::vector<uint8_t> img;
    put_large(img, kIdentifier, bytes("Supermicro Network Adapter  "));
    put_large(img, kVpdR, vpd_r);
    img.push_back(kEndTag);

    auto f = vpd::parse_pci_vpd(img);
    check("product_name trimmed", eq(f.product_name, "Supermicro Network Adapter"));
    check("part_number trimmed", eq(f.part_number, "AOC-S100G-b2C"));
    check("serial_number trimmed", eq(f.serial_number, "OA24CS041598"));
  }

  // Identifier only: PN/SN stay absent.
  {
    std::vector<uint8_t> img;
    put_large(img, kIdentifier, bytes("Some NIC"));
    img.push_back(kEndTag);

    auto f = vpd::parse_pci_vpd(img);
    check("identifier-only product_name", eq(f.product_name, "Some NIC"));
    check("identifier-only no part_number", !f.part_number.has_value());
    check("identifier-only no serial_number", !f.serial_number.has_value());
  }

  // VPD-R only: product name stays absent.
  {
    std::vector<uint8_t> vpd_r;
    put_keyword(vpd_r, "SN", "SERIAL123");
    std::vector<uint8_t> img;
    put_large(img, kVpdR, vpd_r);
    img.push_back(kEndTag);

    auto f = vpd::parse_pci_vpd(img);
    check("vpd-r-only no product_name", !f.product_name.has_value());
    check("vpd-r-only no part_number", !f.part_number.has_value());
    check("vpd-r-only serial_number", eq(f.serial_number, "SERIAL123"));
  }

  // Empty keyword value is treated as absent, not empty string.
  {
    std::vector<uint8_t> vpd_r;
    put_keyword(vpd_r, "PN", "");
    std::vector<uint8_t> img;
    put_large(img, kVpdR, vpd_r);
    img.push_back(kEndTag);

    auto f = vpd::parse_pci_vpd(img);
    check("empty keyword => absent", !f.part_number.has_value());
  }

  // A keyword whose declared length runs past the VPD-R block: parsing stops at
  // the bad entry, but an earlier valid keyword is still returned.
  {
    std::vector<uint8_t> img;
    img.push_back(static_cast<uint8_t>(0x80 | kVpdR));
    // Block payload: a valid PN, then an "SN" with an oversized length byte.
    std::vector<uint8_t> payload;
    put_keyword(payload, "PN", "GOODPN");
    payload.push_back('S');
    payload.push_back('N');
    payload.push_back(200);  // claims 200 bytes but the block ends here
    img.push_back(static_cast<uint8_t>(payload.size() & 0xff));
    img.push_back(static_cast<uint8_t>((payload.size() >> 8) & 0xff));
    img.insert(img.end(), payload.begin(), payload.end());
    img.push_back(kEndTag);

    auto f = vpd::parse_pci_vpd(img);
    check("truncated keyword: earlier PN kept", eq(f.part_number, "GOODPN"));
    check("truncated keyword: bad SN dropped", !f.serial_number.has_value());
  }

  // A large-resource length that runs past the image ends the walk safely.
  {
    std::vector<uint8_t> img;
    img.push_back(static_cast<uint8_t>(0x80 | kIdentifier));
    img.push_back(0xff);  // length low
    img.push_back(0xff);  // length high => far past the buffer
    const auto tail = bytes("short");
    img.insert(img.end(), tail.begin(), tail.end());

    auto f = vpd::parse_pci_vpd(img);
    check("oversized resource length: no crash, no fields", !f.product_name.has_value());
  }

  // Empty image yields all-absent fields.
  {
    auto f = vpd::parse_pci_vpd({});
    check("empty image: all absent", !f.product_name.has_value() && !f.part_number.has_value() &&
                                         !f.serial_number.has_value());
  }

  std::cout << "====================\n";
  std::cout << "Ran " << tests_run << " checks, " << tests_failed << " failed\n";
  return tests_failed == 0 ? 0 : 1;
}
