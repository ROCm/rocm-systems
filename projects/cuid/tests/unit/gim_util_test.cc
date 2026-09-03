// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "unit/gim_util_test.h"

#include <gtest/gtest.h>
#include <sys/stat.h>

#include <cstdint>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

#include "src/gim_util.h"

using cuid::gim::GimAsicInfo;
using cuid::gim::GimClient;
using cuid::gim::GimDeviceEntry;

namespace {

bool gim_dev_present() {
  struct stat st {};
  return ::stat("/dev/gim-smi0", &st) == 0;
}

}  // namespace

// ---------------------------------------------------------------------------
// TestGimClientAvailability
// ---------------------------------------------------------------------------
TestGimClientAvailability::TestGimClientAvailability() {
  SetTitle("GIM Client Availability");
  SetDescription(
      "Verify GimClient reports device-node presence and falls back to "
      "UNSUPPORTED when the GIM driver is absent.");
}

void TestGimClientAvailability::SetUp() {}

void TestGimClientAvailability::Run() {
  // is_available reports the presence of /dev/gim-smi0 and never throws.
  EXPECT_EQ(GimClient::is_available(), gim_dev_present());

  // get_asic_info_for_bdf() should validate BDF format before contacting the
  // driver, regardless of driver presence.
  {
    GimClient client;
    GimAsicInfo info;
    EXPECT_EQ(client.get_asic_info_for_bdf("not-a-bdf", info), AMDCUID_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(client.get_asic_info_for_bdf("", info), AMDCUID_STATUS_INVALID_ARGUMENT);
  }

  if (gim_dev_present()) {
    IF_VERB(1) { printf("  GIM device present; skipping absent-driver behavior checks\n"); }
    return;
  }

  // init() must return UNSUPPORTED on systems with no GIM driver, so callers
  // can safely treat the GIM backend as an optional fallback.
  {
    GimClient client;
    EXPECT_EQ(client.init(), AMDCUID_STATUS_UNSUPPORTED);
    EXPECT_FALSE(client.is_connected());
  }

  // get_devices() must clear the output vector and return UNSUPPORTED when GIM
  // is unavailable.
  {
    GimClient client;
    std::vector<GimDeviceEntry> devices = {{0xdead, "0000:00:00.0", false}};
    EXPECT_EQ(client.get_devices(devices), AMDCUID_STATUS_UNSUPPORTED);
    EXPECT_TRUE(devices.empty());
  }
}

// ---------------------------------------------------------------------------
// TestGimParseAsicSerial
// ---------------------------------------------------------------------------
TestGimParseAsicSerial::TestGimParseAsicSerial() {
  SetTitle("GIM Parse ASIC Serial");
  SetDescription(
      "Verify GimClient::parse_asic_serial accepts canonical hex forms and "
      "rejects malformed input without modifying the output on failure.");
}

void TestGimParseAsicSerial::SetUp() {}

void TestGimParseAsicSerial::Run() {
  // Accepted forms.
  {
    uint64_t value = 0;

    EXPECT_TRUE(GimClient::parse_asic_serial("0x1234abcd", value));
    EXPECT_EQ(value, 0x1234abcdu);

    EXPECT_TRUE(GimClient::parse_asic_serial("FFFFFFFFFFFFFFFF", value));
    EXPECT_EQ(value, UINT64_MAX);

    EXPECT_TRUE(GimClient::parse_asic_serial("0X1", value));
    EXPECT_EQ(value, 0x1u);
  }

  // Rejected forms.
  {
    uint64_t value = 0xCAFEu;

    // Empty input.
    EXPECT_FALSE(GimClient::parse_asic_serial("", value));
    EXPECT_EQ(value, 0xCAFEu) << "value must not be modified on failure";

    // Non-hex character.
    EXPECT_FALSE(GimClient::parse_asic_serial("0xZZZ", value));
    EXPECT_EQ(value, 0xCAFEu);

    // 0x prefix only.
    EXPECT_FALSE(GimClient::parse_asic_serial("0x", value));

    // More than 64 bits worth of hex digits.
    EXPECT_FALSE(GimClient::parse_asic_serial("0x10000000000000000", value));
  }
}

// ---------------------------------------------------------------------------
// TestGimFormatBdf
// ---------------------------------------------------------------------------
TestGimFormatBdf::TestGimFormatBdf() {
  SetTitle("GIM Format BDF");
  SetDescription(
      "Verify GimClient::format_bdf renders packed BDF values in canonical "
      "PCI dddd:bb:dd.f form.");
}

void TestGimFormatBdf::SetUp() {}

void TestGimFormatBdf::Run() {
  // domain=0x0000, bus=0x65, device=0x00, function=0
  const uint64_t packed =
      (uint64_t{0x0000} << 16) | (uint64_t{0x65} << 8) | (uint64_t{0x00} << 3) | uint64_t{0};
  EXPECT_EQ(GimClient::format_bdf(packed), "0000:65:00.0");

  // domain=0x1234, bus=0xab, device=0x1f, function=7 (max field values)
  const uint64_t packed2 =
      (uint64_t{0x1234} << 16) | (uint64_t{0xab} << 8) | (uint64_t{0x1f} << 3) | uint64_t{7};
  EXPECT_EQ(GimClient::format_bdf(packed2), "1234:ab:1f.7");
}

// ---------------------------------------------------------------------------
// TestGimDeviceEnumeration
// ---------------------------------------------------------------------------
TestGimDeviceEnumeration::TestGimDeviceEnumeration() {
  SetTitle("GIM Device Enumeration");
  SetDescription(
      "Enumerate GPUs from a live GIM driver and verify each has a unique, "
      "canonical BDF and a unique, parseable ASIC serial. Detects wire-ABI "
      "regressions that yield zero/garbage devices or collapsed identities.");
}

void TestGimDeviceEnumeration::SetUp() {}

void TestGimDeviceEnumeration::Run() {
  GimClient client;
  ASSERT_EQ(client.init(), AMDCUID_STATUS_SUCCESS)
      << "GIM handshake must succeed when the device node is present and "
         "running as root";
  EXPECT_TRUE(client.is_connected());

  std::vector<GimDeviceEntry> devices;
  // A wrong smi_server_static_info size makes the driver reject the request
  // (status != SUCCESS); a wrong device-entry stride yields garbage entries.
  ASSERT_EQ(client.get_devices(devices), AMDCUID_STATUS_SUCCESS);
  ASSERT_FALSE(devices.empty()) << "GIM reported zero devices";

  std::set<std::string> seen_bdfs;
  std::set<uint64_t> seen_serials;
  for (const auto &dev : devices) {
    // Canonical "dddd:bb:dd.f" is exactly 12 characters and must be unique.
    EXPECT_EQ(dev.bdf.size(), 12u) << "malformed BDF: '" << dev.bdf << "'";
    EXPECT_NE(dev.bdf.find(':'), std::string::npos)
        << "malformed BDF: '" << dev.bdf << "'";
    EXPECT_TRUE(seen_bdfs.insert(dev.bdf).second)
        << "duplicate BDF indicates a wrong ABI stride: " << dev.bdf;

    GimAsicInfo info;
    ASSERT_EQ(client.get_asic_info(dev.dev_id, info), AMDCUID_STATUS_SUCCESS)
        << "GET_ASIC_INFO failed for " << dev.bdf;
    EXPECT_NE(info.vendor_id, 0u) << "zero vendor id for " << dev.bdf;
    EXPECT_FALSE(info.asic_serial.empty()) << "empty serial for " << dev.bdf;

    uint64_t serial = 0;
    EXPECT_TRUE(GimClient::parse_asic_serial(info.asic_serial, serial))
        << "unparseable serial '" << info.asic_serial << "' for " << dev.bdf;
    // The per-GPU ASIC serial is the CUID hardware fingerprint; duplicates
    // would collapse multiple GPUs into a single CUID.
    EXPECT_TRUE(seen_serials.insert(serial).second)
        << "duplicate ASIC serial across devices: " << info.asic_serial;
  }
}
