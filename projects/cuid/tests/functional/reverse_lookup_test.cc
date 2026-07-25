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

#include "functional/reverse_lookup_test.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>

#include "src/cuid_util.h"
#include "src/pci_util.h"
#include "src/smbios_util.h"

// Strip UUIDv8 overhead from a primary CUID to recover the 122 raw data bits.
// Raw bit layout (after stripping):
//   [0..7]   serial number (uint64_t, little-endian)
//   [8]      unit_id lower 8 bits (raw bits 64:71)
//   [9]      revision_id (uint8_t) (raw bits 72:79)
//   [10..11] device_id (uint16_t, little-endian) (raw bits 80:95)
//   [12..13] vendor_id (uint16_t, little-endian) (raw bits 96:111)
//   [14]     bits[4:0]=unit_id upper 5 bits | bit[5]=aux_indicator |
//            bits[7:6]=device_type[1:0]
//   [15]     bits[7:6]=device_type[3:2] | bits[5:0]=padding
static void extract_primary_raw_bits(const amdcuid_id_t& uuid, uint8_t raw_bits[16]) {
  amdcuid_id_t mutable_uuid = uuid;
  CuidUtilities::remove_UUIDv8_bits(&mutable_uuid, raw_bits);
}

// ---------------------------------------------------------------------------
// TestReverseSerialNumber
// ---------------------------------------------------------------------------

TestReverseSerialNumber::TestReverseSerialNumber() {
  SetTitle("Reverse Lookup — Serial Number");
  SetDescription(
      "Verify that the serial number embedded in the primary CUID matches "
      "AMDCUID_QUERY_HARDWARE_FINGERPRINT for each device.");
}

void TestReverseSerialNumber::Run() {
  if (device_handles_.empty()) {
    GTEST_SKIP() << "No devices found; skipping.";
  }

  for (size_t i = 0; i < device_handles_.size(); ++i) {
    char device_node[256] = {0};
    uint32_t node_len = sizeof(device_node);
    amdcuid_query_device_property(device_handles_[i], AMDCUID_QUERY_DEVICE_PATH, device_node,
                                  &node_len);

    uint64_t serial_number = 0;
    uint32_t length = sizeof(serial_number);
    amdcuid_status_t status = amdcuid_query_device_property(
        device_handles_[i], AMDCUID_QUERY_HARDWARE_FINGERPRINT, &serial_number, &length);

    if (status == AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND) {
      // Expect a temporary CUID; build the same fallback fingerprint the
      // library uses so we have something to compare against.
      bool is_temporary = false;
      length = sizeof(is_temporary);
      status = amdcuid_query_device_property(device_handles_[i], AMDCUID_QUERY_TEMPORARY_CUID,
                                             &is_temporary, &length);
      EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);
      EXPECT_TRUE(is_temporary);

      amdcuid_device_type_t device_type;
      length = sizeof(device_type);
      amdcuid_query_device_property(device_handles_[i], AMDCUID_QUERY_DEVICE_TYPE, &device_type,
                                    &length);

      switch (device_type) {
        case AMDCUID_DEVICE_TYPE_PLATFORM: {
          std::string name, family_dummy;
          status = SmbiosUtil::get_product_info(name, family_dummy);
          EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);
          CuidUtilities::make_fallback_fingerprint(name, serial_number);
        } break;
        case AMDCUID_DEVICE_TYPE_CPU: {
          uint16_t physical_id = 0, core_id = 0;
          length = sizeof(physical_id);
          amdcuid_query_device_property(device_handles_[i], AMDCUID_QUERY_PHYSICAL_ID, &physical_id,
                                        &length);
          length = sizeof(core_id);
          amdcuid_query_device_property(device_handles_[i], AMDCUID_QUERY_CORE_ID, &core_id,
                                        &length);
          std::string physical_core_id =
              std::to_string(physical_id) + ":" + std::to_string(core_id);
          CuidUtilities::make_fallback_fingerprint(physical_core_id, serial_number);
        } break;
        case AMDCUID_DEVICE_TYPE_GPU:
        case AMDCUID_DEVICE_TYPE_NIC:
        case AMDCUID_DEVICE_TYPE_NPU: {
          char bdf[32] = {0};
          length = sizeof(bdf);
          amdcuid_query_device_property(device_handles_[i], AMDCUID_QUERY_BDF, bdf, &length);
          CuidUtilities::make_fallback_fingerprint(bdf, serial_number);
        } break;
        default:
          FAIL() << "Unsupported device type for fallback fingerprint";
      }
    } else {
      EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);
      EXPECT_NE(serial_number, 0u);
    }

    amdcuid_id_t primary_id = {};
    length = sizeof(primary_id);
    status = amdcuid_query_device_property(device_handles_[i], AMDCUID_QUERY_PRIMARY_CUID,
                                           &primary_id, &length);
    EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);

    uint8_t raw_bits[16] = {0};
    extract_primary_raw_bits(primary_id, raw_bits);
    uint64_t extracted_serial = 0;
    memcpy(&extracted_serial, raw_bits, sizeof(extracted_serial));

    EXPECT_EQ(serial_number, extracted_serial)
        << "Serial number mismatch for device " << device_node;

    IF_VERB(1) {
      printf("  Device [%s] serial: 0x%016llx (extracted: 0x%016llx)\n", device_node,
             (unsigned long long)serial_number, (unsigned long long)extracted_serial);
    }
  }
}

// ---------------------------------------------------------------------------
// TestReverseVendorId
// ---------------------------------------------------------------------------

TestReverseVendorId::TestReverseVendorId() {
  SetTitle("Reverse Lookup — Vendor ID");
  SetDescription(
      "Verify that the vendor ID embedded in the primary CUID matches "
      "AMDCUID_QUERY_VENDOR_ID for each device.");
}

void TestReverseVendorId::Run() {
  if (device_handles_.empty()) {
    GTEST_SKIP() << "No devices found; skipping.";
  }

  for (size_t i = 0; i < device_handles_.size(); ++i) {
    char device_node[256] = {0};
    uint32_t node_len = sizeof(device_node);
    amdcuid_query_device_property(device_handles_[i], AMDCUID_QUERY_DEVICE_PATH, device_node,
                                  &node_len);

    amdcuid_id_t primary_id;
    uint32_t length = sizeof(primary_id);
    amdcuid_status_t status = amdcuid_query_device_property(
        device_handles_[i], AMDCUID_QUERY_PRIMARY_CUID, &primary_id, &length);
    EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);

    uint8_t raw_bits[16] = {0};
    extract_primary_raw_bits(primary_id, raw_bits);
    uint16_t extracted_vendor =
        static_cast<uint16_t>(raw_bits[12]) | (static_cast<uint16_t>(raw_bits[13]) << 8);

    uint16_t queried_vendor = 0;
    length = sizeof(queried_vendor);
    status = amdcuid_query_device_property(device_handles_[i], AMDCUID_QUERY_VENDOR_ID,
                                           &queried_vendor, &length);
    EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);
    EXPECT_EQ(extracted_vendor, queried_vendor) << "Vendor ID mismatch for device " << device_node;

    IF_VERB(1) { printf("  Device [%s] vendor_id: 0x%04x\n", device_node, queried_vendor); }
  }
}

// ---------------------------------------------------------------------------
// TestReverseDeviceId
// ---------------------------------------------------------------------------

TestReverseDeviceId::TestReverseDeviceId() {
  SetTitle("Reverse Lookup — Device ID");
  SetDescription(
      "Verify that the device ID embedded in the primary CUID matches "
      "AMDCUID_QUERY_DEVICE_ID for each supporting device type.");
}

void TestReverseDeviceId::Run() {
  if (device_handles_.empty()) {
    GTEST_SKIP() << "No devices found; skipping.";
  }

  for (size_t i = 0; i < device_handles_.size(); ++i) {
    char device_node[256] = {0};
    uint32_t node_len = sizeof(device_node);
    amdcuid_query_device_property(device_handles_[i], AMDCUID_QUERY_DEVICE_PATH, device_node,
                                  &node_len);

    amdcuid_id_t primary_id;
    uint32_t length = sizeof(primary_id);
    amdcuid_status_t status = amdcuid_query_device_property(
        device_handles_[i], AMDCUID_QUERY_PRIMARY_CUID, &primary_id, &length);
    EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);

    uint8_t raw_bits[16] = {0};
    extract_primary_raw_bits(primary_id, raw_bits);
    uint16_t extracted_device_id =
        static_cast<uint16_t>(raw_bits[10]) | (static_cast<uint16_t>(raw_bits[11]) << 8);

    uint16_t queried_device_id = 0;
    length = sizeof(queried_device_id);
    status = amdcuid_query_device_property(device_handles_[i], AMDCUID_QUERY_DEVICE_ID,
                                           &queried_device_id, &length);
    if (status == AMDCUID_STATUS_WRONG_DEVICE_TYPE) {
      continue;
    }
    EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);
    EXPECT_EQ(extracted_device_id, queried_device_id)
        << "Device ID mismatch for device " << device_node;

    IF_VERB(1) { printf("  Device [%s] device_id: 0x%04x\n", device_node, queried_device_id); }
  }
}

// ---------------------------------------------------------------------------
// TestReverseRevisionId
// ---------------------------------------------------------------------------

TestReverseRevisionId::TestReverseRevisionId() {
  SetTitle("Reverse Lookup — Revision ID");
  SetDescription(
      "Verify that the revision ID embedded in the primary CUID matches "
      "AMDCUID_QUERY_REVISION_ID for each supporting device type.");
}

void TestReverseRevisionId::Run() {
  if (device_handles_.empty()) {
    GTEST_SKIP() << "No devices found; skipping.";
  }

  for (size_t i = 0; i < device_handles_.size(); ++i) {
    char device_node[256] = {0};
    uint32_t node_len = sizeof(device_node);
    amdcuid_query_device_property(device_handles_[i], AMDCUID_QUERY_DEVICE_PATH, device_node,
                                  &node_len);

    amdcuid_id_t primary_id;
    uint32_t length = sizeof(primary_id);
    amdcuid_status_t status = amdcuid_query_device_property(
        device_handles_[i], AMDCUID_QUERY_PRIMARY_CUID, &primary_id, &length);
    EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);

    uint8_t raw_bits[16] = {0};
    extract_primary_raw_bits(primary_id, raw_bits);
    uint16_t extracted_revision = static_cast<uint16_t>(raw_bits[9]);

    uint16_t queried_revision = 0;
    length = sizeof(queried_revision);
    status = amdcuid_query_device_property(device_handles_[i], AMDCUID_QUERY_REVISION_ID,
                                           &queried_revision, &length);
    if (status == AMDCUID_STATUS_WRONG_DEVICE_TYPE) {
      continue;
    }
    EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);
    EXPECT_EQ(extracted_revision, queried_revision)
        << "Revision ID mismatch for device " << device_node;

    IF_VERB(1) { printf("  Device [%s] revision_id: 0x%04x\n", device_node, queried_revision); }
  }
}

// ---------------------------------------------------------------------------
// TestReverseUnitId
// ---------------------------------------------------------------------------

TestReverseUnitId::TestReverseUnitId() {
  SetTitle("Reverse Lookup — Unit ID");
  SetDescription(
      "Verify that the unit ID embedded in the primary CUID matches "
      "AMDCUID_QUERY_UNIT_ID for each supporting device type.");
}

void TestReverseUnitId::Run() {
  if (device_handles_.empty()) {
    GTEST_SKIP() << "No devices found; skipping.";
  }

  for (size_t i = 0; i < device_handles_.size(); ++i) {
    char device_node[256] = {0};
    uint32_t node_len = sizeof(device_node);
    amdcuid_query_device_property(device_handles_[i], AMDCUID_QUERY_DEVICE_PATH, device_node,
                                  &node_len);

    amdcuid_id_t primary_id;
    uint32_t length = sizeof(primary_id);
    amdcuid_status_t status = amdcuid_query_device_property(
        device_handles_[i], AMDCUID_QUERY_PRIMARY_CUID, &primary_id, &length);
    EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);

    uint8_t raw_bits[16] = {0};
    extract_primary_raw_bits(primary_id, raw_bits);
    // unit_id is 13 bits: lower 8 in raw_bits[8], upper 5 in bits [4:0] of
    // raw_bits[14].
    uint16_t extracted_unit_id =
        static_cast<uint16_t>(raw_bits[8]) | (static_cast<uint16_t>(raw_bits[14] & 0x1F) << 8);

    uint16_t queried_unit_id = 0;
    length = sizeof(queried_unit_id);
    status = amdcuid_query_device_property(device_handles_[i], AMDCUID_QUERY_UNIT_ID,
                                           &queried_unit_id, &length);
    if (status == AMDCUID_STATUS_WRONG_DEVICE_TYPE) {
      continue;
    }
    EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);
    EXPECT_EQ(extracted_unit_id, queried_unit_id) << "Unit ID mismatch for device " << device_node;

    IF_VERB(1) { printf("  Device [%s] unit_id: 0x%04x\n", device_node, queried_unit_id); }
  }
}

// ---------------------------------------------------------------------------
// TestReverseDeviceType
// ---------------------------------------------------------------------------

TestReverseDeviceType::TestReverseDeviceType() {
  SetTitle("Reverse Lookup — Device Type");
  SetDescription(
      "Verify that the device type embedded in the primary CUID matches "
      "AMDCUID_QUERY_DEVICE_TYPE for each device.");
}

void TestReverseDeviceType::Run() {
  if (device_handles_.empty()) {
    GTEST_SKIP() << "No devices found; skipping.";
  }

  for (size_t i = 0; i < device_handles_.size(); ++i) {
    char device_node[256] = {0};
    uint32_t node_len = sizeof(device_node);
    amdcuid_query_device_property(device_handles_[i], AMDCUID_QUERY_DEVICE_PATH, device_node,
                                  &node_len);

    amdcuid_id_t primary_id;
    uint32_t length = sizeof(primary_id);
    amdcuid_status_t status = amdcuid_query_device_property(
        device_handles_[i], AMDCUID_QUERY_PRIMARY_CUID, &primary_id, &length);
    EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);

    uint8_t raw_bits[16] = {0};
    extract_primary_raw_bits(primary_id, raw_bits);
    // device_type is 4 bits: bits [7:6] of raw_bits[14] hold bits [1:0],
    // bits [7:6] of raw_bits[15] hold bits [3:2].
    uint8_t extracted_type =
        static_cast<uint8_t>(((raw_bits[14] >> 6) & 0x3) | ((raw_bits[15] >> 4) & 0xC));

    amdcuid_device_type_t queried_type = AMDCUID_DEVICE_TYPE_NONE;
    length = sizeof(queried_type);
    status = amdcuid_query_device_property(device_handles_[i], AMDCUID_QUERY_DEVICE_TYPE,
                                           &queried_type, &length);
    EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);
    EXPECT_EQ(static_cast<amdcuid_device_type_t>(extracted_type), queried_type)
        << "Device type mismatch for device " << device_node;

    IF_VERB(1) {
      printf("  Device [%s] device_type: %u\n", device_node, static_cast<unsigned>(queried_type));
    }
  }
}
