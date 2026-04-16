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

#include "include/amd_cuid.h"
#include "src/cuid_util.h"
#include "src/pci_util.h"
#include <cstring>
#include <gtest/gtest.h>
#include <unistd.h>
#include <vector>

// write tests to reverse CUID into its components like serial number, product
// family, etc. tests must be run with sudo to be able to access primary CUIDs
// with which to reverse serial number and other components

// Helper: strip UUIDv8 overhead from a primary CUID to recover the 122 raw data
// bits. Raw bit layout (after stripping):
//   [0..7]  serial number (uint64_t, little-endian)
//   [8]     unit_id lower 8 bits
//   [9]     revision_id (uint8_t)
//   [10..11] device_id (uint16_t, little-endian)
//   [12..13] vendor_id (uint16_t, little-endian)
//   [14]    (unit_id upper 6 bits << 2) | (device_type upper 2 bits)
//   [15]    (device_type lower 2 bits << 6) | padding
static void extract_primary_raw_bits(const amdcuid_id_t &uuid,
                                     uint8_t raw_bits[16]) {
  amdcuid_id_t mutable_uuid = uuid;
  CuidUtilities::remove_UUIDv8_bits(&mutable_uuid, raw_bits);
}

void test_cuid_reverse_serial_number() {
  if (geteuid() != 0) {
    GTEST_SKIP()
        << "Skipping test_cuid_reverse_serial_number: requires elevated "
           "permissions";
  }

  uint32_t count = 0;
  std::vector<amdcuid_id_t> handles(count);
  amdcuid_status_t status = amdcuid_get_all_handles(handles.data(), &count);
  EXPECT_TRUE(status == AMDCUID_STATUS_INSUFFICIENT_SIZE);

  handles.resize(count);
  status = amdcuid_get_all_handles(handles.data(), &count);
  EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);
  EXPECT_GT(count, 0);

  for (uint32_t i = 0; i < count; ++i) {
    uint64_t serial_number = 0;
    uint32_t length = sizeof(serial_number);
    status = amdcuid_query_device_property(handles[i],
                                           AMDCUID_QUERY_HARDWARE_FINGERPRINT,
                                           &serial_number, &length);
    EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);
    EXPECT_NE(serial_number, 0); // Assuming valid serial numbers are non-zero

    // get the primary CUID to reverse it
    amdcuid_id_t primary_id = {};
    length = sizeof(primary_id);
    status = amdcuid_query_device_property(
        handles[i], AMDCUID_QUERY_PRIMARY_CUID, &primary_id, &length);
    EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);

    // First 8 bytes in primary CUID should be the serial number in
    // little-endian format, just need to copy those bytes, remove uuidv8 bits
    // and convert to uint64_t
    uint8_t raw_bits[16] = {0};
    extract_primary_raw_bits(primary_id, raw_bits);
    uint64_t extracted_serial = 0;
    memcpy(&extracted_serial, raw_bits, sizeof(extracted_serial));
    EXPECT_EQ(serial_number, extracted_serial);
  }
}

void test_cuid_reverse_vendor_id() {
  if (geteuid() != 0) {
    GTEST_SKIP() << "Skipping test_cuid_reverse_vendor_id: requires elevated "
                    "permissions";
  }

  uint32_t count = 0;
  amdcuid_status_t status = amdcuid_get_all_handles(nullptr, &count);
  ASSERT_GT(count, 0u);

  std::vector<amdcuid_id_t> handles(count);
  status = amdcuid_get_all_handles(handles.data(), &count);
  EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);
  EXPECT_GT(count, 0u);

  for (uint32_t i = 0; i < count; ++i) {
    amdcuid_id_t primary_id;
    uint32_t length = sizeof(primary_id);
    status = amdcuid_query_device_property(
        handles[i], AMDCUID_QUERY_PRIMARY_CUID, &primary_id, &length);
    EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);

    uint8_t raw_bits[16] = {0};
    extract_primary_raw_bits(primary_id, raw_bits);
    // vendor_id is stored at raw_bits[12..13] in little-endian order
    uint16_t extracted_vendor = static_cast<uint16_t>(raw_bits[12]) |
                                (static_cast<uint16_t>(raw_bits[13]) << 8);

    uint16_t queried_vendor = 0;
    length = sizeof(queried_vendor);
    status = amdcuid_query_device_property(handles[i], AMDCUID_QUERY_VENDOR_ID,
                                           &queried_vendor, &length);
    EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);
    EXPECT_EQ(extracted_vendor, queried_vendor);
  }
}

void test_cuid_reverse_device_id() {
  if (geteuid() != 0) {
    GTEST_SKIP() << "Skipping test_cuid_reverse_device_id: requires elevated "
                    "permissions";
  }

  uint32_t count = 0;
  amdcuid_status_t status = amdcuid_get_all_handles(nullptr, &count);
  ASSERT_GT(count, 0u);

  std::vector<amdcuid_id_t> handles(count);
  status = amdcuid_get_all_handles(handles.data(), &count);
  EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);
  EXPECT_GT(count, 0u);

  for (uint32_t i = 0; i < count; ++i) {
    amdcuid_id_t primary_id;
    uint32_t length = sizeof(primary_id);
    status = amdcuid_query_device_property(
        handles[i], AMDCUID_QUERY_PRIMARY_CUID, &primary_id, &length);
    EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);

    uint8_t raw_bits[16] = {0};
    extract_primary_raw_bits(primary_id, raw_bits);
    // device_id is stored at raw_bits[10..11] in little-endian order
    uint16_t extracted_device_id = static_cast<uint16_t>(raw_bits[10]) |
                                   (static_cast<uint16_t>(raw_bits[11]) << 8);

    uint16_t queried_device_id = 0;
    length = sizeof(queried_device_id);
    status = amdcuid_query_device_property(handles[i], AMDCUID_QUERY_DEVICE_ID,
                                           &queried_device_id, &length);
    if (status == AMDCUID_STATUS_WRONG_DEVICE_TYPE) {
      continue; // device_id not supported for this device type (e.g. platform)
    }
    EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);
    EXPECT_EQ(extracted_device_id, queried_device_id);
  }
}

void test_cuid_reverse_revision_id() {
  if (geteuid() != 0) {
    GTEST_SKIP() << "Skipping test_cuid_reverse_revision_id: requires elevated "
                    "permissions";
  }

  uint32_t count = 0;
  amdcuid_status_t status = amdcuid_get_all_handles(nullptr, &count);
  ASSERT_GT(count, 0u);

  std::vector<amdcuid_id_t> handles(count);
  status = amdcuid_get_all_handles(handles.data(), &count);
  EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);
  EXPECT_GT(count, 0u);

  for (uint32_t i = 0; i < count; ++i) {
    amdcuid_id_t primary_id;
    uint32_t length = sizeof(primary_id);
    status = amdcuid_query_device_property(
        handles[i], AMDCUID_QUERY_PRIMARY_CUID, &primary_id, &length);
    EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);

    uint8_t raw_bits[16] = {0};
    extract_primary_raw_bits(primary_id, raw_bits);
    // revision_id is stored as a uint8_t at raw_bits[9]; the API returns it
    // widened to uint16_t
    uint16_t extracted_revision = static_cast<uint16_t>(raw_bits[9]);

    uint16_t queried_revision = 0;
    length = sizeof(queried_revision);
    status = amdcuid_query_device_property(
        handles[i], AMDCUID_QUERY_REVISION_ID, &queried_revision, &length);
    if (status == AMDCUID_STATUS_WRONG_DEVICE_TYPE) {
      continue; // revision_id not supported for this device type (e.g.
                // platform)
    }
    EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);
    EXPECT_EQ(extracted_revision, queried_revision);
  }
}

void test_cuid_reverse_unit_id() {
  if (geteuid() != 0) {
    GTEST_SKIP()
        << "Skipping test_cuid_reverse_unit_id: requires elevated permissions";
  }

  uint32_t count = 0;
  amdcuid_status_t status = amdcuid_get_all_handles(nullptr, &count);
  ASSERT_GT(count, 0u);

  std::vector<amdcuid_id_t> handles(count);
  status = amdcuid_get_all_handles(handles.data(), &count);
  EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);
  EXPECT_GT(count, 0u);

  for (uint32_t i = 0; i < count; ++i) {
    amdcuid_id_t primary_id;
    uint32_t length = sizeof(primary_id);
    status = amdcuid_query_device_property(
        handles[i], AMDCUID_QUERY_PRIMARY_CUID, &primary_id, &length);
    EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);

    uint8_t raw_bits[16] = {0};
    extract_primary_raw_bits(primary_id, raw_bits);
    // unit_id is 14 bits split across two bytes:
    //   lower 8 bits: raw_bits[8]
    //   upper 6 bits: bits [7:2] of raw_bits[14]
    uint16_t extracted_unit_id =
        static_cast<uint16_t>(raw_bits[8]) |
        (static_cast<uint16_t>((raw_bits[14] >> 2) & 0x3F) << 8);

    uint16_t queried_unit_id = 0;
    length = sizeof(queried_unit_id);
    status = amdcuid_query_device_property(handles[i], AMDCUID_QUERY_UNIT_ID,
                                           &queried_unit_id, &length);
    if (status == AMDCUID_STATUS_WRONG_DEVICE_TYPE) {
      continue; // unit_id not supported for this device type (e.g. platform,
                // NIC)
    }
    EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);
    EXPECT_EQ(extracted_unit_id, queried_unit_id);
  }
}

void test_cuid_reverse_device_type() {
  if (geteuid() != 0) {
    GTEST_SKIP() << "Skipping test_cuid_reverse_device_type: requires elevated "
                    "permissions";
  }

  uint32_t count = 0;
  amdcuid_status_t status = amdcuid_get_all_handles(nullptr, &count);
  ASSERT_GT(count, 0u);

  std::vector<amdcuid_id_t> handles(count);
  status = amdcuid_get_all_handles(handles.data(), &count);
  EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);
  EXPECT_GT(count, 0u);

  for (uint32_t i = 0; i < count; ++i) {
    amdcuid_id_t primary_id;
    uint32_t length = sizeof(primary_id);
    status = amdcuid_query_device_property(
        handles[i], AMDCUID_QUERY_PRIMARY_CUID, &primary_id, &length);
    EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);

    uint8_t raw_bits[16] = {0};
    extract_primary_raw_bits(primary_id, raw_bits);
    // device_type is 4 bits split across two bytes:
    //   upper 2 bits: bits [1:0] of raw_bits[14]
    //   lower 2 bits: bits [7:6] of raw_bits[15]
    uint8_t extracted_type = static_cast<uint8_t>(((raw_bits[14] & 0x3) << 2) |
                                                  ((raw_bits[15] >> 6) & 0x3));

    amdcuid_device_type_t queried_type = AMDCUID_DEVICE_TYPE_NONE;
    length = sizeof(queried_type);
    status = amdcuid_query_device_property(
        handles[i], AMDCUID_QUERY_DEVICE_TYPE, &queried_type, &length);
    EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);
    EXPECT_EQ(static_cast<amdcuid_device_type_t>(extracted_type), queried_type);
  }
}

TEST(CUIDReverseLookupTest, ReverseSerialNumber) {
  test_cuid_reverse_serial_number();
}

TEST(CUIDReverseLookupTest, ReverseVendorId) { test_cuid_reverse_vendor_id(); }

TEST(CUIDReverseLookupTest, ReverseDeviceId) { test_cuid_reverse_device_id(); }

TEST(CUIDReverseLookupTest, ReverseRevisionId) {
  test_cuid_reverse_revision_id();
}

TEST(CUIDReverseLookupTest, ReverseUnitId) { test_cuid_reverse_unit_id(); }

TEST(CUIDReverseLookupTest, ReverseDeviceType) {
  test_cuid_reverse_device_type();
}
