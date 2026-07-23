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

#include "functional/sysfs_cuid_test.h"

#include <gtest/gtest.h>
#include <limits.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <fstream>

#include "src/cuid_util.h"

// Read a sysfs UUID string file and parse it into a 16-byte array using the
// same path the library takes: uuid_string_to_uint8.
// Returns false (and leaves out_id zeroed) if the file is absent or empty,
// which means the driver has not yet written the CUID — the calling test
// skips that device rather than failing.
static bool read_sysfs_cuid(const std::string& path, amdcuid_id_t& out_id) {
  std::ifstream f(path);
  if (!f.is_open()) return false;
  std::string uuid_str;
  std::getline(f, uuid_str);
  if (uuid_str.empty()) return false;
  out_id = {};
  amdcuid_status_t status = CuidUtilities::uuid_string_to_uint8(uuid_str, out_id.bytes);
  return status == AMDCUID_STATUS_SUCCESS;
}

// Returns the sysfs render_node path for each GPU device that is driven by
// amdgpu, identified by the driver symlink at <render_node>/device/driver.
static std::vector<std::string> amdgpu_render_nodes(const std::vector<amdcuid_id_t>& handles) {
  std::vector<std::string> nodes;
  for (const auto& handle : handles) {
    amdcuid_device_type_t device_type = AMDCUID_DEVICE_TYPE_NONE;
    uint32_t len = sizeof(device_type);
    if (amdcuid_query_device_property(handle, AMDCUID_QUERY_DEVICE_TYPE, &device_type, &len) !=
            AMDCUID_STATUS_SUCCESS ||
        device_type != AMDCUID_DEVICE_TYPE_GPU)
      continue;

    char path_buf[512] = {};
    len = sizeof(path_buf);
    if (amdcuid_query_device_property(handle, AMDCUID_QUERY_DEVICE_PATH, path_buf, &len) !=
        AMDCUID_STATUS_SUCCESS)
      continue;

    std::string render_node(path_buf);
    char link_buf[PATH_MAX] = {};
    ssize_t link_len =
        readlink((render_node + "/device/driver").c_str(), link_buf, sizeof(link_buf) - 1);
    if (link_len <= 0) continue;
    link_buf[link_len] = '\0';
    const char* name = strrchr(link_buf, '/');
    if (name && strcmp(name + 1, "amdgpu") == 0) nodes.push_back(render_node);
  }
  return nodes;
}

// ---------------------------------------------------------------------------
// TestSysfsReadPrimaryCuid
// ---------------------------------------------------------------------------

TestSysfsReadPrimaryCuid::TestSysfsReadPrimaryCuid() {
  SetTitle("Sysfs Read — Primary CUID");
  SetDescription(
      "For each amdgpu GPU, read cuid_primary directly from sysfs and verify "
      "it matches AMDCUID_QUERY_PRIMARY_CUID. Skips devices where the sysfs "
      "file is absent (driver has not yet written the primary CUID).");
}

void TestSysfsReadPrimaryCuid::Run() {
  if (device_handles_.empty()) {
    GTEST_SKIP() << "No devices found; skipping.";
  }

  auto nodes = amdgpu_render_nodes(device_handles_);
  if (nodes.empty()) {
    GTEST_SKIP() << "No amdgpu devices found; skipping.";
  }

  int checked = 0;
  for (const auto& render_node : nodes) {
    std::string sysfs_path = render_node + "/device/cuid_primary";

    amdcuid_id_t sysfs_id = {};
    if (!read_sysfs_cuid(sysfs_path, sysfs_id)) {
      IF_VERB(1) { printf("  %s: cuid_primary absent; skipping device.\n", render_node.c_str()); }
      continue;
    }

    // Find the handle whose device path matches this render node so we can
    // query AMDCUID_QUERY_PRIMARY_CUID for comparison.
    amdcuid_id_t api_id = {};
    bool handle_found = false;
    for (const auto& handle : device_handles_) {
      char path_buf[512] = {};
      uint32_t len = sizeof(path_buf);
      if (amdcuid_query_device_property(handle, AMDCUID_QUERY_DEVICE_PATH, path_buf, &len) !=
          AMDCUID_STATUS_SUCCESS)
        continue;
      if (render_node != std::string(path_buf)) continue;

      len = sizeof(api_id);
      amdcuid_status_t status =
          amdcuid_query_device_property(handle, AMDCUID_QUERY_PRIMARY_CUID, &api_id, &len);
      EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS)
          << "AMDCUID_QUERY_PRIMARY_CUID failed for " << render_node;
      handle_found = true;
      break;
    }

    if (!handle_found) {
      // Should not happen — we derived the node from handles above.
      ADD_FAILURE() << "Could not find handle for render node: " << render_node;
      continue;
    }

    EXPECT_EQ(memcmp(sysfs_id.bytes, api_id.bytes, sizeof(amdcuid_id_t)), 0)
        << "cuid_primary sysfs content does not match "
           "AMDCUID_QUERY_PRIMARY_CUID for device: "
        << render_node;

    IF_VERB(1) { printf("  %s: cuid_primary verified\n", render_node.c_str()); }
    IF_VERB(2) {
      printf("    sysfs  : %02x%02x%02x%02x...\n", sysfs_id.bytes[0], sysfs_id.bytes[1],
             sysfs_id.bytes[2], sysfs_id.bytes[3]);
      printf("    api    : %02x%02x%02x%02x...\n", api_id.bytes[0], api_id.bytes[1],
             api_id.bytes[2], api_id.bytes[3]);
    }
    ++checked;
  }

  if (checked == 0) {
    GTEST_SKIP() << "No amdgpu devices had a cuid_primary sysfs file; "
                    "driver may not have written it yet.";
  }
}

// ---------------------------------------------------------------------------
// TestSysfsReadSecondaryCuid
// ---------------------------------------------------------------------------

TestSysfsReadSecondaryCuid::TestSysfsReadSecondaryCuid() {
  SetTitle("Sysfs Read — Secondary CUID");
  SetDescription(
      "For each amdgpu GPU, read cuid_secondary directly from sysfs and "
      "verify it matches AMDCUID_QUERY_DERIVED_CUID. Skips devices where the "
      "sysfs file is absent (driver has not yet written the secondary CUID).");
}

void TestSysfsReadSecondaryCuid::Run() {
  if (device_handles_.empty()) {
    GTEST_SKIP() << "No devices found; skipping.";
  }

  auto nodes = amdgpu_render_nodes(device_handles_);
  if (nodes.empty()) {
    GTEST_SKIP() << "No amdgpu devices found; skipping.";
  }

  int checked = 0;
  for (const auto& render_node : nodes) {
    std::string sysfs_path = render_node + "/device/cuid_secondary";

    amdcuid_id_t sysfs_id = {};
    if (!read_sysfs_cuid(sysfs_path, sysfs_id)) {
      IF_VERB(1) { printf("  %s: cuid_secondary absent; skipping device.\n", render_node.c_str()); }
      continue;
    }

    amdcuid_id_t api_id = {};
    bool handle_found = false;
    for (const auto& handle : device_handles_) {
      char path_buf[512] = {};
      uint32_t len = sizeof(path_buf);
      if (amdcuid_query_device_property(handle, AMDCUID_QUERY_DEVICE_PATH, path_buf, &len) !=
          AMDCUID_STATUS_SUCCESS)
        continue;
      if (render_node != std::string(path_buf)) continue;

      len = sizeof(api_id);
      amdcuid_status_t status =
          amdcuid_query_device_property(handle, AMDCUID_QUERY_DERIVED_CUID, &api_id, &len);
      EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS)
          << "AMDCUID_QUERY_DERIVED_CUID failed for " << render_node;
      handle_found = true;
      break;
    }

    if (!handle_found) {
      ADD_FAILURE() << "Could not find handle for render node: " << render_node;
      continue;
    }

    EXPECT_EQ(memcmp(sysfs_id.bytes, api_id.bytes, sizeof(amdcuid_id_t)), 0)
        << "cuid_secondary sysfs content does not match "
           "AMDCUID_QUERY_DERIVED_CUID for device: "
        << render_node;

    IF_VERB(1) { printf("  %s: cuid_secondary verified\n", render_node.c_str()); }
    IF_VERB(2) {
      printf("    sysfs  : %02x%02x%02x%02x...\n", sysfs_id.bytes[0], sysfs_id.bytes[1],
             sysfs_id.bytes[2], sysfs_id.bytes[3]);
      printf("    api    : %02x%02x%02x%02x...\n", api_id.bytes[0], api_id.bytes[1],
             api_id.bytes[2], api_id.bytes[3]);
    }
    ++checked;
  }

  if (checked == 0) {
    GTEST_SKIP() << "No amdgpu devices had a cuid_secondary sysfs file; "
                    "driver may not have written it yet.";
  }
}
