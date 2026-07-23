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

#include "functional/hmac_test.h"

#include <gtest/gtest.h>
#include <limits.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <fstream>

TestHMAC::TestHMAC() {
  SetTitle("HMAC Key Operations");
  SetDescription(
      "Verify amdcuid_generate_hash_key produces a non-zero key and that "
      "amdcuid_set_hash_key accepts it. Both operations require root.");
}

// Device enumeration is needed to verify cuid_seed writes on amdgpu devices.
void TestHMAC::SetUp() {
  TestBase::SetUp();
  // Snapshot cuid_seed contents before the test modifies them.
  amdgpu_nodes_ = collect_amdgpu_render_nodes(device_handles_);
  saved_seeds_ = save_cuid_seeds(amdgpu_nodes_);
}

// Returns the amdgpu render node paths for all discovered GPU devices.
static std::vector<std::string> collect_amdgpu_render_nodes(
    const std::vector<amdcuid_id_t>& handles) {
  std::vector<std::string> nodes;
  for (const auto& handle : handles) {
    amdcuid_device_type_t device_type = AMDCUID_DEVICE_TYPE_NONE;
    uint32_t type_len = sizeof(device_type);
    if (amdcuid_query_device_property(handle, AMDCUID_QUERY_DEVICE_TYPE, &device_type, &type_len) !=
            AMDCUID_STATUS_SUCCESS ||
        device_type != AMDCUID_DEVICE_TYPE_GPU)
      continue;

    char path_buf[512] = {};
    uint32_t path_len = sizeof(path_buf);
    if (amdcuid_query_device_property(handle, AMDCUID_QUERY_DEVICE_PATH, path_buf, &path_len) !=
        AMDCUID_STATUS_SUCCESS)
      continue;

    std::string render_node(path_buf);
    std::string driver_link = render_node + "/device/driver";
    char link_buf[PATH_MAX] = {};
    ssize_t link_len = readlink(driver_link.c_str(), link_buf, sizeof(link_buf) - 1);
    if (link_len <= 0) continue;
    link_buf[link_len] = '\0';
    const char* driver_name = strrchr(link_buf, '/');
    if (driver_name && strcmp(driver_name + 1, "amdgpu") == 0) nodes.push_back(render_node);
  }
  return nodes;
}

// Saves the current contents of cuid_seed for each node, or an empty vector
// entry if the file does not yet exist.
static std::vector<std::vector<uint8_t>> save_cuid_seeds(const std::vector<std::string>& nodes) {
  std::vector<std::vector<uint8_t>> saved;
  for (const auto& node : nodes) {
    std::string seed_path = node + "/device/cuid_seed";
    std::ifstream f(seed_path, std::ios::in | std::ios::binary);
    if (!f.is_open()) {
      saved.emplace_back();  // empty — file did not exist
      continue;
    }
    saved.emplace_back(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
  }
  return saved;
}

// Restores the cuid_seed files from the saved snapshots.  An empty snapshot
// means the file did not exist before the test ran and should be removed.
static void restore_cuid_seeds(const std::vector<std::string>& nodes,
                               const std::vector<std::vector<uint8_t>>& saved) {
  for (size_t i = 0; i < nodes.size(); ++i) {
    std::string seed_path = nodes[i] + "/device/cuid_seed";
    if (saved[i].empty()) {
      // File was absent before the test; remove what the test wrote.
      std::remove(seed_path.c_str());
    } else {
      std::ofstream f(seed_path, std::ios::out | std::ios::binary | std::ios::trunc);
      if (f) f.write(reinterpret_cast<const char*>(saved[i].data()), saved[i].size());
    }
  }
}

// For each amdgpu render node, verifies that cuid_seed contains exactly the
// expected key bytes.
void TestHMAC::VerifyCuidSeedWritten(const uint8_t key[32]) {
  if (device_handles_.empty()) {
    GTEST_SKIP() << "No devices found; skipping cuid_seed verification.\n";
  }

  if (amdgpu_nodes_.empty()) {
    GTEST_SKIP() << "No amdgpu devices found; cuid_seed verification skipped.\n";
  }

  for (const auto& render_node : amdgpu_nodes_) {
    std::string seed_path = render_node + "/device/cuid_seed";
    std::ifstream seed_file(seed_path, std::ios::in | std::ios::binary);
    EXPECT_TRUE(seed_file.is_open())
        << "Could not open " << seed_path
        << " for reading after amdcuid_set_hash_key() for device: " << render_node;

    uint8_t seed_bytes[32] = {};
    seed_file.read(reinterpret_cast<char*>(seed_bytes), 32);
    EXPECT_EQ(seed_file.gcount(), 32)
        << seed_path << " contained fewer than 32 bytes for device: " << render_node;

    EXPECT_EQ(memcmp(seed_bytes, key, 32), 0) << "cuid_seed content does not match the key set via "
                                                 "amdcuid_set_hash_key() for device: "
                                              << render_node;

    IF_VERB(1) { printf("  cuid_seed verified for %s\n", render_node.c_str()); }
  }
}

void TestHMAC::Run() {
  uint8_t generated_key[32] = {0};
  amdcuid_status_t status = amdcuid_generate_hash_key(generated_key);
  CHK_ERR_ASRT(status);

  bool all_zeros = true;
  for (size_t i = 0; i < sizeof(generated_key); ++i) {
    if (generated_key[i] != 0) {
      all_zeros = false;
      break;
    }
  }
  EXPECT_FALSE(all_zeros) << "Generated key is all zeros";

  IF_VERB(2) {
    printf("  Generated key (first 4 bytes): %02x %02x %02x %02x\n", generated_key[0],
           generated_key[1], generated_key[2], generated_key[3]);
  }

  status = amdcuid_set_hash_key(generated_key);
  CHK_ERR_ASRT(status);

  IF_VERB(1) { printf("  amdcuid_set_hash_key: %s\n", amdcuid_status_to_string(status)); }

  VerifyCuidSeedWritten(generated_key);
}

void TestHMAC::Close() {
  // Restore the original cuid_seed contents so that derived CUIDs remain
  // stable after the test completes.
  restore_cuid_seeds(amdgpu_nodes_, saved_seeds_);
}
