// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "functional/hmac_test.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

TestHMAC::TestHMAC() {
  SetTitle("HMAC Key Operations");
  SetDescription(
      "Verify amdcuid_generate_hash_key produces a non-zero key. "
      "amdcuid_set_hash_key requires root.");
}

// No device enumeration needed for HMAC key operations.
void TestHMAC::SetUp() {}

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

  const char* allow = std::getenv("AMDCUID_TEST_ALLOW_SET_KEY");
  if (!allow || std::strcmp(allow, "1") != 0) {
    GTEST_SKIP() << "amdcuid_set_hash_key() would replace this host's node key; "
                    "set AMDCUID_TEST_ALLOW_SET_KEY=1 to run it.";
  }

  status = amdcuid_set_hash_key(generated_key);
  if (status == AMDCUID_STATUS_UNSUPPORTED) GTEST_SKIP() << "no amdgpu device exposes cuid_seed";
  CHK_ERR_ASRT(status);

  IF_VERB(1) { printf("  amdcuid_set_hash_key: %s\n", amdcuid_status_to_string(status)); }
}
