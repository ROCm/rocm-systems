// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "functional/hmac_test.h"

#include <gtest/gtest.h>

#include <cstdio>

TestHMAC::TestHMAC() {
  SetTitle("HMAC Key Operations");
  SetDescription(
      "Verify amdcuid_generate_hash_key produces a non-zero key and that "
      "amdcuid_set_hash_key accepts it. Both operations require root.");
}

// No device enumeration needed for HMAC key operations.
void TestHMAC::SetUp() {}

void TestHMAC::Run() {
  // This test provisions a real seed on the node it runs on, which changes
  // every derived CUID the kernel does not answer for. Put the node's own seed
  // back afterwards.
  KeyStoreGuard key_guard;

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
}
