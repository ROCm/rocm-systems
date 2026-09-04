// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Unit tests for amdsmi_uuid_gen(); no GPU required (pure function of
// serial/did/idx). Guards the bounded-snprintf rewrite against a regression to
// the unbounded writer that overran the UUID buffer.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_uuid.h"

namespace {

bool IsLowerHex(char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }

// A standard UUID is exactly 36 chars: 8-4-4-4-12 lowercase hex with hyphens
// at indices 8, 13, 18, 23 and a NUL at 36.
void ExpectStandardShape(const char* uuid) {
  ASSERT_EQ(std::strlen(uuid), 36u);
  for (int i = 0; i < 36; ++i) {
    if (i == 8 || i == 13 || i == 18 || i == 23) {
      EXPECT_EQ(uuid[i], '-') << "expected hyphen at index " << i;
    } else {
      EXPECT_TRUE(IsLowerHex(uuid[i])) << "expected lowercase hex at index " << i;
    }
  }
}

}  // namespace

TEST(GpuUnit, UuidGenFormatsStandardLayoutForKnownInputs) {
  char uuid[AMDSMI_GPU_UUID_SIZE] = {};

  ASSERT_EQ(amdsmi_uuid_gen(uuid, /*serial=*/0xDEADBEEFull, /*did=*/0x1234, /*idx=*/0x56),
            AMDSMI_STATUS_SUCCESS);

  ExpectStandardShape(uuid);
  // version nibble is fixed at 1 (first char of the third group).
  EXPECT_EQ(uuid[14], '1');
  // Last group is asic_4 (16 bits) then asic_0 (low 32 bits of the serial).
  EXPECT_STREQ(uuid + 24, "0000deadbeef");
}

TEST(GpuUnit, UuidGenStaysWithinUuidBufferOnMaxInputs) {
  // Oversize the buffer and fence it with a canary so any write past the
  // AMDSMI_GPU_UUID_SIZE-bounded region is observable.
  constexpr char kCanary = 0x7e;
  char uuid[AMDSMI_GPU_UUID_SIZE + 8];
  std::memset(uuid, kCanary, sizeof(uuid));

  ASSERT_EQ(amdsmi_uuid_gen(uuid, /*serial=*/0xFFFFFFFFFFFFFFFFull, /*did=*/0xFFFF, /*idx=*/0xFF),
            AMDSMI_STATUS_SUCCESS);

  ExpectStandardShape(uuid);
  // snprintf writes 36 chars + NUL (indices 0..36); every byte from 37 on, both
  // the padding byte inside AMDSMI_GPU_UUID_SIZE and the fence past it, stays canary.
  for (size_t i = 37; i < sizeof(uuid); ++i) {
    EXPECT_EQ(uuid[i], kCanary) << "buffer overrun at index " << i;
  }
}
