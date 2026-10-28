#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string>

#include "amd_smi/impl/amd_smi_uuid.h"

amdsmi_status_t smi_clear_char_and_reinitialize(char buffer[], uint32_t len,
                                               std::string newString);

TEST(SupportUnit, CodeqlEmptyBufferIsRejected) {
  char output = '!';
  EXPECT_EQ(smi_clear_char_and_reinitialize(&output, 0, "value"), AMDSMI_STATUS_INVAL);
  EXPECT_EQ(output, '!');
  EXPECT_EQ(smi_clear_char_and_reinitialize(nullptr, 1, "value"), AMDSMI_STATUS_INVAL);
}

TEST(SupportUnit, CodeqlTextBufferTerminatesAtCapacity) {
  std::array<char, 5> output;
  output.fill('!');
  EXPECT_EQ(smi_clear_char_and_reinitialize(output.data() + 1, 3, "long"),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_STREQ(output.data() + 1, "lo");
  EXPECT_EQ(output.front(), '!');
  EXPECT_EQ(output.back(), '!');
}

TEST(SupportUnit, CodeqlUuidPreservesExactWidth) {
  std::array<char, 39> output;
  output.fill('!');
  EXPECT_EQ(amdsmi_uuid_gen(output.data() + 1, UINT64_MAX, UINT16_MAX, UINT8_MAX),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_STREQ(output.data() + 1, "ffffffff-0000-1000-80ff-ffffffffffff");
  EXPECT_EQ(output.front(), '!');
  EXPECT_EQ(output.back(), '!');
}