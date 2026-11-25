// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// The v1 boot message decoders build a 32-bit word from four message bytes, with byte 7 on top.
// Pins which bytes each decoder uses and that a top byte >= 0x80 comes through. No GPU required.

#include "ras-decode/boot_decode.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

namespace {

// Bytes 7..0 are 0x88 0x77 0x66 0x55 0x44 0x33 0x22 0x11.
constexpr uint64_t kBootMsg = 0x8877665544332211ULL;

std::string DecodedField(boot_decoder_func_t decoder, const char* key) {
  OamBootMsg msg{kBootMsg};
  JsonValue* decoded = decoder(&msg);
  const JsonValue* field = json_object_get(decoded, key);
  const std::string text =
      (field != nullptr && field->type == JSON_STRING) ? field->data.string : "";
  json_free(decoded);
  return text;
}

TEST(GpuUnit, BootDecodeV1BuildsWordFromMessageBytes) {
  EXPECT_EQ(DecodedField(decode_boot_controller_generic_v1, "fw_boot_status"), "0x88776611");
  EXPECT_EQ(DecodedField(decode_boot_success_v1, "fw_boot_status"), "0x88776611");
  EXPECT_EQ(DecodedField(decode_unhandled_error_v1, "fw_boot_status"), "0x88776611");
  EXPECT_EQ(DecodedField(decode_data_abort_v1, "exception_address"), "0x88776655");
}

}  // namespace
