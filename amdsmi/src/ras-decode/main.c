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

/**
 * @file main.c
 * @brief Demo program showing how to use the ACA decoder
 *
 * This is a demonstration program that shows how to use the ACA decoder
 * with sample raw data to decode ACA error information.
 */

#include <aca_decode.h>
#include <aca_version.h>
#include <inttypes.h>
#include <json_printer.h>
#include <ras_decode_api.h>
#include <ras_decode_constants.h>
#include <stdint.h>
#include <stdio.h>

// Function prototype
void print_version_info(void);
void demonstrate_json_decoding(void);

void print_version_info(void) {
  printf("=== ACA Decoder Library Version Information ===\n");
  printf("Version: %s\n", aca_get_version_string());
  printf("Major: %d\n", aca_get_version_major());
  printf("Minor: %d\n", aca_get_version_minor());
  printf("Patch: %d\n", aca_get_version_patch());

  aca_version_info_t version_info = aca_get_version_info();
  printf("Complete version info:\n");
  printf("  Major: %d\n", version_info.major);
  printf("  Minor: %d\n", version_info.minor);
  printf("  Patch: %d\n", version_info.patch);
  printf("  String: %s\n", version_info.string);
  printf("===============================================\n\n");
}

// Function to demonstrate JSON decoding functionality
void demonstrate_json_decoding(void) {
  printf("=== ACA Decoder - JSON Output Examples ===\n");

  // Example 1: HBM FATAL ERROR (32-byte array)
  uint64_t register_array_32[RAS_DECODE_REGISTER_ARRAY_SIZE_32_BYTES] = {
      0xbaa000000004081b, 0x0, 0x209600090f00, 0x5d000000};
  printf("\n--- HBM FATAL ERROR (32-byte array) ---\n");
  printf("Decoded AFID: %d\n",
         decode_afid(register_array_32, RAS_DECODE_REGISTER_ARRAY_SIZE_32_BYTES, 0, 1, 1));

  JsonValue *json_result_32 =
      decode_error_info(register_array_32, RAS_DECODE_REGISTER_ARRAY_SIZE_32_BYTES, 0, 1, 1);
  if (json_result_32) {
    print_json_value(json_result_32);
    json_free(json_result_32);
  }

  // Example 2: GC FATAL ERROR
  uint64_t register_array_test[RAS_DECODE_REGISTER_ARRAY_SIZE_32_BYTES] = {
      0xbea00000003b0000, 0x100000029, 0x1200136430400, 0x20b};
  printf("\n--- GC FATAL ERROR ---\n");
  printf("Decoded AFID: %d\n",
         decode_afid(register_array_test, RAS_DECODE_REGISTER_ARRAY_SIZE_32_BYTES, 0, 1, 1));

  JsonValue *json_result_test =
      decode_error_info(register_array_test, RAS_DECODE_REGISTER_ARRAY_SIZE_32_BYTES, 0, 1, 1);
  if (json_result_test) {
    print_json_value(json_result_test);
    json_free(json_result_test);
  }

  // Example 3: HBM CORRECTED ERROR (128-byte array)
  uint64_t register_array_128[RAS_DECODE_REGISTER_ARRAY_SIZE_128_BYTES] = {0xffff,
                                                                           0xdc2040000000011b,
                                                                           0x0,
                                                                           0xd008000801000000,
                                                                           0x25000001ff,
                                                                           0x209600191f00,
                                                                           0xa000000,
                                                                           0x0,
                                                                           0x0,
                                                                           0x0,
                                                                           0xd008000801000000,
                                                                           0x0,
                                                                           0x0,
                                                                           0x0,
                                                                           0x0,
                                                                           0x0};

  printf("\n--- HBM CORRECTED ERROR (128-byte array) ---\n");
  printf("Decoded AFID: %d\n",
         decode_afid(register_array_128, RAS_DECODE_REGISTER_ARRAY_SIZE_128_BYTES, 0, 1, 1));

  JsonValue *json_result_128 =
      decode_error_info(register_array_128, RAS_DECODE_REGISTER_ARRAY_SIZE_128_BYTES, 0, 1, 1);
  if (json_result_128) {
    print_json_value(json_result_128);
    json_free(json_result_128);
  }

  // Example 4: PCS XGMI Error
  uint64_t register_array_pcs_xgmi[RAS_DECODE_REGISTER_ARRAY_SIZE_128_BYTES] = {0xffffffff,
                                                                                0x9820000000060150,
                                                                                0x0,
                                                                                0xd008000200000000,
                                                                                0x27000001f9,
                                                                                0xe05012109201,
                                                                                0xaf812d4a000000,
                                                                                0x0,
                                                                                0x0,
                                                                                0x0,
                                                                                0x0,
                                                                                0x0,
                                                                                0x0,
                                                                                0x0,
                                                                                0x0,
                                                                                0x0};

  printf("\n--- PCS XGMI Error ---\n");
  printf("Decoded AFID: %d\n",
         decode_afid(register_array_pcs_xgmi, RAS_DECODE_REGISTER_ARRAY_SIZE_128_BYTES, 0, 1, 1));

  JsonValue *json_result_pcs =
      decode_error_info(register_array_pcs_xgmi, RAS_DECODE_REGISTER_ARRAY_SIZE_128_BYTES, 0, 1, 1);
  if (json_result_pcs) {
    print_json_value(json_result_pcs);
    json_free(json_result_pcs);
  }

  // Example 5: Bad page (threshold exceeded flag)
  uint64_t register_array_bad_page[RAS_DECODE_REGISTER_ARRAY_SIZE_128_BYTES] = {
      0x1,           0xb000000000000137,
      0x0,           0x0,
      0x1ff00000002, 0x9600000000,
      0x0,           0x0,
      0x0,           0x0,
      0x0,           0x0,
      0x0,           0x0,
      0x0,           0x0};

  printf("\n--- Bad Page (Threshold Exceeded) ---\n");
  printf("Decoded AFID: %d\n",
         decode_afid(register_array_bad_page, RAS_DECODE_REGISTER_ARRAY_SIZE_128_BYTES,
                     RAS_DECODE_FLAG_THRESHOLD_EXCEEDED, 1, 1));

  JsonValue *json_result_bad_page =
      decode_error_info(register_array_bad_page, RAS_DECODE_REGISTER_ARRAY_SIZE_128_BYTES,
                        RAS_DECODE_FLAG_THRESHOLD_EXCEEDED, 1, 1);
  if (json_result_bad_page) {
    print_json_value(json_result_bad_page);
    json_free(json_result_bad_page);
  }

  // Example 6: Boot Error Demo
  uint64_t boot_messages[8] = {
      0x3c000228a4,  // Oam0bootmsg
      0x3c001228a4,  // Oam1bootmsg
      0x3c002228a4,  // Oam2bootmsg
      0x3c003128a4,  // Oam3bootmsg
      0x3c004328a4,  // Oam4bootmsg
      0x3c005228a4,  // Oam5bootmsg
      0x3c006228a4,  // Oam6bootmsg
      0x3c007228a4   // Oam7bootmsg
  };

  printf("\n--- Boot Error Demo ---\n");
  printf("Decoded AFID: %d\n",
         decode_afid(boot_messages, sizeof(boot_messages) / sizeof(boot_messages[0]), 0, 1, 9));

  JsonValue *json_result_boot =
      decode_error_info(boot_messages, sizeof(boot_messages) / sizeof(boot_messages[0]), 0, 1, 9);
  if (json_result_boot) {
    print_json_value(json_result_boot);
    json_free(json_result_boot);
  } else {
    printf("Failed to decode boot messages\n");
  }

  printf("\n===========================================\n");
}

int main() {
  // Display version information
  print_version_info();

  // Demonstrate the new JSON-based ACA decoding functionality
  demonstrate_json_decoding();

  return 0;
}
