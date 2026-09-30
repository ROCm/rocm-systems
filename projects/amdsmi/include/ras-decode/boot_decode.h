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

#ifndef BOOT_DECODE_H
#define BOOT_DECODE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "json_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Boot message structure representing OAM boot message
 */
typedef struct {
  uint64_t value;  ///< 64-bit boot message value
} OamBootMsg;

/**
 * @brief Decoder function pointer type
 * @param msg Boot message to decode
 * @return JsonValue containing decoded information or NULL on failure
 */
typedef JsonValue* (*boot_decoder_func_t)(OamBootMsg* msg);

/**
 * @brief Decoder mapping entry
 */
typedef struct {
  uint8_t encoding;             ///< Error encoding value
  boot_decoder_func_t decoder;  ///< Decoder function
} boot_decoder_entry_t;

/**
 * @brief Boot message constants
 */
#define BOOT_ERROR_PRESENT_MARKER 0xA4
#define BOOT_IN_BOOT_MARKER 0xBA
#define BOOT_SUCCESS_ENCODING 0xBA

/**
 * @brief Error encoding constants
 */
#define BOOT_ENCODING_HBM_TRAINING 0x01
#define BOOT_ENCODING_FW_LOAD 0x04
#define BOOT_ENCODING_WAFL_LINK 0x05
#define BOOT_ENCODING_XGMI_LINK 0x06
#define BOOT_ENCODING_USR_CP_LINK 0x07
#define BOOT_ENCODING_USR_DP_LINK 0x08
#define BOOT_ENCODING_HBM_MEM_TEST 0x09
#define BOOT_ENCODING_HBM_BIST_TEST 0x0A
#define BOOT_ENCODING_BOOT_CTRL_GEN_V0 0x0B
#define BOOT_ENCODING_BOOT_CTRL_GEN_V1 0x0C
#define BOOT_ENCODING_DATA_ABORT 0x0D

/**
 * @brief HBM stack decoder constants
 */
#define HBM_STACK_0 0x01
#define HBM_STACK_1 0x02
#define HBM_STACK_UNKNOWN -1

/**
 * @brief Extract specific byte from 64-bit value
 * @param value 64-bit value
 * @param byte_index Byte index (0-7)
 * @return Extracted byte value
 */
static inline uint8_t extract_byte(uint64_t value, int byte_index) {
  return (uint8_t)((value >> (byte_index * 8)) & 0xFF);
}

/**
 * @brief Extract specific bits mask
 * @param num_bits Number of bits to extract
 * @return Bit mask
 */
static inline uint32_t extract_bits(int num_bits) { return (1U << num_bits) - 1; }

/**
 * @brief Get boot version from boot message
 * @param msg Boot message
 * @return Boot version (0 or 1)
 */
int get_boot_version(OamBootMsg* msg);

/**
 * @brief Get error encoding from boot message
 * @param msg Boot message
 * @return Error encoding value
 */
int get_error_encoding(OamBootMsg* msg);

/**
 * @brief Check if error is present in boot message
 * @param msg Boot message
 * @return true if error present, false otherwise
 */
bool error_present(OamBootMsg* msg);

/**
 * @brief Check if in boot mode
 * @param msg Boot message
 * @return true if in boot mode, false otherwise
 */
bool in_boot(OamBootMsg* msg);

/**
 * @brief Get socket number from boot message
 * @param msg Boot message
 * @param version Boot version
 * @return Socket number
 */
int get_socket(OamBootMsg* msg, int version);

/**
 * @brief Get AID number from boot message
 * @param msg Boot message
 * @param version Boot version
 * @return AID number
 */
int get_aid(OamBootMsg* msg, int version);

/**
 * @brief Decode HBM stack value
 * @param stack Stack value
 * @return Decoded stack number or HBM_STACK_UNKNOWN
 */
int decode_hbm_stack(uint8_t stack);

/**
 * @brief Create JSON array of failed links
 * @param byte_value Byte containing link status bits
 * @param max_links Maximum number of links to check
 * @return JsonValue array or NULL on failure
 */
JsonValue* create_failed_links_array(uint8_t byte_value, int max_links);

/**
 * @brief Create hex string representation
 * @param value Value to convert
 * @param width Width of hex string (with padding)
 * @return Dynamically allocated hex string or NULL on failure
 */
char* create_hex_string(uint64_t value, int width);

// Decoder functions for Version 0
JsonValue* decode_hbm_training_v0(OamBootMsg* msg);
JsonValue* decode_fw_load_v0(OamBootMsg* msg);
JsonValue* decode_wafl_link_training_v0(OamBootMsg* msg);
JsonValue* decode_xgmi_link_training_v0(OamBootMsg* msg);
JsonValue* decode_usr_cp_link_training_v0(OamBootMsg* msg);
JsonValue* decode_usr_dp_link_training_v0(OamBootMsg* msg);
JsonValue* decode_hbm_mem_test_v0(OamBootMsg* msg);
JsonValue* decode_hbm_bist_test_v0(OamBootMsg* msg);
JsonValue* decode_boot_controller_generic_v0(OamBootMsg* msg);

// Decoder functions for Version 1
JsonValue* decode_hbm_training_v1(OamBootMsg* msg);
JsonValue* decode_fw_load_v1(OamBootMsg* msg);
JsonValue* decode_wafl_link_training_v1(OamBootMsg* msg);
JsonValue* decode_xgmi_link_training_v1(OamBootMsg* msg);
JsonValue* decode_usr_cp_link_training_v1(OamBootMsg* msg);
JsonValue* decode_usr_dp_link_training_v1(OamBootMsg* msg);
JsonValue* decode_hbm_mem_test_v1(OamBootMsg* msg);
JsonValue* decode_hbm_bist_test_v1(OamBootMsg* msg);
JsonValue* decode_boot_controller_generic_v1(OamBootMsg* msg);
JsonValue* decode_data_abort_v1(OamBootMsg* msg);
JsonValue* decode_boot_success_v1(OamBootMsg* msg);

// Unhandled error decoders
JsonValue* decode_unhandled_error_v0(OamBootMsg* msg);
JsonValue* decode_unhandled_error_v1(OamBootMsg* msg);

/**
 * @brief Get appropriate decoder function for boot message
 * @param msg Boot message
 * @return Decoder function pointer or NULL if no decoder found
 */
boot_decoder_func_t get_decoder_function(OamBootMsg* msg);

/**
 * @brief Orchestrate decoding of multiple boot messages
 * @param oam_boot_msgs Array of boot message values
 * @param count Number of boot messages
 * @return JsonValue object containing decoded results or NULL on failure
 */
JsonValue* boot_decode_orchestrator(const uint64_t* oam_boot_msgs, size_t count);

#ifdef __cplusplus
}
#endif

#endif /* BOOT_DECODE_H */
