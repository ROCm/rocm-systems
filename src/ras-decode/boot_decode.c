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

#include "boot_decode.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ras_decode_constants.h"

// Boot decoder mapping tables
static const boot_decoder_entry_t boot_decoder_map_v0[] = {
    {BOOT_ENCODING_HBM_TRAINING, decode_hbm_training_v0},
    {BOOT_ENCODING_FW_LOAD, decode_fw_load_v0},
    {BOOT_ENCODING_WAFL_LINK, decode_wafl_link_training_v0},
    {BOOT_ENCODING_XGMI_LINK, decode_xgmi_link_training_v0},
    {BOOT_ENCODING_USR_CP_LINK, decode_usr_cp_link_training_v0},
    {BOOT_ENCODING_USR_DP_LINK, decode_usr_dp_link_training_v0},
    {BOOT_ENCODING_HBM_MEM_TEST, decode_hbm_mem_test_v0},
    {BOOT_ENCODING_HBM_BIST_TEST, decode_hbm_bist_test_v0},
    {BOOT_ENCODING_BOOT_CTRL_GEN_V0, decode_boot_controller_generic_v0},
    {0, NULL}  // Sentinel
};

static const boot_decoder_entry_t boot_decoder_map_v1[] = {
    {BOOT_ENCODING_HBM_TRAINING, decode_hbm_training_v1},
    {BOOT_ENCODING_FW_LOAD, decode_fw_load_v1},
    {BOOT_ENCODING_WAFL_LINK, decode_wafl_link_training_v1},
    {BOOT_ENCODING_XGMI_LINK, decode_xgmi_link_training_v1},
    {BOOT_ENCODING_USR_CP_LINK, decode_usr_cp_link_training_v1},
    {BOOT_ENCODING_USR_DP_LINK, decode_usr_dp_link_training_v1},
    {BOOT_ENCODING_HBM_MEM_TEST, decode_hbm_mem_test_v1},
    {BOOT_ENCODING_HBM_BIST_TEST, decode_hbm_bist_test_v1},
    {BOOT_ENCODING_BOOT_CTRL_GEN_V1, decode_boot_controller_generic_v1},
    {BOOT_ENCODING_DATA_ABORT, decode_data_abort_v1},
    {BOOT_SUCCESS_ENCODING, decode_boot_success_v1},
    {0, NULL}  // Sentinel
};

int get_boot_version(OamBootMsg *msg) {
  if (!msg) return 0;
  return extract_byte(msg->value, 1) >> 5;
}

int get_error_encoding(OamBootMsg *msg) {
  if (!msg) return 0;
  return (int)(extract_byte(msg->value, 1) & extract_bits(5));
}

bool error_present(OamBootMsg *msg) {
  if (!msg) return false;
  return extract_byte(msg->value, 0) == BOOT_ERROR_PRESENT_MARKER;
}

bool in_boot(OamBootMsg *msg) {
  if (!msg) return false;
  return extract_byte(msg->value, 0) == BOOT_IN_BOOT_MARKER;
}

int get_socket(OamBootMsg *msg, int version) {
  if (!msg) return 0;

  if (version == 0) {
    return extract_byte(msg->value, 4);
  } else {
    return (int)((extract_byte(msg->value, 2) >> 4) & extract_bits(4));
  }
}

int get_aid(OamBootMsg *msg, int version) {
  if (!msg) return 0;

  if (version == 0) {
    return extract_byte(msg->value, 5);
  } else {
    return (int)(extract_byte(msg->value, 2) & extract_bits(4));
  }
}

int decode_hbm_stack(uint8_t stack) {
  switch (stack) {
    case HBM_STACK_0:
      return 0;
    case HBM_STACK_1:
      return 1;
    default:
      return HBM_STACK_UNKNOWN;
  }
}

JsonValue *create_failed_links_array(uint8_t byte_value, int max_links) {
  JsonValue *array = json_create_array();
  if (!array) return NULL;

  for (int i = 0; i < max_links; i++) {
    if ((byte_value >> i) & 0x1) {
      JsonValue *link_num = json_create_number(i);
      if (link_num) {
        json_array_push(array, link_num);
      }
    }
  }

  return array;
}

char *create_hex_string(uint64_t value, int width) {
  if (width < 0) return NULL;
  size_t buffer_size = (size_t)width + 3U;  // '0' + 'x' + width digits + '\0'
  char *hex_str = malloc(buffer_size);
  if (!hex_str) return NULL;

  snprintf(hex_str, buffer_size, "0x%0*llX", width, (unsigned long long)value);
  return hex_str;
}

// Version 0 decoder implementations
JsonValue *decode_hbm_training_v0(OamBootMsg *msg) {
  if (!msg) return NULL;

  JsonValue *result = json_create_object();
  if (!result) return NULL;

  int version = get_boot_version(msg);
  uint8_t byte3 = extract_byte(msg->value, 3);
  uint8_t byte2 = extract_byte(msg->value, 2);

  json_object_set(result, "error_type", json_create_string(RAS_DECODE_ERROR_TYPE_HBM_TRAINING));
  json_object_set(result, "socket", json_create_number(get_socket(msg, version)));
  json_object_set(result, "aid", json_create_number(get_aid(msg, version)));
  json_object_set(result, "hbm_stack", json_create_number(decode_hbm_stack(byte3)));
  json_object_set(result, "hbm_channel", json_create_number(byte2));

  return result;
}

JsonValue *decode_fw_load_v0(OamBootMsg *msg) {
  if (!msg) return NULL;

  JsonValue *result = json_create_object();
  if (!result) return NULL;

  int version = get_boot_version(msg);
  uint8_t byte3 = extract_byte(msg->value, 3);
  uint8_t byte2 = extract_byte(msg->value, 2);
  uint16_t fw_id = (uint16_t)((byte3 << 8) | byte2);

  char *fw_id_str = create_hex_string(fw_id, 4);

  json_object_set(result, "error_type", json_create_string(RAS_DECODE_ERROR_TYPE_FW_LOAD));
  json_object_set(result, "socket", json_create_number(get_socket(msg, version)));
  json_object_set(result, "aid", json_create_number(get_aid(msg, version)));
  json_object_set(result, "fw_id", json_create_string(fw_id_str ? fw_id_str : "0x0000"));

  free(fw_id_str);
  return result;
}

JsonValue *decode_wafl_link_training_v0(OamBootMsg *msg) {
  if (!msg) return NULL;

  JsonValue *result = json_create_object();
  if (!result) return NULL;

  int version = get_boot_version(msg);
  uint8_t byte2 = extract_byte(msg->value, 2);
  JsonValue *failed_links = create_failed_links_array(byte2, 2);

  json_object_set(result, "error_type",
                  json_create_string(RAS_DECODE_ERROR_TYPE_WAFL_LINK_TRAINING));
  json_object_set(result, "socket", json_create_number(get_socket(msg, version)));
  json_object_set(result, "aid", json_create_number(get_aid(msg, version)));
  json_object_set(result, "failed_links", failed_links ? failed_links : json_create_array());

  return result;
}

JsonValue *decode_xgmi_link_training_v0(OamBootMsg *msg) {
  if (!msg) return NULL;

  JsonValue *result = json_create_object();
  if (!result) return NULL;

  int version = get_boot_version(msg);
  uint8_t byte2 = extract_byte(msg->value, 2);
  JsonValue *failed_links = create_failed_links_array(byte2, 8);

  json_object_set(result, "error_type",
                  json_create_string(RAS_DECODE_ERROR_TYPE_XGMI_LINK_TRAINING));
  json_object_set(result, "socket", json_create_number(get_socket(msg, version)));
  json_object_set(result, "aid", json_create_number(get_aid(msg, version)));
  json_object_set(result, "failed_links", failed_links ? failed_links : json_create_array());

  return result;
}

JsonValue *decode_usr_cp_link_training_v0(OamBootMsg *msg) {
  if (!msg) return NULL;

  JsonValue *result = json_create_object();
  if (!result) return NULL;

  int version = get_boot_version(msg);
  uint8_t byte2 = extract_byte(msg->value, 2);
  JsonValue *failed_links = create_failed_links_array(byte2, 2);

  json_object_set(result, "error_type",
                  json_create_string(RAS_DECODE_ERROR_TYPE_USR_CP_LINK_TRAINING));
  json_object_set(result, "socket", json_create_number(get_socket(msg, version)));
  json_object_set(result, "aid", json_create_number(get_aid(msg, version)));
  json_object_set(result, "failed_links", failed_links ? failed_links : json_create_array());

  return result;
}

JsonValue *decode_usr_dp_link_training_v0(OamBootMsg *msg) {
  if (!msg) return NULL;

  JsonValue *result = json_create_object();
  if (!result) return NULL;

  int version = get_boot_version(msg);
  uint8_t byte2 = extract_byte(msg->value, 2);
  JsonValue *failed_links = create_failed_links_array(byte2, 4);

  json_object_set(result, "error_type",
                  json_create_string(RAS_DECODE_ERROR_TYPE_USR_DP_LINK_TRAINING));
  json_object_set(result, "socket", json_create_number(get_socket(msg, version)));
  json_object_set(result, "aid", json_create_number(get_aid(msg, version)));
  json_object_set(result, "failed_links", failed_links ? failed_links : json_create_array());

  return result;
}

JsonValue *decode_hbm_mem_test_v0(OamBootMsg *msg) {
  if (!msg) return NULL;

  JsonValue *result = json_create_object();
  if (!result) return NULL;

  int version = get_boot_version(msg);
  uint8_t byte3 = extract_byte(msg->value, 3);
  uint8_t byte2 = extract_byte(msg->value, 2);

  json_object_set(result, "error_type", json_create_string(RAS_DECODE_ERROR_TYPE_HBM_MEMORY_TEST));
  json_object_set(result, "socket", json_create_number(get_socket(msg, version)));
  json_object_set(result, "aid", json_create_number(get_aid(msg, version)));
  json_object_set(result, "hbm_stack", json_create_number(decode_hbm_stack(byte3)));
  json_object_set(result, "hbm_channel", json_create_number(byte2));

  return result;
}

JsonValue *decode_hbm_bist_test_v0(OamBootMsg *msg) {
  if (!msg) return NULL;

  JsonValue *result = json_create_object();
  if (!result) return NULL;

  int version = get_boot_version(msg);
  uint8_t byte3 = extract_byte(msg->value, 3);
  uint8_t byte2 = extract_byte(msg->value, 2);

  json_object_set(result, "error_type", json_create_string(RAS_DECODE_ERROR_TYPE_HBM_BIST_TEST));
  json_object_set(result, "socket", json_create_number(get_socket(msg, version)));
  json_object_set(result, "aid", json_create_number(get_aid(msg, version)));
  json_object_set(result, "hbm_stack", json_create_number(decode_hbm_stack(byte3)));
  json_object_set(result, "hbm_channel", json_create_number(byte2));

  return result;
}

JsonValue *decode_boot_controller_generic_v0(OamBootMsg *msg) {
  if (!msg) return NULL;

  JsonValue *result = json_create_object();
  if (!result) return NULL;

  int version = get_boot_version(msg);

  json_object_set(result, "error_type",
                  json_create_string(RAS_DECODE_ERROR_TYPE_BOOT_CONTROLLER_GENERIC));
  json_object_set(result, "socket", json_create_number(get_socket(msg, version)));
  json_object_set(result, "aid", json_create_number(get_aid(msg, version)));

  return result;
}

// Version 1 decoder implementations
JsonValue *decode_hbm_training_v1(OamBootMsg *msg) {
  if (!msg) return NULL;

  JsonValue *result = json_create_object();
  if (!result) return NULL;

  int version = get_boot_version(msg);
  uint8_t byte5 = extract_byte(msg->value, 5);
  uint8_t byte4 = extract_byte(msg->value, 4);

  json_object_set(result, "error_type", json_create_string(RAS_DECODE_ERROR_TYPE_HBM_TRAINING));
  json_object_set(result, "socket", json_create_number(get_socket(msg, version)));
  json_object_set(result, "aid", json_create_number(get_aid(msg, version)));
  json_object_set(result, "hbm_stack", json_create_number(decode_hbm_stack(byte5)));
  json_object_set(result, "hbm_channel", json_create_number(byte4));

  return result;
}

JsonValue *decode_fw_load_v1(OamBootMsg *msg) {
  if (!msg) return NULL;

  JsonValue *result = json_create_object();
  if (!result) return NULL;

  int version = get_boot_version(msg);
  uint8_t byte5 = extract_byte(msg->value, 5);
  uint8_t byte4 = extract_byte(msg->value, 4);
  uint16_t fw_id = (uint16_t)((byte5 << 8) | byte4);

  char *fw_id_str = create_hex_string(fw_id, 4);

  json_object_set(result, "error_type", json_create_string(RAS_DECODE_ERROR_TYPE_FW_LOAD));
  json_object_set(result, "socket", json_create_number(get_socket(msg, version)));
  json_object_set(result, "aid", json_create_number(get_aid(msg, version)));
  json_object_set(result, "fw_id", json_create_string(fw_id_str ? fw_id_str : "0x0000"));

  free(fw_id_str);
  return result;
}

JsonValue *decode_wafl_link_training_v1(OamBootMsg *msg) {
  if (!msg) return NULL;

  JsonValue *result = json_create_object();
  if (!result) return NULL;

  int version = get_boot_version(msg);
  uint8_t byte4 = extract_byte(msg->value, 4);
  JsonValue *failed_links = create_failed_links_array(byte4, 2);

  json_object_set(result, "error_type",
                  json_create_string(RAS_DECODE_ERROR_TYPE_WAFL_LINK_TRAINING));
  json_object_set(result, "socket", json_create_number(get_socket(msg, version)));
  json_object_set(result, "aid", json_create_number(get_aid(msg, version)));
  json_object_set(result, "failed_links", failed_links ? failed_links : json_create_array());

  return result;
}

JsonValue *decode_xgmi_link_training_v1(OamBootMsg *msg) {
  if (!msg) return NULL;

  JsonValue *result = json_create_object();
  if (!result) return NULL;

  int version = get_boot_version(msg);
  uint8_t byte4 = extract_byte(msg->value, 4);
  JsonValue *failed_links = create_failed_links_array(byte4, 8);

  json_object_set(result, "error_type",
                  json_create_string(RAS_DECODE_ERROR_TYPE_XGMI_LINK_TRAINING));
  json_object_set(result, "socket", json_create_number(get_socket(msg, version)));
  json_object_set(result, "aid", json_create_number(get_aid(msg, version)));
  json_object_set(result, "failed_links", failed_links ? failed_links : json_create_array());

  return result;
}

JsonValue *decode_usr_cp_link_training_v1(OamBootMsg *msg) {
  if (!msg) return NULL;

  JsonValue *result = json_create_object();
  if (!result) return NULL;

  int version = get_boot_version(msg);
  uint8_t byte4 = extract_byte(msg->value, 4);
  JsonValue *failed_links = create_failed_links_array(byte4, 2);

  json_object_set(result, "error_type",
                  json_create_string(RAS_DECODE_ERROR_TYPE_USR_CP_LINK_TRAINING));
  json_object_set(result, "socket", json_create_number(get_socket(msg, version)));
  json_object_set(result, "aid", json_create_number(get_aid(msg, version)));
  json_object_set(result, "failed_links", failed_links ? failed_links : json_create_array());

  return result;
}

JsonValue *decode_usr_dp_link_training_v1(OamBootMsg *msg) {
  if (!msg) return NULL;

  JsonValue *result = json_create_object();
  if (!result) return NULL;

  int version = get_boot_version(msg);
  uint8_t byte4 = extract_byte(msg->value, 4);
  JsonValue *failed_links = create_failed_links_array(byte4, 4);

  json_object_set(result, "error_type",
                  json_create_string(RAS_DECODE_ERROR_TYPE_USR_DP_LINK_TRAINING));
  json_object_set(result, "socket", json_create_number(get_socket(msg, version)));
  json_object_set(result, "aid", json_create_number(get_aid(msg, version)));
  json_object_set(result, "failed_links", failed_links ? failed_links : json_create_array());

  return result;
}

JsonValue *decode_hbm_mem_test_v1(OamBootMsg *msg) {
  if (!msg) return NULL;

  JsonValue *result = json_create_object();
  if (!result) return NULL;

  int version = get_boot_version(msg);
  uint8_t byte5 = extract_byte(msg->value, 5);
  uint8_t byte4 = extract_byte(msg->value, 4);

  json_object_set(result, "error_type", json_create_string(RAS_DECODE_ERROR_TYPE_HBM_MEMORY_TEST));
  json_object_set(result, "socket", json_create_number(get_socket(msg, version)));
  json_object_set(result, "aid", json_create_number(get_aid(msg, version)));
  json_object_set(result, "hbm_stack", json_create_number(decode_hbm_stack(byte5)));
  json_object_set(result, "hbm_channel", json_create_number(byte4));

  return result;
}

JsonValue *decode_hbm_bist_test_v1(OamBootMsg *msg) {
  if (!msg) return NULL;

  JsonValue *result = json_create_object();
  if (!result) return NULL;

  int version = get_boot_version(msg);
  uint8_t byte5 = extract_byte(msg->value, 5);
  uint8_t byte4 = extract_byte(msg->value, 4);

  json_object_set(result, "error_type", json_create_string(RAS_DECODE_ERROR_TYPE_HBM_BIST_TEST));
  json_object_set(result, "socket", json_create_number(get_socket(msg, version)));
  json_object_set(result, "aid", json_create_number(get_aid(msg, version)));
  json_object_set(result, "hbm_stack", json_create_number(decode_hbm_stack(byte5)));
  json_object_set(result, "hbm_channel", json_create_number(byte4));

  return result;
}

JsonValue *decode_boot_controller_generic_v1(OamBootMsg *msg) {
  if (!msg) return NULL;

  JsonValue *result = json_create_object();
  if (!result) return NULL;

  int version = get_boot_version(msg);
  uint8_t byte4 = extract_byte(msg->value, 4);
  uint8_t byte0 = extract_byte(msg->value, 0);
  uint8_t byte5 = extract_byte(msg->value, 5);
  uint8_t byte6 = extract_byte(msg->value, 6);
  uint8_t byte7 = extract_byte(msg->value, 7);

  char *boot_step_str = create_hex_string(byte4, 2);
  uint32_t boot_status = (uint32_t)((byte7 << 24) | (byte6 << 16) | (byte5 << 8) | byte0);
  char *boot_status_str = create_hex_string(boot_status, 8);

  json_object_set(result, "error_type",
                  json_create_string(RAS_DECODE_ERROR_TYPE_BOOT_CONTROLLER_GENERIC));
  json_object_set(result, "socket", json_create_number(get_socket(msg, version)));
  json_object_set(result, "aid", json_create_number(get_aid(msg, version)));
  json_object_set(result, "last_successful_boot_step_number",
                  json_create_string(boot_step_str ? boot_step_str : "0x00"));
  json_object_set(result, "fw_boot_status",
                  json_create_string(boot_status_str ? boot_status_str : "0x00000000"));

  free(boot_step_str);
  free(boot_status_str);
  return result;
}

JsonValue *decode_data_abort_v1(OamBootMsg *msg) {
  if (!msg) return NULL;

  JsonValue *result = json_create_object();
  if (!result) return NULL;

  int version = get_boot_version(msg);
  uint8_t byte3 = extract_byte(msg->value, 3);
  uint8_t byte4 = extract_byte(msg->value, 4);
  uint8_t byte5 = extract_byte(msg->value, 5);
  uint8_t byte6 = extract_byte(msg->value, 6);
  uint8_t byte7 = extract_byte(msg->value, 7);

  char *boot_step_str = create_hex_string(byte3, 2);
  uint32_t exception_addr = (uint32_t)((byte7 << 24) | (byte6 << 16) | (byte5 << 8) | byte4);
  char *exception_addr_str = create_hex_string(exception_addr, 8);

  json_object_set(result, "error_type",
                  json_create_string(RAS_DECODE_ERROR_TYPE_BOOT_CONTROLLER_DATA_ABORT));
  json_object_set(result, "socket", json_create_number(get_socket(msg, version)));
  json_object_set(result, "aid", json_create_number(get_aid(msg, version)));
  json_object_set(result, "last_successful_boot_step_number",
                  json_create_string(boot_step_str ? boot_step_str : "0x00"));
  json_object_set(result, "exception_address",
                  json_create_string(exception_addr_str ? exception_addr_str : "0x00000000"));

  free(boot_step_str);
  free(exception_addr_str);
  return result;
}

JsonValue *decode_boot_success_v1(OamBootMsg *msg) {
  if (!msg) return NULL;

  JsonValue *result = json_create_object();
  if (!result) return NULL;

  uint8_t byte4 = extract_byte(msg->value, 4);
  uint8_t byte0 = extract_byte(msg->value, 0);
  uint8_t byte5 = extract_byte(msg->value, 5);
  uint8_t byte6 = extract_byte(msg->value, 6);
  uint8_t byte7 = extract_byte(msg->value, 7);

  char *boot_step_str = create_hex_string(byte4, 2);
  uint32_t boot_status = (uint32_t)((byte7 << 24) | (byte6 << 16) | (byte5 << 8) | byte0);
  char *boot_status_str = create_hex_string(boot_status, 8);

  json_object_set(result, "error_type", json_create_string(RAS_DECODE_ERROR_TYPE_BOOT_SUCCESS));
  json_object_set(result, "last_successful_boot_step_number",
                  json_create_string(boot_step_str ? boot_step_str : "0x00"));
  json_object_set(result, "fw_boot_status",
                  json_create_string(boot_status_str ? boot_status_str : "0x00000000"));

  free(boot_step_str);
  free(boot_status_str);
  return result;
}

// Unhandled error decoders
JsonValue *decode_unhandled_error_v0(OamBootMsg *msg) {
  (void)msg;  // Suppress unused parameter warning
  JsonValue *result = json_create_object();
  if (!result) return NULL;

  json_object_set(result, "error_type", json_create_string(RAS_DECODE_ERROR_TYPE_UNHANDLED));

  return result;
}

JsonValue *decode_unhandled_error_v1(OamBootMsg *msg) {
  if (!msg) return NULL;

  JsonValue *result = json_create_object();
  if (!result) return NULL;

  uint8_t byte4 = extract_byte(msg->value, 4);
  uint8_t byte0 = extract_byte(msg->value, 0);
  uint8_t byte5 = extract_byte(msg->value, 5);
  uint8_t byte6 = extract_byte(msg->value, 6);
  uint8_t byte7 = extract_byte(msg->value, 7);

  char *boot_step_str = create_hex_string(byte4, 2);
  uint32_t boot_status = (uint32_t)((byte7 << 24) | (byte6 << 16) | (byte5 << 8) | byte0);
  char *boot_status_str = create_hex_string(boot_status, 8);

  json_object_set(result, "error_type", json_create_string(RAS_DECODE_ERROR_TYPE_UNHANDLED));
  json_object_set(result, "last_successful_boot_step_number",
                  json_create_string(boot_step_str ? boot_step_str : "0x00"));
  json_object_set(result, "fw_boot_status",
                  json_create_string(boot_status_str ? boot_status_str : "0x00000000"));

  free(boot_step_str);
  free(boot_status_str);
  return result;
}

boot_decoder_func_t get_decoder_function(OamBootMsg *msg) {
  if (!msg) return NULL;

  uint8_t byte0 = extract_byte(msg->value, 0);
  if (byte0 == BOOT_IN_BOOT_MARKER) {
    int version = get_boot_version(msg);
    if (version == 1) {
      return decode_boot_success_v1;
    }
  }

  int version = get_boot_version(msg);
  int encoding = get_error_encoding(msg);

  const boot_decoder_entry_t *decoder_map =
      (version == 0) ? boot_decoder_map_v0 : boot_decoder_map_v1;

  for (int i = 0; decoder_map[i].decoder != NULL; i++) {
    if (decoder_map[i].encoding == encoding) {
      return decoder_map[i].decoder;
    }
  }

  return NULL;  // No decoder found
}

JsonValue *boot_decode_orchestrator(const uint64_t *oam_boot_msgs, size_t count) {
  if (!oam_boot_msgs || count == 0) return NULL;

  JsonValue *results = json_create_object();
  if (!results) return NULL;

  // Convert to OamBootMsg structures
  OamBootMsg *msgs = malloc(count * sizeof(OamBootMsg));
  if (!msgs) {
    json_free(results);
    return NULL;
  }

  for (size_t i = 0; i < count; i++) {
    msgs[i].value = oam_boot_msgs[i];
  }

  // Check error markers across all messages
  size_t messages_with_markers = 0;
  bool *has_marker = malloc(count * sizeof(bool));
  if (!has_marker) {
    free(msgs);
    json_free(results);
    return NULL;
  }

  // Count messages with error markers (0xA4) or boot markers (0xBA)
  for (size_t i = 0; i < count; i++) {
    has_marker[i] = error_present(&msgs[i]) || in_boot(&msgs[i]);
    if (has_marker[i]) {
      messages_with_markers++;
    }
  }

  // Determine decoding strategy based on the presence of error markers
  bool decode_all_as_unhandled = (messages_with_markers == 0);
  bool decode_only_marked = (messages_with_markers > 0 && messages_with_markers < count);
  bool decode_all_normally = (messages_with_markers == count);

  // Check if all decoders are NULL (for unhandled error handling)
  bool all_decoders_none = true;
  if (!decode_all_as_unhandled) {
    for (size_t i = 0; i < count; i++) {
      if (has_marker[i] && get_decoder_function(&msgs[i]) != NULL) {
        all_decoders_none = false;
        break;
      }
    }
  }

  // Process each message
  for (size_t i = 0; i < count; i++) {
    char msg_key[32];
    snprintf(msg_key, sizeof(msg_key), "msg%zu", i);

    // Skip messages without markers if we're in selective decode mode
    if (decode_only_marked && !has_marker[i]) {
      continue;
    }

    JsonValue *msg_result = json_create_object();
    if (!msg_result) continue;

    boot_decoder_func_t decoder_func = NULL;

    if (decode_all_as_unhandled) {
      // Rule 3: No messages have markers, decode all as UNHANDLED
      decoder_func = decode_unhandled_error_v1;
    } else if (has_marker[i] || decode_all_normally) {
      // Rule 1 & 2: Decode messages with markers (or all if all have markers)
      if (all_decoders_none) {
        // Use unhandled error decoders
        int encoding = get_error_encoding(&msgs[i]);
        decoder_func = (encoding == 0) ? decode_unhandled_error_v0 : decode_unhandled_error_v1;
      } else {
        decoder_func = get_decoder_function(&msgs[i]);
      }
    }
    // If no decoder function is found, skip this message

    if (decoder_func) {
      JsonValue *decoded = decoder_func(&msgs[i]);
      if (decoded) {
        // Copy all fields from decoded result to msg_result
        for (JsonPair *pair = decoded->data.object; pair != NULL; pair = pair->next) {
          // Create a copy of the value for the new object
          JsonValue *value_copy = NULL;
          switch (pair->value->type) {
            case JSON_STRING:
              value_copy = json_create_string(pair->value->data.string);
              break;
            case JSON_NUMBER:
              value_copy = json_create_number(pair->value->data.number);
              break;
            case JSON_BOOL:
              value_copy = json_create_bool(pair->value->data.boolean);
              break;
            case JSON_NULL:
              value_copy = json_create_null();
              break;
            case JSON_ARRAY:
              // For arrays, we need to copy each element
              value_copy = json_create_array();
              if (value_copy) {
                for (size_t j = 0; j < pair->value->data.array.count; j++) {
                  JsonValue *elem = pair->value->data.array.items[j];
                  JsonValue *elem_copy = NULL;
                  if (elem->type == JSON_NUMBER) {
                    elem_copy = json_create_number(elem->data.number);
                  }
                  if (elem_copy) {
                    json_array_push(value_copy, elem_copy);
                  }
                }
              }
              break;
            default:
              break;
          }

          if (value_copy) {
            json_object_set(msg_result, pair->key, value_copy);
          }
        }
        json_free(decoded);
      }
    }

    json_object_set(results, msg_key, msg_result);
  }

  free(msgs);
  free(has_marker);
  return results;
}
