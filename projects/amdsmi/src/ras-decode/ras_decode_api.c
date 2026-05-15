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

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "aca_decode.h"
#include "boot_decode.h"
#include "error_map.h"
#include "json_util.h"
#include "ras_decode_constants.h"

int decode_afid(const uint64_t *register_array, size_t array_len, uint32_t flag,
                uint16_t hw_revision, uint16_t register_context_type) {
  if (!register_array) {
    return -1;
  }

  // Use decode_error_info to get the JSON result
  JsonValue *json_result =
      decode_error_info(register_array, array_len, flag, hw_revision, register_context_type);
  if (!json_result) {
    return -1;
  }

  // Use the decode_error_info_afid function to extract AFID
  int afid = decode_error_info_afid(json_result);

  json_free(json_result);
  return afid;
}

JsonValue *decode_error_info(const uint64_t *register_array, size_t array_len, uint32_t flag,
                             uint16_t hw_revision, uint16_t register_context_type) {
  if (!register_array) {
    return NULL;
  }

  // Check register context type parameter
  if (register_context_type == 9) {
    // For boot decode, use boot_decode_orchestrator with register_array and array_len
    // Flag is not used in boot decode
    return boot_decode_orchestrator(register_array, array_len);
  } else if (register_context_type == 1) {
    // For ACA decode, use existing logic
    aca_raw_data_t raw_data = {0};

    if (array_len == RAS_DECODE_REGISTER_ARRAY_SIZE_32_BYTES)  // 32 bytes
    {
      raw_data.aca_status = register_array[0];
      raw_data.aca_addr = register_array[1];
      raw_data.aca_ipid = register_array[2];
      raw_data.aca_synd = register_array[3];
    } else if (array_len == RAS_DECODE_REGISTER_ARRAY_SIZE_128_BYTES)  // 128 bytes
    {
      raw_data.aca_status = register_array[1];
      raw_data.aca_addr = register_array[2];
      raw_data.aca_ipid = register_array[5];
      raw_data.aca_synd = register_array[6];
    } else {
      return NULL;  // Unsupported size
    }

    raw_data.flags = flag;
    raw_data.hw_revision = hw_revision;

    return aca_decode(&raw_data);
  } else {
    return NULL;  // Invalid register context type
  }
}

int decode_error_info_afid(JsonValue *error_json) {
  if (!error_json || error_json->type != JSON_OBJECT) {
    return -1;  // Invalid AFID for null or invalid JSON
  }

  // Check if this is MCA error
  JsonValue *category_value = json_object_get(error_json, "error_category");
  JsonValue *type_value = json_object_get(error_json, "error_type");
  JsonValue *severity_value = json_object_get(error_json, "severity");

  if (category_value && type_value && severity_value && category_value->type == JSON_STRING &&
      type_value->type == JSON_STRING && severity_value->type == JSON_STRING) {
    const char *error_category = category_value->data.string;
    const char *error_type = type_value->data.string;
    const char *error_severity = severity_value->data.string;

    // Check for the specific case: HBM Errors + Bad Page Retirement Threshold + Fatal
    if (strcmp(error_category, RAS_DECODE_CATEGORY_HBM_ERRORS) == 0 &&
        strcmp(error_type, RAS_DECODE_ERROR_TYPE_BAD_PAGE_RETIREMENT_THRESHOLD) == 0 &&
        strcmp(error_severity, RAS_DECODE_SEVERITY_FATAL) == 0) {
      // Use the error_type directly as service_error for this case
      return get_error_id(error_category, error_type, error_severity);
    }

    // For other cases, we need to determine the service_error_type based on the logic
    // from get_service_error_type function
    const char *service_error = NULL;

    // Extract bank if needed for service error type determination
    JsonValue *bank_value = json_object_get(error_json, "bank");
    const char *error_bank =
        (bank_value && bank_value->type == JSON_STRING) ? bank_value->data.string : "";

    if (strcmp(error_type, RAS_DECODE_ERROR_TYPE_BAD_PAGE_RETIREMENT_THRESHOLD) == 0) {
      service_error = RAS_DECODE_ERROR_TYPE_BAD_PAGE_RETIREMENT_THRESHOLD;
    } else if (strcmp(error_category, RAS_DECODE_CATEGORY_HBM_ERRORS) == 0 &&
               strcmp(error_severity, RAS_DECODE_SEVERITY_CORRECTED) == 0) {
      service_error = RAS_DECODE_ERROR_TYPE_ALL;
    } else if (strcmp(error_type, "RdCrcErr") == 0) {
      service_error = RAS_DECODE_ERROR_TYPE_END_TO_END_CRC;
    } else if (strcmp(error_category, RAS_DECODE_CATEGORY_HBM_ERRORS) == 0 &&
               strcmp(error_severity, RAS_DECODE_SEVERITY_FATAL) == 0 &&
               strcmp(error_type, RAS_DECODE_ERROR_TYPE_ON_DIE_ECC) != 0 &&
               strcmp(error_type, RAS_DECODE_ERROR_TYPE_END_TO_END_CRC) != 0) {
      service_error = RAS_DECODE_ERROR_TYPE_ALL_OTHERS;
    } else if (strcmp(error_category, RAS_DECODE_CATEGORY_DEVICE_INTERNAL_ERRORS) == 0) {
      if ((strcmp(error_severity, RAS_DECODE_SEVERITY_UNCORRECTED_NON_FATAL) == 0 ||
           strcmp(error_severity, RAS_DECODE_SEVERITY_CORRECTED) == 0 ||
           strcmp(error_severity, RAS_DECODE_SEVERITY_FATAL) == 0) &&
          strcmp(error_type, RAS_DECODE_ERROR_TYPE_HARDWARE_ASSERTION) != 0 &&
          strcmp(error_type, RAS_DECODE_ERROR_TYPE_WATCHDOG_TIMEOUT) != 0) {
        service_error = RAS_DECODE_ERROR_TYPE_ALL_OTHERS;
      }
    } else if (strcmp(error_category, RAS_DECODE_CATEGORY_OFF_PACKAGE_LINK_ERRORS) == 0) {
      if (strcmp(error_bank, RAS_DECODE_BANK_PCS_XGMI) == 0) {
        service_error = RAS_DECODE_ERROR_TYPE_XGMI;
      } else if (strcmp(error_bank, RAS_DECODE_BANK_KPX_WAFL) == 0) {
        service_error = RAS_DECODE_ERROR_TYPE_WAFL;
      }
    }

    if (!service_error) {
      service_error = error_type;  // Fallback to error_type
    }

    return get_error_id(error_category, service_error, error_severity);
  }

  // Check if this is a boot error
  // Find the first msg<i> key to get the error_type
  JsonPair *current_pair = error_json->data.object;
  JsonValue *first_msg = NULL;
  int lowest_msg_index = INT_MAX;

  while (current_pair) {
    if (strncmp(current_pair->key, "msg", 3) == 0) {
      // Extract the message index
      int msg_index = atoi(current_pair->key + 3);
      if (msg_index < lowest_msg_index) {
        lowest_msg_index = msg_index;
        first_msg = current_pair->value;
      }
    }
    current_pair = current_pair->next;
  }

  if (first_msg && first_msg->type == JSON_OBJECT) {
    // This is a boot error - extract error_type from the first message
    JsonValue *boot_error_type = json_object_get(first_msg, "error_type");
    if (boot_error_type && boot_error_type->type == JSON_STRING) {
      const char *service_error = NULL;
      service_error = boot_error_type->data.string;

      // For boot errors, always use Boot-Time Errors category and Fail-to-init severity
      return get_error_id(RAS_DECODE_CATEGORY_BOOT_TIME_ERRORS, service_error,
                          RAS_DECODE_SEVERITY_FAIL_TO_INIT);
    }
  }

  return -1;  // Invalid AFID if neither MCA nor boot error format
}
