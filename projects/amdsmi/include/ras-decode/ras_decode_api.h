// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef RAS_DECODE_API_H
#define RAS_DECODE_API_H

#include <stddef.h>
#include <stdint.h>

#include "aca_version.h"
#include "json_util.h"

/**
 * @brief Structure containing decoded error information
 */
typedef struct {
  const char* bank_ref;       /**< Reference to bank name string */
  const char* error_type_ref; /**< Reference to error type string */
  const char* severity_ref;   /**< Reference to error severity string */
  const char* category_ref;   /**< Reference to error category string */
  const char* instance_ref;   /**< Reference to instance name string */
  int oam;                    /**< OAM value */
  int aid;                    /**< AID value */
  uint64_t raw_status;        /**< Raw status register value */
  uint64_t raw_addr;          /**< Raw address register value */
  uint64_t raw_ipid;          /**< Raw IPID register value */
  uint64_t raw_synd;          /**< Raw syndrome register value */
  uint8_t scrub;              /**< Scrub bit from status */
  uint8_t poison;             /**< Poison bit from status */
  uint8_t deferred;           /**< Deferred bit from status */
  uint8_t error_code_ext;     /**< Extended error code from status */
} aca_error_info_t;

/**
 * @brief Decodes the AFID from a register array
 * @param[in] register_array Pointer to an array of 64-bit register values
 * @param[in] array_len Size of register array in elements
 * @param[in] flag Decoder flags
 * @param[in] hw_revision Hardware revision number
 * @param[in] register_context_type Register context type (16-bit): 1 for ACA decode, 9 for boot
 * decode
 * @return AFID value or -1 if decoding fails
 */
int decode_afid(const uint64_t* register_array, size_t array_len, uint32_t flag,
                uint16_t hw_revision, uint16_t register_context_type);

/**
 * @brief Decodes and returns complete error information from a register array as JSON
 * @param[in] register_array Pointer to an array of 64-bit register values
 * @param[in] array_len Size of register array in elements
 * @param[in] flag Decoder flags
 * @param[in] hw_revision Hardware revision number
 * @param[in] register_context_type Register context type (16-bit): 1 for ACA decode, 9 for boot
 * decode
 * @return JsonValue* containing complete error information, or NULL on failure
 */
JsonValue* decode_error_info(const uint64_t* register_array, size_t array_len, uint32_t flag,
                             uint16_t hw_revision, uint16_t register_context_type);

/**
 * @brief Decodes the AFID from a JSON error object based on error category, type, and severity
 * @param[in] error_json Pointer to JSON object containing error information
 * @return AFID value or -1 if decoding fails or JSON is NULL
 */
int decode_error_info_afid(JsonValue* error_json);

#endif  // RAS_DECODE_API_H
