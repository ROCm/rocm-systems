// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * @file aca_decode.h
 * @brief Internal decoder interface and data structures
 */
#ifndef RAS_DECODE_DECODE_H
#define RAS_DECODE_DECODE_H

#include "aca_fields.h"
#include "json_util.h"
#include "ras_decode_api.h"

/**
 * @brief Internal decoder structure with parsed register fields
 */
typedef struct {
  uint64_t aca_status;  /**< Raw status register value */
  uint64_t aca_addr;    /**< Raw address register value */
  uint64_t aca_ipid;    /**< Raw IPID register value */
  uint64_t aca_synd;    /**< Raw syndrome register value */
  uint32_t flags;       /**< Decoder flags */
  uint16_t hw_revision; /**< Hardware hw_revision */

  aca_status_fields_t status; /**< Parsed status fields */
  aca_ipid_fields_t ipid;     /**< Parsed IPID fields */
  aca_synd_fields_t synd;     /**< Parsed syndrome fields */
} aca_decoder_t;

/**
 * @brief Structure containing raw ACA error data from hardware
 */
typedef struct {
  uint64_t aca_status;  /**< Raw status register value */
  uint64_t aca_addr;    /**< Raw address register value */
  uint64_t aca_ipid;    /**< Raw IPID register value */
  uint64_t aca_synd;    /**< Raw syndrome register value */
  uint32_t flags;       /**< Flags from descriptor */
  uint16_t hw_revision; /**< Hardware hw_revision number */
} aca_raw_data_t;

/**
 * @brief Main decode function that processes raw ACA error data and returns JSON
 * @param[in] raw_data Pointer to structure containing raw ACA error data
 * @return JsonValue* containing the decoded error information, or NULL on failure
 */
JsonValue* aca_decode(const aca_raw_data_t* raw_data);

#endif /* RAS_DECODE_DECODE_H */
