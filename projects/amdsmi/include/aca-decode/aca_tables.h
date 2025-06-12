// SPDX-License-Identifier: MIT
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
 * @file aca_tables.h
 * @brief ACA lookup table definitions and helper functions
 * @details Contains data structures and functions definitions for mapping ACA Registers
 *          into their corresponding names and types.
 */

#ifndef ACA_TABLES_H
#define ACA_TABLES_H

#include <stdint.h>
#include <stddef.h>

/**
 * @brief Structure mapping hardware ID and ACA type to bank names
 */
typedef struct
{
    uint16_t hw_id;    /**< Hardware ID value */
    uint16_t aca_type; /**< ACA type identifier */
    const char *name;  /**< Bank name string */
} aca_bank_entry_t;

/**
 * @brief Structure mapping bank-specific error codes to error types
 */
typedef struct
{
    const char *bank;    /**< Bank name string */
    uint32_t error_code; /**< Error code value */
    const char *type;    /**< Error type string */
} aca_error_type_t;

/**
 * @brief Structure for generic error code to error type mapping
 */
typedef struct
{
    uint32_t error_code; /**< Error code value */
    const char *type;    /**< Error type string */
} aca_error_entry_t;

/**
 * @brief Structure mapping instance_id_hi to OAM and AID values
 */
typedef struct
{
    uint8_t oam; /**< OAM value */
    uint8_t aid; /**< AID value */
} oam_aid_map_t;

/**
 * @brief Structure for mapping bank and instance ID LO to instance name
 */
typedef struct
{
    const char *bank;      /**< Bank name */
    uint32_t instance_id_lo;  /**< Instance ID Lo (masked with 0xFFFFFFFC) */
    const char *name;      /**< Instance name */
} aca_instance_entry_t;

// External table declarations
extern const aca_bank_entry_t bank_table[];
extern const aca_error_type_t error_table[];
extern const aca_error_entry_t xcd_error_table[];
extern const aca_error_entry_t aid_error_table[];

// Table size constants
extern const size_t NUM_BANKS;
extern const size_t NUM_ERRORS;
extern const size_t NUM_XCD_ERRORS;
extern const size_t NUM_AID_ERRORS;

/**
 * @brief Find bank name based on hardware ID and ACA type
 * @param[in] hw_id Hardware ID value
 * @param[in] aca_type ACA type value
 * @param[out] bank_name Pointer to store result string
 * @return 0 on success, 1 if not found, -1 on parameter error
 */
int find_bank_name(uint16_t hw_id, uint16_t aca_type, const char **bank_name);

/**
 * @brief Find error type for a specific bank and error code
 * @param[in] bank Bank name string
 * @param[in] error_code Error code value
 * @param[out] error_type Pointer to store result string
 * @return 0 on success, 1 if not found, -1 on parameter error
 */
int find_error_type_by_bank(const char *bank, uint32_t error_code, const char **error_type);

/**
 * @brief Generic lookup for error codes in an error table
 * @param[in] table Pointer to error table
 * @param[in] table_size Number of table entries
 * @param[in] error_code Error code to look up
 * @param[out] error_type Pointer to store result string
 * @return 0 on success, 1 if not found, -1 on parameter error
 */
int find_error_in_table(const aca_error_entry_t *table, size_t table_size,
                        uint32_t error_code, const char **error_type);

/**
 * @brief Find OAM and AID values based on instance_id_hi
 * @param[in] instance_id_hi Instance ID low value (0x00-0x0F)
 * @param[out] oam_aid Pointer to store OAM and AID values
 * @return 0 on success, 1 if not found, -1 on parameter error
 */
int find_oam_aid(uint8_t instance_id_hi, oam_aid_map_t *oam_aid);

/**
 * @brief Find instance name based on bank and instance ID
 * @param[in] bank Bank name string
 * @param[in] instance_id_lo Instance ID (will be masked with 0xFFFFFFFC)
 * @param[out] instance_name Pointer to store result string
 * @return 0 on success, 1 if not found, -1 on parameter error
 */
int find_instance_name(const char *bank, uint32_t instance_id_lo, const char **instance_name);

#endif
