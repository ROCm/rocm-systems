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
 * @file aca_fields.h
 * @brief ACA register field definitions and manipulation functions
 *
 * Contains structures and functions for decoding and handling
 * ACA register fields. It provides field
 * definitions for status, IPID, and syndrome registers, along with
 * functions to initialize and access these fields.
 */
#ifndef ACA_FIELDS_H
#define ACA_FIELDS_H

#include <stdint.h>

/**
 * @brief Base structure for ACA fields containing raw register value
 */
typedef struct
{
    uint64_t raw_value; /**< Raw 64-bit register value */
} aca_fields_t;

/**
 * @brief Structure containing decoded ACA status register fields
 */
typedef struct
{
    aca_fields_t base;
    uint16_t error_code;
    uint8_t error_code_ext;
    uint8_t reserv22;
    uint8_t addr_lsb;
    uint8_t reserv30;
    uint8_t err_core_id;
    uint8_t reserv38;
    uint8_t scrub;
    uint8_t reserv41;
    uint8_t poison;
    uint8_t deferred;
    uint8_t uecc;
    uint8_t cecc;
    uint8_t reserv47;
    uint8_t synd_v;
    uint8_t reserv54;
    uint8_t tcc;
    uint8_t err_core_id_val;
    uint8_t pcc;
    uint8_t addr_v;
    uint8_t misc_v;
    uint8_t en;
    uint8_t uc;
    uint8_t overflow;
    uint8_t val;
} aca_status_fields_t;

/**
 * @brief Structure containing decoded ACA IPID register fields
 */
typedef struct
{
    aca_fields_t base;
    uint32_t instance_id_lo;
    uint16_t hardware_id;
    uint16_t aca_type;
    uint8_t instance_id_hi;
} aca_ipid_fields_t;

/**
 * @brief Structure containing decoded ACA syndrome register fields
 */
typedef struct
{
    aca_fields_t base;
    uint32_t error_information;
    uint8_t length;
    uint8_t error_priority;
    uint8_t reserved27;
    uint16_t syndrome;
    uint32_t reserved39;
} aca_synd_fields_t;

/**
 * @brief Structure containing decoded ACA address register fields
 */
typedef struct
{
    aca_fields_t base;
    uint64_t error_addr;
    uint64_t reserved;
} aca_addr_fields_t;

/**
 * @brief Reads the raw value from an ACA field structure
 * @param[in] fields Pointer to the ACA fields structure
 * @return The raw 64-bit value stored in the structure
 */
uint64_t aca_fields_read(const aca_fields_t *fields);

/**
 * @brief Initializes ACA status fields from a raw status register value
 * @param[out] fields Pointer to the status fields structure to initialize
 * @param[in] status_reg Raw 64-bit status register value
 */
void aca_status_init(aca_status_fields_t *fields, uint64_t status_reg);

/**
 * @brief Initializes ACA IPID fields from a raw IPID register value
 * @param[out] fields Pointer to the IPID fields structure to initialize
 * @param[in] ipid_reg Raw 64-bit IPID register value
 */
void aca_ipid_init(aca_ipid_fields_t *fields, uint64_t ipid_reg);

/**
 * @brief Initializes ACA syndrome fields from a raw syndrome register value
 * @param[out] fields Pointer to the syndrome fields structure to initialize
 * @param[in] synd_reg Raw 64-bit syndrome register value
 */
void aca_synd_init(aca_synd_fields_t *fields, uint64_t synd_reg);

/**
 * @brief Initializes ACA address fields from a raw address register value
 * @param[out] fields Pointer to the address fields structure to initialize
 * @param[in] addr_reg Raw 64-bit address register value
 */
void aca_addr_init(aca_addr_fields_t *fields, uint64_t addr_reg);

#endif
