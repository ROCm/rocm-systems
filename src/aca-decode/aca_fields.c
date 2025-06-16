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
 * @file aca_fields.c
 * @brief Implementation of ACA register field handling
 *
 * This file contains functions for initializing and reading various ACA register fields
 * including status, IPID, and syndrome registers. Each function
 * extracts specific bit fields from raw register values and populates corresponding
 * field structures.
 */

#include "aca_fields.h"

/**
 * @brief Extracts a bit field from a value
 * @param[in] value The source value to extract bits from
 * @param[in] start Starting bit position
 * @param[in] count Number of bits to extract
 * @param[in] type The type to cast the extracted bits to
 * @return The extracted bits as a value of the specified type
 */
#define EXTRACT_BITS(value, start, count, type) ((type)(((value) >> (start)) & ((1ULL << (count)) - 1)))

uint64_t aca_fields_read(const aca_fields_t *fields)
{
    return fields->raw_value;
}

void aca_status_init(aca_status_fields_t *fields, uint64_t status_reg)
{
    fields->base.raw_value = status_reg;
    fields->error_code = EXTRACT_BITS(status_reg, 0, 16, uint16_t);
    fields->error_code_ext = EXTRACT_BITS(status_reg, 16, 6, uint8_t);
    fields->reserv22 = EXTRACT_BITS(status_reg, 22, 2, uint8_t);
    fields->addr_lsb = EXTRACT_BITS(status_reg, 24, 6, uint8_t);
    fields->reserv30 = EXTRACT_BITS(status_reg, 30, 2, uint8_t);
    fields->err_core_id = EXTRACT_BITS(status_reg, 32, 6, uint8_t);
    fields->reserv38 = EXTRACT_BITS(status_reg, 38, 2, uint8_t);
    fields->scrub = EXTRACT_BITS(status_reg, 40, 1, uint8_t);
    fields->reserv41 = EXTRACT_BITS(status_reg, 41, 2, uint8_t);
    fields->poison = EXTRACT_BITS(status_reg, 43, 1, uint8_t);
    fields->deferred = EXTRACT_BITS(status_reg, 44, 1, uint8_t);
    fields->uecc = EXTRACT_BITS(status_reg, 45, 1, uint8_t);
    fields->cecc = EXTRACT_BITS(status_reg, 46, 1, uint8_t);
    fields->reserv47 = EXTRACT_BITS(status_reg, 47, 5, uint8_t);
    fields->synd_v = EXTRACT_BITS(status_reg, 53, 1, uint8_t);
    fields->reserv54 = EXTRACT_BITS(status_reg, 54, 1, uint8_t);
    fields->tcc = EXTRACT_BITS(status_reg, 55, 1, uint8_t);
    fields->err_core_id_val = EXTRACT_BITS(status_reg, 56, 1, uint8_t);
    fields->pcc = EXTRACT_BITS(status_reg, 57, 1, uint8_t);
    fields->addr_v = EXTRACT_BITS(status_reg, 58, 1, uint8_t);
    fields->misc_v = EXTRACT_BITS(status_reg, 59, 1, uint8_t);
    fields->en = EXTRACT_BITS(status_reg, 60, 1, uint8_t);
    fields->uc = EXTRACT_BITS(status_reg, 61, 1, uint8_t);
    fields->overflow = EXTRACT_BITS(status_reg, 62, 1, uint8_t);
    fields->val = EXTRACT_BITS(status_reg, 63, 1, uint8_t);
}

void aca_ipid_init(aca_ipid_fields_t *fields, uint64_t ipid_reg)
{
    fields->base.raw_value = ipid_reg;
    fields->instance_id_lo = EXTRACT_BITS(ipid_reg, 0, 32, uint32_t);
    fields->hardware_id = EXTRACT_BITS(ipid_reg, 32, 12, uint16_t);
    fields->instance_id_hi = EXTRACT_BITS(ipid_reg, 44, 4, uint8_t);
    fields->aca_type = EXTRACT_BITS(ipid_reg, 48, 16, uint16_t);
}

void aca_synd_init(aca_synd_fields_t *fields, uint64_t synd_reg)
{
    fields->base.raw_value = synd_reg;
    fields->error_information = EXTRACT_BITS(synd_reg, 0, 18, uint32_t);
    fields->length = EXTRACT_BITS(synd_reg, 18, 6, uint8_t);
    fields->error_priority = EXTRACT_BITS(synd_reg, 24, 3, uint8_t);
    fields->reserved27 = EXTRACT_BITS(synd_reg, 27, 5, uint8_t);
    fields->syndrome = EXTRACT_BITS(synd_reg, 32, 7, uint16_t);
    fields->reserved39 = EXTRACT_BITS(synd_reg, 39, 25, uint32_t);
}

void aca_addr_init(aca_addr_fields_t *fields, uint64_t addr_reg)
{
    fields->base.raw_value = addr_reg;
    fields->error_addr = EXTRACT_BITS(addr_reg, 0, 56, uint64_t);
    fields->reserved = EXTRACT_BITS(addr_reg, 56, 8, uint8_t);
}
