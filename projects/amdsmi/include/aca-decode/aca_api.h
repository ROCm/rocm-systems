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

#ifndef ACA_API_H
#define ACA_API_H

#include <stdint.h>
#include <stddef.h>

/**
 * @brief Structure containing decoded error information
 */
typedef struct
{
    const char *bank_ref;       /**< Reference to bank name string */
    const char *error_type_ref; /**< Reference to error type string */
    const char *severity_ref;   /**< Reference to error severity string */
    const char *category_ref;   /**< Reference to error category string */
    const char *instance_ref;   /**< Reference to instance name string */
    int oam;                    /**< OAM value */  
    int aid;                    /**< AID value */
    int afid;                   /**< AFID value (AMD Field ID) */
    uint64_t raw_status;        /**< Raw status register value */ 
    uint64_t raw_addr;          /**< Raw address register value */
    uint64_t raw_ipid;          /**< Raw IPID register value */
    uint64_t raw_synd;          /**< Raw syndrome register value */
    uint8_t scrub;              /**< Scrub bit from status */
    uint8_t error_code_ext;     /**< Extended error code from status */
} aca_error_info_t;

/**
 * @brief Decodes the AFID from a register array
 * @param[in] register_array Pointer to an array of 64-bit register values
 * @param[in] array_len Size of register array in elements
 * @param[in] flag Decoder flags
 * @param[in] hw_revision Hardware revision number
 * @return AFID value or -1 if decoding fails
 */
int decode_afid(const uint64_t *register_array, size_t array_len, uint32_t flag, uint16_t hw_revision);

/**
 * @brief Decodes and returns complete error information from a register array
 * @param[in] register_array Pointer to an array of 64-bit register values
 * @param[in] array_len Size of register array in elements
 * @param[in] flag Decoder flags
 * @param[in] hw_revision Hardware revision number
 * @return Complete error information structure
 */
aca_error_info_t decode_error_info(const uint64_t *register_array, size_t array_len, uint32_t flag, uint16_t hw_revision);

#endif // ACA_API_H
