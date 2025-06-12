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
 * @file aca_decode.h
 * @brief Internal decoder interface and data structures
 */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef ACA_DECODE_H
#define ACA_DECODE_H

#include "aca_api.h"
#include "aca_fields.h"

/**
 * @brief Internal decoder structure with parsed register fields
 */
typedef struct
{
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
typedef struct
{
    uint64_t aca_status;  /**< Raw status register value */
    uint64_t aca_addr;    /**< Raw address register value */
    uint64_t aca_ipid;    /**< Raw IPID register value */
    uint64_t aca_synd;    /**< Raw syndrome register value */
    uint32_t flags;       /**< Flags from descriptor */
    uint16_t hw_revision; /**< Hardware hw_revision number */
} aca_raw_data_t;

/**
 * @brief Main decode function that processes raw ACA error data
 * @param[in] raw_data Pointer to structure containing raw ACA error data
 * @return Decoded error information structure
 */
aca_error_info_t aca_decode(const aca_raw_data_t *raw_data);

#ifdef __cplusplus
}
#endif
#endif /* ACA_DECODE_H */
