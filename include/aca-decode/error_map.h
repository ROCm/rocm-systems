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

#ifndef ERROR_MAP_H
#define ERROR_MAP_H

#include <stdint.h>

/**
 * @brief Structure representing an error mapping entry
 */
typedef struct
{
    uint32_t id;
    const char *error_category;
    const char *error_type;
    const char *method;
    const char *error_severity;
} error_map_entry_t;

/**
 * @brief Get error ID based on category, type and severity
 * @param[in] error_category Error category string
 * @param[in] error_type Error type string
 * @param[in] error_severity Error severity string
 * @return Error ID if found, -1 if not found
 */
int get_error_id(const char *error_category, const char *error_type, const char *error_severity);

#endif /* ERROR_MAP_H */
