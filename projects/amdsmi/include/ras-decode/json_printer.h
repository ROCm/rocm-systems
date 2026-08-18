// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef JSON_PRINTER_H
#define JSON_PRINTER_H

#include "json_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Print a JSON value to stdout in formatted form
 * @param value JSON value to print
 */
void print_json_value(JsonValue* value);

#ifdef __cplusplus
}
#endif

#endif /* JSON_PRINTER_H */
