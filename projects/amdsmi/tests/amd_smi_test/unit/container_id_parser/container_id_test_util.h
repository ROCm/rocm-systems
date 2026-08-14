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

// Test-only convenience wrappers around the production container-ID parser
// (amd::smi::ExtractContainerId). Every test links to the shipped
// implementation so the suite can never drift from production behavior.

#ifndef AMDSMI_TESTS_UNIT_CONTAINER_ID_PARSER_CONTAINER_ID_TEST_UTIL_H_
#define AMDSMI_TESTS_UNIT_CONTAINER_ID_PARSER_CONTAINER_ID_TEST_UTIL_H_

#include <cstddef>
#include <string>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_container_id_parser.h"

namespace amdsmi_test {

// Extract into a caller-supplied buffer; returns bytes written (excluding the
// NUL terminator). Thin pass-through to the production parser.
inline size_t ExtractIdInto(const std::string& line, const char* type_name,
                            char* out, size_t out_cap) {
  return amd::smi::ExtractContainerId(line, type_name, out, out_cap);
}

// Convenience value-return overload for equality-style assertions.
inline std::string ExtractIdString(const std::string& line,
                                   const char* type_name) {
  char buf[AMDSMI_MAX_STRING_LENGTH] = {0};
  ExtractIdInto(line, type_name, buf, sizeof(buf));
  return std::string(buf);
}

}  // namespace amdsmi_test

#endif  // AMDSMI_TESTS_UNIT_CONTAINER_ID_PARSER_CONTAINER_ID_TEST_UTIL_H_
