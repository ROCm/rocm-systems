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

// Shared test inputs. 64-character SHA-256 hex strings match Docker's
// `fullLen = 64` constant in client/pkg/stringid/stringid.go:
// https://github.com/moby/moby/blob/2200f277f9f576886e90ca75929a2bb892b9ef23/client/pkg/stringid/stringid.go#L14-L15

#ifndef AMDSMI_TESTS_UNIT_CONTAINER_ID_PARSER_FIXTURES_H_
#define AMDSMI_TESTS_UNIT_CONTAINER_ID_PARSER_FIXTURES_H_

namespace amdsmi_test {

// Valid 64-char lowercase hex SHA-256, as produced by Docker/containerd.
inline constexpr char kDocker64[] =
    "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";

inline constexpr char kDocker64Alt[] =
    "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";

}  // namespace amdsmi_test

#endif  // AMDSMI_TESTS_UNIT_CONTAINER_ID_PARSER_FIXTURES_H_
