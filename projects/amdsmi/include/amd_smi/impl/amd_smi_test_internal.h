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

#ifndef AMD_SMI_INCLUDE_AMD_SMI_TEST_INTERNAL_H_
#define AMD_SMI_INCLUDE_AMD_SMI_TEST_INTERNAL_H_

#include <cstdint>

#include "amd_smi/amdsmi.h"

// Reserved test-only init flag. Passed to amdsmi_init() / rsmi_init() to
// switch GPU device mutexes from blocking to non-blocking (trylock) mode,
// making AMDSMI_STATUS_BUSY / RSMI_STATUS_BUSY observable by tests.
//
// MUST NOT overlap with any public AMDSMI_INIT_* flag defined in amdsmi.h.
// Current public flags occupy bits [0:3]; this flag uses bit 59 (0x0800_0000_0000_0000).
inline constexpr uint64_t AMD_SMI_INIT_FLAG_RESRV_TEST1 = 0x0800000000000000ULL;

// Internal test-only wrapper around rsmi_test_sleep. Acquires the device mutex
// for |seconds| seconds and returns an amdsmi_status_t so tests do not need to
// extern-declare the rsmi_status_t function directly.
amdsmi_status_t amdsmi_test_sleep(amdsmi_processor_handle processor_handle, uint32_t seconds);

#endif  // AMD_SMI_INCLUDE_AMD_SMI_TEST_INTERNAL_H_
