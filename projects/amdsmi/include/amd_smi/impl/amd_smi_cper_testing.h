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

#pragma once
#include <sys/types.h>

// Library-local test seam: swaps the read() used by the CPER reader so unit
// tests can drive zero/partial/error returns the way an empty debugfs ring does.
// Not amdsmi_-prefixed, so the linker version script keeps it out of
// libamd_smi.so (no public ABI); tests reach it through the static archive.
// Passing nullptr restores POSIX read(). Not thread-safe: only the
// single-threaded tests call it; production never does.
//
// Shared by the definition (src/amd_smi/amd_smi_cper.cc) and the tests
// (tests/amd_smi_test/functional/cper_read.cc) so the signature stays in sync.
void cper_set_read_fn_for_testing(ssize_t (*read_fn)(int, void*, size_t));
