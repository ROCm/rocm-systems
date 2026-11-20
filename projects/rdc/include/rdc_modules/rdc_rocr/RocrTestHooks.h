/*
Copyright (c) 2026 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef RDC_MODULES_RDC_ROCR_ROCRTESTHOOKS_H_
#define RDC_MODULES_RDC_ROCR_ROCRTESTHOOKS_H_

#include <stddef.h>
#include <stdint.h>

// Two functions the ROCr diagnostic module defines only when RDC_ROCR_TEST_HOOKS
// is 1, which only the unit tests define: the installed librdc_rocr.so has
// neither these functions nor the variables behind them, and its kernel search
// list and its wait deadline are constants. They are declared here, inside an
// extern "C" block as RdcDiagnosticLibInterface.h declares the module's entry
// points, so that a test can reach them with dlsym when it loads the module as a
// shared library. Only a test calls them, and only before a run: they are not
// safe to call while a check is running.
extern "C" {

// Replaces the module's list of directories to search for a kernel file with the
// given one, which the module searches in order. The strings are copied. A count
// of 0, or a null array, restores the module's own list.
void rdc_rocr_test_set_kernel_directories(const char* const* directories, size_t count);

// Replaces the deadline of each wait on a kernel or a memory copy, in
// milliseconds, and returns the deadline it replaces, so that a test can read the
// module's default without knowing it.
uint64_t rdc_rocr_test_set_wait_deadline_ms(uint64_t deadline_ms);
}

#endif  // RDC_MODULES_RDC_ROCR_ROCRTESTHOOKS_H_
