/*
Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.

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
#ifndef INCLUDE_RDC_LIB_RDCDIAGNOSTICLIBINTERFACE_H_
#define INCLUDE_RDC_LIB_RDCDIAGNOSTICLIBINTERFACE_H_

// The telemetry interface for libraries, for example, AMD-SMI.
#include <rdc/rdc.h>

extern "C" {

// The library will implement below function

// Which test cases are supported in the library. test_cases[] holds MAX_TEST_CASES
// entries, of which the library writes the *test_case_count it reports.
rdc_status_t rdc_diag_test_cases_query(rdc_diag_test_cases_t test_cases[MAX_TEST_CASES],
                                       uint32_t* test_case_count);

// Run a specific test case.
//
// gpu_index[] holds the group's entity ids, of which the library reads
// min(gpu_count, RDC_MAX_NUM_DEVICES): the RDC_MAX_NUM_DEVICES in the signature bounds
// nothing, as the parameter is a pointer, and the dispatcher passes at most
// RDC_GROUP_MAX_ENTITIES ids. result->gpu_results[] holds RDC_MAX_NUM_DEVICES entries,
// and the library sets result->per_gpu_result_count to the rows it wrote and never
// above that: rdcd sends the count unchanged while it copies at most
// RDC_MAX_NUM_DEVICES rows, and rdci reads gpu_results[] by it.
//
// The call returns RDC_ST_OK for every outcome of the test, which it reports in
// result->status, because rdcd sends no result at all for any other status.
rdc_status_t rdc_diag_test_case_run(rdc_diag_test_cases_t test_case,
                                    uint32_t gpu_index[RDC_MAX_NUM_DEVICES], uint32_t gpu_count,
                                    const char* config, size_t config_size,
                                    rdc_diag_test_result_t* result, rdc_diag_callback_t* callback);

rdc_status_t rdc_diag_init(uint64_t flags);

rdc_status_t rdc_diag_destroy();
}

#endif  // INCLUDE_RDC_LIB_RDCDIAGNOSTICLIBINTERFACE_H_
