/* Copyright (c) 2022 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>


/* Placeholder for calling convention and import/export macros */
#if !defined(TEST_HSA_TOOL_EXPORT_CALL)
#define TEST_HSA_TOOL_EXPORT_CALL
#endif /* !defined (TEST_HSA_TOOL_EXPORT_CALL) */

#if !defined(TEST_HSA_TOOL_EXPORT_DECORATOR)
#if defined(__GNUC__)
#define TEST_HSA_TOOL_EXPORT_DECORATOR __attribute__((visibility("default")))
#elif defined(_MSC_VER)
#define TEST_HSA_TOOL_EXPORT_DECORATOR __declspec(dllexport)
#endif /* defined (_MSC_VER) */
#endif /* !defined (TEST_HSA_TOOL_EXPORT_DECORATOR) */

#if !defined(TEST_HSA_TOOL_IMPORT_DECORATOR)
#if defined(__GNUC__)
#define TEST_HSA_TOOL_IMPORT_DECORATOR
#elif defined(_MSC_VER)
#define TEST_HSA_TOOL_IMPORT_DECORATOR __declspec(dllimport)
#endif /* defined (_MSC_VER) */
#endif /* !defined (TEST_HSA_TOOL_IMPORT_DECORATOR) */
#define TEST_HSA_TOOL_EXPORT TEST_HSA_TOOL_EXPORT_DECORATOR TEST_HSA_TOOL_EXPORT_CALL
#define TEST_HSA_TOOL_IMPORT TEST_HSA_TOOL_IMPORT_DECORATOR TEST_HSA_TOOL_IMPORT_CALL
#if defined(TEST_HSA_TOOL_EXPORTS)
#define TEST_HSA_TOOL_API TEST_HSA_TOOL_EXPORT
#else /* !defined (TEST_HSA_TOOL_EXPORTS) */
#define TEST_HSA_TOOL_API TEST_HSA_TOOL_EXPORT
#endif /* !defined (TEST_HSA_TOOL_EXPORTS) */
typedef int (*rocprofiler_onload_callback)(
    void* table, uint64_t runtime_version, uint64_t failed_tool_count,
    const char* const* failed_tool_names);


TEST_HSA_TOOL_API void SetHSACallback(rocprofiler_onload_callback callback);