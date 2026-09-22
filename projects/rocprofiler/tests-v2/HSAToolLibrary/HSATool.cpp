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

#include "HSATool.h"
extern "C" {


/*
 @brief The HSA_AMD_TOOL_PRIORITY variable must be a constant value type
   initialized by the loader itself, not by code during _init. 'extern const'
   seems do that although that is not a guarantee.
*/

TEST_HSA_TOOL_API extern const uint32_t HSA_AMD_TOOL_PRIORITY = 50;
static rocprofiler_onload_callback rocprofiler_onload_callback_call = nullptr;

/*

 @brief Callback function called upon loading the HSA.
  The function updates the core api table function pointers to point to the
  interceptor functions in this file.

*/

TEST_HSA_TOOL_API bool OnLoad(void* table, uint64_t runtime_version, uint64_t failed_tool_count,
 const char* const* failed_tool_names) {
  rocprofiler_onload_callback_call(table, runtime_version, failed_tool_count, failed_tool_names);
  return true;
}

/*
@brief Callback function upon unloading the HSA.
*/

TEST_HSA_TOOL_API void OnUnload() { printf("\n\nTool is getting unloaded\n\n"); }

}  // extern "C"

TEST_HSA_TOOL_API void SetHSACallback(rocprofiler_onload_callback callback) {
  rocprofiler_onload_callback_call = callback;
}