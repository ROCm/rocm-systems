// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/**
 * @file stale-rocpd-decoy.cpp
 * @brief Stands in for an older librocprofiler-sdk-rocpd.so that is already in the global
 *        symbol scope, which is what a machine with a system ROCm earlier on
 *        LD_LIBRARY_PATH looks like. It is built into a prefix that has no sibling
 *        share/rocprofiler-sdk-rocpd directory, so a library that locates its schema by
 *        asking about the exported rocpd_sql_load_schema symbol lands here and finds no
 *        versions.yml.
 *
 *        It forwards to the real implementation, so the test still exercises the real
 *        library and fails only on the schema path that library resolves for itself.
 */

#include <rocprofiler-sdk-rocpd/rocpd.h>
#include <rocprofiler-sdk-rocpd/sql.h>

#include <dlfcn.h>

extern "C" ROCPD_PUBLIC_API rocpd_status_t
rocpd_sql_load_schema(rocpd_sql_engine_t                        engine,
                      rocpd_sql_schema_kind_t                   kind,
                      rocpd_sql_options_t                       options,
                      rocpd_version_triplet_t                   schema_version,
                      const rocpd_sql_schema_jinja_variables_t* variables,
                      rocpd_sql_load_schema_cb_t                callback,
                      const char**                              schema_path_hints,
                      uint64_t                                  num_schema_path_hints,
                      void*                                     user_data)
{
    using func_t = rocpd_status_t (*)(rocpd_sql_engine_t,
                                      rocpd_sql_schema_kind_t,
                                      rocpd_sql_options_t,
                                      rocpd_version_triplet_t,
                                      const rocpd_sql_schema_jinja_variables_t*,
                                      rocpd_sql_load_schema_cb_t,
                                      const char**,
                                      uint64_t,
                                      void*);

    auto* _next = reinterpret_cast<func_t>(dlsym(RTLD_NEXT, "rocpd_sql_load_schema"));

    if(_next == nullptr) return ROCPD_STATUS_ERROR;

    return _next(engine,
                 kind,
                 options,
                 schema_version,
                 variables,
                 callback,
                 schema_path_hints,
                 num_schema_path_hints,
                 user_data);
}
