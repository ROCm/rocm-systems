// MIT License
//
// Copyright (c) 2023 ROCm Developer Tools
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <rocprofiler/registration.h>
#include "lib/common/defines.hpp"

#include <cstdint>
#include <string>
#include <vector>

extern "C" {
struct HsaApiTable;

using on_load_t = bool (*)(HsaApiTable*, uint64_t, uint64_t, const char* const*);

bool
OnLoad(HsaApiTable*       table,
       uint64_t           runtime_version,
       uint64_t           failed_tool_count,
       const char* const* failed_tool_names) ROCPROFILER_PUBLIC_API;

// this is the "hidden" function that rocprofiler-register invokes to pass
// the API tables to rocprofiler
int
rocprofiler_set_api_table(const char* name,
                          uint64_t    lib_version,
                          uint64_t    lib_instance,
                          void**      tables,
                          uint64_t    num_tables) ROCPROFILER_PUBLIC_API;
}

namespace rocprofiler
{
namespace registration
{
// initialize the clients
void
initialize();

// finalize the clients
void
finalize();

// invoke all rocprofiler_configure symbols
bool
invoke_client_configures();

// invoke initialize functions returned from rocprofiler_configure
bool
invoke_client_initializers();

// invoke finalize functions returned from rocprofiler_configure
bool
invoke_client_finalizers();

// explicitly invoke the initialize function of a specific client
bool invoke_client_initializer(rocprofiler_client_id_t);

// explicitly invoke the finalize function of a specific client
void invoke_client_finalizer(rocprofiler_client_id_t);

int
get_init_status();

int
get_fini_status();

void
set_init_status(int);

void
set_fini_status(int);
}  // namespace registration
}  // namespace rocprofiler
