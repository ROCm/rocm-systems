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
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "prestore.hpp"
#include "queue_registration.hpp"
#include "code_object_registration.hpp"

#include "lib/common/logging.hpp"

void
init_logging()
{
    rocprofiler::common::init_logging("ROCPROFILER");
}

// ensure that logging is always initialized when library is loaded
bool init_logging_at_load = (init_logging(), true);

extern "C" {
    
int
rocprofiler_prestore_set_api_table(
    const char* name,
    uint64_t    lib_version,
    uint64_t    lib_instance,
    void**      tables,
    uint64_t    num_tables) 
{
    ROCP_TRACE << "rocprofiler_prestore_set_api_table called for api " << name;
    (void)lib_version; // unused
    (void)lib_instance; // unused

    if (std::string_view{name} != "hsa")
    {
        ROCP_ERROR << "rocprofiler_prestore_set_api_table was called with a table other than HSA";
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    }

    ROCP_ERROR_IF(num_tables > 1)
        << "rocprofiler expected HSA library to pass 1 API table, not " << num_tables;

    auto* hsa_api_table = static_cast<HsaApiTable*>(*tables);

    // Initialize all registration services in prestore
    rocprofiler::prestore::queue_registration_init(hsa_api_table);
    rocprofiler::prestore::code_object_registration_init(hsa_api_table);

    return 0;
}

int rocprofiler_prestore_get_version()
{
    constexpr int ROCPROFILER_PRESTORE_VERSION = 1;
    return ROCPROFILER_PRESTORE_VERSION;
}

}
