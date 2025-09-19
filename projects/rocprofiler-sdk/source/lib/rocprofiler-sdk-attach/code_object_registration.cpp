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

#include "code_object_registration.h"
#include "code_object_registration.hpp"
#include "table.hpp"

#include <hsa/hsa.h>

#include "lib/common/static_object.hpp"

#include <mutex>

namespace
{
using hsa_executable_freeze_t  = decltype(CoreApiTable::hsa_executable_freeze_fn);
using hsa_executable_destroy_t = decltype(CoreApiTable::hsa_executable_destroy_fn);
using code_object_collection_t = std::vector<hsa_executable_t>;

struct code_object_registration_t
{
    // gates access to code_objects collection
    std::mutex               code_objects_mutex;
    code_object_collection_t code_objects;
    hsa_executable_freeze_t  hsa_executable_freeze_fn  = nullptr;
    hsa_executable_destroy_t hsa_executable_destroy_fn = nullptr;
};

code_object_registration_t*
get_code_object_registration()
{
    static auto*& registration =
        rocprofiler::common::static_object<code_object_registration_t>::construct();
    return registration;
}

hsa_status_t
executable_freeze(hsa_executable_t executable, const char* options)
{
    auto* registration = CHECK_NOTNULL(get_code_object_registration());
    auto  status       = registration->hsa_executable_freeze_fn(executable, options);

    if(status != HSA_STATUS_SUCCESS) return status;

    ROCP_TRACE << "adding code_object " << executable.handle;
    {
        std::lock_guard lg(registration->code_objects_mutex);
        registration->code_objects.emplace_back(executable);
    }
    auto* attach_table = rocprofiler::attach::get_dispatch_table();
    if(attach_table->rocprofiler_attach_notify_new_code_object)
    {
        attach_table->rocprofiler_attach_notify_new_code_object(executable, nullptr);
    }
    return HSA_STATUS_SUCCESS;
}

hsa_status_t
executable_destroy(hsa_executable_t executable)
{
    auto* registration = CHECK_NOTNULL(get_code_object_registration());
    ROCP_TRACE << "removing code_object " << executable.handle;
    {
        std::lock_guard lg(registration->code_objects_mutex);
        auto pred = [&](const hsa_executable_t& a) { return a.handle == executable.handle; };
        auto itr  = std::find_if(
            registration->code_objects.begin(), registration->code_objects.end(), pred);
        if(itr == registration->code_objects.end())
        {
            ROCP_WARNING << "remove code_object could not find " << executable.handle;
        }
        registration->code_objects.erase(itr);
    }

    return registration->hsa_executable_destroy_fn(executable);
}

int
iterate_all_code_objects(rocprof_attach_code_object_iterator_t func, void* data)
{
    auto* registration = CHECK_NOTNULL(get_code_object_registration());

    for(const auto& code_object : registration->code_objects)
    {
        func(code_object, data);
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

}  // namespace

namespace rocprofiler
{
namespace attach
{
void
code_object_registration_init(
    HsaApiTable* table)  // CoreApiTable& core_table, AmdExtTable& ext_table)
{
    ROCP_TRACE << "Initializing Code Object Registration";
    auto*         registration = CHECK_NOTNULL(get_code_object_registration());
    CoreApiTable& core_table   = *table->core_;

    // route executable freeze and destroy to us, but also save the original entrypoint so we can
    // call it
    registration->hsa_executable_freeze_fn  = core_table.hsa_executable_freeze_fn;
    core_table.hsa_executable_freeze_fn     = executable_freeze;
    registration->hsa_executable_destroy_fn = core_table.hsa_executable_destroy_fn;
    core_table.hsa_executable_destroy_fn    = executable_destroy;
}

}  // namespace attach
}  // namespace rocprofiler

ROCPROFILER_EXTERN_C_INIT

int
rocprofiler_attach_iterate_all_code_objects(rocprof_attach_code_object_iterator_t func, void* data)
{
    return iterate_all_code_objects(func, data);
}

ROCPROFILER_EXTERN_C_FINI
