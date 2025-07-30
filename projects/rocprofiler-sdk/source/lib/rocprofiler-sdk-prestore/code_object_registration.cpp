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

#include "details/code_object_registration.hpp"
#include "code_object_registration.hpp"

#include "lib/common/static_object.hpp"

namespace rocprofiler {
namespace prestore {

hsa_status_t
executable_freeze(hsa_executable_t executable, const char* options)
{
    auto cor = CHECK_NOTNULL(get_code_object_registration());
    auto status = cor->get_hsa_executable_freeze_fn()(executable, options);
    if (status)
    {
        return status;
    }
    cor->add_code_object(executable);
    return HSA_STATUS_SUCCESS;
}

hsa_status_t
executable_destroy(hsa_executable_t executable)
{
    auto cor = CHECK_NOTNULL(get_code_object_registration());
    auto status = cor->get_hsa_executable_destroy_fn()(executable);
    if (status)
    {
        return status;
    }
    cor->remove_code_object(executable);
    return HSA_STATUS_SUCCESS;
}

void CodeObjectRegistration::init(CoreApiTable& core_table, AmdExtTable& ext_table)
{
    ROCP_TRACE << "Initializing CodeObjectRegistration";
    (void)ext_table; // unused

    m_hsa_executable_freeze_fn = core_table.hsa_executable_freeze_fn;
    core_table.hsa_executable_freeze_fn = prestore::executable_freeze;
    m_hsa_executable_destroy_fn = core_table.hsa_executable_destroy_fn;
    core_table.hsa_executable_destroy_fn = prestore::executable_destroy;
}

void CodeObjectRegistration::add_code_object(code_object_prestore_t code_object)
{
    ROCP_TRACE << "adding code_object " << code_object.handle;
    m_code_objects.emplace_back(code_object);
}

void CodeObjectRegistration::remove_code_object(code_object_prestore_t code_object)
{
    ROCP_TRACE << "removing code_object " << code_object.handle;
    auto pred = [&](const code_object_prestore_t& a)
    {
        return a.handle == code_object.handle;
    };
    auto itr = std::find_if(m_code_objects.begin(), m_code_objects.end(), pred);
    if (itr == m_code_objects.end())
    {
        ROCP_INFO << "remove_code_object could not find " << code_object.handle;
        return;
    }
    m_code_objects.erase(itr);
}

CodeObjectRegistration*
get_code_object_registration()
{
    static auto*& registration = common::static_object<CodeObjectRegistration>::construct();
    return registration;
}

void code_object_registration_init(HsaApiTable* table)
{
    CHECK_NOTNULL(get_code_object_registration())->init(*table->core_, *table->amd_ext_);
}

}}

extern "C" {

int rocprofiler_prestore_export_all_code_objects(
    hsa_executable_t* executables,
    uint64_t* num_executables)
{
    if (!executables && num_executables)
    {
        *num_executables = CHECK_NOTNULL(rocprofiler::prestore::get_code_object_registration())->get_all_code_objects().size();
        return ROCPROFILER_STATUS_SUCCESS;
    }

    CHECK_NOTNULL(executables);
    CHECK_NOTNULL(num_executables);
    auto cos = CHECK_NOTNULL(rocprofiler::prestore::get_code_object_registration())->get_all_code_objects();
    auto cos_out = executables;
    if (*num_executables < cos.size())
    {
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    }
    std::copy(cos.begin(), cos.end(), cos_out);

    return ROCPROFILER_STATUS_SUCCESS;
}
}