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

#pragma once

#include "lib/rocprofiler-sdk-prestore/code_object_registration.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"

#include <hsa/hsa.h>

#include <cstdint>

namespace rocprofiler
{
namespace prestore
{

class CodeObjectRegistration
{
    using hsa_executable_freeze_t  = decltype(CoreApiTable::hsa_executable_freeze_fn);
    using hsa_executable_destroy_t = decltype(CoreApiTable::hsa_executable_destroy_fn);
    using code_object_prestore_t   = hsa_executable_t;
    using code_object_collection_t = std::vector<code_object_prestore_t>;

public:
    CodeObjectRegistration()  = default;
    ~CodeObjectRegistration() = default;
    // Initializes the CodeObjectRegistration. This must be delayed until
    // HSA has been inited.
    void init(CoreApiTable& core_table, AmdExtTable& ext_table);

    void add_code_object(code_object_prestore_t);
    void remove_code_object(code_object_prestore_t);

    const code_object_collection_t& get_all_code_objects() { return m_code_objects; };
    hsa_executable_freeze_t         get_hsa_executable_freeze_fn() const
    {
        return m_hsa_executable_freeze_fn;
    }
    hsa_executable_destroy_t get_hsa_executable_destroy_fn() const
    {
        return m_hsa_executable_destroy_fn;
    }

private:
    code_object_collection_t m_code_objects;

    hsa_executable_freeze_t  m_hsa_executable_freeze_fn;
    hsa_executable_destroy_t m_hsa_executable_destroy_fn;
};

CodeObjectRegistration*
get_code_object_registration();

}  // namespace prestore
}  // namespace rocprofiler
