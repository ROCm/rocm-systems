// MIT License
//
// Copyright (c) 2024-2025 ROCm Developer Tools
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

#include "lib/common/static_object.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/defines.hpp"

#include <hsa/hsa_api_trace.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <mutex>
#include <rocprofiler-sdk/cxx/codeobj/segment.hpp>
#include <shared_mutex>

namespace rocprofiler
{
namespace pc_sampling
{
namespace code_object
{
using address_range_t        = rocprofiler::sdk::codeobj::segment::address_range_t;
using CodeobjTableTranslator = rocprofiler::sdk::codeobj::segment::CodeobjTableTranslator;

class CodeobjTableTranslatorSynchronized : public CodeobjTableTranslator
{
    using Super            = CodeobjTableTranslator;
    using code_object_id_t = uint64_t;

public:
    // Must acquire write lock
    void insert(address_range_t addr_range)
    {
        auto lock = std::unique_lock{backlog_mut};
        insert_backlog.emplace_back(addr_range);

        if(auto try_lock = std::unique_lock{query_mut, std::try_to_lock}) clear_insert_log();
    }

    // Must acquire write lock
    void remove(address_range_t addr_range)
    {
        auto lock = std::unique_lock{backlog_mut};
        remove_backlog.emplace_back(addr_range);

        if(auto try_lock = std::unique_lock{query_mut, std::try_to_lock}) clear_remove_log();
    }

    void clear_backlog()
    {
        auto backlog_lock = std::unique_lock{backlog_mut};

        if(!remove_backlog.empty() || !insert_backlog.empty())
        {
            auto query_lock = std::unique_lock{query_mut};

            clear_remove_log();
            clear_insert_log();
        }
    }

    std::shared_lock<std::shared_mutex> acquire_query_lock() { return std::shared_lock{query_mut}; }

    // Must acquire read lock
    address_range_t find_codeobj_in_range(uint64_t addr) const
    {
        auto it = this->find(address_range_t{addr, 0, 0});
        // `addr` might originate from an unknown code object.
        if(it == this->end()) return address_range_t{0, 0, ROCPROFILER_CODE_OBJECT_ID_NONE};
        return *it;
    }

    static CodeobjTableTranslatorSynchronized* Get()
    {
        static auto*& _v = common::static_object<CodeobjTableTranslatorSynchronized>::construct();
        return _v;
    }

private:
    void clear_insert_log()
    {
        for(const auto& addr_range : insert_backlog)
            this->Super::insert(addr_range);
        insert_backlog.clear();
    }

    void clear_remove_log()
    {
        for(const auto& addr_range : remove_backlog)
            this->Super::remove(addr_range);
        remove_backlog.clear();
    }

    std::mutex                   backlog_mut{};
    std::shared_mutex            query_mut{};
    std::vector<address_range_t> insert_backlog{};
    std::vector<address_range_t> remove_backlog{};
};

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0

void
initialize(HsaApiTable* table);

void
finalize();

#endif

}  // namespace code_object
}  // namespace pc_sampling
}  // namespace rocprofiler
