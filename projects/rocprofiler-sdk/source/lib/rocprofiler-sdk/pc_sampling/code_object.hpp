// MIT License
//
// Copyright (c) 2024 ROCm Developer Tools
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

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0

#    include <rocprofiler-sdk/callback_tracing.h>
#    include <rocprofiler-sdk/cxx/codeobj/segment.hpp>

#    include <hsa/hsa_api_trace.h>

#    include <mutex>
#    include <shared_mutex>

namespace rocprofiler
{
namespace pc_sampling
{
namespace code_object
{
void
initialize(HsaApiTable* table);

void
finalize();

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
        auto lock = std::unique_lock{mut};
        this->Super::insert(addr_range);
    }

    // Must acquire write lock
    bool remove(address_range_t addr_range)
    {
        auto lock = std::unique_lock{mut};
        return this->Super::remove(addr_range);
    }

    // Must acquire read lock
    address_range_t find_codeobj_in_range(uint64_t addr) const
    {
        // TODO: It would be good to have a way to cache search results
        // (caching could be done easily in the parser)
        auto lock = std::shared_lock{mut};
        auto it   = this->find(address_range_t{addr, 0, 0});
        // `addr` might originate from an unknown code object.
        if(it == this->end()) return address_range_t{0, 0, ROCPROFILER_CODE_OBJECT_ID_NONE};
        return *it;
    }

private:
    mutable std::shared_mutex mut = {};
};

CodeobjTableTranslatorSynchronized*
get_code_object_translator();

}  // namespace code_object
}  // namespace pc_sampling
}  // namespace rocprofiler

#endif
