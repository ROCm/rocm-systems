// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "core/common.hpp"
#include "core/components/fwd.hpp"
#include "core/containers/static_vector.hpp"
#include "core/defines.hpp"
#include "core/timemory.hpp"
#include "library/thread_data.hpp"

#include <timemory/components/base/declaration.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/unwind/cache.hpp>
#include <timemory/unwind/processed_entry.hpp>
#include <timemory/unwind/stack.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <set>
#include <vector>

namespace rocprofsys
{
namespace component
{
struct callchain : comp::empty_base
{
    static constexpr size_t stack_depth = ROCPROFSYS_MAX_UNWIND_DEPTH;

    struct record
    {
        uint64_t                                         timestamp = 0;
        container::static_vector<uintptr_t, stack_depth> data      = {};

        bool operator<(const record& rhs) const;
    };

    using cache_type     = tim::unwind::cache;
    using entry_type     = tim::unwind::processed_entry;
    using value_type     = void;
    using data_t         = container::static_vector<record, 64>;
    using entry_vec_t    = std::vector<entry_type>;
    using ts_entry_vec_t = std::pair<uint64_t, entry_vec_t>;

    static std::string label();
    static std::string description();

    callchain()                     = default;
    ~callchain()                    = default;
    callchain(const callchain&)     = default;
    callchain(callchain&&) noexcept = default;

    callchain& operator=(const callchain&)     = default;
    callchain& operator=(callchain&&) noexcept = default;

    static std::vector<ts_entry_vec_t> filter_and_patch(
        const std::vector<ts_entry_vec_t>&);

    static void start();
    static void stop();

    void                        sample(int = -1);
    bool                        empty() const;
    size_t                      size() const;
    std::vector<ts_entry_vec_t> get() const;
    data_t                      get_data() const { return m_data; }

private:
    data_t m_data = {};
};
}  // namespace component
}  // namespace rocprofsys
