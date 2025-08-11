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
#include "core/defines.hpp"
#include "core/timemory.hpp"
#include "library/causal/data.hpp"
#include "library/causal/fwd.hpp"
#include "library/perf.hpp"

#include <timemory/components/base.hpp>
#include <timemory/macros/language.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/tpls/cereal/cereal/cereal.hpp>
#include <timemory/units.hpp>
#include <timemory/utility/unwind.hpp>

#include <chrono>
#include <cstdint>

namespace rocprofsys
{
namespace causal
{
namespace component
{
struct overflow : comp::empty_base
{
    static constexpr auto alt_stack_size = perf::perf_event::max_batch_size;

    using value_type  = void;
    using callchain_t = container::static_vector<uintptr_t, unwind_depth>;
    using alt_stack_t = container::static_vector<callchain_t, alt_stack_size>;

    static std::string label() { return "causal::overflow"; }
    static void        global_init();

    void sample(int = -1);

    auto        get_selected() const { return m_selected; }
    auto        get_index() const { return m_index; }
    const auto& get_stack() const { return m_stack; }

private:
    int32_t     m_selected = 0;
    uint32_t    m_index    = 0;
    alt_stack_t m_stack    = {};
};

struct backtrace : comp::empty_base
{
    using value_type  = void;
    using callchain_t = container::static_vector<uint64_t, unwind_depth>;

    static std::string label() { return "causal::backtrace"; }
    static void        global_init();

    backtrace()                     = default;
    ~backtrace()                    = default;
    backtrace(const backtrace&)     = default;
    backtrace(backtrace&&) noexcept = default;

    backtrace& operator=(const backtrace&)     = default;
    backtrace& operator=(backtrace&&) noexcept = default;

    void sample(int = -1);

    auto get_selected() const { return m_selected; }
    auto get_index() const { return m_index; }
    auto get_stack() const { return m_stack; }

    template <typename Tp = uint64_t>
    static Tp get_period(uint64_t _units = units::nsec);

private:
    bool                  m_selected = false;
    uint32_t              m_index    = 0;
    causal::unwind_addr_t m_stack    = {};
};
}  // namespace component
}  // namespace causal
}  // namespace rocprofsys
