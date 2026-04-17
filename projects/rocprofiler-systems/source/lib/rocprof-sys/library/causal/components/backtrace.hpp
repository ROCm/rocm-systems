// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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
