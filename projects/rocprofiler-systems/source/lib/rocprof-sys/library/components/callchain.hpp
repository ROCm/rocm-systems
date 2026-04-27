// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
#include "core/common.hpp"
#include "core/components/fwd.hpp"
#include "core/containers/static_vector.hpp"
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
