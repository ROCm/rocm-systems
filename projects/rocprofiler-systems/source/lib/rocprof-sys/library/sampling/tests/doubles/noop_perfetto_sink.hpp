// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "sampling/data/overflow_sample.hpp"
#include "sampling/data/timer_sample.hpp"

#include <cstdint>
#include <vector>

namespace rocprofsys::sampling::test
{

struct noop_perfetto_sink
{
    void emit_timer(int64_t /*tid*/, void const* /*info*/,
                    std::vector<timer_sample> const& /*samples*/)
    {}

    void emit_overflow(int64_t /*tid*/, void const* /*info*/,
                       std::vector<overflow_sample> const& /*samples*/)
    {}
};

}  // namespace rocprofsys::sampling::test
