// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Lightweight value type carrying the thread-info fields that sampling
// policies need. Decouples sink/impl code from library/thread_info.hpp
// (which pulls the full main-library include chain).

#include <cstddef>
#include <cstdint>

namespace rocprofsys::sampling
{

struct thread_info_data
{
    std::size_t system_value  = 0;
    std::size_t sequent_value = 0;
    uint64_t    start_ns      = 0;
    uint64_t    stop_ns       = 0;

    [[nodiscard]] bool is_valid_lifetime(uint64_t beg, uint64_t end) const noexcept
    {
        return beg >= start_ns && end <= stop_ns;
    }
};

}  // namespace rocprofsys::sampling
