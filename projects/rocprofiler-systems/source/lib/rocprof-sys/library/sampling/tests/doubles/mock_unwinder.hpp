// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "sampling/data/stack_frame.hpp"

#include <cstdint>
#include <queue>
#include <vector>

namespace rocprofsys::sampling::test
{

struct mock_unwinder
{
    [[nodiscard]] std::vector<stack_frame> unwind(void const* /*ctx*/) noexcept
    {
        ++m_unwind_call_count;
        if(!m_scripted_frames.empty())
        {
            auto frames = std::move(m_scripted_frames.front());
            m_scripted_frames.pop();
            return frames;
        }
        return {};
    }

    [[nodiscard]] static bool valid_pc(uintptr_t /*pc*/) noexcept { return true; }

    void enqueue_frames(std::vector<stack_frame> frames)
    {
        m_scripted_frames.push(std::move(frames));
    }

    [[nodiscard]] int unwind_call_count() const noexcept { return m_unwind_call_count; }

private:
    std::queue<std::vector<stack_frame>> m_scripted_frames;
    int                                  m_unwind_call_count = 0;
};

}  // namespace rocprofsys::sampling::test
