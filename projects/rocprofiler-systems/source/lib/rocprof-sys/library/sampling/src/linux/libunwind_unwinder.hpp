// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// In-handler libunwind implementation of the unwinder policy concept.
// Captures raw instruction pointers into a thread-local array (async-signal-safe).
// DWARF symbol resolution is deferred to post_process — NOT called from here.
//
// NFR-PORT-3: lives under src/linux/ — not under include/sampling/.

#include "sampling/data/stack_frame.hpp"

#include <cstdint>
#include <libunwind.h>
#include <ucontext.h>
#include <vector>

namespace rocprofsys::sampling
{

class libunwind_unwinder
{
public:
    static constexpr int max_depth = 64;

    // Unwind from the supplied signal context and return a vector of stack_frame
    // with only the address field populated. Symbol resolution is deferred.
    // async-signal-safe: uses only libunwind local-unwind, no malloc, no locks.
    [[nodiscard]] std::vector<stack_frame> unwind(void const* ctx) noexcept
    {
        // Thread-local PC buffer — avoids heap allocation in handler.
        static thread_local uintptr_t pc_buf[max_depth];

        int           depth  = 0;
        unw_cursor_t  cursor = {};
        unw_context_t uctx   = {};

        if(ctx)
        {
            // Re-use the signal ucontext_t directly.
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
            auto* uc = static_cast<ucontext_t const*>(ctx);
            // Copy to mutable local so unw_init_local can modify it.
            uctx = *reinterpret_cast<unw_context_t const*>(uc);
        }
        else
        {
            unw_getcontext(&uctx);
        }

        if(unw_init_local(&cursor, &uctx) != 0) return {};

        while(depth < max_depth)
        {
            unw_word_t ip = 0;
            if(unw_get_reg(&cursor, UNW_REG_IP, &ip) != 0) break;
            if(ip == 0) break;
            pc_buf[depth++] = static_cast<uintptr_t>(ip);
            if(unw_step(&cursor) <= 0) break;
        }

        std::vector<stack_frame> frames;
        frames.reserve(static_cast<size_t>(depth));
        for(int i = 0; i < depth; ++i)
        {
            stack_frame sf;
            sf.address = pc_buf[i];
            frames.push_back(std::move(sf));
        }
        return frames;
    }

    [[nodiscard]] static bool valid_pc(uintptr_t pc) noexcept { return pc != 0; }
};

}  // namespace rocprofsys::sampling
