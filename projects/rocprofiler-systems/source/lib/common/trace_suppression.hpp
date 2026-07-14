// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace rocprofsys
{
inline namespace common
{
/// Tracks, per-thread, whether the calling thread is currently inside code
/// whose pthread mutex/rwlock usage must not be traced (e.g. rocprofsys's
/// own internal locking). A depth counter rather than a bool so nested
/// enter()/exit() pairs on the same thread compose correctly.
class trace_suppression
{
public:
    static void enter() noexcept { ++s_depth; }
    static void exit() noexcept { --s_depth; }

    [[nodiscard]] static bool is_active() noexcept { return s_depth > 0; }

private:
    inline static thread_local std::uint32_t s_depth = 0;
};
}  // namespace common
}  // namespace rocprofsys
