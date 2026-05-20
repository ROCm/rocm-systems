// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace rocprofsys
{
// Owns the per-process Perfetto track-uuid registry and its mutex. Each
// engine instance owns one; the active registry is published via the
// thread-local active_track_registry pointer so emission helpers in
// tracing.cpp can reach it without callsite changes.
class track_registry
{
public:
    using hash_value_t = std::uint64_t;

    track_registry()  = default;
    ~track_registry() = default;

    track_registry(const track_registry&)            = delete;
    track_registry& operator=(const track_registry&) = delete;
    track_registry(track_registry&&)                 = delete;
    track_registry& operator=(track_registry&&)      = delete;

    std::mutex& mutex() noexcept { return m_mutex; }

    std::unordered_map<hash_value_t, std::string>& map() noexcept { return m_map; }

private:
    std::mutex                                    m_mutex;
    std::unordered_map<hash_value_t, std::string> m_map;
};

// Thread-local active registry pointer. The engine calls
// set_active_track_registry(&owned) on each parser thread at start;
// emission helpers in tracing.cpp pull the active pointer for every
// track-create call. When null, tracing.cpp's helpers fall back to a
// process-global default registry.
void
set_active_track_registry(track_registry* registry) noexcept;

track_registry*
get_active_track_registry() noexcept;
}  // namespace rocprofsys
