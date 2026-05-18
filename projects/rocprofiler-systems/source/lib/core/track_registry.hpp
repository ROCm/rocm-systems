// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace rocprofsys
{
// Owns the per-process Perfetto track-uuid registry and its mutex. Replaces the
// Meyer singletons previously at library/tracing.cpp. Constructor injection +
// thread-local active accessor are how the engine (slice B+) wires this in
// without changing emission-template callsites; slice A retains a process-
// global default for backward compat.
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

// Thread-local active registry pointer. Engine (slice B+) calls
// set_active_track_registry(&owned) at start; emission helpers in tracing.cpp
// pull the active pointer for every track-create call. Slice A: the
// process-global default is lazily seeded on first read by tracing.cpp.
void
set_active_track_registry(track_registry* registry) noexcept;

track_registry*
get_active_track_registry() noexcept;
}  // namespace rocprofsys
