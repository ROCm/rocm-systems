// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "core/perfetto/engine.hpp"
#include "core/track_registry.hpp"

#include <thread>

// perfetto_processor_t::prepare_for_processing performs a two-step TLS
// handshake on each parser thread —
//   set_active_track_registry(m_tracks);
//   core::set_emitting_pid(m_process_id);
// before the cached interceptor sees any TRACE_EVENT_*. This file verifies
// the contract that handshake relies on: two threads each issuing the pair
// see only their own values, and the caller thread is unaffected.
//
// Engine-byte isolation (drain bytes don't cross pids) is covered by
// perfetto_engine_cached.drain_two_sources_no_cross_bleed. End-to-end
// integration is covered by tests/pytest/test_cached_perfetto_isolation.py.

namespace
{
struct prepared_state
{
    rocprofsys::track_registry* registry{ nullptr };
    int                         emitting_pid{ -1 };
};

prepared_state
prepare_on_this_thread(rocprofsys::track_registry* tracks, int pid)
{
    rocprofsys::set_active_track_registry(tracks);
    rocprofsys::core::set_emitting_pid(pid);
    return { rocprofsys::get_active_track_registry(),
             rocprofsys::core::get_emitting_pid() };
}
}  // namespace

TEST(emission_handshake_tls_contract,
     prepare_handshake_on_two_threads_sees_no_cross_contamination)
{
    rocprofsys::track_registry tracks_a;
    rocprofsys::track_registry tracks_b;

    prepared_state observed_a{};
    prepared_state observed_b{};

    std::thread thread_a{ [&] { observed_a = prepare_on_this_thread(&tracks_a, 101); } };
    std::thread thread_b{ [&] { observed_b = prepare_on_this_thread(&tracks_b, 202); } };
    thread_a.join();
    thread_b.join();

    EXPECT_EQ(observed_a.registry, &tracks_a);
    EXPECT_EQ(observed_a.emitting_pid, 101);
    EXPECT_EQ(observed_b.registry, &tracks_b);
    EXPECT_EQ(observed_b.emitting_pid, 202);
}

TEST(emission_handshake_tls_contract, prepare_on_worker_does_not_leak_to_caller)
{
    rocprofsys::set_active_track_registry(nullptr);
    rocprofsys::core::set_emitting_pid(-1);

    rocprofsys::track_registry tracks_worker;

    std::thread worker{ [&] { prepare_on_this_thread(&tracks_worker, 999); } };
    worker.join();

    EXPECT_EQ(rocprofsys::get_active_track_registry(), nullptr);
    EXPECT_EQ(rocprofsys::core::get_emitting_pid(), -1);
}

TEST(emission_handshake_tls_contract, the_two_tls_systems_are_independent)
{
    rocprofsys::set_active_track_registry(nullptr);
    rocprofsys::core::set_emitting_pid(-1);

    rocprofsys::track_registry tracks;

    rocprofsys::set_active_track_registry(&tracks);
    EXPECT_EQ(rocprofsys::get_active_track_registry(), &tracks);
    EXPECT_EQ(rocprofsys::core::get_emitting_pid(), -1);

    rocprofsys::core::set_emitting_pid(42);
    EXPECT_EQ(rocprofsys::get_active_track_registry(), &tracks);
    EXPECT_EQ(rocprofsys::core::get_emitting_pid(), 42);

    rocprofsys::set_active_track_registry(nullptr);
    rocprofsys::core::set_emitting_pid(-1);
}
