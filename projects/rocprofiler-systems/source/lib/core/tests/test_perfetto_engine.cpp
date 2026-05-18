// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "core/perfetto_engine.hpp"

#include <thread>

namespace
{
rocprofsys::core::engine_config
make_test_config()
{
    rocprofsys::core::engine_config cfg{};
    cfg.buffer_size_kb     = 1024;
    cfg.shmem_size_hint_kb = 64;
    cfg.flush_period_ms    = 0;
    cfg.fill_policy        = rocprofsys::core::engine_config::fill_policy_t::discard;
    cfg.backend            = rocprofsys::core::engine_config::backend_t::inprocess;
    return cfg;
}
}  // namespace

TEST(perfetto_engine, construct_from_config_literal_no_config_access)
{
    // RQ3: engine instantiable in unit tests with no global pollution.
    // Construction must not touch rocprofsys::config::*; passing a literal
    // engine_config exercises this.
    rocprofsys::core::engine_config cfg = make_test_config();
    rocprofsys::core::perfetto_engine engine{ cfg };

    EXPECT_FALSE(engine.is_running());
}

TEST(perfetto_engine, two_instances_no_shared_running_state)
{
    // RQ3: a second engine constructed after the first sees pristine state.
    rocprofsys::core::perfetto_engine first{ make_test_config() };
    rocprofsys::core::perfetto_engine second{ make_test_config() };

    EXPECT_FALSE(first.is_running());
    EXPECT_FALSE(second.is_running());
}

TEST(perfetto_engine, stop_without_start_is_noop)
{
    // RF6: engine.stop() invoked before start() returns without error.
    rocprofsys::core::perfetto_engine engine{ make_test_config() };

    EXPECT_NO_THROW(engine.stop());
    EXPECT_FALSE(engine.is_running());
}

TEST(perfetto_engine, read_trace_without_session_returns_empty)
{
    // Calling read_trace on a pid with no active session must not crash and
    // returns an empty vector (defensive contract for the perfetto.cpp shim
    // when post_process is invoked twice).
    rocprofsys::core::perfetto_engine engine{ make_test_config() };

    auto bytes = engine.read_trace(static_cast<pid_t>(99999));
    EXPECT_TRUE(bytes.empty());
}

TEST(perfetto_engine, release_session_idempotent)
{
    // release_session on an unknown pid is a no-op; calling it twice on
    // the same pid is also fine. Mirrors fork_gotcha .release()/.reset()
    // call patterns from the child process.
    rocprofsys::core::perfetto_engine engine{ make_test_config() };

    EXPECT_NO_THROW(engine.release_session(static_cast<pid_t>(11111)));
    EXPECT_NO_THROW(engine.release_session(static_cast<pid_t>(11111)));
}

TEST(perfetto_engine, set_emitting_pid_round_trip)
{
    // The thread_local pid tag set by static set_emitting_pid is read back
    // by static get_emitting_pid on the same thread.
    rocprofsys::core::perfetto_engine::set_emitting_pid(4242);
    EXPECT_EQ(rocprofsys::core::perfetto_engine::get_emitting_pid(), 4242);

    // Reset for subsequent test cases on this thread.
    rocprofsys::core::perfetto_engine::set_emitting_pid(0);
}

TEST(perfetto_engine, emitting_pid_is_thread_local)
{
    // D4: emitting pid is per-thread. Setting on one thread must not leak
    // to other threads.
    rocprofsys::core::perfetto_engine::set_emitting_pid(7777);
    EXPECT_EQ(rocprofsys::core::perfetto_engine::get_emitting_pid(), 7777);

    int observed_on_other_thread = -1;
    std::thread other{ [&observed_on_other_thread]() {
        observed_on_other_thread =
            rocprofsys::core::perfetto_engine::get_emitting_pid();
    } };
    other.join();

    EXPECT_EQ(observed_on_other_thread, 0)
        << "emitting pid must default to 0 on a fresh thread";
    EXPECT_EQ(rocprofsys::core::perfetto_engine::get_emitting_pid(), 7777)
        << "main thread's tag must be unchanged by other thread's read";

    rocprofsys::core::perfetto_engine::set_emitting_pid(0);
}

TEST(perfetto_engine, session_ref_returns_empty_slot_for_unknown_pid)
{
    // session_ref creates an empty slot on first access; bridge contract
    // for fork_gotcha which calls .release() on the parent's slot from the
    // child. Slot must exist (even if empty) so .release() is safe.
    rocprofsys::core::perfetto_engine engine{ make_test_config() };

    auto& slot = engine.session_ref(static_cast<pid_t>(54321));
    EXPECT_EQ(slot.get(), nullptr);

    // Calling again returns the same slot.
    auto& slot2 = engine.session_ref(static_cast<pid_t>(54321));
    EXPECT_EQ(&slot, &slot2);
}
