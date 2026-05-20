// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "core/perfetto/engine.hpp"
#include "core/perfetto/sinks.hpp"

#include <algorithm>
#include <thread>
#include <variant>
#include <vector>

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
    rocprofsys::core::engine_config   cfg = make_test_config();
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

TEST(perfetto_engine, destroy_session_idempotent)
{
    // destroy_session on an unknown pid is a no-op; calling it twice on
    // the same pid is also fine. Used for post-stop cleanup of a
    // genuinely-finished session.
    rocprofsys::core::perfetto_engine engine{ make_test_config() };

    EXPECT_NO_THROW(engine.destroy_session(static_cast<pid_t>(11111)));
    EXPECT_NO_THROW(engine.destroy_session(static_cast<pid_t>(11111)));
}

TEST(perfetto_engine, set_emitting_pid_round_trip)
{
    // The thread_local pid tag set by set_emitting_pid is read back by
    // get_emitting_pid on the same thread.
    rocprofsys::core::set_emitting_pid(4242);
    EXPECT_EQ(rocprofsys::core::get_emitting_pid(), 4242);

    // Reset for subsequent test cases on this thread.
    rocprofsys::core::set_emitting_pid(-1);
}

TEST(perfetto_engine, emitting_pid_is_thread_local)
{
    // emitting pid is per-thread. Setting on one thread must not leak
    // to other threads.
    rocprofsys::core::set_emitting_pid(7777);
    EXPECT_EQ(rocprofsys::core::get_emitting_pid(), 7777);

    int         observed_on_other_thread = 0;
    std::thread other{ [&observed_on_other_thread]() {
        observed_on_other_thread = rocprofsys::core::get_emitting_pid();
    } };
    other.join();

    EXPECT_EQ(observed_on_other_thread, -1)
        << "emitting pid must default to -1 on a fresh thread";
    EXPECT_EQ(rocprofsys::core::get_emitting_pid(), 7777)
        << "main thread's tag must be unchanged by other thread's read";

    rocprofsys::core::set_emitting_pid(-1);
}

TEST(perfetto_engine, forget_session_on_unknown_pid_is_noop)
{
    // forget_session is the fork_gotcha bridge for the child to drop the
    // parent's inherited session pointer without destroying the underlying
    // TracingSession. Unknown pid → no-op (no exception, no allocation).
    rocprofsys::core::perfetto_engine engine{ make_test_config() };

    EXPECT_NO_THROW(engine.forget_session(static_cast<pid_t>(54321)));
    EXPECT_NO_THROW(engine.forget_session(static_cast<pid_t>(54321)));
}

// ----------------------------------------------------------------------------
// Cached-interceptor mode
// ----------------------------------------------------------------------------

// Helper: simulate the cached-mode interceptor pushing bytes for a pid.
// Stand-in for `cached_interceptor::OnTracePacket -> collect_packet_bytes`,
// avoiding the heavy perfetto.hpp / TRACE_EVENT include chain in test
// scope. SDK-driven emission is covered end-to-end by the pytest integration
// suite in tests/pytest/.
namespace
{
void
simulate_interceptor_emit(rocprofsys::core::perfetto_engine& engine, int pid,
                          const std::vector<char>& bytes)
{
    engine.collect_packet_bytes(pid, bytes.data(), bytes.size());
}
}  // namespace

TEST(perfetto_engine_cached, start_then_stop_with_no_emission_drains_empty)
{
    // Cached mode with zero emissions: engine.stop() must invoke
    // sink.finalize() and produce zero records — verifies the drain
    // pathway runs without crashing when no thread tagged itself.
    rocprofsys::core::perfetto_engine engine{ make_test_config() };
    engine.init_sdk();

    rocprofsys::core::trace_sink sink{ rocprofsys::core::recording_sink{} };

    engine.start(rocprofsys::core::perfetto_engine::mode::cached_interceptor, sink);
    EXPECT_TRUE(engine.is_running());
    engine.stop();

    EXPECT_FALSE(engine.is_running());
    const auto& rec = std::get<rocprofsys::core::recording_sink>(sink);
    EXPECT_TRUE(rec.finalized());
    EXPECT_TRUE(rec.records().empty());
}

// collect_packet_bytes wraps each raw TracePacket in the Trace.packets
// length-delimited header (tag 0x0A + varint(size)). These helpers
// compute the expected framed byte sequence so tests stay readable.
namespace
{
std::vector<char>
frame_packet(const std::vector<char>& payload)
{
    std::vector<char> out;
    out.reserve(payload.size() + 2);
    out.push_back(static_cast<char>(0x0A));  // field 1, wire type 2
    std::size_t v = payload.size();
    while(v >= 0x80)
    {
        out.push_back(static_cast<char>((v & 0x7F) | 0x80));
        v >>= 7;
    }
    out.push_back(static_cast<char>(v));
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}
}  // namespace

TEST(perfetto_engine_cached, drain_one_source_one_record)
{
    // Simulate one source emitting bytes keyed pid=42; engine.stop() must
    // produce exactly one drained record whose contents are the original
    // payload wrapped in the Trace.packets length-delimited frame.
    rocprofsys::core::perfetto_engine engine{ make_test_config() };
    engine.init_sdk();

    rocprofsys::core::trace_sink sink{ rocprofsys::core::recording_sink{} };
    engine.start(rocprofsys::core::perfetto_engine::mode::cached_interceptor, sink);
    engine.preregister_pids({ 42 });

    const std::vector<char> payload{ 'p', 'a', 'c', 'k', 'e', 't' };
    simulate_interceptor_emit(engine, 42, payload);

    engine.stop();

    const auto& rec = std::get<rocprofsys::core::recording_sink>(sink);
    ASSERT_EQ(rec.records().size(), 1u);
    EXPECT_EQ(rec.records()[0].first, 42);
    EXPECT_EQ(rec.records()[0].second, frame_packet(payload));
    EXPECT_TRUE(rec.finalized());
}

TEST(perfetto_engine_cached, drain_two_sources_no_cross_bleed)
{
    // Two sources with different pids; engine.stop() must produce two
    // records, each containing only its source's framed bytes (no
    // cross-pid bleed in the engine's per-pid collector). Per-source
    // drain order is intentionally unspecified — the engine iterates an
    // unordered_map, so assertions accept either order.
    rocprofsys::core::perfetto_engine engine{ make_test_config() };
    engine.init_sdk();

    rocprofsys::core::trace_sink sink{ rocprofsys::core::recording_sink{} };
    engine.start(rocprofsys::core::perfetto_engine::mode::cached_interceptor, sink);
    engine.preregister_pids({ 101, 202 });

    const std::vector<char> payload_a{ 'a', 'a', 'a' };
    const std::vector<char> payload_b{ 'b', 'b' };
    simulate_interceptor_emit(engine, 101, payload_a);
    simulate_interceptor_emit(engine, 202, payload_b);

    engine.stop();

    const auto& rec = std::get<rocprofsys::core::recording_sink>(sink);
    ASSERT_EQ(rec.records().size(), 2u);

    // Build the expected (source_id, framed_bytes) pairs and assert
    // set-equality. EXPECT_THAT(UnorderedElementsAre) would be cleaner
    // but the existing tests don't pull in <gmock/gmock.h>; manual
    // 2-element check keeps the dependency set unchanged.
    auto       got        = rec.records();
    const auto expected_a = std::make_pair(101, frame_packet(payload_a));
    const auto expected_b = std::make_pair(202, frame_packet(payload_b));
    const bool ab         = got[0] == expected_a && got[1] == expected_b;
    const bool ba         = got[0] == expected_b && got[1] == expected_a;
    EXPECT_TRUE(ab || ba) << "records must contain both sources, order is unspecified";
    EXPECT_TRUE(rec.finalized());
}

TEST(perfetto_engine_cached, multiple_emits_same_pid_concatenate)
{
    // Two emits for the same pid must concatenate into a single drained
    // record — each emit's bytes get their own length-delimited frame
    // header, and concatenation forms a valid Trace proto.
    rocprofsys::core::perfetto_engine engine{ make_test_config() };
    engine.init_sdk();

    rocprofsys::core::trace_sink sink{ rocprofsys::core::recording_sink{} };
    engine.start(rocprofsys::core::perfetto_engine::mode::cached_interceptor, sink);
    engine.preregister_pids({ 7 });

    const std::vector<char> first{ '1', '2' };
    const std::vector<char> second{ '3', '4', '5' };
    simulate_interceptor_emit(engine, 7, first);
    simulate_interceptor_emit(engine, 7, second);

    engine.stop();

    const auto& rec = std::get<rocprofsys::core::recording_sink>(sink);
    ASSERT_EQ(rec.records().size(), 1u);
    EXPECT_EQ(rec.records()[0].first, 7);

    auto expected      = frame_packet(first);
    auto second_framed = frame_packet(second);
    expected.insert(expected.end(), second_framed.begin(), second_framed.end());
    EXPECT_EQ(rec.records()[0].second, expected);
}

// ----------------------------------------------------------------------------
// engine.stop() exception contract
// ----------------------------------------------------------------------------

namespace
{
// Test double whose on_source_drained throws. Used to verify the engine's
// drain contract:
//   - finalize() is always called even if a per-source drain throws
//   - the first thrown exception is rethrown after finalize()
//   - other per-source drains still run after a throw
// Engine reaches this fixture via polymorphic_sink_view (a variant
// alternative of trace_sink) — no virtual base required.
class throwing_sink
{
public:
    explicit throwing_sink(std::vector<int> pids_to_throw_on)
    : m_throw_pids{ std::move(pids_to_throw_on) }
    {}

    void on_source_drained(int source_id, std::vector<char> bytes)
    {
        m_drained_count++;
        if(std::find(m_throw_pids.begin(), m_throw_pids.end(), source_id) !=
           m_throw_pids.end())
        {
            m_throw_count++;
            throw std::runtime_error{ "throwing_sink on_source_drained: pid " +
                                      std::to_string(source_id) };
        }
        m_kept.emplace_back(source_id, std::move(bytes));
    }

    void finalize() { m_finalize_count++; }

    int drained_count() const noexcept { return m_drained_count; }
    int throw_count() const noexcept { return m_throw_count; }
    int finalize_count() const noexcept { return m_finalize_count; }
    const std::vector<std::pair<int, std::vector<char>>>& kept() const noexcept
    {
        return m_kept;
    }

private:
    std::vector<int>                               m_throw_pids;
    std::vector<std::pair<int, std::vector<char>>> m_kept;
    int                                            m_drained_count{ 0 };
    int                                            m_throw_count{ 0 };
    int                                            m_finalize_count{ 0 };
};
}  // namespace

TEST(perfetto_engine_cached, stop_calls_finalize_even_when_drain_throws)
{
    rocprofsys::core::perfetto_engine engine{ make_test_config() };
    engine.init_sdk();
    throwing_sink                target{ { 42 } };  // throws on pid 42
    rocprofsys::core::trace_sink sink{ rocprofsys::core::polymorphic_sink_view{
        target } };

    engine.start(rocprofsys::core::perfetto_engine::mode::cached_interceptor, sink);
    engine.preregister_pids({ 42 });
    simulate_interceptor_emit(engine, 42, std::vector<char>{ 'x' });

    EXPECT_THROW(engine.stop(), std::runtime_error);
    EXPECT_EQ(target.throw_count(), 1);
    EXPECT_EQ(target.finalize_count(), 1)
        << "finalize() must run even after on_source_drained threw";
}

TEST(perfetto_engine_cached, stop_rethrows_first_drain_exception_after_finalize)
{
    rocprofsys::core::perfetto_engine engine{ make_test_config() };
    engine.init_sdk();
    throwing_sink                target{ { 101, 202 } };  // both throw
    rocprofsys::core::trace_sink sink{ rocprofsys::core::polymorphic_sink_view{
        target } };

    engine.start(rocprofsys::core::perfetto_engine::mode::cached_interceptor, sink);
    engine.preregister_pids({ 101, 202, 303 });  // 303 won't throw
    simulate_interceptor_emit(engine, 101, std::vector<char>{ 'a' });
    simulate_interceptor_emit(engine, 202, std::vector<char>{ 'b' });
    simulate_interceptor_emit(engine, 303, std::vector<char>{ 'c' });

    EXPECT_THROW(engine.stop(), std::runtime_error);
    EXPECT_EQ(target.drained_count(), 3)
        << "all three sources must be attempted even after the first throw";
    EXPECT_EQ(target.throw_count(), 2);
    EXPECT_EQ(target.finalize_count(), 1);
    ASSERT_EQ(target.kept().size(), 1u) << "only the non-throwing source's bytes kept";
    EXPECT_EQ(target.kept()[0].first, 303);
}
