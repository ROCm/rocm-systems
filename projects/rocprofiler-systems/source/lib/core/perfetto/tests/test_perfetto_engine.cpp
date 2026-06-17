// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "core/perfetto/engine.hpp"
#include "core/perfetto/engine_impl.hpp"
#include "core/perfetto/packet_framing.hpp"
#include "core/perfetto/session_backend.hpp"
#include "core/perfetto/sinks/trace_sink.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include <unistd.h>

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

namespace
{
class mock_trace_session
{
public:
    virtual ~mock_trace_session() = default;

    MOCK_METHOD(void, SetOnErrorCallback, (std::function<void(int)>) );
    MOCK_METHOD(void, Setup, (const int&, int) );
    MOCK_METHOD(void, StartBlocking, ());
    MOCK_METHOD(void, FlushBlocking, ());
    MOCK_METHOD(void, StopBlocking, ());
};

class mock_session_backend_delegate
{
public:
    MOCK_METHOD(std::unique_ptr<mock_trace_session>, new_trace, ());
    MOCK_METHOD(void, flush_track_events, ());
};

std::unique_ptr<mock_session_backend_delegate> g_mock_session_backend;

struct mock_session_backend
{
    [[nodiscard]] std::unique_ptr<mock_trace_session> new_trace()
    {
        return g_mock_session_backend->new_trace();
    }

    void flush_track_events() { g_mock_session_backend->flush_track_events(); }
};

class session_backend_policy_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        g_mock_session_backend =
            std::make_unique<::testing::StrictMock<mock_session_backend_delegate>>();
    }

    void TearDown() override { g_mock_session_backend.reset(); }
};

struct incomplete_session_backend
{};

static_assert(rocprofsys::core::trace_session_backend<mock_session_backend>);
static_assert(rocprofsys::core::trace_session_backend_for<mock_session_backend, int,
                                                          std::function<void(int)>>);
static_assert(rocprofsys::core::flushable_trace_session_backend<mock_session_backend>);
static_assert(!rocprofsys::core::trace_session_backend<incomplete_session_backend>);
static_assert(
    !rocprofsys::core::flushable_trace_session_backend<incomplete_session_backend>);
static_assert(rocprofsys::core::stoppable_trace_session<mock_trace_session>);
static_assert(
    rocprofsys::core::trace_session<mock_trace_session, int, std::function<void(int)>>);
}  // namespace

TEST_F(session_backend_policy_test, start_and_stop_sequence_is_mockable)
{
    int                  trace_cfg = 0;
    mock_session_backend backend{};

    auto session = std::unique_ptr<mock_trace_session>{
        new ::testing::StrictMock<mock_trace_session>{}
    };
    auto* session_ptr = session.get();

    {
        ::testing::InSequence seq;
        EXPECT_CALL(*g_mock_session_backend, new_trace())
            .WillOnce(::testing::Return(::testing::ByMove(std::move(session))));
        EXPECT_CALL(*session_ptr, SetOnErrorCallback(::testing::_));
        EXPECT_CALL(*session_ptr, Setup(::testing::_, 17));
        EXPECT_CALL(*session_ptr, StartBlocking());
    }

    auto started =
        rocprofsys::core::start_tracing_session(backend, trace_cfg, 17, [](auto) {});

    {
        ::testing::InSequence seq;
        EXPECT_CALL(*g_mock_session_backend, flush_track_events());
        EXPECT_CALL(*session_ptr, FlushBlocking());
        EXPECT_CALL(*session_ptr, StopBlocking());
    }

    rocprofsys::core::flush_and_stop_session(backend, *started);
}

// ----------------------------------------------------------------------------
// basic_cached_perfetto_engine policy backend seam
// ----------------------------------------------------------------------------
namespace
{
struct mock_backend_session
{};

class mock_perfetto_backend_recorder
{
public:
    MOCK_METHOD(void, init_sdk, (const rocprofsys::core::engine_config&) );
    MOCK_METHOD(mock_backend_session*, start_cached_session,
                (const rocprofsys::core::engine_config&) );
    MOCK_METHOD(void, flush_and_stop, (mock_backend_session*) );
};

std::unique_ptr<mock_perfetto_backend_recorder> g_mock_perfetto_backend;

struct mock_perfetto_backend
{
    using session_ptr = mock_backend_session*;

    void init_sdk(const rocprofsys::core::engine_config& cfg) const
    {
        g_mock_perfetto_backend->init_sdk(cfg);
    }

    [[nodiscard]] session_ptr start_cached_session(
        const rocprofsys::core::engine_config& cfg) const
    {
        return g_mock_perfetto_backend->start_cached_session(cfg);
    }

    void flush_and_stop(session_ptr& session) const
    {
        g_mock_perfetto_backend->flush_and_stop(session);
    }
};

static_assert(rocprofsys::core::perfetto_backend<mock_perfetto_backend>);

class perfetto_engine_backend_policy_test : public ::testing::Test
{
protected:
    using mock_engine =
        rocprofsys::core::basic_cached_perfetto_engine<mock_perfetto_backend>;

    void SetUp() override
    {
        g_mock_perfetto_backend =
            std::make_unique<::testing::StrictMock<mock_perfetto_backend_recorder>>();
    }

    void TearDown() override { g_mock_perfetto_backend.reset(); }

    mock_backend_session m_session{};
};
}  // namespace

TEST_F(perfetto_engine_backend_policy_test, init_sdk_delegates_config_to_backend_policy)
{
    auto cfg = make_test_config();
    cfg.disabled_categories.push_back("mock-category");
    cfg.suppress_sdk_log_output = true;
    mock_engine engine{ cfg };

    EXPECT_CALL(*g_mock_perfetto_backend, init_sdk(::testing::_))
        .WillOnce([&cfg](const rocprofsys::core::engine_config& observed) {
            EXPECT_EQ(observed.buffer_size_kb, cfg.buffer_size_kb);
            EXPECT_EQ(observed.shmem_size_hint_kb, cfg.shmem_size_hint_kb);
            EXPECT_EQ(observed.flush_period_ms, cfg.flush_period_ms);
            EXPECT_EQ(observed.fill_policy, cfg.fill_policy);
            EXPECT_EQ(observed.backend, cfg.backend);
            EXPECT_EQ(observed.disabled_categories, cfg.disabled_categories);
            EXPECT_EQ(observed.suppress_sdk_log_output, cfg.suppress_sdk_log_output);
        });

    engine.init_sdk();
}

TEST_F(perfetto_engine_backend_policy_test,
       cached_start_uses_backend_policy_and_stop_drains_collected_bytes)
{
    mock_engine                  engine{ make_test_config() };
    rocprofsys::core::trace_sink sink{ rocprofsys::core::recording_sink{} };

    {
        ::testing::InSequence seq;
        EXPECT_CALL(*g_mock_perfetto_backend, start_cached_session(::testing::_))
            .WillOnce(::testing::Return(&m_session));
        EXPECT_CALL(*g_mock_perfetto_backend, flush_and_stop(&m_session));
    }

    engine.start(sink);
    engine.preregister_pids({ 42 });
    const std::vector<char> payload{ 'm', 'o', 'c', 'k' };
    engine.collect_packet_bytes(42, payload.data(), payload.size());

    engine.stop();

    const auto& rec = std::get<rocprofsys::core::recording_sink>(sink);
    ASSERT_EQ(rec.records().size(), 1u);
    EXPECT_EQ(rec.records()[0].first, 42);
    EXPECT_TRUE(rec.finalized());
}

TEST_F(perfetto_engine_backend_policy_test,
       system_or_all_backend_does_not_start_backend_session)
{
    EXPECT_CALL(*g_mock_perfetto_backend, start_cached_session(::testing::_)).Times(0);

    for(auto backend : { rocprofsys::core::engine_config::backend_t::system,
                         rocprofsys::core::engine_config::backend_t::all })
    {
        auto cfg    = make_test_config();
        cfg.backend = backend;
        mock_engine                  engine{ cfg };
        rocprofsys::core::trace_sink sink{ rocprofsys::core::recording_sink{} };

        engine.start(sink);
        EXPECT_FALSE(engine.is_running());
    }
}

TEST(perfetto_engine, construct_from_config_literal_no_config_access)
{
    // Engine must be instantiable in unit tests with no global pollution:
    // construction must not touch rocprofsys::config::*. Passing a literal
    // engine_config exercises this.
    rocprofsys::core::engine_config          cfg = make_test_config();
    rocprofsys::core::cached_perfetto_engine engine{ cfg };

    EXPECT_FALSE(engine.is_running());
}

TEST(perfetto_engine, two_instances_no_shared_running_state)
{
    // A second engine constructed after the first sees pristine state — no
    // hidden global handed between instances.
    rocprofsys::core::cached_perfetto_engine first{ make_test_config() };
    rocprofsys::core::cached_perfetto_engine second{ make_test_config() };

    EXPECT_FALSE(first.is_running());
    EXPECT_FALSE(second.is_running());
}

TEST(perfetto_engine, stop_without_start_is_noop)
{
    // engine.stop() invoked before start() returns without error.
    rocprofsys::core::cached_perfetto_engine engine{ make_test_config() };

    EXPECT_NO_THROW(engine.stop());
    EXPECT_FALSE(engine.is_running());
}

TEST(perfetto_engine, cached_start_with_system_backend_warns_and_stays_stopped)
{
    auto cfg    = make_test_config();
    cfg.backend = rocprofsys::core::engine_config::backend_t::system;

    rocprofsys::core::cached_perfetto_engine engine{ cfg };
    rocprofsys::core::trace_sink             sink{ rocprofsys::core::recording_sink{} };

    ::testing::internal::CaptureStdout();
    ::testing::internal::CaptureStderr();
    engine.start(sink);
    const auto stderr_output = ::testing::internal::GetCapturedStderr();
    const auto stdout_output = ::testing::internal::GetCapturedStdout();

    EXPECT_FALSE(engine.is_running());
    EXPECT_THAT(stderr_output + stdout_output,
                ::testing::HasSubstr("cached output is unsupported"));

    const auto& rec = std::get<rocprofsys::core::recording_sink>(sink);
    EXPECT_FALSE(rec.finalized()) << "early-return warning path must not bind the sink";
    EXPECT_TRUE(rec.records().empty());
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
simulate_interceptor_emit(rocprofsys::core::cached_perfetto_engine& engine, int pid,
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
    rocprofsys::core::cached_perfetto_engine engine{ make_test_config() };
    engine.init_sdk();

    rocprofsys::core::trace_sink sink{ rocprofsys::core::recording_sink{} };

    engine.start(sink);
    EXPECT_TRUE(engine.is_running());
    engine.stop();

    EXPECT_FALSE(engine.is_running());
    const auto& rec = std::get<rocprofsys::core::recording_sink>(sink);
    EXPECT_TRUE(rec.finalized());
    EXPECT_TRUE(rec.records().empty());
}

// Builds the expected framed byte sequence for a raw TracePacket payload
// by reusing the production helpers — keeps the test in lockstep with the
// engine's framing rather than re-deriving the protobuf wire format here.
namespace
{
std::vector<char>
frame_packet(const std::vector<char>& payload)
{
    std::vector<char> out;
    out.reserve(payload.size() + rocprofsys::core::TRACE_PACKETS_TAG_BYTES +
                rocprofsys::core::VARINT_MAX_BYTES);
    out.push_back(static_cast<char>(rocprofsys::core::TRACE_PACKETS_TAG));
    rocprofsys::core::append_varint(out, static_cast<std::uint64_t>(payload.size()));
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}
}  // namespace

TEST(perfetto_engine_cached, drain_one_source_one_record)
{
    // Simulate one source emitting bytes keyed pid=42; engine.stop() must
    // produce exactly one drained record whose contents are the original
    // payload wrapped in the Trace.packets length-delimited frame.
    rocprofsys::core::cached_perfetto_engine engine{ make_test_config() };
    engine.init_sdk();

    rocprofsys::core::trace_sink sink{ rocprofsys::core::recording_sink{} };
    engine.start(sink);
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
    rocprofsys::core::cached_perfetto_engine engine{ make_test_config() };
    engine.init_sdk();

    rocprofsys::core::trace_sink sink{ rocprofsys::core::recording_sink{} };
    engine.start(sink);
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
    rocprofsys::core::cached_perfetto_engine engine{ make_test_config() };
    engine.init_sdk();

    rocprofsys::core::trace_sink sink{ rocprofsys::core::recording_sink{} };
    engine.start(sink);
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

TEST(perfetto_engine_cached, collect_before_preregister_is_dropped)
{
    rocprofsys::core::cached_perfetto_engine engine{ make_test_config() };
    engine.init_sdk();

    rocprofsys::core::trace_sink sink{ rocprofsys::core::recording_sink{} };
    engine.start(sink);

    const std::vector<char> payload{ 1, 2, 3, 4 };
    simulate_interceptor_emit(engine, 77, payload);

    engine.stop();

    const auto& rec = std::get<rocprofsys::core::recording_sink>(sink);
    EXPECT_TRUE(rec.records().empty());
    EXPECT_TRUE(rec.finalized());
}

TEST(perfetto_engine_cached, collect_for_unregistered_pid_is_dropped)
{
    rocprofsys::core::cached_perfetto_engine engine{ make_test_config() };
    engine.init_sdk();

    rocprofsys::core::trace_sink sink{ rocprofsys::core::recording_sink{} };
    engine.start(sink);
    engine.preregister_pids({ 10 });

    const std::vector<char> kept{ 5, 6, 7, 8 };
    const std::vector<char> dropped{ 9, 10, 11, 12 };
    simulate_interceptor_emit(engine, 10, kept);
    simulate_interceptor_emit(engine, 11, dropped);

    engine.stop();

    const auto& rec = std::get<rocprofsys::core::recording_sink>(sink);
    ASSERT_EQ(rec.records().size(), 1u);
    EXPECT_EQ(rec.records()[0].first, 10);
    EXPECT_EQ(rec.records()[0].second, frame_packet(kept));
    EXPECT_TRUE(rec.finalized());
}

TEST(perfetto_engine_cached, preregister_after_freeze_does_not_add_new_pid)
{
    rocprofsys::core::cached_perfetto_engine engine{ make_test_config() };
    engine.init_sdk();

    rocprofsys::core::trace_sink sink{ rocprofsys::core::recording_sink{} };
    engine.start(sink);
    engine.preregister_pids({ 1 });
    engine.preregister_pids({ 2 });

    const std::vector<char> first{ 13, 14, 15 };
    const std::vector<char> second{ 16, 17, 18 };
    simulate_interceptor_emit(engine, 1, first);
    simulate_interceptor_emit(engine, 2, second);

    engine.stop();

    const auto& rec = std::get<rocprofsys::core::recording_sink>(sink);
    ASSERT_EQ(rec.records().size(), 1u);
    EXPECT_EQ(rec.records()[0].first, 1);
    EXPECT_EQ(rec.records()[0].second, frame_packet(first));
    EXPECT_TRUE(rec.finalized());
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
    rocprofsys::core::cached_perfetto_engine engine{ make_test_config() };
    engine.init_sdk();
    throwing_sink                target{ { 42 } };  // throws on pid 42
    rocprofsys::core::trace_sink sink{ rocprofsys::core::polymorphic_sink_view{
        target } };

    engine.start(sink);
    engine.preregister_pids({ 42 });
    simulate_interceptor_emit(engine, 42, std::vector<char>{ 'x' });

    EXPECT_THROW(engine.stop(), std::runtime_error);
    EXPECT_EQ(target.throw_count(), 1);
    EXPECT_EQ(target.finalize_count(), 1)
        << "finalize() must run even after on_source_drained threw";
}

TEST(perfetto_engine_cached, stop_rethrows_first_drain_exception_after_finalize)
{
    rocprofsys::core::cached_perfetto_engine engine{ make_test_config() };
    engine.init_sdk();
    throwing_sink                target{ { 101, 202 } };  // both throw
    rocprofsys::core::trace_sink sink{ rocprofsys::core::polymorphic_sink_view{
        target } };

    engine.start(sink);
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

TEST(perfetto_engine_cached, concurrent_collect_packet_bytes_no_loss_or_bleed)
{
    // The lock-free hot path: after preregister_pids freezes the per-pid
    // map, parser threads call collect_packet_bytes against their own pid
    // with no synchronization. The contract is single-writer-per-pid
    // across all threads. This stress run launches one parser thread per
    // pid; each pushes a sequence of distinct payloads only into its own
    // pid's slot. After stop(), every preregistered pid's drained record
    // must contain exactly its own payloads in submission order, with no
    // bytes from any other pid mixed in.
    rocprofsys::core::cached_perfetto_engine engine{ make_test_config() };
    engine.init_sdk();

    rocprofsys::core::trace_sink sink{ rocprofsys::core::recording_sink{} };
    engine.start(sink);

    constexpr int pid_count       = 4;
    constexpr int packets_per_pid = 64;

    std::vector<int> pids;
    pids.reserve(pid_count);
    for(int i = 0; i < pid_count; ++i)
        pids.push_back(1000 + i);
    engine.preregister_pids(pids);

    // Per-pid payloads are tagged with (pid, index) so cross-pid bleed
    // is detectable by inspecting any single byte of any record.
    auto make_payload = [](int pid, int idx) {
        std::vector<char> payload;
        payload.reserve(8);
        payload.push_back(static_cast<char>('P'));
        payload.push_back(static_cast<char>((pid >> 8) & 0xff));
        payload.push_back(static_cast<char>(pid & 0xff));
        payload.push_back(static_cast<char>('#'));
        payload.push_back(static_cast<char>((idx >> 8) & 0xff));
        payload.push_back(static_cast<char>(idx & 0xff));
        return payload;
    };

    std::atomic<int>         ready{ 0 };
    std::atomic<bool>        go{ false };
    std::vector<std::thread> threads;
    threads.reserve(pid_count);
    for(int pid : pids)
    {
        threads.emplace_back([&, pid]() {
            ++ready;
            while(!go.load(std::memory_order_acquire))
            {
            }
            for(int i = 0; i < packets_per_pid; ++i)
            {
                const auto payload = make_payload(pid, i);
                engine.collect_packet_bytes(pid, payload.data(), payload.size());
            }
        });
    }
    while(ready.load() < pid_count)
    {
    }
    go.store(true, std::memory_order_release);
    for(auto& t : threads)
        t.join();

    engine.stop();

    const auto& rec = std::get<rocprofsys::core::recording_sink>(sink);
    ASSERT_EQ(rec.records().size(), static_cast<std::size_t>(pid_count));

    // Build the expected per-pid concatenated framed bytes and check
    // that every preregistered pid shows up exactly once with exactly
    // its payloads, in submission order, with no other pid's bytes
    // mixed in. Records come out in unordered_map iteration order, so
    // we lookup-by-pid rather than asserting index order.
    for(int pid : pids)
    {
        const auto it = std::find_if(rec.records().begin(), rec.records().end(),
                                     [pid](const auto& r) { return r.first == pid; });
        ASSERT_NE(it, rec.records().end())
            << "no record drained for preregistered pid " << pid;

        std::vector<char> expected;
        for(int i = 0; i < packets_per_pid; ++i)
        {
            const auto framed = frame_packet(make_payload(pid, i));
            expected.insert(expected.end(), framed.begin(), framed.end());
        }
        EXPECT_EQ(it->second, expected) << "drained bytes for pid " << pid
                                        << " do not match the concatenated submission";
    }

    EXPECT_TRUE(rec.finalized());
}
