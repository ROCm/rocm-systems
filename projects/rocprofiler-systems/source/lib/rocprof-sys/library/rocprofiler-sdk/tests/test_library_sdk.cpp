// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// SUT first: its include chain defines ROCPROFILER_VERSION, _OPENMP, and other
// SDK/OpenMP macros that mock_wrapper.hpp uses in #ifdef guards to pick the real
// enum types (rocprofiler_callback_tracing_kind_t etc.) when available.
// Template instantiation of library_sdk<mock_backend> happens at end-of-TU, so
// both headers are fully visible regardless of include order.
#include "mock_wrapper.hpp"
#include "rocprof-sys/library/rocprofiler-sdk.hpp"
#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>

// ─── mock_externals ───────────────────────────────────────────────────────────
//
// Null-object stub for the Externals policy parameter.  All boolean config
// getters return false so that every Perfetto/timemory code path in library_sdk
// is compiled out at template instantiation time.  The remaining methods are
// no-op templates that satisfy the duck-typed interface without pulling in any
// real subsystem.

struct mock_externals
{
    // ─── Stub types matching the concrete args used in library_sdk ────────────
    //
    // add_thread_info and add_track are called with brace-enclosed initializers
    // whose field count must match the struct layout (C++ cannot deduce a type
    // from a bare brace-initializer in a template call).

    struct null_thread_info
    {
        std::int32_t  parent_process_id = 0;
        std::int32_t  process_id        = 0;
        std::uint64_t thread_id         = 0;
        std::uint32_t start             = 0;
        std::uint32_t end               = 0;
        std::string   extdata;
    };

    struct null_track_info
    {
        std::string                track_name;
        std::optional<std::size_t> thread_id;
        std::string                extdata;
    };

    struct null_thread_index_data
    {
        std::size_t sequent_value = 0;
    };

    struct null_thread_data
    {
        null_thread_index_data  _index{};
        null_thread_index_data* index_data = &_index;
    };

    struct null_registry
    {
        void add_string(std::string_view) {}
        void add_thread_info(const null_thread_info&) {}
        void add_track(const null_track_info&) {}
        void add_queue(const std::uint64_t&) {}
        void add_stream(const std::uint64_t&) {}

        template <typename T>
        void add_code_object(T&&)
        {}
        template <typename T>
        void add_kernel_symbol(T&&)
        {}

        struct null_tracing_info
        {
            template <typename A, typename B>
            std::string_view at(A, B) const
            {
                return {};
            }
        };
        [[nodiscard]] null_tracing_info get_callback_tracing_info() const { return {}; }
    };

    struct null_storage
    {
        template <typename... Args>
        void store(Args&&...)
        {}
    };

    static null_registry& get_metadata_registry()
    {
        static null_registry reg;
        return reg;
    }
    static null_storage& get_buffer_storage()
    {
        static null_storage storage;
        return storage;
    }

    static bool get_use_perfetto() { return false; }
    static bool get_use_timemory() { return false; }
    static bool get_perfetto_annotations() { return false; }
    static bool get_use_rocpd() { return false; }
    static bool get_group_by_queue() { return false; }

    template <typename... Args>
    static void push_timemory(Args&&...)
    {}
    template <typename... Args>
    static void pop_timemory(Args&&...)
    {}
    template <typename... Args>
    static void push_perfetto_ts(Args&&...)
    {}
    template <typename... Args>
    static void pop_perfetto_ts(Args&&...)
    {}
    template <typename... Args>
    static void push_perfetto(Args&&...)
    {}
    template <typename... Args>
    static void pop_perfetto(Args&&...)
    {}
    template <typename... Args>
    static void add_perfetto_annotation(Args&&...)
    {}

    template <typename... Args>
    static std::nullptr_t get_perfetto_track(Args&&...)
    {
        return nullptr;
    }

    template <typename TidT, typename TagT>
    static const null_thread_data* get_thread_info(TidT, TagT)
    {
        static null_thread_data stub{};
        return &stub;
    }

    // ─── MarkerWriterPolicy interface (no-ops) ────────────────────────────────────
    // Called by roctx_client/marker_writer when library_sdk uses Externals as the
    // marker policy.  The variadic templates above (push_timemory, pop_timemory,
    // push_perfetto_ts, pop_perfetto_ts) already cover those signatures.

    static void add_string(std::string_view) {}
    static void store_region(const ::rocprofsys::trace_cache::region_sample&) {}
    static void add_thread_info(const ::rocprofsys::trace_cache::info::thread&) {}

    // ─── PMC interface (no-ops) ───────────────────────────────────────────────────
    static void register_gpu_perf_counter_source(
        const std::vector<std::shared_ptr<::rocprofsys::agent>>&)
    {}
    static void set_pmc_state(::rocprofsys::State) {}
};

// ─── Type aliases ─────────────────────────────────────────────────────────────

namespace mock_ns = ::rocprofsys::mock::rocprofiler_sdk;

using mock_backend_t = mock_ns::backend;
using sut    = ::rocprofsys::rocprofiler_sdk::library_sdk<mock_backend_t, mock_externals>;
using data_t = ::rocprofsys::rocprofiler_sdk::client_data<mock_backend_t>;
using handle_t = mock_backend_t::handle_t;

// Bring in gmock helpers used throughout to avoid per-test repetition.
using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

static constexpr auto SUCCESS = mock_backend_t::STATUS_SUCCESS;

// ─── Fixture ──────────────────────────────────────────────────────────────────

class library_sdk_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mock_ns::g_mock_wrapper = std::make_unique<mock_ns::gmock_wrapper>();

        sut::tool_fini_done.store(false);
        sut::tool_init_done.store(false);
        sut::sdk_configured.store(false);
        sut::g_roctx_client.reset();

        delete sut::tool_data;
        sut::tool_data = new data_t{};
    }

    void TearDown() override { mock_ns::g_mock_wrapper.reset(); }

    mock_ns::gmock_wrapper& mock() { return *mock_ns::g_mock_wrapper; }
};

// ─── start() tests ────────────────────────────────────────────────────────────

// Initialized context that is currently inactive: context_is_active is queried,
// then start_context is called once.
TEST_F(library_sdk_test, start_calls_start_context_for_initialized_inactive_context)
{
    const handle_t ctx          = { 42 };
    sut::tool_data->primary_ctx = ctx;

    EXPECT_CALL(mock(), context_is_active(ctx, _))
        .WillOnce(DoAll(SetArgPointee<1>(0), Return(SUCCESS)));
    EXPECT_CALL(mock(), start_context(ctx)).WillOnce(Return(SUCCESS));

    sut::start();
}

// When the context is already active, context_is_active returns active and
// start_context must NOT be called.
TEST_F(library_sdk_test, start_skips_start_context_when_context_already_active)
{
    const handle_t ctx          = { 42 };
    sut::tool_data->primary_ctx = ctx;

    EXPECT_CALL(mock(), context_is_active(ctx, _))
        .WillOnce(DoAll(SetArgPointee<1>(1), Return(SUCCESS)));
    EXPECT_CALL(mock(), start_context(_)).Times(0);

    sut::start();
}

// All four contexts are zero-handle (not initialized): no SDK call at all.
TEST_F(library_sdk_test, start_makes_no_sdk_calls_when_all_contexts_are_uninitialized)
{
    EXPECT_CALL(mock(), context_is_active(_, _)).Times(0);
    EXPECT_CALL(mock(), start_context(_)).Times(0);

    sut::start();
}

// All four contexts initialized and inactive: start_context called for every one.
TEST_F(library_sdk_test,
       start_calls_start_context_for_all_four_initialized_inactive_contexts)
{
    sut::tool_data->primary_ctx     = { 10 };
    sut::tool_data->counter_ctx     = { 20 };
    sut::tool_data->code_object_ctx = { 30 };
    sut::tool_data->control_ctx     = { 40 };

    for(const handle_t ctx :
        { handle_t{ 10 }, handle_t{ 20 }, handle_t{ 30 }, handle_t{ 40 } })
    {
        EXPECT_CALL(mock(), context_is_active(ctx, _))
            .WillOnce(DoAll(SetArgPointee<1>(0), Return(SUCCESS)));
        EXPECT_CALL(mock(), start_context(ctx)).WillOnce(Return(SUCCESS));
    }

    sut::start();
}

// ─── stop() tests ─────────────────────────────────────────────────────────────

// Active context: context_is_active confirms active, stop_context is called.
TEST_F(library_sdk_test, stop_calls_stop_context_for_initialized_active_context)
{
    const handle_t ctx          = { 42 };
    sut::tool_data->primary_ctx = ctx;

    EXPECT_CALL(mock(), context_is_active(ctx, _))
        .WillOnce(DoAll(SetArgPointee<1>(1), Return(SUCCESS)));
    EXPECT_CALL(mock(), stop_context(ctx)).WillOnce(Return(SUCCESS));

    sut::stop();
}

// Context is initialized but not active: stop_context must NOT be called.
TEST_F(library_sdk_test, stop_skips_stop_context_when_context_is_not_active)
{
    const handle_t ctx          = { 42 };
    sut::tool_data->primary_ctx = ctx;

    EXPECT_CALL(mock(), context_is_active(ctx, _))
        .WillOnce(DoAll(SetArgPointee<1>(0), Return(SUCCESS)));
    EXPECT_CALL(mock(), stop_context(_)).Times(0);

    sut::stop();
}

// All zero-handle contexts: no SDK call.
TEST_F(library_sdk_test, stop_makes_no_sdk_calls_when_all_contexts_are_uninitialized)
{
    EXPECT_CALL(mock(), context_is_active(_, _)).Times(0);
    EXPECT_CALL(mock(), stop_context(_)).Times(0);

    sut::stop();
}

// ─── pause() / resume() ───────────────────────────────────────────────────────
//
// Both methods call flush_counter_tracks_to_zero(0) which instantiates
// counter_storage<Wrapper>::write_zero() → cache_manager::get_instance()
// (defined in librocprof-sys-core.a(cache_manager.cpp.o)), which depends on
// rocprofsys::is_root_process() from rocprofiler-systems-object-library.
// That library is explicitly blocked from unit tests (TSan deadlock in
// library.cpp static initializers — see source/tests/CMakeLists.txt).
// Tests for pause()/resume() belong in integration tests that link the full
// library.

// ─── tool_init() tests ────────────────────────────────────────────────────────

// tool_init guards itself with tool_init_done.exchange(true): a second call
// returns 0 immediately without touching any SDK function.
TEST_F(library_sdk_test,
       tool_init_returns_zero_and_makes_no_sdk_calls_when_already_initialized)
{
    sut::tool_init_done.store(true);

    EXPECT_CALL(mock(), create_context(_)).Times(0);
    EXPECT_CALL(mock(), configure_callback_tracing_service(_, _, _, _, _, _)).Times(0);
    EXPECT_CALL(mock(), create_buffer(_, _, _, _, _, _, _)).Times(0);

    const int ret = sut::tool_init(nullptr, nullptr);

    EXPECT_EQ(ret, 0);
    EXPECT_TRUE(sut::tool_init_done.load());
}

// ─── reset_sdk_session_guards() tests ─────────────────────────────────────────

TEST_F(library_sdk_test, reset_sdk_session_guards_clears_both_done_flags)
{
    sut::tool_fini_done.store(true);
    sut::tool_init_done.store(true);

    sut::reset_sdk_session_guards();

    EXPECT_FALSE(sut::tool_fini_done.load());
    EXPECT_FALSE(sut::tool_init_done.load());
}
