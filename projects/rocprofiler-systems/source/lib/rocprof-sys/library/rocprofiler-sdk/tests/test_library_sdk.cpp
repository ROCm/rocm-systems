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

    // ─── Mock settings ────────────────────────────────────────────────────────────
    // Duck-typed stand-in for tim::settings used by tool_init and sdk_core.
    // Interface needed: settings->at(key)->get_choices().
    //
    // Two-class GMock pattern (GMock objects are non-copyable so they cannot be
    // held by value inside the returning shared_ptr):
    //   gmock_setting_entry  — EXPECT_CALL / ON_CALL target (global unique_ptr)
    //   mock_setting_entry   — thin shared_ptr-able wrapper that delegates

    struct gmock_setting_entry
    {
        MOCK_METHOD(const std::vector<std::string>&, get_choices, (), (const));
    };

    struct mock_setting_entry
    {
        const std::vector<std::string>& get_choices() const
        {
            return g_mock_setting->get_choices();
        }
    };

    struct mock_settings
    {
        std::shared_ptr<mock_setting_entry> at(const std::string& /*key*/) const
        {
            return std::make_shared<mock_setting_entry>();
        }
    };

    static auto* get_settings()
    {
        static mock_settings stub{};
        return &stub;
    }

    static inline std::unique_ptr<::testing::NiceMock<gmock_setting_entry>>
        g_mock_setting;

    // ─── Mock config methods ───────────────────────────────────────────────────────
    // Per-test control of string-returning config accessors used by sdk_core to
    // determine which domains are active.  Tests set ON_CALL expectations on
    // g_mock_config to drive specific domain combinations through tool_init.

    struct gmock_externals_config
    {
        MOCK_METHOD(std::string, get_rocm_domains, ());
        MOCK_METHOD(std::string, get_rocm_events_setting, ());
        MOCK_METHOD(std::string, get_gpu_perf_counters, ());
    };

    static inline std::unique_ptr<::testing::NiceMock<gmock_externals_config>>
        g_mock_config;

    static bool        get_use_perfetto() { return false; }
    static bool        get_use_timemory() { return false; }
    static bool        get_perfetto_annotations() { return false; }
    static bool        get_use_rocpd() { return false; }
    static bool        get_group_by_queue() { return false; }
    static bool        get_use_rcclp() { return false; }
    static bool        get_use_ompt() { return false; }
    static bool        get_use_unified_memory_profiling() { return false; }
    static bool        get_use_process_sampling() { return false; }
    static std::string get_trace_region() { return {}; }
    static std::string get_rocm_domains() { return g_mock_config->get_rocm_domains(); }
    static std::string get_rocm_events_setting()
    {
        return g_mock_config->get_rocm_events_setting();
    }
    static std::string get_gpu_perf_counters()
    {
        return g_mock_config->get_gpu_perf_counters();
    }

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

    void TearDown() override
    {
        mock_ns::g_mock_wrapper.reset();
        mock_externals::g_mock_setting.reset();
        mock_externals::g_mock_config.reset();
    }

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

// ─── tool_init() tests (extended) ────────────────────────────────────────────
//
// Fixture subclass that primes the three sdk_core helpers (get_version,
// get_callback_tracing_names, get_buffer_tracing_names) with AnyNumber()
// expectations.  Those helpers cache their results in function-local statics;
// how many times the mock is actually called depends on which test runs first,
// so we cannot pin the count.
//
// With no ROCM_DOMAINS setting configured, get_callback_domains() and
// get_buffered_domains() both return empty sets.  Therefore tool_init makes
// exactly three create_context calls (primary, code_object, control) plus
// one configure_callback_tracing_service call (CODE_OBJECT on code_object_ctx)
// and one configure_external_correlation_id_request_service call on primary_ctx.
// No buffers are created, so create_callback_thread / assign_callback_thread
// are never called.

class tool_init_test : public library_sdk_test
{
protected:
    void SetUp() override
    {
        library_sdk_test::SetUp();

        EXPECT_CALL(mock(), get_version(_, _, _))
            .Times(testing::AnyNumber())
            .WillRepeatedly(DoAll(SetArgPointee<0>(1U), SetArgPointee<1>(1U),
                                  SetArgPointee<2>(0U), Return(SUCCESS)));
        EXPECT_CALL(mock(), get_callback_tracing_names())
            .Times(testing::AnyNumber())
            .WillRepeatedly(Return(mock_backend_t::callback_name_info_t{}));
        EXPECT_CALL(mock(), get_buffer_tracing_names())
            .Times(testing::AnyNumber())
            .WillRepeatedly(Return(mock_backend_t::buffer_name_info_t{}));

        // Fresh mock for settings->at(key)->get_choices().
        // Default: empty choices — no ROCM_DOMAINS configured.
        mock_externals::g_mock_setting =
            std::make_unique<::testing::NiceMock<mock_externals::gmock_setting_entry>>();
        static const std::vector<std::string> empty_choices{};
        ON_CALL(*mock_externals::g_mock_setting, get_choices())
            .WillByDefault(::testing::ReturnRef(empty_choices));

        // Fresh mock for string config accessors.
        // Default: all empty — no domains, no events, no GPU perf counters.
        mock_externals::g_mock_config = std::make_unique<
            ::testing::NiceMock<mock_externals::gmock_externals_config>>();
        ON_CALL(*mock_externals::g_mock_config, get_rocm_domains())
            .WillByDefault(Return(std::string{}));
        ON_CALL(*mock_externals::g_mock_config, get_rocm_events_setting())
            .WillByDefault(Return(std::string{}));
        ON_CALL(*mock_externals::g_mock_config, get_gpu_perf_counters())
            .WillByDefault(Return(std::string{}));
    }

    // Expects create_context called 3 times in sequence; assigns distinct handles
    // 1/2/3 so callers can match on specific contexts.
    void expect_three_contexts()
    {
        EXPECT_CALL(mock(), create_context(_))
            .WillOnce(DoAll(SetArgPointee<0>(handle_t{ 1 }), Return(SUCCESS)))
            .WillOnce(DoAll(SetArgPointee<0>(handle_t{ 2 }), Return(SUCCESS)))
            .WillOnce(DoAll(SetArgPointee<0>(handle_t{ 3 }), Return(SUCCESS)));
    }

    void expect_code_object_tracing()
    {
        EXPECT_CALL(mock(), configure_callback_tracing_service(
                                handle_t{ 2 },
                                mock_backend_t::CALLBACK_TRACING_CODE_OBJECT, _, _, _, _))
            .WillOnce(Return(SUCCESS));
    }

    void expect_external_correlation()
    {
        EXPECT_CALL(mock(), configure_external_correlation_id_request_service(
                                handle_t{ 1 }, _, _, _, _))
            .WillOnce(Return(SUCCESS));
    }

    void expect_primary_ctx_valid(bool valid = true)
    {
        EXPECT_CALL(mock(), context_is_valid(handle_t{ 1 }, _))
            .WillOnce(DoAll(SetArgPointee<1>(valid ? 1 : 0), Return(SUCCESS)));
    }

    // start() iterates all four contexts; counter_ctx stays handle=0 so only
    // the three initialized ones (1, 2, 3) trigger context_is_active/start_context.
    void expect_start_for_initialized_contexts()
    {
        for(const auto h : { handle_t{ 1 }, handle_t{ 2 }, handle_t{ 3 } })
        {
            EXPECT_CALL(mock(), context_is_active(h, _))
                .WillOnce(DoAll(SetArgPointee<1>(0), Return(SUCCESS)));
            EXPECT_CALL(mock(), start_context(h)).WillOnce(Return(SUCCESS));
        }
    }

    void expect_full_tool_init_success()
    {
        expect_three_contexts();
        expect_code_object_tracing();
        expect_external_correlation();
        expect_primary_ctx_valid();
        expect_start_for_initialized_contexts();
    }
};

// Sets tool_init_done to true so subsequent calls short-circuit.
TEST_F(tool_init_test, sets_tool_init_done_to_true_after_first_call)
{
    expect_full_tool_init_success();

    ASSERT_FALSE(sut::tool_init_done.load());
    sut::tool_init(nullptr, sut::tool_data);
    EXPECT_TRUE(sut::tool_init_done.load());
}

// Three create_context calls happen in a specific order; the resulting handles
// are stored into primary_ctx, code_object_ctx, and control_ctx respectively.
// counter_ctx stays 0 because no counter events are configured.
TEST_F(tool_init_test, assigns_context_handles_to_tool_data_fields)
{
    expect_external_correlation();
    expect_primary_ctx_valid();
    expect_start_for_initialized_contexts();

    // Actual sequence in tool_init:
    //   create primary_ctx       (handle 1)
    //   create code_object_ctx   (handle 2)
    //   configure CODE_OBJECT tracing on code_object_ctx
    //   create control_ctx       (handle 3)
    {
        testing::InSequence seq;
        EXPECT_CALL(mock(), create_context(_))
            .WillOnce(DoAll(SetArgPointee<0>(handle_t{ 1 }), Return(SUCCESS)));
        EXPECT_CALL(mock(), create_context(_))
            .WillOnce(DoAll(SetArgPointee<0>(handle_t{ 2 }), Return(SUCCESS)));
        EXPECT_CALL(mock(), configure_callback_tracing_service(_, _, _, _, _, _))
            .WillOnce(Return(SUCCESS));
        EXPECT_CALL(mock(), create_context(_))
            .WillOnce(DoAll(SetArgPointee<0>(handle_t{ 3 }), Return(SUCCESS)));
    }

    sut::tool_init(nullptr, sut::tool_data);

    EXPECT_EQ(sut::tool_data->primary_ctx, handle_t{ 1 });
    EXPECT_EQ(sut::tool_data->code_object_ctx, handle_t{ 2 });
    EXPECT_EQ(sut::tool_data->control_ctx, handle_t{ 3 });
    EXPECT_EQ(sut::tool_data->counter_ctx, handle_t{ 0 });
}

// The code-object callback tracing service must be configured on code_object_ctx
// (the second context created) with the CODE_OBJECT tracing kind.
TEST_F(tool_init_test, configures_code_object_tracing_on_code_object_ctx)
{
    expect_three_contexts();
    EXPECT_CALL(mock(), configure_callback_tracing_service(
                            handle_t{ 2 }, mock_backend_t::CALLBACK_TRACING_CODE_OBJECT,
                            /*ops=*/nullptr, /*ops_count=*/0, _, _))
        .WillOnce(Return(SUCCESS));
    expect_external_correlation();
    expect_primary_ctx_valid();
    expect_start_for_initialized_contexts();

    sut::tool_init(nullptr, sut::tool_data);
}

// External correlation request service must be configured on primary_ctx.
TEST_F(tool_init_test, configures_external_correlation_service_on_primary_ctx)
{
    expect_three_contexts();
    expect_code_object_tracing();
    EXPECT_CALL(mock(), configure_external_correlation_id_request_service(handle_t{ 1 },
                                                                          _, _, _, _))
        .WillOnce(Return(SUCCESS));
    expect_primary_ctx_valid();
    expect_start_for_initialized_contexts();

    sut::tool_init(nullptr, sut::tool_data);
}

// If context_is_valid reports the primary context as invalid (status = 0),
// tool_init returns -1 and must not call start_context for any context.
TEST_F(tool_init_test, returns_minus_one_when_primary_ctx_is_invalid)
{
    expect_three_contexts();
    expect_code_object_tracing();
    expect_external_correlation();
    EXPECT_CALL(mock(), context_is_valid(handle_t{ 1 }, _))
        .WillOnce(DoAll(SetArgPointee<1>(0), Return(SUCCESS)));

    EXPECT_CALL(mock(), context_is_active(_, _)).Times(0);
    EXPECT_CALL(mock(), start_context(_)).Times(0);

    const int ret = sut::tool_init(nullptr, sut::tool_data);
    EXPECT_EQ(ret, -1);
}

// Happy path: valid primary context → returns 0.
TEST_F(tool_init_test, returns_zero_when_primary_ctx_is_valid)
{
    expect_full_tool_init_success();
    EXPECT_EQ(sut::tool_init(nullptr, sut::tool_data), 0);
}

// start() iterates {primary, counter, code_object, control}.  counter_ctx
// stays handle=0 (no counter events configured) so is_initialized returns
// false for it and no SDK calls are made on its behalf.
TEST_F(tool_init_test, starts_initialized_contexts_and_skips_uninitialized_counter_ctx)
{
    expect_three_contexts();
    expect_code_object_tracing();
    expect_external_correlation();
    expect_primary_ctx_valid();

    for(const auto h : { handle_t{ 1 }, handle_t{ 2 }, handle_t{ 3 } })
    {
        EXPECT_CALL(mock(), context_is_active(h, _))
            .WillOnce(DoAll(SetArgPointee<1>(0), Return(SUCCESS)));
        EXPECT_CALL(mock(), start_context(h)).WillOnce(Return(SUCCESS));
    }
    // counter_ctx has handle 0 — must never be touched by start().
    EXPECT_CALL(mock(), context_is_active(handle_t{ 0 }, _)).Times(0);
    EXPECT_CALL(mock(), start_context(handle_t{ 0 })).Times(0);

    sut::tool_init(nullptr, sut::tool_data);
}

// ─── tool_init() domain tests ─────────────────────────────────────────────────
//
// These tests drive specific domain strings through get_rocm_domains() /
// get_choices() and verify that tool_init calls the expected Wrapper methods.
// set_operation_options() is used to pre-register operation option entries for
// callback kinds so that sdk_core::get_operations() does not throw.

// When "memory_copy" is in the domain list, tool_init must create a buffer for
// memory-copy tracing and configure the buffer tracing service on primary_ctx.
TEST_F(tool_init_test, creates_memory_copy_buffer_when_memory_copy_domain_active)
{
    // Activate the "memory_copy" buffered domain
    static const std::vector<std::string> choices{ "memory_copy" };
    ON_CALL(*mock_externals::g_mock_setting, get_choices())
        .WillByDefault(::testing::ReturnRef(choices));
    ON_CALL(*mock_externals::g_mock_config, get_rocm_domains())
        .WillByDefault(Return(std::string{ "memory_copy" }));

    expect_three_contexts();
    expect_code_object_tracing();
    expect_external_correlation();

    // BUFFER_TRACING_MEMORY_COPY triggers the HIP_STREAM callback registration
    // (compile_time_version >= 700 guard in tool_init).
    EXPECT_CALL(mock(), configure_callback_tracing_service(
                            handle_t{ 1 }, mock_backend_t::CALLBACK_TRACING_HIP_STREAM,
                            nullptr, 0, _, _))
        .WillOnce(Return(SUCCESS));

    // create_buffer called for MEMORY_COPY — assign a non-zero handle so the
    // buffer loop triggers create_callback_thread / assign_callback_thread.
    const handle_t mem_buf{ 99 };
    EXPECT_CALL(mock(), create_buffer(handle_t{ 1 }, _, _, _, _, _, _))
        .WillOnce(DoAll(SetArgPointee<6>(mem_buf), Return(SUCCESS)));
    EXPECT_CALL(mock(), configure_buffer_tracing_service(
                            handle_t{ 1 }, mock_backend_t::BUFFER_TRACING_MEMORY_COPY,
                            nullptr, 0, mem_buf))
        .WillOnce(Return(SUCCESS));

    const handle_t cb_thread{ 7 };
    EXPECT_CALL(mock(), create_callback_thread(_))
        .WillOnce(DoAll(SetArgPointee<0>(cb_thread), Return(SUCCESS)));
    EXPECT_CALL(mock(), assign_callback_thread(mem_buf, cb_thread))
        .WillOnce(Return(SUCCESS));

    expect_primary_ctx_valid();
    expect_start_for_initialized_contexts();

    EXPECT_EQ(sut::tool_init(nullptr, sut::tool_data), 0);
    EXPECT_EQ(sut::tool_data->memory_copy_buffer, mem_buf);
}

// When "hip_api" is in the domain list, tool_init must register callback tracing
// for HIP_RUNTIME_API and HIP_COMPILER_API on primary_ctx.
TEST_F(tool_init_test, configures_hip_api_callback_tracing_when_hip_api_domain_active)
{
    using sut_core =
        ::rocprofsys::rocprofiler_sdk::sdk_core<mock_backend_t, mock_externals>;

    // Pre-register operation options so sdk_core::get_operations() does not throw.
    sut_core::set_operation_options(mock_backend_t::CALLBACK_TRACING_HIP_RUNTIME_API, "",
                                    "", "");
    sut_core::set_operation_options(mock_backend_t::CALLBACK_TRACING_HIP_COMPILER_API, "",
                                    "", "");

    static const std::vector<std::string> choices{ "hip_api" };
    ON_CALL(*mock_externals::g_mock_setting, get_choices())
        .WillByDefault(::testing::ReturnRef(choices));
    ON_CALL(*mock_externals::g_mock_config, get_rocm_domains())
        .WillByDefault(Return(std::string{ "hip_api" }));

    expect_three_contexts();
    expect_code_object_tracing();
    expect_external_correlation();

    // Both HIP kinds must be registered on primary_ctx.
    EXPECT_CALL(mock(), configure_callback_tracing_service(
                            handle_t{ 1 },
                            mock_backend_t::CALLBACK_TRACING_HIP_RUNTIME_API, _, _, _, _))
        .WillOnce(Return(SUCCESS));
    EXPECT_CALL(
        mock(),
        configure_callback_tracing_service(
            handle_t{ 1 }, mock_backend_t::CALLBACK_TRACING_HIP_COMPILER_API, _, _, _, _))
        .WillOnce(Return(SUCCESS));

    expect_primary_ctx_valid();
    expect_start_for_initialized_contexts();

    EXPECT_EQ(sut::tool_init(nullptr, sut::tool_data), 0);
}

// ─── tool_init() — additional coverage ───────────────────────────────────────

// When a non-null fini_func is supplied, tool_init must store it in client_fini.
TEST_F(tool_init_test, stores_client_fini_when_fini_func_is_non_null)
{
    expect_full_tool_init_success();

    auto fini_stub = [](mock_backend_t::client_id_t*, void*) {};

    sut::tool_init(fini_stub, sut::tool_data);

    EXPECT_EQ(sut::tool_data->client_fini, fini_stub);
}

// When "scratch_memory" is active, tool_init creates a buffer and configures
// BUFFER_TRACING_SCRATCH_MEMORY on primary_ctx.
// No HIP_STREAM registration: that guard only fires for KERNEL_DISPATCH or MEMORY_COPY.
//
// "scratch_memory" is not a special-cased domain in get_buffered_domains() — it falls
// through to the name-lookup path.  Override get_buffer_tracing_names() to populate
// the entry at the SCRATCH_MEMORY kind value so the lookup succeeds.
TEST_F(tool_init_test, creates_scratch_memory_buffer_when_scratch_memory_domain_active)
{
    static const std::vector<std::string> choices{ "scratch_memory" };
    ON_CALL(*mock_externals::g_mock_setting, get_choices())
        .WillByDefault(::testing::ReturnRef(choices));
    ON_CALL(*mock_externals::g_mock_config, get_rocm_domains())
        .WillByDefault(Return(std::string{ "scratch_memory" }));

    // Provide name entry so get_buffered_domains() fallback finds SCRATCH_MEMORY.
    mock_backend_t::buffer_name_info_t buf_names;
    buf_names[mock_backend_t::BUFFER_TRACING_SCRATCH_MEMORY.value].name =
        "scratch_memory";
    EXPECT_CALL(mock(), get_buffer_tracing_names())
        .Times(testing::AnyNumber())
        .WillRepeatedly(Return(buf_names));

    expect_three_contexts();
    expect_code_object_tracing();
    expect_external_correlation();

    const handle_t scratch_buf{ 55 };
    EXPECT_CALL(mock(), create_buffer(handle_t{ 1 }, _, _, _, _, _, _))
        .WillOnce(DoAll(SetArgPointee<6>(scratch_buf), Return(SUCCESS)));
    EXPECT_CALL(mock(), configure_buffer_tracing_service(
                            handle_t{ 1 }, mock_backend_t::BUFFER_TRACING_SCRATCH_MEMORY,
                            nullptr, 0, scratch_buf))
        .WillOnce(Return(SUCCESS));

    const handle_t cb_thread{ 8 };
    EXPECT_CALL(mock(), create_callback_thread(_))
        .WillOnce(DoAll(SetArgPointee<0>(cb_thread), Return(SUCCESS)));
    EXPECT_CALL(mock(), assign_callback_thread(scratch_buf, cb_thread))
        .WillOnce(Return(SUCCESS));

    expect_primary_ctx_valid();
    expect_start_for_initialized_contexts();

    EXPECT_EQ(sut::tool_init(nullptr, sut::tool_data), 0);
    EXPECT_EQ(sut::tool_data->scratch_memory_buffer, scratch_buf);
}

// When "kernel_dispatch" is in the buffered domain, tool_init creates a buffer for
// kernel dispatch tracing. Because compile_time_version >= 700, tool_init also
// registers CALLBACK_TRACING_HIP_STREAM on primary_ctx.
//
// "kernel_dispatch" falls through to the name-lookup path in get_buffered_domains(),
// so get_buffer_tracing_names() must expose the KERNEL_DISPATCH name entry.
TEST_F(tool_init_test, creates_kernel_dispatch_buffer_when_kernel_dispatch_domain_active)
{
    static const std::vector<std::string> choices{ "kernel_dispatch" };
    ON_CALL(*mock_externals::g_mock_setting, get_choices())
        .WillByDefault(::testing::ReturnRef(choices));
    ON_CALL(*mock_externals::g_mock_config, get_rocm_domains())
        .WillByDefault(Return(std::string{ "kernel_dispatch" }));

    mock_backend_t::buffer_name_info_t buf_names;
    buf_names[mock_backend_t::BUFFER_TRACING_KERNEL_DISPATCH.value].name =
        "kernel_dispatch";
    EXPECT_CALL(mock(), get_buffer_tracing_names())
        .Times(testing::AnyNumber())
        .WillRepeatedly(Return(buf_names));

    expect_three_contexts();
    expect_code_object_tracing();
    expect_external_correlation();

    // BUFFER_TRACING_KERNEL_DISPATCH triggers HIP_STREAM callback registration.
    EXPECT_CALL(mock(), configure_callback_tracing_service(
                            handle_t{ 1 }, mock_backend_t::CALLBACK_TRACING_HIP_STREAM,
                            nullptr, 0, _, _))
        .WillOnce(Return(SUCCESS));

    const handle_t kd_buf{ 66 };
    EXPECT_CALL(mock(), create_buffer(handle_t{ 1 }, _, _, _, _, _, _))
        .WillOnce(DoAll(SetArgPointee<6>(kd_buf), Return(SUCCESS)));
    EXPECT_CALL(mock(), configure_buffer_tracing_service(
                            handle_t{ 1 }, mock_backend_t::BUFFER_TRACING_KERNEL_DISPATCH,
                            nullptr, 0, kd_buf))
        .WillOnce(Return(SUCCESS));

    const handle_t cb_thread{ 9 };
    EXPECT_CALL(mock(), create_callback_thread(_))
        .WillOnce(DoAll(SetArgPointee<0>(cb_thread), Return(SUCCESS)));
    EXPECT_CALL(mock(), assign_callback_thread(kd_buf, cb_thread))
        .WillOnce(Return(SUCCESS));

    expect_primary_ctx_valid();
    expect_start_for_initialized_contexts();

    EXPECT_EQ(sut::tool_init(nullptr, sut::tool_data), 0);
    EXPECT_EQ(sut::tool_data->kernel_dispatch_buffer, kd_buf);
}

// When "memory_allocation" is in the buffered domain, tool_init creates a buffer
// and configures BUFFER_TRACING_MEMORY_ALLOCATION (SDK >= 600, always true for
// mock's compile_time_version 10301). The buffer handle must be non-zero: tool_init
// aborts if create_buffer returns handle=0 for this domain.
TEST_F(tool_init_test,
       creates_memory_allocation_buffer_when_memory_allocation_domain_active)
{
    static const std::vector<std::string> choices{ "memory_allocation" };
    ON_CALL(*mock_externals::g_mock_setting, get_choices())
        .WillByDefault(::testing::ReturnRef(choices));
    ON_CALL(*mock_externals::g_mock_config, get_rocm_domains())
        .WillByDefault(Return(std::string{ "memory_allocation" }));

    expect_three_contexts();
    expect_code_object_tracing();
    expect_external_correlation();

    const handle_t alloc_buf{ 77 };
    EXPECT_CALL(mock(), create_buffer(handle_t{ 1 }, _, _, _, _, _, _))
        .WillOnce(DoAll(SetArgPointee<6>(alloc_buf), Return(SUCCESS)));
    EXPECT_CALL(mock(),
                configure_buffer_tracing_service(
                    handle_t{ 1 }, mock_backend_t::BUFFER_TRACING_MEMORY_ALLOCATION,
                    nullptr, 0, alloc_buf))
        .WillOnce(Return(SUCCESS));

    const handle_t cb_thread{ 10 };
    EXPECT_CALL(mock(), create_callback_thread(_))
        .WillOnce(DoAll(SetArgPointee<0>(cb_thread), Return(SUCCESS)));
    EXPECT_CALL(mock(), assign_callback_thread(alloc_buf, cb_thread))
        .WillOnce(Return(SUCCESS));

    expect_primary_ctx_valid();
    expect_start_for_initialized_contexts();

    EXPECT_EQ(sut::tool_init(nullptr, sut::tool_data), 0);
    EXPECT_EQ(sut::tool_data->memory_alloc_buffer, alloc_buf);
}

// When "hsa_api" is in the domain list, tool_init must configure callback tracing
// for all four HSA kinds (CORE, AMD_EXT, IMAGE_EXT, FINALIZE_EXT) on primary_ctx.
TEST_F(tool_init_test, configures_hsa_api_callback_tracing_for_all_four_hsa_kinds)
{
    using sut_core =
        ::rocprofsys::rocprofiler_sdk::sdk_core<mock_backend_t, mock_externals>;

    for(auto kind : { mock_backend_t::CALLBACK_TRACING_HSA_CORE_API,
                      mock_backend_t::CALLBACK_TRACING_HSA_AMD_EXT_API,
                      mock_backend_t::CALLBACK_TRACING_HSA_IMAGE_EXT_API,
                      mock_backend_t::CALLBACK_TRACING_HSA_FINALIZE_EXT_API })
        sut_core::set_operation_options(kind, "", "", "");

    static const std::vector<std::string> choices{ "hsa_api" };
    ON_CALL(*mock_externals::g_mock_setting, get_choices())
        .WillByDefault(::testing::ReturnRef(choices));
    ON_CALL(*mock_externals::g_mock_config, get_rocm_domains())
        .WillByDefault(Return(std::string{ "hsa_api" }));

    expect_three_contexts();
    expect_code_object_tracing();
    expect_external_correlation();

    for(auto kind : { mock_backend_t::CALLBACK_TRACING_HSA_CORE_API,
                      mock_backend_t::CALLBACK_TRACING_HSA_AMD_EXT_API,
                      mock_backend_t::CALLBACK_TRACING_HSA_IMAGE_EXT_API,
                      mock_backend_t::CALLBACK_TRACING_HSA_FINALIZE_EXT_API })
    {
        EXPECT_CALL(mock(),
                    configure_callback_tracing_service(handle_t{ 1 }, kind, _, _, _, _))
            .WillOnce(Return(SUCCESS));
    }

    expect_primary_ctx_valid();
    expect_start_for_initialized_contexts();

    EXPECT_EQ(sut::tool_init(nullptr, sut::tool_data), 0);
}

// When two buffers are active (memory_copy + scratch_memory), tool_init must call
// create_callback_thread / assign_callback_thread once per non-zero buffer handle.
// memory_copy is special-cased in get_buffered_domains(); scratch_memory requires a
// name entry in get_buffer_tracing_names() so the fallback lookup succeeds.
TEST_F(tool_init_test, creates_two_callback_thread_pairs_for_two_active_buffers)
{
    static const std::vector<std::string> choices{ "memory_copy", "scratch_memory" };
    ON_CALL(*mock_externals::g_mock_setting, get_choices())
        .WillByDefault(::testing::ReturnRef(choices));
    ON_CALL(*mock_externals::g_mock_config, get_rocm_domains())
        .WillByDefault(Return(std::string{ "memory_copy scratch_memory" }));

    mock_backend_t::buffer_name_info_t buf_names;
    buf_names[mock_backend_t::BUFFER_TRACING_SCRATCH_MEMORY.value].name =
        "scratch_memory";
    EXPECT_CALL(mock(), get_buffer_tracing_names())
        .Times(testing::AnyNumber())
        .WillRepeatedly(Return(buf_names));

    expect_three_contexts();
    expect_code_object_tracing();
    expect_external_correlation();

    // memory_copy activates HIP_STREAM (compile_time_version >= 700).
    EXPECT_CALL(mock(), configure_callback_tracing_service(
                            handle_t{ 1 }, mock_backend_t::CALLBACK_TRACING_HIP_STREAM,
                            nullptr, 0, _, _))
        .WillOnce(Return(SUCCESS));

    // tool_init order: memory_copy buffer first, scratch_memory buffer second.
    const handle_t mem_buf{ 20 };
    const handle_t scratch_buf{ 21 };
    EXPECT_CALL(mock(), create_buffer(handle_t{ 1 }, _, _, _, _, _, _))
        .WillOnce(DoAll(SetArgPointee<6>(mem_buf), Return(SUCCESS)))
        .WillOnce(DoAll(SetArgPointee<6>(scratch_buf), Return(SUCCESS)));
    EXPECT_CALL(mock(), configure_buffer_tracing_service(
                            handle_t{ 1 }, mock_backend_t::BUFFER_TRACING_MEMORY_COPY,
                            nullptr, 0, mem_buf))
        .WillOnce(Return(SUCCESS));
    EXPECT_CALL(mock(), configure_buffer_tracing_service(
                            handle_t{ 1 }, mock_backend_t::BUFFER_TRACING_SCRATCH_MEMORY,
                            nullptr, 0, scratch_buf))
        .WillOnce(Return(SUCCESS));

    // Two non-zero buffer handles → two thread creation/assignment pairs.
    // get_buffers() returns {scratch_memory, memory_copy, ...} so scratch_memory is
    // processed first; use wildcards to avoid coupling to that internal ordering.
    EXPECT_CALL(mock(), create_callback_thread(_))
        .Times(2)
        .WillRepeatedly(DoAll(SetArgPointee<0>(handle_t{ 30 }), Return(SUCCESS)));
    EXPECT_CALL(mock(), assign_callback_thread(_, _))
        .Times(2)
        .WillRepeatedly(Return(SUCCESS));

    expect_primary_ctx_valid();
    expect_start_for_initialized_contexts();

    EXPECT_EQ(sut::tool_init(nullptr, sut::tool_data), 0);
}

// When get_rocm_events_setting() returns a non-empty event list, tool_init creates
// a counter_ctx (4th context), configures KERNEL_DISPATCH callback tracing on it,
// and registers the dispatch counting service. start() then also activates counter_ctx.
//
// initialize_event_info() queries real GPU agents present on the test host via
// iterate_agent_supported_counters — allow those calls with AnyNumber().
TEST_F(tool_init_test, creates_counter_ctx_when_rocm_events_configured)
{
    ON_CALL(*mock_externals::g_mock_config, get_rocm_events_setting())
        .WillByDefault(Return(std::string{ "SQ_CYCLES" }));

    // Four create_context calls in order: primary(1), code_object(2), control(3),
    // counter(4). Chained WillOnce on a single EXPECT_CALL delivers handles in order.
    EXPECT_CALL(mock(), create_context(_))
        .WillOnce(DoAll(SetArgPointee<0>(handle_t{ 1 }), Return(SUCCESS)))
        .WillOnce(DoAll(SetArgPointee<0>(handle_t{ 2 }), Return(SUCCESS)))
        .WillOnce(DoAll(SetArgPointee<0>(handle_t{ 3 }), Return(SUCCESS)))
        .WillOnce(DoAll(SetArgPointee<0>(handle_t{ 4 }), Return(SUCCESS)));

    expect_code_object_tracing();
    expect_external_correlation();

    // KERNEL_DISPATCH callback tracing and dispatch counting service on counter_ctx.
    EXPECT_CALL(mock(), configure_callback_tracing_service(
                            handle_t{ 4 },
                            mock_backend_t::CALLBACK_TRACING_KERNEL_DISPATCH, _, _, _, _))
        .WillOnce(Return(SUCCESS));
    EXPECT_CALL(mock(),
                configure_callback_dispatch_counting_service(handle_t{ 4 }, _, _, _, _))
        .WillOnce(Return(SUCCESS));

    // initialize_event_info() iterates real GPU agents on this host.
    EXPECT_CALL(mock(), iterate_agent_supported_counters(_, _, _))
        .Times(testing::AnyNumber())
        .WillRepeatedly(Return(SUCCESS));

    expect_primary_ctx_valid();

    // All four initialized contexts are started (counter_ctx=4 is now non-zero).
    for(const auto h : { handle_t{ 1 }, handle_t{ 2 }, handle_t{ 3 }, handle_t{ 4 } })
    {
        EXPECT_CALL(mock(), context_is_active(h, _))
            .WillOnce(DoAll(SetArgPointee<1>(0), Return(SUCCESS)));
        EXPECT_CALL(mock(), start_context(h)).WillOnce(Return(SUCCESS));
    }

    EXPECT_EQ(sut::tool_init(nullptr, sut::tool_data), 0);
    EXPECT_EQ(sut::tool_data->counter_ctx, handle_t{ 4 });
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
