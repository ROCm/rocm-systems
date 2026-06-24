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

// ─── mock_unwind::stack<Depth> ────────────────────────────────────────────────
//
// Minimal stand-in for ::tim::unwind::stack<Depth> used by mock_externals.
// Satisfies the interface consumed by populate_backtrace_data():
//   .size(), begin()/end() (range-for), element bool-conversion, element .address().
// Always empty so the inner loop never executes — no timemory dependency in tests.

namespace mock_unwind
{
template <std::size_t Depth>
struct stack
{
    struct entry
    {
        // Always false so the if(itr) body in populate_backtrace_data never runs.
        explicit operator bool() const { return false; }
        // operator-> needed: production code calls itr->address() (dead branch).
        const entry*   operator->() const { return this; }
        std::uintptr_t address() const { return 0; }
    };

    using value_type     = entry;
    using const_iterator = const entry*;

    static constexpr std::size_t size() noexcept { return 0; }
    const_iterator               begin() const noexcept { return nullptr; }
    const_iterator               end() const noexcept { return nullptr; }
};
}  // namespace mock_unwind

// ─── gmock_externals / mock_externals ────────────────────────────────────────
//
// Unified policy-based-DI pattern (see programming-cpp-policy-based-di skill):
//
//   gmock_externals  — GMock class; every non-template Externals method has a
//                      MOCK_METHOD.  EXPECT_CALL / ON_CALL targets live here.
//   g_mock_externals — global unique_ptr<NiceMock<gmock_externals>>; one per
//                      test, created in library_sdk_test::SetUp().
//   mock_externals   — static-method policy struct (Externals template param);
//                      every non-template method delegates to g_mock_externals.
//
// Template perfetto/timemory methods stay as no-ops: they are only reachable
// when get_use_perfetto() / get_use_timemory() return true, and those booleans
// are now fully mocked.

struct gmock_externals
{
    // ─── settings (get_choices returns const& → needs explicit ReturnRef default)
    MOCK_METHOD(const std::vector<std::string>&, get_choices, (), (const));

    // ─── boolean config
    MOCK_METHOD(bool, get_use_perfetto, ());
    MOCK_METHOD(bool, get_use_timemory, ());
    MOCK_METHOD(bool, get_perfetto_annotations, ());
    MOCK_METHOD(bool, get_use_rocpd, ());
    MOCK_METHOD(bool, get_group_by_queue, ());
    MOCK_METHOD(bool, get_use_rcclp, ());
    MOCK_METHOD(bool, get_use_ompt, ());
    MOCK_METHOD(bool, get_use_unified_memory_profiling, ());
    MOCK_METHOD(bool, get_use_process_sampling, ());

    // ─── string config
    MOCK_METHOD(std::string, get_trace_region, ());
    MOCK_METHOD(std::string, get_rocm_domains, ());
    MOCK_METHOD(std::string, get_rocm_events_setting, ());
    MOCK_METHOD(std::string, get_gpu_perf_counters, ());

    // ─── side-effecting / PMC
    MOCK_METHOD(void, set_pmc_state, (::rocprofsys::State));
    MOCK_METHOD(void, gpu_add_device_metadata, ());
    MOCK_METHOD(void, register_gpu_perf_counter_source,
                (const std::vector<std::shared_ptr<::rocprofsys::agent>>&) );

    // Returns mock_unwind::stack<16> — our own type, no timemory dependency.
    // Depth 16 is the compile-time constant used by tool_tracing_callback.
    using unwind_stack_16_t = mock_unwind::stack<16>;
    MOCK_METHOD(unwind_stack_16_t, tim_get_unw_stack, ());
};

inline std::unique_ptr<::testing::StrictMock<gmock_externals>> g_mock_externals;

// ─── mock_externals ───────────────────────────────────────────────────────────

struct mock_externals
{
    // ─── Stub types (kept for library_sdk template instantiation) ─────────────

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

    // ─── settings (get_choices delegates through a thin wrapper) ──────────────

    struct mock_settings
    {
        struct mock_entry
        {
            const std::vector<std::string>& get_choices() const
            {
                return g_mock_externals->get_choices();
            }
        };
        std::shared_ptr<mock_entry> at(const std::string& /*key*/) const
        {
            return std::make_shared<mock_entry>();
        }
    };
    static auto* get_settings()
    {
        static mock_settings s{};
        return &s;
    }

    // ─── boolean config → delegate ────────────────────────────────────────────
    static bool get_use_perfetto() { return g_mock_externals->get_use_perfetto(); }
    static bool get_use_timemory() { return g_mock_externals->get_use_timemory(); }
    static bool get_perfetto_annotations()
    {
        return g_mock_externals->get_perfetto_annotations();
    }
    static bool get_use_rocpd() { return g_mock_externals->get_use_rocpd(); }
    static bool get_group_by_queue() { return g_mock_externals->get_group_by_queue(); }
    static bool get_use_rcclp() { return g_mock_externals->get_use_rcclp(); }
    static bool get_use_ompt() { return g_mock_externals->get_use_ompt(); }
    static bool get_use_unified_memory_profiling()
    {
        return g_mock_externals->get_use_unified_memory_profiling();
    }
    static bool get_use_process_sampling()
    {
        return g_mock_externals->get_use_process_sampling();
    }

    // ─── string config → delegate ─────────────────────────────────────────────
    static std::string get_trace_region() { return g_mock_externals->get_trace_region(); }
    static std::string get_rocm_domains() { return g_mock_externals->get_rocm_domains(); }
    static std::string get_rocm_events_setting()
    {
        return g_mock_externals->get_rocm_events_setting();
    }
    static std::string get_gpu_perf_counters()
    {
        return g_mock_externals->get_gpu_perf_counters();
    }

    // ─── template perfetto / timemory (no-ops; only reachable when flags true) ─
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

    // ─── MarkerWriterPolicy no-ops ─────────────────────────────────────────────
    static void add_string(std::string_view) {}
    static void store_region(const ::rocprofsys::trace_cache::region_sample&) {}
    static void add_thread_info(const ::rocprofsys::trace_cache::info::thread&) {}

    // ─── PMC / state → delegate ───────────────────────────────────────────────
    static void register_gpu_perf_counter_source(
        const std::vector<std::shared_ptr<::rocprofsys::agent>>& agents)
    {
        g_mock_externals->register_gpu_perf_counter_source(agents);
    }
    static void set_pmc_state(::rocprofsys::State s)
    {
        g_mock_externals->set_pmc_state(s);
    }
    static void gpu_add_device_metadata() { g_mock_externals->gpu_add_device_metadata(); }

    // ─── backtrace unwind → mock_unwind::stack (no timemory dependency) ──────────
    // Own type alias — independent of ::tim::unwind::stack<Depth>.
    // Depth==16 delegates to the mock so tests can EXPECT_CALL / Return.
    // Any other depth returns an empty mock_unwind::stack directly.
    template <std::size_t Depth>
    using unwind_stack_t = mock_unwind::stack<Depth>;

    template <std::size_t Depth, std::int64_t Offset = 1, bool WSignalFrame = false>
    static unwind_stack_t<Depth> tim_get_unw_stack()
    {
        if constexpr(Depth == 16)
            return g_mock_externals->tim_get_unw_stack();
        else
            return {};
    }
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

        // StrictMock: every call to g_mock_externals must have an EXPECT_CALL —
        // any unmatched call is an immediate test failure.  For methods called in
        // the base fixtures, use Times(AnyNumber()) so individual tests can add
        // tighter EXPECT_CALLs that override these defaults via GMock's LIFO ordering.
        // get_choices() returns const& and needs ReturnRef to a persistent object.
        g_mock_externals = std::make_unique<::testing::StrictMock<gmock_externals>>();
        static const std::vector<std::string> s_empty_choices{};
        EXPECT_CALL(*g_mock_externals, get_choices())
            .WillRepeatedly(::testing::ReturnRef(s_empty_choices));

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
        g_mock_externals.reset();
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

        // Reset the sdk_core::get_version() static cache so Wrapper::get_version()
        // is called exactly once per test (from get_callback_domains()).
        // get_callback_tracing_names(): 1x from client_data::initialize() +
        //                               1x from sdk_core::get_callback_domains() = 2
        // get_buffer_tracing_names():   1x from client_data::initialize() +
        //   get_callback_tracing_names: 1x from client_data::initialize() +
        //                               1x from sdk_core::get_callback_domains()
        //                             + 1x from sdk_core::get_operations_impl()
        //                               (only when a callback domain is active)
        //   get_buffer_tracing_names:   1x from client_data::initialize() +
        //                               1x from sdk_core::get_buffered_domains() = always
        //                               2
        //
        // Both ops-impl caches (callback + buffer) are reset so the next call
        // to get_operations_impl() always re-queries Wrapper, making Times(N) exact.
        using sut_core_t =
            ::rocprofsys::rocprofiler_sdk::sdk_core<mock_backend_t, mock_externals>;
        sut_core_t::reset_version_cache();
        sut_core_t::reset_tracing_names_cache();

        EXPECT_CALL(mock(), get_version(_, _, _))
            .Times(1)
            .WillOnce(DoAll(SetArgPointee<0>(1U), SetArgPointee<1>(1U),
                            SetArgPointee<2>(0U), Return(SUCCESS)));
        // Tracing-name counts are set per-test via expect_buffer_tracing_names()
        // and expect_callback_tracing_names().  Counter-event tests call
        // initialize_event_info() which triggers a second initialize() call on
        // GPU-less machines, adding a 3rd get_buffer_tracing_names() call.
        // Tests with buffer-domain overrides supply their own buf_names.

        // StrictMock defaults for every Externals method called by tool_init or
        // sdk_core helpers (get_callback_domains, get_buffered_domains, get_rocm_events).
        // Times(AnyNumber()) lets tests add tighter EXPECT_CALLs that take precedence.
        EXPECT_CALL(*g_mock_externals, get_rocm_domains())
            .WillRepeatedly(Return(std::string{}));
        EXPECT_CALL(*g_mock_externals, get_rocm_events_setting())
            .WillRepeatedly(Return(std::string{}));
        EXPECT_CALL(*g_mock_externals, get_gpu_perf_counters())
            .WillRepeatedly(Return(std::string{}));
        EXPECT_CALL(*g_mock_externals, get_trace_region())
            .WillRepeatedly(Return(std::string{}));
        EXPECT_CALL(*g_mock_externals, get_use_rcclp()).WillRepeatedly(Return(false));
        EXPECT_CALL(*g_mock_externals, get_use_ompt()).WillRepeatedly(Return(false));
        EXPECT_CALL(*g_mock_externals, get_use_unified_memory_profiling())
            .WillRepeatedly(Return(false));
        EXPECT_CALL(*g_mock_externals, get_use_process_sampling())
            .WillRepeatedly(Return(false));
        EXPECT_CALL(*g_mock_externals, gpu_add_device_metadata())
            .WillRepeatedly(::testing::Return());
        EXPECT_CALL(*g_mock_externals, register_gpu_perf_counter_source(_))
            .WillRepeatedly(::testing::Return());
    }

    // Pin the exact number of get_callback_tracing_names() calls.
    //   n=2: no active callback domain  (initialize + get_callback_domains)
    //   n=3: an active callback domain  (+ one call from get_operations_impl)
    void expect_callback_tracing_names(int n = 2)
    {
        EXPECT_CALL(mock(), get_callback_tracing_names())
            .Times(n)
            .WillRepeatedly(Return(mock_backend_t::callback_name_info_t{}));
    }

    // Pin the exact number of get_buffer_tracing_names() calls.
    //   n=2: no counter events          (initialize + get_buffered_domains)
    //   n=3: counter events configured  (initialize_event_info calls initialize again
    //                                    on GPU-less machines → 3rd call)
    // Overloaded variant accepts specific buf_names for domain-fallback tests.
    void expect_buffer_tracing_names(int n = 2)
    {
        EXPECT_CALL(mock(), get_buffer_tracing_names())
            .Times(n)
            .WillRepeatedly(Return(mock_backend_t::buffer_name_info_t{}));
    }

    void expect_buffer_tracing_names(int                                       n,
                                     const mock_backend_t::buffer_name_info_t& names)
    {
        EXPECT_CALL(mock(), get_buffer_tracing_names())
            .Times(n)
            .WillRepeatedly(Return(names));
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
        expect_callback_tracing_names(2);
        expect_buffer_tracing_names(2);
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
    expect_callback_tracing_names(2);
    expect_buffer_tracing_names(2);
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
    expect_callback_tracing_names();
    expect_buffer_tracing_names();
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
    expect_callback_tracing_names();
    expect_buffer_tracing_names();
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
    expect_callback_tracing_names();
    expect_buffer_tracing_names();
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
    expect_callback_tracing_names();
    expect_buffer_tracing_names();
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
    EXPECT_CALL(*g_mock_externals, get_choices())
        .WillRepeatedly(::testing::ReturnRef(choices));
    EXPECT_CALL(*g_mock_externals, get_rocm_domains())
        .WillRepeatedly(Return(std::string{ "memory_copy" }));

    expect_callback_tracing_names();
    expect_buffer_tracing_names();
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
    EXPECT_CALL(*g_mock_externals, get_choices())
        .WillRepeatedly(::testing::ReturnRef(choices));
    EXPECT_CALL(*g_mock_externals, get_rocm_domains())
        .WillRepeatedly(Return(std::string{ "hip_api" }));

    // Active callback domain: get_operations_impl cold → 3 callback calls.
    // hip_api has no counter events → no extra initialize() call → 2 buffer calls.
    expect_callback_tracing_names(3);
    expect_buffer_tracing_names(2);
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
    EXPECT_CALL(*g_mock_externals, get_choices())
        .WillRepeatedly(::testing::ReturnRef(choices));
    EXPECT_CALL(*g_mock_externals, get_rocm_domains())
        .WillRepeatedly(Return(std::string{ "scratch_memory" }));

    // Provide name entry so get_buffered_domains() fallback finds SCRATCH_MEMORY.
    mock_backend_t::buffer_name_info_t buf_names;
    buf_names[mock_backend_t::BUFFER_TRACING_SCRATCH_MEMORY].name = "scratch_memory";

    expect_callback_tracing_names();
    expect_buffer_tracing_names(2, buf_names);
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
    EXPECT_CALL(*g_mock_externals, get_choices())
        .WillRepeatedly(::testing::ReturnRef(choices));
    EXPECT_CALL(*g_mock_externals, get_rocm_domains())
        .WillRepeatedly(Return(std::string{ "kernel_dispatch" }));

    mock_backend_t::buffer_name_info_t buf_names;
    buf_names[mock_backend_t::BUFFER_TRACING_KERNEL_DISPATCH].name = "kernel_dispatch";

    expect_callback_tracing_names();
    expect_buffer_tracing_names(2, buf_names);
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
    EXPECT_CALL(*g_mock_externals, get_choices())
        .WillRepeatedly(::testing::ReturnRef(choices));
    EXPECT_CALL(*g_mock_externals, get_rocm_domains())
        .WillRepeatedly(Return(std::string{ "memory_allocation" }));

    expect_callback_tracing_names();
    expect_buffer_tracing_names();
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
    EXPECT_CALL(*g_mock_externals, get_choices())
        .WillRepeatedly(::testing::ReturnRef(choices));
    EXPECT_CALL(*g_mock_externals, get_rocm_domains())
        .WillRepeatedly(Return(std::string{ "hsa_api" }));

    // Active callback domain: get_operations_impl cold → 3 callback calls.
    // hsa_api has no counter events → no extra initialize() call → 2 buffer calls.
    expect_callback_tracing_names(3);
    expect_buffer_tracing_names(2);
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
    EXPECT_CALL(*g_mock_externals, get_choices())
        .WillRepeatedly(::testing::ReturnRef(choices));
    EXPECT_CALL(*g_mock_externals, get_rocm_domains())
        .WillRepeatedly(Return(std::string{ "memory_copy scratch_memory" }));

    mock_backend_t::buffer_name_info_t buf_names;
    buf_names[mock_backend_t::BUFFER_TRACING_SCRATCH_MEMORY].name = "scratch_memory";

    expect_callback_tracing_names();
    expect_buffer_tracing_names(2, buf_names);
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
    EXPECT_CALL(*g_mock_externals, get_rocm_events_setting())
        .WillRepeatedly(Return(std::string{ "SQ_CYCLES" }));

    // Inject a fake GPU agent so iterate_agent_supported_counters is called exactly
    // once (once per agent in gpu_agents) — making the test hardware-independent.
    // The mock returns SUCCESS without invoking the callback, so agent_counter_info
    // stays empty and create_agent_profile takes its early-return path.
    static ::rocprofsys::agent fake_gpu_counter{
        .type      = ::rocprofsys::agent_type::GPU,
        .handle    = 0xABCDABCFULL,
        .device_id = 0,
        .node_id   = 0,
    };
    sut::tool_data->gpu_agents.push_back(
        ::rocprofsys::rocprofiler_sdk::tool_agent{ 0, &fake_gpu_counter });

    const handle_t fake_agent{ fake_gpu_counter.handle };

    // initialize_event_info() calls initialize() again (agent_manager has no real
    // agents even with the fake one) → 3 calls each for both tracing names.
    expect_callback_tracing_names(3);
    expect_buffer_tracing_names(3);

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

    // initialize_event_info calls iterate_agent_supported_counters exactly once —
    // one call per entry in gpu_agents.  Returns SUCCESS without calling the callback,
    // leaving agent_counter_info empty (agent treated as unsupported architecture).
    EXPECT_CALL(mock(), iterate_agent_supported_counters(fake_agent, _, _))
        .Times(1)
        .WillOnce(Return(SUCCESS));

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

// ─── tool_hip_stream_callback coverage tests ──────────────────────────────────
//
// Fixture that calls tool_init with the "memory_copy" buffered domain active.
// This triggers the ROCPROFILER_VERSION >= 700 guard in tool_init and registers
// tool_hip_stream_callback on primary_ctx (handle 1) via
// configure_callback_tracing_service.  SaveArg<4> captures the raw function
// pointer so each test can invoke the private static method directly, covering
// every reachable branch of the callback body.

class hip_stream_callback_test : public tool_init_test
{
protected:
    using cb_t           = mock_backend_t::callback_tracing_cb_t;
    cb_t m_hip_stream_cb = nullptr;

    void SetUp() override
    {
        tool_init_test::SetUp();

        static const std::vector<std::string> choices{ "memory_copy" };
        EXPECT_CALL(*g_mock_externals, get_choices())
            .WillRepeatedly(::testing::ReturnRef(choices));
        EXPECT_CALL(*g_mock_externals, get_rocm_domains())
            .WillRepeatedly(Return(std::string{ "memory_copy" }));

        expect_callback_tracing_names(2);
        expect_buffer_tracing_names(2);
        expect_three_contexts();
        expect_code_object_tracing();
        expect_external_correlation();

        // Exact-match: primary_ctx = handle_t{1}, CALLBACK_TRACING_HIP_STREAM,
        // ops = nullptr, ops_count = 0, cb_data = nullptr.
        // SaveArg<4> captures the tool_hip_stream_callback function pointer.
        EXPECT_CALL(mock(),
                    configure_callback_tracing_service(
                        handle_t{ 1 }, mock_backend_t::CALLBACK_TRACING_HIP_STREAM,
                        nullptr, 0, _, nullptr))
            .WillOnce(DoAll(::testing::SaveArg<4>(&m_hip_stream_cb), Return(SUCCESS)));

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

        ASSERT_EQ(sut::tool_init(nullptr, sut::tool_data), 0);
        ASSERT_NE(m_hip_stream_cb, nullptr);
    }
};

// Callback's first line is `if(record.kind != HIP_STREAM) return;`.
// With a non-HIP_STREAM kind the body is skipped entirely — payload can be
// null because it is never dereferenced.
TEST_F(hip_stream_callback_test,
       ignores_record_whose_kind_is_not_callback_tracing_hip_stream)
{
    mock_ns::callback_tracing_record_t rec{};
    rec.kind    = mock_backend_t::CALLBACK_TRACING_CODE_OBJECT;
    rec.payload = nullptr;

    mock_ns::user_data_t ud{};
    EXPECT_NO_FATAL_FAILURE(m_hip_stream_cb(rec, &ud, nullptr));
}

// HIP_STREAM_CREATE: extracts stream_id from payload, logs the operation,
// and returns without modifying the stream stack.
TEST_F(hip_stream_callback_test, handles_hip_stream_create_without_side_effects)
{
    mock_ns::hip_stream_data_t stream_data{ .stream_id = handle_t{ 10 } };

    mock_ns::callback_tracing_record_t rec{};
    rec.kind      = mock_backend_t::CALLBACK_TRACING_HIP_STREAM;
    rec.operation = mock_backend_t::HIP_STREAM_CREATE;
    rec.phase     = mock_backend_t::CALLBACK_PHASE_NONE;
    rec.payload   = &stream_data;

    mock_ns::user_data_t ud{};
    EXPECT_NO_FATAL_FAILURE(m_hip_stream_cb(rec, &ud, nullptr));
}

// HIP_STREAM_DESTROY: mirrors CREATE — logs and returns, no stack change.
TEST_F(hip_stream_callback_test, handles_hip_stream_destroy_without_side_effects)
{
    mock_ns::hip_stream_data_t stream_data{ .stream_id = handle_t{ 10 } };

    mock_ns::callback_tracing_record_t rec{};
    rec.kind      = mock_backend_t::CALLBACK_TRACING_HIP_STREAM;
    rec.operation = mock_backend_t::HIP_STREAM_DESTROY;
    rec.phase     = mock_backend_t::CALLBACK_PHASE_NONE;
    rec.payload   = &stream_data;

    mock_ns::user_data_t ud{};
    EXPECT_NO_FATAL_FAILURE(m_hip_stream_cb(rec, &ud, nullptr));
}

// HIP_STREAM_SET ENTER pushes the stream_id onto the thread-local stream stack;
// EXIT pops it.  Both phases are exercised in a single paired invocation so
// the thread-local stack is left balanced for subsequent tests.
TEST_F(hip_stream_callback_test, hip_stream_set_enter_pushes_and_exit_pops_stream_id)
{
    mock_ns::hip_stream_data_t stream_data{ .stream_id = handle_t{ 42 } };

    mock_ns::callback_tracing_record_t enter_rec{};
    enter_rec.kind      = mock_backend_t::CALLBACK_TRACING_HIP_STREAM;
    enter_rec.operation = mock_backend_t::HIP_STREAM_SET;
    enter_rec.phase     = mock_backend_t::CALLBACK_PHASE_ENTER;
    enter_rec.payload   = &stream_data;

    mock_ns::callback_tracing_record_t exit_rec{};
    exit_rec.kind      = mock_backend_t::CALLBACK_TRACING_HIP_STREAM;
    exit_rec.operation = mock_backend_t::HIP_STREAM_SET;
    exit_rec.phase     = mock_backend_t::CALLBACK_PHASE_EXIT;
    exit_rec.payload   = &stream_data;

    mock_ns::user_data_t ud{};
    m_hip_stream_cb(enter_rec, &ud, nullptr);
    EXPECT_NO_FATAL_FAILURE(m_hip_stream_cb(exit_rec, &ud, nullptr));
}

// ─── KFD buffer domain tests ──────────────────────────────────────────────────
//
// "kfd_events" is the umbrella domain name added to valid choices when
// compile_time_version >= 10000.  Activating it populates _buffered_domain with
// all six KFD buffer tracing kinds, covering the entire KFD block in tool_init:
//   metadata_initialize + one create_buffer / configure_buffer_tracing_service
//   pair per kind + create_callback_thread / assign_callback_thread per buffer.
//
// KFD_EVENT_QUEUE is the only kind whose configure_buffer_tracing_service call
// passes a non-null ops array (ops_count=1, only RESTORE_RESCHEDULED operation).
// Every other kind uses nullptr/0.

TEST_F(tool_init_test, creates_all_kfd_buffers_when_kfd_events_domain_active)
{
    static const std::vector<std::string> choices{ "kfd_events" };
    EXPECT_CALL(*g_mock_externals, get_choices())
        .WillRepeatedly(::testing::ReturnRef(choices));
    EXPECT_CALL(*g_mock_externals, get_rocm_domains())
        .WillRepeatedly(Return(std::string{ "kfd_events" }));

    expect_callback_tracing_names();
    expect_buffer_tracing_names();
    expect_three_contexts();
    expect_code_object_tracing();
    expect_external_correlation();

    // Six KFD buffers created in the order the if-blocks appear in tool_init.
    const handle_t pf_buf{ 101 };
    const handle_t pm_buf{ 102 };
    const handle_t q_buf{ 103 };
    const handle_t eq_buf{ 104 };
    const handle_t um_buf{ 105 };
    const handle_t de_buf{ 106 };

    EXPECT_CALL(mock(), create_buffer(handle_t{ 1 }, _, _, _, _, _, _))
        .WillOnce(DoAll(SetArgPointee<6>(pf_buf), Return(SUCCESS)))
        .WillOnce(DoAll(SetArgPointee<6>(pm_buf), Return(SUCCESS)))
        .WillOnce(DoAll(SetArgPointee<6>(q_buf), Return(SUCCESS)))
        .WillOnce(DoAll(SetArgPointee<6>(eq_buf), Return(SUCCESS)))
        .WillOnce(DoAll(SetArgPointee<6>(um_buf), Return(SUCCESS)))
        .WillOnce(DoAll(SetArgPointee<6>(de_buf), Return(SUCCESS)));

    EXPECT_CALL(mock(), configure_buffer_tracing_service(
                            handle_t{ 1 }, mock_backend_t::BUFFER_TRACING_KFD_PAGE_FAULT,
                            nullptr, 0, pf_buf))
        .WillOnce(Return(SUCCESS));
    EXPECT_CALL(mock(),
                configure_buffer_tracing_service(
                    handle_t{ 1 }, mock_backend_t::BUFFER_TRACING_KFD_PAGE_MIGRATE,
                    nullptr, 0, pm_buf))
        .WillOnce(Return(SUCCESS));
    EXPECT_CALL(mock(), configure_buffer_tracing_service(
                            handle_t{ 1 }, mock_backend_t::BUFFER_TRACING_KFD_QUEUE,
                            nullptr, 0, q_buf))
        .WillOnce(Return(SUCCESS));
    // Only RESTORE_RESCHEDULED is requested for EVENT_QUEUE — non-null ops, count=1.
    EXPECT_CALL(mock(), configure_buffer_tracing_service(
                            handle_t{ 1 }, mock_backend_t::BUFFER_TRACING_KFD_EVENT_QUEUE,
                            ::testing::NotNull(), 1, eq_buf))
        .WillOnce(Return(SUCCESS));
    EXPECT_CALL(mock(), configure_buffer_tracing_service(
                            handle_t{ 1 },
                            mock_backend_t::BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU,
                            nullptr, 0, um_buf))
        .WillOnce(Return(SUCCESS));
    EXPECT_CALL(mock(), configure_buffer_tracing_service(
                            handle_t{ 1 },
                            mock_backend_t::BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS,
                            nullptr, 0, de_buf))
        .WillOnce(Return(SUCCESS));

    // Six non-zero buffer handles → six create_callback_thread/assign pairs.
    EXPECT_CALL(mock(), create_callback_thread(_))
        .Times(6)
        .WillRepeatedly(DoAll(SetArgPointee<0>(handle_t{ 200 }), Return(SUCCESS)));
    EXPECT_CALL(mock(), assign_callback_thread(_, _))
        .Times(6)
        .WillRepeatedly(Return(SUCCESS));

    expect_primary_ctx_valid();
    expect_start_for_initialized_contexts();

    EXPECT_EQ(sut::tool_init(nullptr, sut::tool_data), 0);

    EXPECT_EQ(sut::tool_data->kfd_page_fault_buffer, pf_buf);
    EXPECT_EQ(sut::tool_data->kfd_page_migrate_buffer, pm_buf);
    EXPECT_EQ(sut::tool_data->kfd_queue_buffer, q_buf);
    EXPECT_EQ(sut::tool_data->kfd_event_queue_buffer, eq_buf);
    EXPECT_EQ(sut::tool_data->kfd_event_unmap_buffer, um_buf);
    EXPECT_EQ(sut::tool_data->kfd_event_dropped_buffer, de_buf);
}

// ─── GPU agent / counter-event loop coverage ──────────────────────────────────
//
// The counter-event loop inside tool_init:
//
//   for(const auto& itr : _data->gpu_agents)
//   {
//       const auto& _agent_id = Wrapper::agent_id{ itr.agent->handle };
//       _data->agent_events.emplace(
//           _agent_id, create_agent_profile(_agent_id, _counter_events, _data));
//   }
//
// never executes on GPU-less machines because agent_manager returns nothing.
//
// Strategy: pre-inject a fake ::rocprofsys::agent into tool_data->gpu_agents.
// set_agents() (called by initialize()) does emplace_back without clearing, so
// the entry survives.  The iterate_agent_supported_counters mock invokes
// counters_supported_callback with 0 counters, which registers an empty entry
// in agent_counter_info for the fake agent.  create_agent_profile then proceeds
// past the "unsupported architecture" guard into the counter-matching loop,
// finds no match for "SQ_CYCLES", and records an empty agent_events entry.

TEST_F(tool_init_test, counter_loop_runs_for_each_gpu_agent_when_no_counters_match)
{
    EXPECT_CALL(*g_mock_externals, get_rocm_events_setting())
        .WillRepeatedly(Return(std::string{ "SQ_CYCLES" }));

    // Inject a fake GPU agent whose handle is used as the agent_id key.
    // Static lifetime avoids a dangling pointer; tool_data->gpu_agents is
    // reset in each SetUp() so there is no cross-test contamination.
    static ::rocprofsys::agent fake_gpu{
        .type      = ::rocprofsys::agent_type::GPU,
        .handle    = 0xABCDABCDULL,
        .device_id = 0,
        .node_id   = 0,
    };
    sut::tool_data->gpu_agents.push_back(
        ::rocprofsys::rocprofiler_sdk::tool_agent{ 0, &fake_gpu });

    const handle_t fake_agent{ fake_gpu.handle };

    // Counter events → initialize_event_info() calls initialize() again on GPU-less
    // machines → 3 calls each.
    expect_callback_tracing_names(3);
    expect_buffer_tracing_names(3);

    // Four contexts: primary(1), code_object(2), control(3), counter(4).
    EXPECT_CALL(mock(), create_context(_))
        .WillOnce(DoAll(SetArgPointee<0>(handle_t{ 1 }), Return(SUCCESS)))
        .WillOnce(DoAll(SetArgPointee<0>(handle_t{ 2 }), Return(SUCCESS)))
        .WillOnce(DoAll(SetArgPointee<0>(handle_t{ 3 }), Return(SUCCESS)))
        .WillOnce(DoAll(SetArgPointee<0>(handle_t{ 4 }), Return(SUCCESS)));

    expect_code_object_tracing();
    expect_external_correlation();

    // initialize_event_info calls iterate_agent_supported_counters for fake_agent.
    // The lambda invokes counters_supported_callback with 0 counters so that
    // agent_counter_info gains an empty entry — allowing create_agent_profile to
    // reach the counter-matching loop instead of taking the early-return path.
    using status_t = mock_backend_t::status_t;
    using cb_t     = mock_backend_t::available_counters_cb_t;
    EXPECT_CALL(mock(), iterate_agent_supported_counters(fake_agent, _, _))
        .WillOnce([](handle_t id, cb_t cb, void* data) -> status_t {
            cb(id, nullptr, 0, data);
            return mock_backend_t::STATUS_SUCCESS;
        });

    EXPECT_CALL(mock(), configure_callback_tracing_service(
                            handle_t{ 4 },
                            mock_backend_t::CALLBACK_TRACING_KERNEL_DISPATCH, _, _, _, _))
        .WillOnce(Return(SUCCESS));
    EXPECT_CALL(mock(),
                configure_callback_dispatch_counting_service(handle_t{ 4 }, _, _, _, _))
        .WillOnce(Return(SUCCESS));

    expect_primary_ctx_valid();

    for(const auto h : { handle_t{ 1 }, handle_t{ 2 }, handle_t{ 3 }, handle_t{ 4 } })
    {
        EXPECT_CALL(mock(), context_is_active(h, _))
            .WillOnce(DoAll(SetArgPointee<1>(0), Return(SUCCESS)));
        EXPECT_CALL(mock(), start_context(h)).WillOnce(Return(SUCCESS));
    }

    EXPECT_EQ(sut::tool_init(nullptr, sut::tool_data), 0);

    // Loop executed for fake_agent: no counter matched "SQ_CYCLES" → empty event list.
    EXPECT_EQ(sut::tool_data->counter_ctx, handle_t{ 4 });
    EXPECT_EQ(sut::tool_data->agent_events.count(fake_agent), 1u);
    EXPECT_TRUE(sut::tool_data->agent_events.at(fake_agent).empty());
}

// When the injected GPU agent's counter_info contains a name that matches a
// requested counter event, create_agent_profile reaches the
//
//   if(!counters_v.empty()) { create_counter_config(...); profile = profile_v; }
//
// branch.  This test drives that path end-to-end.
//
// Call chain inside initialize_event_info (NOT create_agent_profile):
//   iterate_agent_supported_counters → lambda calls counters_supported_callback
//     → query_counter_info   (populates counter_info_v0_t with name="SQ_CYCLES")
//     → iterate_counter_dimensions (no dimensions)
//   agent_counter_info[fake_agent] = [{name="SQ_CYCLES", id=handle_t{42}}]
//
// Then inside the tool_init counter loop → create_agent_profile:
//   "SQ_CYCLES" matches citr.name  → counters_v = [handle_t{42}] (non-empty)
//   create_counter_config called   → profile stored for fake_agent

TEST_F(tool_init_test, creates_counter_profile_when_gpu_agent_counter_name_matches_event)
{
    EXPECT_CALL(*g_mock_externals, get_rocm_events_setting())
        .WillRepeatedly(Return(std::string{ "SQ_CYCLES" }));

    static ::rocprofsys::agent fake_gpu2{
        .type      = ::rocprofsys::agent_type::GPU,
        .handle    = 0xABCDABCEULL,
        .device_id = 1,
        .node_id   = 0,
    };
    sut::tool_data->gpu_agents.push_back(
        ::rocprofsys::rocprofiler_sdk::tool_agent{ 1, &fake_gpu2 });

    const handle_t fake_agent{ fake_gpu2.handle };
    const handle_t sq_handle{ 42 };

    expect_callback_tracing_names(3);
    expect_buffer_tracing_names(3);

    EXPECT_CALL(mock(), create_context(_))
        .WillOnce(DoAll(SetArgPointee<0>(handle_t{ 1 }), Return(SUCCESS)))
        .WillOnce(DoAll(SetArgPointee<0>(handle_t{ 2 }), Return(SUCCESS)))
        .WillOnce(DoAll(SetArgPointee<0>(handle_t{ 3 }), Return(SUCCESS)))
        .WillOnce(DoAll(SetArgPointee<0>(handle_t{ 4 }), Return(SUCCESS)));

    expect_code_object_tracing();
    expect_external_correlation();

    // Provide one counter handle to counters_supported_callback.
    using status_t = mock_backend_t::status_t;
    using cb_t     = mock_backend_t::available_counters_cb_t;
    EXPECT_CALL(mock(), iterate_agent_supported_counters(fake_agent, _, _))
        .WillOnce([sq_handle](handle_t id, cb_t cb, void* data) -> status_t {
            handle_t h = sq_handle;
            cb(id, &h, 1, data);
            return mock_backend_t::STATUS_SUCCESS;
        });

    // counters_supported_callback calls query_counter_info for the counter.
    // Populate name and id so the counter matches "SQ_CYCLES" in create_agent_profile.
    EXPECT_CALL(mock(), query_counter_info(sq_handle, _, _))
        .WillOnce([sq_handle](handle_t, int, void* out) -> status_t {
            auto* info        = static_cast<mock_backend_t::counter_info_v0_t*>(out);
            info->id          = sq_handle;
            info->name        = "SQ_CYCLES";
            info->is_constant = 0;
            info->is_derived  = 0;
            return mock_backend_t::STATUS_SUCCESS;
        });

    // No dimensions for this counter.
    EXPECT_CALL(mock(), iterate_counter_dimensions(sq_handle, _, _))
        .WillOnce(Return(SUCCESS));

    // create_counter_config is called when counters_v is non-empty.
    const handle_t profile_handle{ 77 };
    EXPECT_CALL(mock(), create_counter_config(fake_agent, _, 1, _))
        .WillOnce(DoAll(SetArgPointee<3>(profile_handle), Return(SUCCESS)));

    EXPECT_CALL(mock(), configure_callback_tracing_service(
                            handle_t{ 4 },
                            mock_backend_t::CALLBACK_TRACING_KERNEL_DISPATCH, _, _, _, _))
        .WillOnce(Return(SUCCESS));
    EXPECT_CALL(mock(),
                configure_callback_dispatch_counting_service(handle_t{ 4 }, _, _, _, _))
        .WillOnce(Return(SUCCESS));

    expect_primary_ctx_valid();

    for(const auto h : { handle_t{ 1 }, handle_t{ 2 }, handle_t{ 3 }, handle_t{ 4 } })
    {
        EXPECT_CALL(mock(), context_is_active(h, _))
            .WillOnce(DoAll(SetArgPointee<1>(0), Return(SUCCESS)));
        EXPECT_CALL(mock(), start_context(h)).WillOnce(Return(SUCCESS));
    }

    EXPECT_EQ(sut::tool_init(nullptr, sut::tool_data), 0);

    // create_agent_profile ran for fake_agent: "SQ_CYCLES" matched → profile created.
    ASSERT_TRUE(sut::tool_data->agent_counter_profiles.count(fake_agent) > 0);
    EXPECT_TRUE(sut::tool_data->agent_counter_profiles.at(fake_agent).has_value());
    EXPECT_EQ(*sut::tool_data->agent_counter_profiles.at(fake_agent), profile_handle);

    // agent_events holds the matched counter id.
    ASSERT_EQ(sut::tool_data->agent_events.count(fake_agent), 1u);
    ASSERT_FALSE(sut::tool_data->agent_events.at(fake_agent).empty());
    EXPECT_EQ(sut::tool_data->agent_events.at(fake_agent).front(), sq_handle);
}

// ─── tool_tracing_callback coverage tests ─────────────────────────────────────
//
// tool_tracing_callback is registered as the sdk callback for every active
// callback domain (hip_api, hsa_api, …).  This fixture activates "hip_api",
// captures the callback pointer via SaveArg<4>, and then directly invokes it.
//
// Three control-flow branches are exercised:
//   1. get_state() != Active  → early return at line 1793 (no phase dispatch)
//   2. CALLBACK_PHASE_ENTER   → tool_tracing_callback_start path
//   3. CALLBACK_PHASE_EXIT    → populate_backtrace_data + tool_tracing_callback_stop
//
// mock_externals returns false for get_use_perfetto() / get_use_timemory() /
// get_use_rocpd(), so those branches are dead and only the three mock methods
// below matter when state is Active:
//   - get_timestamp   (called at function entry, before the state guard)
//   - query_callback_op_name  (same)
//   - iterate_callback_tracing_kind_operation_args  (EXIT only)

class tool_tracing_callback_test : public tool_init_test
{
protected:
    using sut_core =
        ::rocprofsys::rocprofiler_sdk::sdk_core<mock_backend_t, mock_externals>;
    using cb_t = mock_backend_t::callback_tracing_cb_t;

    cb_t m_tracing_cb = nullptr;

    void SetUp() override
    {
        tool_init_test::SetUp();

        // set_operation_options must be called before tool_init so that
        // sdk_core::get_operations() does not throw for hip_api kinds.
        sut_core::set_operation_options(mock_backend_t::CALLBACK_TRACING_HIP_RUNTIME_API,
                                        "", "", "");
        sut_core::set_operation_options(mock_backend_t::CALLBACK_TRACING_HIP_COMPILER_API,
                                        "", "", "");

        static const std::vector<std::string> choices{ "hip_api" };
        EXPECT_CALL(*g_mock_externals, get_choices())
            .WillRepeatedly(::testing::ReturnRef(choices));
        EXPECT_CALL(*g_mock_externals, get_rocm_domains())
            .WillRepeatedly(Return(std::string{ "hip_api" }));

        // tool_tracing_callback reads these booleans on every invocation.
        EXPECT_CALL(*g_mock_externals, get_use_perfetto()).WillRepeatedly(Return(false));
        EXPECT_CALL(*g_mock_externals, get_use_timemory()).WillRepeatedly(Return(false));
        EXPECT_CALL(*g_mock_externals, get_perfetto_annotations())
            .WillRepeatedly(Return(false));
        EXPECT_CALL(*g_mock_externals, get_use_rocpd()).WillRepeatedly(Return(false));

        // Active callback domain (hip_api): get_operations_impl cold → 3 callback calls.
        // No counter events → 2 buffer calls (initialize + get_buffered_domains).
        expect_callback_tracing_names(3);
        expect_buffer_tracing_names(2);
        expect_three_contexts();
        expect_code_object_tracing();
        expect_external_correlation();

        // Capture tool_tracing_callback from the HIP_RUNTIME_API registration.
        EXPECT_CALL(mock(),
                    configure_callback_tracing_service(
                        handle_t{ 1 }, mock_backend_t::CALLBACK_TRACING_HIP_RUNTIME_API,
                        _, _, _, _))
            .WillOnce(DoAll(::testing::SaveArg<4>(&m_tracing_cb), Return(SUCCESS)));
        EXPECT_CALL(mock(),
                    configure_callback_tracing_service(
                        handle_t{ 1 }, mock_backend_t::CALLBACK_TRACING_HIP_COMPILER_API,
                        _, _, _, _))
            .WillOnce(Return(SUCCESS));

        expect_primary_ctx_valid();
        expect_start_for_initialized_contexts();

        ASSERT_EQ(sut::tool_init(nullptr, sut::tool_data), 0);
        ASSERT_NE(m_tracing_cb, nullptr);
    }

    void TearDown() override
    {
        // Reset global tool state so other tests start from a clean baseline.
        ::rocprofsys::reset_state();
        tool_init_test::TearDown();
    }

    // Helpers to build records for the two phases under test.
    static mock_ns::callback_tracing_record_t make_hip_enter()
    {
        mock_ns::callback_tracing_record_t rec{};
        rec.kind      = mock_backend_t::CALLBACK_TRACING_HIP_RUNTIME_API;
        rec.operation = 1;
        rec.phase     = mock_backend_t::CALLBACK_PHASE_ENTER;
        return rec;
    }

    static mock_ns::callback_tracing_record_t make_hip_exit()
    {
        mock_ns::callback_tracing_record_t rec{};
        rec.kind      = mock_backend_t::CALLBACK_TRACING_HIP_RUNTIME_API;
        rec.operation = 1;
        rec.phase     = mock_backend_t::CALLBACK_PHASE_EXIT;
        return rec;
    }

    // Every tool_tracing_callback invocation calls get_timestamp and
    // query_callback_op_name before the state guard.
    void expect_common_preamble(std::uint64_t ts_out = 100)
    {
        EXPECT_CALL(mock(), get_timestamp(_))
            .WillOnce(DoAll(SetArgPointee<0>(ts_out), Return(SUCCESS)));
        EXPECT_CALL(mock(), query_callback_op_name(_, _, _, _)).WillOnce(Return(SUCCESS));
    }
};

// State != Active: callback logs a warning and returns before any phase dispatch.
// Only get_timestamp and query_callback_op_name are called.
TEST_F(tool_tracing_callback_test, returns_early_when_tool_state_is_not_active)
{
    expect_common_preamble();

    auto                 rec = make_hip_enter();
    mock_ns::user_data_t ud{};
    EXPECT_NO_FATAL_FAILURE(m_tracing_cb(rec, &ud, nullptr));
}

// CALLBACK_PHASE_ENTER with HIP_RUNTIME_API: stores timestamp in user_data->value
// and dispatches to tool_tracing_callback_start (timemory/perfetto both disabled
// in mock_externals, so it returns immediately after the name lookup).
TEST_F(tool_tracing_callback_test, enter_phase_stores_timestamp_in_user_data)
{
    ::rocprofsys::set_state(::rocprofsys::State::Active);

    expect_common_preamble(/*ts_out=*/42);

    auto                 rec = make_hip_enter();
    mock_ns::user_data_t ud{};
    EXPECT_NO_FATAL_FAILURE(m_tracing_cb(rec, &ud, nullptr));

    EXPECT_EQ(ud.value, 42u);
}

// CALLBACK_PHASE_EXIT with HIP_RUNTIME_API: calls populate_backtrace_data (no-op
// since perfetto+rocpd disabled), tool_tracing_callback_stop, and
// iterate_callback_tracing_kind_operation_args, then caches the region.
TEST_F(tool_tracing_callback_test, exit_phase_calls_stop_and_caches_region)
{
    ::rocprofsys::set_state(::rocprofsys::State::Active);

    expect_common_preamble(/*ts_out=*/200);

    // EXIT path calls iterate_callback_tracing_kind_operation_args inside
    // tool_tracing_callback_stop to serialise function arguments.
    EXPECT_CALL(mock(), iterate_callback_tracing_kind_operation_args(_, _, _, _))
        .WillOnce(Return(SUCCESS));

    auto                 rec = make_hip_exit();
    mock_ns::user_data_t ud{ .value = 100 };  // simulates timestamp from a prior ENTER
    EXPECT_NO_FATAL_FAILURE(m_tracing_cb(rec, &ud, nullptr));
}

// ─── Boolean flag coverage (true / false for every Externals bool) ─────────────
//
// Each test below overrides one or more boolean EXPECT_CALLs set in the fixture
// SetUp (the later registration takes precedence via GMock's LIFO ordering) and
// invokes the captured callback.
//
// Template perfetto / timemory methods (push_timemory, push_perfetto_ts, etc.)
// are no-op stubs — they compile and link without additional mock setup.

// ENTER with timemory=true → tool_tracing_callback_start calls push_timemory.
TEST_F(tool_tracing_callback_test, enter_phase_with_timemory_enabled_calls_push_timemory)
{
    ::rocprofsys::set_state(::rocprofsys::State::Active);

    EXPECT_CALL(*g_mock_externals, get_use_timemory()).WillRepeatedly(Return(true));

    expect_common_preamble(/*ts_out=*/10);

    auto                 rec = make_hip_enter();
    mock_ns::user_data_t ud{};
    EXPECT_NO_FATAL_FAILURE(m_tracing_cb(rec, &ud, nullptr));
}

// EXIT with timemory=true → tool_tracing_callback_stop calls pop_timemory.
TEST_F(tool_tracing_callback_test, exit_phase_with_timemory_enabled_calls_pop_timemory)
{
    ::rocprofsys::set_state(::rocprofsys::State::Active);

    EXPECT_CALL(*g_mock_externals, get_use_timemory()).WillRepeatedly(Return(true));

    expect_common_preamble(/*ts_out=*/20);
    EXPECT_CALL(mock(), iterate_callback_tracing_kind_operation_args(_, _, _, _))
        .WillOnce(Return(SUCCESS));

    auto                 rec = make_hip_exit();
    mock_ns::user_data_t ud{ .value = 5 };
    EXPECT_NO_FATAL_FAILURE(m_tracing_cb(rec, &ud, nullptr));
}

// EXIT with perfetto=true, annotations=false → enters the perfetto block,
// calls push/pop_perfetto_ts (no-op stubs), and one iterate_args for RocPD args.
TEST_F(tool_tracing_callback_test,
       exit_phase_with_perfetto_enabled_no_annotations_calls_push_pop_perfetto)
{
    ::rocprofsys::set_state(::rocprofsys::State::Active);

    EXPECT_CALL(*g_mock_externals, get_use_perfetto()).WillRepeatedly(Return(true));

    expect_common_preamble(/*ts_out=*/30);
    // Only one iterate_args call: the perfetto annotations block is skipped.
    EXPECT_CALL(mock(), iterate_callback_tracing_kind_operation_args(_, _, _, _))
        .WillOnce(Return(SUCCESS));

    auto                 rec = make_hip_exit();
    mock_ns::user_data_t ud{ .value = 5 };
    EXPECT_NO_FATAL_FAILURE(m_tracing_cb(rec, &ud, nullptr));
}

// EXIT with perfetto=true, annotations=true → enters full perfetto block:
//   iterate_args called TWICE (once for save_args inside the block, once for
//   the regular RocPD args after).
TEST_F(tool_tracing_callback_test,
       exit_phase_with_perfetto_and_annotations_enabled_calls_iterate_args_twice)
{
    ::rocprofsys::set_state(::rocprofsys::State::Active);

    EXPECT_CALL(*g_mock_externals, get_use_perfetto()).WillRepeatedly(Return(true));
    EXPECT_CALL(*g_mock_externals, get_perfetto_annotations())
        .WillRepeatedly(Return(true));

    expect_common_preamble(/*ts_out=*/40);
    // Two calls: one for save_args (inside perfetto block), one for
    // iterate_args_callback.
    EXPECT_CALL(mock(), iterate_callback_tracing_kind_operation_args(_, _, _, _))
        .Times(2)
        .WillRepeatedly(Return(SUCCESS));

    auto                 rec = make_hip_exit();
    mock_ns::user_data_t ud{ .value = 5 };
    EXPECT_NO_FATAL_FAILURE(m_tracing_cb(rec, &ud, nullptr));
}

// EXIT with rocpd=true and backtrace_operations populated → populate_backtrace_data()
// enters the unwind branch.  Externals::tim_get_unw_stack() is now a mock that
// notifies gmock_externals and returns an empty stack — zero real libunwind time.
TEST_F(tool_tracing_callback_test,
       exit_phase_with_rocpd_enabled_and_backtrace_op_registered_enters_unwind_path)
{
    ::rocprofsys::set_state(::rocprofsys::State::Active);

    EXPECT_CALL(*g_mock_externals, get_use_rocpd()).WillRepeatedly(Return(true));

    // Insert the record's operation so the (use_rocpd && count>0) condition is true.
    sut::tool_data->backtrace_operations[mock_backend_t::CALLBACK_TRACING_HIP_RUNTIME_API]
        .insert(1);

    // The mock returns a typed empty stack — verifies the path was entered
    // without any real libunwind walking.
    EXPECT_CALL(*g_mock_externals, tim_get_unw_stack())
        .Times(1)
        .WillOnce(Return(mock_externals::unwind_stack_t<16>{}));

    expect_common_preamble(/*ts_out=*/50);
    EXPECT_CALL(mock(), iterate_callback_tracing_kind_operation_args(_, _, _, _))
        .WillOnce(Return(SUCCESS));

    auto                 rec = make_hip_exit();
    mock_ns::user_data_t ud{ .value = 5 };
    EXPECT_NO_FATAL_FAILURE(m_tracing_cb(rec, &ud, nullptr));
}
