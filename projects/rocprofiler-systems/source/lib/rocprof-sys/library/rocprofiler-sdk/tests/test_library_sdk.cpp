// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Include the mock before the SUT so gmock_wrapper / g_mock_wrapper are defined
// in the same TU that instantiates library_sdk<mock_backend>.
#include "mock_wrapper.hpp"
#include "rocprof-sys/library/rocprofiler-sdk.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>

// ─── Type aliases ─────────────────────────────────────────────────────────────

namespace mock_ns = ::rocprofsys::mock::rocprofiler_sdk;

using mock_backend_t = mock_ns::backend;
using sut            = ::rocprofsys::rocprofiler_sdk::library_sdk<mock_backend_t>;
using data_t         = ::rocprofsys::rocprofiler_sdk::client_data<mock_backend_t>;
using handle_t       = mock_backend_t::handle_t;

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
