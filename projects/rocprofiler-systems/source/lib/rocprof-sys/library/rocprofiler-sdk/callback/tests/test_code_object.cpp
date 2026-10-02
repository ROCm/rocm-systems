// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/callback/code_object.hpp"
#include "library/rocprofiler-sdk/tests/mock_domain_service.hpp"
#include "library/rocprofiler-sdk/types.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>

namespace rocprofsys::domains::callback
{
namespace
{

using ::testing::Field;
using ::testing::StrictMock;

using test_support::externals;
using test_support::g_metadata_registry_mock;
using test_support::gmock_metadata_registry;
using test_support::mock_sdk;

}  // namespace

TEST(code_object_test, descriptor_reports_correct_metadata)
{
    constexpr const auto& k_domain = k_code_object<mock_sdk, externals>;

    EXPECT_EQ(k_domain.meta.name, "code_object");
    EXPECT_EQ(k_domain.meta.id, mock_sdk::CALLBACK_TRACING_CODE_OBJECT);
    EXPECT_EQ(k_domain.meta.mode, collection_mode::callback);
    EXPECT_FALSE(k_domain.meta.group.has_value());
}

TEST(code_object_test, on_code_object_enter_handles_call_without_crashing)
{
    const mock_sdk::callback_tracing_record_t record{};
    mock_sdk::user_data_t                     user_data{};

    on_code_object_enter<mock_sdk, externals>(record, &user_data, nullptr);
}

TEST(code_object_test, on_code_object_enter_registers_loaded_code_object)
{
    g_metadata_registry_mock = std::make_unique<StrictMock<gmock_metadata_registry>>();

    auto payload     = mock_sdk::code_object_load_data_t{ .code_object_id = 42 };
    auto record      = mock_sdk::callback_tracing_record_t{};
    record.operation = mock_sdk::CODE_OBJECT_LOAD;
    record.payload   = &payload;
    mock_sdk::user_data_t user_data{};

    EXPECT_CALL(
        *g_metadata_registry_mock,
        add_code_object(Field(&mock_sdk::code_object_load_data_t::code_object_id, 42)));

    on_code_object_enter<mock_sdk, externals>(record, &user_data, nullptr);

    g_metadata_registry_mock.reset();
}

TEST(code_object_test, on_code_object_enter_registers_kernel_symbol)
{
    g_metadata_registry_mock = std::make_unique<StrictMock<gmock_metadata_registry>>();

    auto payload = mock_sdk::code_object_kernel_symbol_register_data_t{ .kernel_id = 7 };
    auto record  = mock_sdk::callback_tracing_record_t{};
    record.operation = mock_sdk::CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER;
    record.payload   = &payload;
    mock_sdk::user_data_t user_data{};

    EXPECT_CALL(*g_metadata_registry_mock,
                add_kernel_symbol(Field(
                    &mock_sdk::code_object_kernel_symbol_register_data_t::kernel_id, 7)));

    on_code_object_enter<mock_sdk, externals>(record, &user_data, nullptr);

    g_metadata_registry_mock.reset();
}

TEST(code_object_test, on_record_dispatches_by_phase_without_crashing)
{
    constexpr const auto& k_domain = k_code_object<mock_sdk, externals>;
    mock_sdk::user_data_t user_data{};

    auto enter_record  = mock_sdk::callback_tracing_record_t{};
    enter_record.phase = mock_sdk::CALLBACK_PHASE_ENTER;
    k_domain.on_record(enter_record, &user_data, nullptr);

    auto exit_record  = mock_sdk::callback_tracing_record_t{};
    exit_record.phase = mock_sdk::CALLBACK_PHASE_EXIT;
    k_domain.on_record(exit_record, &user_data, nullptr);

    auto none_record  = mock_sdk::callback_tracing_record_t{};
    none_record.phase = mock_sdk::CALLBACK_PHASE_NONE;
    k_domain.on_record(none_record, &user_data, nullptr);
}

}  // namespace rocprofsys::domains::callback
