// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/buffered/kfd_page_migrate.hpp"
#include "library/rocprofiler-sdk/tests/mock_domain_service.hpp"
#include "library/rocprofiler-sdk/types.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace rocprofsys::domains::buffered
{
namespace
{

using ::testing::AllOf;
using ::testing::Eq;
using ::testing::Field;
using ::testing::Return;
using ::testing::StrictMock;

using test_support::agent_t;
using test_support::externals;
using test_support::g_externals_mock;
using test_support::gmock_externals;
using test_support::mock_sdk;
using test_support::pmc_info_data_t;

}  // namespace

TEST(kfd_page_migrate_test, descriptor_reports_correct_metadata)
{
    using mock_dispatcher =
        buffered_callback_dispatcher<mock_sdk, mock_sdk::kfd_page_migrate_record,
                                     on_kfd_page_migrate<mock_sdk, externals>>;
    constexpr const auto& k_domain = k_kfd_page_migrate<mock_sdk, externals>;

    EXPECT_EQ(k_domain.meta.name, "kfd_page_migrate");
    EXPECT_EQ(k_domain.meta.id, mock_sdk::BUFFER_TRACING_KFD_PAGE_MIGRATE);
    EXPECT_EQ(k_domain.meta.mode, collection_mode::buffered);
    ASSERT_TRUE(k_domain.meta.group.has_value());
    EXPECT_EQ(k_domain.meta.group.value().name, "kfd_events");
    EXPECT_EQ(k_domain.on_records, &mock_dispatcher::callback);
}

TEST(kfd_page_migrate_test, descriptor_uses_default_buffer_properties)
{
    constexpr const auto& k_domain = k_kfd_page_migrate<mock_sdk, externals>;

    EXPECT_EQ(k_domain.buffer.buffer_size, k_default_buffer_properties.buffer_size);
    EXPECT_EQ(k_domain.buffer.buffer_watermark,
              k_default_buffer_properties.buffer_watermark);
}

TEST(kfd_page_migrate_test,
     on_kfd_page_migrate_handles_empty_record_batch_without_crashing)
{
    mock_sdk::kfd_page_migrate_record record{};

    on_kfd_page_migrate<mock_sdk, externals>(&record, nullptr);
}

TEST(kfd_page_migrate_test,
     on_configure_registers_category_string_and_skips_pmc_info_without_agents)
{
    g_externals_mock = std::make_unique<StrictMock<gmock_externals>>();

    EXPECT_CALL(*g_externals_mock,
                add_string(Eq(externals::k_kfd_page_migrate_category_name)))
        .Times(1);
    EXPECT_CALL(*g_externals_mock, get_agents_by_type(Eq(externals::k_agent_type_gpu)))
        .Times(1)
        .WillOnce(Return(std::vector<std::shared_ptr<agent_t>>{}));
    EXPECT_CALL(*g_externals_mock, get_agents_by_type(Eq(externals::k_agent_type_cpu)))
        .Times(1)
        .WillOnce(Return(std::vector<std::shared_ptr<agent_t>>{}));

    on_kfd_page_migrate_configure<externals>();

    g_externals_mock.reset();
}

TEST(kfd_page_migrate_test, on_configure_registers_pmc_info_for_each_gpu_and_cpu_agent)
{
    g_externals_mock = std::make_unique<StrictMock<gmock_externals>>();

    auto gpu_agent               = std::make_shared<agent_t>();
    gpu_agent->type              = externals::k_agent_type_gpu;
    gpu_agent->device_type_index = 1;
    auto cpu_agent               = std::make_shared<agent_t>();
    cpu_agent->type              = externals::k_agent_type_cpu;
    cpu_agent->device_type_index = 0;

    EXPECT_CALL(*g_externals_mock,
                add_string(Eq(externals::k_kfd_page_migrate_category_name)))
        .Times(1);
    EXPECT_CALL(*g_externals_mock, get_agents_by_type(Eq(externals::k_agent_type_gpu)))
        .Times(1)
        .WillOnce(Return(std::vector<std::shared_ptr<agent_t>>{ gpu_agent }));
    EXPECT_CALL(*g_externals_mock, get_agents_by_type(Eq(externals::k_agent_type_cpu)))
        .Times(1)
        .WillOnce(Return(std::vector<std::shared_ptr<agent_t>>{ cpu_agent }));
    EXPECT_CALL(*g_externals_mock,
                add_pmc_info(AllOf(
                    Field(&pmc_info_data_t::type, Eq(externals::k_agent_type_gpu)),
                    Field(&pmc_info_data_t::agent_type_index, Eq(std::size_t{ 1 })),
                    Field(&pmc_info_data_t::target_arch, Eq(std::string{ "GPU" })),
                    Field(&pmc_info_data_t::name,
                          Eq(std::string{ externals::k_kfd_page_migrate_category_name })),
                    Field(&pmc_info_data_t::symbol,
                          Eq(std::string{ "KFD Page Migration Events" })),
                    Field(&pmc_info_data_t::description,
                          Eq(std::string{
                              externals::k_kfd_page_migrate_category_description })))))
        .Times(1);
    EXPECT_CALL(*g_externals_mock,
                add_pmc_info(AllOf(
                    Field(&pmc_info_data_t::type, Eq(externals::k_agent_type_cpu)),
                    Field(&pmc_info_data_t::agent_type_index, Eq(std::size_t{ 0 })),
                    Field(&pmc_info_data_t::target_arch, Eq(std::string{ "CPU" })),
                    Field(&pmc_info_data_t::name,
                          Eq(std::string{ externals::k_kfd_page_migrate_category_name })),
                    Field(&pmc_info_data_t::symbol,
                          Eq(std::string{ "KFD Page Migration Events" })),
                    Field(&pmc_info_data_t::description,
                          Eq(std::string{
                              externals::k_kfd_page_migrate_category_description })))))
        .Times(1);

    on_kfd_page_migrate_configure<externals>();

    g_externals_mock.reset();
}

}  // namespace rocprofsys::domains::buffered
