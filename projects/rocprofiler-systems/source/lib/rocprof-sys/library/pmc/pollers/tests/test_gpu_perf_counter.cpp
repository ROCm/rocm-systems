// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/agent.hpp"
#include "core/collection/tests/log_capture.hpp"
#include "core/collection/tests/recording_sink.hpp"
#include "library/pmc/collectors/gpu_perf_counter/device.hpp"
#include "library/pmc/collectors/gpu_perf_counter/tests/mock_backend.hpp"
#include "library/pmc/collectors/gpu_perf_counter/types.hpp"
#include "library/pmc/common/types.hpp"
#include "library/pmc/pollers/gpu_perf_counter.hpp"
#include "rocprofiler-systems/categories.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace rocprofsys::pmc::pollers
{
namespace
{
namespace perf_counter = pmc::collectors::gpu_perf_counter;
namespace sdk_mock     = backends::rocprofiler_sdk::testing;

using mock_impl = ::testing::StrictMock<sdk_mock::mock_backend_impl>;
using sdk_mock::mock_backend;
using device_t = perf_counter::device<mock_backend>;
using collection::testing::log_capture;
using collection::testing::recording_sink;
using ::testing::_;
using ::testing::Throw;

constexpr std::uint64_t k_waves_id       = 10;
constexpr std::uint64_t k_valu_id        = 20;
constexpr double        k_waves_value    = 42.0;
constexpr std::size_t   k_gpu_index      = 2;
constexpr std::uint64_t k_timestamp      = 1000;
constexpr std::uint64_t k_next_timestamp = 2000;

struct stub_settings
{};

struct stub_provider
{
    using backend_t = mock_backend;

    std::vector<std::shared_ptr<device_t>> devices;
    bool                                   shutdown_called = false;

    template <typename Device>
    std::vector<std::shared_ptr<Device>> get_devices(device_type /*type*/)
    {
        static_assert(std::is_same_v<Device, device_t>);
        return devices;
    }

    void shutdown() { shutdown_called = true; }
};

using poller_t = gpu_perf_counter<stub_provider, stub_settings>;

std::vector<perf_counter::counter_metadata>
counters()
{
    return { { .counter_id  = k_waves_id,
               .name        = "SQ_WAVES",
               .description = "Waves",
               .block       = "SQ",
               .expression  = "",
               .is_constant = false,
               .is_derived  = false,
               .dimensions  = {} },
             { .counter_id  = k_valu_id,
               .name        = "SQ_INSTS_VALU",
               .description = "",
               .block       = "SQ",
               .expression  = "SQ_INSTS_VALU_ADD",
               .is_constant = false,
               .is_derived  = true,
               .dimensions  = {} } };
}

class gpu_perf_counter_poller_test_interface : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_impl = std::make_shared<mock_impl>();
        mock_backend::bind(m_impl);
    }

    void TearDown() override
    {
        m_poller.reset();
        m_provider.reset();
        mock_backend::unbind();
    }

    [[nodiscard]] std::shared_ptr<device_t> make_device(
        std::size_t index, std::vector<perf_counter::counter_metadata> meta) const
    {
        const auto gpu         = std::make_shared<agent>();
        gpu->type              = agent_type::gpu;
        gpu->device_type_index = index;
        return std::make_shared<device_t>(
            std::make_shared<mock_backend>(), mock_backend::context_id_t{ index }, gpu,
            mock_backend::counter_config_id_t{}, std::move(meta));
    }

    [[nodiscard]] bool describe(std::vector<std::shared_ptr<device_t>> devices)
    {
        m_provider          = std::make_shared<stub_provider>();
        m_provider->devices = std::move(devices);
        m_poller            = std::make_unique<poller_t>(m_provider);
        return m_poller->describe(m_sink);
    }

    /** The next sample of the bound context reports `records` (counter id, value). */
    void expect_sample(std::vector<sdk_mock::mock_counter_record> records) const
    {
        EXPECT_CALL(*m_impl, sample_device_counting_service(_, _, _, _, _))
            .WillOnce([records](mock_backend::context_id_t, mock_backend::user_data_t,
                                mock_backend::counter_flag_t,
                                mock_backend::counter_record_t* out, std::size_t* count) {
                const std::span<mock_backend::counter_record_t> output{ out, *count };
                std::ranges::copy(records, output.begin());
                *count = records.size();
                return mock_backend::status_success;
            })
            .RetiresOnSaturation();
        if(records.empty())
        {
            return;
        }
        EXPECT_CALL(*m_impl, query_record_counter_id(_, _))
            .Times(static_cast<int>(records.size()))
            .WillRepeatedly([](mock_backend::counter_record_t record,
                               mock_backend::counter_id_t*    counter_id) {
                counter_id->handle = record.id;
                return mock_backend::status_success;
            })
            .RetiresOnSaturation();
    }

    /** Each sample reports the records of its context (= GPU index). */
    void expect_samples_by_context(
        std::map<std::uint64_t, std::vector<sdk_mock::mock_counter_record>> by_context)
        const
    {
        std::size_t record_count = 0;
        for(const auto& [context, records] : by_context)
        {
            record_count += records.size();
        }
        EXPECT_CALL(*m_impl, sample_device_counting_service(_, _, _, _, _))
            .Times(static_cast<int>(by_context.size()))
            .WillRepeatedly(
                [by_context](mock_backend::context_id_t context,
                             mock_backend::user_data_t, mock_backend::counter_flag_t,
                             mock_backend::counter_record_t* out, std::size_t* count) {
                    const auto& records = by_context.at(context.handle);
                    std::ranges::copy(records, std::span{ out, *count }.begin());
                    *count = records.size();
                    return mock_backend::status_success;
                });
        EXPECT_CALL(*m_impl, query_record_counter_id(_, _))
            .Times(static_cast<int>(record_count))
            .WillRepeatedly([](mock_backend::counter_record_t record,
                               mock_backend::counter_id_t*    counter_id) {
                counter_id->handle = record.id;
                return mock_backend::status_success;
            });
    }

    std::shared_ptr<mock_impl>     m_impl;
    recording_sink                 m_sink;
    std::shared_ptr<stub_provider> m_provider;
    std::unique_ptr<poller_t>      m_poller;
};

TEST_F(gpu_perf_counter_poller_test_interface, columns_are_the_counters_of_the_device)
{
    ASSERT_TRUE(describe({ make_device(k_gpu_index, counters()) }));

    const auto schemas = m_sink.schemas();
    ASSERT_EQ(schemas.size(), 1U);
    const auto& description = schemas.front();
    EXPECT_EQ(description.agent, agent_type::gpu);
    EXPECT_EQ(description.device, k_gpu_index);
    EXPECT_EQ(description.target_arch, "GPU");
    EXPECT_EQ(description.event_category, "rocm_counter_collection");
    ASSERT_EQ(description.columns.size(), 2U);
    const auto& waves = description.columns[0];
    EXPECT_EQ(waves.pmc_name, "SQ_WAVES");
    EXPECT_EQ(waves.symbol, "SQ_WAVES");
    EXPECT_EQ(waves.description, "Waves");
    EXPECT_EQ(waves.unit, "count");
    EXPECT_EQ(waves.block, "SQ");
    EXPECT_EQ(waves.perfetto_track_name, "GPU [2] SQ_WAVES (S)");
    EXPECT_EQ(waves.rocpd_track_name, "GPU [2] SQ_WAVES (S)");
    EXPECT_EQ(waves.category_id, ROCPROFSYS_CATEGORY_ROCM_COUNTER_COLLECTION);
    const auto& valu = description.columns[1];
    EXPECT_EQ(valu.description, "SDK PMC hardware counter");
    EXPECT_EQ(valu.expression, "SQ_INSTS_VALU_ADD");
    EXPECT_TRUE(valu.is_derived);
    EXPECT_FALSE(valu.is_constant);
}

TEST_F(gpu_perf_counter_poller_test_interface,
       counter_not_reported_in_a_poll_has_no_value)
{
    ASSERT_TRUE(describe({ make_device(k_gpu_index, counters()) }));
    EXPECT_CALL(*m_impl, start_context(_));
    expect_sample({ { .id = k_waves_id, .counter_value = k_waves_value } });

    m_poller->poll(k_timestamp, m_sink);

    const auto batches = m_sink.batches();
    ASSERT_EQ(batches.size(), 1U);
    EXPECT_EQ(batches.front().timestamp, k_timestamp);
    const auto& values = batches.front().values;
    ASSERT_EQ(values.size(), 2U);
    EXPECT_EQ(values[0], k_waves_value);
    EXPECT_TRUE(std::isnan(values[1]));
}

TEST_F(gpu_perf_counter_poller_test_interface, poll_without_records_writes_no_row)
{
    ASSERT_TRUE(describe({ make_device(k_gpu_index, counters()) }));
    EXPECT_CALL(*m_impl, start_context(_));
    expect_sample({});

    m_poller->poll(k_timestamp, m_sink);

    EXPECT_TRUE(m_sink.batches().empty());
}

TEST_F(gpu_perf_counter_poller_test_interface, failed_gpu_is_disabled)
{
    ASSERT_TRUE(describe({ make_device(k_gpu_index, counters()) }));
    EXPECT_CALL(*m_impl, start_context(_));
    EXPECT_CALL(*m_impl, sample_device_counting_service(_, _, _, _, _))
        .WillOnce(Throw(std::runtime_error("device lost")));

    const log_capture logs;
    m_poller->poll(k_timestamp, m_sink);
    m_poller->poll(k_next_timestamp, m_sink);

    EXPECT_TRUE(m_sink.batches().empty());
    EXPECT_EQ(logs.count("error", "gpu_pmc device"), 1U);
}

TEST_F(gpu_perf_counter_poller_test_interface,
       describe_is_false_without_counters_and_shuts_down)
{
    EXPECT_FALSE(describe({ make_device(k_gpu_index, {}) }));

    EXPECT_TRUE(m_sink.schemas().empty());
    EXPECT_TRUE(m_provider->shutdown_called);
}

TEST_F(gpu_perf_counter_poller_test_interface, release_shuts_the_provider_down)
{
    ASSERT_TRUE(describe({ make_device(k_gpu_index, counters()) }));

    m_poller->release();
    m_poller->poll(k_timestamp, m_sink);

    EXPECT_TRUE(m_provider->shutdown_called);
    EXPECT_TRUE(m_sink.batches().empty());
}
TEST_F(gpu_perf_counter_poller_test_interface, each_gpu_writes_only_its_own_counters)
{
    constexpr std::size_t k_other_gpu   = 5;
    constexpr double      k_other_waves = 7.0;
    constexpr double      k_other_valu  = 3.0;
    ASSERT_TRUE(describe(
        { make_device(k_gpu_index, counters()), make_device(k_other_gpu, counters()) }));
    EXPECT_CALL(*m_impl, start_context(_)).Times(2);
    expect_samples_by_context(
        { { k_gpu_index, { { .id = k_waves_id, .counter_value = k_waves_value } } },
          { k_other_gpu,
            { { .id = k_waves_id, .counter_value = k_other_waves },
              { .id = k_valu_id, .counter_value = k_other_valu } } } });

    m_poller->poll(k_timestamp, m_sink);

    const auto schemas = m_sink.schemas();
    ASSERT_EQ(schemas.size(), 2U);
    EXPECT_EQ(schemas[1].device, k_other_gpu);
    EXPECT_EQ(schemas[1].columns.front().perfetto_track_name, "GPU [5] SQ_WAVES (S)");
    const auto first  = m_sink.batches_of(0);
    const auto second = m_sink.batches_of(1);
    ASSERT_EQ(first.size(), 1U);
    ASSERT_EQ(second.size(), 1U);
    EXPECT_EQ(first.front().values[0], k_waves_value);
    EXPECT_TRUE(std::isnan(first.front().values[1]));
    EXPECT_THAT(second.front().values,
                ::testing::ElementsAre(k_other_waves, k_other_valu));
}

TEST_F(gpu_perf_counter_poller_test_interface,
       failed_sample_writes_no_row_and_keeps_the_gpu)
{
    ASSERT_TRUE(describe({ make_device(k_gpu_index, counters()) }));
    EXPECT_CALL(*m_impl, start_context(_));
    EXPECT_CALL(*m_impl, sample_device_counting_service(_, _, _, _, _))
        .WillOnce(::testing::Return(mock_backend::status_error))
        .RetiresOnSaturation();

    m_poller->poll(k_timestamp, m_sink);
    EXPECT_TRUE(m_sink.batches().empty());

    expect_sample({ { .id = k_waves_id, .counter_value = k_waves_value } });
    m_poller->poll(k_next_timestamp, m_sink);

    const auto batches = m_sink.batches();
    ASSERT_EQ(batches.size(), 1U);
    EXPECT_EQ(batches.front().timestamp, k_next_timestamp);
}
}  // namespace
}  // namespace rocprofsys::pmc::pollers
