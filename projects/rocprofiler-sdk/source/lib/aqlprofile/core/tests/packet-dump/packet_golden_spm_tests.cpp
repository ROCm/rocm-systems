// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "packet_golden_fixture.hpp"
#include "packet_golden_options.hpp"
#include "packet_golden_spm_json.hpp"
#include "packet_golden_utils.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <new>
#include <optional>
#include <string>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using namespace aqlprofile::packet_dump;

constexpr uint64_t spm_buffer_size_bytes      = uint64_t{1} << 16;
constexpr uint64_t spm_sample_interval_cycles = 8192;

std::vector<aqlprofile_spm_parameter_t>
spm_dump_parameters(uint64_t buffer_size_bytes = spm_buffer_size_bytes)
{
    return {
        {AQLPROFILE_SPM_PARAMETER_TYPE_BUFFER_SIZE, buffer_size_bytes},
        {AQLPROFILE_SPM_PARAMETER_TYPE_SAMPLE_INTERVAL, spm_sample_interval_cycles},
        {AQLPROFILE_SPM_PARAMETER_TYPE_TIMEOUT, 0},
        {AQLPROFILE_SPM_PARAMETER_TYPE_SAMPLE_MODE, AQLPROFILE_SPM_PARAMETER_SAMPLE_MODE_SCLK},
    };
}

class aqlprofile_spm_packet_golden_test : public aqlprofile_packet_golden_test
{
protected:
    void SetUp() override
    {
        aqlprofile_packet_golden_test::SetUp();
        ASSERT_EQ(::setenv(spm_agent_check_override_env, "1", 1), 0);
    }
};

struct spm_block_events
{
    std::vector<packet_test_event> supported{};
    std::vector<spm_skipped_event> skipped{};
};

spm_block_events
classify_spm_events(aqlprofile_agent_handle_t           agent,
                    const packet_test_case&             test_case,
                    hsa_ven_amd_aqlprofile_block_name_t block)
{
    spm_block_events result{};
    for(const auto& test_event : test_case.skipped_events)
        result.skipped.push_back({test_event, "metric_name_contains_level_token"});
    for(const auto& test_event : test_case.events)
    {
        aqlprofile_pmc_event_t event{};
        event.event_id   = test_event.id;
        event.block_name = block;
        if(!aqlprofile_spm_is_event_supported(agent, event))
            result.skipped.push_back(
                {test_event, "aqlprofile_spm_is_event_supported_returned_false"});
        else
            result.supported.push_back(test_event);
    }
    std::stable_sort(result.skipped.begin(),
                     result.skipped.end(),
                     [](const auto& lhs, const auto& rhs) { return lhs.event.id < rhs.event.id; });
    return result;
}

aqlprofile_pmc_event_t
spm_event(const packet_block& block, uint32_t block_index, uint32_t event_id)
{
    aqlprofile_pmc_event_t event{};
    event.block_index = block_index;
    event.event_id    = event_id;
    event.block_name  = block.id;
    return event;
}

template <typename RequestEvents>
void
append_spm_event_sweep(nlohmann::ordered_json&                        document,
                       aqlprofile_agent_handle_t                      agent,
                       const std::vector<packet_test_event>&          test_events,
                       const std::vector<aqlprofile_spm_parameter_t>& parameters,
                       RequestEvents&&                                request_events)
{
    std::optional<spm_packet_streams> baseline{};
    size_t                            baseline_case_index = 0;
    for(const auto& test_event : test_events)
    {
        SCOPED_TRACE(test_event.name);
        const std::vector<aqlprofile_pmc_event_t> events = request_events(test_event);
        for(const auto& event : events)
            ASSERT_TRUE(aqlprofile_spm_is_event_supported(agent, event));

        auto streams =
            generate_spm_packet_streams(agent, events, parameters, deterministic_address_base);
        if(!baseline)
        {
            baseline_case_index =
                append_full_spm_json(document, test_event, events, parameters, streams);
            baseline = std::move(streams);
            continue;
        }

        ASSERT_EQ(streams.allocations, baseline->allocations);
        ASSERT_EQ(streams.memory_copies, baseline->memory_copies);
        ASSERT_EQ(std::memcmp(&streams.aql, &baseline->aql, sizeof(streams.aql)), 0);
        ASSERT_EQ(streams.stop, baseline->stop);
        ASSERT_EQ(streams.descriptor_allocation_index, baseline->descriptor_allocation_index);
        ASSERT_EQ(streams.descriptor_offset_bytes, baseline->descriptor_offset_bytes);
        ASSERT_EQ(streams.descriptor, baseline->descriptor);
        ASSERT_EQ(streams.decode_queries, baseline->decode_queries);
        append_referenced_spm_json(
            document, test_event, events, parameters, streams, baseline_case_index);
    }
}

hsa_status_t
collect_configurations(const aqlprofile_spm_available_configuration_t* configurations,
                       size_t                                          count,
                       void*                                           userdata) noexcept
{
    auto* result = static_cast<std::vector<aqlprofile_spm_available_configuration_t>*>(userdata);
    if(!result || (!configurations && count != 0)) return HSA_STATUS_ERROR;
    try
    {
        result->assign(configurations, configurations + count);
        return HSA_STATUS_SUCCESS;
    } catch(const std::bad_alloc&)
    {
        return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }
}

TEST_F(aqlprofile_spm_packet_golden_test, dump_spm_configurations)
{
    for(const auto& architecture : get_packet_test_options().spm)
    {
        SCOPED_TRACE(architecture.gfx);
        select_fixture(architecture.gfx);
        const aqlprofile_agent_handle_t agent = register_packet_agent(architecture.gfx);

        std::vector<aqlprofile_spm_available_configuration_t> configurations{};
        ASSERT_EQ(aqlprofile_spm_query_agent_configurations(
                      agent, collect_configurations, &configurations),
                  HSA_STATUS_SUCCESS);

        const fs::path output_directory = packet_dump_path_ / architecture.gfx / "spm";
        ASSERT_TRUE(fs::create_directories(output_directory) || fs::is_directory(output_directory));
        write_spm_configurations_json(
            output_directory / "configurations.json", architecture.gfx, configurations);
    }
}

TEST_F(aqlprofile_spm_packet_golden_test, dump_spm_packets)
{
    const auto parameters = spm_dump_parameters();
    for(const auto& architecture : get_packet_test_options().spm)
    {
        SCOPED_TRACE(architecture.gfx);
        select_fixture(architecture.gfx);
        const aqlprofile_agent_handle_t agent = register_packet_agent(architecture.gfx);

        const fs::path output_directory = packet_dump_path_ / architecture.gfx / "spm";
        ASSERT_TRUE(fs::create_directories(output_directory) || fs::is_directory(output_directory));

        for(const auto& test_case : architecture.cases)
        {
            SCOPED_TRACE(test_case.block);
            const auto block    = query_block(agent, test_case.block);
            const auto events   = classify_spm_events(agent, test_case, block.id);
            auto       document = make_spm_json_document(
                architecture.gfx, test_case.block, block, events.supported, events.skipped);

            for(uint32_t block_index = 0; block_index < block.instance_count; ++block_index)
            {
                SCOPED_TRACE("block_index=" + std::to_string(block_index));
                ASSERT_NO_FATAL_FAILURE(append_spm_event_sweep(
                    document, agent, events.supported, parameters, [&](const auto& test_event) {
                        return std::vector<aqlprofile_pmc_event_t>{
                            spm_event(block, block_index, test_event.id)};
                    }));
            }

            if(block.instance_count > 1)
            {
                SCOPED_TRACE("all-instances");
                ASSERT_NO_FATAL_FAILURE(append_spm_event_sweep(
                    document, agent, events.supported, parameters, [&](const auto& test_event) {
                        return all_instance_events(block.id, test_event.id, block.instance_count);
                    }));
            }

            write_spm_json(output_directory / (test_case.block + ".json"), document);
        }
    }
}

TEST_F(aqlprofile_spm_packet_golden_test, allocation_bases_only_change_aql_targets)
{
    const auto parameters       = spm_dump_parameters();
    const auto allocation_bases = relocated_allocation_bases(relocation_seed(), 16);

    for(const auto& architecture : get_packet_test_options().spm)
    {
        SCOPED_TRACE(architecture.gfx);
        select_fixture(architecture.gfx);
        const aqlprofile_agent_handle_t agent = register_packet_agent(architecture.gfx);

        for(const auto& test_case : architecture.cases)
        {
            SCOPED_TRACE(test_case.block);
            const auto block  = query_block(agent, test_case.block);
            const auto events = classify_spm_events(agent, test_case, block.id);
            for(const auto& test_event : events.supported)
            {
                const std::vector<aqlprofile_pmc_event_t> request{
                    spm_event(block, 0, test_event.id)};
                const auto baseline = generate_spm_packet_streams(
                    agent, request, parameters, deterministic_address_base);
                for(const uintptr_t allocation_base : allocation_bases)
                {
                    SCOPED_TRACE(test_event.name +
                                 " allocation_base=" + std::to_string(allocation_base));
                    const auto moved =
                        generate_spm_packet_streams(agent, request, parameters, allocation_base);

                    EXPECT_EQ(moved.start, baseline.start);
                    EXPECT_EQ(moved.stop, baseline.stop);
                    EXPECT_EQ(moved.memory_copies, baseline.memory_copies);
                    EXPECT_EQ(moved.descriptor, baseline.descriptor);
                    EXPECT_EQ(moved.decode_queries, baseline.decode_queries);
                    expect_only_ib_target_moved(baseline.aql.start_packet,
                                                baseline.command_base,
                                                moved.aql.start_packet,
                                                moved.command_base);
                    expect_only_ib_target_moved(baseline.aql.stop_packet,
                                                baseline.command_base,
                                                moved.aql.stop_packet,
                                                moved.command_base);
                }
            }
        }
    }
}

TEST_F(aqlprofile_spm_packet_golden_test, buffer_size_does_not_change_packets)
{
    const auto parameters         = spm_dump_parameters();
    const auto resized_parameters = spm_dump_parameters(spm_buffer_size_bytes * 16);

    for(const auto& architecture : get_packet_test_options().spm)
    {
        SCOPED_TRACE(architecture.gfx);
        select_fixture(architecture.gfx);
        const aqlprofile_agent_handle_t agent = register_packet_agent(architecture.gfx);

        for(const auto& test_case : architecture.cases)
        {
            SCOPED_TRACE(test_case.block);
            const auto block  = query_block(agent, test_case.block);
            const auto events = classify_spm_events(agent, test_case, block.id);
            if(events.supported.empty()) continue;

            const std::vector<aqlprofile_pmc_event_t> request{
                spm_event(block, 0, events.supported.front().id)};
            const auto baseline =
                generate_spm_packet_streams(agent, request, parameters, deterministic_address_base);
            const auto resized = generate_spm_packet_streams(
                agent, request, resized_parameters, deterministic_address_base);

            EXPECT_EQ(resized.start, baseline.start);
            EXPECT_EQ(resized.stop, baseline.stop);
            EXPECT_EQ(std::memcmp(&resized.aql, &baseline.aql, sizeof(resized.aql)), 0);
            EXPECT_EQ(resized.memory_copies, baseline.memory_copies);
            EXPECT_EQ(resized.descriptor, baseline.descriptor);
            EXPECT_EQ(resized.decode_queries, baseline.decode_queries);
        }
    }
}

}  // namespace
