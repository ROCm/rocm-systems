// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "lib/aqlprofile/core/ip_discovery.h"

#include "packet_golden_json.hpp"
#include "packet_golden_options.hpp"
#include "packet_golden_utils.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <list>
#include <random>
#include <stdexcept>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using namespace aqlprofile::packet_dump;

constexpr auto* ip_discovery_path_env = "AQLPROFILE_CUSTOM_IP_DISCOVERY_PATH";
constexpr auto* fixture_root_path_env = "AQLPROFILE_PACKET_FIXTURE_PATH";
constexpr auto* packet_dump_path_env  = "AQLPROFILE_PACKET_DUMP_PATH";

uint32_t
load_le_dword(const uint8_t* bytes)
{
    return static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8) |
           (static_cast<uint32_t>(bytes[2]) << 16) | (static_cast<uint32_t>(bytes[3]) << 24);
}

std::vector<size_t>
differing_dwords(const std::vector<uint8_t>& lhs, const std::vector<uint8_t>& rhs)
{
    if(lhs.size() != rhs.size() || lhs.size() % sizeof(uint32_t) != 0)
        throw std::runtime_error{"cannot compare differently sized packet streams"};

    std::vector<size_t> differences{};
    for(size_t offset = 0; offset < lhs.size(); offset += sizeof(uint32_t))
        if(load_le_dword(lhs.data() + offset) != load_le_dword(rhs.data() + offset))
            differences.emplace_back(offset / sizeof(uint32_t));
    return differences;
}

struct failing_copy_capture
{
    std::list<std::vector<uint8_t>> allocations{};
    size_t                          deallocation_count   = 0;
    bool                            invalid_deallocation = false;
};

hsa_status_t
allocate_for_failing_copy(void**   pointer,
                          uint64_t size,
                          aqlprofile_buffer_desc_flags_t,
                          void* userdata) noexcept
{
    auto* capture = static_cast<failing_copy_capture*>(userdata);
    if(!capture || !pointer || size > std::numeric_limits<size_t>::max()) return HSA_STATUS_ERROR;
    try
    {
        auto& allocation = capture->allocations.emplace_back(static_cast<size_t>(size));
        *pointer         = allocation.data();
        return HSA_STATUS_SUCCESS;
    } catch(const std::bad_alloc&)
    {
        return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    } catch(...)
    {
        return HSA_STATUS_ERROR;
    }
}

void
deallocate_for_failing_copy(void* pointer, void* userdata) noexcept
{
    auto* capture = static_cast<failing_copy_capture*>(userdata);
    if(!capture) return;
    const auto allocation = std::find_if(
        capture->allocations.begin(), capture->allocations.end(), [pointer](const auto& entry) {
            return entry.data() == pointer;
        });
    if(allocation == capture->allocations.end())
    {
        capture->invalid_deallocation = true;
        return;
    }
    capture->allocations.erase(allocation);
    ++capture->deallocation_count;
}

hsa_status_t
reject_memory_copy(void*, const void*, size_t, void*) noexcept
{
    return HSA_STATUS_ERROR;
}

class aqlprofile_packet_golden_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const char* fixture_root_path = std::getenv(fixture_root_path_env);
        ASSERT_NE(fixture_root_path, nullptr) << fixture_root_path_env << " is not set";
        fixture_root_path_ = fixture_root_path;
        ASSERT_TRUE(fs::is_directory(fixture_root_path_)) << fixture_root_path_;

        const char* packet_dump_path = std::getenv(packet_dump_path_env);
        ASSERT_NE(packet_dump_path, nullptr) << packet_dump_path_env << " is not set";
        packet_dump_path_ = packet_dump_path;
    }

    void select_fixture(const std::string& gfx)
    {
        const fs::path fixture_path = fixture_root_path_ / gfx;
        ASSERT_TRUE(fs::is_directory(fixture_path)) << fixture_path;
        ASSERT_EQ(::setenv(ip_discovery_path_env, fixture_path.c_str(), 1), 0);
    }

    fs::path fixture_root_path_{};
    fs::path packet_dump_path_{};
};

TEST_F(aqlprofile_packet_golden_test, loads_ip_discovery_fixture)
{
    for(const auto& architecture : get_packet_test_options().architectures)
    {
        SCOPED_TRACE(architecture.gfx);
        select_fixture(architecture.gfx);
        const auto entries = parse_ip_discovery(0xFFFF, 0xFFFF);

        const size_t expected_instances = architecture.gfx == "gfx942" ? 8 : 1;
        ASSERT_EQ(entries.size(), expected_instances);
        EXPECT_TRUE(std::all_of(entries.begin(), entries.end(), [](const auto& entry) {
            return entry.die == 0 && entry.ipname == "gc";
        }));
    }
}

TEST_F(aqlprofile_packet_golden_test, copy_failure_releases_packet_allocations)
{
    const auto& architecture = get_packet_test_options().architectures.front();
    select_fixture(architecture.gfx);
    const aqlprofile_agent_handle_t agent     = register_packet_agent(architecture.gfx);
    const auto&                     test_case = architecture.cases.front();

    aqlprofile_pmc_event_t event{};
    event.event_id   = test_case.events.front().id;
    event.block_name = resolve_block(agent, test_case.block);

    failing_copy_capture           capture{};
    aqlprofile_handle_t            handle{};
    aqlprofile_pmc_aql_packets_t   packets{};
    const aqlprofile_pmc_profile_t profile{agent, &event, 1};
    EXPECT_EQ(aqlprofile_pmc_create_packets(&handle,
                                            &packets,
                                            profile,
                                            allocate_for_failing_copy,
                                            deallocate_for_failing_copy,
                                            reject_memory_copy,
                                            &capture),
              HSA_STATUS_ERROR);
    EXPECT_FALSE(capture.invalid_deallocation);
    EXPECT_EQ(capture.deallocation_count, 2);
    EXPECT_TRUE(capture.allocations.empty());
}

TEST_F(aqlprofile_packet_golden_test, dump_topology)
{
    for(const auto& architecture : get_packet_test_options().architectures)
    {
        SCOPED_TRACE(architecture.gfx);
        select_fixture(architecture.gfx);

        const fs::path output_directory = packet_dump_path_ / architecture.gfx;
        ASSERT_TRUE(fs::create_directories(output_directory) || fs::is_directory(output_directory));

        std::error_code error{};
        fs::remove(output_directory / "topology.json", error);
        ASSERT_FALSE(error);

        write_topology_json(output_directory / (architecture.gfx + "_ip_discovery.json"),
                            architecture.gfx,
                            parse_ip_discovery(0xFFFF, 0xFFFF));
    }
}

TEST_F(aqlprofile_packet_golden_test, dump_packets)
{
    const auto& options = get_packet_test_options();
    for(const auto& architecture : options.architectures)
    {
        SCOPED_TRACE(architecture.gfx);
        select_fixture(architecture.gfx);
        const aqlprofile_agent_handle_t agent = register_packet_agent(architecture.gfx);

        const fs::path output_directory = packet_dump_path_ / architecture.gfx;
        ASSERT_TRUE(fs::create_directories(output_directory) || fs::is_directory(output_directory));

        for(const auto& test_case : architecture.cases)
        {
            const auto block = resolve_block(agent, test_case.block);

            aqlprofile_pmc_event_t dimension_event{};
            dimension_event.event_id   = test_case.events.front().id;
            dimension_event.block_name = block;
            const auto dimensions      = event_dimensions(agent, dimension_event);
            auto       json_document   = make_pmc_json_document(
                architecture.gfx, test_case, block, block_counter_count(agent, block), dimensions);

            const uint32_t instance_count = block_instance_count(dimensions);
            for(uint32_t block_index = 0; block_index < instance_count; ++block_index)
            {
                const auto&            baseline_test_event = test_case.events.front();
                aqlprofile_pmc_event_t baseline_event{};
                baseline_event.block_index = block_index;
                baseline_event.event_id    = baseline_test_event.id;
                baseline_event.block_name  = block;

                bool valid_event = false;
                ASSERT_EQ(aqlprofile_validate_pmc_event(agent, &baseline_event, &valid_event),
                          HSA_STATUS_SUCCESS);
                ASSERT_TRUE(valid_event);

                const auto baseline =
                    generate_pmc_packet_streams(agent, baseline_event, deterministic_address_base);
                const size_t baseline_case_index =
                    append_full_pmc_json(json_document,
                                         baseline_test_event,
                                         std::vector<aqlprofile_pmc_event_t>{baseline_event},
                                         baseline);

                size_t changed_start_dword = std::numeric_limits<size_t>::max();
                for(size_t event_index = 1; event_index < test_case.events.size(); ++event_index)
                {
                    const auto& test_event = test_case.events.at(event_index);
                    SCOPED_TRACE(test_case.block + ":" + test_event.name +
                                 ":block_index=" + std::to_string(block_index));

                    aqlprofile_pmc_event_t event{};
                    event.block_index = block_index;
                    event.event_id    = test_event.id;
                    event.block_name  = block;
                    valid_event       = false;
                    ASSERT_EQ(aqlprofile_validate_pmc_event(agent, &event, &valid_event),
                              HSA_STATUS_SUCCESS);
                    ASSERT_TRUE(valid_event);

                    const auto streams =
                        generate_pmc_packet_streams(agent, event, deterministic_address_base);
                    ASSERT_EQ(streams.read, baseline.read);
                    ASSERT_EQ(streams.stop, baseline.stop);
                    ASSERT_EQ(std::memcmp(&streams.aql, &baseline.aql, sizeof(streams.aql)), 0);

                    const auto differences = differing_dwords(streams.start, baseline.start);
                    ASSERT_EQ(differences.size(), 1);
                    if(changed_start_dword == std::numeric_limits<size_t>::max())
                        changed_start_dword = differences.front();
                    else
                        ASSERT_EQ(differences.front(), changed_start_dword);

                    append_atomic_pmc_json(json_document,
                                           test_event,
                                           std::vector<aqlprofile_pmc_event_t>{event},
                                           streams,
                                           baseline_case_index);
                }
            }

            if(instance_count > 1)
            {
                const auto& baseline_test_event = test_case.events.front();
                const auto  baseline_events =
                    all_instance_events(block, baseline_test_event.id, instance_count);
                for(const auto& event : baseline_events)
                {
                    bool valid_event = false;
                    ASSERT_EQ(aqlprofile_validate_pmc_event(agent, &event, &valid_event),
                              HSA_STATUS_SUCCESS);
                    ASSERT_TRUE(valid_event);
                }

                const auto baseline =
                    generate_pmc_packet_streams(agent, baseline_events, deterministic_address_base);
                const size_t baseline_case_index = append_full_pmc_json(
                    json_document, baseline_test_event, baseline_events, baseline);
                std::vector<size_t> changed_start_dwords{};

                for(size_t event_index = 1; event_index < test_case.events.size(); ++event_index)
                {
                    const auto& test_event = test_case.events.at(event_index);
                    SCOPED_TRACE(test_case.block + ":" + test_event.name + ":all-instances");
                    const auto events = all_instance_events(block, test_event.id, instance_count);
                    for(const auto& event : events)
                    {
                        bool valid_event = false;
                        ASSERT_EQ(aqlprofile_validate_pmc_event(agent, &event, &valid_event),
                                  HSA_STATUS_SUCCESS);
                        ASSERT_TRUE(valid_event);
                    }

                    const auto streams =
                        generate_pmc_packet_streams(agent, events, deterministic_address_base);
                    ASSERT_EQ(streams.read, baseline.read);
                    ASSERT_EQ(streams.stop, baseline.stop);
                    ASSERT_EQ(std::memcmp(&streams.aql, &baseline.aql, sizeof(streams.aql)), 0);
                    ASSERT_EQ(streams.allocations, baseline.allocations);
                    ASSERT_EQ(streams.memory_copies, baseline.memory_copies);
                    ASSERT_TRUE(same_sample_layout(streams, baseline));
                    ASSERT_EQ(streams.unreported_output_offsets,
                              baseline.unreported_output_offsets);

                    const auto differences = differing_dwords(streams.start, baseline.start);
                    ASSERT_EQ(differences.size(), instance_count);
                    if(changed_start_dwords.empty())
                        changed_start_dwords = differences;
                    else
                        ASSERT_EQ(differences, changed_start_dwords);

                    append_instance_sweep_pmc_json(
                        json_document, test_event, events, streams, baseline_case_index);
                }
            }
            write_pmc_json(output_directory / (test_case.block + ".json"), json_document);
        }
    }
}

void
expect_event_id_packet_invariants(aqlprofile_agent_handle_t           agent,
                                  hsa_ven_amd_aqlprofile_block_name_t block,
                                  const packet_test_case&             test_case,
                                  uint32_t                            block_index)
{
    if(test_case.events.size() < 2) return;
    SCOPED_TRACE("block_index=" + std::to_string(block_index));

    aqlprofile_pmc_event_t baseline_event{};
    baseline_event.block_index = block_index;
    baseline_event.event_id    = test_case.events.front().id;
    baseline_event.block_name  = block;
    const auto baseline =
        generate_pmc_packet_streams(agent, baseline_event, deterministic_address_base, false);

    size_t changed_start_dword = std::numeric_limits<size_t>::max();
    for(size_t i = 1; i < test_case.events.size(); ++i)
    {
        const auto& test_event = test_case.events.at(i);
        SCOPED_TRACE(test_event.name + ":" + std::to_string(test_event.id));

        aqlprofile_pmc_event_t event{};
        event.block_index = block_index;
        event.event_id    = test_event.id;
        event.block_name  = block;
        const auto current =
            generate_pmc_packet_streams(agent, event, deterministic_address_base, false);

        EXPECT_EQ(current.read, baseline.read);
        EXPECT_EQ(current.stop, baseline.stop);
        EXPECT_EQ(std::memcmp(&current.aql, &baseline.aql, sizeof(current.aql)), 0);

        const auto differences = differing_dwords(current.start, baseline.start);
        ASSERT_EQ(differences.size(), 1);
        if(changed_start_dword == std::numeric_limits<size_t>::max())
            changed_start_dword = differences.front();
        else
            EXPECT_EQ(differences.front(), changed_start_dword);
    }
}

TEST_F(aqlprofile_packet_golden_test, event_id_packet_invariants_at_instance_zero)
{
    for(const auto& architecture : get_packet_test_options().architectures)
    {
        SCOPED_TRACE(architecture.gfx);
        select_fixture(architecture.gfx);
        const aqlprofile_agent_handle_t agent = register_packet_agent(architecture.gfx);

        for(const auto& test_case : architecture.cases)
        {
            SCOPED_TRACE(test_case.block);
            expect_event_id_packet_invariants(
                agent, resolve_block(agent, test_case.block), test_case, 0);
        }
    }
}

TEST_F(aqlprofile_packet_golden_test, event_id_packet_invariants_at_nonzero_instances)
{
    for(const auto& architecture : get_packet_test_options().architectures)
    {
        SCOPED_TRACE(architecture.gfx);
        select_fixture(architecture.gfx);
        const aqlprofile_agent_handle_t agent = register_packet_agent(architecture.gfx);

        for(const auto& test_case : architecture.cases)
        {
            SCOPED_TRACE(test_case.block);
            const auto block = resolve_block(agent, test_case.block);

            aqlprofile_pmc_event_t event{};
            event.event_id        = test_case.events.front().id;
            event.block_name      = block;
            const auto dimensions = event_dimensions(agent, event);
            const auto instance =
                std::find_if(dimensions.begin(), dimensions.end(), [](const auto& dimension) {
                    return dimension.name == "INSTANCE";
                });
            const uint64_t instance_count =
                instance == dimensions.end() ? uint64_t{1} : instance->extent;

            for(uint32_t block_index = 1; block_index < instance_count; ++block_index)
                expect_event_id_packet_invariants(agent, block, test_case, block_index);
        }
    }
}

TEST_F(aqlprofile_packet_golden_test, combined_all_instance_event_id_packet_invariants)
{
    for(const auto& architecture : get_packet_test_options().architectures)
    {
        SCOPED_TRACE(architecture.gfx);
        select_fixture(architecture.gfx);
        const aqlprofile_agent_handle_t agent = register_packet_agent(architecture.gfx);

        for(const auto& test_case : architecture.cases)
        {
            if(test_case.events.size() < 2) continue;
            SCOPED_TRACE(test_case.block);
            const auto block = resolve_block(agent, test_case.block);

            aqlprofile_pmc_event_t dimension_event{};
            dimension_event.event_id   = test_case.events.front().id;
            dimension_event.block_name = block;
            const auto dimensions      = event_dimensions(agent, dimension_event);
            const auto instance_count  = block_instance_count(dimensions);
            if(instance_count < 2) continue;

            const auto baseline_events =
                all_instance_events(block, test_case.events.front().id, instance_count);
            const auto baseline =
                generate_pmc_packet_streams(agent, baseline_events, deterministic_address_base);
            std::vector<size_t> changed_start_dwords{};

            for(size_t event_index = 1; event_index < test_case.events.size(); ++event_index)
            {
                const auto& test_event = test_case.events.at(event_index);
                SCOPED_TRACE(test_event.name + ":" + std::to_string(test_event.id));
                const auto events = all_instance_events(block, test_event.id, instance_count);
                const auto current =
                    generate_pmc_packet_streams(agent, events, deterministic_address_base);

                EXPECT_EQ(current.read, baseline.read);
                EXPECT_EQ(current.stop, baseline.stop);
                EXPECT_EQ(std::memcmp(&current.aql, &baseline.aql, sizeof(current.aql)), 0);
                EXPECT_EQ(current.allocations, baseline.allocations);
                EXPECT_EQ(current.memory_copies, baseline.memory_copies);
                EXPECT_TRUE(same_sample_layout(current, baseline));
                EXPECT_EQ(current.unreported_output_offsets, baseline.unreported_output_offsets);

                const auto differences = differing_dwords(current.start, baseline.start);
                ASSERT_EQ(differences.size(), instance_count);
                if(changed_start_dwords.empty())
                    changed_start_dwords = differences;
                else
                    EXPECT_EQ(differences, changed_start_dwords);
            }
        }
    }
}

TEST_F(aqlprofile_packet_golden_test, allocation_bases_only_change_address_fields)
{
    const auto&  options   = get_packet_test_options();
    const size_t page_size = system_page_size();
    ASSERT_EQ(deterministic_address_base % page_size, 0);
    ASSERT_EQ(deterministic_address_stride % page_size, 0);
    std::mt19937_64 generator{
        static_cast<uint64_t>(::testing::UnitTest::GetInstance()->random_seed())};
    std::uniform_int_distribution<uint64_t> high_slot_distribution{1, 1024};
    std::uniform_int_distribution<uint64_t> low_page_distribution{
        1, (deterministic_address_stride / page_size) - 1};

    std::vector<uintptr_t> allocation_bases = {
        deterministic_address_base + page_size,
        deterministic_address_base + deterministic_address_stride - page_size,
        deterministic_address_base + deterministic_address_stride,
        deterministic_address_base + deterministic_address_stride + page_size,
        deterministic_address_base + deterministic_address_stride + 2 * page_size,
    };
    for(size_t i = 0; i < 64; ++i)
    {
        allocation_bases.emplace_back(
            deterministic_address_base +
            (high_slot_distribution(generator) * deterministic_address_stride) +
            (low_page_distribution(generator) * page_size));
    }

    for(const auto& architecture : options.architectures)
    {
        SCOPED_TRACE(architecture.gfx);
        select_fixture(architecture.gfx);
        const aqlprofile_agent_handle_t agent = register_packet_agent(architecture.gfx);

        for(const auto& test_case : architecture.cases)
        {
            const auto block = resolve_block(agent, test_case.block);
            for(const auto& test_event : test_case.events)
            {
                aqlprofile_pmc_event_t event{};
                event.block_index = 0;
                event.event_id    = test_event.id;
                event.block_name  = block;

                const auto baseline =
                    generate_pmc_packet_streams(agent, event, deterministic_address_base, false);
                const auto normalized_baseline = normalize_pmc_read_addresses(baseline.read);
                for(const uintptr_t allocation_base : allocation_bases)
                {
                    SCOPED_TRACE(test_case.block + ":" + std::to_string(test_event.id) +
                                 " allocation_base=" + std::to_string(allocation_base));
                    const auto randomized =
                        generate_pmc_packet_streams(agent, event, allocation_base, false);

                    EXPECT_EQ(baseline.start, randomized.start);
                    EXPECT_EQ(baseline.stop, randomized.stop);
                    EXPECT_NE(baseline.read, randomized.read);
                    EXPECT_EQ(normalized_baseline, normalize_pmc_read_addresses(randomized.read));
                }
            }
        }
    }
}

}  // namespace
