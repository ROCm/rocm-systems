// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "lib/aqlprofile/core/ip_discovery.h"

#include "packet_golden_options.hpp"
#include "packet_golden_utils.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <string_view>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using namespace aqlprofile::packet_dump;

constexpr auto* ip_discovery_path_env = "AQLPROFILE_CUSTOM_IP_DISCOVERY_PATH";
constexpr auto* fixture_root_path_env = "AQLPROFILE_PACKET_FIXTURE_PATH";
constexpr auto* packet_dump_path_env  = "AQLPROFILE_PACKET_DUMP_PATH";

std::string
to_hex(const std::vector<uint8_t>& bytes)
{
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for(size_t i = 0; i < bytes.size(); ++i)
    {
        if(i > 0) output << ',';
        output << std::setw(2) << static_cast<unsigned>(bytes.at(i));
    }
    return output.str();
}

void
write_packet(std::ostream&               output,
             uint32_t                    event_id,
             std::string_view            packet_name,
             const std::vector<uint8_t>& packet)
{
    output << event_id << ':' << packet_name << "_PACKET:" << packet.size() << ":["
           << to_hex(packet) << "]\n";
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

            std::ofstream output{output_directory / (test_case.block + ".txt"), std::ios::trunc};
            ASSERT_TRUE(output.is_open());

            for(const uint32_t event_id : test_case.events)
            {
                aqlprofile_pmc_event_t event{};
                event.block_index = 0;
                event.event_id    = event_id;
                event.block_name  = block;

                bool valid_event = false;
                ASSERT_EQ(aqlprofile_validate_pmc_event(agent, &event, &valid_event),
                          HSA_STATUS_SUCCESS);
                ASSERT_TRUE(valid_event);

                const auto streams =
                    generate_pmc_packet_streams(agent, event, deterministic_address_base);
                write_packet(output, event_id, "START", streams.start);
                write_packet(output, event_id, "READ", streams.read);
                write_packet(output, event_id, "STOP", streams.stop);
            }
        }
    }
}

TEST_F(aqlprofile_packet_golden_test, allocation_bases_only_change_address_fields)
{
    const auto&     options = get_packet_test_options();
    std::mt19937_64 generator{
        static_cast<uint64_t>(::testing::UnitTest::GetInstance()->random_seed())};
    std::uniform_int_distribution<uint64_t> high_slot_distribution{1, 1024};
    std::uniform_int_distribution<uint64_t> low_page_distribution{
        1, (deterministic_address_stride / 4096) - 1};

    std::vector<uintptr_t> allocation_bases = {
        deterministic_address_base + 0x1000,
        deterministic_address_base + deterministic_address_stride - 0x1000,
        deterministic_address_base + deterministic_address_stride,
        deterministic_address_base + deterministic_address_stride + 0x1000,
        deterministic_address_base + deterministic_address_stride + 0x2000,
    };
    for(size_t i = 0; i < 64; ++i)
    {
        allocation_bases.emplace_back(
            deterministic_address_base +
            (high_slot_distribution(generator) * deterministic_address_stride) +
            (low_page_distribution(generator) * 4096));
    }

    for(const auto& architecture : options.architectures)
    {
        SCOPED_TRACE(architecture.gfx);
        select_fixture(architecture.gfx);
        const aqlprofile_agent_handle_t agent = register_packet_agent(architecture.gfx);

        for(const auto& test_case : architecture.cases)
        {
            const auto block = resolve_block(agent, test_case.block);
            for(const uint32_t event_id : test_case.events)
            {
                aqlprofile_pmc_event_t event{};
                event.block_index = 0;
                event.event_id    = event_id;
                event.block_name  = block;

                const auto baseline =
                    generate_pmc_packet_streams(agent, event, deterministic_address_base);
                const auto normalized_baseline = normalize_pmc_read_addresses(baseline.read);
                for(const uintptr_t allocation_base : allocation_bases)
                {
                    SCOPED_TRACE(test_case.block + ":" + std::to_string(event_id) +
                                 " allocation_base=" + std::to_string(allocation_base));
                    const auto randomized =
                        generate_pmc_packet_streams(agent, event, allocation_base);

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
