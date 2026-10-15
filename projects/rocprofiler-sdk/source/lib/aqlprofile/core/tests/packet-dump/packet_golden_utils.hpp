// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "lib/aqlprofile/aqlprofile.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace aqlprofile
{
namespace packet_dump
{
inline constexpr uintptr_t deterministic_address_base   = 0x100000000000;
inline constexpr uintptr_t deterministic_address_stride = 0x100000000;
inline constexpr uintptr_t userspace_address_limit      = 0x800000000000;

size_t
system_page_size();

enum class packet_allocation_role
{
    output,
    command,
};

struct packet_allocation
{
    packet_allocation_role         role{};
    size_t                         observed_call_index = 0;
    uintptr_t                      base                = 0;
    size_t                         size_bytes          = 0;
    aqlprofile_buffer_desc_flags_t flags{};
};

enum class packet_phase
{
    read,
    start,
    stop,
};

struct packet_memory_copy
{
    packet_phase phase{};
    size_t       observed_call_index = 0;
    size_t       allocation_index    = 0;
    size_t       offset_bytes        = 0;
    size_t       size_bytes          = 0;
};

struct packet_sample_coordinate
{
    std::string name{};
    uint32_t    position   = 0;
    uint32_t    id         = 0;
    uint64_t    extent     = 0;
    uint64_t    coordinate = 0;
};

struct packet_sample
{
    aqlprofile_pmc_event_t                event{};
    uint64_t                              counter_id          = 0;
    size_t                                callback_index      = 0;
    size_t                                output_offset_bytes = 0;
    std::vector<packet_sample_coordinate> coordinates{};
};

struct packet_dimension
{
    std::string name{};
    uint32_t    position = 0;
    uint32_t    id       = 0;
    uint64_t    extent   = 0;
};

struct pmc_packet_streams
{
    std::vector<uint8_t>            start{};
    std::vector<uint8_t>            read{};
    std::vector<uint8_t>            stop{};
    aqlprofile_pmc_aql_packets_t    aql{};
    uintptr_t                       command_base = 0;
    std::vector<packet_allocation>  allocations{};
    std::vector<packet_memory_copy> memory_copies{};
    std::vector<packet_sample>      samples{};
    std::vector<size_t>             unreported_output_offsets{};
};

struct pm4_packet_range
{
    size_t   byte_offset = 0;
    size_t   byte_size   = 0;
    uint32_t opcode      = 0;
};

pmc_packet_streams
generate_pmc_packet_streams(aqlprofile_agent_handle_t     agent,
                            const aqlprofile_pmc_event_t& event,
                            uintptr_t                     allocation_base,
                            bool                          capture_samples = true);

pmc_packet_streams
generate_pmc_packet_streams(aqlprofile_agent_handle_t                  agent,
                            const std::vector<aqlprofile_pmc_event_t>& events,
                            uintptr_t                                  allocation_base,
                            bool                                       capture_samples = true);

std::vector<uint32_t>
normalize_pmc_read_addresses(const std::vector<uint8_t>& bytes);

bool
operator==(const packet_allocation& lhs, const packet_allocation& rhs);

bool
operator==(const packet_memory_copy& lhs, const packet_memory_copy& rhs);

bool
same_sample_layout(const pmc_packet_streams& lhs, const pmc_packet_streams& rhs);

std::vector<pm4_packet_range>
split_pm4_stream(const std::vector<uint8_t>& bytes);

aqlprofile_agent_info_v2_t
packet_agent_info(const std::string& gfx);

aqlprofile_agent_handle_t
register_packet_agent(const std::string& gfx);

hsa_ven_amd_aqlprofile_block_name_t
resolve_block(aqlprofile_agent_handle_t agent, const std::string& block);

uint32_t
block_counter_count(aqlprofile_agent_handle_t agent, hsa_ven_amd_aqlprofile_block_name_t block);

std::vector<packet_dimension>
event_dimensions(aqlprofile_agent_handle_t agent, const aqlprofile_pmc_event_t& event);

uint32_t
block_instance_count(const std::vector<packet_dimension>& dimensions);

std::vector<aqlprofile_pmc_event_t>
all_instance_events(hsa_ven_amd_aqlprofile_block_name_t block,
                    uint32_t                            event_id,
                    uint32_t                            instance_count);

}  // namespace packet_dump
}  // namespace aqlprofile
