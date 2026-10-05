// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "lib/aqlprofile/aqlprofile.hpp"

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

struct pmc_packet_streams
{
    std::vector<uint8_t> start{};
    std::vector<uint8_t> read{};
    std::vector<uint8_t> stop{};
};

pmc_packet_streams
generate_pmc_packet_streams(aqlprofile_agent_handle_t     agent,
                            const aqlprofile_pmc_event_t& event,
                            uintptr_t                     allocation_base);

std::vector<uint32_t>
normalize_pmc_read_addresses(const std::vector<uint8_t>& bytes);

aqlprofile_agent_handle_t
register_packet_agent(const std::string& gfx);

hsa_ven_amd_aqlprofile_block_name_t
resolve_block(aqlprofile_agent_handle_t agent, const std::string& block);

}  // namespace packet_dump
}  // namespace aqlprofile
