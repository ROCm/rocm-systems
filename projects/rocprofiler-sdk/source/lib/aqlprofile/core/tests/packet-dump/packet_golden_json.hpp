// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "lib/aqlprofile/core/ip_discovery.h"

#include "packet_golden_options.hpp"
#include "packet_golden_utils.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace aqlprofile
{
namespace packet_dump
{
std::string
hex_dword(uint32_t value);

std::string
hex_offset(uint64_t offset);

std::vector<std::string>
to_hex_dwords(const std::vector<uint8_t>& bytes);

std::string_view
allocation_role_name(packet_allocation_role role);

nlohmann::ordered_json
allocations_json(const std::vector<packet_allocation>& allocations);

nlohmann::ordered_json
memory_copies_json(const std::vector<packet_memory_copy>& copies,
                   const std::vector<packet_allocation>&  allocations);

nlohmann::ordered_json
phase_json(const std::vector<uint8_t>&         stream,
           const hsa_ext_amd_aql_pm4_packet_t& aql,
           uintptr_t                           command_base);

nlohmann::ordered_json
request_json(const packet_test_event&                   test_event,
             const std::vector<aqlprofile_pmc_event_t>& events);

void
record_document_bases(nlohmann::ordered_json&               document,
                      const std::vector<packet_allocation>& allocations,
                      uintptr_t                             command_base);

void
write_json(const std::filesystem::path& path, const nlohmann::ordered_json& document);

nlohmann::ordered_json
make_pmc_json_document(const std::string&                   gfx,
                       const packet_test_case&              test_case,
                       hsa_ven_amd_aqlprofile_block_name_t  block,
                       uint32_t                             counter_count,
                       const std::vector<packet_dimension>& dimensions);

size_t
append_full_pmc_json(nlohmann::ordered_json&                    document,
                     const packet_test_event&                   test_event,
                     const std::vector<aqlprofile_pmc_event_t>& events,
                     const pmc_packet_streams&                  streams);

void
append_atomic_pmc_json(nlohmann::ordered_json&                    document,
                       const packet_test_event&                   test_event,
                       const std::vector<aqlprofile_pmc_event_t>& events,
                       const pmc_packet_streams&                  streams,
                       size_t                                     baseline_case_index);

void
append_instance_sweep_pmc_json(nlohmann::ordered_json&                    document,
                               const packet_test_event&                   test_event,
                               const std::vector<aqlprofile_pmc_event_t>& events,
                               const pmc_packet_streams&                  streams,
                               size_t                                     baseline_case_index);

void
write_pmc_json(const std::filesystem::path& path, const nlohmann::ordered_json& document);

void
write_topology_json(const std::filesystem::path& path,
                    const std::string&           gfx,
                    discovery_table_t            ip_discovery);

}  // namespace packet_dump
}  // namespace aqlprofile
