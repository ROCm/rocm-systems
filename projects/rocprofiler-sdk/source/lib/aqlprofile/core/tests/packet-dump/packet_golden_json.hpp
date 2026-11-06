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
#include <vector>

namespace aqlprofile
{
namespace packet_dump
{
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
