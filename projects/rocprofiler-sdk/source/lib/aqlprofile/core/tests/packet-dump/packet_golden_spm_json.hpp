// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "packet_golden_options.hpp"
#include "packet_golden_utils.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace aqlprofile
{
namespace packet_dump
{
inline constexpr const char* spm_agent_check_override_env = "AQLPROFILE_SPM_OVERRIDE_AGENT_CHECK";

struct spm_skipped_event
{
    packet_test_event event{};
    std::string_view  reason{};
};

nlohmann::ordered_json
make_spm_json_document(const std::string&                    gfx,
                       const std::string&                    block_name,
                       const packet_block&                   block,
                       const std::vector<packet_test_event>& events,
                       const std::vector<spm_skipped_event>& skipped);

size_t
append_full_spm_json(nlohmann::ordered_json&                        document,
                     const packet_test_event&                       test_event,
                     const std::vector<aqlprofile_pmc_event_t>&     events,
                     const std::vector<aqlprofile_spm_parameter_t>& parameters,
                     const spm_packet_streams&                      streams);

void
append_referenced_spm_json(nlohmann::ordered_json&                        document,
                           const packet_test_event&                       test_event,
                           const std::vector<aqlprofile_pmc_event_t>&     events,
                           const std::vector<aqlprofile_spm_parameter_t>& parameters,
                           const spm_packet_streams&                      streams,
                           size_t                                         baseline_case_index);

void
write_spm_json(const std::filesystem::path& path, const nlohmann::ordered_json& document);

void
write_spm_configurations_json(
    const std::filesystem::path&                                 path,
    const std::string&                                           gfx,
    const std::vector<aqlprofile_spm_available_configuration_t>& configurations);

}  // namespace packet_dump
}  // namespace aqlprofile
