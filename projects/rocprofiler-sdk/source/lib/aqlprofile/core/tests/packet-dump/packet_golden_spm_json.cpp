// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "packet_golden_spm_json.hpp"

#include "packet_golden_json.hpp"

#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace aqlprofile
{
namespace packet_dump
{
namespace
{
using ordered_json = nlohmann::ordered_json;

ordered_json
harness_environment_json()
{
    const char* value = std::getenv(spm_agent_check_override_env);
    return {{spm_agent_check_override_env, value ? ordered_json(value) : ordered_json(nullptr)}};
}

ordered_json
harness_json()
{
    return {{"environment", harness_environment_json()}};
}

std::string_view
parameter_type_name(aqlprofile_spm_parameter_type_t type)
{
    switch(type)
    {
        case AQLPROFILE_SPM_PARAMETER_TYPE_BUFFER_SIZE:
            return "AQLPROFILE_SPM_PARAMETER_TYPE_BUFFER_SIZE";
        case AQLPROFILE_SPM_PARAMETER_TYPE_SAMPLE_INTERVAL:
            return "AQLPROFILE_SPM_PARAMETER_TYPE_SAMPLE_INTERVAL";
        case AQLPROFILE_SPM_PARAMETER_TYPE_TIMEOUT: return "AQLPROFILE_SPM_PARAMETER_TYPE_TIMEOUT";
        case AQLPROFILE_SPM_PARAMETER_TYPE_SAMPLE_MODE:
            return "AQLPROFILE_SPM_PARAMETER_TYPE_SAMPLE_MODE";
        case AQLPROFILE_SPM_PARAMETER_TYPE_LAST: break;
    }
    throw std::runtime_error{"unknown SPM parameter type"};
}

std::string_view
sample_mode_name(aqlprofile_spm_parameter_interval_mode_t mode)
{
    switch(mode)
    {
        case AQLPROFILE_SPM_PARAMETER_SAMPLE_MODE_SCLK:
            return "AQLPROFILE_SPM_PARAMETER_SAMPLE_MODE_SCLK";
        case AQLPROFILE_SPM_PARAMETER_SAMPLE_MODE_REFCLK:
            return "AQLPROFILE_SPM_PARAMETER_SAMPLE_MODE_REFCLK";
    }
    throw std::runtime_error{"unknown SPM sample mode"};
}

std::string_view
decode_query_name(aqlprofile_spm_decode_query_t query)
{
    switch(query)
    {
        case AQLPROFILE_SPM_DECODE_QUERY_SEG_SIZE: return "AQLPROFILE_SPM_DECODE_QUERY_SEG_SIZE";
        case AQLPROFILE_SPM_DECODE_QUERY_NUM_XCC: return "AQLPROFILE_SPM_DECODE_QUERY_NUM_XCC";
        case AQLPROFILE_SPM_DECODE_QUERY_EVENT_COUNT:
            return "AQLPROFILE_SPM_DECODE_QUERY_EVENT_COUNT";
        case AQLPROFILE_SPM_DECODE_QUERY_COUNTER_MAP_BYTE_OFFSET:
            return "AQLPROFILE_SPM_DECODE_QUERY_COUNTER_MAP_BYTE_OFFSET";
        case AQLPROFILE_SPM_DECODE_QUERY_LAST: break;
    }
    throw std::runtime_error{"unknown SPM decode query"};
}

ordered_json
parameter_type_json(aqlprofile_spm_parameter_type_t type)
{
    return {{"name", parameter_type_name(type)}, {"value", static_cast<uint32_t>(type)}};
}

ordered_json
spm_request_json(const packet_test_event&                       test_event,
                 const std::vector<aqlprofile_pmc_event_t>&     events,
                 const std::vector<aqlprofile_spm_parameter_t>& parameters)
{
    auto request_parameters = ordered_json::array();
    for(const auto& parameter : parameters)
        request_parameters.push_back({
            {"type", parameter_type_json(parameter.type)},
            {"value", parameter.value},
        });

    auto request          = request_json(test_event, events);
    request["parameters"] = std::move(request_parameters);
    return request;
}

ordered_json
spm_response_json(const spm_packet_streams& streams)
{
    auto decode_queries = ordered_json::object();
    for(size_t query = 0; query < streams.decode_queries.size(); ++query)
        decode_queries[std::string{decode_query_name(
            static_cast<aqlprofile_spm_decode_query_t>(query))}] = streams.decode_queries.at(query);

    const auto& descriptor_allocation = streams.allocations.at(streams.descriptor_allocation_index);
    return {
        {"allocations", allocations_json(streams.allocations)},
        {"memory_copies", memory_copies_json(streams.memory_copies, streams.allocations)},
        {"start", phase_json(streams.start, streams.aql.start_packet, streams.command_base)},
        {"stop", phase_json(streams.stop, streams.aql.stop_packet, streams.command_base)},
        {"descriptor",
         {
             {"allocation", allocation_role_name(descriptor_allocation.role)},
             {"offset_bytes", hex_offset(streams.descriptor_offset_bytes)},
             {"size_bytes", streams.descriptor.size()},
             {"uint32_t", to_hex_dwords(streams.descriptor)},
         }},
        {"decode_queries", std::move(decode_queries)},
    };
}

void
reference_baseline(ordered_json&       response,
                   const ordered_json& document,
                   size_t              baseline_case_index,
                   const std::string&  section)
{
    if(baseline_case_index >= document.at("spm").size())
        throw std::runtime_error{"SPM baseline reference is not backward"};

    const std::string baseline =
        "/spm/" + std::to_string(baseline_case_index) + "/response" + section;
    const ordered_json::json_pointer local{section};
    if(response.at(local) != document.at(ordered_json::json_pointer{baseline}))
        throw std::runtime_error{"SPM response section " + section + " differs from its baseline"};
    response[local] = {{"$ref", "#" + baseline}};
}
}  // namespace

ordered_json
make_spm_json_document(const std::string&                    gfx,
                       const std::string&                    block_name,
                       const packet_block&                   block,
                       const std::vector<packet_test_event>& events,
                       const std::vector<spm_skipped_event>& skipped)
{
    ordered_json document{
        {"format_version", 1},
        {"gfx", gfx},
        {"topology", "../" + gfx + "_ip_discovery.json"},
        {"configurations", "configurations.json"},
        {"harness", harness_json()},
        {"block",
         {
             {"name", block_name},
             {"__implementation_aqlprofile:block_id", static_cast<uint32_t>(block.id)},
             {"instance_count", block.instance_count},
         }},
        {"events", ordered_json::array()},
        {"skipped", ordered_json::array()},
        {"byte_order", "little-endian"},
        {"output_base", nullptr},
        {"command_base", nullptr},
        {"spm", ordered_json::array()},
    };
    for(const auto& event : events)
        document["events"].push_back({{"name", event.name}, {"event_id", event.id}});
    for(const auto& entry : skipped)
        document["skipped"].push_back({
            {"name", entry.event.name},
            {"event_id", entry.event.id},
            {"reason", entry.reason},
        });
    return document;
}

size_t
append_full_spm_json(ordered_json&                                  document,
                     const packet_test_event&                       test_event,
                     const std::vector<aqlprofile_pmc_event_t>&     events,
                     const std::vector<aqlprofile_spm_parameter_t>& parameters,
                     const spm_packet_streams&                      streams)
{
    record_document_bases(document, streams.allocations, streams.command_base);
    const size_t case_index = document.at("spm").size();
    document["spm"].push_back({
        {"request", spm_request_json(test_event, events, parameters)},
        {"response", spm_response_json(streams)},
    });
    return case_index;
}

void
append_referenced_spm_json(ordered_json&                                  document,
                           const packet_test_event&                       test_event,
                           const std::vector<aqlprofile_pmc_event_t>&     events,
                           const std::vector<aqlprofile_spm_parameter_t>& parameters,
                           const spm_packet_streams&                      streams,
                           size_t                                         baseline_case_index)
{
    record_document_bases(document, streams.allocations, streams.command_base);

    auto response = spm_response_json(streams);
    for(const auto* section : {"/allocations",
                               "/memory_copies",
                               "/start/aql_packet",
                               "/stop",
                               "/descriptor",
                               "/decode_queries"})
        reference_baseline(response, document, baseline_case_index, section);

    document["spm"].push_back({
        {"request", spm_request_json(test_event, events, parameters)},
        {"response", std::move(response)},
    });
}

void
write_spm_json(const std::filesystem::path& path, const ordered_json& document)
{
    if(!document.at("spm").empty() &&
       (document.at("output_base").is_null() || document.at("command_base").is_null()))
        throw std::runtime_error{"SPM dump allocation bases were not recorded"};
    write_json(path, document);
}

void
write_spm_configurations_json(
    const std::filesystem::path&                                 path,
    const std::string&                                           gfx,
    const std::vector<aqlprofile_spm_available_configuration_t>& configurations)
{
    auto entries = ordered_json::array();
    for(const auto& configuration : configurations)
    {
        if(configuration.type != AQLPROFILE_SPM_PARAMETER_TYPE_SAMPLE_INTERVAL)
            throw std::runtime_error{"unsupported SPM configuration type"};
        entries.push_back({
            {"type", parameter_type_json(configuration.type)},
            {"interval",
             {
                 {"min_interval", configuration.interval.min_interval},
                 {"max_interval", configuration.interval.max_interval},
                 {"mode",
                  {
                      {"name", sample_mode_name(configuration.interval.mode)},
                      {"value", static_cast<uint32_t>(configuration.interval.mode)},
                  }},
             }},
        });
    }

    write_json(path,
               {
                   {"format_version", 1},
                   {"gfx", gfx},
                   {"topology", "../" + gfx + "_ip_discovery.json"},
                   {"harness", harness_json()},
                   {"configurations", std::move(entries)},
               });
}

}  // namespace packet_dump
}  // namespace aqlprofile
