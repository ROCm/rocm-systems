// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "packet_golden_json.hpp"

#include "lib/aqlprofile/core/commandbuffermgr.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <utility>

namespace aqlprofile
{
namespace packet_dump
{
namespace
{
namespace fs       = std::filesystem;
using ordered_json = nlohmann::ordered_json;

const size_t command_section_alignment_bytes = aql_profile::CommandBufferMgr::Align(1);

uint32_t
load_le_dword(const uint8_t* bytes)
{
    return static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8) |
           (static_cast<uint32_t>(bytes[2]) << 16) | (static_cast<uint32_t>(bytes[3]) << 24);
}

std::string
hex_dword(uint32_t value)
{
    std::ostringstream output;
    output << "0x" << std::hex << std::setfill('0') << std::setw(8) << value;
    return output.str();
}

std::string
hex_address(uint64_t address)
{
    std::ostringstream output;
    output << "0x" << std::hex << std::setfill('0') << std::setw(16) << address;
    return output.str();
}

std::string
hex_offset(uint64_t offset)
{
    std::ostringstream output;
    output << "0x" << std::hex << offset;
    return output.str();
}

std::vector<std::string>
to_hex_dwords(const uint8_t* bytes, size_t size)
{
    if(size % sizeof(uint32_t) != 0) throw std::runtime_error{"packet size is not dword aligned"};

    std::vector<std::string> dwords{};
    dwords.reserve(size / sizeof(uint32_t));
    for(size_t i = 0; i < size; i += sizeof(uint32_t))
        dwords.emplace_back(hex_dword(load_le_dword(bytes + i)));
    return dwords;
}

std::vector<std::string>
to_hex_dwords(const std::vector<uint8_t>& bytes)
{
    return to_hex_dwords(bytes.data(), bytes.size());
}

struct decoded_aql_packet
{
    std::vector<uint8_t> bytes{};
    uint64_t             target_offset = 0;
    size_t               ib_bytes      = 0;
};

decoded_aql_packet
decode_aql_packet(const hsa_ext_amd_aql_pm4_packet_t& aql,
                  uintptr_t                           command_base,
                  size_t                              stream_size)
{
    const auto*    aql_bytes = reinterpret_cast<const uint8_t*>(&aql);
    const uint64_t target    = (uint64_t{load_le_dword(aql_bytes + 12)} << 32) |
                            (load_le_dword(aql_bytes + 8) & 0xFFFFFFFCu);
    if(target < command_base) throw std::runtime_error{"AQL IB target precedes command allocation"};
    if(command_base % command_section_alignment_bytes != 0 ||
       target % command_section_alignment_bytes != 0)
        throw std::runtime_error{"AQL IB target is not command-section aligned"};

    decoded_aql_packet result{
        {aql_bytes, aql_bytes + sizeof(aql)},
        target - command_base,
        (load_le_dword(aql_bytes + 16) & 0xFFFFFu) * sizeof(uint32_t),
    };
    if(result.ib_bytes != stream_size)
        throw std::runtime_error{"AQL IB size does not match PM4 stream"};
    return result;
}

ordered_json
phase_json(const std::vector<uint8_t>&         stream,
           const hsa_ext_amd_aql_pm4_packet_t& aql,
           uintptr_t                           command_base)
{
    const auto decoded = decode_aql_packet(aql, command_base, stream.size());
    return {
        {"aql_packet",
         {
             {"size_bytes", decoded.bytes.size()},
             {"ib_bytes", decoded.ib_bytes},
             {"ib_target",
              {
                  {"base", "command_base"},
                  {"offset_bytes", hex_offset(decoded.target_offset)},
                  {"align_bytes", command_section_alignment_bytes},
              }},
             {"uint32_t", to_hex_dwords(decoded.bytes)},
         }},
        {"pm4_ib",
         {
             {"size_bytes", stream.size()},
             {"uint32_t", to_hex_dwords(stream)},
         }},
    };
}

std::string_view
allocation_role_name(packet_allocation_role role)
{
    switch(role)
    {
        case packet_allocation_role::output: return "output";
        case packet_allocation_role::command: return "command";
    }
    throw std::runtime_error{"unknown packet allocation role"};
}

std::string_view
phase_name(packet_phase phase)
{
    switch(phase)
    {
        case packet_phase::read: return "read";
        case packet_phase::start: return "start";
        case packet_phase::stop: return "stop";
    }
    throw std::runtime_error{"unknown packet phase"};
}

std::string_view
memory_hint_name(uint32_t hint)
{
    switch(static_cast<aqlprofile_memory_hint_t>(hint))
    {
        case AQLPROFILE_MEMORY_HINT_NONE: return "AQLPROFILE_MEMORY_HINT_NONE";
        case AQLPROFILE_MEMORY_HINT_HOST: return "AQLPROFILE_MEMORY_HINT_HOST";
        case AQLPROFILE_MEMORY_HINT_DEVICE_UNCACHED:
            return "AQLPROFILE_MEMORY_HINT_DEVICE_UNCACHED";
        case AQLPROFILE_MEMORY_HINT_DEVICE_COHERENT:
            return "AQLPROFILE_MEMORY_HINT_DEVICE_COHERENT";
        case AQLPROFILE_MEMORY_HINT_DEVICE_NONCOHERENT:
            return "AQLPROFILE_MEMORY_HINT_DEVICE_NONCOHERENT";
        case AQLPROFILE_MEMORY_HINT_LAST: return "AQLPROFILE_MEMORY_HINT_LAST";
    }
    throw std::runtime_error{"unknown packet memory hint"};
}

ordered_json
allocations_json(const pmc_packet_streams& streams)
{
    auto result = ordered_json::array();
    for(const auto& allocation : streams.allocations)
    {
        result.push_back({
            {"id", allocation_role_name(allocation.role)},
            {"observed_call_index", allocation.observed_call_index},
            {"base", hex_address(allocation.base)},
            {"size_bytes", allocation.size_bytes},
            {"flags",
             {
                 {"raw", hex_dword(allocation.flags.raw)},
                 {"host_access", allocation.flags.host_access != 0},
                 {"device_access", allocation.flags.device_access != 0},
                 {"memory_hint",
                  {
                      {"name", memory_hint_name(allocation.flags.memory_hint)},
                      {"value", allocation.flags.memory_hint},
                  }},
             }},
        });
    }
    return result;
}

ordered_json
memory_copies_json(const pmc_packet_streams& streams)
{
    auto result = ordered_json::array();
    for(const auto& copy : streams.memory_copies)
    {
        const auto& allocation = streams.allocations.at(copy.allocation_index);
        result.push_back({
            {"observed_call_index", copy.observed_call_index},
            {"phase", phase_name(copy.phase)},
            {"destination",
             {
                 {"allocation", allocation_role_name(allocation.role)},
                 {"offset_bytes", hex_offset(copy.offset_bytes)},
             }},
            {"size_bytes", copy.size_bytes},
        });
    }
    return result;
}

ordered_json
samples_json(const pmc_packet_streams&                  streams,
             const std::vector<aqlprofile_pmc_event_t>& requested_events)
{
    auto result = ordered_json::array();
    for(const auto& sample : streams.samples)
    {
        const bool requested = std::any_of(
            requested_events.begin(), requested_events.end(), [&sample](const auto& event) {
                return sample.event.block_name == event.block_name &&
                       sample.event.block_index == event.block_index &&
                       sample.event.event_id == event.event_id &&
                       sample.event.flags.raw == event.flags.raw;
            });
        if(!requested) throw std::runtime_error{"sample callback returned an unexpected event"};

        auto coordinates = ordered_json::array();
        for(const auto& coordinate : sample.coordinates)
            coordinates.push_back({
                {"position", coordinate.position},
                {"id", coordinate.id},
                {"name", coordinate.name},
                {"extent", coordinate.extent},
                {"value", coordinate.coordinate},
            });

        result.push_back({
            {"callback_event",
             {
                 {"block_name", static_cast<uint32_t>(sample.event.block_name)},
                 {"block_index", sample.event.block_index},
                 {"event_id", sample.event.event_id},
                 {"flags", {{"raw", hex_dword(sample.event.flags.raw)}}},
             }},
            {"counter_id", sample.counter_id},
            {"callback_index", sample.callback_index},
            {"output",
             {
                 {"allocation", "output"},
                 {"offset_bytes", hex_offset(sample.output_offset_bytes)},
                 {"size_bytes", sizeof(uint64_t)},
             }},
            {"coordinates", std::move(coordinates)},
        });
    }
    return result;
}

ordered_json
unreported_output_slots_json(const pmc_packet_streams& streams)
{
    auto result = ordered_json::array();
    for(const size_t offset : streams.unreported_output_offsets)
        result.push_back({
            {"allocation", "output"},
            {"offset_bytes", hex_offset(offset)},
            {"size_bytes", sizeof(uint64_t)},
        });
    return result;
}

ordered_json
dimensions_json(const std::vector<packet_dimension>& dimensions)
{
    auto result = ordered_json::array();
    for(const auto& dimension : dimensions)
        result.push_back({
            {"position", dimension.position},
            {"id", dimension.id},
            {"name", dimension.name},
            {"extent", dimension.extent},
        });
    return result;
}

ordered_json
pmc_request_json(const packet_test_event&                   test_event,
                 const std::vector<aqlprofile_pmc_event_t>& events)
{
    auto request_events = ordered_json::array();
    for(const auto& event : events)
        request_events.push_back({
            {"event", test_event.name},
            {"block_index", event.block_index},
            {"flags", {{"raw", hex_dword(event.flags.raw)}}},
        });
    return {
        {"shape", events.size() == 1 ? "single-entry" : "same-event-instance-sweep"},
        {"events", std::move(request_events)},
    };
}

ordered_json
full_response_json(const pmc_packet_streams&                  streams,
                   const std::vector<aqlprofile_pmc_event_t>& requested_events)
{
    return {
        {"allocations", allocations_json(streams)},
        {"memory_copies", memory_copies_json(streams)},
        {"start", phase_json(streams.start, streams.aql.start_packet, streams.command_base)},
        {"read", phase_json(streams.read, streams.aql.read_packet, streams.command_base)},
        {"stop", phase_json(streams.stop, streams.aql.stop_packet, streams.command_base)},
        {"samples", samples_json(streams, requested_events)},
        {"unreported_output_slots", unreported_output_slots_json(streams)},
    };
}

struct packet_allocation_bases
{
    uintptr_t output  = 0;
    uintptr_t command = 0;
};

packet_allocation_bases
allocation_bases(const pmc_packet_streams& streams)
{
    std::optional<uintptr_t> output{};
    std::optional<uintptr_t> command{};
    for(const auto& allocation : streams.allocations)
    {
        auto& base = allocation.role == packet_allocation_role::output ? output : command;
        if(base) throw std::runtime_error{"duplicate packet allocation role"};
        base = allocation.base;
    }
    if(!output || !command || *command != streams.command_base)
        throw std::runtime_error{"packet allocation roles are incomplete or inconsistent"};
    return {*output, *command};
}

void
record_document_bases(ordered_json& document, const pmc_packet_streams& streams)
{
    const auto current = allocation_bases(streams);
    if(document.at("output_base").is_null() && document.at("command_base").is_null())
    {
        document["output_base"]  = hex_address(current.output);
        document["command_base"] = hex_address(current.command);
        return;
    }

    if(document.at("output_base") != hex_address(current.output) ||
       document.at("command_base") != hex_address(current.command))
        throw std::runtime_error{"packet allocation bases changed within one dump document"};
}

std::string
baseline_response_pointer(const ordered_json& document, size_t baseline_case_index)
{
    if(baseline_case_index >= document.at("pmc").size())
        throw std::runtime_error{"PMC baseline reference is not backward"};
    return "#/pmc/" + std::to_string(baseline_case_index) + "/response/";
}

ordered_json
topology_json(const std::string& gfx, discovery_table_t ip_discovery)
{
    const auto               info = packet_agent_info(gfx);
    std::vector<std::string> cu_bitmap{};
    for(uint32_t se = 0; se < AQLPROFILE_DRM_CU_BITMAP_NUM_SE; ++se)
        for(uint32_t sa = 0; sa < AQLPROFILE_DRM_CU_BITMAP_NUM_SA_PER_SE; ++sa)
            cu_bitmap.emplace_back(hex_dword(info.cu_bitmap.bits[se][sa]));

    std::sort(ip_discovery.begin(), ip_discovery.end(), [](const auto& lhs, const auto& rhs) {
        if(lhs.die != rhs.die) return lhs.die < rhs.die;
        if(lhs.ipname != rhs.ipname) return lhs.ipname < rhs.ipname;
        return lhs.id < rhs.id;
    });
    auto   dies  = ordered_json::array();
    size_t index = 0;
    while(index < ip_discovery.size())
    {
        const int die = ip_discovery.at(index).die;
        auto      ips = ordered_json::array();
        while(index < ip_discovery.size() && ip_discovery.at(index).die == die)
        {
            const std::string ip        = ip_discovery.at(index).ipname;
            auto              instances = ordered_json::array();
            while(index < ip_discovery.size() && ip_discovery.at(index).die == die &&
                  ip_discovery.at(index).ipname == ip)
            {
                const auto&              entry = ip_discovery.at(index++);
                std::vector<std::string> segments{};
                segments.reserve(entry.segment_count);
                for(size_t i = 0; i < entry.segment_count; ++i)
                    segments.emplace_back(hex_dword(entry.segments.at(i)));

                instances.push_back({
                    {"id", entry.id},
                    {"major", entry.major},
                    {"minor", entry.minor},
                    {"revision", entry.revision},
                    {"num_instance", entry.instance},
                    {"base_addr", segments},
                });
            }

            std::string name = ip;
            std::transform(name.begin(), name.end(), name.begin(), [](unsigned char character) {
                return static_cast<char>(std::toupper(character));
            });
            ips.push_back({{"name", name}, {"instances", std::move(instances)}});
        }
        dies.push_back({{"id", die}, {"ips", std::move(ips)}});
    }

    return {
        {"format_version", 1},
        {"gfx", gfx},
        {"agent_info_version",
         {
             {"name", "AQLPROFILE_AGENT_VERSION_V2"},
             {"value", static_cast<uint32_t>(AQLPROFILE_AGENT_VERSION_V2)},
         }},
        {"agent_info",
         {
             {"agent_gfxip", info.agent_gfxip},
             {"xcc_num", info.xcc_num},
             {"se_num", info.se_num},
             {"cu_num", info.cu_num},
             {"shader_arrays_per_se", info.shader_arrays_per_se},
             {"domain", info.domain},
             {"location_id", hex_dword(info.location_id)},
             {"cu_bitmap",
              {
                  {"layout", "se-major"},
                  {"dimensions",
                   {
                       {"se", AQLPROFILE_DRM_CU_BITMAP_NUM_SE},
                       {"shader_arrays_per_se", AQLPROFILE_DRM_CU_BITMAP_NUM_SA_PER_SE},
                   }},
                  {"uint32_t", cu_bitmap},
              }},
         }},
        {"ip_discovery", {{"dies", std::move(dies)}}},
    };
}

void
write_json(const fs::path& path, const ordered_json& document)
{
    const std::string contents  = document.dump(2) + '\n';
    fs::path          temporary = path;
    temporary += ".tmp";

    std::ofstream output{temporary, std::ios::trunc};
    if(!output.is_open()) throw std::runtime_error{"failed to open " + temporary.string()};
    output << contents;
    output.close();
    if(!output)
    {
        std::error_code error{};
        fs::remove(temporary, error);
        throw std::runtime_error{"failed to write " + temporary.string()};
    }

    std::error_code error{};
    fs::rename(temporary, path, error);
    if(error)
    {
        fs::remove(temporary);
        throw std::runtime_error{"failed to replace " + path.string() + ": " + error.message()};
    }
}
}  // namespace

ordered_json
make_pmc_json_document(const std::string&                   gfx,
                       const packet_test_case&              test_case,
                       hsa_ven_amd_aqlprofile_block_name_t  block,
                       uint32_t                             counter_count,
                       const std::vector<packet_dimension>& dimensions)
{
    ordered_json document{
        {"format_version", 1},
        {"gfx", gfx},
        {"topology", gfx + "_ip_discovery.json"},
        {"block",
         {
             {"name", test_case.block},
             {"__implementation_aqlprofile:block_id", static_cast<uint32_t>(block)},
             {"counter_count", counter_count},
             {"dimensions", dimensions_json(dimensions)},
         }},
        {"events", ordered_json::array()},
        {"skipped", ordered_json::array()},
        {"byte_order", "little-endian"},
        {"output_base", nullptr},
        {"command_base", nullptr},
        {"pmc", ordered_json::array()},
    };
    for(const auto& test_event : test_case.events)
        document["events"].push_back({{"name", test_event.name}, {"event_id", test_event.id}});
    for(const auto& test_event : test_case.skipped_events)
        document["skipped"].push_back({
            {"name", test_event.name},
            {"event_id", test_event.id},
            {"reason", "metric_name_contains_level_token"},
        });
    return document;
}

size_t
append_full_pmc_json(ordered_json&                              document,
                     const packet_test_event&                   test_event,
                     const std::vector<aqlprofile_pmc_event_t>& events,
                     const pmc_packet_streams&                  streams)
{
    record_document_bases(document, streams);
    const size_t case_index = document.at("pmc").size();
    document["pmc"].push_back({
        {"request", pmc_request_json(test_event, events)},
        {"response", full_response_json(streams, events)},
    });
    return case_index;
}

void
append_atomic_pmc_json(ordered_json&                              document,
                       const packet_test_event&                   test_event,
                       const std::vector<aqlprofile_pmc_event_t>& events,
                       const pmc_packet_streams&                  streams,
                       size_t                                     baseline_case_index)
{
    if(events.size() != 1) throw std::invalid_argument{"atomic PMC case must contain one event"};
    record_document_bases(document, streams);

    auto              response      = full_response_json(streams, events);
    const std::string baseline      = baseline_response_pointer(document, baseline_case_index);
    response["start"]["aql_packet"] = {{"$ref", baseline + "start/aql_packet"}};
    response["read"]                = {{"$ref", baseline + "read"}};
    response["stop"]                = {{"$ref", baseline + "stop"}};
    document["pmc"].push_back({
        {"request", pmc_request_json(test_event, events)},
        {"response", std::move(response)},
    });
}

void
append_instance_sweep_pmc_json(ordered_json&                              document,
                               const packet_test_event&                   test_event,
                               const std::vector<aqlprofile_pmc_event_t>& events,
                               const pmc_packet_streams&                  streams,
                               size_t                                     baseline_case_index)
{
    if(events.size() < 2)
        throw std::invalid_argument{"instance-sweep PMC case must contain multiple events"};
    record_document_bases(document, streams);

    auto              response          = full_response_json(streams, events);
    const std::string baseline          = baseline_response_pointer(document, baseline_case_index);
    response["allocations"]             = {{"$ref", baseline + "allocations"}};
    response["memory_copies"]           = {{"$ref", baseline + "memory_copies"}};
    response["start"]["aql_packet"]     = {{"$ref", baseline + "start/aql_packet"}};
    response["read"]                    = {{"$ref", baseline + "read"}};
    response["stop"]                    = {{"$ref", baseline + "stop"}};
    response["unreported_output_slots"] = {
        {"$ref", baseline + "unreported_output_slots"},
    };
    document["pmc"].push_back({
        {"request", pmc_request_json(test_event, events)},
        {"response", std::move(response)},
    });
}

void
write_pmc_json(const fs::path& path, const ordered_json& document)
{
    if(document.at("pmc").empty()) throw std::runtime_error{"packet dump contains no PMC cases"};
    if(document.at("output_base").is_null() || document.at("command_base").is_null())
        throw std::runtime_error{"packet dump allocation bases were not recorded"};
    write_json(path, document);
}

void
write_topology_json(const fs::path& path, const std::string& gfx, discovery_table_t ip_discovery)
{
    write_json(path, topology_json(gfx, std::move(ip_discovery)));
}

}  // namespace packet_dump
}  // namespace aqlprofile
