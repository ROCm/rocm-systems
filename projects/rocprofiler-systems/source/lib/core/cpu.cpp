// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cpu.hpp"
#include "agent_manager.hpp"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <unistd.h>
#include <unordered_map>

namespace rocprofsys
{
namespace cpu
{
std::vector<cpu_info>
process_cpu_info_data()
{
    std::vector<cpu_info> cpu_data;
    std::ifstream         cpuinfo_file("/proc/cpuinfo");

    if(!cpuinfo_file.is_open())
    {
        return cpu_data;
    }

    std::string line;
    cpu_info    current_cpu;
    bool        has_processor_entry = false;

    auto parse_long = [](const std::string& value) -> long {
        try
        {
            return std::stol(value);
        } catch(const std::exception&)
        {
            return -1;
        }
    };

    auto trim_whitespace = [](const std::string& str) -> std::string {
        size_t start = str.find_first_not_of(" \t");
        if(start == std::string::npos) return "";
        size_t end = str.find_last_not_of(" \t");
        return str.substr(start, end - start + 1);
    };

    static const std::unordered_map<std::string,
                                    std::function<void(cpu_info&, const std::string&)>>
        field_parsers = {
            { "processor",
              [&parse_long](cpu_info& cpu, const std::string& val) {
                  cpu.processor = parse_long(val);
              } },
            { "cpu family",
              [&parse_long](cpu_info& cpu, const std::string& val) {
                  cpu.family = parse_long(val);
              } },
            { "model",
              [&parse_long](cpu_info& cpu, const std::string& val) {
                  cpu.model = parse_long(val);
              } },
            { "physical id",
              [&parse_long](cpu_info& cpu, const std::string& val) {
                  cpu.physical_id = parse_long(val);
              } },
            { "core id",
              [&parse_long](cpu_info& cpu, const std::string& val) {
                  cpu.core_id = parse_long(val);
              } },
            { "apicid",
              [&parse_long](cpu_info& cpu, const std::string& val) {
                  cpu.apicid = parse_long(val);
              } },
            { "vendor_id",
              [](cpu_info& cpu, const std::string& val) { cpu.vendor_id  = val; } },
            { "model name",
              [](cpu_info& cpu, const std::string& val) { cpu.model_name = val; } }
        };

    while(std::getline(cpuinfo_file, line))
    {
        if(line.empty())
        {
            if(has_processor_entry)
            {
                cpu_data.push_back(current_cpu);
                current_cpu         = cpu_info{};
                has_processor_entry = false;
            }
            continue;
        }

        size_t colon_pos = line.find(':');
        if(colon_pos == std::string::npos)
        {
            continue;
        }

        std::string key   = trim_whitespace(line.substr(0, colon_pos));
        std::string value = trim_whitespace(line.substr(colon_pos + 1));

        std::transform(key.begin(), key.end(), key.begin(), ::tolower);

        auto it = field_parsers.find(key);
        if(it != field_parsers.end())
        {
            it->second(current_cpu, value);
            if(key == "processor")
            {
                has_processor_entry = true;
            }
        }
    }

    if(has_processor_entry)
    {
        cpu_data.push_back(current_cpu);
    }

    return cpu_data;
}

std::vector<cpu_info>
get_cpu_info()
{
    static auto _v = process_cpu_info_data();
    return _v;
}

size_t
device_count()
{
    // Return unique socket count from parsed CPU info
    auto           cpu_data = get_cpu_info();
    std::set<long> sockets;
    for(const auto& cpu : cpu_data)
    {
        sockets.insert(std::max(0L, cpu.physical_id));
    }
    return sockets.empty() ? (cpu_data.empty() ? 0 : 1) : sockets.size();
}

void
query_cpu_agents()
{
    auto cpu_data = get_cpu_info();
    if(cpu_data.empty()) return;

    // Group CPUs by socket (physical_id), collect model_name per socket
    std::map<size_t, std::string> socket_model_names;
    std::map<size_t, std::string> socket_vendor_ids;

    for(const auto& cpu : cpu_data)
    {
        const auto socket_id = static_cast<size_t>(std::max(0L, cpu.physical_id));
        if(socket_model_names.find(socket_id) == socket_model_names.end())
        {
            socket_model_names[socket_id] = cpu.model_name;
            socket_vendor_ids[socket_id]  = cpu.vendor_id;
        }
    }

    // Insert one agent per socket in ascending socket_id order
    // so that device_type_index == socket_id
    auto&    mgr        = get_agent_manager_instance();
    uint32_t node_count = 0;

    for(const auto& [socket_id, model_name] : socket_model_names)
    {
        const auto node_id     = node_count++;
        const auto device_name = "CPU" + std::to_string(socket_id);
        auto       cur_agent   = agent{ agent_type::CPU,
                                0,
                                static_cast<uint32_t>(socket_id),
                                node_id,
                                static_cast<int32_t>(socket_id),
                                static_cast<int32_t>(socket_id),
                                device_name,
                                model_name,
                                socket_vendor_ids[socket_id],
                                "",
                                0,
                                0,
                                "" };
        mgr.insert_agent(cur_agent);
    }
}
}  // namespace cpu
}  // namespace rocprofsys
