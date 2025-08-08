#include "cpu.hpp"
#include "agent_manager.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <functional>
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
                return cpu_data;  // Return immediately after first core
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
    auto cpu_data = get_cpu_info();
    return cpu_data.size();
}

void
query_cpu_agents()
{
    int32_t  id_count   = 0;
    uint32_t node_count = 0;
    uint32_t cpu_count  = 0;

    if(device_count() == 0)
    {
        return;
    }

    auto& _agent_manager = agent_manager::get_instance();
    auto  cpu_data       = get_cpu_info();

    for(auto& cpu : cpu_data)
    {
        auto node_id    = node_count++;
        auto logical_id = id_count++;
        auto id         = cpu_count++;
        auto cur_agent  = agent{ agent_type::CPU,
                                id,
                                node_id,
                                logical_id,
                                static_cast<int32_t>(id),
                                cpu.model_name,
                                cpu.model_name,
                                cpu.vendor_id,
                                "" };
        _agent_manager.insert_agent(cur_agent);
    }
}
}  // namespace cpu
}  // namespace rocprofsys
