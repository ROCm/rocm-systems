
#include <iostream>
#include <vector>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <cstring>
#include <ctime>
#include <map>
#include <algorithm>
#include "cuid_util.h"
#include "cuid.h"


void print_help()
{
    std::cout << "Usage: amdcuid [OPTIONS]\n"
              << "Options:\n"
              << "  --type [gpu|cpu|nic|platform]   Filter and show only the specified device type.\n"
              << "  --query_cuid <partial_cuid>     Query and list CUIDs that match the given partial CUID.\n"
              << "  --update_cuid <file_name>       Update the secondary CUID and save to file.\n"
              << "  --update_primary_id             Update the mapping of the primary CUID and secondary CUID for privileged user.\n"
              << "  --monitor                       Continuously monitors device events and updates the secondary CUID.\n"
              << "  --help                          Show this help message and exit.\n";
}



int main(int argc, char *argv[])
{
    if (argc == 1 || (argc == 2 && std::string(argv[1]) == "--help"))
    {
        print_help();
        return 0;
    }

    bool update_cuid = false;
    std::string cuid_file = "cuid_devices.txt";
    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "--update_cuid")
        {
            update_cuid = true;
            // Check if next argument exists and is not another option
            if (i + 1 < argc && argv[i + 1][0] != '-')
            {
                cuid_file = argv[i + 1];
                ++i;
            }
        }
    }

    if (update_cuid)
    {
        // Only handle GPU for this example
        amdcuid_status_t err;
        uint32_t count = 0, available = 0;
        std::vector<amdcuid_handle> handles;
        // Get GPU handles
        do
        {
            count = available;
            handles.resize(count);
            err = amdcuid_get_handles(AMDCUID_DEVICE_TYPE_SET_GPU, count, handles.data(), &available);
            if (err != AMDCUID_STATUS_SUCCESS)
            {
                std::cerr << "Failed to get GPU handles: " << CuidUtilities::cuid_status_to_string(err) << std::endl;
                return 1;
            }
        } while (count != available);

        // Parse existing file
        SectionMap sections = CuidUtilities::parse_cuid_file(cuid_file);

        for (uint32_t i = 0; i < count; ++i)
        {
            amdcuid secondary_id = {};
            err = amdcuid_get_secondary_cuid(handles[i], &secondary_id);
            if (err != AMDCUID_STATUS_SUCCESS)
            {
                std::cerr << "Failed to get secondary CUID for GPU #" << i << std::endl;
                continue;
            }
            char device_node[128] = {0};
            uint32_t device_node_len = sizeof(device_node);
            err = amdcuid_get_render_node(handles[i], device_node, &device_node_len);
            if (err != AMDCUID_STATUS_SUCCESS)
            {
                strcpy(device_node, "");
            }
            std::string section = "[GPU:" + std::to_string(i) + "]";
            sections[section]["secondary_cuid"] = CuidUtilities::get_cuid_as_string(&secondary_id);
            sections[section]["device_node"] = device_node;
            sections[section]["last_update"] = std::to_string(std::time(nullptr));
        }
        CuidUtilities::write_cuid_file(cuid_file, sections);
        std::cout << "Updated secondary CUIDs for GPUs and saved to " << cuid_file << std::endl;
        return 0;
    }

    std::cerr << "Unknown or unimplemented option. Use --help for usage." << std::endl;
    return 1;
}