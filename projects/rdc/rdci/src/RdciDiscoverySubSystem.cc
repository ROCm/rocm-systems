/*
Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <getopt.h>
#include <unistd.h>
#include "rdc_lib/rdc_common.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcException.h"
#include "RdciDiscoverySubSystem.h"


namespace amd {
namespace rdc {

RdciDiscoverySubSystem::RdciDiscoverySubSystem() : show_help_(false) {
}

void RdciDiscoverySubSystem::parse_cmd_opts(int argc, char ** argv) {
    const int HOST_OPTIONS = 1000;
    const int JSON_OPTIONS = 1001;
    const struct option long_options[] = {
        {"host",    required_argument, nullptr, HOST_OPTIONS },
        {"help", optional_argument, nullptr, 'h' },
        {"unauth", optional_argument, nullptr, 'u' },
        {"list", optional_argument, nullptr, 'l' },
        {"json",  optional_argument, nullptr, JSON_OPTIONS },
        { nullptr,  0 , nullptr, 0 }
    };

    int option_index = 0;
    int opt = 0;
    bool is_list = false;

    while ((opt = getopt_long(argc, argv, "hlu",
                long_options, &option_index)) != -1) {
        switch (opt) {
            case HOST_OPTIONS:
                ip_port_ = optarg;
                break;
            case JSON_OPTIONS:
                set_json_output(true);
                break;
            case 'h':
                 show_help_ = true;
                 return;
            case 'u':
                 use_auth_ = false;
                 break;
            case 'l':
                 is_list = true;
                 break;
            default:
                show_help();
                throw RdcException(RDC_ST_BAD_PARAMETER,
                        "Unknown command line options");
        }
    }

    if (!is_list) {
        show_help();
        throw RdcException(RDC_ST_BAD_PARAMETER,
                "Need to specify operations");
    }
}

void RdciDiscoverySubSystem::show_help() const {
    if (is_json_output()) return;
    std::cout << " discovery -- Used to discover and identify GPUs "
        << "and their attributes.\n\n";
    std::cout << "Usage\n";
    std::cout << "    rdci discovery [--host <IP/FQDN>:port] [--json]"
            << " [-u] -l\n";
    std::cout << "\nFlags:\n";
    show_common_usage();
    std::cout << "  --json                         "
              << "Output using json.\n";
    std::cout << "  -l  --list                     list GPU discovered"
              <<" on the system\n";
}


void RdciDiscoverySubSystem::process() {
    if (show_help_) {
        return show_help();
    }

    uint32_t  gpu_index_list[RDC_MAX_NUM_DEVICES];
    uint32_t count = 0;
    rdc_status_t result =  rdc_device_get_all(rdc_handle_,
            gpu_index_list, &count);
    if (result != RDC_ST_OK) {
         throw RdcException(result, "Fail to get device information");
    }
    if (count == 0) {
        if (is_json_output()) {
            std::cout << "\"gpus\" : [], \"status\": \"ok\"";
        } else {
            std::cout << "No GPUs find on the system\n";
        }
        return;
    }

    if (is_json_output()) {
        std::cout << "\"gpus\" : [";
    } else {
        std::cout << count << " GPUs found.\n";
        std::cout << "------------------------------------------------"
                << "-----------------\n";
        std::cout << "GPU Index\t Device Information\n";
    }
    for (uint32_t i = 0; i < count; i++) {
        rdc_device_attributes_t attribute;
        result = rdc_device_get_attributes(rdc_handle_,
                gpu_index_list[i], &attribute);
        if (result != RDC_ST_OK) {
            return;
        }
        if (is_json_output()) {
            std::cout << "{\"gpu_index\": \"" << i << "\", \"device_name\": \""
                << attribute.device_name << "\"}";
            if (i != count -1) {
                std::cout << ",";
            }
        } else {
            std::cout << i << "\t\t" << attribute.device_name <<std::endl;
        }
    }
    if (is_json_output()) {
        std::cout << ']';
    } else {
        std::cout << "------------------------------------------------"
              << "-----------------\n";
    }
}


}  // namespace rdc
}  // namespace amd


