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
#include "RdcException.h"
#include "RdciDiscoverySubSystem.h"


namespace amd {
namespace rdc {

RdciDiscoverySubSystem::RdciDiscoverySubSystem() :show_help_(false) {
}

void RdciDiscoverySubSystem::parse_cmd_opts(int argc, char ** argv) {
    const int HOST_OPTIONS = 1000;
    const struct option long_options[] = {
        {"host",    required_argument, nullptr, HOST_OPTIONS },
        {"help", optional_argument, nullptr, 'h' },
        { nullptr,  0 , nullptr, 0 }
    };

    int option_index = 0;
    int opt = 0;

    while ((opt = getopt_long(argc, argv, "h",
                long_options, &option_index)) != -1) {
        switch (opt) {
            case HOST_OPTIONS:
                ip_port_ = optarg;
                break;
            case 'h':
                 show_help_ = true;
                 return;
            default:
                show_help();
                throw RdcException(RDC_ST_BAD_PARAMETER,
                        "Unknown command line options");
        }
    }
}

void RdciDiscoverySubSystem::show_help() const {
    std::cout << " discovery -- Used to discover and identify GPUs "
        << "and their attributes.\n\n";
    std::cout << "Usage\n";
    std::cout << "    rdci discovery [--host <IP/FQDN>:port]\n";
    std::cout << "\nFlags:\n";
    std::cout << "  --host       <IP/FQDN>:port    Connects to "
            << "specified IP or fully-qualified domain name.\n";
    std::cout << "                                 The port "
            << "must be specified.\n";
    std::cout << "                                 Default: localhost:50051\n";
    std::cout << "  -h  --help                     Displays usage "
              << "information and exits.\n";
}


void RdciDiscoverySubSystem::process() {
    if (show_help_) {
        return show_help();
    }

    uint32_t  gpu_index_list[RDC_MAX_NUM_DEVICES];
    uint32_t count = 0;
    rdc_status_t result =  rdc_get_all_devices(rdc_handle_,
            gpu_index_list, &count);
    if (result != RDC_ST_OK) {
         throw RdcException(result, "Fail to get device information");
    }
    if (count == 0) {
        std::cout << "No GPUs find on the sytem\n";
        return;
    }

    std::cout << count << " GPUs found.\n";
    std::cout << "------------------------------------------------"
              << "-----------------\n";
    std::cout << "GPU Index\t Device Information\n";
    for (uint32_t i = 0; i < count; i++) {
        rdc_device_attributes_t attribute;
        result = rdc_get_device_attributes(rdc_handle_,
                gpu_index_list[i], &attribute);
        if (result != RDC_ST_OK) {
            return;
        }
        std::cout << i << "\t\t" << attribute.device_name <<std::endl;
    }
    std::cout << "------------------------------------------------"
              << "-----------------\n";
}


}  // namespace rdc
}  // namespace amd


