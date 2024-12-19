/*
Copyright (c) 2024 - present Advanced Micro Devices, Inc. All rights reserved.
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
#include "RdciTopologyLinkSubSystem.h"

#include <getopt.h>
#include <signal.h>
#include <unistd.h>

#include "common/rdc_utils.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcException.h"
#include "rdc_lib/rdc_common.h"
namespace amd {
namespace rdc {
RdciTopologyLinkSubSystem::RdciTopologyLinkSubSystem() {}

void RdciTopologyLinkSubSystem::parse_cmd_opts(int argc, char** argv) {
  const int HOST_OPTIONS = 1000;
  const struct option long_options[] = {{"host", required_argument, nullptr, HOST_OPTIONS},
                                        {"help", optional_argument, nullptr, 'h'},
                                        {"unauth", optional_argument, nullptr, 'u'},
                                        {"gpu_index", required_argument, nullptr, 'i'},
                                        {nullptr, 0, nullptr, 0}};
  int option_index = 0;
  int opt = 0;
  while ((opt = getopt_long(argc, argv, "hui:", long_options, &option_index)) != -1) {
    {
      switch (opt) {
        case HOST_OPTIONS:
          ip_port_ = optarg;
          break;
        case 'h':
          show_help();
          break;
        case 'u':
          use_auth_ = false;
          break;
        case 'i':
          if (!IsNumber(optarg)) {
            show_help();
            throw RdcException(RDC_ST_BAD_PARAMETER, "The field group index needs to be a number");
          }
          topology_ops_ = TOPOLOGY_INDEX;
          group_index_ = std::stoi(optarg);
          is_group_index_set = true;
          break;
        default:
          show_help();
          throw RdcException(RDC_ST_BAD_PARAMETER, "Unknown command line options");
      }
    }
  }
  if (is_group_index_set == false) {
    show_help();
    throw RdcException(RDC_ST_BAD_PARAMETER, "Need to specify the group index to notify");
  }
}

void RdciTopologyLinkSubSystem::show_help() const {
  std::cout << " topo -- Used to  topology will show how the GPU connected.\n\n";
  std::cout << "Usage\n";
  std::cout << "    rdci topo [--host <IP/FQDN>:port] [--reg] -g <groupId>\n";
  std::cout << "\nFlags:\n";
  show_common_usage();
  std::cout << "  -i  --index                      display the topology"
            << "information for GPU index 0 .\n";
}

static const char* topology_link_type_to_str(rdc_topology_link_type_t type) {
  if (type == RDC_IOLINK_TYPE_PCIEXPRESS) return "Connected via PCIe \t";
  if (type == RDC_IOLINK_TYPE_XGMI) return "Connected via XGMI \t";
  if (type == RDCI_IOLINK_TYPE_NUMIOLINKTYPES) return "Number of IO Link types";
  return "N/A \t\t\t";
}

void RdciTopologyLinkSubSystem::process() {
  rdc_status_t result = RDC_ST_OK;
  switch (topology_ops_) {
    case TOPOLOGY_INDEX: {
      rdc_device_topology_t topology;
      result = rdc_device_topology_get(rdc_handle_, group_index_, &topology);
      if (result == RDC_ST_OK) {
        std::cout << "+-----------------------+"
                  << "-----------------------------"
                  << "------------------+\n";
        std::cout << "| GPU ID: " << group_index_ << "\t\t"
                  << "| Topology Information \t\t\t\t|\n";
        std::cout << "+=======================+"
                  << "============================="
                  << "==================+\n";
        std::cout << "+-----------------------+"
                  << "-----------------------------"
                  << "------------------+\n";
        for (uint32_t i = 0; i < topology.num_of_gpus; i++) {
          std::cout << "| To GPU " << i + 1 << "\t\t"
                    << "| " << topology_link_type_to_str(RDC_IOLINK_TYPE_XGMI)
                    << "\t\t\t|\n";
        }
        std::cout << "+-----------------------+"
                  << "-----------------------------"
                  << "------------------+\n";
      }
      break;
    }
  }
}
}  // namespace rdc
}  // namespace amd