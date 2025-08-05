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
#include "RdciDiscoverySubSystem.h"

#include <getopt.h>
#include <unistd.h>

#include <cstring>
#include <iomanip>
#include <set>

#include "rdc/rdc.h"
#include "rdc/rdc_private.h"
#include "rdc_lib/RdcException.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

RdciDiscoverySubSystem::RdciDiscoverySubSystem()
    : show_help_(false), is_list_(false), is_partition_(false), show_version_(false) {}

void RdciDiscoverySubSystem::parse_cmd_opts(int argc, char** argv) {
  const int HOST_OPTIONS = 1000;
  const int JSON_OPTIONS = 1001;
  const struct option long_options[] = {{"host", required_argument, nullptr, HOST_OPTIONS},
                                        {"help", optional_argument, nullptr, 'h'},
                                        {"unauth", optional_argument, nullptr, 'u'},
                                        {"list", optional_argument, nullptr, 'l'},
                                        {"json", optional_argument, nullptr, JSON_OPTIONS},
                                        {"version", optional_argument, nullptr, 'v'},
                                        {nullptr, 0, nullptr, 0}};

  int option_index = 0;
  int opt = 0;

  while ((opt = getopt_long(argc, argv, "hliuv", long_options, &option_index)) != -1) {
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
        is_list_ = true;
        break;
      case 'i':
        is_partition_ = true;
        break;
      case 'v':
        show_version_ = true;
        break;
      default:
        show_help();
        throw RdcException(RDC_ST_BAD_PARAMETER, "Unknown command line options");
    }
  }

  int opCount = (is_list_ ? 1 : 0) + (is_partition_ ? 1 : 0) + (show_version_ ? 1 : 0);
  if (opCount != 1) {
    show_help();
    throw RdcException(RDC_ST_BAD_PARAMETER, "Need to specify exactly one operation");
  }
}

void RdciDiscoverySubSystem::show_help() const {
  if (is_json_output()) return;
  std::cout << " discovery -- Used to discover and identify GPUs "
            << "and their attributes, as well as server version information.\n\n";
  std::cout << "Usage\n";
  std::cout << "    rdci discovery [--host <IP/FQDN>:port] [--json]"
            << " [-u] -l -v\n";
  std::cout << "\nFlags:\n";
  show_common_usage();
  std::cout << "  --json                         "
            << "Output using json.\n";
  std::cout << "  -l  --list                     list GPU discovered"
            << " on the system\n";
  std::cout << "  -i  --gpu-instance             list GPU discovered"
            << " on the system with partitions\n";
  std::cout << "  -v  --version                  Display version information of the"
            << " the server and libraries used by the server\n";
}

void RdciDiscoverySubSystem::show_attributes() {
  uint32_t gpu_index_list[RDC_MAX_NUM_DEVICES];
  uint32_t count = 0;
  rdc_status_t result = rdc_device_get_all(rdc_handle_, gpu_index_list, &count);
  if (result != RDC_ST_OK) {
    throw RdcException(result, "Fail to get device information");
  }
  if (count == 0) {
    if (is_json_output()) {
      std::cout << "\"gpus\" : [], \"status\": \"ok\"";
    } else {
      std::cout << "No GPUs found on the system\n";
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
    result = rdc_device_get_attributes(rdc_handle_, gpu_index_list[i], &attribute);
    if (result != RDC_ST_OK) {
      return;
    }
    if (is_json_output()) {
      std::cout << "{\"gpu_index\": \"" << i << "\", \"device_name\": \"" << attribute.device_name
                << "\"}";
      if (i != count - 1) {
        std::cout << ",";
      }
    } else {
      std::cout << i << "\t\t" << attribute.device_name << std::endl;
    }
  }
  if (is_json_output()) {
    std::cout << ']';
  } else {
    std::cout << "------------------------------------------------"
              << "-----------------\n";
  }
}

void RdciDiscoverySubSystem::show_attributes_with_partitions() {
  uint32_t gpu_index_list[RDC_MAX_NUM_DEVICES];
  uint32_t count = 0;
  rdc_status_t result = rdc_device_get_all(rdc_handle_, gpu_index_list, &count);
  if (result != RDC_ST_OK) {
    throw RdcException(result, "Fail to get all devices");
  }

  if (count == 0) {
    if (is_json_output())
      std::cout << "\"gpus\" : [], \"status\": \"ok\"";
    else
      std::cout << "No GPUs found on the system\n";
    return;
  }

  // Print header.
  if (!is_json_output()) {
    std::cout << count << " GPUs found." << std::endl;
    std::cout << "---------------------------------------------------------------------"
              << std::endl;
    std::cout << std::setw(12) << std::left << "GPU Index" << std::setw(20) << "Instance Index"
              << std::setw(25) << "Device Information" << std::setw(8) << "XCC" << std::setw(8)
              << "DECODER" << std::endl;
  } else {
    std::cout << "\"gpus\" : [";
  }

  // Loop over each GPU.
  for (uint32_t i = 0; i < count; i++) {
    rdc_device_attributes_t attribute;
    result = rdc_device_get_attributes(rdc_handle_, gpu_index_list[i], &attribute);
    if (result != RDC_ST_OK) return;

    // Build physical device entity info.
    rdc_entity_info_t phys_info;
    phys_info.device_index = i;
    phys_info.instance_index = 0;
    phys_info.entity_role = RDC_DEVICE_ROLE_PHYSICAL;
    phys_info.device_type = RDC_DEVICE_TYPE_GPU;
    uint32_t phys_entity_index = rdc_get_entity_index_from_info(phys_info);

    rdc_resource_profile_t phys_xcc = {};
    rdc_resource_profile_t phys_decoder_profile = {};
    result =
        rdc_instance_profile_get(rdc_handle_, phys_entity_index, RDC_ACCELERATOR_XCC, &phys_xcc);
    result = rdc_instance_profile_get(rdc_handle_, phys_entity_index, RDC_ACCELERATOR_DECODER,
                                      &phys_decoder_profile);

    std::string phys_xcc_str = std::to_string(phys_xcc.partition_resource);
    std::string phys_decoder_str = std::to_string(phys_decoder_profile.partition_resource);

    if (!is_json_output()) {
      std::cout << std::setw(12) << std::left << i << std::setw(20) << "" << std::setw(25)
                << attribute.device_name << std::setw(8) << phys_xcc_str << std::setw(8)
                << phys_decoder_str << std::endl;
    } else {
      std::cout << "{\"gpu_index\": \"" << i << "\", "
                << "\"device_name\": \"" << attribute.device_name << "\", "
                << "\"physical\": {"
                << "\"xcc\": \"" << phys_xcc_str << "\", "
                << "\"decoder\": \"" << phys_decoder_str << "\" "
                << "}";
    }

    uint16_t num_partition = 0;
    rdc_status_t result = rdc_get_num_partition(rdc_handle_, i, &num_partition);
    if (result != RDC_ST_OK) {
      return;
    }

    // A partitionable device not in partitionable mode will have metrics.num_partition=1
    // Where as, a non-partitionable device will have metrics.num_partition = UINT16_MAX
    if (num_partition != UINT16_MAX && num_partition > 1) {
      if (is_json_output()) {
        std::cout << ", \"partitions\": [";
      }
      for (uint32_t pid = 0; pid < num_partition; pid++) {
        std::string instance_str = "g" + std::to_string(i) + "." + std::to_string(pid);

        rdc_entity_info_t part_info;
        part_info.device_index = i;
        part_info.instance_index = pid;
        part_info.entity_role = RDC_DEVICE_ROLE_PARTITION_INSTANCE;
        part_info.device_type = RDC_DEVICE_TYPE_GPU;
        uint32_t part_entity_index = rdc_get_entity_index_from_info(part_info);

        rdc_resource_profile_t part_xcc = {};
        rdc_resource_profile_t part_decoder = {};
        result = rdc_instance_profile_get(rdc_handle_, part_entity_index, RDC_ACCELERATOR_XCC,
                                          &part_xcc);
        result = rdc_instance_profile_get(rdc_handle_, part_entity_index, RDC_ACCELERATOR_DECODER,
                                          &part_decoder);

        std::string part_decoder_str = std::to_string(part_decoder.partition_resource);
        std::string part_xcc_str = std::to_string(part_xcc.partition_resource);
        std::string starColumn = " ";
        if (part_decoder.num_partitions_share_resource > 1) {
          starColumn = "*";
        }

        if (!is_json_output()) {
          std::cout << std::setw(12) << "" << std::setw(20) << instance_str << std::setw(25) << ""
                    << std::setw(7) << part_xcc_str << std::setw(1) << starColumn << std::setw(8)
                    << part_decoder_str << std::endl;
        } else {
          std::string decoder_shared =
              (part_decoder.num_partitions_share_resource > 1) ? "true" : "false";
          std::cout << "{\"instance_index\": \"" << instance_str << "\", "
                    << "\"xcc\": \"" << part_xcc_str << "\", "
                    << "\"decoder\": \"" << part_decoder_str << "\", "
                    << "\"decoder_shared\": " << decoder_shared << "}";

          if (pid != num_partition - 1) {
            std::cout << ",";
          } else {
            std::cout << "]";
          }
        }
      }
    }

    if (is_json_output()) {
      if (i != count - 1)
        std::cout << "},";
      else
        std::cout << "}";
    }
  }

  if (!is_json_output()) {
    std::cout << "---------------------------------------------------------------------"
              << std::endl;
    std::cout << "* if the resource is shared" << std::endl;
  } else {
    std::cout << ']';
  }
}

void RdciDiscoverySubSystem::show_version() {
  rdc_component_version_t smiv;
  rdc_status_t result = rdc_device_get_component_version(rdc_handle_, RDC_AMDSMI_COMPONENT, &smiv);
  if (result != RDC_ST_OK) {
    return;
  }

  mixed_component_version_t rdcdv;
  uint32_t ret = get_mixed_component_version(rdc_handle_, RDCD_COMPONENT, &rdcdv);
  if (ret) {
    std::cout << "get rdcd version fail" << std::endl;
    return;
  }

  if (is_json_output()) {
    std::cout << "\"version\" : ";
    std::cout << '{';
    std::cout << "\"rdcd\": "
              << "\"" << rdcdv.version << "\", ";
    std::cout << "\"amdsmi_lib\": "
              << "\"" << smiv.version << "\"";
    std::cout << '}';
  } else {
    std::cout << "RDCD : " << rdcdv.version << "  |  "
              << "AMDSMI Library : " << smiv.version << std::endl;
  }

  return;
}

void RdciDiscoverySubSystem::process() {
  if (show_help_) {
    return show_help();
  }

  if (is_list_) {
    return show_attributes();
  }

  if (is_partition_) {
    return show_attributes_with_partitions();
  }

  if (show_version_) {
    return show_version();
  }
}

}  // namespace rdc
}  // namespace amd
