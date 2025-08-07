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
#include "RdciGroupSubSystem.h"

#include <getopt.h>
#include <unistd.h>

#include "common/rdc_utils.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcException.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

RdciGroupSubSystem::RdciGroupSubSystem() : group_ops_(GROUP_UNKNOWN), is_group_set_(false) {}

void RdciGroupSubSystem::parse_cmd_opts(int argc, char** argv) {
  const int HOST_OPTIONS = 1000;
  const int JSON_OPTIONS = 1001;
  const struct option long_options[] = {{"host", required_argument, nullptr, HOST_OPTIONS},
                                        {"help", optional_argument, nullptr, 'h'},
                                        {"unauth", optional_argument, nullptr, 'u'},
                                        {"list", optional_argument, nullptr, 'l'},
                                        {"group", required_argument, nullptr, 'g'},
                                        {"create", required_argument, nullptr, 'c'},
                                        {"add", required_argument, nullptr, 'a'},
                                        {"info", optional_argument, nullptr, 'i'},
                                        {"delete", required_argument, nullptr, 'd'},
                                        {"json", optional_argument, nullptr, JSON_OPTIONS},
                                        {nullptr, 0, nullptr, 0}};

  int option_index = 0;
  int opt = 0;

  while ((opt = getopt_long(argc, argv, "hluic:g:a:d:", long_options, &option_index)) != -1) {
    switch (opt) {
      case HOST_OPTIONS:
        ip_port_ = optarg;
        break;
      case JSON_OPTIONS:
        set_json_output(true);
        break;
      case 'h':
        group_ops_ = GROUP_HELP;
        return;
      case 'u':
        use_auth_ = false;
        break;
      case 'l':
        group_ops_ = GROUP_LIST;
        break;
      case 'g':
        if (!IsNumber(optarg)) {
          show_help();
          throw RdcException(RDC_ST_BAD_PARAMETER, "The group id needs to be a number");
        }
        group_id_ = std::stoi(optarg);
        is_group_set_ = true;
        break;
      case 'c':
        group_ops_ = GROUP_CREATE;
        group_name_ = optarg;
        break;
      case 'a':
        // Create may add GPUs as well.
        if (group_ops_ != GROUP_CREATE) {
          group_ops_ = GROUP_ADD_GPUS;
        }
        gpu_ids_ = optarg;
        break;
      case 'i':
        group_ops_ = GROUP_INFO;
        break;
      case 'd':
        group_ops_ = GROUP_DELETE;
        if (!IsNumber(optarg)) {
          show_help();
          throw RdcException(RDC_ST_BAD_PARAMETER, "The group id needs to be a number");
        }
        group_id_ = std::stoi(optarg);
        is_group_set_ = true;
        break;
      default:
        show_help();
        throw RdcException(RDC_ST_BAD_PARAMETER, "Unknown command line options");
    }
  }

  if (group_ops_ == GROUP_UNKNOWN) {
    show_help();
    throw RdcException(RDC_ST_BAD_PARAMETER, "Must specify a valid operations");
  }
}

void RdciGroupSubSystem::show_help() const {
  if (is_json_output()) return;
  std::cout << " group -- Used to  create and maintain groups of GPUs.\n\n";
  std::cout << "Usage\n";
  std::cout << "    rdci group [--host <IP/FQDN>:port] [--json] [-u] -l\n";
  std::cout << "    rdci group [--host <IP/FQDN>:port] [--json] [-u]"
            << " -c <groupName> [-a <entityId>]\n";
  std::cout << "    rdci group [--host <IP/FQDN>:port] [--json] [-u]"
            << " -g <groupId> [-a <entityId>]\n";
  std::cout << "    rdci group [--host <IP/FQDN>:port] [--json] [-u] "
            << "-g <groupId> [-i]\n";
  std::cout << "    rdci group [--host <IP/FQDN>:port] [--json] [-u] "
            << "-d <groupId>\n";
  std::cout << "\nFlags:\n";
  show_common_usage();
  std::cout << "  --json                         "
            << "Output using json.\n";
  std::cout << "  -l  --list                     "
            << "List the groups that currently exist for a host.\n";
  std::cout << "  -g  --group groupId            "
            << "The GPU group to query on the specified host.\n";
  std::cout << "  -c  --create groupName         "
            << "Create a group on the remote host.\n";
  std::cout << "  -a  --add    gpuIndexes        "
            << "Comma-separated list of the GPU indexes to add to the group.\n";
  std::cout << "  -i  --info                     "
            << "Display the information for the specified group Id\n";
  std::cout << "  -d  --delete groupId           "
            << "Delete a group on the remote host.\n";
}

void RdciGroupSubSystem::process() {
  rdc_status_t result = RDC_ST_OK;
  std::vector<std::string> gpu_ids;
  rdc_group_info_t group_info;
  uint32_t count = 0;
  std::string json_group_ids = "\"gpu_groups\": [";
  switch (group_ops_) {
    case GROUP_HELP:
      show_help();
      break;
    case GROUP_CREATE:
      if (group_name_ == "") {
        show_help();
        throw RdcException(RDC_ST_BAD_PARAMETER, "Must specify the group name when create a group");
      }
      rdc_gpu_group_t group_id;
      result = rdc_group_gpu_create(rdc_handle_, RDC_GROUP_EMPTY, group_name_.c_str(), &group_id);
      if (result != RDC_ST_OK) {
        throw RdcException(result, "Fail to create group " + group_name_);
      }

      gpu_ids = split_string(gpu_ids_, ',');
      for (uint32_t i = 0; i < gpu_ids.size(); i++) {
        if (!IsNumber(gpu_ids[i])) {
          throw RdcException(RDC_ST_BAD_PARAMETER,
                             "The GPU Id " + gpu_ids[i] + " needs to be a number");
        }
        result = rdc_group_gpu_add(rdc_handle_, group_id, std::stoi(gpu_ids[i]));
        if (result != RDC_ST_OK) {
          throw RdcException(result, "Fail to add GPU " + gpu_ids[i] + " to the group");
        }
      }

      if (result == RDC_ST_OK) {
        if (is_json_output()) {
          std::cout << "\"group_id\": \"" << group_id << "\", \"status\": \"ok\"";
        } else {
          std::cout << "Successfully created group with a group ID " << group_id << std::endl;
        }
        return;
      }
      break;
    case GROUP_DELETE:
      if (!is_group_set_) {
        show_help();
        throw RdcException(RDC_ST_BAD_PARAMETER, "Need to specify the group id to delete a group");
      }
      result = rdc_group_gpu_destroy(rdc_handle_, group_id_);
      if (result == RDC_ST_OK) {
        if (is_json_output()) {
          std::cout << "\"group_id\": \"" << group_id_ << "\", \"status\": \"ok\"";
        } else {
          std::cout << "Successfully deleted the group " << group_id_ << std::endl;
        }
        return;
      }
      break;
    case GROUP_LIST:
      rdc_gpu_group_t group_id_list[RDC_MAX_NUM_GROUPS];
      result = rdc_group_get_all_ids(rdc_handle_, group_id_list, &count);
      if (result != RDC_ST_OK) break;

      if (!is_json_output()) {
        std::cout << count << " group found.\n";
        std::cout << "GroupID\t"
                  << "GroupName\t"
                  << "GPUIndex\n";
      }
      for (uint32_t i = 0; i < count; i++) {
        result = rdc_group_gpu_get_info(rdc_handle_, group_id_list[i], &group_info);
        if (result != RDC_ST_OK) {
          throw RdcException(RDC_ST_BAD_PARAMETER, "Fail to get information for group " +
                                                       std::to_string(group_id_list[i]));
        }

        if (!is_json_output()) {
          std::cout << group_id_list[i] << "\t" << group_info.group_name << "\t\t";
        } else {
          json_group_ids += "{\"group_id\": \"";
          json_group_ids += std::to_string(group_id_list[i]);
          json_group_ids += "\", \"group_name\": \"";
          json_group_ids += group_info.group_name;
          json_group_ids += "\", \"gpu_indexes\": [";
        }
        for (uint32_t j = 0; j < group_info.count; j++) {
          if (!is_json_output()) {
            std::cout << group_info.entity_ids[j];
          } else {
            json_group_ids += std::to_string(group_info.entity_ids[j]);
          }
          if (j < group_info.count - 1) {
            if (!is_json_output()) {
              std::cout << ",";
            } else {
              json_group_ids += ",";
            }
          }
        }
        if (!is_json_output()) {
          std::cout << std::endl;
        } else {
          json_group_ids += "]}";
          if (i != count - 1) {
            json_group_ids += ",";
          }
        }
      }
      if (is_json_output()) {
        json_group_ids += "], \"status\": \"ok\"";
        std::cout << json_group_ids;
      }
      break;
    case GROUP_ADD_GPUS:
      if (!is_group_set_) {
        show_help();
        throw RdcException(RDC_ST_BAD_PARAMETER, "Need to specify the group id to add a group");
      }

      gpu_ids = split_string(gpu_ids_, ',');
      for (uint32_t i = 0; i < gpu_ids.size(); i++) {
        if (!IsNumber(gpu_ids[i])) {
          throw RdcException(RDC_ST_BAD_PARAMETER,
                             "The GPU Id " + gpu_ids[i] + " needs to be a number");
        }
        result = rdc_group_gpu_add(rdc_handle_, group_id_, std::stoi(gpu_ids[i]));
        if (result != RDC_ST_OK) {
          throw RdcException(result, "Fail to add GPU " + gpu_ids[i] + " to the group");
        }
      }
      if (result == RDC_ST_OK) {
        if (is_json_output()) {
          std::cout << "\"group_id\": \"" << group_id_ << "\", \"status\": \"ok\"";
        } else {
          std::cout << "Successfully added the GPU " << gpu_ids_ << " to group " << group_id_
                    << std::endl;
        }
        return;
      }
      break;
    case GROUP_INFO:
      if (!is_group_set_) {
        show_help();
        throw RdcException(RDC_ST_BAD_PARAMETER, "Need to specify the group id to show group info");
      }
      result = rdc_group_gpu_get_info(rdc_handle_, group_id_, &group_info);
      if (result == RDC_ST_OK) {
        if (is_json_output()) {
          std::cout << "\"group_name\": \"" << group_info.group_name << "\", \"gpu_indexes\": [";
        } else {
          std::cout << "Group name: " << group_info.group_name << std::endl;
          std::cout << "Gpu indexes: ";
        }
        for (uint32_t i = 0; i < group_info.count; i++) {
          if (is_json_output()) {
            std::cout << group_info.entity_ids[i];
            if (i != group_info.count - 1) {
              std::cout << ",";
            }
          } else {
            std::cout << group_info.entity_ids[i] << " ";
          }
        }
        if (is_json_output()) {
          std::cout << "], \"status\": \"ok\"";
        } else {
          std::cout << std::endl;
        }
        return;
      }
      break;
    default:
      throw RdcException(RDC_ST_BAD_PARAMETER, "Unknown command");
  }

  if (result != RDC_ST_OK) {
    throw RdcException(result, rdc_status_string(result));
  }
}

}  // namespace rdc
}  // namespace amd
