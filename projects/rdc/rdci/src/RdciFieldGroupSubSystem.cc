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
#include "RdciFieldGroupSubSystem.h"

#include <getopt.h>
#include <unistd.h>

#include "common/rdc_fields_supported.h"
#include "common/rdc_utils.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcException.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

RdciFieldGroupSubSystem::RdciFieldGroupSubSystem()
    : field_group_ops_(FIELD_GROUP_UNKNOWN), is_group_set_(false) {}

void RdciFieldGroupSubSystem::parse_cmd_opts(int argc, char** argv) {
  const int HOST_OPTIONS = 1000;
  const int JSON_OPTIONS = 1001;
  const struct option long_options[] = {{"host", required_argument, nullptr, HOST_OPTIONS},
                                        {"help", optional_argument, nullptr, 'h'},
                                        {"unauth", optional_argument, nullptr, 'u'},
                                        {"list", optional_argument, nullptr, 'l'},
                                        {"group", required_argument, nullptr, 'g'},
                                        {"create", required_argument, nullptr, 'c'},
                                        {"fieldids", required_argument, nullptr, 'f'},
                                        {"info", optional_argument, nullptr, 'i'},
                                        {"delete", required_argument, nullptr, 'd'},
                                        {"json", optional_argument, nullptr, JSON_OPTIONS},
                                        {nullptr, 0, nullptr, 0}};

  int option_index = 0;
  int opt = 0;

  while ((opt = getopt_long(argc, argv, "hluif:c:g:d:", long_options, &option_index)) != -1) {
    switch (opt) {
      case HOST_OPTIONS:
        ip_port_ = optarg;
        break;
      case JSON_OPTIONS:
        set_json_output(true);
        break;
      case 'h':
        field_group_ops_ = FIELD_GROUP_HELP;
        return;
      case 'u':
        use_auth_ = false;
        break;
      case 'l':
        field_group_ops_ = FIELD_GROUP_LIST;
        break;
      case 'f':
        field_ids_ = optarg;
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
        field_group_ops_ = FIELD_GROUP_CREATE;
        group_name_ = optarg;
        break;
      case 'i':
        field_group_ops_ = FIELD_GROUP_INFO;
        break;
      case 'd':
        field_group_ops_ = FIELD_GROUP_DELETE;
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

  if (field_group_ops_ == FIELD_GROUP_UNKNOWN) {
    show_help();
    throw RdcException(RDC_ST_BAD_PARAMETER, "Must specify a valid operations");
  }
}

void RdciFieldGroupSubSystem::show_help() const {
  if (is_json_output()) return;
  std::cout << " fieldgroup -- Used to  create and maintain groups "
            << "of field Ids.\n\n";
  std::cout << "Usage\n";
  std::cout << "    rdci fieldgroup [--host <IP/FQDN>:port]"
            << " [--json] [-u] -l\n";
  std::cout << "    rdci fieldgroup [--host <IP/FQDN>:port] [--json]"
            << " [-u] -c <groupName> -f <filedIds>\n";
  std::cout << "    rdci fieldgroup [--host <IP/FQDN>:port] [--json] [-u] "
            << "-g <groupId> -i\n";
  std::cout << "    rdci fieldgroup [--host <IP/FQDN>:port] [--json] [-u] "
            << "-d <groupId>\n";
  std::cout << "\nFlags:\n";
  show_common_usage();
  std::cout << "  --json                         "
            << "Output using json.\n";
  std::cout << "  -l  --list                     "
            << "List the field groups that currently exist for a host.\n";
  std::cout << "  -g  --group groupId            "
            << "The field group to query on the specified host.\n";
  std::cout << "  -c  --create groupName         "
            << "Create a field group on the remote host.\n";
  std::cout << "  -f  --fieldids fieldIds        Comma-separated "
            << "list of the field ids to add to a field group\n";
  std::cout << "  -i  --info                     "
            << "Display the information for the specified group Id\n";
  std::cout << "  -d  --delete groupId           "
            << "Delete a field group on the remote host.\n";
}

void RdciFieldGroupSubSystem::process() {
  rdc_status_t result = RDC_ST_OK;
  rdc_field_group_info_t group_info;
  uint32_t count = 0;
  std::string json_group_ids = "\"field_groups\": [";
  switch (field_group_ops_) {
    case FIELD_GROUP_HELP:
      show_help();
      break;
    case FIELD_GROUP_CREATE: {
      if (group_name_ == "") {
        show_help();
        throw RdcException(RDC_ST_BAD_PARAMETER,
                           "Must specify the group name when create a field group");
      }
      std::vector<std::string> fields = split_string(field_ids_, ',');
      rdc_field_t field_ids[RDC_MAX_FIELD_IDS_PER_FIELD_GROUP];
      for (uint32_t i = 0; i < fields.size(); i++) {
        if (!IsNumber(fields[i])) {
          if (!get_field_id_from_name(fields[i], &field_ids[i])) {
            throw RdcException(RDC_ST_BAD_PARAMETER,
                               "The field name " + fields[i] + " is not valid");
          }
        } else {
          field_ids[i] = static_cast<rdc_field_t>(std::stoi(fields[i]));
        }
      }
      rdc_field_grp_t group_id;
      result = rdc_group_field_create(rdc_handle_, fields.size(), &field_ids[0],
                                      group_name_.c_str(), &group_id);
      if (result == RDC_ST_OK) {
        if (is_json_output()) {
          std::cout << "\"field_group_id\": \"" << group_id << "\", \"status\": \"ok\"";
        } else {
          std::cout << "Successfully created a field group"
                    << " with a group ID " << group_id << std::endl;
          return;
        }
      }
      break;
    }
    case FIELD_GROUP_DELETE:
      if (!is_group_set_) {
        show_help();
        throw RdcException(RDC_ST_BAD_PARAMETER, "Need to specify the group id to delete a group");
      }
      result = rdc_group_field_destroy(rdc_handle_, group_id_);
      if (result == RDC_ST_OK) {
        if (is_json_output()) {
          std::cout << "\"field_group_id\": \"" << group_id_ << "\", \"status\": \"ok\"";
        } else {
          std::cout << "Successfully deleted the field group " << group_id_ << std::endl;
        }
        return;
      }
      break;
    case FIELD_GROUP_LIST:
      rdc_field_grp_t group_id_list[RDC_MAX_NUM_FIELD_GROUPS];
      result = rdc_group_field_get_all_ids(rdc_handle_, group_id_list, &count);
      if (result != RDC_ST_OK) break;

      if (!is_json_output()) {
        std::cout << count << " field group found.\n";
        std::cout << "GroupID\t"
                  << "GroupName\t"
                  << "FieldIds\n";
      }

      for (uint32_t i = 0; i < count; i++) {
        result = rdc_group_field_get_info(rdc_handle_, group_id_list[i], &group_info);
        if (result != RDC_ST_OK) {
          throw RdcException(RDC_ST_BAD_PARAMETER, "Fail to get information for field group " +
                                                       std::to_string(group_id_list[i]));
        }

        if (!is_json_output()) {
          std::cout << group_id_list[i] << "\t" << group_info.group_name << "\t\t";
        } else {
          json_group_ids += "{\"group_id\": \"";
          json_group_ids += std::to_string(group_id_list[i]);
          json_group_ids += "\", \"group_name\": \"";
          json_group_ids += group_info.group_name;
          json_group_ids += "\", \"field_ids\": [";
        }

        for (uint32_t j = 0; j < group_info.count; j++) {
          if (!is_json_output()) {
            std::cout << group_info.field_ids[j];
          } else {
            json_group_ids += std::to_string(group_info.field_ids[j]);
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
    case FIELD_GROUP_INFO:
      if (!is_group_set_) {
        show_help();
        throw RdcException(RDC_ST_BAD_PARAMETER,
                           "Need to specify the group id to show field group info");
      }
      result = rdc_group_field_get_info(rdc_handle_, group_id_, &group_info);
      if (result == RDC_ST_OK) {
        if (is_json_output()) {
          std::cout << "\"group_name\": \"" << group_info.group_name << "\", \"field_ids\": [";
        } else {
          std::cout << "Group name: " << group_info.group_name << std::endl;
          std::cout << "Field Ids: ";
        }
        for (uint32_t i = 0; i < group_info.count; i++) {
          if (is_json_output()) {
            std::cout << group_info.field_ids[i];
            if (i != group_info.count - 1) {
              std::cout << ",";
            }
          } else {
            std::cout << group_info.field_ids[i] << " ";
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
