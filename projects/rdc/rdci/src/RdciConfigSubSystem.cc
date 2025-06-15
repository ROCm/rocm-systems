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
#include "RdciConfigSubSystem.h"

#include <getopt.h>
#include <string.h>
#include <unistd.h>

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "common/rdc_utils.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcException.h"
#include "rdc_lib/rdc_common.h"

static constexpr uint32_t TABLE_COLUMN_WIDTH = 20;

static std::string make_fg_name(rdc_gpu_group_t gid) {
  return std::string("RdciConfigSubSystem_group") + std::to_string(gid);
}

namespace amd {
namespace rdc {

RdciConfigSubSystem::RdciConfigSubSystem()
    : config_cmd_(CONFIG_COMMAND_NONE), power_limit_(0), gfx_max_clock_(0), memory_max_clock_(0) {}

RdciConfigSubSystem::~RdciConfigSubSystem() {}

void RdciConfigSubSystem::parse_cmd_opts(int argc, char** argv) {
  const int JSON_OPTIONS = 1001;
  const struct option long_options[] = {{"set", no_argument, nullptr, 's'},
                                        {"get", no_argument, nullptr, 't'},
                                        {"clear", no_argument, nullptr, 'c'},
                                        {"unauth", optional_argument, nullptr, 'u'},
                                        {"group", required_argument, nullptr, 'g'},
                                        {"powerlimit", required_argument, nullptr, 'p'},
                                        {"gfxmaxclk", required_argument, nullptr, 'x'},
                                        {"memmaxclk", required_argument, nullptr, 'm'},
                                        {"help", optional_argument, nullptr, 'h'},
                                        {"json", optional_argument, nullptr, JSON_OPTIONS},
                                        {nullptr, 0, nullptr, 0}};

  int option_index = 0;
  int opt = 0;
  config_cmd_ = CONFIG_COMMAND_NONE;
  bool group_id_set = false;  // ensure set, get, and clear have a group associated with them

  while ((opt = getopt_long(argc, argv, "stcuhg:p:x:m", long_options, &option_index)) != -1) {
    switch (opt) {
      case 's':
        config_cmd_ = CONFIG_COMMAND_SET;
        break;
      case 't':
        config_cmd_ = CONFIG_COMMAND_GET;
        break;
      case 'c':
        config_cmd_ = CONFIG_COMMAND_CLEAR;
        break;
      case 'u':
        use_auth_ = false;
        break;
      case 'h':
        config_cmd_ = CONFIG_COMMAND_HELP;
        break;
      case 'g':
        group_id_ = std::stoi(optarg);
        group_id_set = true;
        break;
      case 'p':
        power_limit_ = std::stoul(optarg);
        break;
      case 'x':
        gfx_max_clock_ = std::stoul(optarg);
        break;
      case 'm':
        memory_max_clock_ = std::stoul(optarg);
        break;
      case JSON_OPTIONS:
        set_json_output(true);
        break;
      default:
        show_help();
        throw RdcException(RDC_ST_BAD_PARAMETER, "Unknown command line options");
    }
  }

  if (config_cmd_ == CONFIG_COMMAND_NONE) {
    show_help();
    throw RdcException(RDC_ST_BAD_PARAMETER, "Must specify a valid operations");
  }

  // Enforce a mandatory group id for set, get, and clear
  if ((config_cmd_ == CONFIG_COMMAND_SET || config_cmd_ == CONFIG_COMMAND_GET ||
       config_cmd_ == CONFIG_COMMAND_CLEAR) &&
      !group_id_set) {
    show_help();
    throw RdcException(
        RDC_ST_BAD_PARAMETER,
        "Must specify a group ID (-g or --group) for set, get, and clear operations");
  }
}

void RdciConfigSubSystem::show_help() const {
  if (is_json_output()) return;
  std::cout << " config -- Used to configure GPU to have configuration across workloads and across "
               "devices.\n\n";
  std::cout << "Usage\n";
  std::cout << "    rdci config --help\n";
  std::cout << "    rdci config  [-g <group id>]  --set [--powerlimit <value>] [--gfxmaxclk "
               "<value>]  [--memmaxclk <value>]\n";
  std::cout << "    rdci config  [-g <group id>]  --get\n";
  std::cout << "    rdci config  [-g <group id>]  --clear\n";
  show_common_usage();
}

void RdciConfigSubSystem::process() {
  rdc_status_t result = RDC_ST_UNKNOWN_ERROR;
  std::ostringstream json_ss;

  switch (config_cmd_) {
    case CONFIG_COMMAND_SET: {
      if (gfx_max_clock_ == 0 && power_limit_ == 0 && memory_max_clock_ == 0) {
        show_help();
        throw RdcException(
            RDC_ST_BAD_PARAMETER,
            "Must specify at least one of --powerlimit, --gfxmaxclk or --memclk when using --set");
      }

      rdc_config_setting_t setting;

      if (gfx_max_clock_ != 0) {
        setting.type = RDC_CFG_GFX_CLOCK_LIMIT;
        setting.target_value = gfx_max_clock_;
        result = rdc_config_set(rdc_handle_, group_id_, setting);
      }

      if (power_limit_ != 0) {
        setting.type = RDC_CFG_POWER_LIMIT;
        setting.target_value = power_limit_;
        result = rdc_config_set(rdc_handle_, group_id_, setting);
      }

      if (memory_max_clock_ != 0) {
        setting.type = RDC_CFG_MEMORY_CLOCK_LIMIT;
        setting.target_value = memory_max_clock_;
        result = rdc_config_set(rdc_handle_, group_id_, setting);
      }

      if (result != RDC_ST_OK) {
        break;
      }

      // If watch group exists, get existing watched fields, tear group down and make new one
      std::vector<rdc_field_t> job_fields;
      rdc_field_grp_t old_fgid;
      std::string fg_name = make_fg_name(group_id_);
      if (rdc_group_field_find(rdc_handle_, fg_name.c_str(), &old_fgid) == RDC_ST_OK) {
        rdc_field_group_info_t info{};
        result = rdc_group_field_get_info(rdc_handle_, old_fgid, &info);
        if (result != RDC_ST_OK) {
          std::cout << "Error unable to get group info for field group: " << old_fgid << std::endl;
          break;
        }
        for (uint32_t i = 0; i < info.count; ++i) {
          job_fields.push_back(info.field_ids[i]);
        }
        result = rdc_field_unwatch(rdc_handle_, group_id_, old_fgid);
        if (result != RDC_ST_OK) {
          std::cout << "Error unable to unwatch field group: " << old_fgid << std::endl;
          break;
        }
        rdc_group_field_destroy(rdc_handle_, old_fgid);
        if (result != RDC_ST_OK) {
          std::cout << "Error unable to destroy field group: " << old_fgid << std::endl;
          break;
        }
        std::cout << "Successfully removed field group: " << old_fgid << std::endl;
      }

      // Add new and existing fields to watch, ignoring duplicates
      auto add_if_new = [&](rdc_field_t f) {
        if (std::find(job_fields.begin(), job_fields.end(), f) == job_fields.end())
          job_fields.push_back(f);
      };
      if (gfx_max_clock_) add_if_new(RDC_FI_GPU_CLOCK);
      if (power_limit_) add_if_new(RDC_FI_POWER_USAGE);
      if (memory_max_clock_) add_if_new(RDC_FI_MEM_CLOCK);

      rdc_field_grp_t fgid;
      result = rdc_group_field_create(rdc_handle_, job_fields.size(), job_fields.data(),
                                      fg_name.c_str(), &fgid);
      if (result != RDC_ST_OK) {
        std::cout << "Error creating field group: " << fgid << std::endl;
        break;
      }
      std::cout << "Successfully created the field group " << fgid << std::endl;

      // Start watching all config values
      const double max_keep_age = 30060;  // Length of time to keep data in field in seconds
      const int max_keep_samples = 10;
      const int update_frequency = 1000000;  // Once per minute
      result = rdc_field_watch(rdc_handle_, group_id_, fgid, update_frequency, max_keep_age,
                               max_keep_samples);
      if (result != RDC_ST_OK) {
        std::cout << "Error creating field watch for group id: " + group_id_ << std::endl;
        break;
      }

      if (is_json_output()) {
        json_ss << "{"
                << "\"group_id\": \"" << group_id_ << "\", \"status\": \"ok\""
                << "}";
      } else {
        std::cout << "Successfully configured GPU Id belongs to group: " << group_id_ << std::endl;
      }
      std::cout << json_ss.str() << std::endl;
      return;
    }
    case CONFIG_COMMAND_GET: {
      rdc_config_setting_list_t settings = {0, {}};
      result = rdc_config_get(rdc_handle_, group_id_, &settings);
      if (result == RDC_ST_OK) {
        display_config_settings(settings);
      } else if (result == RDC_ST_NOT_FOUND) {
        std::cout << "No configuration found for group " << group_id_ << std::endl;
      } else {
        std::cout << "Error retrieving config for group " << group_id_ << std::endl;
      }

      break;
    }
    case CONFIG_COMMAND_CLEAR: {
      result = rdc_config_clear(rdc_handle_, group_id_);

      // Delete the field group and GPU group
      std::vector<rdc_field_t> job_fields;
      rdc_field_grp_t old_fgid;
      std::string fg_name = make_fg_name(group_id_);
      if (rdc_group_field_find(rdc_handle_, fg_name.c_str(), &old_fgid) == RDC_ST_OK) {
        rdc_field_unwatch(rdc_handle_, group_id_, old_fgid);
        rdc_group_field_destroy(rdc_handle_, old_fgid);
      }
      if (result != RDC_ST_OK) {
        std::cout << "Error delete field group. Return: " << rdc_status_string(result);
        break;
      }
      std::cout << "Successfully deleted the field group " << old_fgid << std::endl;

      if (result == RDC_ST_OK) {
        if (is_json_output()) {
          json_ss << "\"group_id\": \"" << group_id_ << "\", \"status\": \"ok\"";
        } else {
          std::cout << "Successfully cleared all configuration for group: " << group_id_
                    << std::endl;
        }
        std::cout << json_ss.str() << std::endl;
        return;
      }
      break;
    }
    case CONFIG_COMMAND_HELP:
      show_help();
      result = RDC_ST_OK;
      break;
    case CONFIG_COMMAND_NONE:
    default:
      throw RdcException(RDC_ST_BAD_PARAMETER, "Unknown command");
  }
  if (result != RDC_ST_OK) {
    throw RdcException(result, rdc_status_string(result));
  }
}

rdc_status_t RdciConfigSubSystem::rdc_group_field_find(rdc_handle_t p_rdc_handle,
                                                       const char* field_group_name,
                                                       rdc_field_grp_t* out_field_group_id) {
  if (p_rdc_handle == nullptr || field_group_name == nullptr || out_field_group_id == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }

  rdc_field_grp_t field_group_id_list[RDC_MAX_NUM_FIELD_GROUPS];
  uint32_t count = 0;
  rdc_status_t status = rdc_group_field_get_all_ids(p_rdc_handle, field_group_id_list, &count);
  if (status != RDC_ST_OK) {
    return status;
  }

  for (uint32_t i = 0; i < count; ++i) {
    rdc_field_group_info_t info;
    status = rdc_group_field_get_info(p_rdc_handle, field_group_id_list[i], &info);
    if (status != RDC_ST_OK) {
      continue;
    }

    if (strncmp(info.group_name, field_group_name, RDC_MAX_STR_LENGTH) == 0) {
      *out_field_group_id = field_group_id_list[i];
      return RDC_ST_OK;
    }
  }

  return RDC_ST_NOT_FOUND;
}

// Our display config settings is called with rdc config get.
// We pass in rdc_configs_list which is all our config settings for the specified group.
// We read from the rdc cache, and display the live value of that setting.
// Remember, rdc cache is updated with our watcher values upon config set
void RdciConfigSubSystem::display_config_settings(rdc_config_setting_list_t& rdc_configs_list) {
  rdc_status_t result = RDC_ST_OK;
  std::stringstream ss, json_ss;
  rdc_group_info_t rdc_group_info = {0, "", {0}};
  uint32_t gpu_index = 0;
  uint32_t index = 0;

  ss << std::setw(TABLE_COLUMN_WIDTH) << std::left << "configure" << std::setw(TABLE_COLUMN_WIDTH)
     << std::left << "gpu_index" << std::setw(TABLE_COLUMN_WIDTH) << std::left << "config_limit"
     << std::setw(TABLE_COLUMN_WIDTH) << std::left << "current_value" << std::endl;
  json_ss << "\"group_id\": " << group_id_ << ","
          << "\"config_list\" : [";

  result = rdc_group_gpu_get_info(rdc_handle_, group_id_, &rdc_group_info);
  if (result == RDC_ST_OK) {
    std::vector<uint32_t> group_index_array(rdc_group_info.entity_ids,
                                            rdc_group_info.entity_ids + rdc_group_info.count);
    sort(begin(group_index_array), end(group_index_array));
    for (uint32_t i = 0; i < rdc_configs_list.total_settings && i < RDC_GROUP_MAX_ENTITIES; ++i) {
      auto type = rdc_configs_list.settings[i].type;
      auto config_value = rdc_configs_list.settings[i].target_value;

      for (gpu_index = 0; gpu_index < rdc_group_info.count && gpu_index < RDC_GROUP_MAX_ENTITIES;
           ++gpu_index) {
        json_ss << "{\"gpu_index\": " << group_index_array[gpu_index] << ",";

        rdc_field_value value;
        switch (type) {
          case RDC_CFG_GFX_CLOCK_LIMIT: {
            json_ss << "\"GFX Clock Limit\":" << config_value;
            ss << std::setw(TABLE_COLUMN_WIDTH) << std::left << "gfx_clock_limit"
               << std::setw(TABLE_COLUMN_WIDTH) << std::left << group_index_array[gpu_index]
               << std::setw(TABLE_COLUMN_WIDTH) << std::left << config_value
               << std::setw(TABLE_COLUMN_WIDTH) << std::left;

            result = rdc_field_get_latest_value(rdc_handle_, group_index_array[gpu_index],
                                                RDC_FI_GPU_CLOCK, &value);
            json_ss << ",\"GFX Clock Current Value\":";
            if (result != RDC_ST_OK) {
              ss << "N/A";
              json_ss << "\"N/A\"";
            } else {
              if (value.type == INTEGER) {
                double mhz = static_cast<double>(value.value.l_int) / 1'000'000.0;
                ss << std::fixed << std::setprecision(0) << mhz;
                json_ss << mhz;
              } else if (value.type == DOUBLE) {
                double mhz = value.value.dbl / 1'000'000.0;
                ss << std::fixed << std::setprecision(0) << mhz;
                json_ss << mhz;
              } else {
                ss << value.value.str;
                json_ss << value.value.str;
              }
            }
            ss << std::endl;
            break;
          }
          case RDC_CFG_MEMORY_CLOCK_LIMIT:
            json_ss << "\"Memory Clock Limit\":" << config_value;
            ss << std::setw(TABLE_COLUMN_WIDTH) << std::left << "memory_clock_limit"
               << std::setw(TABLE_COLUMN_WIDTH) << std::left << group_index_array[gpu_index]
               << std::setw(TABLE_COLUMN_WIDTH) << std::left << config_value
               << std::setw(TABLE_COLUMN_WIDTH) << std::left;

            result = rdc_field_get_latest_value(rdc_handle_, group_index_array[gpu_index],
                                                RDC_FI_MEM_CLOCK, &value);
            json_ss << ",\"Memory Clock Current Value\":";
            if (result != RDC_ST_OK) {
              ss << "N/A";
              json_ss << "\"N/A\"";
            } else {
              if (value.type == INTEGER) {
                double mhz = static_cast<double>(value.value.l_int) / 1'000'000.0;
                ss << std::fixed << std::setprecision(0) << mhz;
                json_ss << mhz;
              } else if (value.type == DOUBLE) {
                double mhz = value.value.dbl / 1'000'000.0;
                ss << std::fixed << std::setprecision(0) << mhz;
                json_ss << mhz;
              } else {
                ss << value.value.str;
                json_ss << value.value.str;
              }
            }
            ss << std::endl;
            break;
          case RDC_CFG_POWER_LIMIT:
            json_ss << "\"Power Limit\":" << config_value;
            ss << std::setw(TABLE_COLUMN_WIDTH) << std::left << "power_limit"
               << std::setw(TABLE_COLUMN_WIDTH) << std::left << group_index_array[gpu_index]
               << std::setw(TABLE_COLUMN_WIDTH) << std::left << config_value
               << std::setw(TABLE_COLUMN_WIDTH) << std::left;

            result = rdc_field_get_latest_value(rdc_handle_, group_index_array[gpu_index],
                                                RDC_FI_POWER_USAGE, &value);
            json_ss << ",\"Power Current Value\":";
            if (result != RDC_ST_OK) {
              ss << "N/A";
              json_ss << "\"N/A\"";
            } else {
              if (value.type == INTEGER) {
                double watts = static_cast<double>(value.value.l_int) / 1'000'000;
                ss << std::fixed << std::setprecision(3) << watts;
                json_ss << watts;
              } else if (value.type == DOUBLE) {
                double watts = value.value.dbl / 1'000'000;
                ss << std::fixed << std::setprecision(3) << watts;
                json_ss << watts;
              } else {
                ss << value.value.str;
                json_ss << value.value.str;
              }
            }
            ss << std::endl;
            break;
          default:
            break;
        }
        // Set the json seperator
        json_ss << "}";
        if ((gpu_index + 1) != rdc_group_info.count) {
          json_ss << ",";
        }
      }
      if (rdc_group_info.count != 0 && i < rdc_configs_list.total_settings - 1 &&
          i < RDC_GROUP_MAX_ENTITIES - 1) {
        json_ss << ",";
      }
    }
  }
  if (index != 0) {
    json_ss << "}";
  }
  json_ss << "]";
  if (is_json_output()) {
    std::cout << json_ss.str() << std::endl;
  } else {
    std::cout << ss.str() << std::endl;
  }
}

}  // namespace rdc
}  // namespace amd
