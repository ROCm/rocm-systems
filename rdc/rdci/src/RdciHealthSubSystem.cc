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
#include "RdciHealthSubSystem.h"

#include <getopt.h>
#include <signal.h>
#include <unistd.h>

#include <ctime>
#include <iomanip>
#include <limits>

#include "common/rdc_utils.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcException.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

RdciHealthSubSystem::RdciHealthSubSystem() {}

RdciHealthSubSystem::~RdciHealthSubSystem() {}

void RdciHealthSubSystem::parse_cmd_opts(int argc, char** argv) {
  const int HOST_OPTIONS = 1000;
  const int JSON_OPTIONS = 1001;
  const int CLEAR_OPTIONS = 1002;
  const struct option long_options[] = {{"host", required_argument, nullptr, HOST_OPTIONS},
                                        {"unauth", optional_argument, nullptr, 'u'},
                                        {"help", optional_argument, nullptr, 'h'},
                                        {"json", optional_argument, nullptr, JSON_OPTIONS},
                                        {"clear", optional_argument, nullptr, CLEAR_OPTIONS},
                                        {"group", required_argument, nullptr, 'g'},
                                        {"fetch", optional_argument, nullptr, 'f'},
                                        {"set", required_argument, nullptr, 's'},
                                        {"check", optional_argument, nullptr, 'c'},
                                        {nullptr, 0, nullptr, 0}};

  bool group_id_set = false;
  int option_index = 0, opt = 0;
  std::string flags;
  unsigned int components = 0;

  while ((opt = getopt_long(argc, argv, "uhg:fs:c", long_options, &option_index)) != -1) {
    switch (opt) {
      case HOST_OPTIONS:
        ip_port_ = optarg;
        break;

      case JSON_OPTIONS:
        set_json_output(true);
        break;

      case CLEAR_OPTIONS:
        health_ops_ = HEALTH_CLEAR;
        break;

      case 'u':
        use_auth_ = false;
        break;

      case 'h':
        health_ops_ = HEALTH_HELP;
        return;

      case 'g':
        if (!IsNumber(optarg)) {
          show_help();
          throw RdcException(RDC_ST_BAD_PARAMETER, "The group id needs to be a number");
        }
        group_id_ = std::stoi(optarg);
        group_id_set = true;
        break;

      case 'f':
        health_ops_ = HEALTH_FETCH;
        break;

      case 's':
        health_ops_ = HEALTH_SET;

        flags = optarg;
        for (unsigned int i = 0; i < flags.length(); i++) {
          switch (flags.at(i)) {
            case 'a':
              components |= RDC_HEALTH_WATCH_PCIE;
              components |= RDC_HEALTH_WATCH_XGMI;
              components |= RDC_HEALTH_WATCH_MEM;
              components |= RDC_HEALTH_WATCH_EEPROM;
              components |= RDC_HEALTH_WATCH_THERMAL;
              components |= RDC_HEALTH_WATCH_POWER;
              break;

            case 'p':
              components |= RDC_HEALTH_WATCH_PCIE;
              break;

            case 'm':
              components |= RDC_HEALTH_WATCH_MEM;
              break;

            case 'e':
              components |= RDC_HEALTH_WATCH_EEPROM;
              break;

            case 't':
              components |= RDC_HEALTH_WATCH_THERMAL;
              components |= RDC_HEALTH_WATCH_POWER;
              break;

            case 'x':
              components |= RDC_HEALTH_WATCH_XGMI;
              break;

            default:
              throw RdcException(RDC_ST_BAD_PARAMETER, "Invalid flags");
              break;
          }
        }

        if (0 == components) {
            throw RdcException(RDC_ST_BAD_PARAMETER, "No flags");
        } else
          components_ = components;
        break;

      case 'c':
        health_ops_ = HEALTH_CHECK;
        break;

      default:
        show_help();
        throw RdcException(RDC_ST_BAD_PARAMETER, "Unknown command line options");
    }
  }

  if (!group_id_set) {
    show_help();
    throw RdcException(RDC_ST_BAD_PARAMETER, "Need to specify the GPU group id");
  }
}

void RdciHealthSubSystem::show_help() const {
  if (is_json_output()) return;
  std::cout << " health -- Used to manage the health watches of a group. \n"
            << " The health of the GPUs in a group can then be monitored"
            << " during a process.\n\n";
  std::cout << "Usage\n";
  std::cout << "    rdci health [--host <IP/FQDN>:port] [-u] [-j] -g <groupId> -s <flags>\n";
  std::cout << "    rdci health [--host <IP/FQDN>:port] [-u] [-j] -g <groupId> -c\n";
  std::cout << "    rdci health [--host <IP/FQDN>:port] [-u] [-j] -g <groupId> -f\n";
  std::cout << "    rdci health [--host <IP/FQDN>:prot] [-u] [-j] -g <groupId> --clear\n";
  std::cout << "\nFlags:\n";
  show_common_usage();
  std::cout << "  --json                         Output using json.\n";
  std::cout << "  --clear                        Disable all watches being monitored.\n";
  std::cout << "  -g  --group    groupId         The GPU group to query "
            << "on the specified host.\n";
  std::cout << "  -f  --fetch                    Fetch the current watch status.\n";
  std::cout << "  -s  --set      flags           The list of components can be watched. "
            << "[default = pm]\n";
  std::cout << "                                  a - watch all components\n";
  std::cout << "                                  p - watch PCIe\n";
  std::cout << "                                  m - watch Memory\n";
  std::cout << "                                  e - watch EEPROM\n";
  std::cout << "                                  t - watch power and thermal\n";
  std::cout << "                                  x - watch XGMI\n";
  std::cout << "  -c  --check                    Check to see if any errors or warnings have "
            << "occurred in the currently monitored watches.\n";
}

void RdciHealthSubSystem::get_watches() const {
  rdc_status_t result;
  unsigned int components = 0;
  std::string on = "On";
  std::string off = "Off";

  result = rdc_health_get(rdc_handle_, group_id_, &components);
  if (result != RDC_ST_OK) {
    std::string error_msg = rdc_status_string(result);
    if (result == RDC_ST_NOT_FOUND) {
      error_msg = "Cannot find the group " + std::to_string(group_id_) + " or the field.";
    }
    throw RdcException(result, error_msg.c_str());
  }

  if (is_json_output()) {
    std::cout << "\"heading\" : \"Health monitor systems status\", ";
    std::cout << "\"body\" : [";
    std::cout << "{\"Component\" : \"PCIe\", \"Status\" : \"" << ((components & RDC_HEALTH_WATCH_PCIE) ? on : off).c_str() << "\"},";
    std::cout << "{\"Component\" : \"XGMI\", \"Status\" : \"" << ((components & RDC_HEALTH_WATCH_XGMI) ? on : off).c_str() << "\"},";
    std::cout << "{\"Component\" : \"Memory\", \"Status\" : \"" << ((components & RDC_HEALTH_WATCH_MEM) ? on : off).c_str() << "\"},";
    std::cout << "{\"Component\" : \"EEPROM\", \"Status\" : \"" << ((components & RDC_HEALTH_WATCH_EEPROM) ? on : off).c_str() << "\"},";
    std::cout << "{\"Component\" : \"Thermal\", \"Status\" : \"" << ((components & RDC_HEALTH_WATCH_THERMAL) ? on : off).c_str() << "\"},";
    std::cout << "{\"Component\" : \"Power\", \"Status\" : \"" << ((components & RDC_HEALTH_WATCH_POWER) ? on : off).c_str() << "\"}";
    std::cout << "]";
  } else {
    std::cout << "Health monitor systems status:" << std::endl;
    std::cout << "+--------------------+" //"-" width :20
              << "---------------------------------------------------+\n"; //-" width :51
    std::cout << "|" << std::setw(20) << std::left << " PCIe"    << "| "
              << std::setw(50) << std::left << ((components & RDC_HEALTH_WATCH_PCIE) ? on : off).c_str() << "|\n";
    std::cout << "|" << std::setw(20) << std::left << " XGMI"    << "| "
              << std::setw(50) << std::left << ((components & RDC_HEALTH_WATCH_XGMI) ? on : off).c_str() << "|\n";
    std::cout << "|" << std::setw(20) << std::left << " Memory"  << "| "
              << std::setw(50) << std::left << ((components & RDC_HEALTH_WATCH_MEM) ? on : off).c_str() << "|\n";
    std::cout << "|" << std::setw(20) << std::left << " EEPROM" << "| "
              << std::setw(50) << std::left << ((components & RDC_HEALTH_WATCH_EEPROM) ? on : off).c_str() << "|\n";
    std::cout << "|" << std::setw(20) << std::left << " Thermal" << "| "
              << std::setw(50) << std::left << ((components & RDC_HEALTH_WATCH_THERMAL) ? on : off).c_str() << "|\n";
    std::cout << "|" << std::setw(20) << std::left << " Power"   << "| "
              << std::setw(50) << std::left << ((components & RDC_HEALTH_WATCH_POWER) ? on : off).c_str() << "|\n";
    std::cout << "+--------------------+" //"-" width :20
              << "---------------------------------------------------+\n"; //-" width :51
  }
}

void RdciHealthSubSystem::set_watches() const {
  rdc_status_t result;

  result = rdc_health_set(rdc_handle_, group_id_, components_);
  if (result != RDC_ST_OK) {
    std::string error_msg = rdc_status_string(result);
    if (result == RDC_ST_NOT_FOUND) {
      error_msg = "Cannot find the group " + std::to_string(group_id_) + " or the field.";
    }
    throw RdcException(result, error_msg.c_str());
  }

  std::cout << "Group " << group_id_ << " health monitor systems set successfully." << std::endl;
}

std::string RdciHealthSubSystem::health_string(rdc_health_result_t health) const {
  switch (health) {
    case RDC_HEALTH_RESULT_PASS:
      return "Pass";

    case RDC_HEALTH_RESULT_WARN:
      return "Warning";

    case RDC_HEALTH_RESULT_FAIL:
      return "Fail";

    default:
      return "Unknown";
  }
}

std::string RdciHealthSubSystem::component_string(rdc_health_system_t component) const {
    switch (component) {
      case RDC_HEALTH_WATCH_PCIE:
        return "PCIe system: ";

      case RDC_HEALTH_WATCH_XGMI:
        return"XGMI system: ";

      case RDC_HEALTH_WATCH_MEM:
        return "Memory system: ";

      case RDC_HEALTH_WATCH_EEPROM:
        return "EEPROM system: ";

      case RDC_HEALTH_WATCH_THERMAL:
        return "Thermal system:";

      case RDC_HEALTH_WATCH_POWER:
        return "Power system: ";

      default:
        return "Unknown";
    }
}

void RdciHealthSubSystem::output_errstr(const std::string& input) const {
  std::string word, line_str;
  unsigned int width = 60, line_size = 0;
  std::istringstream iss(input);

  while (iss >> word) {
    if (line_size + word.size() >= width) {
      std::cout << "|" << std::setw(20) << " " << "| "
                << std::setw(width) << std::left << line_str << "|\n";

      //add new line string
      line_str = word;
      line_size = word.size();
    } else {
      if (line_size > 0) {
        line_str += " ";
        line_str += word;
        line_size += word.size() + 1;
      } else {
        line_str += word;
        line_size += word.size();
      }
    }
  } //end while

  if (0 < line_size)
      std::cout << "|" << std::setw(20) << " " << "| "
                << std::setw(width) << std::left << line_str << "|\n";
}

unsigned int RdciHealthSubSystem::handle_one_component(rdc_health_response_t &response,
                                                       unsigned int start_index,
                                                       uint32_t gpu_index,
                                                       rdc_health_system_t component,
                                                       rdc_health_result_t &component_health,
                                                       std::vector<std::string> &err_str) const {
  unsigned int count = 0;
  rdc_health_incidents_t *incident;
  std::string all_err_str;

  for (unsigned int i = start_index; i < response.incidents_count; i++) {
    incident = &response.incidents[i];

    //same GPU Index, same component
    if ((incident->gpu_index != gpu_index) ||
        (incident->component != component))
      break;

    //set component health
    if (incident->health > component_health)
      component_health = incident->health;

    all_err_str = " - ";
    all_err_str += incident->error.msg;
    err_str.push_back(all_err_str);

    count++;
  }

  return count;
}

unsigned int RdciHealthSubSystem::handle_one_gpu(rdc_health_response_t &response,
                                                 unsigned int start_index,
                                                 uint32_t gpu_index) const {
  unsigned int count = 0, comp_count = 0;
  rdc_health_incidents_t *incident;
  rdc_health_result_t gpu_health = RDC_HEALTH_RESULT_PASS;
  std::string component_str, health_str, gpu_health_str;
  typedef struct {
    rdc_health_result_t component_health;
    std::vector<std::string> err_str;
  } component_detail_t;
  std::map<rdc_health_system_t, component_detail_t> component_detail_map;

  for (unsigned int i = start_index; i < response.incidents_count; i++) {
    incident = &response.incidents[i];

    //same GPU Index
    if (incident->gpu_index != gpu_index)
      break;

    //set gpu health
    if (incident->health > gpu_health)
      gpu_health = incident->health;

    //handle smae component
    component_detail_t detail;
    detail.component_health = RDC_HEALTH_RESULT_PASS;
    detail.err_str.clear();

    comp_count = handle_one_component(response, i, gpu_index, incident->component, detail.component_health, detail.err_str);
    i += comp_count - 1;
    count += comp_count;

    // Add to the component detail map
    component_detail_map.insert({incident->component, detail});
  }

  //output gpu_index health result
  gpu_health_str = health_string(gpu_health);

  if (is_json_output()) {
    std::cout << "{\"Index\" : \"" << std::to_string(gpu_index) << "\", ";
    std::cout <<  "\"Health\" : \"" << gpu_health_str << "\", ";
    std::cout <<  "\"Error\" : [";

    unsigned int i = 0; 
    for (auto ite : component_detail_map) {
      component_str = component_string(ite.first);
      health_str = health_string(ite.second.component_health);

      std::cout << "{\"Component\" : \"" << component_str << "\", ";
      std::cout << "\"Health\" : \"" << health_str << "\", ";

      std::cout << "\"Message\" : [";
      unsigned int j = 0; 
      for (auto err_ite : ite.second.err_str) {
        std::cout << "\"" << err_ite << "\"";
        j++;
        if (j < ite.second.err_str.size())
          std::cout << ", ";
      }
      std::cout << "]}"; //end Message

      i++;
      if (i < component_detail_map.size()) {
        std::cout << ", ";
      }
    }
    std::cout <<  "]}"; //end Error
  } else {
    std::cout << "|" << std::setw(20) << " GPU ID: " + std::to_string(gpu_index) << "| "
              << std::setw(60) << std::left << gpu_health_str << "|\n";
    std::cout << "|" << std::setw(20) << " " << "| "
              << std::setw(60) << " " << "|\n";

    for (auto ite : component_detail_map) {
      component_str = component_string(ite.first);
      health_str = health_string(ite.second.component_health);
      std::cout << "|" << std::setw(20) << " " << "| "
                << std::setw(60) << std::left << component_str + health_str << "|\n";

      for (auto msg : ite.second.err_str)
        output_errstr(msg);

      std::cout << "|" << std::setw(20) << " " << "| "
                << std::setw(60) << " " << "|\n";
    }
    std::cout << "+--------------------+-" //"-" width :20
              << "------------------------------------------------------------+\n"; //-" width :60
  }

  return count;
}

void RdciHealthSubSystem::health_check() const {
  unsigned int components = 0;
  rdc_status_t result;
  rdc_health_response_t response;

  result = rdc_health_get(rdc_handle_, group_id_, &components);
  if (result != RDC_ST_OK) {
    std::string error_msg = rdc_status_string(result);
    if (result == RDC_ST_NOT_FOUND) {
      error_msg = "Cannot find the group " + std::to_string(group_id_) + " or the field.";
    }
    throw RdcException(result, error_msg.c_str());
  }

  if (0 == components) {
    std::string error_msg = "Health watches not enable, please enable watches first.";
    throw RdcException(RDC_ST_UNKNOWN_ERROR, error_msg.c_str());
  }

  result = rdc_health_check(rdc_handle_, group_id_, &response);
  if (result != RDC_ST_OK) {
    throw RdcException(result, rdc_status_string(result));
  }

  //output headline
  std::string overall_str = health_string(response.overall_health);
  if (is_json_output()) {
    std::cout << "\"heading\" : \"Health monitor report\", ";
    std::cout << "\"body\" : ";
    std::cout <<  "{\"Group\" : \"" << std::to_string(group_id_) << "\", ";
    std::cout <<   "\"Overall Health\" : \"" << overall_str << "\", ";
    std::cout <<   "\"GPU\" : [";
  } else {
    std::cout << "Health monitor report:" << std::endl;
    std::cout << "+--------------------+-" //"-" width :20
              << "------------------------------------------------------------+\n"; //-" width :60
    std::cout << "|" << std::setw(20) << std::left << " Group " + std::to_string(group_id_) << "| "
              << std::setw(60) << std::left << "Overall Health: " + overall_str << "|\n";
    std::cout << "+====================+=" //"=" width :20
              << "============================================================+\n"; //"=" width :60
  }

  //output health of per GPU
  unsigned int index = 0;
  while (index < response.incidents_count) {
    uint32_t gpu_index = response.incidents[index].gpu_index;

    unsigned int count = handle_one_gpu(response, index, gpu_index);
    index += count;
    if (is_json_output() && (index < response.incidents_count))
      std::cout << ",";
  }

  if (is_json_output())
    std::cout <<  "]}"; //end Group
}

void RdciHealthSubSystem::health_clear() const {
  rdc_status_t result;

  result = rdc_health_clear(rdc_handle_, group_id_);
  if (result != RDC_ST_OK) {
    std::string error_msg = rdc_status_string(result);
    if (result == RDC_ST_NOT_FOUND) {
      error_msg = "Cannot find the group " + std::to_string(group_id_) + " or the field.";
    }
    throw RdcException(result, error_msg.c_str());
  }

  std::cout << "Clear Group " << group_id_ << " all health monitor systems." << std::endl;
}

void RdciHealthSubSystem::process() {
  switch (health_ops_) {
    case HEALTH_HELP:
    case HEALTH_UNKNOWN:
      show_help();
      break;

    case HEALTH_FETCH:
      get_watches();
      break;

    case HEALTH_SET:
      set_watches();
      break;

    case HEALTH_CHECK:
      health_check();
      break;

    case HEALTH_CLEAR:
      health_clear();
      break;

    default:
      show_help();
      break;
  }
}

}  // namespace rdc
}  // namespace amd
