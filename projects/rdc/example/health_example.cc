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

#include <unistd.h>

#include <iomanip>
#include <iostream>
#include <vector>
#include <map>

#include "rdc/rdc.h"

rdc_status_t get_watches(rdc_handle_t rdc_handle, rdc_gpu_group_t group_id) {
  unsigned int components;
  rdc_status_t result = rdc_health_get(rdc_handle, group_id, &components);
  if (result == RDC_ST_OK) {
    std::string on = "On";
    std::string off = "Off";

    std::cout << "Health monitor systems status:" << std::endl;
    std::cout << "+--------------------+" //"-" width :20
              << "---------------------------------------------------+\n"; //-" width :51
    std::cout << "|" << std::setw(20) << std::left << " PCIe"    << "| "
              << std::setw(50) << std::left << ((components & RDC_HEALTH_WATCH_PCIE) ? on : off).c_str() << "|\n";
    std::cout << "|" << std::setw(20) << std::left << " XGMI"    << "| "
              << std::setw(50) << std::left << ((components & RDC_HEALTH_WATCH_XGMI) ? on : off).c_str() << "|\n";
    std::cout << "|" << std::setw(20) << std::left << " Memory"  << "| "
              << std::setw(50) << std::left << ((components & RDC_HEALTH_WATCH_MEM) ? on : off).c_str() << "|\n";
    std::cout << "|" << std::setw(20) << std::left << " EEPROM"  << "| "
              << std::setw(50) << std::left << ((components & RDC_HEALTH_WATCH_EEPROM) ? on : off).c_str() << "|\n";
    std::cout << "|" << std::setw(20) << std::left << " Thermal" << "| "
              << std::setw(50) << std::left << ((components & RDC_HEALTH_WATCH_THERMAL) ? on : off).c_str() << "|\n";
    std::cout << "|" << std::setw(20) << std::left << " Power"   << "| "
              << std::setw(50) << std::left << ((components & RDC_HEALTH_WATCH_POWER) ? on : off).c_str() << "|\n";
    std::cout << "+--------------------+" //"-" width :20
              << "---------------------------------------------------+\n"; //-" width :51
  }

  return result;
}

std::string health_string(rdc_health_result_t health) {
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

std::string component_string(rdc_health_system_t component) {
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

void output_errstr(const std::string& input) {
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

unsigned int handle_one_component(rdc_health_response_t &response,
                                  unsigned int start_index,
                                  uint32_t gpu_index,
                                  rdc_health_system_t component,
                                  rdc_health_result_t &component_health,
                                  std::vector<std::string> &err_str) {
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

unsigned int handle_one_gpu(rdc_health_response_t &response,
                            unsigned int start_index,
                            uint32_t gpu_index) {
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

  return count;
}

int main(int, char**) {
  rdc_status_t result;
  rdc_handle_t rdc_handle;
  char hostIpAddress[] = {"127.0.0.1:50051"};
  char group_name[] = {"healthgroup1"};

  std::cout << "Start rdci in Standalone mode\n";

  // Init the rdc
  result = rdc_init(0);

  if (result != RDC_ST_OK) {
    std::cout << "Error initializing RDC. Return: " << rdc_status_string(result) << std::endl;
    goto cleanup;
  } else {
    std::cout << "RDC Initialized.\n";
  }

  result = rdc_connect(hostIpAddress, &rdc_handle, nullptr, nullptr, nullptr);
  if (result != RDC_ST_OK) {
    std::cout << "Error connecting to remote rdcd. Return: " << rdc_status_string(result)
              << std::endl;
    goto cleanup;
  }

  // Now we can use the same API for standalone
  // (1) create group and add GPUs
  rdc_gpu_group_t group_id;
  result = rdc_group_gpu_create(rdc_handle, RDC_GROUP_EMPTY, group_name, &group_id);
  if (result != RDC_ST_OK) {
    std::cout << "Error creating group. Return: " << rdc_status_string(result)
              << std::endl;
    goto cleanup;
  }
  std::cout << "Created the GPU group " << group_id << std::endl;

  result = rdc_group_gpu_add(rdc_handle, group_id, 0);  // Add GPU 0
  if (result != RDC_ST_OK) {
    std::cout << "Error adding group. Return: " << rdc_status_string(result)
              << std::endl;
    goto destroygroup;
  }

  rdc_device_attributes_t attribute;
  result = rdc_device_get_attributes(rdc_handle, 0, &attribute);
  if (result != RDC_ST_OK) {
    std::cout << "Error get GPU attribute. Return: " << rdc_status_string(result);
    goto destroygroup;
  }
  std::cout << "Add GPU 0: " << attribute.device_name << " to group "
            << group_id << std::endl;

  // (2) get heath current watches before setting
  result = get_watches(rdc_handle, group_id);
  if (result != RDC_ST_OK) {
    std::cout << "Error getting health watches. Return: " << rdc_status_string(result)
              << std::endl;
    goto destroygroup;
  }

  // (3) set health watches.
  unsigned int components;
  components = RDC_HEALTH_WATCH_PCIE | RDC_HEALTH_WATCH_XGMI | RDC_HEALTH_WATCH_MEM
                          | RDC_HEALTH_WATCH_EEPROM | RDC_HEALTH_WATCH_THERMAL
                          | RDC_HEALTH_WATCH_POWER;
  result = rdc_health_set(rdc_handle, group_id, components);
  if (result != RDC_ST_OK) {
    std::cout << "Error setting health watches. Return: " << rdc_status_string(result)
              << std::endl;
    goto destroygroup;
  }
  std::cout << "Set health watches to all." << std::endl;

  // (4) get heath current watches after setting
  result = get_watches(rdc_handle, group_id);
  if (result != RDC_ST_OK) {
    std::cout << "Error getting health watches. Return: " << rdc_status_string(result)
              << std::endl;
    goto destroygroup;
  }

  std::cout << "Start to health monitor group:" << group_id
            << std::endl;
  std::cout << "Sleep a few seconds before retreive the data ...\n";
  // For standalone mode, the daemon will update and cache the samples
  // take samples, standalone mode, do nothing
    usleep(5000000);  // sleep 5 seconds before fetch the stats

  // (5) Get the health stats
  rdc_health_response_t response;
  result = rdc_health_check(rdc_handle, group_id, &response);
  if (result != RDC_ST_OK) {
    std::cout << "Error health check. Return: " << rdc_status_string(result)
              << std::endl;
    goto destroygroup;
  } else {
    //output headline
    std::string overall_str = health_string(response.overall_health);
    std::cout << "Health monitor report:" << std::endl;
    std::cout << "+--------------------+-" //"-" width :20
              << "------------------------------------------------------------+\n"; //-" width :60
    std::cout << "|" << std::setw(20) << std::left << " Group " + std::to_string(group_id) << "| "
              << std::setw(60) << std::left << "Overall Health: " + overall_str << "|\n";
    std::cout << "+====================+=" //"=" width :20
              << "============================================================+\n"; //"=" width :60

    //output health of per GPU
    unsigned int index = 0;
    while (index < response.incidents_count) {
      uint32_t gpu_index = response.incidents[index].gpu_index;

      unsigned int count = handle_one_gpu(response, index, gpu_index);
      index += count;
    }
  }

  // (6) Clear the health
  result = rdc_health_clear(rdc_handle, group_id);
  if (result != RDC_ST_OK) {
    std::cout << "Error clear health. Return: " << rdc_status_string(result)
              << std::endl;
    goto destroygroup;
  }
  std::cout << "Clear Group " << group_id << " all health monitor systems." << std::endl;

destroygroup:
  // Delete the GPU group
  result = rdc_group_gpu_destroy(rdc_handle, group_id);
  if (result != RDC_ST_OK) {
    std::cout << "Error delete GPU group. Return: " << rdc_status_string(result);
    goto cleanup;
  }
  std::cout << "Deleted the GPU group " << group_id << std::endl;

  // Cleanup consists of shutting down RDC.
cleanup:
  std::cout << "Cleaning up.\n";
  rdc_disconnect(rdc_handle);
  rdc_shutdown();
  return result;
}
