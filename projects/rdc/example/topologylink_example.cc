/*
Copyright (C) Advanced Micro Devices. All rights reserved.

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

#include <iostream>

#include "rdc/rdc.h"

static const char* topology_link_type_to_str(rdc_topology_link_type_t type) {
  if (type == RDC_IOLINK_TYPE_PCIEXPRESS) return "RDC_IOLINK_TYPE_PCIEXPRESS";
  if (type == RDC_IOLINK_TYPE_XGMI) return "RDC_IOLINK_TYPE_XGMI";
  return "Unknown_Type";
}
int main() {
  rdc_gpu_group_t group_id;
  rdc_status_t result;
  bool standalone = false;
  rdc_handle_t rdc_handle;
  uint32_t count = 0;

  char hostIpAddress[] = {"localhost:50051"};
  char group_name[] = {"group1"};

  // Select the embedded mode and standalone mode dynamically.
  std::cout << "Start rdci in: \n";
  std::cout << "0 - Embedded mode \n";
  std::cout << "1 - Standalone mode \n";
  while (!(std::cin >> standalone)) {
    std::cout << "Invalid input.\n";
    std::cin.clear();
    std::cin.ignore();
  }
  std::cout << std::endl;
  std::cout << (standalone ? "Standalone mode selected.\n" : "Embedded mode selected.\n");

  // Init the rdc
  result = rdc_init(0);

  if (result != RDC_ST_OK) {
    std::cout << "Error initializing RDC. Return: " << rdc_status_string(result) << std::endl;
    goto cleanup;
  } else {
    std::cout << "RDC Initialized.\n";
  }

  if (standalone) {  // standalone
    result = rdc_connect(hostIpAddress, &rdc_handle, nullptr, nullptr, nullptr);
    if (result != RDC_ST_OK) {
      std::cout << "Error connecting to remote rdcd. Return: " << rdc_status_string(result)
                << std::endl;
      goto cleanup;
    }
  } else {  // embedded
    result = rdc_start_embedded(RDC_OPERATION_MODE_AUTO, &rdc_handle);
    if (result != RDC_ST_OK) {
      std::cout << "Error starting embedded RDC engine. Return: " << rdc_status_string(result)
                << std::endl;
      goto cleanup;
    }
  }

  // Now we can use the same API for both standalone and embedded
  // Get the list of devices in the system
  uint32_t gpu_index_list[RDC_MAX_NUM_DEVICES];
  result = rdc_device_get_all(rdc_handle, gpu_index_list, &count);
  if (result != RDC_ST_OK) {
    std::cout << "Error to find devices on the system. Return: " << rdc_status_string(result);
    goto cleanup;
  }
  if (count == 0) {
    std::cout << "No GPUs find on the sytem ";
    goto cleanup;
  } else {
    std::cout << count << " GPUs found in the system.\n";
  }

  // Create the group
  result = rdc_group_gpu_create(rdc_handle, RDC_GROUP_EMPTY, group_name, &group_id);
  if (result != RDC_ST_OK) {
    std::cout << "Error creating group. Return: " << rdc_status_string(result);
    goto cleanup;
  }
  std::cout << "Created the GPU group " << group_id << std::endl;

  // Add all GPUs to the group
  for (uint32_t i = 0; i < count; i++) {
    result = rdc_group_gpu_add(rdc_handle, group_id, gpu_index_list[i]);  // Add GPU 0
    if (result != RDC_ST_OK) {
      std::cout << "Error adding group. Return: " << rdc_status_string(result);
      goto cleanup;
    }
    rdc_device_attributes_t attribute;
    result = rdc_device_get_attributes(rdc_handle, gpu_index_list[i], &attribute);
    if (result != RDC_ST_OK) {
      std::cout << "Error get GPU attribute. Return: " << rdc_status_string(result);
      goto cleanup;
    }
    std::cout << "Add GPU " << gpu_index_list[i] << ":" << attribute.device_name << " to group "
              << group_id << std::endl;
  }
  rdc_group_info_t group_info;
  result = rdc_group_gpu_get_info(rdc_handle, group_id, &group_info);
  if (result != RDC_ST_OK) {
    std::cout << "Error get gpu group info. Return: " << rdc_status_string(result);
    goto cleanup;
  }

  rdc_device_topology_t topo;
  result = rdc_device_topology_get(rdc_handle, group_info.entity_ids[0], &topo);
  if (result != RDC_ST_OK) {
    std::cout << "Error clear topology, Return: " << rdc_status_string(result) << std::endl;
    goto cleanup;
  }

  for (uint32_t i = 0; i < topo.num_of_gpus; i++) {
    std::cout << "Topology Information Return \n "
              << "gpu_index: " << std::to_string(topo.link_infos[i].gpu_index) << "\n"
              << "weight: " << std::to_string(topo.link_infos[i].weight) << "\n"
              << "min_bandwidth: " << std::to_string(topo.link_infos[i].min_bandwidth) << "\n"
              << "max_bandwidth: " << std::to_string(topo.link_infos[i].max_bandwidth) << "\n"
              << "hops: " << std::to_string(topo.link_infos[i].hops) << "\n"
              << "link_type: " << topology_link_type_to_str(topo.link_infos[i].link_type) << "\n"
              << "is_p2p_accessible: " << std::to_string(topo.link_infos[i].is_p2p_accessible)
              << "\n"
              << std::endl;
  }

  rdc_link_status_t link_status;
  result = rdc_link_status_get(rdc_handle, &link_status);
  if (result != RDC_ST_OK) {
    std::cout << "Error clear topology, Return: " << rdc_status_string(result) << std::endl;
    goto cleanup;
  }

  //... clean up
cleanup:
  std::cout << "Cleaning up.\n";
  if (standalone)
    rdc_disconnect(rdc_handle);
  else
    rdc_stop_embedded(rdc_handle);
  rdc_shutdown();
  return result;
}
