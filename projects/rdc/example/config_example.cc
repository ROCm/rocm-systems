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

#include <chrono>
#include <iostream>
#include <thread>

#include "rdc/rdc.h"

int main() {
  rdc_gpu_group_t group_id;
  rdc_status_t result;
  bool standalone = false;
  rdc_handle_t rdc_handle;
  uint32_t count = 0;
  rdc_config_setting_list_t settings_list;
  rdc_config_setting_t setting;
  uint64_t watts;

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

  setting.type = RDC_CFG_POWER_LIMIT;
  // Our targeted value is 195 Watts, which will be converted into Microwatts inside of
  // rdc_config_set
  setting.target_value = 195;
  result = rdc_config_set(rdc_handle, group_id, setting);
  if (result != RDC_ST_OK) {
    std::cout << "Error set config RDC_CFG_POWER_LIMIT, Return: " << rdc_status_string(result)
              << std::endl;
    goto cleanup;
  }

  result = rdc_config_get(rdc_handle, group_id, &settings_list);
  if (result != RDC_ST_OK) {
    std::cout << "Error get config, Return: " << rdc_status_string(result) << std::endl;
    goto cleanup;
  }

  // Prompt user to change amd-smi to other value, and watch rdc config change it back

  std::cout << "Config before wait:" << std::endl;

  result = rdc_config_get(rdc_handle, group_id, &settings_list);
  if (result != RDC_ST_OK) {
    std::cout << "Error get config, Return: " << rdc_status_string(result) << std::endl;
    goto cleanup;
  }

  std::cout << "The config will keep the power limit to 195 Watts" << std::endl;
  std::cout << "You can change the power limit using amd-smi, the RDC config module should be able "
               "to detect it and set it back"
            << std::endl;
  std::cout << "Waiting 3 minutes before exit ..." << std::endl;
  std::this_thread::sleep_for(std::chrono::minutes(3));

  result = rdc_config_clear(rdc_handle, group_id);
  if (result != RDC_ST_OK) {
    std::cout << "Error clear config, Return: " << rdc_status_string(result) << std::endl;
    goto cleanup;
  }

//... clean up
cleanup:
  std::cout << "Cleaning up.\n";

  result = rdc_group_gpu_destroy(rdc_handle, group_id);
  if (result != RDC_ST_OK) {
    std::cout << "Error delete GPU group. Return: " << rdc_status_string(result);
  }
  std::cout << "Deleted the GPU group " << group_id << std::endl;

  if (standalone)
    rdc_disconnect(rdc_handle);
  else
    rdc_stop_embedded(rdc_handle);
  rdc_shutdown();
  return result;
}
