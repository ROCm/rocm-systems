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

static const char* condition_type_to_str(rdc_policy_condition_type_t type) {
  if (type == RDC_POLICY_COND_MAX_PAGE_RETRIED) return "Retried Page Limit";
  if (type == RDC_POLICY_COND_THERMAL) return "Temperature Limit";
  if (type == RDC_POLICY_COND_POWER) return "Power Limit";
  return "Unknown_Type";
}

static time_t last_time = 0;  // last time to print message
int rdc_policy_callback(rdc_policy_callback_response_t* userData) {
  if (userData == nullptr) {
    std::cerr << "The rdc_policy_callback returns null data\n";
    return 1;
  }

  // To avoid flooding too many messages, only print message every 5 seconds
  time_t now = time(NULL);
  if (difftime(now, last_time) < 5) {
    return 0;
  }
  std::cout << "The " << condition_type_to_str(userData->condition.type)
            << " exceeds the threshold " << userData->condition.value << " with the value "
            << userData->value << std::endl;
  last_time = now;  // update the last time
  return 0;
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

  // Define a policy to print out message when temperature is above 30 degree
  // or power usage is more than 150W
  rdc_policy_t policy;
  policy.condition = {RDC_POLICY_COND_THERMAL, 30 * 1000};  // convert to milli degree
  policy.action = RDC_POLICY_ACTION_NONE;                   // Notify only
  result = rdc_policy_set(rdc_handle, group_id, policy);
  if (result != RDC_ST_OK) {
    std::cout << "Error set policy RDC_POLICY_COND_THERMAL, Return: " << rdc_status_string(result)
              << std::endl;
    goto cleanup;
  }

  policy.condition = {RDC_POLICY_COND_POWER, 150000};  // convert to milli degree
  policy.action = RDC_POLICY_ACTION_NONE;               // Notify only
  result = rdc_policy_set(rdc_handle, group_id, policy);
  if (result != RDC_ST_OK) {
    std::cout << "Error set policy RDC_POLICY_COND_POWER, Return: " << rdc_status_string(result)
              << std::endl;
    goto cleanup;
  }

  policy.condition = {RDC_POLICY_COND_MAX_PAGE_RETRIED, 100};  // convert to milli degree
  policy.action = RDC_POLICY_ACTION_NONE;               // Notify only
  result = rdc_policy_set(rdc_handle, group_id, policy);
  if (result != RDC_ST_OK) {
    std::cout << "Error set policy RDC_POLICY_COND_MAX_PAGE_RETRIED, Return: " << rdc_status_string(result)
              << std::endl;
    goto cleanup;
  }

  rdc_policy_t policy_get[RDC_MAX_POLICY_SETTINGS];
  result = rdc_policy_get(rdc_handle, group_id, &count, policy_get);
  if (result != RDC_ST_OK) {
    std::cout << "Error get policy, Return: " << rdc_status_string(result) << std::endl;
    goto cleanup;
  }

  // Register a function to listen to the events
  result = rdc_policy_register(rdc_handle, group_id, rdc_policy_callback);
  if (result != RDC_ST_OK) {
    std::cout << "Error register policy, Return: " << rdc_status_string(result) << std::endl;
    goto cleanup;
  }

  std::cout << "Wait 30 seconds for the events happening ...\n" << std::endl;

  // If the events happening, the callback rdc_policy_register_callback will be called.
  usleep(30 * 1000000);  // sleep 30 seconds

  // Un-register the events
  result = rdc_policy_unregister(rdc_handle, group_id);
  if (result != RDC_ST_OK) {
    std::cout << "Error unregister policy, Return: " << rdc_status_string(result) << std::endl;
    goto cleanup;
  }

  // clear the events
  rdc_policy_condition_type_t condition_type;
  condition_type = RDC_POLICY_COND_THERMAL;
  result = rdc_policy_delete(rdc_handle, group_id, condition_type);
  if (result != RDC_ST_OK) {
    std::cout << "Error clear policy, Return: " << rdc_status_string(result) << std::endl;
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