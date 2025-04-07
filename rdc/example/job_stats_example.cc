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

#include <unistd.h>

#include <iostream>

#include "rdc/rdc.h"

int main(int, char**) {
  rdc_status_t result;
  rdc_handle_t rdc_handle;
  bool standalone = false;
  char hostIpAddress[] = {"127.0.0.1:50051"};
  char group_name[] = {"group1"};
  char job_id[] = {"123"};

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
    result = rdc_start_embedded(RDC_OPERATION_MODE_MANUAL, &rdc_handle);
    if (result != RDC_ST_OK) {
      std::cout << "Error starting embedded RDC engine. Return: " << rdc_status_string(result)
                << std::endl;
      goto cleanup;
    }
  }

  // Now we can use the same API for both standalone and embedded
  // (1) create group and add GPUs
  rdc_gpu_group_t group_id;
  result = rdc_group_gpu_create(rdc_handle, RDC_GROUP_EMPTY, group_name, &group_id);
  if (result != RDC_ST_OK) {
    std::cout << "Error creating group. Return: " << rdc_status_string(result);
    goto cleanup;
  }

  result = rdc_group_gpu_add(rdc_handle, group_id, 0);  // Add GPU 0
  if (result != RDC_ST_OK) {
    std::cout << "Error adding group. Return: " << rdc_status_string(result);
    goto cleanup;
  }

  // (2) start the recording. Set the sample frequency to once per second.
  result = rdc_job_start_stats(rdc_handle, group_id, job_id, 1000000);
  if (result != RDC_ST_OK) {
    std::cout << "Error start job stats. Return: " << rdc_status_string(result);
    goto cleanup;
  }

  // For standalone mode, the daemon will update and cache the samples
  // In manual mode, we must call rdc_field_update_all periodically to
  // take samples.
  if (!standalone) {               // embedded manual mode
    for (int i = 5; i > 0; i--) {  // As an example, we will take 5 samples
      result = rdc_field_update_all(rdc_handle, 0);
      if (result != RDC_ST_OK) {
        std::cout << "Error update all fields. Return: " << rdc_status_string(result);
        goto cleanup;
      }
      usleep(1000000);
    }
  } else {            // standalone mode, do nothing
    usleep(5000000);  // sleep 5 seconds before fetch the stats
  }

  // (3) stop the Slurm job, which will stop the watch
  // We do not have to stop the job to get stats. The rdc_job_get_stats can be
  // called at any time before stop
  result = rdc_job_stop_stats(rdc_handle, job_id);
  if (result != RDC_ST_OK) {
    std::cout << "Error stop job stats. Return: " << rdc_status_string(result);
    goto cleanup;
  }

  // (4) Get the stats
  rdc_job_info_t job_info;
  result = rdc_job_get_stats(rdc_handle, job_id, &job_info);

  if (result == RDC_ST_OK) {
    std::cout << "|------- Execution Stats ----------+"
              << "------------------------------------\n";
    std::cout << "| Start Time *                     | " << job_info.summary.start_time << "\n";
    std::cout << "| End Time *                       | " << job_info.summary.end_time << "\n";
    std::cout << "| Total Execution Time (sec) *     | "
              << (job_info.summary.end_time - job_info.summary.start_time) << "\n";
    std::cout << "+------- Performance Stats --------+"
              << "------------------------------------\n";
    std::cout << "| Energy Consumed (Joules)         | " << job_info.summary.energy_consumed
              << "\n";
    std::cout << "| Power Usage (Watts)              | "
              << "Max: " << job_info.summary.power_usage.max_value
              << " Min: " << job_info.summary.power_usage.min_value
              << " Avg: " << job_info.summary.power_usage.average << "\n";
    std::cout << "| GPU Clock (MHz)                  | "
              << "Max: " << job_info.summary.gpu_clock.max_value
              << " Min: " << job_info.summary.gpu_clock.min_value
              << " Avg: " << job_info.summary.gpu_clock.average << "\n";
    std::cout << "| GPU Utilization (%)              | "
              << "Max: " << job_info.summary.gpu_utilization.max_value
              << " Min: " << job_info.summary.gpu_utilization.min_value
              << " Avg: " << job_info.summary.gpu_utilization.average << "\n";
    std::cout << "| Max GPU Memory Used (bytes) *    | " << job_info.summary.max_gpu_memory_used
              << "\n";
    std::cout << "| Memory Utilization (%)           | "
              << "Max: " << job_info.summary.memory_utilization.max_value
              << " Min: " << job_info.summary.memory_utilization.min_value
              << " Avg: " << job_info.summary.memory_utilization.average << "\n";
    std::cout << "+----------------------------------+"
              << "------------------------------------\n";
  } else {
    std::cout << "No data for job stats found." << std::endl;
  }

// Cleanup consists of shutting down RDC.
cleanup:
  std::cout << "Cleaning up.\n";
  if (standalone)
    rdc_disconnect(rdc_handle);
  else
    rdc_stop_embedded(rdc_handle);
  rdc_shutdown();
  return result;
}
