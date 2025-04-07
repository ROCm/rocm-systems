/* Copyright (c) 2022 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include <gtest/gtest.h>

#include <vector>
#include <mutex>
#include <memory>
#include <stdlib.h>
#include "api/rocprofiler_singleton.h"
#include "src/core/hsa/hsa_support.h"
#define MAX_THREADS 10000
struct devices_t {
  std::vector<hsa_agent_t> cpu_devices;
  std::vector<hsa_agent_t> gpu_devices;
  std::vector<hsa_agent_t> other_devices;
};

hsa_status_t device_cb_tool(hsa_agent_t agent, void* data) {
  hsa_device_type_t device_type;
  devices_t* devices = reinterpret_cast<devices_t*>(data);
  if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type) != HSA_STATUS_SUCCESS) {
    std::cout << "hsa_iterate_agents failed" << std::endl;
    std::exit(-1);
  }
  switch (device_type) {
    case HSA_DEVICE_TYPE_CPU:
      devices->cpu_devices.push_back(agent);
      break;
    case HSA_DEVICE_TYPE_GPU:
      devices->gpu_devices.push_back(agent);
      break;
    default:
      devices->other_devices.push_back(agent);
      break;
  }
  return HSA_STATUS_SUCCESS;
}
void get_hsa_agents_list_tool(devices_t* device_list) {
  // Enumerate the agents.
  if (hsa_iterate_agents(device_cb_tool, device_list) != HSA_STATUS_SUCCESS) {
    std::cout << "hsa_iterate_agents failed" << std::endl;
    std::exit(-1);
  }
}


class TestHSASupportSingleton {
 public:
  uint64_t ref_address;
  TestHSASupportSingleton() {
    rocprofiler::HSASupport_Singleton& hsasupport_singleton =
        rocprofiler::HSASupport_Singleton::GetInstance();
    ref_address = reinterpret_cast<long int>(&hsasupport_singleton);
  }
};

void instanciateHSASupportSingleton(int index, uint64_t *ref_array) {
  TestHSASupportSingleton test;
  *(ref_array+index) = test.ref_address;
}

//Add more threads here
TEST(WhenInvokingHSASingleton, HSASupportSingletonInstanciation) {
  std::vector<std::thread> threads;
  uint64_t *refaddress = (uint64_t*)malloc(sizeof(uint64_t)*(MAX_THREADS));
  for(int i = 0; i < MAX_THREADS; i++) {
    threads.emplace_back(instanciateHSASupportSingleton, i, &refaddress[0]);
  }
  for (auto&& thread : threads) thread.join();
  uint64_t ref_addr = refaddress[0];
  for(int i = 1; i < MAX_THREADS; i++)
   EXPECT_EQ(ref_addr, refaddress[i]) << "HSASingleton Instanciation failed";
}


TEST(WhenInvokingGetHSAInitialize, TestHSASupportSingleton) {
 rocprofiler::HSASupport_Singleton& hsasupport_singleton =
        rocprofiler::HSASupport_Singleton::GetInstance();
 devices_t device_list;
 get_hsa_agents_list_tool(&device_list);
 for(auto it = device_list.gpu_devices.begin(); it !=  device_list.gpu_devices.end(); it++) {
    [[maybe_unused]]rocprofiler::HSAAgentInfo& agent_info = hsasupport_singleton.GetHSAAgentInfo(it->handle);
 }
 EXPECT_EQ(hsasupport_singleton.gpu_agents.size(), device_list.gpu_devices.size()) << "HSAInitialize failed";
}

TEST(WhenInvokingGetHSAAgentInfo, TestHSASupportSingleton) {
 rocprofiler::HSASupport_Singleton& hsasupport_singleton =
        rocprofiler::HSASupport_Singleton::GetInstance();
 devices_t device_list;
 get_hsa_agents_list_tool(&device_list);
 for(auto it = device_list.gpu_devices.begin(); it !=  device_list.gpu_devices.end(); it++) {
    rocprofiler::HSAAgentInfo& agent_info = hsasupport_singleton.GetHSAAgentInfo(it->handle);
    uint32_t gpu_id;
    char name[64];
    hsasupport_singleton.GetCoreApiTable().hsa_agent_get_info_fn(
                    *it, (hsa_agent_info_t)(HSA_AMD_AGENT_INFO_DRIVER_UID), &gpu_id);
    hsasupport_singleton.GetCoreApiTable().hsa_agent_get_info_fn(*it, HSA_AGENT_INFO_NAME, name);

    EXPECT_EQ(agent_info.GetDeviceInfo().getGPUId(), gpu_id) << "HSAAgentInfo has incorrect gpu id for the agent: " << it->handle;
    EXPECT_EQ(strcmp(agent_info.GetDeviceInfo().getName().data(), name), 0) << "HSAAgentInfo has incorrect gpu name for the agent: " << it->handle;
  }
}

TEST(WhenInvokingQueueInterceptors, TestQueueInterceptors) {

 rocprofiler::HSASupport_Singleton& hsasupport_singleton =
          rocprofiler::HSASupport_Singleton::GetInstance();
  hsa_queue_t* queue1 = nullptr, *queue2 = nullptr;

  hsa_status_t status =  hsa_queue_create(hsasupport_singleton.gpu_agents[0], 1024, HSA_QUEUE_TYPE_SINGLE, NULL, NULL, UINT32_MAX,
                            UINT32_MAX, &queue1);


  EXPECT_EQ(status, HSA_STATUS_SUCCESS) << "Queue create interceptor failed";
  status =  hsa_queue_create(hsasupport_singleton.gpu_agents[0], 1024, HSA_QUEUE_TYPE_SINGLE, NULL, NULL, UINT32_MAX,
                            UINT32_MAX, &queue2);
  status =  hsa_queue_destroy(queue1);
  EXPECT_EQ(status, HSA_STATUS_SUCCESS) <<  "Queue destroy interceptor failed";
}

