/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

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

/** \mainpage ROC Profiler Multi Queue Dependency Test
 *
 * \section introduction Introduction
 *
 * The goal of this test is to ensure ROC profiler does not go to deadlock
 * when multiple queue are created and they are dependent on each other
 *
 */

#include "multiqueue_testapp.h"
#include "src/utils/filesystem.hpp"
#include "src/utils/exception.h"

namespace fs = rocprofiler::common::filesystem;
std::vector<hsa_agent_t> Device::all_devices;

std::string GetRunningPath(std::string string_to_erase);
static void init_test_path();

std::string test_app_path;
std::string hasco_path;

int main() {
  hsa_status_t status;
  MQDependencyTest obj;

  // Get Agent info
  obj.DeviceDiscovery();

  char agent_name[64];
  status = hsa_agent_get_info(gpu[0].agent, HSA_AGENT_INFO_NAME, agent_name);
  ASSERT_EQ(status, HSA_STATUS_SUCCESS);

  // set global test path for this test
  init_test_path();
  // Getting Current Path
  std::string app_path = GetRunningPath(test_app_path + "multiqueue_testapp");
  // Getting hasco Path
  std::string ko_path = app_path + hasco_path + std::string(agent_name) + "_copy.hsaco";

  MQDependencyTest::CodeObject code_object;
  if (!obj.LoadCodeObject(ko_path, gpu[0].agent, code_object)) {
    printf("Kernel file not found or not usable with given agent.\n");
    abort();
  }

  MQDependencyTest::Kernel copyA;
  if (!obj.GetKernel(code_object, "copyA", gpu[0].agent, copyA)) {
    printf("Test kernel A not found.\n");
    abort();
  }
  MQDependencyTest::Kernel copyB;
  if (!obj.GetKernel(code_object, "copyB", gpu[0].agent, copyB)) {
    printf("Test kernel B not found.\n");
    abort();
  }
  MQDependencyTest::Kernel copyC;
  if (!obj.GetKernel(code_object, "copyC", gpu[0].agent, copyC)) {
    printf("Test kernel C not found.\n");
    abort();
  }

  struct args_t {
    uint32_t* a;
    uint32_t* b;
    MQDependencyTest::OCLHiddenArgs hidden;
  };

  args_t* args;
  args = static_cast<args_t*>(obj.hsaMalloc(sizeof(args_t), kernarg));
  memset(args, 0, sizeof(args_t));

  uint32_t* a = static_cast<uint32_t*>(obj.hsaMalloc(64 * sizeof(uint32_t), kernarg));
  uint32_t* b = static_cast<uint32_t*>(obj.hsaMalloc(64 * sizeof(uint32_t), kernarg));

  memset(a, 0, 64 * sizeof(uint32_t));
  memset(b, 1, 64 * sizeof(uint32_t));

  // Create queue in gpu agent and prepare a kernel dispatch packet
  hsa_queue_t* queue1;
  status = hsa_queue_create(gpu[0].agent, 1024, HSA_QUEUE_TYPE_SINGLE, NULL, NULL, UINT32_MAX,
                            UINT32_MAX, &queue1);
  ASSERT_EQ(status, HSA_STATUS_SUCCESS);

  // Create a signal with a value of 1 and attach it to the first kernel
  // dispatch packet
  hsa_signal_t completion_signal_1;
  status = hsa_signal_create(1, 0, NULL, &completion_signal_1);
  ASSERT_EQ(status, HSA_STATUS_SUCCESS);

  // First dispath packet on queue 1, Kernel A
  {
    MQDependencyTest::Aql packet{};
    packet.header.type = HSA_PACKET_TYPE_KERNEL_DISPATCH;
    packet.header.barrier = 1;
    packet.header.acquire = HSA_FENCE_SCOPE_SYSTEM;
    packet.header.release = HSA_FENCE_SCOPE_SYSTEM;

    packet.dispatch.setup = 1;
    packet.dispatch.workgroup_size_x = 64;
    packet.dispatch.workgroup_size_y = 1;
    packet.dispatch.workgroup_size_z = 1;
    packet.dispatch.grid_size_x = 64;
    packet.dispatch.grid_size_y = 1;
    packet.dispatch.grid_size_z = 1;

    packet.dispatch.group_segment_size = copyA.group;
    packet.dispatch.private_segment_size = copyA.scratch;
    packet.dispatch.kernel_object = copyA.handle;

    packet.dispatch.kernarg_address = args;
    packet.dispatch.completion_signal = completion_signal_1;

    args->a = a;
    args->b = b;
    // Tell packet processor of A to launch the first kernel dispatch packet
    obj.SubmitPacket(queue1, packet);
  }

  // Create a signal with a value of 1 and attach it to the second kernel
  // dispatch packet
  hsa_signal_t completion_signal_2;
  status = hsa_signal_create(1, 0, NULL, &completion_signal_2);
  ASSERT_EQ(status, HSA_STATUS_SUCCESS);

  hsa_signal_t completion_signal_3;
  status = hsa_signal_create(1, 0, NULL, &completion_signal_3);
  ASSERT_EQ(status, HSA_STATUS_SUCCESS);

  // Create barrier-AND packet that is enqueued in queue 1
  {
    MQDependencyTest::Aql packet{};
    packet.header.type = HSA_PACKET_TYPE_BARRIER_AND;
    packet.header.barrier = 1;
    packet.header.acquire = HSA_FENCE_SCOPE_SYSTEM;
    packet.header.release = HSA_FENCE_SCOPE_SYSTEM;

    packet.barrier_and.dep_signal[0] = completion_signal_2;
    obj.SubmitPacket(queue1, packet);
  }

  // Second dispath packet on queue 1, Kernel C
  {
    MQDependencyTest::Aql packet{};
    packet.header.type = HSA_PACKET_TYPE_KERNEL_DISPATCH;
    packet.header.barrier = 1;
    packet.header.acquire = HSA_FENCE_SCOPE_SYSTEM;
    packet.header.release = HSA_FENCE_SCOPE_SYSTEM;

    packet.dispatch.setup = 1;
    packet.dispatch.workgroup_size_x = 64;
    packet.dispatch.workgroup_size_y = 1;
    packet.dispatch.workgroup_size_z = 1;
    packet.dispatch.grid_size_x = 64;
    packet.dispatch.grid_size_y = 1;
    packet.dispatch.grid_size_z = 1;

    packet.dispatch.group_segment_size = copyC.group;
    packet.dispatch.private_segment_size = copyC.scratch;
    packet.dispatch.kernel_object = copyC.handle;
    packet.dispatch.completion_signal = completion_signal_3;
    packet.dispatch.kernarg_address = args;

    args->a = a;
    args->b = b;
    // Tell packet processor to launch the second kernel dispatch packet
    obj.SubmitPacket(queue1, packet);
  }

  // Create queue 2
  hsa_queue_t* queue2;
  status = hsa_queue_create(gpu[0].agent, 1024, HSA_QUEUE_TYPE_SINGLE, NULL, NULL, UINT32_MAX,
                            UINT32_MAX, &queue2);
  ASSERT_EQ(status, HSA_STATUS_SUCCESS);

  // Create barrier-AND packet that is enqueued in queue 2
  {
    MQDependencyTest::Aql packet{};
    packet.header.type = HSA_PACKET_TYPE_BARRIER_AND;
    packet.header.barrier = 1;
    packet.header.acquire = HSA_FENCE_SCOPE_SYSTEM;
    packet.header.release = HSA_FENCE_SCOPE_SYSTEM;

    packet.barrier_and.dep_signal[0] = completion_signal_1;
    obj.SubmitPacket(queue2, packet);
  }

  // Third dispath packet on queue 2, Kernel B
  {
    MQDependencyTest::Aql packet{};
    packet.header.type = HSA_PACKET_TYPE_KERNEL_DISPATCH;
    packet.header.barrier = 1;
    packet.header.acquire = HSA_FENCE_SCOPE_SYSTEM;
    packet.header.release = HSA_FENCE_SCOPE_SYSTEM;

    packet.dispatch.setup = 1;
    packet.dispatch.workgroup_size_x = 64;
    packet.dispatch.workgroup_size_y = 1;
    packet.dispatch.workgroup_size_z = 1;
    packet.dispatch.grid_size_x = 64;
    packet.dispatch.grid_size_y = 1;
    packet.dispatch.grid_size_z = 1;

    packet.dispatch.group_segment_size = copyB.group;
    packet.dispatch.private_segment_size = copyB.scratch;
    packet.dispatch.kernel_object = copyB.handle;

    packet.dispatch.kernarg_address = args;
    packet.dispatch.completion_signal = completion_signal_2;

    args->a = a;
    args->b = b;
    // Tell packet processor to launch the third kernel dispatch packet
    obj.SubmitPacket(queue2, packet);
  }

  // Wait on the completion signal
  hsa_signal_wait_relaxed(completion_signal_1, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                          HSA_WAIT_STATE_BLOCKED);

  // Wait on the completion signal
  hsa_signal_wait_relaxed(completion_signal_2, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                          HSA_WAIT_STATE_BLOCKED);

  // Wait on the completion signal
  hsa_signal_wait_relaxed(completion_signal_3, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                          HSA_WAIT_STATE_BLOCKED);

  for (int i = 0; i < 64; i++) {
    if (a[i] != b[i]) {
      printf("error at %d: expected %d, got %d\n", i, b[i], a[i]);
      abort();
    }
  }

  // Clearing data structures and memory
  status = hsa_signal_destroy(completion_signal_1);
  ASSERT_EQ(status, HSA_STATUS_SUCCESS);

  status = hsa_signal_destroy(completion_signal_2);
  ASSERT_EQ(status, HSA_STATUS_SUCCESS);

  status = hsa_signal_destroy(completion_signal_3);
  ASSERT_EQ(status, HSA_STATUS_SUCCESS);

  if (queue1 != nullptr) {
    status = hsa_queue_destroy(queue1);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);
  }

  if (queue2 != nullptr) {
    status = hsa_queue_destroy(queue2);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);
  }

  status = hsa_memory_free(a);
  ASSERT_EQ(status, HSA_STATUS_SUCCESS);
  status = hsa_memory_free(b);
  ASSERT_EQ(status, HSA_STATUS_SUCCESS);

  status = hsa_executable_destroy(code_object.executable);
  ASSERT_EQ(status, HSA_STATUS_SUCCESS);

  status = hsa_code_object_reader_destroy(code_object.code_obj_rdr);
  ASSERT_EQ(status, HSA_STATUS_SUCCESS);
  close(code_object.file);
}

// This function returns the running path of executable
std::string GetRunningPath(std::string string_to_erase) {
  std::string path;
  char* real_path;
  Dl_info dl_info;

  if (0 != dladdr(reinterpret_cast<void*>(main), &dl_info)) {
    std::string to_erase = string_to_erase;
    path = dl_info.dli_fname;
    real_path = realpath(path.c_str(), NULL);
    if (real_path == nullptr) {
      throw(std::string("Error! in extracting real path"));
    }
    path.clear();  // reset path
    path.append(real_path);

    size_t pos = path.find(to_erase);
    if (pos != std::string::npos) path.erase(pos, to_erase.length());
  } else {
    throw(std::string("Error! in extracting real path"));
  }
  return path;
}

bool is_installed_path() {
  std::string path;
  char* real_path;
  Dl_info dl_info;

  if (0 != dladdr(reinterpret_cast<void*>(main), &dl_info)) {
    path = dl_info.dli_fname;
    real_path = realpath(path.c_str(), NULL);
    if (real_path == nullptr) {
      throw(std::string("Error! in extracting real path"));
    }
    path.clear();  // reset path
    path.append(real_path);
    if (path.find("/opt") != std::string::npos) {
      return true;
    }
  }
  return false;
}

static void init_test_path() {
  if (is_installed_path()) {
    test_app_path = "share/rocprofiler/tests/featuretests/profiler/apps/";
    hasco_path = "share/rocprofiler/tests/";
  } else {
    test_app_path = "tests-v2/featuretests/profiler/apps/";
    hasco_path = "tests-v2/featuretests/profiler/";
  }
}
