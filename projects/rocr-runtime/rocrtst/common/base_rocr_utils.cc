/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2017, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */

/// \file
/// Utility functions that act on BaseRocR objects.

#include "common/base_rocr_utils.h"
#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string>
#include "common/base_rocr.h"
#include "common/helper_funcs.h"
#include "common/os.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"

namespace rocrtst {


#define RET_IF_HSA_UTILS_ERR(err)                                                                  \
  {                                                                                                \
    if ((err) != HSA_STATUS_SUCCESS) {                                                             \
      const char* msg = 0;                                                                         \
      hsa_status_string(err, &msg);                                                                \
      EXPECT_EQ(HSA_STATUS_SUCCESS, err) << msg;                                                   \
      return (err);                                                                                \
    }                                                                                              \
  }

#define RET_IF_HSA_UTILS_ERR_RET(err, ret)                                                             \
  {                                                                                                \
    if ((err) != HSA_STATUS_SUCCESS) {                                                             \
      const char* msg = 0;                                                                         \
      hsa_status_string(err, &msg);                                                                \
      EXPECT_EQ(HSA_STATUS_SUCCESS, err) << msg;                                                   \
      return (ret);                                                                                \
    }                                                                                              \
  }
// Clean up some of the common handles and memory used by BaseRocR code, then
// shut down hsa. Restore HSA_ENABLE_INTERRUPT to original value, if necessary
hsa_status_t CommonCleanUp(BaseRocR* test) {
  hsa_status_t err;

  assert(test != nullptr);

  if (nullptr != test->kernarg_buffer()) {
    err = hsa_amd_memory_pool_free(test->kernarg_buffer());
    RET_IF_HSA_UTILS_ERR(err);
    test->set_kernarg_buffer(nullptr);
  }

  if (nullptr != test->main_queue()) {
    err = hsa_queue_destroy(test->main_queue());
    RET_IF_HSA_UTILS_ERR(err);
    test->set_main_queue(nullptr);
  }

  if (test->aql().completion_signal.handle != 0) {
    err = hsa_signal_destroy(test->aql().completion_signal);
    RET_IF_HSA_UTILS_ERR(err);
  }

  test->clear_code_object();
  err = hsa_shut_down();
  RET_IF_HSA_UTILS_ERR(err);

  // Ensure that HSA is actually closed.
  hsa_status_t check = hsa_shut_down();
  if (check != HSA_STATUS_ERROR_NOT_INITIALIZED) {
    EXPECT_EQ(HSA_STATUS_ERROR_NOT_INITIALIZED, check) << "hsa_init reference count was too high.";
    return HSA_STATUS_ERROR;
  }

  std::string intr_val;

  if (test->orig_hsa_enable_interrupt() == nullptr) {
    intr_val = "";
  } else {
    intr_val = test->orig_hsa_enable_interrupt();
  }

  SetEnv("HSA_ENABLE_INTERRUPT", intr_val.c_str());

  return err;
}

static const char* PROFILE_STR[] = {"HSA_PROFILE_BASE", "HSA_PROFILE_FULL", };

/// Verify that the machine running the test has the required profile.
/// This function will verify that the execution machine meets any specific
/// test requirement for a profile (HSA_PROFILE_BASE or HSA_PROFILE_FULL).
/// \param[in] test Test that provides profile requirements.
/// \returns bool
///          - true Machine meets test requirements
///          - false Machine does not meet test requirements
bool CheckProfileAndInform(BaseRocR* test) {
  if (test->verbosity() > 0) {
    std::cout << "Target HW Profile is "
              << PROFILE_STR[test->profile()] << std::endl;
  }

  if (test->requires_profile() == -1) {
    if (test->verbosity() > 0) {
      std::cout << "Test can run on any profile. OK." << std::endl;
    }
    return true;
  } else {
    std::cout << "Test requires " << PROFILE_STR[test->requires_profile()]
              << ". ";

    if (test->requires_profile() != test->profile()) {
      std::cout << "Not Running." << std::endl;
      return false;
    } else {
      std::cout << "OK." << std::endl;
      return true;
    }
  }
}

/// Helper function to process error returned from
///  iterate function like hsa_amd_agent_iterate_memory_pools
/// \param[in] Error returned from iterate call
/// \returns HSA_STATUS_SUCCESS iff iterate call succeeds in finding
///  what was being searched for
static hsa_status_t ProcessIterateError(hsa_status_t err) {
  if (err == HSA_STATUS_INFO_BREAK) {
    err = HSA_STATUS_SUCCESS;
  } else if (err == HSA_STATUS_SUCCESS) {
    // This actually means no pool was found.
    err = HSA_STATUS_ERROR;
  }
  return err;
}

// Find pools for cpu, gpu and for kernel arguments. These pools have
// common basic requirements, but are not suitable for all cases. In
// that case, set cpu_pool(), device_pool() and/or kern_arg_pool()
// yourself instead of using this function.
hsa_status_t SetPoolsTypical(BaseRocR* test) {
  hsa_status_t err;
  if (test->profile() == HSA_PROFILE_FULL) {
    err = hsa_amd_agent_iterate_memory_pools(*test->cpu_device(),
          rocrtst::FindAPUStandardPool, &test->cpu_pool());
    RET_IF_HSA_UTILS_ERR(rocrtst::ProcessIterateError(err));

    err = hsa_amd_agent_iterate_memory_pools(*test->cpu_device(),
          rocrtst::FindAPUStandardPool, &test->device_pool());
    RET_IF_HSA_UTILS_ERR(rocrtst::ProcessIterateError(err));

    err = hsa_amd_agent_iterate_memory_pools(*test->cpu_device(),
          rocrtst::FindAPUStandardPool, &test->kern_arg_pool());
    RET_IF_HSA_UTILS_ERR(rocrtst::ProcessIterateError(err));

  } else {
    err = hsa_amd_agent_iterate_memory_pools(*test->cpu_device(),
          rocrtst::FindStandardPool, &test->cpu_pool());
    RET_IF_HSA_UTILS_ERR(rocrtst::ProcessIterateError(err));

    err = hsa_amd_agent_iterate_memory_pools(*test->gpu_device1(),
          rocrtst::FindStandardPool, &test->device_pool());
    RET_IF_HSA_UTILS_ERR(rocrtst::ProcessIterateError(err));

    err = hsa_amd_agent_iterate_memory_pools(*test->cpu_device(),
          rocrtst::FindKernArgPool, &test->kern_arg_pool());
    RET_IF_HSA_UTILS_ERR(rocrtst::ProcessIterateError(err));
  }

  return HSA_STATUS_SUCCESS;
}

// Enable interrupts if necessary, and call hsa_init()
hsa_status_t InitAndSetupHSA(BaseRocR* test) {
  hsa_status_t err;

  if (test->enable_interrupt()) {
    SetEnv("HSA_ENABLE_INTERRUPT", "1");
  }

  err = hsa_init();
  RET_IF_HSA_UTILS_ERR(err);

  return HSA_STATUS_SUCCESS;
}

// Attempt to find and set test->cpu_device and test->gpu_device1
hsa_status_t SetDefaultAgents(BaseRocR* test) {
  hsa_agent_t gpu_device1;
  hsa_agent_t cpu_device;
  hsa_status_t err;

  gpu_device1.handle = 0;
  err = hsa_iterate_agents(FindGPUDevice, &gpu_device1);
  RET_IF_HSA_UTILS_ERR(rocrtst::ProcessIterateError(err));
  test->set_gpu_device1(gpu_device1);

  cpu_device.handle = 0;
  err = hsa_iterate_agents(FindCPUDevice, &cpu_device);
  RET_IF_HSA_UTILS_ERR(rocrtst::ProcessIterateError(err));
  test->set_cpu_device(cpu_device);

  if (0 == gpu_device1.handle) {
    std::cout << "GPU Device is not Created properly!" << std::endl;
    RET_IF_HSA_UTILS_ERR(HSA_STATUS_ERROR);
  }

  if (0 == cpu_device.handle) {
    std::cout << "CPU Device is not Created properly!" << std::endl;
    RET_IF_HSA_UTILS_ERR(HSA_STATUS_ERROR);
  }

  if (test->verbosity() > 0) {
    char name[64] = {0};
    err = hsa_agent_get_info(gpu_device1, HSA_AGENT_INFO_NAME, name);
    RET_IF_HSA_UTILS_ERR(err);
    std::cout << "The gpu device name is " << name << std::endl;
  }

  hsa_profile_t profile;
  err = hsa_agent_get_info(gpu_device1, HSA_AGENT_INFO_PROFILE, &profile);
  RET_IF_HSA_UTILS_ERR(err);
  test->set_profile(profile);

  if (!CheckProfileAndInform(test)) {
    return HSA_STATUS_ERROR;
  }
  return HSA_STATUS_SUCCESS;
}

// See if the profile of the target matches any required profile by the
// test program.
bool CheckProfile(BaseRocR const* test) {
  if (test->requires_profile() == -1) {
    return true;
  } else {
    return (test->requires_profile() == test->profile());
  }
}

/// Locate file using local and device named file paths.
std::string LocateKernelFile(std::string filename, hsa_agent_t agent) {
  char agent_name[64];
  std::string obj_file;
  hsa_status_t err = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, agent_name);
  RET_IF_HSA_UTILS_ERR_RET(err, obj_file);

  obj_file = "./" + filename;
  int file_handle = open(obj_file.c_str(), O_RDONLY);
  if (file_handle < 0) {
    obj_file = "./" + std::string(agent_name) + "/" + filename;
    file_handle = open(obj_file.c_str(), O_RDONLY);
    if(file_handle < 0)
      std::runtime_error("Could not open file.\n");
  }

  close(file_handle);
  return obj_file;
}

// Load the specified kernel code from the specified file, inspect and fill
// in BaseRocR member variables related to the kernel and executable.
// Required Input BaseRocR member variables:
// - gpu_device1()
// - kernel_file_name()
// - kernel_name()
//
// Written BaseRocR member variables:
//  -kernel_object()
//  -private_segment_size()
//  -group_segment_size()
//  -kernarg_size()
//  -kernarg_align()
hsa_status_t LoadKernelFromObjFile(BaseRocR* test, hsa_agent_t* agent) {
  hsa_status_t err;
  Kernel kern;
  std::string kern_name;
  char agent_name[64];
  std::string obj_file;
  CodeObject* obj;

  assert(test != nullptr);
  if (agent == nullptr) {
    agent = test->gpu_device1();  // Assume GPU agent for now
  }

  obj_file = LocateKernelFile(test->kernel_file_name(), *agent);
  Device *gpu = (Device*)(agent - offsetof(Device, agent));
  obj = new CodeObject(obj_file, *gpu);
  test->set_code_object(obj);
  kern_name = test->kernel_name() + ".kd";

  if(!obj->GetKernel(kern_name, kern)) {
      ADD_FAILURE();
      return HSA_STATUS_ERROR;
  }

  test->set_kernel_object(kern.handle);
  test->set_private_segment_size(kern.scratch);
  test->set_group_segment_size(kern.group);
  test->set_kernarg_size(kern.kernarg_size);
  assert(kern.kernarg_align >= 16 && "Reported kernarg size is too small.");
  kern.kernarg_size = (kern.kernarg_size == 0) ? 16 : kern.kernarg_size;
  test->set_kernarg_align(kern.kernarg_size);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t CreateQueue(hsa_agent_t device, hsa_queue_t** queue,
                         uint32_t num_pkts) {
  hsa_status_t err;

  if (num_pkts == 0) {
    err = hsa_agent_get_info(device, HSA_AGENT_INFO_QUEUE_MAX_SIZE,
                             &num_pkts);
    RET_IF_HSA_UTILS_ERR(err);
  }

  err = hsa_queue_create(device, num_pkts, HSA_QUEUE_TYPE_MULTI, NULL,
                         NULL, UINT32_MAX, UINT32_MAX, queue);
  RET_IF_HSA_UTILS_ERR(err);

  return HSA_STATUS_SUCCESS;
}
// Initialize the provided aql packet with standard default values, and
// values from provided BaseRocR object.
hsa_status_t InitializeAQLPacket(const BaseRocR* test,
                         hsa_kernel_dispatch_packet_t* aql) {
  hsa_status_t err;

  assert(aql != nullptr);

  if (aql == nullptr) {
    return HSA_STATUS_ERROR;
  }
  
  // Initialize Packet type as Invalid
  // Update packet type to Kernel Dispatch
  // right before ringing doorbell
  aql->header = 1;

  aql->setup = 1;
  aql->workgroup_size_x = 256;
  aql->workgroup_size_y = 1;
  aql->workgroup_size_z = 1;

  aql->grid_size_x = (uint64_t) 256;  // manual_input*group_input; workg max sz
  aql->grid_size_y = 1;
  aql->grid_size_z = 1;

  aql->private_segment_size = test->private_segment_size();

  aql->group_segment_size = test->group_segment_size();

  // Pin kernel code and the kernel argument buffer to the aql packet->
  aql->kernel_object = test->kernel_object();

  // aql->kernarg_address may be filled in by AllocAndSetKernArgs() if it is
  // called before this function, so we don't want overwrite it, therefore
  // we ignore it in this function.

  if (!aql->completion_signal.handle)
    err = hsa_signal_create(1, 0, NULL, &aql->completion_signal);
  else
    err = HSA_STATUS_SUCCESS;

  return err;
}

// Copy BaseRocR aql object values to the BaseRocR object queue in the
// specified queue position (ind)
hsa_kernel_dispatch_packet_t * WriteAQLToQueue(BaseRocR* test, uint64_t *ind) {
  assert(test);
  assert(test->main_queue());

  void *queue_base = test->main_queue()->base_address;
  const uint32_t queue_mask = test->main_queue()->size - 1;
  uint64_t que_idx = hsa_queue_add_write_index_relaxed(test->main_queue(), 1);
  *ind = que_idx;

  hsa_kernel_dispatch_packet_t* staging_aql_packet = &test->aql();
  hsa_kernel_dispatch_packet_t* queue_aql_packet;

  queue_aql_packet =
       &(reinterpret_cast<hsa_kernel_dispatch_packet_t*>(queue_base))
                                                        [que_idx & queue_mask];

  queue_aql_packet->workgroup_size_x = staging_aql_packet->workgroup_size_x;
  queue_aql_packet->workgroup_size_y = staging_aql_packet->workgroup_size_y;
  queue_aql_packet->workgroup_size_z = staging_aql_packet->workgroup_size_z;
  queue_aql_packet->grid_size_x = staging_aql_packet->grid_size_x;
  queue_aql_packet->grid_size_y = staging_aql_packet->grid_size_y;
  queue_aql_packet->grid_size_z = staging_aql_packet->grid_size_z;
  queue_aql_packet->private_segment_size =
                                     staging_aql_packet->private_segment_size;
  queue_aql_packet->group_segment_size =
                                       staging_aql_packet->group_segment_size;
  queue_aql_packet->kernel_object = staging_aql_packet->kernel_object;
  queue_aql_packet->kernarg_address = staging_aql_packet->kernarg_address;
  queue_aql_packet->completion_signal = staging_aql_packet->completion_signal;

  return queue_aql_packet;
}

void
WriteAQLToQueueLoc(hsa_queue_t *queue, uint64_t indx,
                                      hsa_kernel_dispatch_packet_t *aql_pkt) {
  assert(queue);
  assert(aql_pkt);

  void *queue_base = queue->base_address;
  const uint32_t queue_mask = queue->size - 1;
  hsa_kernel_dispatch_packet_t* queue_aql_packet;

  queue_aql_packet =
       &(reinterpret_cast<hsa_kernel_dispatch_packet_t*>(queue_base))
                                                        [indx & queue_mask];

  queue_aql_packet->workgroup_size_x = aql_pkt->workgroup_size_x;
  queue_aql_packet->workgroup_size_y = aql_pkt->workgroup_size_y;
  queue_aql_packet->workgroup_size_z = aql_pkt->workgroup_size_z;
  queue_aql_packet->grid_size_x = aql_pkt->grid_size_x;
  queue_aql_packet->grid_size_y = aql_pkt->grid_size_y;
  queue_aql_packet->grid_size_z = aql_pkt->grid_size_z;
  queue_aql_packet->private_segment_size =
                                     aql_pkt->private_segment_size;
  queue_aql_packet->group_segment_size =
                                       aql_pkt->group_segment_size;
  queue_aql_packet->kernel_object = aql_pkt->kernel_object;
  queue_aql_packet->kernarg_address = aql_pkt->kernarg_address;
  queue_aql_packet->completion_signal = aql_pkt->completion_signal;
}

// Allocate a buffer in the kern_arg_pool for the kernel arguments and write
// the arguments to buffer
hsa_status_t AllocAndSetKernArgs(BaseRocR* test, void* args, size_t arg_size) {
  void* kern_arg_buf = nullptr;
  hsa_status_t err;
  size_t buf_size;
  size_t req_align;
  assert(args != nullptr);
  assert(test != nullptr);

  req_align = test->kernarg_align();
  // Allocate enough extra space for alignment adjustments if ncessary
  buf_size = arg_size + (req_align << 1);

  err = hsa_amd_memory_pool_allocate(test->kern_arg_pool(), buf_size, 0,
                                     reinterpret_cast<void**>(&kern_arg_buf));
  RET_IF_HSA_UTILS_ERR(err);

  test->set_kernarg_buffer(kern_arg_buf);

  void *adj_kern_arg_buf = rocrtst::AlignUp(kern_arg_buf, req_align);

  assert(arg_size >= test->kernarg_size());
  assert(((uintptr_t)adj_kern_arg_buf + arg_size) <
                                        ((uintptr_t)kern_arg_buf + buf_size));

  hsa_agent_t ag_list[2] = {*test->gpu_device1(), *test->cpu_device()};
  err = hsa_amd_agents_allow_access(2, ag_list, NULL, kern_arg_buf);
  RET_IF_HSA_UTILS_ERR(err);

  err = hsa_memory_copy(adj_kern_arg_buf, args, arg_size);
  RET_IF_HSA_UTILS_ERR(err);

  test->aql().kernarg_address = adj_kern_arg_buf;

  return HSA_STATUS_SUCCESS;
}

#undef RET_IF_HSA_UTILS_ERR

}  // namespace rocrtst
