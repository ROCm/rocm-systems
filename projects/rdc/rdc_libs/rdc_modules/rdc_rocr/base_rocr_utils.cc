/*
Copyright (c) 2021 - present Advanced Micro Devices, Inc. All rights reserved.

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

#include "rdc_modules/rdc_rocr/base_rocr_utils.h"

#include <assert.h>
#include <fcntl.h>
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <stdexcept>
#include <string>

#include "hsa/hsa.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

// Clean up some of the common handles and memory used by RdcRocrBase code, then
// shut down hsa. Restore HSA_ENABLE_INTERRUPT to original value, if necessary
hsa_status_t CommonCleanUp(RdcRocrBase* test) {
  hsa_status_t err;

  assert(test != nullptr);

  if (nullptr != test->kernarg_buffer()) {
    err = hsa_amd_memory_pool_free(test->kernarg_buffer());
    throw_if_error(err);
    test->set_kernarg_buffer(nullptr);
  }

  if (nullptr != test->main_queue()) {
    err = hsa_queue_destroy(test->main_queue());
    throw_if_error(err);
    test->set_main_queue(nullptr);
  }

  if (test->aql().completion_signal.handle != 0) {
    err = hsa_signal_destroy(test->aql().completion_signal);
    throw_if_error(err);
  }

  err = hsa_shut_down();
  throw_if_error(err);

  std::string intr_val;

  if (test->orig_hsa_enable_interrupt() == nullptr) {
    intr_val = "";
  } else {
    intr_val = test->orig_hsa_enable_interrupt();
  }

  SetEnv("HSA_ENABLE_INTERRUPT", intr_val.c_str());

  return err;
}

static const char* PROFILE_STR[] = {
    "HSA_PROFILE_BASE",
    "HSA_PROFILE_FULL",
};

/// Verify that the machine running the test has the required profile.
/// This function will verify that the execution machine meets any specific
/// test requirement for a profile (HSA_PROFILE_BASE or HSA_PROFILE_FULL).
/// \param[in] test Test that provides profile requirements.
/// \returns bool
///          - true Machine meets test requirements
///          - false Machine does not meet test requirements
bool CheckProfileAndInform(RdcRocrBase* test) {
  if (test->verbosity() > 0) {
    RDC_LOG(RDC_DEBUG, "Target HW Profile is " << PROFILE_STR[test->profile()]);
  }

  if (test->requires_profile() == -1) {
    if (test->verbosity() > 0) {
      RDC_LOG(RDC_DEBUG, "Test can run on any profile. OK.");
    }
    return true;
  } else {
    RDC_LOG(RDC_DEBUG, "Test requires " << PROFILE_STR[test->requires_profile()] << ". ");
    if (test->requires_profile() != test->profile()) {
      RDC_LOG(RDC_DEBUG, "Not Running.");
      return false;
    } else {
      RDC_LOG(RDC_DEBUG, "OK.");
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
hsa_status_t SetPoolsTypical(RdcRocrBase* test) {
  hsa_status_t err;
  if (test->profile() == HSA_PROFILE_FULL) {
    err = hsa_amd_agent_iterate_memory_pools(*test->cpu_device(), FindAPUStandardPool,
                                             &test->cpu_pool());
    throw_if_error(ProcessIterateError(err));

    err = hsa_amd_agent_iterate_memory_pools(*test->cpu_device(), FindAPUStandardPool,
                                             &test->device_pool());
    throw_if_error(ProcessIterateError(err));

    err = hsa_amd_agent_iterate_memory_pools(*test->cpu_device(), FindAPUStandardPool,
                                             &test->kern_arg_pool());
    throw_if_error(ProcessIterateError(err));

  } else {
    err = hsa_amd_agent_iterate_memory_pools(*test->cpu_device(), FindStandardPool,
                                             &test->cpu_pool());
    throw_if_error(ProcessIterateError(err));

    err = hsa_amd_agent_iterate_memory_pools(*test->gpu_device1(), FindStandardPool,
                                             &test->device_pool());
    throw_if_error(ProcessIterateError(err));

    err = hsa_amd_agent_iterate_memory_pools(*test->cpu_device(), FindKernArgPool,
                                             &test->kern_arg_pool());
    throw_if_error(ProcessIterateError(err));
  }

  return HSA_STATUS_SUCCESS;
}

// Enable interrupts if necessary, and call hsa_init()
hsa_status_t InitAndSetupHSA(RdcRocrBase* test) {
  hsa_status_t err;

  if (test->enable_interrupt()) {
    SetEnv("HSA_ENABLE_INTERRUPT", "1");
  }

  err = hsa_init();
  throw_if_error(err);

  return HSA_STATUS_SUCCESS;
}

// Attempt to find and set test->cpu_device and test->gpu_device1
hsa_status_t SetDefaultAgents(RdcRocrBase* test) {
  hsa_agent_t gpu_device1;
  hsa_agent_t cpu_device;
  hsa_status_t err;

  gpu_device1.handle = 0;
  err = hsa_iterate_agents(FindGPUDevice, &gpu_device1);
  throw_if_error(ProcessIterateError(err));
  test->set_gpu_device1(gpu_device1);

  cpu_device.handle = 0;
  err = hsa_iterate_agents(FindCPUDevice, &cpu_device);
  throw_if_error(ProcessIterateError(err));
  test->set_cpu_device(cpu_device);

  if (0 == gpu_device1.handle) {
    RDC_LOG(RDC_ERROR, "GPU Device is not Created properly!");
    throw_if_error(HSA_STATUS_ERROR, "GPU Device is not Created properly!");
  }

  if (0 == cpu_device.handle) {
    RDC_LOG(RDC_ERROR, "CPU Device is not Created properly!");
    throw_if_error(HSA_STATUS_ERROR, "CPU Device is not Created properly!");
  }

  if (test->verbosity() > 0) {
    char name[64] = {0};
    err = hsa_agent_get_info(gpu_device1, HSA_AGENT_INFO_NAME, name);
    throw_if_error(err);
    RDC_LOG(RDC_DEBUG, "The gpu device name is " << name);
  }

  hsa_profile_t profile;
  err = hsa_agent_get_info(gpu_device1, HSA_AGENT_INFO_PROFILE, &profile);
  throw_if_error(err);
  test->set_profile(profile);

  if (!CheckProfileAndInform(test)) {
    return HSA_STATUS_ERROR;
  }
  return HSA_STATUS_SUCCESS;
}

// See if the profile of the target matches any required profile by the
// test program.
bool CheckProfile(RdcRocrBase const* test) {
  if (test->requires_profile() == -1) {
    return true;
  } else {
    return (test->requires_profile() == test->profile());
  }
}
// Load the specified kernel code from the specified file, inspect and fill
// in RdcRocrBase member variables related to the kernel and executable.
// Required Input RdcRocrBase member variables:
// - gpu_device1()
// - kernel_file_name()
// - kernel_name()
//
// Written RdcRocrBase member variables:
//  -kernel_object()
//  -private_segment_size()
//  -group_segment_size()
//  -kernarg_size()
//  -kernarg_align()
hsa_status_t LoadKernelFromObjFile(RdcRocrBase* test, hsa_agent_t* agent) {
  hsa_status_t err;
  hsa_code_object_reader_t code_obj_rdr = {0};
  hsa_executable_t executable = {0};

  assert(test != nullptr);
  if (agent == nullptr) {
    agent = test->gpu_device1();  // Assume GPU agent for now
  }

  // if agent name is not set, then set the agent name
  if (!test->get_agent_name().size()) {
    char agent_name[64];
    err = hsa_agent_get_info(*agent, HSA_AGENT_INFO_NAME, agent_name);
    throw_if_error(err);
    test->set_agent_name(agent_name);
  }

  std::string kern_name = test->kernel_name();
  std::string obj_file =
      search_hsaco_full_path(test->kernel_file_name().c_str(), test->get_agent_name().c_str());
  if (obj_file == "") {
    RDC_LOG(RDC_ERROR, "failed to find " << test->kernel_file_name() << " at line " << __LINE__
                                         << ", errno: " << errno);
    std::string msg("fail to open ");
    msg += test->kernel_file_name();
    throw_if_skip(msg);
    return HSA_STATUS_ERROR;
  }

  hsa_file_t file_handle = open(obj_file.c_str(), O_RDONLY);

  if (file_handle == -1) {
    RDC_LOG(RDC_ERROR, "failed to open " << obj_file.c_str() << " at line " << __LINE__
                                         << ", file: " << __FILE__);
    return (hsa_status_t)errno;
  }

  err = hsa_code_object_reader_create_from_file(file_handle, &code_obj_rdr);
  throw_if_error(err);
  close(file_handle);

  err = hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, NULL,
                                  &executable);
  throw_if_error(err);
  err = hsa_executable_load_agent_code_object(executable, *agent, code_obj_rdr, NULL, NULL);
  throw_if_error(err);
  err = hsa_executable_freeze(executable, NULL);
  throw_if_error(err);

  hsa_executable_symbol_t kern_sym;
  err = hsa_executable_get_symbol(executable, NULL, (kern_name + ".kd").c_str(), *agent, 0,
                                  &kern_sym);
  throw_if_error(err);

  uint64_t codeHandle;
  err = hsa_executable_symbol_get_info(kern_sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                       &codeHandle);
  throw_if_error(err);
  test->set_kernel_object(codeHandle);

  uint32_t val;
  err = hsa_executable_symbol_get_info(
      kern_sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &val);
  throw_if_error(err);
  test->set_private_segment_size(val);

  err = hsa_executable_symbol_get_info(kern_sym,
                                       HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &val);
  throw_if_error(err);
  test->set_group_segment_size(val);

  // Remaining queries only supported on code object v3.
  err = hsa_executable_symbol_get_info(
      kern_sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &val);
  throw_if_error(err);
  test->set_kernarg_size(val);

  err = hsa_executable_symbol_get_info(
      kern_sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT, &val);
  throw_if_error(err);
  assert(val >= 16 && "Reported kernarg size is too small.");
  val = (val == 0) ? 16 : val;
  test->set_kernarg_align(val);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t CreateQueue(hsa_agent_t device, hsa_queue_t** queue, uint32_t num_pkts) {
  hsa_status_t err;

  if (num_pkts == 0) {
    err = hsa_agent_get_info(device, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &num_pkts);
    throw_if_error(err);
  }

  err = hsa_queue_create(device, num_pkts, HSA_QUEUE_TYPE_MULTI, NULL, NULL, UINT32_MAX, UINT32_MAX,
                         queue);
  throw_if_error(err);

  return HSA_STATUS_SUCCESS;
}
// Initialize the provided aql packet with standard default values, and
// values from provided RdcRocrBase object.
hsa_status_t InitializeAQLPacket(const RdcRocrBase* test, hsa_kernel_dispatch_packet_t* aql) {
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

  aql->grid_size_x = (uint64_t)256;  // manual_input*group_input; workg max sz
  aql->grid_size_y = 1;
  aql->grid_size_z = 1;

  aql->private_segment_size = test->private_segment_size();

  aql->group_segment_size = test->group_segment_size();

  // Pin kernel code and the kernel argument buffer to the aql packet->
  aql->kernel_object = test->kernel_object();

  // aql->kernarg_address may be filled in by AllocAndSetKernArgs() if it is
  // called before this function, so we don't want overwrite it, therefore
  // we ignore it in this function.

  err = hsa_signal_create(1, 0, NULL, &aql->completion_signal);

  return err;
}

// Copy RdcRocrBase aql object values to the RdcRocrBase object queue in the
// specified queue position (ind)
hsa_kernel_dispatch_packet_t* WriteAQLToQueue(RdcRocrBase* test, uint64_t* ind) {
  assert(test);
  assert(test->main_queue());

  void* queue_base = test->main_queue()->base_address;
  const uint32_t queue_mask = test->main_queue()->size - 1;
  uint64_t que_idx = hsa_queue_add_write_index_relaxed(test->main_queue(), 1);
  *ind = que_idx;

  hsa_kernel_dispatch_packet_t* staging_aql_packet = &test->aql();
  hsa_kernel_dispatch_packet_t* queue_aql_packet;

  queue_aql_packet =
      &(reinterpret_cast<hsa_kernel_dispatch_packet_t*>(queue_base))[que_idx & queue_mask];

  queue_aql_packet->workgroup_size_x = staging_aql_packet->workgroup_size_x;
  queue_aql_packet->workgroup_size_y = staging_aql_packet->workgroup_size_y;
  queue_aql_packet->workgroup_size_z = staging_aql_packet->workgroup_size_z;
  queue_aql_packet->grid_size_x = staging_aql_packet->grid_size_x;
  queue_aql_packet->grid_size_y = staging_aql_packet->grid_size_y;
  queue_aql_packet->grid_size_z = staging_aql_packet->grid_size_z;
  queue_aql_packet->private_segment_size = staging_aql_packet->private_segment_size;
  queue_aql_packet->group_segment_size = staging_aql_packet->group_segment_size;
  queue_aql_packet->kernel_object = staging_aql_packet->kernel_object;
  queue_aql_packet->kernarg_address = staging_aql_packet->kernarg_address;
  queue_aql_packet->completion_signal = staging_aql_packet->completion_signal;

  return queue_aql_packet;
}

void WriteAQLToQueueLoc(hsa_queue_t* queue, uint64_t indx, hsa_kernel_dispatch_packet_t* aql_pkt) {
  assert(queue);
  assert(aql_pkt);

  void* queue_base = queue->base_address;
  const uint32_t queue_mask = queue->size - 1;
  hsa_kernel_dispatch_packet_t* queue_aql_packet;

  queue_aql_packet =
      &(reinterpret_cast<hsa_kernel_dispatch_packet_t*>(queue_base))[indx & queue_mask];

  queue_aql_packet->workgroup_size_x = aql_pkt->workgroup_size_x;
  queue_aql_packet->workgroup_size_y = aql_pkt->workgroup_size_y;
  queue_aql_packet->workgroup_size_z = aql_pkt->workgroup_size_z;
  queue_aql_packet->grid_size_x = aql_pkt->grid_size_x;
  queue_aql_packet->grid_size_y = aql_pkt->grid_size_y;
  queue_aql_packet->grid_size_z = aql_pkt->grid_size_z;
  queue_aql_packet->private_segment_size = aql_pkt->private_segment_size;
  queue_aql_packet->group_segment_size = aql_pkt->group_segment_size;
  queue_aql_packet->kernel_object = aql_pkt->kernel_object;
  queue_aql_packet->kernarg_address = aql_pkt->kernarg_address;
  queue_aql_packet->completion_signal = aql_pkt->completion_signal;
}

// Allocate a buffer in the kern_arg_pool for the kernel arguments and write
// the arguments to buffer
hsa_status_t AllocAndSetKernArgs(RdcRocrBase* test, void* args, size_t arg_size) {
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
  throw_if_error(err);

  test->set_kernarg_buffer(kern_arg_buf);

  void* adj_kern_arg_buf = AlignUp(kern_arg_buf, req_align);

  assert(arg_size >= test->kernarg_size());
  assert(((uintptr_t)adj_kern_arg_buf + arg_size) < ((uintptr_t)kern_arg_buf + buf_size));

  hsa_agent_t ag_list[2] = {*test->gpu_device1(), *test->cpu_device()};
  err = hsa_amd_agents_allow_access(2, ag_list, NULL, kern_arg_buf);
  throw_if_error(err);

  err = hsa_memory_copy(adj_kern_arg_buf, args, arg_size);
  throw_if_error(err);

  test->aql().kernarg_address = adj_kern_arg_buf;

  return HSA_STATUS_SUCCESS;
}

std::string get_lib_dir(const char* lib_name) {
  std::string result;
  char line[1024 * 8];

  FILE* file = fopen("/proc/self/maps", "r");
  if (file == NULL) return result;
  std::string lib_path = "/";
  lib_path += lib_name;
  // 7f4eacb46000 r-xp 00000 08:01 17183106 /lib/x86_64-linux-gnu/libc-2.27.so
  while (fgets(line, sizeof(line), file)) {
    char* end = strstr(line, lib_path.c_str());
    if (end != NULL) {
      char* start = end;
      while (start > line) {
        if (isspace(*start)) {
          start++;
          break;
        }
        start--;
      }
      result = std::string(start, end - start);
      break;
    }
  }
  fclose(file);

  return result;
}

std::string get_app_dir() {
  char buf[1024 * 8];
  int ret = readlink("/proc/self/exe", buf, 1024 * 8);
  if ((ret != -1) && ret < (1024 * 8 - 1)) {
    buf[ret] = '\0';
    return dirname(buf);
  }
  return "";
}

std::string search_hsaco_full_path(const char* hsaco_file_name, const char* agent_name) {
  const std::string lib_dir = get_lib_dir("librdc_rocr.so");
  const std::string app_dir = get_app_dir();

  std::vector<std::string> path_to_search;
  path_to_search.push_back(std::string("./") + hsaco_file_name);
  path_to_search.push_back(app_dir + "/" + hsaco_file_name);
  path_to_search.push_back(lib_dir + "/" + hsaco_file_name);
  path_to_search.push_back(lib_dir + "/rdc/hsaco/" + agent_name + "/" + hsaco_file_name);
  path_to_search.push_back(lib_dir + "/hsaco/" + agent_name + "/" + hsaco_file_name);
  // for dev structure
  path_to_search.push_back(lib_dir + "/../../rdc_libs/rdc_modules/kernels/hsaco/" + agent_name +
                           "/" + hsaco_file_name);
  for (std::size_t i = 0; i < path_to_search.size(); i++) {
    if (::access(path_to_search[i].c_str(), F_OK) == 0) {
      RDC_LOG(RDC_DEBUG, "Use the file " << path_to_search[i]);
      return path_to_search[i];
    }
    RDC_LOG(RDC_DEBUG, "Skip not exists file " << path_to_search[i]);
  }
  return "";
}

}  // namespace rdc
}  // namespace amd
