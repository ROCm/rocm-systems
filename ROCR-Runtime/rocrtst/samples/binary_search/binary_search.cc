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

#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <string>
#include <iostream>
#include <climits>
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

#define RET_IF_HSA_ERR(err) { \
  if ((err) != HSA_STATUS_SUCCESS) { \
    std::cout << "hsa api call failure at line " << __LINE__ << ", file: " << \
              __FILE__ << ". Call returned " << err << std::endl; \
    return (err); \
  } \
}

#ifndef ROCRTST_EMULATOR_BUILD
static const uint32_t kBinarySearchLength = 512;
static const uint32_t kBinarySearchFindMe = 108;
static const uint32_t kWorkGroupSize = 256;
#else
static const uint32_t kBinarySearchLength = 16;
static const uint32_t kBinarySearchFindMe = 6;
static const uint32_t kWorkGroupSize = 8;
#endif

// Hold all the info specific to binary search
typedef struct BinarySearch {
  // Binary Search parameters
  uint32_t length;
  uint32_t work_group_size;
  uint32_t work_grid_size;
  uint32_t num_sub_divisions;
  uint32_t find_me;

  // Buffers needed for this application
  uint32_t* input;
  uint32_t* input_arr;
  uint32_t* input_arr_local;
  uint32_t* output;
  // Keneral argument buffers and addresses
  void* kern_arg_buffer;  // Begin of allocated memory
  //  this pointer to be deallocated
  void* kern_arg_address;  // Properly aligned address to be used in aql
  // packet (don't use for deallocation)

  // Kernel code
  std::string kernel_file_name;
  std::string kernel_name;
  uint32_t kernarg_size;
  uint32_t kernarg_align;

  // HSA/RocR objects needed for this application
  hsa_agent_t gpu_dev;
  hsa_agent_t cpu_dev;
  hsa_signal_t signal;
  hsa_queue_t* queue;
  hsa_amd_memory_pool_t cpu_pool;
  hsa_amd_memory_pool_t gpu_pool;
  hsa_amd_memory_pool_t kern_arg_pool;

  // Other items we need to populate AQL packet
  uint64_t kernel_object;
  uint32_t group_segment_size;   ///< Kernel group seg size
  uint32_t private_segment_size;   ///< Kernel private seg size
} BinarySearch;

void InitializeBinarySearch(BinarySearch* bs) {
  bs->kernel_file_name = "./binary_search_kernels.hsaco";
  bs->kernel_name = "binarySearch.kd";
  bs->length = kBinarySearchLength;
  bs->find_me = kBinarySearchFindMe;
  bs->work_group_size = kWorkGroupSize;
  bs->num_sub_divisions = bs->length / bs->work_group_size;
}

// This function is called by the call-back functions used to find an agent of
// the specified hsa_device_type_t. Note that it cannot be called directly from
// hsa_iterate_agents() as it does not match the prototype of the call-back
// function. It must be wrapped by a function with the correct prototype.
//
// Return values:
//  HSA_STATUS_INFO_BREAK -- "agent" is of the specified type (dev_type)
//  HSA_STATUS_SUCCESS -- "agent" is not of the specified type
//  Other -- Some error occurred
static hsa_status_t FindAgent(hsa_agent_t agent, void* data,
                              hsa_device_type_t dev_type) {
  if (data == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  // See if the provided agent matches the input type (dev_type)
  hsa_device_type_t hsa_device_type;
  hsa_status_t hsa_error_code = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE,
                                &hsa_device_type);
  RET_IF_HSA_ERR(hsa_error_code);

  if (hsa_device_type == dev_type) {
    *(reinterpret_cast<hsa_agent_t*>(data)) = agent;
    return HSA_STATUS_INFO_BREAK;
  }

  return HSA_STATUS_SUCCESS;
}

// This is the call-back function used to find a GPU type agent. Note that the
// prototype of this function is dictated by the HSA specification
hsa_status_t FindGPUDevice(hsa_agent_t agent, void* data) {
  return FindAgent(agent, data, HSA_DEVICE_TYPE_GPU);
}

// This is the call-back function used to find a CPU type agent. Note that the
// prototype of this function is dictated by the HSA specification
hsa_status_t FindCPUDevice(hsa_agent_t agent, void* data) {
  return FindAgent(agent, data, HSA_DEVICE_TYPE_CPU);
}

// Find the CPU and GPU agents we need to run this sample, and save them in the
// BinarySearch structure for later use.
hsa_status_t FindDevices(BinarySearch* bs) {
  hsa_status_t err;

  // Note that hsa_iterate_agents iterate through all known agents until
  // HSA_STATUS_SUCCESS is not returned. The call-backs are implemented such
  // that HSA_STATUS_INFO_BREAK means we found an agent of the specified type.
  // This value is returned by hsa_iterate_agents.
  bs->gpu_dev.handle = 0;
  err = hsa_iterate_agents(FindGPUDevice, &bs->gpu_dev);

  if (err != HSA_STATUS_INFO_BREAK) {
    return HSA_STATUS_ERROR;
  }

  bs->cpu_dev.handle = 0;
  err = hsa_iterate_agents(FindCPUDevice, &bs->cpu_dev);

  if (err != HSA_STATUS_INFO_BREAK) {
    return HSA_STATUS_ERROR;
  }

  if (0 == bs->gpu_dev.handle) {
    std::cout << "GPU Device is not Created properly!" << std::endl;
    RET_IF_HSA_ERR(HSA_STATUS_ERROR);
  }

  if (0 == bs->cpu_dev.handle) {
    std::cout << "CPU Device is not Created properly!" << std::endl;
    RET_IF_HSA_ERR(HSA_STATUS_ERROR);
  }

  return HSA_STATUS_SUCCESS;
}

// This function checks to see if the provided
// pool has the HSA_AMD_SEGMENT_GLOBAL property. If the kern_arg flag is true,
// the function adds an additional requirement that the pool have the
// HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT property. If kern_arg is false,
// pools must NOT have this property.
// Upon finding a pool that meets these conditions, HSA_STATUS_INFO_BREAK is
// returned. HSA_STATUS_SUCCESS is returned if no errors were encountered, but
// no pool was found meeting the requirements. If an error is encountered, we
// return that error.

// Note that this function does not match the required prototype for the
// hsa_amd_agent_iterate_memory_pools call back function, and therefore must be
// wrapped by a function with the correct prototype.
static hsa_status_t
FindGlobalPool(hsa_amd_memory_pool_t pool, void* data, bool kern_arg) {
  hsa_status_t err;
  hsa_amd_segment_t segment;
  uint32_t flag;

  if (nullptr == data) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                                     &segment);
  RET_IF_HSA_ERR(err);

  if (HSA_AMD_SEGMENT_GLOBAL != segment) {
    return HSA_STATUS_SUCCESS;
  }

  err = hsa_amd_memory_pool_get_info(pool,
                                HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flag);
  RET_IF_HSA_ERR(err);

  uint32_t karg_st = flag & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT;

  if ((karg_st == 0 && kern_arg) ||
      (karg_st != 0 && !kern_arg)) {
    return HSA_STATUS_SUCCESS;
  }

  *(reinterpret_cast<hsa_amd_memory_pool_t*>(data)) = pool;
  return HSA_STATUS_INFO_BREAK;
}

// This is the call-back function for hsa_amd_agent_iterate_memory_pools() that
// finds a pool with the properties of HSA_AMD_SEGMENT_GLOBAL and that is NOT
// HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT
hsa_status_t FindStandardPool(hsa_amd_memory_pool_t pool, void* data) {
  return FindGlobalPool(pool, data, false);
}

// This is the call-back function for hsa_amd_agent_iterate_memory_pools() that
// finds a pool with the properties of HSA_AMD_SEGMENT_GLOBAL and that IS
// HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT
hsa_status_t FindKernArgPool(hsa_amd_memory_pool_t pool, void* data) {
  return FindGlobalPool(pool, data, true);
}

// Find memory pools that we will need to allocate from for this sample
// application. We will need memory associated with the host CPU, the GPU
// executing the kernels, and for kernel arguments. This function will
// save the found pools to the BinarySearch structure for use elsewhere
// in this program.
hsa_status_t FindPools(BinarySearch* bs) {
  hsa_status_t err;

  err = hsa_amd_agent_iterate_memory_pools(bs->cpu_dev, FindStandardPool,
        &bs->cpu_pool);

  if (err != HSA_STATUS_INFO_BREAK) {
    return HSA_STATUS_ERROR;
  }

  err = hsa_amd_agent_iterate_memory_pools(bs->gpu_dev, FindStandardPool,
        &bs->gpu_pool);

  if (err != HSA_STATUS_INFO_BREAK) {
    return HSA_STATUS_ERROR;
  }

  err = hsa_amd_agent_iterate_memory_pools(bs->cpu_dev,
        FindKernArgPool, &bs->kern_arg_pool);

  if (err != HSA_STATUS_INFO_BREAK) {
    return HSA_STATUS_ERROR;
  }

  return HSA_STATUS_SUCCESS;
}

// Once the needed memory pools have been found and the BinarySearch structure
// has been updated with these handles, this function is then used to allocate
// memory from those pools.
// Devices with which a pool is associated already have access to the pool.
// However, other devices may also need to read or write to that memory. Below,
// we see how we can grant access to other devices to address this issue.
hsa_status_t AllocateAndInitBuffers(BinarySearch* bs) {
  hsa_status_t err;
  uint32_t out_length = 4 * sizeof(uint32_t);
  uint32_t in_length = bs->num_sub_divisions * 2 * sizeof(uint32_t);

  // In all of these examples, we want both the cpu and gpu to have access to
  // the buffer in question. We use the array of agents below in the susequent
  // calls to hsa_amd_agents_allow_access() for this purpose.
  hsa_agent_t ag_list[2] = {bs->gpu_dev, bs->cpu_dev};

  err = hsa_amd_memory_pool_allocate(bs->cpu_pool, in_length, 0,
                                     reinterpret_cast<void**>(&bs->input));
  RET_IF_HSA_ERR(err);
  err = hsa_amd_agents_allow_access(2, ag_list, NULL, bs->input);
  RET_IF_HSA_ERR(err);
  (void)memset(bs->input, 0, in_length);

  err = hsa_amd_memory_pool_allocate(bs->cpu_pool, out_length, 0,
                                     reinterpret_cast<void**>(&bs->output));
  RET_IF_HSA_ERR(err);
  err = hsa_amd_agents_allow_access(2, ag_list, NULL, bs->output);
  RET_IF_HSA_ERR(err);
  (void)memset(bs->input, 0, in_length);

  err = hsa_amd_memory_pool_allocate(bs->cpu_pool, in_length, 0,
                                     reinterpret_cast<void**>(&bs->input_arr));
  RET_IF_HSA_ERR(err);
  err = hsa_amd_agents_allow_access(2, ag_list, NULL, bs->input_arr);
  RET_IF_HSA_ERR(err);
  (void)memset(bs->input, 0, in_length);

  err = hsa_amd_memory_pool_allocate(bs->cpu_pool, in_length, 0,
                               reinterpret_cast<void**>(&bs->input_arr_local));
  RET_IF_HSA_ERR(err);
  err = hsa_amd_agents_allow_access(2, ag_list, NULL, bs->input_arr_local);
  RET_IF_HSA_ERR(err);

  // Binary-search application specific code...
  // Initialize input buffer with random values in an increasing order
  uint32_t max = bs->length * 20;
  bs->input[0] = 0;

  uint32_t seed = (unsigned int)time(NULL);
  srand(seed);

  for (uint32_t i = 1; i < bs->length; ++i) {
    bs->input[i] = bs->input[i - 1] +
     static_cast<uint32_t>(max * rand_r(&seed) / static_cast<float>(RAND_MAX));
  }

// #define VERBOSE 1
#ifdef VERBOSE
  std::cout << "Input array values:" << std::endl;

  for (uint32_t i = 0; i < bs->length; ++i) {
    std::cout << "input[" << i << "] = " << bs->input[i] << " ";

    if (i % 4 == 0) {
      std::cout << std::endl;
    }
  }

  std::cout << std::endl;
#endif

  return err;
}

// The code in this function illustrates how to load a kernel from
// pre-compiled code. The goal is to get a handle that can be later
// used in an AQL packet and also to extract information about kernel
// that we will need. All of the information hand kernel handle will
// be saved to the BinarySearch structure. It will be used when we
// populate the AQL packet.
hsa_status_t LoadKernelFromObjFile(BinarySearch* bs) {
  hsa_status_t err;
  hsa_code_object_reader_t code_obj_rdr = {0};
  hsa_executable_t executable = {0};

  hsa_file_t file_handle = open(bs->kernel_file_name.c_str(), O_RDONLY);

  if (file_handle == -1) {
    char agent_name[64];
    err = hsa_agent_get_info(bs->gpu_dev, HSA_AGENT_INFO_NAME, agent_name);
    RET_IF_HSA_ERR(err);
    std::string fileName = std::string("./") + agent_name + "/" + bs->kernel_file_name;
    hsa_file_t file_handle = open(fileName.c_str(), O_RDONLY);
  }

  if (file_handle == -1) {
    std::cout << "failed to open " << bs->kernel_file_name.c_str() <<
              " at line " << __LINE__ << ", errno: " << errno << std::endl;
    return HSA_STATUS_ERROR;
  }

  err = hsa_code_object_reader_create_from_file(file_handle, &code_obj_rdr);
  close(file_handle);
  RET_IF_HSA_ERR(err);

  err = hsa_executable_create_alt(HSA_PROFILE_FULL,
                HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, NULL, &executable);
  RET_IF_HSA_ERR(err);

  err = hsa_executable_load_agent_code_object(executable, bs->gpu_dev,
        code_obj_rdr, NULL, NULL);
  RET_IF_HSA_ERR(err);

  err = hsa_executable_freeze(executable, NULL);
  RET_IF_HSA_ERR(err);

  hsa_executable_symbol_t kern_sym;
  err = hsa_executable_get_symbol(executable, NULL, bs->kernel_name.c_str(),
                                  bs->gpu_dev, 0, &kern_sym);
  RET_IF_HSA_ERR(err);

  err = hsa_executable_symbol_get_info(kern_sym,
                                    HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                                          &bs->kernel_object);
  RET_IF_HSA_ERR(err);

  err = hsa_executable_symbol_get_info(kern_sym,
                      HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                                                   &bs->private_segment_size);
  RET_IF_HSA_ERR(err);

  err = hsa_executable_symbol_get_info(kern_sym,
                        HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                                                     &bs->group_segment_size);
  RET_IF_HSA_ERR(err);

  // Remaining queries not supported on code object v3.
  err = hsa_executable_symbol_get_info(kern_sym,
                      HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
                                                           &bs->kernarg_size);
  RET_IF_HSA_ERR(err);

  err = hsa_executable_symbol_get_info(kern_sym,
                 HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT,
                                                          &bs->kernarg_align);
  RET_IF_HSA_ERR(err);
  assert(bs->kernarg_align >= 16 && "Reported kernarg size is too small.");
  bs->kernarg_align = (bs->kernarg_align == 0) ? 16 : bs->kernarg_align;

  return err;
}

// This function shows how to do an asynchronous copy. We have to create a
// signal and use the signal to notify us when the copy has completed.
hsa_status_t AgentMemcpy(void* dst, const void* src,
                         size_t size, hsa_agent_t dst_ag, hsa_agent_t src_ag) {
  hsa_signal_t s;
  hsa_status_t err;

  err = hsa_signal_create(1, 0, NULL, &s);
  RET_IF_HSA_ERR(err);

  err = hsa_amd_memory_async_copy(dst, dst_ag, src, src_ag, size, 0, NULL, s);
  RET_IF_HSA_ERR(err);

  if (hsa_signal_wait_scacquire(s, HSA_SIGNAL_CONDITION_LT, 1,
                                UINT64_MAX, HSA_WAIT_STATE_BLOCKED) != 0) {
    err = HSA_STATUS_ERROR;
    std::cout << "Async copy signal error" << std::endl;

    RET_IF_HSA_ERR(err);
  }

  err = hsa_signal_destroy(s);

  RET_IF_HSA_ERR(err);

  return err;
}

// AlignDown and AlignUp are 2 utility functions we use to find an aligned
// boundary either below or above a given value (address). The function will
// return a value that has the specified alignment.
static intptr_t
AlignDown(intptr_t value, size_t alignment) {
  assert(alignment != 0 && "Zero alignment");
  return (intptr_t) (value & ~(alignment - 1));
}
static void*
AlignUp(void* value, size_t alignment) {
  return reinterpret_cast<void*>(AlignDown((uintptr_t)
           (reinterpret_cast<uintptr_t>(value) + alignment - 1), alignment));
}

// This function populates the AQL patch with the information
// we have collected and stored in the BinarySearch structure thus far.
void PopulateAQLPacket(BinarySearch const* bs,
                       hsa_kernel_dispatch_packet_t* aql) {
  aql->header = 0;  // Dummy val. for now. Set this right before doorbell ring
  aql->setup = 1;
  aql->workgroup_size_x = bs->work_group_size;
  aql->workgroup_size_y = 1;
  aql->workgroup_size_z = 1;
  aql->grid_size_x = bs->work_grid_size;
  aql->grid_size_y = 1;
  aql->grid_size_z = 1;
  aql->private_segment_size = bs->private_segment_size;
  aql->group_segment_size = bs->group_segment_size;
  aql->kernel_object = bs->kernel_object;
  aql->kernarg_address = bs->kern_arg_address;
  aql->completion_signal = bs->signal;

  return;
}
/*
 * Write everything in the provided AQL packet to the queue except the first 32
 * bits which include the header and setup fields. That should be done
 * last.
 */
void WriteAQLToQueue(hsa_kernel_dispatch_packet_t const* in_aql,
                     hsa_queue_t* q) {
  void* queue_base = q->base_address;
  const uint32_t queue_mask = q->size - 1;
  uint64_t que_idx = hsa_queue_add_write_index_relaxed(q, 1);

  hsa_kernel_dispatch_packet_t* queue_aql_packet;

  queue_aql_packet =
    &(reinterpret_cast<hsa_kernel_dispatch_packet_t*>(queue_base))
    [que_idx & queue_mask];

  queue_aql_packet->workgroup_size_x = in_aql->workgroup_size_x;
  queue_aql_packet->workgroup_size_y = in_aql->workgroup_size_y;
  queue_aql_packet->workgroup_size_z = in_aql->workgroup_size_z;
  queue_aql_packet->grid_size_x = in_aql->grid_size_x;
  queue_aql_packet->grid_size_y = in_aql->grid_size_y;
  queue_aql_packet->grid_size_z = in_aql->grid_size_z;
  queue_aql_packet->private_segment_size = in_aql->private_segment_size;
  queue_aql_packet->group_segment_size = in_aql->group_segment_size;
  queue_aql_packet->kernel_object = in_aql->kernel_object;
  queue_aql_packet->kernarg_address = in_aql->kernarg_address;
  queue_aql_packet->completion_signal = in_aql->completion_signal;
}

// This function allocates memory from the kern_arg pool we already found, and
// then sets the argument values needed by the kernel code.
hsa_status_t AllocAndSetKernArgs(BinarySearch* bs, void* args,
                                 size_t arg_size, void** aql_buf_ptr) {
  void* kern_arg_buf = nullptr;
  hsa_status_t err;
  size_t buf_size;
  size_t req_align;

  // The kernel code must be written to memory at the correct alignment. We
  // already queried the executable to get the correct alignment, which is
  // stored in bs->kernarg_align. In case the memory returned from
  // hsa_amd_memory_pool is not of the correct alignment, we request a little
  // more than what we need in case we need to adjust.
  req_align = bs->kernarg_align;
  // Allocate enough extra space for alignment adjustments if ncessary
  buf_size = arg_size + (req_align << 1);

  err = hsa_amd_memory_pool_allocate(bs->kern_arg_pool, buf_size, 0,
                                     reinterpret_cast<void**>(&kern_arg_buf));
  RET_IF_HSA_ERR(err);

  // Address of the allocated buffer
  bs->kern_arg_buffer = kern_arg_buf;

  // Addr. of kern arg start.
  bs->kern_arg_address = AlignUp(kern_arg_buf, req_align);

  assert(arg_size >= bs->kernarg_size);
  assert(((uintptr_t)bs->kern_arg_address + arg_size) <
         ((uintptr_t)bs->kern_arg_buffer + buf_size));

  (void)memcpy(bs->kern_arg_address, args, arg_size);
  RET_IF_HSA_ERR(err);

  // Make sure both the CPU and GPU can access the kernel arguments
  hsa_agent_t ag_list[2] = {bs->gpu_dev, bs->cpu_dev};
  err = hsa_amd_agents_allow_access(2, ag_list, NULL, bs->kern_arg_buffer);
  RET_IF_HSA_ERR(err);

  // Save this info in our BinarySearch structure for later.
  *aql_buf_ptr = bs->kern_arg_address;

  return HSA_STATUS_SUCCESS;
}

// This wrapper atomically writes the provided header and setup to the
// provided AQL packet. The provided AQL packet address should be in the
// queue memory space.
inline void AtomicSetPacketHeader(uint16_t header, uint16_t setup,
                                  hsa_kernel_dispatch_packet_t* queue_packet) {
  __atomic_store_n(reinterpret_cast<uint32_t*>(queue_packet),
                   header | (setup << 16), __ATOMIC_RELEASE);
}

// Once all the required data for kernel execution is collected (in this
// application it is stored in the BinarySearch structure) we can put it in
// an AQL packet and ring the queue door bell to tell the command processor to
// execute it.
hsa_status_t Run(BinarySearch* bs) {
  hsa_status_t err;

  std::cout << "Executing kernel " << bs->kernel_name << std::endl;

  // Adjust the size of workgroup
  // This is mostly application specific.
  if (bs->work_group_size > 64) {
    bs->work_group_size = 64;
    bs->num_sub_divisions = bs->length / bs->work_group_size;
  }
  if (bs->num_sub_divisions < bs->work_group_size) {
    bs->num_sub_divisions = bs->work_group_size;
  }

  bs->work_grid_size = bs->num_sub_divisions;

  // Explanation of BinarySearch algorithm.
  /*
   * Since a plain binary search on the GPU would not achieve much benefit
   * over the GPU we are doing an N'ary search. We split the array into N
   * segments every pass and therefore get log (base N) passes instead of log
   * (base 2) passes.
   *
   * In every pass, only the thread that can potentially have the element we
   * are looking for writes to the output array. For ex: if we are looking to
   * find 4567 in the array and every thread is searching over a segment of
   * 1000 values and the input array is 1, 2, 3, 4,... then the first thread
   * is searching in 1 to 1000, the second one from 1001 to 2000, etc. The
   * first one does not write to the output. The second one doesn't either.
   * The fifth one however is from 4001 to 5000. So it can potentially have
   * the element 4567 which lies between them.
   *
   * This particular thread writes to the output the lower bound, upper bound
   * and whether the element equals the lower bound element. So, it would be
   * 4001, 5000, 0
   *
   * The next pass would subdivide 4001 to 5000 into smaller segments and
   * continue the same process from there.
   *
   * When a pass returns 1 in the third element, it means the element has been
   * found and we can stop executing the kernel. If the element is not found,
   * then the execution stops after looking at segment of size 1.
   */

  uint32_t global_lower_bound = 0;
  uint32_t global_upper_bound = bs->length - 1;
  uint32_t sub_div_size = (global_upper_bound - global_lower_bound + 1) /
                          bs->num_sub_divisions;

  if ((bs->input[0] > bs->find_me) ||
      (bs->input[bs->length - 1] < bs->find_me)) {
    bs->output[0] = 0;
    bs->output[1] = bs->length - 1;
    bs->output[2] = 0;
    std::cout << "Returning too early" << std::endl;
    return HSA_STATUS_SUCCESS;
  }

  bs->output[3] = 1;

  // Setup the kernel args
  // See the meta-data for the compiled OpenCL kernel code to ascertain
  // the sizes, padding and alignment required for kernel arguments.
  // This can be seen by executing
  // $ amdgcn-amd-amdhsa-readelf -aw ./binary_search_kernels.hsaco
  // The kernel code will expect the following arguments aligned as shown.
  typedef uint32_t uint2[2];
  typedef uint32_t uint4[4];
  struct __attribute__((aligned(16))) local_args_t {
    uint4* outputArray;
    uint2*  sortedArray;
    uint32_t findMe;
    uint32_t pad;
    uint64_t global_offset_x;
    uint64_t global_offset_y;
    uint64_t global_offset_z;
    uint64_t printf_buffer;
    uint64_t default_queue;
    uint64_t completion_action;
  } local_args;

  local_args.outputArray = reinterpret_cast<uint4*>(bs->output);
  local_args.sortedArray = reinterpret_cast<uint2*>(bs->input_arr_local);
  local_args.findMe = bs->find_me;
  local_args.global_offset_x = 0;
  local_args.global_offset_y = 0;
  local_args.global_offset_z = 0;
  local_args.printf_buffer = 0;
  local_args.default_queue = 0;
  local_args.completion_action = 0;

  // Copy the kernel args structure into kernel arg memory
  err = AllocAndSetKernArgs(bs, &local_args, sizeof(local_args),
                            &bs->kern_arg_address);
  RET_IF_HSA_ERR(err);

  // Populate an AQL packet with the info we've gathered
  hsa_kernel_dispatch_packet_t aql;
  PopulateAQLPacket(bs, &aql);

  uint32_t in_length = bs->num_sub_divisions * 2 * sizeof(uint32_t);

  while ((sub_div_size > 1) && (bs->output[3] != 0)) {
    for (uint32_t i = 0 ; i < bs->num_sub_divisions; i++) {
      int idx1 = i * sub_div_size;
      int idx2 = ((i + 1) * sub_div_size) - 1;
      bs->input_arr[2 * i] = bs->input[idx1];
      bs->input_arr[2 * i + 1] = bs->input[idx2];
    }

    // Copy kernel parameter from system memory to local memory
    err = AgentMemcpy(reinterpret_cast<uint8_t*>(bs->input_arr_local),
                      reinterpret_cast<uint8_t*>(bs->input_arr),
                                        in_length, bs->gpu_dev, bs->cpu_dev);

    RET_IF_HSA_ERR(err);

    // Reset output buffer to zero
    bs->output[3] = 0;

    // Dispatch kernel with global work size, work group size with ONE dimesion
    // and wait for kernel to complete

    // Compute the write index of queue and copy Aql packet into it
    uint64_t que_idx = hsa_queue_load_write_index_relaxed(bs->queue);

    const uint32_t mask = bs->queue->size - 1;

    // This function simply copies the data we've collected so far into our
    // local AQL packet, except the the setup and header fields.
    WriteAQLToQueue(&aql, bs->queue);

    uint32_t aql_header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
    aql_header |= HSA_FENCE_SCOPE_SYSTEM <<
                  HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
    aql_header |= HSA_FENCE_SCOPE_SYSTEM <<
                  HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;

    // Set the packet's type, acquire and release fences. This should be done
    // atomically after all the other fields have been set, using release
    // memory ordering to ensure all the fields are set when the door bell
    // signal is activated.
    void* q_base = bs->queue->base_address;

    AtomicSetPacketHeader(aql_header, aql.setup,
                      &(reinterpret_cast<hsa_kernel_dispatch_packet_t*>
                                                   (q_base))[que_idx & mask]);

    // Increment the write index and ring the doorbell to dispatch kernel.
    hsa_queue_store_write_index_relaxed(bs->queue, (que_idx + 1));
    hsa_signal_store_relaxed(bs->queue->doorbell_signal, que_idx);

    // Wait on the dispatch signal until the kernel is finished.
    // Modify the wait condition to HSA_WAIT_STATE_ACTIVE (instead of
    // HSA_WAIT_STATE_BLOCKED) if polling is needed instead of blocking, as we
    // have below.
    // The call below will block until the condition is met. Below we have said
    // the condition is that the signal value (initiailzed to 1) associated with
    // the queue is less than 1. When the kernel associated with the queued AQL
    // packet has completed execution, the signal value is automatically
    // decremented by the packet processor.
    hsa_signal_value_t value = hsa_signal_wait_scacquire(bs->signal,
                               HSA_SIGNAL_CONDITION_LT, 1,
                               UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

    // value should be 0, or we timed-out
    if (value) {
      std::cout << "Timed out waiting for kernel to complete?" << std::endl;
      RET_IF_HSA_ERR(HSA_STATUS_ERROR);
    }

    // Reset the signal to its initial value for the next iteration
    hsa_signal_store_screlease(bs->signal, 1);

    // Binary search algorithm stuff...
    global_lower_bound = bs->output[0] * sub_div_size;
    global_upper_bound = global_lower_bound + sub_div_size - 1;
    sub_div_size = (global_upper_bound - global_lower_bound + 1) /
                   bs->num_sub_divisions;
  }

  uint32_t element_index = UINT_MAX;

  for (uint32_t i = global_lower_bound; i <= global_upper_bound; i++) {
    if (bs->input[i] == bs->find_me) {
      element_index = i;
      bs->output[0] = i;
      bs->output[1] = i + 1;
      bs->output[2] = 1;
      break;
    }

    // Element is not found in region specified
    // by global lower bound to global upper bound
    bs->output[2] = 0;
  }

  uint32_t is_elem_found = bs->output[2];

  std::cout << "Lower bound = " << global_lower_bound << std::endl;
  std::cout << "Upper bound = " << global_upper_bound << std::endl;
  std::cout << "Element search for = " << bs->find_me << std::endl;


  if (is_elem_found == 1) {
    std::cout << "Element found at index " << element_index << std::endl;
  } else {
    std::cout << "Element value " << bs->find_me << " not found" << std::endl;
  }

  return HSA_STATUS_SUCCESS;
}

// Release all the RocR resources we have acquired in this application.
hsa_status_t CleanUp(BinarySearch* bs) {
  hsa_status_t err;

  err = hsa_amd_memory_pool_free(bs->input);
  RET_IF_HSA_ERR(err);

  err = hsa_amd_memory_pool_free(bs->output);
  RET_IF_HSA_ERR(err);

  err = hsa_amd_memory_pool_free(bs->input_arr);
  RET_IF_HSA_ERR(err);

  err = hsa_amd_memory_pool_free(bs->kern_arg_buffer);
  RET_IF_HSA_ERR(err);

  err = hsa_queue_destroy(bs->queue);
  RET_IF_HSA_ERR(err);

  err = hsa_signal_destroy(bs->signal);
  RET_IF_HSA_ERR(err);

  err = hsa_shut_down();
  RET_IF_HSA_ERR(err);

  return HSA_STATUS_SUCCESS;
}

int main(int argc, char* argv[]) {
  // This BinarySearch structure (bs) below holds all of the appl. specific
  // info we need to run the sample. This includes algorithm specific
  // information as well as handles to RocR/HSA objects.

  // The basic structure of this sample is to fill in this structure with the
  // required RocR/HSA handles to RocR resources (e.g., agents, memory pools,
  // queues, etc.) and then dispatch the packets to the queue, and examine the
  // output.

  BinarySearch bs;
  hsa_status_t err;

  // Set some working values specific to this application
  InitializeBinarySearch(&bs);

  // hsa_init() initializes internal data structures and causes devices
  // (agents), memory pools and other resources to be discovered.
  err = hsa_init();
  RET_IF_HSA_ERR(err);

  // Find the agents needed for the sample
  err = FindDevices(&bs);
  RET_IF_HSA_ERR(err);

  // Create the completion signal used when dispatching a packet
  err = hsa_signal_create(1, 0, NULL, &bs.signal);
  RET_IF_HSA_ERR(err);

  // Create a queue to submit our binary search AQL packets
  err = hsa_queue_create(bs.gpu_dev, 128, HSA_QUEUE_TYPE_MULTI, NULL, NULL,
                         UINT32_MAX, UINT32_MAX, &bs.queue);
  RET_IF_HSA_ERR(err);

  // Find the HSA memory pools we need to run this sample
  err = FindPools(&bs);
  RET_IF_HSA_ERR(err);

  // Allocate memory from the correct memory pool, and initialize them as
  // neeeded for the algorihm.
  err = AllocateAndInitBuffers(&bs);
  RET_IF_HSA_ERR(err);

  // Create a kernel object from the pre-compiled kernel, and read some
  // attributes associated with the kernel that we will need.
  err = LoadKernelFromObjFile(&bs);
  RET_IF_HSA_ERR(err);

  // Fill in the AQL packet, assign the kernel arguments, enqueue the packet,
  // "ring" the doorbell, and wait for completion.
  err = Run(&bs);
  RET_IF_HSA_ERR(err);

  // Release all the RocR resources we've acquired and shutdown HSA.
  err = CleanUp(&bs);

  return 0;
}


#undef RET_IF_HSA_ERR
