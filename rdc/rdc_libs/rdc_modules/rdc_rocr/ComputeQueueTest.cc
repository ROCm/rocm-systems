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
#include "rdc_modules/rdc_rocr/ComputeQueueTest.h"

#include <assert.h>
#include <fcntl.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include <algorithm>
#include <climits>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/rdc_common.h"
#include "rdc_modules/rdc_rocr/base_rocr_utils.h"
#include "rdc_modules/rdc_rocr/common.h"

namespace amd {
namespace rdc {

static const uint32_t kNumBufferElements = 256;

ComputeQueueTest::ComputeQueueTest(uint32_t gpu_index) : TestBase(gpu_index) {
  set_num_iteration(10);  // Number of iterations to execute of the main test;
                          // This is a default value which can be overridden
                          // on the command line.
  set_title("ComputeQueue Test");
  set_description("This test will run binary search compute task via AQL.");
}

ComputeQueueTest::~ComputeQueueTest(void) {}

// Any 1-time setup involving member variables used in the rest of the test
// should be done here.
hsa_status_t ComputeQueueTest::SetUp(void) {
  hsa_status_t err = HSA_STATUS_SUCCESS;

  TestBase::SetUp();

  err = SetDefaultAgents(this);
  if (err != HSA_STATUS_SUCCESS) return err;

  err = SetPoolsTypical(this);
  return err;
}

void ComputeQueueTest::Run(void) {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!CheckProfile(this)) {
    return;
  }

  TestBase::Run();
}

void ComputeQueueTest::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void ComputeQueueTest::DisplayResults(void) const {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!CheckProfile(this)) {
    return;
  }

  return;
}

void ComputeQueueTest::Close() {
  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

static const uint32_t kBinarySearchLength = 512;
static const uint32_t kBinarySearchFindMe = 108;
static const uint32_t kWorkGroupSize = 256;

void ComputeQueueTest::InitializeBinarySearch(BinarySearch* bs) {
  bs->kernel_file_name = "binary_search_kernels.hsaco";
  bs->kernel_name = "binarySearch.kd";
  bs->length = kBinarySearchLength;
  bs->find_me = kBinarySearchFindMe;
  bs->work_group_size = kWorkGroupSize;
  bs->num_sub_divisions = bs->length / bs->work_group_size;
}

// This function shows how to do an asynchronous copy. We have to create a
// signal and use the signal to notify us when the copy has completed.
hsa_status_t ComputeQueueTest::AgentMemcpy(void* dst, const void* src, size_t size,
                                           hsa_agent_t dst_ag, hsa_agent_t src_ag) {
  hsa_signal_t s;
  hsa_status_t err;

  err = hsa_signal_create(1, 0, NULL, &s);
  throw_if_error(err);

  err = hsa_amd_memory_async_copy(dst, dst_ag, src, src_ag, size, 0, NULL, s);
  throw_if_error(err);

  if (hsa_signal_wait_scacquire(s, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
                                HSA_WAIT_STATE_BLOCKED) != 0) {
    err = HSA_STATUS_ERROR;
    RDC_LOG(RDC_ERROR, "Async copy signal error");

    throw_if_error(err);
  }

  err = hsa_signal_destroy(s);

  throw_if_error(err);

  return err;
}

hsa_status_t ComputeQueueTest::FindPools(BinarySearch* bs) {
  hsa_status_t err;

  err = hsa_amd_agent_iterate_memory_pools(bs->cpu_dev, FindStandardPool, &bs->cpu_pool);

  if (err != HSA_STATUS_INFO_BREAK) {
    return HSA_STATUS_ERROR;
  }

  err = hsa_amd_agent_iterate_memory_pools(bs->gpu_dev, FindStandardPool, &bs->gpu_pool);

  if (err != HSA_STATUS_INFO_BREAK) {
    return HSA_STATUS_ERROR;
  }

  err = hsa_amd_agent_iterate_memory_pools(bs->cpu_dev, FindKernArgPool, &bs->kern_arg_pool);

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
hsa_status_t ComputeQueueTest::AllocateAndInitBuffers(BinarySearch* bs) {
  hsa_status_t err;
  uint32_t out_length = 4 * sizeof(uint32_t);
  uint32_t in_length = bs->num_sub_divisions * 2 * sizeof(uint32_t);

  // In all of these examples, we want both the cpu and gpu to have access to
  // the buffer in question. We use the array of agents below in the susequent
  // calls to hsa_amd_agents_allow_access() for this purpose.
  hsa_agent_t ag_list[2] = {bs->gpu_dev, bs->cpu_dev};

  err = hsa_amd_memory_pool_allocate(bs->cpu_pool, in_length, 0,
                                     reinterpret_cast<void**>(&bs->input));
  throw_if_error(err);
  err = hsa_amd_agents_allow_access(2, ag_list, NULL, bs->input);
  throw_if_error(err);
  (void)memset(bs->input, 0, in_length);

  err = hsa_amd_memory_pool_allocate(bs->cpu_pool, out_length, 0,
                                     reinterpret_cast<void**>(&bs->output));
  throw_if_error(err);
  err = hsa_amd_agents_allow_access(2, ag_list, NULL, bs->output);
  throw_if_error(err);
  (void)memset(bs->input, 0, in_length);

  err = hsa_amd_memory_pool_allocate(bs->cpu_pool, in_length, 0,
                                     reinterpret_cast<void**>(&bs->input_arr));
  throw_if_error(err);
  err = hsa_amd_agents_allow_access(2, ag_list, NULL, bs->input_arr);
  throw_if_error(err);
  (void)memset(bs->input, 0, in_length);

  err = hsa_amd_memory_pool_allocate(bs->cpu_pool, in_length, 0,
                                     reinterpret_cast<void**>(&bs->input_arr_local));
  throw_if_error(err);
  err = hsa_amd_agents_allow_access(2, ag_list, NULL, bs->input_arr_local);
  throw_if_error(err);

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

  return err;
}

// The code in this function illustrates how to load a kernel from
// pre-compiled code. The goal is to get a handle that can be later
// used in an AQL packet and also to extract information about kernel
// that we will need. All of the information hand kernel handle will
// be saved to the BinarySearch structure. It will be used when we
// populate the AQL packet.
hsa_status_t ComputeQueueTest::LoadKernelFromObjFile(BinarySearch* bs) {
  hsa_status_t err;
  char agent_name[512];
  hsa_code_object_reader_t code_obj_rdr = {0};
  hsa_executable_t executable = {0};

  err = hsa_agent_get_info(bs->gpu_dev, HSA_AGENT_INFO_NAME, agent_name);
  throw_if_error(err);
  std::string kernel_file = search_hsaco_full_path(bs->kernel_file_name.c_str(), agent_name);
  if (kernel_file == "") {
    RDC_LOG(RDC_ERROR, "failed to open " << bs->kernel_file_name.c_str() << " at line " << __LINE__
                                         << ", errno: " << errno);
    std::string msg("fail to open ");
    msg += bs->kernel_file_name;
    throw_if_skip(msg);
    return HSA_STATUS_ERROR;
  }

  hsa_file_t file_handle = open(kernel_file.c_str(), O_RDONLY);
  if (file_handle == -1) {
    RDC_LOG(RDC_ERROR, "failed to open " << bs->kernel_file_name.c_str() << " at line " << __LINE__
                                         << ", errno: " << errno);
    return HSA_STATUS_ERROR;
  }

  err = hsa_code_object_reader_create_from_file(file_handle, &code_obj_rdr);
  throw_if_error(err);
  close(file_handle);

  err = hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, NULL,
                                  &executable);
  throw_if_error(err);

  err = hsa_executable_load_agent_code_object(executable, bs->gpu_dev, code_obj_rdr, NULL, NULL);
  throw_if_error(err);

  err = hsa_executable_freeze(executable, NULL);
  throw_if_error(err);

  hsa_executable_symbol_t kern_sym;
  err = hsa_executable_get_symbol(executable, NULL, bs->kernel_name.c_str(), bs->gpu_dev, 0,
                                  &kern_sym);
  throw_if_error(err);

  err = hsa_executable_symbol_get_info(kern_sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                       &bs->kernel_object);
  throw_if_error(err);

  err = hsa_executable_symbol_get_info(
      kern_sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &bs->private_segment_size);
  throw_if_error(err);

  err = hsa_executable_symbol_get_info(
      kern_sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &bs->group_segment_size);
  throw_if_error(err);

  // Remaining queries not supported on code object v3.
  err = hsa_executable_symbol_get_info(
      kern_sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &bs->kernarg_size);
  throw_if_error(err);

  err = hsa_executable_symbol_get_info(
      kern_sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT, &bs->kernarg_align);
  throw_if_error(err);
  assert(bs->kernarg_align >= 16 && "Reported kernarg size is too small.");
  bs->kernarg_align = (bs->kernarg_align == 0) ? 16 : bs->kernarg_align;

  return err;
}

// This function populates the AQL patch with the information
// we have collected and stored in the BinarySearch structure thus far.
void ComputeQueueTest::PopulateAQLPacket(BinarySearch const* bs,
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
}

void ComputeQueueTest::WriteAQLToQueue(hsa_kernel_dispatch_packet_t const* in_aql, hsa_queue_t* q) {
  void* queue_base = q->base_address;
  const uint32_t queue_mask = q->size - 1;
  uint64_t que_idx = hsa_queue_add_write_index_relaxed(q, 1);

  hsa_kernel_dispatch_packet_t* queue_aql_packet;

  queue_aql_packet =
      &(reinterpret_cast<hsa_kernel_dispatch_packet_t*>(queue_base))[que_idx & queue_mask];

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
hsa_status_t ComputeQueueTest::AllocAndSetKernArgs(BinarySearch* bs, void* args, size_t arg_size,
                                                   void** aql_buf_ptr) {
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
  throw_if_error(err);

  // Address of the allocated buffer
  bs->kern_arg_buffer = kern_arg_buf;

  // Addr. of kern arg start.
  bs->kern_arg_address = AlignUp(kern_arg_buf, req_align);

  assert(arg_size >= bs->kernarg_size);
  assert(((uintptr_t)bs->kern_arg_address + arg_size) <
         ((uintptr_t)bs->kern_arg_buffer + buf_size));

  (void)memcpy(bs->kern_arg_address, args, arg_size);
  throw_if_error(err);

  // Make sure both the CPU and GPU can access the kernel arguments
  hsa_agent_t ag_list[2] = {bs->gpu_dev, bs->cpu_dev};
  err = hsa_amd_agents_allow_access(2, ag_list, NULL, bs->kern_arg_buffer);
  throw_if_error(err);

  // Save this info in our BinarySearch structure for later.
  *aql_buf_ptr = bs->kern_arg_address;

  return HSA_STATUS_SUCCESS;
}

// Once all the required data for kernel execution is collected (in this
// application it is stored in the BinarySearch structure) we can put it in
// an AQL packet and ring the queue door bell to tell the command processor to
// execute it.
hsa_status_t ComputeQueueTest::Run(BinarySearch* bs) {
  hsa_status_t err;
  RDC_LOG(RDC_DEBUG, "Executing kernel " << bs->kernel_name);

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
  uint32_t sub_div_size = (global_upper_bound - global_lower_bound + 1) / bs->num_sub_divisions;

  if ((bs->input[0] > bs->find_me) || (bs->input[bs->length - 1] < bs->find_me)) {
    bs->output[0] = 0;
    bs->output[1] = bs->length - 1;
    bs->output[2] = 0;
    RDC_LOG(RDC_DEBUG, "Returning too early");
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
    uint2* sortedArray;
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
  err = AllocAndSetKernArgs(bs, &local_args, sizeof(local_args), &bs->kern_arg_address);
  throw_if_error(err);

  // Populate an AQL packet with the info we've gathered
  hsa_kernel_dispatch_packet_t aql;
  PopulateAQLPacket(bs, &aql);

  uint32_t in_length = bs->num_sub_divisions * 2 * sizeof(uint32_t);

  while ((sub_div_size > 1) && (bs->output[3] != 0)) {
    for (uint32_t i = 0; i < bs->num_sub_divisions; i++) {
      int idx1 = i * sub_div_size;
      int idx2 = ((i + 1) * sub_div_size) - 1;
      bs->input_arr[2 * i] = bs->input[idx1];
      bs->input_arr[2 * i + 1] = bs->input[idx2];
    }

    // Copy kernel parameter from system memory to local memory
    err =
        AgentMemcpy(reinterpret_cast<uint8_t*>(bs->input_arr_local),
                    reinterpret_cast<uint8_t*>(bs->input_arr), in_length, bs->gpu_dev, bs->cpu_dev);

    throw_if_error(err);

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
    aql_header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
    aql_header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;

    // Set the packet's type, acquire and release fences. This should be done
    // atomically after all the other fields have been set, using release
    // memory ordering to ensure all the fields are set when the door bell
    // signal is activated.
    void* q_base = bs->queue->base_address;

    AtomicSetPacketHeader(
        aql_header, aql.setup,
        &(reinterpret_cast<hsa_kernel_dispatch_packet_t*>(q_base))[que_idx & mask]);

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
    hsa_signal_value_t value = hsa_signal_wait_scacquire(bs->signal, HSA_SIGNAL_CONDITION_LT, 1,
                                                         UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

    // value should be 0, or we timed-out
    if (value) {
      RDC_LOG(RDC_ERROR, "Timed out waiting for kernel to complete?");
      throw_if_error(HSA_STATUS_ERROR);
    }

    // Reset the signal to its initial value for the next iteration
    hsa_signal_store_screlease(bs->signal, 1);

    // Binary search algorithm stuff...
    global_lower_bound = bs->output[0] * sub_div_size;
    global_upper_bound = global_lower_bound + sub_div_size - 1;
    sub_div_size = (global_upper_bound - global_lower_bound + 1) / bs->num_sub_divisions;
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
  RDC_LOG(RDC_DEBUG, "Lower bound = " << global_lower_bound);
  RDC_LOG(RDC_DEBUG, "Upper bound = " << global_upper_bound);
  RDC_LOG(RDC_DEBUG, "Element search for = " << bs->find_me);

  if (is_elem_found == 1) {
    RDC_LOG(RDC_DEBUG, "Element found at index " << element_index);
  } else {
    RDC_LOG(RDC_DEBUG, "Element value " << bs->find_me << " not found");
  }

  return HSA_STATUS_SUCCESS;
}

// Release all the RocR resources we have acquired in this application.
hsa_status_t ComputeQueueTest::CleanUp(BinarySearch* bs) {
  hsa_status_t err = HSA_STATUS_SUCCESS;

  err = hsa_amd_memory_pool_free(bs->input);

  err = hsa_amd_memory_pool_free(bs->output);

  err = hsa_amd_memory_pool_free(bs->input_arr);

  err = hsa_amd_memory_pool_free(bs->kern_arg_buffer);

  err = hsa_queue_destroy(bs->queue);

  err = hsa_signal_destroy(bs->signal);

  // shutdown will be called at destructor
  // err = hsa_shut_down();

  return err;
}

hsa_status_t ComputeQueueTest::RunBinarySearchTest(void) {
  BinarySearch bs;
  hsa_status_t err;

  InitializeBinarySearch(&bs);

  hsa_agent_t current_gpu;
  err = get_agent_by_gpu_index(gpu_index_, &current_gpu);
  throw_if_error(err, "Get agent by GPU index fail.");
  bs.gpu_dev.handle = current_gpu.handle;

  // find all cpu agents
  std::vector<hsa_agent_t> cpus;
  err = hsa_iterate_agents(IterateCPUAgents, &cpus);
  throw_if_error(err);
  bs.cpu_dev.handle = cpus[0].handle;

  err = hsa_signal_create(1, 0, NULL, &bs.signal);
  throw_if_error(err, "Fail to create signal.");

  err = hsa_queue_create(bs.gpu_dev, 128, HSA_QUEUE_TYPE_MULTI, NULL, NULL, UINT32_MAX, UINT32_MAX,
                         &bs.queue);
  throw_if_error(err, "Fail to create queue.");

  err = FindPools(&bs);
  throw_if_error(err, "Fail to find pools.");

  // Allocate memory from the correct memory pool, and initialize them as
  // neeeded for the algorihm.
  err = AllocateAndInitBuffers(&bs);
  throw_if_error(err, "Allocate and initBuffers fail.");

  err = LoadKernelFromObjFile(&bs);
  throw_if_error(err, "Load kernel from Object file fail.");

  err = Run(&bs);
  throw_if_error(err, "Run binary search fail.");

  CleanUp(&bs);

  gpu_info_ += "Run binary search task on GPU ";
  gpu_info_ += std::to_string(gpu_index_);
  gpu_info_ += " Pass.";

  return err;
}

}  // namespace rdc
}  // namespace amd
