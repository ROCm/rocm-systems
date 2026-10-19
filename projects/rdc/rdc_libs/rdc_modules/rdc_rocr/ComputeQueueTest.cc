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

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <string>

#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/rdc_common.h"
#include "rdc_modules/rdc_rocr/base_rocr_utils.h"
#include "rdc_modules/rdc_rocr/common.h"

namespace amd {
namespace rdc {

ComputeQueueTest::ComputeQueueTest(uint32_t gpu_index, uint32_t node_id, GpuOutcome* outcome)
    : TestBase(gpu_index, node_id, outcome) {}

ComputeQueueTest::~ComputeQueueTest(void) {}

static const uint32_t kBinarySearchLength = 512;
static const uint32_t kBinarySearchFindMe = 108;
static const uint32_t kWorkGroupSize = 256;

// The largest work group this search dispatches.
static const uint32_t kMaxWorkGroupSize = 64;

// What output[0] holds before each dispatch: a kernel that writes nothing leaves it there, and
// it is outside every range the check accepts below.
static const uint32_t kNoSubDivision = UINT32_MAX;

// What the check says of a kernel that answered with a sub-division outside the input.
static const char kIndexOutOfRange[] = "The kernel returned an index out of range.";

// The arguments the search kernel reads, in the order and at the offsets its metadata gives
// (amdgcn-amd-amdhsa-readelf -aw ./binary_search_kernels.hsaco). They are written into the
// kernel's argument segment one member at a time, at the offsets below, so that `pad` and the
// padding after the members stay the zeros the segment was cleared with.
typedef uint32_t search_uint2_t[2];
typedef uint32_t search_uint4_t[4];
struct __attribute__((aligned(16))) search_args_t {
  search_uint4_t* outputArray;
  search_uint2_t* sortedArray;
  uint32_t findMe;
  uint32_t pad;
  uint64_t global_offset_x;
  uint64_t global_offset_y;
  uint64_t global_offset_z;
  uint64_t printf_buffer;
  uint64_t default_queue;
  uint64_t completion_action;
};

// The bytes of arguments the kernel is given, without the padding the structure's own alignment
// adds after them: a kernel whose argument segment is shorter than this cannot be dispatched.
static const size_t kSearchArgumentBytes =
    offsetof(search_args_t, completion_action) + sizeof(uint64_t);

void ComputeQueueTest::InitializeBinarySearch(BinarySearch* bs) {
  bs->kernel_file_name = "binary_search_kernels.hsaco";
  bs->kernel_name = "binarySearch.kd";
  bs->length = kBinarySearchLength;
  bs->find_me = kBinarySearchFindMe;
  bs->work_group_size = kWorkGroupSize;

  // The run's work group and its sub-divisions, decided here because they size the buffers the
  // check allocates: a work group is at most kMaxWorkGroupSize work items, and the search needs
  // at least one sub-division per work item.
  if (bs->work_group_size > kMaxWorkGroupSize) {
    bs->work_group_size = kMaxWorkGroupSize;
  }
  bs->num_sub_divisions = bs->length / bs->work_group_size;
  if (bs->num_sub_divisions < bs->work_group_size) {
    bs->num_sub_divisions = bs->work_group_size;
  }
  bs->work_grid_size = bs->num_sub_divisions;
}

// This function shows how to do an asynchronous copy. We have to create a
// signal and use the signal to notify us when the copy has completed.
void ComputeQueueTest::AgentMemcpy(void* dst, const void* src, size_t size, hsa_agent_t dst_ag,
                                   hsa_agent_t src_ag) {
  hsa_signal_t handle = {0};
  hsa_status_t err;

  // The hint the wait below carries is read before the signal and the copy are made: a
  // frequency ROCr cannot report, or one of 0, is this GPU's failure with nothing in flight,
  // where a throw during the copy would unwind over the buffer the copy writes into.
  (void)wait_hint();

  // The copy's signal is released on every path out of this function, a failed copy among them.
  SignalHolder copy_signal(release_context());
  err = hsa_signal_create(1, 0, NULL, &handle);
  if (err == HSA_STATUS_SUCCESS) {
    copy_signal.reset(handle);
  }
  throw_if_error(err, "Fail to create the memory copy's signal.");

  const LoggedStep step(gpu_index_, "Memory copy");
  err = hsa_amd_memory_async_copy(dst, dst_ag, src, src_ag, size, 0, NULL, copy_signal.get());
  throw_if_error(err, "Fail to start a memory copy.");

  // The copy runs on ROCr's own path rather than on this check's queue, so its wait reads no
  // queue error and ends on the copy's own signal or at the deadline; an error raised meanwhile
  // is seen by the check before the doorbell, once the copy is done.
  wait_for_copy(copy_signal.get());
}

// The pools this check allocates from. An iteration whose callback stops at the pool it wants
// answers HSA_STATUS_INFO_BREAK; HSA_STATUS_SUCCESS means the agent has no pool of that kind,
// which is that GPU's skip, and any other status is its failure, with the call and the status.
void ComputeQueueTest::FindPools(BinarySearch* bs) {
  struct PoolSearch {
    hsa_agent_t agent;
    hsa_status_t (*callback)(hsa_amd_memory_pool_t, void*);
    hsa_amd_memory_pool_t* pool;
    const char* none;
  };
  const PoolSearch searches[] = {
      {bs->cpu_dev, FindStandardPool, &bs->cpu_pool, "The CPU has no memory pool for the check."},
      {bs->cpu_dev, FindKernArgPool, &bs->kern_arg_pool, "No pool for kernel arguments."}};

  for (const PoolSearch& search : searches) {
    const hsa_status_t err =
        hsa_amd_agent_iterate_memory_pools(search.agent, search.callback, search.pool);
    if (err == HSA_STATUS_INFO_BREAK) {
      continue;
    }
    if (err == HSA_STATUS_SUCCESS) {
      throw_if_skip(search.none);
    }
    throw_if_error(err, "Fail to call hsa_amd_agent_iterate_memory_pools().");
  }
}

// Once the needed memory pools have been found and the BinarySearch structure
// has been updated with these handles, this function is then used to allocate
// memory from those pools.
// Devices with which a pool is associated already have access to the pool.
// However, other devices may also need to read or write to that memory. Below,
// we see how we can grant access to other devices to address this issue.
void ComputeQueueTest::AllocateAndInitBuffers(BinarySearch* bs) {
  hsa_status_t err;
  uint32_t out_length = 4 * sizeof(uint32_t);

  // The input holds the whole sorted sequence, and the two arrays the run dispatches hold the
  // bounds of each of its sub-divisions, which were decided before this allocation.
  uint32_t input_length = bs->length * sizeof(uint32_t);
  uint32_t in_length = bs->num_sub_divisions * 2 * sizeof(uint32_t);

  // In all of these examples, we want both the cpu and gpu to have access to
  // the buffer in question. We use the array of agents below in the subsequent
  // calls to hsa_amd_agents_allow_access() for this purpose.
  hsa_agent_t ag_list[2] = {bs->gpu_dev, bs->cpu_dev};

  // Each block is taken by its holder as soon as the pool hands it out, so that a failure of any
  // call below returns every block this check has allocated.
  struct Allocation {
    PoolBlockHolder* holder;
    uint32_t size;
  };
  const Allocation allocations[] = {{&bs->input, input_length},
                                    {&bs->output, out_length},
                                    {&bs->input_arr, in_length},
                                    {&bs->input_arr_local, in_length}};

  for (const Allocation& allocation : allocations) {
    void* block = nullptr;
    err = hsa_amd_memory_pool_allocate(bs->cpu_pool, allocation.size, 0, &block);
    if (err == HSA_STATUS_SUCCESS) {
      allocation.holder->reset(block);
    }
    throw_if_error(err, "Fail to allocate memory for the check.");

    err = hsa_amd_agents_allow_access(2, ag_list, NULL, allocation.holder->get());
    throw_if_error(err, "Fail to grant access to the memory the check allocated.");
  }

  // Every block the check reads is cleared once: what no work item writes is then a zero the
  // host can see rather than what the pool's memory held.
  uint32_t* const input = bs->input.as<uint32_t>();
  (void)memset(input, 0, input_length);
  (void)memset(bs->output.get(), 0, out_length);
  (void)memset(bs->input_arr.get(), 0, in_length);
  (void)memset(bs->input_arr_local.get(), 0, in_length);

  // Binary-search application specific code...
  // The input is the same sorted sequence on every run and on every GPU, and it holds the value
  // to find, at input[54]: a random sequence makes a failure the next run cannot repeat.
  for (uint32_t i = 0; i < bs->length; ++i) {
    input[i] = 2 * i;
  }
}

// The code in this function illustrates how to load a kernel from
// pre-compiled code. The goal is to get a handle that can be later
// used in an AQL packet and also to extract information about kernel
// that we will need. All of the information hand kernel handle will
// be saved to the BinarySearch structure. It will be used when we
// populate the AQL packet.
void ComputeQueueTest::LoadKernelFromObjFile(BinarySearch* bs) {
  hsa_status_t err;
  hsa_code_object_reader_t code_obj_rdr = {0};
  hsa_executable_t executable = {0};

  const std::string agent_name = read_agent_name(bs->gpu_dev);
  const LoggedStep step(gpu_index_, "Kernel " + bs->kernel_file_name + " for " + agent_name);

  // The descriptor the search opened is the one the reader is made from, so that the file that is
  // loaded is the file that was found. It is held so that every path out of this function closes
  // it, and it is closed as soon as the reader has been made, whatever that call returned.
  FileDescriptorHolder file = open_kernel_file(bs->kernel_file_name, agent_name);

  err = hsa_code_object_reader_create_from_file(file.get(), &code_obj_rdr);
  file.reset();
  if (err == HSA_STATUS_SUCCESS) {
    bs->reader.reset(code_obj_rdr);
  }
  throw_if_error(err, "Fail to read the kernel file.");

  err = hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, NULL,
                                  &executable);
  if (err == HSA_STATUS_SUCCESS) {
    bs->executable.reset(executable);
  }
  throw_if_error(err, "Fail to create the kernel's executable.");

  err = hsa_executable_load_agent_code_object(bs->executable.get(), bs->gpu_dev, bs->reader.get(),
                                              NULL, NULL);
  throw_if_error(err, "Fail to load the kernel.");

  err = hsa_executable_freeze(bs->executable.get(), NULL);
  throw_if_error(err, "Fail to freeze the kernel's executable.");

  hsa_executable_symbol_t kern_sym;
  err = hsa_executable_get_symbol(bs->executable.get(), NULL, bs->kernel_name.c_str(), bs->gpu_dev,
                                  0, &kern_sym);
  throw_if_error(err, "Fail to find the kernel in its executable.");

  err = hsa_executable_symbol_get_info(kern_sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                       &bs->kernel_object);
  throw_if_error(err, "Fail to read the kernel's code handle.");

  err = hsa_executable_symbol_get_info(
      kern_sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &bs->private_segment_size);
  throw_if_error(err, "Fail to read the kernel's private segment size.");

  err = hsa_executable_symbol_get_info(
      kern_sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &bs->group_segment_size);
  throw_if_error(err, "Fail to read the kernel's group segment size.");

  // Remaining queries not supported on code object v3.
  err = hsa_executable_symbol_get_info(
      kern_sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &bs->kernarg_size);
  throw_if_error(err, "Fail to read the kernel's argument segment size.");

  err = hsa_executable_symbol_get_info(
      kern_sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT, &bs->kernarg_align);
  throw_if_error(err, "Fail to read the kernel's argument segment alignment.");
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
  aql->completion_signal = bs->signal.get();
}

void ComputeQueueTest::WriteAQLToQueue(hsa_kernel_dispatch_packet_t const* in_aql, hsa_queue_t* q,
                                       uint64_t que_idx) {
  void* queue_base = q->base_address;
  const uint32_t queue_mask = q->size - 1;

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

  // reserved0 and reserved2 must be 0 (hsa.h), and ROCr leaves the packet contents of a new ring
  // uninitialized, so they are written from the staging packet, which starts zeroed.
  queue_aql_packet->reserved0 = in_aql->reserved0;
  queue_aql_packet->reserved2 = in_aql->reserved2;
}

// This function allocates the kernel's argument segment from the kern_arg pool we already
// found. Its size and its alignment are checked against that pool before anything is allocated,
// so that a kernel this check cannot serve costs no allocation and no dispatch, and the block is
// zeroed whole, so that every byte the arguments do not fill is a zero the GPU reads rather than
// what the pool's memory held.
void ComputeQueueTest::AllocateKernArgBuffer(BinarySearch* bs) {
  check_kernarg_segment(bs->kern_arg_pool, &bs->kernarg_align, bs->kernarg_size,
                        kSearchArgumentBytes);

  void* kern_arg_buf = nullptr;
  hsa_status_t err =
      hsa_amd_memory_pool_allocate(bs->kern_arg_pool, bs->kernarg_size, 0, &kern_arg_buf);
  if (err == HSA_STATUS_SUCCESS) {
    bs->kern_arg_buffer.reset(kern_arg_buf);
  }
  throw_if_error(err, "Fail to allocate the kernel argument buffer.");

  // ROCr rounds every pool allocation up to whole pages and carves its fragments from a
  // page-aligned block, so the address it hands out meets the kernel's alignment; one that does
  // not is this GPU's failure rather than something the check works around.
  check_kernarg_address(bs->kern_arg_buffer.get(), bs->kernarg_align);
  (void)memset(bs->kern_arg_buffer.get(), 0, bs->kernarg_size);

  // Make sure both the CPU and GPU can access the kernel arguments
  hsa_agent_t ag_list[2] = {bs->gpu_dev, bs->cpu_dev};
  err = hsa_amd_agents_allow_access(2, ag_list, NULL, bs->kern_arg_buffer.get());
  throw_if_error(err, "Fail to grant access to the kernel argument buffer.");

  // Addr. of kern arg start: the block itself, a view into it rather than an allocation.
  bs->kern_arg_address = bs->kern_arg_buffer.get();
}

// Once all the required data for kernel execution is collected (in this
// application it is stored in the BinarySearch structure) we can put it in
// an AQL packet and ring the queue door bell to tell the command processor to
// execute it.
hsa_status_t ComputeQueueTest::Run(BinarySearch* bs) {
  RDC_LOG(RDC_DEBUG, "Executing kernel " << bs->kernel_name);

  // What this check allocated and made, read from the holders that own it. The run's work group
  // and its sub-divisions were decided before those buffers were allocated, which is what sizes
  // them.
  uint32_t* const input = bs->input.as<uint32_t>();
  uint32_t* const output = bs->output.as<uint32_t>();
  uint32_t* const input_arr = bs->input_arr.as<uint32_t>();
  uint32_t* const input_arr_local = bs->input_arr_local.as<uint32_t>();
  hsa_queue_t* const queue = bs->queue.get();
  const hsa_signal_t signal = bs->signal.get();

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

  // The input holds the value by construction, so an input that does not is this GPU's skip:
  // nothing was dispatched, and a pass would say a kernel ran when none did.
  if ((input[0] > bs->find_me) || (input[bs->length - 1] < bs->find_me)) {
    RDC_LOG(RDC_DEBUG, "Returning too early");
    throw_if_skip("The kernel's input does not hold the value to find.");
  }

  output[3] = 1;

  // The kernel's argument segment, and each argument written at its own offset: everything the
  // arguments do not fill, `pad` and the padding after the members among it, stays the zeros the
  // segment was cleared with.
  AllocateKernArgBuffer(bs);

  search_uint4_t* const output_array = reinterpret_cast<search_uint4_t*>(output);
  search_uint2_t* const sorted_array = reinterpret_cast<search_uint2_t*>(input_arr_local);
  uint8_t* const segment = static_cast<uint8_t*>(bs->kern_arg_address);
  (void)memcpy(segment + offsetof(search_args_t, outputArray), &output_array, sizeof(output_array));
  (void)memcpy(segment + offsetof(search_args_t, sortedArray), &sorted_array, sizeof(sorted_array));
  (void)memcpy(segment + offsetof(search_args_t, findMe), &bs->find_me, sizeof(bs->find_me));

  // Populate an AQL packet with the info we've gathered. It starts zeroed, so that the fields
  // the AQL protocol reserves reach the queue's ring as the zeros it asks for.
  hsa_kernel_dispatch_packet_t aql = {};
  PopulateAQLPacket(bs, &aql);

  uint32_t in_length = bs->num_sub_divisions * 2 * sizeof(uint32_t);

  while ((sub_div_size > 1) && (output[3] != 0)) {
    for (uint32_t i = 0; i < bs->num_sub_divisions; i++) {
      // The bounds of this sub-division of the range the last pass chose, which starts at the
      // lower bound: a pass that read from 0 would hand the kernel another range than the one
      // its answer is checked against below.
      uint32_t idx1 = global_lower_bound + (i * sub_div_size);
      uint32_t idx2 = global_lower_bound + ((i + 1) * sub_div_size) - 1;
      input_arr[2 * i] = input[idx1];
      input_arr[2 * i + 1] = input[idx2];
    }

    // Both blocks are the CPU pool's, so this copies the sub-division bounds from one block
    // of system memory to another. It is kept because it is what checks ROCr's copy path, the
    // asynchronous copy and its signal, which the GPU's own reads of that block rest on.
    AgentMemcpy(reinterpret_cast<uint8_t*>(input_arr_local), reinterpret_cast<uint8_t*>(input_arr),
                in_length, bs->gpu_dev, bs->cpu_dev);

    // The flag the kernel raises when it found the value, and the sentinel it writes its
    // sub-division over: a kernel that writes nothing leaves an answer the check refuses below.
    output[3] = 0;
    output[0] = kNoSubDivision;

    // Dispatch kernel with global work size, work group size with ONE dimension
    // and wait for kernel to complete

    // An error ROCr reported earlier suspended the queue, so nothing put on it now would ever
    // complete: it is seen here, before the doorbell, with the hint the wait below carries.
    before_doorbell(bs->status.get());

    // Compute the write index of queue and copy Aql packet into it
    uint64_t que_idx = hsa_queue_load_write_index_relaxed(queue);

    const uint32_t mask = queue->size - 1;

    // The packet is copied into the ring at the index loaded above, except for the setup and
    // header fields, which the store below writes as one. The index stays where it is until
    // this function's caller advances it.
    WriteAQLToQueue(&aql, queue, que_idx);

    uint32_t aql_header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
    aql_header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
    aql_header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;

    // Set the packet's type, acquire and release fences. This should be done
    // atomically after all the other fields have been set, using release
    // memory ordering to ensure all the fields are set when the door bell
    // signal is activated.
    void* q_base = queue->base_address;

    AtomicSetPacketHeader(
        aql_header, aql.setup,
        &(reinterpret_cast<hsa_kernel_dispatch_packet_t*>(q_base))[que_idx & mask]);

    // Increment the write index and ring the doorbell to dispatch kernel.
    hsa_queue_store_write_index_relaxed(queue, (que_idx + 1));
    {
      const LoggedStep step(gpu_index_, "Dispatch of " + bs->kernel_file_name);
      hsa_signal_store_relaxed(queue->doorbell_signal, que_idx);

      // Wait for the kernel, blocked, in hints of a tenth of a second each: the wait ends when
      // the packet processor has taken the signal below 1, when the queue reports an error, or
      // at the deadline, which abandons the work rather than destroying a queue a kernel still
      // runs on.
      wait_for_kernel(signal, bs->status.get());
    }

    // Reset the signal to its initial value for the next iteration
    hsa_signal_store_screlease(signal, 1);

    // Binary search algorithm stuff...
    // The sub-division the kernel answered with, checked before it is multiplied: a product
    // that wraps would land back inside the input, and a kernel that wrote nothing leaves the
    // sentinel above here. The answer numbers the sub-divisions of the range this pass handed
    // the kernel, so it is added to that range's lower bound, and the bound below holds.
    if (output[0] >= bs->num_sub_divisions) {
      throw_failure(kIndexOutOfRange);
    }
    global_lower_bound += output[0] * sub_div_size;
    global_upper_bound = global_lower_bound + sub_div_size - 1;
    if (global_upper_bound >= bs->length) {
      throw_failure(kIndexOutOfRange);
    }
    sub_div_size = (global_upper_bound - global_lower_bound + 1) / bs->num_sub_divisions;
  }

  // The range the kernel chose must hold the value, at input[54] with this input: a range that
  // does not is this GPU's failure, where a miss used to be a debug line and a pass.
  uint32_t element_index = UINT32_MAX;

  for (uint32_t i = global_lower_bound; i <= global_upper_bound; i++) {
    if (input[i] == bs->find_me) {
      element_index = i;
      break;
    }
  }

  RDC_LOG(RDC_DEBUG, "Lower bound = " << global_lower_bound);
  RDC_LOG(RDC_DEBUG, "Upper bound = " << global_upper_bound);
  RDC_LOG(RDC_DEBUG, "Element search for = " << bs->find_me);

  if (element_index == UINT32_MAX) {
    RDC_LOG(RDC_DEBUG, "Element value " << bs->find_me << " not found");
    throw_failure("The kernel's search did not find the value it was given.");
  }
  RDC_LOG(RDC_DEBUG, "Element found at index " << element_index);

  return HSA_STATUS_SUCCESS;
}

void ComputeQueueTest::RunBinarySearchTest(void) {
  // Everything this check makes is owned by one of the holders below, so every path out of this
  // function releases it, in the order they are declared in: the queue, the signal, the
  // executable and its reader, and the buffers last.
  BinarySearch bs(release_context());
  hsa_status_t err;

  InitializeBinarySearch(&bs);

  hsa_agent_t current_gpu;
  get_agent_by_node_id(node_id_, &current_gpu);
  bs.gpu_dev.handle = current_gpu.handle;

  bs.cpu_dev = get_cpu_agent();

  // The kernel is found and loaded before anything is created, so that a GPU whose kernel is not
  // installed gets no queue, no signal and no block, and so reserves no scratch on it.
  // A kernel file that was found and then could not be read or loaded is an error of the run,
  // logged once here with the file and what failed; a kernel that is not installed for the agent
  // is this check's skip, which the search keeps at debug level.
  try {
    LoadKernelFromObjFile(&bs);
  } catch (const SkipException&) {
    throw;
  } catch (const std::exception& failure) {
    RDC_LOG(RDC_ERROR,
            "Fail to load the kernel file " << bs.kernel_file_name << ": " << failure.what());
    throw;
  }

  hsa_signal_t signal = {0};
  err = hsa_signal_create(1, 0, NULL, &signal);
  if (err == HSA_STATUS_SUCCESS) {
    bs.signal.reset(signal);
  }
  throw_if_error(err, "Fail to create signal.");

  // A queue of the packets this check needs, at most what the agent allows, and with no private
  // or group segment reserved: what the kernel needs of either comes from its packet, where a
  // reservation of UINT32_MAX costs an APU every scratch slot of every compute unit.
  // The queue carries an error callback and a status block of its own: ROCr reports an error of
  // the queue only through it, and suspends the queue, so a dispatch on a queue without one
  // never completes.
  err = create_check_queue(bs.gpu_dev, &bs.status, &bs.queue);
  throw_if_error(err, "Fail to call hsa_queue_create().");

  FindPools(&bs);

  // Allocate memory from the correct memory pool, and initialize them as
  // needed for the algorithm.
  AllocateAndInitBuffers(&bs);

  err = Run(&bs);
  throw_if_error(err, "Run binary search fail.");
}

}  // namespace rdc
}  // namespace amd
