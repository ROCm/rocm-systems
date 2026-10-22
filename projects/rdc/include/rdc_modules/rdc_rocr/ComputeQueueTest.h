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
#ifndef RDC_MODULES_RDC_ROCR_COMPUTEQUEUETEST_H_
#define RDC_MODULES_RDC_ROCR_COMPUTEQUEUETEST_H_

#include "hsa/hsa.h"
#include "rdc_modules/rdc_rocr/TestBase.h"

namespace amd {
namespace rdc {

// Hold all the info specific to binary search
typedef struct BinarySearch {
  // The holders below report a release that failed where this check's outcome is put together.
  explicit BinarySearch(ReleaseContext* context)
      : input(context),
        output(context),
        input_arr(context),
        input_arr_local(context),
        kern_arg_buffer(context),
        reader(context),
        executable(context),
        signal(context),
        status(context),
        queue(context) {}

  // Binary Search parameters
  uint32_t length = 0;
  uint32_t work_group_size = 0;
  uint32_t work_grid_size = 0;
  uint32_t num_sub_divisions = 0;
  uint32_t find_me = 0;

  // Properly aligned address inside kern_arg_buffer, to be used in the aql packet. It is a view
  // into that block, not an allocation of its own.
  void* kern_arg_address = nullptr;

  // Kernel code
  std::string kernel_file_name;
  std::string kernel_name;
  uint32_t kernarg_size = 0;
  uint32_t kernarg_align = 0;

  // HSA/RocR objects needed for this application
  hsa_agent_t gpu_dev = {};
  hsa_agent_t cpu_dev = {};
  hsa_amd_memory_pool_t cpu_pool = {};
  hsa_amd_memory_pool_t kern_arg_pool = {};

  // Other items we need to populate AQL packet
  uint64_t kernel_object = 0;
  uint32_t group_segment_size = 0;    ///< Kernel group seg size
  uint32_t private_segment_size = 0;  ///< Kernel private seg size

  // What this check makes. They are declared in the reverse of the order in which they are
  // released, so that C++ releases them in that order whatever the order in which they were
  // made: the queue first, then its status block, the signal, the executable and its code
  // object reader, and the buffers last.
  PoolBlockHolder input;
  PoolBlockHolder output;
  PoolBlockHolder input_arr;
  PoolBlockHolder input_arr_local;
  PoolBlockHolder kern_arg_buffer;
  CodeObjectReaderHolder reader;
  ExecutableHolder executable;
  SignalHolder signal;
  QueueStatusHolder status;
  QueueHolder queue;
} BinarySearch;

class ComputeQueueTest : public TestBase {
 public:
  ComputeQueueTest(uint32_t gpu_index, uint32_t node_id, GpuOutcome* outcome);

  // @Brief: Destructor for test case of ComputeQueueTest
  virtual ~ComputeQueueTest();

  // Every call that fails is this GPU's failure, named with its status, and what the check
  // cannot run is its skip: both leave these functions by throwing.
  void RunBinarySearchTest(void);

 private:
  void InitializeBinarySearch(BinarySearch* bs);
  void FindPools(BinarySearch* bs);
  void AllocateAndInitBuffers(BinarySearch* bs);
  void LoadKernelFromObjFile(BinarySearch* bs);
  hsa_status_t Run(BinarySearch* bs);
  void PopulateAQLPacket(BinarySearch const* bs, hsa_kernel_dispatch_packet_t* aql);
  void AgentMemcpy(void* dst, const void* src, size_t size, hsa_agent_t dst_ag, hsa_agent_t src_ag);
  void AllocateKernArgBuffer(BinarySearch* bs);
  // Writes the packet into the ring at the write index the caller holds, which it leaves
  // where it is: Run loads that index and stores the next one.
  void WriteAQLToQueue(hsa_kernel_dispatch_packet_t const* in_aql, hsa_queue_t* q,
                       uint64_t que_idx);
};

}  // namespace rdc
}  // namespace amd

#endif  // RDC_MODULES_RDC_ROCR_COMPUTEQUEUETEST_H_
