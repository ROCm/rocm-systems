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
  uint32_t group_segment_size;    ///< Kernel group seg size
  uint32_t private_segment_size;  ///< Kernel private seg size
} BinarySearch;

class ComputeQueueTest : public TestBase {
 public:
  explicit ComputeQueueTest(uint32_t gpu_index);

  // @Brief: Destructor for test case of ComputeQueueTest
  virtual ~ComputeQueueTest();

  // @Brief: Setup the environment for measurement
  virtual hsa_status_t SetUp();

  // @Brief: Core measurement execution
  virtual void Run();

  // @Brief: Clean up and retrive the resource
  virtual void Close();

  // @Brief: Display  results
  virtual void DisplayResults() const;

  // @Brief: Display information about what this test does
  virtual void DisplayTestInfo(void);

  hsa_status_t RunBinarySearchTest(void);

 private:
  void InitializeBinarySearch(BinarySearch* bs);
  hsa_status_t FindPools(BinarySearch* bs);
  hsa_status_t AllocateAndInitBuffers(BinarySearch* bs);
  hsa_status_t LoadKernelFromObjFile(BinarySearch* bs);
  hsa_status_t Run(BinarySearch* bs);
  hsa_status_t CleanUp(BinarySearch* bs);
  void PopulateAQLPacket(BinarySearch const* bs, hsa_kernel_dispatch_packet_t* aql);
  hsa_status_t AgentMemcpy(void* dst, const void* src, size_t size, hsa_agent_t dst_ag,
                           hsa_agent_t src_ag);
  hsa_status_t AllocAndSetKernArgs(BinarySearch* bs, void* args, size_t arg_size,
                                   void** aql_buf_ptr);
  void WriteAQLToQueue(hsa_kernel_dispatch_packet_t const* in_aql, hsa_queue_t* q);
};

}  // namespace rdc
}  // namespace amd

#endif  // RDC_MODULES_RDC_ROCR_COMPUTEQUEUETEST_H_
