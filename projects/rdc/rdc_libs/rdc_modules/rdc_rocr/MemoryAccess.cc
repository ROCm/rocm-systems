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

#include "rdc_modules/rdc_rocr/MemoryAccess.h"

#include <stddef.h>
#include <string.h>

#include <vector>

#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/rdc_common.h"
#include "rdc_modules/rdc_rocr/base_rocr_utils.h"
#include "rdc_modules/rdc_rocr/common.h"

namespace amd {
namespace rdc {

MemoryAccessTest::MemoryAccessTest(uint32_t gpu_index, uint32_t node_id, GpuOutcome* outcome)
    : TestBase(gpu_index, node_id, outcome) {}

MemoryAccessTest::~MemoryAccessTest(void) {}

typedef struct __attribute__((aligned(16))) args_t {
  int* a;
  int* b;
  int* c;
} args;

static void PrintMemorySubtestHeader(const char* header) {
  RDC_LOG(RDC_DEBUG, "  *** Memory Subtest: " << header << " ***");
}

// The elements of one int each that the read and write sub-test allocates on both sides and
// dispatches one work item per.
static const int kMemoryAllocSize = 1024;

// The GPU memory the CPU-access sub-test writes and reads back through the BAR: two pages where
// the pool's page is 4 KiB, which shows that the CPU reaches the GPU's memory and that the
// mapping's two pages are distinct, and little enough that a GPU costs milliseconds of BAR reads
// rather than seconds.
static const size_t kCpuAccessBytes = 8 * 1024;

// The bytes of arguments the read and write kernel is given, without the padding the structure's
// own alignment adds after them.
static const size_t kAccessArgumentBytes = offsetof(args_t, c) + sizeof(int*);

// The GPU pool the CPU-access sub-test uses, and what an iteration that found none passed over.
struct CpuAccessPoolSearch {
  hsa_agent_t cpu_agent = {};
  size_t wanted_bytes = 0;
  hsa_amd_memory_pool_t pool = {};
  bool passed_over_for_access = false;
};

// The first of the GPU's pools that is global and coarse-grained, allows allocation, has a
// maximum allocation that covers what the sub-test asks of it (ROCr refuses a larger request
// before it allocates anything) and can be reached by the CPU. A pool that matched but for that
// access is remembered, so that the sub-test's skip says which it was. A callback never throws:
// ROCr turns an exception into a status of its own and the reason is lost.
static hsa_status_t FindCpuAccessPool(hsa_amd_memory_pool_t pool, void* data) {
  if (data == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  CpuAccessPoolSearch* const search = static_cast<CpuAccessPoolSearch*>(data);

  hsa_amd_segment_t segment = {};
  hsa_status_t err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }
  if (HSA_AMD_SEGMENT_GLOBAL != segment) {
    return HSA_STATUS_SUCCESS;
  }

  uint32_t flags = 0;
  err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }
  // The flag is read as the bit it is: a pool that carries another one as well is still
  // coarse-grained memory the CPU can write.
  if ((flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) == 0) {
    return HSA_STATUS_SUCCESS;
  }

  bool alloc_allowed = false;
  err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
                                     &alloc_allowed);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }
  if (!alloc_allowed) {
    return HSA_STATUS_SUCCESS;
  }

  uint64_t max_size = 0;
  err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE, &max_size);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }
  if (max_size < search->wanted_bytes) {
    return HSA_STATUS_SUCCESS;
  }

  hsa_amd_memory_pool_access_t access = HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
  err = hsa_amd_agent_memory_pool_get_info(search->cpu_agent, pool,
                                           HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }
  if (access == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
    search->passed_over_for_access = true;
    return HSA_STATUS_SUCCESS;
  }

  search->pool = pool;
  return HSA_STATUS_INFO_BREAK;
}

// Test to check GPU can read & write to system memory
void MemoryAccessTest::GPUAccessToCPUMemoryTest(hsa_agent_t cpuAgent, hsa_agent_t gpuAgent) {
  hsa_status_t err;

  // What this sub-test makes, declared in the reverse of the order in which it is released: the
  // queue first, then its status block, the signal, the executable and its code object reader,
  // the buffers last.
  PoolBlockHolder cpu_result_holder(release_context());
  PoolBlockHolder sys_data_holder(release_context());
  PoolBlockHolder dup_sys_data_holder(release_context());
  PoolBlockHolder gpu_result_holder(release_context());
  PoolBlockHolder kern_args_holder(release_context());
  CodeObjectReaderHolder reader_holder(release_context());
  ExecutableHolder executable_holder(release_context());
  SignalHolder signal_holder(release_context());
  QueueStatusHolder status_holder(release_context());
  QueueHolder queue_holder(release_context());

  // What this sub-test allocates from each of its two global pools in one block, which is what
  // those pools have to be able to hand out.
  const size_t buffer_bytes = static_cast<size_t>(kMemoryAllocSize) * sizeof(int);

  // Get Global Memory Pool on the gpuAgent to allocate gpu buffers. The callback records the
  // last pool that matches and never stops the iteration, so what it recorded says whether the
  // agent has one.
  PoolRequest gpu_request;
  gpu_request.wanted_bytes = buffer_bytes;
  err = hsa_amd_agent_iterate_memory_pools(gpuAgent, GetGlobalMemoryPool, &gpu_request);
  throw_if_error(err, "Fail to list the GPU's memory pools.");
  const hsa_amd_memory_pool_t gpu_pool = gpu_request.pool;
  if (gpu_pool.handle == 0) {
    throw_if_skip("The GPU has no memory pool to check.");
  }

  // Get System Memory Pool on the cpuAgent to allocate host side buffers
  PoolRequest cpu_request;
  cpu_request.wanted_bytes = buffer_bytes;
  err = hsa_amd_agent_iterate_memory_pools(cpuAgent, GetGlobalMemoryPool, &cpu_request);
  throw_if_error(err, "Fail to list the CPU's memory pools.");
  const hsa_amd_memory_pool_t global_pool = cpu_request.pool;
  if (global_pool.handle == 0) {
    throw_if_skip("The CPU has no memory pool for the check.");
  }

  // Find a memory pool that supports kernel arguments.
  hsa_amd_memory_pool_t kernarg_pool = {};
  err = hsa_amd_agent_iterate_memory_pools(cpuAgent, GetKernArgMemoryPool, &kernarg_pool);
  throw_if_error(err, "Fail to list the CPU's memory pools.");
  if (kernarg_pool.handle == 0) {
    throw_if_skip("No pool for kernel arguments.");
  }

  // Both of the accesses this sub-test needs, read before it makes anything: it reads its result
  // back from the GPU's pool, and the GPU reads and writes the CPU's. A GPU that is short of
  // either is a skip that says which, rather than a run that proves nothing.
  hsa_amd_memory_pool_access_t access = HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
  err = hsa_amd_agent_memory_pool_get_info(cpuAgent, gpu_pool,
                                           HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access);
  throw_if_error(err, "Fail to read the CPU's access to the GPU's memory pool.");
  if (access == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
    throw_if_skip("The CPU cannot read the GPU's memory pool.");
  }

  access = HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
  err = hsa_amd_agent_memory_pool_get_info(gpuAgent, global_pool,
                                           HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access);
  throw_if_error(err, "Fail to read the GPU's access to the CPU's memory pool.");
  if (access == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
    throw_if_skip("The GPU cannot access the CPU's memory pool.");
  }

  // Create the executable, get symbol by name and load the code object. A kernel that is not
  // installed for this agent is this sub-test's skip, and it is found before anything is created,
  // so a GPU whose kernel is missing gets no queue, no signal, no block and no scratch. The
  // executable and its reader live until the dispatch is done: destroying the executable would
  // make the kernel object it reports invalid.
  set_kernel_file_name("gpuReadWrite_kernels.hsaco");
  set_kernel_name("gpuReadWrite");
  set_agent_name(read_agent_name(gpuAgent));
  {
    const LoggedStep step(gpu_index_, "Kernel " + kernel_file_name() + " for " + get_agent_name());
    // A kernel file that was found and then could not be read or loaded is an error of the run,
    // logged once here with the file and what failed; a kernel that is not installed for the
    // agent is this sub-test's skip, which the search keeps at debug level.
    try {
      LoadKernelFromObjFile(this, &gpuAgent, &reader_holder, &executable_holder);
    } catch (const SkipException&) {
      throw;
    } catch (const std::exception& failure) {
      RDC_LOG(RDC_ERROR,
              "Fail to load the kernel file " << kernel_file_name() << ": " << failure.what());
      throw;
    }
  }

  // create queue: the packets this one dispatch needs, at most what the agent allows, rather
  // than the largest queue the agent has, which is 8 MiB of ring on ROCr.
  // It carries an error callback and a status block of its own: ROCr reports an error of the
  // queue only through it, and suspends the queue, so a dispatch on a queue without one never
  // completes.
  err = create_check_queue(gpuAgent, &status_holder, &queue_holder);
  throw_if_error(err, "Fail to call hsa_queue_create().");
  hsa_queue_t* const queue = queue_holder.get();  // command queue

  // Allocate the host side buffers (sys_data, dup_sys_data, cpuResult) on system memory and the
  // GPU's result buffer on its own pool, each of kMemoryAllocSize elements of one int, which is
  // what the grid below writes. Each block is taken by its holder as the pool hands it out, so
  // a later failure returns every block this sub-test allocated.
  struct Allocation {
    PoolBlockHolder* holder;
    hsa_amd_memory_pool_t pool;
    size_t size;
  };
  const Allocation allocations[] = {{&cpu_result_holder, global_pool, buffer_bytes},
                                    {&sys_data_holder, global_pool, buffer_bytes},
                                    {&dup_sys_data_holder, global_pool, buffer_bytes},
                                    {&gpu_result_holder, gpu_pool, buffer_bytes}};

  for (const Allocation& allocation : allocations) {
    void* block = nullptr;
    err = hsa_amd_memory_pool_allocate(allocation.pool, allocation.size, 0, &block);
    if (err == HSA_STATUS_SUCCESS) {
      allocation.holder->reset(block);
    }
    throw_if_error(err, "Fail to allocate memory for the check.");
  }

  int* const cpuResult = cpu_result_holder.as<int>();
  int* const sys_data = sys_data_holder.as<int>();
  int* const dup_sys_data = dup_sys_data_holder.as<int>();
  int* const gpuResult = gpu_result_holder.as<int>();

  // initialize the host buffers, each element with a value of its own, so that a read-back that
  // is off by an element holds values that do not match rather than values that do.
  for (int i = 0; i < kMemoryAllocSize; ++i) {
    sys_data[i] = i + 1;
    dup_sys_data[i] = sys_data[i];
  }

  memset(cpuResult, 0, buffer_bytes);

  // For the dGPU, we have coarse grained local memory, so the GPU's result buffer above
  // comes from the GPU's own GLOBAL segment.

  // Allow cpuAgent access to all allocated GPU memory.
  err = hsa_amd_agents_allow_access(1, &cpuAgent, NULL, gpuResult);
  throw_if_error(err, "Fail to grant the CPU access to the GPU's memory.");
  memset(gpuResult, 0, buffer_bytes);

  // Allow gpuAgent access to all allocated system memory.
  err = hsa_amd_agents_allow_access(1, &gpuAgent, NULL, cpuResult);
  throw_if_error(err, "Fail to grant the GPU access to the CPU's memory.");
  err = hsa_amd_agents_allow_access(1, &gpuAgent, NULL, sys_data);
  throw_if_error(err, "Fail to grant the GPU access to the CPU's memory.");
  err = hsa_amd_agents_allow_access(1, &gpuAgent, NULL, dup_sys_data);
  throw_if_error(err, "Fail to grant the GPU access to the CPU's memory.");

  // The kernel's argument segment, checked against the pool before anything is allocated for it
  // and zeroed whole once it is, with the three pointers written at their offsets: every byte
  // past them is a zero the kernel reads rather than what the pool's memory held.
  uint32_t alignment = kernarg_align();
  check_kernarg_segment(kernarg_pool, &alignment, kernarg_size(), kAccessArgumentBytes);

  void* kern_args_block = nullptr;
  err = hsa_amd_memory_pool_allocate(kernarg_pool, kernarg_size(), 0, &kern_args_block);
  if (err == HSA_STATUS_SUCCESS) {
    kern_args_holder.reset(kern_args_block);
  }
  throw_if_error(err, "Fail to allocate the kernel argument buffer.");
  check_kernarg_address(kern_args_holder.get(), alignment);
  memset(kern_args_holder.get(), 0, kernarg_size());

  err = hsa_amd_agents_allow_access(1, &gpuAgent, NULL, kern_args_holder.get());
  throw_if_error(err, "Fail to grant the GPU access to the kernel arguments.");

  uint8_t* const kernArgs = kern_args_holder.as<uint8_t>();
  memcpy(kernArgs + offsetof(args_t, a), &sys_data, sizeof(sys_data));
  // system memory passed to gpu for write, and gpu memory to verify that gpu read system data
  memcpy(kernArgs + offsetof(args_t, b), &cpuResult, sizeof(cpuResult));
  memcpy(kernArgs + offsetof(args_t, c), &gpuResult, sizeof(gpuResult));

  // Fill the dispatch packet with
  // workgroup_size, grid_size, kernelArgs and completion signal
  // Put it on the queue and launch the kernel by ringing the doorbell

  // create completion signal
  hsa_signal_t signal = {0};
  err = hsa_signal_create(1, 0, NULL, &signal);
  if (err == HSA_STATUS_SUCCESS) {
    signal_holder.reset(signal);
  }
  throw_if_error(err, "Fail to create signal.");

  // create aql packet
  hsa_kernel_dispatch_packet_t aql;
  memset(&aql, 0, sizeof(aql));

  // initialize aql packet
  aql.workgroup_size_x = 256;
  aql.workgroup_size_y = 1;
  aql.workgroup_size_z = 1;
  aql.grid_size_x = kMemoryAllocSize;
  aql.grid_size_y = 1;
  aql.grid_size_z = 1;
  // What the kernel asks for, rather than none of either: ROCr gives a queue scratch only for a
  // dispatch that asks, so a kernel rebuilt with scratch or LDS would be dispatched without it.
  aql.private_segment_size = private_segment_size();
  aql.group_segment_size = group_segment_size();
  aql.kernel_object = kernel_object();  // kernel_code;
  aql.kernarg_address = kernArgs;
  aql.completion_signal = signal;

  // const uint32_t queue_size = queue->size;
  const uint32_t queue_mask = queue->size - 1;

  // An error ROCr reported earlier suspended the queue, so this dispatch would never complete:
  // it is seen here, before the doorbell, with the hint the wait below carries.
  before_doorbell(status_holder.get());

  // write to command queue
  uint64_t index = hsa_queue_load_write_index_relaxed(queue);
  hsa_queue_store_write_index_relaxed(queue, index + 1);

  WriteAQLToQueueLoc(queue, index, &aql);

  hsa_kernel_dispatch_packet_t* q_base_addr =
      reinterpret_cast<hsa_kernel_dispatch_packet_t*>(queue->base_address);
  AtomicSetPacketHeader(
      (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
          (1 << HSA_PACKET_HEADER_BARRIER) |
          (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
          (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE),
      (1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS),
      reinterpret_cast<hsa_kernel_dispatch_packet_t*>(&q_base_addr[index & queue_mask]));

  {
    const LoggedStep step(gpu_index_, "Dispatch of " + kernel_file_name());

    // ringdoor bell
    hsa_signal_store_relaxed(queue->doorbell_signal, index);
    // Wait for the kernel, blocked rather than spinning a core, in hints of a tenth of a second
    // each: the wait ends when the packet processor has taken the signal below 1, when the queue
    // reports an error, or at the deadline, which abandons the work.
    wait_for_kernel(signal, status_holder.get());
  }
  hsa_signal_store_relaxed(signal, 1);

  {
    // compare device and host side results
    const LoggedStep step(gpu_index_, "GPU access read-back");

    RDC_LOG(RDC_DEBUG, "check gpu has read the system memory");
    for (int i = 0; i < kMemoryAllocSize; ++i) {
      if (gpuResult[i] != dup_sys_data[i]) {
        RDC_LOG(RDC_DEBUG, "The GPU read " << gpuResult[i] << " at element " << i
                                           << ", where the CPU wrote " << dup_sys_data[i]);
        throw_failure("The GPU read other values from the system memory than the CPU wrote.");
      }
    }

    RDC_LOG(RDC_DEBUG, "gpu has read the system memory successfully");
    RDC_LOG(RDC_DEBUG, "check gpu has written to system memory");
    for (int i = 0; i < kMemoryAllocSize; ++i) {
      if (cpuResult[i] != i) {
        RDC_LOG(RDC_DEBUG, "The system memory holds " << cpuResult[i] << " at element " << i
                                                      << ", where the check asked for " << i);
        throw_failure(
            "The GPU wrote other values into the system memory than the check asked for.");
      }
    }

    RDC_LOG(RDC_DEBUG, "gpu has written to system memory successfully");
  }

  // The holders above release the queue, the signal, the executable, its reader and every
  // block when this function returns, on this path and on every other.
}

// Test to check cpu can read & write to GPU memory
void MemoryAccessTest::CPUAccessToGPUMemoryTest(hsa_agent_t cpuAgent, hsa_amd_memory_pool_t pool) {
  // Two pages of the GPU's memory, written in one pass and only then read back in a second: the
  // reads go through a volatile pointer, so that the compiler cannot answer the comparison with
  // the values it stored rather than with what the GPU's memory holds.
  const unsigned int count = kCpuAccessBytes / sizeof(unsigned int);
  std::vector<unsigned int> host_data(count, 0);

  PoolBlockHolder gpu_data_holder(release_context());
  void* block = nullptr;
  hsa_status_t err = hsa_amd_memory_pool_allocate(pool, kCpuAccessBytes, 0, &block);
  if (err == HSA_STATUS_SUCCESS) {
    gpu_data_holder.reset(block);
  }
  throw_if_error(err, "Fail to allocate memory from the GPU's pool.");

  err = hsa_amd_agents_allow_access(1, &cpuAgent, NULL, gpu_data_holder.get());
  throw_if_error(err, "Fail to grant the CPU access to the GPU's memory.");

  unsigned int* const gpu_data = gpu_data_holder.as<unsigned int>();
  memset(gpu_data, 0, kCpuAccessBytes);

  {
    // Verify CPU can read & write to GPU memory
    const LoggedStep step(gpu_index_, "CPU access");

    RDC_LOG(RDC_DEBUG, "Verify CPU can read & write to GPU memory");
    for (unsigned int i = 0; i < count; ++i) {
      host_data[i] = i;
      gpu_data[i] = i;  // Write to gpu memory directly
    }

    volatile const unsigned int* const read_back = gpu_data;
    for (unsigned int i = 0; i < count; ++i) {
      if (read_back[i] != host_data[i]) {  // Reading GPU memory
        RDC_LOG(RDC_DEBUG, "The GPU's memory holds " << read_back[i] << " at element " << i
                                                     << ", where the CPU wrote " << host_data[i]);
        throw_failure("The CPU read other values from the GPU's memory than it wrote.");
      }
    }
  }

  RDC_LOG(RDC_DEBUG, "CPU have read & write to GPU memory successfully");
}

void MemoryAccessTest::CPUAccessToGPUMemoryTest(void) {
  PrintMemorySubtestHeader("CPUAccessToGPUMemoryTest in Memory Pools");

  const hsa_agent_t cpu_agent = get_cpu_agent();

  // the GPU this check was asked about, and no other
  hsa_agent_t current_gpu;
  get_agent_by_node_id(node_id_, &current_gpu);

  // The pool this sub-test writes and reads back. An iteration that found none is this GPU's
  // skip, which says whether a pool was passed over for the CPU's access alone.
  CpuAccessPoolSearch search;
  search.cpu_agent = cpu_agent;
  search.wanted_bytes = kCpuAccessBytes;
  const hsa_status_t err =
      hsa_amd_agent_iterate_memory_pools(current_gpu, FindCpuAccessPool, &search);
  if (err == HSA_STATUS_SUCCESS) {
    throw_if_skip(search.passed_over_for_access
                      ? "The CPU cannot access the GPU's memory pool."
                      : "The GPU has no coarse-grained memory pool to check.");
  }
  if (err != HSA_STATUS_INFO_BREAK) {
    throw_if_error(err, "Fail to list the GPU's memory pools.");
  }

  CPUAccessToGPUMemoryTest(cpu_agent, search.pool);
  RDC_LOG(RDC_DEBUG, "subtest Passed");
}

void MemoryAccessTest::GPUAccessToCPUMemoryTest(void) {
  PrintMemorySubtestHeader("GPUAccessToCPUMemoryTest in Memory Pools");

  const hsa_agent_t cpu_agent = get_cpu_agent();

  // find current gpu
  hsa_agent_t current_gpu;
  get_agent_by_node_id(node_id_, &current_gpu);

  GPUAccessToCPUMemoryTest(cpu_agent, current_gpu);
  RDC_LOG(RDC_DEBUG, "subtest Passed");
}

}  // namespace rdc
}  // namespace amd
