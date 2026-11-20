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

#include "rdc_modules/rdc_rocr/MemoryTest.h"

#include <algorithm>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/rdc_common.h"
#include "rdc_modules/rdc_rocr/base_rocr_utils.h"
#include "rdc_modules/rdc_rocr/common.h"

namespace amd {
namespace rdc {

MemoryTest::MemoryTest(uint32_t gpu_index, uint32_t node_id, GpuOutcome* outcome)
    : TestBase(gpu_index, node_id, outcome) {}

MemoryTest::~MemoryTest(void) {}

namespace {

// What one allocation of a pool takes: enough to show the pool hands memory out and takes it
// back, little enough that the check does not hold memory the GPU's own work needs. ROCr rounds
// a request up to its page.
const uint64_t kPoolAllocationBytes = 64ull * 1024 * 1024;

// A number of bytes as the GPU's report reads it.
std::string bytes_text(uint64_t bytes) {
  if (bytes >= 1024 * 1024 && (bytes % (1024 * 1024)) == 0) {
    return std::to_string(bytes / (1024 * 1024)) + " MiB";
  }
  if (bytes >= 1024 && (bytes % 1024) == 0) {
    return std::to_string(bytes / 1024) + " KiB";
  }
  return std::to_string(bytes) + " bytes";
}

// The agent's pools, each handle recorded and no attribute read: a callback that reads one makes
// its failure the iteration's, which ROCr reports as a status of its own, and the pools after it
// are never seen. A callback never throws.
hsa_status_t ListPools(hsa_amd_memory_pool_t pool, void* data) {
  if (data == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  try {
    static_cast<std::vector<hsa_amd_memory_pool_t>*>(data)->push_back(pool);
  } catch (const std::bad_alloc&) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  return HSA_STATUS_SUCCESS;
}

}  // namespace

// Each pool of the GPU that allows allocation is allocated from and freed at once, and what the
// check reports is that it did so: a user reads this check to learn that the GPU's memory can be
// taken and given back, not how large a block the GPU happens to have free, which is what the
// other processes on it decide. The pool's own attributes are read after the iteration has
// returned, so that a read that fails is that pool's failure and the pools after it are checked
// all the same.
void MemoryTest::MaxSingleAllocationTest(void) {
  hsa_agent_t current_gpu = {};
  get_agent_by_node_id(node_id_, &current_gpu);

  std::vector<hsa_amd_memory_pool_t> pools;
  const hsa_status_t listed = hsa_amd_agent_iterate_memory_pools(current_gpu, ListPools, &pools);
  throw_if_error(listed, "Fail to call hsa_amd_agent_iterate_memory_pools().");

  uint32_t counted = 0;
  for (size_t i = 0; i < pools.size(); i++) {
    const std::string number = std::to_string(i + 1);

    bool alloc_allowed = false;
    hsa_status_t err = hsa_amd_memory_pool_get_info(
        pools[i], HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc_allowed);
    if (err != HSA_STATUS_SUCCESS) {
      say(RDC_DIAG_RESULT_FAIL,
          hsa_error_text(
              "Fail to call hsa_amd_memory_pool_get_info() for memory pool " + number + ".", err));
      continue;
    }
    if (!alloc_allowed) {
      RDC_LOG(RDC_DEBUG, "Memory pool " << number << " does not allow allocation: not checked.");
      continue;
    }

    uint64_t max_size = 0;
    err =
        hsa_amd_memory_pool_get_info(pools[i], HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE, &max_size);
    if (err != HSA_STATUS_SUCCESS) {
      say(RDC_DIAG_RESULT_FAIL,
          hsa_error_text(
              "Fail to call hsa_amd_memory_pool_get_info() for memory pool " + number + ".", err));
      continue;
    }
    // ROCr refuses a request of no bytes, so a pool that reports no maximum cannot be checked.
    if (max_size == 0) {
      RDC_LOG(RDC_DEBUG, "Memory pool " << number << " reports a maximum of 0: not checked.");
      continue;
    }

    const uint64_t bytes = std::min<uint64_t>(kPoolAllocationBytes, max_size);
    bool freed = false;
    {
      const LoggedStep step(gpu_index_, "Pool " + number + " allocate and free");

      // The block goes back to its pool as soon as it was handed out, and a free that fails is
      // this GPU's failure, through the holder.
      PoolBlockHolder block(release_context());
      void* address = nullptr;
      err = hsa_amd_memory_pool_allocate(pools[i], bytes, 0, &address);
      if (err == HSA_STATUS_SUCCESS) {
        block.reset(address);
      }
      // The block is given back here, inside the step, so that the sentence below says what this
      // pool did rather than what it was asked to do.
      block.reset();
      freed = !block.release_failed();
    }

    // A pool with no free memory says so and is not counted: the GPU's memory is in use, which
    // is not a fault of the GPU.
    if (err == HSA_STATUS_ERROR_OUT_OF_RESOURCES) {
      say(RDC_DIAG_RESULT_SKIP, "Memory pool " + number + " had no free memory.");
      continue;
    }
    if (err != HSA_STATUS_SUCCESS) {
      say(RDC_DIAG_RESULT_FAIL,
          hsa_error_text(
              "Fail to call hsa_amd_memory_pool_allocate() for memory pool " + number + ".", err));
      continue;
    }

    // A pool whose block could not be given back says that it was allocated, is not counted
    // and is that pool's failure: the holder has already made that release this GPU's failure,
    // and a pool the check did not finish with is not a pool that passed.
    if (freed) {
      counted++;
    }
    const std::string did = freed ? ": allocated and freed " : ": allocated ";
    say(freed ? RDC_DIAG_RESULT_PASS : RDC_DIAG_RESULT_FAIL,
        "Memory pool " + number + did + bytes_text(bytes) + ".");
  }

  if (counted == 0) {
    throw_if_skip("No memory pool could be checked.");
  }
}

}  // namespace rdc
}  // namespace amd
