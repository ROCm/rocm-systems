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

/// \file
/// Implementation of utility functions used by RocR applications
#include "rdc_modules/rdc_rocr/common.h"

#include <assert.h>
#include <stdlib.h>

#include <memory>
#include <sstream>
#include <string>

#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

std::string hsa_error_text(const std::string& msg, hsa_status_t err) {
  // A call that gave no message of its own leaves the status to name it, with no space before
  // it: every sentence a GPU reads starts with a word.
  std::string text = msg.empty() ? "" : msg + " ";
  text += "hsa error code: " + std::to_string(err);
  // ROCr leaves the caller's pointer as it was for a status it has no string for, so what it
  // wrote is used only when it says it wrote something.
  const char* errstr = nullptr;
  if (hsa_status_string(err, &errstr) == HSA_STATUS_SUCCESS && errstr != nullptr) {
    text += " ";
    text += errstr;
  }
  return text;
}

void throw_if_error(hsa_status_t err, const std::string& msg) {
  if (err != HSA_STATUS_SUCCESS) {
    throw std::runtime_error(hsa_error_text(msg, err));
  }
}

void throw_if_skip(const std::string& msg) { throw SkipException(msg.c_str()); }

void throw_failure(const std::string& msg) { throw std::runtime_error(msg); }

// Whether the agent's pool allows a check to allocate from it. A pool that does not is never
// chosen, so that a check skips it here rather than failing at its first allocation from it.
static hsa_status_t PoolAllowsAllocation(hsa_amd_memory_pool_t pool, bool* allowed) {
  bool alloc_allowed = false;
  const hsa_status_t err = hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc_allowed);
  *allowed = (err == HSA_STATUS_SUCCESS) && alloc_allowed;
  return err;
}

// Find CPU Agents. A callback never throws: ROCr turns an exception into a status of its own and
// the reason is lost, so every query's status is returned as it is.
hsa_status_t IterateCPUAgents(hsa_agent_t agent, void* data) {
  assert(data != nullptr);
  if (data == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  hsa_device_type_t device_type = {};
  const hsa_status_t status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  if (HSA_DEVICE_TYPE_CPU != device_type) {
    return HSA_STATUS_SUCCESS;
  }
  try {
    static_cast<std::vector<hsa_agent_t>*>(data)->push_back(agent);
  } catch (const std::bad_alloc&) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  return HSA_STATUS_SUCCESS;
}

// Find GPU Agents
hsa_status_t IterateGPUAgents(hsa_agent_t agent, void* data) {
  assert(data != nullptr);
  if (data == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  hsa_device_type_t device_type = {};
  const hsa_status_t status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  if (HSA_DEVICE_TYPE_GPU != device_type) {
    return HSA_STATUS_SUCCESS;
  }
  try {
    static_cast<std::vector<hsa_agent_t>*>(data)->push_back(agent);
  } catch (const std::bad_alloc&) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  return HSA_STATUS_SUCCESS;
}

// Find coarse grained system memory. This callback records the last pool that matches and never
// stops the iteration, so its caller decides from what it recorded whether the agent has one.
hsa_status_t GetGlobalMemoryPool(hsa_amd_memory_pool_t pool, void* data) {
  if (nullptr == data) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  PoolRequest* const request = static_cast<PoolRequest*>(data);

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

  bool alloc_allowed = false;
  err = PoolAllowsAllocation(pool, &alloc_allowed);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }
  if (!alloc_allowed) {
    return HSA_STATUS_SUCCESS;
  }

  // The pool must be able to hand out what the sub-test allocates from it in one block: ROCr
  // refuses a larger request before it allocates anything.
  uint64_t max_size = 0;
  err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE, &max_size);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }
  if (max_size < request->wanted_bytes) {
    return HSA_STATUS_SUCCESS;
  }

  // A dGPU's pool is coarse-grained; on an APU it has to be fine-grained.
  const uint32_t wanted =
      HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED | HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED;
  if ((flags & wanted) != 0) {
    request->pool = pool;
  }
  return HSA_STATUS_SUCCESS;
}

// Find  a memory pool that can be used for kernarg locations. It records the last match too.
hsa_status_t GetKernArgMemoryPool(hsa_amd_memory_pool_t pool, void* data) {
  if (nullptr == data) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

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

  bool alloc_allowed = false;
  err = PoolAllowsAllocation(pool, &alloc_allowed);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }
  if (!alloc_allowed) {
    return HSA_STATUS_SUCCESS;
  }

  if ((flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) != 0) {
    *reinterpret_cast<hsa_amd_memory_pool_t*>(data) = pool;
  }

  return HSA_STATUS_SUCCESS;
}

/// Ennumeration that indicates whether a pool property must be present or not.
/// This is meant to be used by FindPool
typedef enum {
  POOL_PROP_OFF = 0,   ///< The property must not be present.
  POOL_PROP_ON,        ///< The property must be present.
  POOL_PROP_DONT_CARE  ///< We don't care if the property is present or not.
} pool_prop_t;

// The pool a check asks for, which must also allow allocation. This callback stops the iteration
// at the first pool that matches, so its caller reads HSA_STATUS_INFO_BREAK as a match,
// HSA_STATUS_SUCCESS as no such pool and any other status as that GPU's failure.
static hsa_status_t FindPool(hsa_amd_memory_pool_t pool, void* data, hsa_amd_segment_t in_segment,
                             pool_prop_t accessible_by_all, pool_prop_t kern_arg,
                             pool_prop_t fine_grain) {
  if (nullptr == data) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  hsa_status_t err;
  hsa_amd_segment_t segment = {};
  uint32_t flag = 0;

  err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }

  if (in_segment != segment) {
    return HSA_STATUS_SUCCESS;
  }

  bool alloc_allowed = false;
  err = PoolAllowsAllocation(pool, &alloc_allowed);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }
  if (!alloc_allowed) {
    return HSA_STATUS_SUCCESS;
  }

  if (HSA_AMD_SEGMENT_GLOBAL == in_segment) {
    err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flag);
    if (err != HSA_STATUS_SUCCESS) {
      return err;
    }

    if (kern_arg != POOL_PROP_DONT_CARE) {
      uint32_t karg_st = flag & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT;
      if ((karg_st == 0 && kern_arg == POOL_PROP_ON) ||
          (karg_st != 0 && kern_arg == POOL_PROP_OFF)) {
        return HSA_STATUS_SUCCESS;
      }
    }
    if (fine_grain != POOL_PROP_DONT_CARE) {
      uint32_t fg_st = flag & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED;
      if ((fg_st == 0 && fine_grain == POOL_PROP_ON) ||
          (fg_st != 0 && fine_grain == POOL_PROP_OFF)) {
        return HSA_STATUS_SUCCESS;
      }
    }
  }

  if (accessible_by_all != POOL_PROP_DONT_CARE) {
    bool access_read = false;
    err = hsa_amd_memory_pool_get_info(
        pool, (hsa_amd_memory_pool_info_t)HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL, &access_read);
    if (err != HSA_STATUS_SUCCESS) {
      return err;
    }

    if (((!access_read) && accessible_by_all == POOL_PROP_ON) ||
        (access_read && (accessible_by_all == POOL_PROP_OFF))) {
      return HSA_STATUS_SUCCESS;
    }
  }

  *(reinterpret_cast<hsa_amd_memory_pool_t*>(data)) = pool;
  return HSA_STATUS_INFO_BREAK;
}

hsa_status_t FindStandardPool(hsa_amd_memory_pool_t pool, void* data) {
  return FindPool(pool, data, HSA_AMD_SEGMENT_GLOBAL, POOL_PROP_DONT_CARE, POOL_PROP_OFF,
                  POOL_PROP_DONT_CARE);
}

hsa_status_t FindKernArgPool(hsa_amd_memory_pool_t pool, void* data) {
  return FindPool(pool, data, HSA_AMD_SEGMENT_GLOBAL, POOL_PROP_DONT_CARE, POOL_PROP_ON,
                  POOL_PROP_DONT_CARE);
}

}  // namespace rdc
}  // namespace amd
