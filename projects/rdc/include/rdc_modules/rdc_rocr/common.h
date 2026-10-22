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
/// RocR related helper functions for sequeneces that come up frequently

#ifndef RDC_MODULES_RDC_ROCR_COMMON_H_
#define RDC_MODULES_RDC_ROCR_COMMON_H_

#include <stdio.h>
#include <string.h>

#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

namespace amd {
namespace rdc {

// define below should be deleted. Leaving in commented out until code that
// refers to it has been corrected
// #define HSA_ARGUMENT_ALIGN_BYTES 16

/// What a sub-test asks of a pool and what the search found: ROCr refuses an allocation above a
/// pool's maximum before it allocates anything, so a pool that cannot hand out what the sub-test
/// writes is not a pool it can use.
struct PoolRequest {
  uint64_t wanted_bytes = 0;
  hsa_amd_memory_pool_t pool = {};
};

/// If the provided agent is associated with a CPU, return that agent through
/// output parameter. This function is meant to be the call-back function used
/// with hsa_iterate_agents to find all the CPU agents.
/// \param[in] agent Agent to evaluate if CPU
/// \param[out] data If agent is associated with a CPU, this pointer will point
///  to the agent upon return
/// \returns HSA_STATUS_SUCCESS if no errors are encountered.
hsa_status_t IterateCPUAgents(hsa_agent_t agent, void* data);

/// If the provided agent is associated with a GPU, return that agent through
/// output parameter. This function is meant to be the call-back function used
/// with hsa_iterate_agents to find  all the GPU agents.
/// \param[in] agent Agent to evaluate if GPU
/// \param[out] data If agent is associated with a GPU, this pointer will point
///  to the agent upon return
/// \returns HSA_STATUS_SUCCESS if no errors are encountered.
hsa_status_t IterateGPUAgents(hsa_agent_t agent, void* data);

/// Find a GLOBAL memory pool. By this, we mean not a kernel args pool.
/// This function is meant to be the call-back function used
/// with hsa_amd_agent_iterate_memory_pools. It records the last pool that
/// meets the criteria, allocation allowed and a maximum allocation that covers
/// the request's bytes among them, and never stops the iteration, so the caller
/// decides from what it recorded whether the agent has such a pool.
/// \param[in] pool Pool to evaluate for required properties
/// \param[in] data The PoolRequest this search is for, whose pool is written
///  when the pool meets the criteria
/// \returns hsa_status_t
///      -HSA_STATUS_SUCCESS - the pool was evaluated
///      -else the status of the query that failed
hsa_status_t GetGlobalMemoryPool(hsa_amd_memory_pool_t pool, void* data);

/// Find a "kernel arg" pool.
/// This function is meant to be the call-back function used
/// with hsa_amd_agent_iterate_memory_pools. It records the last pool that
/// meets the criteria and never stops the iteration, so the caller decides
/// from what it recorded whether the agent has such a pool.
/// \param[in] pool Pool to evaluate for required properties
/// \param[in] data If pool meets criteria, this pointer will point
///  to the pool upon return
/// \returns hsa_status_t
///      -HSA_STATUS_SUCCESS - the pool was evaluated
///      -else the status of the query that failed
hsa_status_t GetKernArgMemoryPool(hsa_amd_memory_pool_t pool, void* data);

/// Find a "standard" pool. By this, we mean not a kernel args pool.
/// The pool found will have the following properties:
///     HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL: Don't care
///     HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT: Off
///     HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED: Don't care
/// This function is meant to be the call-back function used
/// with hsa_amd_agent_iterate_memory_pools.
/// \param[in] pool Pool to evaluate for required properties
/// \param[in] data If pool meets criteria, this pointer will point
///  to the pool upon return
/// \returns hsa_status_t
///      -HSA_STATUS_INFO_BREAK - we found a pool that meets criteria
///      -HSA_STATUS_SUCCESS - we did not find a pool that meets the criteria
///      -else return an appropriate error code for any error encountered
hsa_status_t FindStandardPool(hsa_amd_memory_pool_t pool, void* data);

/// Find a "kernel arg" pool.
/// The pool found will have the following properties:
///     HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL: Don't care
///     HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT: On
///     HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED: Don't care
/// This function is meant to be the call-back function used
/// with hsa_amd_agent_iterate_memory_pools.
/// \param[in] pool Pool to evaluate for required properties
/// \param[in] data If pool meets criteria, this pointer will point
///  to the pool upon return
/// \returns hsa_status_t
///      -HSA_STATUS_INFO_BREAK - we found a pool that meets criteria
///      -HSA_STATUS_SUCCESS - we did not find a pool that meets the criteria
///      -else return an appropriate error code for any error encountered
hsa_status_t FindKernArgPool(hsa_amd_memory_pool_t pool, void* data);

// The sentence a failed HSA call gives a GPU: the message, the status as a number and, when
// ROCr has a string for that status, that string. A status it has none for leaves its output
// untouched (hsa.cpp), so the number is what names it.
std::string hsa_error_text(const std::string& msg, hsa_status_t err);

void throw_if_error(hsa_status_t err, const std::string& msg = "");

void throw_if_skip(const std::string& msg);

// A failure of the check itself, with the sentence that names it, where no HSA call failed.
void throw_failure(const std::string& msg);

// The customize exception when the test has to be skipped
class SkipException : public std::exception {
 public:
  explicit SkipException(const char* msg) : _msg(msg) {}
  virtual const char* what() const noexcept { return _msg.c_str(); }

 private:
  std::string _msg;
};

}  // namespace rdc
}  // namespace amd

#endif  // RDC_MODULES_RDC_ROCR_COMMON_H_
