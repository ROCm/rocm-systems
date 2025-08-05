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

/// \file
/// Implementation of utility functions used by RocR applications
#include "common/common.h"
#include <assert.h>
#include <sstream>
#include <string>
#include <memory>

namespace rocrtst {


#define RET_IF_HSA_COMMON_ERR(err) { \
  if ((err) != HSA_STATUS_SUCCESS) { \
    std::cout << "hsa api call failure at line " << __LINE__ << ", file: " << \
              __FILE__ << ". Call returned " << err << std::endl; \
    return (err); \
  } \
}

static hsa_status_t FindAgent(hsa_agent_t agent, void* data,
                                                hsa_device_type_t dev_type) {
  assert(data != nullptr);

  if (data == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  hsa_device_type_t hsa_device_type;
  hsa_status_t hsa_error_code = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE,
                                &hsa_device_type);
  RET_IF_HSA_COMMON_ERR(hsa_error_code);

  if (hsa_device_type == dev_type) {
    *(reinterpret_cast<hsa_agent_t*>(data)) = agent;
    return HSA_STATUS_INFO_BREAK;
  }

  return HSA_STATUS_SUCCESS;
}

// Find CPU Agents
hsa_status_t IterateCPUAgents(hsa_agent_t agent, void *data) {
  hsa_status_t status;
  assert(data != nullptr);
  if (data == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  std::vector<hsa_agent_t>* cpus = static_cast<std::vector<hsa_agent_t>*>(data);
  hsa_device_type_t device_type;
  status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
  RET_IF_HSA_COMMON_ERR(status);
  if (HSA_STATUS_SUCCESS == status && HSA_DEVICE_TYPE_CPU == device_type) {
    cpus->push_back(agent);
  }
  return status;
}



// Find GPU Agents
hsa_status_t IterateGPUAgents(hsa_agent_t agent, void *data) {
  hsa_status_t status;
  assert(data != nullptr);
  if (data == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  std::vector<hsa_agent_t>* gpus = static_cast<std::vector<hsa_agent_t>*>(data);
  hsa_device_type_t device_type;
  status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
  RET_IF_HSA_COMMON_ERR(status);
  if (HSA_STATUS_SUCCESS == status && HSA_DEVICE_TYPE_GPU == device_type) {
    gpus->push_back(agent);
  }
  return status;
}

// Find coarse grained device memory if this exists.  Fine grain otherwise.
hsa_status_t GetGlobalMemoryPool(hsa_amd_memory_pool_t pool, void* data) {
  hsa_amd_segment_t segment;
  hsa_status_t err;
  hsa_amd_memory_pool_t* ret = reinterpret_cast<hsa_amd_memory_pool_t*>(data);

  err = hsa_amd_memory_pool_get_info(pool,
                                         HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                                         &segment);
  RET_IF_HSA_COMMON_ERR(err);
  if (HSA_AMD_SEGMENT_GLOBAL != segment)
    return HSA_STATUS_SUCCESS;

  hsa_amd_memory_pool_global_flag_t flags;
  err = hsa_amd_memory_pool_get_info(pool,
                                        HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
                                        &flags);
  RET_IF_HSA_COMMON_ERR(err);

  // this is valid for dGPUs. But on APUs, it has to be FINE_GRAINED
  if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) {
    *ret = pool;
  } else {  // this is for APUs
    if ((ret == nullptr) && (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED)) {
      *ret = pool;
    }
  }
  return HSA_STATUS_SUCCESS;
}

// Find  a memory pool that can be used for kernarg locations.
hsa_status_t GetKernArgMemoryPool(hsa_amd_memory_pool_t pool, void* data) {
  hsa_status_t err;
  if (nullptr == data) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  hsa_amd_segment_t segment;
  err = hsa_amd_memory_pool_get_info(pool,
                                         HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                                         &segment);
  RET_IF_HSA_COMMON_ERR(err);
  if (HSA_AMD_SEGMENT_GLOBAL != segment) {
    return HSA_STATUS_SUCCESS;
  }

  hsa_amd_memory_pool_global_flag_t flags;
  err = hsa_amd_memory_pool_get_info(pool,
                                         HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
                                         &flags);
  RET_IF_HSA_COMMON_ERR(err);

  if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) {
    hsa_amd_memory_pool_t* ret =
                                reinterpret_cast<hsa_amd_memory_pool_t*>(data);
    *ret = pool;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t FindGPUDevice(hsa_agent_t agent, void* data) {
  return FindAgent(agent, data, HSA_DEVICE_TYPE_GPU);
}

hsa_status_t FindCPUDevice(hsa_agent_t agent, void* data) {
  return FindAgent(agent, data, HSA_DEVICE_TYPE_CPU);
}

/// Ennumeration that indicates whether a pool property must be present or not.
/// This is meant to be used by FindPool
typedef enum {
  POOL_PROP_OFF = 0,   ///< The property must be present.
  POOL_PROP_ON,        ///< The property must not be present.
  POOL_PROP_DONT_CARE  ///< We don't care if the property is present or not.
} pool_prop_t;

static hsa_status_t
FindPool(hsa_amd_memory_pool_t pool, void* data, hsa_amd_segment_t in_segment,
    pool_prop_t accessible_by_all, pool_prop_t kern_arg,
                                                    pool_prop_t fine_grain) {
  if (nullptr == data) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  hsa_status_t err;
  hsa_amd_segment_t segment;
  uint32_t flag;

  err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                                     &segment);
  RET_IF_HSA_COMMON_ERR(err);

  if (in_segment != segment) {
    return HSA_STATUS_SUCCESS;
  }

  if (HSA_AMD_SEGMENT_GLOBAL == in_segment) {
    err = hsa_amd_memory_pool_get_info(pool,
                               HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flag);
    RET_IF_HSA_COMMON_ERR(err);

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
    bool access_read;
    err = hsa_amd_memory_pool_get_info(pool,
          (hsa_amd_memory_pool_info_t)
                    HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL, &access_read);
    RET_IF_HSA_COMMON_ERR(err);

    if (((!access_read) && accessible_by_all == POOL_PROP_ON) ||
        (access_read  && (accessible_by_all == POOL_PROP_OFF))) {
      return HSA_STATUS_SUCCESS;
    }
  }

  *(reinterpret_cast<hsa_amd_memory_pool_t*>(data)) = pool;
  return HSA_STATUS_INFO_BREAK;
}

hsa_status_t FindStandardPool(hsa_amd_memory_pool_t pool, void* data) {
  return FindPool(pool, data, HSA_AMD_SEGMENT_GLOBAL, POOL_PROP_DONT_CARE,
                                          POOL_PROP_OFF, POOL_PROP_DONT_CARE);
}

hsa_status_t FindKernArgPool(hsa_amd_memory_pool_t pool, void* data) {
    return FindPool(pool, data, HSA_AMD_SEGMENT_GLOBAL, POOL_PROP_DONT_CARE,
                                            POOL_PROP_ON, POOL_PROP_DONT_CARE);
}
hsa_status_t FindGlobalPool(hsa_amd_memory_pool_t pool, void* data) {
  return FindPool(pool, data, HSA_AMD_SEGMENT_GLOBAL, POOL_PROP_ON,
                                          POOL_PROP_OFF, POOL_PROP_DONT_CARE);
}

hsa_status_t FindAPUStandardPool(hsa_amd_memory_pool_t pool, void* data) {
  return FindPool(pool, data, HSA_AMD_SEGMENT_GLOBAL, POOL_PROP_DONT_CARE,
                                          POOL_PROP_DONT_CARE, POOL_PROP_DONT_CARE);
}

// Populate the vector with handles to all agents and pools
hsa_status_t
GetAgentPools(std::vector<std::shared_ptr<agent_pools_t>> *agent_pools) {
  hsa_status_t err;

  assert(agent_pools != nullptr);

  auto save_agent = [](hsa_agent_t a, void *data)->hsa_status_t {
    std::vector<std::shared_ptr<agent_pools_t>> *ag_vec;
    hsa_status_t err;
    assert(data != nullptr);
    ag_vec =
        reinterpret_cast<std::vector<std::shared_ptr<agent_pools_t>> *>(data);
    std::shared_ptr<agent_pools_t> ag(new agent_pools_t);
    ag->agent = a;


    auto save_pool = [](hsa_amd_memory_pool_t p, void *data)->hsa_status_t {
      assert(data != nullptr);
      std::vector<hsa_amd_memory_pool_t> *p_list =
                 reinterpret_cast<std::vector<hsa_amd_memory_pool_t> *>(data);
      p_list->push_back(p);

      return HSA_STATUS_SUCCESS;
    };

    err = hsa_amd_agent_iterate_memory_pools(a, save_pool,
                                        reinterpret_cast<void *>(&ag->pools));
    ag_vec->push_back(ag);
    return err;
  };

  err = hsa_iterate_agents(save_agent, reinterpret_cast<void *>(agent_pools));
  return err;
}

static hsa_status_t MakeGlobalFlagsString(const pool_info_t *pool_i,
                                        std::string* out_str) {
  uint32_t global_flag = pool_i->global_flag;

  assert(out_str != nullptr);

  *out_str = "";

  std::vector < std::string > flags;

  if (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT & global_flag) {
    flags.push_back("KERNARG");
  }

  if (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED & global_flag) {
    flags.push_back("FINE GRAINED");
  }

  if (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED & global_flag) {
    flags.push_back("EXT-SCOPE FINE GRAINED");
  }

  if (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED & global_flag) {
    flags.push_back("COARSE GRAINED");
  }

  if (flags.size() > 0) {
    *out_str += flags[0];
  }

  for (size_t i = 1; i < flags.size(); i++) {
    *out_str += ", " + flags[i];
  }

  return HSA_STATUS_SUCCESS;
}
static hsa_status_t DumpSegment(const pool_info_t *pool_i,
                                 std::string const *ind_lvl) {
  hsa_status_t err;

  fprintf(stdout, "%s%-28s", ind_lvl->c_str(), "Pool Segment:");
  std::string seg_str = "";
  std::string tmp_str;

  switch (pool_i->segment) {
    case HSA_AMD_SEGMENT_GLOBAL:
      err = MakeGlobalFlagsString(pool_i, &tmp_str);
      RET_IF_HSA_COMMON_ERR(err);

      seg_str += "GLOBAL; FLAGS: " + tmp_str;
      break;

    case HSA_AMD_SEGMENT_READONLY:
      seg_str += "READONLY";
      break;

    case HSA_AMD_SEGMENT_PRIVATE:
      seg_str += "PRIVATE";
      break;

    case HSA_AMD_SEGMENT_GROUP:
      seg_str += "GROUP";
      break;

    default:
      std::cout << "Not Supported" << std::endl;
      break;
  }

  fprintf(stdout, "%-35s\n", seg_str.c_str());

  return HSA_STATUS_SUCCESS;
}

hsa_status_t AcquirePoolInfo(hsa_amd_memory_pool_t pool,
                                                        pool_info_t *pool_i) {
  hsa_status_t err;

  err = hsa_amd_memory_pool_get_info(pool,
                  HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &pool_i->global_flag);
  RET_IF_HSA_COMMON_ERR(err);

  err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                                                             &pool_i->segment);
  RET_IF_HSA_COMMON_ERR(err);

  // Get the size of the POOL
  err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SIZE,
                                                          &pool_i->size);
  RET_IF_HSA_COMMON_ERR(err);

#ifdef ROCRTST_EMULATOR_BUILD
  // Limit pool sizes to 2 GB on emulator
  const size_t max_pool_size = 2*1024*1024*1024UL;
  pool_i->size = std::min(pool_i->size, max_pool_size);
#endif

  err = hsa_amd_memory_pool_get_info(pool,
             HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
                                                      &pool_i->alloc_allowed);
  RET_IF_HSA_COMMON_ERR(err);

  err = hsa_amd_memory_pool_get_info(pool,
             HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,
                                                      &pool_i->alloc_granule);
  RET_IF_HSA_COMMON_ERR(err);

  err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE,
                                     &pool_i->alloc_rec_granule);
  RET_IF_HSA_COMMON_ERR(err);

  err = hsa_amd_memory_pool_get_info(pool,
                           HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT,
                                               &pool_i->alloc_alignment);
  RET_IF_HSA_COMMON_ERR(err);

  err = hsa_amd_memory_pool_get_info(pool,
                      HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL,
                                                  &pool_i->accessible_by_all);
  RET_IF_HSA_COMMON_ERR(err);

  err = hsa_amd_memory_pool_get_info(pool,
                       HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE,
                       &pool_i->aggregate_alloc_max);
  RET_IF_HSA_COMMON_ERR(err);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t DumpMemoryPoolInfo(const pool_info_t *pool_i,
                                uint32_t indent) {
  std::string ind_lvl(indent, ' ');

  DumpSegment(pool_i, &ind_lvl);

  std::string sz_str = std::to_string(pool_i->size / 1024) + "KB";
  fprintf(stdout, "%s%-28s%-36s\n", ind_lvl.c_str(), "Pool Size:",
          sz_str.c_str());

  fprintf(stdout, "%s%-28s%-36s\n", ind_lvl.c_str(), "Pool Allocatable:",
          (pool_i->alloc_allowed ? "TRUE" : "FALSE"));

  std::string gr_str = std::to_string(pool_i->alloc_granule / 1024) + "KB";
  fprintf(stdout, "%s%-28s%-36s\n", ind_lvl.c_str(), "Pool Alloc Granule:",
          gr_str.c_str());

  std::string recgr_str = std::to_string(pool_i->alloc_rec_granule / 1024) + "KB";
  fprintf(stdout, "%s%-28s%-36s\n", ind_lvl.c_str(),
          "Pool Alloc Recommended Granule:", recgr_str.c_str());

  std::string al_str =
                   std::to_string(pool_i->alloc_alignment / 1024) + "KB";
  fprintf(stdout, "%s%-28s%-36s\n", ind_lvl.c_str(), "Pool Alloc Alignment:",
          al_str.c_str());

  fprintf(stdout, "%s%-28s%-36s\n", ind_lvl.c_str(), "Pool Acessible by all:",
          (pool_i->accessible_by_all ? "TRUE" : "FALSE"));

  std::string agg_str =
              std::to_string(pool_i->aggregate_alloc_max / 1024) + "KB";
  fprintf(stdout, "%s%-28s%-36s\n", ind_lvl.c_str(), "Pool Aggregate Alloc Size:",
          agg_str.c_str());

  return HSA_STATUS_SUCCESS;
}

static const char* Types[] = {"HSA_EXT_POINTER_TYPE_UNKNOWN",
                              "HSA_EXT_POINTER_TYPE_HSA",
                              "HSA_EXT_POINTER_TYPE_LOCKED",
                              "HSA_EXT_POINTER_TYPE_GRAPHICS",
                              "HSA_EXT_POINTER_TYPE_IPC"
                             };

hsa_status_t DumpPointerInfo(void* ptr) {
  hsa_amd_pointer_info_t info;
  hsa_agent_t* agents;
  uint32_t count;
  hsa_status_t err;

  err = hsa_amd_pointer_info(ptr, &info, malloc, &count, &agents);
  RET_IF_HSA_COMMON_ERR(err);

  std::cout << "Info for ptr: " << ptr << std::endl;
  std::cout << "CPU ptr: " << reinterpret_cast<void*>(info.hostBaseAddress) <<
                                                                     std::endl;
  std::cout << "GPU ptr: " << reinterpret_cast<void*>(info.agentBaseAddress)
                                                                  << std::endl;
  std::cout << "Size: " << info.sizeInBytes << std::endl;
  std::cout << "Type: " << Types[info.type] << std::endl;
  std::cout << "UsrPtr " << reinterpret_cast<void*>(info.userData) <<
                                                                     std::endl;
  std::cout << "Accessible by: ";

  for (uint32_t i = 0; i < count; i++) {
    std::cout << agents[i].handle << " ";
  }

  std::cout << " ;[EOM]" << std::endl;
  free(agents);
  return HSA_STATUS_SUCCESS;
}


/*! \brief Writes to the buffer and increments the write pointer to the
 *         buffer. Also, ensures that the argument is written to an
 *         aligned memory as specified. Return the new write pointer.
 *
 * @param dst The write pointer to the buffer
 * @param src The source pointer
 * @param size The size in bytes to copy
 * @param alignment The alignment to follow while writing to the buffer
 */
#if 0
inline void *
addArg(void * dst, const void* src, size_t size, uint32_t alignment) {
    dst = rocrtst::AlignUp(dst, alignment);
    ::memcpy(dst, src, size);
    return dst + size;
}
#endif
#undef RET_IF_HSA_COMMON_ERR

}  // namespace rocrtst
