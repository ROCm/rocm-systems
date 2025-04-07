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
#include <stdio.h>
#include <vector>
#include <string>
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

#define RET_IF_HSA_ERR(err) { \
  if ((err) != HSA_STATUS_SUCCESS) { \
    printf("hsa api call failure at line %d, file: %s. Call returned %d\n", \
                                                   __LINE__, __FILE__, err); \
    return (err); \
  } \
}

// This structure holds system information acquired through hsa info related
// calls, and is later used for reference when displaying the information.
struct system_info_t {
    uint16_t major, minor;
    uint64_t timestamp_frequency = 0;
    uint64_t max_wait = 0;
    hsa_endianness_t endianness;
    hsa_machine_model_t machine_model;
};

// This structure holds agent information acquired through hsa info related
// calls, and is later used for reference when displaying the information.
struct agent_info_t {
  char name[64];
  char vendor_name[64];
  hsa_agent_feature_t agent_feature;
  hsa_profile_t agent_profile;
  hsa_default_float_rounding_mode_t float_rounding_mode;
  uint32_t max_queue;
  uint32_t queue_min_size;
  uint32_t queue_max_size;
  hsa_queue_type_t queue_type;
  uint32_t node;
  hsa_device_type_t device_type;
  uint32_t cache_size[4];
  uint32_t chip_id;
  uint32_t cacheline_size;
  uint32_t max_clock_freq;
  uint32_t compute_unit;
  uint32_t wavefront_size;
  uint32_t workgroup_max_size;
  uint32_t grid_max_size;
  uint32_t fbarrier_max_size;
  uint32_t waves_per_cu;
  hsa_isa_t agent_isa;
  hsa_dim3_t grid_max_dim;
  uint16_t workgroup_max_dim[3];
  uint16_t bdf_id;
  bool fast_f16;
};

// This structure holds memory pool information acquired through hsa info
// related calls, and is later used for reference when displaying the
// information.
struct pool_info_t {
    uint32_t segment;
    size_t pool_size;
    bool alloc_allowed;
    size_t alloc_granule;
    size_t alloc_recommended_granule;
    size_t pool_alloc_alignment;
    bool pl_access;
    uint32_t global_flag;
};

// This structure holds ISA information acquired through hsa info
// related calls, and is later used for reference when displaying the
// information.
struct isa_info_t {
    char *name_str;
    uint32_t workgroup_max_size;
    hsa_dim3_t grid_max_dim;
    uint64_t grid_max_size;
    uint32_t fbarrier_max_size;
    uint16_t workgroup_max_dim[3];
    bool def_rounding_modes[3];
    bool base_rounding_modes[3];
    bool mach_models[2];
    bool profiles[2];
    bool fast_f16;
};

// This structure holds cache information acquired through hsa info
// related calls, and is later used for reference when displaying the
// information.
struct cache_info_t {
    char *name_str;
    uint8_t level;
    uint32_t size;
};

static const uint32_t kLabelFieldSize = 25;
static const uint32_t kValueFieldSize = 35;
static const uint32_t kIndentSize = 2;

static void printLabelInt(char const *l, int d, uint32_t indent_lvl = 0) {
  std::string ind(kIndentSize * indent_lvl, ' ');

  printf("%s%-*s%-*u\n", ind.c_str(), kLabelFieldSize, l, kValueFieldSize, d);
}
static void printLabelStr(char const *l, char const *s,
                                                    uint32_t indent_lvl = 0) {
  std::string ind(kIndentSize * indent_lvl, ' ');
  printf("%s%-*s%-*s\n", ind.c_str(), kLabelFieldSize, l, kValueFieldSize, s);
}
static void printLabel(char const *l, bool newline = false,
                                                    uint32_t indent_lvl = 0) {
  std::string ind(kIndentSize * indent_lvl, ' ');

  printf("%s%-*s", ind.c_str(), kLabelFieldSize, l);

  if (newline) {
    printf("\n");
  }
}
static void printValueStr(char const *s, bool newline = true) {
  printf("%-*s\n", kValueFieldSize, s);
}

// Acquire system information
static hsa_status_t AcquireSystemInfo(system_info_t *sys_info) {
  hsa_status_t err;

  // Get Major and Minor version of runtime
  err = hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MAJOR, &sys_info->major);
  RET_IF_HSA_ERR(err);
  err = hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MINOR, &sys_info->minor);
  RET_IF_HSA_ERR(err);

  // Get timestamp frequency
  err = hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY,
                                              &sys_info->timestamp_frequency);
  RET_IF_HSA_ERR(err);

  // Get maximum duration of a signal wait operation
  err = hsa_system_get_info(HSA_SYSTEM_INFO_SIGNAL_MAX_WAIT,
                                                         &sys_info->max_wait);
  RET_IF_HSA_ERR(err);

  // Get Endianness of the system
  err = hsa_system_get_info(HSA_SYSTEM_INFO_ENDIANNESS, &sys_info->endianness);
  RET_IF_HSA_ERR(err);

  // Get machine model info
  err = hsa_system_get_info(HSA_SYSTEM_INFO_MACHINE_MODEL,
                                                     &sys_info->machine_model);
  RET_IF_HSA_ERR(err);
  return err;
}

static void DisplaySystemInfo(system_info_t const *sys_info) {
  printLabel("Runtime Version:");
  printf("%d.%d\n", sys_info->major, sys_info->minor);
  printLabel("System Timestamp Freq.:");
  printf("%fMHz\n", sys_info->timestamp_frequency / 1e6);
  printLabel("Sig. Max Wait Duration:");
  printf("%lu (number of timestamp)\n", sys_info->max_wait);

  printLabel("Machine Model:");
  if (HSA_MACHINE_MODEL_SMALL == sys_info->machine_model) {
    printValueStr("SMALL");
  } else if (HSA_MACHINE_MODEL_LARGE == sys_info->machine_model) {
    printValueStr("LARGE");
  }

  printLabel("System Endianness:");
  if (HSA_ENDIANNESS_LITTLE == sys_info->endianness) {
    printValueStr("LITTLE");
  } else if (HSA_ENDIANNESS_BIG == sys_info->endianness) {
    printValueStr("BIG");
  }
  printf("\n");
}

static hsa_status_t
AcquireAgentInfo(hsa_agent_t agent, agent_info_t *agent_i) {
  hsa_status_t err;
  // Get agent name and vendor
  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, agent_i->name);
  RET_IF_HSA_ERR(err);
  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_VENDOR_NAME,
                                                       &agent_i->vendor_name);
  RET_IF_HSA_ERR(err);

  // Get agent feature
  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_FEATURE,
                                                     &agent_i->agent_feature);
  RET_IF_HSA_ERR(err);

  // Get profile supported by the agent
  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_PROFILE,
                                                     &agent_i->agent_profile);
  RET_IF_HSA_ERR(err);

  // Get floating-point rounding mode
  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE,
                                               &agent_i->float_rounding_mode);
  RET_IF_HSA_ERR(err);

  // Get max number of queue
  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUES_MAX,
                                                         &agent_i->max_queue);
  RET_IF_HSA_ERR(err);

  // Get queue min size
  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUE_MIN_SIZE,
                                                    &agent_i->queue_min_size);
  RET_IF_HSA_ERR(err);

  // Get queue max size
  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE,
                                                    &agent_i->queue_max_size);
  RET_IF_HSA_ERR(err);

  // Get queue type
  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUE_TYPE,
                                                        &agent_i->queue_type);
  RET_IF_HSA_ERR(err);

  // Get agent node
  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_NODE, &agent_i->node);
  RET_IF_HSA_ERR(err);

  // Get device type
  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE,
                                                       &agent_i->device_type);
  RET_IF_HSA_ERR(err);

  if (HSA_DEVICE_TYPE_GPU == agent_i->device_type) {
    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_ISA, &agent_i->agent_isa);
    RET_IF_HSA_ERR(err);
  }

  // Get cache size
  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_CACHE_SIZE,
                                                        agent_i->cache_size);
  RET_IF_HSA_ERR(err);

  // Get chip id
  err = hsa_agent_get_info(agent,
                           (hsa_agent_info_t) HSA_AMD_AGENT_INFO_CHIP_ID,
                                                           &agent_i->chip_id);
  RET_IF_HSA_ERR(err);

  // Get cacheline size
  err = hsa_agent_get_info(agent,
                       (hsa_agent_info_t) HSA_AMD_AGENT_INFO_CACHELINE_SIZE,
                                                    &agent_i->cacheline_size);
  RET_IF_HSA_ERR(err);

  // Get Max clock frequency
  err = hsa_agent_get_info(agent,
                  (hsa_agent_info_t) HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY,
                                                    &agent_i->max_clock_freq);
  RET_IF_HSA_ERR(err);

  // Get Agent BDFID
  err = hsa_agent_get_info(agent,
                (hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID, &agent_i->bdf_id);
  RET_IF_HSA_ERR(err);

  // Get number of Compute Unit
  err = hsa_agent_get_info(agent,
                   (hsa_agent_info_t) HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT,
                                                      &agent_i->compute_unit);
  RET_IF_HSA_ERR(err);

  // Check if the agent is kernel agent
  if (agent_i->agent_feature & HSA_AGENT_FEATURE_KERNEL_DISPATCH) {
    // Get flaf of fast_f16 operation
    err = hsa_agent_get_info(agent,
                       HSA_AGENT_INFO_FAST_F16_OPERATION, &agent_i->fast_f16);
    RET_IF_HSA_ERR(err);

    // Get wavefront size
    err = hsa_agent_get_info(agent,
                     HSA_AGENT_INFO_WAVEFRONT_SIZE, &agent_i->wavefront_size);
    RET_IF_HSA_ERR(err);

    // Get max total number of work-items in a workgroup
    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_WORKGROUP_MAX_SIZE,
                                                &agent_i->workgroup_max_size);
    RET_IF_HSA_ERR(err);

    // Get max number of work-items of each dimension of a work-group
    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_WORKGROUP_MAX_DIM,
                                                 &agent_i->workgroup_max_dim);
    RET_IF_HSA_ERR(err);

    // Get max number of a grid per dimension
    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_GRID_MAX_DIM,
                                                      &agent_i->grid_max_dim);
    RET_IF_HSA_ERR(err);

    // Get max total number of work-items in a grid
    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_GRID_MAX_SIZE,
                                                     &agent_i->grid_max_size);
    RET_IF_HSA_ERR(err);

    // Get max number of fbarriers per work group
    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_FBARRIER_MAX_SIZE,
                                                 &agent_i->fbarrier_max_size);
    RET_IF_HSA_ERR(err);

    err = hsa_agent_get_info(agent,
                    (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU,
                                                      &agent_i->waves_per_cu);
    RET_IF_HSA_ERR(err);
  }
  return err;
}

static void DisplayAgentInfo(agent_info_t *agent_i) {
  printLabelStr("Name:", agent_i->name, 1);
  printLabelStr("Vendor Name:", agent_i->vendor_name, 1);

  printLabel("Feature:", false, 1);
  if (agent_i->agent_feature & HSA_AGENT_FEATURE_KERNEL_DISPATCH
      && agent_i->agent_feature & HSA_AGENT_FEATURE_AGENT_DISPATCH) {
    printValueStr("KERNEL_DISPATCH & AGENT_DISPATCH");
  } else if (agent_i->agent_feature & HSA_AGENT_FEATURE_KERNEL_DISPATCH) {
    printValueStr("KERNEL_DISPATCH");
  } else if (agent_i->agent_feature & HSA_AGENT_FEATURE_AGENT_DISPATCH) {
    printValueStr("AGENT_DISPATCH");
  } else {
    printValueStr("None specified");
  }

  printLabel("Profile:", false, 1);
  if (HSA_PROFILE_BASE == agent_i->agent_profile) {
    printValueStr("BASE_PROFILE");
  } else if (HSA_PROFILE_FULL == agent_i->agent_profile) {
    printValueStr("FULL_PROFILE");
  } else {
    printValueStr("Unknown");
  }

  printLabel("Float Round Mode:", false, 1);
  if (HSA_DEFAULT_FLOAT_ROUNDING_MODE_ZERO == agent_i->float_rounding_mode) {
    printValueStr("ZERO");
  } else if (HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR ==
                                               agent_i->float_rounding_mode) {
    printValueStr("NEAR");
  } else {
    printValueStr("Not Supported");
  }

  printLabelInt("Max Queue Number:", agent_i->max_queue, 1);
  printLabelInt("Queue Min Size:", agent_i->queue_min_size, 1);
  printLabelInt("Queue Max Size:", agent_i->queue_max_size, 1);

  if (HSA_QUEUE_TYPE_MULTI == agent_i->queue_type) {
    printLabelStr("Queue Type:", "MULTI", 1);
  } else if (HSA_QUEUE_TYPE_SINGLE == agent_i->queue_type) {
    printLabelStr("Queue Type:", "SINGLE", 1);
  } else {
    printLabelStr("Queue Type:", "Unknown", 1);
  }

  printLabelInt("Node:", agent_i->node, 1);

  printLabel("Device Type:", false, 1);
  if (HSA_DEVICE_TYPE_CPU == agent_i->device_type) {
    printValueStr("CPU");
  } else if (HSA_DEVICE_TYPE_GPU == agent_i->device_type) {
    printValueStr("GPU");
  } else {
    printValueStr("DSP");
  }

  printLabel("Cache Info:", true, 1);

  for (int i = 0; i < 4; i++) {
    if (agent_i->cache_size[i]) {
      std::string tmp_str("L");
      tmp_str += std::to_string(i+1);
      tmp_str += ":";
      printLabel(tmp_str.c_str(), false, 2);

      tmp_str = std::to_string(agent_i->cache_size[i]/1024);
      tmp_str += "KB";
      printValueStr(tmp_str.c_str());
    }
  }

  printLabelInt("Chip ID:", agent_i->chip_id, 1);
  printLabelInt("Cacheline Size:", agent_i->cacheline_size, 1);
  printLabelInt("Max Clock Frequency (MHz):", agent_i->max_clock_freq, 1);
  printLabelInt("BDFID:", agent_i->bdf_id, 1);
  printLabelInt("Compute Unit:", agent_i->compute_unit, 1);

  printLabel("Features:", false, 1);
  if (agent_i->agent_feature & HSA_AGENT_FEATURE_KERNEL_DISPATCH) {
    printf("%s", "KERNEL_DISPATCH ");
  }
  if (agent_i->agent_feature & HSA_AGENT_FEATURE_AGENT_DISPATCH) {
    printf("%s", "AGENT_DISPATCH");
  }
  if (agent_i->agent_feature == 0) {
    printf("None");
  }
  printf("\n");

  if (agent_i->agent_feature & HSA_AGENT_FEATURE_KERNEL_DISPATCH) {
    printLabelStr("Fast F16 Operation:", agent_i->fast_f16 ? "TRUE":"FALSE", 1);

    printLabelInt("Wavefront Size:", agent_i->wavefront_size, 1);
    printLabelInt("Workgroup Max Size:", agent_i->workgroup_max_size, 1);

    printLabel("Workgroup Max Size Per Dimension:", true, 1);
    std::string dim;
    for (int i = 0; i < 3; i++) {
      dim = "Dim[" + std::to_string(i) + "]:";
      printLabelInt(dim.c_str(),
              reinterpret_cast<uint32_t*>(&agent_i->workgroup_max_dim)[i], 2);
    }
    printLabelInt("Grid Max Size:", agent_i->grid_max_size, 1);
    printLabelInt("Waves Per CU:", agent_i->waves_per_cu, 1);
    printLabelInt("Max Work-item Per CU:",
                            agent_i->wavefront_size*agent_i->waves_per_cu, 1);
    printLabel("Grid Max Size per Dimension:", true, 1);
    for (int i = 0; i < 3; i++) {
      dim = "Dim[" + std::to_string(i) + "]:";
      printLabelInt(dim.c_str(),
                 reinterpret_cast<uint32_t*>(&agent_i->grid_max_dim)[i], 2);
    }

    printLabelInt("Max number Of fbarriers Per Workgroup:",
                                             agent_i->fbarrier_max_size, 1);
  }
}

static hsa_status_t AcquirePoolInfo(hsa_amd_memory_pool_t pool,
                                                        pool_info_t *pool_i) {
  hsa_status_t err;

  err = hsa_amd_memory_pool_get_info(pool,
                  HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &pool_i->global_flag);
  RET_IF_HSA_ERR(err);

  err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                                                             &pool_i->segment);
  RET_IF_HSA_ERR(err);

  // Get the size of the POOL
  err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SIZE,
                                                          &pool_i->pool_size);
  RET_IF_HSA_ERR(err);

  err = hsa_amd_memory_pool_get_info(pool,
             HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
                                                      &pool_i->alloc_allowed);
  RET_IF_HSA_ERR(err);

  err = hsa_amd_memory_pool_get_info(pool,
             HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,
                                                      &pool_i->alloc_granule);
  RET_IF_HSA_ERR(err);

  err = hsa_amd_memory_pool_get_info(pool,
                           HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT,
                                               &pool_i->pool_alloc_alignment);
  RET_IF_HSA_ERR(err);

  err =
      hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE,
                                   &pool_i->alloc_recommended_granule);
  RET_IF_HSA_ERR(err);

  err = hsa_amd_memory_pool_get_info(pool,
                      HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL,
                                                          &pool_i->pl_access);
  RET_IF_HSA_ERR(err);

  return HSA_STATUS_SUCCESS;
}

static void MakeGlobalFlagsString(uint32_t global_flag, std::string* out_str) {
  *out_str = "";

  std::vector<std::string> flags;

  if (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT & global_flag) {
    flags.push_back("KERNARG");
  }

  if (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED & global_flag) {
    flags.push_back("FINE GRAINED");
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
}

static void DumpSegment(pool_info_t *pool_i, uint32_t ind_lvl) {
  std::string seg_str;
  std::string tmp_str;

  printLabel("Segment:", false, ind_lvl);

  switch (pool_i->segment) {
    case HSA_AMD_SEGMENT_GLOBAL:
      MakeGlobalFlagsString(pool_i->global_flag, &tmp_str);
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
      printf("Not Supported\n");
      break;
  }
  printValueStr(seg_str.c_str());
}

static void DisplayPoolInfo(pool_info_t *pool_i, uint32_t indent) {
  DumpSegment(pool_i, indent);

  std::string sz_str = std::to_string(pool_i->pool_size/1024) + "KB";
  printLabelStr("Size:", sz_str.c_str(), indent);
  printLabelStr("Allocatable:", (pool_i->alloc_allowed ? "TRUE" : "FALSE"),
                                                                      indent);
  std::string gr_str = std::to_string(pool_i->alloc_granule/1024)+"KB";
  printLabelStr("Alloc Granule:", gr_str.c_str(), indent);

  std::string al_str = std::to_string(pool_i->pool_alloc_alignment/1024)+"KB";
  printLabelStr("Alloc Alignment:", al_str.c_str(), indent);

  printLabelStr("Acessible by all:", (pool_i->pl_access ? "TRUE" : "FALSE"),
                                                                      indent);
}

static hsa_status_t
AcquireAndDisplayMemPoolInfo(const hsa_amd_memory_pool_t pool,
                                                            uint32_t indent) {
  hsa_status_t err;
  pool_info_t pool_i;

  err = AcquirePoolInfo(pool, &pool_i);
  RET_IF_HSA_ERR(err);

  DisplayPoolInfo(&pool_i, 3);

  return err;
}

static hsa_status_t get_pool_info(hsa_amd_memory_pool_t pool, void* data) {
  hsa_status_t err;
  int* p_int = reinterpret_cast<int*>(data);
  (*p_int)++;

  std::string pool_str("Pool ");
  pool_str += std::to_string(*p_int);
  printLabel(pool_str.c_str(), true, 2);

  err = AcquireAndDisplayMemPoolInfo(pool, 3);
  RET_IF_HSA_ERR(err);

  return err;
}

static hsa_status_t AcquireISAInfo(hsa_isa_t isa, isa_info_t *isa_i) {
  hsa_status_t err;
  uint32_t name_len;
  err = hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME_LENGTH, &name_len);
  RET_IF_HSA_ERR(err);

  isa_i->name_str = new char[name_len];
  if (isa_i->name_str == nullptr) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  err = hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, isa_i->name_str);
  RET_IF_HSA_ERR(err);

  err = hsa_isa_get_info_alt(isa, HSA_ISA_INFO_MACHINE_MODELS,
                                                          isa_i->mach_models);
  RET_IF_HSA_ERR(err);

  err = hsa_isa_get_info_alt(isa, HSA_ISA_INFO_PROFILES, isa_i->profiles);
  RET_IF_HSA_ERR(err);

  err = hsa_isa_get_info_alt(isa, HSA_ISA_INFO_DEFAULT_FLOAT_ROUNDING_MODES,
                                                   isa_i->def_rounding_modes);
  RET_IF_HSA_ERR(err);

  err = hsa_isa_get_info_alt(isa,
                    HSA_ISA_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES,
                                                  isa_i->base_rounding_modes);
  RET_IF_HSA_ERR(err);

  err = hsa_isa_get_info_alt(isa, HSA_ISA_INFO_FAST_F16_OPERATION,
                                                            &isa_i->fast_f16);
  RET_IF_HSA_ERR(err);

  err = hsa_isa_get_info_alt(isa, HSA_ISA_INFO_WORKGROUP_MAX_DIM,
                                                   &isa_i->workgroup_max_dim);
  RET_IF_HSA_ERR(err);

  err = hsa_isa_get_info_alt(isa, HSA_ISA_INFO_WORKGROUP_MAX_SIZE,
                                                  &isa_i->workgroup_max_size);
  RET_IF_HSA_ERR(err);

  err = hsa_isa_get_info_alt(isa, HSA_ISA_INFO_GRID_MAX_DIM,
                                                        &isa_i->grid_max_dim);
  RET_IF_HSA_ERR(err);

  err = hsa_isa_get_info_alt(isa, HSA_ISA_INFO_GRID_MAX_SIZE,
                                                        &isa_i->grid_max_size);
  RET_IF_HSA_ERR(err);

  err = hsa_isa_get_info_alt(isa, HSA_ISA_INFO_FBARRIER_MAX_SIZE,
                                                    &isa_i->fbarrier_max_size);
  RET_IF_HSA_ERR(err);

  return err;
}

static void DisplayISAInfo(isa_info_t *isa_i, uint32_t indent) {
  printLabelStr("Name:", isa_i->name_str, indent);

  std::string models("");
  if (isa_i->mach_models[HSA_MACHINE_MODEL_SMALL]) {
    models = "HSA_MACHINE_MODEL_SMALL ";
  }
  if (isa_i->mach_models[HSA_MACHINE_MODEL_LARGE]) {
    models += "HSA_MACHINE_MODEL_LARGE";
  }
  printLabelStr("Machine Models:", models.c_str(), indent);

  std::string profiles("");
  if (isa_i->profiles[HSA_PROFILE_BASE]) {
    profiles = "HSA_PROFILE_BASE ";
  }
  if (isa_i->profiles[HSA_PROFILE_FULL]) {
    profiles += "HSA_PROFILE_FULL";
  }
  printLabelStr("Profiles:", profiles.c_str(), indent);

  std::string rounding_modes("");
  if (isa_i->def_rounding_modes[HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT]) {
    rounding_modes = "DEFAULT ";
  }
  if (isa_i->def_rounding_modes[HSA_DEFAULT_FLOAT_ROUNDING_MODE_ZERO]) {
    rounding_modes += "ZERO ";
  }
  if (isa_i->def_rounding_modes[HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR]) {
    rounding_modes += "NEAR";
  }
  printLabelStr("Default Rounding Mode:", rounding_modes.c_str(), indent);

  rounding_modes = "";
  if (isa_i->base_rounding_modes[HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT]) {
    rounding_modes = "DEFAULT ";
  }
  if (isa_i->base_rounding_modes[HSA_DEFAULT_FLOAT_ROUNDING_MODE_ZERO]) {
    rounding_modes += "ZERO ";
  }
  if (isa_i->base_rounding_modes[HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR]) {
    rounding_modes += "NEAR";
  }
  printLabelStr("Default Rounding Mode:", rounding_modes.c_str(), indent);

  printLabelStr("Fast f16:", (isa_i->fast_f16 ? "TRUE" : "FALSE"), indent);

  printLabel("Workgroup Max Dimension:", true, indent);
  std::string dim;
  for (int i = 0; i < 3; i++) {
    dim = "Dim[" + std::to_string(i) + "]:";
    printLabelInt(dim.c_str(),
         reinterpret_cast<uint32_t*>(&isa_i->workgroup_max_dim)[i], indent+1);
  }

  printLabelInt("Workgroup Max Size:", isa_i->workgroup_max_size, indent);

  printLabel("Grid Max Dimension:", true, indent);
  printLabelInt("x", isa_i->grid_max_dim.x, indent+1);
  printLabelInt("y", isa_i->grid_max_dim.y, indent+1);
  printLabelInt("z", isa_i->grid_max_dim.z, indent+1);

  printLabelInt("Grid Max Size:", isa_i->grid_max_size, indent);
  printLabelInt("FBarrier Max Size:", isa_i->fbarrier_max_size, indent);
}

static hsa_status_t
AcquireAndDisplayISAInfo(const hsa_isa_t isa, uint32_t indent) {
  hsa_status_t err;
  isa_info_t isa_i;

  isa_i.name_str = nullptr;
  err = AcquireISAInfo(isa, &isa_i);
  RET_IF_HSA_ERR(err);

  DisplayISAInfo(&isa_i, 3);

  if (isa_i.name_str != nullptr) {
    delete []isa_i.name_str;
  }
  return err;
}
static hsa_status_t get_isa_info(hsa_isa_t isa, void* data) {
  hsa_status_t err;
  int* isa_int = reinterpret_cast<int*>(data);
  (*isa_int)++;

  std::string isa_str("ISA ");
  isa_str += std::to_string(*isa_int);
  printLabel(isa_str.c_str(), true, 2);

  err = AcquireAndDisplayISAInfo(isa, 3);
  RET_IF_HSA_ERR(err);

  return err;
}
// Cache info dump is ifdef'd out as it generates a lot of output that is
// not that interesting. Define ENABLE_CACHE_DUMP if this is of interest.
#ifdef ENABLE_CACHE_DUMP
static void DisplayCacheInfo(cache_info_t *cache_i, uint32_t indent) {
  printLabelStr("Name:", cache_i->name_str, indent);

  printLabelInt("Level:", cache_i->level, indent);
  printLabelInt("Size:", cache_i->size, indent);
}

static hsa_status_t AcquireCacheInfo(hsa_cache_t cache, cache_info_t *cache_i) {
  hsa_status_t err;
  uint32_t name_len;
  err = hsa_cache_get_info(cache, HSA_CACHE_INFO_NAME_LENGTH, &name_len);
  RET_IF_HSA_ERR(err);

  cache_i->name_str = new char[name_len];
  if (cache_i->name_str == nullptr) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  err = hsa_cache_get_info(cache, HSA_CACHE_INFO_NAME, cache_i->name_str);
  RET_IF_HSA_ERR(err);

  err = hsa_cache_get_info(cache, HSA_CACHE_INFO_LEVEL, &cache_i->level);
  RET_IF_HSA_ERR(err);

  err = hsa_cache_get_info(cache, HSA_CACHE_INFO_SIZE, &cache_i->size);
  RET_IF_HSA_ERR(err);
  return err;
}

static hsa_status_t
AcquireAndDisplayCacheInfo(const hsa_cache_t cache, uint32_t indent) {
  hsa_status_t err;
  cache_info_t cache_i;

  err = AcquireCacheInfo(cache, &cache_i);
  RET_IF_HSA_ERR(err);

  DisplayCacheInfo(&cache_i, 3);

  if (cache_i.name_str != nullptr) {
    delete []cache_i.name_str;
  }

  return err;
}

static hsa_status_t get_cache_info(hsa_cache_t cache, void* data) {
  hsa_status_t err;
  int* cache_int = reinterpret_cast<int*>(data);
  (*cache_int)++;

  std::string cache_str("Cache L");
  cache_str += std::to_string(*cache_int);
  printLabel(cache_str.c_str(), true, 2);

  err = AcquireAndDisplayCacheInfo(cache, 3);
  RET_IF_HSA_ERR(err);

  return err;
}
#endif  // ENABLE_CACHE_DUMP
static hsa_status_t
AcquireAndDisplayAgentInfo(hsa_agent_t agent, void* data) {
  int pool_number = 0;
  int isa_number = 0;

  hsa_status_t err;
  agent_info_t agent_i;

  int *agent_number = reinterpret_cast<int*>(data);
  (*agent_number)++;

  err = AcquireAgentInfo(agent, &agent_i);
  RET_IF_HSA_ERR(err);

  std::string ind(kIndentSize, ' ');

  printLabel("*******", true);
  std::string agent_ind("Agent ");
  agent_ind += std::to_string(*agent_number).c_str();
  printLabel(agent_ind.c_str(), true);
  printLabel("*******", true);

  DisplayAgentInfo(&agent_i);

  printLabel("Pool Info:", true, 1);
  err = hsa_amd_agent_iterate_memory_pools(agent, get_pool_info, &pool_number);
  RET_IF_HSA_ERR(err);

  printLabel("ISA Info:", true, 1);
  err = hsa_agent_iterate_isas(agent, get_isa_info, &isa_number);
  if (err == HSA_STATUS_ERROR_INVALID_AGENT) {
    printLabel("N/A", true, 2);
    return HSA_STATUS_SUCCESS;
  }
  RET_IF_HSA_ERR(err);

#if ENABLE_CACHE_DUMP
  int cache_number = 0;
  printLabel("Cache Info:", true, 1);
  err = hsa_agent_iterate_caches(agent, get_cache_info, &cache_number);
  if (err == HSA_STATUS_ERROR_INVALID_AGENT) {
    printLabel("N/A", true, 2);
    return HSA_STATUS_SUCCESS;
  }
#endif
  RET_IF_HSA_ERR(err);

  return HSA_STATUS_SUCCESS;
}

// Print out all static information known to HSA about the target system.
// Throughout this program, the Acquire-type functions make HSA calls to
// interate through HSA objects and then perform HSA get_info calls to
// acccumulate information about those objects. Corresponding to each
// Acquire-type function is a Display* function which display the
// accumulated data in a formatted way.
int main(int argc, char* argv[]) {
  hsa_status_t err;

  err = hsa_init();
  RET_IF_HSA_ERR(err);

  // Acquire and display system information
  system_info_t sys_info;

  // This function will call HSA get_info functions to gather information
  // about the system.
  err = AcquireSystemInfo(&sys_info);
  RET_IF_HSA_ERR(err);

  printLabel("=====================", true);
  printLabel("HSA System Attributes", true);
  printLabel("=====================", true);
  DisplaySystemInfo(&sys_info);

  // Iterate through every agent and get and display their info
  printLabel("==========", true);
  printLabel("HSA Agents", true);
  printLabel("==========", true);
  uint32_t agent_ind = 0;
  err = hsa_iterate_agents(AcquireAndDisplayAgentInfo, &agent_ind);
  RET_IF_HSA_ERR(err);

  printLabel("*** Done ***", true);

  err = hsa_shut_down();
  RET_IF_HSA_ERR(err);
}

#undef RET_IF_HSA_ERR
