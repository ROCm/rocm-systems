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
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <grp.h>
#include <unistd.h>
#include <pwd.h>

#include <fstream>
#include <vector>
#include <string>
#include <sstream>

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

#define COL_BLU  "\x1B[34m"
#define COL_KCYN  "\x1B[36m"
#define COL_GRN  "\x1B[32m"
#define COL_NRM  "\x1B[0m"
#define COL_RED  "\x1B[31m"
#define COL_MAG  "\x1B[35m"
#define COL_WHT  "\x1B[37m"
#define COL_YEL  "\x1B[33m"
#define COL_RESET "\033[0m"

#define RET_IF_HSA_ERR(err) { \
  if ((err) != HSA_STATUS_SUCCESS) { \
    char err_val[12];                                                         \
    char* err_str = NULL;                                                     \
    if (hsa_status_string(err,                                                \
            (const char**)&err_str) != HSA_STATUS_SUCCESS) {                  \
      snprintf(&(err_val[0]), sizeof(err_val), "%#x", (uint32_t)err);         \
      err_str = &(err_val[0]);                                                \
    }                                                                         \
    printf("%shsa api call failure at: %s:%d\n",                              \
                      COL_RED, __FILE__, __LINE__);                           \
    printf("%sCall returned %s\n", COL_RED, err_str);                         \
    printf("%s", COL_RESET);                                                  \
    return (err);                                                             \
  }                                                                           \
}

// This structure holds system information acquired through hsa info related
// calls, and is later used for reference when displaying the information.
struct system_info_t {
    uint16_t major, minor;
    uint16_t ext_major, ext_minor;
    uint64_t timestamp_frequency = 0;
    uint64_t max_wait = 0;
    hsa_endianness_t endianness;
    hsa_machine_model_t machine_model;
    bool mwaitx_enabled;
    bool xnack_enabled;
    bool dmabuf_support;
    bool vmm_support;
};

// This structure holds agent information acquired through hsa info related
// calls, and is later used for reference when displaying the information.
struct agent_info_t {
  char name[64];
  char uuid[24];
  char vendor_name[64];
  char device_mkt_name[64];
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
  uint32_t asic_revision;
  uint32_t cacheline_size;
  uint32_t max_clock_freq;
  uint32_t internal_node_id;
  uint32_t max_addr_watch_pts;
  // HSA_AMD_AGENT_INFO_MEMORY_WIDTH is deprecated, so exclude
  // uint32_t mem_max_freq; Not supported by get_info
  uint32_t compute_unit;
  uint32_t wavefront_size;
  uint32_t workgroup_max_size;
  uint32_t grid_max_size;
  uint32_t fbarrier_max_size;
  uint32_t max_waves_per_cu;
  uint32_t simds_per_cu;
  uint32_t shader_engs;
  uint32_t shader_arrs_per_sh_eng;
  hsa_isa_t agent_isa;
  hsa_dim3_t grid_max_dim;
  uint16_t workgroup_max_dim[3];
  uint16_t bdf_id;
  bool fast_f16;
  bool coherent_host_access;
  uint32_t pkt_processor_ucode_ver;
  uint32_t sdma_ucode_ver;
  hsa_amd_iommu_version_t iommu_support;
  uint8_t memory_properties[8];
};

// This structure holds memory pool information acquired through hsa info
// related calls, and is later used for reference when displaying the
// information.
typedef struct {
    uint32_t segment;
    size_t pool_size;
    bool alloc_allowed;
    size_t alloc_granule;
    size_t alloc_rec_granule;
    size_t pool_alloc_alignment;
    bool pl_access;
    uint32_t global_flag;
} pool_info_t;

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

static bool wsl_env = false;

enum rocmi_int_format {
  ROCMI_INT_FORMAT_DEC = 1,
  ROCMI_INT_FORMAT_HEX = 2,
};

// Make the most common format the default
std::string int_to_string(uint32_t i,
                   uint32_t fmt = ROCMI_INT_FORMAT_DEC|ROCMI_INT_FORMAT_HEX) {
  std::stringstream sd;
  bool need_parens = false;

  if (fmt & ROCMI_INT_FORMAT_DEC) {
    if (need_parens) {
      sd << "(";
    }
    sd << i;
    if (need_parens) {
      sd << ") ";
    }
    need_parens = true;
  }

  if (fmt & ROCMI_INT_FORMAT_HEX) {
    if (need_parens) {
      sd << "(0x";
    }
    sd << std::hex << i;
    if (need_parens) {
      sd << ") ";
    }
  }

  return sd.str();
}

static void DetectWSLEnvironment() {
  const char *filePath = "/dev/dxg";
  FILE *file = fopen(filePath, "r");
  if (file) {
    printf("WSL environment detected.\n");
    fclose(file);
    wsl_env = true;
  }
}

static void printLabelInt(char const *l, int d, uint32_t indent_lvl = 0) {
  std::string ind(kIndentSize * indent_lvl, ' ');

  printf("%s%-*s%-*d\n", ind.c_str(), kLabelFieldSize, l, kValueFieldSize, d);
}
static void printLabelStr(char const *l, char const *s,
                                                    uint32_t indent_lvl = 0) {
  std::string ind(kIndentSize * indent_lvl, ' ');
  printf("%s%-*s%-*s\n", ind.c_str(), kLabelFieldSize, l, kValueFieldSize, s);
}
static void printLabelStr(char const *l, std::string const &s,
                                                    uint32_t indent_lvl = 0) {
  std::string ind(kIndentSize * indent_lvl, ' ');
  printf("%s%-*s%-*s\n", ind.c_str(), kLabelFieldSize, l, kValueFieldSize,
                                                                   s.c_str());
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

  // Get HSA Ext Interface version
  err = hsa_system_get_info(HSA_AMD_SYSTEM_INFO_EXT_VERSION_MAJOR,
                                                     &sys_info->ext_major);
  RET_IF_HSA_ERR(err);
  err = hsa_system_get_info(HSA_AMD_SYSTEM_INFO_EXT_VERSION_MINOR,
                                                     &sys_info->ext_minor);
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

  // Get mwaitx mode
  err = hsa_system_get_info(HSA_AMD_SYSTEM_INFO_MWAITX_ENABLED,
                                                     &sys_info->mwaitx_enabled);
  RET_IF_HSA_ERR(err);

  // Get DMABuf support
  err = hsa_system_get_info(HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED,
                                                     &sys_info->dmabuf_support);
  RET_IF_HSA_ERR(err);

  // Get Xnack Enabled
  err = hsa_system_get_info(HSA_AMD_SYSTEM_INFO_XNACK_ENABLED,
                                                     &sys_info->xnack_enabled);
  RET_IF_HSA_ERR(err);

  // Get VMM supported
  err = hsa_system_get_info(HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED,
                                                     &sys_info->vmm_support);
  RET_IF_HSA_ERR(err);

  return err;
}

static void DisplaySystemInfo(system_info_t const *sys_info) {
  printLabel("Runtime Version:");
  printf("%d.%d\n", sys_info->major, sys_info->minor);
  printLabel("Runtime Ext Version:");
  printf("%d.%d\n", sys_info->ext_major, sys_info->ext_minor);
  printLabel("System Timestamp Freq.:");
  printf("%fMHz\n", sys_info->timestamp_frequency / 1e6);
  printLabel("Sig. Max Wait Duration:");
  printf("%lu (0x%lX) (timestamp count)\n", sys_info->max_wait,
                                                           sys_info->max_wait);

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

  printLabel("Mwaitx:");
  printf("%s\n", sys_info->mwaitx_enabled ? "ENABLED" : "DISABLED");

  printLabel("XNACK enabled:");
  printf("%s\n", sys_info->xnack_enabled ? "YES" : "NO");

  printLabel("DMAbuf Support:");
  printf("%s\n", sys_info->dmabuf_support ? "YES" : "NO");

  printLabel("VMM Support:");
  printf("%s\n", sys_info->vmm_support ? "YES" : "NO");

  printf("\n");
}

static hsa_status_t
AcquireAgentInfo(hsa_agent_t agent, agent_info_t *agent_i) {
  hsa_status_t err;
  // Get agent name and vendor
  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, agent_i->name);
  RET_IF_HSA_ERR(err);

  // Get UUID, an Ascii string, of a ROCm device
  err = hsa_agent_get_info(agent,
                         (hsa_agent_info_t)HSA_AMD_AGENT_INFO_UUID,
                                                   &agent_i->uuid);

  // Get device's vendor name
  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_VENDOR_NAME,
                                                       &agent_i->vendor_name);
  RET_IF_HSA_ERR(err);

  // Get device marketing name
  err = hsa_agent_get_info(agent,
                         (hsa_agent_info_t)HSA_AMD_AGENT_INFO_PRODUCT_NAME,
                                                   &agent_i->device_mkt_name);
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

  // Get asic revision
  err = hsa_agent_get_info(agent,
                           (hsa_agent_info_t) HSA_AMD_AGENT_INFO_ASIC_REVISION,
                                                           &agent_i->asic_revision);
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

  // Internal Driver node ID
  err = hsa_agent_get_info(agent,
                  (hsa_agent_info_t) HSA_AMD_AGENT_INFO_DRIVER_NODE_ID,
                                                  &agent_i->internal_node_id);
  RET_IF_HSA_ERR(err);

  // Max number of watch points on mem. addr. ranges to generate exeception
  // events
  err = hsa_agent_get_info(agent,
              (hsa_agent_info_t) HSA_AMD_AGENT_INFO_MAX_ADDRESS_WATCH_POINTS,
                                                &agent_i->max_addr_watch_pts);
  RET_IF_HSA_ERR(err);

  // Get Agent BDFID
  err = hsa_agent_get_info(agent,
                (hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID, &agent_i->bdf_id);
  RET_IF_HSA_ERR(err);

  // Get Max Memory Clock
  // Not supported by hsa_agent_get_info
  //  err = hsa_agent_get_info(agent,d
  //              (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MEMORY_MAX_FREQUENCY,
  //                                                      &agent_i->mem_max_freq);
  //  RET_IF_HSA_ERR(err);

  // Get Num SIMDs per CU
  err = hsa_agent_get_info(agent,
              (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU,
                                                      &agent_i->simds_per_cu);
  RET_IF_HSA_ERR(err);

  // Get Num Shader Engines
  err = hsa_agent_get_info(agent,
              (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES,
                                                      &agent_i->shader_engs);
  RET_IF_HSA_ERR(err);

  // Get Num Shader Arrays per Shader engine
  err = hsa_agent_get_info(agent,
              (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE,
                                            &agent_i->shader_arrs_per_sh_eng);
  RET_IF_HSA_ERR(err);

  // Get number of Compute Unit
  err = hsa_agent_get_info(agent,
                   (hsa_agent_info_t) HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT,
                                                      &agent_i->compute_unit);
  RET_IF_HSA_ERR(err);

  // Get coherent Host access
  err = hsa_agent_get_info(agent,
                           (hsa_agent_info_t) HSA_AMD_AGENT_INFO_SVM_DIRECT_HOST_ACCESS,
                           &agent_i->coherent_host_access);
  RET_IF_HSA_ERR(err);

  // Get memory properties
  err = hsa_agent_get_info(agent,
                           (hsa_agent_info_t) HSA_AMD_AGENT_INFO_MEMORY_PROPERTIES,
                           agent_i->memory_properties);
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
                                                  &agent_i->max_waves_per_cu);
    RET_IF_HSA_ERR(err);

    err = hsa_agent_get_info(agent,
                    (hsa_agent_info_t)HSA_AMD_AGENT_INFO_UCODE_VERSION,
                                                  &agent_i->pkt_processor_ucode_ver);
    RET_IF_HSA_ERR(err);
    err = hsa_agent_get_info(agent,
                    (hsa_agent_info_t)HSA_AMD_AGENT_INFO_SDMA_UCODE_VERSION,
                                                  &agent_i->sdma_ucode_ver);
    RET_IF_HSA_ERR(err);
    err = hsa_agent_get_info(agent,
                    (hsa_agent_info_t)HSA_AMD_AGENT_INFO_IOMMU_SUPPORT,
                                                  &agent_i->iommu_support);
    RET_IF_HSA_ERR(err);
  }
  return err;
}

static void DisplayAgentInfo(agent_info_t *agent_i) {
  printLabelStr("Name:", agent_i->name, 1);
  printLabelStr("Uuid:", agent_i->uuid, 1);
  printLabelStr("Marketing Name:", agent_i->device_mkt_name, 1);
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

  printLabelStr("Max Queue Number:",  int_to_string(agent_i->max_queue), 1);

  printLabelStr("Queue Min Size:",  int_to_string(agent_i->queue_min_size), 1);

  printLabelStr("Queue Max Size:",  int_to_string(agent_i->queue_max_size), 1);

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

    //  tmp_str = std::to_string(agent_i->cache_size[i]/1024);
      tmp_str = int_to_string(agent_i->cache_size[i]/1024);
      tmp_str += "KB";
      printValueStr(tmp_str.c_str());
    }
  }

  printLabelStr("Chip ID:", int_to_string(agent_i->chip_id), 1);
  if (!wsl_env)
    printLabelStr("ASIC Revision:", int_to_string(agent_i->asic_revision), 1);
  printLabelStr("Cacheline Size:", int_to_string(agent_i->cacheline_size), 1);
  if (!wsl_env || HSA_DEVICE_TYPE_GPU == agent_i->device_type)
    printLabelInt("Max Clock Freq. (MHz):", agent_i->max_clock_freq, 1);
  printLabelInt("BDFID:", agent_i->bdf_id, 1);
  printLabelInt("Internal Node ID:", agent_i->internal_node_id, 1);
  printLabelInt("Compute Unit:", agent_i->compute_unit, 1);
  printLabelInt("SIMDs per CU:", agent_i->simds_per_cu, 1);
  printLabelInt("Shader Engines:", agent_i->shader_engs, 1);
  printLabelInt("Shader Arrs. per Eng.:", agent_i->shader_arrs_per_sh_eng, 1);
  if (!wsl_env)
    printLabelInt("WatchPts on Addr. Ranges:", agent_i->max_addr_watch_pts, 1);

  if (agent_i->device_type == HSA_DEVICE_TYPE_GPU)
    printLabelStr("Coherent Host Access:", agent_i->coherent_host_access ? "TRUE":"FALSE", 1);

  printLabel("Memory Properties:", false, 1);
  if (hsa_flag_isset64(agent_i->memory_properties, HSA_AMD_MEMORY_PROPERTY_AGENT_IS_APU))
    printf("%s", "APU");
  printf("\n");

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
    printLabelStr("Fast F16 Operation:",
                                       agent_i->fast_f16 ? "TRUE":"FALSE", 1);

    printLabelStr("Wavefront Size:",
                                   int_to_string(agent_i->wavefront_size), 1);

    printLabelStr("Workgroup Max Size:",
        int_to_string(agent_i->workgroup_max_size), 1);
    printLabel("Workgroup Max Size per Dimension:", true, 1);
    printLabelStr("x",
      int_to_string(static_cast<uint32_t>(agent_i->workgroup_max_dim[0])), 2);
    printLabelStr("y",
      int_to_string(static_cast<uint32_t>(agent_i->workgroup_max_dim[1])), 2);
    printLabelStr("z",
      int_to_string(static_cast<uint32_t>(agent_i->workgroup_max_dim[2])), 2);

    printLabelStr("Max Waves Per CU:",
                                 int_to_string(agent_i->max_waves_per_cu), 1);
    printLabelStr("Max Work-item Per CU:",
         int_to_string(agent_i->wavefront_size*agent_i->max_waves_per_cu), 1);

    printLabelStr("Grid Max Size:", int_to_string(agent_i->grid_max_size), 1);
    printLabel("Grid Max Size per Dimension:", true, 1);
    printLabelStr("x", int_to_string(agent_i->grid_max_dim.x), 2);
    printLabelStr("y", int_to_string(agent_i->grid_max_dim.y), 2);
    printLabelStr("z", int_to_string(agent_i->grid_max_dim.z), 2);

    printLabelInt("Max fbarriers/Workgrp:", agent_i->fbarrier_max_size, 1);

    printLabelInt("Packet Processor uCode::", agent_i->pkt_processor_ucode_ver, 1);
    printLabelInt("SDMA engine uCode::", agent_i->sdma_ucode_ver, 1);
    printLabelStr("IOMMU Support::",
                    agent_i->iommu_support == HSA_IOMMU_SUPPORT_V2 ? "V2" : "None", 1);
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
                                     HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE,
                                     &pool_i->alloc_rec_granule);
  RET_IF_HSA_ERR(err);

  err = hsa_amd_memory_pool_get_info(pool,
                           HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT,
                                               &pool_i->pool_alloc_alignment);
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

  if (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED & global_flag)
  {
    flags.push_back("EXTENDED FINE GRAINED");
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

  size_t sz = pool_i->pool_size/1024;
  printLabelStr("Size:", int_to_string(sz) + "KB", indent);
  printLabelStr("Allocatable:", (pool_i->alloc_allowed ? "TRUE" : "FALSE"),
                                                                      indent);
  std::string gr_str = std::to_string(pool_i->alloc_granule/1024)+"KB";
  printLabelStr("Alloc Granule:", gr_str.c_str(), indent);

  std::string rgr_str = std::to_string(pool_i->alloc_rec_granule / 1024) + "KB";
  printLabelStr("Alloc Recommended Granule:", rgr_str.c_str(), indent);

  std::string al_str = std::to_string(pool_i->pool_alloc_alignment/1024)+"KB";
  printLabelStr("Alloc Alignment:", al_str.c_str(), indent);

  printLabelStr("Accessible by all:", (pool_i->pl_access ? "TRUE" : "FALSE"),
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

  printLabelStr("Workgroup Max Size:",
                            int_to_string(isa_i->workgroup_max_size), indent);
  printLabel("Workgroup Max Size per Dimension:", true, indent);
  printLabelStr("x", int_to_string(
               static_cast<uint32_t>(isa_i->workgroup_max_dim[0])), indent+1);
  printLabelStr("y", int_to_string(
               static_cast<uint32_t>(isa_i->workgroup_max_dim[1])), indent+1);
  printLabelStr("z", int_to_string(
               static_cast<uint32_t>(isa_i->workgroup_max_dim[2])), indent+1);

  printLabelStr("Grid Max Size:", int_to_string(isa_i->grid_max_size), indent);
  printLabel("Grid Max Size per Dimension:", true, indent);
  printLabelStr("x", int_to_string(isa_i->grid_max_dim.x), indent+1);
  printLabelStr("y", int_to_string(isa_i->grid_max_dim.y), indent+1);
  printLabelStr("z", int_to_string(isa_i->grid_max_dim.z), indent+1);

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

int CheckInitialState(void) {
  // Check kernel module for ROCk is loaded

  std::ifstream amdgpu_initstate("/sys/module/amdgpu/initstate");
  if (amdgpu_initstate){
    std::stringstream buffer;
    buffer << amdgpu_initstate.rdbuf();
    amdgpu_initstate.close();

    std::string line;
    bool is_live = false;
    while (std::getline(buffer, line)){
      if (line.find( "live" ) != std::string::npos){
        is_live = true;
        break;
      }
    }
    if (is_live){
      std::ifstream amdgpu_version("/sys/module/amdgpu/version");
      if (amdgpu_version){
        std::stringstream buffer;
        buffer << amdgpu_version.rdbuf();
        std::string vers;
        std::getline(buffer, vers);
        amdgpu_version.close();
        printf("%sROCk module version %s is loaded%s\n", COL_WHT, vers.c_str(), COL_RESET);
      } else {
        printf("%sROCk module is loaded%s\n", COL_WHT, COL_RESET);
      }
    } else {
      printf("%sROCk module is NOT live, possibly no GPU devices%s\n",
                                                          COL_RED, COL_RESET);
      return -1;
    }
  } else {
    int module_dir;
    module_dir = open("/sys/module/amdgpu", O_DIRECTORY);
    if (module_dir < 0) {
      printf("%sROCk module is NOT loaded, possibly no GPU devices%s\n",
                                                            COL_RED, COL_RESET);
      return -1;
    }
    close(module_dir);
  }

  // Check if user belongs to the group for /dev/kfd (e.g. "video" or
  // "render")
  // @note: User who are not members of "video"
  // group cannot access DRM services
  char u_name[32];
  bool member = false;
  struct passwd *pw;
  int num_groups = 0;
  gid_t *groups;

  // Check if we can open /dev/kfd as read-write. If not, try to
  // diagnose common reasons why you can't.
  int open_kfd = open("/dev/kfd", O_RDWR);
  if (open_kfd >= 0) {
      close(open_kfd);
      return 0;
  }

  printf("%sUnable to open /dev/kfd read-write: %s%s\n",
         COL_RED, strerror(errno), COL_RESET);

  const char *kfd_gr_name = NULL;

  struct stat sb;
  if (stat("/dev/kfd", &sb) == 0) {
      // The owner of kfd was renamed, so avoid hard-coding the
      // name. Check whatever group owns it.
      if (struct group *kfd_gr = getgrgid(sb.st_gid))
          kfd_gr_name = kfd_gr->gr_name;
  }

  if (!kfd_gr_name)
      kfd_gr_name = "video";

  struct group *gr_s = getgrnam(kfd_gr_name);  // NOLINT
  if (gr_s == nullptr) {
    printf("%sFailed to get group info to check"
           " for %s group membership%s\n", COL_RED, kfd_gr_name,
           COL_RESET);
    return -1;
  }

  if (getlogin_r(u_name, 32)) {
    printf("%sFailed to get user name to check for"
           " %s group membership%s\n", COL_RED, kfd_gr_name,
           COL_RESET);
    return -1;
  }

  pw = getpwnam(u_name); // NOLINT
  if (pw == NULL) {
    printf("%sFailed to find pwd entry for user %s%s\n",
                                                  COL_RED, u_name, COL_RESET);
    return -1;
  }

  (void)getgrouplist(u_name, pw->pw_gid, NULL, &num_groups);
  groups = new gid_t[num_groups];
  if (getgrouplist(u_name, pw->pw_gid, groups, &num_groups) == -1) {
    printf("%sFailed to get user group list%s\n", COL_RED, COL_RESET);
    delete []groups;
    return -1;
  }

  for (int i = 0; i < num_groups; ++i) {
    if (gr_s->gr_gid == groups[i]) {
      printf("%s%s is member of %s group%s\n", COL_WHT, u_name, kfd_gr_name, COL_RESET);
      member = true;
      break;
    }
  }
  if (member == false) {
    printf("%s%s is not member of \"%s\" group, the default DRM access "
     "group. Users must be a member of the \"%s\" group or another"
        " DRM access group in order for ROCm applications to run "
           "successfully%s.\n", COL_RED, u_name, kfd_gr_name, kfd_gr_name, COL_RESET);
  }

  delete []groups;
  return -1;
}

// Print out all static information known to HSA about the target system.
// Throughout this program, the Acquire-type functions make HSA calls to
// interate through HSA objects and then perform HSA get_info calls to
// acccumulate information about those objects. Corresponding to each
// Acquire-type function is a Display* function which display the
// accumulated data in a formatted way.
int main(int argc, char* argv[]) {
  hsa_status_t err;

  DetectWSLEnvironment();

  if (!wsl_env && CheckInitialState()) {
    return 1;
  }
  err = hsa_init();
  RET_IF_HSA_ERR(err)

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
  return 0;
}

#undef RET_IF_HSA_ERR
