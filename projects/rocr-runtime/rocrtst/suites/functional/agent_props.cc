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

#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

#include "suites/functional/agent_props.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "common/hsatimer.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

namespace {

// gfx12.5 and later place a workgroup cluster on one shader engine.
// Processor names are "gfx12<minor>..." with an optional "-strict" suffix.
bool WorkgroupClustersSupported(const char* name) {
  if (std::strncmp(name, "gfx12", 5) != 0)
    return false;
  const char minor = name[5];
  return minor >= '5' && minor <= '9';
}

}  // namespace

static const uint32_t kNumBufferElements = 256;

#define RET_IF_HSA_ERR(err) { \
  if ((err) != HSA_STATUS_SUCCESS) { \
    const char* msg = 0; \
    hsa_status_string(err, &msg); \
    std::cout << "hsa api call failure at line " << __LINE__ << ", file: " << \
                          __FILE__ << ". Call returned " << err << std::endl; \
    std::cout << msg << std::endl; \
    return (err); \
  } \
}

static const char kSubTestSeparator[] = "  **************************";

static void PrintAgentPropsSubtestHeader(const char *header) {
  std::cout << "  *** " << header << " ***" << std::endl;
}

AgentPropTest::AgentPropTest(void) :
    TestBase() {
  set_num_iteration(10);  // Number of iterations to execute of the main test;
                          // This is a default value which can be overridden
                          // on the command line.
  set_title("  *** Query RocR Agent Properties ***");
  set_description("  *** Checks properties of Agent's on a system ***");
}

AgentPropTest::~AgentPropTest(void) {
}

// Any 1-time setup involving member variables used in the rest of the test
// should be done here.
void AgentPropTest::SetUp(void) {
  TestBase::SetUp();
  if (test_skipped_) return;
  std::cout << "  *** Initialize ROCr Runtime and " 
            << "acquire handles of agents" << " ***" << std::endl;
}

void AgentPropTest::Run(void) {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::Run();
}

void AgentPropTest::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void AgentPropTest::DisplayResults(void) const {
  TestBase::DisplayResults();
  std::cout << std::endl;
  for (uint32_t idx = 0 ; idx < this->propList_.size(); ++idx) {
    std::cout << this->propList_[idx] << std::endl;
  }
  return;
}

void AgentPropTest::Close() {
  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

// Extend this method to query for agent properties that are
// currently not tested
void AgentPropTest::QueryAgentProp(hsa_agent_t agent,
                                   hsa_agent_info_t prop) {
  hsa_status_t err;
  hsa_device_type_t agType;
  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &agType);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  std::stringstream ss;
  ss << "  Agent " << "(";
  switch (agType) {
    case HSA_DEVICE_TYPE_CPU:
      ss << "CPU) : ";
      break;
    case HSA_DEVICE_TYPE_GPU:
      ss << "GPU) : ";
      break;
    case HSA_DEVICE_TYPE_DSP:
      ss << "DSP) : ";
      break;
    case HSA_DEVICE_TYPE_AIE:
      ss << "AIE) : ";
      break;
  }

  // Print the agent property
  uint32_t key = uint32_t(prop);
  switch (key) {
  // Retrieves UUID property value of the agent
  case HSA_AMD_AGENT_INFO_UUID: {
    char uuid[32];
    err = hsa_agent_get_info(agent, prop, (void*)&uuid[0]);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    ss << uuid;
    propList_.push_back(ss.str());
    break;
  }
  case HSA_AMD_AGENT_INFO_CLOCK_COUNTERS: {
    std::stringstream str_s;

    hsa_amd_clock_counters_t counters = {0};

    err = hsa_agent_get_info(agent, prop, &counters);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    str_s << "\n\n  Clock Counters";
    str_s << "\n  Clock Frequency: "  << counters.system_clock_frequency << "\n";
    str_s << "  GPU Clock counter: "  << counters.gpu_clock_counter << "\n";
    str_s << "  System system_clock_counter: "  << counters.system_clock_counter << "\n";
    str_s << "  CPU Clock counter: "  << counters.cpu_clock_counter << "\n";
    propList_.push_back(str_s.str());

    ASSERT_NE(0, counters.system_clock_frequency);
    ASSERT_NE(0, counters.gpu_clock_counter);
    ASSERT_NE(0, counters.system_clock_counter);
    ASSERT_NE(0, counters.cpu_clock_counter);

    break;
  }
  default:
    FAIL();
  }

}

void AgentPropTest::QueryAgentUUID() {
  hsa_status_t err;
  if (verbosity() > 0) {
    PrintAgentPropsSubtestHeader("Query GPU and CPU Agent's UUID");
  }

  // find all cpu agents
  std::vector<hsa_agent_t> cpus;
  err = hsa_iterate_agents(rocrtst::IterateCPUAgents, &cpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // find all gpu agents
  std::vector<hsa_agent_t> gpus;
  err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  for (uint32_t idx = 0 ; idx < cpus.size(); ++idx) {
    QueryAgentProp(cpus[idx], (hsa_agent_info_t)HSA_AMD_AGENT_INFO_UUID);
  }

  for (uint32_t idx = 0 ; idx < gpus.size(); ++idx) {
    QueryAgentProp(gpus[idx], (hsa_agent_info_t)HSA_AMD_AGENT_INFO_UUID);
  }

  if (verbosity() > 0) {
    std::cout << "  *** Execution completed - subtest Passed " << " ***" << std::endl;
  }
}

void AgentPropTest::QueryAgentClockCounters() {
  hsa_status_t err;
  if (verbosity() > 0) {
    PrintAgentPropsSubtestHeader("Query Agent's Clock Counters");
  }

  // find all gpu agents
  std::vector<hsa_agent_t> gpus;
  err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  for (uint32_t idx = 0 ; idx < gpus.size(); ++idx) {
    QueryAgentProp(gpus[idx], (hsa_agent_info_t)HSA_AMD_AGENT_INFO_CLOCK_COUNTERS);
  }

  if (verbosity() > 0) {
    std::cout << "  *** Execution completed - subtest Passed " << " ***" << std::endl;
  }
}

void AgentPropTest::QueryGpuShaderEngineTopology() {
  hsa_status_t err;
  if (verbosity() > 0) {
    PrintAgentPropsSubtestHeader("Validate GPU shader-engine topology");
  }

  std::vector<hsa_agent_t> gpus;
  err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  for (uint32_t idx = 0; idx < gpus.size(); ++idx) {
    const hsa_agent_t agent = gpus[idx];
    char name[64] = {};
    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS) << "GPU index " << idx;

    uint32_t num_shader_engines = 0;
    uint32_t num_simds_per_cu = 0;
    uint32_t num_cus = 0;
    err = hsa_agent_get_info(
        agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES,
        &num_shader_engines);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS) << name;
    err = hsa_agent_get_info(
        agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU,
        &num_simds_per_cu);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS) << name;
    err = hsa_agent_get_info(
        agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT, &num_cus);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS) << name;

    // Zero shader engines, zero SIMDs per CU, or fewer active CUs than shader
    // engines cannot describe a cluster. That topology is invalid.
    ASSERT_GT(num_shader_engines, 0u) << name << " has no shader engines";
    ASSERT_GT(num_simds_per_cu, 0u) << name << " has no SIMDs per CU";
    ASSERT_GT(num_cus / num_shader_engines, 0u)
        << name << " has " << num_cus << " CUs and " << num_shader_engines
        << " shader engines";

    hsa_amd_dim3_t cluster_dim = {};
    uint64_t cluster_max = 0;
    err = hsa_agent_get_info(
        agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_CLUSTER_MAX_DIM, &cluster_dim);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS) << name;
    err = hsa_agent_get_info(
        agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_CLUSTER_MAX_SIZE, &cluster_max);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS) << name;

    // gfx12.5 reports the active CUs in one shader engine. Other GPUs report 1.
    const uint64_t expected = WorkgroupClustersSupported(name)
                                  ? num_cus / num_shader_engines
                                  : 1;
    ASSERT_EQ(cluster_max, expected) << name;
    ASSERT_EQ(cluster_dim.x, expected) << name;
    ASSERT_EQ(cluster_dim.y, expected) << name;
    ASSERT_EQ(cluster_dim.z, expected) << name;
  }

  if (verbosity() > 0) {
    std::cout << "  *** Execution completed - subtest Passed " << " ***" << std::endl;
  }
}

void AgentPropTest::QuerySystemInfo() {
  if (verbosity() > 0) {
    PrintAgentPropsSubtestHeader("Query system info attributes");
  }

  uint16_t major = 0;
  uint16_t minor = 0;
  ASSERT_SUCCESS(hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MAJOR, &major));
  ASSERT_SUCCESS(hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MINOR, &minor));
  // ROCr implements HSA 1.x.
  ASSERT_EQ(major, 1);
  ASSERT_GE(minor, 0);

  // The system timestamp must tick forward at the reported frequency.
  uint64_t freq = 0;
  ASSERT_SUCCESS(hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &freq));
  ASSERT_NE(freq, 0u);

  uint64_t ts_before = 0;
  uint64_t ts_after = 0;
  ASSERT_SUCCESS(hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP, &ts_before));
  ASSERT_SUCCESS(hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP, &ts_after));
  ASSERT_NE(ts_before, 0u);
  ASSERT_GE(ts_after, ts_before);

  uint64_t max_wait = 0;
  ASSERT_SUCCESS(hsa_system_get_info(HSA_SYSTEM_INFO_SIGNAL_MAX_WAIT, &max_wait));
  ASSERT_EQ(max_wait, UINT64_MAX);

  hsa_endianness_t endianness;
  ASSERT_SUCCESS(hsa_system_get_info(HSA_SYSTEM_INFO_ENDIANNESS, &endianness));
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  ASSERT_EQ(endianness, HSA_ENDIANNESS_BIG);
#else
  ASSERT_EQ(endianness, HSA_ENDIANNESS_LITTLE);
#endif

  hsa_machine_model_t model;
  ASSERT_SUCCESS(hsa_system_get_info(HSA_SYSTEM_INFO_MACHINE_MODEL, &model));
  ASSERT_EQ(model, sizeof(void*) == 8 ? HSA_MACHINE_MODEL_LARGE : HSA_MACHINE_MODEL_SMALL);

  // The extension bit mask must agree with hsa_system_extension_supported().
  uint8_t extensions[128];
  memset(extensions, 0xFF, sizeof(extensions));
  ASSERT_SUCCESS(hsa_system_get_info(HSA_SYSTEM_INFO_EXTENSIONS, extensions));
  auto ext_bit = [&](uint16_t ext) { return (extensions[ext / 8] >> (ext % 8)) & 1; };
  ASSERT_TRUE(ext_bit(HSA_EXTENSION_AMD_PROFILER));
  bool supported = false;
  ASSERT_SUCCESS(hsa_system_extension_supported(HSA_EXTENSION_AMD_PROFILER, 1, 0, &supported));
  ASSERT_TRUE(supported);
  // Bits past the highest extension the runtime defines are never set.
  ASSERT_EQ(extensions[127], 0);

  const char* build_version = nullptr;
  ASSERT_SUCCESS(hsa_system_get_info(HSA_AMD_SYSTEM_INFO_BUILD_VERSION, &build_version));
  ASSERT_NE(build_version, nullptr);

  bool svm_supported = false;
  bool svm_by_default = false;
  bool xnack = false;
  bool mwaitx = false;
  bool dmabuf = false;
  bool vmem = false;
  bool fabric = false;
  bool host_dmabuf = false;
  ASSERT_SUCCESS(hsa_system_get_info(HSA_AMD_SYSTEM_INFO_SVM_SUPPORTED, &svm_supported));
  ASSERT_SUCCESS(hsa_system_get_info(HSA_AMD_SYSTEM_INFO_SVM_ACCESSIBLE_BY_DEFAULT, &svm_by_default));
  ASSERT_SUCCESS(hsa_system_get_info(HSA_AMD_SYSTEM_INFO_XNACK_ENABLED, &xnack));
  ASSERT_SUCCESS(hsa_system_get_info(HSA_AMD_SYSTEM_INFO_MWAITX_ENABLED, &mwaitx));
  ASSERT_SUCCESS(hsa_system_get_info(HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED, &dmabuf));
  ASSERT_SUCCESS(hsa_system_get_info(HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED, &vmem));
  ASSERT_SUCCESS(hsa_system_get_info(HSA_AMD_SYSTEM_INFO_FABRIC_HANDLES_SUPPORTED, &fabric));
  ASSERT_SUCCESS(hsa_system_get_info(HSA_AMD_SYSTEM_INFO_HOST_ALLOC_DMA_BUF_SUPPORTED, &host_dmabuf));
  // Both queries are defined as the XNACK state of the system.
  ASSERT_EQ(svm_by_default, xnack);
  // Host-memory dma-buf allocation is built on the virtual memory APIs.
  if (host_dmabuf) {
    ASSERT_TRUE(vmem);
  }

  uint16_t ext_major = 0;
  uint16_t ext_minor = 0;
  ASSERT_SUCCESS(hsa_system_get_info(HSA_AMD_SYSTEM_INFO_EXT_VERSION_MAJOR, &ext_major));
  ASSERT_SUCCESS(hsa_system_get_info(HSA_AMD_SYSTEM_INFO_EXT_VERSION_MINOR, &ext_minor));
  ASSERT_EQ(ext_major, HSA_AMD_INTERFACE_VERSION_MAJOR);
  ASSERT_EQ(ext_minor, HSA_AMD_INTERFACE_VERSION_MINOR);

  if (verbosity() > 0) {
    std::cout << "  HSA " << major << "." << minor << ", AMD ext " << ext_major << "."
              << ext_minor << ", build " << build_version << ", svm=" << svm_supported
              << " xnack=" << xnack << " dmabuf=" << dmabuf << " vmem=" << vmem
              << " fabric=" << fabric << " host_dmabuf=" << host_dmabuf << " mwaitx=" << mwaitx
              << std::endl;
  }

  // Invalid arguments.
  uint64_t scratch = 0;
  ASSERT_EQ(hsa_system_get_info(static_cast<hsa_system_info_t>(0x7FFF), &scratch),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);
  ASSERT_EQ(hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MAJOR, nullptr),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);

  if (verbosity() > 0) {
    std::cout << "  *** Execution completed - subtest Passed " << " ***" << std::endl;
  }
}

namespace {

struct AgentIterCounts {
  uint32_t regions = 0;
  uint32_t caches = 0;
  uint32_t isas = 0;
  uint32_t pools = 0;
  bool region_info_ok = true;
  bool cache_info_ok = true;
  bool isa_info_ok = true;
};

hsa_status_t CountRegion(hsa_region_t region, void* data) {
  auto* counts = static_cast<AgentIterCounts*>(data);
  ++counts->regions;
  hsa_region_segment_t segment;
  size_t size = 0;
  if (hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment) != HSA_STATUS_SUCCESS ||
      hsa_region_get_info(region, HSA_REGION_INFO_SIZE, &size) != HSA_STATUS_SUCCESS) {
    counts->region_info_ok = false;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t CountCache(hsa_cache_t cache, void* data) {
  auto* counts = static_cast<AgentIterCounts*>(data);
  ++counts->caches;
  uint8_t level = 0;
  uint32_t size = 0;
  if (hsa_cache_get_info(cache, HSA_CACHE_INFO_LEVEL, &level) != HSA_STATUS_SUCCESS ||
      hsa_cache_get_info(cache, HSA_CACHE_INFO_SIZE, &size) != HSA_STATUS_SUCCESS || level == 0) {
    counts->cache_info_ok = false;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t CountIsa(hsa_isa_t isa, void* data) {
  auto* counts = static_cast<AgentIterCounts*>(data);
  ++counts->isas;
  uint32_t name_len = 0;
  if (hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME_LENGTH, &name_len) != HSA_STATUS_SUCCESS ||
      name_len == 0) {
    counts->isa_info_ok = false;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t CountPool(hsa_amd_memory_pool_t pool, void* data) {
  auto* counts = static_cast<AgentIterCounts*>(data);
  ++counts->pools;
  return HSA_STATUS_SUCCESS;
}

// Stops the iteration after the first element and reports the stop reason.
hsa_status_t StopIteration(hsa_region_t, void*) { return HSA_STATUS_INFO_BREAK; }

}  // namespace

void AgentPropTest::QueryGpuAgentInfo() {
  hsa_status_t err;
  if (verbosity() > 0) {
    PrintAgentPropsSubtestHeader("Query GPU agent info attributes");
  }

  std::vector<hsa_agent_t> gpus;
  err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // Attributes every GPU agent must be able to report. Image attributes are
  // left out because they depend on the image extension being loaded.
  static const uint32_t kAttributes[] = {
      HSA_AGENT_INFO_NAME,
      HSA_AGENT_INFO_VENDOR_NAME,
      HSA_AGENT_INFO_FEATURE,
      HSA_AGENT_INFO_MACHINE_MODEL,
      HSA_AGENT_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES,
      HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE,
      HSA_AGENT_INFO_FAST_F16_OPERATION,
      HSA_AGENT_INFO_PROFILE,
      HSA_AGENT_INFO_WAVEFRONT_SIZE,
      HSA_AGENT_INFO_WORKGROUP_MAX_DIM,
      HSA_AGENT_INFO_WORKGROUP_MAX_SIZE,
      HSA_AGENT_INFO_GRID_MAX_DIM,
      HSA_AGENT_INFO_GRID_MAX_SIZE,
      HSA_AGENT_INFO_FBARRIER_MAX_SIZE,
      HSA_AGENT_INFO_QUEUES_MAX,
      HSA_AGENT_INFO_QUEUE_MIN_SIZE,
      HSA_AGENT_INFO_QUEUE_MAX_SIZE,
      HSA_AGENT_INFO_QUEUE_TYPE,
      HSA_AGENT_INFO_NODE,
      HSA_AGENT_INFO_DEVICE,
      HSA_AGENT_INFO_CACHE_SIZE,
      HSA_AGENT_INFO_ISA,
      HSA_AGENT_INFO_EXTENSIONS,
      HSA_AGENT_INFO_VERSION_MAJOR,
      HSA_AGENT_INFO_VERSION_MINOR,
      HSA_AMD_AGENT_INFO_CHIP_ID,
      HSA_AMD_AGENT_INFO_CACHELINE_SIZE,
      HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT,
      HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY,
      HSA_AMD_AGENT_INFO_DRIVER_NODE_ID,
      HSA_AMD_AGENT_INFO_MAX_ADDRESS_WATCH_POINTS,
      HSA_AMD_AGENT_INFO_BDFID,
      HSA_AMD_AGENT_INFO_MEMORY_WIDTH,
      HSA_AMD_AGENT_INFO_MEMORY_MAX_FREQUENCY,
      HSA_AMD_AGENT_INFO_PRODUCT_NAME,
      HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU,
      HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU,
      HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES,
      HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE,
      HSA_AMD_AGENT_INFO_HDP_FLUSH,
      HSA_AMD_AGENT_INFO_DOMAIN,
      HSA_AMD_AGENT_INFO_COOPERATIVE_QUEUES,
      HSA_AMD_AGENT_INFO_UUID,
      HSA_AMD_AGENT_INFO_ASIC_REVISION,
      HSA_AMD_AGENT_INFO_SVM_DIRECT_HOST_ACCESS,
      HSA_AMD_AGENT_INFO_COOPERATIVE_COMPUTE_UNIT_COUNT,
      HSA_AMD_AGENT_INFO_MEMORY_AVAIL,
      HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY,
      HSA_AMD_AGENT_INFO_ASIC_FAMILY_ID,
      HSA_AMD_AGENT_INFO_UCODE_VERSION,
      HSA_AMD_AGENT_INFO_SDMA_UCODE_VERSION,
      HSA_AMD_AGENT_INFO_NUM_SDMA_ENG,
      HSA_AMD_AGENT_INFO_NUM_SDMA_XGMI_ENG,
      HSA_AMD_AGENT_INFO_IOMMU_SUPPORT,
      HSA_AMD_AGENT_INFO_NUM_XCC,
      HSA_AMD_AGENT_INFO_DRIVER_UID,
      HSA_AMD_AGENT_INFO_NEAREST_CPU,
      HSA_AMD_AGENT_INFO_MEMORY_PROPERTIES,
      HSA_AMD_AGENT_INFO_AQL_EXTENSIONS,
      HSA_AMD_AGENT_INFO_SCRATCH_LIMIT_MAX,
      HSA_AMD_AGENT_INFO_SCRATCH_LIMIT_CURRENT,
      HSA_AMD_AGENT_INFO_CLOCK_COUNTERS,
      HSA_AMD_AGENT_INFO_PM4_EMULATION,
      HSA_AMD_AGENT_INFO_LUID,
      HSA_AMD_AGENT_INFO_HAS_EXPERT_SCHED_MODE,
      HSA_AMD_AGENT_INFO_CUID,
      HSA_AMD_AGENT_INFO_KERNEL_WG_MAX_SIZE,
      HSA_AMD_AGENT_INFO_KERNEL_CLUSTER_MAX_DIM,
      HSA_AMD_AGENT_INFO_KERNEL_CLUSTER_MAX_SIZE,
      HSA_AMD_AGENT_INFO_CLUSTER_MAX_DIM,
      HSA_AMD_AGENT_INFO_CLUSTER_MAX_SIZE,
      HSA_AMD_AGENT_INFO_KERNEL_WG_MAX_DIM,
      HSA_AMD_AGENT_INFO_MAX_DATA_PREFETCH_REGIONS,
      HSA_AMD_AGENT_INFO_HOST_ALLOC_DMABUF_SUPPORTED,
      HSA_AMD_AGENT_INFO_REQUEST_PERSISTING_L2_CACHE_SIZE,
      HSA_AMD_AGENT_INFO_MAX_PERSISTING_L2_CACHE_SIZE,
      HSA_AMD_AGENT_INFO_PCIE_STORES_NEED_FLUSH,
      HSA_AMD_AGENT_INFO_HOST_ATOMICS_SUPPORTED,
  };

  ASSERT_GT(gpus.size(), 0u);
  for (size_t idx = 0; idx < gpus.size(); ++idx) {
    const hsa_agent_t agent = gpus[idx];
    char name[64] = {};
    ASSERT_SUCCESS(hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name));

    // Larger and more aligned than any attribute value.
    alignas(16) uint8_t value[1024];
    for (uint32_t attribute : kAttributes) {
      memset(value, 0, sizeof(value));
      err = hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(attribute), value);
      ASSERT_EQ(err, HSA_STATUS_SUCCESS)
          << name << ": attribute 0x" << std::hex << attribute << " failed";
    }

    // Spot-check values that must be self-consistent.
    uint32_t wave_size = 0;
    ASSERT_SUCCESS(hsa_agent_get_info(agent, HSA_AGENT_INFO_WAVEFRONT_SIZE, &wave_size));
    ASSERT_TRUE(wave_size == 32 || wave_size == 64) << name << " wave size " << wave_size;

    uint32_t queue_min = 0;
    uint32_t queue_max = 0;
    ASSERT_SUCCESS(hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUE_MIN_SIZE, &queue_min));
    ASSERT_SUCCESS(hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_max));
    ASSERT_GT(queue_min, 0u);
    ASSERT_LE(queue_min, queue_max);

    hsa_device_type_t type;
    ASSERT_SUCCESS(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type));
    ASSERT_EQ(type, HSA_DEVICE_TYPE_GPU);

    uint32_t cache_size[4] = {};
    ASSERT_SUCCESS(hsa_agent_get_info(agent, HSA_AGENT_INFO_CACHE_SIZE, cache_size));
    ASSERT_NE(cache_size[0], 0u) << name << " reports no L1 cache";

    uint64_t scratch_max = 0;
    uint64_t scratch_cur = 0;
    ASSERT_SUCCESS(hsa_agent_get_info(
        agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_SCRATCH_LIMIT_MAX), &scratch_max));
    ASSERT_SUCCESS(hsa_agent_get_info(
        agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_SCRATCH_LIMIT_CURRENT),
        &scratch_cur));
    ASSERT_GT(scratch_max, 0u);

    // The agent's own CPU must be a CPU agent.
    hsa_agent_t nearest_cpu = {};
    ASSERT_SUCCESS(hsa_agent_get_info(
        agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_NEAREST_CPU), &nearest_cpu));
    ASSERT_SUCCESS(hsa_agent_get_info(nearest_cpu, HSA_AGENT_INFO_DEVICE, &type));
    ASSERT_EQ(type, HSA_DEVICE_TYPE_CPU);

    uint64_t mem_avail = 0;
    ASSERT_SUCCESS(hsa_agent_get_info(
        agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_MEMORY_AVAIL), &mem_avail));
    ASSERT_NE(mem_avail, 0u);

    // Invalid arguments.
    ASSERT_EQ(hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(0x7FFFF), value),
              HSA_STATUS_ERROR_INVALID_ARGUMENT);
    ASSERT_EQ(hsa_agent_get_info(agent, HSA_AGENT_INFO_WAVEFRONT_SIZE, nullptr),
              HSA_STATUS_ERROR_INVALID_ARGUMENT);

    // Walk regions, caches, ISAs and pools.
    AgentIterCounts counts;
    ASSERT_SUCCESS(hsa_agent_iterate_regions(agent, CountRegion, &counts));
    ASSERT_SUCCESS(hsa_agent_iterate_caches(agent, CountCache, &counts));
    ASSERT_SUCCESS(hsa_agent_iterate_isas(agent, CountIsa, &counts));
    ASSERT_SUCCESS(hsa_amd_agent_iterate_memory_pools(agent, CountPool, &counts));
    EXPECT_GT(counts.regions, 0u) << name;
    EXPECT_GT(counts.caches, 0u) << name;
    EXPECT_GT(counts.isas, 0u) << name;
    EXPECT_GT(counts.pools, 0u) << name;
    EXPECT_TRUE(counts.region_info_ok) << name;
    EXPECT_TRUE(counts.cache_info_ok) << name;
    EXPECT_TRUE(counts.isa_info_ok) << name;

    // A callback that stops the walk has its status returned to the caller.
    EXPECT_EQ(hsa_agent_iterate_regions(agent, StopIteration, nullptr), HSA_STATUS_INFO_BREAK)
        << name;

    if (verbosity() > 0) {
      std::cout << "  " << name << ": regions=" << counts.regions << " caches=" << counts.caches
                << " isas=" << counts.isas << " pools=" << counts.pools << std::endl;
    }
  }

  if (verbosity() > 0) {
    std::cout << "  *** Execution completed - subtest Passed " << " ***" << std::endl;
  }
}

#undef RET_IF_HSA_ERR
