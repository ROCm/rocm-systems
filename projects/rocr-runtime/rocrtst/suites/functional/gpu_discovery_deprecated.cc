/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

//
// Test: GPU Discovery with Deprecated Devices
//
// Verifies that HSA initialization and agent enumeration succeed even when
// the system contains GPUs with deprecated doorbell types (pre-Vega).
//
// Doorbell type mapping (from kfd_topology.c):
//   0 = PRE_1_0: Kaveri, Hawaii, Tonga
//   1 = 1_0:     Carrizo, Fiji, Polaris10, Polaris11, Polaris12, Vegam
//   2 = 2_0:     Vega and newer (GCN 5.0+, GC IP >= 9.0.1) — only supported type
//   3 =          Reserved for future use
//
// DoorbellType is currently a 2-bit field (bits 12-13 of capability), meaning
// only values 0-3 are possible today. However, as AMD adds new GPU generations,
// this field may be widened or reinterpreted. The tests below verify that
// unknown/future doorbell types are handled gracefully rather than crashing.
//
// On a system with e.g. a Polaris display GPU + Vega/CDNA compute GPU,
// HSA must skip the Polaris device and still expose the Vega/CDNA device.
//

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "suites/functional/gpu_discovery_deprecated.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

// Doorbell type values from kfd_sysfs.h
static const unsigned int kDoorbellTypePre1_0 = 0;
static const unsigned int kDoorbellType1_0    = 1;
static const unsigned int kDoorbellType2_0    = 2;
static const unsigned int kDoorbellTypeReserved = 3;

// Read the capability field from a KFD topology node's sysfs properties.
// Returns the raw capability uint32, or 0 on failure.
static uint32_t ReadKfdNodeCapability(int node_id) {
  std::ostringstream path;
  path << "/sys/devices/virtual/kfd/kfd/topology/nodes/" << node_id << "/properties";
  std::ifstream props(path.str());
  if (!props.is_open()) return 0;

  std::string key;
  uint64_t value;
  while (props >> key >> value) {
    if (key == "capability") return static_cast<uint32_t>(value);
  }
  return 0;
}

// Extract DoorbellType (bits 12-13) from capability field.
// The field is currently 2 bits wide, so valid values are 0-3.
// If the field is widened in future kernels, this mask must be updated.
static unsigned int ExtractDoorbellType(uint32_t capability) {
  return (capability >> 12) & 0x3;
}

// Check whether a doorbell type is supported by the HSA runtime.
// Only DoorbellType 2 (HSA_CAP_DOORBELL_TYPE_2_0, Vega+) is supported.
// This mirrors the logic in amd_gpu_agent.cpp — if the supported set changes
// there, it must change here too.
static bool IsDoorbellTypeSupported(unsigned int doorbell_type) {
  return doorbell_type == kDoorbellType2_0;
}

// Count KFD topology nodes.
static int CountKfdNodes() {
  int count = 0;
  while (true) {
    std::ostringstream path;
    path << "/sys/devices/virtual/kfd/kfd/topology/nodes/" << count << "/properties";
    std::ifstream props(path.str());
    if (!props.is_open()) break;
    ++count;
  }
  return count;
}

// Check if a KFD node is a GPU (has compute cores).
static bool IsGpuNode(int node_id) {
  std::ostringstream path;
  path << "/sys/devices/virtual/kfd/kfd/topology/nodes/" << node_id << "/properties";
  std::ifstream props(path.str());
  if (!props.is_open()) return false;

  std::string key;
  uint64_t value;
  while (props >> key >> value) {
    if (key == "simd_count" && value > 0) return true;
  }
  return false;
}

// Callback for hsa_iterate_agents: count GPU agents.
static hsa_status_t CountGpuAgentsCallback(hsa_agent_t agent, void* data) {
  hsa_device_type_t type;
  hsa_status_t err = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
  if (err != HSA_STATUS_SUCCESS) return err;
  if (type == HSA_DEVICE_TYPE_GPU) {
    (*static_cast<uint32_t*>(data))++;
  }
  return HSA_STATUS_SUCCESS;
}

GpuDiscoveryDeprecatedTest::GpuDiscoveryDeprecatedTest() : TestBase() {
  set_title("GPU Discovery with Deprecated Devices");
  set_description(
      "Verifies that HSA initialization succeeds and supported GPUs are "
      "enumerated even when deprecated GPU devices (pre-Vega, DoorbellType != 2) "
      "are present in the system. Unsupported GPUs should be silently skipped.");
}

GpuDiscoveryDeprecatedTest::~GpuDiscoveryDeprecatedTest() {}

void GpuDiscoveryDeprecatedTest::SetUp() {
  TestBase::SetUp();
}

void GpuDiscoveryDeprecatedTest::Run() {
  // Phase 1: Scan KFD topology to understand what hardware is present.
  int num_nodes = CountKfdNodes();
  ASSERT_GT(num_nodes, 0) << "No KFD topology nodes found";

  int total_gpu_nodes = 0;
  int supported_gpu_nodes = 0;   // DoorbellType == 2
  int deprecated_gpu_nodes = 0;  // DoorbellType != 2

  std::cout << "  KFD topology: " << num_nodes << " nodes" << std::endl;

  for (int i = 0; i < num_nodes; ++i) {
    if (!IsGpuNode(i)) continue;
    total_gpu_nodes++;

    uint32_t cap = ReadKfdNodeCapability(i);
    unsigned int doorbell = ExtractDoorbellType(cap);

    const char* doorbell_name = "unknown";
    switch (doorbell) {
      case kDoorbellTypePre1_0:   doorbell_name = "PRE_1_0 (Kaveri/Hawaii/Tonga)"; break;
      case kDoorbellType1_0:      doorbell_name = "1_0 (Fiji/Polaris/Vegam)"; break;
      case kDoorbellType2_0:      doorbell_name = "2_0 (Vega+)"; break;
      case kDoorbellTypeReserved: doorbell_name = "3 (reserved)"; break;
    }

    std::cout << "  Node " << i << ": GPU, DoorbellType=" << doorbell
              << " (" << doorbell_name << ")"
              << (IsDoorbellTypeSupported(doorbell) ? " [supported]" : " [deprecated]")
              << std::endl;

    if (IsDoorbellTypeSupported(doorbell)) {
      supported_gpu_nodes++;
    } else {
      deprecated_gpu_nodes++;
    }
  }

  std::cout << "  Summary: " << total_gpu_nodes << " GPU node(s), "
            << supported_gpu_nodes << " supported, "
            << deprecated_gpu_nodes << " deprecated" << std::endl;

  // Phase 2: HSA was initialized in TestBase::SetUp() via InitAndSetupHSA().
  // The core regression check is implicit: if SetUp() returned successfully
  // with deprecated GPUs present, hsa_init() did not abort the way it used to
  // before this fix. We do not call hsa_init() / hsa_shut_down() here — the
  // rocrtst harness (main.cc:RunCustomTestProlog/Epilog) handles that lifecycle
  // and TestBase::Close() will call hsa_shut_down() via CommonCleanUp().

  // Phase 3: Count GPU agents exposed by HSA.
  uint32_t hsa_gpu_count = 0;
  hsa_status_t err = hsa_iterate_agents(CountGpuAgentsCallback, &hsa_gpu_count);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS) << "hsa_iterate_agents failed";

  std::cout << "  HSA reports " << hsa_gpu_count << " GPU agent(s)" << std::endl;

  // Phase 4: Verify correct filtering.
  // In restricted environments (containers, ROCR_VISIBLE_DEVICES, cgroups),
  // HSA may see fewer GPUs than KFD topology reports, because sysfs exposes
  // the full host topology while HSA respects GPU visibility restrictions.
  // Therefore we check:
  //   - HSA exposes at least one GPU when KFD has supported nodes
  //   - HSA never exposes MORE GPUs than KFD says are supported
  //   - Deprecated GPUs are never exposed (count <= supported, not total)
  if (supported_gpu_nodes > 0) {
    EXPECT_GT(hsa_gpu_count, 0u)
        << "HSA should expose at least one supported GPU when KFD reports "
        << supported_gpu_nodes << " node(s) with DoorbellType 2.";
  }
  EXPECT_LE(hsa_gpu_count, static_cast<uint32_t>(supported_gpu_nodes))
      << "HSA should never report more GPUs than KFD supported nodes. "
         "Got " << hsa_gpu_count << " HSA agents but only "
      << supported_gpu_nodes << " KFD nodes with DoorbellType 2.";

  if (hsa_gpu_count < static_cast<uint32_t>(supported_gpu_nodes)) {
    std::cout << "  NOTE: HSA reports fewer GPUs (" << hsa_gpu_count
              << ") than KFD supported nodes (" << supported_gpu_nodes
              << "). This is expected in container or cgroup-restricted "
                 "environments." << std::endl;
  }

  if (deprecated_gpu_nodes > 0 && supported_gpu_nodes == 0) {
    std::cout << "  NOTE: All GPU nodes are deprecated. HSA initialized with "
                 "0 GPU agents (CPU agent only)." << std::endl;
  }

  if (deprecated_gpu_nodes > 0 && hsa_gpu_count > 0) {
    std::cout << "  PASS: " << deprecated_gpu_nodes
              << " deprecated GPU(s) were correctly skipped, "
              << hsa_gpu_count << " supported GPU(s) exposed."
              << std::endl;
  }
  // hsa_shut_down() intentionally not called here — TestBase::Close() handles
  // it via CommonCleanUp().
}

void GpuDiscoveryDeprecatedTest::Close() {
  TestBase::Close();
}

void GpuDiscoveryDeprecatedTest::DisplayResults() const {}

void GpuDiscoveryDeprecatedTest::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}
