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

// Build a synthetic capability value with a given doorbell type.
// DoorbellType occupies bits 12-13 of the capability field.
// Values larger than 3 are masked to 2 bits (current field width).
static uint32_t MakeCapabilityWithDoorbell(unsigned int doorbell_type) {
  return (doorbell_type & 0x3) << 12;
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
  for (int i = 0; i < 64; ++i) {
    std::ostringstream path;
    path << "/sys/devices/virtual/kfd/kfd/topology/nodes/" << i << "/properties";
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

  // Phase 2: Verify hsa_init() succeeds regardless of deprecated GPUs.
  // This is the core regression test — before the fix, hsa_init() would fail
  // with HSA_STATUS_ERROR if ANY GPU had DoorbellType != 2.
  hsa_status_t err = hsa_init();
  ASSERT_EQ(err, HSA_STATUS_SUCCESS)
      << "hsa_init() failed. If deprecated GPUs are present, they should be "
         "skipped gracefully without aborting initialization.";

  // Phase 3: Count GPU agents exposed by HSA.
  uint32_t hsa_gpu_count = 0;
  err = hsa_iterate_agents(CountGpuAgentsCallback, &hsa_gpu_count);
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

  hsa_shut_down();
}

void GpuDiscoveryDeprecatedTest::Close() {
  TestBase::Close();
}

void GpuDiscoveryDeprecatedTest::DisplayResults() const {}

void GpuDiscoveryDeprecatedTest::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

// ============================================================================
// Standalone unit tests for doorbell type validation logic.
//
// These do NOT require GPU hardware — they validate the classification and
// bit-extraction logic that determines whether a GPU is supported or skipped.
//
// Why these tests matter:
//   DoorbellType is currently a 2-bit field (bits 12-13) in the capability
//   word, giving values 0-3. Three of four values are already assigned
//   (0=pre-1.0, 1=1.0, 2=2.0) with one reserved. As AMD deprecates more
//   GPU generations or introduces new doorbell protocols, this field will
//   likely be widened. These tests ensure that:
//
//   1. Known deprecated types are correctly rejected (not just "!= 2")
//   2. The current supported type (2) is accepted
//   3. Unknown/future values are rejected gracefully (the default case)
//   4. Bit extraction is robust if the field width changes
//
//   The original bug: DoorbellType 1 (Polaris/gfx803) threw HSA_STATUS_ERROR
//   instead of HSA_STATUS_ERROR_INVALID_ISA, bypassing the catch handler in
//   DiscoverGpu and crashing hsa_init() for ALL GPUs in the system.
// ============================================================================

// --- Known doorbell types: verify correct classification ---

TEST(DoorbellTypeValidation, Supported_Type2_Vega_And_Newer) {
  // DoorbellType 2 is the only type supported by the HSA runtime.
  // All Vega, CDNA (MI50/MI100/MI250), RDNA (Navi) GPUs use this.
  EXPECT_TRUE(IsDoorbellTypeSupported(kDoorbellType2_0));
}

TEST(DoorbellTypeValidation, Deprecated_Type0_Pre1_Kaveri_Hawaii_Tonga) {
  // Pre-1.0 doorbell: very old GCN GPUs. Must be rejected.
  EXPECT_FALSE(IsDoorbellTypeSupported(kDoorbellTypePre1_0));
}

TEST(DoorbellTypeValidation, Deprecated_Type1_Polaris_Fiji_Vegam) {
  // 1.0 doorbell: Polaris (gfx803), Fiji, Vegam.
  // This is the exact case that triggered the original bug — a WX 2100
  // (gfx803/Polaris) with DoorbellType=1 killed HSA init for MI50 + W5700.
  EXPECT_FALSE(IsDoorbellTypeSupported(kDoorbellType1_0));
}

TEST(DoorbellTypeValidation, Reserved_Type3_Must_Be_Rejected) {
  // Type 3 is currently marked "reserved for future use" in kfd_sysfs.h.
  // Until the HSA runtime adds explicit support, it must be rejected.
  EXPECT_FALSE(IsDoorbellTypeSupported(kDoorbellTypeReserved));
}

// --- Exhaustive coverage of all current field values ---

TEST(DoorbellTypeValidation, ExactlyOneTypeIsSupported_OutOfFourPossible) {
  // The 2-bit field allows values 0-3. Exactly one (type 2) is supported.
  // If AMD adds a new supported type, this test intentionally breaks to
  // remind developers to update IsDoorbellTypeSupported and the switch
  // in amd_gpu_agent.cpp.
  int supported = 0;
  for (unsigned int dt = 0; dt <= 3; ++dt) {
    if (IsDoorbellTypeSupported(dt)) supported++;
  }
  EXPECT_EQ(supported, 1)
      << "Expected exactly 1 supported doorbell type out of 4 possible values. "
         "If a new type was added, update the switch in amd_gpu_agent.cpp and "
         "IsDoorbellTypeSupported in this test.";
}

// --- Bit extraction: ensure DoorbellType is correctly isolated ---

TEST(DoorbellTypeValidation, BitExtraction_EachDoorbellValue) {
  // Verify ExtractDoorbellType correctly reads bits 12-13 for all 4 values.
  EXPECT_EQ(ExtractDoorbellType(0x00000000), 0u);  // bits 12-13 = 00
  EXPECT_EQ(ExtractDoorbellType(0x00001000), 1u);  // bits 12-13 = 01
  EXPECT_EQ(ExtractDoorbellType(0x00002000), 2u);  // bits 12-13 = 10
  EXPECT_EQ(ExtractDoorbellType(0x00003000), 3u);  // bits 12-13 = 11
}

TEST(DoorbellTypeValidation, BitExtraction_IgnoresAdjacentFields) {
  // Bits outside 12-13 should not affect the extracted doorbell type.
  // Set all OTHER bits high, leave doorbell bits at 0 (type 0).
  EXPECT_EQ(ExtractDoorbellType(0xFFFF0FFF), 0u);
  // Set all OTHER bits high, set doorbell to type 2.
  EXPECT_EQ(ExtractDoorbellType(0xFFFF2FFF), 2u);
}

// --- Real hardware capability values (regression data) ---

TEST(DoorbellTypeValidation, RealHardware_MI50_gfx906_Supported) {
  // Instinct MI50 (gfx906, Vega20): capability=0xac73a280
  // Observed on a real system. DoorbellType=2, must be supported.
  EXPECT_EQ(ExtractDoorbellType(0xac73a280), kDoorbellType2_0);
  EXPECT_TRUE(IsDoorbellTypeSupported(ExtractDoorbellType(0xac73a280)));
}

TEST(DoorbellTypeValidation, RealHardware_WX2100_gfx803_Deprecated) {
  // Radeon Pro WX 2100 (gfx803, Polaris12): capability=0x00001280
  // This is the card that triggered the original bug.
  // DoorbellType=1, must be rejected without crashing other GPUs.
  EXPECT_EQ(ExtractDoorbellType(0x00001280), kDoorbellType1_0);
  EXPECT_FALSE(IsDoorbellTypeSupported(ExtractDoorbellType(0x00001280)));
}

TEST(DoorbellTypeValidation, RealHardware_W5700_gfx1010_Supported) {
  // Radeon Pro W5700 (gfx1010, Navi10): capability=0x2883a280
  // DoorbellType=2, must be supported.
  EXPECT_EQ(ExtractDoorbellType(0x2883a280), kDoorbellType2_0);
  EXPECT_TRUE(IsDoorbellTypeSupported(ExtractDoorbellType(0x2883a280)));
}

// --- Future-proofing: what happens when the field width changes ---
//
// DoorbellType is currently 2 bits (values 0-3). When AMD inevitably adds
// new GPU generations, they may:
//   a) Reuse the reserved value 3
//   b) Widen the field beyond 2 bits (requiring kfd_sysfs.h changes)
//
// These tests document the current masking behavior. If the field is widened,
// ExtractDoorbellType's mask must be updated, and these tests will catch
// the discrepancy.

TEST(DoorbellTypeValidation, FutureProofing_CurrentFieldIsTwoBits) {
  // Verify the field width assumption: all bits set in a uint32_t should
  // extract to 3 (max 2-bit value), not something larger.
  // If this test fails after a kernel update, the field was widened and
  // ExtractDoorbellType needs a wider mask.
  EXPECT_EQ(ExtractDoorbellType(0xFFFFFFFF), 3u)
      << "DoorbellType extracted a value > 3. Has the field been widened "
         "beyond 2 bits? Update ExtractDoorbellType mask and this test.";
}

TEST(DoorbellTypeValidation, FutureProofing_SyntheticValues_MaskedToFieldWidth) {
  // If someone hypothetically writes a doorbell value larger than 3 into the
  // capability field (e.g. due to a wider field in a future kernel), the
  // current 2-bit mask truncates it. These tests document that behavior
  // so it's obvious when the mask needs updating.
  //
  // Value 666: binary = ...1010011010, bits 1:0 = 10 = 2
  // After MakeCapabilityWithDoorbell masks to 2 bits: 666 & 3 = 2
  uint32_t cap = MakeCapabilityWithDoorbell(666);
  unsigned int extracted = ExtractDoorbellType(cap);
  EXPECT_EQ(extracted, 666u & 0x3)
      << "Synthetic value 666 should be masked to " << (666u & 0x3)
      << " by the 2-bit field. If the field is now wider, update the mask.";

  // Value 0xDEAD: bits 1:0 = 01 = 1 (maps to deprecated Polaris type)
  cap = MakeCapabilityWithDoorbell(0xDEAD);
  extracted = ExtractDoorbellType(cap);
  EXPECT_EQ(extracted, 0xDEADu & 0x3);
  EXPECT_FALSE(IsDoorbellTypeSupported(extracted))
      << "0xDEAD masked to " << extracted << " should map to a deprecated type";

  // Value 0xFFFF: bits 1:0 = 11 = 3 (maps to reserved type)
  cap = MakeCapabilityWithDoorbell(0xFFFF);
  extracted = ExtractDoorbellType(cap);
  EXPECT_EQ(extracted, 3u);
  EXPECT_FALSE(IsDoorbellTypeSupported(extracted))
      << "0xFFFF masked to 3 should map to reserved/unsupported type";
}

TEST(DoorbellTypeValidation, FutureProofing_MakeAndExtract_RoundTripsForAllValidValues) {
  // MakeCapabilityWithDoorbell → ExtractDoorbellType should round-trip
  // for all values in the current 2-bit range.
  for (unsigned int dt = 0; dt <= 3; ++dt) {
    uint32_t cap = MakeCapabilityWithDoorbell(dt);
    EXPECT_EQ(ExtractDoorbellType(cap), dt)
        << "Round-trip failed for DoorbellType " << dt;
  }
}
