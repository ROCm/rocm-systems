// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Unit tests for cross-plugin device correlation: making two differently
// ordered backends agree on which physical GPU is which, tested in isolation
// with synthetic plugin device lists.

#include "device_registry.h"

#include <gtest/gtest.h>

#include <cstring>

using namespace gpumetrics;

namespace {

gpum_device_identity Id(uint64_t bdf, uint32_t kfd, uint32_t socket = GPUM_SOCKET_UNKNOWN,
                        int32_t partition = -1, uint32_t local = 0) {
  gpum_device_identity id{};
  id.bdf = bdf;
  id.oam_id = GPUM_ID_UNKNOWN;
  id.kfd_node_id = kfd;
  id.socket_id = socket;
  id.partition_index = partition;
  id.plugin_local_index = local;
  return id;
}

gpum_device_identity IdUuid(const char* tag) {
  gpum_device_identity id{};
  id.oam_id = GPUM_ID_UNKNOWN;
  std::memset(id.uuid, 0, 16);
  std::strncpy(reinterpret_cast<char*>(id.uuid), tag, 15);
  id.partition_index = -1;
  return id;
}

}  // namespace

// --- SamePhysicalGpu: key priority ----------------------------------------

TEST(SamePhysicalGpu, MatchesByBdf) {
  EXPECT_TRUE(SamePhysicalGpu(Id(0x6300, 0), Id(0x6300, 0)));
}

TEST(SamePhysicalGpu, BdfMismatchWinsOverKfdMatch) {
  // BDF is highest priority: differing BDFs mean different GPUs even if kfd ids
  // collide.
  EXPECT_FALSE(SamePhysicalGpu(Id(0x6300, 7), Id(0x6400, 7)));
}

TEST(SamePhysicalGpu, FallsBackToKfdWhenBdfMissing) {
  EXPECT_TRUE(SamePhysicalGpu(Id(0, 5), Id(0, 5)));
  EXPECT_FALSE(SamePhysicalGpu(Id(0, 5), Id(0, 6)));
}

TEST(SamePhysicalGpu, OneSideMissingBdfFallsToKfd) {
  // One side has BDF, the other only kfd: they share only kfd, so match on it.
  EXPECT_TRUE(SamePhysicalGpu(Id(0x6300, 9), Id(0, 9)));
}

TEST(SamePhysicalGpu, FallsBackToUuid) {
  EXPECT_TRUE(SamePhysicalGpu(IdUuid("GPU-abc"), IdUuid("GPU-abc")));
  EXPECT_FALSE(SamePhysicalGpu(IdUuid("GPU-abc"), IdUuid("GPU-xyz")));
}

TEST(SamePhysicalGpu, NoSharedKeyIsNotAMatch) {
  gpum_device_identity a{};
  gpum_device_identity b{};
  a.oam_id = GPUM_ID_UNKNOWN;
  b.oam_id = GPUM_ID_UNKNOWN;
  a.bdf = 0x6300;      // only bdf
  b.kfd_node_id = 3;   // only kfd
  EXPECT_FALSE(SamePhysicalGpu(a, b));
}

TEST(SamePhysicalGpu, MatchesByMaskedBdfAcrossPartitions) {
  // Two CPX partitions of one GPU: same bus/device, different function and
  // kfd/uuid. The masked BDF still groups them.
  EXPECT_TRUE(SamePhysicalGpu(Id(0x6300 + 0, 100), Id(0x6300 + 1, 101)));
  // Different device does not match.
  EXPECT_FALSE(SamePhysicalGpu(Id(0x6300, 100), Id(0x6308, 101)));
}

TEST(SamePhysicalGpu, MatchesByOamWhenBdfAbsent) {
  gpum_device_identity a{}, b{};
  a.oam_id = 6;
  b.oam_id = 6;
  EXPECT_TRUE(SamePhysicalGpu(a, b));
  b.oam_id = 7;
  EXPECT_FALSE(SamePhysicalGpu(a, b));
}

// --- DeviceRegistry::Build ------------------------------------------------

TEST(DeviceRegistry, CorrelatesTwoPluginsSameGpu) {
  // Both plugins report the same 2 GPUs with different local indices and key
  // coverage. Expect 2 canonical GPUs, each with both providers.
  std::vector<PluginDeviceRef> refs = {
      {"amdsmi", Id(0x6300, 100, /*socket*/ 0, -1, /*local*/ 0), "W6800"},
      {"amdsmi", Id(0x6400, 101, 1, -1, 1), "W6800"},
      // rocprofiler reports them in a DIFFERENT order, no socket hint.
      {"rocprofiler", Id(0x6400, 101, GPUM_SOCKET_UNKNOWN, -1, 0), "gfx1030"},
      {"rocprofiler", Id(0x6300, 100, GPUM_SOCKET_UNKNOWN, -1, 1), "gfx1030"},
  };
  auto reg = DeviceRegistry::Build(refs, {"amdsmi", "rocprofiler"});
  ASSERT_EQ(reg.devices().size(), 2u);

  // Ordinals assigned by socket then BDF: gpu0 = bdf 0x6300, gpu1 = 0x6400.
  EXPECT_EQ(reg.devices()[0].identity.bdf, 0x6300u);
  EXPECT_EQ(reg.devices()[1].identity.bdf, 0x6400u);
  // amdsmi wins the name (higher provider priority).
  EXPECT_EQ(reg.devices()[0].name, "W6800");
  // both providers attached
  EXPECT_EQ(reg.devices()[0].providers.size(), 2u);
}

TEST(DeviceRegistry, RoutingMapsCanonicalToPluginLocalIndex) {
  std::vector<PluginDeviceRef> refs = {
      {"amdsmi", Id(0x6300, 100, 0, -1, /*local*/ 5), "A"},
      {"rocprofiler", Id(0x6300, 100, GPUM_SOCKET_UNKNOWN, -1, /*local*/ 9), "A"},
  };
  auto reg = DeviceRegistry::Build(refs, {"amdsmi", "rocprofiler"});
  ASSERT_EQ(reg.devices().size(), 1u);
  auto handles = reg.handlesFor(reg.devices()[0].id);
  ASSERT_EQ(handles.size(), 2u);
  // amdsmi local index 5, rocprofiler local index 9
  bool saw_amdsmi = false, saw_rocp = false;
  for (const auto& h : handles) {
    if (h.plugin == "amdsmi") {
      EXPECT_EQ(h.plugin_local_index, 5u);
      saw_amdsmi = true;
    }
    if (h.plugin == "rocprofiler") {
      EXPECT_EQ(h.plugin_local_index, 9u);
      saw_rocp = true;
    }
  }
  EXPECT_TRUE(saw_amdsmi && saw_rocp);
}

TEST(DeviceRegistry, PartitionsBecomeSeparateEntities) {
  // One physical GPU with two partitions reported by amdsmi (whole + 2 parts).
  std::vector<PluginDeviceRef> refs = {
      {"amdsmi", Id(0x6300, 100, 0, /*part*/ -1, 0), "MI300"},
      {"amdsmi", Id(0x6300, 100, 0, /*part*/ 0, 1), "MI300"},
      {"amdsmi", Id(0x6300, 100, 0, /*part*/ 1, 2), "MI300"},
  };
  auto reg = DeviceRegistry::Build(refs, {"amdsmi"});
  ASSERT_EQ(reg.devices().size(), 1u);
  EXPECT_EQ(reg.devices()[0].partitions.size(), 2u);

  gpum_entity_id part0;
  part0.kind = GPUM_ENTITY_GPU_PARTITION;
  part0.socket = 0;
  part0.gpu = 0;
  part0.partition = 1;
  auto handles = reg.handlesFor(part0);
  ASSERT_EQ(handles.size(), 1u);
  EXPECT_EQ(handles[0].plugin_local_index, 2u);  // partition 1 -> local 2
  EXPECT_EQ(handles[0].partition, 1);
}

TEST(DeviceRegistry, SocketGroupingFromHint) {
  std::vector<PluginDeviceRef> refs = {
      {"amdsmi", Id(0x6300, 100, /*socket*/ 0, -1, 0), "A"},
      {"amdsmi", Id(0x6400, 101, /*socket*/ 0, -1, 1), "B"},
      {"amdsmi", Id(0x6500, 102, /*socket*/ 1, -1, 2), "C"},
  };
  auto reg = DeviceRegistry::Build(refs, {"amdsmi"});
  ASSERT_EQ(reg.sockets().size(), 2u);
  EXPECT_EQ(reg.sockets()[0].gpus.size(), 2u);
  EXPECT_EQ(reg.sockets()[1].gpus.size(), 1u);
}

TEST(DeviceRegistry, UnknownSocketHintGivesEachGpuOwnSocket) {
  std::vector<PluginDeviceRef> refs = {
      {"rocprofiler", Id(0x6300, 100, GPUM_SOCKET_UNKNOWN, -1, 0), "A"},
      {"rocprofiler", Id(0x6400, 101, GPUM_SOCKET_UNKNOWN, -1, 1), "B"},
  };
  auto reg = DeviceRegistry::Build(refs, {"rocprofiler"});
  EXPECT_EQ(reg.sockets().size(), 2u);
}

TEST(DeviceRegistry, LookupByBdfAndUuid) {
  std::vector<PluginDeviceRef> refs = {
      {"amdsmi", Id(0x6300, 100, 0, -1, 0), "A"},
  };
  auto reg = DeviceRegistry::Build(refs, {"amdsmi"});
  ASSERT_NE(reg.gpuByBdf(0x6300), nullptr);
  EXPECT_EQ(reg.gpuByBdf(0x6300)->id.gpu, 0u);
  EXPECT_EQ(reg.gpuByBdf(0x9999), nullptr);
}
