/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Pins ncclChannelMaskNthChannelId, not its ncclKernelMain call sites: block or warp n gets the n-th set bit.

#include "DeviceTestBase.hpp"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "device.h"

namespace RcclUnitTesting {

namespace {

constexpr int kThreadsPerBlock = 256;  // NCCL_MAX_NTHREADS; only warp 0 runs the lookup, as in ncclKernelMain.
constexpr int kMaxWarpsPerBlock = kThreadsPerBlock / 32;  // Wave32 has the most warps per block.
constexpr int kMinWarpsPerBlock = kThreadsPerBlock / 64;

// Mirrors ncclKernelMain's case 0, and also counts owning lanes and reports the compiled WARP_SIZE.
__global__ void kernelBlockToChannel(channelMasks mask, int* channelIds, int* owners, int* deviceWarpSize) {
  __shared__ int channelId;
  __shared__ int ownerCount;
  if (threadIdx.x == 0) {
    channelId = -1;
    ownerCount = 0;
  }
  __syncthreads();
  if (threadIdx.x < WARP_SIZE) {
    int id = ncclChannelMaskNthChannelId(mask, blockIdx.x, threadIdx.x);
    if (id >= 0) {
      channelId = id;
      atomicAdd(&ownerCount, 1);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    channelIds[blockIdx.x] = channelId;
    owners[blockIdx.x] = ownerCount;
    if (blockIdx.x == 0) {
      *deviceWarpSize = WARP_SIZE;
    }
  }
}

// Mirrors the ENABLE_WARP_SPEED per-warp lookup in ncclKernelMain.
__global__ void kernelWarpToChannel(channelMasks mask, int* channelIds, int* owners, int* deviceWarpSize) {
  __shared__ int warpChannelId[kMaxWarpsPerBlock];
  __shared__ int warpOwners[kMaxWarpsPerBlock];
  const int warpCount = blockDim.x / WARP_SIZE;
  const int localWarpId = threadIdx.x / WARP_SIZE;
  const int globalWarpId = warpCount * blockIdx.x + localWarpId;
  const int laneId = threadIdx.x % WARP_SIZE;
  if (laneId == 0) {
    warpChannelId[localWarpId] = -1;
    warpOwners[localWarpId] = 0;
  }
  __syncthreads();
  int id = ncclChannelMaskNthChannelId(mask, globalWarpId, laneId);
  if (id >= 0) {
    warpChannelId[localWarpId] = id;
    atomicAdd(&warpOwners[localWarpId], 1);
  }
  __syncthreads();
  if (laneId == 0) {
    channelIds[globalWarpId] = warpChannelId[localWarpId];
    owners[globalWarpId] = warpOwners[localWarpId];
    if (globalWarpId == 0) {
      *deviceWarpSize = WARP_SIZE;
    }
  }
}

struct MaskCase {
  std::string name;
  std::vector<int> channels;  // Ascending; block or warp n must map to channels[n].
};

// Names the failing case instead of dumping its raw bytes.
void PrintTo(const MaskCase& maskCase, std::ostream* os) {
  *os << maskCase.name;
}

std::vector<int> Range(int lo, int hi, int step = 1) {
  std::vector<int> v;
  for (int c = lo; c < hi; c += step) {
    v.push_back(c);
  }
  return v;
}

channelMasks MaskOf(const std::vector<int>& channels) {
  channelMasks mask = {};
  for (int c : channels) {
    mask.masks[c / CHANNELS_PER_MASK_WORD] |= 1ull << (c % CHANNELS_PER_MASK_WORD);
  }
  return mask;
}

std::vector<MaskCase> MaskCases() {
  return {
    {"Dense0To127", Range(0, 128)},
    {"Dense0To255", Range(0, 256)},
    {"Only64To95", Range(64, 96)},
    {"Only96To127", Range(96, 128)},
    {"EveryOther0To255", Range(0, 256, 2)},
    {"SparsePerWord", {1, 33, 63, 64, 70, 95, 100, 127, 130, 191, 200, 255}},
    {"LowBitsEachWord", {0, 1, 2, 3, 64, 65, 66, 67, 128, 129, 192, 193}},
  };
}

// Slot s must hold want[s] with one owning lane; slots past the end hold -1 with none. Returns "" if all match.
std::string Mismatches(const std::vector<int>& want, const std::vector<int>& got, const std::vector<int>& owners) {
  std::ostringstream report;
  int bad = 0;
  for (size_t s = 0; s < got.size(); s++) {
    const int wantId = s < want.size() ? want[s] : -1;
    const int wantOwners = s < want.size() ? 1 : 0;
    if (got[s] != wantId || owners[s] != wantOwners) {
      if (bad < 8) {
        report << " [slot " << s << ": channel " << got[s] << " (" << owners[s] << " owners), want " << wantId << "]";
      }
      bad++;
    }
  }
  if (bad == 0) {
    return "";
  }
  return std::to_string(bad) + " of " + std::to_string(got.size()) + " slots wrong:" + report.str();
}

}  // namespace

class ChannelMaskMapTest : public DeviceTestBase, public ::testing::WithParamInterface<MaskCase> {
protected:
  int HostWarpSize() {
    int device = 0;
    int warpSize = 0;
    EXPECT_EQ(hipGetDevice(&device), hipSuccess);
    EXPECT_EQ(hipDeviceGetAttribute(&warpSize, hipDeviceAttributeWarpSize, device), hipSuccess);
    return warpSize > 0 ? warpSize : 32;
  }

  // A wave32 GPU must run the wave32 code object, else the WARP_SIZE + lane path is not what was tested.
  void ExpectCompiledWarpSize(const DeviceBuffer<int>& deviceWarpSize, int* compiledWarpSizeOut = nullptr) {
    const int compiledWarpSize = deviceWarpSize.download();
    if (compiledWarpSizeOut != nullptr) {
      *compiledWarpSizeOut = compiledWarpSize;
    }
    RecordProperty("compiledWarpSize", compiledWarpSize);
    std::cout << "[ INFO     ] device code WARP_SIZE " << compiledWarpSize << std::endl;
    ASSERT_EQ(compiledWarpSize, HostWarpSize());
  }
};

TEST_P(ChannelMaskMapTest, BlockMapsToNthSetBit) {
  const std::vector<int>& want = GetParam().channels;
  const int slots = static_cast<int>(want.size()) + 1;  // One extra block past the last enabled channel.
  DeviceBuffer<int> channelIds(slots), owners(slots), deviceWarpSize(1);

  kernelBlockToChannel<<<slots, kThreadsPerBlock>>>(MaskOf(want), channelIds.ptr, owners.ptr, deviceWarpSize.ptr);
  syncAndCheck();

  ASSERT_NO_FATAL_FAILURE(ExpectCompiledWarpSize(deviceWarpSize));
  EXPECT_EQ(Mismatches(want, channelIds.copyTo(), owners.copyTo()), "") << "warpSize " << HostWarpSize();
}

TEST_P(ChannelMaskMapTest, WarpMapsToNthSetBit) {
  const std::vector<int>& want = GetParam().channels;
  const int blocks = static_cast<int>(want.size()) / kMinWarpsPerBlock + 1;  // At least one spare warp at the end.
  DeviceBuffer<int> channelIds(blocks * kMaxWarpsPerBlock), owners(blocks * kMaxWarpsPerBlock), deviceWarpSize(1);

  kernelWarpToChannel<<<blocks, kThreadsPerBlock>>>(MaskOf(want), channelIds.ptr, owners.ptr, deviceWarpSize.ptr);
  syncAndCheck();

  int compiledWarpSize = 0;
  ASSERT_NO_FATAL_FAILURE(ExpectCompiledWarpSize(deviceWarpSize, &compiledWarpSize));
  const size_t slots = static_cast<size_t>(blocks) * (kThreadsPerBlock / compiledWarpSize);
  std::vector<int> got = channelIds.copyTo();
  std::vector<int> gotOwners = owners.copyTo();
  got.resize(slots);
  gotOwners.resize(slots);
  EXPECT_EQ(Mismatches(want, got, gotOwners), "") << "warpSize " << compiledWarpSize;
}

INSTANTIATE_TEST_SUITE_P(Masks, ChannelMaskMapTest, ::testing::ValuesIn(MaskCases()),
                         [](const ::testing::TestParamInfo<MaskCase>& info) {
                           return info.param.name;
                         });

}  // namespace RcclUnitTesting
