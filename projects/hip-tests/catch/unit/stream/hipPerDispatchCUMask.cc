/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * Per-dispatch CU mask validation via hipLaunchAttributeExtCUMask.
 *
 * Uses __smid() to read the hardware SE/WGP identity from each wave and
 * verifies that dispatches with different CU masks execute on disjoint
 * CU sets matching the requested masks.
 *
 * Requires gfx1250 (Format 4 / launch descriptor CU enable support).
 * Skipped on all other architectures.
 */

#include <hip_test_common.hh>
#include <vector>
#include <cstring>

static bool isGfx1250() {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  // gcnArchName is e.g. "gfx1250" or "gfx1250:sramecc+:xnack-"
  return strncmp(prop.gcnArchName, "gfx1250", 7) == 0;
}

static int getNumCUs() {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  return prop.multiProcessorCount;
}

// Kernel: each workgroup records its SE and WGP into a bitmap.
// __smid() returns a packed value: on gfx12, bits encode (xcc, se, sa, wgp, cu).
// We extract SE and WGP to build a flat CU index = se * wgpsPerSE + wgp.
__global__ void probe_cu_ids(uint32_t* cu_bitmap, uint32_t wgpsPerSE) {
  if (threadIdx.x != 0) return;  // one thread per block is enough

  unsigned smid = __smid();
  // __smid() packing for gfx12: cu_bits(0) | wgp_bits(4) | sa_bits(1) | se_bits(4) | xcc
  // Extract WGP (bits 0-3), SA (bit 4), SE (bits 5-8) for gfx12
  unsigned wgp = smid & 0xF;
  unsigned sa  = (smid >> 4) & 0x1;
  unsigned se  = (smid >> 5) & 0xF;

  // Flat CU index: combine SE and WGP
  unsigned flat_id = se * wgpsPerSE + wgp;
  atomicOr(&cu_bitmap[flat_id / 32], 1u << (flat_id % 32));
}

// Helper: build a contiguous CU mask bitmask for [startCU, startCU+count) across numSEs
static std::vector<uint32_t> buildCUMask(uint32_t startCU, uint32_t count,
                                          uint32_t cusPerSE, uint32_t numSEs) {
  uint32_t totalCUs = numSEs * cusPerSE;
  uint32_t numWords = (totalCUs + 31) / 32;
  std::vector<uint32_t> mask(numWords, 0);
  for (uint32_t se = 0; se < numSEs; ++se) {
    for (uint32_t cu = startCU; cu < startCU + count && cu < cusPerSE; ++cu) {
      uint32_t bitIdx = se * cusPerSE + cu;
      mask[bitIdx / 32] |= (1u << (bitIdx % 32));
    }
  }
  return mask;
}

// Helper: collect set bits from a bitmap
static std::vector<uint32_t> collectSetBits(const uint32_t* bitmap, uint32_t numWords) {
  std::vector<uint32_t> bits;
  for (uint32_t w = 0; w < numWords; ++w) {
    for (uint32_t b = 0; b < 32; ++b) {
      if (bitmap[w] & (1u << b)) {
        bits.push_back(w * 32 + b);
      }
    }
  }
  return bits;
}

/**
 * Test: launch the same kernel twice with different per-dispatch CU masks.
 * Verify the observed CU sets are within the respective masks.
 */
TEST_CASE("PerDispatchCUMask_DisjointMasks") {
  if (!isGfx1250()) {
    HipTest::HIP_SKIP_TEST("Per-dispatch CU mask requires gfx1250");
    return;
  }

  int numCUs = getNumCUs();
  REQUIRE(numCUs >= 4);

  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));

  // Estimate WGPs per SE (WGP = 2 CUs in WGP mode)
  // multiProcessorCount reports WGPs on RDNA
  uint32_t numSEs = 2;  // gfx1250 has 2 SEs
  uint32_t wgpsPerSE = numCUs / numSEs;

  uint32_t totalCUs = numSEs * wgpsPerSE;
  uint32_t bitmapWords = (totalCUs + 31) / 32;

  // Allocate bitmap on device
  uint32_t* d_bitmap;
  HIP_CHECK(hipMalloc(&d_bitmap, bitmapWords * sizeof(uint32_t)));

  // ---- Dispatch 1: CUs [0, half) in each SE ----
  uint32_t half = wgpsPerSE / 2;
  REQUIRE(half > 0);
  REQUIRE(half <= 15);  // must fit in 4-bit cu_count

  auto maskA = buildCUMask(0, half, wgpsPerSE, numSEs);
  hipExtCUMaskConfig cuCfgA = {
    .cuMaskSize = static_cast<uint32_t>(maskA.size()),
    .cuMask = maskA.data()
  };

  HIP_CHECK(hipMemset(d_bitmap, 0, bitmapWords * sizeof(uint32_t)));

  hipLaunchAttribute attrA = {};
  attrA.id = hipLaunchAttributeExtCUMask;
  attrA.val.cuMask = &cuCfgA;

  hipLaunchConfig_t configA = {};
  configA.gridDim = dim3(numCUs * 16);  // oversaturate to hit all enabled CUs
  configA.blockDim = dim3(64);
  configA.numAttrs = 1;
  configA.attrs = &attrA;
  configA.stream = nullptr;

  void* argsA[] = {&d_bitmap, &wgpsPerSE};
  HIP_CHECK(hipLaunchKernelExC(&configA, reinterpret_cast<const void*>(probe_cu_ids), argsA));
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<uint32_t> hostBitmapA(bitmapWords);
  HIP_CHECK(hipMemcpy(hostBitmapA.data(), d_bitmap, bitmapWords * sizeof(uint32_t),
                       hipMemcpyDeviceToHost));
  auto observedA = collectSetBits(hostBitmapA.data(), bitmapWords);

  // Verify: all observed CUs should be within maskA
  for (auto cu : observedA) {
    uint32_t word = cu / 32;
    uint32_t bit = cu % 32;
    CAPTURE(cu, word, bit);
    REQUIRE(word < maskA.size());
    CHECK((maskA[word] & (1u << bit)) != 0);
  }
  // Should have observed at least some CUs
  CHECK(!observedA.empty());

  // ---- Dispatch 2: CUs [half, wgpsPerSE) in each SE ----
  uint32_t countB = wgpsPerSE - half;
  if (countB > 15) countB = 15;

  auto maskB = buildCUMask(half, countB, wgpsPerSE, numSEs);
  hipExtCUMaskConfig cuCfgB = {
    .cuMaskSize = static_cast<uint32_t>(maskB.size()),
    .cuMask = maskB.data()
  };

  HIP_CHECK(hipMemset(d_bitmap, 0, bitmapWords * sizeof(uint32_t)));

  hipLaunchAttribute attrB = {};
  attrB.id = hipLaunchAttributeExtCUMask;
  attrB.val.cuMask = &cuCfgB;

  hipLaunchConfig_t configB = {};
  configB.gridDim = dim3(numCUs * 16);
  configB.blockDim = dim3(64);
  configB.numAttrs = 1;
  configB.attrs = &attrB;
  configB.stream = nullptr;

  void* argsB[] = {&d_bitmap, &wgpsPerSE};
  HIP_CHECK(hipLaunchKernelExC(&configB, reinterpret_cast<const void*>(probe_cu_ids), argsB));
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<uint32_t> hostBitmapB(bitmapWords);
  HIP_CHECK(hipMemcpy(hostBitmapB.data(), d_bitmap, bitmapWords * sizeof(uint32_t),
                       hipMemcpyDeviceToHost));
  auto observedB = collectSetBits(hostBitmapB.data(), bitmapWords);

  // Verify: all observed CUs should be within maskB
  for (auto cu : observedB) {
    uint32_t word = cu / 32;
    uint32_t bit = cu % 32;
    CAPTURE(cu, word, bit);
    REQUIRE(word < maskB.size());
    CHECK((maskB[word] & (1u << bit)) != 0);
  }
  CHECK(!observedB.empty());

  // Verify: the two sets are disjoint
  for (auto cu : observedA) {
    for (auto cuB : observedB) {
      CHECK(cu != cuB);
    }
  }

  HIP_CHECK(hipFree(d_bitmap));
}

/**
 * Test: no per-dispatch CU mask (baseline). Verify kernel uses CUs from
 * multiple SEs — proves the default is not accidentally restricted.
 */
TEST_CASE("PerDispatchCUMask_NoMaskBaseline") {
  if (!isGfx1250()) {
    HipTest::HIP_SKIP_TEST("Per-dispatch CU mask requires gfx1250");
    return;
  }

  int numCUs = getNumCUs();
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));

  uint32_t numSEs = 2;
  uint32_t wgpsPerSE = numCUs / numSEs;
  uint32_t totalCUs = numSEs * wgpsPerSE;
  uint32_t bitmapWords = (totalCUs + 31) / 32;

  uint32_t* d_bitmap;
  HIP_CHECK(hipMalloc(&d_bitmap, bitmapWords * sizeof(uint32_t)));
  HIP_CHECK(hipMemset(d_bitmap, 0, bitmapWords * sizeof(uint32_t)));

  // Launch without CU mask attribute
  probe_cu_ids<<<dim3(numCUs * 16), dim3(64)>>>(d_bitmap, wgpsPerSE);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<uint32_t> hostBitmap(bitmapWords);
  HIP_CHECK(hipMemcpy(hostBitmap.data(), d_bitmap, bitmapWords * sizeof(uint32_t),
                       hipMemcpyDeviceToHost));
  auto observed = collectSetBits(hostBitmap.data(), bitmapWords);

  // Without masking, we should see CUs from multiple SEs
  bool seenSE0 = false, seenSE1 = false;
  for (auto cu : observed) {
    if (cu < wgpsPerSE) seenSE0 = true;
    else seenSE1 = true;
  }
  CHECK(seenSE0);
  CHECK(seenSE1);
  // Should see more CUs than any restricted test
  CHECK(observed.size() > 2);

  HIP_CHECK(hipFree(d_bitmap));
}

/**
 * Test: compose per-dispatch CU mask with stream-level CU mask.
 * Stream mask enables CUs [0, wgpsPerSE). Per-dispatch mask enables CUs [half, wgpsPerSE).
 * Observed CUs should be in the intersection: [half, wgpsPerSE).
 */
TEST_CASE("PerDispatchCUMask_ComposedWithStreamMask") {
  if (!isGfx1250()) {
    HipTest::HIP_SKIP_TEST("Per-dispatch CU mask requires gfx1250");
    return;
  }

  int numCUs = getNumCUs();
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));

  uint32_t numSEs = 2;
  uint32_t wgpsPerSE = numCUs / numSEs;
  uint32_t half = wgpsPerSE / 2;
  uint32_t totalCUs = numSEs * wgpsPerSE;
  uint32_t bitmapWords = (totalCUs + 31) / 32;

  REQUIRE(half > 0);
  REQUIRE(half <= 15);

  // Stream-level mask: all CUs enabled (full mask)
  auto streamMask = buildCUMask(0, wgpsPerSE, wgpsPerSE, numSEs);
  hipStream_t stream;
  HIP_CHECK(hipExtStreamCreateWithCUMask(&stream,
                                          static_cast<uint32_t>(streamMask.size()),
                                          streamMask.data()));

  // Per-dispatch mask: second half only
  uint32_t countB = wgpsPerSE - half;
  if (countB > 15) countB = 15;
  auto dispatchMask = buildCUMask(half, countB, wgpsPerSE, numSEs);

  hipExtCUMaskConfig cuCfg = {
    .cuMaskSize = static_cast<uint32_t>(dispatchMask.size()),
    .cuMask = dispatchMask.data()
  };

  uint32_t* d_bitmap;
  HIP_CHECK(hipMalloc(&d_bitmap, bitmapWords * sizeof(uint32_t)));
  HIP_CHECK(hipMemset(d_bitmap, 0, bitmapWords * sizeof(uint32_t)));

  hipLaunchAttribute attr = {};
  attr.id = hipLaunchAttributeExtCUMask;
  attr.val.cuMask = &cuCfg;

  hipLaunchConfig_t config = {};
  config.gridDim = dim3(numCUs * 16);
  config.blockDim = dim3(64);
  config.numAttrs = 1;
  config.attrs = &attr;
  config.stream = stream;

  void* args[] = {&d_bitmap, &wgpsPerSE};
  HIP_CHECK(hipLaunchKernelExC(&config, reinterpret_cast<const void*>(probe_cu_ids), args));
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<uint32_t> hostBitmap(bitmapWords);
  HIP_CHECK(hipMemcpy(hostBitmap.data(), d_bitmap, bitmapWords * sizeof(uint32_t),
                       hipMemcpyDeviceToHost));
  auto observed = collectSetBits(hostBitmap.data(), bitmapWords);

  // All observed CUs must be in the per-dispatch mask (the tighter constraint)
  for (auto cu : observed) {
    uint32_t word = cu / 32;
    uint32_t bit = cu % 32;
    CAPTURE(cu, word, bit);
    REQUIRE(word < dispatchMask.size());
    CHECK((dispatchMask[word] & (1u << bit)) != 0);
  }
  CHECK(!observed.empty());

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(d_bitmap));
}

/**
 * Test: non-contiguous mask returns error.
 */
TEST_CASE("PerDispatchCUMask_NonContiguousReturnsError") {
  if (!isGfx1250()) {
    HipTest::HIP_SKIP_TEST("Per-dispatch CU mask requires gfx1250");
    return;
  }

  // Non-contiguous: CUs 0,2 (gap at CU 1)
  uint32_t mask[] = {0x05};
  hipExtCUMaskConfig cuCfg = {
    .cuMaskSize = 1,
    .cuMask = mask
  };

  hipLaunchAttribute attr = {};
  attr.id = hipLaunchAttributeExtCUMask;
  attr.val.cuMask = &cuCfg;

  hipLaunchConfig_t config = {};
  config.gridDim = dim3(1);
  config.blockDim = dim3(64);
  config.numAttrs = 1;
  config.attrs = &attr;
  config.stream = nullptr;

  uint32_t wgpsPerSE = 8;
  uint32_t* d_out;
  HIP_CHECK(hipMalloc(&d_out, sizeof(uint32_t)));
  void* args[] = {&d_out, &wgpsPerSE};

  hipError_t err = hipLaunchKernelExC(&config,
                                       reinterpret_cast<const void*>(probe_cu_ids), args);
  CHECK(err == hipErrorInvalidValue);

  HIP_CHECK(hipFree(d_out));
}
