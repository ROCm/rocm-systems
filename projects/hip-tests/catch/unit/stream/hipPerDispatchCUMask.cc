/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * Per-dispatch CU mask validation via hipLaunchAttributeExtCUMask.
 *
 * Uses __smid() to read the hardware SE/WGP identity from each thread
 * and verifies that dispatches with CU masks execute only on the
 * configured Shader Engines and CU ranges.
 *
 * Seven tests:
 *   1. Baseline — no mask, both SEs used
 *   2. SE isolation — seEn=0x1 restricts to SE0, seEn=0x2 to SE1
 *   3. Invalid parameters — out-of-range values return hipErrorInvalidValue
 *   4. Count reduction — smaller cuCount produces fewer observed WGPs
 *   5. Mask non-persistence — masked dispatch followed by unmasked uses full GPU
 *   6. Per-thread audit — every thread individually verified (1M threads)
 *   7. Atomic correctness — atomicAdd under mask produces exact count
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
  return strncmp(prop.gcnArchName, "gfx1250", 7) == 0;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Per-block probe: thread 0 records SE:WGP in a bitmap.
__global__ void probe_wgp_bitmap(uint32_t* bitmap, uint32_t maxWGPs) {
  if (threadIdx.x != 0) return;
  unsigned smid = __smid();
  unsigned wgp = smid & 0xF;
  unsigned se  = (smid >> 5) & 0xF;
  unsigned flat_id = se * 16 + wgp;
  if (flat_id < maxWGPs * 32)
    atomicOr(&bitmap[flat_id / 32], 1u << (flat_id % 32));
}

// Per-thread probe: EVERY thread records its SE:WGP.
__global__ void record_cu_per_thread(uint8_t* out, uint32_t n) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  unsigned smid = __smid();
  out[tid] = (uint8_t)(((smid >> 5) & 0xF) << 4 | (smid & 0xF));
}

// Atomic counter + CU recording.
__global__ void atomic_count_and_record(unsigned long long* counter,
                                         uint8_t* cuRecord, uint32_t n) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  atomicAdd(counter, 1ULL);
  unsigned smid = __smid();
  cuRecord[tid] = (uint8_t)(((smid >> 5) & 0xF) << 4 | (smid & 0xF));
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::vector<uint32_t> collectSetBits(const uint32_t* bitmap, uint32_t words) {
  std::vector<uint32_t> bits;
  for (uint32_t w = 0; w < words; ++w)
    for (uint32_t b = 0; b < 32; ++b)
      if (bitmap[w] & (1u << b))
        bits.push_back(w * 32 + b);
  return bits;
}

static hipError_t launchMasked(const void* kernel, void** args,
                               int grid, int block,
                               uint32_t cuStart, uint32_t cuCount, uint32_t seEn,
                               hipStream_t stream = nullptr) {
  hipExtCUMaskConfig cuCfg;
  cuCfg.cuStart = cuStart;
  cuCfg.cuCount = cuCount;
  cuCfg.seEn    = seEn;

  hipLaunchAttribute attr;
  memset(&attr, 0, sizeof(attr));
  attr.id = hipLaunchAttributeExtCUMask;
  attr.val.cuMask = cuCfg;

  hipLaunchConfig_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.gridDim  = dim3(grid);
  cfg.blockDim = dim3(block);
  cfg.numAttrs = 1;
  cfg.attrs    = &attr;
  cfg.stream   = stream;

  return hipLaunchKernelExC(&cfg, kernel, args);
}

static std::vector<uint32_t> probeWGPs(uint32_t cuStart, uint32_t cuCount, uint32_t seEn,
                                        uint32_t* d_bitmap, uint32_t bitmapWords, int grid) {
  HIP_CHECK(hipMemset(d_bitmap, 0, bitmapWords * sizeof(uint32_t)));
  uint32_t mw = bitmapWords;
  void* args[] = {&d_bitmap, &mw};
  HIP_CHECK(launchMasked(reinterpret_cast<const void*>(probe_wgp_bitmap),
                         args, grid, 64, cuStart, cuCount, seEn));
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<uint32_t> host(bitmapWords);
  HIP_CHECK(hipMemcpy(host.data(), d_bitmap, bitmapWords * sizeof(uint32_t),
                       hipMemcpyDeviceToHost));
  return collectSetBits(host.data(), bitmapWords);
}

// ============================================================================
// Test 1: Baseline — no mask, verify both SEs are used
// ============================================================================
TEST_CASE("PerDispatchCUMask_Baseline") {
  if (!isGfx1250()) {
    HipTest::HIP_SKIP_TEST("Per-dispatch CU mask requires gfx1250");
    return;
  }

  const uint32_t bitmapWords = 2;
  const int grid = 1024;

  uint32_t* d_bitmap;
  HIP_CHECK(hipMalloc(&d_bitmap, bitmapWords * sizeof(uint32_t)));
  HIP_CHECK(hipMemset(d_bitmap, 0, bitmapWords * sizeof(uint32_t)));

  uint32_t mw = bitmapWords;
  probe_wgp_bitmap<<<dim3(grid), dim3(64)>>>(d_bitmap, mw);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<uint32_t> host(bitmapWords);
  HIP_CHECK(hipMemcpy(host.data(), d_bitmap, bitmapWords * sizeof(uint32_t),
                       hipMemcpyDeviceToHost));
  auto observed = collectSetBits(host.data(), bitmapWords);

  bool seenSE0 = false, seenSE1 = false;
  for (auto flat : observed) {
    if (flat / 16 == 0) seenSE0 = true;
    if (flat / 16 == 1) seenSE1 = true;
  }
  CHECK(seenSE0);
  CHECK(seenSE1);
  CHECK(observed.size() > 2);

  HIP_CHECK(hipFree(d_bitmap));
}

// ============================================================================
// Test 2: SE isolation — seEn restricts to correct Shader Engine
// ============================================================================
TEST_CASE("PerDispatchCUMask_SEIsolation") {
  if (!isGfx1250()) {
    HipTest::HIP_SKIP_TEST("Per-dispatch CU mask requires gfx1250");
    return;
  }

  const uint32_t bitmapWords = 2;
  const int grid = 1024;

  uint32_t* d_bitmap;
  HIP_CHECK(hipMalloc(&d_bitmap, bitmapWords * sizeof(uint32_t)));

  // SE0 only
  auto se0 = probeWGPs(0, 0, 0x1, d_bitmap, bitmapWords, grid);
  REQUIRE(!se0.empty());
  for (auto flat : se0) {
    CAPTURE(flat);
    CHECK(flat / 16 == 0);
  }

  // SE1 only
  auto se1 = probeWGPs(0, 0, 0x2, d_bitmap, bitmapWords, grid);
  REQUIRE(!se1.empty());
  for (auto flat : se1) {
    CAPTURE(flat);
    CHECK(flat / 16 == 1);
  }

  // Both SEs (seEn=0x3 and seEn=0x0 should both use both SEs)
  auto both = probeWGPs(0, 0, 0x3, d_bitmap, bitmapWords, grid);
  bool s0 = false, s1 = false;
  for (auto flat : both) {
    if (flat / 16 == 0) s0 = true;
    if (flat / 16 == 1) s1 = true;
  }
  CHECK(s0);
  CHECK(s1);

  // Verify SE0 and SE1 sets are disjoint
  for (auto a : se0) {
    for (auto b : se1) {
      CHECK(a != b);
    }
  }

  HIP_CHECK(hipFree(d_bitmap));
}

// ============================================================================
// Test 3: Invalid parameters return hipErrorInvalidValue
// ============================================================================
TEST_CASE("PerDispatchCUMask_InvalidParams") {
  if (!isGfx1250()) {
    HipTest::HIP_SKIP_TEST("Per-dispatch CU mask requires gfx1250");
    return;
  }

  const uint32_t bitmapWords = 2;
  uint32_t* d_bitmap;
  HIP_CHECK(hipMalloc(&d_bitmap, bitmapWords * sizeof(uint32_t)));
  HIP_CHECK(hipMemset(d_bitmap, 0, bitmapWords * sizeof(uint32_t)));

  uint32_t mw = bitmapWords;
  void* args[] = {&d_bitmap, &mw};
  const void* kernel = reinterpret_cast<const void*>(probe_wgp_bitmap);

  // cuStart > 15
  CHECK(launchMasked(kernel, args, 1, 64, 16, 2, 0x1) == hipErrorInvalidValue);

  // cuCount > 15
  CHECK(launchMasked(kernel, args, 1, 64, 0, 16, 0x1) == hipErrorInvalidValue);

  // seEn > 3
  CHECK(launchMasked(kernel, args, 1, 64, 0, 2, 4) == hipErrorInvalidValue);

  // All invalid
  CHECK(launchMasked(kernel, args, 1, 64, 16, 16, 4) == hipErrorInvalidValue);

  HIP_CHECK(hipFree(d_bitmap));
}

// ============================================================================
// Test 4: Count reduction — fewer CUs → fewer observed WGPs
// ============================================================================
TEST_CASE("PerDispatchCUMask_CountReduction") {
  if (!isGfx1250()) {
    HipTest::HIP_SKIP_TEST("Per-dispatch CU mask requires gfx1250");
    return;
  }

  const uint32_t bitmapWords = 2;
  const int grid = 1024;

  uint32_t* d_bitmap;
  HIP_CHECK(hipMalloc(&d_bitmap, bitmapWords * sizeof(uint32_t)));

  // All CUs on SE0
  auto allCUs = probeWGPs(0, 0, 0x1, d_bitmap, bitmapWords, grid);

  // 2 CUs on SE0
  auto twoCUs = probeWGPs(0, 2, 0x1, d_bitmap, bitmapWords, grid);

  REQUIRE(!allCUs.empty());
  REQUIRE(!twoCUs.empty());

  // twoCUs must be a subset of allCUs
  for (auto flat : twoCUs) {
    bool found = false;
    for (auto a : allCUs) {
      if (a == flat) { found = true; break; }
    }
    CAPTURE(flat);
    CHECK(found);
  }

  // twoCUs must be strictly smaller
  CHECK(twoCUs.size() < allCUs.size());

  HIP_CHECK(hipFree(d_bitmap));
}

// ============================================================================
// Test 5: Mask non-persistence — mask doesn't leak to next dispatch
// ============================================================================
TEST_CASE("PerDispatchCUMask_NoLeak") {
  if (!isGfx1250()) {
    HipTest::HIP_SKIP_TEST("Per-dispatch CU mask requires gfx1250");
    return;
  }

  const uint32_t bitmapWords = 2;
  const int grid = 1024;

  uint32_t* d_bitmap;
  HIP_CHECK(hipMalloc(&d_bitmap, bitmapWords * sizeof(uint32_t)));

  // Dispatch 1: SE0 only
  auto masked = probeWGPs(0, 2, 0x1, d_bitmap, bitmapWords, grid);
  REQUIRE(!masked.empty());
  for (auto flat : masked) {
    CHECK(flat / 16 == 0);
  }

  // Dispatch 2: normal <<<>>> launch, no mask
  HIP_CHECK(hipMemset(d_bitmap, 0, bitmapWords * sizeof(uint32_t)));
  uint32_t mw = bitmapWords;
  probe_wgp_bitmap<<<dim3(grid), dim3(64)>>>(d_bitmap, mw);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<uint32_t> host(bitmapWords);
  HIP_CHECK(hipMemcpy(host.data(), d_bitmap, bitmapWords * sizeof(uint32_t),
                       hipMemcpyDeviceToHost));
  auto unmasked = collectSetBits(host.data(), bitmapWords);

  // Unmasked dispatch must use both SEs
  bool seenSE0 = false, seenSE1 = false;
  for (auto flat : unmasked) {
    if (flat / 16 == 0) seenSE0 = true;
    if (flat / 16 == 1) seenSE1 = true;
  }
  CHECK(seenSE0);
  CHECK(seenSE1);

  // Unmasked must use strictly more WGPs
  CHECK(unmasked.size() > masked.size());

  HIP_CHECK(hipFree(d_bitmap));
}

// ============================================================================
// Test 6: Per-thread audit — every thread individually checked (1M threads)
// ============================================================================
TEST_CASE("PerDispatchCUMask_PerThreadAudit") {
  if (!isGfx1250()) {
    HipTest::HIP_SKIP_TEST("Per-dispatch CU mask requires gfx1250");
    return;
  }

  const uint32_t N = 1024 * 1024;
  const int BLOCK = 256;
  const int GRID = (N + BLOCK - 1) / BLOCK;

  uint8_t* d_out;
  HIP_CHECK(hipMalloc(&d_out, N));

  // SE0 only, cuCount=4
  HIP_CHECK(hipMemset(d_out, 0xFF, N));
  uint32_t n = N;
  void* args[] = {&d_out, &n};
  HIP_CHECK(launchMasked(reinterpret_cast<const void*>(record_cu_per_thread),
                         args, GRID, BLOCK, 0, 4, 0x1));
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<uint8_t> host(N);
  HIP_CHECK(hipMemcpy(host.data(), d_out, N, hipMemcpyDeviceToHost));

  uint32_t violations = 0;
  for (uint32_t i = 0; i < N; ++i) {
    if (host[i] == 0xFF) continue;
    uint8_t se = (host[i] >> 4) & 0xF;
    if (se != 0) violations++;
  }

  CAPTURE(violations, N);
  CHECK(violations == 0);

  // SE1 only, cuCount=4
  HIP_CHECK(hipMemset(d_out, 0xFF, N));
  HIP_CHECK(launchMasked(reinterpret_cast<const void*>(record_cu_per_thread),
                         args, GRID, BLOCK, 0, 4, 0x2));
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(host.data(), d_out, N, hipMemcpyDeviceToHost));

  violations = 0;
  for (uint32_t i = 0; i < N; ++i) {
    if (host[i] == 0xFF) continue;
    uint8_t se = (host[i] >> 4) & 0xF;
    if (se != 1) violations++;
  }

  CAPTURE(violations, N);
  CHECK(violations == 0);

  HIP_CHECK(hipFree(d_out));
}

// ============================================================================
// Test 7: Atomic correctness — exact count under CU mask
// ============================================================================
TEST_CASE("PerDispatchCUMask_AtomicCorrectness") {
  if (!isGfx1250()) {
    HipTest::HIP_SKIP_TEST("Per-dispatch CU mask requires gfx1250");
    return;
  }

  const uint32_t N = 1024 * 1024;
  const int BLOCK = 256;
  const int GRID = (N + BLOCK - 1) / BLOCK;

  unsigned long long* d_counter;
  uint8_t* d_cuRecord;
  HIP_CHECK(hipMalloc(&d_counter, sizeof(unsigned long long)));
  HIP_CHECK(hipMalloc(&d_cuRecord, N));

  struct { uint32_t cs, cc, se; } masks[] = {
    {0, 2, 0x1},
    {0, 4, 0x2},
    {0, 0, 0x3},
  };

  for (auto& m : masks) {
    HIP_CHECK(hipMemset(d_counter, 0, sizeof(unsigned long long)));
    HIP_CHECK(hipMemset(d_cuRecord, 0xFF, N));

    uint32_t n = N;
    void* args[] = {&d_counter, &d_cuRecord, &n};
    HIP_CHECK(launchMasked(reinterpret_cast<const void*>(atomic_count_and_record),
                           args, GRID, BLOCK, m.cs, m.cc, m.se));
    HIP_CHECK(hipDeviceSynchronize());

    unsigned long long h_counter = 0;
    HIP_CHECK(hipMemcpy(&h_counter, d_counter, sizeof(unsigned long long),
                         hipMemcpyDeviceToHost));

    CAPTURE(m.cs, m.cc, m.se, h_counter, N);
    CHECK(h_counter == static_cast<unsigned long long>(N));

    // Verify SE constraint
    std::vector<uint8_t> h_cu(N);
    HIP_CHECK(hipMemcpy(h_cu.data(), d_cuRecord, N, hipMemcpyDeviceToHost));

    uint32_t effectiveSeEn = (m.se == 0) ? 0x3 : m.se;
    uint32_t seViolations = 0;
    for (uint32_t i = 0; i < N; ++i) {
      if (h_cu[i] == 0xFF) continue;
      uint8_t se = (h_cu[i] >> 4) & 0xF;
      if (!(effectiveSeEn & (1u << se))) seViolations++;
    }

    CAPTURE(seViolations);
    CHECK(seViolations == 0);
  }

  HIP_CHECK(hipFree(d_counter));
  HIP_CHECK(hipFree(d_cuRecord));
}
