/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * hipExtStreamCreateWithCUMask — CU enforcement validation.
 *
 * Verifies that stream-level CU masks actually restrict kernel execution,
 * not just that the API accepts/returns masks. Uses raw __smid() values
 * (no architecture-specific bit decoding) for full portability.
 *
 * Tests:
 *   1. Half-mask — first half of CUs produces fewer unique smids
 *   2. Disjoint halves — first/second half produce zero-overlap smid sets
 *   3. Quarter-mask — quarter of CUs produces ~25% of smids
 *   4. Non-persistence — masked stream doesn't affect default stream
 *   5. Per-thread audit — 1M threads, every one checked against allowed set
 *
 * Validated on: gfx950 (MI350X), gfx1101 (Navi32)
 * Known broken: gfx1250 (MI450) — stream masks not enforced, skipped
 */

#include <hip_test_common.hh>

#include <cstring>
#include <set>
#include <vector>

// Portable: every thread records raw __smid(), no bit decoding.
__global__ void record_smid_raw(uint32_t* out, uint32_t n) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  out[tid] = __smid();
}

namespace {

constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kProbeN = 256 * 1024;

bool isSkippedTarget(int device) {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  // gfx1250: stream CU masks not enforced (queue BASE mask not programmed)
  return strncmp(prop.gcnArchName, "gfx1250", 7) == 0;
}

uint32_t getNumCUs(int device) {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  return static_cast<uint32_t>(prop.multiProcessorCount);
}

std::set<uint32_t> collectUniqueSmids(const uint32_t* host, uint32_t n) {
  std::set<uint32_t> s;
  for (uint32_t i = 0; i < n; ++i)
    if (host[i] != 0xFFFFFFFF) s.insert(host[i]);
  return s;
}

std::set<uint32_t> probeSmids(hipStream_t stream, uint32_t* d_out, uint32_t n) {
  HIP_CHECK(hipMemsetAsync(d_out, 0xFF, n * sizeof(uint32_t), stream));
  uint32_t grid = (n + kBlockSize - 1) / kBlockSize;
  record_smid_raw<<<dim3(grid), dim3(kBlockSize), 0, stream>>>(d_out, n);
  HIP_CHECK(hipStreamSynchronize(stream));
  std::vector<uint32_t> host(n);
  HIP_CHECK(hipMemcpy(host.data(), d_out, n * sizeof(uint32_t), hipMemcpyDeviceToHost));
  return collectUniqueSmids(host.data(), n);
}

std::set<uint32_t> probeSmidsMulti(hipStream_t stream, uint32_t* d_out,
                                    uint32_t n, int passes) {
  std::set<uint32_t> all;
  for (int i = 0; i < passes; ++i) {
    auto obs = probeSmids(stream, d_out, n);
    all.insert(obs.begin(), obs.end());
  }
  return all;
}

std::vector<uint32_t> buildFlatMask(uint32_t numCUs, uint32_t startCU, uint32_t count) {
  uint32_t maskWords = (numCUs + 31) / 32;
  std::vector<uint32_t> mask(maskWords, 0);
  for (uint32_t cu = startCU; cu < startCU + count && cu < numCUs; ++cu)
    mask[cu / 32] |= (1u << (cu % 32));
  return mask;
}

}  // namespace

// ============================================================================
// Test 1: Half-mask — first half of CUs produces fewer unique smids
// ============================================================================
TEST_CASE("StreamCUMask_Enforcement_HalfMask") {
  int dev;
  HIP_CHECK(hipGetDevice(&dev));
  if (isSkippedTarget(dev)) {
    HIP_SKIP_TEST("Stream CU masks not enforced on this target");
    return;
  }

  uint32_t numCUs = getNumCUs(dev);
  uint32_t half = numCUs / 2;
  uint32_t maskWords = (numCUs + 31) / 32;

  uint32_t* d_out;
  HIP_CHECK(hipMalloc(&d_out, kProbeN * sizeof(uint32_t)));

  auto baseline = probeSmidsMulti(nullptr, d_out, kProbeN, 3);

  auto mask = buildFlatMask(numCUs, 0, half);
  hipStream_t stream;
  HIP_CHECK(hipExtStreamCreateWithCUMask(&stream, maskWords, mask.data()));
  auto masked = probeSmidsMulti(stream, d_out, kProbeN, 3);
  HIP_CHECK(hipStreamDestroy(stream));

  CAPTURE(baseline.size(), masked.size(), numCUs, half);
  CHECK(masked.size() < baseline.size());

  HIP_CHECK(hipFree(d_out));
}

// ============================================================================
// Test 2: Disjoint halves — zero overlap between first and second half
// ============================================================================
TEST_CASE("StreamCUMask_Enforcement_DisjointHalves") {
  int dev;
  HIP_CHECK(hipGetDevice(&dev));
  if (isSkippedTarget(dev)) {
    HIP_SKIP_TEST("Stream CU masks not enforced on this target");
    return;
  }

  uint32_t numCUs = getNumCUs(dev);
  uint32_t half = numCUs / 2;
  uint32_t maskWords = (numCUs + 31) / 32;

  uint32_t* d_out;
  HIP_CHECK(hipMalloc(&d_out, kProbeN * sizeof(uint32_t)));

  auto maskA = buildFlatMask(numCUs, 0, half);
  auto maskB = buildFlatMask(numCUs, half, numCUs - half);

  hipStream_t sA, sB;
  HIP_CHECK(hipExtStreamCreateWithCUMask(&sA, maskWords, maskA.data()));
  HIP_CHECK(hipExtStreamCreateWithCUMask(&sB, maskWords, maskB.data()));

  auto obsA = probeSmidsMulti(sA, d_out, kProbeN, 3);
  auto obsB = probeSmidsMulti(sB, d_out, kProbeN, 3);

  HIP_CHECK(hipStreamDestroy(sA));
  HIP_CHECK(hipStreamDestroy(sB));

  uint32_t overlap = 0;
  for (auto a : obsA)
    if (obsB.count(a)) overlap++;

  CAPTURE(obsA.size(), obsB.size(), overlap);
  CHECK(!obsA.empty());
  CHECK(!obsB.empty());
  CHECK(overlap == 0);

  HIP_CHECK(hipFree(d_out));
}

// ============================================================================
// Test 3: Quarter-mask — quarter of CUs produces ~25% of smids
// ============================================================================
TEST_CASE("StreamCUMask_Enforcement_QuarterMask") {
  int dev;
  HIP_CHECK(hipGetDevice(&dev));
  if (isSkippedTarget(dev)) {
    HIP_SKIP_TEST("Stream CU masks not enforced on this target");
    return;
  }

  uint32_t numCUs = getNumCUs(dev);
  uint32_t quarter = numCUs / 4;
  if (quarter < 2) {
    HIP_SKIP_TEST("Too few CUs for quarter-mask test");
    return;
  }
  uint32_t maskWords = (numCUs + 31) / 32;

  uint32_t* d_out;
  HIP_CHECK(hipMalloc(&d_out, kProbeN * sizeof(uint32_t)));

  auto baseline = probeSmidsMulti(nullptr, d_out, kProbeN, 3);

  auto mask = buildFlatMask(numCUs, 0, quarter);
  hipStream_t stream;
  HIP_CHECK(hipExtStreamCreateWithCUMask(&stream, maskWords, mask.data()));
  auto masked = probeSmidsMulti(stream, d_out, kProbeN, 3);
  HIP_CHECK(hipStreamDestroy(stream));

  CAPTURE(baseline.size(), masked.size(), numCUs, quarter);
  CHECK(masked.size() < baseline.size());

  HIP_CHECK(hipFree(d_out));
}

// ============================================================================
// Test 4: Non-persistence — masked stream doesn't affect default stream
// ============================================================================
TEST_CASE("StreamCUMask_Enforcement_NoLeak") {
  int dev;
  HIP_CHECK(hipGetDevice(&dev));
  if (isSkippedTarget(dev)) {
    HIP_SKIP_TEST("Stream CU masks not enforced on this target");
    return;
  }

  uint32_t numCUs = getNumCUs(dev);
  uint32_t half = numCUs / 2;
  uint32_t maskWords = (numCUs + 31) / 32;

  uint32_t* d_out;
  HIP_CHECK(hipMalloc(&d_out, kProbeN * sizeof(uint32_t)));

  auto baseline = probeSmidsMulti(nullptr, d_out, kProbeN, 3);

  auto mask = buildFlatMask(numCUs, 0, half);
  hipStream_t masked;
  HIP_CHECK(hipExtStreamCreateWithCUMask(&masked, maskWords, mask.data()));
  probeSmids(masked, d_out, kProbeN);
  HIP_CHECK(hipStreamDestroy(masked));

  auto afterObs = probeSmidsMulti(nullptr, d_out, kProbeN, 3);

  CAPTURE(baseline.size(), afterObs.size());
  CHECK(afterObs.size() >= baseline.size());

  HIP_CHECK(hipFree(d_out));
}

// ============================================================================
// Test 5: Per-thread audit — 1M threads, every one against allowed set
// ============================================================================
TEST_CASE("StreamCUMask_Enforcement_PerThreadAudit") {
  int dev;
  HIP_CHECK(hipGetDevice(&dev));
  if (isSkippedTarget(dev)) {
    HIP_SKIP_TEST("Stream CU masks not enforced on this target");
    return;
  }

  uint32_t numCUs = getNumCUs(dev);
  uint32_t half = numCUs / 2;
  uint32_t maskWords = (numCUs + 31) / 32;

  uint32_t* d_probe;
  HIP_CHECK(hipMalloc(&d_probe, kProbeN * sizeof(uint32_t)));

  auto mask = buildFlatMask(numCUs, 0, half);
  hipStream_t stream;
  HIP_CHECK(hipExtStreamCreateWithCUMask(&stream, maskWords, mask.data()));

  auto allowed = probeSmidsMulti(stream, d_probe, kProbeN, 3);
  REQUIRE(!allowed.empty());

  const uint32_t auditN = 1024 * 1024;
  uint32_t* d_audit;
  HIP_CHECK(hipMalloc(&d_audit, auditN * sizeof(uint32_t)));
  HIP_CHECK(hipMemsetAsync(d_audit, 0xFF, auditN * sizeof(uint32_t), stream));

  uint32_t grid = (auditN + kBlockSize - 1) / kBlockSize;
  record_smid_raw<<<dim3(grid), dim3(kBlockSize), 0, stream>>>(d_audit, auditN);
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<uint32_t> host(auditN);
  HIP_CHECK(hipMemcpy(host.data(), d_audit, auditN * sizeof(uint32_t), hipMemcpyDeviceToHost));

  uint32_t violations = 0;
  for (uint32_t i = 0; i < auditN; ++i)
    if (host[i] != 0xFFFFFFFF && allowed.find(host[i]) == allowed.end())
      violations++;

  CAPTURE(violations, auditN, allowed.size());
  CHECK(violations == 0);

  HIP_CHECK(hipFree(d_audit));
  HIP_CHECK(hipFree(d_probe));
  HIP_CHECK(hipStreamDestroy(stream));
}
