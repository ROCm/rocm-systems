/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <hip_test_common.hh>
#include <hip_test_context.hh>

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <vector>

// ============================================================================
// Section A: Target skip-list and environment variable check
// ============================================================================

namespace HipTest {

// Queue-level CU mask enforcement is not supported on all targets.
// gfx1250/gfx1251: CP firmware does not load COMPUTE_STATIC_THREAD_MGMT_SEn
// from MQD, so hipExtStreamCreateWithCUMask has no hardware effect.
// TODO(SWDEV-XXXXXX): track gfx1250/gfx1251 queue-level CU mask limitation
inline bool isCuMaskEnforcementSupported(int device = -1) {
#if HT_NVIDIA
  return false;
#elif HT_AMD
  if (device < 0) {
    HIP_CHECK(hipGetDevice(&device));
  }
  hipDeviceProp_t props{};
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  std::string arch(props.gcnArchName);
  auto pos = arch.find(':');
  if (pos != std::string::npos) arch.resize(pos);
  if (arch == "gfx1250" || arch == "gfx1251") return false;
  return true;
#else
  return false;
#endif
}

inline bool isCuMaskEnforcementDisabled() {
  const std::string val = TestContext::getEnvVar("HIP_SKIP_CU_MASK_ENFORCEMENT");
  return !val.empty() && val != "0";
}

namespace SkipReason {
inline constexpr char const kCuMaskEnforcementUnsupported[] =
    "queue-level CU mask enforcement is not supported on this target.";
inline constexpr char const kCuMaskEnforcementDisabledByEnv[] =
    "CU mask enforcement validation disabled via HIP_SKIP_CU_MASK_ENFORCEMENT.";
inline constexpr char const kCuMaskEnforcementTooFewCUs[] =
    "device has too few CUs to partition for this enforcement test.";
}  // namespace SkipReason

}  // namespace HipTest

// ============================================================================
// Section B: Skip macro
// ============================================================================

#define CHECK_CU_MASK_ENFORCEMENT_SUPPORT(minCUs)                              \
  do {                                                                         \
    if (HipTest::isCuMaskEnforcementDisabled()) {                              \
      HIP_SKIP_TEST(HipTest::SkipReason::kCuMaskEnforcementDisabledByEnv);    \
    }                                                                          \
    if (!HipTest::isCuMaskEnforcementSupported()) {                            \
      HIP_SKIP_TEST(HipTest::SkipReason::kCuMaskEnforcementUnsupported);      \
    }                                                                          \
    {                                                                          \
      int cumask_dev_ = 0;                                                     \
      HIP_CHECK(hipGetDevice(&cumask_dev_));                                   \
      hipDeviceProp_t cumask_props_{};                                         \
      HIP_CHECK(hipGetDeviceProperties(&cumask_props_, cumask_dev_));          \
      if (cumask_props_.multiProcessorCount < (minCUs)) {                      \
        HIP_SKIP_TEST(HipTest::SkipReason::kCuMaskEnforcementTooFewCUs);      \
      }                                                                        \
    }                                                                          \
  } while (0)

// ============================================================================
// Section C: Mask building utilities
// ============================================================================

namespace HipTest {
namespace CuMask {

constexpr uint32_t kBitsPerWord = 32;

inline std::vector<uint32_t> buildFullMask(int numCUs) {
  const size_t words = (static_cast<size_t>(numCUs) + kBitsPerWord - 1) / kBitsPerWord;
  std::vector<uint32_t> mask(words, 0);
  for (int cu = 0; cu < numCUs; ++cu) {
    mask[static_cast<size_t>(cu) / kBitsPerWord] |=
        uint32_t{1} << (static_cast<uint32_t>(cu) % kBitsPerWord);
  }
  return mask;
}

inline std::vector<uint32_t> buildRangeMask(int numCUs, int startCU, int count) {
  const size_t words = (static_cast<size_t>(numCUs) + kBitsPerWord - 1) / kBitsPerWord;
  std::vector<uint32_t> mask(words, 0);
  for (int i = startCU; i < startCU + count && i < numCUs; ++i) {
    mask[static_cast<size_t>(i) / kBitsPerWord] |=
        uint32_t{1} << (static_cast<uint32_t>(i) % kBitsPerWord);
  }
  return mask;
}

inline std::vector<uint32_t> buildZeroMask(int numCUs) {
  const size_t words = (static_cast<size_t>(numCUs) + kBitsPerWord - 1) / kBitsPerWord;
  return std::vector<uint32_t>(words, 0);
}

}  // namespace CuMask
}  // namespace HipTest

// ============================================================================
// Section D: SMID topology decoder
// ============================================================================

namespace HipTest {
namespace CuMask {

struct SmidTopology {
  int num_xcc;
  int num_se_per_xcc;
  int num_sa_per_se;
  int num_wgp_per_sa;

  uint32_t cuIndexToSmid(uint32_t cu_index) const {
    const int cus_per_sa = num_wgp_per_sa;
    const int cus_per_se = cus_per_sa * num_sa_per_se;
    const int cus_per_xcc = cus_per_se * num_se_per_xcc;

    uint32_t xcc = cu_index / cus_per_xcc;
    uint32_t rem = cu_index % cus_per_xcc;
    uint32_t se = rem / cus_per_se;
    rem = rem % cus_per_se;
    uint32_t sa = rem / cus_per_sa;
    uint32_t wgp = rem % cus_per_sa;

    // gfx1250 packing: xcc(4) << 9 | se(4) << 5 | sa(1) << 4 | wgp(4)
    // Other architectures will need their own encoding added here.
    return (xcc << 9) | (se << 5) | (sa << 4) | wgp;
  }

  std::set<uint32_t> maskToExpectedSmids(const std::vector<uint32_t>& mask,
                                          int numCUs) const {
    std::set<uint32_t> expected;
    for (int cu = 0; cu < numCUs; ++cu) {
      size_t word = static_cast<size_t>(cu) / kBitsPerWord;
      uint32_t bit = static_cast<uint32_t>(cu) % kBitsPerWord;
      if (word < mask.size() && (mask[word] & (uint32_t{1} << bit))) {
        expected.insert(cuIndexToSmid(static_cast<uint32_t>(cu)));
      }
    }
    return expected;
  }
};

// Returns topology for the current device, or nullopt if the architecture's
// SMID encoding is not yet characterized.
inline std::optional<SmidTopology> getSmidTopology(int device = -1) {
#if HT_AMD
  if (device < 0) {
    HIP_CHECK(hipGetDevice(&device));
  }
  hipDeviceProp_t props{};
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  std::string arch(props.gcnArchName);
  auto pos = arch.find(':');
  if (pos != std::string::npos) arch.resize(pos);

  // gfx1250: 8 XCC, 2 SE/XCC, 2 SA/SE, 8 WGP/SA (some harvested)
  // SMID packing: xcc(4)<<9 | se(4)<<5 | sa(1)<<4 | wgp(4)
  if (arch == "gfx1250") {
    int totalCUs = props.multiProcessorCount;
    int wgp_per_sa = totalCUs / (8 * 2 * 2);
    if (wgp_per_sa < 1) wgp_per_sa = 8;
    return SmidTopology{8, 2, 2, wgp_per_sa};
  }

  // TODO: Add encodings for gfx942, gfx950, gfx1100, gfx1101
  // after running smid_decode probes on those targets.

  return std::nullopt;
#else
  return std::nullopt;
#endif
}

}  // namespace CuMask
}  // namespace HipTest

// ============================================================================
// Section E: SMID collection helpers
// ============================================================================

namespace HipTest {
namespace CuMask {

constexpr uint32_t kSmidSentinel = 0xFFFFFFFF;

inline std::set<uint32_t> collectUniqueSmids(const uint32_t* host, uint32_t n) {
  std::set<uint32_t> smids;
  for (uint32_t i = 0; i < n; ++i) {
    if (host[i] != kSmidSentinel) {
      smids.insert(host[i]);
    }
  }
  return smids;
}

// Probe SMIDs using a caller-provided kernel launcher.
// LaunchFn signature: void(hipStream_t stream, uint32_t* d_out, uint32_t n)
// Returns the union of unique SMIDs observed across `passes` invocations.
template <typename LaunchFn>
std::set<uint32_t> probeSmidsMulti(hipStream_t stream, uint32_t* d_out,
                                    uint32_t* h_out, uint32_t n,
                                    int passes, LaunchFn launchFn) {
  std::set<uint32_t> allSmids;
  for (int p = 0; p < passes; ++p) {
    HIP_CHECK(hipMemsetAsync(d_out, 0xFF, n * sizeof(uint32_t), stream));
    launchFn(stream, d_out, n);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemcpy(h_out, d_out, n * sizeof(uint32_t), hipMemcpyDeviceToHost));
    auto smids = collectUniqueSmids(h_out, n);
    allSmids.insert(smids.begin(), smids.end());
  }
  return allSmids;
}

}  // namespace CuMask
}  // namespace HipTest

// ============================================================================
// Section F: CuMaskStreamGuard (RAII)
// ============================================================================

namespace HipTest {

class CuMaskStreamGuard {
 public:
  explicit CuMaskStreamGuard(const std::vector<uint32_t>& mask) {
    HIP_CHECK(hipExtStreamCreateWithCUMask(
        &stream_, static_cast<uint32_t>(mask.size()), mask.data()));
  }

  ~CuMaskStreamGuard() {
    if (stream_ != nullptr) {
      static_cast<void>(hipStreamDestroy(stream_));
    }
  }

  CuMaskStreamGuard(const CuMaskStreamGuard&) = delete;
  CuMaskStreamGuard& operator=(const CuMaskStreamGuard&) = delete;

  CuMaskStreamGuard(CuMaskStreamGuard&& o) noexcept : stream_{o.stream_} {
    o.stream_ = nullptr;
  }

  CuMaskStreamGuard& operator=(CuMaskStreamGuard&& o) noexcept {
    if (this != &o) {
      if (stream_ != nullptr) {
        static_cast<void>(hipStreamDestroy(stream_));
      }
      stream_ = o.stream_;
      o.stream_ = nullptr;
    }
    return *this;
  }

  hipStream_t stream() const { return stream_; }

 private:
  hipStream_t stream_ = nullptr;
};

}  // namespace HipTest
