// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// HIP data hazard end-to-end tests.
//
// Each test launches a kernel through the rocjitsu emulator with the
// data_hazard plugin selected in the test config. After hipDeviceSynchronize()
// the test parses the hazard log to verify that hazards were (or were not)
// reported.
//
// ctest runs each TEST() in its own process (via --gtest_filter) so that each
// test gets a fresh output directory (RJ_SINK_DIR). The plugin streams each
// finding to RJ_SINK_DIR/data_hazard.log as it is made. It also writes a JSON
// report, but only once the run ends, which here is process exit -- after the
// test body has already run -- so these tests read the log instead.
//
// Naming convention:
//   _hazard suffix = expects a hazard, no suffix = expects none.
//   Paired tests share a base name (e.g. vgpr_waitcnt / vgpr_waitcnt_hazard).
//
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// ============================================================
// Hazard log parser
// ============================================================

struct HazardRecord {
  uint64_t pc = 0;
  int wave = -1;
  std::string message;    // diagnostic text
  std::string suggestion; // remediation text, empty when the plugin gave none
};

namespace {

/// Parse the hazard lines from RJ_SINK_DIR/data_hazard.log.
///
/// Example log file content:
///
///   [data_hazard] pc=0x7f07e042a04c wave=0: RAW hazard: VGPR v2 read before
///       async load completes (load at pc 0x7f07e042a040, instruction 610)
///       | Add s_wait_loadcnt 0 before reading this register
///   [data_hazard] 1 hazard detected
///
/// Each hazard is one line of the form
/// "[data_hazard] pc=<hex> wave=<n>: <message>[ | <suggestion>]". The
/// "N hazards detected" summary line carries no "pc=" and is skipped.
std::vector<HazardRecord> parse_hazard_log() {
  const char *dir = std::getenv("RJ_SINK_DIR");
  if (!dir)
    return {};

  std::ifstream f(std::string(dir) + "/data_hazard.log");
  std::vector<HazardRecord> records;
  std::string line;
  while (std::getline(f, line)) {
    const size_t pc_pos = line.find("pc=0x");
    if (line.rfind("[data_hazard] ", 0) != 0 || pc_pos == std::string::npos)
      continue;

    const size_t wave_pos = line.find(" wave=", pc_pos);
    const size_t colon_pos =
        wave_pos == std::string::npos ? std::string::npos : line.find(": ", wave_pos);
    if (colon_pos == std::string::npos)
      continue;

    HazardRecord r;
    r.pc = std::strtoull(line.c_str() + pc_pos + 5, nullptr, 16);
    r.wave = std::atoi(line.c_str() + wave_pos + 6);

    std::string rest = line.substr(colon_pos + 2);
    const size_t sep = rest.find(" | ");
    if (sep == std::string::npos) {
      r.message = rest;
    } else {
      r.message = rest.substr(0, sep);
      r.suggestion = rest.substr(sep + 3);
    }
    records.push_back(std::move(r));
  }
  return records;
}

} // namespace

// ============================================================
// Fixture
// ============================================================

class Gfx950DataHazardTest : public ::testing::Test {
protected:
  void TearDown() override {
    for (void *p : allocations_)
      (void)hipFree(p);
    allocations_.clear();
  }

  template <typename T> T *alloc(size_t count) {
    void *p = nullptr;
    EXPECT_EQ(hipMalloc(&p, count * sizeof(T)), hipSuccess);
    EXPECT_EQ(hipMemset(p, 0, count * sizeof(T)), hipSuccess);
    allocations_.push_back(p);
    return static_cast<T *>(p);
  }

  template <typename T> T *allocWithData(size_t count) {
    T *p = alloc<T>(count);
    std::vector<T> host(count);
    for (size_t i = 0; i < count; ++i)
      host[i] = static_cast<T>(i);
    EXPECT_EQ(hipMemcpy(p, host.data(), count * sizeof(T), hipMemcpyHostToDevice), hipSuccess);
    return p;
  }

  void sync() { ASSERT_EQ(hipDeviceSynchronize(), hipSuccess); }

  /// Hazards found so far. Valid once the dispatch has been synchronized,
  /// since the plugin streams each finding as it is made.
  std::vector<HazardRecord> records() { return parse_hazard_log(); }

  /// Guards against a vacuous pass: an absent log would otherwise be
  /// indistinguishable from a run in which the plugin found nothing.
  static void ExpectPluginRan() {
    const char *dir = std::getenv("RJ_SINK_DIR");
    ASSERT_NE(dir, nullptr) << "RJ_SINK_DIR is not set";
    const std::string path = std::string(dir) + "/data_hazard.log";
    ASSERT_TRUE(std::ifstream(path).good()) << "plugin wrote no log at " << path;
  }

  static std::string describe(const std::vector<HazardRecord> &records) {
    std::ostringstream os;
    for (const HazardRecord &r : records)
      os << "\n  wave " << r.wave << ": " << r.message;
    return os.str();
  }

private:
  std::vector<void *> allocations_;
};

// ============================================================
// Kernels and cases
// ============================================================

// --- VGPR waitcnt ---

__global__ void vgpr_no_hazard_kernel(const float *src, float *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float val;
  asm volatile("global_load_dword %0, %1, off\n"
               "s_waitcnt vmcnt(0)\n"
               : "=v"(val)
               : "v"(&src[tid]));
  dst[tid] = val;
}

__global__ void vgpr_hazard_kernel(const float *src, float *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float val;
  asm volatile("global_load_dword %0, %1, off\n"
               // BUG: missing s_waitcnt vmcnt(0).
               "s_nop 0\n"
               : "=v"(val)
               : "v"(&src[tid]));
  dst[tid] = val;
}

TEST_F(Gfx950DataHazardTest, vgpr_waitcnt) {
  auto *src = allocWithData<float>(64);
  auto *dst = alloc<float>(64);
  vgpr_no_hazard_kernel<<<1, 64>>>(src, dst);
  sync();
  ASSERT_NO_FATAL_FAILURE(ExpectPluginRan());
  const auto r = records();
  EXPECT_TRUE(r.empty()) << "unexpected hazards:" << describe(r);
}

TEST_F(Gfx950DataHazardTest, vgpr_waitcnt_hazard) {
  auto *src = allocWithData<float>(64);
  auto *dst = alloc<float>(64);
  vgpr_hazard_kernel<<<1, 64>>>(src, dst);
  sync();
  const auto r = records();
  ASSERT_EQ(r.size(), 1u) << "expected one hazard on the unwaited global_load_dword:"
                          << describe(r);
  EXPECT_EQ(r[0].wave, 0);
  EXPECT_NE(r[0].message.find("RAW hazard"), std::string::npos) << r[0].message;
  EXPECT_NE(r[0].message.find("VGPR v2"), std::string::npos) << r[0].message;
  EXPECT_FALSE(r[0].suggestion.empty());
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
