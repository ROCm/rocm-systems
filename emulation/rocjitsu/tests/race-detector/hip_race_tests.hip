// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// HIP race detection end-to-end tests.
//
// Each test launches one or more kernels through the rocjitsu emulator
// with race detection enabled (RJ_RACE=1). After hipDeviceSynchronize(),
// the test parses the structured race log to verify that races were
// (or were not) detected.
//
// ctest runs each TEST() in its own process (via --gtest_filter) so that
// each test gets a fresh output directory (RJ_SINK_DIR). The race plugin
// writes to RJ_SINK_DIR/race.log via the plugin sink system.
//
// Naming convention:
//   _race suffix = expects race, no suffix = expects no race.
//   Paired tests share a base name (e.g. vgpr_waitcnt / vgpr_waitcnt_race).
//
// TODO(newling): Add wave-32 (RDNA 3.5) cross-wave LDS test. Requires:
//   1. A gfx1150_rdna3_5_kmd.json config with wave_front_size=32
//   2. Compiling with --offload-arch=gfx1150 (or similar RDNA 3.5 target)
//   3. A separate ctest entry pointing at the RDNA 3.5 config
//   The test itself is the same lds_cross_wave pattern but with the
//   cross-wave boundary at lane 32 instead of 64.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

// ============================================================
// Race log parser
// ============================================================

struct RaceRecord {
  std::string type;     // "VGPR", "SGPR", "LDS"
  int reg = -1;         // register index (VGPR/SGPR) or byte address (LDS)
  int wave = -1;
  int lane = -1;
  std::string wg;       // "x,y,z"
  std::string conflict; // mnemonic of the instruction that initiated the
                        // unresolved memory operation (e.g. global_load_dword)
  std::string message;  // full diagnostic text between RACE header and END_RACE
};

/// Parse structured RACE blocks from the file at RJ_SINK_DIR/race.log.
///
/// Example log file content:
///
///   RACE type=VGPR reg=2 wave=0 lane=0 wg=0,0,0 conflict=global_load_dword
///   Race on VGPR v2 [workgroup (0, 0, 0), wave 0, lane 0]
///    ==>  0x5400021940  global_load_dword v2, v[2:3], off
///         0x5400021948  s_nop 0
///    ==>  0x540002194c  v_mov_b32_e32 v4, v2
///   END_RACE
///
/// The first line ("RACE ...") is a machine-parseable header with key=value
/// pairs. Lines between the header and "END_RACE" are the human-readable
/// diagnostic message. Each such block becomes one RaceRecord.
static std::vector<RaceRecord> parse_race_log() {
  const char *dir = std::getenv("RJ_SINK_DIR");
  if (!dir) return {};
  std::string path = std::string(dir) + "/race.log";

  std::ifstream f(path);
  std::vector<RaceRecord> records;
  std::string line;
  while (std::getline(f, line)) {
    if (line.substr(0, 5) != "RACE ") continue;

    RaceRecord r;

    // Parse key=value pairs from the RACE header line.
    std::string rest = line.substr(5);
    size_t pos = 0;
    while (pos < rest.size()) {
      size_t eq = rest.find('=', pos);
      if (eq == std::string::npos) break;
      size_t sp = rest.find(' ', eq);
      if (sp == std::string::npos) sp = rest.size();
      std::string key = rest.substr(pos, eq - pos);
      std::string val = rest.substr(eq + 1, sp - eq - 1);
      if (key == "type")          r.type = val;
      else if (key == "reg")      r.reg = std::stoi(val);
      else if (key == "wave")     r.wave = val.empty() ? -1 : std::stoi(val);
      else if (key == "lane")     r.lane = val.empty() ? -1 : std::stoi(val);
      else if (key == "wg")       r.wg = val;
      else if (key == "conflict") r.conflict = val;
      pos = sp + 1;
    }

    // Collect message body until END_RACE.
    while (std::getline(f, line)) {
      if (line == "END_RACE") break;
      if (!r.message.empty()) r.message += '\n';
      r.message += line;
    }

    records.push_back(std::move(r));
  }
  return records;
}

// ============================================================
// Trace section parser
// ============================================================

struct TraceSections {
  std::string header;
  std::vector<std::string> before;
  std::string mark0;
  std::vector<std::string> between;
  std::string mark1;
};

static std::string stripMarkPrefix(const std::string &line) {
  auto pos = line.find("==>");
  if (pos == std::string::npos) return line;
  std::string rest = line.substr(pos + 3);
  auto hex = rest.find("0x");
  if (hex != std::string::npos) {
    auto sp = rest.find(' ', hex);
    if (sp != std::string::npos)
      sp = rest.find_first_not_of(' ', sp);
    if (sp != std::string::npos)
      return rest.substr(sp);
  }
  return rest;
}

static TraceSections parseTrace(const RaceRecord &r) {
  TraceSections t;
  std::istringstream ss(r.message);
  std::string line;
  int marks_seen = 0;
  while (std::getline(ss, line)) {
    if (t.header.empty()) { t.header = line; continue; }
    bool is_mark = line.find("==>") != std::string::npos;
    if (is_mark) {
      if (marks_seen == 0) t.mark0 = stripMarkPrefix(line);
      else                 t.mark1 = stripMarkPrefix(line);
      ++marks_seen;
    } else {
      if (marks_seen == 0)      t.before.push_back(line);
      else if (marks_seen == 1) t.between.push_back(line);
    }
  }
  return t;
}

static bool anyContains(const std::vector<std::string> &lines,
                        const std::string &s) {
  return std::any_of(lines.begin(), lines.end(),
      [&](const std::string &l) { return l.find(s) != std::string::npos; });
}

static constexpr auto npos = std::string::npos;

// ============================================================
// Test fixture
// ============================================================

class RaceTest : public ::testing::Test {
protected:
  // Returned pointers are device pointers. Not freed (process exits).

  template <typename T>
  T *alloc(int n) {
    T *p;
    (void)hipMalloc(&p, n * sizeof(T));
    return p;
  }

  template <typename T>
  T *allocWithData(int n) {
    T *p = alloc<T>(n);
    std::vector<T> h(n);
    for (int i = 0; i < n; ++i) h[i] = static_cast<T>(i);
    (void)hipMemcpy(p, h.data(), n * sizeof(T), hipMemcpyHostToDevice);
    return p;
  }

  void sync() { (void)hipDeviceSynchronize(); }

  std::vector<RaceRecord> records() { return parse_race_log(); }

  void ExpectNoRace() {
    auto r = records();
    if (!r.empty()) {
      for (auto &rec : r)
        fprintf(stderr, "  [%s reg=%d wg=%s] %s\n",
                rec.type.c_str(), rec.reg, rec.wg.c_str(),
                rec.message.c_str());
    }
    EXPECT_TRUE(r.empty()) << "Expected no races, got " << r.size();
  }

  void ExpectRace() {
    auto r = records();
    EXPECT_FALSE(r.empty()) << "Expected at least one race, got none";
  }

  void ExpectNRaces(int n) {
    auto r = records();
    EXPECT_EQ(static_cast<int>(r.size()), n)
        << "Expected " << n << " race(s), got " << r.size();
  }
};

// ============================================================
// Tests
// ============================================================

// --- VGPR waitcnt ---

__global__ void vgpr_no_race_kernel(const float *src, float *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float val;
  asm volatile(
      "global_load_dword %0, %1, off\n"
      "s_waitcnt vmcnt(0)\n"
      : "=v"(val) : "v"(&src[tid])
  );
  dst[tid] = val;
}

__global__ void vgpr_race_kernel(const float *src, float *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float val;
  asm volatile(
      "global_load_dword %0, %1, off\n"
      // BUG: missing s_waitcnt vmcnt(0).
      "s_nop 0\n"
      : "=v"(val) : "v"(&src[tid])
  );
  dst[tid] = val;
}

TEST_F(RaceTest, vgpr_waitcnt) {
  auto *src = allocWithData<float>(64), *dst = alloc<float>(64);
  vgpr_no_race_kernel<<<1, 64>>>(src, dst);
  sync();
  ExpectNoRace();
}

TEST_F(RaceTest, vgpr_waitcnt_race) {
  auto *src = allocWithData<float>(64), *dst = alloc<float>(64);
  vgpr_race_kernel<<<1, 64>>>(src, dst);
  sync();
  auto r = records();
  ASSERT_EQ(r.size(), 1u);
  auto t = parseTrace(r[0]);
  EXPECT_NE(t.header.find("VGPR"), npos);
  EXPECT_NE(t.header.find("wave 0"), npos);
  EXPECT_NE(t.header.find("lane 0"), npos);
  EXPECT_NE(t.mark0.find("global_load_dword"), npos) << "mark0: " << t.mark0;
  EXPECT_NE(t.mark1.find("global_store_dword"), npos) << "mark1: " << t.mark1;
}

// --- SGPR waitcnt ---

__global__ void sgpr_no_race_kernel(const int *src, int *dst) {
  int val;
  asm volatile(
      "s_load_dword %0, %1, 0x0\n"
      "s_waitcnt lgkmcnt(0)\n"
      "s_add_u32 %0, %0, 1\n"
      : "=s"(val) : "s"(src)
  );
  dst[threadIdx.x] = val;
}

__global__ void sgpr_race_kernel(const int *src, int *dst) {
  int val;
  asm volatile(
      "s_load_dword %0, %1, 0x0\n"
      // BUG: missing s_waitcnt lgkmcnt(0).
      "s_add_u32 %0, %0, 1\n"
      : "=s"(val) : "s"(src)
  );
  dst[threadIdx.x] = val;
}

TEST_F(RaceTest, sgpr_waitcnt) {
  auto *src = allocWithData<int>(64), *dst = alloc<int>(64);
  sgpr_no_race_kernel<<<1, 64>>>(src, dst);
  sync();
  ExpectNoRace();
}

TEST_F(RaceTest, sgpr_waitcnt_race) {
  auto *src = allocWithData<int>(64), *dst = alloc<int>(64);
  sgpr_race_kernel<<<1, 64>>>(src, dst);
  sync();
  auto r = records();
  ASSERT_EQ(r.size(), 1u);
  auto t = parseTrace(r[0]);
  EXPECT_NE(t.header.find("SGPR"), npos);
  EXPECT_NE(t.header.find("wave 0"), npos);
  EXPECT_NE(t.mark0.find("s_load_dword"), npos) << "mark0: " << t.mark0;
  EXPECT_NE(t.mark1.find("s_add_u32"), npos) << "mark1: " << t.mark1;
}

// --- LDS cross-wave ---

__attribute__((amdgpu_flat_work_group_size(128, 128)))
__global__ void lds_cross_wave_safe_kernel(int *out) {
  __shared__ int lds[128];
  int tid = threadIdx.x;
  lds[tid] = tid + 100;
  __syncthreads();
  out[tid] = lds[(tid + 64) % 128];
}

__attribute__((amdgpu_flat_work_group_size(128, 128)))
__global__ void lds_cross_wave_race_kernel(int *out) {
  __shared__ int lds[128];
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;
  lds[tid] = tid + 100;
  // BUG: no __syncthreads() before cross-wave read.
  if (blockIdx.x == 3)
    out[gid] = lds[(tid + 1) % 128];
  else
    out[gid] = lds[tid];
}

TEST_F(RaceTest, lds_cross_wave) {
  lds_cross_wave_safe_kernel<<<1, 128>>>(alloc<int>(128));
  sync();
  ExpectNoRace();
}

TEST_F(RaceTest, lds_cross_wave_race) {
  lds_cross_wave_race_kernel<<<5, 128>>>(alloc<int>(5 * 128));
  sync();
  auto r = records();
  ASSERT_GE(r.size(), 1u);
  auto t = parseTrace(r[0]);
  EXPECT_NE(t.header.find("LDS"), npos);
  EXPECT_NE(t.header.find("wave 0"), npos);
  EXPECT_NE(t.mark0.find("ds_write_b32"), npos) << "mark0: " << t.mark0;
  EXPECT_NE(t.mark1.find("ds_read_b32"), npos) << "mark1: " << t.mark1;
}

// --- Global-to-LDS buffer ---
// These kernels use buffer_load_dword with the LDS flag to load data from
// global memory directly into LDS (bypassing VGPRs). Designed for 128-thread
// workgroups (2 waves of 64). Each wave sets M0 to its LDS region offset and
// uses soffset for its global byte offset within the workgroup's buffer slice.

__attribute__((amdgpu_flat_work_group_size(128, 128)))
__global__ void buffer_load_lds_kernel(const int *src, int *out) {
  __shared__ int lds[128];
  int tid = threadIdx.x;
  int lane = tid % 64;
  int wave = tid / 64;
  lds[tid] = -1;
  __syncthreads();

  const int *wg_src = src + blockIdx.x * blockDim.x;
  uintptr_t base = reinterpret_cast<uintptr_t>(wg_src);
  unsigned w0 = static_cast<unsigned>(base);
  unsigned w1 = static_cast<unsigned>(base >> 32) & 0xFFFFu;
  unsigned w2 = 128 * sizeof(int);
  unsigned w3 = (4u) | (7u << 12) | (4u << 15);

  unsigned wave_byte_offset =
      __builtin_amdgcn_readfirstlane(wave * 64 * sizeof(int));
  asm volatile(
      "s_mov_b32 s12, %0\n"
      "s_mov_b32 s13, %1\n"
      "s_mov_b32 s14, %2\n"
      "s_mov_b32 s15, %3\n"
      "s_mov_b32 m0, %5\n"
      "buffer_load_dword %4, s[12:15], %6 offen lds\n"
      "s_waitcnt vmcnt(0)\n"
      :
      : "s"(w0), "s"(w1), "s"(w2), "s"(w3),
        "v"(lane * 4), "s"(wave_byte_offset), "s"(wave_byte_offset)
      : "s12", "s13", "s14", "s15", "memory"
  );

  __syncthreads();
  int gid = blockIdx.x * blockDim.x + tid;
  out[gid] = lds[tid];
}

__attribute__((amdgpu_flat_work_group_size(128, 128)))
__global__ void buffer_load_lds_race_kernel(const int *src, int *out) {
  __shared__ int lds[128];
  int tid = threadIdx.x;
  int lane = tid % 64;
  int wave = tid / 64;

  const int *wg_src = src + blockIdx.x * blockDim.x;
  uintptr_t base = reinterpret_cast<uintptr_t>(wg_src);
  unsigned w0 = static_cast<unsigned>(base);
  unsigned w1 = static_cast<unsigned>(base >> 32) & 0xFFFFu;
  unsigned w2 = 128 * sizeof(int);
  unsigned w3 = (4u) | (7u << 12) | (4u << 15);

  unsigned wave_byte_offset =
      __builtin_amdgcn_readfirstlane(wave * 64 * sizeof(int));
  asm volatile(
      "s_mov_b32 s12, %0\n"
      "s_mov_b32 s13, %1\n"
      "s_mov_b32 s14, %2\n"
      "s_mov_b32 s15, %3\n"
      "s_mov_b32 m0, %5\n"
      "buffer_load_dword %4, s[12:15], %6 offen lds\n"
      "s_waitcnt vmcnt(3)\n"
      "s_waitcnt vmcnt(2)\n"
      "s_waitcnt vmcnt(1)\n"
      // BUG: missing s_waitcnt vmcnt(0).
      :
      : "s"(w0), "s"(w1), "s"(w2), "s"(w3),
        "v"(lane * 4), "s"(wave_byte_offset), "s"(wave_byte_offset)
      : "s12", "s13", "s14", "s15", "memory"
  );

  int val;
  asm volatile(
      "ds_read_b32 %0, %1\n"
      "s_waitcnt lgkmcnt(0)\n"
      : "=v"(val) : "v"(tid * 4)
  );
  int gid = blockIdx.x * blockDim.x + tid;
  out[gid] = val;
}

TEST_F(RaceTest, global_to_lds_buffer) {
  constexpr int N = 3 * 128;
  auto *src = allocWithData<int>(N), *out = alloc<int>(N);
  buffer_load_lds_kernel<<<3, 128>>>(src, out);
  sync();

  std::vector<int> h_src(N), h_out(N);
  for (int i = 0; i < N; ++i) h_src[i] = i;
  (void)hipMemcpy(h_out.data(), out, N * sizeof(int), hipMemcpyDeviceToHost);
  int errors = 0;
  for (int i = 0; i < N; ++i)
    if (h_out[i] != h_src[i] && errors++ < 5)
      fprintf(stderr, "MISMATCH [%d]: got %d, expected %d\n",
              i, h_out[i], h_src[i]);
  EXPECT_EQ(errors, 0) << errors << "/" << N << " mismatches";
}

TEST_F(RaceTest, global_to_lds_buffer_race) {
  buffer_load_lds_race_kernel<<<3, 128>>>(alloc<int>(384), alloc<int>(384));
  sync();
  auto r = records();
  ASSERT_GE(r.size(), 1u);
  auto t = parseTrace(r[0]);
  EXPECT_NE(t.header.find("LDS"), npos);
  EXPECT_NE(t.header.find("wave 0"), npos);
  EXPECT_NE(t.header.find("lane 0"), npos);
  EXPECT_NE(t.mark0.find("buffer_load_dword"), npos) << "mark0: " << t.mark0;
  EXPECT_TRUE(anyContains(t.between, "s_waitcnt"));
  EXPECT_NE(t.mark1.find("ds_read_b32"), npos) << "mark1: " << t.mark1;
}

// --- Partial vmcnt ---

__global__ void partial_vmcnt_kernel(const float *src_a, const float *src_b,
                                     float *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float a, b;
  asm volatile(
      "global_load_dword %0, %2, off\n"   // oldest (vmcnt event 0)
      "global_load_dword %1, %3, off\n"   // newest (vmcnt event 1)
      "s_waitcnt vmcnt(1)\n"              // drain oldest only
      : "=v"(a), "=v"(b)
      : "v"(&src_a[tid]), "v"(&src_b[tid])
  );
  // Reading b races (still pending). Reading a is safe.
  dst[tid] = a + b;
}

TEST_F(RaceTest, partial_vmcnt_race) {
  partial_vmcnt_kernel<<<1, 64>>>(alloc<float>(64), alloc<float>(64),
                                   alloc<float>(64));
  sync();
  auto r = records();
  ASSERT_EQ(r.size(), 1u);
  auto t = parseTrace(r[0]);
  EXPECT_NE(t.header.find("VGPR"), npos);
  EXPECT_NE(t.header.find("wave 0"), npos);
  EXPECT_NE(t.header.find("lane 0"), npos);
  EXPECT_NE(t.mark0.find("global_load_dword"), npos) << "mark0: " << t.mark0;
  EXPECT_TRUE(anyContains(t.between, "s_waitcnt"));
  EXPECT_NE(t.mark1.find("v_add_f32"), npos) << "mark1: " << t.mark1;
}

// --- Multiple race ---

__global__ void two_races_kernel(const float *src_a, const float *src_b,
                                 float *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float a, b;
  asm volatile(
      "global_load_dword %0, %2, off\n"
      "global_load_dword %1, %3, off\n"
      // No waitcnt at all.
      : "=v"(a), "=v"(b)
      : "v"(&src_a[tid]), "v"(&src_b[tid])
  );
  dst[tid] = a + b;
}

TEST_F(RaceTest, multiple_race) {
  two_races_kernel<<<1, 64>>>(alloc<float>(64), alloc<float>(64),
                               alloc<float>(64));
  sync();
  auto r = records();
  // Both loads race against v_add_f32, but dedup by PC collapses them to 1.
  ASSERT_GE(r.size(), 1u);
  auto t = parseTrace(r[0]);
  EXPECT_NE(t.header.find("VGPR"), npos);
  EXPECT_NE(t.mark0.find("global_load_dword"), npos) << "mark0: " << t.mark0;
  EXPECT_NE(t.mark1.find("v_add_f32"), npos) << "mark1: " << t.mark1;
}

// --- Multi-kernel ---

__global__ void clean_kernel_a(const float *src, float *dst) {
  dst[threadIdx.x] = src[threadIdx.x];
}

__global__ void clean_kernel_b(const float *src, float *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  dst[tid] = src[tid] + 1.0f;
}

__global__ void racy_kernel(const float *src, float *dst) {
  int tid = threadIdx.x;
  float val;
  asm volatile(
      "global_load_dword %0, %1, off\n"
      // No waitcnt.
      : "=v"(val) : "v"(&src[tid])
  );
  dst[tid] = val;
}

TEST_F(RaceTest, multi_kernel) {
  auto *s = alloc<float>(1000), *d = alloc<float>(1000);
  clean_kernel_a<<<1, 64>>>(s, d);
  clean_kernel_b<<<3, 128>>>(s, d);
  clean_kernel_a<<<5, 64>>>(s, d);
  clean_kernel_b<<<2, 64>>>(s, d);
  sync();
  ExpectNoRace();
}

TEST_F(RaceTest, multi_kernel_race) {
  auto *s = alloc<float>(1000), *d = alloc<float>(1000);
  clean_kernel_a<<<1, 64>>>(s, d);
  clean_kernel_a<<<3, 128>>>(s, d);
  racy_kernel<<<1, 64>>>(s, d);
  clean_kernel_a<<<5, 64>>>(s, d);
  sync();
  auto r = records();
  ASSERT_EQ(r.size(), 1u);
  auto t = parseTrace(r[0]);
  EXPECT_NE(t.header.find("VGPR"), npos);
  EXPECT_NE(t.header.find("wave 0"), npos);
  EXPECT_NE(t.header.find("lane 0"), npos);
  EXPECT_NE(t.mark0.find("global_load_dword"), npos) << "mark0: " << t.mark0;
  EXPECT_NE(t.mark1.find("global_store_dword"), npos) << "mark1: " << t.mark1;
}

// --- Multi-workgroup ---

__global__ void multi_wg_clean_kernel(const float *src, float *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float val;
  asm volatile(
      "global_load_dword %0, %1, off\n"
      "s_waitcnt vmcnt(0)\n"
      : "=v"(val) : "v"(&src[tid])
  );
  dst[tid] = val;
}

__global__ void multi_wg_race_kernel(const float *src, float *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float val;
  asm volatile(
      "global_load_dword %0, %1, off\n"
      // No waitcnt.
      : "=v"(val) : "v"(&src[tid])
  );
  dst[tid] = val;
}

TEST_F(RaceTest, multi_workgroup) {
  auto *s = alloc<float>(256), *d = alloc<float>(256);
  multi_wg_clean_kernel<<<4, 64>>>(s, d);
  sync();
  ExpectNoRace();
}

TEST_F(RaceTest, multi_workgroup_race) {
  auto *s = alloc<float>(401 * 64), *d = alloc<float>(401 * 64);
  multi_wg_race_kernel<<<401, 64>>>(s, d);
  sync();
  auto r = records();
  ASSERT_GE(r.size(), 1u);
  auto t = parseTrace(r[0]);
  EXPECT_NE(t.header.find("VGPR"), npos);
  EXPECT_NE(t.header.find("wave 0"), npos);
  EXPECT_NE(t.header.find("lane 0"), npos);
  EXPECT_NE(t.mark0.find("global_load_dword"), npos) << "mark0: " << t.mark0;
  EXPECT_NE(t.mark1.find("global_store_dword"), npos) << "mark1: " << t.mark1;
}

// --- Mixed counters ---

__global__ void mixed_counters_kernel(const float *vsrc, const int *ssrc,
                                      float *dst) {
  int tid = threadIdx.x;
  float vval;
  int sval;
  asm volatile(
      "global_load_dword %0, %2, off\n"
      "s_load_dword %1, %3, 0x0\n"
      "s_waitcnt vmcnt(0)\n"   // drains global_load only
      // lgkmcnt not drained.
      : "=v"(vval), "=s"(sval)
      : "v"(&vsrc[tid]), "s"(ssrc)
  );
  dst[tid] = vval + static_cast<float>(sval);
}

TEST_F(RaceTest, mixed_counters_race) {
  mixed_counters_kernel<<<1, 64>>>(alloc<float>(64), alloc<int>(64),
                                    alloc<float>(64));
  sync();
  auto r = records();
  ASSERT_GE(r.size(), 1u);
  auto t = parseTrace(r[0]);
  EXPECT_NE(t.header.find("SGPR"), npos);
  EXPECT_NE(t.header.find("wave 0"), npos);
  EXPECT_NE(t.mark0.find("s_load_dword"), npos) << "mark0: " << t.mark0;
  EXPECT_TRUE(anyContains(t.between, "s_waitcnt"));
  EXPECT_NE(t.mark1.find("v_cvt_f32_i32"), npos) << "mark1: " << t.mark1;
}

// --- Stress: wavefront reuse ---

__global__ void stress_clean_a(const float *src, float *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  dst[tid] = src[tid];
}

__global__ void stress_clean_b(const float *src, float *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  dst[tid] = src[tid] + 1.0f;
}

__global__ void stress_racy(const float *src, float *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float val;
  asm volatile(
      "global_load_dword %0, %1, off\n"
      // No waitcnt.
      : "=v"(val) : "v"(&src[tid])
  );
  dst[tid] = val;
}

TEST_F(RaceTest, stress_wavefront_reuse) {
  auto *s = alloc<float>(64000), *d = alloc<float>(64000);
  stress_clean_a<<<1000, 64>>>(s, d);
  stress_clean_b<<<1000, 64>>>(s, d);
  sync();
  ExpectNoRace();
}

TEST_F(RaceTest, stress_wavefront_reuse_race) {
  auto *s = alloc<float>(64000), *d = alloc<float>(64000);
  stress_clean_a<<<1000, 64>>>(s, d);
  stress_racy<<<1000, 64>>>(s, d);
  sync();
  auto r = records();
  ASSERT_GE(r.size(), 1u);
  auto t = parseTrace(r[0]);
  EXPECT_NE(t.header.find("VGPR"), npos);
  EXPECT_NE(t.header.find("wave 0"), npos);
  EXPECT_NE(t.header.find("lane 0"), npos);
  EXPECT_NE(t.mark0.find("global_load_dword"), npos) << "mark0: " << t.mark0;
  EXPECT_NE(t.mark1.find("global_store_dword"), npos) << "mark1: " << t.mark1;
}

// --- EXEC mask ---
// 128 threads (2 waves). Only first 60 lanes per wave write to LDS.
// Wave 1 reads from wave 0's LDS region. With barrier = safe; without = race.

__attribute__((amdgpu_flat_work_group_size(128, 128)))
__global__ void exec_mask_safe_kernel(int *out) {
  __shared__ int lds[128];
  int tid = threadIdx.x;
  if (tid % 64 < 60)
    lds[tid] = tid + 100;
  __syncthreads();
  if (tid >= 64)
    out[tid] = lds[tid - 64];
}

__attribute__((amdgpu_flat_work_group_size(128, 128)))
__global__ void exec_mask_race_kernel(int *out) {
  __shared__ int lds[128];
  int tid = threadIdx.x;
  if (tid % 64 < 60)
    lds[tid] = tid + 100;
  // BUG: no barrier before cross-wave read.
  if (tid >= 64)
    out[tid] = lds[tid - 64];
}

TEST_F(RaceTest, exec_mask) {
  exec_mask_safe_kernel<<<1, 128>>>(alloc<int>(128));
  sync();
  ExpectNoRace();
}

TEST_F(RaceTest, exec_mask_race) {
  exec_mask_race_kernel<<<1, 128>>>(alloc<int>(128));
  sync();
  auto r = records();
  ASSERT_GE(r.size(), 1u);
  auto t = parseTrace(r[0]);
  EXPECT_NE(t.header.find("LDS"), npos);
  EXPECT_NE(t.header.find("wave 1"), npos);
  EXPECT_NE(t.header.find("lane 0"), npos);
  EXPECT_NE(t.mark0.find("ds_write_b32"), npos) << "mark0: " << t.mark0;
  EXPECT_NE(t.mark1.find("ds_read_b32"), npos) << "mark1: " << t.mark1;
}

// --- EXEC mask: same-wave vs cross-wave boundary ---
//
// 256 threads (4 waves of 64). All threads write lds[tid] = tid, barrier,
// then a subset of lanes overwrite lds[tid+32] = tid. Finally all threads
// read out[tid] = lds[tid].
//
// The conditional overwrite runs under a partial EXEC mask (the if-branch).
// The key variable is WHICH lanes are active:
//
//   lanes [0,32):  lds[tid+32] stays within the same wave (max target
//                  within wave is lds[63] for wave 0). No cross-wave write.
//
//   lanes [1,33):  lane 32 writes lds[tid+32] = lds[64], which is wave 1's
//                  region. One lane crosses the wave boundary.
//
// Synchronization between the overwrite and the read:
//
//   s_waitcnt lgkmcnt(0) — drains the wave's own pending LDS writes.
//       Sufficient for same-wave cross-lane dependencies (lane 0 writes
//       lds[32], lane 32 reads lds[32]). NOT sufficient for cross-wave
//       dependencies (wave 0 writes lds[64], wave 1 reads lds[64]).
//
//   s_barrier (__syncthreads) — synchronizes all waves in the workgroup.
//       Required for cross-wave dependencies.

__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void exec_mask_same_wave_kernel(int *out) {
  __shared__ int lds[256];
  int tid = threadIdx.x;
  lds[tid] = tid;
  __syncthreads();
  if (tid % 64 < 32)
    lds[(tid + 32) % 256] = tid;
  asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
  out[tid] = lds[tid];
}

__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void exec_mask_cross_wave_race_kernel(int *out) {
  __shared__ int lds[256];
  int tid = threadIdx.x;
  lds[tid] = tid;
  __syncthreads();
  int lane = tid % 64;
  if (lane >= 1 && lane < 33 && tid + 32 < 256)
    lds[tid + 32] = tid;
  // lgkmcnt drain resolves same-wave deps but NOT cross-wave — race.
  asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
  out[tid] = lds[tid];
}

__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void exec_mask_cross_wave_barrier_kernel(int *out) {
  __shared__ int lds[256];
  int tid = threadIdx.x;
  lds[tid] = tid;
  __syncthreads();
  int lane = tid % 64;
  if (lane >= 1 && lane < 33 && tid + 32 < 256)
    lds[tid + 32] = tid;
  // Full barrier resolves cross-wave deps — safe.
  __syncthreads();
  out[tid] = lds[tid];
}

// Host-side simulation of the LDS state for numerical validation.
static std::vector<int> simulate_exec_mask_lds(int n, int lo, int hi) {
  std::vector<int> lds(n);
  for (int i = 0; i < n; ++i) lds[i] = i;
  for (int i = 0; i < n; ++i) {
    int lane = i % 64;
    if (lane >= lo && lane < hi && i + 32 < n)
      lds[i + 32] = i;
  }
  return lds; // lds[tid] == expected out[tid]
}

TEST_F(RaceTest, exec_mask_same_wave) {
  auto *out = alloc<int>(256);
  exec_mask_same_wave_kernel<<<1, 256>>>(out);
  sync();
  ExpectNoRace();

  auto expected = simulate_exec_mask_lds(256, 0, 32);
  std::vector<int> h_out(256);
  (void)hipMemcpy(h_out.data(), out, 256 * sizeof(int), hipMemcpyDeviceToHost);
  int errors = 0;
  for (int i = 0; i < 256; ++i)
    if (h_out[i] != expected[i] && errors++ < 5)
      fprintf(stderr, "MISMATCH [%d]: got %d, expected %d\n",
              i, h_out[i], expected[i]);
  EXPECT_EQ(errors, 0) << errors << "/256 mismatches";
}

TEST_F(RaceTest, exec_mask_cross_wave_race) {
  exec_mask_cross_wave_race_kernel<<<1, 256>>>(alloc<int>(256));
  sync();
  auto r = records();
  ASSERT_GE(r.size(), 1u);
  auto t = parseTrace(r[0]);
  EXPECT_NE(t.header.find("LDS"), npos);
  EXPECT_NE(t.header.find("wave 1"), npos);
  EXPECT_NE(t.header.find("lane 0"), npos);
  EXPECT_NE(t.mark0.find("ds_write_b32"), npos) << "mark0: " << t.mark0;
  EXPECT_TRUE(anyContains(t.between, "s_waitcnt"));
  EXPECT_NE(t.mark1.find("ds_read_b32"), npos) << "mark1: " << t.mark1;
}

TEST_F(RaceTest, exec_mask_cross_wave) {
  auto *out = alloc<int>(256);
  exec_mask_cross_wave_barrier_kernel<<<1, 256>>>(out);
  sync();
  ExpectNoRace();

  auto expected = simulate_exec_mask_lds(256, 1, 33);
  std::vector<int> h_out(256);
  (void)hipMemcpy(h_out.data(), out, 256 * sizeof(int), hipMemcpyDeviceToHost);
  int errors = 0;
  for (int i = 0; i < 256; ++i) {
    if (h_out[i] != expected[i]) {
      if (errors++ < 10)
        fprintf(stderr, "MISMATCH [%d]: got %d, expected %d\n",
                i, h_out[i], expected[i]);
    }
  }
  EXPECT_EQ(errors, 0) << errors << "/256 mismatches";
}

// --- LDS transpose ---
//
// 128 threads (2 waves of 64) swap data through LDS. Each thread writes its
// tid to lds[tid], then reads from the other wave's half: lds[(tid+64)%128].
// Wave 0 (tids 0-63) reads addresses 64-127, wave 1 (tids 64-127) reads
// addresses 0-63 — every read crosses the wave boundary.
//
// s_waitcnt lgkmcnt(0) drains this wave's own LDS writes but does NOT
// synchronize with the other wave. Only s_barrier guarantees that both
// waves have completed their writes before either wave reads.

__attribute__((amdgpu_flat_work_group_size(128, 128)))
__global__ void lds_transpose_safe_kernel(int *out) {
  __shared__ int lds[128];
  int tid = threadIdx.x;
  int base = static_cast<int>(reinterpret_cast<uintptr_t>(lds) & 0xFFFF);
  int write_addr = base + tid * 4;
  int read_addr = base + ((tid + 64) % 128) * 4;
  int val;
  asm volatile(
      "ds_write_b32 %1, %2\n"
      "s_waitcnt lgkmcnt(0)\n"
      "s_barrier\n"
      "ds_read_b32 %0, %3\n"
      "s_waitcnt lgkmcnt(0)\n"
      : "=v"(val)
      : "v"(write_addr), "v"(tid), "v"(read_addr)
  );
  out[tid] = val;
}

__attribute__((amdgpu_flat_work_group_size(128, 128)))
__global__ void lds_transpose_race_kernel(int *out) {
  __shared__ int lds[128];
  int tid = threadIdx.x;
  int base = static_cast<int>(reinterpret_cast<uintptr_t>(lds) & 0xFFFF);
  int write_addr = base + tid * 4;
  int read_addr = base + ((tid + 64) % 128) * 4;
  int val;
  asm volatile(
      "ds_write_b32 %1, %2\n"
      "s_waitcnt lgkmcnt(0)\n"
      // BUG: missing s_barrier — wave 0 reads wave 1's region before it's written.
      "ds_read_b32 %0, %3\n"
      "s_waitcnt lgkmcnt(0)\n"
      : "=v"(val)
      : "v"(write_addr), "v"(tid), "v"(read_addr)
  );
  out[tid] = val;
}

TEST_F(RaceTest, lds_transpose) {
  auto *out = alloc<int>(128);
  lds_transpose_safe_kernel<<<1, 128>>>(out);
  sync();
  ExpectNoRace();

  std::vector<int> h_out(128);
  (void)hipMemcpy(h_out.data(), out, 128 * sizeof(int), hipMemcpyDeviceToHost);
  int errors = 0;
  for (int i = 0; i < 128; ++i)
    if (h_out[i] != (i + 64) % 128 && errors++ < 5)
      fprintf(stderr, "MISMATCH [%d]: got %d, expected %d\n",
              i, h_out[i], (i + 64) % 128);
  EXPECT_EQ(errors, 0) << errors << "/128 mismatches";
}

TEST_F(RaceTest, lds_transpose_race) {
  lds_transpose_race_kernel<<<1, 128>>>(alloc<int>(128));
  sync();
  auto r = records();
  ASSERT_GE(r.size(), 1u);
  auto t = parseTrace(r[0]);
  EXPECT_NE(t.header.find("LDS"), npos);
  EXPECT_NE(t.mark0.find("ds_write_b32"), npos) << "mark0: " << t.mark0;
  EXPECT_NE(t.mark1.find("ds_read_b32"), npos) << "mark1: " << t.mark1;
}

// --- Dual-offset LDS ---
// ds_write2_b32 writes two dwords per lane to LDS at two offsets.
// Cross-wave read without barrier = race; with barrier = safe.

__attribute__((amdgpu_flat_work_group_size(128, 128)))
__global__ void dual_offset_lds_safe_kernel(int *out) {
  __shared__ int lds[256];
  int tid = threadIdx.x;
  asm volatile(
      "ds_write2_b32 %0, %1, %2 offset0:0 offset1:1\n"
      "s_waitcnt lgkmcnt(0)\n"
      : : "v"(tid * 8), "v"(tid), "v"(tid + 100) : "memory"
  );
  __syncthreads();
  if (tid >= 64) {
    int val;
    asm volatile(
        "ds_read_b32 %0, %1\n"
        "s_waitcnt lgkmcnt(0)\n"
        : "=v"(val) : "v"((tid - 64) * 8)
    );
    out[tid] = val;
  }
}

__attribute__((amdgpu_flat_work_group_size(128, 128)))
__global__ void dual_offset_lds_race_kernel(int *out) {
  __shared__ int lds[256];
  int tid = threadIdx.x;
  asm volatile(
      "ds_write2_b32 %0, %1, %2 offset0:0 offset1:1\n"
      "s_waitcnt lgkmcnt(0)\n"
      : : "v"(tid * 8), "v"(tid), "v"(tid + 100) : "memory"
  );
  // BUG: no barrier before cross-wave read.
  if (tid >= 64) {
    int val;
    asm volatile(
        "ds_read_b32 %0, %1\n"
        "s_waitcnt lgkmcnt(0)\n"
        : "=v"(val) : "v"((tid - 64) * 8)
    );
    out[tid] = val;
  }
}

TEST_F(RaceTest, dual_offset_lds) {
  dual_offset_lds_safe_kernel<<<1, 128>>>(alloc<int>(128));
  sync();
  ExpectNoRace();
}

TEST_F(RaceTest, dual_offset_lds_race) {
  dual_offset_lds_race_kernel<<<1, 128>>>(alloc<int>(128));
  sync();
  auto r = records();
  ASSERT_GE(r.size(), 1u);
  auto t = parseTrace(r[0]);
  EXPECT_NE(t.header.find("LDS"), npos);
  EXPECT_NE(t.header.find("wave 1"), npos);
  EXPECT_NE(t.header.find("lane 0"), npos);
  EXPECT_NE(t.mark0.find("ds_write2_b32"), npos) << "mark0: " << t.mark0;
  EXPECT_TRUE(anyContains(t.between, "s_waitcnt"));
  EXPECT_NE(t.mark1.find("ds_read_b32"), npos) << "mark1: " << t.mark1;
}

// --- Scratch store vmcnt accounting ---
// scratch_store consumes a vmcnt slot. A global_load followed by two
// scratch_stores has 3 outstanding vmcnt events. vmcnt(2) drains the
// oldest (the global_load), so reading the loaded register is safe.

__global__ void scratch_vmcnt_safe_kernel(const float *src, float *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float val;
  float tmp1 = 1.0f, tmp2 = 2.0f;
  asm volatile(
      "global_load_dword %0, %3, off\n"       // vmcnt event 0 (oldest)
      "scratch_store_byte %4, %1, off\n"       // vmcnt event 1
      "scratch_store_byte %4, %2, off offset:1\n" // vmcnt event 2 (newest)
      "s_waitcnt vmcnt(2)\n"                   // drain oldest → load complete
      : "=v"(val)
      : "v"(tmp1), "v"(tmp2), "v"(&src[tid]), "v"(0)
  );
  dst[tid] = val;
}

__global__ void scratch_vmcnt_race_kernel(const float *src, float *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float val;
  float tmp1 = 1.0f;
  asm volatile(
      "global_load_dword %0, %2, off\n"    // vmcnt event 0 (oldest)
      "scratch_store_byte %3, %1, off\n"   // vmcnt event 1 (newest)
      "s_waitcnt vmcnt(2)\n"               // BUG: 2 outstanding, need vmcnt(1)
      : "=v"(val)
      : "v"(tmp1), "v"(&src[tid]), "v"(0)
  );
  dst[tid] = val;
}

TEST_F(RaceTest, scratch_vmcnt) {
  auto *src = allocWithData<float>(64), *dst = alloc<float>(64);
  scratch_vmcnt_safe_kernel<<<1, 64>>>(src, dst);
  sync();
  ExpectNoRace();
}

TEST_F(RaceTest, scratch_vmcnt_race) {
  auto *src = allocWithData<float>(64), *dst = alloc<float>(64);
  scratch_vmcnt_race_kernel<<<1, 64>>>(src, dst);
  sync();
  auto r = records();
  ASSERT_GE(r.size(), 1u);
  auto t = parseTrace(r[0]);
  EXPECT_NE(t.header.find("VGPR"), npos);
  EXPECT_NE(t.mark0.find("global_load_dword"), npos) << "mark0: " << t.mark0;
}

// TODO(newling): D16 (partial-register) HIP tests.
//
// Concrete gap: the rocjitsu CU's read_vgpr always reads the full 32-bit
// register, so checkVgprRead is called with byteMask=0xF. Sub-dword store
// instructions (ds_write_b16, global_store_short) only consume 16 bits of
// the source VGPR, but trigger a full-register race check — false positive
// when the other half has a pending D16 load.
//
// Fix: add read_vgpr_masked(reg, lane, byteMask) to the CU, and have the
// codegen (_gen_ds_write in codegen.py, esz <= 2) emit the masked variant.
// See compute_unit.h for details. The original race-emulator solved this
// with Wave::getHalfVgpr(reg, lane, hi) which passed byteMask=0x3/0xC.
// See /home/jnewling/workspace/rocm-libraries/shared/race-emulator/tests/
// race_tests.cpp (D16_LdsLoad_NoFalsePositive, D16_HiLoadOutstanding_LoReadSafe)
// for the original tests exercising this.
//
// The write-side byte mask propagation IS done: registerEvent/registerLdsEvent
// pass 0x3/0xC from VectorMemState::d16_lo/hi. The core D16 tests using
// RaceTestBuilder (which bypasses read_vgpr) pass correctly.
//
// Deeper observation: even if an instruction reads "garbage" from pending
// bytes, it only matters if that garbage propagates to an observable side
// effect (global store, host-visible memory). Garbage churned through
// computation but never stored is arguably not a user-visible race. The
// current race detector flags at the point of register read, which is
// conservative. A more precise (but much harder) approach would track
// whether tainted values reach observable side effects.

// --- WAW: global load then ALU write to same register ---
//
// TODO(newling): the detector does not currently flag this. A global load targets
// a VGPR, then an ALU instruction overwrites the same VGPR before
// s_waitcnt retires the load. The final register value is ambiguous —
// if the load completes after the ALU write, it silently clobbers the
// ALU result. This is a WAW (write-after-write) race that should be
// detected but isn't, because the detector only checks reads against
// pending loads (checkVgprRead), not ALU writes.

__global__ void waw_global_load_then_alu_kernel(const float *src, float *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float val;
  asm volatile(
      "global_load_dword %0, %1, off\n"
      // BUG: no s_waitcnt before ALU write to the same register.
      "v_add_f32 %0, %0, %0\n"
      : "=v"(val) : "v"(&src[tid])
  );
  asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
  dst[tid] = val;
}

TEST_F(RaceTest, waw_global_load_then_alu) {
  auto *src = allocWithData<float>(64), *dst = alloc<float>(64);
  waw_global_load_then_alu_kernel<<<1, 64>>>(src, dst);
  sync();
  // TODO(newling): this should detect a VGPR WAW race, but currently
  // doesn't. When WAW detection is implemented, change to:
  //   auto r = records();
  //   ASSERT_GE(r.size(), 1u);
  //   EXPECT_NE(r[0].type.find("VGPR"), npos);
  ExpectNoRace();
}

// --- f64 VGPR race (64-bit global load + use without waitcnt) ---
// Loads a double via global_load_dwordx2 into a VGPR pair, then
// consumes it with v_fma_f64 without an intervening s_waitcnt.
// Exercises 64-bit VGPR race detection through the scalar read_lane64
// path (v_fma_f64 does not currently have a SIMD fast-path probe).

__global__ void f64_vgpr_race_kernel(const double *src, double *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  double val, result;
  asm volatile(
      "global_load_dwordx2 %0, %2, off\n"
      "s_nop 0\n"
      "v_fma_f64 %1, %0, 1.0, 1.0\n"
      : "=&v"(val), "=v"(result) : "v"(&src[tid])
  );
  dst[tid] = result;
}

__global__ void f64_vgpr_safe_kernel(const double *src, double *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  double val, result;
  asm volatile(
      "global_load_dwordx2 %0, %2, off\n"
      "s_waitcnt vmcnt(0)\n"
      "v_fma_f64 %1, %0, 1.0, 1.0\n"
      : "=&v"(val), "=v"(result) : "v"(&src[tid])
  );
  dst[tid] = result;
}

TEST_F(RaceTest, f64_vgpr_race) {
  double *src = allocWithData<double>(64);
  double *dst = alloc<double>(64);
  f64_vgpr_race_kernel<<<1, 64>>>(src, dst);
  sync();
  auto r = records();
  ASSERT_GE(r.size(), 1u);
  EXPECT_NE(r[0].type.find("VGPR"), npos);
}

TEST_F(RaceTest, f64_vgpr_safe) {
  double *src = allocWithData<double>(64);
  double *dst = alloc<double>(64);
  f64_vgpr_safe_kernel<<<1, 64>>>(src, dst);
  sync();
  ExpectNoRace();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
