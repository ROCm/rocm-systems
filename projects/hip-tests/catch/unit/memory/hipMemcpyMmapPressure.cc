/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Regression test for rocm-systems#12528: hipMemcpy from mmap'd pageable memory
// stalls under host memory pressure because hsa_amd_memory_lock triggers KFD
// queue evictions during reclaim.
//
// The test creates deterministic memory pressure (via cgroup memory.max if
// available, otherwise by allocating most of host RAM), then streams a
// file-backed mmap to the GPU and asserts on completion time and per-chunk
// latency.

#include <hip_test_common.hh>

#if defined(__linux__)

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace {

constexpr size_t kFileSizeMiB = 512;
constexpr size_t kFileSize = kFileSizeMiB * 1024 * 1024;
constexpr size_t kChunkSize = 32 * 1024 * 1024;  // 32 MiB, matches default PinXferSize

// Maximum allowed time for the full copy, as a multiple of the no-pressure baseline.
// Under pressure with the bug, copies take 10-100x longer; with the fix, <3x.
constexpr double kMaxSlowdown = 5.0;

// Maximum allowed single-chunk latency in milliseconds.
// Under pressure with the bug, single chunks can take 60+ seconds.
constexpr double kMaxChunkMs = 5000.0;

int64_t readMemAvailable() {
  FILE* f = fopen("/proc/meminfo", "r");
  if (!f) return -1;
  char line[256];
  int64_t avail_kb = -1;
  while (fgets(line, sizeof(line), f)) {
    if (strncmp(line, "MemAvailable:", 13) == 0) {
      avail_kb = atoll(line + 13);
      break;
    }
  }
  fclose(f);
  return avail_kb * 1024LL;
}

// Creates a temporary file with random-ish content and returns its fd.
// The file is unlinked immediately so it's cleaned up on close.
int createTempFile(size_t size) {
  char path[] = "/tmp/hip_pressure_test_XXXXXX";
  int fd = mkstemp(path);
  REQUIRE(fd >= 0);
  unlink(path);

  // Write in chunks to avoid needing a huge buffer
  std::vector<char> buf(1024 * 1024);
  for (size_t i = 0; i < buf.size(); i += sizeof(uint64_t)) {
    // Simple pseudo-random fill so pages aren't all zeros
    uint64_t val = i ^ (i << 13) ^ (i >> 7);
    memcpy(&buf[i], &val, sizeof(val));
  }
  size_t written = 0;
  while (written < size) {
    size_t chunk = std::min(buf.size(), size - written);
    ssize_t n = write(fd, buf.data(), chunk);
    REQUIRE(n > 0);
    written += n;
  }
  fsync(fd);
  return fd;
}

// Returns the effective cgroup memory limit, or 0 if unconstrained.
int64_t readCgroupMemLimit() {
  FILE* f = fopen("/proc/self/cgroup", "r");
  if (!f) return 0;
  char line[512];
  std::string cg_path;
  while (fgets(line, sizeof(line), f)) {
    if (strncmp(line, "0::", 3) == 0) {
      cg_path = line + 3;
      while (!cg_path.empty() && (cg_path.back() == '\n' || cg_path.back() == ' '))
        cg_path.pop_back();
      break;
    }
  }
  fclose(f);
  if (cg_path.empty()) return 0;

  // Walk ancestry for the tightest limit
  int64_t effective = 0;
  std::string path = cg_path;
  while (!path.empty()) {
    std::string max_path = "/sys/fs/cgroup" + path + "/memory.max";
    FILE* mf = fopen(max_path.c_str(), "r");
    if (mf) {
      char buf[64];
      if (fgets(buf, sizeof(buf), mf) && strncmp(buf, "max", 3) != 0) {
        int64_t val = atoll(buf);
        if (val > 0 && (effective == 0 || val < effective)) effective = val;
      }
      fclose(mf);
    }
    auto pos = path.rfind('/');
    if (pos == std::string::npos || pos == 0) break;
    path = path.substr(0, pos);
  }
  return effective;
}

struct MemoryHog {
  std::vector<void*> blocks_;
  static constexpr size_t kBlockSize = 128 * 1024 * 1024;

  // Allocate anonymous memory until MemAvailable drops below target_bytes.
  // Respects cgroup memory limits to avoid OOM kills in containers.
  void start(int64_t target_bytes) {
    int64_t cg_limit = readCgroupMemLimit();
    while (readMemAvailable() > target_bytes + static_cast<int64_t>(kBlockSize)) {
      // In a cgroup, don't allocate past 90% of the limit
      if (cg_limit > 0) {
        int64_t cg_current = 0;
        FILE* f = fopen("/sys/fs/cgroup/memory.current", "r");
        if (f) {
          char buf[64];
          if (fgets(buf, sizeof(buf), f)) cg_current = atoll(buf);
          fclose(f);
        }
        if (cg_current + static_cast<int64_t>(kBlockSize) > cg_limit * 9 / 10) break;
      }
      void* block =
          mmap(nullptr, kBlockSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (block == MAP_FAILED) break;
      memset(block, 0x42, kBlockSize);
      blocks_.push_back(block);
    }
  }

  void stop() {
    for (void* block : blocks_) {
      munmap(block, kBlockSize);
    }
    blocks_.clear();
  }

  ~MemoryHog() { stop(); }
};

struct CopyResult {
  double total_s;
  double worst_chunk_ms;
  size_t chunks;
  bool completed;
};

CopyResult runMmapCopy(int fd, size_t file_size) {
  void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  REQUIRE(mapped != MAP_FAILED);

  // Evict file pages before the test
  posix_fadvise(fd, 0, file_size, POSIX_FADV_DONTNEED);
  madvise(mapped, file_size, MADV_DONTNEED);

  void* d_buf = nullptr;
  HIP_CHECK(hipMalloc(&d_buf, kChunkSize));

  const char* src = static_cast<const char*>(mapped);
  size_t total_copied = 0;
  size_t n_chunks = 0;
  double worst_ms = 0;

  auto t0 = std::chrono::steady_clock::now();

  while (total_copied < file_size) {
    size_t this_chunk = std::min(kChunkSize, file_size - total_copied);

    auto tc0 = std::chrono::steady_clock::now();
    hipError_t err = hipMemcpy(d_buf, src + total_copied, this_chunk, hipMemcpyHostToDevice);
    auto tc1 = std::chrono::steady_clock::now();

    if (err != hipSuccess) {
      HIP_CHECK(hipFree(d_buf));
      munmap(mapped, file_size);
      FAIL("hipMemcpy failed: " << hipGetErrorString(err));
    }

    double ms = std::chrono::duration<double, std::milli>(tc1 - tc0).count();
    worst_ms = std::max(worst_ms, ms);
    n_chunks++;
    total_copied += this_chunk;
  }

  auto t1 = std::chrono::steady_clock::now();
  double total_s = std::chrono::duration<double>(t1 - t0).count();

  HIP_CHECK(hipFree(d_buf));
  munmap(mapped, file_size);

  return {total_s, worst_ms, n_chunks, true};
}

bool canCreatePressure() {
  int64_t avail = readMemAvailable();
  // Need at least 4 GiB to create meaningful pressure (leave ~2 GiB free)
  return avail > 4LL * 1024 * 1024 * 1024;
}

}  // anonymous namespace

// Baseline: copy mmap'd file to GPU without pressure. Establishes the reference time.
HIP_TEST_CASE(Unit_hipMemcpyMmapPressure_Baseline) {
  int fd = createTempFile(kFileSize);
  auto result = runMmapCopy(fd, kFileSize);
  close(fd);

  double mibps = (kFileSize / (1024.0 * 1024.0)) / result.total_s;
  INFO("Baseline: " << kFileSizeMiB << " MiB in " << result.total_s << " s (" << mibps
                    << " MiB/s), worst chunk: " << result.worst_chunk_ms << " ms");
  REQUIRE(result.completed);
  // Sanity: should complete in a reasonable time without pressure
  REQUIRE(result.total_s < 30.0);
}

// Regression test: copy mmap'd file to GPU under memory pressure.
// Without the fix (rocm-systems#12528), this stalls for minutes.
// With the fix, it completes within kMaxSlowdown of the baseline.
HIP_TEST_CASE(Unit_hipMemcpyMmapPressure_UnderPressure) {
  if (!canCreatePressure()) {
    HIP_SKIP_TEST("Not enough memory to create pressure (need >4 GiB available)");
  }

  // Step 1: baseline without pressure
  int fd = createTempFile(kFileSize);
  auto baseline = runMmapCopy(fd, kFileSize);
  INFO("Baseline: " << baseline.total_s << " s");

  // Step 2: create pressure — target 2 GiB available
  constexpr int64_t kTargetAvail = 2LL * 1024 * 1024 * 1024;
  MemoryHog hog;
  hog.start(kTargetAvail);

  int64_t actual_avail = readMemAvailable();
  INFO("MemAvailable after hog: " << (actual_avail / (1024 * 1024)) << " MiB");
  if (actual_avail > 4LL * 1024 * 1024 * 1024) {
    hog.stop();
    close(fd);
    HIP_SKIP_TEST("Could not reduce MemAvailable below 4 GiB");
  }

  // Verify the pressure detector's conditions are actually met before testing.
  // The detector uses max(2 GiB, 5% of total) as its threshold and PSI > 5%.
  // If these aren't met, the test would pass vacuously without exercising the fix.
  int64_t mem_total = 0;
  {
    FILE* f = fopen("/proc/meminfo", "r");
    if (f) {
      char line[256];
      while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "MemTotal:", 9) == 0) {
          mem_total = atoll(line + 9) * 1024LL;
          break;
        }
      }
      fclose(f);
    }
  }
  int64_t cg_limit = readCgroupMemLimit();
  int64_t effective_total = cg_limit > 0 ? cg_limit : mem_total;
  int64_t threshold = std::max(2LL * 1024 * 1024 * 1024, effective_total * 5 / 100);
  INFO("Pressure detector threshold: " << (threshold / (1024 * 1024)) << " MiB"
                                       << " (effective_total: " << (effective_total / (1024 * 1024))
                                       << " MiB)");
  if (actual_avail >= threshold) {
    hog.stop();
    close(fd);
    HIP_SKIP_TEST("MemAvailable ("
                  << (actual_avail / (1024 * 1024)) << " MiB) is above the pressure threshold ("
                  << (threshold / (1024 * 1024)) << " MiB); cannot exercise the fix");
  }

  // Step 3: copy under pressure
  auto pressure = runMmapCopy(fd, kFileSize);
  hog.stop();
  close(fd);

  double mibps = (kFileSize / (1024.0 * 1024.0)) / pressure.total_s;
  double slowdown = pressure.total_s / baseline.total_s;
  INFO("Under pressure: " << pressure.total_s << " s (" << mibps << " MiB/s)"
                          << ", slowdown: " << slowdown << "x"
                          << ", worst chunk: " << pressure.worst_chunk_ms << " ms");

  // Assert: should complete without catastrophic slowdown
  REQUIRE(pressure.completed);
  CHECK(slowdown < kMaxSlowdown);
  CHECK(pressure.worst_chunk_ms < kMaxChunkMs);
}

#endif  // __linux__
