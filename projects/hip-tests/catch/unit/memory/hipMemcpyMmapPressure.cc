/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Regression test for rocm-systems#12528: hipMemcpy from mmap'd pageable memory
// stalls under host memory pressure because hsa_amd_memory_lock triggers KFD
// queue evictions during reclaim.
//
// The test drives the host into reclaim with an anonymous-memory hog, checks that
// the runtime's pressure detector inputs (available memory and PSI, computed the
// same way the runtime computes them) are met, then streams a file-backed mmap to
// the GPU. It asserts the copy completes with correct data and without a stall.
// It skips, rather than passing vacuously, when it cannot create those conditions
// within a bounded time and memory budget.

#include <hip_test_common.hh>

#if defined(__linux__)

#include <fcntl.h>
#include <linux/magic.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/vfs.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int64_t kMiB = 1024LL * 1024;
constexpr int64_t kGiB = 1024 * kMiB;
constexpr size_t kFileSize = 512 * kMiB;
constexpr size_t kChunkSize = 32 * kMiB;  // matches the default pinned transfer size

// Budgets that keep the test bounded on large CI hosts.
constexpr int64_t kMaxHogBytes = 32 * kGiB;
constexpr auto kHogBudget = std::chrono::seconds(60);
constexpr auto kPsiWait = std::chrono::seconds(20);

// Stall guards. With the bug, single chunks took over 60 s and full copies timed out.
constexpr double kMaxChunkMs = 30000.0;
constexpr double kMaxTotalS = 120.0;

using Clock = std::chrono::steady_clock;

int64_t readSingleInt(const std::string& path) {
  FILE* f = fopen(path.c_str(), "r");
  if (!f) return -1;
  char buf[64];
  int64_t val = -1;
  if (fgets(buf, sizeof(buf), f)) val = atoll(buf);
  fclose(f);
  return val;
}

int64_t readKeyedValue(const std::string& path, const char* key, int64_t scale) {
  FILE* f = fopen(path.c_str(), "r");
  if (!f) return -1;
  char line[256];
  size_t key_len = strlen(key);
  int64_t val = -1;
  while (fgets(line, sizeof(line), f)) {
    if (strncmp(line, key, key_len) == 0) {
      val = atoll(line + key_len) * scale;
      break;
    }
  }
  fclose(f);
  return val;
}

// The cgroup v2 directory holding the tightest memory.max in this process's
// ancestry, mirroring the runtime's detector. Empty dir means unconstrained.
struct Cgroup {
  std::string dir;
  int64_t limit = 0;
};

Cgroup findLimitingCgroup() {
  Cgroup cg;
  FILE* f = fopen("/proc/self/cgroup", "r");
  if (!f) return cg;
  char line[512];
  std::string path;
  while (fgets(line, sizeof(line), f)) {
    if (strncmp(line, "0::", 3) == 0) {
      path = line + 3;
      while (!path.empty() && (path.back() == '\n' || path.back() == ' ')) path.pop_back();
      break;
    }
  }
  fclose(f);
  while (!path.empty()) {
    std::string dir = "/sys/fs/cgroup" + path;
    FILE* mf = fopen((dir + "/memory.max").c_str(), "r");
    if (mf) {
      char buf[64];
      if (fgets(buf, sizeof(buf), mf) && strncmp(buf, "max", 3) != 0) {
        int64_t val = atoll(buf);
        if (val > 0 && (cg.limit == 0 || val < cg.limit)) {
          cg.limit = val;
          cg.dir = dir;
        }
      }
      fclose(mf);
    }
    auto pos = path.rfind('/');
    if (pos == std::string::npos || pos == 0) break;
    path = path.substr(0, pos);
  }
  if (!cg.dir.empty() && readSingleInt(cg.dir + "/memory.current") < 0) cg = Cgroup{};
  return cg;
}

// Available memory as the runtime's detector computes it.
int64_t availableBytes(const Cgroup& cg) {
  if (cg.dir.empty()) return readKeyedValue("/proc/meminfo", "MemAvailable:", 1024);
  int64_t current = readSingleInt(cg.dir + "/memory.current");
  int64_t inactive_file = std::max<int64_t>(readKeyedValue(cg.dir + "/memory.stat",
                                                           "inactive_file ", 1), 0);
  return cg.limit - std::max<int64_t>(current - inactive_file, 0);
}

// PSI "some avg10" in percent, or -1 if PSI is unavailable.
double psiSomeAvg10(const Cgroup& cg) {
  std::string path = cg.dir.empty() ? "/proc/pressure/memory" : cg.dir + "/memory.pressure";
  FILE* f = fopen(path.c_str(), "r");
  if (!f) return -1;
  char line[256];
  double val = -1;
  if (fgets(line, sizeof(line), f)) {
    const char* p = strstr(line, "avg10=");
    if (p) val = atof(p + 6);
  }
  fclose(f);
  return val;
}

// The runtime's thresholds, honoring the same GPU_PIN_PRESSURE_* overrides.
struct Thresholds {
  int64_t avail_bytes;
  double psi_percent;  // 0 = available memory only
};

Thresholds detectorThresholds(const Cgroup& cg) {
  Thresholds t{};
  if (const char* mb = getenv("GPU_PIN_PRESSURE_THRESHOLD_MB")) {
    t.avail_bytes = atoll(mb) * kMiB;
  } else {
    int64_t total = cg.dir.empty() ? readKeyedValue("/proc/meminfo", "MemTotal:", 1024) : cg.limit;
    t.avail_bytes = std::max(std::min(2 * kGiB, total / 4), total * 5 / 100);
  }
  const char* psi = getenv("GPU_PIN_PRESSURE_PSI_THRESHOLD");
  t.psi_percent = psi ? atof(psi) : 5.0;
  return t;
}

bool detectorWouldTrigger(const Cgroup& cg, const Thresholds& t) {
  if (availableBytes(cg) >= t.avail_bytes) return false;
  return t.psi_percent <= 0 || psiSomeAvg10(cg) > t.psi_percent;
}

// Creates the test file on a disk-backed filesystem so the mmap is file-backed
// page cache, as in the issue. Returns -1 if no suitable directory exists.
int createDiskBackedFile(size_t size) {
  std::vector<std::string> dirs;
  if (const char* tmpdir = getenv("TMPDIR")) dirs.push_back(tmpdir);
  dirs.push_back("/var/tmp");
  dirs.push_back(".");
  for (const auto& dir : dirs) {
    struct statfs fs;
    struct statvfs vfs;
    if (statfs(dir.c_str(), &fs) != 0 || fs.f_type == TMPFS_MAGIC) continue;
    if (statvfs(dir.c_str(), &vfs) != 0 ||
        static_cast<uint64_t>(vfs.f_bavail) * vfs.f_frsize < 2 * size) {
      continue;
    }
    std::string path = dir + "/hip_mmap_pressure_XXXXXX";
    int fd = mkstemp(&path[0]);
    if (fd < 0) continue;
    unlink(path.c_str());
    std::vector<uint64_t> buf(kMiB / sizeof(uint64_t));
    for (size_t written = 0; written < size; written += kMiB) {
      for (size_t i = 0; i < buf.size(); ++i) {
        uint64_t x = written / sizeof(uint64_t) + i;
        buf[i] = x * 0x9E3779B97F4A7C15ULL ^ (x >> 7);
      }
      if (write(fd, buf.data(), kMiB) != static_cast<ssize_t>(kMiB)) {
        close(fd);
        return -1;
      }
    }
    fsync(fd);
    return fd;
  }
  return -1;
}

// Anonymous memory that drives the host (or cgroup) toward a target of available
// memory, within a time and size budget.
struct MemoryHog {
  static constexpr size_t kBlockSize = 128 * kMiB;
  std::vector<void*> blocks_;

  // Returns true if available memory reached target_avail within budget.
  bool fill(const Cgroup& cg, int64_t target_avail) {
    auto deadline = Clock::now() + kHogBudget;
    while (availableBytes(cg) > target_avail + static_cast<int64_t>(kBlockSize)) {
      if (Clock::now() > deadline) return false;
      if (static_cast<int64_t>(blocks_.size() * kBlockSize) >= kMaxHogBytes) return false;
      // Leave headroom under a cgroup limit so the cgroup OOM killer never fires.
      if (!cg.dir.empty() && readSingleInt(cg.dir + "/memory.current") +
                                     static_cast<int64_t>(kBlockSize) > cg.limit * 9 / 10) {
        break;
      }
      void* block =
          mmap(nullptr, kBlockSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (block == MAP_FAILED) break;
      memset(block, 0x42, kBlockSize);
      blocks_.push_back(block);
    }
    return availableBytes(cg) <= target_avail + static_cast<int64_t>(kBlockSize);
  }

  void release() {
    for (void* block : blocks_) munmap(block, kBlockSize);
    blocks_.clear();
  }

  ~MemoryHog() { release(); }
};

struct CopyResult {
  double total_s = 0;
  double worst_chunk_ms = 0;
  bool data_ok = false;
};

// Streams the file-backed mapping to the GPU in chunks, after evicting it from
// the page cache, and verifies the last chunk against the file contents.
CopyResult copyMappedFile(int fd, size_t size) {
  void* mapped = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  REQUIRE(mapped != MAP_FAILED);
  posix_fadvise(fd, 0, size, POSIX_FADV_DONTNEED);
  madvise(mapped, size, MADV_DONTNEED);

  void* d_buf = nullptr;
  HIP_CHECK(hipMalloc(&d_buf, kChunkSize));
  const char* src = static_cast<const char*>(mapped);

  CopyResult r;
  auto t0 = Clock::now();
  size_t offset = 0;
  size_t last_chunk = 0;
  while (offset < size) {
    last_chunk = std::min(kChunkSize, size - offset);
    auto c0 = Clock::now();
    HIP_CHECK(hipMemcpy(d_buf, src + offset, last_chunk, hipMemcpyHostToDevice));
    double ms = std::chrono::duration<double, std::milli>(Clock::now() - c0).count();
    r.worst_chunk_ms = std::max(r.worst_chunk_ms, ms);
    offset += last_chunk;
  }
  r.total_s = std::chrono::duration<double>(Clock::now() - t0).count();

  std::vector<char> check(last_chunk);
  HIP_CHECK(hipMemcpy(check.data(), d_buf, last_chunk, hipMemcpyDeviceToHost));
  r.data_ok = memcmp(check.data(), src + size - last_chunk, last_chunk) == 0;

  HIP_CHECK(hipFree(d_buf));
  munmap(mapped, size);
  return r;
}

}  // namespace

// Copies a file-backed mapping to the GPU without memory pressure.
HIP_TEST_CASE(Unit_hipMemcpyMmapPressure_Baseline) {
  int fd = createDiskBackedFile(kFileSize);
  if (fd < 0) HIP_SKIP_TEST("No disk-backed directory with enough free space");
  CopyResult r = copyMappedFile(fd, kFileSize);
  close(fd);

  INFO("Baseline: " << r.total_s << " s, worst chunk " << r.worst_chunk_ms << " ms");
  REQUIRE(r.data_ok);
  CHECK(r.worst_chunk_ms < kMaxChunkMs);
  CHECK(r.total_s < kMaxTotalS);
}

// Regression test: copies a file-backed mapping to the GPU while the host is in
// reclaim and the runtime's pressure detector conditions hold. Without the fix,
// pinning that memory triggers KFD queue evictions and single chunks stall for
// tens of seconds.
HIP_TEST_CASE(Unit_hipMemcpyMmapPressure_UnderPressure) {
  const char* skip = getenv("GPU_SKIP_PIN_UNDER_PRESSURE");
  if (skip && atoi(skip) == 0) {
    HIP_SKIP_TEST("GPU_SKIP_PIN_UNDER_PRESSURE=0 disables the behavior under test");
  }
  Cgroup cg = findLimitingCgroup();
  Thresholds t = detectorThresholds(cg);
  if (t.psi_percent > 0 && psiSomeAvg10(cg) < 0) {
    HIP_SKIP_TEST("Memory PSI is unavailable, so the runtime's pressure detector is off");
  }
  int64_t to_allocate = availableBytes(cg) - t.avail_bytes * 3 / 4;
  if (to_allocate > kMaxHogBytes) {
    HIP_SKIP_TEST("Creating pressure would need " << to_allocate / kGiB
                                                  << " GiB of allocations (budget "
                                                  << kMaxHogBytes / kGiB << " GiB)");
  }

  int fd = createDiskBackedFile(kFileSize);
  if (fd < 0) HIP_SKIP_TEST("No disk-backed directory with enough free space");
  CopyResult baseline = copyMappedFile(fd, kFileSize);

  // Aim a quarter below the detector threshold so reclaim is active.
  MemoryHog hog;
  bool reached = hog.fill(cg, t.avail_bytes * 3 / 4);
  INFO("Available after hog: " << availableBytes(cg) / kMiB << " MiB, threshold "
                               << t.avail_bytes / kMiB << " MiB");
  if (!reached) {
    hog.release();
    close(fd);
    HIP_SKIP_TEST("Could not reach the target available memory within the budget");
  }

  // Touch the evicted file through the page cache until PSI reflects reclaim stalls.
  auto deadline = Clock::now() + kPsiWait;
  std::vector<char> scratch(kChunkSize);
  while (!detectorWouldTrigger(cg, t) && Clock::now() < deadline) {
    posix_fadvise(fd, 0, kFileSize, POSIX_FADV_DONTNEED);
    for (off_t off = 0; off < static_cast<off_t>(kFileSize); off += kChunkSize) {
      if (pread(fd, scratch.data(), kChunkSize, off) <= 0) break;
    }
  }
  if (!detectorWouldTrigger(cg, t)) {
    hog.release();
    close(fd);
    HIP_SKIP_TEST("Detector conditions not met (PSI some avg10 " << psiSomeAvg10(cg)
                                                                 << "%, threshold "
                                                                 << t.psi_percent << "%)");
  }

  CopyResult pressure = copyMappedFile(fd, kFileSize);
  hog.release();
  close(fd);

  INFO("Baseline " << baseline.total_s << " s; under pressure " << pressure.total_s
                   << " s (" << pressure.total_s / baseline.total_s << "x), worst chunk "
                   << pressure.worst_chunk_ms << " ms");
  REQUIRE(pressure.data_ok);
  CHECK(pressure.worst_chunk_ms < kMaxChunkMs);
  CHECK(pressure.total_s < kMaxTotalS);
}

#endif  // __linux__
