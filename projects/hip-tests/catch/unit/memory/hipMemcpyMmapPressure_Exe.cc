/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Helper for hipMemcpyMmapPressure.cc: copies a file-backed mapping to the GPU in
// pinned-transfer-sized chunks and verifies every chunk. Runs in its own process
// so the parent test can set GPU_* / AMD_LOG_* variables, which the runtime reads
// at startup.
//
// Usage: hipMemcpyMmapPressureHelper <size_mib>
// Exit codes: 0 success, 1 HIP error, 2 data mismatch, 3 no disk-backed directory.

#include <hip/hip_runtime.h>

#include <fcntl.h>
#include <linux/magic.h>
#include <sys/mman.h>
#include <sys/statvfs.h>
#include <sys/vfs.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr size_t kMiB = 1024 * 1024;
constexpr size_t kChunkSize = 32 * kMiB;  // matches the default pinned transfer size

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

}  // namespace

#define HELPER_CHECK(expr)                                                                         \
  do {                                                                                             \
    hipError_t err_ = (expr);                                                                      \
    if (err_ != hipSuccess) {                                                                      \
      fprintf(stderr, "%s failed: %s\n", #expr, hipGetErrorString(err_));                         \
      return 1;                                                                                    \
    }                                                                                              \
  } while (0)

int main(int argc, char** argv) {
  size_t size = (argc > 1 ? strtoull(argv[1], nullptr, 10) : 256) * kMiB;
  int fd = createDiskBackedFile(size);
  if (fd < 0) return 3;

  void* mapped = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (mapped == MAP_FAILED) return 3;
  posix_fadvise(fd, 0, size, POSIX_FADV_DONTNEED);
  madvise(mapped, size, MADV_DONTNEED);

  void* d_buf = nullptr;
  HELPER_CHECK(hipMalloc(&d_buf, kChunkSize));
  const char* src = static_cast<const char*>(mapped);
  std::vector<char> check(kChunkSize);
  for (size_t offset = 0; offset < size; offset += kChunkSize) {
    size_t n = std::min(kChunkSize, size - offset);
    HELPER_CHECK(hipMemcpy(d_buf, src + offset, n, hipMemcpyHostToDevice));
    HELPER_CHECK(hipMemcpy(check.data(), d_buf, n, hipMemcpyDeviceToHost));
    if (memcmp(check.data(), src + offset, n) != 0) {
      fprintf(stderr, "data mismatch at offset %zu\n", offset);
      return 2;
    }
  }
  HELPER_CHECK(hipFree(d_buf));
  munmap(mapped, size);
  close(fd);
  printf("copied %zu MiB\n", size / kMiB);
  return 0;
}
