/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * Regression coverage for a large pageable host->device hipMemcpy whose source
 * is a file-backed mmap the kernel has evicted from the page cache (e.g. an
 * mmap'd weight file being streamed to the GPU with the host under memory
 * pressure). clr pins that source one GPU_PINNED_XFER_SIZE chunk at a time via
 * hsa_amd_memory_lock(); before ROCm/rocm-systems#11171 every chunk faulted its
 * pages back in one at a time under get_user_pages(). The PR issues an up-front
 * madvise(MADV_WILLNEED) over the whole range so the reclaimed pages come back
 * with clustered readahead instead.
 *
 * This test exercises that exact path and checks the copy is correct (the
 * WILLNEED hint must not perturb the data) for both the sync and the async API,
 * with the source deliberately evicted first.
 */

#include <hip_test_common.hh>

#ifdef __linux__
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

constexpr uint64_t kMiB = 1024ULL * 1024ULL;

// Big enough to span several pin chunks (GPU_PINNED_XFER_SIZE defaults to tens
// of MiB) so the chunked pin loop in clr is actually exercised; small enough
// for CI. Quick runs stay modest.
uint64_t transferSize() { return (isQuickLevel() ? 64ULL : 384ULL) * kMiB; }

struct ScopedFile {
  std::string path;
  int fd = -1;
  ~ScopedFile() {
    if (fd >= 0) close(fd);
    if (!path.empty()) unlink(path.c_str());
  }
};

// Create a file of `size` bytes filled with a byte pattern derived from the
// offset, and return it open (read-only) with the page cache for it dropped.
bool makeEvictedFile(ScopedFile* sf, uint64_t size) {
  sf->path = "hipMemcpyPageableFileBacked.bin";
  int wfd = open(sf->path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
  if (wfd < 0) return false;

  std::vector<uint8_t> buf(1 * kMiB);
  for (uint64_t off = 0; off < size;) {
    const uint64_t n = std::min<uint64_t>(buf.size(), size - off);
    for (uint64_t i = 0; i < n; ++i) buf[i] = static_cast<uint8_t>((off + i) * 131 + 7);
    if (write(wfd, buf.data(), n) != static_cast<ssize_t>(n)) {
      close(wfd);
      return false;
    }
    off += n;
  }
  fsync(wfd);
  close(wfd);

  sf->fd = open(sf->path.c_str(), O_RDONLY);
  if (sf->fd < 0) return false;
  // Drop it from the page cache so the copy path has to fault it back in.
  posix_fadvise(sf->fd, 0, 0, POSIX_FADV_DONTNEED);
  return true;
}

uint8_t expectedByte(uint64_t idx) { return static_cast<uint8_t>(idx * 131 + 7); }

}  // namespace

HIP_TEST_CASE(Unit_hipMemcpy_PageableFileBackedSource_Evicted) {
  const bool async = GENERATE(false, true);
  INFO("async = " << async);

  const uint64_t size = transferSize();

  size_t freeMem = 0, totalMem = 0;
  HIP_CHECK(hipMemGetInfo(&freeMem, &totalMem));
  if (freeMem < size + 64 * kMiB) {
    HIP_SKIP_TEST(HipTest::SkipReason::kNotEnoughFreeGpuMemory);
    return;
  }

  ScopedFile sf;
  if (!makeEvictedFile(&sf, size)) {
    HIP_SKIP_TEST("could not create/evict the backing file");
    return;
  }

  void* mapped = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, sf.fd, 0);
  REQUIRE(mapped != MAP_FAILED);
  // Second best-effort eviction: drop any pages the write left behind.
  madvise(mapped, size, MADV_DONTNEED);
  posix_fadvise(sf.fd, 0, 0, POSIX_FADV_DONTNEED);
  const auto* src = static_cast<const uint8_t*>(mapped);

  uint8_t* devPtr = nullptr;
  HIP_CHECK(hipMalloc(&devPtr, size));

  if (async) {
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipMemcpyAsync(devPtr, src, size, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipStreamDestroy(stream));
  } else {
    HIP_CHECK(hipMemcpy(devPtr, src, size, hipMemcpyHostToDevice));
  }

  std::vector<uint8_t> out(size);
  HIP_CHECK(hipMemcpy(out.data(), devPtr, size, hipMemcpyDeviceToHost));

  // Check the whole range at page granularity plus every byte of the first and
  // last chunk, so a wrong offset or a dropped chunk in the pin loop shows.
  uint64_t firstBad = size;
  for (uint64_t i = 0; i < size; i += 4096) {
    if (out[i] != expectedByte(i)) { firstBad = i; break; }
  }
  const uint64_t edge = std::min<uint64_t>(size, 8 * kMiB);
  for (uint64_t i = 0; i < edge && firstBad == size; ++i)
    if (out[i] != expectedByte(i)) firstBad = i;
  for (uint64_t i = size - edge; i < size && firstBad == size; ++i)
    if (out[i] != expectedByte(i)) firstBad = i;

  INFO("first mismatching byte offset = " << firstBad);
  REQUIRE(firstBad == size);

  HIP_CHECK(hipFree(devPtr));
  munmap(mapped, size);
}

#else  // !__linux__

HIP_TEST_CASE(Unit_hipMemcpy_PageableFileBackedSource_Evicted) {
  HIP_SKIP_TEST(HipTest::SkipReason::kRequiresLinux);
}

#endif
