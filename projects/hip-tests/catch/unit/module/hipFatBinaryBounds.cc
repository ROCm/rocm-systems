/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Robustness tests for the fat-binary image bounds checks that were added
// together with amd::Os::GetReadableSizeFromAddress. The pointer-based load
// APIs (hipModuleLoadData / hipModuleLoadFatBinary) carry no length, so the
// runtime recovers the readable extent of a *file-backed* mapping from the OS
// and rejects images that are too small to safely parse (magic strings, the
// compressed bundle header, an out-of-bounds totalSize, or an ELF header).
//
// To exercise those checks the crafted image must live in a file-backed
// mapping (heap buffers report an unknown size, so the checks are skipped by
// design). Each malformed image is placed flush against the end of a
// file-backed page that is immediately followed by an unmapped PROT_NONE guard
// page. With the fix the runtime returns hipErrorInvalidImage; without it the
// parser reads past the declared image and faults into the guard page, turning
// a silent out-of-bounds read into a hard, detectable failure.

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>

#if HT_AMD && defined(__linux__)

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Maps `payload` so that its final byte sits flush against the end of a
// file-backed page, immediately followed by an unmapped PROT_NONE guard page.
// amd::Os::GetReadableSizeFromAddress (via /proc/self/maps) then reports the
// image's readable extent as exactly payload.size().
class TailMappedImage {
 public:
  explicit TailMappedImage(const std::vector<unsigned char>& payload) {
    page_size_ = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    REQUIRE(!payload.empty());
    REQUIRE(payload.size() <= page_size_);

    char tmpl[] = "/tmp/hip_fatbin_bounds_XXXXXX";
    fd_ = mkstemp(tmpl);
    REQUIRE(fd_ >= 0);
    path_ = tmpl;

    // File is exactly one page; payload occupies the very end of it.
    std::vector<unsigned char> file_bytes(page_size_, 0);
    std::memcpy(file_bytes.data() + (page_size_ - payload.size()), payload.data(),
                payload.size());
    REQUIRE(write(fd_, file_bytes.data(), file_bytes.size()) ==
            static_cast<ssize_t>(file_bytes.size()));

    // Reserve two pages, then drop a file mapping over the first one. The
    // second page stays PROT_NONE and serves as the guard page.
    region_len_ = 2 * page_size_;
    region_ = mmap(nullptr, region_len_, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    REQUIRE(region_ != MAP_FAILED);

    void* file_page = mmap(region_, page_size_, PROT_READ, MAP_PRIVATE | MAP_FIXED, fd_, 0);
    REQUIRE(file_page == region_);

    image_ = static_cast<unsigned char*>(region_) + (page_size_ - payload.size());
  }

  ~TailMappedImage() {
    if (region_ != nullptr && region_ != MAP_FAILED) {
      munmap(region_, region_len_);
    }
    if (fd_ >= 0) {
      close(fd_);
    }
    if (!path_.empty()) {
      unlink(path_.c_str());
    }
  }

  TailMappedImage(const TailMappedImage&) = delete;
  TailMappedImage& operator=(const TailMappedImage&) = delete;

  const void* image() const { return image_; }

 private:
  size_t page_size_ = 0;
  int fd_ = -1;
  std::string path_;
  void* region_ = nullptr;
  size_t region_len_ = 0;
  unsigned char* image_ = nullptr;
};

}  // namespace

/**
 * Test Description
 * ------------------------
 * - Feeds truncated / malformed fat-binary images, mapped flush against an
 *   unmapped guard page, to the pointer-based module load APIs and verifies
 *   they are rejected with hipErrorInvalidImage instead of reading out of
 *   bounds. Each section targets one of the bounds checks in
 *   FatBinaryInfo::ExtractFatBinaryUsingCOMGR.
 * Test source
 * ------------------------
 * - catch/unit/module/hipFatBinaryBounds.cc
 * Test requirements
 * ------------------------
 * - Linux, AMD backend
 */
HIP_TEST_CASE(Unit_hipModuleLoadData_Bounds_TruncatedImages) {
  HIP_CHECK(hipFree(nullptr));
  hipModule_t module = nullptr;

  SECTION("compressed magic but image smaller than the compressed header") {
    const std::vector<unsigned char> payload = {'C', 'C', 'O', 'B'};
    TailMappedImage img(payload);
    HIP_CHECK_ERROR(hipModuleLoadData(&module, img.image()), hipErrorInvalidImage);
  }

  SECTION("compressed header with out-of-bounds totalSize") {
    // Full compressed header (>= 24 bytes) with a bogus totalSize that far
    // exceeds the readable extent. totalSize lives at byte offset 8.
    std::vector<unsigned char> payload(32, 0);
    std::memcpy(payload.data(), "CCOB", 4);
    const uint32_t total_size = 0xFFFFFFFFu;
    std::memcpy(payload.data() + 8, &total_size, sizeof(total_size));
    TailMappedImage img(payload);
    HIP_CHECK_ERROR(hipModuleLoadData(&module, img.image()), hipErrorInvalidImage);
  }

  SECTION("ELF-like header smaller than Elf64_Ehdr") {
    const std::vector<unsigned char> payload = {0x7f, 'E', 'L', 'F', 2, 1, 1, 0};
    TailMappedImage img(payload);
    HIP_CHECK_ERROR(hipModuleLoadData(&module, img.image()), hipErrorInvalidImage);
  }

  SECTION("uncompressed bundle magic truncated below magic length") {
    const char* magic = "__CLANG_OFFLOAD_BUNDLE__";
    const std::vector<unsigned char> payload(magic, magic + 10);
    TailMappedImage img(payload);
    HIP_CHECK_ERROR(hipModuleLoadData(&module, img.image()), hipErrorInvalidImage);
  }

  SECTION("hipModuleLoadFatBinary rejects a truncated compressed image") {
    const std::vector<unsigned char> payload = {'C', 'C', 'O', 'B'};
    TailMappedImage img(payload);
    HIP_CHECK_ERROR(hipModuleLoadFatBinary(&module, img.image()), hipErrorInvalidImage);
  }
}

/**
 * Test Description
 * ------------------------
 * - Maps a legitimately sized compressed module from disk (file-backed) and
 *   loads it through hipModuleLoadData. This guards against the bounds checks
 *   regressing into false positives: GetReadableSizeFromAddress must report an
 *   extent large enough that a valid image still loads.
 * Test source
 * ------------------------
 * - catch/unit/module/hipFatBinaryBounds.cc
 * Test requirements
 * ------------------------
 * - Linux, AMD backend
 */
HIP_TEST_CASE(Unit_hipModuleLoadData_Bounds_ValidFileBackedImageLoads) {
  HIP_CHECK(hipFree(nullptr));

  constexpr const char* kPath = "copyKernelCompressed.code";
  const int fd = open(kPath, O_RDONLY);
  REQUIRE(fd >= 0);

  struct stat st {};
  REQUIRE(fstat(fd, &st) == 0);
  REQUIRE(st.st_size > 0);

  void* mapped =
      mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
  REQUIRE(mapped != MAP_FAILED);

  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoadData(&module, mapped));
  REQUIRE(module != nullptr);
  HIP_CHECK(hipModuleUnload(module));

  munmap(mapped, static_cast<size_t>(st.st_size));
  close(fd);
}

#endif  // HT_AMD && defined(__linux__)
