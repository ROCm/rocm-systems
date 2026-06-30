/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Robustness tests for readable-size bounds checks in fat-binary parsing.
// Pointer-based load APIs carry no length, so the runtime can only bound
// parsing when the image comes from a file-backed mapping.
//
// Malformed pointer inputs are tail-mapped before a no-access guard page so
// pre-fix over-reads fault, while fixed code returns hipErrorInvalidImage.

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <hip_test_filesystem.hh>

#if HT_AMD

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace {

// Magic for ROCm compressed clang offload bundles.
constexpr char kCompressedBundleMagic[] = "CCOB";

// Read-only file mapping used by the valid-image false-positive guard.
class FileBackedMapping {
 public:
  explicit FileBackedMapping(const char* path) {
#if defined(_WIN32)
    file_ = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                        FILE_ATTRIBUTE_NORMAL, nullptr);
    REQUIRE(file_ != INVALID_HANDLE_VALUE);

    LARGE_INTEGER file_size{};
    REQUIRE(GetFileSizeEx(file_, &file_size));
    REQUIRE(file_size.QuadPart > 0);
    size_ = static_cast<size_t>(file_size.QuadPart);

    mapping_ = CreateFileMappingA(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    REQUIRE(mapping_ != nullptr);

    data_ = MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0);
    REQUIRE(data_ != nullptr);
#else
    fd_ = open(path, O_RDONLY);
    REQUIRE(fd_ >= 0);

    struct stat st {};
    REQUIRE(fstat(fd_, &st) == 0);
    REQUIRE(st.st_size > 0);
    size_ = static_cast<size_t>(st.st_size);

    data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    REQUIRE(data_ != MAP_FAILED);
#endif
  }

  ~FileBackedMapping() {
#if defined(_WIN32)
    if (data_ != nullptr) {
      UnmapViewOfFile(data_);
    }
    if (mapping_ != nullptr) {
      CloseHandle(mapping_);
    }
    if (file_ != INVALID_HANDLE_VALUE) {
      CloseHandle(file_);
    }
#else
    if (data_ != nullptr && data_ != MAP_FAILED) {
      munmap(data_, size_);
    }
    if (fd_ >= 0) {
      close(fd_);
    }
#endif
  }

  FileBackedMapping(const FileBackedMapping&) = delete;
  FileBackedMapping& operator=(const FileBackedMapping&) = delete;

  const void* data() const { return data_; }
  size_t size() const { return size_; }

 private:
#if defined(_WIN32)
  HANDLE file_ = INVALID_HANDLE_VALUE;
  HANDLE mapping_ = nullptr;
#else
  int fd_ = -1;
#endif
  void* data_ = nullptr;
  size_t size_ = 0;
};

// Temporary file for hipModuleLoad file-path tests.
class TempFile {
 public:
  explicit TempFile(const std::vector<unsigned char>& bytes) {
#if defined(_WIN32)
    char dir[MAX_PATH] = {0};
    const DWORD dir_len = GetTempPathA(static_cast<DWORD>(sizeof(dir)), dir);
    REQUIRE(dir_len > 0);
    REQUIRE(dir_len < sizeof(dir));
    char path[MAX_PATH] = {0};
    REQUIRE(GetTempFileNameA(dir, "hip", 0, path) != 0);
    path_ = path;
#else
    char tmpl[] = "/tmp/hip_fatbin_bounds_XXXXXX";
    const int fd = mkstemp(tmpl);
    REQUIRE(fd >= 0);
    close(fd);
    path_ = tmpl;
#endif

    std::ofstream file(path_, std::ios::binary | std::ios::trunc);
    REQUIRE(file.is_open());
    if (!bytes.empty()) {
      file.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
      REQUIRE(file.good());
    }
    file.close();
    REQUIRE(file.good());
  }

  ~TempFile() {
    if (!path_.empty()) {
      std::error_code ec;
      fs::remove(path_, ec);
    }
  }

  TempFile(const TempFile&) = delete;
  TempFile& operator=(const TempFile&) = delete;

  const char* path() const { return path_.c_str(); }

 private:
  std::string path_;
};

// Places `payload` at the end of a file-backed page immediately followed by a
// no-access guard page. The runtime then sees exactly payload.size() readable
// bytes, and any read past the payload faults instead of succeeding silently.
#if defined(_WIN32)

class TailMappedImage {
 public:
  explicit TailMappedImage(const std::vector<unsigned char>& payload) {
    SYSTEM_INFO si{};
    GetSystemInfo(&si);
    page_size_ = si.dwPageSize;
    REQUIRE(!payload.empty());
    REQUIRE(payload.size() <= page_size_);

    char dir[MAX_PATH] = {0};
    const DWORD dir_len = GetTempPathA(static_cast<DWORD>(sizeof(dir)), dir);
    REQUIRE(dir_len > 0);
    REQUIRE(dir_len < sizeof(dir));
    char path[MAX_PATH] = {0};
    REQUIRE(GetTempFileNameA(dir, "hip", 0, path) != 0);
    path_ = path;

    // Back the image with a one-page file holding the payload at its very end
    // (leading bytes zero-padded), so the payload's last byte sits flush against
    // the page boundary once mapped.
    std::vector<unsigned char> file_bytes(page_size_, 0);
    std::memcpy(file_bytes.data() + (page_size_ - payload.size()), payload.data(),
                payload.size());

    file_ = CreateFileA(path_.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                        FILE_ATTRIBUTE_NORMAL, nullptr);
    REQUIRE(file_ != INVALID_HANDLE_VALUE);
    DWORD written = 0;
    REQUIRE(WriteFile(file_, file_bytes.data(), static_cast<DWORD>(file_bytes.size()), &written,
                      nullptr));
    REQUIRE(written == file_bytes.size());

    // Map exactly the one-page file. The view occupies a full allocation-
    // granularity slot, but only this page is backed; the pages after it stay
    // unmapped and act as the guard, so any read past the page faults.
    mapping_ = CreateFileMappingA(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    REQUIRE(mapping_ != nullptr);
    base_ = MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0);
    REQUIRE(base_ != nullptr);

    image_ = static_cast<unsigned char*>(base_) + (page_size_ - payload.size());
  }

  ~TailMappedImage() {
    if (base_ != nullptr) {
      UnmapViewOfFile(base_);
    }
    if (mapping_ != nullptr) {
      CloseHandle(mapping_);
    }
    if (file_ != INVALID_HANDLE_VALUE) {
      CloseHandle(file_);
    }
    if (!path_.empty()) {
      DeleteFileA(path_.c_str());
    }
  }

  TailMappedImage(const TailMappedImage&) = delete;
  TailMappedImage& operator=(const TailMappedImage&) = delete;

  const void* image() const { return image_; }

 private:
  size_t page_size_ = 0;
  std::string path_;
  HANDLE file_ = INVALID_HANDLE_VALUE;
  HANDLE mapping_ = nullptr;
  void* base_ = nullptr;
  unsigned char* image_ = nullptr;
};

#else

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

    // Back the image with a one-page file holding the payload at its very end
    // (leading bytes zero-padded). Mapped at a page boundary below, this puts
    // the payload's last byte flush against the page boundary.
    std::vector<unsigned char> file_bytes(page_size_, 0);
    std::memcpy(file_bytes.data() + (page_size_ - payload.size()), payload.data(),
                payload.size());
    REQUIRE(write(fd_, file_bytes.data(), file_bytes.size()) ==
            static_cast<ssize_t>(file_bytes.size()));

    // Reserve two adjacent pages with no access. The second one stays
    // unmapped/no-access as a guard page, so any read past the first page faults.
    region_len_ = 2 * page_size_;
    region_ = mmap(nullptr, region_len_, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    REQUIRE(region_ != MAP_FAILED);

    // Map the file read-only over the first page only, leaving the guard page
    // intact immediately after it.
    void* file_page = mmap(region_, page_size_, PROT_READ, MAP_PRIVATE | MAP_FIXED, fd_, 0);
    REQUIRE(file_page == region_);

    // Point at the payload's start; image_ + payload.size() lands exactly on
    // the guard page, so reads past the payload fault.
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

#endif  // defined(_WIN32)

}  // namespace

/**
 * Feeds guard-page-backed malformed images to the pointer load APIs and
 * expects hipErrorInvalidImage instead of an out-of-bounds read. Regression
 * coverage for the clr fat-binary out-of-bounds read security fix.
 */
HIP_TEST_CASE(Unit_hipModuleLoadData_Negative_TruncatedImages) {
  hipModule_t module = nullptr;

  SECTION("compressed magic with too little readable image") {
    // Only 4 readable bytes (magic, no header). The fix rejects it; without it
    // the runtime reads past the image while probing for a header.
    const std::vector<unsigned char> payload(kCompressedBundleMagic,
                                             kCompressedBundleMagic + 4);
    TailMappedImage img(payload);
    HIP_CHECK_ERROR(hipModuleLoadData(&module, img.image()), hipErrorInvalidImage);
  }

  SECTION("compressed header with out-of-bounds totalSize") {
    // Valid 32-byte header, but its declared totalSize (offset 8) claims a far
    // larger image than exists. The fix rejects it; without it the runtime
    // trusts that size and reads past the image.
    std::vector<unsigned char> payload(32, 0);
    std::memcpy(payload.data(), kCompressedBundleMagic, 4);
    const uint32_t total_size = 0xFFFFFFFFu;
    std::memcpy(payload.data() + 8, &total_size, sizeof(total_size));
    TailMappedImage img(payload);
    HIP_CHECK_ERROR(hipModuleLoadData(&module, img.image()), hipErrorInvalidImage);
  }

  SECTION("AMDGPU ELF header smaller than Elf64_Ehdr") {
    // 40-byte AMDGPU ELF header: big enough that the magic checks stay in
    // bounds, but smaller than a full Elf64_Ehdr (64 bytes). The fix rejects
    // it; without it the runtime reads ELF header fields past the image.
    std::vector<unsigned char> payload(40, 0);
    payload[0] = 0x7f;
    payload[1] = 'E';
    payload[2] = 'L';
    payload[3] = 'F';
    payload[4] = 2;     // EI_CLASS  = ELFCLASS64
    payload[5] = 1;     // EI_DATA   = ELFDATA2LSB (little-endian)
    payload[6] = 1;     // EI_VERSION = EV_CURRENT
    payload[7] = 64;    // EI_OSABI  = ELFOSABI_AMDGPU_HSA
    payload[18] = 224;  // e_machine (low byte) = EM_AMDGPU
    TailMappedImage img(payload);
    HIP_CHECK_ERROR(hipModuleLoadData(&module, img.image()), hipErrorInvalidImage);
  }

  SECTION("uncompressed bundle magic truncated below magic length") {
    // Only 10 readable bytes of the 24-byte uncompressed magic. The fix rejects
    // it; without it the runtime compares the full magic and reads past the image.
    const char* magic = "__CLANG_OFFLOAD_BUNDLE__";
    const std::vector<unsigned char> payload(magic, magic + 10);
    TailMappedImage img(payload);
    HIP_CHECK_ERROR(hipModuleLoadData(&module, img.image()), hipErrorInvalidImage);
  }

  SECTION("hipModuleLoadFatBinary enforces the totalSize bound") {
    // Reuses the out-of-bounds totalSize image, but loads it via
    // hipModuleLoadFatBinary to confirm that API enforces the bound too.
    std::vector<unsigned char> payload(32, 0);
    std::memcpy(payload.data(), kCompressedBundleMagic, 4);
    const uint32_t total_size = 0xFFFFFFFFu;
    std::memcpy(payload.data() + 8, &total_size, sizeof(total_size));
    TailMappedImage img(payload);
    HIP_CHECK_ERROR(hipModuleLoadFatBinary(&module, img.image()), hipErrorInvalidImage);
  }
}

/**
 * Loads a valid file-backed compressed module and runs its kernel, guarding
 * against the bounds checks regressing into false positives.
 */
HIP_TEST_CASE(Unit_hipModuleLoadData_Positive_ValidFileBackedImage) {
  FileBackedMapping mapping("copyKernelCompressed.code");

  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoadData(&module, mapping.data()));
  REQUIRE(module != nullptr);

  hipFunction_t kernel = nullptr;
  HIP_CHECK(hipModuleGetFunction(&kernel, module, "copy_ker"));
  REQUIRE(kernel != nullptr);

  constexpr unsigned int kLen = 64;
  constexpr size_t kBytes = kLen * sizeof(int);
  std::vector<int> host_in(kLen), host_out(kLen, 0);
  for (unsigned int i = 0; i < kLen; ++i) {
    host_in[i] = static_cast<int>(i);
  }

  int* device_in = nullptr;
  int* device_out = nullptr;
  HIP_CHECK(hipMalloc(&device_in, kBytes));
  HIP_CHECK(hipMalloc(&device_out, kBytes));
  HIP_CHECK(hipMemcpy(device_in, host_in.data(), kBytes, hipMemcpyHostToDevice));

  struct {
    void* Ad;
    void* Bd;
    size_t size;
  } args{device_in, device_out, kLen};
  size_t args_size = sizeof(args);
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                    &args_size, HIP_LAUNCH_PARAM_END};
  HIP_CHECK(hipModuleLaunchKernel(kernel, 1, 1, 1, kLen, 1, 1, 0, nullptr, nullptr, config));
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(host_out.data(), device_out, kBytes, hipMemcpyDeviceToHost));
  for (unsigned int i = 0; i < kLen; ++i) {
    REQUIRE(host_out[i] == host_in[i]);
  }

  HIP_CHECK(hipFree(device_in));
  HIP_CHECK(hipFree(device_out));
  HIP_CHECK(hipModuleUnload(module));
}

/**
 * Loads a file whose compressed header declares a totalSize larger than the
 * file. hipModuleLoad must reject it with hipErrorInvalidImage. Regression
 * coverage for the clr fat-binary out-of-bounds read security fix.
 */
HIP_TEST_CASE(Unit_hipModuleLoad_Negative_OutOfBoundsTotalSize) {
  // Full header, but totalSize exceeds the file size.
  std::vector<unsigned char> payload(32, 0);
  std::memcpy(payload.data(), kCompressedBundleMagic, 4);
  const uint32_t total_size = 0xFFFFFFFFu;
  std::memcpy(payload.data() + 8, &total_size, sizeof(total_size));

  TempFile image(payload);
  hipModule_t module = nullptr;
  HIP_CHECK_ERROR(hipModuleLoad(&module, image.path()), hipErrorInvalidImage);
}

#endif  // HT_AMD
