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
// over-reads fault instead of silently succeeding, while correct parsing
// returns hipErrorInvalidImage.
//
// This suite is Linux-only (built under unit/oob, which is gated to UNIX); the
// bundle-header/magic and ELF-header-size bounds it exercises complement the
// getElfSize ELF-internal cases in oob_module.cc.

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <hip_test_filesystem.hh>

#if HT_AMD

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <system_error>
#include <vector>

namespace {

// Magic strings for ROCm clang offload bundles.
constexpr char kCompressedBundleMagic[] = "CCOB";
constexpr char kUncompressedBundleMagic[] = "__CLANG_OFFLOAD_BUNDLE__";

// AMDGPU ELF identity values the runtime checks. Defined locally so this test
// stays independent of runtime headers.
constexpr unsigned char kElfOsabiAmdgpuHsa = 64;  // ELFOSABI_AMDGPU_HSA
constexpr unsigned char kEmAmdgpuLowByte = 224;   // EM_AMDGPU (0x00E0), low byte

// Builds a `size`-byte compressed offload bundle: the "CCOB" magic with
// `total_size` written to the totalSize field (offset 8); remaining bytes zero.
std::vector<unsigned char> MakeCompressedBundle(size_t size, uint32_t total_size) {
  REQUIRE(size >= 12);  // magic (4) through totalSize (offset 8, 4 bytes)
  std::vector<unsigned char> image(size, 0);
  std::memcpy(image.data(), kCompressedBundleMagic, 4);
  std::memcpy(image.data() + 8, &total_size, sizeof(total_size));
  return image;
}

// Builds a `size`-byte little-endian AMDGPU ELFCLASS64 image whose identity
// fields (EM_AMDGPU + ELFOSABI_AMDGPU_HSA) pass the runtime's ELF check. Only
// the e_ident bytes and e_machine are set; all other fields are zero.
std::vector<unsigned char> MakeAmdgpuElf64Image(size_t size) {
  REQUIRE(size >= 20);  // the last field written, spans offsets 18-19
  std::vector<unsigned char> image(size, 0);
  image[0] = 0x7f;
  image[1] = 'E';
  image[2] = 'L';
  image[3] = 'F';
  image[4] = 2;  // EI_CLASS   = ELFCLASS64
  image[5] = 1;  // EI_DATA    = ELFDATA2LSB (little-endian)
  image[6] = 1;  // EI_VERSION = EV_CURRENT
  image[7] = kElfOsabiAmdgpuHsa;   // EI_OSABI
  image[18] = kEmAmdgpuLowByte;    // e_machine low byte (little-endian)
  return image;
}

// Read-only file mapping used by the valid-image false-positive guard.
class FileBackedMapping {
 public:
  explicit FileBackedMapping(const char* path) {
    fd_ = open(path, O_RDONLY);
    REQUIRE(fd_ >= 0);

    struct stat st {};
    REQUIRE(fstat(fd_, &st) == 0);
    REQUIRE(st.st_size > 0);
    size_ = static_cast<size_t>(st.st_size);

    data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    REQUIRE(data_ != MAP_FAILED);
  }

  ~FileBackedMapping() {
    if (data_ != nullptr && data_ != MAP_FAILED) {
      munmap(data_, size_);
    }
    if (fd_ >= 0) {
      close(fd_);
    }
  }

  FileBackedMapping(const FileBackedMapping&) = delete;
  FileBackedMapping& operator=(const FileBackedMapping&) = delete;

  const void* data() const { return data_; }
  size_t size() const { return size_; }

 private:
  int fd_ = -1;
  void* data_ = nullptr;
  size_t size_ = 0;
};

// Temporary file for hipModuleLoad file-path tests.
class TempFile {
 public:
  explicit TempFile(const std::vector<unsigned char>& bytes) {
    path_ = (fs::temp_directory_path() /
             ("hip_fatbin_bounds_" + std::to_string(std::random_device{}()) + ".code"))
                .string();
    std::ofstream file(path_, std::ios::binary | std::ios::trunc);
    REQUIRE(file.is_open());
    file.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    file.close();
    REQUIRE(file.good());
  }

  ~TempFile() {
    std::error_code ec;
    fs::remove(path_, ec);
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

}  // namespace

/**
 * Feeds guard-page-backed malformed images to the pointer load APIs and
 * expects hipErrorInvalidImage instead of an out-of-bounds read.
 */
HIP_TEST_CASE(Unit_hipModuleLoadData_Negative_TruncatedImages) {
  hipModule_t module = nullptr;

  SECTION("compressed magic shorter than the magic itself") {
    // Only 3 readable bytes, fewer than the 4-byte compressed magic. Checks that
    // the magic comparison doesn't read past the image.
    const std::vector<unsigned char> payload(kCompressedBundleMagic,
                                             kCompressedBundleMagic + 3);
    TailMappedImage img(payload);
    HIP_CHECK_ERROR(hipModuleLoadData(&module, img.image()), hipErrorInvalidImage);
  }

  SECTION("compressed magic with too little readable image") {
    // Only 4 readable bytes (magic, no header). Checks that probing for a header
    // doesn't read past the image.
    const std::vector<unsigned char> payload(kCompressedBundleMagic,
                                             kCompressedBundleMagic + 4);
    TailMappedImage img(payload);
    HIP_CHECK_ERROR(hipModuleLoadData(&module, img.image()), hipErrorInvalidImage);
  }

  SECTION("compressed header with out-of-bounds totalSize") {
    // Valid 32-byte header, but its declared totalSize (offset 8) claims a far
    // larger image than exists. Checks that the runtime doesn't trust that size
    // and read past the image.
    const std::vector<unsigned char> payload = MakeCompressedBundle(32, 0xFFFFFFFFu);
    TailMappedImage img(payload);
    HIP_CHECK_ERROR(hipModuleLoadData(&module, img.image()), hipErrorInvalidImage);
  }

  SECTION("AMDGPU ELF header smaller than Elf64_Ehdr") {
    // 40-byte AMDGPU ELF header: big enough that the magic checks stay in
    // bounds, but smaller than a full Elf64_Ehdr (64 bytes). Checks that reading
    // ELF header fields doesn't read past the image.
    const std::vector<unsigned char> payload = MakeAmdgpuElf64Image(40);
    TailMappedImage img(payload);
    HIP_CHECK_ERROR(hipModuleLoadData(&module, img.image()), hipErrorInvalidImage);
  }

  SECTION("uncompressed bundle magic truncated below magic length") {
    // Only 10 readable bytes of the 24-byte uncompressed magic. Checks that
    // comparing the full magic doesn't read past the image.
    const std::vector<unsigned char> payload(kUncompressedBundleMagic,
                                             kUncompressedBundleMagic + 10);
    TailMappedImage img(payload);
    HIP_CHECK_ERROR(hipModuleLoadData(&module, img.image()), hipErrorInvalidImage);
  }

  SECTION("AMDGPU ELF whose declared size exceeds the readable image") {
    // Full 64-byte Elf64_Ehdr so the header checks pass, but e_shoff points a
    // section table far past the image. e_shnum stays 0 so computing the ELF
    // size does not itself walk out of bounds; the computed size is what is out
    // of bounds. Checks that the oversized size isn't used to read past the image
    // when the code object is handed off for loading.
    std::vector<unsigned char> payload = MakeAmdgpuElf64Image(64);
    const uint64_t e_shoff = 0x1000000;                           // section table far past 64 bytes
    std::memcpy(payload.data() + 40, &e_shoff, sizeof(e_shoff));  // Elf64_Ehdr::e_shoff
    TailMappedImage img(payload);
    HIP_CHECK_ERROR(hipModuleLoadData(&module, img.image()), hipErrorInvalidImage);
  }

  SECTION("uncompressed bundle magic with no readable body") {
    // Exactly the 24-byte magic and nothing else. Checks that reading the bundle
    // header is clamped to the readable size rather than a fixed 4096-byte slice
    // that reads past the image.
    const std::vector<unsigned char> payload(kUncompressedBundleMagic,
                                             kUncompressedBundleMagic + 24);
    TailMappedImage img(payload);
    HIP_CHECK_ERROR(hipModuleLoadData(&module, img.image()), hipErrorInvalidImage);
  }
}

/**
 * Loads a valid file-backed compressed module and runs its kernel, guarding
 * against the bounds checks regressing into false positives.
 */
HIP_TEST_CASE(Unit_hipModuleLoadData_Positive_ValidFileBackedImage) {
  FileBackedMapping mapping("oob_copyKernelCompressed.code");

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
 * file. hipModuleLoad must reject it with hipErrorInvalidImage.
 */
HIP_TEST_CASE(Unit_hipModuleLoad_Negative_OutOfBoundsTotalSize) {
  // Full header, but totalSize exceeds the file size.
  const std::vector<unsigned char> payload = MakeCompressedBundle(32, 0xFFFFFFFFu);

  TempFile image(payload);
  hipModule_t module = nullptr;
  HIP_CHECK_ERROR(hipModuleLoad(&module, image.path()), hipErrorInvalidImage);
}

#endif  // HT_AMD
