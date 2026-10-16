/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

// Negative tests for the llvm-offload-binary package container. A package is
// recognized purely by its 4-byte magic, so a truncated or self-inconsistent one must be
// rejected as an invalid image rather than trusted far enough to read out of bounds.

namespace {

constexpr std::array<uint8_t, 4> kPackageMagic = {0x10, 0xFF, 0x10, 0xAD};
constexpr uint32_t kPackageVersion = 2;
// Magic[4] + Version[4] + Size[8] + EntriesOffset[8] + EntriesCount[8], per
// llvm::object::OffloadBinary::Header.
constexpr size_t kPackageHeaderSize = 32;
// ImageKind[2] + OffloadKind[2] + Flags[4] + StringOffset[8] + NumStrings[8] + ImageOffset[8] +
// ImageSize[8], per llvm::object::OffloadBinary::Entry. A package whose declared size cannot hold
// one entry is rejected on size alone, before any offset or count is looked at, so the sections
// below that target those checks must declare at least a header plus an entry.
constexpr size_t kPackageEntrySize = 40;
constexpr uint64_t kOutOfBoundsOffset = 0xFFFFFFF0ULL;

void AppendUint32LE(std::vector<uint8_t>& bytes, uint32_t value) {
  for (unsigned int i = 0; i < sizeof(value); ++i) {
    bytes.push_back(static_cast<uint8_t>(value >> (i * 8)));
  }
}

void AppendUint64LE(std::vector<uint8_t>& bytes, uint64_t value) {
  for (unsigned int i = 0; i < sizeof(value); ++i) {
    bytes.push_back(static_cast<uint8_t>(value >> (i * 8)));
  }
}

std::vector<uint8_t> BuildPackageHeader(uint64_t total_size, uint64_t entries_offset,
                                        uint64_t entries_count) {
  std::vector<uint8_t> bytes;
  bytes.insert(bytes.end(), kPackageMagic.begin(), kPackageMagic.end());
  AppendUint32LE(bytes, kPackageVersion);
  AppendUint64LE(bytes, total_size);
  AppendUint64LE(bytes, entries_offset);
  AppendUint64LE(bytes, entries_count);
  REQUIRE(bytes.size() == kPackageHeaderSize);
  return bytes;
}

class ScopedPackageFile {
 public:
  explicit ScopedPackageFile(const std::vector<uint8_t>& bytes) {
    const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("hipModuleLoadMalformedPackage_" + std::to_string(timestamp) + "_" +
             std::to_string(reinterpret_cast<uintptr_t>(bytes.data())) + ".code");

    std::ofstream file(path_, std::ios::binary | std::ios::trunc);
    REQUIRE(file.is_open());
    file.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    REQUIRE(file.good());
  }

  ~ScopedPackageFile() {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

  std::string path() const { return path_.string(); }

 private:
  std::filesystem::path path_;
};

void ExpectFileLoadRejected(const std::vector<uint8_t>& package) {
  const ScopedPackageFile file(package);
  const std::string path = file.path();
  hipModule_t module = nullptr;
  HIP_CHECK_ERROR(hipModuleLoad(&module, path.c_str()), hipErrorInvalidImage);
}

}  // namespace

HIP_TEST_CASE(Unit_hipModuleLoad_Negative_MalformedPackageBounds) {
  HIP_CHECK(hipFree(nullptr));

  // hipModuleLoadData is not checked here: it has no length parameter, so a malformed image on
  // that path is undefined behavior.
  SECTION("package header is truncated") {
    auto package = BuildPackageHeader(kPackageHeaderSize, kPackageHeaderSize, 1);
    package.resize(kPackageHeaderSize / 2);
    ExpectFileLoadRejected(package);
  }

  SECTION("declared size is larger than the mapped file") {
    ExpectFileLoadRejected(BuildPackageHeader(kOutOfBoundsOffset, kPackageHeaderSize, 1));
  }

  SECTION("declared size is smaller than the header itself") {
    ExpectFileLoadRejected(BuildPackageHeader(kPackageHeaderSize / 2, kPackageHeaderSize, 1));
  }

  SECTION("entries offset is outside the readable image") {
    // Declares room for exactly one entry, so the declared size itself is consistent and the
    // out-of-bounds entries offset is what must be caught.
    constexpr uint64_t kTotalSize = kPackageHeaderSize + kPackageEntrySize;
    auto package = BuildPackageHeader(kTotalSize, kOutOfBoundsOffset, 1);
    package.resize(kTotalSize, 0);
    ExpectFileLoadRejected(package);
  }

  SECTION("entries count exceeds what the declared size can hold") {
    constexpr uint64_t kTotalSize = kPackageHeaderSize + kPackageEntrySize;
    auto package = BuildPackageHeader(kTotalSize, kPackageHeaderSize, 0xFFFFFFFFULL);
    package.resize(kTotalSize, 0);
    ExpectFileLoadRejected(package);
  }

  SECTION("magic is present but the body is absent") {
    std::vector<uint8_t> package(kPackageMagic.begin(), kPackageMagic.end());
    ExpectFileLoadRejected(package);
  }
}
