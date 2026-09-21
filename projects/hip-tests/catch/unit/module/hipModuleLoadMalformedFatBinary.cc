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

namespace {

constexpr char kBundleMagic[] = "__CLANG_OFFLOAD_BUNDLE__";
constexpr uint64_t kOutOfBoundsOffset = 0xFFFFFFF0ULL;
constexpr uint64_t kCodeObjectSize = 0x1000ULL;
constexpr size_t kComgrHeaderWindow = 4096;

void AppendUint64LE(std::vector<uint8_t>& bytes, uint64_t value) {
  for (unsigned int i = 0; i < sizeof(value); ++i) {
    bytes.push_back(static_cast<uint8_t>(value >> (i * 8)));
  }
}

void AppendBundleEntry(std::vector<uint8_t>& bytes, const std::string& id, uint64_t offset,
                       uint64_t size) {
  AppendUint64LE(bytes, offset);
  AppendUint64LE(bytes, size);
  AppendUint64LE(bytes, static_cast<uint64_t>(id.size()));
  bytes.insert(bytes.end(), id.begin(), id.end());
}

std::vector<uint8_t> BuildBundle(uint64_t offset, uint64_t size) {
  static const std::array<std::string, 4> kBundleIds = {
      "hip-spirv64-amd-amdhsa--amdgcnspirv",
      "hip-spirv64-amd-amdhsa-unknown-amdgcnspirv",
      "hipv4-spirv64-amd-amdhsa--amdgcnspirv",
      "hipv4-spirv64-amd-amdhsa-unknown-amdgcnspirv",
  };

  std::vector<uint8_t> bytes;
  bytes.insert(bytes.end(), kBundleMagic, kBundleMagic + sizeof(kBundleMagic) - 1);
  AppendUint64LE(bytes, static_cast<uint64_t>(kBundleIds.size()));
  for (const auto& id : kBundleIds) {
    AppendBundleEntry(bytes, id, offset, size);
  }
  return bytes;
}

class ScopedBundleFile {
 public:
  explicit ScopedBundleFile(const std::vector<uint8_t>& bytes) {
    const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("hipModuleLoadMalformedFatBinary_" + std::to_string(timestamp) + "_" +
             std::to_string(reinterpret_cast<uintptr_t>(bytes.data())) + ".code");

    std::ofstream file(path_, std::ios::binary | std::ios::trunc);
    REQUIRE(file.is_open());
    file.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    REQUIRE(file.good());
  }

  ~ScopedBundleFile() {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

  std::string path() const { return path_.string(); }

 private:
  std::filesystem::path path_;
};

void ExpectFileLoadRejected(const std::vector<uint8_t>& bundle) {
  const ScopedBundleFile file(bundle);
  const std::string path = file.path();
  hipModule_t module = nullptr;
  HIP_CHECK_ERROR(hipModuleLoad(&module, path.c_str()), hipErrorInvalidImage);
}

}  // namespace

HIP_TEST_CASE(Unit_hipModuleLoad_Negative_MalformedFatBinaryBounds) {
  HIP_CHECK(hipFree(nullptr));

  // hipModuleLoadData is not checked here: it has no length parameter, so a
  // malformed image on that path is undefined behavior.
  SECTION("code object offset is outside the readable image") {
    ExpectFileLoadRejected(BuildBundle(kOutOfBoundsOffset, kCodeObjectSize));
  }

  SECTION("code object size crosses the mapped file boundary") {
    auto bundle = BuildBundle(kComgrHeaderWindow - 1, 2);
    REQUIRE(bundle.size() < kComgrHeaderWindow);
    bundle.resize(kComgrHeaderWindow);
    ExpectFileLoadRejected(bundle);
  }
}
