// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <msgpack.hpp>
#include <string>
#include <thread>
#include <vector>

#include "kpack_internal.h"
#include "rocm_kpack/kpack.h"

// Helper to get test assets directory
static std::string get_test_assets_dir() {
  const char* dir = std::getenv("ROCM_KPACK_TEST_ASSETS_DIR");
  return dir ? dir : "";
}

// RAII wrapper for cache
class CacheGuard {
 public:
  CacheGuard() : cache_(nullptr) {}
  ~CacheGuard() {
    if (cache_) {
      kpack_cache_destroy(cache_);
    }
  }

  kpack_cache_t* ptr() { return &cache_; }
  kpack_cache_t get() const { return cache_; }

 private:
  kpack_cache_t cache_;
};

// RAII wrapper for environment variable
// Note: setenv() and unsetenv() are POSIX, not C++ standard.
// For Windows, use _putenv_s() instead.
class EnvGuard {
 public:
  EnvGuard(const char* name, const char* value) : name_(name) {
    const char* old = std::getenv(name);
    if (old) {
      saved_ = old;
      had_value_ = true;
    } else {
      had_value_ = false;
    }
    setenv(name, value, 1);
  }

  ~EnvGuard() {
    if (had_value_) {
      setenv(name_.c_str(), saved_.c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

 private:
  std::string name_;
  std::string saved_;
  bool had_value_;
};

//
// kpack_discover_binary_path tests
//

TEST(LoaderAPITest, DiscoverBinaryPath_NullAddress) {
  char path[1024];
  kpack_error_t err =
      kpack_discover_binary_path(nullptr, path, sizeof(path), nullptr);
  EXPECT_EQ(err, KPACK_ERROR_INVALID_ARGUMENT);
}

TEST(LoaderAPITest, DiscoverBinaryPath_NullPathOut) {
  // Use address of this test function as a valid address
  const void* addr = reinterpret_cast<const void*>(&get_test_assets_dir);
  kpack_error_t err = kpack_discover_binary_path(addr, nullptr, 1024, nullptr);
  EXPECT_EQ(err, KPACK_ERROR_INVALID_ARGUMENT);
}

TEST(LoaderAPITest, DiscoverBinaryPath_ZeroSize) {
  const void* addr = reinterpret_cast<const void*>(&get_test_assets_dir);
  char path[1024];
  kpack_error_t err = kpack_discover_binary_path(addr, path, 0, nullptr);
  EXPECT_EQ(err, KPACK_ERROR_INVALID_ARGUMENT);
}

TEST(LoaderAPITest, DiscoverBinaryPath_ValidAddress) {
  // Use address of a function in this test binary
  const void* addr = reinterpret_cast<const void*>(&get_test_assets_dir);
  char path[1024];
  size_t offset = 0;

  kpack_error_t err =
      kpack_discover_binary_path(addr, path, sizeof(path), &offset);

  // This should succeed on Linux
#ifdef __linux__
  EXPECT_EQ(err, KPACK_SUCCESS);
  // Path should be non-empty and contain the test executable name
  EXPECT_GT(std::strlen(path), 0u);
  // The path should end with the test executable name
  std::string path_str(path);
  EXPECT_NE(path_str.find("test_kpack_runtime"), std::string::npos);
#else
  // On Windows, we expect NOT_IMPLEMENTED until Windows support is added
  EXPECT_EQ(err, KPACK_ERROR_NOT_IMPLEMENTED);
#endif
}

TEST(LoaderAPITest, DiscoverBinaryPath_BufferTooSmall) {
  const void* addr = reinterpret_cast<const void*>(&get_test_assets_dir);
  char path[4];  // Too small for any real path

  kpack_error_t err =
      kpack_discover_binary_path(addr, path, sizeof(path), nullptr);

#ifdef __linux__
  // Should fail because buffer is too small
  EXPECT_EQ(err, KPACK_ERROR_INVALID_ARGUMENT);
#else
  EXPECT_EQ(err, KPACK_ERROR_NOT_IMPLEMENTED);
#endif
}

//
// kpack_cache_create / kpack_cache_destroy tests
//

TEST(LoaderAPITest, CacheCreate_Success) {
  CacheGuard cache;
  kpack_error_t err = kpack_cache_create(cache.ptr());
  EXPECT_EQ(err, KPACK_SUCCESS);
  EXPECT_NE(cache.get(), nullptr);
}

TEST(LoaderAPITest, CacheCreate_NullPointer) {
  kpack_error_t err = kpack_cache_create(nullptr);
  EXPECT_EQ(err, KPACK_ERROR_INVALID_ARGUMENT);
}

TEST(LoaderAPITest, CacheDestroy_Null) {
  // Should not crash when passed NULL
  kpack_cache_destroy(nullptr);
}

TEST(LoaderAPITest, CacheCreate_ResolvesEnvVars) {
  // Set environment variables before creating cache
  EnvGuard path_guard("ROCM_KPACK_PATH", "/test/path1:/test/path2");
  EnvGuard prefix_guard("ROCM_KPACK_PATH_PREFIX", "/prefix/path");

  CacheGuard cache;
  kpack_error_t err = kpack_cache_create(cache.ptr());
  EXPECT_EQ(err, KPACK_SUCCESS);
  ASSERT_NE(cache.get(), nullptr);

  // kpack_cache struct is defined in kpack_internal.h
  kpack_cache* internal = cache.get();

  // Verify ROCM_KPACK_PATH was split correctly
  ASSERT_EQ(internal->env_path_override.size(), 2u);
  EXPECT_EQ(internal->env_path_override[0], "/test/path1");
  EXPECT_EQ(internal->env_path_override[1], "/test/path2");

  // Verify ROCM_KPACK_PATH_PREFIX was split correctly
  ASSERT_EQ(internal->env_path_prefix.size(), 1u);
  EXPECT_EQ(internal->env_path_prefix[0], "/prefix/path");
}

//
// kpack_load_code_object tests
//

TEST(LoaderAPITest, LoadCodeObject_NullCache) {
  const char* arch_list[] = {"gfx1100"};
  void* code_object = nullptr;
  size_t size = 0;

  kpack_error_t err =
      kpack_load_code_object(nullptr, "test", "/nonexistent/binary.so", 0,
                             arch_list, 1, &code_object, &size);
  EXPECT_EQ(err, KPACK_ERROR_INVALID_ARGUMENT);
}

TEST(LoaderAPITest, LoadCodeObject_NullMetadata) {
  CacheGuard cache;
  ASSERT_EQ(kpack_cache_create(cache.ptr()), KPACK_SUCCESS);

  const char* arch_list[] = {"gfx1100"};
  void* code_object = nullptr;
  size_t size = 0;

  kpack_error_t err =
      kpack_load_code_object(cache.get(), nullptr, "/nonexistent/binary.so", 0,
                             arch_list, 1, &code_object, &size);
  EXPECT_EQ(err, KPACK_ERROR_INVALID_ARGUMENT);
}

TEST(LoaderAPITest, LoadCodeObject_NullBinaryPath) {
  CacheGuard cache;
  ASSERT_EQ(kpack_cache_create(cache.ptr()), KPACK_SUCCESS);

  const char metadata[] = "test";
  const char* arch_list[] = {"gfx1100"};
  void* code_object = nullptr;
  size_t size = 0;

  kpack_error_t err = kpack_load_code_object(cache.get(), metadata, nullptr, 0,
                                             arch_list, 1, &code_object, &size);
  EXPECT_EQ(err, KPACK_ERROR_INVALID_ARGUMENT);
}

TEST(LoaderAPITest, LoadCodeObject_NullArchList) {
  CacheGuard cache;
  ASSERT_EQ(kpack_cache_create(cache.ptr()), KPACK_SUCCESS);

  const char metadata[] = "test";
  void* code_object = nullptr;
  size_t size = 0;

  kpack_error_t err =
      kpack_load_code_object(cache.get(), metadata, "/nonexistent/binary.so", 0,
                             nullptr, 1, &code_object, &size);
  EXPECT_EQ(err, KPACK_ERROR_INVALID_ARGUMENT);
}

TEST(LoaderAPITest, LoadCodeObject_ZeroArchCount) {
  CacheGuard cache;
  ASSERT_EQ(kpack_cache_create(cache.ptr()), KPACK_SUCCESS);

  const char metadata[] = "test";
  const char* arch_list[] = {"gfx1100"};
  void* code_object = nullptr;
  size_t size = 0;

  kpack_error_t err =
      kpack_load_code_object(cache.get(), metadata, "/nonexistent/binary.so", 0,
                             arch_list, 0, &code_object, &size);
  EXPECT_EQ(err, KPACK_ERROR_INVALID_ARGUMENT);
}

TEST(LoaderAPITest, LoadCodeObject_NullCodeObjectOut) {
  CacheGuard cache;
  ASSERT_EQ(kpack_cache_create(cache.ptr()), KPACK_SUCCESS);

  const char metadata[] = "test";
  const char* arch_list[] = {"gfx1100"};
  size_t size = 0;

  kpack_error_t err =
      kpack_load_code_object(cache.get(), metadata, "/nonexistent/binary.so", 0,
                             arch_list, 1, nullptr, &size);
  EXPECT_EQ(err, KPACK_ERROR_INVALID_ARGUMENT);
}

TEST(LoaderAPITest, LoadCodeObject_NullSizeOut) {
  CacheGuard cache;
  ASSERT_EQ(kpack_cache_create(cache.ptr()), KPACK_SUCCESS);

  const char metadata[] = "test";
  const char* arch_list[] = {"gfx1100"};
  void* code_object = nullptr;

  kpack_error_t err =
      kpack_load_code_object(cache.get(), metadata, "/nonexistent/binary.so", 0,
                             arch_list, 1, &code_object, nullptr);
  EXPECT_EQ(err, KPACK_ERROR_INVALID_ARGUMENT);
}

TEST(LoaderAPITest, LoadCodeObject_InvalidMetadata) {
  CacheGuard cache;
  ASSERT_EQ(kpack_cache_create(cache.ptr()), KPACK_SUCCESS);

  // Random bytes that are not valid msgpack
  const char metadata[] = "this is not valid msgpack data!";
  const char* arch_list[] = {"gfx1100"};
  void* code_object = nullptr;
  size_t size = 0;

  kpack_error_t err =
      kpack_load_code_object(cache.get(), metadata, "/nonexistent/binary.so", 0,
                             arch_list, 1, &code_object, &size);
  EXPECT_EQ(err, KPACK_ERROR_INVALID_METADATA);
}

//
// kpack_free_code_object tests
//

TEST(LoaderAPITest, FreeCodeObject_Null) {
  // Should not crash when passed NULL
  kpack_free_code_object(nullptr);
}

//
// kpack_enumerate_architectures tests
//

TEST(LoaderAPITest, EnumerateArchitectures_NullPath) {
  auto callback = [](const char*, void*) -> bool { return true; };
  kpack_error_t err = kpack_enumerate_architectures(nullptr, callback, nullptr);
  EXPECT_EQ(err, KPACK_ERROR_INVALID_ARGUMENT);
}

TEST(LoaderAPITest, EnumerateArchitectures_NullCallback) {
  kpack_error_t err = kpack_enumerate_architectures(
      "/nonexistent/binary.so.kpack", nullptr, nullptr);
  EXPECT_EQ(err, KPACK_ERROR_INVALID_ARGUMENT);
}

TEST(LoaderAPITest, EnumerateArchitectures_FileNotFound) {
  auto callback = [](const char*, void*) -> bool { return true; };
  kpack_error_t err = kpack_enumerate_architectures("/nonexistent/test.kpack",
                                                    callback, nullptr);
  EXPECT_EQ(err, KPACK_ERROR_FILE_NOT_FOUND);
}

// Callback that collects architectures into a vector
struct ArchCollector {
  std::vector<std::string> archs;

  static bool collect(const char* arch, void* user_data) {
    auto* self = static_cast<ArchCollector*>(user_data);
    self->archs.emplace_back(arch);
    return true;  // continue
  }

  static bool stop_after_one(const char* arch, void* user_data) {
    auto* self = static_cast<ArchCollector*>(user_data);
    self->archs.emplace_back(arch);
    return false;  // stop after first
  }
};

TEST(LoaderAPITest, EnumerateArchitectures_ValidArchive) {
  std::string assets_dir = get_test_assets_dir();
  if (assets_dir.empty()) {
    GTEST_SKIP() << "ROCM_KPACK_TEST_ASSETS_DIR not set";
  }

  std::string archive_path = assets_dir + "/test_noop.kpack";

  ArchCollector collector;
  kpack_error_t err = kpack_enumerate_architectures(
      archive_path.c_str(), ArchCollector::collect, &collector);

  EXPECT_EQ(err, KPACK_SUCCESS);
  // The test archive should have at least one architecture
  EXPECT_GT(collector.archs.size(), 0u);
}

TEST(LoaderAPITest, EnumerateArchitectures_EarlyTermination) {
  std::string assets_dir = get_test_assets_dir();
  if (assets_dir.empty()) {
    GTEST_SKIP() << "ROCM_KPACK_TEST_ASSETS_DIR not set";
  }

  std::string archive_path = assets_dir + "/test_noop.kpack";

  // First, count total architectures
  ArchCollector full_collector;
  kpack_error_t err = kpack_enumerate_architectures(
      archive_path.c_str(), ArchCollector::collect, &full_collector);
  ASSERT_EQ(err, KPACK_SUCCESS);

  if (full_collector.archs.size() <= 1) {
    GTEST_SKIP()
        << "Archive has only one architecture, cannot test early termination";
  }

  // Now test early termination
  ArchCollector stop_collector;
  err = kpack_enumerate_architectures(
      archive_path.c_str(), ArchCollector::stop_after_one, &stop_collector);

  EXPECT_EQ(err, KPACK_SUCCESS);
  EXPECT_EQ(stop_collector.archs.size(), 1u);
}

//
// Environment variable tests
//

TEST(LoaderAPITest, LoadCodeObject_DisabledViaEnv) {
  // Set ROCM_KPACK_DISABLE before creating cache
  EnvGuard env_guard("ROCM_KPACK_DISABLE", "1");

  CacheGuard cache;
  ASSERT_EQ(kpack_cache_create(cache.ptr()), KPACK_SUCCESS);

  // Create minimal valid-looking msgpack that would parse
  // (but we should hit the disable check first)
  const char metadata[] = "test";
  const char* arch_list[] = {"gfx1100"};
  void* code_object = nullptr;
  size_t size = 0;

  kpack_error_t err =
      kpack_load_code_object(cache.get(), metadata, "/nonexistent/binary.so", 0,
                             arch_list, 1, &code_object, &size);
  EXPECT_EQ(err, KPACK_ERROR_NOT_IMPLEMENTED);
}

TEST(LoaderAPITest, CacheReusesArchives) {
  std::string assets_dir = get_test_assets_dir();
  if (assets_dir.empty()) {
    GTEST_SKIP() << "ROCM_KPACK_TEST_ASSETS_DIR not set";
  }

  CacheGuard cache;
  ASSERT_EQ(kpack_cache_create(cache.ptr()), KPACK_SUCCESS);
  ASSERT_NE(cache.get(), nullptr);

  // kpack_cache struct is defined in kpack_internal.h
  kpack_cache* internal = cache.get();

  // Initially no archives are cached
  EXPECT_EQ(internal->archives.size(), 0u);
}

//
// Helper to create HIPK-format msgpack metadata
//
// HIPK metadata structure (msgpack map):
// {
//   "kernel_name": "<binary_name>",  // binary name in kpack TOC
//   "kpack_search_paths": ["path1", "path2", ...]  // relative to binary_path
// }
//
static std::vector<char> make_hipk_metadata(
    const std::string& kernel_name,
    const std::vector<std::string>& search_paths) {
  msgpack::sbuffer sbuf;
  msgpack::packer<msgpack::sbuffer> pk(&sbuf);
  pk.pack_map(2);
  pk.pack("kernel_name");
  pk.pack(kernel_name);
  pk.pack("kpack_search_paths");
  pk.pack(search_paths);
  return std::vector<char>(sbuf.data(), sbuf.data() + sbuf.size());
}

//
// kpack_load_code_object integration tests
//

TEST(LoaderAPITest, LoadCodeObject_FromNoopArchive) {
  std::string assets_dir = get_test_assets_dir();
  if (assets_dir.empty()) {
    GTEST_SKIP() << "ROCM_KPACK_TEST_ASSETS_DIR not set";
  }

  CacheGuard cache;
  ASSERT_EQ(kpack_cache_create(cache.ptr()), KPACK_SUCCESS);

  // Create HIPK metadata pointing directly to test_noop.kpack archive.
  // The archive contains gfx900 and gfx906 architectures.
  auto metadata = make_hipk_metadata("lib/libtest.so", {"test_noop.kpack"});

  // binary_path is used to resolve relative search paths
  std::string binary_path = assets_dir + "/fake_binary.so";

  const char* arch_list[] = {"gfx900"};
  void* code_object = nullptr;
  size_t size = 0;

  kpack_error_t err =
      kpack_load_code_object(cache.get(), metadata.data(), binary_path.c_str(),
                             0, arch_list, 1, &code_object, &size);

  ASSERT_EQ(err, KPACK_SUCCESS);
  EXPECT_GT(size, 0u);
  EXPECT_NE(code_object, nullptr);

  // Verify the kernel content matches what we expect from the test archive
  // test_noop.kpack has kernel starting with "KERNEL1_GFX900_DATA"
  EXPECT_EQ(std::memcmp(code_object, "KERNEL1_GFX900_DATA", 19), 0);

  kpack_free_code_object(code_object);
}

TEST(LoaderAPITest, LoadCodeObject_FromZstdArchive) {
  std::string assets_dir = get_test_assets_dir();
  if (assets_dir.empty()) {
    GTEST_SKIP() << "ROCM_KPACK_TEST_ASSETS_DIR not set";
  }

  CacheGuard cache;
  ASSERT_EQ(kpack_cache_create(cache.ptr()), KPACK_SUCCESS);

  // Create HIPK metadata pointing directly to test_zstd.kpack archive.
  // The archive contains gfx1100 and gfx1101 architectures.
  auto metadata = make_hipk_metadata("lib/libhip.so", {"test_zstd.kpack"});

  std::string binary_path = assets_dir + "/fake_binary.so";

  const char* arch_list[] = {"gfx1100"};
  void* code_object = nullptr;
  size_t size = 0;

  kpack_error_t err =
      kpack_load_code_object(cache.get(), metadata.data(), binary_path.c_str(),
                             0, arch_list, 1, &code_object, &size);

  ASSERT_EQ(err, KPACK_SUCCESS);
  EXPECT_GT(size, 0u);
  EXPECT_NE(code_object, nullptr);

  // Verify the kernel content matches expected from zstd archive
  EXPECT_EQ(std::memcmp(code_object, "HIP_KERNEL_GFX1100_", 19), 0);

  kpack_free_code_object(code_object);
}

TEST(LoaderAPITest, LoadCodeObject_ArchitecturePriority) {
  std::string assets_dir = get_test_assets_dir();
  if (assets_dir.empty()) {
    GTEST_SKIP() << "ROCM_KPACK_TEST_ASSETS_DIR not set";
  }

  CacheGuard cache;
  ASSERT_EQ(kpack_cache_create(cache.ptr()), KPACK_SUCCESS);

  // Request [gfx906, gfx900] - should get gfx906 (first match)
  auto metadata = make_hipk_metadata("lib/libtest.so", {"test_noop.kpack"});
  std::string binary_path = assets_dir + "/fake_binary.so";

  const char* arch_list[] = {"gfx906", "gfx900"};
  void* code_object = nullptr;
  size_t size = 0;

  kpack_error_t err =
      kpack_load_code_object(cache.get(), metadata.data(), binary_path.c_str(),
                             0, arch_list, 2, &code_object, &size);

  ASSERT_EQ(err, KPACK_SUCCESS);
  EXPECT_NE(code_object, nullptr);

  // Should get gfx906 kernel since it's first in priority list
  EXPECT_EQ(std::memcmp(code_object, "KERNEL2_GFX906_DATA", 19), 0);

  kpack_free_code_object(code_object);
}

TEST(LoaderAPITest, LoadCodeObject_ArchNotFound) {
  std::string assets_dir = get_test_assets_dir();
  if (assets_dir.empty()) {
    GTEST_SKIP() << "ROCM_KPACK_TEST_ASSETS_DIR not set";
  }

  CacheGuard cache;
  ASSERT_EQ(kpack_cache_create(cache.ptr()), KPACK_SUCCESS);

  auto metadata = make_hipk_metadata("lib/libtest.so", {"test_noop.kpack"});
  std::string binary_path = assets_dir + "/fake_binary.so";

  // Request architecture that doesn't exist in the archive.
  const char* arch_list[] = {"gfx9999"};
  void* code_object = nullptr;
  size_t size = 0;

  kpack_error_t err =
      kpack_load_code_object(cache.get(), metadata.data(), binary_path.c_str(),
                             0, arch_list, 1, &code_object, &size);

  EXPECT_EQ(err, KPACK_ERROR_ARCH_NOT_FOUND);
  EXPECT_EQ(code_object, nullptr);
}

TEST(LoaderAPITest, LoadCodeObject_ArchiveNotFound) {
  std::string assets_dir = get_test_assets_dir();
  if (assets_dir.empty()) {
    GTEST_SKIP() << "ROCM_KPACK_TEST_ASSETS_DIR not set";
  }

  CacheGuard cache;
  ASSERT_EQ(kpack_cache_create(cache.ptr()), KPACK_SUCCESS);

  // Point to non-existent archive
  auto metadata = make_hipk_metadata("lib/libtest.so", {"nonexistent.kpack"});
  std::string binary_path = assets_dir + "/fake_binary.so";

  const char* arch_list[] = {"gfx900"};
  void* code_object = nullptr;
  size_t size = 0;

  kpack_error_t err =
      kpack_load_code_object(cache.get(), metadata.data(), binary_path.c_str(),
                             0, arch_list, 1, &code_object, &size);

  EXPECT_EQ(err, KPACK_ERROR_ARCHIVE_NOT_FOUND);
  EXPECT_EQ(code_object, nullptr);
}

TEST(LoaderAPITest, LoadCodeObject_CacheReusesArchive) {
  std::string assets_dir = get_test_assets_dir();
  if (assets_dir.empty()) {
    GTEST_SKIP() << "ROCM_KPACK_TEST_ASSETS_DIR not set";
  }

  CacheGuard cache;
  ASSERT_EQ(kpack_cache_create(cache.ptr()), KPACK_SUCCESS);

  auto metadata = make_hipk_metadata("lib/libtest.so", {"test_noop.kpack"});
  std::string binary_path = assets_dir + "/fake_binary.so";

  // First load
  const char* arch_list1[] = {"gfx900"};
  void* code_object1 = nullptr;
  size_t size1 = 0;

  kpack_error_t err =
      kpack_load_code_object(cache.get(), metadata.data(), binary_path.c_str(),
                             0, arch_list1, 1, &code_object1, &size1);
  ASSERT_EQ(err, KPACK_SUCCESS);

  // Check cache has one archive
  kpack_cache* internal = cache.get();
  EXPECT_EQ(internal->archives.size(), 1u);

  // Second load - same archive, different arch
  const char* arch_list2[] = {"gfx906"};
  void* code_object2 = nullptr;
  size_t size2 = 0;

  err =
      kpack_load_code_object(cache.get(), metadata.data(), binary_path.c_str(),
                             0, arch_list2, 1, &code_object2, &size2);
  ASSERT_EQ(err, KPACK_SUCCESS);

  // Cache should still have only one archive (reused)
  EXPECT_EQ(internal->archives.size(), 1u);

  // Verify both kernels are valid and different
  EXPECT_EQ(std::memcmp(code_object1, "KERNEL1_GFX900_DATA", 19), 0);
  EXPECT_EQ(std::memcmp(code_object2, "KERNEL2_GFX906_DATA", 19), 0);

  kpack_free_code_object(code_object1);
  kpack_free_code_object(code_object2);
}

TEST(LoaderAPITest, LoadCodeObject_EnvPathOverride) {
  std::string assets_dir = get_test_assets_dir();
  if (assets_dir.empty()) {
    GTEST_SKIP() << "ROCM_KPACK_TEST_ASSETS_DIR not set";
  }

  // Set ROCM_KPACK_PATH to full path to kpack file (overrides search paths)
  std::string kpack_path = assets_dir + "/test_noop.kpack";
  EnvGuard env_guard("ROCM_KPACK_PATH", kpack_path.c_str());

  CacheGuard cache;
  ASSERT_EQ(kpack_cache_create(cache.ptr()), KPACK_SUCCESS);

  // Metadata points to different path, but env override will be used instead
  auto metadata = make_hipk_metadata("lib/libtest.so", {"wrong_path.kpack"});

  // binary_path doesn't matter since env override takes precedence
  std::string binary_path = "/some/other/path/binary.so";

  const char* arch_list[] = {"gfx900"};
  void* code_object = nullptr;
  size_t size = 0;

  kpack_error_t err =
      kpack_load_code_object(cache.get(), metadata.data(), binary_path.c_str(),
                             0, arch_list, 1, &code_object, &size);

  ASSERT_EQ(err, KPACK_SUCCESS);
  EXPECT_NE(code_object, nullptr);
  EXPECT_EQ(std::memcmp(code_object, "KERNEL1_GFX900_DATA", 19), 0);

  kpack_free_code_object(code_object);
}

//
// Target triple stripping tests
//

TEST(LoaderAPITest, LoadCodeObject_TargetTripleStripping) {
  std::string assets_dir = get_test_assets_dir();
  if (assets_dir.empty()) {
    GTEST_SKIP() << "ROCM_KPACK_TEST_ASSETS_DIR not set";
  }

  CacheGuard cache;
  ASSERT_EQ(kpack_cache_create(cache.ptr()), KPACK_SUCCESS);

  auto metadata = make_hipk_metadata("lib/libtest.so", {"test_noop.kpack"});
  std::string binary_path = assets_dir + "/fake_binary.so";

  // Use full target triple with amdgcn-amd-amdhsa-- prefix.
  // The loader should strip this to "gfx900" for archive architecture matching.
  const char* arch_list[] = {"amdgcn-amd-amdhsa--gfx900"};
  void* code_object = nullptr;
  size_t size = 0;

  kpack_error_t err =
      kpack_load_code_object(cache.get(), metadata.data(), binary_path.c_str(),
                             0, arch_list, 1, &code_object, &size);

  ASSERT_EQ(err, KPACK_SUCCESS);
  EXPECT_NE(code_object, nullptr);
  EXPECT_EQ(std::memcmp(code_object, "KERNEL1_GFX900_DATA", 19), 0);

  kpack_free_code_object(code_object);
}

TEST(LoaderAPITest, LoadCodeObject_ArchNotInArchive) {
  std::string assets_dir = get_test_assets_dir();
  if (assets_dir.empty()) {
    GTEST_SKIP() << "ROCM_KPACK_TEST_ASSETS_DIR not set";
  }

  CacheGuard cache;
  ASSERT_EQ(kpack_cache_create(cache.ptr()), KPACK_SUCCESS);

  // test_zstd.kpack only contains gfx1100 and gfx1101
  auto metadata = make_hipk_metadata("lib/libtest.so", {"test_zstd.kpack"});
  std::string binary_path = assets_dir + "/fake_binary.so";

  // Request gfx900 which is NOT in this archive
  const char* arch_list[] = {"gfx900"};
  void* code_object = nullptr;
  size_t size = 0;

  kpack_error_t err =
      kpack_load_code_object(cache.get(), metadata.data(), binary_path.c_str(),
                             0, arch_list, 1, &code_object, &size);

  EXPECT_EQ(err, KPACK_ERROR_ARCH_NOT_FOUND);
  EXPECT_EQ(code_object, nullptr);
}

TEST(LoaderAPITest, LoadCodeObject_KernelNotFound) {
  std::string assets_dir = get_test_assets_dir();
  if (assets_dir.empty()) {
    GTEST_SKIP() << "ROCM_KPACK_TEST_ASSETS_DIR not set";
  }

  CacheGuard cache;
  ASSERT_EQ(kpack_cache_create(cache.ptr()), KPACK_SUCCESS);

  // test_noop.kpack contains gfx900, and the kernel "lib/libtest.so#0".
  // Request co_index=999 — the archive will be found and the architecture
  // will match, but "lib/libtest.so#999" is not in the TOC.
  auto metadata = make_hipk_metadata("lib/libtest.so", {"test_noop.kpack"});
  std::string binary_path = assets_dir + "/fake_binary.so";

  const char* arch_list[] = {"gfx900"};
  void* code_object = nullptr;
  size_t size = 0;

  kpack_error_t err =
      kpack_load_code_object(cache.get(), metadata.data(), binary_path.c_str(),
                             999, arch_list, 1, &code_object, &size);

  EXPECT_EQ(err, KPACK_ERROR_KERNEL_NOT_FOUND);
  EXPECT_EQ(code_object, nullptr);
}

TEST(LoaderAPITest, LoadCodeObject_ArchiveFileNotFound) {
  std::string assets_dir = get_test_assets_dir();
  if (assets_dir.empty()) {
    GTEST_SKIP() << "ROCM_KPACK_TEST_ASSETS_DIR not set";
  }

  CacheGuard cache;
  ASSERT_EQ(kpack_cache_create(cache.ptr()), KPACK_SUCCESS);

  // Point to an archive file that doesn't exist
  auto metadata =
      make_hipk_metadata("lib/libtest.so", {"does_not_exist.kpack"});
  std::string binary_path = assets_dir + "/fake_binary.so";

  const char* arch_list[] = {"gfx900"};
  void* code_object = nullptr;
  size_t size = 0;

  kpack_error_t err =
      kpack_load_code_object(cache.get(), metadata.data(), binary_path.c_str(),
                             0, arch_list, 1, &code_object, &size);

  EXPECT_EQ(err, KPACK_ERROR_ARCHIVE_NOT_FOUND);
  EXPECT_EQ(code_object, nullptr);
}

//
// Thread safety tests
//
// The API claims:
// - kpack_get_kernel(): "Thread-safe when called concurrently on SAME archive"
// - kpack_load_code_object(): "Thread-safe with same cache from multiple
// threads"
//

TEST(LoaderAPITest, ThreadSafety_ConcurrentLoadCodeObject) {
  std::string assets_dir = get_test_assets_dir();
  if (assets_dir.empty()) {
    GTEST_SKIP() << "ROCM_KPACK_TEST_ASSETS_DIR not set";
  }

  CacheGuard cache;
  ASSERT_EQ(kpack_cache_create(cache.ptr()), KPACK_SUCCESS);

  auto metadata = make_hipk_metadata("lib/libtest.so", {"test_noop.kpack"});
  std::string binary_path = assets_dir + "/fake_binary.so";

  constexpr int kNumThreads = 8;
  constexpr int kIterationsPerThread = 50;

  std::atomic<int> success_count{0};
  std::atomic<int> failure_count{0};

  auto worker = [&](int thread_id) {
    // Each thread alternates between gfx900 and gfx906
    const char* arch = (thread_id % 2 == 0) ? "gfx900" : "gfx906";
    const char* expected_prefix =
        (thread_id % 2 == 0) ? "KERNEL1_GFX900_DATA" : "KERNEL2_GFX906_DATA";

    for (int i = 0; i < kIterationsPerThread; ++i) {
      const char* arch_list[] = {arch};
      void* code_object = nullptr;
      size_t size = 0;

      kpack_error_t err = kpack_load_code_object(
          cache.get(), metadata.data(), binary_path.c_str(), 0, arch_list, 1,
          &code_object, &size);

      if (err == KPACK_SUCCESS && code_object != nullptr &&
          std::memcmp(code_object, expected_prefix, 19) == 0) {
        success_count++;
      } else {
        failure_count++;
      }

      if (code_object) {
        kpack_free_code_object(code_object);
      }
    }
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back(worker, i);
  }

  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(success_count.load(), kNumThreads * kIterationsPerThread);
  EXPECT_EQ(failure_count.load(), 0);
}

TEST(LoaderAPITest, ThreadSafety_ConcurrentArchiveCaching) {
  std::string assets_dir = get_test_assets_dir();
  if (assets_dir.empty()) {
    GTEST_SKIP() << "ROCM_KPACK_TEST_ASSETS_DIR not set";
  }

  CacheGuard cache;
  ASSERT_EQ(kpack_cache_create(cache.ptr()), KPACK_SUCCESS);

  // Multiple threads try to load from same archive simultaneously
  // First call should cache, others should reuse
  auto metadata = make_hipk_metadata("lib/libtest.so", {"test_noop.kpack"});
  std::string binary_path = assets_dir + "/fake_binary.so";

  constexpr int kNumThreads = 10;
  std::atomic<int> success_count{0};

  auto worker = [&]() {
    const char* arch_list[] = {"gfx900"};
    void* code_object = nullptr;
    size_t size = 0;

    kpack_error_t err = kpack_load_code_object(
        cache.get(), metadata.data(), binary_path.c_str(), 0, arch_list, 1,
        &code_object, &size);

    if (err == KPACK_SUCCESS) {
      success_count++;
      kpack_free_code_object(code_object);
    }
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back(worker);
  }

  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(success_count.load(), kNumThreads);

  // Verify archive was cached (only one entry, not N)
  kpack_cache* internal = cache.get();
  EXPECT_EQ(internal->archives.size(), 1u);
}

// Test concurrent kpack_get_kernel() calls on same archive
TEST(ArchiveIntegrationTest, ThreadSafety_ConcurrentGetKernel) {
  std::string assets_dir = get_test_assets_dir();
  if (assets_dir.empty()) {
    GTEST_SKIP() << "ROCM_KPACK_TEST_ASSETS_DIR not set";
  }

  std::string test_kpack = assets_dir + "/test_noop.kpack";

  kpack_archive_t archive;
  kpack_error_t err = kpack_open(test_kpack.c_str(), &archive);
  ASSERT_EQ(err, KPACK_SUCCESS);

  constexpr int kNumThreads = 8;
  constexpr int kIterationsPerThread = 50;

  std::atomic<int> success_count{0};
  std::atomic<int> failure_count{0};

  auto worker = [&](int thread_id) {
    // Each thread alternates between different binary/arch combinations
    const char* binary =
        (thread_id % 2 == 0) ? "lib/libtest.so#0" : "bin/testapp#0";
    const char* arch = "gfx900";
    const char* expected_prefix =
        (thread_id % 2 == 0) ? "KERNEL1_GFX900_DATA" : "KERNEL3_APP_GFX900";

    for (int i = 0; i < kIterationsPerThread; ++i) {
      void* kernel_data = nullptr;
      size_t kernel_size = 0;

      kpack_error_t result =
          kpack_get_kernel(archive, binary, arch, &kernel_data, &kernel_size);

      if (result == KPACK_SUCCESS && kernel_data != nullptr &&
          std::memcmp(kernel_data, expected_prefix,
                      std::strlen(expected_prefix)) == 0) {
        success_count++;
      } else {
        failure_count++;
      }

      if (kernel_data) {
        kpack_free_kernel(kernel_data);
      }
    }
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back(worker, i);
  }

  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(success_count.load(), kNumThreads * kIterationsPerThread);
  EXPECT_EQ(failure_count.load(), 0);

  kpack_close(archive);
}

//
// HIPK metadata parsing edge case tests
//

TEST(LoaderAPITest, HIPKMetadata_MissingKernelName) {
  std::string assets_dir = get_test_assets_dir();
  if (assets_dir.empty()) {
    GTEST_SKIP() << "ROCM_KPACK_TEST_ASSETS_DIR not set";
  }

  CacheGuard cache;
  ASSERT_EQ(kpack_cache_create(cache.ptr()), KPACK_SUCCESS);

  // Create metadata without kernel_name field
  msgpack::sbuffer sbuf;
  msgpack::packer<msgpack::sbuffer> pk(&sbuf);
  pk.pack_map(1);
  pk.pack("kpack_search_paths");
  pk.pack(std::vector<std::string>{"test_noop.kpack"});

  std::string binary_path = assets_dir + "/fake_binary.so";
  const char* arch_list[] = {"gfx900"};
  void* code_object = nullptr;
  size_t size = 0;

  kpack_error_t err =
      kpack_load_code_object(cache.get(), sbuf.data(), binary_path.c_str(), 0,
                             arch_list, 1, &code_object, &size);

  EXPECT_EQ(err, KPACK_ERROR_INVALID_METADATA);
  EXPECT_EQ(code_object, nullptr);
}

TEST(LoaderAPITest, HIPKMetadata_MissingSearchPaths) {
  std::string assets_dir = get_test_assets_dir();
  if (assets_dir.empty()) {
    GTEST_SKIP() << "ROCM_KPACK_TEST_ASSETS_DIR not set";
  }

  CacheGuard cache;
  ASSERT_EQ(kpack_cache_create(cache.ptr()), KPACK_SUCCESS);

  // Create metadata without kpack_search_paths field
  msgpack::sbuffer sbuf;
  msgpack::packer<msgpack::sbuffer> pk(&sbuf);
  pk.pack_map(1);
  pk.pack("kernel_name");
  pk.pack("lib/libtest.so");

  std::string binary_path = assets_dir + "/fake_binary.so";
  const char* arch_list[] = {"gfx900"};
  void* code_object = nullptr;
  size_t size = 0;

  kpack_error_t err =
      kpack_load_code_object(cache.get(), sbuf.data(), binary_path.c_str(), 0,
                             arch_list, 1, &code_object, &size);

  EXPECT_EQ(err, KPACK_ERROR_INVALID_METADATA);
  EXPECT_EQ(code_object, nullptr);
}

TEST(LoaderAPITest, HIPKMetadata_EmptySearchPaths) {
  std::string assets_dir = get_test_assets_dir();
  if (assets_dir.empty()) {
    GTEST_SKIP() << "ROCM_KPACK_TEST_ASSETS_DIR not set";
  }

  CacheGuard cache;
  ASSERT_EQ(kpack_cache_create(cache.ptr()), KPACK_SUCCESS);

  // Create metadata with empty search paths array
  auto metadata = make_hipk_metadata("lib/libtest.so", {});

  std::string binary_path = assets_dir + "/fake_binary.so";
  const char* arch_list[] = {"gfx900"};
  void* code_object = nullptr;
  size_t size = 0;

  kpack_error_t err =
      kpack_load_code_object(cache.get(), metadata.data(), binary_path.c_str(),
                             0, arch_list, 1, &code_object, &size);

  EXPECT_EQ(err, KPACK_ERROR_INVALID_METADATA);
  EXPECT_EQ(code_object, nullptr);
}

TEST(LoaderAPITest, HIPKMetadata_WrongTypeKernelName) {
  std::string assets_dir = get_test_assets_dir();
  if (assets_dir.empty()) {
    GTEST_SKIP() << "ROCM_KPACK_TEST_ASSETS_DIR not set";
  }

  CacheGuard cache;
  ASSERT_EQ(kpack_cache_create(cache.ptr()), KPACK_SUCCESS);

  // Create metadata with kernel_name as int instead of string
  msgpack::sbuffer sbuf;
  msgpack::packer<msgpack::sbuffer> pk(&sbuf);
  pk.pack_map(2);
  pk.pack("kernel_name");
  pk.pack(12345);  // wrong type
  pk.pack("kpack_search_paths");
  pk.pack(std::vector<std::string>{"test_noop.kpack"});

  std::string binary_path = assets_dir + "/fake_binary.so";
  const char* arch_list[] = {"gfx900"};
  void* code_object = nullptr;
  size_t size = 0;

  kpack_error_t err =
      kpack_load_code_object(cache.get(), sbuf.data(), binary_path.c_str(), 0,
                             arch_list, 1, &code_object, &size);

  EXPECT_EQ(err, KPACK_ERROR_INVALID_METADATA);
  EXPECT_EQ(code_object, nullptr);
}

TEST(LoaderAPITest, HIPKMetadata_WrongTypeSearchPaths) {
  std::string assets_dir = get_test_assets_dir();
  if (assets_dir.empty()) {
    GTEST_SKIP() << "ROCM_KPACK_TEST_ASSETS_DIR not set";
  }

  CacheGuard cache;
  ASSERT_EQ(kpack_cache_create(cache.ptr()), KPACK_SUCCESS);

  // Create metadata with kpack_search_paths as string instead of array
  msgpack::sbuffer sbuf;
  msgpack::packer<msgpack::sbuffer> pk(&sbuf);
  pk.pack_map(2);
  pk.pack("kernel_name");
  pk.pack("lib/libtest.so");
  pk.pack("kpack_search_paths");
  pk.pack("test_noop.kpack");  // should be array

  std::string binary_path = assets_dir + "/fake_binary.so";
  const char* arch_list[] = {"gfx900"};
  void* code_object = nullptr;
  size_t size = 0;

  kpack_error_t err =
      kpack_load_code_object(cache.get(), sbuf.data(), binary_path.c_str(), 0,
                             arch_list, 1, &code_object, &size);

  EXPECT_EQ(err, KPACK_ERROR_INVALID_METADATA);
  EXPECT_EQ(code_object, nullptr);
}

TEST(LoaderAPITest, HIPKMetadata_NotAMap) {
  std::string assets_dir = get_test_assets_dir();
  if (assets_dir.empty()) {
    GTEST_SKIP() << "ROCM_KPACK_TEST_ASSETS_DIR not set";
  }

  CacheGuard cache;
  ASSERT_EQ(kpack_cache_create(cache.ptr()), KPACK_SUCCESS);

  // Create metadata that is an array instead of a map
  msgpack::sbuffer sbuf;
  msgpack::packer<msgpack::sbuffer> pk(&sbuf);
  pk.pack(std::vector<std::string>{"lib/libtest.so", "test_noop.kpack"});

  std::string binary_path = assets_dir + "/fake_binary.so";
  const char* arch_list[] = {"gfx900"};
  void* code_object = nullptr;
  size_t size = 0;

  kpack_error_t err =
      kpack_load_code_object(cache.get(), sbuf.data(), binary_path.c_str(), 0,
                             arch_list, 1, &code_object, &size);

  EXPECT_EQ(err, KPACK_ERROR_INVALID_METADATA);
  EXPECT_EQ(code_object, nullptr);
}

//
// Environment variable edge case tests
//

TEST(LoaderAPITest, EnvPath_EmptyComponents) {
  std::string assets_dir = get_test_assets_dir();
  if (assets_dir.empty()) {
    GTEST_SKIP() << "ROCM_KPACK_TEST_ASSETS_DIR not set";
  }

  // Path with empty components: "path1::path2" should ignore empty
  std::string kpack_path = assets_dir + "/test_noop.kpack";
  std::string path_with_empty = kpack_path + "::" + kpack_path;
  EnvGuard env_guard("ROCM_KPACK_PATH", path_with_empty.c_str());

  CacheGuard cache;
  ASSERT_EQ(kpack_cache_create(cache.ptr()), KPACK_SUCCESS);

  auto metadata = make_hipk_metadata("lib/libtest.so", {"wrong.kpack"});
  std::string binary_path = "/some/path/binary.so";

  const char* arch_list[] = {"gfx900"};
  void* code_object = nullptr;
  size_t size = 0;

  kpack_error_t err =
      kpack_load_code_object(cache.get(), metadata.data(), binary_path.c_str(),
                             0, arch_list, 1, &code_object, &size);

  ASSERT_EQ(err, KPACK_SUCCESS);
  EXPECT_NE(code_object, nullptr);
  kpack_free_code_object(code_object);
}

TEST(LoaderAPITest, EnvPath_TrailingColon) {
  std::string assets_dir = get_test_assets_dir();
  if (assets_dir.empty()) {
    GTEST_SKIP() << "ROCM_KPACK_TEST_ASSETS_DIR not set";
  }

  // Path with trailing colon should be parsed correctly
  std::string kpack_path = assets_dir + "/test_noop.kpack:";
  EnvGuard env_guard("ROCM_KPACK_PATH", kpack_path.c_str());

  CacheGuard cache;
  ASSERT_EQ(kpack_cache_create(cache.ptr()), KPACK_SUCCESS);

  auto metadata = make_hipk_metadata("lib/libtest.so", {"wrong.kpack"});
  std::string binary_path = "/some/path/binary.so";

  const char* arch_list[] = {"gfx900"};
  void* code_object = nullptr;
  size_t size = 0;

  kpack_error_t err =
      kpack_load_code_object(cache.get(), metadata.data(), binary_path.c_str(),
                             0, arch_list, 1, &code_object, &size);

  ASSERT_EQ(err, KPACK_SUCCESS);
  EXPECT_NE(code_object, nullptr);
  kpack_free_code_object(code_object);
}

TEST(LoaderAPITest, EnvDisable_WithZero) {
  // ROCM_KPACK_DISABLE="0" should be ENABLED (not disabled)
  EnvGuard env_guard("ROCM_KPACK_DISABLE", "0");

  CacheGuard cache;
  ASSERT_EQ(kpack_cache_create(cache.ptr()), KPACK_SUCCESS);

  // Verify disabled flag is NOT set
  kpack_cache* internal = cache.get();
  EXPECT_FALSE(internal->disabled);
}

TEST(LoaderAPITest, EnvDisable_WithEmpty) {
  // ROCM_KPACK_DISABLE="" should be ENABLED (not disabled)
  EnvGuard env_guard("ROCM_KPACK_DISABLE", "");

  CacheGuard cache;
  ASSERT_EQ(kpack_cache_create(cache.ptr()), KPACK_SUCCESS);

  // Verify disabled flag is NOT set
  kpack_cache* internal = cache.get();
  EXPECT_FALSE(internal->disabled);
}

TEST(LoaderAPITest, EnvPathPrefix_WithOverride) {
  std::string assets_dir = get_test_assets_dir();
  if (assets_dir.empty()) {
    GTEST_SKIP() << "ROCM_KPACK_TEST_ASSETS_DIR not set";
  }

  // When both PATH and PATH_PREFIX are set, PATH should take precedence
  std::string kpack_path = assets_dir + "/test_noop.kpack";
  EnvGuard path_guard("ROCM_KPACK_PATH", kpack_path.c_str());
  EnvGuard prefix_guard("ROCM_KPACK_PATH_PREFIX", "/should/be/ignored");

  CacheGuard cache;
  ASSERT_EQ(kpack_cache_create(cache.ptr()), KPACK_SUCCESS);

  // Verify env_path_override is set and prefix is stored but not used
  kpack_cache* internal = cache.get();
  EXPECT_EQ(internal->env_path_override.size(), 1u);
  EXPECT_EQ(internal->env_path_override[0], kpack_path);

  // Load should succeed using the override path
  auto metadata = make_hipk_metadata("lib/libtest.so", {"wrong.kpack"});
  std::string binary_path = "/some/path/binary.so";

  const char* arch_list[] = {"gfx900"};
  void* code_object = nullptr;
  size_t size = 0;

  kpack_error_t err =
      kpack_load_code_object(cache.get(), metadata.data(), binary_path.c_str(),
                             0, arch_list, 1, &code_object, &size);

  ASSERT_EQ(err, KPACK_SUCCESS);
  kpack_free_code_object(code_object);
}
