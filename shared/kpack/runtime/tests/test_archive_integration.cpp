// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstring>

#include "rocm_kpack/kpack.h"
#include "test_utils.h"

// Integration tests for full archive reading with generated test data

TEST(ArchiveIntegrationTest, NoOpArchive) {
  std::string test_kpack =
      test_utils::get_test_assets_dir() / "test_noop.kpack";
  ASSERT_TRUE(std::filesystem::exists(test_kpack))
      << "Test asset not found: " << test_kpack;

  // Open archive
  kpack_archive_t archive;
  kpack_error_t err = kpack_open(test_kpack.c_str(), &archive);
  ASSERT_EQ(err, KPACK_SUCCESS) << "Failed to open NoOp test archive";

  // Verify architectures
  size_t arch_count;
  err = kpack_get_architecture_count(archive, &arch_count);
  ASSERT_EQ(err, KPACK_SUCCESS);
  EXPECT_EQ(arch_count, 2);  // gfx900, gfx906

  const char* arch0;
  err = kpack_get_architecture(archive, 0, &arch0);
  ASSERT_EQ(err, KPACK_SUCCESS);
  EXPECT_STREQ(arch0, "gfx900");

  const char* arch1;
  err = kpack_get_architecture(archive, 1, &arch1);
  ASSERT_EQ(err, KPACK_SUCCESS);
  EXPECT_STREQ(arch1, "gfx906");

  // Verify binaries
  size_t binary_count;
  err = kpack_get_binary_count(archive, &binary_count);
  ASSERT_EQ(err, KPACK_SUCCESS);
  EXPECT_EQ(binary_count, 2);  // lib/libtest.so#0, bin/testapp#0

  // Load and verify kernel 1: lib/libtest.so#0 @ gfx900
  void* kernel_data;
  size_t kernel_size;
  err = kpack_get_kernel(archive, "lib/libtest.so#0", "gfx900", &kernel_data,
                         &kernel_size);
  ASSERT_EQ(err, KPACK_SUCCESS);
  EXPECT_EQ(kernel_size, 119);
  EXPECT_EQ(0, memcmp(kernel_data, "KERNEL1_GFX900_DATA", 19));
  kpack_free_kernel(kernel_data);

  // Load and verify kernel 2: lib/libtest.so#0 @ gfx906
  err = kpack_get_kernel(archive, "lib/libtest.so#0", "gfx906", &kernel_data,
                         &kernel_size);
  ASSERT_EQ(err, KPACK_SUCCESS);
  EXPECT_EQ(kernel_size, 219);
  EXPECT_EQ(0, memcmp(kernel_data, "KERNEL2_GFX906_DATA", 19));
  kpack_free_kernel(kernel_data);

  // Load and verify kernel 3: bin/testapp#0 @ gfx900
  err = kpack_get_kernel(archive, "bin/testapp#0", "gfx900", &kernel_data,
                         &kernel_size);
  ASSERT_EQ(err, KPACK_SUCCESS);
  EXPECT_EQ(kernel_size, 168);
  EXPECT_EQ(0, memcmp(kernel_data, "KERNEL3_APP_GFX900", 18));
  kpack_free_kernel(kernel_data);

  // Verify kernel not found
  err = kpack_get_kernel(archive, "nonexistent/binary", "gfx900", &kernel_data,
                         &kernel_size);
  EXPECT_EQ(err, KPACK_ERROR_KERNEL_NOT_FOUND);

  err = kpack_get_kernel(archive, "lib/libtest.so#0", "gfx908", &kernel_data,
                         &kernel_size);
  EXPECT_EQ(err, KPACK_ERROR_KERNEL_NOT_FOUND);

  kpack_close(archive);
}

TEST(ArchiveIntegrationTest, ZstdArchive) {
  std::string test_kpack =
      test_utils::get_test_assets_dir() / "test_zstd.kpack";
  ASSERT_TRUE(std::filesystem::exists(test_kpack))
      << "Test asset not found: " << test_kpack;

  // Open archive
  kpack_archive_t archive;
  kpack_error_t err = kpack_open(test_kpack.c_str(), &archive);
  ASSERT_EQ(err, KPACK_SUCCESS) << "Failed to open Zstd test archive";

  // Verify architectures
  size_t arch_count;
  err = kpack_get_architecture_count(archive, &arch_count);
  ASSERT_EQ(err, KPACK_SUCCESS);
  EXPECT_EQ(arch_count, 2);  // gfx1100, gfx1101

  const char* arch0;
  err = kpack_get_architecture(archive, 0, &arch0);
  ASSERT_EQ(err, KPACK_SUCCESS);
  EXPECT_STREQ(arch0, "gfx1100");

  const char* arch1;
  err = kpack_get_architecture(archive, 1, &arch1);
  ASSERT_EQ(err, KPACK_SUCCESS);
  EXPECT_STREQ(arch1, "gfx1101");

  // Verify binaries
  size_t binary_count;
  err = kpack_get_binary_count(archive, &binary_count);
  ASSERT_EQ(err, KPACK_SUCCESS);
  EXPECT_EQ(binary_count, 2);  // lib/libhip.so#0, bin/hiptest#0

  // Load and verify kernel 1: lib/libhip.so#0 @ gfx1100
  void* kernel_data;
  size_t kernel_size;
  err = kpack_get_kernel(archive, "lib/libhip.so#0", "gfx1100", &kernel_data,
                         &kernel_size);
  ASSERT_EQ(err, KPACK_SUCCESS);
  EXPECT_EQ(kernel_size, 1019);
  EXPECT_EQ(0, memcmp(kernel_data, "HIP_KERNEL_GFX1100_", 19));
  kpack_free_kernel(kernel_data);

  // Load and verify kernel 2: lib/libhip.so#0 @ gfx1101
  err = kpack_get_kernel(archive, "lib/libhip.so#0", "gfx1101", &kernel_data,
                         &kernel_size);
  ASSERT_EQ(err, KPACK_SUCCESS);
  EXPECT_EQ(kernel_size, 619);
  EXPECT_EQ(0, memcmp(kernel_data, "HIP_KERNEL_GFX1101_", 19));
  kpack_free_kernel(kernel_data);

  // Load and verify kernel 3: bin/hiptest#0 @ gfx1100
  err = kpack_get_kernel(archive, "bin/hiptest#0", "gfx1100", &kernel_data,
                         &kernel_size);
  ASSERT_EQ(err, KPACK_SUCCESS);
  EXPECT_EQ(kernel_size, 1018);
  EXPECT_EQ(0, memcmp(kernel_data, "TEST_APP_KERNEL___", 18));
  kpack_free_kernel(kernel_data);

  kpack_close(archive);
}

// Test kpack_get_binary retrieves binary names correctly
TEST(ArchiveIntegrationTest, GetBinaryNames) {
  std::string test_kpack =
      test_utils::get_test_assets_dir() / "test_noop.kpack";
  ASSERT_TRUE(std::filesystem::exists(test_kpack))
      << "Test asset not found: " << test_kpack;

  kpack_archive_t archive;
  kpack_error_t err = kpack_open(test_kpack.c_str(), &archive);
  ASSERT_EQ(err, KPACK_SUCCESS);

  // Verify binary count
  size_t binary_count;
  err = kpack_get_binary_count(archive, &binary_count);
  ASSERT_EQ(err, KPACK_SUCCESS);
  EXPECT_EQ(binary_count, 2u);

  // Get binary 0 (expected: bin/testapp#0 based on alphabetical ordering)
  const char* binary0;
  err = kpack_get_binary(archive, 0, &binary0);
  ASSERT_EQ(err, KPACK_SUCCESS);
  EXPECT_STREQ(binary0, "bin/testapp#0");

  // Get binary 1 (expected: lib/libtest.so#0)
  const char* binary1;
  err = kpack_get_binary(archive, 1, &binary1);
  ASSERT_EQ(err, KPACK_SUCCESS);
  EXPECT_STREQ(binary1, "lib/libtest.so#0");

  // Verify pointer stability - get again, should be same pointer
  const char* binary0_again;
  err = kpack_get_binary(archive, 0, &binary0_again);
  ASSERT_EQ(err, KPACK_SUCCESS);
  EXPECT_EQ(binary0, binary0_again);  // Same pointer, archive owns memory

  kpack_close(archive);
}

// Test kpack_get_binary retrieves binaries from zstd archive
TEST(ArchiveIntegrationTest, GetBinaryNames_Zstd) {
  std::string test_kpack =
      test_utils::get_test_assets_dir() / "test_zstd.kpack";
  ASSERT_TRUE(std::filesystem::exists(test_kpack))
      << "Test asset not found: " << test_kpack;

  kpack_archive_t archive;
  kpack_error_t err = kpack_open(test_kpack.c_str(), &archive);
  ASSERT_EQ(err, KPACK_SUCCESS);

  // Verify binary count
  size_t binary_count;
  err = kpack_get_binary_count(archive, &binary_count);
  ASSERT_EQ(err, KPACK_SUCCESS);
  EXPECT_EQ(binary_count, 2u);

  // Get both binaries
  const char* binary0;
  err = kpack_get_binary(archive, 0, &binary0);
  ASSERT_EQ(err, KPACK_SUCCESS);
  EXPECT_STREQ(binary0, "bin/hiptest#0");

  const char* binary1;
  err = kpack_get_binary(archive, 1, &binary1);
  ASSERT_EQ(err, KPACK_SUCCESS);
  EXPECT_STREQ(binary1, "lib/libhip.so#0");

  kpack_close(archive);
}

// Test that multiple kernel loads return independent copies
TEST(ArchiveIntegrationTest, KernelCacheOverwrite) {
  std::string test_kpack =
      test_utils::get_test_assets_dir() / "test_noop.kpack";

  kpack_archive_t archive;
  kpack_error_t err = kpack_open(test_kpack.c_str(), &archive);
  ASSERT_EQ(err, KPACK_SUCCESS);

  // Load first kernel
  void* kernel1_data;
  size_t kernel1_size;
  err = kpack_get_kernel(archive, "lib/libtest.so#0", "gfx900", &kernel1_data,
                         &kernel1_size);
  ASSERT_EQ(err, KPACK_SUCCESS);
  EXPECT_EQ(kernel1_size, 119);

  // Load second kernel - returns independent copy
  void* kernel2_data;
  size_t kernel2_size;
  err = kpack_get_kernel(archive, "lib/libtest.so#0", "gfx906", &kernel2_data,
                         &kernel2_size);
  ASSERT_EQ(err, KPACK_SUCCESS);

  // Verify second kernel is correct
  EXPECT_EQ(kernel2_size, 219);
  EXPECT_EQ(0, memcmp(kernel2_data, "KERNEL2_GFX906_DATA", 19));

  // Verify first kernel is STILL valid (independent copy, not overwritten)
  EXPECT_EQ(0, memcmp(kernel1_data, "KERNEL1_GFX900_DATA", 19));

  // Load first kernel again - verify it still works
  void* kernel1_reload;
  size_t kernel1_reload_size;
  err = kpack_get_kernel(archive, "lib/libtest.so#0", "gfx900", &kernel1_reload,
                         &kernel1_reload_size);
  ASSERT_EQ(err, KPACK_SUCCESS);
  EXPECT_EQ(kernel1_reload_size, 119);
  EXPECT_EQ(0, memcmp(kernel1_reload, "KERNEL1_GFX900_DATA", 19));

  // Free all kernels
  kpack_free_kernel(kernel1_data);
  kpack_free_kernel(kernel2_data);
  kpack_free_kernel(kernel1_reload);

  kpack_close(archive);
}
