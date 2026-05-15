// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "rocm_kpack/kpack.h"

// Helper to create a temp file with specific content
// Uses mkstemp() on POSIX for race-free temp file creation
class TempFile {
 public:
  TempFile(const void* data, size_t size) {
#ifdef _WIN32
    // Use GetTempPath2 (Windows 10 20H2+) for better security, fall back to
    // GetTempPath
    wchar_t temp_path[32768];  // No MAX_PATH limitation
    DWORD len = GetTempPath2W(32768, temp_path);
    if (len == 0) {
      len = GetTempPathW(32768, temp_path);
    }
    wchar_t temp_file[32768];
    if (len > 0 && GetTempFileNameW(temp_path, L"kpk", 0, temp_file)) {
      path_ = std::filesystem::path(temp_file);
      FILE* f = _wfopen(temp_file, L"wb");
      if (f) {
        fwrite(data, 1, size, f);
        fclose(f);
      }
    }
#else
    std::string tmpl =
        (std::filesystem::temp_directory_path() / "kpack_test_XXXXXX").string();
    int fd = mkstemp(tmpl.data());
    if (fd >= 0) {
      ssize_t written = write(fd, data, size);
      (void)written;  // Ignore return in tests
      close(fd);
      path_ = tmpl;
    }
#endif
  }

  TempFile(const std::string& content)
      : TempFile(content.data(), content.size()) {}

  ~TempFile() { std::filesystem::remove(path_); }

  const std::filesystem::path& path() const { return path_; }
  std::string str() const { return path_.string(); }

 private:
  std::filesystem::path path_;
};

// Test that library links and basic error handling works
TEST(KpackAPITest, NullArguments) {
  kpack_error_t err;

  // kpack_open with NULL path
  kpack_archive_t archive;
  err = kpack_open(nullptr, &archive);
  EXPECT_EQ(err, KPACK_ERROR_INVALID_ARGUMENT);

  // kpack_open with NULL archive pointer (path can be any valid-looking path)
  std::string temp_path =
      (std::filesystem::temp_directory_path() / "test.kpack").string();
  err = kpack_open(temp_path.c_str(), nullptr);
  EXPECT_EQ(err, KPACK_ERROR_INVALID_ARGUMENT);

  // kpack_get_architecture_count with NULL archive
  size_t count;
  err = kpack_get_architecture_count(nullptr, &count);
  EXPECT_EQ(err, KPACK_ERROR_INVALID_ARGUMENT);

  // kpack_get_kernel with NULL arguments
  void* data;
  size_t size;
  err = kpack_get_kernel(nullptr, "test", "gfx1100", &data, &size);
  EXPECT_EQ(err, KPACK_ERROR_INVALID_ARGUMENT);
}

// Test file not found
TEST(KpackAPITest, FileNotFound) {
  kpack_archive_t archive;
  kpack_error_t err = kpack_open("/nonexistent/test.kpack", &archive);
  EXPECT_EQ(err, KPACK_ERROR_FILE_NOT_FOUND);
}

// Test kpack_close with NULL (should not crash)
TEST(KpackAPITest, CloseNull) {
  kpack_close(nullptr);
  // Should not crash
}

// Test kpack_get_binary with NULL arguments
TEST(KpackAPITest, GetBinary_NullArchive) {
  const char* binary;
  kpack_error_t err = kpack_get_binary(nullptr, 0, &binary);
  EXPECT_EQ(err, KPACK_ERROR_INVALID_ARGUMENT);
}

TEST(KpackAPITest, GetBinary_NullBinaryOut) {
  // Need a valid archive to test this - skip if no test assets
  const char* dir = std::getenv("ROCM_KPACK_TEST_ASSETS_DIR");
  if (!dir) {
    GTEST_SKIP() << "ROCM_KPACK_TEST_ASSETS_DIR not set";
  }

  std::string test_kpack = std::string(dir) + "/test_noop.kpack";
  kpack_archive_t archive;
  kpack_error_t err = kpack_open(test_kpack.c_str(), &archive);
  ASSERT_EQ(err, KPACK_SUCCESS);

  err = kpack_get_binary(archive, 0, nullptr);
  EXPECT_EQ(err, KPACK_ERROR_INVALID_ARGUMENT);

  kpack_close(archive);
}

TEST(KpackAPITest, GetBinary_IndexOutOfRange) {
  const char* dir = std::getenv("ROCM_KPACK_TEST_ASSETS_DIR");
  if (!dir) {
    GTEST_SKIP() << "ROCM_KPACK_TEST_ASSETS_DIR not set";
  }

  std::string test_kpack = std::string(dir) + "/test_noop.kpack";
  kpack_archive_t archive;
  kpack_error_t err = kpack_open(test_kpack.c_str(), &archive);
  ASSERT_EQ(err, KPACK_SUCCESS);

  size_t binary_count;
  err = kpack_get_binary_count(archive, &binary_count);
  ASSERT_EQ(err, KPACK_SUCCESS);
  ASSERT_GT(binary_count, 0u);

  // Test index exactly at boundary (should fail)
  const char* binary;
  err = kpack_get_binary(archive, binary_count, &binary);
  EXPECT_EQ(err, KPACK_ERROR_INVALID_ARGUMENT);

  // Test way out of range
  err = kpack_get_binary(archive, 99999, &binary);
  EXPECT_EQ(err, KPACK_ERROR_INVALID_ARGUMENT);

  kpack_close(archive);
}

// Test kpack_get_architecture with invalid index
TEST(KpackAPITest, GetArchitecture_IndexOutOfRange) {
  const char* dir = std::getenv("ROCM_KPACK_TEST_ASSETS_DIR");
  if (!dir) {
    GTEST_SKIP() << "ROCM_KPACK_TEST_ASSETS_DIR not set";
  }

  std::string test_kpack = std::string(dir) + "/test_noop.kpack";
  kpack_archive_t archive;
  kpack_error_t err = kpack_open(test_kpack.c_str(), &archive);
  ASSERT_EQ(err, KPACK_SUCCESS);

  size_t arch_count;
  err = kpack_get_architecture_count(archive, &arch_count);
  ASSERT_EQ(err, KPACK_SUCCESS);
  ASSERT_GT(arch_count, 0u);

  // Test index exactly at boundary (should fail)
  const char* arch;
  err = kpack_get_architecture(archive, arch_count, &arch);
  EXPECT_EQ(err, KPACK_ERROR_INVALID_ARGUMENT);

  // Test way out of range
  err = kpack_get_architecture(archive, 99999, &arch);
  EXPECT_EQ(err, KPACK_ERROR_INVALID_ARGUMENT);

  kpack_close(archive);
}

// Test kpack_get_binary_count with NULL count pointer
TEST(KpackAPITest, GetBinaryCount_NullCount) {
  const char* dir = std::getenv("ROCM_KPACK_TEST_ASSETS_DIR");
  if (!dir) {
    GTEST_SKIP() << "ROCM_KPACK_TEST_ASSETS_DIR not set";
  }

  std::string test_kpack = std::string(dir) + "/test_noop.kpack";
  kpack_archive_t archive;
  kpack_error_t err = kpack_open(test_kpack.c_str(), &archive);
  ASSERT_EQ(err, KPACK_SUCCESS);

  err = kpack_get_binary_count(archive, nullptr);
  EXPECT_EQ(err, KPACK_ERROR_INVALID_ARGUMENT);

  kpack_close(archive);
}

// Test kpack_get_architecture_count with NULL count pointer
TEST(KpackAPITest, GetArchitectureCount_NullCount) {
  const char* dir = std::getenv("ROCM_KPACK_TEST_ASSETS_DIR");
  if (!dir) {
    GTEST_SKIP() << "ROCM_KPACK_TEST_ASSETS_DIR not set";
  }

  std::string test_kpack = std::string(dir) + "/test_noop.kpack";
  kpack_archive_t archive;
  kpack_error_t err = kpack_open(test_kpack.c_str(), &archive);
  ASSERT_EQ(err, KPACK_SUCCESS);

  err = kpack_get_architecture_count(archive, nullptr);
  EXPECT_EQ(err, KPACK_ERROR_INVALID_ARGUMENT);

  kpack_close(archive);
}

// Test kpack_free_kernel with NULL (should not crash)
TEST(KpackAPITest, FreeKernel_Null) {
  kpack_free_kernel(nullptr);
  // Should not crash
}

//
// Invalid archive format tests
//

TEST(KpackAPITest, InvalidArchive_EmptyFile) {
  TempFile empty_file("", 0);

  kpack_archive_t archive;
  kpack_error_t err = kpack_open(empty_file.str().c_str(), &archive);
  // Empty file should fail - can't read header
  EXPECT_NE(err, KPACK_SUCCESS);
}

TEST(KpackAPITest, InvalidArchive_WrongMagic) {
  // Create a file with wrong magic bytes
  // Header format: magic (4), version (4), compression (4), toc_offset (8)
  struct __attribute__((packed)) FakeHeader {
    char magic[4] = {'X', 'X', 'X', 'X'};  // Wrong magic
    uint32_t version = 1;
    uint32_t compression = 0;
    uint64_t toc_offset = 20;
  };
  FakeHeader header;

  TempFile bad_magic(&header, sizeof(header));

  kpack_archive_t archive;
  kpack_error_t err = kpack_open(bad_magic.str().c_str(), &archive);
  EXPECT_EQ(err, KPACK_ERROR_INVALID_FORMAT);
}

TEST(KpackAPITest, InvalidArchive_UnsupportedVersion) {
  // Create a file with correct magic but wrong version
  struct __attribute__((packed)) FakeHeader {
    char magic[4] = {'K', 'P', 'A', 'K'};  // Correct magic
    uint32_t version = 999;                // Unsupported version
    uint32_t compression = 0;
    uint64_t toc_offset = 20;
  };
  FakeHeader header;

  TempFile bad_version(&header, sizeof(header));

  kpack_archive_t archive;
  kpack_error_t err = kpack_open(bad_version.str().c_str(), &archive);
  EXPECT_EQ(err, KPACK_ERROR_UNSUPPORTED_VERSION);
}

TEST(KpackAPITest, InvalidArchive_TruncatedHeader) {
  // Create a file with only partial header (8 bytes instead of 20)
  const char partial_header[] = "KPAK\x01\x00\x00\x00";  // magic + version only

  TempFile truncated(partial_header, 8);

  kpack_archive_t archive;
  kpack_error_t err = kpack_open(truncated.str().c_str(), &archive);
  // Should fail - can't read full header
  EXPECT_NE(err, KPACK_SUCCESS);
}

TEST(KpackAPITest, InvalidArchive_TOCOffsetBeyondFile) {
  // Create a file with valid header but TOC offset beyond file size
  struct __attribute__((packed)) FakeHeader {
    char magic[4] = {'K', 'P', 'A', 'K'};
    uint32_t version = 1;
    uint32_t compression = 0;
    uint64_t toc_offset = 999999;  // Way beyond file size
  };
  FakeHeader header;

  TempFile bad_offset(&header, sizeof(header));

  kpack_archive_t archive;
  kpack_error_t err = kpack_open(bad_offset.str().c_str(), &archive);
  // Should fail with INVALID_FORMAT - TOC offset is beyond file size
  EXPECT_EQ(err, KPACK_ERROR_INVALID_FORMAT);
}
