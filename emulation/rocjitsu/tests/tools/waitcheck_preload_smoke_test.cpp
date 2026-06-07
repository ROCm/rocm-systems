// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file waitcheck_preload_smoke_test.cpp
/// @brief Smoke tests for the rocjitsu_waitcheck LD_PRELOAD hook.

#include "waitcheck_fixture.h"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#endif
#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_loader.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include <gtest/gtest.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <string>
#include <string_view>
#include <vector>

#include <unistd.h>

namespace {

struct CapturedCreate {
  hsa_status_t status = HSA_STATUS_ERROR;
  std::string stderr_text;
};

bool contains(std::string_view haystack, std::string_view needle) {
  return haystack.find(needle) != std::string_view::npos;
}

hsa_status_t create_reader(const std::vector<uint8_t> &image) {
  hsa_code_object_reader_t reader{};
  const hsa_status_t status =
      hsa_code_object_reader_create_from_memory(image.data(), image.size(), &reader);
  if (status == HSA_STATUS_SUCCESS)
    (void)hsa_code_object_reader_destroy(reader);
  return status;
}

using CreateFromFileWithOffsetSizeFn = hsa_status_t (*)(hsa_file_t, size_t, size_t,
                                                        hsa_code_object_reader_t *);

CreateFromFileWithOffsetSizeFn create_from_file_with_offset_size_fn() {
  return reinterpret_cast<CreateFromFileWithOffsetSizeFn>(dlsym(
      RTLD_DEFAULT, "hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size"));
}

template <typename Fn> CapturedCreate capture_create_stderr(Fn &&create) {
  std::array<int, 2> pipe_fds{-1, -1};
  if (pipe(pipe_fds.data()) != 0) {
    ADD_FAILURE() << "pipe failed";
    return {};
  }

  std::fflush(stderr);
  const int saved_stderr = dup(STDERR_FILENO);
  if (saved_stderr < 0) {
    ADD_FAILURE() << "dup stderr failed";
    close(pipe_fds[0]);
    close(pipe_fds[1]);
    return {};
  }

  if (dup2(pipe_fds[1], STDERR_FILENO) < 0) {
    ADD_FAILURE() << "redirect stderr failed";
    close(saved_stderr);
    close(pipe_fds[0]);
    close(pipe_fds[1]);
    return {};
  }
  close(pipe_fds[1]);

  CapturedCreate captured;
  captured.status = create();

  std::fflush(stderr);
  if (dup2(saved_stderr, STDERR_FILENO) < 0)
    ADD_FAILURE() << "restore stderr failed";
  close(saved_stderr);

  std::array<char, 4096> buffer{};
  for (;;) {
    const ssize_t nread = read(pipe_fds[0], buffer.data(), buffer.size());
    if (nread <= 0)
      break;
    captured.stderr_text.append(buffer.data(), static_cast<size_t>(nread));
  }
  close(pipe_fds[0]);
  return captured;
}

CapturedCreate create_reader_with_captured_stderr(const std::vector<uint8_t> &image) {
  return capture_create_stderr([&] { return create_reader(image); });
}

struct TempFile {
  std::string path;
  int fd = -1;

  ~TempFile() {
    if (fd >= 0)
      close(fd);
    if (!path.empty())
      unlink(path.c_str());
  }
};

struct HsaRuntimeScope {
  hsa_status_t status = hsa_init();

  ~HsaRuntimeScope() {
    if (status == HSA_STATUS_SUCCESS)
      (void)hsa_shut_down();
  }
};

TempFile write_temp_bytes(const std::vector<uint8_t> &bytes) {
  std::array<char, 64> path_template{};
  std::snprintf(path_template.data(), path_template.size(), "/tmp/waitcheck_co_%ld_XXXXXX",
                static_cast<long>(getpid()));
  TempFile file;
  file.fd = mkstemp(path_template.data());
  if (file.fd < 0) {
    ADD_FAILURE() << "mkstemp failed";
    return file;
  }
  file.path = path_template.data();

  size_t written = 0;
  while (written < bytes.size()) {
    const ssize_t nwritten = write(file.fd, bytes.data() + written, bytes.size() - written);
    if (nwritten <= 0) {
      ADD_FAILURE() << "write failed";
      return file;
    }
    written += static_cast<size_t>(nwritten);
  }
  if (lseek(file.fd, 0, SEEK_SET) < 0)
    ADD_FAILURE() << "lseek failed";
  return file;
}

TempFile write_temp_code_object(const std::vector<uint8_t> &image) {
  return write_temp_bytes(image);
}

CapturedCreate create_file_reader_with_captured_stderr(hsa_file_t file) {
  return capture_create_stderr([&] {
    hsa_code_object_reader_t reader{};
    const hsa_status_t status = hsa_code_object_reader_create_from_file(file, &reader);
    if (status == HSA_STATUS_SUCCESS)
      (void)hsa_code_object_reader_destroy(reader);
    return status;
  });
}

CapturedCreate create_file_range_reader_with_captured_stderr(hsa_file_t file, size_t offset,
                                                             size_t size) {
  return capture_create_stderr([&] {
    auto *create = create_from_file_with_offset_size_fn();
    if (!create)
      return HSA_STATUS_ERROR_NOT_INITIALIZED;

    hsa_code_object_reader_t reader{};
    const hsa_status_t status = create(file, offset, size, &reader);
    if (status == HSA_STATUS_SUCCESS)
      (void)hsa_code_object_reader_destroy(reader);
    return status;
  });
}

void enable_waitcheck(bool fail_on_hazard) {
  setenv("ROCJITSU_WAITCHECK", "1", 1);
  setenv("ROCJITSU_WAITCHECK_FAIL", fail_on_hazard ? "1" : "0", 1);
}

} // namespace

TEST(WaitcheckPreload, MissingWaitReportsAndFailsWhenRequested) {
  enable_waitcheck(true);

  const auto image = rocjitsu::waitcheck_test::make_gfx1200_missing_wait_code_object();
  const CapturedCreate captured = create_reader_with_captured_stderr(image);

  EXPECT_EQ(captured.status, HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
  EXPECT_TRUE(contains(captured.stderr_text, "rocjitsu-waitcheck: 1 waitcnt hazard"))
      << captured.stderr_text;
  EXPECT_TRUE(contains(captured.stderr_text, "missing s_wait_loadcnt <= 0"))
      << captured.stderr_text;
}

TEST(WaitcheckPreload, CorrectWaitDoesNotReportHazard) {
  enable_waitcheck(true);

  const auto image = rocjitsu::waitcheck_test::make_gfx1200_correct_wait_code_object();
  const CapturedCreate captured = create_reader_with_captured_stderr(image);

  EXPECT_FALSE(contains(captured.stderr_text, "rocjitsu-waitcheck:")) << captured.stderr_text;
}

TEST(WaitcheckPreload, FileBackedReaderReportsAndFailsWhenRequested) {
  enable_waitcheck(true);

  const auto image = rocjitsu::waitcheck_test::make_gfx1200_missing_wait_code_object();
  TempFile file = write_temp_code_object(image);
  ASSERT_GE(file.fd, 0);
  const CapturedCreate captured = create_file_reader_with_captured_stderr(file.fd);

  EXPECT_EQ(captured.status, HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
  EXPECT_TRUE(contains(captured.stderr_text, "rocjitsu-waitcheck: 1 waitcnt hazard"))
      << captured.stderr_text;
  EXPECT_TRUE(contains(captured.stderr_text, "missing s_wait_loadcnt <= 0"))
      << captured.stderr_text;
}

TEST(WaitcheckPreload, FileBackedOffsetSizeReaderReportsEmbeddedCodeObject) {
  enable_waitcheck(true);

  const auto image = rocjitsu::waitcheck_test::make_gfx1200_missing_wait_code_object();
  std::vector<uint8_t> file_bytes = {0x7f, 'n', 'o', 't', '-', 'e', 'l', 'f'};
  const size_t code_object_offset = file_bytes.size();
  file_bytes.insert(file_bytes.end(), image.begin(), image.end());
  file_bytes.insert(file_bytes.end(), {0xde, 0xad, 0xbe, 0xef});
  TempFile file = write_temp_bytes(file_bytes);
  ASSERT_GE(file.fd, 0);

  const CapturedCreate captured =
      create_file_range_reader_with_captured_stderr(file.fd, code_object_offset, image.size());

  EXPECT_EQ(captured.status, HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
  EXPECT_TRUE(contains(captured.stderr_text, "rocjitsu-waitcheck: 1 waitcnt hazard"))
      << captured.stderr_text;
  EXPECT_TRUE(contains(captured.stderr_text, "missing s_wait_loadcnt <= 0"))
      << captured.stderr_text;
}

TEST(WaitcheckPreload, LoaderExtensionTableRoutesOffsetSizeReaderThroughPreload) {
  enable_waitcheck(true);
  HsaRuntimeScope hsa;
  ASSERT_EQ(hsa.status, HSA_STATUS_SUCCESS);

  hsa_ven_amd_loader_1_03_pfn_t table{};
  hsa_status_t status =
      hsa_system_get_major_extension_table(HSA_EXTENSION_AMD_LOADER, 1, sizeof(table), &table);
  ASSERT_EQ(status, HSA_STATUS_SUCCESS);
  ASSERT_NE(table.hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size, nullptr);

  const auto image = rocjitsu::waitcheck_test::make_gfx1200_missing_wait_code_object();
  std::vector<uint8_t> file_bytes = {'p', 'a', 'd'};
  const size_t code_object_offset = file_bytes.size();
  file_bytes.insert(file_bytes.end(), image.begin(), image.end());
  file_bytes.push_back(0);
  TempFile file = write_temp_bytes(file_bytes);
  ASSERT_GE(file.fd, 0);

  const CapturedCreate captured = capture_create_stderr([&] {
    hsa_code_object_reader_t reader{};
    const hsa_status_t reader_status =
        table.hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size(
            file.fd, code_object_offset, image.size(), &reader);
    if (reader_status == HSA_STATUS_SUCCESS)
      (void)hsa_code_object_reader_destroy(reader);
    return reader_status;
  });

  EXPECT_EQ(captured.status, HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
  EXPECT_TRUE(contains(captured.stderr_text, "rocjitsu-waitcheck: 1 waitcnt hazard"))
      << captured.stderr_text;
  EXPECT_TRUE(contains(captured.stderr_text, "missing s_wait_loadcnt <= 0"))
      << captured.stderr_text;
}

TEST(WaitcheckPreload, AnalysisFailureReportsAndFailsWhenRequested) {
  enable_waitcheck(true);

  const auto image = rocjitsu::waitcheck_test::make_gfx1200_invalid_instruction_code_object();
  const CapturedCreate captured = create_reader_with_captured_stderr(image);

  EXPECT_EQ(captured.status, HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
  EXPECT_TRUE(
      contains(captured.stderr_text, "rocjitsu-waitcheck: analysis failed for gfx1200 code object"))
      << captured.stderr_text;
  EXPECT_TRUE(contains(captured.stderr_text,
                       "decode failed while building CFG: Invalid instruction opcode: 800000"))
      << captured.stderr_text;
}

TEST(WaitcheckPreload, FileBackedCorrectReaderChainsToRuntime) {
  enable_waitcheck(true);
  HsaRuntimeScope hsa;
  ASSERT_EQ(hsa.status, HSA_STATUS_SUCCESS);

  const auto image = rocjitsu::waitcheck_test::make_gfx1200_correct_wait_code_object();
  TempFile file = write_temp_code_object(image);
  ASSERT_GE(file.fd, 0);
  const CapturedCreate captured = create_file_reader_with_captured_stderr(file.fd);

  EXPECT_EQ(captured.status, HSA_STATUS_SUCCESS);
  EXPECT_FALSE(contains(captured.stderr_text, "rocjitsu-waitcheck:")) << captured.stderr_text;
}

TEST(WaitcheckPreload, DisabledHookDoesNotReportHazard) {
  setenv("ROCJITSU_WAITCHECK", "0", 1);
  setenv("ROCJITSU_WAITCHECK_FAIL", "1", 1);

  const auto image = rocjitsu::waitcheck_test::make_gfx1200_missing_wait_code_object();
  const CapturedCreate captured = create_reader_with_captured_stderr(image);

  EXPECT_FALSE(contains(captured.stderr_text, "rocjitsu-waitcheck:")) << captured.stderr_text;
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
