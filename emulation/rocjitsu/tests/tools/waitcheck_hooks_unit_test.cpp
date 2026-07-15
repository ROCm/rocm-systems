// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "hsa/hsa_api_trace_minimal.h"
#include "waitcheck_fixture.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <unistd.h>

extern "C" bool OnLoad(HsaApiTable *table, uint64_t runtime_version, uint64_t failed_tool_count,
                       const char *const *failed_tool_names);
extern "C" void OnUnload();

namespace {

using CreateFromFileWithOffsetSizeFn = hsa_status_t (*)(hsa_file_t, size_t, size_t,
                                                        hsa_code_object_reader_t *);

struct AmdLoaderTablePrefix {
  void *query_host_address = nullptr;
  void *query_segment_descriptors = nullptr;
  void *query_executable = nullptr;
  void *executable_iterate_loaded_code_objects = nullptr;
  void *loaded_code_object_get_info = nullptr;
  CreateFromFileWithOffsetSizeFn create_from_file_with_offset_size = nullptr;
};

uint64_t g_next_reader = 1;
int g_load_calls = 0;
int g_destroy_calls = 0;
int g_first_tool_memory_calls = 0;
int g_first_tool_load_calls = 0;

decltype(hsa_code_object_reader_create_from_memory) *g_inner_create = nullptr;
decltype(hsa_code_object_reader_destroy) *g_inner_destroy = nullptr;
decltype(hsa_executable_load_agent_code_object) *g_inner_load = nullptr;
std::vector<uint8_t> g_replacement_bytes;

hsa_status_t HSA_API fake_get_extension_table(uint16_t, uint16_t, uint16_t, void *) {
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

hsa_status_t HSA_API fake_reader_create_from_file(hsa_file_t, hsa_code_object_reader_t *reader) {
  if (reader == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  reader->handle = g_next_reader++;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_reader_create_from_memory(const void *, size_t,
                                                    hsa_code_object_reader_t *reader) {
  if (reader == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  reader->handle = g_next_reader++;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API first_tool_reader_create_from_memory(const void *bytes, size_t size,
                                                          hsa_code_object_reader_t *reader) {
  ++g_first_tool_memory_calls;
  return fake_reader_create_from_memory(bytes, size, reader);
}

hsa_status_t HSA_API fake_reader_create_from_file_range(hsa_file_t, size_t, size_t,
                                                        hsa_code_object_reader_t *reader) {
  if (reader == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  reader->handle = g_next_reader++;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_get_major_extension_table(uint16_t extension, uint16_t version_major,
                                                    size_t table_length, void *table) {
  if (extension != HSA_EXTENSION_AMD_LOADER || version_major != 1 || table == nullptr ||
      table_length < sizeof(AmdLoaderTablePrefix)) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  auto *loader = static_cast<AmdLoaderTablePrefix *>(table);
  *loader = {};
  loader->create_from_file_with_offset_size = fake_reader_create_from_file_range;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_reader_destroy(hsa_code_object_reader_t) {
  ++g_destroy_calls;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_load(hsa_executable_t, hsa_agent_t, hsa_code_object_reader_t,
                               const char *, hsa_loaded_code_object_t *) {
  ++g_load_calls;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API first_tool_load(hsa_executable_t executable, hsa_agent_t agent,
                                     hsa_code_object_reader_t reader, const char *options,
                                     hsa_loaded_code_object_t *loaded) {
  ++g_first_tool_load_calls;
  return fake_load(executable, agent, reader, options, loaded);
}

hsa_status_t HSA_API outer_create(const void *bytes, size_t size,
                                  hsa_code_object_reader_t *reader) {
  return g_inner_create(bytes, size, reader);
}

hsa_status_t HSA_API outer_load(hsa_executable_t executable, hsa_agent_t agent,
                                hsa_code_object_reader_t, const char *options,
                                hsa_loaded_code_object_t *loaded) {
  hsa_code_object_reader_t replacement{};
  hsa_status_t status =
      g_inner_create(g_replacement_bytes.data(), g_replacement_bytes.size(), &replacement);
  if (status != HSA_STATUS_SUCCESS)
    return status;
  status = g_inner_load(executable, agent, replacement, options, loaded);
  (void)g_inner_destroy(replacement);
  return status;
}

CoreApiTable make_core_table() {
  CoreApiTable core{};
  core.version.major_id = 1;
  core.version.minor_id = sizeof(CoreApiTable);
  core.hsa_system_get_extension_table_fn = fake_get_extension_table;
  core.hsa_system_get_major_extension_table_fn = fake_get_major_extension_table;
  core.hsa_code_object_reader_create_from_file_fn = fake_reader_create_from_file;
  core.hsa_code_object_reader_create_from_memory_fn = fake_reader_create_from_memory;
  core.hsa_code_object_reader_destroy_fn = fake_reader_destroy;
  core.hsa_executable_load_agent_code_object_fn = fake_load;
  return core;
}

HsaApiTable make_api_table(CoreApiTable *core) {
  HsaApiTable table{};
  table.version.major_id = 1;
  table.version.minor_id = sizeof(HsaApiTable);
  table.core_ = core;
  return table;
}

struct CapturedCall {
  hsa_status_t status = HSA_STATUS_ERROR;
  std::string stderr_text;
};

CapturedCall capture_stderr(const std::function<hsa_status_t()> &call) {
  std::array<int, 2> pipe_fds{-1, -1};
  if (pipe(pipe_fds.data()) != 0)
    return {};
  std::fflush(stderr);
  const int saved_stderr = dup(STDERR_FILENO);
  if (saved_stderr < 0)
    return {};
  if (dup2(pipe_fds[1], STDERR_FILENO) < 0)
    return {};
  close(pipe_fds[1]);

  CapturedCall captured;
  captured.status = call();

  std::fflush(stderr);
  (void)dup2(saved_stderr, STDERR_FILENO);
  close(saved_stderr);
  std::array<char, 4096> buffer{};
  for (;;) {
    const ssize_t read_size = read(pipe_fds[0], buffer.data(), buffer.size());
    if (read_size <= 0)
      break;
    captured.stderr_text.append(buffer.data(), static_cast<size_t>(read_size));
  }
  close(pipe_fds[0]);
  return captured;
}

bool contains(std::string_view text, std::string_view needle) {
  return text.find(needle) != std::string_view::npos;
}

struct TempFile {
  std::string path;
  int fd = -1;

  TempFile() = default;
  TempFile(const TempFile &) = delete;
  TempFile &operator=(const TempFile &) = delete;
  TempFile(TempFile &&other) noexcept : path(std::move(other.path)), fd(other.fd) { other.fd = -1; }
  ~TempFile() {
    if (fd >= 0)
      close(fd);
    if (!path.empty())
      unlink(path.c_str());
  }
};

TempFile write_temp_bytes(const std::vector<uint8_t> &bytes) {
  std::array<char, 64> name{};
  std::snprintf(name.data(), name.size(), "/tmp/waitcheck_hooks_%ld_XXXXXX",
                static_cast<long>(getpid()));
  TempFile file;
  file.fd = mkstemp(name.data());
  if (file.fd < 0)
    return file;
  file.path = name.data();
  size_t written = 0;
  while (written < bytes.size()) {
    const ssize_t write_size = write(file.fd, bytes.data() + written, bytes.size() - written);
    if (write_size <= 0)
      break;
    written += static_cast<size_t>(write_size);
  }
  return file;
}

class WaitcheckHooksTest : public ::testing::Test {
protected:
  void SetUp() override {
    g_next_reader = 1;
    g_load_calls = 0;
    g_destroy_calls = 0;
    g_first_tool_memory_calls = 0;
    g_first_tool_load_calls = 0;
    setenv("ROCJITSU_WAITCHECK", "1", 1);
    setenv("ROCJITSU_WAITCHECK_FAIL", "1", 1);
    unsetenv("ROCJITSU_WAITCHECK_SUMMARY");
    core_ = make_core_table();
    table_ = make_api_table(&core_);
    ASSERT_TRUE(OnLoad(&table_, 1, 0, nullptr));
    loaded_ = true;
  }

  void TearDown() override {
    if (loaded_)
      OnUnload();
    unsetenv("ROCJITSU_WAITCHECK");
    unsetenv("ROCJITSU_WAITCHECK_FAIL");
    unsetenv("ROCJITSU_WAITCHECK_SUMMARY");
  }

  CapturedCall load_memory(const std::vector<uint8_t> &bytes) {
    hsa_code_object_reader_t reader{};
    EXPECT_EQ(
        core_.hsa_code_object_reader_create_from_memory_fn(bytes.data(), bytes.size(), &reader),
        HSA_STATUS_SUCCESS);
    CapturedCall captured = capture_stderr([&] {
      return core_.hsa_executable_load_agent_code_object_fn({}, {}, reader, nullptr, nullptr);
    });
    EXPECT_EQ(core_.hsa_code_object_reader_destroy_fn(reader), HSA_STATUS_SUCCESS);
    return captured;
  }

  CoreApiTable core_{};
  HsaApiTable table_{};
  bool loaded_ = false;
};

TEST_F(WaitcheckHooksTest, RejectsMissingWaitAtLoadRatherThanReaderCreation) {
  const auto bytes = rocjitsu::waitcheck_test::make_gfx1200_missing_wait_code_object();
  hsa_code_object_reader_t reader{};
  EXPECT_EQ(core_.hsa_code_object_reader_create_from_memory_fn(bytes.data(), bytes.size(), &reader),
            HSA_STATUS_SUCCESS);

  CapturedCall captured = capture_stderr([&] {
    return core_.hsa_executable_load_agent_code_object_fn({}, {}, reader, nullptr, nullptr);
  });
  EXPECT_EQ(captured.status, HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
  EXPECT_EQ(g_load_calls, 0);
  EXPECT_TRUE(contains(captured.stderr_text, "1 waitcnt hazard")) << captured.stderr_text;
  EXPECT_TRUE(contains(captured.stderr_text, "missing s_wait_loadcnt <= 0"))
      << captured.stderr_text;
  EXPECT_EQ(core_.hsa_code_object_reader_destroy_fn(reader), HSA_STATUS_SUCCESS);
}

TEST_F(WaitcheckHooksTest, CleanMemoryReaderChainsToRuntime) {
  const CapturedCall captured =
      load_memory(rocjitsu::waitcheck_test::make_gfx1200_correct_wait_code_object());
  EXPECT_EQ(captured.status, HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_load_calls, 1);
  EXPECT_FALSE(contains(captured.stderr_text, "rocjitsu-waitcheck:")) << captured.stderr_text;
}

TEST_F(WaitcheckHooksTest, ReportOnlyModeChainsHazardToRuntimeOncePerReader) {
  setenv("ROCJITSU_WAITCHECK_FAIL", "0", 1);
  const auto bytes = rocjitsu::waitcheck_test::make_gfx1200_missing_wait_code_object();
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(core_.hsa_code_object_reader_create_from_memory_fn(bytes.data(), bytes.size(), &reader),
            HSA_STATUS_SUCCESS);
  const CapturedCall first = capture_stderr([&] {
    return core_.hsa_executable_load_agent_code_object_fn({}, {}, reader, nullptr, nullptr);
  });
  const CapturedCall second = capture_stderr([&] {
    return core_.hsa_executable_load_agent_code_object_fn({}, {}, reader, nullptr, nullptr);
  });
  EXPECT_EQ(first.status, HSA_STATUS_SUCCESS);
  EXPECT_EQ(second.status, HSA_STATUS_SUCCESS);
  EXPECT_TRUE(contains(first.stderr_text, "1 waitcnt hazard"));
  EXPECT_TRUE(second.stderr_text.empty()) << second.stderr_text;
  EXPECT_EQ(g_load_calls, 2);
  EXPECT_EQ(core_.hsa_code_object_reader_destroy_fn(reader), HSA_STATUS_SUCCESS);
}

TEST_F(WaitcheckHooksTest, FileReaderIsCheckedAtLoad) {
  const auto bytes = rocjitsu::waitcheck_test::make_gfx1200_missing_wait_code_object();
  TempFile file = write_temp_bytes(bytes);
  ASSERT_GE(file.fd, 0);
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(core_.hsa_code_object_reader_create_from_file_fn(file.fd, &reader), HSA_STATUS_SUCCESS);
  const CapturedCall captured = capture_stderr([&] {
    return core_.hsa_executable_load_agent_code_object_fn({}, {}, reader, nullptr, nullptr);
  });
  EXPECT_EQ(captured.status, HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
  EXPECT_TRUE(contains(captured.stderr_text, "1 waitcnt hazard")) << captured.stderr_text;
  EXPECT_EQ(core_.hsa_code_object_reader_destroy_fn(reader), HSA_STATUS_SUCCESS);
}

TEST_F(WaitcheckHooksTest, LoaderExtensionFileRangeIsCheckedAtLoad) {
  const auto bytes = rocjitsu::waitcheck_test::make_gfx1200_missing_wait_code_object();
  std::vector<uint8_t> file_bytes = {'p', 'a', 'd'};
  const size_t offset = file_bytes.size();
  file_bytes.insert(file_bytes.end(), bytes.begin(), bytes.end());
  file_bytes.push_back(0);
  TempFile file = write_temp_bytes(file_bytes);
  ASSERT_GE(file.fd, 0);

  AmdLoaderTablePrefix loader{};
  ASSERT_EQ(core_.hsa_system_get_major_extension_table_fn(HSA_EXTENSION_AMD_LOADER, 1,
                                                          sizeof(loader), &loader),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(loader.create_from_file_with_offset_size, nullptr);
  ASSERT_NE(loader.create_from_file_with_offset_size, fake_reader_create_from_file_range);

  hsa_code_object_reader_t reader{};
  ASSERT_EQ(loader.create_from_file_with_offset_size(file.fd, offset, bytes.size(), &reader),
            HSA_STATUS_SUCCESS);
  const CapturedCall captured = capture_stderr([&] {
    return core_.hsa_executable_load_agent_code_object_fn({}, {}, reader, nullptr, nullptr);
  });
  EXPECT_EQ(captured.status, HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
  EXPECT_TRUE(contains(captured.stderr_text, "1 waitcnt hazard")) << captured.stderr_text;
  EXPECT_EQ(core_.hsa_code_object_reader_destroy_fn(reader), HSA_STATUS_SUCCESS);
}

TEST_F(WaitcheckHooksTest, DisabledHookPassesHazardThroughWithoutDiagnostics) {
  setenv("ROCJITSU_WAITCHECK", "0", 1);
  const CapturedCall captured =
      load_memory(rocjitsu::waitcheck_test::make_gfx1200_missing_wait_code_object());
  EXPECT_EQ(captured.status, HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_load_calls, 1);
  EXPECT_TRUE(captured.stderr_text.empty()) << captured.stderr_text;
}

TEST_F(WaitcheckHooksTest, SavesChainsAndRestoresPreviouslyInstalledWrappers) {
  OnUnload();
  loaded_ = false;
  core_ = make_core_table();
  core_.hsa_code_object_reader_create_from_memory_fn = first_tool_reader_create_from_memory;
  core_.hsa_executable_load_agent_code_object_fn = first_tool_load;
  table_ = make_api_table(&core_);
  ASSERT_TRUE(OnLoad(&table_, 1, 0, nullptr));
  loaded_ = true;

  const CapturedCall captured =
      load_memory(rocjitsu::waitcheck_test::make_gfx1200_correct_wait_code_object());
  EXPECT_EQ(captured.status, HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_first_tool_memory_calls, 1);
  EXPECT_EQ(g_first_tool_load_calls, 1);

  OnUnload();
  loaded_ = false;
  EXPECT_EQ(core_.hsa_code_object_reader_create_from_memory_fn,
            first_tool_reader_create_from_memory);
  EXPECT_EQ(core_.hsa_executable_load_agent_code_object_fn, first_tool_load);
}

TEST_F(WaitcheckHooksTest, InnerWaitcheckAnalyzesOuterToolsFinalReplacement) {
  g_inner_create = core_.hsa_code_object_reader_create_from_memory_fn;
  g_inner_destroy = core_.hsa_code_object_reader_destroy_fn;
  g_inner_load = core_.hsa_executable_load_agent_code_object_fn;
  core_.hsa_code_object_reader_create_from_memory_fn = outer_create;
  core_.hsa_executable_load_agent_code_object_fn = outer_load;

  g_replacement_bytes = rocjitsu::waitcheck_test::make_gfx1200_correct_wait_code_object();
  CapturedCall captured =
      load_memory(rocjitsu::waitcheck_test::make_gfx1200_missing_wait_code_object());
  EXPECT_EQ(captured.status, HSA_STATUS_SUCCESS);
  EXPECT_FALSE(contains(captured.stderr_text, "waitcnt hazard")) << captured.stderr_text;

  g_replacement_bytes = rocjitsu::waitcheck_test::make_gfx1200_missing_wait_code_object();
  captured = load_memory(rocjitsu::waitcheck_test::make_gfx1200_correct_wait_code_object());
  EXPECT_EQ(captured.status, HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
  EXPECT_TRUE(contains(captured.stderr_text, "1 waitcnt hazard")) << captured.stderr_text;

  core_.hsa_code_object_reader_create_from_memory_fn = g_inner_create;
  core_.hsa_executable_load_agent_code_object_fn = g_inner_load;
}

TEST(WaitcheckHooksValidationTest, RejectsTruncatedCoreTable) {
  CoreApiTable core = make_core_table();
  core.version.minor_id = offsetof(CoreApiTable, hsa_executable_load_agent_code_object_fn);
  HsaApiTable table = make_api_table(&core);
  EXPECT_FALSE(OnLoad(&table, 1, 0, nullptr));
}

} // namespace
