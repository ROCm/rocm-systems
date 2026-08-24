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
using LoadedCodeObjectGetInfoFn = hsa_status_t (*)(hsa_loaded_code_object_t, uint32_t, void *);

struct AmdLoaderTablePrefix {
  void *query_host_address = nullptr;
  void *query_segment_descriptors = nullptr;
  void *query_executable = nullptr;
  void *executable_iterate_loaded_code_objects = nullptr;
  LoadedCodeObjectGetInfoFn loaded_code_object_get_info = nullptr;
  CreateFromFileWithOffsetSizeFn create_from_file_with_offset_size = nullptr;
};

uint64_t g_next_reader = 1;
int g_load_calls = 0;
int g_first_tool_memory_calls = 0;
int g_first_tool_load_calls = 0;
int g_writer_calls = 0;
int64_t g_load_delta = 0x100000;
hsa_amd_queue_intercept_handler_t g_intercept_handler = nullptr;
void *g_intercept_data = nullptr;
hsa_queue_t g_intercept_queue{};
hsa_queue_t g_descriptor_queue{};
int g_amd_queue_create_calls = 0;
hsa_queue_t *g_last_destroyed_queue = nullptr;
hsa_queue_t *g_priority_queue = nullptr;
hsa_amd_queue_priority_t g_priority = HSA_AMD_QUEUE_PRIORITY_NORMAL;
hsa_queue_t *g_cu_mask_queue = nullptr;
uint32_t g_cu_mask_count = 0;
uint32_t g_cu_mask_first_word = 0;

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
  loader->loaded_code_object_get_info = [](hsa_loaded_code_object_t, uint32_t attribute,
                                           void *value) -> hsa_status_t {
    if (attribute != 8 || value == nullptr)
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    *static_cast<int64_t *>(value) = g_load_delta;
    return HSA_STATUS_SUCCESS;
  };
  loader->create_from_file_with_offset_size = fake_reader_create_from_file_range;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_reader_destroy(hsa_code_object_reader_t) { return HSA_STATUS_SUCCESS; }

hsa_status_t HSA_API fake_load(hsa_executable_t, hsa_agent_t, hsa_code_object_reader_t,
                               const char *, hsa_loaded_code_object_t *loaded) {
  ++g_load_calls;
  if (loaded != nullptr)
    loaded->handle = static_cast<uint64_t>(g_load_calls);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_executable_freeze(hsa_executable_t, const char *) {
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_executable_destroy(hsa_executable_t) { return HSA_STATUS_SUCCESS; }

hsa_status_t HSA_API fake_queue_destroy(hsa_queue_t *queue) {
  g_last_destroyed_queue = queue;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_queue_intercept_create(hsa_agent_t, uint32_t, hsa_queue_type32_t,
                                                 void (*)(hsa_status_t, hsa_queue_t *, void *),
                                                 void *, uint32_t, uint32_t, hsa_queue_t **queue) {
  if (queue == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *queue = &g_intercept_queue;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_queue_intercept_register(hsa_queue_t *,
                                                   hsa_amd_queue_intercept_handler_t handler,
                                                   void *data) {
  g_intercept_handler = handler;
  g_intercept_data = data;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_amd_queue_create(hsa_agent_t, hsa_amd_queue_create_desc_t *descs,
                                           uint32_t num_descs) {
  ++g_amd_queue_create_calls;
  if (descs == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  for (uint32_t index = 0; index < num_descs; ++index) {
    if (descs[index].engine_type == HSA_AMD_QUEUE_ENGINE_COMPUTE)
      descs[index].queue = &g_descriptor_queue;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_queue_set_priority(hsa_queue_t *queue,
                                             hsa_amd_queue_priority_t priority) {
  g_priority_queue = queue;
  g_priority = priority;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_queue_cu_set_mask(const hsa_queue_t *queue, uint32_t count,
                                            const uint32_t *mask) {
  g_cu_mask_queue = const_cast<hsa_queue_t *>(queue);
  g_cu_mask_count = count;
  g_cu_mask_first_word = mask == nullptr ? 0u : mask[0];
  return HSA_STATUS_SUCCESS;
}

void fake_packet_writer(const void *, uint64_t packet_count) {
  g_writer_calls += static_cast<int>(packet_count);
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
  core.hsa_queue_create_fn = fake_queue_intercept_create;
  core.hsa_queue_destroy_fn = fake_queue_destroy;
  core.hsa_executable_destroy_fn = fake_executable_destroy;
  core.hsa_executable_freeze_fn = fake_executable_freeze;
  core.hsa_code_object_reader_create_from_file_fn = fake_reader_create_from_file;
  core.hsa_code_object_reader_create_from_memory_fn = fake_reader_create_from_memory;
  core.hsa_code_object_reader_destroy_fn = fake_reader_destroy;
  core.hsa_executable_load_agent_code_object_fn = fake_load;
  return core;
}

AmdExtTable make_amd_ext_table() {
  AmdExtTable table{};
  table.version.major_id = 1;
  table.version.minor_id = sizeof(AmdExtTable);
  table.hsa_amd_queue_intercept_create_fn = fake_queue_intercept_create;
  table.hsa_amd_queue_intercept_register_fn = fake_queue_intercept_register;
  table.hsa_amd_queue_set_priority_fn = fake_queue_set_priority;
  table.hsa_amd_queue_cu_set_mask_fn = fake_queue_cu_set_mask;
  table.hsa_amd_queue_create_fn = fake_amd_queue_create;
  return table;
}

HsaApiTable make_api_table(CoreApiTable *core, AmdExtTable *amd_ext = nullptr) {
  HsaApiTable table{};
  table.version.major_id = 1;
  table.version.minor_id = sizeof(HsaApiTable);
  table.core_ = core;
  table.amd_ext_ = amd_ext;
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

uint64_t kernel_descriptor_vaddr(const std::vector<uint8_t> &image, std::string_view symbol_name) {
  if (image.size() < sizeof(rocjitsu::Elf64_Ehdr))
    return 0;
  rocjitsu::Elf64_Ehdr ehdr{};
  std::memcpy(&ehdr, image.data(), sizeof(ehdr));
  if (ehdr.e_shoff > image.size() ||
      static_cast<uint64_t>(ehdr.e_shnum) * sizeof(rocjitsu::Elf64_Shdr) >
          image.size() - ehdr.e_shoff) {
    return 0;
  }
  const auto *shdrs = reinterpret_cast<const rocjitsu::Elf64_Shdr *>(image.data() + ehdr.e_shoff);
  for (uint16_t i = 0; i < ehdr.e_shnum; ++i) {
    const auto &symtab = shdrs[i];
    if (symtab.sh_type != rocjitsu::SHT_SYMTAB ||
        symtab.sh_entsize != sizeof(rocjitsu::Elf64_Sym) || symtab.sh_link >= ehdr.e_shnum)
      continue;
    const auto &strtab = shdrs[symtab.sh_link];
    const auto *symbols =
        reinterpret_cast<const rocjitsu::Elf64_Sym *>(image.data() + symtab.sh_offset);
    const char *names = reinterpret_cast<const char *>(image.data() + strtab.sh_offset);
    for (size_t j = 0; j < symtab.sh_size / sizeof(rocjitsu::Elf64_Sym); ++j) {
      if (symbols[j].st_name < strtab.sh_size &&
          std::string_view(names + symbols[j].st_name) == symbol_name) {
        return symbols[j].st_value;
      }
    }
  }
  return 0;
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
    g_first_tool_memory_calls = 0;
    g_first_tool_load_calls = 0;
    g_writer_calls = 0;
    g_intercept_handler = nullptr;
    g_intercept_data = nullptr;
    g_amd_queue_create_calls = 0;
    g_last_destroyed_queue = nullptr;
    g_priority_queue = nullptr;
    g_priority = HSA_AMD_QUEUE_PRIORITY_NORMAL;
    g_cu_mask_queue = nullptr;
    g_cu_mask_count = 0;
    g_cu_mask_first_word = 0;
    setenv("ROCJITSU_WAITCHECK", "1", 1);
    setenv("ROCJITSU_WAITCHECK_FAIL", "1", 1);
    setenv("ROCJITSU_WAITCHECK_MODE", "eager", 1);
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
    unsetenv("ROCJITSU_WAITCHECK_MODE");
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
  AmdExtTable amd_ext_{};
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

TEST_F(WaitcheckHooksTest, FailClosedRejectsIncompleteAnalysis) {
  const CapturedCall captured =
      load_memory(rocjitsu::waitcheck_test::make_gfx950_incomplete_direct_to_lds_code_object());

  EXPECT_EQ(captured.status, HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
  EXPECT_EQ(g_load_calls, 0);
  EXPECT_TRUE(contains(captured.stderr_text, "analysis incomplete")) << captured.stderr_text;
  EXPECT_TRUE(
      contains(captured.stderr_text, "direct-to-LDS producer range is not statically known"))
      << captured.stderr_text;
  EXPECT_FALSE(contains(captured.stderr_text, "waitcnt hazard")) << captured.stderr_text;
}

TEST_F(WaitcheckHooksTest, ReportOnlyModeDoesNotDescribeIncompleteAnalysisAsClean) {
  setenv("ROCJITSU_WAITCHECK_FAIL", "0", 1);
  const CapturedCall captured =
      load_memory(rocjitsu::waitcheck_test::make_gfx950_incomplete_direct_to_lds_code_object());

  EXPECT_EQ(captured.status, HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_load_calls, 1);
  EXPECT_TRUE(contains(captured.stderr_text, "analysis incomplete")) << captured.stderr_text;
  EXPECT_FALSE(contains(captured.stderr_text, "waitcnt hazard")) << captured.stderr_text;
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

TEST_F(WaitcheckHooksTest, DispatchModeChecksOnlySelectedKernelAndCachesIt) {
  OnUnload();
  loaded_ = false;
  setenv("ROCJITSU_WAITCHECK_MODE", "dispatch", 1);
  setenv("ROCJITSU_WAITCHECK_FAIL", "0", 1);

  core_ = make_core_table();
  amd_ext_ = make_amd_ext_table();
  table_ = make_api_table(&core_, &amd_ext_);
  ASSERT_TRUE(OnLoad(&table_, 1, 0, nullptr));
  loaded_ = true;

  std::vector<uint32_t> clean_words;
  rocjitsu::waitcheck_test::append_inst(clean_words, rocjitsu::waitcheck_test::global_load_b32(0));
  rocjitsu::waitcheck_test::append_inst(clean_words, rocjitsu::waitcheck_test::s_wait_loadcnt(0));
  rocjitsu::waitcheck_test::append_inst(clean_words, rocjitsu::waitcheck_test::v_mov_b32(1, 0));
  clean_words.push_back(0xBFB00000U); // s_endpgm
  std::vector<uint32_t> hazardous_words;
  rocjitsu::waitcheck_test::append_inst(hazardous_words,
                                        rocjitsu::waitcheck_test::global_load_b32(0));
  rocjitsu::waitcheck_test::append_inst(hazardous_words, rocjitsu::waitcheck_test::v_mov_b32(1, 0));
  hazardous_words.push_back(0xBFB00000U); // s_endpgm
  const auto bytes = rocjitsu::waitcheck_test::make_gfx_multi_kernel_code_object(
      {{"clean", clean_words}, {"hazardous", hazardous_words}},
      rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1200);
  const uint64_t clean_descriptor = kernel_descriptor_vaddr(bytes, "clean.kd");
  const uint64_t hazardous_descriptor = kernel_descriptor_vaddr(bytes, "hazardous.kd");
  ASSERT_NE(clean_descriptor, 0U);
  ASSERT_NE(hazardous_descriptor, 0U);

  hsa_code_object_reader_t reader{};
  ASSERT_EQ(core_.hsa_code_object_reader_create_from_memory_fn(bytes.data(), bytes.size(), &reader),
            HSA_STATUS_SUCCESS);
  constexpr hsa_executable_t executable{77};
  const CapturedCall load = capture_stderr([&] {
    return core_.hsa_executable_load_agent_code_object_fn(executable, {}, reader, nullptr, nullptr);
  });
  EXPECT_EQ(load.status, HSA_STATUS_SUCCESS);
  EXPECT_TRUE(load.stderr_text.empty()) << load.stderr_text;
  ASSERT_EQ(core_.hsa_executable_freeze_fn(executable, nullptr), HSA_STATUS_SUCCESS);

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(core_.hsa_queue_create_fn({}, 64, HSA_QUEUE_TYPE_MULTI, nullptr, nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(queue, &g_intercept_queue);
  ASSERT_NE(g_intercept_handler, nullptr);

  hsa_kernel_dispatch_packet_t packet{};
  packet.header = static_cast<uint16_t>(HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);
  packet.kernel_object = clean_descriptor + static_cast<uint64_t>(g_load_delta);
  const CapturedCall clean = capture_stderr([&] {
    g_intercept_handler(&packet, 1, 0, g_intercept_data, fake_packet_writer);
    return HSA_STATUS_SUCCESS;
  });
  packet.kernel_object = hazardous_descriptor + static_cast<uint64_t>(g_load_delta);
  const CapturedCall first_hazard = capture_stderr([&] {
    g_intercept_handler(&packet, 1, 0, g_intercept_data, fake_packet_writer);
    return HSA_STATUS_SUCCESS;
  });
  const CapturedCall cached_hazard = capture_stderr([&] {
    g_intercept_handler(&packet, 1, 0, g_intercept_data, fake_packet_writer);
    return HSA_STATUS_SUCCESS;
  });

  EXPECT_TRUE(clean.stderr_text.empty()) << clean.stderr_text;
  EXPECT_TRUE(contains(first_hazard.stderr_text, "kernel hazardous")) << first_hazard.stderr_text;
  EXPECT_TRUE(contains(first_hazard.stderr_text, "1 waitcnt hazard")) << first_hazard.stderr_text;
  EXPECT_TRUE(cached_hazard.stderr_text.empty()) << cached_hazard.stderr_text;
  EXPECT_EQ(g_writer_calls, 3);
  EXPECT_EQ(core_.hsa_executable_destroy_fn(executable), HSA_STATUS_SUCCESS);
  EXPECT_EQ(core_.hsa_code_object_reader_destroy_fn(reader), HSA_STATUS_SUCCESS);
}

TEST_F(WaitcheckHooksTest, DispatchModeWrapsDirectInterceptQueueCreationAndRestoresIt) {
  OnUnload();
  loaded_ = false;
  setenv("ROCJITSU_WAITCHECK_MODE", "dispatch", 1);

  core_ = make_core_table();
  amd_ext_ = make_amd_ext_table();
  table_ = make_api_table(&core_, &amd_ext_);
  ASSERT_TRUE(OnLoad(&table_, 1, 0, nullptr));
  loaded_ = true;

  ASSERT_NE(amd_ext_.hsa_amd_queue_intercept_create_fn, fake_queue_intercept_create);
  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(amd_ext_.hsa_amd_queue_intercept_create_fn({}, 64, HSA_QUEUE_TYPE_MULTI, nullptr,
                                                       nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(queue, &g_intercept_queue);
  EXPECT_NE(g_intercept_handler, nullptr);

  OnUnload();
  loaded_ = false;
  EXPECT_EQ(amd_ext_.hsa_amd_queue_intercept_create_fn, fake_queue_intercept_create);
}

TEST_F(WaitcheckHooksTest, DispatchModeReplacesDescriptorComputeQueueAndRestoresIt) {
  OnUnload();
  loaded_ = false;
  setenv("ROCJITSU_WAITCHECK_MODE", "dispatch", 1);

  core_ = make_core_table();
  amd_ext_ = make_amd_ext_table();
  table_ = make_api_table(&core_, &amd_ext_);
  ASSERT_TRUE(OnLoad(&table_, 1, 0, nullptr));
  loaded_ = true;

  ASSERT_NE(amd_ext_.hsa_amd_queue_create_fn, fake_amd_queue_create);
  constexpr uint32_t cu_mask = 0x5u;
  hsa_amd_queue_create_desc_t desc{};
  desc.version = HSA_AMD_QUEUE_CREATE_DESC_VERSION;
  desc.engine_type = HSA_AMD_QUEUE_ENGINE_COMPUTE;
  desc.queue_size_bytes = 64u * sizeof(hsa_kernel_dispatch_packet_t);
  desc.priority = HSA_AMD_QUEUE_PRIORITY_HIGH;
  desc.engine.compute = {
      .cu_mask = &cu_mask,
      .type = HSA_QUEUE_TYPE_MULTI,
      .private_segment_size = HSA_AMD_PRIVATE_SEGMENT_SIZE_DEFAULT,
      .cu_mask_count = 1,
      .reserved = {},
  };

  ASSERT_EQ(amd_ext_.hsa_amd_queue_create_fn({}, &desc, 1), HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_amd_queue_create_calls, 1);
  EXPECT_EQ(desc.queue, &g_intercept_queue);
  EXPECT_EQ(g_last_destroyed_queue, &g_descriptor_queue);
  EXPECT_NE(g_intercept_handler, nullptr);
  EXPECT_EQ(g_priority_queue, &g_intercept_queue);
  EXPECT_EQ(g_priority, HSA_AMD_QUEUE_PRIORITY_HIGH);
  EXPECT_EQ(g_cu_mask_queue, &g_intercept_queue);
  EXPECT_EQ(g_cu_mask_count, 1u);
  EXPECT_EQ(g_cu_mask_first_word, cu_mask);

  OnUnload();
  loaded_ = false;
  EXPECT_EQ(amd_ext_.hsa_amd_queue_create_fn, fake_amd_queue_create);
}

TEST(WaitcheckHooksValidationTest, RejectsTruncatedCoreTable) {
  CoreApiTable core = make_core_table();
  core.version.minor_id = offsetof(CoreApiTable, hsa_executable_load_agent_code_object_fn);
  HsaApiTable table = make_api_table(&core);
  EXPECT_FALSE(OnLoad(&table, 1, 0, nullptr));
}

} // namespace
