// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "hsa/hsa_api_trace_minimal.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/rj_gfx1250_b0_to_a0.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "support/gfx1250_test_code_object.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <mutex>
#include <new>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

extern "C" bool OnLoad(HsaApiTable *, uint64_t, uint64_t, const char *const *);
extern "C" void OnUnload();
extern "C" size_t rj_test_retained_executable_buffer_count();
extern "C" void rj_test_clear_retained_storage();
extern "C" void rj_test_log_translation(uint64_t source_id, size_t changed);
extern "C" void
rj_test_log_translation_diagnostic(uint64_t source_id,
                                   const rj_gfx1250_b0_to_a0_diagnostic_t *diagnostic);
extern "C" uint64_t rj_test_translation_count();
extern "C" uint64_t rj_test_translation_memo_bytes();
extern "C" void rj_test_set_translation_memo_capacity(uint64_t bytes);
extern "C" void rj_test_close_translation_gate();
extern "C" void rj_test_open_translation_gate();
extern "C" uint64_t rj_test_translation_waiters();
extern "C" void rj_test_force_next_translation_status(int status);
extern "C" uint64_t rj_test_dump_path_count();
extern "C" uint64_t rj_test_sample_fingerprint(const void *bytes, size_t size);
extern "C" void rj_test_retain_completed_claims(bool retain);
extern "C" void rj_test_fail_next_memo_admission(int stage);
extern "C" void rj_test_set_pretranslation_root(const char *root);
extern "C" uint64_t rj_test_pretranslation_hits();

namespace {

constexpr hsa_agent_t kA0Agent{1250};
constexpr hsa_executable_t kExecutable{42};
constexpr uint32_t kAsicRevisionAttribute = 0xA012;

struct ReaderView {
  const uint8_t *bytes;
  size_t size;
};

using VendorReaderCreate = hsa_status_t (*)(hsa_file_t, size_t, size_t, hsa_code_object_reader_t *);

struct FakeVendorLoaderTable {
  void (*query_host_address)();
  void (*query_segment_descriptors)();
  void (*query_executable)();
  void (*iterate_loaded_code_objects)();
  void (*loaded_code_object_get_info)();
  VendorReaderCreate create_reader_from_file;
};

uint64_t g_next_reader = 1;
std::unordered_map<uint64_t, ReaderView> g_readers;
std::vector<uint8_t> g_loaded_bytes;
hsa_code_object_reader_t g_loaded_reader{};
hsa_status_t g_reader_destroy_status = HSA_STATUS_SUCCESS;
hsa_status_t g_executable_destroy_status = HSA_STATUS_SUCCESS;
hsa_status_t g_load_agent_status = HSA_STATUS_SUCCESS;
uint32_t g_asic_revision = 0;
int g_reader_destroy_calls = 0;
int g_load_agent_calls = 0;
int g_load_program_calls = 0;
int g_load_deprecated_calls = 0;
int g_get_extension_table_calls = 0;
int g_vendor_reader_calls = 0;
hsa_file_t g_vendor_file = -1;
size_t g_vendor_offset = 0;
size_t g_vendor_size = 0;
bool g_throw_from_deprecated_load = false;

void reset_fakes() {
  g_next_reader = 1;
  g_readers.clear();
  g_loaded_bytes.clear();
  g_loaded_reader = {};
  g_reader_destroy_status = HSA_STATUS_SUCCESS;
  g_executable_destroy_status = HSA_STATUS_SUCCESS;
  g_load_agent_status = HSA_STATUS_SUCCESS;
  g_asic_revision = 0;
  g_reader_destroy_calls = 0;
  g_load_agent_calls = 0;
  g_load_program_calls = 0;
  g_load_deprecated_calls = 0;
  g_get_extension_table_calls = 0;
  g_vendor_reader_calls = 0;
  g_vendor_file = -1;
  g_vendor_offset = 0;
  g_vendor_size = 0;
  g_throw_from_deprecated_load = false;
}

hsa_status_t HSA_API fake_iterate_agents(hsa_status_t (*callback)(hsa_agent_t, void *),
                                         void *data) {
  return callback == nullptr ? HSA_STATUS_ERROR_INVALID_ARGUMENT : callback(kA0Agent, data);
}

hsa_status_t HSA_API fake_agent_get_info(hsa_agent_t agent, hsa_agent_info_t attribute,
                                         void *value) {
  if (agent.handle != kA0Agent.handle || value == nullptr ||
      static_cast<uint32_t>(attribute) != kAsicRevisionAttribute)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *static_cast<uint32_t *>(value) = g_asic_revision;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_agent_iterate_isas(hsa_agent_t agent,
                                             hsa_status_t (*callback)(hsa_isa_t, void *),
                                             void *data) {
  if (agent.handle != kA0Agent.handle || callback == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  return callback(hsa_isa_t{1250}, data);
}

hsa_status_t HSA_API fake_isa_get_info(hsa_isa_t isa, hsa_isa_info_t attribute, void *value) {
  constexpr char kIsaName[] = "amdgcn-amd-amdhsa--gfx1250";
  if (isa.handle != 1250 || value == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (attribute == HSA_ISA_INFO_NAME_LENGTH) {
    *static_cast<uint32_t *>(value) = sizeof(kIsaName);
    return HSA_STATUS_SUCCESS;
  }
  if (attribute == HSA_ISA_INFO_NAME) {
    std::memcpy(value, kIsaName, sizeof(kIsaName));
    return HSA_STATUS_SUCCESS;
  }
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

// Serialises the loader fakes' shared bookkeeping. Only the concurrency cases call
// them from more than one thread, but the state they touch is plain globals, so
// without this those cases would be racing rather than testing.
std::mutex g_fake_mutex;

hsa_status_t HSA_API fake_create_file(hsa_file_t, hsa_code_object_reader_t *reader) {
  if (reader == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  std::lock_guard fake_lock(g_fake_mutex);
  *reader = hsa_code_object_reader_t{g_next_reader++};
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_create_memory(const void *bytes, size_t size,
                                        hsa_code_object_reader_t *reader) {
  if (bytes == nullptr || size == 0 || reader == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  std::lock_guard fake_lock(g_fake_mutex);
  *reader = hsa_code_object_reader_t{g_next_reader++};
  g_readers.emplace(reader->handle, ReaderView{static_cast<const uint8_t *>(bytes), size});
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_destroy_reader(hsa_code_object_reader_t reader) {
  std::lock_guard fake_lock(g_fake_mutex);
  ++g_reader_destroy_calls;
  if (g_reader_destroy_status == HSA_STATUS_SUCCESS)
    g_readers.erase(reader.handle);
  return g_reader_destroy_status;
}

hsa_status_t HSA_API fake_destroy_executable(hsa_executable_t) {
  return g_executable_destroy_status;
}

hsa_status_t HSA_API fake_load_agent(hsa_executable_t, hsa_agent_t, hsa_code_object_reader_t reader,
                                     const char *, hsa_loaded_code_object_t *loaded) {
  std::lock_guard fake_lock(g_fake_mutex);
  ++g_load_agent_calls;
  g_loaded_reader = reader;
  if (g_load_agent_status != HSA_STATUS_SUCCESS)
    return g_load_agent_status;
  const auto it = g_readers.find(reader.handle);
  if (it == g_readers.end())
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER;
  g_loaded_bytes.assign(it->second.bytes, it->second.bytes + it->second.size);
  if (loaded != nullptr)
    loaded->handle = 99;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_load_program(hsa_executable_t, hsa_code_object_reader_t, const char *,
                                       hsa_loaded_code_object_t *) {
  ++g_load_program_calls;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_load_deprecated(hsa_executable_t, hsa_agent_t, hsa_code_object_t,
                                          const char *) {
  ++g_load_deprecated_calls;
  if (g_throw_from_deprecated_load)
    throw std::bad_alloc();
  return HSA_STATUS_SUCCESS;
}

hsa_status_t fake_vendor_reader_create(hsa_file_t file, size_t offset, size_t size,
                                       hsa_code_object_reader_t *reader) {
  ++g_vendor_reader_calls;
  g_vendor_file = file;
  g_vendor_offset = offset;
  g_vendor_size = size;
  if (reader == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *reader = hsa_code_object_reader_t{g_next_reader++};
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_get_extension_table(uint16_t extension, uint16_t version_major,
                                              size_t table_length, void *table) {
  ++g_get_extension_table_calls;
  constexpr size_t reader_field_end =
      offsetof(FakeVendorLoaderTable, create_reader_from_file) + sizeof(VendorReaderCreate);
  if (extension == HSA_EXTENSION_AMD_LOADER && version_major == 1 && table != nullptr &&
      table_length >= reader_field_end) {
    static_cast<FakeVendorLoaderTable *>(table)->create_reader_from_file =
        fake_vendor_reader_create;
  }
  return HSA_STATUS_SUCCESS;
}

std::vector<uint8_t> make_invalid_gfx1250_elf() {
  std::vector<uint8_t> image(sizeof(rocjitsu::Elf64_Ehdr), 0);
  auto *header = reinterpret_cast<rocjitsu::Elf64_Ehdr *>(image.data());
  std::memcpy(header->e_ident, rocjitsu::EI_MAGIC, rocjitsu::EI_MAGIC_SIZE);
  header->e_ident[rocjitsu::EI_CLASS] = rocjitsu::ELFCLASS64;
  header->e_machine = rocjitsu::EM_AMDGPU;
  header->e_flags = rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1250;
  return image;
}

/// @brief A directory that exists for one test case.
class ScopedTempDirectory {
public:
  ScopedTempDirectory() {
    char pattern[] = "/tmp/rocjitsu-hotswap-dump-XXXXXX";
    const char *created = mkdtemp(pattern);
    if (created != nullptr)
      path_ = created;
  }
  ScopedTempDirectory(const ScopedTempDirectory &) = delete;
  ScopedTempDirectory &operator=(const ScopedTempDirectory &) = delete;
  ~ScopedTempDirectory() {
    if (path_.empty())
      return;
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  const std::string &path() const { return path_; }

private:
  std::string path_;
};

size_t count_occurrences(const std::string &text, std::string_view needle) {
  size_t count = 0;
  for (size_t at = text.find(needle); at != std::string::npos;
       at = text.find(needle, at + needle.size()))
    ++count;
  return count;
}

/// @brief An invalid object whose identity differs from every other salt.
std::vector<uint8_t> make_invalid_gfx1250_elf(uint8_t salt) {
  std::vector<uint8_t> image = make_invalid_gfx1250_elf();
  image.push_back(salt);
  return image;
}

void expect_failure_dump(const std::string &log_text, const std::vector<uint8_t> &source) {
  std::smatch match;
  const std::regex path_pattern(
      R"(path=([^;\s]+); please file a bug report and attach this code object)");
  ASSERT_TRUE(std::regex_search(log_text, match, path_pattern)) << log_text;
  const std::string path = match[1].str();
  std::ifstream input(path, std::ios::binary);
  ASSERT_TRUE(input) << path;
  const std::vector<uint8_t> dumped((std::istreambuf_iterator<char>(input)),
                                    std::istreambuf_iterator<char>());
  EXPECT_EQ(dumped, source);
  input.close();
  EXPECT_EQ(unlink(path.c_str()), 0) << path;
}

#ifdef GFX1250_B0_TO_A0_FIXTURE
std::vector<uint8_t> read_translation_fixture() {
  std::ifstream input(GFX1250_B0_TO_A0_FIXTURE, std::ios::binary);
  if (!input)
    return {};
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(input),
                              std::istreambuf_iterator<char>());
}
#endif

struct FakeApi {
  CoreApiTable core{};
  HsaApiTable table{};

  FakeApi() {
    core.version.minor_id = sizeof(core);
    table.version.minor_id = sizeof(table);
    table.core_ = &core;
    core.hsa_iterate_agents_fn = fake_iterate_agents;
    core.hsa_agent_get_info_fn = fake_agent_get_info;
    core.hsa_agent_iterate_isas_fn = fake_agent_iterate_isas;
    core.hsa_isa_get_info_alt_fn = fake_isa_get_info;
    core.hsa_code_object_reader_create_from_file_fn = fake_create_file;
    core.hsa_code_object_reader_create_from_memory_fn = fake_create_memory;
    core.hsa_code_object_reader_destroy_fn = fake_destroy_reader;
    core.hsa_executable_destroy_fn = fake_destroy_executable;
    core.hsa_executable_load_agent_code_object_fn = fake_load_agent;
    core.hsa_executable_load_program_code_object_fn = fake_load_program;
    core.hsa_executable_load_code_object_fn = fake_load_deprecated;
    core.hsa_system_get_major_extension_table_fn = fake_get_extension_table;
  }
};

class HsaHotswapHookTest : public ::testing::Test {
protected:
  void SetUp() override {
    for (const char *name : kIsolatedEnvironment) {
      const char *value = std::getenv(name);
      saved_environment_.emplace_back(name, value != nullptr ? std::optional<std::string>(value)
                                                             : std::nullopt);
      (void)unsetenv(name);
    }
    OnUnload();
    // Production storage is process-lifetime (not freed on reinstall), so clear it
    // here to isolate the retention lifecycle between test cases.
    rj_test_clear_retained_storage();
    reset_fakes();
  }
  void TearDown() override {
    OnUnload();
    rj_test_clear_retained_storage();
    for (const auto &[name, value] : saved_environment_) {
      if (value)
        (void)setenv(name, value->c_str(), 1);
      else
        (void)unsetenv(name);
    }
    saved_environment_.clear();
  }

  // Loads @p source through a fresh reader, the way a caller that re-registers
  // the same object does -- a new handle every time, so nothing but the content
  // itself can connect one load to the next.
  hsa_status_t load_through_new_reader(const std::vector<uint8_t> &source,
                                       hsa_executable_t executable = kExecutable) {
    hsa_code_object_reader_t reader{};
    const hsa_status_t create_status = api.core.hsa_code_object_reader_create_from_memory_fn(
        source.data(), source.size(), &reader);
    if (create_status != HSA_STATUS_SUCCESS)
      return create_status;
    return api.core.hsa_executable_load_agent_code_object_fn(executable, kA0Agent, reader, nullptr,
                                                             nullptr);
  }

  // Every case runs with these cleared and gets whatever the process had back,
  // so one case cannot decide what a later one observes.
  static constexpr std::array<const char *, 4> kIsolatedEnvironment = {
      "HSA_HOTSWAP_VERBOSE", "HSA_HOTSWAP_DUMP_SOURCE", "HSA_HOTSWAP_DUMP_DIR", "TMPDIR"};

  FakeApi api;
  std::vector<std::pair<const char *, std::optional<std::string>>> saved_environment_;
};

TEST_F(HsaHotswapHookTest, InstallsOnlyTheEightEntryEagerSurface) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));

  EXPECT_NE(api.core.hsa_code_object_reader_create_from_file_fn, fake_create_file);
  EXPECT_NE(api.core.hsa_code_object_reader_create_from_memory_fn, fake_create_memory);
  EXPECT_NE(api.core.hsa_code_object_reader_destroy_fn, fake_destroy_reader);
  EXPECT_NE(api.core.hsa_executable_destroy_fn, fake_destroy_executable);
  EXPECT_NE(api.core.hsa_executable_load_agent_code_object_fn, fake_load_agent);
  EXPECT_NE(api.core.hsa_executable_load_program_code_object_fn, fake_load_program);
  EXPECT_NE(api.core.hsa_executable_load_code_object_fn, fake_load_deprecated);
  EXPECT_NE(api.core.hsa_system_get_major_extension_table_fn, fake_get_extension_table);

  EXPECT_EQ(api.core.hsa_iterate_agents_fn, fake_iterate_agents);
  EXPECT_EQ(api.core.hsa_agent_get_info_fn, fake_agent_get_info);
  EXPECT_EQ(api.core.hsa_agent_iterate_isas_fn, fake_agent_iterate_isas);
  EXPECT_EQ(api.core.hsa_isa_get_info_alt_fn, fake_isa_get_info);
}

TEST_F(HsaHotswapHookTest, UnloadRestoresAndAllowsReinstall) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  OnUnload();
  EXPECT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn, fake_create_memory);
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn, fake_load_agent);

  EXPECT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
}

TEST_F(HsaHotswapHookTest, EmitsConcurrentTranslationRecordsAtomically) {
  constexpr size_t kThreadCount = 8;
  constexpr size_t kRecordsPerThread = 32;

  ASSERT_EQ(setenv("HSA_HOTSWAP_VERBOSE", "1", 1), 0);
  testing::internal::CaptureStderr();
  std::vector<std::thread> workers;
  workers.reserve(kThreadCount);
  for (size_t thread_index = 0; thread_index < kThreadCount; ++thread_index) {
    workers.emplace_back([thread_index] {
      for (size_t record_index = 0; record_index < kRecordsPerThread; ++record_index) {
        const uint64_t source_id = thread_index * kRecordsPerThread + record_index;
        rj_test_log_translation(source_id, record_index);
      }
    });
  }
  for (auto &worker : workers)
    worker.join();
  const std::string log_text = testing::internal::GetCapturedStderr();
  ASSERT_EQ(unsetenv("HSA_HOTSWAP_VERBOSE"), 0);

  const std::regex record_pattern(
      R"(^\[hsa-hotswap-rj\] eager translation source_id=fnv1a64:[0-9a-f]{16} input_revision=b0 output_revision=a0 outcome=translated changed=[0-9]+ input_bytes=64 output_bytes=96 translation_status=0 status=0$)");
  std::istringstream lines(log_text);
  std::string line;
  size_t record_count = 0;
  while (std::getline(lines, line)) {
    EXPECT_TRUE(std::regex_match(line, record_pattern)) << line;
    ++record_count;
  }
  EXPECT_EQ(record_count, kThreadCount * kRecordsPerThread);
}

// A rejected install must not leave the hook stuck. install() commits g_state.core
// only AFTER its one fallible step (building the saved-table snapshot), so a
// rejected or throwing install cannot latch core non-null and permanently reject
// every later install. A failed OnLoad is followed by a successful one here.
TEST_F(HsaHotswapHookTest, RejectedInstallLeavesHookInstallable) {
  // Missing a required lower entry makes install() reject before committing state.
  FakeApi incomplete;
  incomplete.core.hsa_executable_load_agent_code_object_fn = nullptr;
  EXPECT_FALSE(OnLoad(&incomplete.table, 0, 0, nullptr));

  // The hook is not latched: a subsequent valid install succeeds.
  EXPECT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  EXPECT_NE(api.core.hsa_executable_load_agent_code_object_fn, fake_load_agent);
}

TEST_F(HsaHotswapHookTest, UsesImmutableSnapshotForForwardedA0Load) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  std::vector<uint8_t> source{1, 2, 3, 4};
  const std::vector<uint8_t> expected = source;
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(
      api.core.hsa_code_object_reader_create_from_memory_fn(source.data(), source.size(), &reader),
      HSA_STATUS_SUCCESS);
  source.assign(source.size(), 0xff);

  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kExecutable, kA0Agent, reader,
                                                              nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_loaded_bytes, expected);
  EXPECT_NE(g_loaded_reader.handle, reader.handle);
  EXPECT_EQ(g_load_agent_calls, 1);
  EXPECT_EQ(g_reader_destroy_calls, 1);
  EXPECT_EQ(g_readers.count(g_loaded_reader.handle), 0u);
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(kExecutable), HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_reader_destroy_calls, 1);
}

#ifdef GFX1250_B0_TO_A0_FIXTURE
TEST_F(HsaHotswapHookTest, TranslatesGfx1250AndRetainsOutputUntilDestroy) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  const std::vector<uint8_t> source = read_translation_fixture();
  ASSERT_FALSE(source.empty()) << GFX1250_B0_TO_A0_FIXTURE;

  hsa_code_object_reader_t reader{};
  ASSERT_EQ(
      api.core.hsa_code_object_reader_create_from_memory_fn(source.data(), source.size(), &reader),
      HSA_STATUS_SUCCESS);
  ASSERT_EQ(setenv("HSA_HOTSWAP_VERBOSE", "1", 1), 0);
  testing::internal::CaptureStderr();
  const hsa_status_t load_status = api.core.hsa_executable_load_agent_code_object_fn(
      kExecutable, kA0Agent, reader, nullptr, nullptr);
  const std::string log_text = testing::internal::GetCapturedStderr();
  ASSERT_EQ(unsetenv("HSA_HOTSWAP_VERBOSE"), 0);
  ASSERT_EQ(load_status, HSA_STATUS_SUCCESS);

  ASSERT_GE(g_loaded_bytes.size(), rocjitsu::EI_MAGIC_SIZE);
  EXPECT_EQ(std::memcmp(g_loaded_bytes.data(), rocjitsu::EI_MAGIC, rocjitsu::EI_MAGIC_SIZE), 0);
  EXPECT_NE(g_loaded_bytes, source);
  EXPECT_NE(g_loaded_reader.handle, reader.handle);
  EXPECT_EQ(g_load_agent_calls, 1);
  EXPECT_EQ(rj_test_retained_executable_buffer_count(), 1u);

  uint64_t identity = 14695981039346656037ULL;
  for (uint8_t byte : source) {
    identity ^= byte;
    identity *= 1099511628211ULL;
  }
  std::ostringstream expected_identity;
  expected_identity << "source_id=fnv1a64:" << std::hex << std::setfill('0') << std::setw(16)
                    << identity;
  EXPECT_NE(log_text.find("[hsa-hotswap-rj] eager translation "), std::string::npos) << log_text;
  EXPECT_NE(log_text.find(expected_identity.str()), std::string::npos) << log_text;
  EXPECT_NE(log_text.find(" input_revision=b0 output_revision=a0 outcome=translated changed="),
            std::string::npos)
      << log_text;
  EXPECT_EQ(log_text.find(" changed=0 "), std::string::npos) << log_text;
  EXPECT_NE(log_text.find(" translation_status=0 status=0"), std::string::npos) << log_text;

  EXPECT_EQ(api.core.hsa_executable_destroy_fn(kExecutable), HSA_STATUS_SUCCESS);
  EXPECT_EQ(rj_test_retained_executable_buffer_count(), 0u);
}
#endif

TEST_F(HsaHotswapHookTest, TranslationFailureDoesNotLoadOrRetain) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  const std::vector<uint8_t> source = make_invalid_gfx1250_elf();

  hsa_code_object_reader_t reader{};
  ASSERT_EQ(
      api.core.hsa_code_object_reader_create_from_memory_fn(source.data(), source.size(), &reader),
      HSA_STATUS_SUCCESS);
  ScopedTempDirectory capture_directory;
  ASSERT_FALSE(capture_directory.path().empty());
  ASSERT_EQ(setenv("HSA_HOTSWAP_DUMP_SOURCE", "1", 1), 0);
  ASSERT_EQ(setenv("HSA_HOTSWAP_DUMP_DIR", capture_directory.path().c_str(), 1), 0);
  testing::internal::CaptureStderr();
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kExecutable, kA0Agent, reader,
                                                              nullptr, nullptr),
            HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
  const std::string log_text = testing::internal::GetCapturedStderr();
  EXPECT_EQ(g_load_agent_calls, 0);
  EXPECT_EQ(rj_test_retained_executable_buffer_count(), 0u);
  EXPECT_NE(log_text.find("[hsa-hotswap-rj] error: eager translation "), std::string::npos)
      << log_text;
  EXPECT_NE(log_text.find("[hsa-hotswap-rj] error: translation diagnostic "), std::string::npos)
      << log_text;
  EXPECT_NE(log_text.find(" severity=error kind=input-invalid-code-object "), std::string::npos)
      << log_text;
  EXPECT_NE(log_text.find(" message=source is not a valid gfx1250 AMDGPU code object"),
            std::string::npos)
      << log_text;
  EXPECT_NE(log_text.find(" outcome=translation_failed "), std::string::npos) << log_text;
  EXPECT_NE(log_text.find(" translation_status="), std::string::npos) << log_text;
  EXPECT_NE(log_text.find(" status="), std::string::npos) << log_text;
  expect_failure_dump(log_text, source);
}

TEST_F(HsaHotswapHookTest, RendersTranslatorDiagnosticsAndDumpsFailedSource) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  constexpr auto conversion =
      rocjitsu::cdna5::build_sop1(rocjitsu::cdna5::kSBarrierSignalIsfirstSop1, {.ssrc0 = 195});
  constexpr uint32_t kEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 2> text = {conversion[0], kEndpgm};
  const auto source = rocjitsu::test_support::make_gfx1250_code_object(text);
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(
      api.core.hsa_code_object_reader_create_from_memory_fn(source.data(), source.size(), &reader),
      HSA_STATUS_SUCCESS);

  ScopedTempDirectory capture_directory;
  ASSERT_FALSE(capture_directory.path().empty());
  ASSERT_EQ(setenv("HSA_HOTSWAP_DUMP_SOURCE", "1", 1), 0);
  ASSERT_EQ(setenv("HSA_HOTSWAP_DUMP_DIR", capture_directory.path().c_str(), 1), 0);
  testing::internal::CaptureStderr();
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kExecutable, kA0Agent, reader,
                                                              nullptr, nullptr),
            HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
  const std::string log_text = testing::internal::GetCapturedStderr();
  EXPECT_EQ(g_load_agent_calls, 0);
  EXPECT_NE(log_text.find("[hsa-hotswap-rj] error: translation diagnostic "), std::string::npos)
      << log_text;
  EXPECT_NE(log_text.find(" severity=error kind=translator-expand-failed "), std::string::npos)
      << log_text;
  EXPECT_NE(log_text.find(" guest_offset=.text+0x0 mnemonic=s_barrier_signal_isfirst "),
            std::string::npos)
      << log_text;
  EXPECT_NE(log_text.find(" message="), std::string::npos) << log_text;
  EXPECT_NE(log_text.find(" required=Use a different barrier id, or signal it without the first-"
                          "signal form."),
            std::string::npos)
      << log_text;
  expect_failure_dump(log_text, source);
}

TEST_F(HsaHotswapHookTest, DiagnosticPrefixMatchesWarningSeverity) {
  const rj_gfx1250_b0_to_a0_diagnostic_t diagnostic{
      "warning", "translator-data-only", 0, 0, "", "test warning", 0,
  };
  testing::internal::CaptureStderr();
  rj_test_log_translation_diagnostic(0x1234, &diagnostic);
  const std::string log_text = testing::internal::GetCapturedStderr();
  EXPECT_NE(log_text.find("[hsa-hotswap-rj] warning: translation diagnostic "), std::string::npos)
      << log_text;
  EXPECT_NE(log_text.find(" severity=warning kind=translator-data-only "), std::string::npos)
      << log_text;
  EXPECT_EQ(log_text.find("[hsa-hotswap-rj] error:"), std::string::npos) << log_text;
}

TEST_F(HsaHotswapHookTest, RendersRequiredWorkDiagnostic) {
  const rj_gfx1250_b0_to_a0_diagnostic_t diagnostic{
      "error", "translator-expand-missing", 1, 8, "v_test", "required step", 1,
  };
  testing::internal::CaptureStderr();
  rj_test_log_translation_diagnostic(0x1234, &diagnostic);
  const std::string log_text = testing::internal::GetCapturedStderr();
  EXPECT_NE(log_text.find(" severity=error kind=translator-expand-missing "), std::string::npos)
      << log_text;
  EXPECT_NE(log_text.find(" guest_offset=.text+0x8 mnemonic=v_test "), std::string::npos)
      << log_text;
  EXPECT_NE(log_text.find(" required=required step"), std::string::npos) << log_text;
  EXPECT_EQ(log_text.find(" message="), std::string::npos) << log_text;
}

// The translated backing storage retained for an A0 load must SURVIVE OnUnload()
// (ROCr destroys its loader after OnUnload but before closing the DSO, so the bytes
// the loader still references must outlive OnUnload) AND a runtime-generation
// reinstall (OnLoad again): a consumer whose lifetime is not bounded by the HSA
// generation -- rocprofiler-register, which finalizes its records at process-exit
// atexit -- can still reference these bytes, so a reinstall must not free them. They
// are released only at executable destroy (or process exit). Regression guard for
// the process-lifetime storage-retention lifecycle.
TEST_F(HsaHotswapHookTest, FailedTranslationWritesNothingByDefault) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  const std::vector<uint8_t> source = make_invalid_gfx1250_elf();

  hsa_code_object_reader_t reader{};
  ASSERT_EQ(
      api.core.hsa_code_object_reader_create_from_memory_fn(source.data(), source.size(), &reader),
      HSA_STATUS_SUCCESS);
  testing::internal::CaptureStderr();
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kExecutable, kA0Agent, reader,
                                                              nullptr, nullptr),
            HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
  const std::string log_text = testing::internal::GetCapturedStderr();

  // Nothing was written, and the report says how to ask for the artifact.
  EXPECT_EQ(log_text.find("; please file a bug report and attach this code object"),
            std::string::npos)
      << log_text;
  EXPECT_NE(log_text.find("set HSA_HOTSWAP_DUMP_SOURCE=1 to save the source code object"),
            std::string::npos)
      << log_text;
  EXPECT_EQ(rj_test_dump_path_count(), 0u);
}

TEST_F(HsaHotswapHookTest, CapturedSourceHonorsTheConfiguredDirectory) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  const std::vector<uint8_t> source = make_invalid_gfx1250_elf();
  ScopedTempDirectory directory;
  ASSERT_FALSE(directory.path().empty());
  ASSERT_EQ(setenv("HSA_HOTSWAP_DUMP_SOURCE", "1", 1), 0);
  ASSERT_EQ(setenv("HSA_HOTSWAP_DUMP_DIR", directory.path().c_str(), 1), 0);

  hsa_code_object_reader_t reader{};
  ASSERT_EQ(
      api.core.hsa_code_object_reader_create_from_memory_fn(source.data(), source.size(), &reader),
      HSA_STATUS_SUCCESS);
  testing::internal::CaptureStderr();
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kExecutable, kA0Agent, reader,
                                                              nullptr, nullptr),
            HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
  const std::string log_text = testing::internal::GetCapturedStderr();

  std::smatch match;
  const std::regex path_pattern(
      R"(path=([^;\s]+); please file a bug report and attach this code object)");
  ASSERT_TRUE(std::regex_search(log_text, match, path_pattern)) << log_text;
  EXPECT_EQ(match[1].str().rfind(directory.path() + "/", 0), 0u) << match[1].str();
  expect_failure_dump(log_text, source);
}

TEST_F(HsaHotswapHookTest, RepeatedEnvironmentalFailuresCaptureOneArtifact) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  const std::vector<uint8_t> source = make_invalid_gfx1250_elf();
  ScopedTempDirectory capture_directory;
  ASSERT_FALSE(capture_directory.path().empty());
  ASSERT_EQ(setenv("HSA_HOTSWAP_DUMP_SOURCE", "1", 1), 0);
  ASSERT_EQ(setenv("HSA_HOTSWAP_DUMP_DIR", capture_directory.path().c_str(), 1), 0);
  constexpr size_t kLoads = 8;

  // A throwing translator is not remembered, so every load translates again.
  // Capture must still leave exactly one file behind.
  for (size_t load = 0; load < kLoads; ++load) {
    rj_test_force_next_translation_status(1 /* ROCJITSU_STATUS_ERROR */);
    hsa_code_object_reader_t reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(source.data(), source.size(),
                                                                    &reader),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kExecutable, kA0Agent, reader,
                                                                nullptr, nullptr),
              HSA_STATUS_ERROR_INVALID_CODE_OBJECT)
        << load;
  }

  // The point of the case: every load really did translate again, and the eight
  // attempts still left one artifact.
  EXPECT_EQ(rj_test_translation_count(), kLoads);
  EXPECT_EQ(rj_test_dump_path_count(), 1u);
}

TEST_F(HsaHotswapHookTest, AnOutOfResourcesFailureCapturesNothing) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  const std::vector<uint8_t> source = make_invalid_gfx1250_elf();
  ScopedTempDirectory capture_directory;
  ASSERT_FALSE(capture_directory.path().empty());
  ASSERT_EQ(setenv("HSA_HOTSWAP_DUMP_SOURCE", "1", 1), 0);
  ASSERT_EQ(setenv("HSA_HOTSWAP_DUMP_DIR", capture_directory.path().c_str(), 1), 0);
  rj_test_force_next_translation_status(3 /* ROCJITSU_STATUS_OUT_OF_RESOURCES */);

  hsa_code_object_reader_t reader{};
  ASSERT_EQ(
      api.core.hsa_code_object_reader_create_from_memory_fn(source.data(), source.size(), &reader),
      HSA_STATUS_SUCCESS);
  testing::internal::CaptureStderr();
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kExecutable, kA0Agent, reader,
                                                              nullptr, nullptr),
            HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
  const std::string log_text = testing::internal::GetCapturedStderr();

  // Copying a large input after an allocation failure adds pressure without
  // diagnosing anything: the bytes are not why it failed.
  EXPECT_EQ(rj_test_dump_path_count(), 0u);
  EXPECT_EQ(log_text.find("; please file a bug report and attach this code object"),
            std::string::npos)
      << log_text;
}

TEST_F(HsaHotswapHookTest, EnablingCaptureAfterAHintStillProducesTheArtifact) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  const std::vector<uint8_t> source = make_invalid_gfx1250_elf();

  // The hint names the variable that turns capture on. Acting on it inside the
  // same process must work: a hint that consumed the source's one capture slot
  // would tell the operator to do something that then cannot succeed. Both
  // attempts force a status the memo refuses to remember, so the second load
  // really translates again instead of replaying the first verdict.
  rj_test_force_next_translation_status(1 /* ROCJITSU_STATUS_ERROR */);
  testing::internal::CaptureStderr();
  EXPECT_EQ(load_through_new_reader(source), HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
  const std::string hint_text = testing::internal::GetCapturedStderr();
  ASSERT_NE(hint_text.find("set HSA_HOTSWAP_DUMP_SOURCE=1 to save the source code object"),
            std::string::npos)
      << hint_text;
  ASSERT_EQ(rj_test_dump_path_count(), 0u);

  ScopedTempDirectory capture_directory;
  ASSERT_FALSE(capture_directory.path().empty());
  ASSERT_EQ(setenv("HSA_HOTSWAP_DUMP_SOURCE", "1", 1), 0);
  ASSERT_EQ(setenv("HSA_HOTSWAP_DUMP_DIR", capture_directory.path().c_str(), 1), 0);
  rj_test_force_next_translation_status(1 /* ROCJITSU_STATUS_ERROR */);
  testing::internal::CaptureStderr();
  EXPECT_EQ(load_through_new_reader(source), HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
  const std::string capture_text = testing::internal::GetCapturedStderr();

  EXPECT_EQ(rj_test_dump_path_count(), 1u);
  expect_failure_dump(capture_text, source);
}

TEST_F(HsaHotswapHookTest, CorrectingTheCaptureDirectoryRetriesTheArtifact) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  const std::vector<uint8_t> source = make_invalid_gfx1250_elf();
  ASSERT_EQ(setenv("HSA_HOTSWAP_DUMP_SOURCE", "1", 1), 0);
  ASSERT_EQ(setenv("HSA_HOTSWAP_DUMP_DIR", "/nonexistent-rocjitsu-hotswap-dump", 1), 0);

  // Forced so the memo does not remember it: every load below translates again
  // and retries the write, which is what makes the reporting bound matter.
  constexpr size_t kFailedLoads = 4;
  testing::internal::CaptureStderr();
  for (size_t load = 0; load < kFailedLoads; ++load) {
    rj_test_force_next_translation_status(1 /* ROCJITSU_STATUS_ERROR */);
    EXPECT_EQ(load_through_new_reader(source), HSA_STATUS_ERROR_INVALID_CODE_OBJECT) << load;
  }
  const std::string failed_text = testing::internal::GetCapturedStderr();
  // The destination stays broken, so the write keeps being retried -- but the
  // operator is told about it once, not once per load.
  EXPECT_EQ(count_occurrences(failed_text, "could not be saved"), 1u) << failed_text;
  ASSERT_EQ(rj_test_dump_path_count(), 0u);

  // A capture that could not be written must not spend the source's slot: the
  // operator fixes the destination and the next failure produces the artifact.
  ScopedTempDirectory directory;
  ASSERT_FALSE(directory.path().empty());
  ASSERT_EQ(setenv("HSA_HOTSWAP_DUMP_DIR", directory.path().c_str(), 1), 0);
  rj_test_force_next_translation_status(1 /* ROCJITSU_STATUS_ERROR */);
  testing::internal::CaptureStderr();
  EXPECT_EQ(load_through_new_reader(source), HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
  const std::string capture_text = testing::internal::GetCapturedStderr();

  EXPECT_EQ(rj_test_dump_path_count(), 1u);
  expect_failure_dump(capture_text, source);
}

TEST_F(HsaHotswapHookTest, CaptureFallsBackToTmpdirWhenNoDirectoryIsNamed) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  const std::vector<uint8_t> source = make_invalid_gfx1250_elf();
  ScopedTempDirectory directory;
  ASSERT_FALSE(directory.path().empty());
  // HSA_HOTSWAP_DUMP_DIR stays unset: TMPDIR is the next choice, and /tmp only
  // after that.
  ASSERT_EQ(setenv("HSA_HOTSWAP_DUMP_SOURCE", "1", 1), 0);
  ASSERT_EQ(setenv("TMPDIR", directory.path().c_str(), 1), 0);

  testing::internal::CaptureStderr();
  EXPECT_EQ(load_through_new_reader(source), HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
  const std::string log_text = testing::internal::GetCapturedStderr();

  std::smatch match;
  const std::regex path_pattern(
      R"(path=([^;\s]+); please file a bug report and attach this code object)");
  ASSERT_TRUE(std::regex_search(log_text, match, path_pattern)) << log_text;
  EXPECT_EQ(match[1].str().rfind(directory.path() + "/", 0), 0u) << match[1].str();
  expect_failure_dump(log_text, source);
}

TEST_F(HsaHotswapHookTest, CaptureStopsAtTheSourceCapAndSaysWhy) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  ScopedTempDirectory capture_directory;
  ASSERT_FALSE(capture_directory.path().empty());
  ASSERT_EQ(setenv("HSA_HOTSWAP_DUMP_SOURCE", "1", 1), 0);
  ASSERT_EQ(setenv("HSA_HOTSWAP_DUMP_DIR", capture_directory.path().c_str(), 1), 0);
  // kMaxCapturedSources in the hook. One more source than the registry holds.
  constexpr size_t kCap = 32;

  testing::internal::CaptureStderr();
  for (size_t index = 0; index <= kCap; ++index) {
    const std::vector<uint8_t> source = make_invalid_gfx1250_elf(static_cast<uint8_t>(index));
    EXPECT_EQ(load_through_new_reader(source), HSA_STATUS_ERROR_INVALID_CODE_OBJECT) << index;
  }
  const std::string log_text = testing::internal::GetCapturedStderr();

  // Refusing silently would be indistinguishable from a capture the operator
  // then cannot find, so the cap explains itself -- once.
  EXPECT_EQ(rj_test_dump_path_count(), kCap);
  EXPECT_EQ(count_occurrences(log_text, "have already been captured"), 1u) << log_text;
}

TEST_F(HsaHotswapHookTest, RetainedStorageSurvivesUnloadAndReinstall) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  ASSERT_EQ(rj_test_retained_executable_buffer_count(), 0u);

  // A forwarded A0 load (non-gfx1250 source) retains its owned bytes under the
  // executable for the loader's lifetime.
  const std::vector<uint8_t> source{1, 2, 3, 4};
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(
      api.core.hsa_code_object_reader_create_from_memory_fn(source.data(), source.size(), &reader),
      HSA_STATUS_SUCCESS);
  ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kExecutable, kA0Agent, reader,
                                                              nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(rj_test_retained_executable_buffer_count(), 1u);

  // OnUnload must NOT free the buffer: the loader still references it until ROCr
  // destroys the loader (which happens after OnUnload, before the DSO closes).
  OnUnload();
  EXPECT_EQ(rj_test_retained_executable_buffer_count(), 1u);

  // A reinstall (next runtime generation) must NOT free the previous generation's
  // storage either: an old-generation profiler record can still point into it.
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  EXPECT_EQ(rj_test_retained_executable_buffer_count(), 1u);

  // Storage is released when the executable is destroyed.
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(kExecutable), HSA_STATUS_SUCCESS);
  EXPECT_EQ(rj_test_retained_executable_buffer_count(), 0u);
}

TEST_F(HsaHotswapHookTest, B0LoadsUseOnlyTheOriginalApi) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  g_asic_revision = 1;
  const std::vector<uint8_t> source{1, 2, 3, 4};
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(
      api.core.hsa_code_object_reader_create_from_memory_fn(source.data(), source.size(), &reader),
      HSA_STATUS_SUCCESS);

  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kExecutable, kA0Agent, reader,
                                                              nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_loaded_reader.handle, reader.handle);
  EXPECT_EQ(g_loaded_bytes, source);
  EXPECT_EQ(g_reader_destroy_calls, 0);

  EXPECT_EQ(
      api.core.hsa_executable_load_program_code_object_fn(kExecutable, reader, nullptr, nullptr),
      HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_load_program_calls, 1);
  EXPECT_EQ(api.core.hsa_executable_load_code_object_fn(kExecutable, kA0Agent, hsa_code_object_t{1},
                                                        nullptr),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_load_deprecated_calls, 1);
}

TEST_F(HsaHotswapHookTest, FailedLoadDestroysTransientReaderImmediately) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  const std::vector<uint8_t> source{1, 2, 3, 4};
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(
      api.core.hsa_code_object_reader_create_from_memory_fn(source.data(), source.size(), &reader),
      HSA_STATUS_SUCCESS);

  g_load_agent_status = HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kExecutable, kA0Agent, reader,
                                                              nullptr, nullptr),
            HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
  EXPECT_EQ(g_reader_destroy_calls, 1);
  EXPECT_EQ(g_readers.count(g_loaded_reader.handle), 0u);
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(kExecutable), HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_reader_destroy_calls, 1);
}

// A failed lower load must NOT retain the translated bytes. The failure may be a
// pre-publication rejection (null/invalid executable, profile/ISA mismatch) where
// ROCr never referenced the bytes, and there is no successful executable-destroy to
// release a stranded blob -- so retaining on failure would grow storage without
// bound across repeated invalid loads. Retention happens only on success.
TEST_F(HsaHotswapHookTest, FailedLoadDoesNotRetainAndDoesNotGrow) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  ASSERT_EQ(rj_test_retained_executable_buffer_count(), 0u);

  const std::vector<uint8_t> source{1, 2, 3, 4};
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(
      api.core.hsa_code_object_reader_create_from_memory_fn(source.data(), source.size(), &reader),
      HSA_STATUS_SUCCESS);

  // Repeated failing loads must not accumulate retained buffers.
  g_load_agent_status = HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kExecutable, kA0Agent, reader,
                                                                nullptr, nullptr),
              HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
  }
  EXPECT_EQ(rj_test_retained_executable_buffer_count(), 0u);
}

// A successful load retains the bytes until the executable is destroyed (ROCr
// aliases the ELF pointer, so a live loaded object references them for its lifetime).
TEST_F(HsaHotswapHookTest, SuccessfulLoadRetainsUntilDestroy) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  ASSERT_EQ(rj_test_retained_executable_buffer_count(), 0u);

  const std::vector<uint8_t> source{1, 2, 3, 4};
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(
      api.core.hsa_code_object_reader_create_from_memory_fn(source.data(), source.size(), &reader),
      HSA_STATUS_SUCCESS);
  ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kExecutable, kA0Agent, reader,
                                                              nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(rj_test_retained_executable_buffer_count(), 1u);
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(kExecutable), HSA_STATUS_SUCCESS);
  EXPECT_EQ(rj_test_retained_executable_buffer_count(), 0u);
}

TEST_F(HsaHotswapHookTest, FailedReaderDestroyKeepsCapturedBytes) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  const std::vector<uint8_t> source{1, 2, 3, 4};
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(
      api.core.hsa_code_object_reader_create_from_memory_fn(source.data(), source.size(), &reader),
      HSA_STATUS_SUCCESS);

  g_reader_destroy_status = HSA_STATUS_ERROR;
  EXPECT_EQ(api.core.hsa_code_object_reader_destroy_fn(reader), HSA_STATUS_ERROR);
  g_reader_destroy_status = HSA_STATUS_SUCCESS;
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kExecutable, kA0Agent, reader,
                                                              nullptr, nullptr),
            HSA_STATUS_SUCCESS);
}

TEST_F(HsaHotswapHookTest, FileCaptureFailureDestroysCreatedReader) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  hsa_code_object_reader_t reader{777};
  EXPECT_EQ(api.core.hsa_code_object_reader_create_from_file_fn(-1, &reader),
            HSA_STATUS_ERROR_OUT_OF_RESOURCES);
  EXPECT_EQ(reader.handle, 0u);
  EXPECT_EQ(g_reader_destroy_calls, 1);
}

TEST_F(HsaHotswapHookTest, VendorLoaderReaderIsReplacedCapturedAndForwarded) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));

  FakeVendorLoaderTable loader{};
  ASSERT_EQ(api.core.hsa_system_get_major_extension_table_fn(HSA_EXTENSION_AMD_LOADER, 1,
                                                             sizeof(loader), &loader),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(loader.create_reader_from_file, nullptr);
  EXPECT_NE(loader.create_reader_from_file, fake_vendor_reader_create);
  EXPECT_EQ(g_get_extension_table_calls, 1);

  constexpr std::array<uint8_t, 8> kFileBytes = {0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80};
  constexpr size_t kOffset = 2;
  constexpr size_t kSize = 4;
  std::FILE *file = std::tmpfile();
  ASSERT_NE(file, nullptr);
  ASSERT_EQ(std::fwrite(kFileBytes.data(), 1, kFileBytes.size(), file), kFileBytes.size());
  ASSERT_EQ(std::fflush(file), 0);
  const int file_descriptor = fileno(file);
  ASSERT_GE(file_descriptor, 0);

  hsa_code_object_reader_t reader{};
  ASSERT_EQ(loader.create_reader_from_file(file_descriptor, kOffset, kSize, &reader),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_vendor_reader_calls, 1);
  EXPECT_EQ(g_vendor_file, file_descriptor);
  EXPECT_EQ(g_vendor_offset, kOffset);
  EXPECT_EQ(g_vendor_size, kSize);
  ASSERT_EQ(std::fclose(file), 0);

  ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kExecutable, kA0Agent, reader,
                                                              nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  const std::vector<uint8_t> expected(kFileBytes.begin() + kOffset,
                                      kFileBytes.begin() + kOffset + kSize);
  EXPECT_EQ(g_loaded_bytes, expected);
  EXPECT_EQ(rj_test_retained_executable_buffer_count(), 1u);
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(kExecutable), HSA_STATUS_SUCCESS);
  EXPECT_EQ(rj_test_retained_executable_buffer_count(), 0u);

  EXPECT_EQ(loader.create_reader_from_file(file_descriptor, 0, 0, nullptr),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(g_vendor_reader_calls, 1);

  FakeVendorLoaderTable short_loader{};
  short_loader.create_reader_from_file = fake_vendor_reader_create;
  constexpr size_t reader_field_end =
      offsetof(FakeVendorLoaderTable, create_reader_from_file) + sizeof(VendorReaderCreate);
  ASSERT_GT(reader_field_end, 0u);
  ASSERT_EQ(api.core.hsa_system_get_major_extension_table_fn(HSA_EXTENSION_AMD_LOADER, 1,
                                                             reader_field_end - 1, &short_loader),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(short_loader.create_reader_from_file, fake_vendor_reader_create);
}

// A null output reader pointer is rejected BEFORE forwarding to the lower/vendor
// API, which may write through it without checking. Fast-fail with
// INVALID_ARGUMENT and never invoke the underlying create (g_next_reader unchanged).
TEST_F(HsaHotswapHookTest, RejectsNullReaderPointerBeforeForwarding) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  const uint64_t reader_id_before = g_next_reader;

  const std::vector<uint8_t> source{1, 2, 3, 4};
  EXPECT_EQ(
      api.core.hsa_code_object_reader_create_from_memory_fn(source.data(), source.size(), nullptr),
      HSA_STATUS_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(api.core.hsa_code_object_reader_create_from_file_fn(0, nullptr),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);

  // The underlying create was never called, so no reader handle was allocated.
  EXPECT_EQ(g_next_reader, reader_id_before);
}

TEST_F(HsaHotswapHookTest, RefusesAgentlessAndDeprecatedA0Loads) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  EXPECT_EQ(api.core.hsa_executable_load_program_code_object_fn(
                kExecutable, hsa_code_object_reader_t{1}, nullptr, nullptr),
            HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS);
  EXPECT_EQ(g_load_program_calls, 0);
  EXPECT_EQ(api.core.hsa_executable_load_code_object_fn(kExecutable, kA0Agent, hsa_code_object_t{1},
                                                        nullptr),
            HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS);
  EXPECT_EQ(g_load_deprecated_calls, 0);
}

TEST_F(HsaHotswapHookTest, ContainsExceptionsAtTheHsaBoundary) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  g_asic_revision = 1;
  g_throw_from_deprecated_load = true;
  ASSERT_EQ(unsetenv("HSA_HOTSWAP_VERBOSE"), 0);
  testing::internal::CaptureStderr();
  EXPECT_EQ(api.core.hsa_executable_load_code_object_fn(kExecutable, kA0Agent, hsa_code_object_t{1},
                                                        nullptr),
            HSA_STATUS_ERROR_OUT_OF_RESOURCES);
  const std::string log_text = testing::internal::GetCapturedStderr();
  EXPECT_EQ(g_load_deprecated_calls, 1);
  EXPECT_NE(log_text.find("[hsa-hotswap-rj] error: "), std::string::npos) << log_text;
  EXPECT_NE(log_text.find("operation=hsa_executable_load_code_object"), std::string::npos)
      << log_text;
  EXPECT_NE(log_text.find("exception=std::bad_alloc"), std::string::npos) << log_text;
  EXPECT_NE(log_text.find("status="), std::string::npos) << log_text;
}

TEST_F(HsaHotswapHookTest, CallbackApiSnapshotIsSafeDuringUnload) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  auto create = api.core.hsa_code_object_reader_create_from_memory_fn;
  std::atomic<bool> started{false};
  std::atomic<bool> stop{false};
  std::atomic<int> unexpected_statuses{0};
  const std::vector<uint8_t> source{1, 2, 3, 4};

  std::thread worker([&] {
    started.store(true, std::memory_order_release);
    while (!stop.load(std::memory_order_acquire)) {
      hsa_code_object_reader_t reader{};
      const hsa_status_t status = create(source.data(), source.size(), &reader);
      if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_ERROR)
        unexpected_statuses.fetch_add(1, std::memory_order_relaxed);
    }
  });
  while (!started.load(std::memory_order_acquire)) {
  }
  OnUnload();
  stop.store(true, std::memory_order_release);
  worker.join();

  EXPECT_EQ(unexpected_statuses.load(std::memory_order_relaxed), 0);
}

// Translation is a pure function of the source bytes, so a caller that registers
// the same object repeatedly should pay for it once. The cases below assert on the
// number of translations actually performed rather than on elapsed time: two loads
// producing equal bytes is also true when both translated, and only the count
// distinguishes reuse from repetition.
class HsaHotswapMemoTest : public HsaHotswapHookTest {
protected:
  // Spin until @p expected threads are asleep on an in-flight translation, or a
  // generous deadline passes. Returns what was actually observed so the caller can
  // release the gate before asserting on it.
  static uint64_t wait_for_memo_waiters(uint64_t expected) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    uint64_t waiters = rj_test_translation_waiters();
    while (waiters < expected && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      waiters = rj_test_translation_waiters();
    }
    return waiters;
  }

  // Build a second object of the SAME size as @p base whose sampled windows are
  // identical but whose bytes are not, by disturbing one byte of trailing padding
  // that the sampling happens to miss. Returns empty if no such byte exists, so a
  // caller can fail loudly rather than quietly test something weaker. The
  // fingerprint is queried rather than assumed: hand-picking an offset would stop
  // testing a collision the day the window layout changed.
  static std::vector<uint8_t> make_fingerprint_twin(const std::vector<uint8_t> &base) {
    constexpr size_t kPadding = 4096;
    std::vector<uint8_t> original = base;
    original.resize(base.size() + kPadding, 0);
    const uint64_t target = rj_test_sample_fingerprint(original.data(), original.size());
    for (size_t offset = base.size(); offset < original.size(); ++offset) {
      std::vector<uint8_t> twin = original;
      twin[offset] = 0x5a;
      if (rj_test_sample_fingerprint(twin.data(), twin.size()) == target)
        return twin;
    }
    return {};
  }

  // Overwrite the first instruction of @p base with @p word. The result is still a
  // structurally valid gfx1250 object, so it reaches translation rather than being
  // rejected up front -- which is how the two ways a translation can fail are told
  // apart: a body the translator cannot render dispatchable, versus one that makes
  // it throw. Returns empty if .text cannot be located.
  static std::vector<uint8_t> patch_first_instruction(const std::vector<uint8_t> &base,
                                                      uint32_t word) {
    if (base.size() < sizeof(rocjitsu::Elf64_Ehdr))
      return {};
    rocjitsu::Elf64_Ehdr header{};
    std::memcpy(&header, base.data(), sizeof(header));
    // Bounds-check what the fixture claims before believing it. A truncated or
    // malformed one should fail the case, not read past the buffer and take the
    // whole test process down with it.
    const size_t table_bytes = size_t{header.e_shnum} * sizeof(rocjitsu::Elf64_Shdr);
    if (header.e_shoff > base.size() || table_bytes > base.size() - header.e_shoff ||
        header.e_shstrndx >= header.e_shnum)
      return {};
    const auto *sections =
        reinterpret_cast<const rocjitsu::Elf64_Shdr *>(base.data() + header.e_shoff);
    const size_t names_offset = sections[header.e_shstrndx].sh_offset;
    if (names_offset >= base.size())
      return {};
    const char *names = reinterpret_cast<const char *>(base.data() + names_offset);
    const size_t names_limit = base.size() - names_offset;
    for (uint16_t index = 0; index < header.e_shnum; ++index) {
      if (sections[index].sh_name >= names_limit)
        return {};
      const char *name = names + sections[index].sh_name;
      const size_t name_limit = names_limit - sections[index].sh_name;
      if (std::memchr(name, '\0', name_limit) == nullptr)
        return {}; // Unterminated: comparing it would run off the end.
      if (std::strcmp(name, ".text") != 0)
        continue;
      if (sections[index].sh_offset + sizeof(word) > base.size())
        return {};
      std::vector<uint8_t> patched = base;
      std::memcpy(patched.data() + sections[index].sh_offset, &word, sizeof(word));
      return patched;
    }
    return {};
  }
};

// The ticket this exists for: a caller registered two distinct code objects 509
// times each across four devices and paid for 2036 translations.
TEST_F(HsaHotswapMemoTest, RemembersARefusalInsteadOfRepeatingIt) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  ASSERT_EQ(unsetenv("HSA_HOTSWAP_VERBOSE"), 0);
  const std::vector<uint8_t> source = make_invalid_gfx1250_elf();
  constexpr size_t kLoads = 256;

  testing::internal::CaptureStderr();
  for (size_t load = 0; load < kLoads; ++load)
    EXPECT_EQ(load_through_new_reader(source), HSA_STATUS_ERROR_INVALID_CODE_OBJECT) << load;
  const std::string log_text = testing::internal::GetCapturedStderr();

  // A refusal is a property of the bytes: reaching it once is enough, and every
  // later load must reach the same answer without re-deriving it.
  EXPECT_EQ(rj_test_translation_count(), 1u);
  EXPECT_EQ(g_load_agent_calls, 0);
  EXPECT_EQ(rj_test_retained_executable_buffer_count(), 0u);
  // The refusal is reported the once it is reached. The 255 reuses that follow
  // are the memo working, not new failures, so they stay on the verbose channel.
  EXPECT_EQ(count_occurrences(log_text, " outcome=translation_failed "), 1u) << log_text;
  EXPECT_EQ(count_occurrences(log_text, " outcome=reused_failure "), 0u) << log_text;
}

TEST_F(HsaHotswapMemoTest, VerboseLoggingStillShowsEveryRememberedRefusal) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  ASSERT_EQ(setenv("HSA_HOTSWAP_VERBOSE", "1", 1), 0);
  const std::vector<uint8_t> source = make_invalid_gfx1250_elf();
  constexpr size_t kLoads = 4;

  testing::internal::CaptureStderr();
  for (size_t load = 0; load < kLoads; ++load)
    EXPECT_EQ(load_through_new_reader(source), HSA_STATUS_ERROR_INVALID_CODE_OBJECT) << load;
  const std::string log_text = testing::internal::GetCapturedStderr();

  EXPECT_EQ(count_occurrences(log_text, " outcome=translation_failed "), 1u) << log_text;
  EXPECT_EQ(count_occurrences(log_text, " outcome=reused_failure "), kLoads - 1) << log_text;
}

#ifdef GFX1250_B0_TO_A0_FIXTURE
TEST_F(HsaHotswapMemoTest, RepeatedLoadsOfOneObjectTranslateOnce) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  const std::vector<uint8_t> source = read_translation_fixture();
  ASSERT_FALSE(source.empty()) << GFX1250_B0_TO_A0_FIXTURE;
  constexpr size_t kLoads = 256;

  ASSERT_EQ(load_through_new_reader(source), HSA_STATUS_SUCCESS);
  const std::vector<uint8_t> first_output = g_loaded_bytes;
  ASSERT_NE(first_output, source);
  ASSERT_EQ(rj_test_translation_count(), 1u);

  for (size_t load = 1; load < kLoads; ++load) {
    g_loaded_bytes.clear();
    ASSERT_EQ(load_through_new_reader(source), HSA_STATUS_SUCCESS) << load;
    ASSERT_EQ(g_loaded_bytes, first_output) << load;
  }

  EXPECT_EQ(rj_test_translation_count(), 1u);
  EXPECT_EQ(g_load_agent_calls, static_cast<int>(kLoads));
  // One entry is held, not one per load: the reuses shared that buffer rather
  // than each taking a copy of it. An entry is its source plus its output,
  // because identity is settled by comparing bytes rather than trusting a digest.
  EXPECT_EQ(rj_test_translation_memo_bytes(), source.size() + first_output.size());
}

// Identity is exact. Two objects the sampling cannot tell apart -- same size, same
// fingerprint, different bytes -- must not be mistaken for each other, because the
// consequence is running the wrong code. The count is the whole check: a confused
// pair is served from the memo and never reaches a second translation.
TEST_F(HsaHotswapMemoTest, ObjectsInOneFingerprintBucketAreNotConfused) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  const std::vector<uint8_t> base = read_translation_fixture();
  ASSERT_FALSE(base.empty()) << GFX1250_B0_TO_A0_FIXTURE;

  std::vector<uint8_t> altered = base;
  altered.resize(base.size() + 4096, 0);
  const std::vector<uint8_t> twin = make_fingerprint_twin(base);
  ASSERT_FALSE(twin.empty()) << "no colliding twin found; the sampling changed";
  ASSERT_EQ(altered.size(), twin.size());
  ASSERT_NE(altered, twin);
  ASSERT_EQ(rj_test_sample_fingerprint(altered.data(), altered.size()),
            rj_test_sample_fingerprint(twin.data(), twin.size()));

  ASSERT_EQ(load_through_new_reader(altered), HSA_STATUS_SUCCESS);
  ASSERT_EQ(load_through_new_reader(twin), HSA_STATUS_SUCCESS);
  EXPECT_EQ(rj_test_translation_count(), 2u);
}

TEST_F(HsaHotswapMemoTest, DistinctObjectsAreTranslatedOncePerContent) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  const std::vector<uint8_t> base = read_translation_fixture();
  ASSERT_FALSE(base.empty()) << GFX1250_B0_TO_A0_FIXTURE;

  // Trailing bytes past the end of the ELF change the content without changing
  // what translation makes of it, which is what a second distinct object needs to
  // be here: different bytes, same outcome.
  std::vector<std::vector<uint8_t>> objects{base, base};
  objects[1].push_back(0x5a);
  constexpr size_t kLoadsPerObject = 64;

  for (size_t load = 0; load < kLoadsPerObject; ++load)
    for (const auto &object : objects)
      ASSERT_EQ(load_through_new_reader(object), HSA_STATUS_SUCCESS) << load;

  EXPECT_EQ(rj_test_translation_count(), objects.size());
}

// A control for the case above: without the memo these loads would each translate,
// so the count must follow the number of DISTINCT objects and not the number of
// loads. Clearing the memo mid-run makes the same object translate again, which
// shows the count is measuring translation rather than being pinned at one.
TEST_F(HsaHotswapMemoTest, ClearingTheMemoMakesTheSameObjectTranslateAgain) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  const std::vector<uint8_t> source = read_translation_fixture();
  ASSERT_FALSE(source.empty()) << GFX1250_B0_TO_A0_FIXTURE;

  ASSERT_EQ(load_through_new_reader(source), HSA_STATUS_SUCCESS);
  ASSERT_EQ(load_through_new_reader(source), HSA_STATUS_SUCCESS);
  ASSERT_EQ(rj_test_translation_count(), 1u);

  rj_test_clear_retained_storage();
  // Without this the case proves nothing: the clear also resets the counter, so
  // the final 1 below would read the same whether a translation happened or not.
  ASSERT_EQ(rj_test_translation_count(), 0u);

  ASSERT_EQ(load_through_new_reader(source), HSA_STATUS_SUCCESS);
  EXPECT_EQ(rj_test_translation_count(), 1u);
}

TEST_F(HsaHotswapMemoTest, ConcurrentLoadsOfOneObjectTranslateOnce) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  const std::vector<uint8_t> source = read_translation_fixture();
  ASSERT_FALSE(source.empty()) << GFX1250_B0_TO_A0_FIXTURE;
  constexpr size_t kThreadCount = 8;

  // Hold the first translation open. The fixture translates in well under a
  // millisecond, so left to itself the claimer would publish before its peers
  // even reached the memo, and the count below would read 1 whether or not
  // single-flight waiting works at all.
  rj_test_close_translation_gate();

  std::atomic<int> failures{0};
  std::vector<std::thread> workers;
  workers.reserve(kThreadCount);
  for (size_t thread_index = 0; thread_index < kThreadCount; ++thread_index) {
    workers.emplace_back([&] {
      if (load_through_new_reader(source) != HSA_STATUS_SUCCESS)
        failures.fetch_add(1, std::memory_order_relaxed);
    });
  }

  // Exactly one thread claims and is held at the gate; the rest must be asleep
  // inside the memo rather than translating alongside it. Sample before opening
  // the gate, and open it whatever the sample said -- an assertion that fired
  // here while threads were still held would abort the process instead of failing
  // the case.
  const uint64_t waiters = wait_for_memo_waiters(kThreadCount - 1);
  const uint64_t translations_while_held = rj_test_translation_count();
  rj_test_open_translation_gate();
  for (auto &worker : workers)
    worker.join();

  EXPECT_EQ(waiters, kThreadCount - 1);
  EXPECT_EQ(translations_while_held, 1u);

  EXPECT_EQ(failures.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(rj_test_translation_waiters(), 0u);
  // The waiters resumed onto the published entry rather than each translating.
  EXPECT_EQ(rj_test_translation_count(), 1u);
}

// Two objects that merely sample alike must translate side by side. The
// fingerprint only narrows, so waiting on it would park an independent object
// behind a translation that for the image RCCL loads runs for 197 s -- and
// because the sampling is deterministic, so is the stall.
TEST_F(HsaHotswapMemoTest, ObjectsThatShareAFingerprintDoNotWaitForEachOther) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  const std::vector<uint8_t> base = read_translation_fixture();
  ASSERT_FALSE(base.empty()) << GFX1250_B0_TO_A0_FIXTURE;

  const std::vector<uint8_t> first = [&] {
    std::vector<uint8_t> padded = base;
    padded.resize(base.size() + 4096, 0);
    return padded;
  }();
  const std::vector<uint8_t> second = make_fingerprint_twin(base);
  ASSERT_FALSE(second.empty()) << "no colliding twin found; the sampling changed";
  ASSERT_EQ(first.size(), second.size());
  ASSERT_NE(first, second);
  ASSERT_EQ(rj_test_sample_fingerprint(first.data(), first.size()),
            rj_test_sample_fingerprint(second.data(), second.size()));

  // Hold both translations open. If the claim were keyed on the fingerprint, the
  // second object would be asleep in the memo instead of translating.
  rj_test_close_translation_gate();
  std::atomic<int> failures{0};
  std::vector<std::thread> workers;
  for (const std::vector<uint8_t> *object : {&first, &second}) {
    workers.emplace_back([&, object] {
      if (load_through_new_reader(*object) != HSA_STATUS_SUCCESS)
        failures.fetch_add(1, std::memory_order_relaxed);
    });
  }

  // Both should reach the gate, so two translations are in flight and nobody is
  // waiting. Sample, then release regardless of what was seen.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (rj_test_translation_count() < 2 && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  const uint64_t translations_while_held = rj_test_translation_count();
  const uint64_t waiters = rj_test_translation_waiters();
  rj_test_open_translation_gate();
  for (auto &worker : workers)
    worker.join();

  EXPECT_EQ(failures.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(translations_while_held, 2u) << "a colliding object was serialised behind the other";
  EXPECT_EQ(waiters, 0u);
}

// The other way a claim ends. A claimer that gives up without publishing must
// release its waiters, and one of them must then go on to do the work -- if the
// claim leaked instead, every later load of these bytes would sleep forever.
TEST_F(HsaHotswapMemoTest, WaitersResumeWhenAClaimIsReleasedWithoutPublishing) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  const std::vector<uint8_t> source = read_translation_fixture();
  ASSERT_FALSE(source.empty()) << GFX1250_B0_TO_A0_FIXTURE;
  constexpr size_t kThreadCount = 4;

  // The first translation reports an environmental failure, which is not
  // remembered, so its claim is released rather than published.
  rj_test_force_next_translation_status(3 /* ROCJITSU_STATUS_OUT_OF_RESOURCES */);
  rj_test_close_translation_gate();

  std::atomic<int> succeeded{0};
  std::vector<std::thread> workers;
  workers.reserve(kThreadCount);
  for (size_t thread_index = 0; thread_index < kThreadCount; ++thread_index) {
    workers.emplace_back([&] {
      if (load_through_new_reader(source) == HSA_STATUS_SUCCESS)
        succeeded.fetch_add(1, std::memory_order_relaxed);
    });
  }

  const uint64_t waiters = wait_for_memo_waiters(kThreadCount - 1);
  rj_test_open_translation_gate();
  for (auto &worker : workers)
    worker.join();

  EXPECT_EQ(waiters, kThreadCount - 1);
  // One load hit the forced failure; the rest resumed and one of them translated
  // for real, so the object is usable again despite the transient fault.
  EXPECT_EQ(succeeded.load(std::memory_order_relaxed), static_cast<int>(kThreadCount) - 1);
  EXPECT_EQ(rj_test_translation_waiters(), 0u);
  EXPECT_GE(rj_test_translation_count(), 2u);
}

// Deterministic failures are verdicts on the bytes and are remembered. The
// header-only object used above never reaches translation at all, so it cannot
// exercise either path.
TEST_F(HsaHotswapMemoTest, NonDispatchableAndInvalidInstructionVerdictsAreRemembered) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  const std::vector<uint8_t> base = read_translation_fixture();
  ASSERT_FALSE(base.empty()) << GFX1250_B0_TO_A0_FIXTURE;

  // A zero opcode translates to completion but yields nothing dispatchable.
  const std::vector<uint8_t> non_dispatchable = patch_first_instruction(base, 0x00000000u);
  ASSERT_FALSE(non_dispatchable.empty()) << "could not locate .text";
  for (size_t load = 0; load < 4; ++load)
    ASSERT_EQ(load_through_new_reader(non_dispatchable), HSA_STATUS_ERROR_INVALID_CODE_OBJECT)
        << load;
  EXPECT_EQ(rj_test_translation_count(), 1u) << "a verdict on the bytes was re-derived";

  // An invalid instruction is now a structured, deterministic translation
  // diagnostic, so repeated loads reuse the same negative verdict.
  rj_test_clear_retained_storage();
  const std::vector<uint8_t> invalid_instruction = patch_first_instruction(base, 0xffffffffu);
  ASSERT_FALSE(invalid_instruction.empty());
  for (size_t load = 0; load < 4; ++load)
    ASSERT_EQ(load_through_new_reader(invalid_instruction), HSA_STATUS_ERROR_INVALID_CODE_OBJECT)
        << load;
  EXPECT_EQ(rj_test_translation_count(), 1u) << "an invalid-instruction verdict was re-derived";
}

// A failure that says nothing about the bytes must not become a verdict on them.
// The translator reports an allocation it could not make with the same kind of
// status as a real refusal, and remembering one would refuse a perfectly valid
// object for the life of the process.
TEST_F(HsaHotswapMemoTest, ATransientFailureIsNotRememberedAndTheNextLoadSucceeds) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  const std::vector<uint8_t> source = read_translation_fixture();
  ASSERT_FALSE(source.empty()) << GFX1250_B0_TO_A0_FIXTURE;

  rj_test_force_next_translation_status(3 /* ROCJITSU_STATUS_OUT_OF_RESOURCES */);
  EXPECT_EQ(load_through_new_reader(source), HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
  EXPECT_EQ(rj_test_translation_count(), 1u);
  EXPECT_EQ(rj_test_translation_memo_bytes(), 0u) << "a transient failure was remembered";

  // The control: the same bytes, and nothing was learned from the bad moment.
  EXPECT_EQ(load_through_new_reader(source), HSA_STATUS_SUCCESS);
  EXPECT_EQ(rj_test_translation_count(), 2u);
  EXPECT_GT(rj_test_translation_memo_bytes(), 0u);
}

// A ceiling of zero switches the memo off entirely: nothing is looked up and
// nothing is kept, so every load translates exactly as it did before the memo
// existed.
TEST_F(HsaHotswapMemoTest, ACapacityOfZeroDisablesTheMemo) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  const std::vector<uint8_t> source = read_translation_fixture();
  ASSERT_FALSE(source.empty()) << GFX1250_B0_TO_A0_FIXTURE;

  rj_test_set_translation_memo_capacity(0);
  constexpr size_t kLoads = 4;
  for (size_t load = 0; load < kLoads; ++load)
    ASSERT_EQ(load_through_new_reader(source), HSA_STATUS_SUCCESS) << load;

  EXPECT_EQ(rj_test_translation_count(), kLoads);
  EXPECT_EQ(rj_test_translation_memo_bytes(), 0u);
}

TEST_F(HsaHotswapMemoTest, DisplacedTranslationsAreTranslatedAgainRatherThanLost) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  const std::vector<uint8_t> base = read_translation_fixture();
  ASSERT_FALSE(base.empty()) << GFX1250_B0_TO_A0_FIXTURE;

  // Two objects of identical size, so one entry's worth of room holds exactly one
  // of them and the second must displace the first.
  const std::vector<uint8_t> other = make_fingerprint_twin(base);
  ASSERT_FALSE(other.empty());
  std::vector<uint8_t> first = base;
  first.resize(base.size() + 4096, 0);
  ASSERT_EQ(first.size(), other.size());

  ASSERT_EQ(load_through_new_reader(first), HSA_STATUS_SUCCESS);
  const uint64_t one_entry = rj_test_translation_memo_bytes();
  ASSERT_GT(one_entry, 0u);

  rj_test_set_translation_memo_capacity(one_entry);
  ASSERT_EQ(load_through_new_reader(other), HSA_STATUS_SUCCESS);
  EXPECT_EQ(rj_test_translation_memo_bytes(), one_entry) << "both entries were held";
  ASSERT_EQ(rj_test_translation_count(), 2u);

  // Displacement costs throughput and nothing else: the object still loads, it is
  // simply translated again.
  ASSERT_EQ(load_through_new_reader(first), HSA_STATUS_SUCCESS);
  EXPECT_EQ(rj_test_translation_count(), 3u);
}

// Declining to cache an oversized result must not turn its waiters into a convoy.
// They asked for these exact bytes and the work is done, so the cohort is served
// from it before it is dropped; waking them to find nothing would make each
// translate the object again in turn -- N x 197 s for the image RCCL loads, worse
// than switching the memo off.
TEST_F(HsaHotswapMemoTest, AnOversizedResultStillServesTheWaitersItAlreadyHas) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  const std::vector<uint8_t> source = read_translation_fixture();
  ASSERT_FALSE(source.empty()) << GFX1250_B0_TO_A0_FIXTURE;
  constexpr size_t kThreadCount = 8;

  rj_test_set_translation_memo_capacity(1); // Nothing can be admitted.
  rj_test_close_translation_gate();

  std::atomic<int> failures{0};
  std::vector<std::thread> workers;
  workers.reserve(kThreadCount);
  for (size_t thread_index = 0; thread_index < kThreadCount; ++thread_index) {
    workers.emplace_back([&] {
      if (load_through_new_reader(source) != HSA_STATUS_SUCCESS)
        failures.fetch_add(1, std::memory_order_relaxed);
    });
  }

  const uint64_t waiters = wait_for_memo_waiters(kThreadCount - 1);
  rj_test_open_translation_gate();
  for (auto &worker : workers)
    worker.join();

  EXPECT_EQ(failures.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(waiters, kThreadCount - 1);
  // One translation for the whole cohort, and not a byte kept afterwards.
  EXPECT_EQ(rj_test_translation_count(), 1u);
  EXPECT_LE(rj_test_translation_memo_bytes(), 1u);

  // The result really was transient: a load arriving after the cohort drained
  // finds nothing and translates again.
  ASSERT_EQ(load_through_new_reader(source), HSA_STATUS_SUCCESS);
  EXPECT_EQ(rj_test_translation_count(), 2u);
  EXPECT_LE(rj_test_translation_memo_bytes(), 1u);
}

// A load arriving while a finished result is still draining its cohort must take
// it rather than start again. The window is normally microseconds, but an
// enrolled waiter can be descheduled for arbitrarily long, and the duplicate
// costs another 197 s and 15.9 GB for the object RCCL loads.
TEST_F(HsaHotswapMemoTest, ALateLoadTakesAResultThatIsStillDraining) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  const std::vector<uint8_t> source = read_translation_fixture();
  ASSERT_FALSE(source.empty()) << GFX1250_B0_TO_A0_FIXTURE;

  // Oversized, so the result is handed to the cohort rather than cached; and held
  // open afterwards, so the window this is about does not close before the late
  // load can enter it.
  rj_test_set_translation_memo_capacity(1);
  rj_test_retain_completed_claims(true);
  rj_test_close_translation_gate();

  std::atomic<int> failures{0};
  std::vector<std::thread> workers;
  workers.reserve(2);
  for (size_t thread_index = 0; thread_index < 2; ++thread_index) {
    workers.emplace_back([&] {
      if (load_through_new_reader(source) != HSA_STATUS_SUCCESS)
        failures.fetch_add(1, std::memory_order_relaxed);
    });
  }
  const uint64_t waiters = wait_for_memo_waiters(1);
  rj_test_open_translation_gate();
  for (auto &worker : workers)
    worker.join();

  EXPECT_EQ(failures.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(waiters, 1u);
  ASSERT_EQ(rj_test_translation_count(), 1u);

  // Arriving inside the drain window: the answer is right there.
  EXPECT_EQ(load_through_new_reader(source), HSA_STATUS_SUCCESS);
  EXPECT_EQ(rj_test_translation_count(), 1u) << "a ready result was ignored";
}

// Admitting an entry allocates twice, and either allocation can come up short --
// most plausibly right after a translation that peaked at 15.9 GB. Neither
// failure may cost the waiters their answer: they would translate the same object
// one at a time, N x 197 s, which is the convoy in a new place.
class HsaHotswapAdmissionFaultTest : public HsaHotswapMemoTest,
                                     public ::testing::WithParamInterface<int> {};

TEST_P(HsaHotswapAdmissionFaultTest, AFailedAdmissionStillServesTheCohort) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  const std::vector<uint8_t> source = read_translation_fixture();
  ASSERT_FALSE(source.empty()) << GFX1250_B0_TO_A0_FIXTURE;
  constexpr size_t kThreadCount = 6;

  // Room to spare, so the entry would be admitted if the allocation succeeded.
  rj_test_close_translation_gate();
  rj_test_fail_next_memo_admission(GetParam());

  std::atomic<int> failures{0};
  std::vector<std::thread> workers;
  workers.reserve(kThreadCount);
  for (size_t thread_index = 0; thread_index < kThreadCount; ++thread_index) {
    workers.emplace_back([&] {
      if (load_through_new_reader(source) != HSA_STATUS_SUCCESS)
        failures.fetch_add(1, std::memory_order_relaxed);
    });
  }
  const uint64_t waiters = wait_for_memo_waiters(kThreadCount - 1);
  rj_test_open_translation_gate();
  for (auto &worker : workers)
    worker.join();

  EXPECT_EQ(waiters, kThreadCount - 1);
  // Every load succeeded, and exactly one translation served all of them.
  EXPECT_EQ(failures.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(rj_test_translation_count(), 1u) << "the cohort was made to translate again";
  // Nothing was cached, which is the cost of the failed admission and all of it.
  EXPECT_EQ(rj_test_translation_memo_bytes(), 0u);

  // A later load finds no entry and translates, confirming the memo really is
  // empty rather than holding something unreachable.
  ASSERT_EQ(load_through_new_reader(source), HSA_STATUS_SUCCESS);
  EXPECT_EQ(rj_test_translation_count(), 2u);
}

INSTANTIATE_TEST_SUITE_P(ListNodeAndIndexNode, HsaHotswapAdmissionFaultTest,
                         ::testing::Values(1, 2));

// Eviction has to run to the ceiling in one go, however many entries that takes.
// Stopping part way -- as it would if anything in the loop could fail -- leaves
// the memo over its limit with nothing that will ever retry it, because a hit
// does not evict.
TEST_F(HsaHotswapMemoTest, ShrinkingTheCeilingDisplacesEveryEntryItHasTo) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  const std::vector<uint8_t> base = read_translation_fixture();
  ASSERT_FALSE(base.empty()) << GFX1250_B0_TO_A0_FIXTURE;
  constexpr size_t kObjects = 5;

  // Distinct objects of the same shape: trailing bytes past the ELF change the
  // content without changing what translation makes of it.
  uint64_t one_entry = 0;
  for (size_t index = 0; index < kObjects; ++index) {
    std::vector<uint8_t> object = base;
    object.resize(base.size() + 64 + index, 0x11);
    ASSERT_EQ(load_through_new_reader(object), HSA_STATUS_SUCCESS) << index;
    if (index == 0)
      one_entry = rj_test_translation_memo_bytes();
  }
  ASSERT_EQ(rj_test_translation_count(), kObjects);
  const uint64_t all_entries = rj_test_translation_memo_bytes();
  ASSERT_GT(all_entries, 2 * one_entry) << "the objects did not all stay resident";

  // Room for roughly one of them, so four have to go at once.
  rj_test_set_translation_memo_capacity(one_entry + 8);
  EXPECT_LE(rj_test_translation_memo_bytes(), one_entry + 8) << "eviction stopped short";
}

// The ceiling is a promise about resident memory, and these bytes are never
// reclaimed, so an entry that does not fit is declined rather than admitted and
// exempted. The cost is a retranslation per load, which is the operator's to
// weigh -- the log names the variable that would buy the entry back.
TEST_F(HsaHotswapMemoTest, AnObjectLargerThanTheCapIsNotRemembered) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  const std::vector<uint8_t> source = read_translation_fixture();
  ASSERT_FALSE(source.empty()) << GFX1250_B0_TO_A0_FIXTURE;

  rj_test_set_translation_memo_capacity(1);
  constexpr size_t kLoads = 4;
  for (size_t load = 0; load < kLoads; ++load)
    ASSERT_EQ(load_through_new_reader(source), HSA_STATUS_SUCCESS) << load;

  // Never over the ceiling, and never a failed load -- only slower ones.
  EXPECT_LE(rj_test_translation_memo_bytes(), 1u);
  EXPECT_EQ(rj_test_translation_count(), kLoads);
}

TEST_F(HsaHotswapMemoTest, ReuseIsReportedAgainstTheSameSourceObject) {
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  const std::vector<uint8_t> source = read_translation_fixture();
  ASSERT_FALSE(source.empty()) << GFX1250_B0_TO_A0_FIXTURE;

  ASSERT_EQ(setenv("HSA_HOTSWAP_VERBOSE", "1", 1), 0);
  testing::internal::CaptureStderr();
  ASSERT_EQ(load_through_new_reader(source), HSA_STATUS_SUCCESS);
  ASSERT_EQ(load_through_new_reader(source), HSA_STATUS_SUCCESS);
  const std::string log_text = testing::internal::GetCapturedStderr();
  ASSERT_EQ(unsetenv("HSA_HOTSWAP_VERBOSE"), 0);

  uint64_t identity = 14695981039346656037ULL;
  for (uint8_t byte : source) {
    identity ^= byte;
    identity *= 1099511628211ULL;
  }
  std::ostringstream expected_identity;
  expected_identity << "source_id=fnv1a64:" << std::hex << std::setfill('0') << std::setw(16)
                    << identity;

  // A reuse must stay attributable to its source object. Almost every load becomes
  // a reuse once the memo is warm, so a reuse that reported less would blind the
  // translation record exactly when it carries the most traffic.
  EXPECT_NE(log_text.find(" outcome=translated changed="), std::string::npos) << log_text;
  EXPECT_NE(log_text.find(" outcome=reused changed="), std::string::npos) << log_text;
  std::istringstream lines(log_text);
  std::string line;
  size_t reuse_records = 0;
  while (std::getline(lines, line)) {
    EXPECT_NE(line.find(expected_identity.str()), std::string::npos) << line;
    if (line.find(" outcome=reused ") != std::string::npos)
      ++reuse_records;
  }
  EXPECT_EQ(reuse_records, 1u);
}
#endif

#if defined(GFX1250_B0_TO_A0_FIXTURE) && defined(RJ_PRETRANSLATE_TOOL)
/// @brief A private pre-translated tier for one test.
///
/// @details A failed mkdtemp points the store at a path that cannot exist rather
/// than passing nullptr, which would restore the REAL install tier: on a machine
/// where someone had pre-translated, a test could then pass without ever touching
/// its own directory.
class ScopedPretranslationRoot {
public:
  static constexpr const char *kUnusableRoot = "/dev/null/rj-fixture-has-no-root";

  ScopedPretranslationRoot() {
    std::strcpy(path_, "/tmp/rj-hotswap-aot-XXXXXX");
    if (mkdtemp(path_) == nullptr)
      path_[0] = '\0';
    rj_test_set_pretranslation_root(path_[0] == '\0' ? kUnusableRoot : path_);
  }
  ~ScopedPretranslationRoot() {
    rj_test_set_pretranslation_root(nullptr);
    if (path_[0] != '\0')
      std::filesystem::remove_all(path_);
  }
  ScopedPretranslationRoot(const ScopedPretranslationRoot &) = delete;
  ScopedPretranslationRoot &operator=(const ScopedPretranslationRoot &) = delete;

  [[nodiscard]] bool has_root() const { return path_[0] != '\0'; }
  [[nodiscard]] const char *path() const { return path_; }

  /// @brief Re-run the tier's checks after filling its directory.
  /// @details It opens once and holds the descriptor, so a tier populated after
  /// the first lookup would otherwise stay closed.
  void reopen() const { rj_test_set_pretranslation_root(path_); }

private:
  char path_[64] = {};
};

class HsaHotswapPretranslationTest : public HsaHotswapHookTest {
protected:
  void SetUp() override {
    HsaHotswapHookTest::SetUp();
    ASSERT_TRUE(store_root.has_root()) << "fixture could not create its private store root";
    ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
    source = read_translation_fixture();
    ASSERT_FALSE(source.empty()) << GFX1250_B0_TO_A0_FIXTURE;
  }

  /// @brief Run one A0 load of @p bytes and return what reached the loader.
  std::vector<uint8_t> load(const std::vector<uint8_t> &bytes) {
    hsa_code_object_reader_t reader{};
    EXPECT_EQ(
        api.core.hsa_code_object_reader_create_from_memory_fn(bytes.data(), bytes.size(), &reader),
        HSA_STATUS_SUCCESS);
    EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kExecutable, kA0Agent, reader,
                                                                nullptr, nullptr),
              HSA_STATUS_SUCCESS);
    const std::vector<uint8_t> loaded = g_loaded_bytes;
    EXPECT_EQ(api.core.hsa_executable_destroy_fn(kExecutable), HSA_STATUS_SUCCESS);
    return loaded;
  }

  /// @brief Fill the tier by running the real tool over @p bytes.
  void pretranslate(const std::vector<uint8_t> &bytes) {
    const std::filesystem::path input = std::filesystem::path(store_root.path()) / "source.co";
    {
      std::ofstream out(input, std::ios::binary);
      out.write(reinterpret_cast<const char *>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
      ASSERT_TRUE(out.good());
    }
    const std::string command = std::string(RJ_PRETRANSLATE_TOOL) + " --store-root '" +
                                store_root.path() + "' '" + input.string() + "' > /dev/null 2>&1";
    ASSERT_EQ(std::system(command.c_str()), 0) << command;
    std::filesystem::remove(input);
    store_root.reopen();
  }

  ScopedPretranslationRoot store_root;
  std::vector<uint8_t> source;
};

// The tier is only worth anything if a DIFFERENT program can fill it. Both sides
// key entries on the portable-prebuilt contract and locate the tree relative to
// the same shared object, so they agree by construction -- but nothing reports
// a disagreement. A tool writing keys the hook never computes would produce no
// error, no warning, and no reuse whatsoever.
//
// So this runs the real tool as a separate process rather than writing entries
// through a test seam. A seam would exercise the hook's own key derivation twice
// and prove nothing about the pair.
TEST_F(HsaHotswapPretranslationTest, WhatTheToolWritesAheadOfTimeServesALoad) {
  pretranslate(source);

  const std::vector<uint8_t> loaded = load(source);
  ASSERT_FALSE(loaded.empty());
  EXPECT_EQ(rj_test_pretranslation_hits(), 1u);

  // Nothing was added to the tier: it is read-only to this process, so a runtime
  // never writes back what it was handed.
  size_t objects = 0;
  for (const auto &item : std::filesystem::recursive_directory_iterator(store_root.path()))
    objects += item.path().extension() == ".obj" ? 1 : 0;
  EXPECT_EQ(objects, 1u);
}

TEST_F(HsaHotswapPretranslationTest, APreTranslatedObjectIsWhatTranslatingWouldHaveProduced) {
  // Reuse is only correct if it is indistinguishable from doing the work. The
  // tool and the hook run the same translator, so this should hold trivially --
  // which is exactly why it is worth an assertion rather than an assumption.
  const std::vector<uint8_t> translated = load(source);
  ASSERT_FALSE(translated.empty());
  rj_test_clear_retained_storage();

  pretranslate(source);

  EXPECT_EQ(load(source), translated);
  EXPECT_EQ(rj_test_pretranslation_hits(), 1u);
}
#endif

} // namespace
