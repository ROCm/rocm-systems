// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "hsa/hsa_api_trace_minimal.h"
#include "rocjitsu/code/amdgpu_elf.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <new>
#include <thread>
#include <unordered_map>
#include <vector>

extern "C" bool OnLoad(HsaApiTable *, uint64_t, uint64_t, const char *const *);
extern "C" void OnUnload();
extern "C" size_t rj_test_retained_executable_buffer_count();
extern "C" void rj_test_clear_retained_storage();

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

hsa_status_t HSA_API fake_create_file(hsa_file_t, hsa_code_object_reader_t *reader) {
  if (reader == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *reader = hsa_code_object_reader_t{g_next_reader++};
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_create_memory(const void *bytes, size_t size,
                                        hsa_code_object_reader_t *reader) {
  if (bytes == nullptr || size == 0 || reader == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *reader = hsa_code_object_reader_t{g_next_reader++};
  g_readers.emplace(reader->handle, ReaderView{static_cast<const uint8_t *>(bytes), size});
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_destroy_reader(hsa_code_object_reader_t reader) {
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
    OnUnload();
    // Production storage is process-lifetime (not freed on reinstall), so clear it
    // here to isolate the retention lifecycle between test cases.
    rj_test_clear_retained_storage();
    reset_fakes();
  }
  void TearDown() override { OnUnload(); }

  FakeApi api;
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
  ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kExecutable, kA0Agent, reader,
                                                              nullptr, nullptr),
            HSA_STATUS_SUCCESS);

  ASSERT_GE(g_loaded_bytes.size(), rocjitsu::EI_MAGIC_SIZE);
  EXPECT_EQ(std::memcmp(g_loaded_bytes.data(), rocjitsu::EI_MAGIC, rocjitsu::EI_MAGIC_SIZE), 0);
  EXPECT_NE(g_loaded_bytes, source);
  EXPECT_NE(g_loaded_reader.handle, reader.handle);
  EXPECT_EQ(g_load_agent_calls, 1);
  EXPECT_EQ(rj_test_retained_executable_buffer_count(), 1u);

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
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kExecutable, kA0Agent, reader,
                                                              nullptr, nullptr),
            HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
  EXPECT_EQ(g_load_agent_calls, 0);
  EXPECT_EQ(rj_test_retained_executable_buffer_count(), 0u);
}

// The translated backing storage retained for an A0 load must SURVIVE OnUnload()
// (ROCr destroys its loader after OnUnload but before closing the DSO, so the bytes
// the loader still references must outlive OnUnload) AND a runtime-generation
// reinstall (OnLoad again): a consumer whose lifetime is not bounded by the HSA
// generation -- rocprofiler-register, which finalizes its records at process-exit
// atexit -- can still reference these bytes, so a reinstall must not free them. They
// are released only at executable destroy (or process exit). Regression guard for
// the process-lifetime storage-retention lifecycle.
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
  EXPECT_EQ(api.core.hsa_executable_load_code_object_fn(kExecutable, kA0Agent, hsa_code_object_t{1},
                                                        nullptr),
            HSA_STATUS_ERROR_OUT_OF_RESOURCES);
  EXPECT_EQ(g_load_deprecated_calls, 1);
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

} // namespace
