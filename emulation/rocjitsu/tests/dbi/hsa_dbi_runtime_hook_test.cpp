// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// The DBI runtime hook against a fake HSA API table: what it installs, which
// bytes reach the lower loader, and how long the instrumented bytes live.

#include "dbi_test_util.h"
#include "hsa/hsa_api_trace.h"
#include "rocjitsu/hooks/dbi_runtime_sites.h"

#include "rocjitsu/code/amdgpu_code_object.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

extern "C" bool OnLoad(HsaApiTable *, uint64_t, uint64_t, const char *const *);
extern "C" void OnUnload();
extern "C" bool rj_test_dbi_set_probe(const void *elf, size_t size, const char *symbol);
extern "C" uint64_t rj_test_dbi_loads();
extern "C" uint64_t rj_test_dbi_unknown_readers();
extern "C" uint64_t rj_test_dbi_decisions(int decision);
extern "C" uint64_t rj_test_dbi_sites();
extern "C" size_t rj_test_dbi_retained_buffer_count();
extern "C" void rj_test_dbi_reset();

namespace {
// Set to make the next allocation in the process throw std::bad_alloc.
bool g_fail_next_allocation = false;
bool g_allocated_here = false;
} // namespace

// The testing hook library resolves operator new to this binary's, so a test
// can fail the hook's next allocation. Weak, so a static sanitizer runtime's
// own operator new wins at link time rather than colliding; the allocation
// tests then skip.
[[gnu::weak]] void *operator new(std::size_t size) {
  g_allocated_here = true;
  if (std::exchange(g_fail_next_allocation, false))
    throw std::bad_alloc();
  if (void *memory = std::malloc(size == 0 ? 1 : size))
    return memory;
  throw std::bad_alloc();
}

[[gnu::weak]] void operator delete(void *memory) noexcept { std::free(memory); }

[[gnu::weak]] void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }

namespace rocjitsu::hooks {
namespace {

using test::kMovV2Zero;
using test::kMovV3V2;
using test::kProbeSetpcS30S31;

constexpr hsa_agent_t kAgent{7};
constexpr hsa_executable_t kExecutable{42};
constexpr const char *kProbeSymbol = "rj_test_probe";

// Two blocks, each anchorable at its first instruction (gfx950):
// 0x00 [v_mov, s_cbranch_scc0 -> 0x0c], 0x08 [s_endpgm], 0x0c [v_mov, s_endpgm].
const std::vector<uint32_t> kKernelText = {kMovV2Zero, 0xBF840001u, 0xBF810000u, kMovV3V2,
                                           0xBF810000u};

using VendorReaderCreate = hsa_status_t (*)(hsa_file_t, size_t, size_t, hsa_code_object_reader_t *);

struct FakeVendorLoaderTable {
  void (*query_host_address)();
  void (*query_segment_descriptors)();
  void (*query_executable)();
  void (*iterate_loaded_code_objects)();
  void (*loaded_code_object_get_info)();
  VendorReaderCreate create_reader_from_file;
};

struct ReaderView {
  const uint8_t *bytes;
  size_t size;
};

uint64_t g_next_reader = 1;
// Set to fail the hook's first allocation after the next reader a fake creates.
bool g_fail_allocation_after_create = false;
hsa_status_t g_destroy_executable_status = HSA_STATUS_SUCCESS;
// Set to fail the hook's first allocation after the next executable destroy.
bool g_fail_allocation_after_destroy = false;
// Runs inside the next executable destroy, once the executable is gone.
std::function<void()> g_during_destroy;
std::unordered_map<uint64_t, ReaderView> g_readers;
std::vector<uint8_t> g_loaded_bytes;
hsa_code_object_reader_t g_loaded_reader{};
hsa_status_t g_load_agent_status = HSA_STATUS_SUCCESS;
int g_reader_destroy_calls = 0;

void reset_fakes() {
  g_next_reader = 1;
  g_readers.clear();
  g_loaded_bytes.clear();
  g_loaded_reader = {};
  g_load_agent_status = HSA_STATUS_SUCCESS;
  g_reader_destroy_calls = 0;
  g_fail_allocation_after_create = false;
  g_destroy_executable_status = HSA_STATUS_SUCCESS;
  g_fail_allocation_after_destroy = false;
  g_during_destroy = nullptr;
  g_fail_next_allocation = false;
}

void arm_allocation_failure() {
  g_fail_next_allocation = std::exchange(g_fail_allocation_after_create, false);
}

hsa_status_t HSA_API fake_create_file(hsa_file_t, hsa_code_object_reader_t *reader) {
  *reader = hsa_code_object_reader_t{g_next_reader++};
  arm_allocation_failure();
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_create_memory(const void *bytes, size_t size,
                                        hsa_code_object_reader_t *reader) {
  *reader = hsa_code_object_reader_t{g_next_reader++};
  g_readers.emplace(reader->handle, ReaderView{static_cast<const uint8_t *>(bytes), size});
  arm_allocation_failure();
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_destroy_reader(hsa_code_object_reader_t reader) {
  ++g_reader_destroy_calls;
  g_readers.erase(reader.handle);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_destroy_executable(hsa_executable_t) {
  if (g_during_destroy)
    g_during_destroy();
  g_fail_next_allocation = std::exchange(g_fail_allocation_after_destroy, false);
  return g_destroy_executable_status;
}

hsa_status_t HSA_API fake_load_agent(hsa_executable_t, hsa_agent_t, hsa_code_object_reader_t reader,
                                     const char *, hsa_loaded_code_object_t *) {
  g_loaded_reader = reader;
  if (g_load_agent_status != HSA_STATUS_SUCCESS)
    return g_load_agent_status;
  const auto it = g_readers.find(reader.handle);
  if (it != g_readers.end())
    g_loaded_bytes.assign(it->second.bytes, it->second.bytes + it->second.size);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_load_program(hsa_executable_t, hsa_code_object_reader_t, const char *,
                                       hsa_loaded_code_object_t *) {
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_load_deprecated(hsa_executable_t, hsa_agent_t, hsa_code_object_t,
                                          const char *) {
  return HSA_STATUS_SUCCESS;
}

hsa_status_t fake_vendor_reader_create(hsa_file_t, size_t, size_t,
                                       hsa_code_object_reader_t *reader) {
  *reader = hsa_code_object_reader_t{g_next_reader++};
  arm_allocation_failure();
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_get_extension_table(uint16_t extension, uint16_t version_major,
                                              size_t table_length, void *table) {
  if (extension == HSA_EXTENSION_AMD_LOADER && version_major == 1 &&
      table_length >= sizeof(FakeVendorLoaderTable))
    static_cast<FakeVendorLoaderTable *>(table)->create_reader_from_file =
        fake_vendor_reader_create;
  return HSA_STATUS_SUCCESS;
}

struct FakeApi {
  CoreApiTable core{};
  HsaApiTable table{};

  FakeApi() {
    core.version.minor_id = sizeof(core);
    table.version.minor_id = sizeof(table);
    table.core_ = &core;
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

uint64_t decisions(DbiLoadDecision decision) {
  return rj_test_dbi_decisions(static_cast<int>(decision));
}

class HsaDbiRuntimeHookTest : public ::testing::Test {
protected:
  void SetUp() override {
    OnUnload();
    rj_test_dbi_reset();
    reset_fakes();
    kernel_ = test::make_gfx950_kernel_elf(kKernelText, /*private_bytes=*/0);
    probe_ = test::make_gfx950_probe_elf(kProbeSymbol, {kProbeSetpcS30S31});
  }

  void TearDown() override {
    OnUnload();
    rj_test_dbi_reset();
  }

  void set_probe() {
    ASSERT_TRUE(rj_test_dbi_set_probe(probe_.data(), probe_.size(), kProbeSymbol));
  }

  hsa_code_object_reader_t create_reader(const std::vector<uint8_t> &bytes) {
    hsa_code_object_reader_t reader{};
    EXPECT_EQ(
        api_.core.hsa_code_object_reader_create_from_memory_fn(bytes.data(), bytes.size(), &reader),
        HSA_STATUS_SUCCESS);
    return reader;
  }

  hsa_status_t load(hsa_code_object_reader_t reader) {
    return api_.core.hsa_executable_load_agent_code_object_fn(kExecutable, kAgent, reader, nullptr,
                                                              nullptr);
  }

  // What the hook should load for kernel_: the same selection, run directly.
  std::vector<uint8_t> expected_instrumented_kernel() const {
    const AmdGpuCodeObject probe(probe_.data(), probe_.size());
    return instrument_each_block(kernel_, DbiProbe{&probe, kProbeSymbol}).elf_bytes;
  }

  // A temporary file holding @p bytes after @p padding zero bytes.
  static std::unique_ptr<FILE, int (*)(FILE *)> temp_file(const std::vector<uint8_t> &bytes,
                                                          size_t padding = 0) {
    std::unique_ptr<FILE, int (*)(FILE *)> file(std::tmpfile(), &std::fclose);
    if (file != nullptr) {
      const std::vector<uint8_t> zeros(padding, 0);
      std::fwrite(zeros.data(), 1, zeros.size(), file.get());
      std::fwrite(bytes.data(), 1, bytes.size(), file.get());
      std::fflush(file.get());
    }
    return file;
  }

  FakeApi api_;
  std::vector<uint8_t> kernel_;
  std::vector<uint8_t> probe_;
};

TEST_F(HsaDbiRuntimeHookTest, InstallsOnlyTheAgentLoadSurface) {
  ASSERT_TRUE(OnLoad(&api_.table, 0, 0, nullptr));

  EXPECT_NE(api_.core.hsa_code_object_reader_create_from_file_fn, fake_create_file);
  EXPECT_NE(api_.core.hsa_code_object_reader_create_from_memory_fn, fake_create_memory);
  EXPECT_NE(api_.core.hsa_code_object_reader_destroy_fn, fake_destroy_reader);
  EXPECT_NE(api_.core.hsa_executable_destroy_fn, fake_destroy_executable);
  EXPECT_NE(api_.core.hsa_executable_load_agent_code_object_fn, fake_load_agent);
  EXPECT_NE(api_.core.hsa_system_get_major_extension_table_fn, fake_get_extension_table);

  EXPECT_EQ(api_.core.hsa_executable_load_program_code_object_fn, fake_load_program);
  EXPECT_EQ(api_.core.hsa_executable_load_code_object_fn, fake_load_deprecated);
}

TEST_F(HsaDbiRuntimeHookTest, UnloadRestoresAndAllowsReinstall) {
  ASSERT_TRUE(OnLoad(&api_.table, 0, 0, nullptr));
  EXPECT_FALSE(OnLoad(&api_.table, 0, 0, nullptr));
  OnUnload();
  EXPECT_EQ(api_.core.hsa_code_object_reader_create_from_file_fn, fake_create_file);
  EXPECT_EQ(api_.core.hsa_code_object_reader_create_from_memory_fn, fake_create_memory);
  EXPECT_EQ(api_.core.hsa_code_object_reader_destroy_fn, fake_destroy_reader);
  EXPECT_EQ(api_.core.hsa_executable_destroy_fn, fake_destroy_executable);
  EXPECT_EQ(api_.core.hsa_executable_load_agent_code_object_fn, fake_load_agent);
  EXPECT_EQ(api_.core.hsa_system_get_major_extension_table_fn, fake_get_extension_table);
  EXPECT_TRUE(OnLoad(&api_.table, 0, 0, nullptr));
}

TEST_F(HsaDbiRuntimeHookTest, RejectedInstallLeavesHookInstallable) {
  EXPECT_FALSE(OnLoad(nullptr, 0, 0, nullptr));
  FakeApi no_core;
  no_core.table.core_ = nullptr;
  EXPECT_FALSE(OnLoad(&no_core.table, 0, 0, nullptr));
  // An older runtime's core table, ending before the load-agent entry.
  FakeApi truncated;
  truncated.core.version.minor_id =
      static_cast<uint32_t>(offsetof(CoreApiTable, hsa_executable_load_agent_code_object_fn));
  EXPECT_FALSE(OnLoad(&truncated.table, 0, 0, nullptr));
  FakeApi incomplete;
  incomplete.core.hsa_executable_load_agent_code_object_fn = nullptr;
  EXPECT_FALSE(OnLoad(&incomplete.table, 0, 0, nullptr));
  EXPECT_TRUE(OnLoad(&api_.table, 0, 0, nullptr));
}

TEST_F(HsaDbiRuntimeHookTest, ForwardsTheCallersReaderWithoutAProbe) {
  ASSERT_TRUE(OnLoad(&api_.table, 0, 0, nullptr));
  const hsa_code_object_reader_t reader = create_reader(kernel_);
  ASSERT_EQ(load(reader), HSA_STATUS_SUCCESS);

  EXPECT_EQ(g_loaded_reader.handle, reader.handle);
  EXPECT_EQ(g_loaded_bytes, kernel_);
  EXPECT_EQ(rj_test_dbi_loads(), 1u);
  EXPECT_EQ(decisions(DbiLoadDecision::Instrumented), 0u);
}

TEST_F(HsaDbiRuntimeHookTest, LoadsTheInstrumentedObjectAndRetainsItUntilDestroy) {
  ASSERT_TRUE(OnLoad(&api_.table, 0, 0, nullptr));
  set_probe();
  const hsa_code_object_reader_t reader = create_reader(kernel_);
  ASSERT_EQ(load(reader), HSA_STATUS_SUCCESS);

  EXPECT_EQ(decisions(DbiLoadDecision::Instrumented), 1u);
  EXPECT_EQ(rj_test_dbi_sites(), 2u);
  EXPECT_NE(g_loaded_reader.handle, reader.handle);
  EXPECT_EQ(g_loaded_bytes, expected_instrumented_kernel());
  // The hook's own reader is destroyed as soon as the load returns; the bytes
  // it read are held for the executable.
  EXPECT_EQ(g_reader_destroy_calls, 1);
  EXPECT_EQ(rj_test_dbi_retained_buffer_count(), 1u);

  OnUnload();
  EXPECT_EQ(rj_test_dbi_retained_buffer_count(), 1u);
  ASSERT_TRUE(OnLoad(&api_.table, 0, 0, nullptr));
  ASSERT_EQ(api_.core.hsa_executable_destroy_fn(kExecutable), HSA_STATUS_SUCCESS);
  EXPECT_EQ(rj_test_dbi_retained_buffer_count(), 0u);
}

TEST_F(HsaDbiRuntimeHookTest, KeepsTheBytesOfAnExecutableThatReusesADestroyedHandle) {
  ASSERT_TRUE(OnLoad(&api_.table, 0, 0, nullptr));
  set_probe();
  ASSERT_EQ(load(create_reader(kernel_)), HSA_STATUS_SUCCESS);
  // A new executable gets the handle as soon as the lower destroy frees it,
  // and loads before the hook's destroy returns.
  const hsa_code_object_reader_t next = create_reader(kernel_);
  g_during_destroy = [&] { EXPECT_EQ(load(next), HSA_STATUS_SUCCESS); };
  ASSERT_EQ(api_.core.hsa_executable_destroy_fn(kExecutable), HSA_STATUS_SUCCESS);

  EXPECT_EQ(decisions(DbiLoadDecision::Instrumented), 2u);
  EXPECT_EQ(rj_test_dbi_retained_buffer_count(), 1u);
}

TEST_F(HsaDbiRuntimeHookTest, ForwardsTheCallersReaderWhenNotInstrumented) {
  ASSERT_TRUE(OnLoad(&api_.table, 0, 0, nullptr));
  set_probe();
  const std::vector<uint8_t> gfx90a_kernel = test::make_amdgpu_kernel_elf(
      kKernelText, /*private_bytes=*/0, /*granulated_sgpr_count=*/3, EF_AMDGPU_MACH_AMDGCN_GFX90A);
  const hsa_code_object_reader_t reader = create_reader(gfx90a_kernel);
  ASSERT_EQ(load(reader), HSA_STATUS_SUCCESS);

  EXPECT_EQ(decisions(DbiLoadDecision::UnsupportedArch), 1u);
  EXPECT_EQ(g_loaded_reader.handle, reader.handle);
  EXPECT_EQ(rj_test_dbi_retained_buffer_count(), 0u);
}

TEST_F(HsaDbiRuntimeHookTest, ForwardsAReaderCreatedBeforeInstall) {
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(fake_create_memory(kernel_.data(), kernel_.size(), &reader), HSA_STATUS_SUCCESS);
  ASSERT_TRUE(OnLoad(&api_.table, 0, 0, nullptr));
  set_probe();
  ASSERT_EQ(load(reader), HSA_STATUS_SUCCESS);

  EXPECT_EQ(rj_test_dbi_unknown_readers(), 1u);
  EXPECT_EQ(g_loaded_reader.handle, reader.handle);
}

TEST_F(HsaDbiRuntimeHookTest, ReleasesInstrumentedBytesWhenTheLoadFails) {
  ASSERT_TRUE(OnLoad(&api_.table, 0, 0, nullptr));
  set_probe();
  const hsa_code_object_reader_t reader = create_reader(kernel_);
  g_load_agent_status = HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  EXPECT_EQ(load(reader), HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
  EXPECT_EQ(rj_test_dbi_retained_buffer_count(), 0u);
  EXPECT_EQ(g_reader_destroy_calls, 1);
}

TEST_F(HsaDbiRuntimeHookTest, InstrumentsAnObjectReadFromAFile) {
  ASSERT_TRUE(OnLoad(&api_.table, 0, 0, nullptr));
  set_probe();
  const auto file = temp_file(kernel_);
  ASSERT_NE(file, nullptr);
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api_.core.hsa_code_object_reader_create_from_file_fn(fileno(file.get()), &reader),
            HSA_STATUS_SUCCESS);

  ASSERT_EQ(load(reader), HSA_STATUS_SUCCESS);
  EXPECT_EQ(decisions(DbiLoadDecision::Instrumented), 1u);
  EXPECT_EQ(g_loaded_bytes, expected_instrumented_kernel());
}

TEST_F(HsaDbiRuntimeHookTest, KeepsAReaderWhoseBytesCannotBeRead) {
  ASSERT_TRUE(OnLoad(&api_.table, 0, 0, nullptr));
  set_probe();
  // The fake accepts any descriptor; the hook cannot read this one.
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api_.core.hsa_code_object_reader_create_from_file_fn(-1, &reader), HSA_STATUS_SUCCESS);
  EXPECT_NE(reader.handle, 0u);
  EXPECT_EQ(g_reader_destroy_calls, 0);

  (void)load(reader);
  EXPECT_EQ(rj_test_dbi_unknown_readers(), 1u);
  EXPECT_EQ(g_loaded_reader.handle, reader.handle);
}

TEST_F(HsaDbiRuntimeHookTest, KeepsAnUnreadableReaderOffAnEarlierReadersBytes) {
  ASSERT_TRUE(OnLoad(&api_.table, 0, 0, nullptr));
  set_probe();
  const hsa_code_object_reader_t earlier = create_reader(kernel_);
  // Destroyed below the hook, which keeps its bytes stored; the runtime then
  // hands out the same handle again.
  ASSERT_EQ(fake_destroy_reader(earlier), HSA_STATUS_SUCCESS);
  g_next_reader = earlier.handle;
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api_.core.hsa_code_object_reader_create_from_file_fn(-1, &reader), HSA_STATUS_SUCCESS);
  ASSERT_EQ(reader.handle, earlier.handle);

  (void)load(reader);
  EXPECT_EQ(rj_test_dbi_unknown_readers(), 1u);
  EXPECT_EQ(g_loaded_reader.handle, reader.handle);
}

TEST_F(HsaDbiRuntimeHookTest, ForwardsAReaderCreatedBeforeReinstallUnderAnEarlierHandle) {
  ASSERT_TRUE(OnLoad(&api_.table, 0, 0, nullptr));
  // Never destroyed while the hook is installed, so its bytes stay stored.
  const hsa_code_object_reader_t earlier = create_reader(kernel_);
  OnUnload();
  // A new runtime instance hands out the same handle again before the hook is
  // reinstalled.
  reset_fakes();
  const std::vector<uint8_t> other = test::make_gfx942_kernel_elf(kKernelText, 0);
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(fake_create_memory(other.data(), other.size(), &reader), HSA_STATUS_SUCCESS);
  ASSERT_EQ(reader.handle, earlier.handle);
  ASSERT_TRUE(OnLoad(&api_.table, 0, 0, nullptr));
  set_probe();

  ASSERT_EQ(load(reader), HSA_STATUS_SUCCESS);
  EXPECT_EQ(rj_test_dbi_unknown_readers(), 1u);
  EXPECT_EQ(g_loaded_reader.handle, reader.handle);
  EXPECT_EQ(g_loaded_bytes, other);
}

TEST_F(HsaDbiRuntimeHookTest, ForgetsTheBytesOfADestroyedReader) {
  ASSERT_TRUE(OnLoad(&api_.table, 0, 0, nullptr));
  set_probe();
  const hsa_code_object_reader_t reader = create_reader(kernel_);
  ASSERT_EQ(api_.core.hsa_code_object_reader_destroy_fn(reader), HSA_STATUS_SUCCESS);

  (void)load(reader);
  EXPECT_EQ(rj_test_dbi_unknown_readers(), 1u);
  EXPECT_EQ(decisions(DbiLoadDecision::Instrumented), 0u);
}

TEST_F(HsaDbiRuntimeHookTest, InstrumentsAnObjectReadThroughTheAmdLoaderExtension) {
  ASSERT_TRUE(OnLoad(&api_.table, 0, 0, nullptr));
  set_probe();
  // The object sits past a header, as in a bundle, so the hook must read the
  // region the caller named rather than the whole file.
  constexpr size_t kHeaderBytes = 64;
  const auto file = temp_file(kernel_, kHeaderBytes);
  ASSERT_NE(file, nullptr);

  FakeVendorLoaderTable loader{};
  ASSERT_EQ(api_.core.hsa_system_get_major_extension_table_fn(HSA_EXTENSION_AMD_LOADER, 1,
                                                              sizeof(loader), &loader),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(loader.create_reader_from_file, nullptr);
  EXPECT_NE(loader.create_reader_from_file, fake_vendor_reader_create);
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(
      loader.create_reader_from_file(fileno(file.get()), kHeaderBytes, kernel_.size(), &reader),
      HSA_STATUS_SUCCESS);

  ASSERT_EQ(load(reader), HSA_STATUS_SUCCESS);
  EXPECT_EQ(decisions(DbiLoadDecision::Instrumented), 1u);
  EXPECT_NE(g_loaded_reader.handle, reader.handle);
  EXPECT_EQ(g_loaded_bytes, expected_instrumented_kernel());
}

// Cases that fail one of the hook's allocations through operator new above.
class HsaDbiRuntimeHookAllocationTest : public HsaDbiRuntimeHookTest {
protected:
  void SetUp() override {
    HsaDbiRuntimeHookTest::SetUp();
    g_allocated_here = false;
    ::operator delete(::operator new(1));
    if (!g_allocated_here)
      GTEST_SKIP() << "a static sanitizer runtime supplies operator new";
  }
};

TEST_F(HsaDbiRuntimeHookAllocationTest, DestroysAMemoryReaderWhoseBytesCannotBeKept) {
  ASSERT_TRUE(OnLoad(&api_.table, 0, 0, nullptr));
  g_fail_allocation_after_create = true;
  hsa_code_object_reader_t reader{};
  EXPECT_EQ(api_.core.hsa_code_object_reader_create_from_memory_fn(kernel_.data(), kernel_.size(),
                                                                   &reader),
            HSA_STATUS_ERROR_OUT_OF_RESOURCES);

  EXPECT_FALSE(g_fail_next_allocation) << "the injected failure was never reached";
  EXPECT_EQ(reader.handle, 0u);
  EXPECT_EQ(g_reader_destroy_calls, 1);
  EXPECT_TRUE(g_readers.empty());
}

TEST_F(HsaDbiRuntimeHookAllocationTest, DestroysAFileReaderWhoseBytesCannotBeKept) {
  ASSERT_TRUE(OnLoad(&api_.table, 0, 0, nullptr));
  const auto file = temp_file(kernel_);
  ASSERT_NE(file, nullptr);
  g_fail_allocation_after_create = true;
  hsa_code_object_reader_t reader{};
  EXPECT_EQ(api_.core.hsa_code_object_reader_create_from_file_fn(fileno(file.get()), &reader),
            HSA_STATUS_ERROR_OUT_OF_RESOURCES);

  EXPECT_FALSE(g_fail_next_allocation) << "the injected failure was never reached";
  EXPECT_EQ(reader.handle, 0u);
  EXPECT_EQ(g_reader_destroy_calls, 1);
}

TEST_F(HsaDbiRuntimeHookAllocationTest, DestroysAnAmdLoaderReaderWhoseBytesCannotBeKept) {
  ASSERT_TRUE(OnLoad(&api_.table, 0, 0, nullptr));
  const auto file = temp_file(kernel_);
  ASSERT_NE(file, nullptr);
  FakeVendorLoaderTable loader{};
  ASSERT_EQ(api_.core.hsa_system_get_major_extension_table_fn(HSA_EXTENSION_AMD_LOADER, 1,
                                                              sizeof(loader), &loader),
            HSA_STATUS_SUCCESS);
  g_fail_allocation_after_create = true;
  hsa_code_object_reader_t reader{};
  EXPECT_EQ(loader.create_reader_from_file(fileno(file.get()), 0, kernel_.size(), &reader),
            HSA_STATUS_ERROR_OUT_OF_RESOURCES);

  EXPECT_FALSE(g_fail_next_allocation) << "the injected failure was never reached";
  EXPECT_EQ(reader.handle, 0u);
  EXPECT_EQ(g_reader_destroy_calls, 1);
}

TEST_F(HsaDbiRuntimeHookAllocationTest, DestroysItsOwnReaderWhenTheBytesCannotBeRetained) {
  ASSERT_TRUE(OnLoad(&api_.table, 0, 0, nullptr));
  set_probe();
  const hsa_code_object_reader_t reader = create_reader(kernel_);
  // The next reader created is the hook's own, for the instrumented bytes.
  g_fail_allocation_after_create = true;
  EXPECT_EQ(load(reader), HSA_STATUS_ERROR_OUT_OF_RESOURCES);

  EXPECT_FALSE(g_fail_next_allocation) << "the injected failure was never reached";
  EXPECT_EQ(g_reader_destroy_calls, 1);
  ASSERT_EQ(g_readers.size(), 1u);
  EXPECT_EQ(g_readers.begin()->first, reader.handle);
  EXPECT_EQ(g_loaded_reader.handle, 0u);
  EXPECT_EQ(rj_test_dbi_retained_buffer_count(), 0u);
}

TEST_F(HsaDbiRuntimeHookAllocationTest, KeepsTheBytesOfAnExecutableWhoseDestroyFails) {
  ASSERT_TRUE(OnLoad(&api_.table, 0, 0, nullptr));
  set_probe();
  ASSERT_EQ(load(create_reader(kernel_)), HSA_STATUS_SUCCESS);
  ASSERT_EQ(rj_test_dbi_retained_buffer_count(), 1u);
  g_destroy_executable_status = HSA_STATUS_ERROR_INVALID_EXECUTABLE;
  g_fail_allocation_after_destroy = true;
  EXPECT_EQ(api_.core.hsa_executable_destroy_fn(kExecutable), HSA_STATUS_ERROR_INVALID_EXECUTABLE);

  EXPECT_TRUE(std::exchange(g_fail_next_allocation, false))
      << "the hook allocated after the destroy failed";
  EXPECT_EQ(rj_test_dbi_retained_buffer_count(), 1u);
}

} // namespace
} // namespace rocjitsu::hooks
