//===- hotswap_loader_test_utils.hpp - Loader test support ----------------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ROCR_HOTSWAP_LOADER_TEST_UTILS_HPP
#define ROCR_HOTSWAP_LOADER_TEST_UTILS_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include <hsa.h>
#include <hsa_api_trace.h>

namespace {

struct FakeEnv {
  std::string agent_isa = "amdgcn-amd-amdhsa--gfx1250";
  uint32_t asic_revision = 0;
  int retarget_calls = 0;
  int retarget_status = 0;
  uint64_t next_reader_handle = 100;
  uint64_t last_loaded_reader = 0;
  std::string retarget_source_isa;
  std::string retarget_target_isa;
};

FakeEnv g_env;
CoreApiTable g_test_core;

} // namespace

#include "../hotswap_tool.cpp"

namespace rocr::hotswap {

// The production helper uses COMGR to read a real code object's ISA metadata.
// This test double only supplies a source ISA so the test can stay focused on
// loader policy without linking COMGR.
std::string GetCodeObjectIsaName(const void *elf_data, size_t elf_size) {
  if (!elf_data || elf_size == 0) {
    return {};
  }

  constexpr char IsaPrefix[] = "amdgcn-amd-amdhsa--";
  const char *begin = static_cast<const char *>(elf_data);
  const char *end = begin + elf_size;
  const char *match = std::search(begin, end, IsaPrefix,
                                  IsaPrefix + sizeof(IsaPrefix) - 1);
  if (match == end) {
    return {};
  }

  const char *limit = match;
  while (limit != end) {
    const char c = *limit;
    if (c == '\0' || c == '\'' || c == '"' || c == '\n' || c == '\r' ||
        c == ' ' || c == '\t') {
      break;
    }
    ++limit;
  }
  return std::string(match, limit);
}

int RetargetCodeObject(const void *elf_data, size_t elf_size,
                       const char *source_isa, const char *target_isa,
                       void **out_data, size_t *out_size) {
  ++g_env.retarget_calls;
  g_env.retarget_source_isa = source_isa ? source_isa : "";
  g_env.retarget_target_isa = target_isa ? target_isa : "";

  if (!out_data || !out_size) {
    return -1;
  }
  *out_data = const_cast<void *>(elf_data);
  *out_size = elf_size;

  if (g_env.retarget_status != 0) {
    return g_env.retarget_status;
  }

  void *copy = std::malloc(elf_size);
  if (!copy) {
    return -1;
  }
  std::memcpy(copy, elf_data, elf_size);
  *out_data = copy;
  *out_size = elf_size;
  return 0;
}

} // namespace rocr::hotswap

extern "C" {

hsa_status_t hsa_agent_iterate_isas(hsa_agent_t /*agent*/,
                                    hsa_status_t (*callback)(hsa_isa_t isa,
                                                             void *data),
                                    void *data) {
  hsa_isa_t isa{};
  isa.handle = 1;
  const hsa_status_t status = callback(isa, data);
  return status == HSA_STATUS_INFO_BREAK ? HSA_STATUS_SUCCESS : status;
}

hsa_status_t hsa_isa_get_info_alt(hsa_isa_t /*isa*/, hsa_isa_info_t attribute,
                                  void *value) {
  if (attribute == HSA_ISA_INFO_NAME_LENGTH) {
    *static_cast<uint32_t *>(value) =
        static_cast<uint32_t>(g_env.agent_isa.size() + 1);
    return HSA_STATUS_SUCCESS;
  }
  if (attribute == HSA_ISA_INFO_NAME) {
    std::memcpy(value, g_env.agent_isa.c_str(), g_env.agent_isa.size() + 1);
    return HSA_STATUS_SUCCESS;
  }
  return HSA_STATUS_ERROR;
}

hsa_status_t hsa_agent_get_info(hsa_agent_t /*agent*/,
                                hsa_agent_info_t attribute, void *value) {
  if (attribute ==
      static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_ASIC_REVISION)) {
    *static_cast<uint32_t *>(value) = g_env.asic_revision;
    return HSA_STATUS_SUCCESS;
  }
  return HSA_STATUS_ERROR;
}

} // extern "C"

namespace {

constexpr const char *kGfx1250Isa = "amdgcn-amd-amdhsa--gfx1250";
constexpr const char *kGfx942Isa = "amdgcn-amd-amdhsa--gfx942";
constexpr const char *kGfx1251Isa = "amdgcn-amd-amdhsa--gfx1251";
constexpr const char *kGfx12_5GenericIsa =
    "amdgcn-amd-amdhsa--gfx12-5-generic";
constexpr const char *kGfx1250B0Isa =
    "amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+";
constexpr const char *kGfx1250A0Isa =
    "amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific-";

int tests_passed = 0;
int tests_failed = 0;

void check(bool cond, const char *name) {
  if (cond) {
    ++tests_passed;
    std::printf("  PASS: %s\n", name);
  } else {
    ++tests_failed;
    std::fprintf(stderr, "  FAIL: %s\n", name);
  }
}

void begin_test(const char *name, const char *description) {
  std::printf("TEST %s...\n  %s\n", name, description);
}

void set_gfx12_5_rewrite_env(const char *value) {
#ifdef _WIN32
  _putenv_s("AMD_COMGR_HOTSWAP_ENTRY_TRAMPOLINES", value ? value : "");
#else
  if (value) {
    setenv("AMD_COMGR_HOTSWAP_ENTRY_TRAMPOLINES", value, 1);
  } else {
    unsetenv("AMD_COMGR_HOTSWAP_ENTRY_TRAMPOLINES");
  }
#endif
}

void reset_state() {
  {
    std::scoped_lock lock(g_reader_map_mutex);
    g_reader_map.clear();
  }
  {
    std::scoped_lock lock(g_rewritten_elfs_mutex);
    g_rewritten_elfs.clear();
  }
  g_core_table = nullptr;
  g_orig_reader_create_from_memory = nullptr;
  g_orig_reader_create_from_file = nullptr;
  g_orig_reader_destroy = nullptr;
  g_orig_load_agent_code_object = nullptr;
  rocr::hotswap::reset_gfx_revision_cache();
  set_gfx12_5_rewrite_env(nullptr);
  g_env = FakeEnv{};
}

std::vector<uint8_t> make_code_object(const std::string &isa) {
  const std::string metadata = "---\namdhsa.target: '" + isa + "'\n";
  return std::vector<uint8_t>(metadata.begin(), metadata.end());
}

hsa_status_t HSA_API fake_reader_create_from_memory(
    const void * /*code_object*/, size_t /*size*/,
    hsa_code_object_reader_t *code_object_reader) {
  code_object_reader->handle = g_env.next_reader_handle++;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_reader_create_from_file(
    hsa_file_t /*file*/, hsa_code_object_reader_t *code_object_reader) {
  code_object_reader->handle = g_env.next_reader_handle++;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API
fake_reader_destroy(hsa_code_object_reader_t code_object_reader) {
  (void)code_object_reader;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_load_agent_code_object(
    hsa_executable_t /*executable*/, hsa_agent_t /*agent*/,
    hsa_code_object_reader_t code_object_reader, const char * /*options*/,
    hsa_loaded_code_object_t *loaded_code_object) {
  g_env.last_loaded_reader = code_object_reader.handle;
  if (loaded_code_object) {
    loaded_code_object->handle = 0xC0DE;
  }
  return HSA_STATUS_SUCCESS;
}

CoreApiTable &install_tool() {
  g_test_core = CoreApiTable{};
  g_test_core.hsa_code_object_reader_create_from_memory_fn =
      fake_reader_create_from_memory;
  g_test_core.hsa_code_object_reader_create_from_file_fn =
      fake_reader_create_from_file;
  g_test_core.hsa_code_object_reader_destroy_fn = fake_reader_destroy;
  g_test_core.hsa_executable_load_agent_code_object_fn =
      fake_load_agent_code_object;

  HsaApiTable table{};
  table.core_ = &g_test_core;
  check(OnLoad(&table, 0, 0, nullptr), "OnLoad installs with complete table");
  return g_test_core;
}

hsa_agent_t fake_agent() {
  hsa_agent_t agent{};
  agent.handle = 42;
  return agent;
}

hsa_code_object_reader_t create_memory_reader(CoreApiTable &core,
                                              const std::vector<uint8_t> &elf) {
  hsa_code_object_reader_t reader{};
  const hsa_status_t status = core.hsa_code_object_reader_create_from_memory_fn(
      elf.data(), elf.size(), &reader);
  check(status == HSA_STATUS_SUCCESS, "memory reader creation succeeds");
  return reader;
}

hsa_status_t load_reader(CoreApiTable &core, hsa_code_object_reader_t reader) {
  hsa_loaded_code_object_t loaded{};
  return core.hsa_executable_load_agent_code_object_fn(
      hsa_executable_t{}, fake_agent(), reader, nullptr, &loaded);
}

struct LoadResult {
  hsa_status_t status = HSA_STATUS_SUCCESS;
  uint64_t original_reader = 0;
  uint64_t loaded_reader = 0;
  int retarget_calls = 0;
  std::string source_isa;
  std::string target_isa;
  size_t retained_elfs = 0;
};

LoadResult load_once(const char *code_object_isa, const char *agent_isa,
                     const char *gfx12_5_rewrite_flag,
                     uint32_t asic_revision = 1,
                     int retarget_status = 0) {
  reset_state();
  set_gfx12_5_rewrite_env(gfx12_5_rewrite_flag);
  g_env.agent_isa = agent_isa;
  g_env.asic_revision = asic_revision;
  g_env.retarget_status = retarget_status;

  CoreApiTable &core = install_tool();
  std::vector<uint8_t> elf = make_code_object(code_object_isa);
  const hsa_code_object_reader_t reader = create_memory_reader(core, elf);

  LoadResult result;
  result.original_reader = reader.handle;
  result.status = load_reader(core, reader);
  result.loaded_reader = g_env.last_loaded_reader;
  result.retarget_calls = g_env.retarget_calls;
  result.source_isa = g_env.retarget_source_isa;
  result.target_isa = g_env.retarget_target_isa;
  result.retained_elfs = g_rewritten_elfs.size();
  return result;
}

void check_original_load(const LoadResult &result, const char *name) {
  check(result.status == HSA_STATUS_SUCCESS, "load succeeds");
  check(result.retarget_calls == 0, name);
  check(result.loaded_reader == result.original_reader,
        "original reader is loaded");
}

} // namespace

#endif // ROCR_HOTSWAP_LOADER_TEST_UTILS_HPP
