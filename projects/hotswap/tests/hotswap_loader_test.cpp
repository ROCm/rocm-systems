//===- hotswap_loader_test.cpp - Tests for HSA tools loader path ----------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This test includes hotswap_tool.cpp directly so it can drive the wrapped HSA
// API-table entry points without a GPU or a real HSA runtime. The original HSA
// functions and HotSwap COMGR calls are replaced with stubs below.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <hsa.h>

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

} // namespace

#include "../hotswap_tool.cpp"

namespace rocr::hotswap {

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

void set_entry_trampoline_env(const char *value) {
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
  set_entry_trampoline_env(nullptr);
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

CoreApiTable install_tool() {
  CoreApiTable core{};
  core.hsa_code_object_reader_create_from_memory_fn =
      fake_reader_create_from_memory;
  core.hsa_code_object_reader_create_from_file_fn = fake_reader_create_from_file;
  core.hsa_code_object_reader_destroy_fn = fake_reader_destroy;
  core.hsa_executable_load_agent_code_object_fn = fake_load_agent_code_object;

  HsaApiTable table{};
  table.core_ = &core;
  check(OnLoad(&table, 0, 0, nullptr), "OnLoad installs with complete table");
  return core;
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

void begin_test(const char *name, const char *description) {
  std::printf("TEST %s...\n  %s\n", name, description);
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
                     const char *entry_trampoline_flag,
                     uint32_t asic_revision = 1,
                     int retarget_status = 0) {
  reset_state();
  set_entry_trampoline_env(entry_trampoline_flag);
  g_env.agent_isa = agent_isa;
  g_env.asic_revision = asic_revision;
  g_env.retarget_status = retarget_status;

  CoreApiTable core = install_tool();
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

void test_DisabledFlagLoadsOriginal() {
  begin_test("DisabledFlagLoadsOriginal",
             "Unset, empty, and 0 trampoline flags must leave non-A0 gfx1250 "
             "on the original reader.");
  struct FlagCase {
    const char *flag_value;
    const char *expectation;
  };
  const FlagCase cases[] = {
      {nullptr, "unset flag skips non-A0 gfx1250 rewrite"},
      {"", "empty flag skips non-A0 gfx1250 rewrite"},
      {"0", "flag value 0 skips non-A0 gfx1250 rewrite"},
  };
  for (const FlagCase &c : cases) {
    const LoadResult result =
        load_once(kGfx1250Isa, kGfx1250Isa, c.flag_value);
    check_original_load(result, c.expectation);
  }
}

void test_FlagRoutesGfx1250B0() {
  begin_test("FlagRoutesGfx1250B0",
             "The trampoline flag must route non-A0 gfx1250 through COMGR "
             "without selecting the A0 patch path.");
  const LoadResult result = load_once(kGfx1250Isa, kGfx1250Isa, "1");
  check(result.status == HSA_STATUS_SUCCESS, "load succeeds");
  check(result.retarget_calls == 1, "flag routes B0 gfx1250 through COMGR");
  check(result.source_isa == kGfx1250B0Isa,
        "source ISA is tagged as B0");
  check(result.target_isa == kGfx1250B0Isa,
        "non-A0 target ISA is tagged as B0");
  check(result.loaded_reader != result.original_reader,
        "rewritten reader is loaded instead of original reader");
  check(result.retained_elfs == 1,
        "rewritten ELF is retained after successful load");
}

void test_FlagRoutesGfx12_5Family() {
  begin_test("FlagRoutesGfx12_5Family",
             "The trampoline flag must route gfx125* and gfx12-5-generic "
             "without adding gfx1250 stepping features.");
  const char *cases[] = {kGfx1251Isa, kGfx12_5GenericIsa};
  for (const char *isa : cases) {
    const LoadResult result = load_once(isa, isa, "1");
    check(result.status == HSA_STATUS_SUCCESS, "load succeeds");
    check(result.retarget_calls == 1, "flag routes gfx12.5 target through COMGR");
    check(result.source_isa == isa, "source ISA is preserved");
    check(result.target_isa == isa, "target ISA is preserved");
  }
}

void test_GenericSourceUsesGenericTarget() {
  begin_test("GenericSourceUsesGenericTarget",
             "A gfx12-5-generic source loaded on a concrete gfx125* agent "
             "must stay generic to avoid processor retargeting.");
  const LoadResult result = load_once(kGfx12_5GenericIsa, kGfx1251Isa, "1");
  check(result.status == HSA_STATUS_SUCCESS, "load succeeds");
  check(result.retarget_calls == 1,
        "generic source on concrete gfx125 agent routes through COMGR");
  check(result.source_isa == kGfx12_5GenericIsa,
        "generic source ISA is preserved");
  check(result.target_isa == kGfx12_5GenericIsa,
        "generic target ISA is preserved to avoid processor retargeting");
}

void test_A0RetargetsWithoutFlag() {
  begin_test("A0RetargetsWithoutFlag",
             "The existing gfx1250 A0 rewrite path must still run without "
             "the trampoline flag.");
  const LoadResult result = load_once(kGfx1250Isa, kGfx1250Isa, nullptr, 0);
  check(result.status == HSA_STATUS_SUCCESS, "load succeeds");
  check(result.retarget_calls == 1, "A0 gfx1250 keeps default rewrite path");
  check(result.source_isa == kGfx1250B0Isa,
        "source code object ISA is tagged as B0");
  check(result.target_isa == kGfx1250A0Isa,
        "A0 agent ISA is tagged as A0");
}

void test_FlagOneBlocksNonGfx12_5() {
  begin_test("FlagOneBlocksNonGfx12_5",
             "The trampoline flag must not become a global rewrite enable.");
  const LoadResult result = load_once(kGfx942Isa, kGfx942Isa, "1", 0);
  check_original_load(result,
                      "entry trampoline flag does not route non-gfx12.5");
}

void test_RetargetFailureFallsBackToOriginalReader() {
  begin_test("RetargetFailureFallsBackToOriginalReader",
             "If COMGR rejects a gated rewrite, the loader must still load "
             "the original reader.");
  const LoadResult result = load_once(kGfx1250Isa, kGfx1250Isa, "1", 1, -1);
  check(result.status == HSA_STATUS_SUCCESS, "fallback load succeeds");
  check(result.retarget_calls == 1, "COMGR retarget was attempted");
  check(result.loaded_reader == result.original_reader,
        "retarget failure falls back to original reader");
}

} // namespace

int main() {
  test_DisabledFlagLoadsOriginal();
  test_FlagRoutesGfx1250B0();
  test_FlagRoutesGfx12_5Family();
  test_GenericSourceUsesGenericTarget();
  test_A0RetargetsWithoutFlag();
  test_FlagOneBlocksNonGfx12_5();
  test_RetargetFailureFallsBackToOriginalReader();
  reset_state();

  std::printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
  return tests_failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
