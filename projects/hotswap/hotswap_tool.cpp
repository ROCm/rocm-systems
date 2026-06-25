//===- hotswap_tool.cpp - HSA tools lib for HotSwap ISA rewriting ---------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// HSA_TOOLS_LIB entry point for HotSwap. Intercepts code object reader
// creation and executable loading to transparently rewrite code objects
// via COMGR's amd_comgr_hotswap_rewrite.
//
// Usage:
//   HSA_TOOLS_LIB=libhsa-hotswap.so ./my_app
//
//===----------------------------------------------------------------------===//

#include "hotswap.hpp"
#include "hotswap_gfx_query.hpp"
#include "hotswap_platform_io.hpp"
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <hsa.h>
#include <hsa_api_trace.h>
#include <hsa_ext_amd.h>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// The agent's gfx target and ASIC revision are read through the HSA runtime
// (HSA_AMD_AGENT_INFO_ASIC_REVISION) in hotswap_gfx_query.{hpp,cpp}, which is
// portable across Linux and Windows. 

#define HSA_HOTSWAP_EXPORT __attribute__((visibility("default")))

namespace {

namespace hotswap_io = rocr::hotswap::platform_io;

using rocr::hotswap::AgentGfxRevision;
using rocr::hotswap::gate_allows_hotswap;
using rocr::hotswap::get_agent_isa_name;
using rocr::hotswap::query_agent_gfx_revision;

using ByteVec = std::shared_ptr<std::vector<uint8_t>>;
using OwnedElf = std::unique_ptr<void, decltype(&std::free)>;

struct ReaderEntry {
  ByteVec bytes;
  bool from_file = false;
  bool keepalive_after_load = false;
};

// These four are deliberately immortal (heap-allocated, never destroyed). HSA
// calls OnUnload from hsa_shut_down, which is driven by libamdhip64's atexit
// handler -- this runs AFTER this library's C++ static destructors (the tool is
// dlopen'd by hsa_init, so its __cxa_atexit dtors are LIFO-earlier-to-run than
// HIP's shutdown handler). If these were ordinary statics, their destructors
// would free the backing storage first, and OnUnload's clear()/lock would then
// touch freed memory -> use-after-free SIGSEGV. Making them never-destroyed
// singletons removes that hazard. This is NOT a buffer leak: OnUnload still
// clear()s the containers and frees every malloc'd ELF buffer; only the small
// container shells persist to process exit (reclaimed by the OS).
std::mutex &reader_map_mutex() {
  static std::mutex *m = new std::mutex();
  return *m;
}
std::unordered_map<uint64_t, ReaderEntry> &reader_map() {
  static auto *m = new std::unordered_map<uint64_t, ReaderEntry>();
  return *m;
}

// Rewritten ELF buffers must outlive the executable because ROCR's
// LoadedCodeObjectImpl stores a raw pointer to the ELF data (used by
// debuggers, profilers, and hsa_ven_amd_loader queries). We keep them
// alive until OnUnload.
std::mutex &rewritten_elfs_mutex() {
  static std::mutex *m = new std::mutex();
  return *m;
}
std::vector<OwnedElf> &rewritten_elfs() {
  static auto *v = new std::vector<OwnedElf>();
  return *v;
}

// Opt-in diagnostic logging (HSA_HOTSWAP_VERBOSE=1). Off by default so
// production stays quiet; lets us confirm which code objects are intercepted,
// gated, and rewritten. Runtime-gated (no rebuild to toggle); when off the cost
// is a single cached bool load per call site.
bool verbose() {
  static const bool v = [] {
    const char *e = std::getenv("HSA_HOTSWAP_VERBOSE");
    return e && e[0] && e[0] != '0';
  }();
  return v;
}

#define HOTSWAP_LOG(...)                                                        \
  do {                                                                         \
    if (verbose())                                                            \
      fprintf(stderr, __VA_ARGS__);                                           \
  } while (0)

CoreApiTable *g_core_table = nullptr;

decltype(hsa_code_object_reader_create_from_memory)
    *g_orig_reader_create_from_memory = nullptr;
decltype(hsa_code_object_reader_create_from_file)
    *g_orig_reader_create_from_file = nullptr;
decltype(hsa_code_object_reader_destroy) *g_orig_reader_destroy = nullptr;
decltype(hsa_executable_load_agent_code_object) *g_orig_load_agent_code_object =
    nullptr;

void stash_bytes(uint64_t handle, const uint8_t *data, size_t size) {
  auto vec = std::make_shared<std::vector<uint8_t>>(data, data + size);
  std::scoped_lock lock(reader_map_mutex());
  reader_map()[handle] = ReaderEntry{std::move(vec), false, false};
}

bool try_get_reader_entry(uint64_t handle, ByteVec *bytes, bool *from_file) {
  std::scoped_lock lock(reader_map_mutex());
  const auto it = reader_map().find(handle);
  if (it == reader_map().end()) {
    return false;
  }
  *bytes = it->second.bytes;
  *from_file = it->second.from_file;
  return true;
}

void retain_rewritten_elf(OwnedElf elf) {
  try {
    std::scoped_lock lock(rewritten_elfs_mutex());
    rewritten_elfs().push_back(std::move(elf));
  } catch (const std::bad_alloc &) {
    // Intentionally leak to preserve debugger/profiler correctness if the
    // keepalive vector itself cannot grow.
    (void)elf.release();
  }
}

void mark_reader_keepalive(uint64_t handle) {
  std::scoped_lock lock(reader_map_mutex());
  auto it = reader_map().find(handle);
  if (it != reader_map().end()) {
    it->second.keepalive_after_load = true;
  }
}

hsa_status_t HSA_API hotswap_reader_create_from_memory(
    const void *code_object, size_t size,
    hsa_code_object_reader_t *code_object_reader) {
  hsa_code_object_reader_t reader = {};
  const hsa_status_t status =
      g_orig_reader_create_from_memory(code_object, size, &reader);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  try {
    stash_bytes(reader.handle, static_cast<const uint8_t *>(code_object), size);
    HOTSWAP_LOG("hotswap: reader_create_from_memory handle=%lu size=%zu\n",
                static_cast<unsigned long>(reader.handle), size);
  } catch (const std::bad_alloc &) {
    // Fall back to the original load path without rewrite support.
  }
  *code_object_reader = reader;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API hotswap_reader_create_from_file(
    hsa_file_t file, hsa_code_object_reader_t *code_object_reader) {
  // Read file into memory so we can inspect/rewrite later.
  // NOTE: this converts the file-based reader to a memory-based reader,
  // which loses URI provenance metadata (affects profiler/debugger traces).
  hotswap_io::file_pos_t saved_pos = 0;
  hotswap_io::file_pos_t file_size = 0;
  if (!hotswap_io::get_file_bounds(file, &saved_pos, &file_size)) {
    return g_orig_reader_create_from_file(file, code_object_reader);
  }
  hotswap_io::restore_file_pos(file, 0);

  hsa_code_object_reader_t reader = {};
  try {
    auto vec = std::make_shared<std::vector<uint8_t>>(file_size);
    if (!hotswap_io::read_all(file, vec->data(), vec->size())) {
      hotswap_io::restore_file_pos(file, saved_pos);
      return g_orig_reader_create_from_file(file, code_object_reader);
    }

    const hsa_status_t status =
        g_orig_reader_create_from_memory(vec->data(), vec->size(), &reader);
    hotswap_io::restore_file_pos(file, saved_pos);
    if (status != HSA_STATUS_SUCCESS) {
      return status;
    }

    {
      std::scoped_lock lock(reader_map_mutex());
      reader_map()[reader.handle] = ReaderEntry{std::move(vec), true, false};
    }
    HOTSWAP_LOG("hotswap: reader_create_from_file handle=%lu size=%lld\n",
                static_cast<unsigned long>(reader.handle),
                static_cast<long long>(file_size));
    *code_object_reader = reader;
    return HSA_STATUS_SUCCESS;
  } catch (const std::bad_alloc &) {
    hotswap_io::restore_file_pos(file, saved_pos);
    if (reader.handle != 0) {
      g_orig_reader_destroy(reader);
    }
    return g_orig_reader_create_from_file(file, code_object_reader);
  }
}

hsa_status_t HSA_API
hotswap_reader_destroy(hsa_code_object_reader_t code_object_reader) {
  {
    std::scoped_lock lock(reader_map_mutex());
    auto it = reader_map().find(code_object_reader.handle);
    if (it != reader_map().end() && !it->second.keepalive_after_load) {
      reader_map().erase(it);
    }
  }
  return g_orig_reader_destroy(code_object_reader);
}

hsa_status_t load_original_reader(hsa_executable_t executable, hsa_agent_t agent,
                                  hsa_code_object_reader_t code_object_reader,
                                  const char *options,
                                  hsa_loaded_code_object_t *loaded_code_object,
                                  bool reader_from_file) {
  const hsa_status_t status = g_orig_load_agent_code_object(
      executable, agent, code_object_reader, options, loaded_code_object);
  if (status == HSA_STATUS_SUCCESS && reader_from_file) {
    mark_reader_keepalive(code_object_reader.handle);
  }
  return status;
}

hsa_status_t load_rewritten_reader(hsa_executable_t executable, hsa_agent_t agent,
                                   const char *options,
                                   hsa_loaded_code_object_t *loaded_code_object,
                                   void *out_elf, size_t out_elf_size) {
  OwnedElf owned_elf(out_elf, &std::free);
  hsa_code_object_reader_t new_reader = {};
  hsa_status_t status =
      g_orig_reader_create_from_memory(owned_elf.get(), out_elf_size, &new_reader);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  status = g_orig_load_agent_code_object(executable, agent, new_reader, options,
                                         loaded_code_object);
  g_orig_reader_destroy(new_reader);

  if (status == HSA_STATUS_SUCCESS) {
    // ROCR's LoadedCodeObjectImpl holds a raw pointer to the ELF data for
    // debugger/profiler queries. The buffer must outlive the executable.
    retain_rewritten_elf(std::move(owned_elf));
  }

  return status;
}

hsa_status_t try_retarget_and_load(hsa_executable_t executable, hsa_agent_t agent,
                                   hsa_code_object_reader_t code_object_reader,
                                   const char *options,
                                   hsa_loaded_code_object_t *loaded_code_object,
                                   const ByteVec &local_bytes) {
  // Source ISA from the code object, target ISA from the running GPU.
  const std::string source_isa = rocr::hotswap::GetCodeObjectIsaName(
      local_bytes->data(), local_bytes->size());
  const std::string target_isa = get_agent_isa_name(agent);
  if (source_isa.empty() || target_isa.empty()) {
    HOTSWAP_LOG("hotswap: rewrite SKIP empty isa (src='%s' tgt='%s' size=%zu)\n",
                source_isa.c_str(), target_isa.c_str(), local_bytes->size());
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }

  void *out_elf = nullptr;
  size_t out_elf_size = 0;
  const int rc = rocr::hotswap::RetargetCodeObject(
      local_bytes->data(), local_bytes->size(), source_isa.c_str(),
      target_isa.c_str(), &out_elf, &out_elf_size);

  HOTSWAP_LOG("hotswap: rewrite src=%s tgt=%s in=%zu rc=%d out=%zu changed=%d\n",
              source_isa.c_str(), target_isa.c_str(), local_bytes->size(), rc,
              out_elf_size, out_elf != local_bytes->data());

  if (rc != 0 || out_elf == local_bytes->data()) {
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }
  OwnedElf owned_elf(out_elf, &std::free);
  if (!owned_elf || out_elf_size == 0) {
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }

  return load_rewritten_reader(executable, agent, options, loaded_code_object,
                               owned_elf.release(), out_elf_size);
}

hsa_status_t HSA_API hotswap_load_agent_code_object(
    hsa_executable_t executable, hsa_agent_t agent,
    hsa_code_object_reader_t code_object_reader, const char *options,
    hsa_loaded_code_object_t *loaded_code_object) {
  ByteVec local_bytes;
  bool reader_from_file = false;

  try {
    try_get_reader_entry(code_object_reader.handle, &local_bytes,
                         &reader_from_file);

    HOTSWAP_LOG("hotswap: load_agent_code_object handle=%lu has_bytes=%d\n",
                static_cast<unsigned long>(code_object_reader.handle),
                local_bytes ? 1 : 0);

    if (!local_bytes) {
      return load_original_reader(executable, agent, code_object_reader,
                                  options, loaded_code_object,
                                  reader_from_file);
    }

    // Gate HotSwap to gfx1250 A0 silicon. On any other GPU or stepping, load
    // the original code object unchanged instead of routing through COMGR.
    const AgentGfxRevision gfx = query_agent_gfx_revision(agent);
    if (!gate_allows_hotswap(gfx)) {
      HOTSWAP_LOG("hotswap: gate BLOCKED (gfx=%s rev=%u valid=%d)\n",
                  gfx.gfx_target.c_str(), gfx.asic_revision, gfx.revision_valid);
      return load_original_reader(executable, agent, code_object_reader,
                                  options, loaded_code_object,
                                  reader_from_file);
    }

    const hsa_status_t status = try_retarget_and_load(
        executable, agent, code_object_reader, options, loaded_code_object,
        local_bytes);
    if (status == HSA_STATUS_SUCCESS) {
      return status;
    }
    fprintf(stderr,
            "hotswap: rewrite failed (status=%d), falling back to "
            "original code object\n",
            static_cast<int>(status));
    return load_original_reader(executable, agent, code_object_reader, options,
                                loaded_code_object, reader_from_file);
  } catch (const std::bad_alloc &) {
    fprintf(stderr, "hotswap: OOM during rewrite, falling back to "
                    "original code object\n");
    return load_original_reader(executable, agent, code_object_reader, options,
                                loaded_code_object, reader_from_file);
  }
}

} // anonymous namespace

extern "C" {

HSA_HOTSWAP_EXPORT
bool OnLoad(HsaApiTable *table, uint64_t runtime_version, uint64_t failed_count,
            const char *const *failed_names) {
  (void)runtime_version;
  (void)failed_count;
  (void)failed_names;

  if (!table || !table->core_) {
    return false;
  }

  CoreApiTable *core = table->core_;

  if (!core->hsa_code_object_reader_create_from_memory_fn ||
      !core->hsa_code_object_reader_create_from_file_fn ||
      !core->hsa_code_object_reader_destroy_fn ||
      !core->hsa_executable_load_agent_code_object_fn) {
    return false;
  }

  g_core_table = core;

  g_orig_reader_create_from_memory =
      core->hsa_code_object_reader_create_from_memory_fn;
  g_orig_reader_create_from_file =
      core->hsa_code_object_reader_create_from_file_fn;
  g_orig_reader_destroy = core->hsa_code_object_reader_destroy_fn;
  g_orig_load_agent_code_object =
      core->hsa_executable_load_agent_code_object_fn;

  core->hsa_code_object_reader_create_from_memory_fn =
      hotswap_reader_create_from_memory;
  core->hsa_code_object_reader_create_from_file_fn =
      hotswap_reader_create_from_file;
  core->hsa_code_object_reader_destroy_fn = hotswap_reader_destroy;
  core->hsa_executable_load_agent_code_object_fn =
      hotswap_load_agent_code_object;

  fprintf(stderr, "hotswap: tool loaded, intercepting code object loading\n");
  return true;
}

HSA_HOTSWAP_EXPORT
void OnUnload() {
  if (g_core_table) {
    g_core_table->hsa_code_object_reader_create_from_memory_fn =
        g_orig_reader_create_from_memory;
    g_core_table->hsa_code_object_reader_create_from_file_fn =
        g_orig_reader_create_from_file;
    g_core_table->hsa_code_object_reader_destroy_fn = g_orig_reader_destroy;
    g_core_table->hsa_executable_load_agent_code_object_fn =
        g_orig_load_agent_code_object;
    g_core_table = nullptr;
  }

  g_orig_reader_create_from_memory = nullptr;
  g_orig_reader_create_from_file = nullptr;
  g_orig_reader_destroy = nullptr;
  g_orig_load_agent_code_object = nullptr;

  {
    std::scoped_lock lock(reader_map_mutex());
    reader_map().clear();
  }

  {
    std::scoped_lock lock(rewritten_elfs_mutex());
    rewritten_elfs().clear();
  }

  fprintf(stderr, "hotswap: tool unloaded\n");
}

} // extern "C"
