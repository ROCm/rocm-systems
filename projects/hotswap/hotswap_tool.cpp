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

std::mutex g_reader_map_mutex;
std::unordered_map<uint64_t, ReaderEntry> g_reader_map;

// Rewritten ELF buffers must outlive the executable because ROCR's
// LoadedCodeObjectImpl stores a raw pointer to the ELF data (used by
// debuggers, profilers, and hsa_ven_amd_loader queries). We keep them
// alive until OnUnload.
std::mutex g_rewritten_elfs_mutex;
std::vector<OwnedElf> g_rewritten_elfs;

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
  std::scoped_lock lock(g_reader_map_mutex);
  g_reader_map[handle] = ReaderEntry{std::move(vec), false, false};
}

bool try_get_reader_entry(uint64_t handle, ByteVec *bytes, bool *from_file) {
  std::scoped_lock lock(g_reader_map_mutex);
  const auto it = g_reader_map.find(handle);
  if (it == g_reader_map.end()) {
    return false;
  }
  *bytes = it->second.bytes;
  *from_file = it->second.from_file;
  return true;
}

void retain_rewritten_elf(OwnedElf elf) {
  try {
    std::scoped_lock lock(g_rewritten_elfs_mutex);
    g_rewritten_elfs.push_back(std::move(elf));
  } catch (const std::bad_alloc &) {
    // Intentionally leak to preserve debugger/profiler correctness if the
    // keepalive vector itself cannot grow.
    (void)elf.release();
  }
}

void mark_reader_keepalive(uint64_t handle) {
  std::scoped_lock lock(g_reader_map_mutex);
  auto it = g_reader_map.find(handle);
  if (it != g_reader_map.end()) {
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
      std::scoped_lock lock(g_reader_map_mutex);
      g_reader_map[reader.handle] = ReaderEntry{std::move(vec), true, false};
    }
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
    std::scoped_lock lock(g_reader_map_mutex);
    auto it = g_reader_map.find(code_object_reader.handle);
    if (it != g_reader_map.end() && !it->second.keepalive_after_load) {
      g_reader_map.erase(it);
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
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }

  void *out_elf = nullptr;
  size_t out_elf_size = 0;
  const int rc = rocr::hotswap::RetargetCodeObject(
      local_bytes->data(), local_bytes->size(), source_isa.c_str(),
      target_isa.c_str(), &out_elf, &out_elf_size);

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

    if (!local_bytes) {
      return load_original_reader(executable, agent, code_object_reader,
                                  options, loaded_code_object,
                                  reader_from_file);
    }

    // Gate HotSwap to gfx1250 A0 silicon. On any other GPU or stepping, load
    // the original code object unchanged instead of routing through COMGR.
    const AgentGfxRevision gfx = query_agent_gfx_revision(agent);
    if (!gate_allows_hotswap(gfx)) {
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
    std::scoped_lock lock(g_reader_map_mutex);
    g_reader_map.clear();
  }

  {
    std::scoped_lock lock(g_rewritten_elfs_mutex);
    g_rewritten_elfs.clear();
  }

  fprintf(stderr, "hotswap: tool unloaded\n");
}

} // extern "C"
