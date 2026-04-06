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
#include "hotswap_comgr_client.hpp"
#include <hsa.h>
#include <hsa_api_trace.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#define HSA_HOTSWAP_EXPORT __attribute__((visibility("default")))

namespace {

struct CodeObjectData {
  std::vector<uint8_t> bytes;
};

std::mutex g_reader_map_mutex;
std::unordered_map<uint64_t, CodeObjectData> g_reader_map;

CoreApiTable* g_core_table = nullptr;

decltype(hsa_code_object_reader_create_from_memory)* g_orig_reader_create_from_memory = nullptr;
decltype(hsa_code_object_reader_create_from_file)* g_orig_reader_create_from_file = nullptr;
decltype(hsa_code_object_reader_destroy)* g_orig_reader_destroy = nullptr;
decltype(hsa_executable_load_agent_code_object)* g_orig_load_agent_code_object = nullptr;

hsa_status_t HSA_API hotswap_reader_create_from_memory(
    const void* code_object, size_t size,
    hsa_code_object_reader_t* code_object_reader) {
  hsa_status_t status = g_orig_reader_create_from_memory(code_object, size, code_object_reader);
  if (status != HSA_STATUS_SUCCESS)
    return status;

  std::lock_guard<std::mutex> lock(g_reader_map_mutex);
  auto& data = g_reader_map[code_object_reader->handle];
  data.bytes.assign(static_cast<const uint8_t*>(code_object),
                    static_cast<const uint8_t*>(code_object) + size);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API hotswap_reader_create_from_file(
    hsa_file_t file, hsa_code_object_reader_t* code_object_reader) {
  // Read file contents into memory so we can inspect/rewrite later.
  off_t file_size = lseek(file, 0, SEEK_END);
  if (file_size <= 0) {
    return g_orig_reader_create_from_file(file, code_object_reader);
  }
  lseek(file, 0, SEEK_SET);

  std::vector<uint8_t> buf(static_cast<size_t>(file_size));
  ssize_t bytes_read = read(file, buf.data(), buf.size());
  if (bytes_read != static_cast<ssize_t>(file_size)) {
    lseek(file, 0, SEEK_SET);
    return g_orig_reader_create_from_file(file, code_object_reader);
  }

  hsa_status_t status = g_orig_reader_create_from_memory(
      buf.data(), buf.size(), code_object_reader);
  if (status != HSA_STATUS_SUCCESS)
    return status;

  std::lock_guard<std::mutex> lock(g_reader_map_mutex);
  g_reader_map[code_object_reader->handle].bytes = std::move(buf);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API hotswap_reader_destroy(
    hsa_code_object_reader_t code_object_reader) {
  {
    std::lock_guard<std::mutex> lock(g_reader_map_mutex);
    g_reader_map.erase(code_object_reader.handle);
  }
  return g_orig_reader_destroy(code_object_reader);
}

static std::string get_agent_isa_name(hsa_agent_t agent) {
  auto cb = [](hsa_isa_t isa, void* data) -> hsa_status_t {
    auto* name = static_cast<std::string*>(data);
    uint32_t len = 0;
    hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME_LENGTH, &len);
    name->resize(len);
    hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, name->data());
    if (!name->empty() && name->back() == '\0')
      name->pop_back();
    return HSA_STATUS_INFO_BREAK;
  };
  std::string name;
  hsa_agent_iterate_isas(agent, cb, &name);
  return name;
}

static std::string read_elf_isa_note(const uint8_t* elf, size_t size) {
  const char prefix[] = "amdgcn-amd-amdhsa--";
  const size_t prefix_len = sizeof(prefix) - 1;
  for (size_t i = 0; i + prefix_len <= size; ++i) {
    if (memcmp(elf + i, prefix, prefix_len) == 0) {
      const char* start = reinterpret_cast<const char*>(elf + i);
      size_t len = 0;
      while (i + len < size && start[len] != '\0' && start[len] != '\n'
             && start[len] != ' ' && len < 128)
        ++len;
      return std::string(start, len);
    }
  }
  return {};
}

hsa_status_t HSA_API hotswap_load_agent_code_object(
    hsa_executable_t executable, hsa_agent_t agent,
    hsa_code_object_reader_t code_object_reader,
    const char* options, hsa_loaded_code_object_t* loaded_code_object) {

  std::vector<uint8_t> local_bytes;
  bool have_bytes = false;

  {
    std::lock_guard<std::mutex> lock(g_reader_map_mutex);
    auto it = g_reader_map.find(code_object_reader.handle);
    if (it != g_reader_map.end()) {
      local_bytes = it->second.bytes;
      have_bytes = true;
    }
  }

  if (!have_bytes || !rocr::hotswap::ComgrHotswapAvailable()) {
    return g_orig_load_agent_code_object(executable, agent, code_object_reader,
                                         options, loaded_code_object);
  }

  std::string source_isa = read_elf_isa_note(local_bytes.data(), local_bytes.size());
  std::string target_isa = get_agent_isa_name(agent);

  if (source_isa.empty() || target_isa.empty() || source_isa == target_isa) {
    return g_orig_load_agent_code_object(executable, agent, code_object_reader,
                                         options, loaded_code_object);
  }

  void* out_elf = nullptr;
  size_t out_elf_size = 0;
  int rc = rocr::hotswap::ComgrHotswapRewrite(
      local_bytes.data(), local_bytes.size(),
      source_isa.c_str(), target_isa.c_str(),
      &out_elf, &out_elf_size);

  if (rc != 0 || !out_elf) {
    fprintf(stderr, "hotswap: COMGR rewrite failed for %s -> %s (rc=%d), "
            "falling back to original\n",
            source_isa.c_str(), target_isa.c_str(), rc);
    return g_orig_load_agent_code_object(executable, agent, code_object_reader,
                                         options, loaded_code_object);
  }

  hsa_code_object_reader_t new_reader;
  hsa_status_t status = g_orig_reader_create_from_memory(
      out_elf, out_elf_size, &new_reader);
  if (status != HSA_STATUS_SUCCESS) {
    std::free(out_elf);
    return g_orig_load_agent_code_object(executable, agent, code_object_reader,
                                         options, loaded_code_object);
  }

  status = g_orig_load_agent_code_object(executable, agent, new_reader,
                                         options, loaded_code_object);
  g_orig_reader_destroy(new_reader);
  std::free(out_elf);
  return status;
}

} // anonymous namespace

extern "C" {

HSA_HOTSWAP_EXPORT
bool OnLoad(HsaApiTable* table, uint64_t runtime_version,
            uint64_t failed_count, const char* const* failed_names) {
  (void)runtime_version;
  (void)failed_count;
  (void)failed_names;

  if (!table || !table->core_)
    return false;

  CoreApiTable* core = table->core_;

  if (!core->hsa_code_object_reader_create_from_memory_fn ||
      !core->hsa_code_object_reader_create_from_file_fn ||
      !core->hsa_code_object_reader_destroy_fn ||
      !core->hsa_executable_load_agent_code_object_fn)
    return false;

  g_core_table = core;

  g_orig_reader_create_from_memory =
      core->hsa_code_object_reader_create_from_memory_fn;
  g_orig_reader_create_from_file =
      core->hsa_code_object_reader_create_from_file_fn;
  g_orig_reader_destroy =
      core->hsa_code_object_reader_destroy_fn;
  g_orig_load_agent_code_object =
      core->hsa_executable_load_agent_code_object_fn;

  core->hsa_code_object_reader_create_from_memory_fn =
      hotswap_reader_create_from_memory;
  core->hsa_code_object_reader_create_from_file_fn =
      hotswap_reader_create_from_file;
  core->hsa_code_object_reader_destroy_fn =
      hotswap_reader_destroy;
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
    g_core_table->hsa_code_object_reader_destroy_fn =
        g_orig_reader_destroy;
    g_core_table->hsa_executable_load_agent_code_object_fn =
        g_orig_load_agent_code_object;
    g_core_table = nullptr;
  }

  g_orig_reader_create_from_memory = nullptr;
  g_orig_reader_create_from_file = nullptr;
  g_orig_reader_destroy = nullptr;
  g_orig_load_agent_code_object = nullptr;

  std::lock_guard<std::mutex> lock(g_reader_map_mutex);
  g_reader_map.clear();

  fprintf(stderr, "hotswap: tool unloaded\n");
}

} // extern "C"
