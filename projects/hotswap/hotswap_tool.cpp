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
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <elf.h>
#include <memory>
#include <mutex>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#define HSA_HOTSWAP_EXPORT __attribute__((visibility("default")))

namespace {

using ByteVec = std::shared_ptr<std::vector<uint8_t>>;

struct CodeObjectData {
  ByteVec bytes;
};

std::mutex g_reader_map_mutex;
std::unordered_map<uint64_t, CodeObjectData> g_reader_map;

// Rewritten ELF buffers must outlive the executable because ROCR's
// LoadedCodeObjectImpl stores a raw pointer to the ELF data (used by
// debuggers, profilers, and hsa_ven_amd_loader queries). We keep them
// alive until OnUnload.
std::mutex g_rewritten_elfs_mutex;
std::vector<void*> g_rewritten_elfs;

CoreApiTable* g_core_table = nullptr;

decltype(hsa_code_object_reader_create_from_memory)* g_orig_reader_create_from_memory = nullptr;
decltype(hsa_code_object_reader_create_from_file)* g_orig_reader_create_from_file = nullptr;
decltype(hsa_code_object_reader_destroy)* g_orig_reader_destroy = nullptr;
decltype(hsa_executable_load_agent_code_object)* g_orig_load_agent_code_object = nullptr;

static void stash_bytes(uint64_t handle, const uint8_t* data, size_t size) {
  auto vec = std::make_shared<std::vector<uint8_t>>(data, data + size);
  std::lock_guard<std::mutex> lock(g_reader_map_mutex);
  g_reader_map[handle].bytes = std::move(vec);
}

static ssize_t read_all(int fd, void* buf, size_t count) {
  size_t total = 0;
  while (total < count) {
    ssize_t n = read(fd, static_cast<uint8_t*>(buf) + total, count - total);
    if (n > 0) {
      total += static_cast<size_t>(n);
    } else if (n == 0) {
      break;
    } else if (errno != EINTR) {
      return -1;
    }
  }
  return static_cast<ssize_t>(total);
}

// Parse ELF PT_NOTE segments to find the AMDGPU ISA note (NT_AMDGPU_HSA_ISA
// type 3, owner "AMDGPU") and return the ISA name string.
static std::string read_elf_isa_note(const uint8_t* elf, size_t size) {
  if (size < sizeof(Elf64_Ehdr))
    return {};
  const auto* ehdr = reinterpret_cast<const Elf64_Ehdr*>(elf);
  if (memcmp(ehdr->e_ident, ELFMAG, SELFMAG) != 0)
    return {};
  if (ehdr->e_ident[EI_CLASS] != ELFCLASS64)
    return {};

  const size_t phoff = ehdr->e_phoff;
  const uint16_t phnum = ehdr->e_phnum;
  const uint16_t phentsize = ehdr->e_phentsize;
  if (phoff == 0 || phnum == 0 || phentsize < sizeof(Elf64_Phdr))
    return {};

  for (uint16_t i = 0; i < phnum; ++i) {
    size_t hdr_offset = phoff + static_cast<size_t>(i) * phentsize;
    if (hdr_offset + sizeof(Elf64_Phdr) > size)
      break;
    const auto* phdr = reinterpret_cast<const Elf64_Phdr*>(elf + hdr_offset);
    if (phdr->p_type != PT_NOTE)
      continue;

    size_t note_offset = phdr->p_offset;
    const size_t note_end = note_offset + phdr->p_filesz;
    if (note_end > size)
      continue;

    while (note_offset + sizeof(Elf64_Nhdr) <= note_end) {
      const auto* nhdr = reinterpret_cast<const Elf64_Nhdr*>(elf + note_offset);
      size_t name_off = note_offset + sizeof(Elf64_Nhdr);
      size_t name_sz_aligned = (nhdr->n_namesz + 3) & ~3u;
      size_t desc_off = name_off + name_sz_aligned;
      size_t desc_sz_aligned = (nhdr->n_descsz + 3) & ~3u;
      size_t next_note = desc_off + desc_sz_aligned;

      if (next_note > note_end)
        break;

      // NT_AMDGPU_METADATA (type 32, owner "AMDGPU") contains msgpack
      // metadata with the ISA triple for v3+ code objects.
      constexpr uint32_t NT_AMDGPU_METADATA = 32;
      if (nhdr->n_type == NT_AMDGPU_METADATA &&
          nhdr->n_descsz > 0 && desc_off + nhdr->n_descsz <= note_end) {
        const char* desc = reinterpret_cast<const char*>(elf + desc_off);
        const char prefix[] = "amdgcn-amd-amdhsa--";
        const size_t prefix_len = sizeof(prefix) - 1;
        for (size_t j = 0; j + prefix_len <= nhdr->n_descsz; ++j) {
          if (memcmp(desc + j, prefix, prefix_len) == 0) {
            size_t len = 0;
            while (j + len < nhdr->n_descsz && desc[j + len] != '\0' &&
                   desc[j + len] != '\n' && desc[j + len] != '\'' &&
                   desc[j + len] != '"' && desc[j + len] != ' ')
              ++len;
            return std::string(desc + j, len);
          }
        }
      }

      note_offset = next_note;
    }
  }
  return {};
}

hsa_status_t HSA_API hotswap_reader_create_from_memory(
    const void* code_object, size_t size,
    hsa_code_object_reader_t* code_object_reader) {
  hsa_status_t status = g_orig_reader_create_from_memory(code_object, size, code_object_reader);
  if (status != HSA_STATUS_SUCCESS)
    return status;

  stash_bytes(code_object_reader->handle,
              static_cast<const uint8_t*>(code_object), size);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API hotswap_reader_create_from_file(
    hsa_file_t file, hsa_code_object_reader_t* code_object_reader) {
  // Read file into memory so we can inspect/rewrite later.
  // NOTE: this converts the file-based reader to a memory-based reader,
  // which loses URI provenance metadata (affects profiler/debugger traces).
  off_t saved_pos = lseek(file, 0, SEEK_CUR);
  off_t file_size = lseek(file, 0, SEEK_END);
  if (file_size <= 0) {
    lseek(file, saved_pos, SEEK_SET);
    return g_orig_reader_create_from_file(file, code_object_reader);
  }
  lseek(file, 0, SEEK_SET);

  std::vector<uint8_t> buf(static_cast<size_t>(file_size));
  ssize_t bytes_read = read_all(file, buf.data(), buf.size());
  if (bytes_read != static_cast<ssize_t>(file_size)) {
    lseek(file, saved_pos, SEEK_SET);
    return g_orig_reader_create_from_file(file, code_object_reader);
  }

  hsa_status_t status = g_orig_reader_create_from_memory(
      buf.data(), buf.size(), code_object_reader);
  if (status != HSA_STATUS_SUCCESS) {
    lseek(file, saved_pos, SEEK_SET);
    return status;
  }

  auto vec = std::make_shared<std::vector<uint8_t>>(std::move(buf));
  std::lock_guard<std::mutex> lock(g_reader_map_mutex);
  g_reader_map[code_object_reader->handle].bytes = std::move(vec);
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
    if (hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME_LENGTH, &len) !=
        HSA_STATUS_SUCCESS)
      return HSA_STATUS_ERROR;
    name->resize(len);
    if (hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, name->data()) !=
        HSA_STATUS_SUCCESS) {
      name->clear();
      return HSA_STATUS_ERROR;
    }
    if (!name->empty() && name->back() == '\0')
      name->pop_back();
    return HSA_STATUS_INFO_BREAK;
  };
  std::string name;
  hsa_agent_iterate_isas(agent, cb, &name);
  return name;
}

hsa_status_t HSA_API hotswap_load_agent_code_object(
    hsa_executable_t executable, hsa_agent_t agent,
    hsa_code_object_reader_t code_object_reader,
    const char* options, hsa_loaded_code_object_t* loaded_code_object) {

  ByteVec local_bytes;

  {
    std::lock_guard<std::mutex> lock(g_reader_map_mutex);
    auto it = g_reader_map.find(code_object_reader.handle);
    if (it != g_reader_map.end())
      local_bytes = it->second.bytes;
  }

  if (!local_bytes || !rocr::hotswap::ComgrHotswapAvailable()) {
    return g_orig_load_agent_code_object(executable, agent, code_object_reader,
                                         options, loaded_code_object);
  }

  std::string source_isa = read_elf_isa_note(local_bytes->data(), local_bytes->size());
  std::string target_isa = get_agent_isa_name(agent);

  if (source_isa.empty() || target_isa.empty()) {
    return g_orig_load_agent_code_object(executable, agent, code_object_reader,
                                         options, loaded_code_object);
  }

  // Do NOT skip when source == target: B0-to-A0 patching uses the same ISA
  // name on both sides. Let COMGR decide whether rewriting is needed.

  void* out_elf = nullptr;
  size_t out_elf_size = 0;
  int rc = rocr::hotswap::ComgrHotswapRewrite(
      local_bytes->data(), local_bytes->size(),
      source_isa.c_str(), target_isa.c_str(),
      &out_elf, &out_elf_size);

  if (rc != 0 || !out_elf) {
    if (out_elf)
      std::free(out_elf);
    if (source_isa != target_isa) {
      fprintf(stderr, "hotswap: COMGR rewrite failed for %s -> %s (rc=%d), "
              "falling back to original\n",
              source_isa.c_str(), target_isa.c_str(), rc);
    }
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

  // ROCR's LoadedCodeObjectImpl holds a raw pointer to the ELF data for
  // debugger/profiler queries. The buffer must outlive the executable.
  {
    std::lock_guard<std::mutex> lock(g_rewritten_elfs_mutex);
    g_rewritten_elfs.push_back(out_elf);
  }

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

  {
    std::lock_guard<std::mutex> lock(g_reader_map_mutex);
    g_reader_map.clear();
  }

  {
    std::lock_guard<std::mutex> lock(g_rewritten_elfs_mutex);
    for (void* p : g_rewritten_elfs)
      std::free(p);
    g_rewritten_elfs.clear();
  }

  fprintf(stderr, "hotswap: tool unloaded\n");
}

} // extern "C"
