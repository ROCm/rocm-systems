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
#include "hotswap_platform_io.hpp"
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <elf.h>
#include <hsa.h>
#include <hsa_api_trace.h>
#include <memory>
#include <mutex>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#define HSA_HOTSWAP_EXPORT __attribute__((visibility("default")))

namespace {

namespace hotswap_io = rocr::hotswap::platform_io;

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

bool checked_add(size_t lhs, size_t rhs, size_t *out) {
  if (lhs > std::numeric_limits<size_t>::max() - rhs) {
    return false;
  }
  *out = lhs + rhs;
  return true;
}

bool checked_mul(size_t lhs, size_t rhs, size_t *out) {
  if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
    return false;
  }
  *out = lhs * rhs;
  return true;
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

// Validate ELF64 header and return pointer, or nullptr on failure.
const Elf64_Ehdr *validate_elf64(const uint8_t *elf, size_t size) {
  if (size < sizeof(Elf64_Ehdr)) {
    return nullptr;
  }
  const auto *ehdr = reinterpret_cast<const Elf64_Ehdr *>(elf);
  if (memcmp(ehdr->e_ident, ELFMAG, SELFMAG) != 0) {
    return nullptr;
  }
  if (ehdr->e_ident[EI_CLASS] != ELFCLASS64) {
    return nullptr;
  }
  return ehdr;
}

bool validate_program_header_table(const Elf64_Ehdr *ehdr, size_t size) {
  return ehdr->e_phoff != 0 && ehdr->e_phoff <= size && ehdr->e_phnum != 0 &&
         ehdr->e_phentsize >= sizeof(Elf64_Phdr);
}

bool compute_program_header_offset(const Elf64_Ehdr *ehdr, size_t size,
                                   uint16_t index, size_t *hdr_offset) {
  size_t hdr_index_offset = 0;
  if (!checked_mul(static_cast<size_t>(index),
                   static_cast<size_t>(ehdr->e_phentsize),
                   &hdr_index_offset) ||
      !checked_add(ehdr->e_phoff, hdr_index_offset, hdr_offset) ||
      *hdr_offset > size || sizeof(Elf64_Phdr) > size - *hdr_offset) {
    return false;
  }
  return true;
}

bool compute_note_segment_bounds(const Elf64_Phdr *phdr, size_t size,
                                 size_t *note_offset, size_t *note_end) {
  *note_offset = phdr->p_offset;
  return *note_offset <= size && phdr->p_filesz <= size - *note_offset &&
         checked_add(*note_offset, phdr->p_filesz, note_end);
}

bool compute_note_layout(size_t note_offset, size_t note_end,
                         const Elf64_Nhdr *nhdr, size_t *desc_off,
                         size_t *next_note) {
  size_t raw_name_size = 0;
  size_t raw_desc_size = 0;
  size_t name_sz_aligned = 0;
  size_t desc_sz_aligned = 0;
  if (!checked_add(static_cast<size_t>(nhdr->n_namesz), 3, &raw_name_size) ||
      !checked_add(static_cast<size_t>(nhdr->n_descsz), 3, &raw_desc_size)) {
    return false;
  }

  name_sz_aligned = raw_name_size & ~size_t{3};
  desc_sz_aligned = raw_desc_size & ~size_t{3};
  return checked_add(note_offset, sizeof(Elf64_Nhdr), desc_off) &&
         checked_add(*desc_off, name_sz_aligned, desc_off) &&
         checked_add(*desc_off, desc_sz_aligned, next_note) &&
         *next_note <= note_end;
}

// Search a single NT_AMDGPU_METADATA note descriptor for the ISA triple.
std::string find_isa_in_metadata(const char *desc, size_t desc_size) {
  const char prefix[] = "amdgcn-amd-amdhsa--";
  const size_t prefix_len = sizeof(prefix) - 1;
  for (size_t j = 0; j + prefix_len <= desc_size; ++j) {
    if (memcmp(desc + j, prefix, prefix_len) == 0) {
      size_t len = 0;
      while (j + len < desc_size && desc[j + len] != '\0' &&
             desc[j + len] != '\n' && desc[j + len] != '\'' &&
             desc[j + len] != '"' && desc[j + len] != ' ') {
        ++len;
      }
      return std::string(desc + j, len);
    }
  }
  return {};
}

std::string read_elf_isa_from_note_segment(const uint8_t *elf, size_t note_offset,
                                           size_t note_end) {
  while (note_offset <= note_end &&
         sizeof(Elf64_Nhdr) <= note_end - note_offset) {
    const auto *nhdr = reinterpret_cast<const Elf64_Nhdr *>(elf + note_offset);
    size_t desc_off = 0;
    size_t next_note = 0;
    if (!compute_note_layout(note_offset, note_end, nhdr, &desc_off,
                             &next_note)) {
      break;
    }

    constexpr uint32_t NT_AMDGPU_METADATA = 32;
    if (nhdr->n_type == NT_AMDGPU_METADATA && nhdr->n_descsz > 0 &&
        desc_off + nhdr->n_descsz <= note_end) {
      const char *desc = reinterpret_cast<const char *>(elf + desc_off);
      std::string result = find_isa_in_metadata(desc, nhdr->n_descsz);
      if (!result.empty()) {
        return result;
      }
    }

    note_offset = next_note;
  }
  return {};
}

// Parse ELF PT_NOTE segments to find the AMDGPU ISA name from
// NT_AMDGPU_METADATA (type 32) notes in v3+ code objects.
std::string read_elf_isa_note(const uint8_t *elf, size_t size) {
  const Elf64_Ehdr *ehdr = validate_elf64(elf, size);
  if (!ehdr || !validate_program_header_table(ehdr, size)) {
    return {};
  }

  for (uint16_t i = 0; i < ehdr->e_phnum; ++i) {
    size_t hdr_offset = 0;
    if (!compute_program_header_offset(ehdr, size, i, &hdr_offset)) {
      break;
    }
    const auto *phdr =
        reinterpret_cast<const Elf64_Phdr *>(elf + hdr_offset);
    if (phdr->p_type != PT_NOTE) {
      continue;
    }

    size_t note_offset = 0;
    size_t note_end = 0;
    if (!compute_note_segment_bounds(phdr, size, &note_offset, &note_end)) {
      continue;
    }

    std::string result = read_elf_isa_from_note_segment(elf, note_offset,
                                                        note_end);
    if (!result.empty()) {
      return result;
    }
  }
  return {};
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

std::string get_agent_isa_name(hsa_agent_t agent) {
  auto cb = [](hsa_isa_t isa, void *data) -> hsa_status_t {
    try {
      auto *name = static_cast<std::string *>(data);
      uint32_t len = 0;
      if (hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME_LENGTH, &len) !=
          HSA_STATUS_SUCCESS) {
        return HSA_STATUS_ERROR;
      }
      name->resize(len);
      if (hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, name->data()) !=
          HSA_STATUS_SUCCESS) {
        name->clear();
        return HSA_STATUS_ERROR;
      }
      if (!name->empty() && name->back() == '\0') {
        name->pop_back();
      }
      return HSA_STATUS_INFO_BREAK;
    } catch (const std::bad_alloc &) {
      auto *name = static_cast<std::string *>(data);
      name->clear();
      return HSA_STATUS_ERROR;
    }
  };
  std::string name;
  hsa_agent_iterate_isas(agent, cb, &name);
  return name;
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
  const std::string source_isa =
      read_elf_isa_note(local_bytes->data(), local_bytes->size());
  const std::string target_isa = get_agent_isa_name(agent);

  if (source_isa.empty() || target_isa.empty()) {
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }

  // Route through RetargetCodeObject for unified logging, validation,
  // and COMGR interaction. Do NOT skip when source == target: B0-to-A0
  // patching uses the same ISA name on both sides.
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
