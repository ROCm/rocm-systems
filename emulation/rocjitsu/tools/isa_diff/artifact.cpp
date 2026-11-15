// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "artifact.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/kernel_descriptor_abi.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "util/sha256.h"

#include <elf.h>

#include <cstddef>
#include <cstring>
#include <fstream>
#include <iterator>
#include <optional>

namespace rocjitsu::isa_diff {
namespace {

template <typename T> bool read_struct(std::span<const uint8_t> image, uint64_t offset, T &out) {
  if (offset > image.size() || image.size() - offset < sizeof(T))
    return false;
  std::memcpy(&out, image.data() + offset, sizeof(T));
  return true;
}

struct ElfText {
  uint64_t offset = 0;
  uint64_t size = 0;
  uint64_t vaddr = 0;
  size_t index = 0;
};

struct ElfKernelSymbol {
  std::string name;
  uint64_t value = 0;
  uint64_t size = 0;
};

bool scan_elf(std::span<const uint8_t> image, ElfText &text, std::vector<ElfKernelSymbol> &funcs,
              std::vector<std::pair<uint64_t, uint64_t>> &notes, std::string &error) {
  Elf64_Ehdr eh{};
  if (!read_struct(image, 0, eh) || std::memcmp(eh.e_ident, ELFMAG, SELFMAG) != 0 ||
      eh.e_ident[EI_CLASS] != ELFCLASS64) {
    error = "not an ELF64 image";
    return false;
  }
  if (eh.e_shentsize != sizeof(Elf64_Shdr)) {
    error = "unexpected section header size";
    return false;
  }
  std::vector<Elf64_Shdr> sections(eh.e_shnum);
  for (size_t i = 0; i < sections.size(); ++i)
    if (!read_struct(image, eh.e_shoff + i * sizeof(Elf64_Shdr), sections[i])) {
      error = "truncated section header table";
      return false;
    }
  if (eh.e_shstrndx >= sections.size()) {
    error = "bad section name table index";
    return false;
  }
  const Elf64_Shdr &shstr = sections[eh.e_shstrndx];
  auto section_name = [&](const Elf64_Shdr &sh) -> std::string {
    const uint64_t off = shstr.sh_offset + sh.sh_name;
    if (off >= image.size())
      return {};
    const char *begin = reinterpret_cast<const char *>(image.data() + off);
    return std::string(begin, strnlen(begin, image.size() - off));
  };
  bool have_text = false;
  for (size_t i = 0; i < sections.size(); ++i) {
    const Elf64_Shdr &sh = sections[i];
    if (sh.sh_type == SHT_NOTE)
      notes.emplace_back(sh.sh_offset, sh.sh_size);
    if (section_name(sh) == ".text") {
      if (have_text) {
        error = "multiple .text sections are not supported in P0";
        return false;
      }
      have_text = true;
      text = {sh.sh_offset, sh.sh_size, sh.sh_addr, i};
    }
  }
  if (!have_text) {
    error = "no .text section";
    return false;
  }
  if (text.offset > image.size() || image.size() - text.offset < text.size) {
    error = ".text extends past the image";
    return false;
  }
  for (const Elf64_Shdr &sh : sections) {
    if (sh.sh_type != SHT_SYMTAB || sh.sh_entsize != sizeof(Elf64_Sym) ||
        sh.sh_link >= sections.size())
      continue;
    const Elf64_Shdr &strtab = sections[sh.sh_link];
    const uint64_t count = sh.sh_size / sizeof(Elf64_Sym);
    for (uint64_t i = 0; i < count; ++i) {
      Elf64_Sym sym{};
      if (!read_struct(image, sh.sh_offset + i * sizeof(Elf64_Sym), sym))
        break;
      if (ELF64_ST_TYPE(sym.st_info) != STT_FUNC || sym.st_shndx != text.index)
        continue;
      const uint64_t off = strtab.sh_offset + sym.st_name;
      if (off >= image.size())
        continue;
      const char *begin = reinterpret_cast<const char *>(image.data() + off);
      funcs.push_back(
          {std::string(begin, strnlen(begin, image.size() - off)), sym.st_value, sym.st_size});
    }
  }
  return true;
}

} // namespace

std::string sha256_hex(std::span<const uint8_t> bytes) {
  util::Sha256 sha;
  sha.update(bytes.data(), bytes.size());
  const auto digest = sha.finish();
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(64);
  for (uint8_t b : digest) {
    out += kHex[b >> 4];
    out += kHex[b & 15];
  }
  return out;
}

std::string sha256_hex(std::string_view text) {
  return sha256_hex(
      std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(text.data()), text.size()));
}

Artifact artifact_from_words(std::span<const uint32_t> words, std::string label,
                             uint64_t text_vaddr) {
  Artifact a;
  a.label = std::move(label);
  a.kind = "raw_words";
  a.text.resize(words.size() * sizeof(uint32_t));
  std::memcpy(a.text.data(), words.data(), a.text.size());
  a.sha256 = sha256_hex(a.text);
  a.text_vaddr = text_vaddr;
  a.kernel_begin = 0;
  a.kernel_end = a.text.size();
  a.kernel_name = "<raw>";
  return a;
}

bool artifact_from_code_object(std::span<const uint8_t> image, const std::string &label,
                               const std::string &kernel, Artifact &out, std::string &error) {
  AmdGpuCodeObject co(image.data(), image.size());
  if (!co.is_valid()) {
    error = "invalid AMDGPU code object";
    return false;
  }
  if (co.target_id() != ROCJITSU_CODE_TARGET_GFX950) {
    error = "code object target is not gfx950 (P0 target)";
    return false;
  }
  ElfText text;
  std::vector<ElfKernelSymbol> funcs;
  std::vector<std::pair<uint64_t, uint64_t>> notes;
  if (!scan_elf(image, text, funcs, notes, error))
    return false;
  const auto kernels = scan_kernel_descriptors(image, text.offset, text.size, text.index);
  const KernelDescriptorInfo *chosen = nullptr;
  for (const auto &k : kernels) {
    if (kernel.empty() || k.kernel_name == kernel) {
      if (chosen && kernel.empty()) {
        error = "code object has several kernels; pass --kernel";
        return false;
      }
      chosen = &k;
    }
  }
  if (!chosen) {
    error = kernel.empty() ? "code object has no kernel descriptor"
                           : "kernel '" + kernel + "' not found";
    return false;
  }
  const ElfKernelSymbol *sym = nullptr;
  for (const auto &f : funcs)
    if (f.name == chosen->kernel_name)
      sym = &f;
  if (!sym || sym->size == 0) {
    error = "kernel symbol '" + chosen->kernel_name + "' is missing or has st_size 0";
    return false;
  }
  if (sym->value < text.vaddr || sym->value - text.vaddr != chosen->entry_text_offset) {
    error = "kernel symbol and descriptor disagree on the entry";
    return false;
  }
  out = {};
  out.label = label;
  out.kind = "code_object";
  out.sha256 = sha256_hex(image);
  out.text.assign(image.begin() + static_cast<std::ptrdiff_t>(text.offset),
                  image.begin() + static_cast<std::ptrdiff_t>(text.offset + text.size));
  out.text_vaddr = text.vaddr;
  out.kernel_begin = chosen->entry_text_offset;
  out.kernel_end = chosen->entry_text_offset + sym->size;
  if (out.kernel_end > text.size) {
    error = "kernel symbol extends past .text";
    return false;
  }
  out.kernel_name = chosen->kernel_name;
  out.descriptor.resize(sizeof(chosen->descriptor));
  std::memcpy(out.descriptor.data(), &chosen->descriptor, sizeof(chosen->descriptor));
  util::Sha256 notes_sha;
  for (const auto &[off, size] : notes)
    if (off <= image.size() && image.size() - off >= size)
      notes_sha.update(image.data() + off, size);
  const auto digest = notes_sha.finish();
  out.metadata_sha256 = sha256_hex(std::span<const uint8_t>(digest.data(), digest.size()));
  return true;
}

bool load_artifact(const std::string &path, const std::string &kernel, Artifact &out,
                   std::string &error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    error = "cannot open " + path;
    return false;
  }
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                             std::istreambuf_iterator<char>());
  if (bytes.size() >= 4 && std::memcmp(bytes.data(), ELFMAG, SELFMAG) == 0)
    return artifact_from_code_object(bytes, path, kernel, out, error);
  if (bytes.empty() || bytes.size() % 4 != 0) {
    error = path + ": raw word stream must be a non-empty multiple of 4 bytes";
    return false;
  }
  std::vector<uint32_t> words(bytes.size() / 4);
  std::memcpy(words.data(), bytes.data(), bytes.size());
  out = artifact_from_words(words, path);
  return true;
}

std::vector<std::string> descriptor_semantic_differences(const std::vector<uint8_t> &a,
                                                         const std::vector<uint8_t> &b,
                                                         bool &layout_only_change) {
  layout_only_change = false;
  std::vector<std::string> out;
  if (a.size() != b.size()) {
    out.push_back("descriptor presence");
    return out;
  }
  if (a.empty())
    return out;
  amdhsa::kernel_descriptor_t da{}, db{};
  std::memcpy(&da, a.data(), sizeof(da));
  std::memcpy(&db, b.data(), sizeof(db));
#define RJ_ISA_DIFF_FIELD(name)                                                                    \
  if (std::memcmp(&da.name, &db.name, sizeof(da.name)) != 0)                                       \
    out.push_back(#name);
  RJ_ISA_DIFF_FIELD(group_segment_fixed_size)
  RJ_ISA_DIFF_FIELD(private_segment_fixed_size)
  RJ_ISA_DIFF_FIELD(kernarg_size)
  RJ_ISA_DIFF_FIELD(reserved0)
  RJ_ISA_DIFF_FIELD(reserved1)
  RJ_ISA_DIFF_FIELD(compute_pgm_rsrc3)
  RJ_ISA_DIFF_FIELD(compute_pgm_rsrc1)
  RJ_ISA_DIFF_FIELD(compute_pgm_rsrc2)
  RJ_ISA_DIFF_FIELD(kernel_code_properties)
  RJ_ISA_DIFF_FIELD(kernarg_preload)
  RJ_ISA_DIFF_FIELD(reserved3)
#undef RJ_ISA_DIFF_FIELD
  if (da.kernel_code_entry_byte_offset != db.kernel_code_entry_byte_offset && out.empty())
    layout_only_change = true;
  return out;
}

} // namespace rocjitsu::isa_diff
