// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "artifact.h"
#include "location.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdhsa_metadata.h"
#include "rocjitsu/code/kernel_descriptor_abi.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "util/sha256.h"

#include <elf.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iterator>
#include <optional>
#include <tuple>

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

/// Inline-asm markers (symbols in .text) and the record section the LLVM plugin wrote.
struct ElfInlineAsm {
  std::vector<ElfKernelSymbol> markers;
  std::string records;
};

bool scan_elf(std::span<const uint8_t> image, ElfText &text, std::vector<ElfKernelSymbol> &funcs,
              std::vector<std::pair<uint64_t, uint64_t>> &notes,
              std::vector<TextRelocation> &relocs, ElfInlineAsm &inline_asm, std::string &error) {
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
    if (section_name(sh) == kInlineAsmSection && sh.sh_type == SHT_PROGBITS &&
        sh.sh_offset <= image.size() && image.size() - sh.sh_offset >= sh.sh_size)
      inline_asm.records.assign(reinterpret_cast<const char *>(image.data() + sh.sh_offset),
                                sh.sh_size);
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
      if (sym.st_shndx != text.index)
        continue;
      const uint64_t off = strtab.sh_offset + sym.st_name;
      if (off >= image.size())
        continue;
      const char *begin = reinterpret_cast<const char *>(image.data() + off);
      std::string name(begin, strnlen(begin, image.size() - off));
      if (name.starts_with(kInlineAsmMarkerPrefix))
        inline_asm.markers.push_back({std::move(name), sym.st_value, 0});
      else if (ELF64_ST_TYPE(sym.st_info) == STT_FUNC)
        funcs.push_back({std::move(name), sym.st_value, sym.st_size});
    }
  }
  // Relocations that patch .text: a static section aimed at .text (sh_info), or any
  // relocation whose target address falls inside .text.
  auto symbol_name = [&](const Elf64_Shdr &rel_sec, uint64_t sym_index) -> std::string {
    if (sym_index == 0 || rel_sec.sh_link >= sections.size())
      return {};
    const Elf64_Shdr &symtab = sections[rel_sec.sh_link];
    if (symtab.sh_entsize != sizeof(Elf64_Sym) || symtab.sh_link >= sections.size())
      return {};
    Elf64_Sym sym{};
    if (!read_struct(image, symtab.sh_offset + sym_index * sizeof(Elf64_Sym), sym))
      return {};
    const uint64_t off = sections[symtab.sh_link].sh_offset + sym.st_name;
    if (off >= image.size())
      return {};
    const char *begin = reinterpret_cast<const char *>(image.data() + off);
    return std::string(begin, strnlen(begin, image.size() - off));
  };
  for (const Elf64_Shdr &sh : sections) {
    const bool rela = sh.sh_type == SHT_RELA;
    if (!rela && sh.sh_type != SHT_REL)
      continue;
    const uint64_t entsize = rela ? sizeof(Elf64_Rela) : sizeof(Elf64_Rel);
    if (sh.sh_entsize != entsize)
      continue;
    const bool aimed_at_text = sh.sh_info == text.index && !(sh.sh_flags & SHF_ALLOC);
    for (uint64_t off = 0; off + entsize <= sh.sh_size; off += entsize) {
      Elf64_Rela r{};
      if (rela) {
        if (!read_struct(image, sh.sh_offset + off, r))
          break;
      } else {
        Elf64_Rel rr{};
        if (!read_struct(image, sh.sh_offset + off, rr))
          break;
        r.r_offset = rr.r_offset;
        r.r_info = rr.r_info;
      }
      uint64_t where = 0;
      if (aimed_at_text && eh.e_type == ET_REL)
        where = r.r_offset;
      else if (r.r_offset >= text.vaddr && r.r_offset - text.vaddr < text.size)
        where = r.r_offset - text.vaddr;
      else
        continue;
      relocs.push_back({where, static_cast<uint32_t>(ELF64_R_TYPE(r.r_info)),
                        symbol_name(sh, ELF64_R_SYM(r.r_info)), r.r_addend});
    }
  }
  std::ranges::sort(relocs, {}, [](const TextRelocation &r) { return r.offset; });
  return true;
}

/// Flatten one MessagePack value: maps join keys ('.'-prefixed keys append directly),
/// arrays append [i], scalars print in a stable form.
void flatten_metadata(const amdhsa::MetadataValue &v, const std::string &prefix,
                      std::map<std::string, std::string> &out) {
  using Kind = amdhsa::MetadataValue::Kind;
  switch (v.kind) {
  case Kind::Map:
    for (const auto &[k, child] : v.map) {
      const std::string key(k);
      flatten_metadata(
          child, prefix.empty() ? key : prefix + (!key.empty() && key[0] == '.' ? "" : ".") + key,
          out);
    }
    if (v.map.empty())
      out[prefix] = "{}";
    return;
  case Kind::Array:
    for (size_t i = 0; i < v.array.size(); ++i)
      flatten_metadata(v.array[i], prefix + "[" + std::to_string(i) + "]", out);
    if (v.array.empty())
      out[prefix] = "[]";
    return;
  case Kind::Integer:
    out[prefix] = std::to_string(v.integer);
    return;
  case Kind::String:
    out[prefix] = "\"" + std::string(v.string) + "\"";
    return;
  case Kind::Boolean:
    out[prefix] = v.integer ? "true" : "false";
    return;
  case Kind::Nil:
    out[prefix] = "nil";
    return;
  }
}

/// Attach the inline-asm blocks inside the kernel range. The record section holds one
/// NUL-terminated record per block: "<marker>\t<before_label 0|1>\t<text>".
bool read_inline_asm(const ElfInlineAsm &info, uint64_t text_vaddr, Artifact &out,
                     std::string &error) {
  std::map<std::string, std::pair<bool, std::string>> records;
  size_t pos = 0;
  while (pos < info.records.size()) {
    size_t end = info.records.find('\0', pos);
    if (end == std::string::npos)
      end = info.records.size();
    const std::string rec = info.records.substr(pos, end - pos);
    pos = end + 1;
    if (rec.empty())
      continue;
    const size_t t1 = rec.find('\t');
    const size_t t2 = t1 == std::string::npos ? t1 : rec.find('\t', t1 + 1);
    if (t2 == std::string::npos) {
      error = "malformed inline-asm record '" + rec + "'";
      return false;
    }
    records[rec.substr(0, t1)] = {rec.substr(t1 + 1, t2 - t1 - 1) == "1", rec.substr(t2 + 1)};
  }
  // Address order; blocks at one address (nothing assembled between them) keep their source
  // order, which the marker's sequence number records.
  std::vector<std::tuple<uint64_t, uint64_t, InlineAsmBlock>> found;
  for (const auto &m : info.markers) {
    if (m.value < text_vaddr)
      continue;
    const uint64_t off = m.value - text_vaddr;
    if (off < out.kernel_begin || off >= out.kernel_end)
      continue;
    auto it = records.find(m.name);
    if (it == records.end()) {
      error = "inline-asm marker '" + m.name + "' has no record";
      return false;
    }
    uint64_t seq = 0;
    const std::string digits = m.name.substr(kInlineAsmMarkerPrefix.size());
    if (digits.empty() || digits.find_first_not_of("0123456789") != std::string::npos) {
      error = "malformed inline-asm marker '" + m.name + "'";
      return false;
    }
    seq = std::stoull(digits);
    found.emplace_back(off, seq, InlineAsmBlock{off, it->second.second, it->second.first});
  }
  std::sort(found.begin(), found.end(), [](const auto &a, const auto &b) {
    return std::tie(std::get<0>(a), std::get<1>(a)) < std::tie(std::get<0>(b), std::get<1>(b));
  });
  for (auto &f : found)
    out.inline_asm.push_back(std::move(std::get<2>(f)));
  return true;
}

/// Parse the AMDHSA metadata note and keep this kernel's map plus the global fields.
void read_metadata(std::span<const uint8_t> image, const std::string &kernel, Artifact &out) {
  std::string error;
  auto doc = amdhsa::MetadataDocument::parse(image, &error);
  if (doc.failed())
    return;
  const amdhsa::MetadataValue &root = doc.value().root();
  if (root.kind != amdhsa::MetadataValue::Kind::Map)
    return;
  const amdhsa::MetadataValue *kernels = nullptr;
  for (const auto &[k, v] : root.map) {
    if (k == "amdhsa.kernels")
      kernels = &v;
    else
      flatten_metadata(v, std::string(k), out.global_metadata);
  }
  if (!kernels || kernels->kind != amdhsa::MetadataValue::Kind::Array)
    return;
  for (const auto &entry : kernels->array) {
    if (entry.text_field_is(".name", kernel) || entry.text_field_is(".symbol", kernel + ".kd")) {
      out.kernel_metadata.clear();
      flatten_metadata(entry, "", out.kernel_metadata);
      out.metadata_parsed = true;
      return;
    }
  }
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
  std::vector<TextRelocation> relocs;
  ElfInlineAsm asm_info;
  if (!scan_elf(image, text, funcs, notes, relocs, asm_info, error))
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
  // Without a descriptor, a function symbol is a callable function: the named one, or the
  // only one when no name is given.
  const ElfKernelSymbol *callable = nullptr;
  if (!chosen) {
    for (const auto &f : funcs) {
      if (f.size == 0 || !(kernel.empty() || f.name == kernel))
        continue;
      if (callable && kernel.empty()) {
        error = "code object has several functions and no kernel; pass --kernel";
        return false;
      }
      callable = &f;
    }
  }
  if (!chosen && !callable) {
    error = kernel.empty() ? "code object has no kernel descriptor"
                           : "kernel or function '" + kernel + "' not found";
    return false;
  }
  const std::string name = chosen ? chosen->kernel_name : callable->name;
  const ElfKernelSymbol *sym = callable;
  for (const auto &f : funcs)
    if (!sym && f.name == name)
      sym = &f;
  if (!sym || sym->size == 0) {
    error = "kernel symbol '" + name + "' is missing or has st_size 0";
    return false;
  }
  if (sym->value < text.vaddr || (chosen && sym->value - text.vaddr != chosen->entry_text_offset)) {
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
  out.kernel_begin = sym->value - text.vaddr;
  out.kernel_end = out.kernel_begin + sym->size;
  if (out.kernel_end > text.size) {
    error = "kernel symbol extends past .text";
    return false;
  }
  out.kernel_name = name;
  out.callable = chosen == nullptr;
  out.entry_vgprs = out.callable ? 32 : 3; // argument VGPRs v0-v31 / workitem IDs v0-v2
  out.entry_sgprs = 30;                    // inreg arguments and special inputs: s0-s29
  if (chosen) {
    // A kernel: the user SGPRs, then the enabled system SGPRs (workgroup IDs, workgroup
    // info, private segment wave offset).
    using namespace amdhsa;
    const uint32_t r2 = chosen->descriptor.compute_pgm_rsrc2;
    out.entry_sgprs =
        static_cast<int>(RJ_AMDHSA_BITS_GET(r2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT) +
                         RJ_AMDHSA_BITS_GET(r2, COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X) +
                         RJ_AMDHSA_BITS_GET(r2, COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y) +
                         RJ_AMDHSA_BITS_GET(r2, COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z) +
                         RJ_AMDHSA_BITS_GET(r2, COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO) +
                         RJ_AMDHSA_BITS_GET(r2, COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT));
  }
  if (chosen) {
    out.descriptor.resize(sizeof(chosen->descriptor));
    std::memcpy(out.descriptor.data(), &chosen->descriptor, sizeof(chosen->descriptor));
  }
  if (!read_inline_asm(asm_info, text.vaddr, out, error))
    return false;
  util::Sha256 notes_sha;
  for (const auto &[off, size] : notes)
    if (off <= image.size() && image.size() - off >= size)
      notes_sha.update(image.data() + off, size);
  const auto digest = notes_sha.finish();
  out.metadata_sha256 = sha256_hex(std::span<const uint8_t>(digest.data(), digest.size()));
  out.text_relocations = std::move(relocs);
  read_metadata(image, out.kernel_name, out);
  return true;
}

uint32_t initial_mode(const Artifact &artifact) {
  return artifact.callable ? artifact.function_mode : initial_mode(artifact.descriptor);
}

uint32_t initial_mode(const std::vector<uint8_t> &descriptor) {
  if (descriptor.size() < sizeof(amdhsa::kernel_descriptor_t))
    return kDefaultInitialMode;
  amdhsa::kernel_descriptor_t kd{};
  std::memcpy(&kd, descriptor.data(), sizeof(kd));
  using namespace amdhsa;
  const uint32_t rsrc1 = kd.compute_pgm_rsrc1;
  uint32_t mode = 0;
  mode |= RJ_AMDHSA_BITS_GET(rsrc1, COMPUTE_PGM_RSRC1_FLOAT_ROUND_MODE_32) << 0;
  mode |= RJ_AMDHSA_BITS_GET(rsrc1, COMPUTE_PGM_RSRC1_FLOAT_ROUND_MODE_16_64) << 2;
  mode |= RJ_AMDHSA_BITS_GET(rsrc1, COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_32) << 4;
  mode |= RJ_AMDHSA_BITS_GET(rsrc1, COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_16_64) << 6;
  mode |= RJ_AMDHSA_BITS_GET(rsrc1, COMPUTE_PGM_RSRC1_ENABLE_DX10_CLAMP) << 8;
  mode |= RJ_AMDHSA_BITS_GET(rsrc1, COMPUTE_PGM_RSRC1_ENABLE_IEEE_MODE) << 9;
  mode |= RJ_AMDHSA_BITS_GET(rsrc1, COMPUTE_PGM_RSRC1_DEBUG_MODE) << 11;
  if (RJ_AMDHSA_BITS_GET(rsrc1, COMPUTE_PGM_RSRC1_FP16_OVFL))
    mode |= 1u << 23;
  return mode;
}

std::string describe_relocation(const TextRelocation &r) {
  std::string s = "type " + std::to_string(r.type);
  if (!r.symbol.empty())
    s += " " + r.symbol;
  if (r.addend)
    s += (r.addend < 0 ? "-" : "+") + std::to_string(r.addend < 0 ? -r.addend : r.addend);
  return s;
}

namespace {
bool load_artifact_untimed(const std::string &path, const std::string &kernel, Artifact &out,
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
} // namespace

bool load_artifact(const std::string &path, const std::string &kernel, Artifact &out,
                   std::string &error) {
  const auto t0 = std::chrono::steady_clock::now();
  const bool ok = load_artifact_untimed(path, kernel, out, error);
  out.load_ns = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0)
          .count());
  return ok;
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
