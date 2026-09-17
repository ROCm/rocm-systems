/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "core/inc/amd_aie_elf.h"

#include <elf.h>

#include <cstring>
#include <limits>

namespace rocr {
namespace AMD {
namespace aie_elf {

namespace {

/// @brief ELF OS/ABI identifying an aie2p AIE ELF.
constexpr uint8_t kElfAmdAie2p = 69;

/// @brief Relocation types, matching the patch schemes the NPU firmware and XRT use.
enum class PatchScheme : uint32_t {
  /// @brief Fold a buffer address into a shim DMA buffer descriptor. Used for kernel arguments.
  kShimDma48 = 5,
  /// @brief Store a plain 64-bit address. Used for the PDI address.
  kAddress64 = 8,
};

/// @brief Highest argument index this reader accepts. A kernel argument list is short; the bound
/// keeps a malformed symbol name from being used to size a vector.
constexpr uint32_t kMaxArgIndex = 4095;

/// @brief One relocation found while scanning `.rela.dyn`, before it is known to be a PDI site or
/// an argument site.
struct RelocSite {
  uint32_t offset = 0;  // byte offset into the control code
  uint32_t addend = 0;  // added to the address before it is written
  PatchScheme scheme = PatchScheme::kShimDma48;
};

// Bounds-checked view over the ELF image. Returns nullptr rather than walking off the end, so a
// truncated or hostile file is a clean error.
class Image {
 public:
  Image(const uint8_t* data, size_t size) : data_(data), size_(size) {}

  const uint8_t* At(uint64_t offset, uint64_t count) const {
    if (offset > size_ || count > size_ - offset) return nullptr;
    return data_ + offset;
  }

  template <typename T> const T* As(uint64_t offset, uint64_t count = 1) const {
    if (count != 0 && sizeof(T) > UINT64_MAX / count) return nullptr;
    // Little-endian ELF32 read on little-endian x86-64, so the on-disk layout is the host layout
    // and an unaligned section offset still reads correctly.
    return reinterpret_cast<const T*>(At(offset, sizeof(T) * count));
  }

 private:
  const uint8_t* data_;
  size_t size_;
};

const char* StringAt(const Image& image, const Elf32_Shdr& strtab, uint32_t offset) {
  if (offset >= strtab.sh_size) return nullptr;
  const uint8_t* base = image.At(strtab.sh_offset, strtab.sh_size);
  if (base == nullptr) return nullptr;
  const auto* str = reinterpret_cast<const char*>(base + offset);
  // The table has to contain the terminator, otherwise the string runs off the section.
  if (std::memchr(str, '\0', strtab.sh_size - offset) == nullptr) return nullptr;
  return str;
}

// "_Z4mainPcPcPc" -> "main". Only the simple `_Z<len><name>` form aiecc emits is handled; anything
// else is returned unchanged, which at worst makes the kernel name uglier, never wrong.
std::string KernelNameFromSymbol(const std::string& symbol) {
  if (symbol.rfind("_Z", 0) != 0) return symbol;
  size_t i = 2;
  size_t len = 0;
  while (i < symbol.size() && symbol[i] >= '0' && symbol[i] <= '9') {
    len = len * 10 + static_cast<size_t>(symbol[i] - '0');
    ++i;
  }
  if (len == 0 || i + len > symbol.size()) return symbol;
  return symbol.substr(i, len);
}

bool ParseArgIndex(const char* name, uint32_t* index) {
  if (name == nullptr || *name == '\0') return false;
  uint32_t value = 0;
  for (const char* p = name; *p != '\0'; ++p) {
    if (*p < '0' || *p > '9') return false;
    const auto digit = static_cast<uint32_t>(*p - '0');
    // Refuse rather than wrap. The caller bounds the result against kMaxArgIndex, and a value
    // that wrapped can land back under that bound and name a different argument than the ELF
    // asked for -- so the overflow has to be caught here, not there.
    if (value > (std::numeric_limits<uint32_t>::max() - digit) / 10) return false;
    value = value * 10 + digit;
  }
  *index = value;
  return true;
}

}  // namespace

hsa_status_t Parse(const void* image_data, size_t image_size, std::map<std::string, Kernel>* out,
                    std::string* error) {
  if (out == nullptr || error == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  auto fail = [&](const std::string& msg) -> hsa_status_t {
    *error = msg;
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  };

  const Image image(static_cast<const uint8_t*>(image_data), image_size);

  const auto* ehdr = image.As<Elf32_Ehdr>(0);
  if (ehdr == nullptr || std::memcmp(ehdr->e_ident, ELFMAG, SELFMAG) != 0 ||
      ehdr->e_ident[EI_CLASS] != ELFCLASS32 || ehdr->e_ident[EI_DATA] != ELFDATA2LSB) {
    return fail("not a little-endian ELF32");
  }
  if (ehdr->e_ident[EI_OSABI] != kElfAmdAie2p) {
    return fail("not an aie2p AIE ELF");
  }
  if (ehdr->e_shentsize != sizeof(Elf32_Shdr) || ehdr->e_shnum == 0) {
    return fail("malformed section headers");
  }
  const uint8_t abi_version = ehdr->e_ident[EI_ABIVERSION];

  const auto* shdrs = image.As<Elf32_Shdr>(ehdr->e_shoff, ehdr->e_shnum);
  if (shdrs == nullptr || ehdr->e_shstrndx >= ehdr->e_shnum) {
    return fail("malformed section headers");
  }
  const Elf32_Shdr& shstrtab = shdrs[ehdr->e_shstrndx];

  auto section_name = [&](uint32_t index) -> const char* {
    if (index >= ehdr->e_shnum) return nullptr;
    return StringAt(image, shstrtab, shdrs[index].sh_name);
  };

  const Elf32_Shdr* symtab = nullptr;
  const Elf32_Shdr* strtab = nullptr;
  const Elf32_Shdr* dynsym = nullptr;
  const Elf32_Shdr* dynstr = nullptr;
  const Elf32_Shdr* rela = nullptr;
  for (uint32_t i = 0; i < ehdr->e_shnum; ++i) {
    const char* name = section_name(i);
    if (name == nullptr) continue;
    if (std::strcmp(name, ".symtab") == 0)
      symtab = &shdrs[i];
    else if (std::strcmp(name, ".strtab") == 0)
      strtab = &shdrs[i];
    else if (std::strcmp(name, ".dynsym") == 0)
      dynsym = &shdrs[i];
    else if (std::strcmp(name, ".dynstr") == 0)
      dynstr = &shdrs[i];
    else if (std::strcmp(name, ".rela.dyn") == 0)
      rela = &shdrs[i];
  }
  if (symtab == nullptr || strtab == nullptr || symtab->sh_entsize != sizeof(Elf32_Sym)) {
    return fail("missing or malformed .symtab");
  }

  const uint32_t symtab_count = symtab->sh_size / sizeof(Elf32_Sym);
  const auto* symbols = image.As<Elf32_Sym>(symtab->sh_offset, symtab_count);
  if (symbols == nullptr) return fail("malformed .symtab");

  // Section index -> its bytes, for the PDI and control-code sections.
  auto section_bytes = [&](uint32_t index, std::vector<uint8_t>* bytes) -> bool {
    const uint8_t* data = image.At(shdrs[index].sh_offset, shdrs[index].sh_size);
    if (data == nullptr) return false;
    bytes->assign(data, data + shdrs[index].sh_size);
    return true;
  };

  // A group's sh_info is the .symtab index of its instance symbol, whose st_shndx in turn is the
  // .symtab index of the kernel's function symbol. That is an overload of st_shndx specific to
  // this ELF flavour, not a section index.
  struct Group {
    std::string name;
    uint32_t ctrltext_section = 0;
    uint32_t pdi_section = 0;
  };
  std::map<uint32_t, Group> groups;           // group section index -> group
  std::map<uint32_t, uint32_t> sec2grp;  // member section index -> group section index

  for (uint32_t i = 0; i < ehdr->e_shnum; ++i) {
    if (shdrs[i].sh_type != SHT_GROUP) continue;
    if (shdrs[i].sh_info >= symtab_count) return fail("bad group signature symbol");

    const Elf32_Sym& instance_sym = symbols[shdrs[i].sh_info];
    const char* instance_name = StringAt(image, *strtab, instance_sym.st_name);
    if (instance_name == nullptr || instance_sym.st_shndx >= symtab_count) {
      return fail("bad group signature symbol");
    }
    const char* kernel_sym = StringAt(image, *strtab, symbols[instance_sym.st_shndx].st_name);
    if (kernel_sym == nullptr) return fail("bad kernel symbol");

    Group group;
    group.name = KernelNameFromSymbol(kernel_sym) + ":" + instance_name;

    // Group data is a flags word followed by the member section indices.
    const uint32_t word_count = shdrs[i].sh_size / sizeof(Elf32_Word);
    const auto* words = image.As<Elf32_Word>(shdrs[i].sh_offset, word_count);
    if (words == nullptr) return fail("malformed group section");
    for (uint32_t w = 1; w < word_count; ++w) {
      const uint32_t member = words[w];
      if (member >= ehdr->e_shnum) return fail("group member out of range");
      sec2grp[member] = i;
      const char* member_name = section_name(member);
      if (member_name != nullptr && std::strncmp(member_name, ".ctrltext", 9) == 0) {
        if (shdrs[member].sh_type != SHT_PROGBITS) {
          return fail("control code section holds no data");
        }
        group.ctrltext_section = member;
      }
    }
    groups.emplace(i, std::move(group));
  }
  if (groups.empty()) return fail("no COMDAT groups: not a group ELF");

  // Collect relocations. Each names a symbol whose st_shndx is the section being patched and
  // whose name says what address to write: ".pdi.N" for a PDI, a decimal string for an argument.
  std::map<uint32_t, std::map<uint32_t, std::vector<RelocSite>>> group_arg_sites;
  std::map<uint32_t, RelocSite> group_pdi_site;

  if (rela != nullptr && dynsym != nullptr && dynstr != nullptr) {
    if (rela->sh_entsize != sizeof(Elf32_Rela) || dynsym->sh_entsize != sizeof(Elf32_Sym)) {
      return fail("malformed .rela.dyn");
    }
    const uint32_t rela_count = rela->sh_size / sizeof(Elf32_Rela);
    const auto* relocs = image.As<Elf32_Rela>(rela->sh_offset, rela_count);
    const uint32_t dynsym_count = dynsym->sh_size / sizeof(Elf32_Sym);
    const auto* dynsyms = image.As<Elf32_Sym>(dynsym->sh_offset, dynsym_count);
    if (relocs == nullptr || dynsyms == nullptr) return fail("malformed .rela.dyn");

    for (uint32_t r = 0; r < rela_count; ++r) {
      const uint32_t sym_index = ELF32_R_SYM(relocs[r].r_info);
      if (sym_index >= dynsym_count) return fail("relocation symbol out of range");
      const Elf32_Sym& sym = dynsyms[sym_index];
      const char* sym_name = StringAt(image, *dynstr, sym.st_name);
      if (sym_name == nullptr) return fail("bad relocation symbol name");

      auto grp_it = sec2grp.find(sym.st_shndx);
      if (grp_it == sec2grp.end()) continue;
      Group& group = groups.at(grp_it->second);
      if (sym.st_shndx != group.ctrltext_section) continue;  // only control code is patched

      RelocSite site;
      site.offset = relocs[r].r_offset;
      if (abi_version == 1) {
        // In ABI version 1 the scheme lives in the low bits of the addend rather than in r_info.
        site.addend = static_cast<uint32_t>(relocs[r].r_addend) >> 4;
        site.scheme = static_cast<PatchScheme>(relocs[r].r_addend & 0xF);
      } else {
        site.addend = static_cast<uint32_t>(relocs[r].r_addend);
        site.scheme = static_cast<PatchScheme>(ELF32_R_TYPE(relocs[r].r_info));
      }

      if (std::strncmp(sym_name, ".pdi", 4) == 0) {
        if (site.scheme != PatchScheme::kAddress64) {
          return fail("unexpected patch scheme for PDI symbol");
        }
        // Find the PDI section this symbol names.
        uint32_t pdi_section = 0;
        for (uint32_t s = 0; s < ehdr->e_shnum; ++s) {
          const char* n = section_name(s);
          if (n != nullptr && std::strcmp(n, sym_name) == 0) {
            pdi_section = s;
            break;
          }
        }
        if (pdi_section == 0) return fail("PDI section not found");
        if (group.pdi_section != 0 && group.pdi_section != pdi_section) {
          return fail("more than one PDI per kernel is not supported");
        }
        // A Kernel carries a single PDI patch offset, so a second site would be dropped and its
        // load_pdi left pointing at whatever placeholder the ELF holds.
        if (group_pdi_site.count(grp_it->second) != 0) {
          return fail("more than one PDI patch site per kernel is not supported");
        }
        group.pdi_section = pdi_section;
        group_pdi_site[grp_it->second] = site;
        continue;
      }

      uint32_t arg_index = 0;
      if (!ParseArgIndex(sym_name, &arg_index)) {
        // A scratch pad, control packet or similar. Skipping it would leave a dangling address in
        // the control code, so refuse instead.
        return fail(std::string("unsupported relocation symbol: ") + sym_name);
      }
      if (site.scheme != PatchScheme::kShimDma48) {
        return fail("unsupported patch scheme for a kernel argument");
      }
      if (arg_index > kMaxArgIndex) {
        return fail("kernel argument index out of range");
      }
      group_arg_sites[grp_it->second][arg_index].push_back(site);
    }
  }

  std::map<std::string, Kernel> kernels;
  for (auto& [grp_index, group] : groups) {
    if (group.ctrltext_section == 0) continue;  // nothing to dispatch

    Kernel k;
    k.name = group.name;
    if (!section_bytes(group.ctrltext_section, &k.ctrl_code)) {
      return fail("section extends past end of file");
    }
    if (k.ctrl_code.empty()) return fail("empty control code");

    if (group.pdi_section != 0) {
      if (!section_bytes(group.pdi_section, &k.pdi)) {
        return fail("section extends past end of file");
      }
      k.pdi_patch_offset = group_pdi_site.at(grp_index).offset;
      k.has_pdi_patch = true;
      // The runtime writes a 64-bit address here, so it has to lie wholly inside the control code.
      if (k.pdi_patch_offset + sizeof(uint64_t) > k.ctrl_code.size() ||
          k.pdi_patch_offset % sizeof(uint32_t) != 0) {
        return fail("PDI patch site does not fit the control code");
      }
      // A real full-ELF control code opens with a 16-byte transaction header, so a legitimate PDI
      // patch site is never at offset 0. This check no longer disambiguates a packet field (there
      // is none left to disambiguate) -- it is retained purely as a well-formedness check on the
      // ELF itself, catching a relocation the parser mis-attributed rather than a legal kernel.
      if (k.pdi_patch_offset == 0) {
        return fail("PDI patch site at offset 0 is indistinguishable from no patch");
      }
    }

    auto args_it = group_arg_sites.find(grp_index);
    if (args_it != group_arg_sites.end() && !args_it->second.empty()) {
      const uint32_t max_arg = args_it->second.rbegin()->first;
      k.arg_sites.resize(max_arg + 1);
      for (auto& [arg_index, sites] : args_it->second) {
        auto& dst = k.arg_sites[arg_index];
        dst.reserve(sites.size());
        for (const RelocSite& site : sites) {
          // PatchShimDma48 reads and writes the two dwords following the site, so three dwords
          // starting at the offset have to lie wholly inside the control code, and the offset has
          // to be dword aligned. Checked here rather than where the relocation was read, because
          // this is the first point at which the control code's size is known. Without it a
          // malformed ELF makes the driver patch past the end of the buffer it sized from
          // ctrl_code_size -- the same bound the PDI patch site is checked against above.
          if ((site.offset % sizeof(uint32_t)) != 0 ||
              site.offset + 3 * sizeof(uint32_t) > k.ctrl_code.size()) {
            return fail("argument patch site does not fit the control code");
          }
          dst.push_back(PatchSite{site.offset, site.addend});
        }
      }
    }

    kernels.emplace(k.name, std::move(k));
  }
  if (kernels.empty()) return fail("no dispatchable kernels");

  out->swap(kernels);
  return HSA_STATUS_SUCCESS;
}

void PatchShimDma48(uint32_t* site, uint64_t addr) {
  constexpr uint64_t kDdrAieAddrOffset = 0x80000000;
  uint64_t base =
      ((static_cast<uint64_t>(site[2]) & 0xFFFF) << 32) | static_cast<uint64_t>(site[1]);
  base += addr + kDdrAieAddrOffset;
  site[1] = static_cast<uint32_t>(base & 0xFFFFFFFC);
  site[2] = (site[2] & 0xFFFF0000) | static_cast<uint32_t>(base >> 32);
}

}  // namespace aie_elf
}  // namespace AMD
}  // namespace rocr
