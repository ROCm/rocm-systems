/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "core/inc/amd_aie_elf.h"

#include <elf.h>

#include <algorithm>
#include <cstring>
#include <limits>

namespace rocr {
namespace AMD {
namespace aie_elf {

namespace {

/// @brief ELF OS/ABI identifying an aie2p AIE ELF.
constexpr uint8_t kElfAmdAie2p = 69;

/// @brief ELF OS/ABI identifying an aie4 (AIE4/Medusa) AIE ELF, one past kElfAmdAie2p; confirmed
/// against an aiecc-produced npu3 full ELF.
constexpr uint8_t kElfAmdAie4 = 70;

/// @brief Returns the ELF OS/ABI a full ELF for @p arch carries, or 0 if @p arch has no full-ELF
/// format this reader implements. aie2 (Phoenix) has none at all: XRT defines no platform value
/// for it.
uint8_t ElfOsAbiForArch(std::string_view arch) {
  if (arch == "aie2p") return kElfAmdAie2p;
  if (arch == "aie4") return kElfAmdAie4;
  return 0;
}

/// @brief Relocation types, matching aiebu's symbol::patch_schema (src/cpp/common/symbol.h) --
/// the values the NPU firmware and XRT use.
enum class PatchScheme : uint32_t {
  /// @brief Fold a buffer address into a shim DMA buffer descriptor. Used for kernel arguments on
  /// aie2p: a 48-bit field split across two dwords (site[1] low 32 bits, site[2] bits 0-15 for
  /// bits 32-47).
  kShimDma48 = 5,
  /// @brief Same as kShimDma48 but for aie4's wider shim DMA buffer descriptor: a 57-bit field
  /// (site[1] low 32 bits, site[0] bits 0-24 for bits 32-56; aiebu's write57_aie4 in
  /// src/cpp/analyzer/transform_manager.cpp). See PatchShimDma57Aie4.
  kShimDma57Aie4 = 6,
  /// @brief Store a plain 64-bit address. Used for the PDI address.
  kAddress64 = 8,
};

/// @brief Highest argument index this reader accepts. A kernel argument list is short; the bound
/// keeps a malformed symbol name from being used to size a vector.
constexpr uint32_t kMaxArgIndex = 4095;

/// @brief Size of the transaction header a full-ELF control code opens with. Used both to rebase a
/// relocation offset from the combined ctrltext+ctrldata address space into ctrldata's own byte
/// offset, and to recognize a PDI site at offset 0 as malformed rather than legal.
constexpr uint32_t kControlCodeHeaderSize = 16;

/// @brief One relocation found while scanning `.rela.dyn`, before it is known to be a PDI site or
/// an argument site.
struct RelocSite {
  // Which page's section the relocation's symbol is bound to, and which buffer within it:
  // the "<col>.<page>.<id>" suffix shared by that page's ctrltext/ctrldata section names (see
  // Group::pages below).
  std::string page_key;
  bool in_ctrldata = false;
  uint32_t offset = 0;  // byte offset into the named buffer, already rebased if in_ctrldata
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

/// @brief True if `name` is `prefix` followed by '.' and more characters, or exactly `prefix`
/// with nothing after (in which case `*suffix` is set empty). Used to find a ctrltext/ctrldata
/// section's "<col>.<page>.<id>" suffix, which is how its ctrldata/ctrltext counterpart (if any)
/// is identified: same suffix, different prefix.
bool StripPrefix(const char* name, const char* prefix, std::string* suffix) {
  const size_t plen = std::strlen(prefix);
  if (std::strncmp(name, prefix, plen) != 0) return false;
  if (name[plen] == '\0') {
    suffix->clear();
    return true;
  }
  if (name[plen] != '.') return false;
  *suffix = name + plen + 1;
  return true;
}

bool ParseArgIndex(const char* name, uint32_t* index) {
  if (name == nullptr || *name == '\0') return false;
  uint32_t value = 0;
  for (const char* p = name; *p != '\0'; ++p) {
    if (*p < '0' || *p > '9') return false;
    const auto digit = static_cast<uint32_t>(*p - '0');
    // Refuse rather than wrap: a wrapped value can land back under the caller's kMaxArgIndex
    // bound and name a different argument than the ELF asked for.
    if (value > (std::numeric_limits<uint32_t>::max() - digit) / 10) return false;
    value = value * 10 + digit;
  }
  *index = value;
  return true;
}

/// @brief Splits a control-code section suffix, "<col>.<page>[.<id>]", into its column and page
/// numbers.
bool ParsePageKey(const std::string& key, uint32_t* col, uint32_t* page) {
  const size_t first = key.find('.');
  if (first == std::string::npos) return false;
  const size_t second = key.find('.', first + 1);
  const std::string col_str = key.substr(0, first);
  const std::string page_str = key.substr(first + 1, second == std::string::npos
                                                         ? std::string::npos
                                                         : second - first - 1);
  return ParseArgIndex(col_str.c_str(), col) && ParseArgIndex(page_str.c_str(), page);
}

}  // namespace

hsa_status_t Parse(const void* image_data, size_t image_size, std::string_view arch,
                   std::map<std::string, Kernel>* out, std::string* error) {
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
  const uint8_t osabi = ElfOsAbiForArch(arch);
  if (osabi == 0) {
    return fail("no full-ELF format for arch " + std::string(arch));
  }
  // The hsaco section the ELF came from names the arch; an ELF built for a different one would load
  // here and only fail on the device.
  if (ehdr->e_ident[EI_OSABI] != osabi) {
    return fail("not an " + std::string(arch) + " AIE ELF");
  }
  if (ehdr->e_shentsize != sizeof(Elf32_Shdr) || ehdr->e_shnum == 0) {
    return fail("malformed section headers");
  }
  const uint8_t abi_version = ehdr->e_ident[EI_ABIVERSION];
  const bool is_aie4 = arch == "aie4";

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
  const Elf32_Shdr* dynsym = nullptr;
  const Elf32_Shdr* rela = nullptr;
  for (uint32_t i = 0; i < ehdr->e_shnum; ++i) {
    const char* name = section_name(i);
    if (name == nullptr) continue;
    if (std::strcmp(name, ".symtab") == 0)
      symtab = &shdrs[i];
    else if (std::strcmp(name, ".dynsym") == 0)
      dynsym = &shdrs[i];
    else if (std::strcmp(name, ".rela.dyn") == 0)
      rela = &shdrs[i];
  }
  // A symbol table names its string table through sh_link, as ELF defines it, rather than by the
  // string table's section name. mlir-aie's hsaco packer reads it the same way, so the two cannot
  // disagree about the kernel names in an ELF that names its string tables differently.
  auto linked_strtab = [&](const Elf32_Shdr* table) -> const Elf32_Shdr* {
    if (table == nullptr || table->sh_link == 0 || table->sh_link >= ehdr->e_shnum) return nullptr;
    return &shdrs[table->sh_link];
  };
  const Elf32_Shdr* strtab = linked_strtab(symtab);
  const Elf32_Shdr* dynstr = linked_strtab(dynsym);
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
  //
  // A kernel's control code may be split into several pages -- aie2p has exactly one, aie4 bundles
  // several per kernel -- each a (ctrltext, ctrldata) section pair sharing a name suffix (see
  // StripPrefix). ctrldata is optional (aie2p never has one).
  struct PageSections {
    uint32_t ctrltext_section = 0;
    uint32_t ctrldata_section = 0;
  };
  struct Group {
    std::string name;
    // Keyed by the page's name suffix; an ordered map just so iteration order is stable from one
    // parse to the next; nothing here depends on it being numeric or otherwise meaningful.
    std::map<std::string, PageSections> pages;
  };
  // Role of a ctrltext/ctrldata section that is a group member, for resolving a relocation's
  // st_shndx straight to the (group, page, buffer) it patches.
  struct SectionRole {
    uint32_t group = 0;
    std::string page_key;
    bool in_ctrldata = false;
  };
  std::map<uint32_t, Group> groups;              // group section index -> group
  std::map<uint32_t, SectionRole> section_role;  // member section index -> role

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
    if (words == nullptr || word_count == 0) return fail("malformed group section");
    // Producers emit only COMDAT groups, and a plain group has no known meaning in a full ELF, so
    // refuse it, as mlir-aie's hsaco packer does.
    if ((words[0] & GRP_COMDAT) == 0) return fail("group is not a COMDAT group");
    for (uint32_t w = 1; w < word_count; ++w) {
      const uint32_t member = words[w];
      if (member >= ehdr->e_shnum) return fail("group member out of range");
      const char* member_name = section_name(member);
      if (member_name == nullptr) continue;
      std::string suffix;
      if (StripPrefix(member_name, ".ctrltext", &suffix)) {
        if (shdrs[member].sh_type != SHT_PROGBITS) {
          return fail("control code section holds no data");
        }
        group.pages[suffix].ctrltext_section = member;
        section_role[member] = SectionRole{i, suffix, false};
      } else if (StripPrefix(member_name, ".ctrldata", &suffix)) {
        if (shdrs[member].sh_type != SHT_PROGBITS) {
          return fail("control-data section holds no data");
        }
        group.pages[suffix].ctrldata_section = member;
        section_role[member] = SectionRole{i, suffix, true};
      }
    }
    groups.emplace(i, std::move(group));
  }
  if (groups.empty()) return fail("no COMDAT groups: not a group ELF");

  // Collect relocations. Each names a symbol whose st_shndx is the section being patched and
  // whose name says what address to write: ".pdi.N" for a PDI, a decimal string for an argument.
  std::map<uint32_t, std::map<uint32_t, std::vector<RelocSite>>> group_arg_sites;
  // One PDI relocation, before it is known which Kernel::pages index its page_key will become.
  struct PdiSite {
    uint32_t pdi_section = 0;
    std::string page_key;
    uint32_t offset = 0;  // already rebased; always into ctrltext (PDI sites in ctrldata are
                          // refused: see below)
  };
  std::map<uint32_t, std::vector<PdiSite>> group_pdi_sites;  // group section index -> PDI sites

  // PDI sections are not group members, so a relocation names its PDI by section name only, and a
  // multi-kernel ELF can repeat a name (MLIR-AIR repeats .pdi.1.. for each kernel). A name is
  // resolved to its first section, and every other section with that name must hold the same
  // bytes: then the choice cannot matter, and otherwise there is no telling which was meant.
  auto find_pdi_section = [&](const char* name, uint32_t* found) -> bool {
    *found = 0;
    for (uint32_t s = 1; s < ehdr->e_shnum; ++s) {
      const char* n = section_name(s);
      if (n == nullptr || std::strcmp(n, name) != 0) continue;
      if (*found == 0) {
        *found = s;
        continue;
      }
      const Elf32_Shdr& a = shdrs[*found];
      const Elf32_Shdr& b = shdrs[s];
      const uint8_t* a_data = image.At(a.sh_offset, a.sh_size);
      const uint8_t* b_data = image.At(b.sh_offset, b.sh_size);
      if (a_data == nullptr || b_data == nullptr || a.sh_size != b.sh_size ||
          std::memcmp(a_data, b_data, a.sh_size) != 0) {
        return false;
      }
    }
    return true;
  };

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

      auto role_it = section_role.find(sym.st_shndx);
      if (role_it == section_role.end()) continue;  // not a control-code section
      const SectionRole& role = role_it->second;
      const Group& group = groups.at(role.group);
      const PageSections& page_secs = group.pages.at(role.page_key);

      uint32_t offset = relocs[r].r_offset;
      bool in_ctrldata = role.in_ctrldata;
      if (is_aie4) {
        // aiebu's convention (elf_reader_gen2plus::init_patchers): a control-code relocation's
        // r_offset is relative to its page's payload, which starts 16 bytes into the page, past
        // the transaction header the control code opens with -- whichever of the page's sections
        // the symbol names. The page is its ctrltext followed directly by its ctrldata, so the
        // site is in ctrltext or in ctrldata depending on where that position falls. 64-bit so a
        // huge r_offset (untrusted ELF input) is caught rather than wrapping back into range.
        if (page_secs.ctrltext_section == 0) return fail("relocation into a page with no ctrltext");
        const uint32_t ctrltext_size = shdrs[page_secs.ctrltext_section].sh_size;
        uint64_t in_page = static_cast<uint64_t>(offset) + kControlCodeHeaderSize;
        in_ctrldata = in_page >= ctrltext_size;
        if (in_ctrldata) {
          if (page_secs.ctrldata_section == 0) {
            return fail("relocation offset past the end of ctrltext, but the page has no ctrldata");
          }
          in_page -= ctrltext_size;
        }
        if (in_page > std::numeric_limits<uint32_t>::max()) {
          return fail("control-code relocation offset overflows");
        }
        offset = static_cast<uint32_t>(in_page);
      } else if (!in_ctrldata && page_secs.ctrltext_section != 0) {
        const uint32_t ctrltext_size = shdrs[page_secs.ctrltext_section].sh_size;
        if (offset >= ctrltext_size) {
          if (page_secs.ctrldata_section == 0) {
            return fail("relocation offset past the end of ctrltext, but the page has no ctrldata");
          }
          // 64-bit so a huge r_offset (untrusted ELF input) is caught by the overflow check below
          // rather than wrapping back into range.
          const uint64_t rebased =
              static_cast<uint64_t>(offset) - ctrltext_size + kControlCodeHeaderSize;
          if (rebased > std::numeric_limits<uint32_t>::max()) {
            return fail("control-data relocation offset overflows");
          }
          offset = static_cast<uint32_t>(rebased);
          in_ctrldata = true;
        }
      }

      RelocSite site;
      site.page_key = role.page_key;
      site.in_ctrldata = in_ctrldata;
      site.offset = offset;
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
        if (site.in_ctrldata) return fail("PDI patch site in control-data is not supported");
        uint32_t pdi_section = 0;
        if (!find_pdi_section(sym_name, &pdi_section)) {
          return fail(std::string("ambiguous PDI: sections named ") + sym_name + " differ");
        }
        if (pdi_section == 0) return fail("PDI section not found");
        group_pdi_sites[role.group].push_back(PdiSite{pdi_section, role.page_key, site.offset});
        continue;
      }

      uint32_t arg_index = 0;
      if (!ParseArgIndex(sym_name, &arg_index)) {
        // A scratch pad, control packet or similar. Skipping it would leave a dangling address in
        // the control code, so refuse instead.
        return fail(std::string("unsupported relocation symbol: ") + sym_name);
      }
      // The dispatch path applies the arch's own scheme to every site, so a site written for the
      // other arch's buffer descriptor would be patched wrongly.
      if (site.scheme != (is_aie4 ? PatchScheme::kShimDma57Aie4 : PatchScheme::kShimDma48)) {
        return fail("unsupported patch scheme for a kernel argument on " + std::string(arch));
      }
      if (arg_index > kMaxArgIndex) {
        return fail("kernel argument index out of range");
      }
      group_arg_sites[role.group][arg_index].push_back(site);
    }
  }

  std::map<std::string, Kernel> kernels;
  for (auto& [grp_index, group] : groups) {
    Kernel k;
    k.name = group.name;

    // page_key -> index into k.pages, for resolving the sites collected above. A page with
    // ctrldata but no ctrltext has nothing to execute, so it is dropped rather than failing the
    // whole group: other pages, or other groups, may still be valid.
    std::map<std::string, uint32_t> page_index;
    // On aie4 a page's place in the control code is its number, so pages are taken in numeric
    // order of the "<col>.<page>" their section names carry. Only column 0 is supported: each
    // column's control code goes to its own microcontroller, which needs an indirect packet that
    // UMQ dispatch does not build.
    std::map<uint32_t, const std::string*> numbered;  // page number -> page_key
    for (const auto& [page_key, secs] : group.pages) {
      if (!is_aie4) break;
      uint32_t col = 0, number = 0;
      if (!ParsePageKey(page_key, &col, &number)) {
        return fail("control-code section without a <col>.<page> suffix");
      }
      if (col != 0) return fail("multi-column control code is not supported");
      if (!numbered.emplace(number, &page_key).second) return fail("duplicate control-code page");
    }
    std::vector<std::pair<uint32_t, const std::string*>> ordered;
    if (is_aie4) {
      ordered.assign(numbered.begin(), numbered.end());
    } else {
      for (const auto& [page_key, secs] : group.pages) ordered.emplace_back(0, &page_key);
    }
    for (const auto& [number, key] : ordered) {
      const std::string& page_key = *key;
      const PageSections& secs = group.pages.at(page_key);
      if (secs.ctrltext_section == 0) continue;
      Page page;
      page.number = number;
      if (!section_bytes(secs.ctrltext_section, &page.ctrltext)) {
        return fail("section extends past end of file");
      }
      if (page.ctrltext.empty()) return fail("empty control code");
      if (secs.ctrldata_section != 0 && !section_bytes(secs.ctrldata_section, &page.ctrldata)) {
        return fail("section extends past end of file");
      }
      page_index.emplace(page_key, static_cast<uint32_t>(k.pages.size()));
      k.pages.push_back(std::move(page));
    }
    if (k.pages.empty()) continue;  // nothing to dispatch

    auto pdi_sites_it = group_pdi_sites.find(grp_index);
    if (pdi_sites_it != group_pdi_sites.end()) {
      for (const PdiSite& s : pdi_sites_it->second) {
        auto page_it = page_index.find(s.page_key);
        // A PDI site is only ever recorded against a page found via its ctrltext section (sites
        // bound to ctrldata are refused above), so that page is never one dropped for lacking
        // ctrltext; fail rather than silently drop if that invariant is ever violated.
        if (page_it == page_index.end()) return fail("PDI patch site's page has no control code");
        std::vector<uint8_t>& ctrltext = k.pages[page_it->second].ctrltext;
        // A 64-bit address is written here, so it must lie wholly inside the control code.
        if (s.offset + sizeof(uint64_t) > ctrltext.size() || s.offset % sizeof(uint32_t) != 0) {
          return fail("PDI patch site does not fit the control code");
        }
        // A full-ELF control code opens with a kControlCodeHeaderSize-byte transaction header, so
        // a PDI patch site at offset 0 means a malformed ELF, not a legal kernel.
        if (s.offset == 0) {
          return fail("PDI patch site at offset 0 is indistinguishable from no patch");
        }
        std::vector<uint8_t> bytes;
        if (!section_bytes(s.pdi_section, &bytes)) return fail("section extends past end of file");
        // Each distinct PDI is placed in device memory once, however many sites load it.
        auto pdi = std::find_if(k.pdis.begin(), k.pdis.end(),
                                [&](const Pdi& p) { return p.bytes == bytes; });
        if (pdi == k.pdis.end()) pdi = k.pdis.insert(k.pdis.end(), Pdi{std::move(bytes), {}});
        pdi->patch_sites.push_back(PdiPatchSite{page_it->second, s.offset});
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
          auto page_it = page_index.find(site.page_key);
          // Unlike a PDI site, an argument site's symbol may bind directly to a ctrldata section,
          // which can reference a page whose ctrldata has no matching ctrltext (dropped above as
          // having nothing to execute); fail rather than silently drop the patch, since a dropped
          // site would leave the ELF's placeholder address in place at dispatch instead of the
          // caller's real one.
          if (page_it == page_index.end()) {
            return fail("argument patch site's page has no control code");
          }
          const std::vector<uint8_t>& target = site.in_ctrldata ? k.pages[page_it->second].ctrldata
                                                                : k.pages[page_it->second].ctrltext;
          // kShimDma48 updates the two dwords after the site (site[1], site[2]), so three dwords
          // from the offset must lie inside the named buffer; kShimDma57Aie4 updates the site
          // itself and the dword after (site[0], site[1]), so two. The offset must also be dword
          // aligned; otherwise the driver patches past the end of its per-dispatch buffer. Checked
          // here because this is the first point at which the buffer's size is known.
          const size_t site_dwords = site.scheme == PatchScheme::kShimDma48 ? 3 : 2;
          if ((site.offset % sizeof(uint32_t)) != 0 ||
              site.offset + site_dwords * sizeof(uint32_t) > target.size()) {
            return fail("argument patch site does not fit the control code");
          }
          dst.push_back(PatchSite{page_it->second, site.in_ctrldata, site.offset, site.addend});
        }
      }
    }

    // Kernels are looked up by name, so two groups with one name would make the lookup ambiguous.
    if (!kernels.emplace(k.name, std::move(k)).second) return fail("duplicate kernel name");
  }
  if (kernels.empty()) return fail("no dispatchable kernels");

  out->swap(kernels);
  return HSA_STATUS_SUCCESS;
}

}  // namespace aie_elf
}  // namespace AMD
}  // namespace rocr
