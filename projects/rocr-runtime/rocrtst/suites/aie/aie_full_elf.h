/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Minimal reader for the full-ELF kernel binaries aiecc emits (`aiecc --get-full-elf`).
//
// A full ELF carries the PDI and the control code in one file, along with the relocations that
// say where addresses have to be written into the control code. The runtime deliberately knows
// nothing about any of that: an application extracts the pieces, allocates them from the agent's
// device memory pool, patches its own argument addresses in, and names the buffers in an ordinary
// dispatch packet. The one address it cannot know is the PDI's device address, so it passes the
// offset of that patch site in hsa_amd_aie_kernel_dispatch_packet_t::pdi_patch_offset and the
// runtime fills it in; that non-zero offset is also what selects the full-ELF dispatch shape.
//
// Only what the vector_scalar_add design needs is supported: one PDI, one control-code section
// per kernel, and buffer arguments. Control packets, preemption save/restore sections and
// scalar arguments are not handled, and an ELF using them is rejected rather than silently
// mispatched.

#ifndef ROCRTST_SUITES_AIE_AIE_FULL_ELF_H_
#define ROCRTST_SUITES_AIE_AIE_FULL_ELF_H_

#include <elf.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace aie_full_elf {

// ELF OS/ABI identifying an aie2p AIE ELF.
inline constexpr std::uint8_t kElfAmdAie2p = 69;

// Relocation types, matching the patch schemes the NPU firmware and XRT use.
enum class PatchScheme : std::uint32_t {
  // Fold a buffer address into a shim DMA buffer descriptor. Used for kernel arguments.
  kShimDma48 = 5,
  // Store a plain 64-bit address. Used for the PDI address.
  kAddress64 = 8,
};

/// Highest argument index this reader accepts. A kernel argument list is short; the bound just
/// keeps a malformed symbol name from being used to size a vector.
inline constexpr std::uint32_t kMaxArgIndex = 4095;

// One place in the control code that takes an address.
struct PatchSite {
  std::uint32_t offset = 0;  // byte offset into the control code
  std::uint32_t addend = 0;  // added to the address before it is written
  PatchScheme scheme = PatchScheme::kShimDma48;
};

// A parsed full-ELF kernel: the bytes to load and where addresses go.
struct Kernel {
  std::string name;  // "<kernel>:<instance>", e.g. "main:sequence"
  std::vector<std::uint8_t> pdi;
  std::vector<std::uint8_t> ctrl_code;
  // Where the PDI's device address goes in the control code. Passed to the runtime as
  // hsa_amd_aie_kernel_dispatch_packet_t::pdi_patch_offset.
  std::uint64_t pdi_patch_offset = 0;
  bool has_pdi_patch = false;
  // Patch sites per argument index. Entries may be empty for unused arguments.
  std::vector<std::vector<PatchSite>> arg_sites;

  // Number of arguments the control code actually references.
  std::uint32_t num_args() const { return static_cast<std::uint32_t>(arg_sites.size()); }
};

namespace detail {

// Bounds-checked view over the ELF image. Returns nullptr rather than walking off the end, so a
// truncated or hostile file is a clean error.
class Image {
 public:
  Image(const std::uint8_t* data, std::size_t size) : data_(data), size_(size) {}

  const std::uint8_t* At(std::uint64_t offset, std::uint64_t count) const {
    if (offset > size_ || count > size_ - offset) return nullptr;
    return data_ + offset;
  }

  template <typename T> const T* As(std::uint64_t offset, std::uint64_t count = 1) const {
    if (count != 0 && sizeof(T) > UINT64_MAX / count) return nullptr;
    // Little-endian ELF32 read on little-endian x86-64, so the on-disk layout is the host layout
    // and an unaligned section offset still reads correctly.
    return reinterpret_cast<const T*>(At(offset, sizeof(T) * count));
  }

 private:
  const std::uint8_t* data_;
  std::size_t size_;
};

inline const char* StringAt(const Image& image, const Elf32_Shdr& strtab, std::uint32_t offset) {
  if (offset >= strtab.sh_size) return nullptr;
  const std::uint8_t* base = image.At(strtab.sh_offset, strtab.sh_size);
  if (base == nullptr) return nullptr;
  const auto* str = reinterpret_cast<const char*>(base + offset);
  // The table has to contain the terminator, otherwise the string runs off the section.
  if (std::memchr(str, '\0', strtab.sh_size - offset) == nullptr) return nullptr;
  return str;
}

// "_Z4mainPcPcPc" -> "main". Only the simple `_Z<len><name>` form aiecc emits is handled; anything
// else is returned unchanged, which at worst makes the kernel name uglier, never wrong.
inline std::string KernelNameFromSymbol(const std::string& symbol) {
  if (symbol.rfind("_Z", 0) != 0) return symbol;
  std::size_t i = 2;
  std::size_t len = 0;
  while (i < symbol.size() && symbol[i] >= '0' && symbol[i] <= '9') {
    len = len * 10 + static_cast<std::size_t>(symbol[i] - '0');
    ++i;
  }
  if (len == 0 || i + len > symbol.size()) return symbol;
  return symbol.substr(i, len);
}

inline bool ParseArgIndex(const char* name, std::uint32_t* index) {
  if (name == nullptr || *name == '\0') return false;
  std::uint32_t value = 0;
  for (const char* p = name; *p != '\0'; ++p) {
    if (*p < '0' || *p > '9') return false;
    value = value * 10 + static_cast<std::uint32_t>(*p - '0');
  }
  *index = value;
  return true;
}

}  // namespace detail

// Parses `image` and returns every dispatchable kernel in it, keyed by "<kernel>:<instance>".
//
// Throws std::runtime_error if the image is not a well-formed aie2p full ELF or uses a feature
// this reader does not implement.
inline std::map<std::string, Kernel> Parse(const std::uint8_t* image_data, std::size_t image_size) {
  const detail::Image image(image_data, image_size);

  const auto* ehdr = image.As<Elf32_Ehdr>(0);
  if (ehdr == nullptr || std::memcmp(ehdr->e_ident, ELFMAG, SELFMAG) != 0 ||
      ehdr->e_ident[EI_CLASS] != ELFCLASS32 || ehdr->e_ident[EI_DATA] != ELFDATA2LSB) {
    throw std::runtime_error("not a little-endian ELF32");
  }
  if (ehdr->e_ident[EI_OSABI] != kElfAmdAie2p) {
    throw std::runtime_error("not an aie2p AIE ELF");
  }
  if (ehdr->e_shentsize != sizeof(Elf32_Shdr) || ehdr->e_shnum == 0) {
    throw std::runtime_error("malformed section headers");
  }
  const std::uint8_t abi_version = ehdr->e_ident[EI_ABIVERSION];

  const auto* shdrs = image.As<Elf32_Shdr>(ehdr->e_shoff, ehdr->e_shnum);
  if (shdrs == nullptr || ehdr->e_shstrndx >= ehdr->e_shnum) {
    throw std::runtime_error("malformed section headers");
  }
  const Elf32_Shdr& shstrtab = shdrs[ehdr->e_shstrndx];

  auto section_name = [&](std::uint32_t index) -> const char* {
    if (index >= ehdr->e_shnum) return nullptr;
    return detail::StringAt(image, shstrtab, shdrs[index].sh_name);
  };

  const Elf32_Shdr* symtab = nullptr;
  const Elf32_Shdr* strtab = nullptr;
  const Elf32_Shdr* dynsym = nullptr;
  const Elf32_Shdr* dynstr = nullptr;
  const Elf32_Shdr* rela = nullptr;
  for (std::uint32_t i = 0; i < ehdr->e_shnum; ++i) {
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
    throw std::runtime_error("missing or malformed .symtab");
  }

  const std::uint32_t symtab_count = symtab->sh_size / sizeof(Elf32_Sym);
  const auto* symbols = image.As<Elf32_Sym>(symtab->sh_offset, symtab_count);
  if (symbols == nullptr) throw std::runtime_error("malformed .symtab");

  // Section index -> its bytes, for the PDI and control-code sections.
  auto section_bytes = [&](std::uint32_t index) {
    const std::uint8_t* data = image.At(shdrs[index].sh_offset, shdrs[index].sh_size);
    if (data == nullptr) throw std::runtime_error("section extends past end of file");
    return std::vector<std::uint8_t>(data, data + shdrs[index].sh_size);
  };

  // A group's sh_info is the .symtab index of its instance symbol, whose st_shndx in turn is the
  // .symtab index of the kernel's function symbol. That is an overload of st_shndx specific to
  // this ELF flavour, not a section index.
  struct Group {
    std::string name;
    std::uint32_t ctrltext_section = 0;
    std::uint32_t pdi_section = 0;
  };
  std::map<std::uint32_t, Group> groups;           // group section index -> group
  std::map<std::uint32_t, std::uint32_t> sec2grp;  // member section index -> group section index

  for (std::uint32_t i = 0; i < ehdr->e_shnum; ++i) {
    if (shdrs[i].sh_type != SHT_GROUP) continue;
    if (shdrs[i].sh_info >= symtab_count) throw std::runtime_error("bad group signature symbol");

    const Elf32_Sym& instance_sym = symbols[shdrs[i].sh_info];
    const char* instance_name = detail::StringAt(image, *strtab, instance_sym.st_name);
    if (instance_name == nullptr || instance_sym.st_shndx >= symtab_count) {
      throw std::runtime_error("bad group signature symbol");
    }
    const char* kernel_sym =
        detail::StringAt(image, *strtab, symbols[instance_sym.st_shndx].st_name);
    if (kernel_sym == nullptr) throw std::runtime_error("bad kernel symbol");

    Group group;
    group.name = detail::KernelNameFromSymbol(kernel_sym) + ":" + instance_name;

    // Group data is a flags word followed by the member section indices.
    const std::uint32_t word_count = shdrs[i].sh_size / sizeof(Elf32_Word);
    const auto* words = image.As<Elf32_Word>(shdrs[i].sh_offset, word_count);
    if (words == nullptr) throw std::runtime_error("malformed group section");
    for (std::uint32_t w = 1; w < word_count; ++w) {
      const std::uint32_t member = words[w];
      if (member >= ehdr->e_shnum) throw std::runtime_error("group member out of range");
      sec2grp[member] = i;
      const char* member_name = section_name(member);
      if (member_name != nullptr && std::strncmp(member_name, ".ctrltext", 9) == 0) {
        if (shdrs[member].sh_type != SHT_PROGBITS) {
          throw std::runtime_error("control code section holds no data");
        }
        group.ctrltext_section = member;
      }
    }
    groups.emplace(i, std::move(group));
  }
  if (groups.empty()) throw std::runtime_error("no COMDAT groups: not a group ELF");

  // Collect relocations. Each names a symbol whose st_shndx is the section being patched and
  // whose name says what address to write: ".pdi.N" for a PDI, a decimal string for an argument.
  std::map<std::uint32_t, std::map<std::uint32_t, std::vector<PatchSite>>> group_arg_sites;
  std::map<std::uint32_t, PatchSite> group_pdi_site;

  if (rela != nullptr && dynsym != nullptr && dynstr != nullptr) {
    if (rela->sh_entsize != sizeof(Elf32_Rela) || dynsym->sh_entsize != sizeof(Elf32_Sym)) {
      throw std::runtime_error("malformed .rela.dyn");
    }
    const std::uint32_t rela_count = rela->sh_size / sizeof(Elf32_Rela);
    const auto* relocs = image.As<Elf32_Rela>(rela->sh_offset, rela_count);
    const std::uint32_t dynsym_count = dynsym->sh_size / sizeof(Elf32_Sym);
    const auto* dynsyms = image.As<Elf32_Sym>(dynsym->sh_offset, dynsym_count);
    if (relocs == nullptr || dynsyms == nullptr) throw std::runtime_error("malformed .rela.dyn");

    for (std::uint32_t r = 0; r < rela_count; ++r) {
      const std::uint32_t sym_index = ELF32_R_SYM(relocs[r].r_info);
      if (sym_index >= dynsym_count) throw std::runtime_error("relocation symbol out of range");
      const Elf32_Sym& sym = dynsyms[sym_index];
      const char* sym_name = detail::StringAt(image, *dynstr, sym.st_name);
      if (sym_name == nullptr) throw std::runtime_error("bad relocation symbol name");

      auto grp_it = sec2grp.find(sym.st_shndx);
      if (grp_it == sec2grp.end()) continue;
      Group& group = groups.at(grp_it->second);
      if (sym.st_shndx != group.ctrltext_section) continue;  // only control code is patched

      PatchSite site;
      site.offset = relocs[r].r_offset;
      if (abi_version == 1) {
        // In ABI version 1 the scheme lives in the low bits of the addend rather than in r_info.
        site.addend = static_cast<std::uint32_t>(relocs[r].r_addend) >> 4;
        site.scheme = static_cast<PatchScheme>(relocs[r].r_addend & 0xF);
      } else {
        site.addend = static_cast<std::uint32_t>(relocs[r].r_addend);
        site.scheme = static_cast<PatchScheme>(ELF32_R_TYPE(relocs[r].r_info));
      }

      if (std::strncmp(sym_name, ".pdi", 4) == 0) {
        if (site.scheme != PatchScheme::kAddress64) {
          throw std::runtime_error("unexpected patch scheme for PDI symbol");
        }
        // Find the PDI section this symbol names.
        std::uint32_t pdi_section = 0;
        for (std::uint32_t s = 0; s < ehdr->e_shnum; ++s) {
          const char* n = section_name(s);
          if (n != nullptr && std::strcmp(n, sym_name) == 0) {
            pdi_section = s;
            break;
          }
        }
        if (pdi_section == 0) throw std::runtime_error("PDI section not found");
        if (group.pdi_section != 0 && group.pdi_section != pdi_section) {
          throw std::runtime_error("more than one PDI per kernel is not supported");
        }
        // A Kernel carries a single PDI patch offset, so a second site would be dropped and its
        // load_pdi left pointing at whatever placeholder the ELF holds.
        if (group_pdi_site.count(grp_it->second) != 0) {
          throw std::runtime_error("more than one PDI patch site per kernel is not supported");
        }
        group.pdi_section = pdi_section;
        group_pdi_site[grp_it->second] = site;
        continue;
      }

      std::uint32_t arg_index = 0;
      if (!detail::ParseArgIndex(sym_name, &arg_index)) {
        // A scratch pad, control packet or similar. Skipping it would leave a dangling address in
        // the control code, so refuse instead.
        throw std::runtime_error(std::string("unsupported relocation symbol: ") + sym_name);
      }
      if (site.scheme != PatchScheme::kShimDma48) {
        throw std::runtime_error("unsupported patch scheme for a kernel argument");
      }
      if (arg_index > kMaxArgIndex) {
        throw std::runtime_error("kernel argument index out of range");
      }
      group_arg_sites[grp_it->second][arg_index].push_back(site);
    }
  }

  std::map<std::string, Kernel> kernels;
  for (auto& [grp_index, group] : groups) {
    if (group.ctrltext_section == 0) continue;  // nothing to dispatch

    Kernel k;
    k.name = group.name;
    k.ctrl_code = section_bytes(group.ctrltext_section);
    if (k.ctrl_code.empty()) throw std::runtime_error("empty control code");

    if (group.pdi_section != 0) {
      k.pdi = section_bytes(group.pdi_section);
      k.pdi_patch_offset = group_pdi_site.at(grp_index).offset;
      k.has_pdi_patch = true;
      // The runtime writes a 64-bit address here, so it has to lie wholly inside the control code.
      if (k.pdi_patch_offset + sizeof(std::uint64_t) > k.ctrl_code.size() ||
          k.pdi_patch_offset % sizeof(std::uint32_t) != 0) {
        throw std::runtime_error("PDI patch site does not fit the control code");
      }
    }

    auto args_it = group_arg_sites.find(grp_index);
    if (args_it != group_arg_sites.end() && !args_it->second.empty()) {
      const std::uint32_t max_arg = args_it->second.rbegin()->first;
      k.arg_sites.resize(max_arg + 1);
      for (auto& [arg_index, sites] : args_it->second) {
        k.arg_sites[arg_index] = sites;
      }
    }

    kernels.emplace(k.name, std::move(k));
  }
  if (kernels.empty()) throw std::runtime_error("no dispatchable kernels");
  return kernels;
}

// Reads an ELF from disk and parses it.
inline std::map<std::string, Kernel> ParseFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + path);
  const auto size = static_cast<std::size_t>(f.tellg());
  f.seekg(0);
  std::vector<std::uint8_t> bytes(size);
  f.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
  if (static_cast<std::size_t>(f.gcount()) != size) throw std::runtime_error("short read " + path);
  return Parse(bytes.data(), bytes.size());
}

// Folds a buffer address into a shim DMA buffer descriptor, the scheme the NPU firmware defines.
// Note this *adds* to the descriptor already in place, so it must only ever be applied to a
// pristine copy of the control code -- see WriteControlCode.
inline void PatchShimDma48(std::uint32_t* site, std::uint64_t addr) {
  constexpr std::uint64_t kDdrAieAddrOffset = 0x80000000;
  std::uint64_t base =
      ((static_cast<std::uint64_t>(site[2]) & 0xFFFF) << 32) | static_cast<std::uint64_t>(site[1]);
  base += addr + kDdrAieAddrOffset;
  site[1] = static_cast<std::uint32_t>(base & 0xFFFFFFFC);
  site[2] = (site[2] & 0xFFFF0000) | static_cast<std::uint32_t>(base >> 32);
}

// Writes a dispatch-ready copy of `kernel`'s control code into `dst`, with `arg_addrs` patched in.
//
// `dst` must be at least kernel.ctrl_code.size() bytes and allocated from the agent's device
// memory pool. Always writes the pristine control code first: the shim DMA scheme is additive, so
// patching over a previous result would accumulate.
inline void WriteControlCode(const Kernel& kernel, void* dst, std::size_t dst_size,
                             const std::vector<std::uint64_t>& arg_addrs) {
  if (dst_size < kernel.ctrl_code.size()) throw std::runtime_error("control code buffer too small");
  if (arg_addrs.size() < kernel.num_args()) throw std::runtime_error("too few argument addresses");

  auto* out = static_cast<std::uint8_t*>(dst);
  std::memcpy(out, kernel.ctrl_code.data(), kernel.ctrl_code.size());

  for (std::uint32_t arg = 0; arg < kernel.arg_sites.size(); ++arg) {
    for (const PatchSite& site : kernel.arg_sites[arg]) {
      // The scheme reads and writes three dwords from the patch site.
      if (site.offset % sizeof(std::uint32_t) != 0 ||
          site.offset + 3 * sizeof(std::uint32_t) > kernel.ctrl_code.size()) {
        throw std::runtime_error("argument patch site out of range");
      }
      PatchShimDma48(reinterpret_cast<std::uint32_t*>(out + site.offset),
                     arg_addrs[arg] + site.addend);
    }
  }
}

}  // namespace aie_full_elf

#endif  // ROCRTST_SUITES_AIE_AIE_FULL_ELF_H_
