// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/amdgpu_elf_reader.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/instruction_builder.h"

#include "hsa/AMDHSAKernelDescriptor.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <optional>
#include <string_view>

namespace rocjitsu {

namespace {

using KernelDescriptor = rocr::llvm::amdhsa::kernel_descriptor_t;
namespace kd = rocr::llvm::amdhsa;

inline constexpr uint16_t kScalarLiteral = 255;
inline constexpr uint32_t kMaxInlinePositiveImm = 64;

struct EntryCounterProbeRegisters {
  uint8_t state_sgpr = 20;
  uint8_t workitem_vgpr = 20;
  uint8_t tmp0_vgpr = 21;
};

uint32_t build_v_mov_b32(uint8_t vdst, uint16_t src0) {
  return (0x3Fu << 25) | (static_cast<uint32_t>(vdst) << 17) | (1u << 9) | (src0 & 0x1FFu);
}

uint16_t amdgpu_positive_inline_const(uint32_t value) {
  return static_cast<uint16_t>(rocjitsu::kScalarPositiveInlineBase + value);
}

bool uses_gfx9_flat_global_encoding(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA2 || arch == ROCJITSU_CODE_ARCH_CDNA3 ||
         arch == ROCJITSU_CODE_ARCH_CDNA4;
}

bool uses_gfx10_flat_global_encoding(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2;
}

bool uses_gfx11_global_encoding(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5;
}

bool is_cdna_arch(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA1 || arch == ROCJITSU_CODE_ARCH_CDNA2 ||
         arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4;
}

bool is_rdna_arch(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2 ||
         arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
         arch == ROCJITSU_CODE_ARCH_RDNA4;
}

bool uses_gfx9_or_gfx10_waitcnt(rj_code_arch_t arch) {
  return uses_gfx9_flat_global_encoding(arch) || uses_gfx10_flat_global_encoding(arch);
}

uint32_t descriptor_vgpr_granularity_for_wavefront(rj_code_arch_t arch, uint32_t wavefront_size) {
  if (arch == ROCJITSU_CODE_ARCH_CDNA1)
    return 4;
  if (is_cdna_arch(arch))
    return 8;
  if (is_rdna_arch(arch))
    return wavefront_size == 32 ? 8 : 4;
  return 1;
}

uint32_t granulated_count_to_registers(uint32_t granulated, uint32_t granularity) {
  return (granulated + 1) * std::max(granularity, 1u);
}

uint32_t register_count_to_granulated(uint32_t registers, uint32_t granularity) {
  granularity = std::max(granularity, 1u);
  if (registers == 0)
    return 0;
  return (registers + granularity - 1) / granularity - 1;
}

void append_s_mov_b64_literal(std::vector<uint32_t> &words, uint8_t sdst_lo, uint64_t value,
                              rj_code_arch_t arch) {
  words.push_back(build_s_mov_b32(sdst_lo, kScalarLiteral, arch));
  words.push_back(static_cast<uint32_t>(value));
  words.push_back(build_s_mov_b32(static_cast<uint16_t>(sdst_lo + 1), kScalarLiteral, arch));
  words.push_back(static_cast<uint32_t>(value >> 32));
}

void append_s_wait_kmcnt(std::vector<uint32_t> &words, rj_code_arch_t arch) {
  if (uses_gfx9_or_gfx10_waitcnt(arch)) {
    words.push_back(pack_sopp(/*s_waitcnt=*/12, /*all counters complete=*/0));
    return;
  }
  if (uses_gfx11_global_encoding(arch)) {
    words.push_back(pack_sopp(/*s_waitcnt=*/9, /*all counters complete=*/0));
    return;
  }
  words.push_back(pack_sopp(/*s_wait_kmcnt=*/0x47, /*imm=*/0));
}

/// @brief Preserve the mbcnt dependency ordering across supported targets.
///
/// Newer targets have s_delay_alu for a precise VALU dependency barrier. Older
/// flat/global encodings used by the minimal probe do not, so a one-cycle NOP is
/// enough for this fixed sequence.
void append_valu_dep_2_barrier(std::vector<uint32_t> &words, rj_code_arch_t arch) {
  if (uses_gfx9_or_gfx10_waitcnt(arch)) {
    words.push_back(build_s_nop());
    return;
  }
  words.push_back(build_s_delay_alu(/*VALU_DEP_2=*/2, arch));
}

/// @brief Emit the counter update for the selected ISA memory encoding.
///
/// The entry probe always addresses counter slot 0, so the vector address
/// register is zero and @p saddr_lo/@p saddr_lo+1 hold the device counter base.
void append_global_store_u32(std::vector<uint32_t> &words, uint8_t vaddr, uint8_t vdata,
                             uint8_t saddr_lo, rj_code_arch_t arch) {
  if (uses_gfx9_flat_global_encoding(arch)) {
    words.push_back(0xDC708000u);
    words.push_back((static_cast<uint32_t>(saddr_lo) << 16) | (static_cast<uint32_t>(vdata) << 8) |
                    vaddr);
    return;
  }
  if (uses_gfx10_flat_global_encoding(arch)) {
    words.push_back(0xDC708000u);
    words.push_back((static_cast<uint32_t>(saddr_lo) << 16) | (static_cast<uint32_t>(vdata) << 8) |
                    vaddr);
    return;
  }
  if (uses_gfx11_global_encoding(arch)) {
    words.push_back(0xDC6A0000u);
    words.push_back((static_cast<uint32_t>(saddr_lo) << 16) | (static_cast<uint32_t>(vdata) << 8) |
                    vaddr);
    words.push_back(build_s_nop());
    return;
  }
  words.push_back(0xEE068000u | saddr_lo);
  words.push_back(static_cast<uint32_t>(vdata) << 23);
  words.push_back(vaddr);
}

std::optional<EntryCounterProbeRegisters>
allocate_entry_probe_registers(const KernelDescriptor &desc, rj_code_arch_t arch) {
  const bool wave32 =
      is_rdna_arch(arch) && AMDHSA_BITS_GET(desc.kernel_code_properties,
                                            kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32) != 0;
  const uint32_t wavefront_size = wave32 ? 32 : 64;
  const uint32_t vgpr_granularity = descriptor_vgpr_granularity_for_wavefront(arch, wavefront_size);
  const uint32_t vgpr_granulated =
      AMDHSA_BITS_GET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  const uint32_t vgprs = granulated_count_to_registers(vgpr_granulated, vgpr_granularity);

  constexpr uint32_t sgpr_granularity = 8;
  const uint32_t sgpr_granulated = AMDHSA_BITS_GET(
      desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
  const uint32_t sgprs = granulated_count_to_registers(sgpr_granulated, sgpr_granularity);
  if (sgprs > std::numeric_limits<uint8_t>::max() - 1 ||
      vgprs > std::numeric_limits<uint8_t>::max() - 1)
    return std::nullopt;

  // Place the fixed entry probe above the kernel's current allocation so the
  // prologue cannot clobber values that the original entry expects to consume.
  EntryCounterProbeRegisters regs;
  regs.state_sgpr = static_cast<uint8_t>(sgprs);
  regs.workitem_vgpr = static_cast<uint8_t>(vgprs);
  regs.tmp0_vgpr = static_cast<uint8_t>(vgprs + 1);
  return regs;
}

void reserve_entry_probe_registers(KernelDescriptor &desc, rj_code_arch_t arch,
                                   const EntryCounterProbeRegisters &regs) {
  const bool wave32 =
      is_rdna_arch(arch) && AMDHSA_BITS_GET(desc.kernel_code_properties,
                                            kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32) != 0;
  const uint32_t wavefront_size = wave32 ? 32 : 64;
  const uint32_t vgpr_granularity = descriptor_vgpr_granularity_for_wavefront(arch, wavefront_size);
  const uint32_t required_vgprs = std::max(static_cast<uint32_t>(regs.workitem_vgpr) + 1,
                                           static_cast<uint32_t>(regs.tmp0_vgpr) + 1);
  const uint32_t vgpr_granulated =
      AMDHSA_BITS_GET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  const uint32_t vgprs = granulated_count_to_registers(vgpr_granulated, vgpr_granularity);
  if (vgprs < required_vgprs) {
    AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                    register_count_to_granulated(required_vgprs, vgpr_granularity));
  }

  constexpr uint32_t sgpr_granularity = 8;
  const uint32_t required_sgprs = static_cast<uint32_t>(regs.state_sgpr) + 2;
  const uint32_t sgpr_granulated = AMDHSA_BITS_GET(
      desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
  const uint32_t sgprs = granulated_count_to_registers(sgpr_granulated, sgpr_granularity);
  if (sgprs < required_sgprs) {
    AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                    register_count_to_granulated(required_sgprs, sgpr_granularity));
  }
}

bool image_contains_range(size_t image_size, uint64_t offset, uint64_t size) {
  return offset <= image_size && size <= image_size - offset;
}

template <typename T> std::optional<T> read_at(std::span<const uint8_t> image, uint64_t offset) {
  if (!image_contains_range(image.size(), offset, sizeof(T)))
    return std::nullopt;

  T value{};
  std::memcpy(&value, image.data() + offset, sizeof(T));
  return value;
}

bool is_elf_header(const Elf64_Ehdr &ehdr) {
  return std::memcmp(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE) == 0;
}

bool supported_machine_flags(uint32_t flags) {
  switch (flags & EF_AMDGPU_MACH) {
  case EF_AMDGPU_MACH_AMDGCN_GFX942:
  case EF_AMDGPU_MACH_AMDGCN_GFX950:
  case EF_AMDGPU_MACH_AMDGCN_GFX1200:
  case EF_AMDGPU_MACH_AMDGCN_GFX1201:
    return true;
  default:
    return false;
  }
}

std::optional<Elf64_Ehdr> parse_supported_header(std::span<const uint8_t> image) {
  auto ehdr = read_at<Elf64_Ehdr>(image, 0);
  if (!ehdr.has_value())
    return std::nullopt;
  if (!is_elf_header(*ehdr) || ehdr->e_ident[EI_CLASS] != ELFCLASS64 ||
      ehdr->e_ident[EI_OSABI] != ELFOSABI_AMDGPU_HSA || ehdr->e_machine != EM_AMDGPU)
    return std::nullopt;
  if (ehdr->e_type != ET_DYN && ehdr->e_type != ET_REL)
    return std::nullopt;
  if (ehdr->e_shentsize != sizeof(Elf64_Shdr) || ehdr->e_shnum == 0)
    return std::nullopt;
  if (!image_contains_range(image.size(), ehdr->e_shoff,
                            static_cast<uint64_t>(ehdr->e_shnum) * sizeof(Elf64_Shdr)))
    return std::nullopt;
  if (!supported_machine_flags(ehdr->e_flags))
    return std::nullopt;
  return ehdr;
}

std::optional<std::vector<Elf64_Shdr>> parse_section_headers(std::span<const uint8_t> image,
                                                             const Elf64_Ehdr &ehdr) {
  std::vector<Elf64_Shdr> shdrs(ehdr.e_shnum);
  std::memcpy(shdrs.data(), image.data() + ehdr.e_shoff, shdrs.size() * sizeof(Elf64_Shdr));
  return shdrs;
}

bool kernel_descriptor_symbol(const Elf64_Sym &symbol, const char *strtab, size_t strtab_size) {
  if (symbol.st_size != sizeof(KernelDescriptor))
    return false;
  if (elf_symbol_type(symbol.st_info) != kElfSymbolTypeObject ||
      elf_symbol_bind(symbol.st_info) != kElfSymbolBindGlobal)
    return false;
  if (strtab == nullptr || strtab_size == 0 || symbol.st_name == 0 || symbol.st_name >= strtab_size)
    return false;

  const char *name = strtab + symbol.st_name;
  const size_t len = strnlen(name, strtab_size - symbol.st_name);
  return len > 3 && std::string_view(name, len).ends_with(".kd");
}

std::string kernel_name_from_symbol(const Elf64_Sym &symbol, const char *strtab,
                                    size_t strtab_size) {
  const char *name = strtab + symbol.st_name;
  const size_t len = strnlen(name, strtab_size - symbol.st_name);
  return std::string(name, len - 3);
}

std::optional<uint64_t> symbol_file_offset(const Elf64_Sym &symbol,
                                           std::span<const Elf64_Shdr> shdrs, size_t image_size) {
  if (symbol.st_shndx == SHN_UNDEF || symbol.st_shndx == SHN_ABS || symbol.st_shndx >= shdrs.size())
    return std::nullopt;

  const Elf64_Shdr &section = shdrs[symbol.st_shndx];
  if (symbol.st_value < section.sh_addr)
    return std::nullopt;

  const uint64_t section_offset = symbol.st_value - section.sh_addr;
  if (section_offset > section.sh_size)
    return std::nullopt;

  const uint64_t file_offset = section.sh_offset + section_offset;
  if (!image_contains_range(image_size, file_offset, sizeof(KernelDescriptor)))
    return std::nullopt;

  return file_offset;
}

std::optional<uint64_t> add_signed_offset(uint64_t base, int64_t offset) {
  if (offset >= 0) {
    const uint64_t unsigned_offset = static_cast<uint64_t>(offset);
    if (base > UINT64_MAX - unsigned_offset)
      return std::nullopt;
    return base + unsigned_offset;
  }

  const uint64_t magnitude = static_cast<uint64_t>(-(offset + 1)) + 1;
  if (base < magnitude)
    return std::nullopt;
  return base - magnitude;
}

std::optional<uint64_t> executable_vaddr_to_file_offset(uint64_t vaddr,
                                                        std::span<const Elf64_Shdr> shdrs,
                                                        size_t image_size) {
  for (const Elf64_Shdr &section : shdrs) {
    if ((section.sh_flags & SHF_EXECINSTR) == 0)
      continue;
    if (vaddr < section.sh_addr || vaddr >= section.sh_addr + section.sh_size)
      continue;

    const uint64_t file_offset = section.sh_offset + (vaddr - section.sh_addr);
    if (!image_contains_range(image_size, file_offset, 1))
      return std::nullopt;
    return file_offset;
  }
  return std::nullopt;
}

std::vector<uint32_t>
build_amdgpu_entry_counter_probe_words(uint64_t state_pointer, rj_code_arch_t arch,
                                       const EntryCounterProbeRegisters &regs) {
  std::vector<uint32_t> words;
  append_s_mov_b64_literal(words, regs.state_sgpr, state_pointer, arch);
  constexpr uint32_t slot_byte_offset = 0;
  static_assert(slot_byte_offset <= kMaxInlinePositiveImm);
  words.push_back(
      build_v_mov_b32(regs.workitem_vgpr, amdgpu_positive_inline_const(slot_byte_offset)));
  words.push_back(build_v_mov_b32(regs.tmp0_vgpr, amdgpu_positive_inline_const(1)));
  append_valu_dep_2_barrier(words, arch);
  append_s_wait_kmcnt(words, arch);
  append_global_store_u32(words, regs.workitem_vgpr, regs.tmp0_vgpr, regs.state_sgpr, arch);
  append_s_wait_kmcnt(words, arch);
  return words;
}

} // namespace

bool is_supported_amdgpu_elf(std::span<const uint8_t> image) {
  return parse_supported_header(image).has_value();
}

std::vector<AmdGpuKernelSite> discover_amdgpu_kernel_sites(std::span<const uint8_t> image) {
  const auto ehdr = parse_supported_header(image);
  if (!ehdr.has_value())
    return {};

  const auto shdrs = parse_section_headers(image, *ehdr);
  if (!shdrs.has_value())
    return {};

  std::vector<AmdGpuKernelSite> sites;
  for (const Elf64_Shdr &symtab : *shdrs) {
    if (symtab.sh_type != SHT_SYMTAB && symtab.sh_type != SHT_DYNSYM)
      continue;
    if (symtab.sh_entsize != sizeof(Elf64_Sym) || symtab.sh_link >= shdrs->size())
      continue;
    if (!image_contains_range(image.size(), symtab.sh_offset, symtab.sh_size))
      continue;

    const Elf64_Shdr &strtab_section = (*shdrs)[symtab.sh_link];
    if (!image_contains_range(image.size(), strtab_section.sh_offset, strtab_section.sh_size))
      continue;

    const char *strtab = reinterpret_cast<const char *>(image.data() + strtab_section.sh_offset);
    const size_t strtab_size = static_cast<size_t>(strtab_section.sh_size);
    const size_t symbol_count = symtab.sh_size / sizeof(Elf64_Sym);
    for (size_t i = 0; i < symbol_count; ++i) {
      const auto symbol = read_at<Elf64_Sym>(image, symtab.sh_offset + i * sizeof(Elf64_Sym));
      if (!symbol.has_value() || !kernel_descriptor_symbol(*symbol, strtab, strtab_size))
        continue;

      const auto descriptor_offset = symbol_file_offset(*symbol, *shdrs, image.size());
      if (!descriptor_offset.has_value())
        continue;

      const auto descriptor = read_at<KernelDescriptor>(image, *descriptor_offset);
      if (!descriptor.has_value())
        continue;

      const auto entry_vaddr =
          add_signed_offset(symbol->st_value, descriptor->kernel_code_entry_byte_offset);
      if (!entry_vaddr.has_value())
        continue;

      const auto entry_offset = executable_vaddr_to_file_offset(*entry_vaddr, *shdrs, image.size());
      if (!entry_offset.has_value())
        continue;

      sites.push_back(AmdGpuKernelSite{
          .kernel_name = kernel_name_from_symbol(*symbol, strtab, strtab_size),
          .descriptor_file_offset = *descriptor_offset,
          .entry_file_offset = *entry_offset,
      });
    }
  }

  std::sort(sites.begin(), sites.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.descriptor_file_offset < rhs.descriptor_file_offset;
  });
  // HIPRTC code objects may publish the same `.kd` symbol through both
  // `.dynsym` and `.symtab`; treat those aliases as one kernel descriptor.
  sites.erase(std::unique(sites.begin(), sites.end(),
                          [](const auto &lhs, const auto &rhs) {
                            return lhs.descriptor_file_offset == rhs.descriptor_file_offset;
                          }),
              sites.end());
  return sites;
}

/// @brief Return entry-probe words for a patched raw AMDGPU ELF kernel.
///
/// Example use: the HSA reader rewrite path can call this with the runtime's
/// device counter buffer address, append the returned words to an executable
/// cave, and redirect each discovered kernel descriptor entry to that cave. A
/// vector-add AFL smoke then observes device coverage whenever the HIPRTC kernel
/// launches, even before general basic-block edge instrumentation exists.
std::vector<uint32_t> build_amdgpu_entry_counter_probe_words(uint64_t state_pointer,
                                                             rj_code_arch_t arch) {
  return build_amdgpu_entry_counter_probe_words(state_pointer, arch, EntryCounterProbeRegisters{});
}

std::vector<uint8_t> patch_amdgpu_elf_kernel_entries(std::span<const uint8_t> image,
                                                     uint64_t state_pointer) {
  std::vector<uint8_t> fail_open(image.begin(), image.end());
  const auto header = parse_supported_header(image);
  if (!header.has_value())
    return fail_open;

  const auto arch = arch_from_elf_flags(header->e_flags);
  if (arch == ROCJITSU_CODE_ARCH_INVALID)
    return fail_open;

  const auto sites = discover_amdgpu_kernel_sites(image);
  if (sites.empty())
    return fail_open;

  AmdGpuCodeObject code_object(image.data(), image.size());
  if (!code_object.is_valid())
    return fail_open;
  if (code_object.text_sections().empty())
    return fail_open;

  CodeObjectPatcher patcher(code_object);
  const uint64_t text_offset = patcher.text_offset();
  patcher.set_cave_start(patcher.text_size());

  for (const auto &site : sites) {
    if (site.entry_file_offset < text_offset)
      return fail_open;
    const uint64_t entry_text_offset = site.entry_file_offset - text_offset;

    auto descriptor = read_at<KernelDescriptor>(patcher.image_bytes(), site.descriptor_file_offset);
    if (!descriptor.has_value())
      return fail_open;

    const auto regs = allocate_entry_probe_registers(*descriptor, arch);
    if (!regs.has_value())
      return fail_open;
    const auto probe_words = build_amdgpu_entry_counter_probe_words(state_pointer, arch, *regs);
    if (probe_words.empty())
      return fail_open;

    const auto new_entry =
        patcher.append_kernel_entry_prologue(entry_text_offset, probe_words, arch);
    if (!new_entry.has_value())
      return fail_open;
    if (!patcher.redirect_kernel_entry(site.descriptor_file_offset, entry_text_offset, *new_entry))
      return fail_open;
    descriptor = read_at<KernelDescriptor>(patcher.image_bytes(), site.descriptor_file_offset);
    if (!descriptor.has_value())
      return fail_open;
    reserve_entry_probe_registers(*descriptor, arch, *regs);
    if (!patcher.patch_kernel_descriptor(
            site.descriptor_file_offset,
            std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(&*descriptor),
                                     sizeof(*descriptor))))
      return fail_open;
  }

  if (!patcher.append_cave_section(".rj_translations"))
    return fail_open;
  return patcher.emit();
}

} // namespace rocjitsu
