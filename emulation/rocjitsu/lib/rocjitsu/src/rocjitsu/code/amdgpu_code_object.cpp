// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/amdgpu_code_object.h"

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/file_io.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rdna_isa_base.h"
#include "rocjitsu/isa/target_registry.h"

#include "hsa/AMDHSAKernelDescriptor.h" // Check SGPR allocation

#include <algorithm>
#include <cstring>
#include <iterator>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace rocjitsu {

namespace {

class HsaHeader : public Header {
public:
  explicit HsaHeader(const Elf64_Ehdr &ehdr) : ehdr_(ehdr) {}

  uint64_t programHeaderOff() const override { return ehdr_.e_phoff; }
  int numProgramHeaders() const override { return static_cast<int>(ehdr_.e_phnum); }
  uint64_t sectionHeaderOff() const override { return ehdr_.e_shoff; }
  int numSectionHeaders() const override { return static_cast<int>(ehdr_.e_shnum); }
  int sectionHeaderStrIdx() const override { return static_cast<int>(ehdr_.e_shstrndx); }
  uint32_t flags() const override { return ehdr_.e_flags; }

private:
  Elf64_Ehdr ehdr_;
};

class HsaSection : public Section {
public:
  HsaSection(std::string name, std::unique_ptr<char[]> data, const Elf64_Shdr &shdr,
             size_t section_index)
      : Section(std::move(name), std::move(data)), shdr_(shdr), section_index_(section_index) {}

  std::size_t size() const override { return shdr_.sh_size; }
  uint64_t flags() const override { return shdr_.sh_flags; }
  uint64_t vaddr() const override { return shdr_.sh_addr; }
  uint32_t sectionHeaderNameIdx() const override { return shdr_.sh_name; }
  std::optional<size_t> sectionHeaderIndex() const override { return section_index_; }
  uint64_t sectionOffset() const override { return shdr_.sh_offset; }

private:
  Elf64_Shdr shdr_;
  size_t section_index_;
};

bool is_elf(const Elf64_Ehdr &ehdr) { return !std::memcmp(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE); }

using detail::fits_in_bounds;

rj_code_target_id_t target_from_machine_flags(uint32_t flags) {
  const IsaGpuTargetDescription *binding =
      default_isa_target_registry().find_gpu_target_by_elf_machine(flags & EF_AMDGPU_MACH);
  return binding == nullptr ? ROCJITSU_CODE_TARGET_INVALID : binding->public_id;
}

rj_code_target_id_t target_from_code_object_id(std::string_view id) {
  const IsaGpuTargetDescription *binding =
      default_isa_target_registry().find_gpu_target_by_code_object_id(id);
  return binding == nullptr ? ROCJITSU_CODE_TARGET_INVALID : binding->public_id;
}

} // namespace

AmdGpuCodeObject::AmdGpuCodeObject(AmdGpuCodeObject &&other) noexcept
    : target_id_(other.target_id_), offload_kind_(std::move(other.offload_kind_)),
      target_triple_(std::move(other.target_triple_)) {
  is_valid_ = other.is_valid_;
  image_ = std::move(other.image_);
  header_ = std::move(other.header_);
  sections_ = std::move(other.sections_);
  text_sections_ = std::move(other.text_sections_);
  executable_nobits_sections_ = std::move(other.executable_nobits_sections_);
  allocated_executable_sections_ = std::move(other.allocated_executable_sections_);
  rodata_sections_ = std::move(other.rodata_sections_);
}

AmdGpuCodeObject::AmdGpuCodeObject(const std::string &elf_path) {
  try {
    image_ = detail::read_file_bytes(elf_path);
  } catch (const std::exception &) {
    is_valid_ = false;
    return;
  }

  if (image_.size() < sizeof(Elf64_Ehdr)) {
    is_valid_ = false;
    return;
  }

  Elf64_Ehdr ehdr;
  std::memcpy(&ehdr, image_.data(), sizeof(Elf64_Ehdr));

  if (!is_elf(ehdr) || ehdr.e_ident[EI_CLASS] != ELFCLASS64 ||
      ehdr.e_ident[EI_OSABI] != ELFOSABI_AMDGPU_HSA) {
    is_valid_ = false;
    return;
  }

  header_ = std::make_unique<HsaHeader>(ehdr);
  load_sections();
  if (!is_valid_)
    return;
  target_id_ = target_from_machine_flags(header_->flags());
}

AmdGpuCodeObject::AmdGpuCodeObject(const uint8_t *elf_bytes, size_t elf_size) {
  if (elf_size < sizeof(Elf64_Ehdr)) {
    is_valid_ = false;
    return;
  }

  image_.assign(reinterpret_cast<const char *>(elf_bytes),
                reinterpret_cast<const char *>(elf_bytes) + elf_size);

  Elf64_Ehdr ehdr;
  std::memcpy(&ehdr, image_.data(), sizeof(Elf64_Ehdr));

  if (!is_elf(ehdr) || ehdr.e_ident[EI_CLASS] != ELFCLASS64 ||
      ehdr.e_ident[EI_OSABI] != ELFOSABI_AMDGPU_HSA) {
    is_valid_ = false;
    return;
  }

  header_ = std::make_unique<HsaHeader>(ehdr);
  load_sections();
  if (!is_valid_)
    return;
  target_id_ = target_from_machine_flags(header_->flags());
}

AmdGpuCodeObject::AmdGpuCodeObject(const uint8_t *elf_bytes, size_t elf_size,
                                   std::string offload_kind, std::string target_triple)
    : offload_kind_(std::move(offload_kind)), target_triple_(std::move(target_triple)) {
  if (elf_size < sizeof(Elf64_Ehdr)) {
    is_valid_ = false;
    return;
  }

  image_.assign(reinterpret_cast<const char *>(elf_bytes),
                reinterpret_cast<const char *>(elf_bytes) + elf_size);

  Elf64_Ehdr ehdr;
  std::memcpy(&ehdr, image_.data(), sizeof(Elf64_Ehdr));

  if (!is_elf(ehdr) || ehdr.e_ident[EI_CLASS] != ELFCLASS64 ||
      ehdr.e_ident[EI_OSABI] != ELFOSABI_AMDGPU_HSA) {
    is_valid_ = false;
    return;
  }

  header_ = std::make_unique<HsaHeader>(ehdr);
  load_sections();
  if (!is_valid_)
    return;
  const rj_code_target_id_t bundle_target = target_from_code_object_id(target_triple_);
  const rj_code_target_id_t elf_target = target_from_machine_flags(header_->flags());
  // A recognized bundle target and inner ELF machine are two declarations of
  // the same concrete GPU identity. Accepting a disagreement here would let
  // callers select a decoder using the bundle while patching code for another
  // target. When only one declaration is recognized, retain that concrete
  // identity instead of degrading it to the INVALID wildcard.
  if (bundle_target != ROCJITSU_CODE_TARGET_INVALID && elf_target != ROCJITSU_CODE_TARGET_INVALID &&
      bundle_target != elf_target) {
    is_valid_ = false;
    return;
  }
  target_id_ = bundle_target != ROCJITSU_CODE_TARGET_INVALID ? bundle_target : elf_target;
}

AmdGpuCodeObject::~AmdGpuCodeObject() = default;

void AmdGpuCodeObject::load_sections() {
  const auto shoff = header_->sectionHeaderOff();
  const int num_shdrs = header_->numSectionHeaders();
  if (num_shdrs < 0 || !fits_in_bounds(shoff, static_cast<uint64_t>(num_shdrs) * sizeof(Elf64_Shdr),
                                       image_.size())) {
    is_valid_ = false;
    return;
  }

  std::vector<Elf64_Shdr> section_hdrs(static_cast<size_t>(num_shdrs));
  std::memcpy(section_hdrs.data(), image_.data() + shoff, section_hdrs.size() * sizeof(Elf64_Shdr));

  int shstrndx = header_->sectionHeaderStrIdx();
  if (shstrndx < 0 || static_cast<size_t>(shstrndx) >= section_hdrs.size()) {
    is_valid_ = false;
    return;
  }

  auto &shstrtab = section_hdrs[shstrndx];
  if (!fits_in_bounds(shstrtab.sh_offset, shstrtab.sh_size, image_.size())) {
    is_valid_ = false;
    return;
  }
  const char *shstrtab_data = image_.data() + shstrtab.sh_offset;

  for (size_t section_index = 0; section_index < section_hdrs.size(); ++section_index) {
    const Elf64_Shdr &shdr = section_hdrs[section_index];
    if (shdr.sh_type == SHT_NULL)
      continue;
    const bool allocated_executable =
        (shdr.sh_flags & (SHF_ALLOC | SHF_EXECINSTR)) == (SHF_ALLOC | SHF_EXECINSTR);
    if (shdr.sh_type == SHT_NOBITS && !allocated_executable)
      continue;
    if (shdr.sh_name >= shstrtab.sh_size) {
      if (allocated_executable) {
        is_valid_ = false;
        return;
      }
      continue;
    }
    if (shdr.sh_type != SHT_NOBITS &&
        !fits_in_bounds(shdr.sh_offset, shdr.sh_size, image_.size())) {
      is_valid_ = false;
      return;
    }

    size_t max_len = shstrtab.sh_size - shdr.sh_name;
    std::string sec_name(shstrtab_data + shdr.sh_name,
                         strnlen(shstrtab_data + shdr.sh_name, max_len));

    std::unique_ptr<char[]> sec_data;
    if (shdr.sh_type != SHT_NOBITS) {
      sec_data = std::make_unique<char[]>(shdr.sh_size);
      std::memcpy(sec_data.get(), image_.data() + shdr.sh_offset, shdr.sh_size);
    }
    // SHT_NOBITS has no executable bytes to decode. Retain allocated executable
    // metadata above so the translator's sole-PROGBITS-.text gate rejects the
    // layout, but keep all_sections() restricted to readable section data.
    if (shdr.sh_type == SHT_NOBITS) {
      executable_nobits_sections_.emplace_back(
          std::make_unique<HsaSection>(sec_name, nullptr, shdr, section_index));
      allocated_executable_sections_.push_back(executable_nobits_sections_.back().get());
      continue;
    }

    sections_.emplace_back(
        std::make_unique<HsaSection>(sec_name, std::move(sec_data), shdr, section_index));

    if (allocated_executable)
      allocated_executable_sections_.push_back(sections_.back().get());
    if (sec_name == ".text")
      text_sections_.push_back(sections_.back().get());
    else if (sec_name == ".rodata")
      rodata_sections_.push_back(sections_.back().get());
  }

  // Parse symbol table for kernel descriptor offsets.
  // Scan both SHT_SYMTAB and SHT_DYNSYM — stripped code objects may
  // only have the latter.
  for (size_t i = 0; i < section_hdrs.size(); ++i) {
    if (section_hdrs[i].sh_type != SHT_SYMTAB && section_hdrs[i].sh_type != SHT_DYNSYM)
      continue;
    auto &symtab_shdr = section_hdrs[i];
    if (symtab_shdr.sh_entsize == 0)
      continue;
    if (symtab_shdr.sh_entsize < sizeof(Elf64_Sym))
      continue;
    if (!fits_in_bounds(symtab_shdr.sh_offset, symtab_shdr.sh_size, image_.size()))
      continue;

    // Read the string table linked to this symtab.
    if (symtab_shdr.sh_link >= section_hdrs.size())
      continue;
    auto &strtab_shdr = section_hdrs[symtab_shdr.sh_link];
    if (!fits_in_bounds(strtab_shdr.sh_offset, strtab_shdr.sh_size, image_.size()))
      continue;
    const char *sym_strtab = image_.data() + strtab_shdr.sh_offset;

    // Read symbols.
    size_t num_syms = symtab_shdr.sh_size / symtab_shdr.sh_entsize;
    const char *symtab_data = image_.data() + symtab_shdr.sh_offset;

    for (size_t sym_index = 0; sym_index < num_syms; ++sym_index) {
      const char *sym_data = symtab_data + sym_index * symtab_shdr.sh_entsize;
      Elf64_Sym sym;
      std::memcpy(&sym, sym_data, sizeof(sym));
      if (sym.st_name >= strtab_shdr.sh_size)
        continue;
      std::string sym_name(sym_strtab + sym.st_name,
                           strnlen(sym_strtab + sym.st_name, strtab_shdr.sh_size - sym.st_name));
      // AMDHSA kernel descriptors have a ".kd" suffix symbol.
      if (sym_name.size() > 3 && sym_name.substr(sym_name.size() - 3) == ".kd") {
        std::string kernel_name = sym_name.substr(0, sym_name.size() - 3);
        kd_offsets_[kernel_name] = sym.st_value;
      }
    }
  }

  is_valid_ = true;
}

uint64_t AmdGpuCodeObject::kernel_descriptor_offset(const std::string &kernel_name) const {
  auto it = kd_offsets_.find(kernel_name);
  return it != kd_offsets_.end() ? it->second : 0;
}

namespace {

// CDNA targets encode the wavefront SGPR count in the descriptor even when the
// granulated field is 0; RDNA-style targets treat a granulated 0 as "use the
// fixed per-wave SGPR pool". Mirrors the command processor's
// sgpr_count_is_descriptor_encoded(); kept as the short, stable CDNA
// list so a new non-CDNA family falls through to the RDNA-style branch by
// default. (Canonical copy: code/dbt/kernel_descriptor_translator.cpp.)
[[nodiscard]] bool is_cdna_arch(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA1 || arch == ROCJITSU_CODE_ARCH_CDNA2 ||
         arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4;
}

// Decode one kernel descriptor's per-wave SGPR count from its granulated field.
[[nodiscard]] uint32_t sgpr_count_from_granulated(uint32_t granulated, rj_code_arch_t arch) {
  // Descriptor-encoded (granulated != 0, or a CDNA target): (granulated + 1) * 8.
  // Otherwise the field is an RDNA-style sentinel and the wave owns the fixed
  // per-wave SGPR pool.
  if (granulated != 0 || is_cdna_arch(arch))
    return (granulated + 1) * 8;
  return amdgpu::RdnaIsaBase::MAX_SGPRS_PER_WF;
}

} // namespace

std::optional<uint32_t>
AmdGpuCodeObject::min_kernel_sgpr_count(rj_code_arch_t arch,
                                        std::span<const KernelDescriptorInfo> kernels) {
  namespace kd = rocr::llvm::amdhsa;

  std::optional<uint32_t> min_count;
  for (const KernelDescriptorInfo &kernel : kernels) {
    const uint32_t granulated = AMDHSA_BITS_GET(
        kernel.descriptor.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
    const uint32_t count = sgpr_count_from_granulated(granulated, arch);
    min_count = min_count ? std::min(*min_count, count) : count;
  }
  return min_count;
}

std::optional<uint32_t> AmdGpuCodeObject::min_kernel_sgpr_count(rj_code_arch_t arch) const {
  const std::span<const uint8_t> image{reinterpret_cast<const uint8_t *>(image_data()),
                                       image_size()};
  // Discover descriptors through the shared scanner rather than re-walking sections
  // here. scan_kernel_descriptors() filters to a single .text extent, so visit each
  // text section to cover every kernel, then reduce via the span overload.
  std::vector<KernelDescriptorInfo> kernels;
  for (const Section *text : text_sections()) {
    std::vector<KernelDescriptorInfo> found =
        scan_kernel_descriptors(image, text->sectionOffset(), text->size());
    kernels.insert(kernels.end(), std::make_move_iterator(found.begin()),
                   std::make_move_iterator(found.end()));
  }
  return min_kernel_sgpr_count(arch, kernels);
}

} // namespace rocjitsu
