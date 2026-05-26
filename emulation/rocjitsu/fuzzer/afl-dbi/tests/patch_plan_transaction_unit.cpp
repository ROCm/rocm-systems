#include "instrumentation_planner.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu_fuzzer/afl_dbi_plan.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <array>
#include <cstdlib>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace {

using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
namespace kd = rocr::llvm::amdhsa;

void check(bool condition, const char *message) {
  if (!condition) {
    fprintf(stderr, "patch_plan_transaction_unit: %s\n", message);
    std::exit(1);
  }
}

uint32_t add_elf_name(std::vector<uint8_t> &names, std::string_view name) {
  const uint32_t offset = static_cast<uint32_t>(names.size());
  names.insert(names.end(), name.begin(), name.end());
  names.push_back('\0');
  return offset;
}

uint64_t align_up_for_test(uint64_t value, uint64_t alignment) {
  const uint64_t remainder = value % alignment;
  return remainder == 0 ? value : value + alignment - remainder;
}

void append_msgpack_string(std::vector<uint8_t> &bytes, std::string_view value) {
  check(value.size() <= 31, "test msgpack string should fit fixstr");
  bytes.push_back(static_cast<uint8_t>(0xa0u | value.size()));
  bytes.insert(bytes.end(), value.begin(), value.end());
}

void append_msgpack_uint(std::vector<uint8_t> &bytes, uint32_t value) {
  if (value <= 0x7fu) {
    bytes.push_back(static_cast<uint8_t>(value));
    return;
  }
  if (value <= 0xffu) {
    bytes.push_back(0xcc);
    bytes.push_back(static_cast<uint8_t>(value));
    return;
  }
  bytes.push_back(0xcd);
  bytes.push_back(static_cast<uint8_t>((value >> 8) & 0xffu));
  bytes.push_back(static_cast<uint8_t>(value & 0xffu));
}

std::vector<uint8_t>
make_minimal_amdgpu_metadata_note(std::optional<uint32_t> sgpr_count = std::nullopt) {
  std::vector<uint8_t> metadata;
  metadata.push_back(0x82); // map(2)
  append_msgpack_string(metadata, "amdhsa.kernels");
  metadata.push_back(0x91); // array(1)
  metadata.push_back(static_cast<uint8_t>(0x83u + (sgpr_count ? 1u : 0u)));
  append_msgpack_string(metadata, ".name");
  append_msgpack_string(metadata, "kernel");
  append_msgpack_string(metadata, ".symbol");
  append_msgpack_string(metadata, "kernel.kd");
  append_msgpack_string(metadata, ".private_segment_fixed_size");
  metadata.push_back(0x00);
  if (sgpr_count) {
    append_msgpack_string(metadata, ".sgpr_count");
    append_msgpack_uint(metadata, *sgpr_count);
  }
  append_msgpack_string(metadata, "amdhsa.version");
  metadata.push_back(0x92); // array(2)
  metadata.push_back(0x01);
  metadata.push_back(0x02);

  struct NoteHeader {
    uint32_t namesz = 0;
    uint32_t descsz = 0;
    uint32_t type = 0;
  };
  NoteHeader note;
  note.namesz = 7;
  note.descsz = static_cast<uint32_t>(metadata.size());
  note.type = rocjitsu::NT_AMDGPU_METADATA;

  std::vector<uint8_t> note_bytes(sizeof(note), 0);
  std::memcpy(note_bytes.data(), &note, sizeof(note));
  const char owner[] = "AMDGPU";
  note_bytes.insert(note_bytes.end(), owner, owner + sizeof(owner));
  while (note_bytes.size() % 4 != 0)
    note_bytes.push_back(0);
  note_bytes.insert(note_bytes.end(), metadata.begin(), metadata.end());
  while (note_bytes.size() % 4 != 0)
    note_bytes.push_back(0);
  return note_bytes;
}

std::vector<uint8_t> snapshot(rocjitsu::CodeObjectPatcher &patcher) {
  return {patcher.image_bytes().begin(), patcher.image_bytes().end()};
}

KD read_kernel_descriptor_for_test(std::span<const uint8_t> image, uint64_t offset) {
  check(offset <= image.size() && sizeof(KD) <= image.size() - offset,
        "kernel descriptor offset should be in range");
  KD desc{};
  std::memcpy(&desc, image.data() + offset, sizeof(desc));
  return desc;
}

std::vector<rocjitsu::Elf64_Phdr> read_program_headers_for_test(
    std::span<const uint8_t> image) {
  check(image.size() >= sizeof(rocjitsu::Elf64_Ehdr),
        "ELF header should be in range");
  rocjitsu::Elf64_Ehdr ehdr{};
  std::memcpy(&ehdr, image.data(), sizeof(ehdr));
  check(ehdr.e_phentsize == sizeof(rocjitsu::Elf64_Phdr),
        "fixture should use native program headers");
  check(ehdr.e_phoff <= image.size() &&
            static_cast<uint64_t>(ehdr.e_phnum) * sizeof(rocjitsu::Elf64_Phdr) <=
                image.size() - ehdr.e_phoff,
        "program headers should be in range");
  std::vector<rocjitsu::Elf64_Phdr> phdrs(ehdr.e_phnum);
  std::memcpy(phdrs.data(), image.data() + ehdr.e_phoff,
              phdrs.size() * sizeof(phdrs[0]));
  return phdrs;
}

bool has_appended_pt_note(std::span<const uint8_t> image, uint64_t original_size) {
  for (const rocjitsu::Elf64_Phdr &phdr : read_program_headers_for_test(image)) {
    if (phdr.p_type == rocjitsu::PT_NOTE && phdr.p_offset >= original_size)
      return true;
  }
  return false;
}

rocjitsu::fuzzer::afl_dbi::KernelSite
single_kernel_site_for_test(rocjitsu::CodeObjectPatcher &patcher) {
  std::vector<rocjitsu::fuzzer::afl_dbi::KernelSite> sites =
      rocjitsu::fuzzer::afl_dbi::find_kernel_sites(patcher.image_bytes());
  check(sites.size() == 1, "fixture should expose one kernel descriptor");
  return sites[0];
}

void check_image_unchanged(rocjitsu::CodeObjectPatcher &patcher,
                           const std::vector<uint8_t> &original,
                           const char *message) {
  check(snapshot(patcher) == original, message);
}

std::vector<uint8_t>
make_minimal_descriptor_elf(bool include_metadata = false,
                            uint32_t elf_mach = rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX950,
                            std::optional<uint32_t> metadata_sgpr_count = std::nullopt) {
  constexpr uint64_t text_offset = 0x100;
  constexpr uint64_t text_vaddr = 0x1100;
  constexpr uint64_t text_size = 8;
  constexpr uint64_t load_align = 0x1000;
  constexpr uint64_t rodata_size = sizeof(KD);
  const std::vector<uint8_t> metadata_note =
      include_metadata ? make_minimal_amdgpu_metadata_note(metadata_sgpr_count)
                       : std::vector<uint8_t>{};

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_elf_name(shstrtab, ".text");
  const uint32_t rodata_name = add_elf_name(shstrtab, ".rodata");
  const uint32_t note_name = include_metadata ? add_elf_name(shstrtab, ".note") : 0;
  const uint32_t symtab_name = add_elf_name(shstrtab, ".symtab");
  const uint32_t strtab_name = add_elf_name(shstrtab, ".strtab");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");

  std::vector<uint8_t> strtab{'\0'};
  const uint32_t kd_symbol_name = add_elf_name(strtab, "kernel.kd");

  const uint64_t rodata_offset = text_offset + text_size;
  const uint64_t rodata_vaddr = text_vaddr + text_size + load_align;
  const uint64_t note_offset = rodata_offset + rodata_size;
  const uint64_t strtab_offset = include_metadata ? note_offset + metadata_note.size()
                                                  : rodata_offset + rodata_size;
  const uint64_t symtab_offset = align_up_for_test(strtab_offset + strtab.size(), 8);
  constexpr size_t sym_count = 2;
  const uint64_t shstrtab_offset = symtab_offset + sym_count * sizeof(rocjitsu::Elf64_Sym);
  const uint64_t shoff = align_up_for_test(shstrtab_offset + shstrtab.size(), 8);
  const uint16_t symtab_index = include_metadata ? 4 : 3;
  const uint16_t strtab_index = include_metadata ? 5 : 4;
  const uint16_t shstrtab_index = include_metadata ? 6 : 5;
  const uint16_t section_count = include_metadata ? 7 : 6;
  const uint16_t phdr_count = include_metadata ? 3 : 2;

  std::vector<uint8_t> image(shoff + section_count * sizeof(rocjitsu::Elf64_Shdr), 0);

  rocjitsu::Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, rocjitsu::EI_MAGIC, rocjitsu::EI_MAGIC_SIZE);
  ehdr.e_ident[rocjitsu::EI_CLASS] = rocjitsu::ELFCLASS64;
  ehdr.e_ident[rocjitsu::EI_OSABI] = rocjitsu::ELFOSABI_AMDGPU_HSA;
  ehdr.e_type = rocjitsu::ET_DYN;
  ehdr.e_machine = rocjitsu::EM_AMDGPU;
  ehdr.e_version = 1;
  ehdr.e_phoff = sizeof(rocjitsu::Elf64_Ehdr);
  ehdr.e_shoff = shoff;
  ehdr.e_flags = elf_mach;
  ehdr.e_ehsize = sizeof(rocjitsu::Elf64_Ehdr);
  ehdr.e_phentsize = sizeof(rocjitsu::Elf64_Phdr);
  ehdr.e_phnum = phdr_count;
  ehdr.e_shentsize = sizeof(rocjitsu::Elf64_Shdr);
  ehdr.e_shnum = section_count;
  ehdr.e_shstrndx = shstrtab_index;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  std::vector<rocjitsu::Elf64_Phdr> phdrs(phdr_count);
  phdrs[0].p_type = rocjitsu::PT_LOAD;
  phdrs[0].p_flags = 0x5;
  phdrs[0].p_offset = text_offset;
  phdrs[0].p_vaddr = text_vaddr;
  phdrs[0].p_paddr = text_vaddr;
  phdrs[0].p_filesz = text_size;
  phdrs[0].p_memsz = text_size;
  phdrs[0].p_align = load_align;
  phdrs[1].p_type = rocjitsu::PT_LOAD;
  phdrs[1].p_flags = 0x4;
  phdrs[1].p_offset = rodata_offset;
  phdrs[1].p_vaddr = rodata_vaddr;
  phdrs[1].p_paddr = rodata_vaddr;
  phdrs[1].p_filesz = rodata_size;
  phdrs[1].p_memsz = rodata_size;
  phdrs[1].p_align = load_align;
  if (include_metadata) {
    phdrs[2].p_type = rocjitsu::PT_NOTE;
    phdrs[2].p_flags = 0x4;
    phdrs[2].p_offset = note_offset;
    phdrs[2].p_vaddr = note_offset;
    phdrs[2].p_paddr = note_offset;
    phdrs[2].p_filesz = metadata_note.size();
    phdrs[2].p_memsz = metadata_note.size();
    phdrs[2].p_align = 4;
  }
  std::memcpy(image.data() + ehdr.e_phoff, phdrs.data(), phdrs.size() * sizeof(phdrs[0]));

  const std::array<uint32_t, 2> text_words = {0xbf800000u, 0xbf800000u};
  std::memcpy(image.data() + text_offset, text_words.data(), text_size);

  KD kd{};
  kd.kernel_code_entry_byte_offset =
      static_cast<int64_t>(text_vaddr) - static_cast<int64_t>(rodata_vaddr);
  std::memcpy(image.data() + rodata_offset, &kd, sizeof(kd));
  if (include_metadata)
    std::memcpy(image.data() + note_offset, metadata_note.data(), metadata_note.size());
  std::memcpy(image.data() + strtab_offset, strtab.data(), strtab.size());

  std::array<rocjitsu::Elf64_Sym, sym_count> syms{};
  syms[1].st_name = kd_symbol_name;
  syms[1].st_info =
      rocjitsu::elf_symbol_info(rocjitsu::kElfSymbolBindGlobal,
                                rocjitsu::kElfSymbolTypeObject);
  syms[1].st_shndx = 2;
  syms[1].st_value = rodata_vaddr;
  syms[1].st_size = sizeof(KD);
  std::memcpy(image.data() + symtab_offset, syms.data(), syms.size() * sizeof(syms[0]));
  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  std::vector<rocjitsu::Elf64_Shdr> shdrs(section_count);
  shdrs[1].sh_name = text_name;
  shdrs[1].sh_type = rocjitsu::SHT_PROGBITS;
  shdrs[1].sh_flags = rocjitsu::SHF_ALLOC | rocjitsu::SHF_EXECINSTR;
  shdrs[1].sh_addr = text_vaddr;
  shdrs[1].sh_offset = text_offset;
  shdrs[1].sh_size = text_size;
  shdrs[1].sh_addralign = sizeof(uint32_t);
  shdrs[2].sh_name = rodata_name;
  shdrs[2].sh_type = rocjitsu::SHT_PROGBITS;
  shdrs[2].sh_flags = rocjitsu::SHF_ALLOC;
  shdrs[2].sh_addr = rodata_vaddr;
  shdrs[2].sh_offset = rodata_offset;
  shdrs[2].sh_size = rodata_size;
  shdrs[2].sh_addralign = 64;
  if (include_metadata) {
    shdrs[3].sh_name = note_name;
    shdrs[3].sh_type = rocjitsu::SHT_NOTE;
    shdrs[3].sh_flags = rocjitsu::SHF_ALLOC;
    shdrs[3].sh_offset = note_offset;
    shdrs[3].sh_size = metadata_note.size();
    shdrs[3].sh_addralign = 4;
  }
  shdrs[symtab_index].sh_name = symtab_name;
  shdrs[symtab_index].sh_type = rocjitsu::SHT_SYMTAB;
  shdrs[symtab_index].sh_offset = symtab_offset;
  shdrs[symtab_index].sh_size = syms.size() * sizeof(syms[0]);
  shdrs[symtab_index].sh_link = strtab_index;
  shdrs[symtab_index].sh_info = 1;
  shdrs[symtab_index].sh_addralign = 8;
  shdrs[symtab_index].sh_entsize = sizeof(rocjitsu::Elf64_Sym);
  shdrs[strtab_index].sh_name = strtab_name;
  shdrs[strtab_index].sh_type = rocjitsu::SHT_STRTAB;
  shdrs[strtab_index].sh_offset = strtab_offset;
  shdrs[strtab_index].sh_size = strtab.size();
  shdrs[strtab_index].sh_addralign = 1;
  shdrs[shstrtab_index].sh_name = shstrtab_name;
  shdrs[shstrtab_index].sh_type = rocjitsu::SHT_STRTAB;
  shdrs[shstrtab_index].sh_offset = shstrtab_offset;
  shdrs[shstrtab_index].sh_size = shstrtab.size();
  shdrs[shstrtab_index].sh_addralign = 1;
  std::memcpy(image.data() + shoff, shdrs.data(), shdrs.size() * sizeof(shdrs[0]));
  return image;
}

std::vector<uint8_t> make_minimal_elf_without_text() {
  constexpr uint64_t rodata_offset = 0x100;
  constexpr uint64_t rodata_size = 4;

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t rodata_name = add_elf_name(shstrtab, ".rodata");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");
  const uint64_t shstrtab_offset = rodata_offset + rodata_size;
  const uint64_t shoff = align_up_for_test(shstrtab_offset + shstrtab.size(), 8);
  constexpr uint16_t section_count = 3;

  std::vector<uint8_t> image(shoff + section_count * sizeof(rocjitsu::Elf64_Shdr), 0);

  rocjitsu::Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, rocjitsu::EI_MAGIC, rocjitsu::EI_MAGIC_SIZE);
  ehdr.e_ident[rocjitsu::EI_CLASS] = rocjitsu::ELFCLASS64;
  ehdr.e_ident[rocjitsu::EI_OSABI] = rocjitsu::ELFOSABI_AMDGPU_HSA;
  ehdr.e_type = rocjitsu::ET_REL;
  ehdr.e_machine = rocjitsu::EM_AMDGPU;
  ehdr.e_version = 1;
  ehdr.e_shoff = shoff;
  ehdr.e_flags = rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX950;
  ehdr.e_ehsize = sizeof(rocjitsu::Elf64_Ehdr);
  ehdr.e_shentsize = sizeof(rocjitsu::Elf64_Shdr);
  ehdr.e_shnum = section_count;
  ehdr.e_shstrndx = 2;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  const uint32_t rodata_word = 0xa5a55a5au;
  std::memcpy(image.data() + rodata_offset, &rodata_word, sizeof(rodata_word));
  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  std::array<rocjitsu::Elf64_Shdr, section_count> shdrs{};
  shdrs[1].sh_name = rodata_name;
  shdrs[1].sh_type = rocjitsu::SHT_PROGBITS;
  shdrs[1].sh_flags = rocjitsu::SHF_ALLOC;
  shdrs[1].sh_offset = rodata_offset;
  shdrs[1].sh_size = rodata_size;
  shdrs[1].sh_addralign = sizeof(uint32_t);
  shdrs[2].sh_name = shstrtab_name;
  shdrs[2].sh_type = rocjitsu::SHT_STRTAB;
  shdrs[2].sh_offset = shstrtab_offset;
  shdrs[2].sh_size = shstrtab.size();
  shdrs[2].sh_addralign = 1;
  std::memcpy(image.data() + shoff, shdrs.data(), shdrs.size() * sizeof(shdrs[0]));
  return image;
}

rocjitsu::AmdGpuCodeObject make_code_object(const std::vector<uint8_t> &image) {
  rocjitsu::AmdGpuCodeObject co(image.data(), image.size());
  check(co.is_valid(), "synthetic AMDGPU ELF should parse");
  return co;
}

void check_descriptor_resource_plan_failure_preserves_image() {
  using namespace rocjitsu::fuzzer::afl_dbi;

  const std::vector<uint8_t> image = make_minimal_descriptor_elf();
  rocjitsu::AmdGpuCodeObject co = make_code_object(image);
  rocjitsu::CodeObjectPatcher patcher(co);
  const std::vector<uint8_t> original = snapshot(patcher);

  KernelSite invalid_site;
  invalid_site.name = "kernel";
  invalid_site.descriptor_file_offset = original.size();
  invalid_site.wave32 = true;

  ProbeRegisterRequirements requirements;
  requirements.sgprs = 128;
  requirements.vgprs = 128;
  check(!plan_kernel_descriptor_resources(patcher.image_bytes(), invalid_site, requirements),
        "descriptor planning should reject out-of-range descriptors");
  check(!patch_kernel_descriptor_for_requirements(patcher, invalid_site, requirements),
        "descriptor patch helper should reject out-of-range descriptors");
  check(patcher.cave_body().empty(), "descriptor planning failure should not append cave bytes");
  check_image_unchanged(patcher, original, "descriptor planning failure mutated image bytes");
}

void check_descriptor_resource_private_segment_plan_and_patch() {
  using namespace rocjitsu::fuzzer::afl_dbi;

  const std::vector<uint8_t> image = make_minimal_descriptor_elf(/*include_metadata=*/true);
  rocjitsu::AmdGpuCodeObject co = make_code_object(image);
  rocjitsu::CodeObjectPatcher patcher(co);
  const KernelSite site = single_kernel_site_for_test(patcher);

  KD baseline = read_kernel_descriptor_for_test(patcher.image_bytes(),
                                               site.descriptor_file_offset);
  baseline.private_segment_fixed_size = 16;
  AMDHSA_BITS_SET(baseline.compute_pgm_rsrc2,
                  kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT, 0);
  const auto *baseline_bytes = reinterpret_cast<const uint8_t *>(&baseline);
  check(patcher.patch_kernel_descriptor(site.descriptor_file_offset,
                                        {baseline_bytes, sizeof(baseline)}),
        "fixture descriptor setup should patch");
  const std::vector<uint8_t> original = snapshot(patcher);

  ProbeRegisterRequirements requirements;
  requirements.sgprs = 8;
  requirements.vgprs = 8;
  requirements.private_segment_bytes = 64;
  std::optional<KernelDescriptorResourceSummary> summary =
      plan_kernel_descriptor_resources(patcher.image_bytes(), site, requirements);
  check(summary.has_value(), "private segment descriptor planning should succeed");
  check(summary->old_private_segment_fixed_size == 16,
        "descriptor plan should report old private segment size");
  check(summary->patched_private_segment_fixed_size == 64,
        "descriptor plan should grow private segment size");
  check(summary->spill_bytes == 48, "descriptor plan should report added spill bytes");
  check(!summary->old_private_segment_enabled,
        "descriptor plan should report old private segment enable bit");
  check(summary->patched_private_segment_enabled,
        "descriptor plan should enable private segment when scratch grows");
  check(summary->resource_fields_changed,
        "descriptor plan should report private segment resource changes");
  check_image_unchanged(patcher, original,
                        "private segment descriptor planning mutated image bytes");
  const char *metadata_failure = nullptr;
  std::optional<AmdgpuMetadataPrivateSegmentPatch> metadata_patch =
      plan_amdgpu_metadata_private_segment_patch(
          patcher.image_bytes(), site.name, summary->patched_private_segment_fixed_size,
          &metadata_failure);
  check(metadata_patch.has_value(),
        "private segment metadata planning should find kernel metadata");
  check(metadata_failure == nullptr,
        "private segment metadata planning should not report failure");
  check(metadata_patch->kind ==
            AmdgpuMetadataPrivateSegmentPatch::Kind::InPlaceBytes,
        "small private segment metadata growth should use an in-place patch");
  check(metadata_patch->size == 1,
        "metadata fixture private segment should be in-place patchable");

  std::optional<AmdgpuMetadataPrivateSegmentPatch::Kind> applied_metadata_patch;
  check(patch_kernel_descriptor_resources(patcher, *summary, nullptr,
                                          &applied_metadata_patch),
        "private segment descriptor patch should succeed");
  check(applied_metadata_patch.has_value() &&
            *applied_metadata_patch ==
                AmdgpuMetadataPrivateSegmentPatch::Kind::InPlaceBytes,
        "private segment descriptor patch should report in-place metadata update");
  const KD patched = read_kernel_descriptor_for_test(patcher.image_bytes(),
                                                    site.descriptor_file_offset);
  check(patched.private_segment_fixed_size == 64,
        "descriptor patch should grow private segment fixed size");
  check(AMDHSA_BITS_GET(patched.compute_pgm_rsrc2,
                        kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT) == 1,
        "descriptor patch should enable private segment");
  check(patcher.image_bytes()[metadata_patch->file_offset] == 64,
        "descriptor patch should update loader-visible private segment metadata");
}

void check_descriptor_resource_private_segment_rebuilds_metadata_note() {
  using namespace rocjitsu::fuzzer::afl_dbi;

  const std::vector<uint8_t> image = make_minimal_descriptor_elf(/*include_metadata=*/true);
  rocjitsu::AmdGpuCodeObject co = make_code_object(image);
  rocjitsu::CodeObjectPatcher patcher(co);
  const KernelSite site = single_kernel_site_for_test(patcher);

  KD baseline = read_kernel_descriptor_for_test(patcher.image_bytes(),
                                               site.descriptor_file_offset);
  baseline.private_segment_fixed_size = 16;
  const auto *baseline_bytes = reinterpret_cast<const uint8_t *>(&baseline);
  check(patcher.patch_kernel_descriptor(site.descriptor_file_offset,
                                        {baseline_bytes, sizeof(baseline)}),
        "fixture descriptor setup should patch");
  const std::vector<uint8_t> original = snapshot(patcher);

  ProbeRegisterRequirements requirements;
  requirements.sgprs = 8;
  requirements.vgprs = 8;
  requirements.private_segment_bytes = 256;
  std::optional<KernelDescriptorResourceSummary> summary =
      plan_kernel_descriptor_resources(patcher.image_bytes(), site, requirements);
  check(summary.has_value(), "large private segment descriptor planning should succeed");
  check(summary->patched_private_segment_fixed_size == 256,
        "descriptor plan should request a value that does not fit fixint metadata");

  const char *metadata_failure = nullptr;
  std::optional<AmdgpuMetadataPrivateSegmentPatch> metadata_patch =
      plan_amdgpu_metadata_private_segment_patch(
          patcher.image_bytes(), site.name, summary->patched_private_segment_fixed_size,
          &metadata_failure);
  check(metadata_patch.has_value(),
        "large private segment metadata planning should rebuild the note");
  check(metadata_failure == nullptr,
        "large private segment metadata planning should not report failure");
  check(metadata_patch->kind ==
            AmdgpuMetadataPrivateSegmentPatch::Kind::RebuiltNoteSection,
        "large private segment metadata growth should rebuild the note section");
  check(!metadata_patch->note_section_bytes.empty(),
        "rebuilt metadata note section should contain replacement bytes");
  check_image_unchanged(patcher, original,
                        "large private segment metadata planning mutated image bytes");

  std::optional<AmdgpuMetadataPrivateSegmentPatch::Kind> applied_metadata_patch;
  check(patch_kernel_descriptor_resources(patcher, *summary, nullptr,
                                          &applied_metadata_patch),
        "large private segment descriptor patch should succeed");
  check(applied_metadata_patch.has_value() &&
            *applied_metadata_patch ==
                AmdgpuMetadataPrivateSegmentPatch::Kind::RebuiltNoteSection,
        "large private segment descriptor patch should report rebuilt metadata update");
  check(patcher.image_bytes().size() > original.size(),
        "metadata note rebuild should append replacement note bytes");
  check(has_appended_pt_note(patcher.image_bytes(), original.size()),
        "metadata note rebuild should retarget PT_NOTE to appended bytes");

  const KD patched = read_kernel_descriptor_for_test(patcher.image_bytes(),
                                                    site.descriptor_file_offset);
  check(patched.private_segment_fixed_size == 256,
        "descriptor patch should grow private segment fixed size past fixint range");

  const char *reparse_failure = nullptr;
  std::optional<AmdgpuMetadataPrivateSegmentPatch> followup_patch =
      plan_amdgpu_metadata_private_segment_patch(patcher.image_bytes(), site.name, 512,
                                                 &reparse_failure);
  check(followup_patch.has_value(),
        "rebuilt metadata note should remain parseable for later planning");
  check(reparse_failure == nullptr,
        "rebuilt metadata note reparse should not report failure");
  check(followup_patch->kind == AmdgpuMetadataPrivateSegmentPatch::Kind::InPlaceBytes,
        "rebuilt uint16 metadata should support later in-place private size updates");
  check(followup_patch->size == 2,
        "rebuilt uint16 metadata should patch the payload bytes in place");
}

void check_private_segment_growth_requires_metadata() {
  using namespace rocjitsu::fuzzer::afl_dbi;

  const std::vector<uint8_t> image = make_minimal_descriptor_elf();
  rocjitsu::AmdGpuCodeObject co = make_code_object(image);
  rocjitsu::CodeObjectPatcher patcher(co);
  const KernelSite site = single_kernel_site_for_test(patcher);

  ProbeRegisterRequirements requirements;
  requirements.sgprs = 8;
  requirements.vgprs = 8;
  requirements.private_segment_bytes = 64;
  std::optional<KernelDescriptorResourceSummary> summary =
      plan_kernel_descriptor_resources(patcher.image_bytes(), site, requirements);
  check(summary.has_value(), "private segment descriptor planning should succeed");

  const std::vector<uint8_t> original = snapshot(patcher);
  const char *failure = nullptr;
  std::optional<AmdgpuMetadataPrivateSegmentPatch::Kind> applied_metadata_patch;
  check(!patch_kernel_descriptor_resources(patcher, *summary, &failure,
                                           &applied_metadata_patch),
        "private segment growth should fail without loader-visible metadata");
  check(!applied_metadata_patch.has_value(),
        "failed private segment growth should not report an applied metadata patch");
  check(failure != nullptr &&
            std::string_view(failure) == "AMDGPU metadata note is missing",
        "private segment growth failure should explain missing metadata");
  check_image_unchanged(patcher, original,
                        "metadata planning failure should not mutate descriptor bytes");
}

void check_gfx1201_sgpr_metadata_overrides_zero_descriptor_count() {
  using namespace rocjitsu::fuzzer::afl_dbi;

  const std::vector<uint8_t> image = make_minimal_descriptor_elf(
      /*include_metadata=*/true, rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1201,
      /*metadata_sgpr_count=*/46);
  rocjitsu::AmdGpuCodeObject co = make_code_object(image);
  rocjitsu::CodeObjectPatcher patcher(co);
  const KernelSite site = single_kernel_site_for_test(patcher);

  check(site.elf_mach == rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1201,
        "gfx1201 fixture should preserve ELF machine ID");
  check(!site.descriptor_sgpr_count_effective,
        "gfx1201 should not treat descriptor SGPR granules as authoritative");
  check(site.descriptor_sgpr_count == 8,
        "zero descriptor SGPR granules decode to the legacy minimum");
  check(site.has_metadata_sgpr_count && site.metadata_sgpr_count == 46,
        "gfx1201 fixture should expose metadata SGPR count");
  check(site.allocated_sgpr_count == 46,
        "gfx1201 planning should use metadata SGPR count");
  check(site.fresh_sgpr_growth_supported,
        "gfx1201 fresh SGPR growth should use metadata-backed patching");

  ProbeRegisterRequirements requirements;
  requirements.sgprs = 44;
  requirements.vgprs = 4;
  std::optional<KernelDescriptorResourceSummary> summary =
      plan_kernel_descriptor_resources(patcher.image_bytes(), site, requirements);
  check(summary.has_value(),
        "requirements within metadata SGPR count should not need descriptor SGPR growth");
  check(summary->has_metadata_sgpr_count && summary->metadata_sgpr_count == 46,
        "descriptor summary should report metadata SGPR count");
  check(!summary->descriptor_sgpr_count_effective,
        "descriptor summary should report ignored SGPR descriptor field");
  check(summary->old_sgpr_count == 46 && summary->patched_sgpr_count == 46,
        "descriptor summary should size SGPRs from metadata without fake growth");
  check(summary->old_sgpr_granulated == 0 && summary->patched_sgpr_granulated == 0,
        "descriptor summary should not patch ignored gfx1201 SGPR granules");
  check(!summary->resource_fields_changed,
        "metadata-sized SGPR requirements should not report descriptor changes");

  const char *failure = nullptr;
  check(patch_kernel_descriptor_resources(patcher, *summary, &failure),
        "no-growth gfx1201 descriptor patch should succeed");
  check(failure == nullptr, "no-growth gfx1201 descriptor patch should not fail");
  const KD patched = read_kernel_descriptor_for_test(patcher.image_bytes(),
                                                    site.descriptor_file_offset);
  check(AMDHSA_BITS_GET(
            patched.compute_pgm_rsrc1,
            kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT) == 0,
        "gfx1201 descriptor patch should leave ignored SGPR granules unchanged");

  requirements.sgprs = 48;
  summary = plan_kernel_descriptor_resources(patcher.image_bytes(), site, requirements);
  check(summary.has_value(),
        "fresh SGPR growth past metadata count should plan metadata patch on gfx1201");
  check(summary->old_sgpr_count == 46 && summary->patched_sgpr_count == 48,
        "descriptor summary should report metadata-backed SGPR growth");
  check(summary->old_sgpr_granulated == 0 && summary->patched_sgpr_granulated == 0,
        "metadata-backed SGPR growth should leave ignored descriptor granules unchanged");
  check(summary->resource_fields_changed,
        "metadata-backed SGPR growth should report resource changes");
  std::optional<AmdgpuMetadataPrivateSegmentPatch::Kind> sgpr_metadata_patch;
  check(patch_kernel_descriptor_resources(patcher, *summary, &failure,
                                          /*applied_private_segment_metadata_patch=*/nullptr,
                                          &sgpr_metadata_patch),
        "metadata-backed SGPR growth patch should succeed on gfx1201");
  check(sgpr_metadata_patch.has_value() &&
            *sgpr_metadata_patch == AmdgpuMetadataPrivateSegmentPatch::Kind::InPlaceBytes,
        "metadata-backed SGPR growth should patch fixint metadata in place");
  const KernelSite grown_site = single_kernel_site_for_test(patcher);
  check(grown_site.has_metadata_sgpr_count && grown_site.metadata_sgpr_count == 48,
        "patched gfx1201 metadata should expose grown SGPR count");
  check(grown_site.allocated_sgpr_count == 48,
        "patched gfx1201 planning should use grown metadata SGPR count");
}

void check_entry_prologue_plan_failure_preserves_image() {
  const std::vector<uint8_t> image = make_minimal_descriptor_elf();
  rocjitsu::AmdGpuCodeObject co = make_code_object(image);
  rocjitsu::CodeObjectPatcher patcher(co);
  const std::vector<uint8_t> original = snapshot(patcher);

  patcher.set_cave_start(/*offset=*/200000);
  const std::array<uint32_t, 1> prologue = {
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4)};
  const auto prologue_entry =
      patcher.append_kernel_entry_prologue(/*entry_text_offset=*/0, prologue,
                                           ROCJITSU_CODE_ARCH_RDNA4);
  check(!prologue_entry, "out-of-range entry prologue branch should fail planning");
  check(patcher.cave_body().empty(), "entry planning failure should not append cave bytes");
  check_image_unchanged(patcher, original, "entry planning failure mutated image bytes");
}

void check_edge_trampoline_plan_failure_preserves_image() {
  using namespace rocjitsu::fuzzer::afl_dbi;

  const std::vector<uint8_t> image = make_minimal_descriptor_elf();
  rocjitsu::AmdGpuCodeObject co = make_code_object(image);
  rocjitsu::CodeObjectPatcher patcher(co);
  const std::vector<uint8_t> original_image = snapshot(patcher);
  const std::vector<uint8_t> original_text(patcher.text_bytes().begin(),
                                           patcher.text_bytes().end());

  EdgeSite site;
  site.kind = EdgePatchKind::BranchTerminator;
  site.kernel_name = "kernel";
  site.patch_text_offset = 0;
  site.return_text_offset = 0;
  site.first_inst_size = sizeof(uint32_t);
  site.bb_id = 0x12345678u;
  site.slot_policy = EdgeSlotPolicyKind::FixedCounter;
  site.fixed_slot = 1;
  site.self_contained_probe = true;

  LocalTextCaveAllocator local_caves(original_text);
  const char *failure_reason = nullptr;
  auto planned = plan_edge_trampoline(site, original_text, /*appended_cave_body_size=*/0,
                                      /*cave_start=*/200000, local_caves,
                                      ROCJITSU_CODE_ARCH_RDNA4,
                                      /*state_pointer=*/0x1234567887654321ull,
                                      &failure_reason);
  check(!planned, "out-of-range edge trampoline should fail planning");
  check(failure_reason != nullptr, "edge trampoline failure should report a reason");
  check(original_text == std::vector<uint8_t>(patcher.text_bytes().begin(),
                                             patcher.text_bytes().end()),
        "edge trampoline planning failure mutated text bytes");
  check(patcher.cave_body().empty(),
        "edge trampoline planning failure should not append cave bytes");
  check_image_unchanged(patcher, original_image,
                        "edge trampoline planning failure mutated image bytes");
}

void check_emit_failure_preserves_image() {
  const std::vector<uint8_t> image = make_minimal_elf_without_text();
  rocjitsu::AmdGpuCodeObject co = make_code_object(image);
  check(co.text_sections().empty(), "emit failure fixture should not contain .text");

  rocjitsu::CodeObjectPatcher patcher(co);
  const std::vector<uint8_t> original = snapshot(patcher);
  const std::array<uint32_t, 1> cave_words = {
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4)};
  patcher.append_cave_body(cave_words);

  check(!patcher.append_cave_section(".rj_afl_entry"),
        "cave materialization without a .text section should fail");
  check(patcher.emit() == original, "emit failure exposed a partially mutated image");
}

} // namespace

int main() {
  check_descriptor_resource_plan_failure_preserves_image();
  check_descriptor_resource_private_segment_plan_and_patch();
  check_descriptor_resource_private_segment_rebuilds_metadata_note();
  check_private_segment_growth_requires_metadata();
  check_gfx1201_sgpr_metadata_overrides_zero_descriptor_count();
  check_entry_prologue_plan_failure_preserves_image();
  check_edge_trampoline_plan_failure_preserves_image();
  check_emit_failure_preserves_image();
  return 0;
}
