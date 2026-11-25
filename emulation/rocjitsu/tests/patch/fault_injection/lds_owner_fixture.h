// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "lds_fixture.h"
#include "rocjitsu/isa/target_registry.h"

namespace rocjitsu::test {

inline std::vector<uint8_t>
make_shared_lds_elf(bool indirect, size_t trailing_nops = 0, bool cdna = false,
                    uint32_t cdna_machine = EF_AMDGPU_MACH_AMDGCN_GFX950,
                    uint32_t rdna_machine = EF_AMDGPU_MACH_AMDGCN_GFX1201) {
  const IsaFamily arch = !cdna                                          ? IsaFamily::Rdna4
                         : cdna_machine == EF_AMDGPU_MACH_AMDGCN_GFX942 ? IsaFamily::Cdna3
                                                                        : IsaFamily::Cdna4;
  const uint32_t nop = build_s_nop(0, arch).value();
  const uint32_t end = build_s_endpgm(arch).value();
  std::vector<uint32_t> words(128, nop);
  for (uint32_t entry : {0u, 256u}) {
    const size_t start = entry / 4;
    if (!indirect) {
      words[start] = build_s_call_b64(30, (512 - entry - 4) / 4, arch).value();
      words[start + 1] = end;
    } else {
      words[start] = build_s_getpc_b64(8, arch).value();
      words[start + 1] = build_s_add_u32(8, 8, 255, arch).value();
      words[start + 2] = 512 - entry - 4;
      words[start + 3] = build_s_addc_u32(9, 9, 128, arch).value();
      words[start + 4] = build_s_swappc_b64(30, 8, arch).value();
      words[start + 5] = end;
    }
  }
  words.insert(words.end(), {cdna ? 0xd81a0004u : 0xd8340004u, 0x00000302u,
                             build_s_setpc_b64(30, arch).value()});
  words.insert(words.end(), trailing_nops, nop);
  // gfx942 needs 40 requested SGPRs (48 physical) to keep s[30:31] unaliased.
  const uint32_t granulated_sgprs = arch == IsaFamily::Cdna3 ? 4u : 3u;
  std::vector<uint8_t> image =
      make_amdgpu_two_kernel_elf(words, 0, granulated_sgprs, cdna ? cdna_machine : rdna_machine);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  const size_t offset =
      0x100 + words.size() * sizeof(uint32_t) + sizeof(amdhsa::kernel_descriptor_t);
  amdhsa::kernel_descriptor_t sibling = {};
  std::memcpy(&sibling, image.data() + offset, sizeof(sibling));
  sibling.kernel_code_entry_byte_offset += 256;
  std::memcpy(image.data() + offset, &sibling, sizeof(sibling));
  return image;
}

inline std::vector<uint8_t> make_shared_lds_with_unrelated_owner() {
  std::vector<uint8_t> image = make_shared_lds_elf(false);
  const uint32_t end = build_s_endpgm(IsaFamily::Rdna4).value();
  std::memcpy(image.data() + 0x100 + 128, &end, sizeof(end));
  Elf64_Ehdr header = {};
  std::memcpy(&header, image.data(), sizeof(header));
  std::vector<Elf64_Shdr> sections(header.e_shnum);
  std::memcpy(sections.data(), image.data() + header.e_shoff, sections.size() * sizeof(Elf64_Shdr));
  constexpr uint64_t descriptor_address = 0x900000;
  amdhsa::kernel_descriptor_t descriptor = {};
  descriptor.kernel_code_entry_byte_offset =
      int64_t{0x1100 + 128} - static_cast<int64_t>(descriptor_address);
  const size_t descriptor_offset = image.size();
  image.resize(image.size() + sizeof(descriptor));
  std::memcpy(image.data() + descriptor_offset, &descriptor, sizeof(descriptor));
  const Elf64_Phdr segment = {.p_type = PT_LOAD,
                              .p_flags = 4,
                              .p_offset = descriptor_offset,
                              .p_vaddr = descriptor_address,
                              .p_paddr = descriptor_address,
                              .p_filesz = sizeof(descriptor),
                              .p_memsz = sizeof(descriptor),
                              .p_align = 1};
  const size_t segment_offset = header.e_phoff + header.e_phnum * header.e_phentsize;
  std::memcpy(image.data() + segment_offset, &segment, sizeof(segment));
  ++header.e_phnum;
  const Elf64_Shdr descriptor_section = {.sh_name = 0,
                                         .sh_type = SHT_PROGBITS,
                                         .sh_flags = SHF_ALLOC,
                                         .sh_addr = descriptor_address,
                                         .sh_offset = descriptor_offset,
                                         .sh_size = sizeof(descriptor),
                                         .sh_link = 0,
                                         .sh_info = 0,
                                         .sh_addralign = 1,
                                         .sh_entsize = 0};
  const uint16_t descriptor_index = static_cast<uint16_t>(sections.size());
  sections.push_back(descriptor_section);
  Elf64_Shdr &strings = sections[4];
  std::vector<uint8_t> names(image.begin() + strings.sh_offset,
                             image.begin() + strings.sh_offset + strings.sh_size);
  const uint32_t name = static_cast<uint32_t>(names.size());
  constexpr char owner_name[] = "unrelated.kd";
  names.insert(names.end(), owner_name, owner_name + sizeof(owner_name));
  strings.sh_offset = image.size();
  strings.sh_size = names.size();
  image.insert(image.end(), names.begin(), names.end());
  Elf64_Shdr &symbols = sections[3];
  std::vector<uint8_t> table(image.begin() + symbols.sh_offset,
                             image.begin() + symbols.sh_offset + symbols.sh_size);
  const Elf64_Sym symbol = {.st_name = name,
                            .st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject),
                            .st_other = 0,
                            .st_shndx = descriptor_index,
                            .st_value = descriptor_address,
                            .st_size = sizeof(descriptor)};
  const size_t symbol_offset = table.size();
  table.resize(table.size() + sizeof(symbol));
  std::memcpy(table.data() + symbol_offset, &symbol, sizeof(symbol));
  image.resize((image.size() + 7) & ~size_t{7});
  symbols.sh_offset = image.size();
  symbols.sh_size = table.size();
  image.insert(image.end(), table.begin(), table.end());
  image.resize((image.size() + 7) & ~size_t{7});
  header.e_shoff = image.size();
  header.e_shnum = static_cast<uint16_t>(sections.size());
  image.resize(image.size() + sections.size() * sizeof(Elf64_Shdr));
  std::memcpy(image.data() + header.e_shoff, sections.data(), sections.size() * sizeof(Elf64_Shdr));
  std::memcpy(image.data(), &header, sizeof(header));
  return image;
}

// Verify the existing trampoline representation, including the emitted guest,
// continuation and independent whole-image/descriptor oracle.
inline bool lds_owner_output_matches(const FaultMutationContext &source, uint64_t offset,
                                     uint32_t replacement_vgpr,
                                     std::span<const uint8_t> candidate) {
  const Section &original = *source.source().text_sections().front();
  if (offset > original.size() || original.size() - offset < 8 || replacement_vgpr > 255)
    return false;
  const auto *bytes = reinterpret_cast<const uint8_t *>(original.data());
  std::vector<uint8_t> expected_bytes(bytes + offset, bytes + offset + 8);
  std::vector<uint8_t> replacement = expected_bytes;
  replacement[4] = static_cast<uint8_t>(replacement_vgpr);
  const std::array<ExpectedReplacement, 1> edits = {ExpectedReplacement{
      .offset = offset, .expected = expected_bytes, .replacement = replacement}};
  const rj_code_target_id_t target = source.source().target_id();
  const IsaTargetDescriptor *binding = default_isa_target_registry().find(target);
  if (!binding)
    return false;
  const std::optional<std::vector<uint8_t>> expected =
      expected_replacement_image(source.source(), edits, binding->family);
  if (!expected || !std::ranges::equal(*expected, candidate))
    return false;
  const AmdGpuCodeObject mutant(candidate.data(), candidate.size());
  if (!mutant.is_valid() || mutant.text_sections().size() != 1 || mutant.target_id() != target)
    return false;
  const Section &text = *mutant.text_sections().front();
  const std::optional<uint64_t> guest = lds_emitted_guest(text, offset, target);
  if (!guest || *guest > text.size() || text.size() - *guest < 12)
    return false;
  std::unique_ptr<Decoder> decoder = Decoder::create(default_isa_target_registry(), target);
  if (!decoder || !branch_reaches(*decoder, text, *guest + 8, offset + 8))
    return false;
  std::array<uint32_t, 2> words = {};
  std::memcpy(words.data(), text.data() + *guest, sizeof(words));
  const DecodeResult decoded = decoder->decode_window(words, *guest);
  return decoded.succeeded() && decoded.value()->size() == 8 && decoded.value()->src_operand(0) &&
         decoded.value()->src_operand(0)->encoding_value() == static_cast<int>(replacement_vgpr);
}

} // namespace rocjitsu::test
