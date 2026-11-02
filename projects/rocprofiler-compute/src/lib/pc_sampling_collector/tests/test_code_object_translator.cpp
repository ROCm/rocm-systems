// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_code_object_translator.h"

#include <elf.h>
#include <hsa/amd_hsa_elf.h>

#include <array>
#include <cstring>

std::vector<char> TestCodeObjectTranslator::make_code_object(const std::vector<uint32_t>& instructions)
{
    constexpr size_t kStringOffset        = 0x210;
    constexpr size_t kSymbolOffset        = 0x240;
    constexpr size_t kSectionHeaderOffset = 0x300;
    constexpr char   kSymbolName[]        = "\0kernel";
    const size_t     text_size            = instructions.size() * kInstructionSize;

    Elf64_Ehdr header{};
    std::memcpy(header.e_ident, ELFMAG, SELFMAG);
    header.e_ident[EI_CLASS]      = ELFCLASS64;
    header.e_ident[EI_DATA]       = ELFDATA2LSB;
    header.e_ident[EI_VERSION]    = EV_CURRENT;
    header.e_ident[EI_OSABI]      = ELF::ELFOSABI_AMDGPU_HSA;
    header.e_ident[EI_ABIVERSION] = ELF::ELFABIVERSION_AMDGPU_HSA_V4;
    header.e_type                 = ET_DYN;
    header.e_machine              = ELF::EM_AMDGPU;
    header.e_version              = EV_CURRENT;
    header.e_flags                = ELF::EF_AMDGPU_MACH_AMDGCN_GFX900;
    header.e_ehsize               = sizeof(header);
    header.e_phoff                = sizeof(header);
    header.e_phentsize            = sizeof(Elf64_Phdr);
    header.e_phnum                = 1;
    header.e_shoff                = kSectionHeaderOffset;
    header.e_shentsize            = sizeof(Elf64_Shdr);
    header.e_shnum                = 4;

    Elf64_Phdr segment{};
    segment.p_type   = PT_LOAD;
    segment.p_flags  = PF_R | PF_X;
    segment.p_offset = kTextFileOffset;
    segment.p_vaddr  = kTextVirtualAddress;
    segment.p_filesz = text_size;
    segment.p_memsz  = text_size;
    segment.p_align  = kInstructionSize;

    std::array<Elf64_Sym, 2> symbols{};
    symbols[1].st_name  = 1;
    symbols[1].st_info  = ELF64_ST_INFO(STB_GLOBAL, STT_FUNC);
    symbols[1].st_shndx = 1;
    symbols[1].st_value = kTextVirtualAddress;
    symbols[1].st_size  = text_size;

    std::array<Elf64_Shdr, 4> sections{};
    sections[1].sh_type      = SHT_PROGBITS;
    sections[1].sh_flags     = SHF_ALLOC | SHF_EXECINSTR;
    sections[1].sh_addr      = kTextVirtualAddress;
    sections[1].sh_offset    = kTextFileOffset;
    sections[1].sh_size      = text_size;
    sections[1].sh_addralign = kInstructionSize;
    sections[2].sh_type      = SHT_STRTAB;
    sections[2].sh_offset    = kStringOffset;
    sections[2].sh_size      = sizeof(kSymbolName);
    sections[2].sh_addralign = 1;
    sections[3].sh_type      = SHT_SYMTAB;
    sections[3].sh_offset    = kSymbolOffset;
    sections[3].sh_size      = sizeof(symbols);
    sections[3].sh_link      = 2;
    sections[3].sh_info      = 1;
    sections[3].sh_addralign = alignof(Elf64_Sym);
    sections[3].sh_entsize   = sizeof(Elf64_Sym);

    std::vector<char> binary(kSectionHeaderOffset + sizeof(sections));
    std::memcpy(binary.data(), &header, sizeof(header));
    std::memcpy(binary.data() + header.e_phoff, &segment, sizeof(segment));
    std::memcpy(binary.data() + kTextFileOffset, instructions.data(), text_size);
    std::memcpy(binary.data() + kStringOffset, kSymbolName, sizeof(kSymbolName));
    std::memcpy(binary.data() + kSymbolOffset, symbols.data(), sizeof(symbols));
    std::memcpy(binary.data() + kSectionHeaderOffset, sections.data(), sizeof(sections));
    return binary;
}

TEST_F(TestCodeObjectTranslator, ReusedLoadAddress_DecodesRequestedObjectWithinAndBeyondOldRange)
{
    constexpr size_t   kOldObjectId  = 5;
    constexpr size_t   kNewObjectId  = 39;
    constexpr uint32_t kMoveOne      = 0xbe800081;
    constexpr uint32_t kMoveTwo      = 0xbe800082;
    constexpr uint32_t kEndProgram   = 0xbf810000;
    constexpr uint64_t kFirstAddress = kLoadAddress + kTextVirtualAddress;
    const auto         old_binary    = make_code_object({kMoveOne});
    const auto         new_binary    = make_code_object({kMoveTwo, kEndProgram});

    m_translator.add_code_object(reinterpret_cast<uint64_t>(old_binary.data()),
                                 old_binary.size(),
                                 kOldObjectId,
                                 kLoadAddress,
                                 kTextVirtualAddress + kInstructionSize);
    m_translator.add_code_object(reinterpret_cast<uint64_t>(new_binary.data()),
                                 new_binary.size(),
                                 kNewObjectId,
                                 kLoadAddress,
                                 kTextVirtualAddress + 2 * kInstructionSize);

    const auto old_instruction = m_translator.get_instruction(kOldObjectId, kFirstAddress);
    EXPECT_EQ(old_instruction.name, "s_mov_b32 s0, 1");
    EXPECT_EQ(old_instruction.size, kInstructionSize);
    EXPECT_EQ(old_instruction.virtual_address, kFirstAddress);
    EXPECT_EQ(old_instruction.code_obj_offset, kTextFileOffset);

    const auto new_instruction = m_translator.get_instruction(kNewObjectId, kFirstAddress);
    EXPECT_EQ(new_instruction.name, "s_mov_b32 s0, 2");
    EXPECT_EQ(new_instruction.size, kInstructionSize);
    EXPECT_EQ(new_instruction.virtual_address, kFirstAddress);
    EXPECT_EQ(new_instruction.code_obj_offset, kTextFileOffset);

    const auto last_instruction = m_translator.get_instruction(kNewObjectId,
                                                               kFirstAddress + kInstructionSize);
    EXPECT_EQ(last_instruction.name, "s_endpgm");
    EXPECT_EQ(last_instruction.size, kInstructionSize);
    EXPECT_EQ(last_instruction.virtual_address, kFirstAddress + kInstructionSize);
    EXPECT_EQ(last_instruction.code_obj_offset, kTextFileOffset + kInstructionSize);

    EXPECT_EQ(m_translator.get_instruction(kOldObjectId, kFirstAddress).name, old_instruction.name);
}
