/**
 * Device Linker - Merges specialized kernel device objects into a single ELF.
 *
 * Two-pass design:
 *   Pass 1: Collect all sections, compute final sizes
 *   Pass 2: Layout sections sequentially, compute addresses
 *   Pass 3: Patch sections with final addresses, write output
 */

// LLVM headers for proper DWARF handling - must come BEFORE <elf.h> to avoid macro conflicts
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFCompileUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLine.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugAbbrev.h"
#include "llvm/DebugInfo/DWARF/DWARFAbbreviationDeclaration.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/WithColor.h"
#include "llvm/BinaryFormat/Dwarf.h"

// DWARFLinker includes
#include "llvm/DWARFLinker/Classic/DWARFLinker.h"
#include "llvm/DWARFLinker/Classic/DWARFLinkerCompileUnit.h"
#include "CustomStreamer.h"
#include "llvm/DWARFLinker/DWARFLinkerBase.h"
#include "llvm/DWARFLinker/AddressesMap.h"
#include "llvm/DWARFLinker/DWARFFile.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/DIE.h"
#include "llvm/CodeGen/NonRelocatableStringpool.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/SourceMgr.h"

// AMDGPU target initialization - will be linked via shared library
extern "C" void LLVMInitializeAMDGPUTargetInfo();
extern "C" void LLVMInitializeAMDGPUTarget();
extern "C" void LLVMInitializeAMDGPUTargetMC();
extern "C" void LLVMInitializeAMDGPUAsmPrinter();
// DWARFExpression is included via AddressesMap.h -> LowLevel/DWARFExpression.h

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <atomic>
#include <algorithm>
#include <thread>
#include <mutex>
#include <functional>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <regex>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <elf.h>

namespace fs = std::filesystem;

// Set by LLVM DWARF error handler when any error is reported; checked so device linker exits on errors
static std::atomic<bool> g_llvm_dwarf_error{false};
// DWARFContext::create expects std::function<void(Error)>, not void(const std::string&)
static void llvmDwarfErrorHandler(llvm::Error e) {
    g_llvm_dwarf_error = true;
    llvm::handleAllErrors(std::move(e), [](const llvm::ErrorInfoBase& ei) {
        std::string msg_str;
        llvm::raw_string_ostream msg(msg_str);
        ei.log(msg);
        // Always log .debug_str_offsets errors for debugging
        if (msg_str.find(".debug_str_offsets") != std::string::npos) {
            fprintf(stderr, "DWARF ERROR: ");
            ei.log(llvm::errs());
            llvm::errs() << "\n";
        }
    });
}
static void llvmDwarfWarningHandler(llvm::Error e) {
    g_llvm_dwarf_error = true;  // treat warnings as fatal so we exit on e.g. .debug_str_offsets issues
    llvm::handleAllErrors(std::move(e), [](const llvm::ErrorInfoBase& ei) {
        ei.log(llvm::errs());
        llvm::errs() << "\n";
    });
}

// ============================================================================
// LLVM DWARF helpers for finding attribute positions that need patching
// ============================================================================

// Structure to hold positions of attributes that need patching
struct DwarfAttrPositions {
    std::vector<std::pair<size_t, uint32_t>> ranges;           // DW_AT_ranges
    std::vector<std::pair<size_t, uint32_t>> str_offsets_base; // DW_AT_str_offsets_base
    std::vector<std::pair<size_t, uint32_t>> addr_base;        // DW_AT_addr_base
    std::vector<std::pair<size_t, uint32_t>> rnglists_base;    // DW_AT_rnglists_base
    std::vector<std::pair<size_t, uint32_t>> stmt_list;        // DW_AT_stmt_list
    std::vector<std::pair<size_t, uint64_t>> low_pc;           // DW_AT_low_pc (8-byte addresses, DWARF4)
    std::vector<std::pair<size_t, uint64_t>> high_pc;          // DW_AT_high_pc (when DW_FORM_addr, 8-byte addresses)
};

// Return DWARF version from the first CU in .debug_info (version at offset 4). Returns 0 if invalid/short.
static uint16_t getDwarfVersionFromDebugInfo(const uint8_t* data, size_t size) {
    if (data == nullptr || size < 6) return 0;
    uint16_t version;
    memcpy(&version, data + 4, 2);
    return version;
}

// DWARF5 .debug_str_offsets minimum header: 20 bytes to match compiler output format.
// unit_length=16 (version 2 + padding 2 + 12 bytes offset entries), version=5.
// unit_length excludes itself, so: 4 (unit_length) + 16 (version + padding + 12 bytes offsets) = 20 bytes total
// This matches what the compiler generates for minimal kernels
static const uint8_t kMinimalStrOffsetsHeader[20] = { 
    16, 0, 0, 0,  // unit_length = 16
    5, 0,         // version = 5
    0, 0,         // padding
    0, 0, 0, 0,   // offset entries (all zeros - empty .debug_str)
    0, 0, 0, 0,
    0, 0, 0, 0
};

// Minimal .debug_str_offsets size: 20 bytes (one DWARF5 header with offset entries). We pass debug_info with
// DW_AT_str_offsets_base patched to 8 (patchStrOffsetsBaseToZero) so LLVM reads offsets starting after the header.
static constexpr size_t kMinimalStrOffsetsSize = 20;

// Helper to create a minimal ELF for LLVM parsing. Includes all necessary DWARF sections
// so that llvm-dwarfdump --verify passes and readelf works correctly.
static std::vector<uint8_t> createMinimalElfForDwarf(
    const std::vector<uint8_t>& debug_info,
    const std::vector<uint8_t>& debug_abbrev,
    const std::vector<uint8_t>& debug_str = {},
    const std::vector<uint8_t>& debug_line = {},
    const std::vector<uint8_t>& debug_line_str = {},
    const std::vector<uint8_t>& debug_addr = {}) {

    constexpr size_t ELF_HDR_SIZE = sizeof(Elf64_Ehdr);
    constexpr size_t SHDR_SIZE = sizeof(Elf64_Shdr);
    
    // Count sections: always have .debug_info, .debug_abbrev, .debug_str_offsets, .debug_str, .shstrtab
    // Optionally add .debug_line, .debug_line_str, .debug_addr
    size_t num_sections = 5;  // null, debug_info, debug_abbrev, debug_str_offsets, debug_str
    if (!debug_line.empty()) num_sections++;
    if (!debug_line_str.empty()) num_sections++;
    if (!debug_addr.empty()) num_sections++;
    num_sections++;  // shstrtab

    // Build shstrtab with all section names (must be null-separated)
    std::vector<uint8_t> shstrtab_data;
    shstrtab_data.push_back(0);  // First entry is always empty (for section 0)
    
    uint32_t shstrtab_debug_info = shstrtab_data.size();
    shstrtab_data.insert(shstrtab_data.end(), (const uint8_t*)".debug_info", (const uint8_t*)".debug_info" + 11);
    shstrtab_data.push_back(0);
    
    uint32_t shstrtab_debug_abbrev = shstrtab_data.size();
    shstrtab_data.insert(shstrtab_data.end(), (const uint8_t*)".debug_abbrev", (const uint8_t*)".debug_abbrev" + 13);
    shstrtab_data.push_back(0);
    
    uint32_t shstrtab_debug_str_offsets = shstrtab_data.size();
    shstrtab_data.insert(shstrtab_data.end(), (const uint8_t*)".debug_str_offsets", (const uint8_t*)".debug_str_offsets" + 18);
    shstrtab_data.push_back(0);
    
    uint32_t shstrtab_debug_str = shstrtab_data.size();
    shstrtab_data.insert(shstrtab_data.end(), (const uint8_t*)".debug_str", (const uint8_t*)".debug_str" + 10);
    shstrtab_data.push_back(0);
    
    uint32_t shstrtab_debug_line = 0;
    uint32_t shstrtab_debug_line_str = 0;
    uint32_t shstrtab_debug_addr = 0;
    
    if (!debug_line.empty()) {
        shstrtab_debug_line = shstrtab_data.size();
        shstrtab_data.insert(shstrtab_data.end(), (const uint8_t*)".debug_line", (const uint8_t*)".debug_line" + 11);
        shstrtab_data.push_back(0);
        
        if (!debug_line_str.empty()) {
            shstrtab_debug_line_str = shstrtab_data.size();
            shstrtab_data.insert(shstrtab_data.end(), (const uint8_t*)".debug_line_str", (const uint8_t*)".debug_line_str" + 15);
            shstrtab_data.push_back(0);
            
            if (!debug_addr.empty()) {
                shstrtab_debug_addr = shstrtab_data.size();
                shstrtab_data.insert(shstrtab_data.end(), (const uint8_t*)".debug_addr", (const uint8_t*)".debug_addr" + 11);
                shstrtab_data.push_back(0);
            }
        } else if (!debug_addr.empty()) {
            shstrtab_debug_addr = shstrtab_data.size();
            shstrtab_data.insert(shstrtab_data.end(), (const uint8_t*)".debug_addr", (const uint8_t*)".debug_addr" + 11);
            shstrtab_data.push_back(0);
        }
    } else {
        if (!debug_line_str.empty()) {
            shstrtab_debug_line_str = shstrtab_data.size();
            shstrtab_data.insert(shstrtab_data.end(), (const uint8_t*)".debug_line_str", (const uint8_t*)".debug_line_str" + 15);
            shstrtab_data.push_back(0);
            
            if (!debug_addr.empty()) {
                shstrtab_debug_addr = shstrtab_data.size();
                shstrtab_data.insert(shstrtab_data.end(), (const uint8_t*)".debug_addr", (const uint8_t*)".debug_addr" + 11);
                shstrtab_data.push_back(0);
            }
        } else if (!debug_addr.empty()) {
            shstrtab_debug_addr = shstrtab_data.size();
            shstrtab_data.insert(shstrtab_data.end(), (const uint8_t*)".debug_addr", (const uint8_t*)".debug_addr" + 11);
            shstrtab_data.push_back(0);
        }
    }
    
    uint32_t shstrtab_shstrtab = shstrtab_data.size();
    shstrtab_data.insert(shstrtab_data.end(), (const uint8_t*)".shstrtab", (const uint8_t*)".shstrtab" + 9);
    shstrtab_data.push_back(0);

    // Use actual debug_str if provided, otherwise minimal empty string
    const std::vector<uint8_t>& actual_debug_str = debug_str.empty() ? 
        std::vector<uint8_t>(1, 0) : debug_str;
    
    // Build .debug_str_offsets section based on actual strings
    // Parse .debug_str to find all string offsets
    std::vector<uint32_t> str_offsets;
    if (!actual_debug_str.empty()) {
        size_t offset = 0;
        while (offset < actual_debug_str.size()) {
            str_offsets.push_back(offset);
            // Find next null terminator
            while (offset < actual_debug_str.size() && actual_debug_str[offset] != 0) {
                offset++;
            }
            if (offset < actual_debug_str.size()) {
                offset++;  // Skip null terminator
            }
        }
    } else {
        str_offsets.push_back(0);  // At least one entry for empty string
    }
    
    // Ensure we have at least 3 offset entries (12 bytes) to match compiler output format
    while (str_offsets.size() < 3) {
        str_offsets.push_back(0);
    }
    
    // Build .debug_str_offsets: header (8 bytes) + offset entries (4 bytes each)
    // unit_length excludes the 4-byte unit_length field itself
    uint32_t str_offsets_data_size = 2 + 2 + str_offsets.size() * 4;  // version(2) + padding(2) + offsets
    uint32_t unit_length = str_offsets_data_size;  // unit_length = rest of contribution
    std::vector<uint8_t> debug_str_offsets_data;
    debug_str_offsets_data.resize(4 + str_offsets_data_size);
    memcpy(&debug_str_offsets_data[0], &unit_length, 4);
    uint16_t version = 5;
    uint16_t padding = 0;
    memcpy(&debug_str_offsets_data[4], &version, 2);
    memcpy(&debug_str_offsets_data[6], &padding, 2);
    for (size_t i = 0; i < str_offsets.size(); i++) {
        uint32_t offset = str_offsets[i];
        memcpy(&debug_str_offsets_data[8 + i * 4], &offset, 4);
    }

    // Calculate offsets
    size_t shdr_offset = ELF_HDR_SIZE;
    size_t debug_info_offset = shdr_offset + num_sections * SHDR_SIZE;
    size_t debug_abbrev_offset = debug_info_offset + debug_info.size();
    size_t debug_str_offsets_offset = debug_abbrev_offset + debug_abbrev.size();
    size_t debug_str_offset = debug_str_offsets_offset + debug_str_offsets_data.size();
    size_t debug_line_offset = debug_str_offset + actual_debug_str.size();
    size_t debug_line_str_offset = debug_line_offset + (debug_line.empty() ? 0 : debug_line.size());
    size_t debug_addr_offset = debug_line_str_offset + (debug_line_str.empty() ? 0 : debug_line_str.size());
    size_t shstrtab_offset = debug_addr_offset + (debug_addr.empty() ? 0 : debug_addr.size());
    size_t total_size = shstrtab_offset + shstrtab_data.size();

    std::vector<uint8_t> elf_data(total_size, 0);

    // ELF header
    Elf64_Ehdr* ehdr = reinterpret_cast<Elf64_Ehdr*>(elf_data.data());
    ehdr->e_ident[EI_MAG0] = ELFMAG0;
    ehdr->e_ident[EI_MAG1] = ELFMAG1;
    ehdr->e_ident[EI_MAG2] = ELFMAG2;
    ehdr->e_ident[EI_MAG3] = ELFMAG3;
    ehdr->e_ident[EI_CLASS] = ELFCLASS64;
    ehdr->e_ident[EI_DATA] = ELFDATA2LSB;
    ehdr->e_ident[EI_VERSION] = EV_CURRENT;
    ehdr->e_ident[EI_OSABI] = 64;
    ehdr->e_type = ET_REL;
    ehdr->e_machine = 224;
    ehdr->e_version = EV_CURRENT;
    ehdr->e_shoff = shdr_offset;
    ehdr->e_ehsize = ELF_HDR_SIZE;
    ehdr->e_shentsize = SHDR_SIZE;
    ehdr->e_shnum = num_sections;
    ehdr->e_shstrndx = num_sections - 1;  // shstrtab is last section

    // Section headers
    Elf64_Shdr* shdrs = reinterpret_cast<Elf64_Shdr*>(elf_data.data() + shdr_offset);
    size_t shdr_idx = 1;

    shdrs[shdr_idx].sh_name = shstrtab_debug_info;
    shdrs[shdr_idx].sh_type = SHT_PROGBITS;
    shdrs[shdr_idx].sh_offset = debug_info_offset;
    shdrs[shdr_idx].sh_size = debug_info.size();
    shdrs[shdr_idx].sh_addralign = 1;
    shdr_idx++;

    shdrs[shdr_idx].sh_name = shstrtab_debug_abbrev;
    shdrs[shdr_idx].sh_type = SHT_PROGBITS;
    shdrs[shdr_idx].sh_offset = debug_abbrev_offset;
    shdrs[shdr_idx].sh_size = debug_abbrev.size();
    shdrs[shdr_idx].sh_addralign = 1;
    shdr_idx++;

    shdrs[shdr_idx].sh_name = shstrtab_debug_str_offsets;
    shdrs[shdr_idx].sh_type = SHT_PROGBITS;
    shdrs[shdr_idx].sh_offset = debug_str_offsets_offset;
    shdrs[shdr_idx].sh_size = debug_str_offsets_data.size();
    shdrs[shdr_idx].sh_addralign = 1;
    shdr_idx++;

    shdrs[shdr_idx].sh_name = shstrtab_debug_str;
    shdrs[shdr_idx].sh_type = SHT_PROGBITS;
    shdrs[shdr_idx].sh_offset = debug_str_offset;
    shdrs[shdr_idx].sh_size = actual_debug_str.size();
    shdrs[shdr_idx].sh_addralign = 1;
    shdr_idx++;

    if (!debug_line.empty()) {
        shdrs[shdr_idx].sh_name = shstrtab_debug_line;
        shdrs[shdr_idx].sh_type = SHT_PROGBITS;
        shdrs[shdr_idx].sh_offset = debug_line_offset;
        shdrs[shdr_idx].sh_size = debug_line.size();
        shdrs[shdr_idx].sh_addralign = 1;
        shdr_idx++;
    }

    if (!debug_line_str.empty()) {
        shdrs[shdr_idx].sh_name = shstrtab_debug_line_str;
        shdrs[shdr_idx].sh_type = SHT_PROGBITS;
        shdrs[shdr_idx].sh_offset = debug_line_str_offset;
        shdrs[shdr_idx].sh_size = debug_line_str.size();
        shdrs[shdr_idx].sh_addralign = 1;
        shdr_idx++;
    }

    if (!debug_addr.empty()) {
        shdrs[shdr_idx].sh_name = shstrtab_debug_addr;
        shdrs[shdr_idx].sh_type = SHT_PROGBITS;
        shdrs[shdr_idx].sh_offset = debug_addr_offset;
        shdrs[shdr_idx].sh_size = debug_addr.size();
        shdrs[shdr_idx].sh_addralign = 1;
        shdr_idx++;
    }

    shdrs[shdr_idx].sh_name = shstrtab_shstrtab;
    shdrs[shdr_idx].sh_type = SHT_STRTAB;
    shdrs[shdr_idx].sh_offset = shstrtab_offset;
    shdrs[shdr_idx].sh_size = shstrtab_data.size();
    shdrs[shdr_idx].sh_addralign = 1;

    // Copy section data
    memcpy(elf_data.data() + debug_info_offset, debug_info.data(), debug_info.size());
    memcpy(elf_data.data() + debug_abbrev_offset, debug_abbrev.data(), debug_abbrev.size());
    memcpy(elf_data.data() + debug_str_offsets_offset, debug_str_offsets_data.data(), debug_str_offsets_data.size());
    memcpy(elf_data.data() + debug_str_offset, actual_debug_str.data(), actual_debug_str.size());
    if (!debug_line.empty()) {
        memcpy(elf_data.data() + debug_line_offset, debug_line.data(), debug_line.size());
    }
    if (!debug_line_str.empty()) {
        memcpy(elf_data.data() + debug_line_str_offset, debug_line_str.data(), debug_line_str.size());
    }
    if (!debug_addr.empty()) {
        memcpy(elf_data.data() + debug_addr_offset, debug_addr.data(), debug_addr.size());
    }
    memcpy(elf_data.data() + shstrtab_offset, shstrtab_data.data(), shstrtab_data.size());

    return elf_data;
}

// Phase 2: Minimal ELF for parsing a single .debug_line chunk with LLVM.
// Contains one CU with DW_AT_stmt_list=0 so getLineTableForUnit parses from offset 0 of .debug_line.
static std::vector<uint8_t> createMinimalElfForLineTable(
    const std::vector<uint8_t>& debug_line,
    const std::vector<uint8_t>& debug_line_str) {

    // Minimal .debug_abbrev: one entry code 1, DW_TAG_compile_unit, no children, DW_AT_stmt_list DW_FORM_sec_offset, (0,0)
    const uint8_t minimal_abbrev[] = { 1, 0x11, 0, 0x10, 0x17, 0, 0 };  // code, tag, no_children, AT_stmt_list, FORM_sec_offset, 0,0
    // Minimal .debug_info: DWARF5 CU header + one DIE (abbrev 1, stmt_list=0)
    // unit_length(4)=13, version(2)=5, unit_type(1)=0, addr_size(1)=8, abbrev_offset(4)=0, ULEB(1), 4-byte 0
    const uint8_t minimal_info[] = {
        13, 0, 0, 0,   // unit_length (rest of unit)
        5, 0,          // version
        0,             // unit_type (DW_UT_compile)
        8,             // addr_size
        0, 0, 0, 0,    // abbrev_offset
        1,             // abbrev code 1
        0, 0, 0, 0     // DW_AT_stmt_list = 0
    };

    constexpr size_t ELF_HDR_SIZE = sizeof(Elf64_Ehdr);
    constexpr size_t SHDR_SIZE = sizeof(Elf64_Shdr);
    const size_t NUM_SECTIONS = 6;  // null, debug_info, debug_abbrev, debug_line, debug_line_str, shstrtab

    const char shstrtab[] = "\0.debug_info\0.debug_abbrev\0.debug_line\0.debug_line_str\0.shstrtab\0";
    constexpr size_t SHSTRTAB_SIZE = sizeof(shstrtab);
    uint32_t debug_info_name = 1, debug_abbrev_name = 14, debug_line_name = 27, debug_line_str_name = 39, shstrtab_name = 55;

    size_t shdr_offset = ELF_HDR_SIZE;
    size_t debug_info_offset = shdr_offset + NUM_SECTIONS * SHDR_SIZE;
    size_t debug_abbrev_offset = debug_info_offset + sizeof(minimal_info);
    size_t debug_line_offset = debug_abbrev_offset + sizeof(minimal_abbrev);
    size_t debug_line_str_offset = debug_line_offset + debug_line.size();
    size_t shstrtab_offset = debug_line_str_offset + debug_line_str.size();
    size_t total_size = shstrtab_offset + SHSTRTAB_SIZE;

    std::vector<uint8_t> elf_data(total_size, 0);

    Elf64_Ehdr* ehdr = reinterpret_cast<Elf64_Ehdr*>(elf_data.data());
    ehdr->e_ident[EI_MAG0] = ELFMAG0;
    ehdr->e_ident[EI_MAG1] = ELFMAG1;
    ehdr->e_ident[EI_MAG2] = ELFMAG2;
    ehdr->e_ident[EI_MAG3] = ELFMAG3;
    ehdr->e_ident[EI_CLASS] = ELFCLASS64;
    ehdr->e_ident[EI_DATA] = ELFDATA2LSB;
    ehdr->e_ident[EI_VERSION] = EV_CURRENT;
    ehdr->e_ident[EI_OSABI] = 64;
    ehdr->e_type = ET_REL;
    ehdr->e_machine = 224;
    ehdr->e_version = EV_CURRENT;
    ehdr->e_shoff = shdr_offset;
    ehdr->e_ehsize = ELF_HDR_SIZE;
    ehdr->e_shentsize = SHDR_SIZE;
    ehdr->e_shnum = NUM_SECTIONS;
    ehdr->e_shstrndx = 5;

    Elf64_Shdr* shdrs = reinterpret_cast<Elf64_Shdr*>(elf_data.data() + shdr_offset);
    shdrs[1].sh_name = debug_info_name;
    shdrs[1].sh_type = SHT_PROGBITS;
    shdrs[1].sh_offset = debug_info_offset;
    shdrs[1].sh_size = sizeof(minimal_info);
    shdrs[1].sh_addralign = 1;

    shdrs[2].sh_name = debug_abbrev_name;
    shdrs[2].sh_type = SHT_PROGBITS;
    shdrs[2].sh_offset = debug_abbrev_offset;
    shdrs[2].sh_size = sizeof(minimal_abbrev);
    shdrs[2].sh_addralign = 1;

    shdrs[3].sh_name = debug_line_name;
    shdrs[3].sh_type = SHT_PROGBITS;
    shdrs[3].sh_offset = debug_line_offset;
    shdrs[3].sh_size = debug_line.size();
    shdrs[3].sh_addralign = 1;

    shdrs[4].sh_name = debug_line_str_name;
    shdrs[4].sh_type = SHT_PROGBITS;
    shdrs[4].sh_offset = debug_line_str_offset;
    shdrs[4].sh_size = debug_line_str.size();
    shdrs[4].sh_addralign = 1;

    shdrs[5].sh_name = shstrtab_name;
    shdrs[5].sh_type = SHT_STRTAB;
    shdrs[5].sh_offset = shstrtab_offset;
    shdrs[5].sh_size = SHSTRTAB_SIZE;
    shdrs[5].sh_addralign = 1;

    memcpy(elf_data.data() + debug_info_offset, minimal_info, sizeof(minimal_info));
    memcpy(elf_data.data() + debug_abbrev_offset, minimal_abbrev, sizeof(minimal_abbrev));
    if (!debug_line.empty())
        memcpy(elf_data.data() + debug_line_offset, debug_line.data(), debug_line.size());
    if (!debug_line_str.empty())
        memcpy(elf_data.data() + debug_line_str_offset, debug_line_str.data(), debug_line_str.size());
    memcpy(elf_data.data() + shstrtab_offset, shstrtab, SHSTRTAB_SIZE);

    return elf_data;
}

// Append ULEB128 to buffer (for Phase 3 line table emission)
static void appendULEB128(std::vector<uint8_t>& out, uint64_t value) {
    // Standard ULEB128 encoding: check BEFORE shifting
    while (value >= 0x80) {
        out.push_back((value & 0x7f) | 0x80);
        value >>= 7;
    }
    out.push_back(value & 0x7f);
}

// Helper: decode ULEB128 from data, return value and bytes consumed
static uint64_t decodeULEB128(const uint8_t* data, size_t max_len, size_t& bytes_read) {
    uint64_t result = 0;
    unsigned shift = 0;
    bytes_read = 0;

    while (bytes_read < max_len) {
        uint8_t byte = data[bytes_read++];
        result |= (uint64_t)(byte & 0x7f) << shift;
        if ((byte & 0x80) == 0) break;
        shift += 7;
    }
    return result;
}

// Helper: decode SLEB128 from data
static int64_t decodeSLEB128(const uint8_t* data, size_t max_len, size_t& bytes_read) {
    int64_t result = 0;
    unsigned shift = 0;
    bytes_read = 0;
    uint8_t byte;

    while (bytes_read < max_len) {
        byte = data[bytes_read++];
        result |= (int64_t)(byte & 0x7f) << shift;
        shift += 7;
        if ((byte & 0x80) == 0) break;
    }

    // Sign extend if needed
    if (shift < 64 && (byte & 0x40)) {
        result |= -(1LL << shift);
    }
    return result;
}

// Calculate size of a DWARF form in bytes (returns 0 for variable-size forms)
// For variable-size forms, we need to decode them
static size_t getFormFixedSize(uint16_t form, uint8_t addr_size, uint16_t dwarf_version) {
    switch (form) {
        case llvm::dwarf::DW_FORM_addr: return addr_size;
        case llvm::dwarf::DW_FORM_data1:
        case llvm::dwarf::DW_FORM_ref1:
        case llvm::dwarf::DW_FORM_flag:
        case llvm::dwarf::DW_FORM_strx1:
        case llvm::dwarf::DW_FORM_addrx1:
            return 1;
        case llvm::dwarf::DW_FORM_data2:
        case llvm::dwarf::DW_FORM_ref2:
        case llvm::dwarf::DW_FORM_strx2:
        case llvm::dwarf::DW_FORM_addrx2:
            return 2;
        case llvm::dwarf::DW_FORM_strx3:
        case llvm::dwarf::DW_FORM_addrx3:
            return 3;
        case llvm::dwarf::DW_FORM_data4:
        case llvm::dwarf::DW_FORM_ref4:
        case llvm::dwarf::DW_FORM_ref_sup4:
        case llvm::dwarf::DW_FORM_strx4:
        case llvm::dwarf::DW_FORM_addrx4:
            return 4;
        case llvm::dwarf::DW_FORM_data8:
        case llvm::dwarf::DW_FORM_ref8:
        case llvm::dwarf::DW_FORM_ref_sig8:
        case llvm::dwarf::DW_FORM_ref_sup8:
            return 8;
        case llvm::dwarf::DW_FORM_sec_offset:
        case llvm::dwarf::DW_FORM_strp:
        case llvm::dwarf::DW_FORM_line_strp:
        case llvm::dwarf::DW_FORM_ref_addr:
            return 4;  // DWARF5 32-bit format
        case llvm::dwarf::DW_FORM_data16:
            return 16;
        case llvm::dwarf::DW_FORM_flag_present:
        case llvm::dwarf::DW_FORM_implicit_const:
            return 0;  // No data in DIE
        default:
            return 0;  // Variable size - need to decode
    }
}

// Skip over a variable-length form, return bytes consumed
static size_t skipVariableForm(uint16_t form, const uint8_t* data, size_t max_len) {
    size_t bytes;
    switch (form) {
        case llvm::dwarf::DW_FORM_string: {
            // Null-terminated string
            size_t len = strnlen((const char*)data, max_len);
            return len + 1;  // Include null terminator
        }
        case llvm::dwarf::DW_FORM_exprloc:
        case llvm::dwarf::DW_FORM_block: {
            uint64_t len = decodeULEB128(data, max_len, bytes);
            return bytes + len;
        }
        case llvm::dwarf::DW_FORM_block1:
            return 1 + data[0];
        case llvm::dwarf::DW_FORM_block2:
            return 2 + (data[0] | (data[1] << 8));
        case llvm::dwarf::DW_FORM_block4:
            return 4 + (data[0] | (data[1] << 8) | (data[2] << 16) | (data[3] << 24));
        case llvm::dwarf::DW_FORM_sdata:
            decodeSLEB128(data, max_len, bytes);
            return bytes;
        case llvm::dwarf::DW_FORM_udata:
        case llvm::dwarf::DW_FORM_ref_udata:
        case llvm::dwarf::DW_FORM_strx:
        case llvm::dwarf::DW_FORM_addrx:
        case llvm::dwarf::DW_FORM_loclistx:
        case llvm::dwarf::DW_FORM_rnglistx:
            decodeULEB128(data, max_len, bytes);
            return bytes;
        case llvm::dwarf::DW_FORM_indirect: {
            // Recursive: first is a ULEB128 form, then the actual data
            uint64_t actual_form = decodeULEB128(data, max_len, bytes);
            size_t fixed = getFormFixedSize(actual_form, 8, 5);
            if (fixed > 0) return bytes + fixed;
            return bytes + skipVariableForm(actual_form, data + bytes, max_len - bytes);
        }
        case llvm::dwarf::DW_FORM_flag_present:
            // No data in DIE; used in line table prologue too
            return 0;
        default:
            return 0;
    }
}

// Abbrev entry: (attr, form) for patching specific attributes without LLVM.
using AbbrevAttrsForms = std::vector<std::pair<uint16_t, uint16_t>>;
// Parse .debug_abbrev and return map: abbrev_code -> (has_children, (attr,form)*).
static std::map<uint64_t, std::pair<bool, AbbrevAttrsForms>> parseAbbrevTableWithAttrs(
    const uint8_t* data, size_t size) {
    std::map<uint64_t, std::pair<bool, AbbrevAttrsForms>> map;
    const uint8_t* p = data;
    const uint8_t* end = data + size;
    while (p < end) {
        size_t n;
        uint64_t code = decodeULEB128(p, (size_t)(end - p), n);
        p += n;
        if (code == 0) break;
        uint64_t tag = decodeULEB128(p, (size_t)(end - p), n);
        p += n;
        if (p >= end) break;
        uint8_t has_children = *p++;
        AbbrevAttrsForms attrs_forms;
        while (p < end) {
            uint64_t attr = decodeULEB128(p, (size_t)(end - p), n);
            p += n;
            uint64_t form = decodeULEB128(p, (size_t)(end - p), n);
            p += n;
            if (attr == 0 && form == 0) break;
            attrs_forms.push_back({(uint16_t)attr, (uint16_t)form});
        }
        map[code] = { (has_children != llvm::dwarf::DW_CHILDREN_no), std::move(attrs_forms) };
    }
    return map;
}

// Parse .debug_abbrev and return map: abbrev_code -> (has_children, forms).
// Abbrev format: code(ULEB), tag(ULEB), has_children(1), (attr(ULEB), form(ULEB))* (0,0).
static std::map<uint64_t, std::pair<bool, std::vector<uint16_t>>> parseAbbrevTable(
    const uint8_t* data, size_t size) {
    std::map<uint64_t, std::pair<bool, std::vector<uint16_t>>> map;
    const uint8_t* p = data;
    const uint8_t* end = data + size;
    while (p < end) {
        size_t n;
        uint64_t code = decodeULEB128(p, (size_t)(end - p), n);
        p += n;
        if (code == 0) break;
        uint64_t tag = decodeULEB128(p, (size_t)(end - p), n);
        p += n;
        if (p >= end) break;
        uint8_t has_children = *p++;
        std::vector<uint16_t> forms;
        while (p < end) {
            uint64_t attr = decodeULEB128(p, (size_t)(end - p), n);
            p += n;
            uint64_t form = decodeULEB128(p, (size_t)(end - p), n);
            p += n;
            if (attr == 0 && form == 0) break;
            forms.push_back((uint16_t)form);
        }
        map[code] = { (has_children != llvm::dwarf::DW_CHILDREN_no), std::move(forms) };
    }
    return map;
}

// Find all DW_FORM_line_strp (offset, value) in one CU's .debug_info by scanning raw bytes.
// Does not use LLVM, so avoids .debug_str_offsets / .debug_addr and abbrev resolution.
static std::vector<std::pair<size_t, uint32_t>> findLineStrpInChunkManual(
    const uint8_t* info_data, size_t info_size,
    const std::map<uint64_t, std::pair<bool, std::vector<uint16_t>>>& abbrev_map,
    uint8_t addr_size, uint16_t version) {

    std::vector<std::pair<size_t, uint32_t>> result;
    if (info_size < 8 || abbrev_map.empty()) return result;

    // DWARF5 CU header: unit_length(4), version(2), unit_type(1), addr_size(1), abbrev_offset(4) = 12
    constexpr size_t kDWARF5CuHeaderSize = 12;
    size_t die_start = kDWARF5CuHeaderSize;
    if (die_start > info_size) return result;

    std::function<size_t(size_t)> parseDIE;
    parseDIE = [&](size_t pos) -> size_t {
        if (pos >= info_size) return info_size;
        size_t n;
        uint64_t code = decodeULEB128(info_data + pos, info_size - pos, n);
        pos += n;
        if (code == 0) return pos;  // null DIE
        auto it = abbrev_map.find(code);
        if (it == abbrev_map.end()) return info_size;
        const std::vector<uint16_t>& forms = it->second.second;
        bool has_children = it->second.first;
        for (uint16_t form : forms) {
            if (pos >= info_size) return info_size;
            size_t form_sz = getFormFixedSize(form, addr_size, version);
            if (form_sz == 0) form_sz = skipVariableForm(form, info_data + pos, info_size - pos);
            if (form_sz == 0) return info_size;
            if (form == llvm::dwarf::DW_FORM_line_strp && form_sz >= 4 && pos + 4 <= info_size) {
                uint32_t val;
                memcpy(&val, info_data + pos, 4);
                result.push_back({pos, val});
            }
            pos += form_sz;
        }
        if (has_children) {
            while (pos < info_size) {
                size_t next = parseDIE(pos);
                if (next <= pos) return next;
                // Null DIE (abbrev code 0, one byte) ends list of children
                if (next == pos + 1 && info_data[pos] == 0) { pos = next; break; }
                pos = next;
            }
        }
        return pos;
    };

    // Single CU: one root DIE, then its children (until null DIE).
    (void)parseDIE(die_start);
    return result;
}

// Patch all DW_AT_str_offsets_base to 8 in a copy of debug_info so the minimal ELF
// matches compiler output format (offsets start after the 8-byte header).
// DWARF4 has no DW_AT_str_offsets_base; return copy unchanged.
static std::vector<uint8_t> patchStrOffsetsBaseToZero(
    const std::vector<uint8_t>& debug_info,
    const std::vector<uint8_t>& debug_abbrev) {
    std::vector<uint8_t> out = debug_info;
    if (out.size() < 8 || debug_abbrev.empty()) return out;
    uint16_t version;
    memcpy(&version, &out[4], 2);
    if (version == 4) return out;  // DWARF4: no str_offsets_base to patch

    auto abbrev_map = parseAbbrevTableWithAttrs(debug_abbrev.data(), debug_abbrev.size());
    if (abbrev_map.empty()) return out;

    size_t pos = 0;
    while (pos + 4 <= out.size()) {
        uint32_t unit_length;
        memcpy(&unit_length, &out[pos], 4);
        if (unit_length == 0xffffffff || unit_length + 4 > out.size() - pos) break;
        size_t cu_end = pos + 4 + unit_length;
        // DWARF5 CU header: unit_length(4), version(2), unit_type(1), addr_size(1), abbrev_offset(4) = 12
        constexpr size_t kDWARF5CuHeaderSize = 12;
        if (pos + kDWARF5CuHeaderSize > cu_end) { pos = cu_end; continue; }
        size_t die_start = pos + kDWARF5CuHeaderSize;
        size_t n;
        uint64_t code = decodeULEB128(out.data() + die_start, cu_end - die_start, n);
        size_t attr_pos = die_start + n;
        auto it = abbrev_map.find(code);
        if (it == abbrev_map.end()) { pos = cu_end; continue; }
        uint8_t addr_size = out[pos + 8];  // DWARF5: addr_size at offset 8
        constexpr uint16_t kDWARFVersion = 5;
        for (const auto& [attr, form] : it->second.second) {
            if (attr_pos >= cu_end) break;
            size_t form_sz = getFormFixedSize(form, addr_size, kDWARFVersion);
            if (form_sz == 0) form_sz = skipVariableForm(form, out.data() + attr_pos, cu_end - attr_pos);
            if (form_sz == 0) break;
            if (attr == llvm::dwarf::DW_AT_str_offsets_base && form_sz == 4 && attr_pos + 4 <= cu_end) {
                uint32_t base_offset = 8;  // Offsets start after the 8-byte header (matches compiler output)
                memcpy(&out[attr_pos], &base_offset, 4);
            }
            attr_pos += form_sz;
        }
        pos = cu_end;
    }
    return out;
}

// Phase 1a: Use real ELF with LLVM (dispatcher). LLVM sees all sections including .debug_str_offsets.
static DwarfAttrPositions findDwarfAttrPositionsFromElf(const uint8_t* elf_data, size_t elf_size) {
    DwarfAttrPositions result;
    if (elf_data == nullptr || elf_size == 0) return result;

    llvm::StringRef elf_ref(reinterpret_cast<const char*>(elf_data), elf_size);
    std::unique_ptr<llvm::MemoryBuffer> buf = llvm::MemoryBuffer::getMemBufferCopy(elf_ref, "");
    llvm::Expected<std::unique_ptr<llvm::object::ObjectFile>> obj_or =
        llvm::object::ObjectFile::createObjectFile(buf->getMemBufferRef());
    if (!obj_or) {
        llvm::consumeError(obj_or.takeError());
        return result;
    }
    std::unique_ptr<llvm::object::ObjectFile> obj = std::move(obj_or.get());
    std::unique_ptr<llvm::DWARFContext> ctx =
        llvm::DWARFContext::create(*obj, llvm::DWARFContext::ProcessDebugRelocations::Ignore,
                                    nullptr, "", llvmDwarfErrorHandler,
                                    llvmDwarfWarningHandler, false);
    if (!ctx) return result;

    // Get .debug_info section content from the object for offset calculations
    const uint8_t* info_data = nullptr;
    size_t info_size = 0;
    for (auto it = obj->section_begin(); it != obj->section_end(); ++it) {
        llvm::Expected<llvm::StringRef> name_or = it->getName();
        if (!name_or || *name_or != ".debug_info") continue;
        llvm::Expected<llvm::StringRef> contents_or = it->getContents();
        if (!contents_or) break;
        info_data = reinterpret_cast<const uint8_t*>(contents_or->data());
        info_size = contents_or->size();
        break;
    }
    if (info_data == nullptr || info_size == 0) return result;

    for (const std::unique_ptr<llvm::DWARFUnit>& unit_ptr : ctx->info_section_units()) {
        llvm::DWARFUnit* unit = unit_ptr.get();
        llvm::DWARFDie die = unit->getUnitDIE(false);
        if (!die.isValid())
            continue;

        uint64_t die_offset = die.getOffset();
        if (die_offset >= info_size)
            continue;

        size_t uleb_bytes = 0;
        (void)decodeULEB128(info_data + die_offset, info_size - die_offset, uleb_bytes);
        size_t pos = die_offset + uleb_bytes;

        for (const auto& attr_spec : die.attributes()) {
            llvm::dwarf::Attribute attr = attr_spec.Attr;
            llvm::dwarf::Form form = attr_spec.Value.getForm();
            size_t attr_value_offset = pos;
            size_t form_sz = getFormFixedSize(static_cast<uint16_t>(form),
                                              unit->getAddressByteSize(),
                                              unit->getVersion());
            if (form_sz == 0)
                form_sz = skipVariableForm(static_cast<uint16_t>(form), info_data + pos, info_size - pos);
            if (form_sz == 0)
                break;

            if ((form == llvm::dwarf::DW_FORM_sec_offset || form == llvm::dwarf::DW_FORM_data4) &&
                attr_value_offset + 4 <= info_size) {
                uint32_t val;
                memcpy(&val, info_data + attr_value_offset, 4);
                switch (attr) {
                    case llvm::dwarf::DW_AT_ranges:
                        result.ranges.push_back({attr_value_offset, val});
                        break;
                    case llvm::dwarf::DW_AT_str_offsets_base:
                        result.str_offsets_base.push_back({attr_value_offset, val});
                        break;
                    case llvm::dwarf::DW_AT_addr_base:
                        result.addr_base.push_back({attr_value_offset, val});
                        break;
                    case llvm::dwarf::DW_AT_rnglists_base:
                        result.rnglists_base.push_back({attr_value_offset, val});
                        break;
                    case llvm::dwarf::DW_AT_stmt_list:
                        result.stmt_list.push_back({attr_value_offset, val});
                        break;
                    default:
                        break;
                }
            } else if (form == llvm::dwarf::DW_FORM_addr && attr_value_offset + 8 <= info_size) {
                // DW_AT_low_pc and DW_AT_high_pc can use DW_FORM_addr (8-byte address) - needed for DWARF4 address patching
                uint64_t val;
                memcpy(&val, info_data + attr_value_offset, 8);
                if (attr == llvm::dwarf::DW_AT_low_pc) {
                    result.low_pc.push_back({attr_value_offset, val});
                } else if (attr == llvm::dwarf::DW_AT_high_pc) {
                    result.high_pc.push_back({attr_value_offset, val});
                }
            }
            pos += form_sz;
            if (pos >= info_size)
                break;
        }
    }
    return result;
}

// Phase 1b: Use minimal ELF with LLVM (kernels). Kernel .o files often have .debug_str_offsets
// too small or missing, so we build a minimal ELF with patched str_offsets_base and proper sections.
static DwarfAttrPositions findDwarfAttrPositionsUsingLLVM(
    const std::vector<uint8_t>& debug_info,
    const std::vector<uint8_t>& debug_abbrev,
    const std::vector<uint8_t>& debug_str = {},
    const std::vector<uint8_t>& debug_line = {},
    const std::vector<uint8_t>& debug_line_str = {},
    const std::vector<uint8_t>& debug_addr = {}) {
    DwarfAttrPositions result;
    if (debug_info.empty() || debug_abbrev.empty()) return result;
    
    // Reset error flag before parsing
    g_llvm_dwarf_error = false;
    
    std::vector<uint8_t> info_for_llvm = patchStrOffsetsBaseToZero(debug_info, debug_abbrev);
    std::vector<uint8_t> elf_data = createMinimalElfForDwarf(info_for_llvm, debug_abbrev, 
                                                              debug_str, debug_line, debug_line_str, debug_addr);
    
    // Verify minimal ELF has correct .debug_str_offsets header
    // Find .debug_str_offsets section by searching section headers
    Elf64_Ehdr* ehdr = reinterpret_cast<Elf64_Ehdr*>(elf_data.data());
    Elf64_Shdr* shdrs = reinterpret_cast<Elf64_Shdr*>(elf_data.data() + ehdr->e_shoff);
    const char* shstrtab = reinterpret_cast<const char*>(elf_data.data() + shdrs[ehdr->e_shstrndx].sh_offset);
    
    for (size_t i = 1; i < ehdr->e_shnum; i++) {
        if (strcmp(shstrtab + shdrs[i].sh_name, ".debug_str_offsets") == 0) {
            uint32_t unit_length;
            uint16_t version;
            memcpy(&unit_length, &elf_data[shdrs[i].sh_offset], 4);
            memcpy(&version, &elf_data[shdrs[i].sh_offset + 4], 2);
            if (version != 5) {
                fprintf(stderr, "WARNING: Minimal ELF .debug_str_offsets header invalid: version=%u (expected 5)\n", version);
            }
            break;
        }
    }
    
    return findDwarfAttrPositionsFromElf(elf_data.data(), elf_data.size());
}

// Find all DW_FORM_line_strp (offset, value) in .debug_info for patching when merging .debug_line_str.
// DWARF5 uses line_strp in .debug_info (e.g. DW_AT_decl_file); offsets must be adjusted per chunk.
static std::vector<std::pair<size_t, uint32_t>> findLineStrpPositionsInDebugInfo(
    const std::vector<uint8_t>& debug_info,
    const std::vector<uint8_t>& debug_abbrev) {

    std::vector<std::pair<size_t, uint32_t>> result;
    if (debug_info.empty() || debug_abbrev.empty()) return result;

    std::vector<uint8_t> info_for_llvm = patchStrOffsetsBaseToZero(debug_info, debug_abbrev);
    std::vector<uint8_t> elf_data = createMinimalElfForDwarf(info_for_llvm, debug_abbrev, {}, {}, {}, {});
    llvm::StringRef elf_ref(reinterpret_cast<const char*>(elf_data.data()), elf_data.size());
    std::unique_ptr<llvm::MemoryBuffer> buf = llvm::MemoryBuffer::getMemBufferCopy(elf_ref, "");
    llvm::Expected<std::unique_ptr<llvm::object::ObjectFile>> obj_or =
        llvm::object::ObjectFile::createObjectFile(buf->getMemBufferRef());
    if (!obj_or) {
        llvm::consumeError(obj_or.takeError());
        return result;
    }
    std::unique_ptr<llvm::object::ObjectFile> obj = std::move(obj_or.get());
    std::unique_ptr<llvm::DWARFContext> ctx =
        llvm::DWARFContext::create(*obj, llvm::DWARFContext::ProcessDebugRelocations::Ignore,
                                    nullptr, "", llvmDwarfErrorHandler,
                                    llvmDwarfWarningHandler, false);
    if (!ctx) return result;

    const uint8_t* info_data = debug_info.data();
    const size_t info_size = debug_info.size();

    std::function<size_t(llvm::DWARFDie)> processDIE;
    processDIE = [&](llvm::DWARFDie die) -> size_t {
        if (!die.isValid()) return 0;
        uint64_t die_offset = die.getOffset();
        if (die_offset >= info_size) return die_offset;

        size_t uleb_bytes = 0;
        (void)decodeULEB128(info_data + die_offset, info_size - die_offset, uleb_bytes);
        size_t pos = die_offset + uleb_bytes;

        llvm::DWARFUnit* unit = die.getDwarfUnit();
        uint8_t addr_size = unit ? unit->getAddressByteSize() : 8;
        uint16_t version = unit ? unit->getVersion() : 4;

        for (const auto& attr_spec : die.attributes()) {
            llvm::dwarf::Form form = attr_spec.Value.getForm();
            size_t form_sz = getFormFixedSize(static_cast<uint16_t>(form), addr_size, version);
            if (form_sz == 0)
                form_sz = skipVariableForm(static_cast<uint16_t>(form), info_data + pos, info_size - pos);
            if (form_sz == 0) break;

            if (form == llvm::dwarf::DW_FORM_line_strp && form_sz >= 4 && pos + 4 <= info_size) {
                uint32_t val;
                memcpy(&val, info_data + pos, 4);
                result.push_back({static_cast<size_t>(pos), val});
            }
            pos += form_sz;
            if (pos >= info_size) break;
        }

        for (llvm::DWARFDie child = die.getFirstChild(); child.isValid(); child = child.getSibling())
            pos = processDIE(child);
        return pos;
    };

    for (const std::unique_ptr<llvm::DWARFUnit>& unit_ptr : ctx->info_section_units()) {
        llvm::DWARFDie root = unit_ptr->getUnitDIE(false);
        if (root.isValid())
            processDIE(root);
    }
    return result;
}

// ============================================================================
// Constants
// ============================================================================

constexpr int FUNC_COUNT = 859;
constexpr int FUNC_ALIGNMENT = 256;

// ============================================================================
// Memory-mapped file wrapper
// ============================================================================

class MappedFile {
public:
    MappedFile(const std::string& path) : fd_(-1), data_(nullptr), size_(0) {
        fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ < 0) return;

        struct stat st;
        if (fstat(fd_, &st) < 0) { close(fd_); fd_ = -1; return; }

        size_ = st.st_size;
        data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (data_ == MAP_FAILED) { data_ = nullptr; close(fd_); fd_ = -1; }
    }

    ~MappedFile() {
        if (data_) munmap(data_, size_);
        if (fd_ >= 0) close(fd_);
    }

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    bool valid() const { return data_ != nullptr; }
    size_t size() const { return size_; }
    const void* data() const { return data_; }

    template<typename T> const T* at(size_t offset) const {
        return reinterpret_cast<const T*>(static_cast<const char*>(data_) + offset);
    }
    const char* str(size_t offset) const { return static_cast<const char*>(data_) + offset; }

private:
    int fd_;
    void* data_;
    size_t size_;
};

// ============================================================================
// ELF Section Info
// ============================================================================

struct SectionInfo {
    std::string name;
    uint32_t type = 0;
    uint64_t flags = 0;
    uint64_t alignment = 1;
    uint64_t entsize = 0;
    std::vector<uint8_t> data;
    size_t nobits_size = 0;  // For SHT_NOBITS sections

    // Computed during layout
    uint64_t addr = 0;
    uint64_t offset = 0;

    size_t size() const { return type == SHT_NOBITS ? nobits_size : data.size(); }
    size_t fileSize() const { return type == SHT_NOBITS ? 0 : data.size(); }
    bool isAlloc() const { return flags & SHF_ALLOC; }
};

// ============================================================================
// ELF Parser (read-only)
// ============================================================================

class ElfParser {
public:
    ElfParser(const MappedFile& file) : file_(file), ehdr_(nullptr) {
        if (!file.valid()) return;
        ehdr_ = file.at<Elf64_Ehdr>(0);
        if (memcmp(ehdr_->e_ident, ELFMAG, SELFMAG) != 0) { ehdr_ = nullptr; return; }

        const Elf64_Shdr* shdrs = file.at<Elf64_Shdr>(ehdr_->e_shoff);
        const char* shstrtab = file.str(shdrs[ehdr_->e_shstrndx].sh_offset);

        for (int i = 0; i < ehdr_->e_shnum; i++) {
            sections_.push_back({
                shstrtab + shdrs[i].sh_name,
                shdrs[i].sh_type,
                shdrs[i].sh_flags,
                shdrs[i].sh_addr,
                shdrs[i].sh_offset,
                shdrs[i].sh_size,
                shdrs[i].sh_addralign,
                shdrs[i].sh_entsize,
                (uint16_t)i
            });
        }
    }

    bool valid() const { return ehdr_ != nullptr; }
    const Elf64_Ehdr* ehdr() const { return ehdr_; }

    struct ParsedSection {
        std::string name;
        uint32_t type;
        uint64_t flags;
        uint64_t addr;
        uint64_t offset;
        uint64_t size;
        uint64_t align;
        uint64_t entsize;
        uint16_t index;
    };

    const ParsedSection* find(const std::string& name) const {
        for (const auto& s : sections_) if (s.name == name) return &s;
        return nullptr;
    }

    std::vector<uint8_t> getBytes(const ParsedSection& s) const {
        const uint8_t* p = file_.at<uint8_t>(s.offset);
        return std::vector<uint8_t>(p, p + s.size);
    }

    // Access to symbol table entries (for finding function table locations)
    template<typename T> const T* fileAt(size_t offset) const { return file_.at<T>(offset); }
    const char* fileStr(size_t offset) const { return file_.str(offset); }

private:
    const MappedFile& file_;
    const Elf64_Ehdr* ehdr_;
    std::vector<ParsedSection> sections_;
};

// ============================================================================
// Kernel extraction utilities
// ============================================================================

std::string g_target_arch;  // Default: detect from local GPU (includes feature flags)

std::string detectLocalGpu() {
    // Try to get full target with features (e.g., gfx950:sramecc+:xnack-)
    // This matches what HIP/clang-offload-bundler expects
    FILE* fp = popen("rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-z]+:[^[:space:]]+' | head -1", "r");
    if (fp) {
        char buf[128] = {0};
        if (fgets(buf, sizeof(buf), fp)) {
            pclose(fp);
            std::string arch(buf);
            while (!arch.empty() && (arch.back() == '\n' || arch.back() == '\r'))
                arch.pop_back();
            if (!arch.empty()) return arch;
        } else {
            pclose(fp);
        }
    }

    // Fallback: just get the gfx arch without features
    fp = popen("rocminfo 2>/dev/null | grep -m1 'gfx[0-9]*' | grep -oE 'gfx[0-9a-z]+'", "r");
    if (!fp) return "";
    char buf[64] = {0};
    if (fgets(buf, sizeof(buf), fp)) {
        pclose(fp);
        std::string arch(buf);
        while (!arch.empty() && (arch.back() == '\n' || arch.back() == '\r'))
            arch.pop_back();
        return arch;
    }
    pclose(fp);
    return "";
}

std::vector<uint8_t> extractDeviceCode(const std::string& path) {
    MappedFile file(path);
    if (!file.valid()) return {};

    const Elf64_Ehdr* ehdr = file.at<Elf64_Ehdr>(0);
    if (memcmp(ehdr->e_ident, ELFMAG, SELFMAG) != 0) return {};

    // Already device ELF?
    if (ehdr->e_machine == 224) {
        const uint8_t* p = static_cast<const uint8_t*>(file.data());
        return std::vector<uint8_t>(p, p + file.size());
    }

    // Extract from host object
    std::string fatbin = path + ".fatbin.tmp";
    std::string device = path + ".device.tmp";

    std::string cmd1 = "/opt/rocm/llvm/bin/llvm-objcopy --dump-section=.hip_fatbin=\"" + fatbin + "\" \"" + path + "\" 2>/dev/null";
    if (system(cmd1.c_str()) != 0) return {};

    std::string cmd2 = "/opt/rocm/llvm/bin/clang-offload-bundler --type=o --targets=hipv4-amdgcn-amd-amdhsa--" +
                       g_target_arch + " --input=\"" + fatbin + "\" --output=\"" + device + "\" --unbundle 2>/dev/null";
    int ret = system(cmd2.c_str());
    unlink(fatbin.c_str());
    if (ret != 0) return {};

    std::vector<uint8_t> result;
    MappedFile df(device);
    if (df.valid()) {
        const uint8_t* p = static_cast<const uint8_t*>(df.data());
        result.assign(p, p + df.size());
    }
    unlink(device.c_str());
    return result;
}

struct KernelInfo {
    std::string name;
    std::string source_file;  // Original file path for error reporting
    std::vector<uint8_t> code;
    uint64_t func_offset = 0;       // Offset of ncclDevFunc_ within code (for function table)
    std::vector<uint8_t> kd;        // 64-byte kernel descriptor from .rodata
    std::vector<uint8_t> note;      // kernel's .note section
    std::vector<uint8_t> debug_line; // .debug_line section (if compiled with -gline-tables-only)
    std::vector<uint8_t> debug_line_str; // .debug_line_str section (DWARF5 string table)
    // Additional DWARF sections for llvm-objdump --source support
    std::vector<uint8_t> debug_abbrev;
    std::vector<uint8_t> debug_info;
    std::vector<uint8_t> debug_str;
    std::vector<uint8_t> debug_str_offsets;
    std::vector<uint8_t> debug_addr;
    std::vector<uint8_t> debug_rnglists;
    std::vector<uint8_t> debug_ranges;  // Legacy .debug_ranges (DWARF5 uses .debug_rnglists)
    uint16_t dwarf_version = 0;        // DWARF version from first CU (0 if no .debug_info); device linker supports 4 or 5
    DwarfAttrPositions dwarf_attr_positions;  // Positions of various DWARF attributes that need patching
    uint64_t orig_text_addr = 0;    // Original .text address for debug_line patching
    int vgpr = 0, sgpr = 0, lds = 0, stack = 0;
    // Symbol info from .dynsym (to preserve compiler's binding/visibility)
    uint8_t dynsym_st_info = 0;     // st_info (binding + type) from original .dynsym
    uint8_t dynsym_st_other = 0;    // st_other (visibility) from original .dynsym
    uint64_t dynsym_st_size = 0;    // st_size from original .dynsym
};

// Read one msgpack integer at data[off]: fixint, uint8 (0xcc), or uint16 (0xcd). Returns value and bytes consumed (0 if invalid).
static int readMsgpackIntAt(const uint8_t* data, size_t size, size_t off, size_t* out_consumed) {
    if (off >= size) { *out_consumed = 0; return 0; }
    uint8_t b = data[off];
    if (b <= 0x7f) { *out_consumed = 1; return b; }
    if (b == 0xcc && off + 1 < size) { *out_consumed = 2; return data[off + 1]; }
    if (b == 0xcd && off + 2 < size) { *out_consumed = 3; return ((uint32_t)data[off + 1] << 8) | data[off + 2]; }
    *out_consumed = 0;
    return 0;
}

// Find all occurrences of key in note data and return the maximum integer value after each key.
static int maxIntInNote(const std::vector<uint8_t>& data, const char* key) {
    size_t key_len = strlen(key);
    if (data.size() < key_len + 1) return 0;
    std::string_view view((const char*)data.data(), data.size());
    int max_val = 0;
    size_t pos = 0;
    while (pos + key_len < data.size()) {
        size_t found = view.find(key, pos);
        if (found == std::string_view::npos) break;
        size_t val_off = found + key_len;
        size_t consumed = 0;
        int v = readMsgpackIntAt(data.data(), data.size(), val_off, &consumed);
        if (v > max_val) max_val = v;
        pos = consumed ? val_off + consumed : val_off + 1;
    }
    return max_val;
}

// Max of first max_count occurrences of key (0 = no limit). Used to exclude oneRank from dispatcher LDS.
static int maxIntInNoteFirstN(const std::vector<uint8_t>& data, const char* key, int max_count) {
    if (max_count <= 0) return maxIntInNote(data, key);
    size_t key_len = strlen(key);
    if (data.size() < key_len + 1) return 0;
    std::string_view view((const char*)data.data(), data.size());
    int max_val = 0;
    size_t pos = 0;
    int count = 0;
    while (count < max_count && pos + key_len < data.size()) {
        size_t found = view.find(key, pos);
        if (found == std::string_view::npos) break;
        size_t val_off = found + key_len;
        size_t consumed = 0;
        int v = readMsgpackIntAt(data.data(), data.size(), val_off, &consumed);
        if (v > max_val) max_val = v;
        count++;
        pos = consumed ? val_off + consumed : val_off + 1;
    }
    return max_val;
}

// Find occurrences of key in note data, print each with index (up to max_kernels; 0 = all), return values. If any differ in that set, warn.
static std::vector<int> printAllLDSInNote(const std::vector<uint8_t>& data, const char* key, const char* label, int max_kernels = 0) {
    std::vector<int> values;
    size_t key_len = strlen(key);
    if (data.size() < key_len + 1) return values;
    std::string_view view((const char*)data.data(), data.size());
    size_t pos = 0;
    int idx = 0;
    while (pos + key_len < data.size()) {
        if (max_kernels > 0 && idx >= max_kernels) break;
        size_t found = view.find(key, pos);
        if (found == std::string_view::npos) break;
        size_t val_off = found + key_len;
        size_t consumed = 0;
        int v = readMsgpackIntAt(data.data(), data.size(), val_off, &consumed);
        printf("  .note LDS %s kernel %d: %d\n", label, idx, v);
        values.push_back(v);
        idx++;
        pos = consumed ? val_off + consumed : val_off + 1;
    }
    if (values.size() > 1) {
        int first = values[0];
        for (size_t i = 1; i < values.size(); i++) {
            if (values[i] != first) {
                printf("WARNING: %s has different LDS values (e.g. kernel 0=%d, kernel %zu=%d)\n",
                       label, first, i, values[i]);
                break;
            }
        }
    }
    return values;
}

KernelInfo parseKernel(const std::vector<uint8_t>& elf_data, const std::string& source_file = "") {
    KernelInfo info;
    info.source_file = source_file;
    if (elf_data.empty()) return info;

    std::string tmp = "/tmp/kern_" + std::to_string(rand()) + ".o";
    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f) return info;
    fwrite(elf_data.data(), 1, elf_data.size(), f);
    fclose(f);

    MappedFile file(tmp);
    ElfParser elf(file);
    if (!elf.valid()) { unlink(tmp.c_str()); return info; }

    // Find ncclDevFunc symbol
    auto* symtab = elf.find(".symtab");
    auto* strtab = elf.find(".strtab");
    auto* text = elf.find(".text");
    if (!symtab || !strtab || !text) { unlink(tmp.c_str()); return info; }

    const char* strings = file.str(strtab->offset);
    const Elf64_Sym* syms = file.at<Elf64_Sym>(symtab->offset);
    size_t nsyms = symtab->size / sizeof(Elf64_Sym);

    // Require exactly one ncclDevFunc_ - the top-level device function (must have noinline in DEFINE_ncclDevFunc).
    // We do NOT fall back to ncclDevKernel_; the kernel ends with s_endpgm and cannot be called as a function.
    int nccl_devfunc_count = 0;
    size_t nccl_devfunc_idx = 0;
    for (size_t i = 0; i < nsyms; i++) {
        if (ELF64_ST_TYPE(syms[i].st_info) == STT_FUNC) {
            const char* n = strings + syms[i].st_name;
            if (strstr(n, "ncclDevFunc_")) {
                nccl_devfunc_count++;
                nccl_devfunc_idx = i;
            }
        }
    }
    if (nccl_devfunc_count != 1) {
        fprintf(stderr, "Error: %s: expected exactly one ncclDevFunc_ symbol, found %d.\n",
                source_file.empty() ? "(object)" : source_file.c_str(), nccl_devfunc_count);
        unlink(tmp.c_str());
        exit(1);
    }
    {
        const char* n = strings + syms[nccl_devfunc_idx].st_name;
        info.name = n;
        uint64_t func_start = syms[nccl_devfunc_idx].st_value;
        uint64_t func_end = func_start + syms[nccl_devfunc_idx].st_size;
        uint64_t extract_size = func_end - text->addr;

        const uint8_t* p = file.at<uint8_t>(text->offset);
        info.code.assign(p, p + extract_size);
        info.func_offset = func_start - text->addr;  // Offset of ncclDevFunc_ within extracted code
    }

    // Parse .note for resources and save full note
    auto* note = elf.find(".note");
    if (note) {
        auto data = elf.getBytes(*note);
        info.note = data;  // Save full note section

        std::string_view view((char*)data.data(), data.size());

        auto findInt = [&](const char* key) -> int {
            auto pos = view.find(key);
            if (pos == std::string_view::npos) return 0;
            pos += strlen(key);
            if (pos >= view.size()) return 0;
            uint8_t b = view[pos];
            if (b <= 0x7f) return b;
            if (b == 0xcc && pos + 1 < view.size()) return (uint8_t)view[pos + 1];
            if (b == 0xcd && pos + 2 < view.size()) return ((uint8_t)view[pos + 1] << 8) | (uint8_t)view[pos + 2];
            return 0;
        };

        info.vgpr = findInt(".vgpr_count");
        info.sgpr = findInt(".sgpr_count");
        info.lds = findInt(".group_segment_fixed_size");
        info.stack = findInt(".private_segment_fixed_size");
        printf("  .note LDS (%s): %d\n", info.name.empty() ? "(no name)" : info.name.c_str(), info.lds);
    }

    // Extract kernel descriptor from .rodata (64 bytes)
    auto* rodata = elf.find(".rodata");
    if (rodata && rodata->size >= 64) {
        auto data = elf.getBytes(*rodata);
        info.kd.assign(data.begin(), data.begin() + 64);
    }

    // Extract .debug_line section if present (compiled with -gline-tables-only)
    auto* debug_line = elf.find(".debug_line");
    if (debug_line && debug_line->size > 0) {
        info.debug_line = elf.getBytes(*debug_line);
        info.orig_text_addr = text->addr;  // Save original .text address for patching

        // Also extract .debug_line_str (DWARF5 string table for file names)
        auto* debug_line_str = elf.find(".debug_line_str");
        if (debug_line_str && debug_line_str->size > 0) {
            info.debug_line_str = elf.getBytes(*debug_line_str);
        }

        // Apply .rela.debug_line relocations to resolve string offsets
        // DWARF5 .debug_line sections have R_X86_64_32 relocations for DW_FORM_line_strp offsets
        auto* rela_debug_line = elf.find(".rela.debug_line");
        if (rela_debug_line && rela_debug_line->size > 0 && rela_debug_line->entsize >= sizeof(Elf64_Rela)) {
            const Elf64_Rela* relas = elf.fileAt<Elf64_Rela>(rela_debug_line->offset);
            size_t num_relas = rela_debug_line->size / sizeof(Elf64_Rela);

            for (size_t i = 0; i < num_relas; i++) {
                const Elf64_Rela& r = relas[i];
                uint32_t rtype = ELF64_R_TYPE(r.r_info);

                // R_X86_64_32 (type 10) writes a 32-bit value (addend only for section symbols)
                // R_AMDGPU_ABS32 (type 3) is similar for AMDGPU
                if ((rtype == 10 || rtype == 3) && r.r_offset + 4 <= info.debug_line.size()) {
                    // Write the addend as the 32-bit offset into .debug_line_str
                    uint32_t val = (uint32_t)r.r_addend;
                    memcpy(&info.debug_line[r.r_offset], &val, 4);
                }
                // R_X86_64_64 (type 1) / R_AMDGPU_ABS64 (type 2) for .text addresses
                else if ((rtype == 1 || rtype == 2) && r.r_offset + 8 <= info.debug_line.size()) {
                    // Write the addend (relative to .text start)
                    uint64_t val = (uint64_t)r.r_addend;
                    memcpy(&info.debug_line[r.r_offset], &val, 8);
                }
            }
        }

        // Extract additional DWARF sections for llvm-objdump --source
        auto* debug_abbrev = elf.find(".debug_abbrev");
        if (debug_abbrev && debug_abbrev->size > 0) {
            info.debug_abbrev = elf.getBytes(*debug_abbrev);
        }
        auto* debug_info = elf.find(".debug_info");
        if (debug_info && debug_info->size > 0) {
            info.debug_info = elf.getBytes(*debug_info);
            info.dwarf_version = getDwarfVersionFromDebugInfo(info.debug_info.data(), info.debug_info.size());
        }
        auto* debug_str = elf.find(".debug_str");
        if (debug_str && debug_str->size > 0) {
            info.debug_str = elf.getBytes(*debug_str);
        }
        auto* debug_str_offsets = elf.find(".debug_str_offsets");
        if (debug_str_offsets && debug_str_offsets->size > 0) {
            info.debug_str_offsets = elf.getBytes(*debug_str_offsets);
        }
        auto* debug_addr = elf.find(".debug_addr");
        if (debug_addr && debug_addr->size > 0) {
            info.debug_addr = elf.getBytes(*debug_addr);
        }
        auto* debug_rnglists = elf.find(".debug_rnglists");
        if (debug_rnglists && debug_rnglists->size > 0) {
            info.debug_rnglists = elf.getBytes(*debug_rnglists);
        }

        // Apply relocations to debug sections before merging
        // These relocations resolve section-relative offsets that need to be applied
        auto applyRelocations = [&](const char* rela_name, std::vector<uint8_t>& section_data) {
            auto* rela_sec = elf.find(rela_name);
            if (!rela_sec || rela_sec->size == 0 || rela_sec->entsize < sizeof(Elf64_Rela)) return;
            
            const Elf64_Rela* relas = elf.fileAt<Elf64_Rela>(rela_sec->offset);
            size_t num_relas = rela_sec->size / sizeof(Elf64_Rela);
            
            for (size_t i = 0; i < num_relas; i++) {
                const Elf64_Rela& r = relas[i];
                uint32_t rtype = ELF64_R_TYPE(r.r_info);
                
                // Apply relocation if offset is within section bounds
                // For section-relative relocations (R_AMDGPU_ABS32 type 6, R_X86_64_32 type 10, R_AMDGPU_ABS32 type 3),
                // the addend contains the offset into the target section
                if (r.r_offset + 4 <= section_data.size()) {
                    if (rtype == 10 || rtype == 3 || rtype == 6) {  // R_X86_64_32, R_AMDGPU_ABS32 (various types) - 32-bit
                        uint32_t val = (uint32_t)r.r_addend;
                        memcpy(&section_data[r.r_offset], &val, 4);
                    } else if (rtype == 1 || rtype == 2) {  // R_X86_64_64 or R_AMDGPU_ABS64 - 64-bit
                        if (r.r_offset + 8 <= section_data.size()) {
                            uint64_t val = (uint64_t)r.r_addend;
                            memcpy(&section_data[r.r_offset], &val, 8);
                        }
                    }
                }
            }
        };

        // Apply relocations to each debug section
        if (!info.debug_info.empty()) {
            applyRelocations(".rela.debug_info", info.debug_info);
        }
        if (!info.debug_str_offsets.empty()) {
            applyRelocations(".rela.debug_str_offsets", info.debug_str_offsets);
        }
        if (!info.debug_addr.empty()) {
            applyRelocations(".rela.debug_addr", info.debug_addr);
        }
        auto* debug_ranges = elf.find(".debug_ranges");
        if (debug_ranges && debug_ranges->size > 0) {
            info.debug_ranges = elf.getBytes(*debug_ranges);
        }

        // Use LLVM to find DWARF attribute positions (minimal ELF: kernel .o files often have bad .debug_str_offsets)
        if (!info.debug_info.empty() && !info.debug_abbrev.empty()) {
            info.dwarf_attr_positions = findDwarfAttrPositionsUsingLLVM(info.debug_info, info.debug_abbrev,
                                                                        info.debug_str, info.debug_line, 
                                                                        info.debug_line_str, info.debug_addr);
        }
    }

    unlink(tmp.c_str());
    return info;
}

// ============================================================================
// FuncId mapping
// ============================================================================

struct FuncIdMapping { int id; int unroll; };

std::unordered_map<std::string, FuncIdMapping> parseHostTable(const std::string& path) {
    std::unordered_map<std::string, FuncIdMapping> map;
    std::ifstream file(path);
    if (!file) return map;

    std::regex pat(R"(\{(\d+),\s*(\d+)\},\s*//\s*(.+))");
    std::string line;

    while (std::getline(file, line)) {
        std::smatch m;
        if (std::regex_search(line, m, pat)) {
            int id = std::stoi(m[2]);
            std::istringstream iss(m[3]);
            std::vector<std::string> parts;
            std::string p;
            while (iss >> p) parts.push_back(p);

            if (parts.size() >= 8) {
                std::string key = parts[0] + "_" + parts[1] + "_" + parts[2] + "_" +
                                  parts[3] + "_" + parts[4] + "_" + parts[5] + "_" + parts[6];
                map[key] = {id, std::stoi(parts[7])};
            }
        }
    }
    return map;
}

std::pair<std::string, int> demangleFunc(const std::string& mangled) {
    if (mangled.substr(0, 2) != "_Z") return {"", 0};
    size_t i = 2;
    while (i < mangled.size() && isdigit(mangled[i])) i++;
    if (i == 2) return {"", 0};

    int len = std::stoi(mangled.substr(2, i - 2));
    std::string name = mangled.substr(i, len);

    bool isSpecialized = false;

    // Strip prefix: ncclDevFunc_ (12 chars) or ncclDevKernel_ (14 chars)
    if (name.substr(0, 14) == "ncclDevKernel_") {
        name = name.substr(14);
        isSpecialized = true;
    } else if (name.substr(0, 12) == "ncclDevFunc_") {
        name = name.substr(12);
    } else {
        return {"", 0};
    }

    // Split by underscore
    std::vector<std::string> parts;
    std::istringstream iss(name);
    std::string part;
    while (std::getline(iss, part, '_')) parts.push_back(part);

    int unroll = 0;
    std::string key;

    if (isSpecialized) {
        // Format: AllReduce_RING_LL128_Sum_f32_acc1_unroll2_Specialized
        // or: AllReduce_RING_LL128_Sum_f32_unroll2_Specialized (no acc)
        int acc = 0;
        std::vector<std::string> filtered;
        for (const auto& p : parts) {
            if (p == "Specialized") continue;
            if (p.size() > 6 && p.substr(0, 6) == "unroll") {
                unroll = std::stoi(p.substr(6));
                continue;
            }
            if (p.size() > 3 && p.substr(0, 3) == "acc") {
                acc = std::stoi(p.substr(3));
                continue;
            }
            filtered.push_back(p);
        }
        if (filtered.size() < 5) return {"", 0};

        // Build key: Coll_Algo_Proto_Redop_Type_Acc_Pipeline
        for (size_t j = 0; j < filtered.size(); j++) {
            if (j > 0) key += "_";
            key += filtered[j];
        }
        key += "_" + std::to_string(acc) + "_0";
    } else {
        // Format: AllGather_PAT_LL128_Sum_i8_0_0_2
        // Last element is unroll, key is everything else
        if (parts.size() < 8) return {"", 0};

        unroll = std::stoi(parts.back());

        // Key is all parts except the last (unroll)
        for (size_t j = 0; j < parts.size() - 1; j++) {
            if (j > 0) key += "_";
            key += parts[j];
        }
    }

    return {key, unroll};
}

// ============================================================================
// Device Linker - Two-Pass Design
// ============================================================================

// Forward declaration
class DeviceLinker;

// Custom AddressesMap for DeviceLinker that maps original addresses to new addresses
class DeviceLinkerAddressesMap : public llvm::dwarf_linker::AddressesMap {
    uint64_t orig_text_start_;  // Original .text address in input object
    uint64_t new_text_start_;    // New .text address in merged output
    uint64_t text_size_;         // Size of .text section
    bool has_relocs_;            // Whether relocations are valid
    
public:
    DeviceLinkerAddressesMap(uint64_t orig_start, uint64_t new_start, uint64_t size)
        : orig_text_start_(orig_start), new_text_start_(new_start),
          text_size_(size), has_relocs_(true) {}
    
    bool hasValidRelocs() override {
        return has_relocs_;
    }
    
    std::optional<int64_t> getSubprogramRelocAdjustment(
        const llvm::DWARFDie& DIE, bool Verbose) override {
        // Get address from DIE's DW_AT_low_pc attribute
        auto low_pc = DIE.find(llvm::dwarf::DW_AT_low_pc);
        if (!low_pc) return std::nullopt;
        
        if (std::optional<uint64_t> addr_opt = low_pc->getAsAddress()) {
            uint64_t addr = *addr_opt;
            // Check if address is within this object's .text section
            if (addr >= orig_text_start_ && addr < orig_text_start_ + text_size_) {
                int64_t adjustment = (int64_t)new_text_start_ - (int64_t)orig_text_start_;
                if (Verbose) {
                    fprintf(stderr, "  Address map: subprogram at 0x%lx -> adjustment 0x%lx\n",
                           addr, (uint64_t)adjustment);
                }
                return adjustment;
            }
        }
        return std::nullopt;
    }
    
    std::optional<int64_t> getExprOpAddressRelocAdjustment(
        llvm::DWARFUnit& U, const llvm::DWARFExpression::Operation& Op,
        uint64_t StartOffset, uint64_t EndOffset, bool Verbose) override {
        // Handle address operations in DWARF expressions (DW_OP_addr, DW_OP_addrx, etc.)
        uint64_t addr = 0;
        bool has_addr = false;
        
        switch (Op.getCode()) {
            case llvm::dwarf::DW_OP_addr: {
                // DW_OP_addr has the address as an immediate operand
                if (Op.getRawOperands().size() >= U.getAddressByteSize()) {
                    addr = 0;
                    // Read address in little-endian format (AMDGPU uses little-endian)
                    for (unsigned i = 0; i < U.getAddressByteSize() && i < Op.getRawOperands().size(); i++) {
                        addr |= ((uint64_t)Op.getRawOperands()[i]) << (i * 8);
                    }
                    has_addr = true;
                }
                break;
            }
            case llvm::dwarf::DW_OP_addrx:
            case llvm::dwarf::DW_OP_constx: {
                // DW_OP_addrx/DW_OP_constx reference .debug_addr table via index
                if (Op.getRawOperands().size() > 0) {
                    uint64_t index = Op.getRawOperands()[0];
                    // Get the address offset in .debug_addr section
                    // For DW_OP_addrx, we can't easily get the actual address here without
                    // parsing .debug_addr. Return adjustment if index offset exists.
                    // DWARFLinker will handle the actual address patching.
                    if (std::optional<uint64_t> addr_offset = U.getIndexedAddressOffset(index)) {
                        // Assume addresses in .debug_addr are valid if index exists
                        // Return adjustment - DWARFLinker will verify the address
                        int64_t adjustment = (int64_t)new_text_start_ - (int64_t)orig_text_start_;
                        if (Verbose) {
                            fprintf(stderr, "  Address map: expression op addrx index %lu -> adjustment 0x%lx\n",
                                   index, (uint64_t)adjustment);
                        }
                        return adjustment;
                    }
                }
                break;
            }
            default:
                return std::nullopt;
        }
        
        if (has_addr) {
            // Check if address is within this object's .text section
            if (addr >= orig_text_start_ && addr < orig_text_start_ + text_size_) {
                int64_t adjustment = (int64_t)new_text_start_ - (int64_t)orig_text_start_;
                if (Verbose) {
                    fprintf(stderr, "  Address map: expression op at 0x%lx -> adjustment 0x%lx\n",
                           addr, (uint64_t)adjustment);
                }
                return adjustment;
            }
        }
        
        return std::nullopt;
    }
    
    std::optional<llvm::StringRef> getLibraryInstallName() override {
        // Not applicable for device linker (no shared libraries)
        return std::nullopt;
    }
    
    bool applyValidRelocs(llvm::MutableArrayRef<char> Data, uint64_t BaseOffset,
                         bool IsLittleEndian) override {
        // DWARFLinker handles relocations internally, so we don't need to apply them here
        (void)Data;
        (void)BaseOffset;
        (void)IsLittleEndian;
        return false;
    }
    
    bool needToSaveValidRelocs() override {
        // We don't need to save relocations - DWARFLinker handles them
        return false;
    }
    
    void updateAndSaveValidRelocs(bool IsDWARF5,
                                 uint64_t OriginalUnitOffset,
                                 int64_t LinkedOffset,
                                 uint64_t StartOffset,
                                 uint64_t EndOffset) override {
        // Not needed - we don't save relocations
        (void)IsDWARF5;
        (void)OriginalUnitOffset;
        (void)LinkedOffset;
        (void)StartOffset;
        (void)EndOffset;
    }
    
    void updateRelocationsWithUnitOffset(uint64_t OriginalUnitOffset,
                                       uint64_t OutputUnitOffset) override {
        // Not needed - we don't track relocations by unit offset
        (void)OriginalUnitOffset;
        (void)OutputUnitOffset;
    }
    
    void clear() override {
        // Nothing to clear - addresses are fixed at construction time
    }
};

class DeviceLinker {
public:
    DeviceLinker(const std::string& dispatcher_path) : disp_path_(dispatcher_path) {}

    bool load() {
        disp_file_ = std::make_unique<MappedFile>(disp_path_);
        if (!disp_file_->valid()) return false;
        disp_ = std::make_unique<ElfParser>(*disp_file_);
        if (!disp_->valid()) return false;

        // Set debug compilation directory based on dispatcher path
        // If dispatcher is at X/device_linker_output/dispatcher.elf, comp_dir is X
        // Use realpath to get absolute path for GDB source lookup
        size_t pos = disp_path_.rfind("/device_linker_output/");
        if (pos != std::string::npos) {
            std::string rel_dir = disp_path_.substr(0, pos);
            char* abs_path = realpath(rel_dir.c_str(), nullptr);
            if (abs_path) {
                debug_comp_dir_ = abs_path;
                free(abs_path);
            } else {
                debug_comp_dir_ = rel_dir;  // fallback to relative
            }
            printf("Debug compilation directory: %s\n", debug_comp_dir_.c_str());
        }

        return true;
    }

    // Set GPU target and parse architecture number (e.g., "gfx942:sramecc+:xnack-" -> 942)
    void setGpuTarget(const std::string& target) {
        gpu_target_ = target;
        // Parse architecture number from target string
        std::regex arch_regex("gfx([0-9]+)");
        std::smatch match;
        if (std::regex_search(target, match, arch_regex) && match.size() > 1) {
            gpu_arch_ = std::stoi(match[1].str());
            printf("Detected GPU architecture: gfx%d\n", gpu_arch_);
        } else {
            printf("Warning: Could not parse GPU arch from '%s', using default gfx%d\n",
                   target.c_str(), gpu_arch_);
        }
    }

    void addKernel(const KernelInfo& k) { kernels_.push_back(k); }
    void setFuncIdMap(const std::unordered_map<std::string, FuncIdMapping>& m) { funcid_map_ = m; }
    void setOmitDwarf(bool v) { omit_dwarf_ = v; }

    bool link(const std::string& output_path) {
        fatal_error_ = false;
        printf("=== Pass 1: Collect and Size ===\n");
        if (!collectSections()) return false;

        printf("\n=== Pass 2: Layout ===\n");
        computeLayout();

        // Merge and patch debug sections after layout (needs text_addr_)
        if (!omit_dwarf_) {
            printf("\n=== Pass 2b: Merge Debug Info ===\n");
            mergeDebugInfoWithDWARFLinker();
            if (fatal_error_) return false;
        } else {
            printf("\n=== Pass 2b: Omitting DWARF (--omit-dwarf) ===\n");
        }

        printf("\n=== Pass 3: Patch and Write ===\n");
        patchSections();
        if (fatal_error_) return false;

        return writeOutput(output_path);
    }

    // Write header file with funcId -> mangled name mapping for host-side tracing
    bool writeFuncIdHeader(const std::string& path) {
        FILE* f = fopen(path.c_str(), "w");
        if (!f) { fprintf(stderr, "Cannot write %s\n", path.c_str()); return false; }

        fprintf(f, "// Auto-generated by device_linker - funcId to mangled function name mapping\n");
        fprintf(f, "// Include this file for host-side tracing of device function calls\n");
        fprintf(f, "#pragma once\n\n");
        fprintf(f, "#define NCCL_FUNC_COUNT %d\n\n", FUNC_COUNT);

        // Write arrays for each unroll factor
        auto writeTable = [&](const char* name, const std::vector<std::string>& names) {
            fprintf(f, "static const char* %s[NCCL_FUNC_COUNT] = {\n", name);
            for (int i = 0; i < FUNC_COUNT; i++) {
                if (names[i].empty()) {
                    fprintf(f, "    nullptr,  // %d\n", i);
                } else {
                    fprintf(f, "    \"%s\",  // %d\n", names[i].c_str(), i);
                }
            }
            fprintf(f, "};\n\n");
        };

        writeTable("ncclDevFuncNames_1", names_1_);
        writeTable("ncclDevFuncNames_2", names_2_);
        writeTable("ncclDevFuncNames_4", names_4_);

        // Write a helper function
        fprintf(f, "// Helper to get function name for a given funcId and unroll factor\n");
        fprintf(f, "static inline const char* ncclDevFuncName(int funcId, int unroll) {\n");
        fprintf(f, "    if (funcId < 0 || funcId >= NCCL_FUNC_COUNT) return nullptr;\n");
        fprintf(f, "    switch (unroll) {\n");
        fprintf(f, "        case 1: return ncclDevFuncNames_1[funcId];\n");
        fprintf(f, "        case 2: return ncclDevFuncNames_2[funcId];\n");
        fprintf(f, "        case 4: return ncclDevFuncNames_4[funcId];\n");
        fprintf(f, "        default: return nullptr;\n");
        fprintf(f, "    }\n");
        fprintf(f, "}\n");

        fclose(f);
        printf("Wrote %s: funcId -> name mapping for %d functions\n", path.c_str(), FUNC_COUNT);
        return true;
    }

private:
    std::string disp_path_;
    bool omit_dwarf_ = false;
    std::unique_ptr<MappedFile> disp_file_;
    std::unique_ptr<ElfParser> disp_;
    std::vector<KernelInfo> kernels_;
    std::vector<std::pair<const KernelInfo*, uint64_t>> kernel_text_offsets_;  // kernel -> text offset
    std::vector<std::pair<size_t, uint64_t>> specialized_kd_offsets_;  // KD offset in .rodata -> text offset
    std::vector<uint64_t> onerank_text_offsets_;  // oneRankReduce kernel offsets in dispatcher .text
    std::vector<std::string> onerank_names_;      // oneRankReduce kernel names (parallel to onerank_text_offsets_)
    std::unordered_map<std::string, FuncIdMapping> funcid_map_;

    // Pre-allocation tracking for .dynsym/.dynstr (set during collectSections, used in patchSections)
    size_t prealloc_dynstr_orig_size_ = 0;     // Original .dynstr size before pre-allocation
    size_t prealloc_dynsym_orig_count_ = 0;    // Original .dynsym symbol count before pre-allocation
    size_t prealloc_dynsym_insert_idx_ = 0;    // Index where new LOCAL symbols were pre-allocated

    // Debug line info: (offset in merged .debug_line, size, orig_text_addr, new_text_offset)
    struct DebugLineChunk {
        size_t merged_offset;      // Offset in merged .debug_line section
        size_t size;               // Size of this chunk
        uint64_t orig_text_addr;   // Original .text address in kernel's ELF
        uint64_t new_text_offset;  // New offset in merged .text section
        size_t str_offset_base;    // Base offset for this chunk's strings in merged .debug_line_str
        size_t orig_str_size;      // Original .debug_line_str size for this kernel
        std::string comp_dir;      // Compilation directory from .debug_line header
    };
    std::vector<DebugLineChunk> debug_line_chunks_;
    std::vector<uint8_t> merged_debug_line_str_;  // Merged .debug_line_str data
    std::string debug_comp_dir_;  // Compilation directory for debug info

    // Merged debug sections for proper DWARF merging
    struct DebugInfoChunk {
        size_t merged_offset;      // Offset in merged section
        size_t size;               // Size of this chunk
        uint64_t orig_text_addr;   // Original .text address in kernel's ELF
        uint64_t new_text_offset;  // New offset in merged .text section
        size_t abbrev_base;        // Base offset in merged .debug_abbrev
        size_t str_base;           // Base offset in merged .debug_str
        size_t str_offsets_base;   // Base offset in merged .debug_str_offsets
        size_t addr_base;          // Base offset in merged .debug_addr
        size_t rnglists_base;      // Base offset in merged .debug_rnglists
        size_t ranges_base;        // Base offset in merged .debug_ranges (legacy; DWARF5 uses .debug_rnglists)
        size_t line_base;          // Base offset in merged .debug_line
        size_t line_str_offset_base;  // Base offset in merged .debug_line_str (for patching DW_FORM_line_strp in .debug_info)
        DwarfAttrPositions dwarf_attr_positions;  // Positions of DWARF attributes in this chunk
        uint16_t dwarf_version;    // 4 or 5 (CU header layout and attributes differ)
        std::string source_file;   // Original file path for error reporting
    };
    std::vector<DebugInfoChunk> debug_info_chunks_;
    std::vector<uint8_t> merged_debug_abbrev_;
    std::vector<uint8_t> merged_debug_str_;
    std::vector<uint8_t> merged_debug_str_offsets_;
    std::vector<uint8_t> merged_debug_addr_;
    std::vector<uint8_t> merged_debug_rnglists_;
    std::vector<uint8_t> merged_debug_ranges_;  // Legacy .debug_ranges (DWARF5 prefers .debug_rnglists)
    std::vector<uint8_t> merged_debug_info_;

    // Output sections
    std::vector<SectionInfo> sections_;
    uint32_t elf_flags_ = 0;

    // Resource maxima
    int max_vgpr_ = 0, max_sgpr_ = 0, max_lds_ = 0, max_stack_ = 0;

    // GPU target info for LDS calculation
    std::string gpu_target_;
    int gpu_arch_ = 942;  // Default gfx942

    // Function tables (text offsets per funcId)
    std::vector<uint64_t> table_1_, table_2_, table_4_;

    // Function names for each funcId (for debugging/tracing)
    std::vector<std::string> names_1_, names_2_, names_4_;

    // Function table offsets within .rodata (for const tables)
    uint64_t rodata_table_1_off_ = 0;
    uint64_t rodata_table_2_off_ = 0;
    uint64_t rodata_table_4_off_ = 0;

    // Addresses computed during layout
    uint64_t text_addr_ = 0;
    uint64_t rodata_addr_ = 0;       // .rodata section
    uint64_t data_addr_ = 0;
    uint64_t data_rel_ro_addr_ = 0;  // Function tables (if not in .rodata)
    uint64_t dyn_addr_ = 0;          // .dynamic section
    uint64_t relro_end_ = 0;         // End of RELRO segment
    uint64_t rela_addr_ = 0;
    size_t rela_size_ = 0;
    size_t table_spacing_ = 6880;

    bool fatal_error_ = false;
    void setFatalError(const char* msg = nullptr) {
        fatal_error_ = true;
        if (msg) fprintf(stderr, "Error: %s\n", msg);
    }

    // ========== Pass 1: Collect ==========
    bool collectSections() {
        elf_flags_ = disp_->ehdr()->e_flags;

        // Include dispatcher .note in max. Only first 6 kernels (Generic + Debug) have LDS; oneRank (6-17) excluded.
        static const int DISPATCHER_LDS_KERNEL_COUNT = 6;
        // Allow 0 in dispatcher LDS for current debug kernels; remove when fixed.
        static const bool DISPATCHER_LDS_ALLOW_ZERO = true;
        auto* disp_note = disp_->find(".note");
        if (!disp_note) {
            fprintf(stderr, "Error: dispatcher has no .note section\n");
            return false;
        }
        int disp_lds_max = 0;
        {
            std::vector<uint8_t> note_data = disp_->getBytes(*disp_note);
            std::vector<int> disp_lds_vals = printAllLDSInNote(note_data, ".group_segment_fixed_size", "dispatcher", DISPATCHER_LDS_KERNEL_COUNT);
            int d_vgpr = maxIntInNote(note_data, ".vgpr_count");
            int d_sgpr = maxIntInNote(note_data, ".sgpr_count");
            int d_lds = maxIntInNoteFirstN(note_data, ".group_segment_fixed_size", DISPATCHER_LDS_KERNEL_COUNT);
            int d_stack = maxIntInNote(note_data, ".private_segment_fixed_size");
            disp_lds_max = d_lds;
            max_vgpr_ = std::max(max_vgpr_, d_vgpr);
            max_sgpr_ = std::max(max_sgpr_, d_sgpr);
            max_lds_ = std::max(max_lds_, d_lds);
            max_stack_ = std::max(max_stack_, d_stack);
            // Check: first 6 dispatcher kernels should all be same size or 0 (0 allowed for debug kernels only).
            int nonzero = -1;
            for (int v : disp_lds_vals) {
                if (v != 0) {
                    if (nonzero >= 0 && v != nonzero) {
                        printf("RED FLAG: dispatcher LDS inconsistent (expected all same or 0): saw %d and %d\n", nonzero, v);
                        break;
                    }
                    nonzero = v;
                }
            }
            if (!DISPATCHER_LDS_ALLOW_ZERO && nonzero >= 0) {
                for (int v : disp_lds_vals)
                    if (v == 0) { printf("RED FLAG: dispatcher kernel has LDS 0 (DISPATCHER_LDS_ALLOW_ZERO will be removed when fixed)\n"); break; }
            }
            printf("Max resources from dispatcher .note (LDS from first %d kernels only): VGPR=%d, SGPR=%d, LDS=%d, Stack=%d\n",
                   DISPATCHER_LDS_KERNEL_COUNT, d_vgpr, d_sgpr, d_lds, d_stack);
        }

        // Compute max resources from specialized kernels. Take max over dispatcher and all specialized.
        // Check: we expect all sizes the same; smaller than dispatcher = red flag, bigger = warn but take max.
        int first_spec_lds = -1;
        for (const auto& k : kernels_) {
            max_vgpr_ = std::max(max_vgpr_, k.vgpr);
            max_sgpr_ = std::max(max_sgpr_, k.sgpr);
            max_lds_ = std::max(max_lds_, k.lds);
            max_stack_ = std::max(max_stack_, k.stack);
            if (first_spec_lds < 0) first_spec_lds = k.lds;
            else if (k.lds != first_spec_lds)
                printf("WARNING: specialized kernel LDS differs (%s: %d vs first %d)\n", k.name.c_str(), k.lds, first_spec_lds);
            if (disp_lds_max >= 0 && k.lds < disp_lds_max)
                printf("RED FLAG: kernel %s LDS %d < dispatcher %d\n", k.name.c_str(), k.lds, disp_lds_max);
            if (disp_lds_max >= 0 && k.lds > disp_lds_max)
                printf("WARNING: kernel %s LDS %d > dispatcher %d (using max; kernel may use extra shared memory)\n", k.name.c_str(), k.lds, disp_lds_max);
        }
        if (disp_lds_max >= 0 && !kernels_.empty() && first_spec_lds >= 0 && disp_lds_max != first_spec_lds)
            printf("WARNING: dispatcher .note LDS (%d) != specialized kernel LDS (%d)\n", disp_lds_max, first_spec_lds);
        printf("Max resources from kernels (dispatcher + specialized): VGPR=%d, SGPR=%d, LDS=%d, Stack=%d\n",
               max_vgpr_, max_sgpr_, max_lds_, max_stack_);
        printf("Final LDS size: %d bytes (from .note/KD only)\n", max_lds_);

        // Build .text (dispatcher + all kernel code)
        auto* disp_text = disp_->find(".text");
        if (!disp_text) { fprintf(stderr, "No .text in dispatcher\n"); return false; }

        SectionInfo text;
        text.name = ".text";
        text.type = SHT_PROGBITS;
        text.flags = SHF_ALLOC | SHF_EXECINSTR;
        text.alignment = 256;
        text.data = disp_->getBytes(*disp_text);

        // Align for first function
        while (text.data.size() % FUNC_ALIGNMENT != 0) text.data.push_back(0);

        // Initialize tables
        table_1_.resize(FUNC_COUNT, 0);
        table_2_.resize(FUNC_COUNT, 0);
        table_4_.resize(FUNC_COUNT, 0);
        names_1_.resize(FUNC_COUNT);
        names_2_.resize(FUNC_COUNT);
        names_4_.resize(FUNC_COUNT);

        // Map kernels to tables, append code
        int mapped = 0;
        for (const auto& k : kernels_) {
            if (k.name.empty() || k.code.empty()) continue;

            auto [key, unroll] = demangleFunc(k.name);
            if (key.empty()) continue;

            auto it = funcid_map_.find(key);
            if (it == funcid_map_.end()) continue;

            int funcid = it->second.id;
            if (funcid < 0 || funcid >= FUNC_COUNT) continue;

            // Record address (will be fixed in layout)
            // rel_addr is where the extracted code block starts
            // func_offset is where ncclDevFunc_ is within that block
            uint64_t rel_addr = text.data.size();  // Relative to .text start
            uint64_t func_addr = rel_addr + k.func_offset;  // Where ncclDevFunc_ actually is

            switch (unroll) {
                case 1: table_1_[funcid] = func_addr; names_1_[funcid] = k.name; break;
                case 2: table_2_[funcid] = func_addr; names_2_[funcid] = k.name; break;
                case 4: table_4_[funcid] = func_addr; names_4_[funcid] = k.name; break;
                default: continue;
            }

            // Track this kernel's code offset for KD patching
            kernel_text_offsets_.push_back({&k, rel_addr});

            text.data.insert(text.data.end(), k.code.begin(), k.code.end());
            while (text.data.size() % FUNC_ALIGNMENT != 0) text.data.push_back(0);
            mapped++;
        }
        printf("Mapped %d kernel functions, total .text size: %zu bytes\n", mapped, text.data.size());
        sections_.push_back(std::move(text));

        // NOTE: We do NOT create a separate .data.rel.ro section for function tables.
        // The function tables are already in .rodata at the correct PC-relative offset
        // from the dispatcher code. We'll make .rodata writable so relocations can fill them.
        // See rodata_table_1_off_, rodata_table_2_off_, rodata_table_4_off_ which are
        // populated when we process .rodata below.

        // Build .data (copy from dispatcher - contains __clang_gpu_used_external)
        // The dispatcher's .data contains:
        //   - __clang_gpu_used_external (96 bytes = 12 * 8-byte pointers to oneRankReduce kernels)
        // These pointers are filled in by R_AMDGPU_RELATIVE64 relocations.
        // We MUST copy the dispatcher's .data, not zero it out!
        auto* disp_data = disp_->find(".data");
        SectionInfo data;
        data.name = ".data";
        data.type = SHT_PROGBITS;
        data.flags = SHF_ALLOC | SHF_WRITE;
        data.alignment = 16;
        if (disp_data && disp_data->size > 0) {
            data.data = disp_->getBytes(*disp_data);
            printf("Copied dispatcher .data: %zu bytes\n", data.data.size());
        } else {
            // Fallback: create empty .data if dispatcher doesn't have one
            size_t scratch_size = 0x60;  // Match IFC's .data size
            data.data.resize(scratch_size, 0);
            printf("Created empty .data: %zu bytes (dispatcher had no .data)\n", scratch_size);
        }
        sections_.push_back(std::move(data));

        // Build .bss (uninitialized data - NOBITS)
        // The dispatcher's .bss originally contained:
        //   - ncclDevFuncTable_1/2/4 (6872 * 3 = 20616 bytes) - NOW MOVED to .data.rel.ro
        //   - __hip_cuid_* (1 byte)
        // So our .bss should ONLY contain __hip_cuid_* (1 byte + alignment)
        SectionInfo bss;
        bss.name = ".bss";
        bss.type = SHT_NOBITS;
        bss.flags = SHF_ALLOC | SHF_WRITE;
        bss.alignment = 1;
        // Only need space for __hip_cuid_* (1 byte) - IFC-like size
        size_t bss_size = 0x6b;  // Match IFC size (contains just cuid markers)
        bss.data.resize(0);  // NOBITS has no data
        bss.nobits_size = bss_size;
        sections_.push_back(std::move(bss));

        // Build .note (dispatcher only - specialized kernels are called as functions, not launched)
        if (disp_note) {
            SectionInfo note;
            note.name = ".note";
            note.type = SHT_NOTE;
            note.flags = SHF_ALLOC;
            note.alignment = 4;

            auto orig = disp_->getBytes(*disp_note);
            // Patch uses_dynamic_stack to false for multi-GPU compatibility
            patchNote(orig, note.data);
            printf("Built .note: %zu bytes (dispatcher only)\n", note.data.size());
            sections_.push_back(std::move(note));
        }

        // Build .rodata (KDs + function tables from dispatcher)
        // IMPORTANT: We make this section WRITABLE (SHF_WRITE) so relocations can fill
        // the function table entries at load time. The dispatcher code uses PC-relative
        // addressing to find the tables at their original offsets within this section.
        auto* disp_rodata = disp_->find(".rodata");
        if (disp_rodata) {
            SectionInfo rodata;
            rodata.name = ".rodata";
            rodata.type = SHT_PROGBITS;
            rodata.flags = SHF_ALLOC | SHF_WRITE;  // WRITE needed for relocations!
            rodata.alignment = 64;
            rodata.data = disp_->getBytes(*disp_rodata);

            // Find function table offsets within .rodata using symbols
            // Tables are declared const, so they're in .rodata now
            uint64_t rodata_base = disp_rodata->addr;
            rodata_table_1_off_ = rodata_table_2_off_ = rodata_table_4_off_ = 0;

            // Read symbol table to find function tables
            auto* symtab = disp_->find(".symtab");
            auto* strtab = disp_->find(".strtab");
            if (symtab && strtab) {
                const char* strings = disp_->fileStr(strtab->offset);
                const Elf64_Sym* syms = disp_->fileAt<Elf64_Sym>(symtab->offset);
                size_t nsyms = symtab->size / sizeof(Elf64_Sym);

                for (size_t i = 0; i < nsyms; i++) {
                    const char* name = strings + syms[i].st_name;
                    uint16_t shndx = syms[i].st_shndx;

                    // Check if symbol is in .rodata section
                    if (shndx == disp_rodata->index) {
                        if (strcmp(name, "ncclDevFuncTable_1") == 0) {
                            rodata_table_1_off_ = syms[i].st_value - rodata_base;
                        } else if (strcmp(name, "ncclDevFuncTable_2") == 0) {
                            rodata_table_2_off_ = syms[i].st_value - rodata_base;
                        } else if (strcmp(name, "ncclDevFuncTable_4") == 0) {
                            rodata_table_4_off_ = syms[i].st_value - rodata_base;
                        }
                    }
                }
            }

            // Zero out function tables - relocations will fill them at load time
            // This is critical for symbol-based relocations to work correctly
            if (rodata_table_1_off_ > 0 || rodata_table_2_off_ > 0 || rodata_table_4_off_ > 0) {
                size_t table_size = FUNC_COUNT * 8;  // Each table is FUNC_COUNT * 8 bytes
                if (rodata_table_1_off_ > 0 && rodata_table_1_off_ + table_size <= rodata.data.size()) {
                    memset(rodata.data.data() + rodata_table_1_off_, 0, table_size);
                }
                if (rodata_table_2_off_ > 0 && rodata_table_2_off_ + table_size <= rodata.data.size()) {
                    memset(rodata.data.data() + rodata_table_2_off_, 0, table_size);
                }
                if (rodata_table_4_off_ > 0 && rodata_table_4_off_ + table_size <= rodata.data.size()) {
                    memset(rodata.data.data() + rodata_table_4_off_, 0, table_size);
                }
                printf("  Zeroed function tables in .rodata (relocations will fill them)\n");
            }

            printf("Built .rodata: %zu bytes (KDs + const function tables)\n", rodata.data.size());
            if (rodata_table_1_off_ || rodata_table_2_off_ || rodata_table_4_off_) {
                printf("  Function tables in .rodata at offsets: 0x%lx, 0x%lx, 0x%lx\n",
                       rodata_table_1_off_, rodata_table_2_off_, rodata_table_4_off_);
            }
            sections_.push_back(std::move(rodata));

        }

        // Collect oneRankReduce kernel offsets for __clang_gpu_used_external
        // These kernels are in dispatcher .text - we need their offsets for relocations
        // NOTE: This is outside the disp_rodata block because we need it even if rodata isn't found
        {
            auto* disp_text_for_onerank = disp_->find(".text");
            auto* symtab_for_onerank = disp_->find(".symtab");
            auto* strtab_for_onerank = disp_->find(".strtab");
            printf("  Collecting oneRankReduce: text=%p, symtab=%p, strtab=%p\n",
                   (void*)disp_text_for_onerank, (void*)symtab_for_onerank, (void*)strtab_for_onerank);
            if (disp_text_for_onerank && symtab_for_onerank && strtab_for_onerank) {
                const char* strings = disp_->fileStr(strtab_for_onerank->offset);
                const Elf64_Sym* syms = disp_->fileAt<Elf64_Sym>(symtab_for_onerank->offset);
                size_t nsyms = symtab_for_onerank->size / sizeof(Elf64_Sym);

                uint16_t text_idx = disp_text_for_onerank->index;
                printf("  .text section index=%u, nsyms=%zu\n", text_idx, nsyms);
                int onerank_candidates = 0;
                for (size_t i = 0; i < nsyms; i++) {
                    const char* name = strings + syms[i].st_name;
                    // Look for oneRankReduce FUNC symbols in .text
                    bool is_func = ELF64_ST_TYPE(syms[i].st_info) == STT_FUNC;
                    bool in_text = syms[i].st_shndx == text_idx;
                    bool has_onerank = strstr(name, "oneRankReduce") != nullptr;
                    bool not_kd = strstr(name, ".kd") == nullptr;

                    if (has_onerank && not_kd) {
                        onerank_candidates++;
                        if (onerank_candidates <= 3) {
                            printf("    Candidate: %s (func=%d, shndx=%u vs text_idx=%u, in_text=%d)\n",
                                   name, is_func, syms[i].st_shndx, text_idx, in_text);
                        }
                    }

                    if (is_func && in_text && has_onerank && not_kd) {
                        // Store offset within .text and name
                        onerank_text_offsets_.push_back(syms[i].st_value - disp_text_for_onerank->addr);
                        onerank_names_.push_back(name);
                    }
                }
                printf("  Found %d oneRankReduce candidates, %zu matched\n",
                       onerank_candidates, onerank_text_offsets_.size());
                if (!onerank_text_offsets_.empty()) {
                    printf("  Found %zu oneRankReduce kernels for __clang_gpu_used_external\n",
                           onerank_text_offsets_.size());
                }
            }
        }

        // Copy other sections
        for (const char* name : {".dynsym", ".gnu.hash", ".hash", ".dynstr"}) {
            auto* s = disp_->find(name);
            if (s) {
                SectionInfo sec;
                sec.name = name;
                sec.type = s->type;
                sec.flags = s->flags;
                sec.alignment = s->align;
                sec.entsize = s->entsize;
                sec.data = disp_->getBytes(*s);
                sections_.push_back(std::move(sec));
            }
        }

        // Pre-allocate space in .dynsym and .dynstr for symbols to be added in Pass 3
        // This ensures computeLayout() accounts for the final section sizes
        {
            SectionInfo* dynsym_sec = nullptr;
            SectionInfo* dynstr_sec = nullptr;
            for (auto& s : sections_) {
                if (s.name == ".dynsym") dynsym_sec = &s;
                if (s.name == ".dynstr") dynstr_sec = &s;
            }
            
            if (dynsym_sec && dynstr_sec && (!kernel_text_offsets_.empty() || !onerank_text_offsets_.empty())) {
                // Calculate space needed for symbol names in .dynstr
                size_t names_size = 0;
                for (const auto& [kern, text_off] : kernel_text_offsets_) {
                    names_size += kern->name.size() + 1;  // +1 for null terminator
                }
                for (const auto& name : onerank_names_) {
                    names_size += name.size() + 1;
                }
                
                size_t num_new_symbols = kernel_text_offsets_.size() + onerank_text_offsets_.size();
                
                // Record original sizes for later reference
                prealloc_dynstr_orig_size_ = dynstr_sec->data.size();
                prealloc_dynsym_orig_count_ = dynsym_sec->data.size() / 24;
                
                // Find first global symbol position in .dynsym
                size_t first_global_idx = dynsym_sec->data.size() / 24;
                for (size_t i = 0; i * 24 < dynsym_sec->data.size(); i++) {
                    uint8_t* sym = dynsym_sec->data.data() + i * 24;
                    uint8_t binding = ELF64_ST_BIND(sym[4]);  // st_info byte
                    if (binding == STB_GLOBAL || binding == STB_WEAK) {
                        first_global_idx = i;
                        break;
                    }
                }
                prealloc_dynsym_insert_idx_ = first_global_idx;
                
                // Pre-allocate .dynstr (append space at end)
                dynstr_sec->data.resize(dynstr_sec->data.size() + names_size);
                
                // Pre-allocate .dynsym (insert placeholder LOCAL symbols before first GLOBAL)
                // This shifts GLOBAL symbols down, maintaining proper LOCAL/GLOBAL ordering
                std::vector<uint8_t> placeholder_syms(num_new_symbols * 24, 0);
                if (first_global_idx * 24 < dynsym_sec->data.size()) {
                    dynsym_sec->data.insert(dynsym_sec->data.begin() + first_global_idx * 24,
                                           placeholder_syms.begin(), placeholder_syms.end());
                } else {
                    dynsym_sec->data.insert(dynsym_sec->data.end(),
                                           placeholder_syms.begin(), placeholder_syms.end());
                }
                
                printf("Pre-allocated for %zu symbols: .dynstr +%zu bytes (now %zu), .dynsym +%zu entries (now %zu) at idx %zu\n",
                       num_new_symbols, names_size, dynstr_sec->data.size(),
                       num_new_symbols, dynsym_sec->data.size() / 24, first_global_idx);
            }
        }

        // Placeholder for .rela.dyn (allocated, will be populated during patchSections)
        // Size will be updated after we know how many relocations are needed
        {
            SectionInfo rela;
            rela.name = ".rela.dyn";
            rela.type = SHT_RELA;
            rela.flags = SHF_ALLOC;
            rela.alignment = 8;
            rela.entsize = 24;
            // Actual data populated in buildRelocations()
            sections_.push_back(std::move(rela));
        }

        // .dynamic - will be populated during layout
        {
            SectionInfo dyn;
            dyn.name = ".dynamic";
            dyn.type = SHT_DYNAMIC;
            dyn.flags = SHF_ALLOC | SHF_WRITE;
            dyn.alignment = 8;
            dyn.entsize = 16;
            // Actual data populated in computeLayout()
            sections_.push_back(std::move(dyn));
        }

        // .relro_padding - NOBITS section for page alignment after RELRO segment
        {
            SectionInfo relro_pad;
            relro_pad.name = ".relro_padding";
            relro_pad.type = SHT_NOBITS;
            relro_pad.flags = SHF_ALLOC | SHF_WRITE;
            relro_pad.alignment = 1;
            relro_pad.nobits_size = 0;  // Will be computed during layout
            sections_.push_back(std::move(relro_pad));
        }

        // Non-allocated sections
        for (const char* name : {".symtab", ".strtab"}) {
            auto* s = disp_->find(name);
            if (s) {
                SectionInfo sec;
                sec.name = name;
                sec.type = s->type;
                sec.flags = 0;  // Non-allocated
                sec.alignment = s->align;
                sec.entsize = s->entsize;
                sec.data = disp_->getBytes(*s);
                sections_.push_back(std::move(sec));
            }
        }

        // Add empty .AMDGPU.gpr_maximums section (expected by COMGR)
        {
            SectionInfo gpr_max;
            gpr_max.name = ".AMDGPU.gpr_maximums";
            gpr_max.type = SHT_PROGBITS;
            gpr_max.flags = 0;  // Non-allocated
            gpr_max.alignment = 1;
            // Empty section
            sections_.push_back(std::move(gpr_max));
        }

        // Build merged .debug_line and .debug_line_str sections from dispatcher and specialized kernels
        // (if any have debug info from -gline-tables-only compilation). Skipped when --omit-dwarf.
        if (!omit_dwarf_) {
        {
            SectionInfo debug_line;
            debug_line.name = ".debug_line";
            debug_line.type = SHT_PROGBITS;
            debug_line.flags = 0;  // Non-allocated
            debug_line.alignment = 1;

            // First, add dispatcher's .debug_line and .debug_line_str if present
            auto* disp_debug_line = disp_->find(".debug_line");
            auto* disp_debug_line_str = disp_->find(".debug_line_str");
            if (disp_debug_line && disp_debug_line->size > 0) {
                auto disp_dl = disp_->getBytes(*disp_debug_line);
                auto* disp_text = disp_->find(".text");
                uint64_t disp_text_addr = disp_text ? disp_text->addr : 0;

                // Apply .rela.debug_line relocations to resolve string offsets
                // (same as done for specialized kernels in extractKernelInfo)
                auto* rela_debug_line = disp_->find(".rela.debug_line");
                if (rela_debug_line && rela_debug_line->size > 0 && rela_debug_line->entsize >= sizeof(Elf64_Rela)) {
                    const Elf64_Rela* relas = disp_->fileAt<Elf64_Rela>(rela_debug_line->offset);
                    size_t num_relas = rela_debug_line->size / sizeof(Elf64_Rela);

                    for (size_t i = 0; i < num_relas; i++) {
                        const Elf64_Rela& r = relas[i];
                        uint32_t rtype = ELF64_R_TYPE(r.r_info);

                        // R_X86_64_32 (type 10) / R_AMDGPU_ABS32 (type 3) for string offsets
                        if ((rtype == 10 || rtype == 3) && r.r_offset + 4 <= disp_dl.size()) {
                            uint32_t val = (uint32_t)r.r_addend;
                            memcpy(&disp_dl[r.r_offset], &val, 4);
                        }
                        // R_X86_64_64 (type 1) / R_AMDGPU_ABS64 (type 2) for .text addresses
                        else if ((rtype == 1 || rtype == 2) && r.r_offset + 8 <= disp_dl.size()) {
                            uint64_t val = (uint64_t)r.r_addend;
                            memcpy(&disp_dl[r.r_offset], &val, 8);
                        }
                    }
                }

                size_t str_base = merged_debug_line_str_.size();
                size_t orig_str_size = 0;
                if (disp_debug_line_str && disp_debug_line_str->size > 0) {
                    auto disp_dls = disp_->getBytes(*disp_debug_line_str);
                    orig_str_size = disp_dls.size();
                    merged_debug_line_str_.insert(merged_debug_line_str_.end(),
                                                  disp_dls.begin(), disp_dls.end());
                }

                debug_line_chunks_.push_back({
                    debug_line.data.size(),  // merged_offset
                    disp_dl.size(),          // size
                    disp_text_addr,          // orig_text_addr
                    0,                       // new_text_offset (dispatcher is at offset 0)
                    str_base,                // str_offset_base
                    orig_str_size,           // orig_str_size
                    {}                       // comp_dir (will use debug_comp_dir_)
                });
                debug_line.data.insert(debug_line.data.end(), disp_dl.begin(), disp_dl.end());
            }

            // Then add each specialized kernel's .debug_line and .debug_line_str
            for (const auto& [kern, text_off] : kernel_text_offsets_) {
                if (!kern->debug_line.empty()) {
                    size_t str_base = merged_debug_line_str_.size();
                    size_t orig_str_size = kern->debug_line_str.size();

                    // Append this kernel's .debug_line_str to merged
                    if (!kern->debug_line_str.empty()) {
                        merged_debug_line_str_.insert(merged_debug_line_str_.end(),
                                                      kern->debug_line_str.begin(),
                                                      kern->debug_line_str.end());
                    }

                    debug_line_chunks_.push_back({
                        debug_line.data.size(),    // merged_offset
                        kern->debug_line.size(),   // size
                        kern->orig_text_addr,      // orig_text_addr
                        text_off,                  // new_text_offset
                        str_base,                  // str_offset_base
                        orig_str_size,             // orig_str_size
                        {}                         // comp_dir (will use debug_comp_dir_)
                    });
                    debug_line.data.insert(debug_line.data.end(),
                                          kern->debug_line.begin(),
                                          kern->debug_line.end());
                }
            }

            if (!debug_line.data.empty()) {
                // Pad .debug_line to 8-byte boundary so .debug_line_str (which follows) starts aligned
                // This ensures .debug_abbrev (which comes after .debug_line_str) also starts aligned
                size_t padding = (8 - (debug_line.data.size() % 8)) % 8;
                if (padding > 0) {
                    debug_line.data.insert(debug_line.data.end(), padding, 0);
                    printf("Built .debug_line: %zu bytes (+ %zu padding = %zu total, %zu chunks)\n",
                           debug_line.data.size() - padding, padding, debug_line.data.size(), debug_line_chunks_.size());
                } else {
                    printf("Built .debug_line: %zu bytes (%zu chunks)\n",
                           debug_line.data.size(), debug_line_chunks_.size());
                }
                sections_.push_back(std::move(debug_line));
            }

            // Add merged .debug_line_str section
            if (!merged_debug_line_str_.empty()) {
                SectionInfo debug_line_str;
                debug_line_str.name = ".debug_line_str";
                debug_line_str.type = SHT_PROGBITS;
                debug_line_str.flags = SHF_MERGE | SHF_STRINGS;
                debug_line_str.alignment = 1;
                debug_line_str.entsize = 1;
                debug_line_str.data = merged_debug_line_str_;
                // Pad to 8-byte boundary so .debug_abbrev (which follows) starts aligned
                size_t padding = (8 - (debug_line_str.data.size() % 8)) % 8;
                if (padding > 0) {
                    debug_line_str.data.insert(debug_line_str.data.end(), padding, 0);
                    printf("Built .debug_line_str: %zu bytes (+ %zu padding = %zu total)\n", 
                           merged_debug_line_str_.size(), padding, debug_line_str.data.size());
                } else {
                    printf("Built .debug_line_str: %zu bytes\n", merged_debug_line_str_.size());
                }
                sections_.push_back(std::move(debug_line_str));
            }

            // Merge other debug sections (.debug_abbrev, .debug_str, .debug_str_offsets,
            // .debug_addr, .debug_rnglists, .debug_info) from each kernel
            // These will be patched later in mergeDebugInfo() after layout
            {
                // Track current offset in each merged section
                size_t line_offset = 0;  // Track where each kernel's debug_line starts

                // First add dispatcher's debug sections if present
                auto* disp_debug_abbrev = disp_->find(".debug_abbrev");
                auto* disp_debug_str = disp_->find(".debug_str");
                auto* disp_debug_str_offsets = disp_->find(".debug_str_offsets");
                auto* disp_debug_addr = disp_->find(".debug_addr");
                auto* disp_debug_rnglists = disp_->find(".debug_rnglists");
                auto* disp_debug_ranges = disp_->find(".debug_ranges");  // Legacy section
                auto* disp_debug_info = disp_->find(".debug_info");
                auto* disp_text = disp_->find(".text");

                if (disp_debug_info && disp_debug_info->size > 0) {
                    auto disp_di = disp_->getBytes(*disp_debug_info);
                    uint16_t disp_version = getDwarfVersionFromDebugInfo(disp_di.data(), disp_di.size());
                    if (disp_version != 0 && disp_version != 4 && disp_version != 5) {
                        setFatalError("Dispatcher has unsupported DWARF version; device linker requires DWARF4 or DWARF5");
                        return false;
                    }

                    DebugInfoChunk chunk;
                    chunk.merged_offset = merged_debug_info_.size();
                    chunk.size = disp_di.size();
                    chunk.orig_text_addr = disp_text ? disp_text->addr : 0;
                    chunk.new_text_offset = 0;  // Dispatcher is at offset 0
                    chunk.abbrev_base = merged_debug_abbrev_.size();
                    chunk.str_base = merged_debug_str_.size();
                    chunk.str_offsets_base = merged_debug_str_offsets_.size();
                    chunk.addr_base = merged_debug_addr_.size();
                    chunk.rnglists_base = merged_debug_rnglists_.size();
                    chunk.ranges_base = merged_debug_ranges_.size();  // Legacy .debug_ranges
                    chunk.line_base = 0;  // First debug_line chunk
                    chunk.line_str_offset_base = debug_line_chunks_.empty() ? 0 : debug_line_chunks_[0].str_offset_base;
                    chunk.dwarf_version = (disp_version != 0) ? disp_version : 5;
                    chunk.source_file = "(dispatcher)";

                    // Append dispatcher's debug sections
                    if (disp_debug_abbrev && disp_debug_abbrev->size > 0) {
                        auto data = disp_->getBytes(*disp_debug_abbrev);
                        merged_debug_abbrev_.insert(merged_debug_abbrev_.end(), data.begin(), data.end());
                    }
                    if (disp_debug_str && disp_debug_str->size > 0) {
                        auto data = disp_->getBytes(*disp_debug_str);
                        merged_debug_str_.insert(merged_debug_str_.end(), data.begin(), data.end());
                    }
                    if (disp_debug_str_offsets && disp_debug_str_offsets->size > 0) {
                        auto data = disp_->getBytes(*disp_debug_str_offsets);
                        // DWARF5 .debug_str_offsets unit needs at least 8 bytes (unit_length + version + padding)
                        if (data.size() >= 8)
                            merged_debug_str_offsets_.insert(merged_debug_str_offsets_.end(), data.begin(), data.end());
                        else
                            merged_debug_str_offsets_.insert(merged_debug_str_offsets_.end(),
                                kMinimalStrOffsetsHeader, kMinimalStrOffsetsHeader + 8);
                    } else {
                        // Chunk has .debug_info (may have DW_AT_str_offsets_base) but no section: add valid 8-byte header
                        merged_debug_str_offsets_.insert(merged_debug_str_offsets_.end(),
                            kMinimalStrOffsetsHeader, kMinimalStrOffsetsHeader + 8);
                    }
                    if (disp_debug_addr && disp_debug_addr->size > 0) {
                        auto data = disp_->getBytes(*disp_debug_addr);
                        merged_debug_addr_.insert(merged_debug_addr_.end(), data.begin(), data.end());
                    }
                    if (disp_debug_rnglists && disp_debug_rnglists->size > 0) {
                        auto data = disp_->getBytes(*disp_debug_rnglists);
                        merged_debug_rnglists_.insert(merged_debug_rnglists_.end(), data.begin(), data.end());
                    }
                    if (disp_debug_ranges && disp_debug_ranges->size > 0) {
                        auto data = disp_->getBytes(*disp_debug_ranges);
                        merged_debug_ranges_.insert(merged_debug_ranges_.end(), data.begin(), data.end());
                    }

                    // Find DWARF attribute positions in dispatcher's debug_info (use real ELF so LLVM sees all sections)
                    chunk.dwarf_attr_positions = findDwarfAttrPositionsFromElf(
                        static_cast<const uint8_t*>(disp_file_->data()), disp_file_->size());
                    merged_debug_info_.insert(merged_debug_info_.end(), disp_di.begin(), disp_di.end());

                    debug_info_chunks_.push_back(chunk);

                    // Track debug_line offset for dispatcher
                    if (!debug_line_chunks_.empty()) {
                        line_offset = debug_line_chunks_[0].size;
                    }
                }

                // Then add each specialized kernel's debug sections
                size_t kern_idx = debug_info_chunks_.empty() ? 0 : 1;
                for (const auto& [kern, text_off] : kernel_text_offsets_) {
                    if (!kern->debug_info.empty()) {
                        DebugInfoChunk chunk;
                        chunk.merged_offset = merged_debug_info_.size();
                        chunk.size = kern->debug_info.size();
                        chunk.orig_text_addr = kern->orig_text_addr;
                        chunk.new_text_offset = text_off;
                        chunk.abbrev_base = merged_debug_abbrev_.size();
                        chunk.str_base = merged_debug_str_.size();
                        chunk.str_offsets_base = merged_debug_str_offsets_.size();
                        chunk.addr_base = merged_debug_addr_.size();
                        chunk.rnglists_base = merged_debug_rnglists_.size();
                        chunk.ranges_base = merged_debug_ranges_.size();  // Legacy .debug_ranges
                        chunk.line_base = line_offset;
                        chunk.line_str_offset_base = (!kern->debug_line.empty() && kern_idx < debug_line_chunks_.size())
                            ? debug_line_chunks_[kern_idx].str_offset_base : 0;
                        chunk.dwarf_version = (kern->dwarf_version != 0) ? kern->dwarf_version : 5;
                        chunk.source_file = kern->source_file.empty() ? "(unknown)" : kern->source_file;

                        // Append kernel's debug sections
                        if (!kern->debug_abbrev.empty()) {
                            merged_debug_abbrev_.insert(merged_debug_abbrev_.end(),
                                kern->debug_abbrev.begin(), kern->debug_abbrev.end());
                        }
                        if (!kern->debug_str.empty()) {
                            merged_debug_str_.insert(merged_debug_str_.end(),
                                kern->debug_str.begin(), kern->debug_str.end());
                        }
                        if (!kern->debug_str_offsets.empty()) {
                            // DWARF5 .debug_str_offsets unit needs at least 8 bytes (unit_length + version + padding)
                            if (kern->debug_str_offsets.size() >= 8)
                                merged_debug_str_offsets_.insert(merged_debug_str_offsets_.end(),
                                    kern->debug_str_offsets.begin(), kern->debug_str_offsets.end());
                            else
                                merged_debug_str_offsets_.insert(merged_debug_str_offsets_.end(),
                                    kMinimalStrOffsetsHeader, kMinimalStrOffsetsHeader + 8);
                        } else {
                            // Chunk has .debug_info (may have DW_AT_str_offsets_base) but no section: add valid 8-byte header
                            merged_debug_str_offsets_.insert(merged_debug_str_offsets_.end(),
                                kMinimalStrOffsetsHeader, kMinimalStrOffsetsHeader + 8);
                        }
                        if (!kern->debug_addr.empty()) {
                            merged_debug_addr_.insert(merged_debug_addr_.end(),
                                kern->debug_addr.begin(), kern->debug_addr.end());
                        }
                        if (!kern->debug_rnglists.empty()) {
                            merged_debug_rnglists_.insert(merged_debug_rnglists_.end(),
                                kern->debug_rnglists.begin(), kern->debug_rnglists.end());
                        }
                        if (!kern->debug_ranges.empty()) {
                            merged_debug_ranges_.insert(merged_debug_ranges_.end(),
                                kern->debug_ranges.begin(), kern->debug_ranges.end());
                        }
                        merged_debug_info_.insert(merged_debug_info_.end(),
                            kern->debug_info.begin(), kern->debug_info.end());

                        // Copy DWARF attribute positions for later patching
                        chunk.dwarf_attr_positions = kern->dwarf_attr_positions;

                        debug_info_chunks_.push_back(chunk);

                        // Track debug_line offset
                        if (kern_idx < debug_line_chunks_.size()) {
                            line_offset += debug_line_chunks_[kern_idx].size;
                        }
                        kern_idx++;
                    }
                }

                printf("Merged debug sections: abbrev=%zu, str=%zu, str_offsets=%zu, addr=%zu, rnglists=%zu, ranges=%zu, info=%zu bytes (%zu chunks)\n",
                       merged_debug_abbrev_.size(), merged_debug_str_.size(),
                       merged_debug_str_offsets_.size(), merged_debug_addr_.size(),
                       merged_debug_rnglists_.size(), merged_debug_ranges_.size(),
                       merged_debug_info_.size(), debug_info_chunks_.size());
            }

            // Note: mergeDebugInfo() is called later from link() after
            // computeLayout() sets text_addr_
        }
        }  // !omit_dwarf_

        return true;
    }

    void patchNote(const std::vector<uint8_t>& orig, std::vector<uint8_t>& out) {
        // Copy and patch various metadata fields
        out = orig;

        // 1. Keep uses_dynamic_stack as-is (IFC has true, we should match)
        // Previously we patched true->false but IFC production builds keep it true
        printf("  .note: keeping uses_dynamic_stack unchanged (matching IFC)\n");

        // 2. Set private_segment_fixed_size to max_stack_ for specialized functions
        // Unlike IFC which compiles everything together, we call separate specialized
        // functions that have their own scratch requirements
        printf("  .note: will patch private_segment_fixed_size to %d\n", max_stack_);

        // NOTE: We still patch VGPR/SGPR counts below to match our max resource usage

        // Legacy code for .private_segment_fixed_size patching (disabled)
        // The field name is 27 chars: ".private_segment_fixed_size"
        // Encoded as: 0xbb (fixstr 27) + 27 chars + value
        // Original value 0 is encoded as single byte 0x00
        // New value (e.g., 1124 = 0x464) needs uint16: 0xcd 0x04 0x64
        // This changes the size! We need to handle this carefully.
        //
        // For now, if max_stack_ fits in a single byte (0-127), we can patch in place.
        // If max_stack_ > 127 but original is 0, we can't expand in place easily.
        //
        // Alternative: patch to 0 (leave unchanged) and rely on KD patching only.
        // The HSA runtime uses KD values, not metadata, for actual execution.
        // But COMGR might validate consistency...
        //
        // Patch .private_segment_fixed_size in msgpack
        // Original value 0 is encoded as single byte fixint (0x00)
        // For values > 127, we need uint16 encoding (0xcd + 2 bytes big-endian)
        const char* target2 = ".private_segment_fixed_size";
        size_t tlen2 = strlen(target2);
        int patch_count2 = 0;

        // Find all occurrences and patch (may need to expand encoding)
        std::vector<size_t> patch_positions;
        for (size_t i = 0; i + tlen2 + 1 < out.size(); i++) {
            // fixstr(27) = 0xbb
            if (out[i] == 0xbb && memcmp(out.data() + i + 1, target2, tlen2) == 0) {
                size_t val_off = i + 1 + tlen2;
                if (val_off < out.size() && out[val_off] <= 0x7f) {
                    patch_positions.push_back(val_off);
                }
            }
        }

        // Helper to expand msgpack integer field (handles fixint -> uint8/uint16 expansion)
        auto expandIntField = [&](const char* field_name, int new_val) -> int {
            size_t flen = strlen(field_name);
            std::vector<size_t> positions;

            for (size_t i = 0; i + flen + 2 < out.size(); i++) {
                uint8_t prefix = out[i];
                size_t str_len = 0, hdr = 1;
                if (prefix >= 0xa0 && prefix <= 0xbf) str_len = prefix - 0xa0;
                else if (prefix == 0xbb) str_len = 27;  // fixstr(27)
                else if (prefix == 0xd9 && i + 1 < out.size()) { str_len = out[i + 1]; hdr = 2; }

                if (str_len == flen && memcmp(out.data() + i + hdr, field_name, flen) == 0) {
                    positions.push_back(i + hdr + flen);
                }
            }
            if (positions.empty()) return 0;

            std::sort(positions.rbegin(), positions.rend());
            int patched = 0;
            for (auto off : positions) {
                if (off >= out.size()) continue;
                uint8_t b = out[off];

                if (b <= 0x7f) {  // fixint
                    if (new_val <= 127) { out[off] = new_val; }
                    else if (new_val <= 255) { out.insert(out.begin() + off, 1, 0); out[off] = 0xcc; out[off+1] = new_val; }
                    else { out.insert(out.begin() + off, 2, 0); out[off] = 0xcd; out[off+1] = (new_val>>8)&0xff; out[off+2] = new_val&0xff; }
                    patched++;
                } else if (b == 0xcc && off + 1 < out.size()) {  // uint8
                    if (new_val <= 255) { out[off+1] = new_val; }
                    else { out.insert(out.begin() + off + 1, 1, 0); out[off] = 0xcd; out[off+1] = (new_val>>8)&0xff; out[off+2] = new_val&0xff; }
                    patched++;
                } else if (b == 0xcd && off + 2 < out.size()) {  // uint16
                    out[off+1] = (new_val>>8)&0xff; out[off+2] = new_val&0xff;
                    patched++;
                }
            }
            return patched;
        };

        // Patch private_segment_fixed_size to max_stack_ for specialized function scratch
        if (!patch_positions.empty()) {
            if (max_stack_ <= 127) {
                for (auto pos : patch_positions) { out[pos] = (uint8_t)max_stack_; patch_count2++; }
            } else {
                std::sort(patch_positions.rbegin(), patch_positions.rend());
                for (auto pos : patch_positions) {
                    if (max_stack_ <= 65535) {
                        out.insert(out.begin() + pos, 2, 0);
                        out[pos] = 0xcd; out[pos+1] = (max_stack_>>8)&0xff; out[pos+2] = max_stack_&0xff;
                    }
                    patch_count2++;
                }
            }
        }
        if (patch_count2 > 0) {
            printf("  .note: patched %d private_segment_fixed_size to %d\n", patch_count2, max_stack_);
        }

        // Patch group_segment_fixed_size to max_lds_ so .note matches patched KD
        int lds_patched = expandIntField(".group_segment_fixed_size", max_lds_);
        if (lds_patched > 0) {
            printf("  .note: patched %d .group_segment_fixed_size to %d\n", lds_patched, max_lds_);
        }

        int vgpr_patched = expandIntField(".vgpr_count", max_vgpr_);
        int sgpr_patched = expandIntField(".sgpr_count", max_sgpr_);
        if (vgpr_patched > 0 || sgpr_patched > 0) {
            printf("  .note: patched %d .vgpr_count to %d, %d .sgpr_count to %d\n",
                   vgpr_patched, max_vgpr_, sgpr_patched, max_sgpr_);
        }

        // Update descsz ONCE after all modifications
        // Note header: namesz(4) + descsz(4) + type(4) + name("AMDGPU\0" aligned to 4 = 8) = 20 bytes
        uint32_t new_descsz = out.size() - 20;
        memcpy(out.data() + 4, &new_descsz, 4);

        // Pad descriptor to 4-byte boundary if needed
        size_t pad = (4 - (new_descsz % 4)) % 4;
        for (size_t i = 0; i < pad; i++) out.push_back(0);
        if (pad > 0) {
            new_descsz = out.size() - 20;
            memcpy(out.data() + 4, &new_descsz, 4);
        }
    }

    // ========== Pass 2: Layout ==========
    void computeLayout() {
        // Layout preserving PC-relative references:
        // The dispatcher uses PC-relative addressing to find function tables.
        // Original layout has function tables in .rodata BEFORE .text.
        // We put .data.rel.ro (function tables) before .text to preserve this.
        //
        // LOAD R:  [ELF header + phdrs] [.note] [.dynsym] [.gnu.hash] [.hash] [.dynstr] [.rela.dyn] [.rodata] [.data.rel.ro]
        // LOAD RX: [.text] (page-aligned)
        // LOAD RW: [.dynamic] [.relro_padding] [.data] [.bss]
        // Non-alloc: [.symtab] [.strtab] [.shstrtab] [section headers]

        printf("Computing layout (preserving PC-relative offsets)...\n");

        // Pre-calculate .rela.dyn size based on function table entries + oneRankReduce
        int rela_count = 0;
        for (int i = 0; i < FUNC_COUNT; i++) {
            if (table_1_[i]) rela_count++;
            if (table_2_[i]) rela_count++;
            if (table_4_[i]) rela_count++;
        }
        rela_count += onerank_text_offsets_.size();  // For __clang_gpu_used_external
        for (auto& s : sections_) {
            if (s.name == ".rela.dyn") {
                s.data.resize(rela_count * 24);
                break;
            }
        }

        const size_t ehdr_size = 64;
        const size_t phdr_size = 56;
        const size_t num_phdrs = 9;  // Match IFC: PHDR, LOAD R, LOAD RX, LOAD RW(RELRO), LOAD RW, DYNAMIC, GNU_RELRO, GNU_STACK, NOTE
        uint64_t addr = ehdr_size + num_phdrs * phdr_size;

        // Sort allocated sections - CRITICAL: .rodata (with function tables) BEFORE .text
        // The dispatcher uses PC-relative addressing to find function tables in .rodata.
        // We preserve this layout: .rodata before .text at the original relative offset.
        auto order = [](const std::string& n) -> int {
            // LOAD R segment (read-only, but .rodata is now writable for relocations)
            if (n == ".note") return 0;
            if (n == ".dynsym") return 1;
            if (n == ".gnu.hash") return 2;
            if (n == ".hash") return 3;
            if (n == ".dynstr") return 4;
            if (n == ".rela.dyn") return 5;
            if (n == ".rodata") return 6;  // Function tables are HERE at PC-relative offset
            // LOAD RX segment
            if (n == ".text") return 7;
            // LOAD RW segment - .dynamic, .relro_padding, .data, .bss
            if (n == ".dynamic") return 8;
            if (n == ".relro_padding") return 9;
            if (n == ".data") return 10;
            if (n == ".bss") return 11;
            return 100;
        };

        std::sort(sections_.begin(), sections_.end(), [&](const SectionInfo& a, const SectionInfo& b) {
            bool a_alloc = a.isAlloc(), b_alloc = b.isAlloc();
            if (a_alloc != b_alloc) return a_alloc;
            if (a_alloc) return order(a.name) < order(b.name);
            return false;
        });

        // First pass: compute addresses for sections before .text
        // Function tables are in .rodata at their original PC-relative offsets
        for (auto& s : sections_) {
            if (!s.isAlloc()) continue;
            if (s.name == ".text") break;  // Stop before .text

            addr = (addr + s.alignment - 1) & ~(s.alignment - 1);
            s.addr = addr;
            s.offset = addr;
            addr += s.size();

            if (s.name == ".rodata") {
                rodata_addr_ = s.addr;
                // Function tables are inside .rodata - set data_rel_ro_addr_ to point there
                // This is used by relocation building code
                data_rel_ro_addr_ = rodata_addr_ + rodata_table_1_off_;
            }

            printf("  %-16s @ 0x%08lx  size=0x%06zx  align=%lu\n",
                   s.name.c_str(), s.addr, s.size(), s.alignment);
        }

        // Page-align for .text segment (LOAD RX)
        uint64_t text_start = (addr + 0xFFF) & ~0xFFFUL;
        printf("  [page align for .text: 0x%lx -> 0x%lx]\n", addr, text_start);
        addr = text_start;

        // Compute .text address
        for (auto& s : sections_) {
            if (s.name == ".text") {
                addr = (addr + s.alignment - 1) & ~(s.alignment - 1);
                s.addr = addr;
                s.offset = addr;
                text_addr_ = addr;
                addr += s.size();
                printf("  %-16s @ 0x%08lx  size=0x%06zx  align=%lu\n",
                       s.name.c_str(), s.addr, s.size(), s.alignment);
                break;
            }
        }

        // Page-align for RW segment
        uint64_t rw_start = (addr + 0xFFF) & ~0xFFFUL;
        printf("  [page align for RW: 0x%lx -> 0x%lx]\n", addr, rw_start);
        addr = rw_start;

        // Compute .dynamic address and populate data
        for (auto& s : sections_) {
            if (s.name == ".dynamic") {
                addr = (addr + s.alignment - 1) & ~(s.alignment - 1);
                s.addr = addr;
                s.offset = addr;
                dyn_addr_ = addr;
                // Data will be populated after all addresses are known
                printf("  %-16s @ 0x%08lx  (data populated later)\n", s.name.c_str(), s.addr);
                break;
            }
        }

        // Compute .relro_padding size to align RELRO segment end to page boundary
        // RELRO covers .data.rel.ro + .dynamic + .relro_padding
        uint64_t relro_end_before_pad = addr;
        uint64_t relro_end_aligned = (relro_end_before_pad + 0xFFF) & ~0xFFFUL;
        size_t relro_pad_size = relro_end_aligned - relro_end_before_pad;

        for (auto& s : sections_) {
            if (s.name == ".relro_padding") {
                s.nobits_size = relro_pad_size;
                s.addr = addr;
                s.offset = addr;  // NOBITS doesn't take file space
                printf("  %-16s @ 0x%08lx  size=0x%06zx  (NOBITS)\n",
                       s.name.c_str(), s.addr, s.size());
                addr += relro_pad_size;
                break;
            }
        }

        relro_end_ = addr;  // End of RELRO segment

        // Page-align for .data/.bss segment (LOAD RW #2)
        uint64_t data_start = (addr + 0xFFF) & ~0xFFFUL;
        printf("  [page align for data: 0x%lx -> 0x%lx]\n", addr, data_start);
        addr = data_start;

        // Assign addresses to .data and .bss
        for (auto& s : sections_) {
            if (!s.isAlloc()) continue;
            if (s.name != ".data" && s.name != ".bss") continue;

            addr = (addr + s.alignment - 1) & ~(s.alignment - 1);
            s.addr = addr;
            s.offset = s.type == SHT_NOBITS ? data_start : addr;  // NOBITS shares offset with previous
            if (s.name == ".data") data_addr_ = addr;
            addr += s.size();

            printf("  %-16s @ 0x%08lx  size=0x%06zx  %s\n",
                   s.name.c_str(), s.addr, s.size(), s.type == SHT_NOBITS ? "(NOBITS)" : "");
        }

        // Reserve space for .dynamic section (11 entries * 16 bytes = 176 bytes)
        // Actual data will be populated in populateDynamicSection() after buildRelocations()
        size_t dyn_size = 11 * 16;  // RELA, RELASZ, RELAENT, RELACOUNT, SYMTAB, SYMENT, STRTAB, STRSZ, GNU_HASH, HASH, NULL

        // Update .relro_padding based on reserved .dynamic size
        uint64_t addr_after_dyn = dyn_addr_ + dyn_size;
        for (auto& s : sections_) {
            if (s.name == ".relro_padding") {
                uint64_t relro_end_aligned = (addr_after_dyn + 0xFFF) & ~0xFFFUL;
                s.nobits_size = relro_end_aligned - addr_after_dyn;
                s.addr = addr_after_dyn;
                s.offset = addr_after_dyn;
                relro_end_ = addr_after_dyn + s.nobits_size;
                printf("  %-16s @ 0x%08lx  size=0x%06zx  (reserved)\n", ".dynamic", dyn_addr_, dyn_size);
                printf("  %-16s @ 0x%08lx  size=0x%06zx  (NOBITS)\n",
                       s.name.c_str(), s.addr, s.size());
                break;
            }
        }

        // Non-allocated sections - offset only, no address
        // Need to recalculate offset starting from end of last allocated section
        uint64_t nonalloc_off = 0;
        for (const auto& s : sections_) {
            if (s.isAlloc() && s.type != SHT_NOBITS) {
                nonalloc_off = std::max(nonalloc_off, s.offset + s.fileSize());
            }
        }

        for (auto& s : sections_) {
            if (s.isAlloc()) continue;
            nonalloc_off = (nonalloc_off + s.alignment - 1) & ~(s.alignment - 1);
            s.addr = 0;
            s.offset = nonalloc_off;
            nonalloc_off += s.fileSize();
            printf("  %-16s @ offset 0x%08lx  size=0x%06zx (non-alloc)\n",
                   s.name.c_str(), s.offset, s.size());
        }

        printf("Layout complete. Total size: 0x%lx\n", nonalloc_off);
    }

    // ========== Pass 3: Patch ==========
    void patchSections() {
        // Function tables - check if in .rodata (const tables) or .data.rel.ro
        SectionInfo* rodata_sec = nullptr;
        SectionInfo* data_rel_ro_sec = nullptr;
        for (auto& s : sections_) {
            if (s.name == ".rodata") rodata_sec = &s;
            if (s.name == ".data.rel.ro") data_rel_ro_sec = &s;
        }

        // Function tables should be zeroed - relocations will fill them at load time
        // This applies whether tables are in .rodata or .data.rel.ro
        // Symbol-based relocations (R_AMDGPU_ABS64) will resolve to the correct addresses
        if (rodata_sec && rodata_table_2_off_ > 0) {
            printf("Function tables in .rodata at offsets 0x%lx, 0x%lx, 0x%lx (zeroed, will be filled by relocations)\n",
                   rodata_table_1_off_, rodata_table_2_off_, rodata_table_4_off_);
            // Tables are already zeroed when .rodata section is created
        } else if (data_rel_ro_sec) {
            printf("Function tables in .data.rel.ro at 0x%lx (zeroed, will be filled by relocations)\n", data_rel_ro_addr_);
            // Tables are already zeroed when .data.rel.ro section is created
        }

        // Patch .text with PC-relative table references
        SectionInfo* text_sec = nullptr;
        for (auto& s : sections_) if (s.name == ".text") { text_sec = &s; break; }

        if (text_sec) {
            // Determine table addresses based on where they are (rodata or data.rel.ro)
            uint64_t table_addrs[3];
            uint64_t old_table_base;
            auto* disp_rodata = disp_->find(".rodata");
            auto* disp_bss = disp_->find(".bss");

            if (rodata_table_2_off_ > 0 && disp_rodata) {
                // Tables are in .rodata (const case)
                table_addrs[0] = rodata_addr_ + rodata_table_1_off_;
                table_addrs[1] = rodata_addr_ + rodata_table_2_off_;
                table_addrs[2] = rodata_addr_ + rodata_table_4_off_;
                old_table_base = disp_rodata->addr + rodata_table_1_off_;
                printf("PC-relative patching: tables in .rodata at 0x%lx, 0x%lx, 0x%lx\n",
                       table_addrs[0], table_addrs[1], table_addrs[2]);
            } else {
                // Tables in .data.rel.ro (old non-const case)
                table_addrs[0] = data_rel_ro_addr_;
                table_addrs[1] = data_rel_ro_addr_ + table_spacing_;
                table_addrs[2] = data_rel_ro_addr_ + table_spacing_ * 2;
                old_table_base = disp_bss ? disp_bss->addr : 0x8000;
            }

            // Find s_getpc_b64 + s_add_u32 patterns
            uint32_t s_getpc = 0xBE801C00;
            uint32_t s_add = 0x8000FF00;

            auto* disp_text = disp_->find(".text");
            uint64_t disp_text_addr = disp_text ? disp_text->addr : 0;

            int patches = 0;
            for (size_t off = 0; off + 12 <= text_sec->data.size(); off += 4) {
                uint32_t i1, i2;
                memcpy(&i1, text_sec->data.data() + off, 4);
                memcpy(&i2, text_sec->data.data() + off + 4, 4);

                if (i1 == s_getpc && i2 == s_add) {
                    uint32_t old_lit;
                    memcpy(&old_lit, text_sec->data.data() + off + 8, 4);

                    // pc is the new address in merged ELF
                    uint64_t pc = text_addr_ + off + 4;
                    // original_pc is the address in the original dispatcher
                    uint64_t original_pc = disp_text_addr + off + 4;
                    // old_target is the original target (relative to original dispatcher layout)
                    int32_t signed_lit = (int32_t)old_lit;
                    uint64_t old_target = (uint64_t)((int64_t)original_pc + signed_lit);

                    int idx = -1;
                    // Check if target is one of the tables
                    if (rodata_table_2_off_ > 0 && disp_rodata) {
                        uint64_t rodata_base = disp_rodata->addr;
                        if (old_target >= rodata_base + rodata_table_1_off_ &&
                            old_target < rodata_base + rodata_table_1_off_ + FUNC_COUNT * 8) {
                            idx = 0;
                        } else if (old_target >= rodata_base + rodata_table_2_off_ &&
                                   old_target < rodata_base + rodata_table_2_off_ + FUNC_COUNT * 8) {
                            idx = 1;
                        } else if (old_target >= rodata_base + rodata_table_4_off_ &&
                                   old_target < rodata_base + rodata_table_4_off_ + FUNC_COUNT * 8) {
                            idx = 2;
                        }
                    } else {
                        if (old_target >= old_table_base && old_target < old_table_base + 0x6000) {
                            uint64_t rel = old_target - old_table_base;
                            if (rel < table_spacing_) idx = 0;
                            else if (rel < table_spacing_ * 2) idx = 1;
                            else idx = 2;
                        }
                    }

                    if (idx >= 0) {
                        int32_t new_lit = (int32_t)(table_addrs[idx] - pc);
                        memcpy(text_sec->data.data() + off + 8, &new_lit, 4);
                        printf("  Patched table ref: PC=0x%lx, table %d -> 0x%lx (lit=0x%x)\n",
                               pc, idx, table_addrs[idx], (uint32_t)new_lit);
                        patches++;
                    }
                }
            }
            printf("Patched %d PC-relative table references\n", patches);
        }

        // Patch .rodata (KDs) - both dispatcher and specialized
        // (rodata_sec was found earlier in this function)
        if (rodata_sec) {
            // Helper to patch a single KD with max resources
            auto patchKD = [&](uint8_t* kd, size_t kd_off, bool is_specialized, uint64_t text_off = 0) {
                // LDS - patch to max across all kernels
                uint32_t lds = max_lds_;
                memcpy(kd + 0, &lds, 4);

                // Stack - set to max needed by specialized functions
                // The specialized functions we call need scratch space for their local variables
                // Unlike IFC which compiles everything together and can inline/optimize,
                // we call separate functions that have explicit scratch requirements
                uint32_t stack = max_stack_;
                memcpy(kd + 4, &stack, 4);

                // RSRC1 at offset 0x30: VGPR/SGPR (6 bits VGPR granulated, 4 bits SGPR granulated)
                uint32_t rsrc1;
                memcpy(&rsrc1, kd + 0x30, 4);
                int vgpr_g = (max_vgpr_ + 3) / 4 - 1;
                int sgpr_g = (max_sgpr_ + 7) / 8 - 1;
                if (vgpr_g > 0x3F) {
                    printf("  WARNING: clamping VGPR granulated %d to 63 (max 256 VGPRs); max_vgpr_=%d\n", vgpr_g, max_vgpr_);
                    vgpr_g = 0x3F;
                }
                if (sgpr_g > 0xF) {
                    printf("  WARNING: clamping SGPR granulated %d to 15 (max 128 SGPRs); max_sgpr_=%d\n", sgpr_g, max_sgpr_);
                    sgpr_g = 0xF;
                }
                rsrc1 = (rsrc1 & ~0x3FF) | (vgpr_g & 0x3F) | ((sgpr_g & 0xF) << 6);
                memcpy(kd + 0x30, &rsrc1, 4);

                // RSRC2 at offset 0x34: Enable scratch
                // Note: We preserve the original USER_SGPR and properties because the
                // dispatcher code was compiled expecting a specific SGPR layout
                uint32_t rsrc2;
                memcpy(&rsrc2, kd + 0x34, 4);
                if (stack > 0) {
                    rsrc2 |= 1;  // SCRATCH_EN = 1
                }
                memcpy(kd + 0x34, &rsrc2, 4);

                // Patch reserved field at offset 0x2c to match IFC (0x1f vs 0x0f)
                // This field may affect wavefront scheduling or resource allocation
                // IFC builds have 0x1f, our dispatcher compiles with 0x0f
                uint32_t reserved_2c = 0x1f;
                memcpy(kd + 0x2c, &reserved_2c, 4);

                // Patch kernel_code_entry_byte_offset
                // KD offset 0x10: kernel_code_entry_byte_offset (int64_t, relative to KD address)
                uint64_t rodata_addr = 0;
                for (const auto& s : sections_) if (s.name == ".rodata") { rodata_addr = s.addr; break; }

                int64_t entry_offset;
                if (is_specialized) {
                    // For specialized: calculate from scratch
                    entry_offset = (int64_t)(text_addr_ + text_off) - (int64_t)(rodata_addr + kd_off);
                } else {
                    // For dispatcher: adjust original entry_offset by the layout delta
                    // Original layout: disp_text_addr - disp_rodata_addr
                    // New layout: text_addr_ - rodata_addr
                    // delta = new_layout - old_layout
                    auto* disp_text = disp_->find(".text");
                    auto* disp_rodata = disp_->find(".rodata");
                    uint64_t disp_text_addr = disp_text ? disp_text->addr : 0;
                    uint64_t disp_rodata_addr = disp_rodata ? disp_rodata->addr : 0;

                    int64_t old_entry_offset;
                    memcpy(&old_entry_offset, kd + 0x10, 8);

                    int64_t delta = (int64_t)(text_addr_ - rodata_addr) -
                                   (int64_t)(disp_text_addr - disp_rodata_addr);
                    entry_offset = old_entry_offset + delta;

                    printf("  KD[%zu]: old_entry=%ld, delta=%ld, new_entry=%ld\n",
                           kd_off / 64, old_entry_offset, delta, entry_offset);
                }
                memcpy(kd + 0x10, &entry_offset, 8);

                uint16_t props;
                memcpy(&props, kd + 0x3c, 2);
                if (!is_specialized) {
                    printf("  KD[%zu] (dispatcher): LDS=%d, stack=%d, RSRC2=0x%08x\n",
                           kd_off / 64, lds, stack, rsrc2);
                }
            };

            // Patch dispatcher KDs. .rodata has 18 KDs: 3 Generic, 3 Debug (profiling), 12 oneRank.
            // Patch the first 6 (Generic + Debug) with layout delta and max resources; oneRank
            // KDs are from separately compiled code in dispatcher .text and are left as-is.
            for (size_t kd_off : {0UL, 64UL, 128UL, 192UL, 256UL, 320UL}) {
                if (kd_off + 64 > rodata_sec->data.size()) continue;
                uint8_t* kd = rodata_sec->data.data() + kd_off;
                patchKD(kd, kd_off, false);
            }

            // Patch specialized KDs
            for (const auto& [kd_offset, text_off] : specialized_kd_offsets_) {
                if (kd_offset + 64 > rodata_sec->data.size()) continue;
                uint8_t* kd = rodata_sec->data.data() + kd_offset;
                patchKD(kd, kd_offset, true, text_off);
            }
            printf("Patched 6 dispatcher KDs (3 Generic + 3 Debug) + %zu specialized KDs\n",
                   specialized_kd_offsets_.size());
        }

        // Patch .dynsym symbol values for all relocated sections
        SectionInfo* dynsym_sec = nullptr;
        for (auto& s : sections_) if (s.name == ".dynsym") { dynsym_sec = &s; break; }

        // Get old section addresses from dispatcher
        auto* disp_bss = disp_->find(".bss");
        auto* disp_rodata = disp_->find(".rodata");
        auto* disp_text_sec = disp_->find(".text");

        uint64_t old_data = disp_bss ? disp_bss->addr : 0;
        uint64_t old_rodata = disp_rodata ? disp_rodata->addr : 0;
        uint64_t old_text = disp_text_sec ? disp_text_sec->addr : 0;

        // Get new section addresses
        uint64_t new_rodata = 0;
        for (const auto& s : sections_) {
            if (s.name == ".rodata") { new_rodata = s.addr; break; }
        }

        int64_t data_delta = (int64_t)data_addr_ - (int64_t)old_data;
        int64_t rodata_delta = (int64_t)new_rodata - (int64_t)old_rodata;
        int64_t text_delta = (int64_t)text_addr_ - (int64_t)old_text;

        printf("Symbol deltas: .rodata=%+ld, .text=%+ld, .data=%+ld\n",
               rodata_delta, text_delta, data_delta);

        // Build section index map for merged ELF: section name -> index (+1 for NULL)
        std::unordered_map<std::string, uint16_t> new_section_indices;
        for (size_t i = 0; i < sections_.size(); i++) {
            new_section_indices[sections_[i].name] = i + 1;  // +1 for NULL section
        }

        // Build address range to section index map for merged ELF
        struct SectionRange {
            uint64_t start, end;
            uint16_t index;
            std::string name;
        };
        std::vector<SectionRange> merged_ranges;
        for (size_t i = 0; i < sections_.size(); i++) {
            if (sections_[i].isAlloc() && sections_[i].size() > 0) {
                merged_ranges.push_back({
                    sections_[i].addr,
                    sections_[i].addr + sections_[i].size(),
                    (uint16_t)(i + 1),  // +1 for NULL section
                    sections_[i].name
                });
            }
        }

        // Find old section indices from dispatcher
        uint16_t old_bss_idx = 0, old_rodata_idx = 0, old_text_idx = 0;
        {
            auto* bss = disp_->find(".bss");
            auto* rodata = disp_->find(".rodata");
            auto* text = disp_->find(".text");
            if (bss) old_bss_idx = bss->index;
            if (rodata) old_rodata_idx = rodata->index;
            if (text) old_text_idx = text->index;
        }
        printf("Old section indices: .bss=%d, .rodata=%d, .text=%d\n", old_bss_idx, old_rodata_idx, old_text_idx);
        printf("New section indices: .data=%d, .bss=%d, .rodata=%d, .text=%d\n",
               new_section_indices[".data"], new_section_indices[".bss"],
               new_section_indices[".rodata"], new_section_indices[".text"]);

        // Helper to find which merged section an address falls into
        auto findSectionForAddr = [&](uint64_t addr) -> uint16_t {
            for (const auto& r : merged_ranges) {
                if (addr >= r.start && addr < r.end) {
                    return r.index;
                }
            }
            return 0;  // Not found
        };

        // Get strtab for looking up symbol names
        SectionInfo* dynsym_strtab = nullptr;
        for (auto& s : sections_) if (s.name == ".dynstr") { dynsym_strtab = &s; break; }

        // Helper to patch symbol entry
        // strtab_data is the string table for looking up symbol names (can be nullptr)
        auto patchSymbol = [&](uint8_t* sym, const uint8_t* strtab_data, size_t strtab_size) {
            uint32_t name_idx;
            uint64_t val;
            uint16_t shndx;
            memcpy(&name_idx, sym, 4);
            memcpy(&val, sym + 8, 8);
            memcpy(&shndx, sym + 6, 2);

            // Skip ABS symbols (st_shndx == SHN_ABS == 0xFFF1)
            if (shndx == 0xFFF1 || shndx == 0) {
                return;  // Don't modify ABS or undefined symbols
            }

            // Get symbol name if available
            const char* sym_name = nullptr;
            if (strtab_data && name_idx < strtab_size) {
                sym_name = (const char*)strtab_data + name_idx;
            }

            // Special handling for ncclDevFuncTable_* symbols:
            // These should be mapped to .data.rel.ro, not .data/.bss
            // Their relative offset within the old .bss maps to offset within .data.rel.ro
            if (sym_name && strstr(sym_name, "ncclDevFuncTable_")) {
                // Extract table number (1, 2, or 4)
                int table_num = 0;
                if (strstr(sym_name, "ncclDevFuncTable_1")) table_num = 1;
                else if (strstr(sym_name, "ncclDevFuncTable_2")) table_num = 2;
                else if (strstr(sym_name, "ncclDevFuncTable_4")) table_num = 4;

                if (table_num > 0) {
                    // Map to .rodata address where function tables are
                    uint64_t new_val;
                    switch (table_num) {
                        case 1: new_val = rodata_addr_ + rodata_table_1_off_; break;
                        case 2: new_val = rodata_addr_ + rodata_table_2_off_; break;
                        case 4: new_val = rodata_addr_ + rodata_table_4_off_; break;
                        default: new_val = val; break;
                    }

                    // Update symbol value
                    memcpy(sym + 8, &new_val, 8);

                    // Update section index to .rodata
                    uint16_t rodata_idx = new_section_indices[".rodata"];
                    memcpy(sym + 6, &rodata_idx, 2);

                    printf("  Patched %s: 0x%lx -> 0x%lx (section %d -> %d)\n",
                           sym_name, val, new_val, shndx, rodata_idx);
                    return;
                }
            }

            // Standard handling for other symbols
            uint64_t new_val = val;
            int64_t delta = 0;
            if (old_data && val >= old_data && val < old_data + 0x10000) {
                delta = data_delta;
            } else if (old_rodata && val >= old_rodata && val < old_rodata + 0x1000) {
                delta = rodata_delta;
            } else if (old_text && val >= old_text && val < old_text + 0x10000) {
                delta = text_delta;
            }

            if (delta != 0 && val != 0) {
                new_val = (uint64_t)((int64_t)val + delta);
                memcpy(sym + 8, &new_val, 8);
            }

            // Update st_shndx based on the NEW address (after patching)
            // Find which section the symbol's new address falls into
            uint16_t new_shndx = findSectionForAddr(new_val);
            if (new_shndx != 0 && new_shndx != shndx) {
                memcpy(sym + 6, &new_shndx, 2);
            }
        };

        if (dynsym_sec && dynsym_strtab) {
            printf("Patching .dynsym symbols...\n");
            for (size_t i = 0; i + 24 <= dynsym_sec->data.size(); i += 24) {
                uint8_t* sym = dynsym_sec->data.data() + i;

                // Get symbol name
                uint32_t name_idx;
                memcpy(&name_idx, sym, 4);
                const char* sym_name = "";
                if (name_idx < dynsym_strtab->data.size()) {
                    sym_name = (const char*)dynsym_strtab->data.data() + name_idx;
                }

                // ncclDevFuncTable_* symbols: patch their addresses to .rodata
                // These are function pointer tables filled by relocations
                if (strncmp(sym_name, "ncclDevFuncTable_", 17) == 0) {
                    uint64_t new_val = 0;
                    if (strstr(sym_name, "ncclDevFuncTable_1")) new_val = rodata_addr_ + rodata_table_1_off_;
                    else if (strstr(sym_name, "ncclDevFuncTable_2")) new_val = rodata_addr_ + rodata_table_2_off_;
                    else if (strstr(sym_name, "ncclDevFuncTable_4")) new_val = rodata_addr_ + rodata_table_4_off_;

                    if (new_val != 0) {
                        // Update st_value
                        memcpy(sym + 8, &new_val, 8);
                        // Update st_shndx to .rodata
                        uint16_t rodata_idx = 0;
                        for (size_t j = 0; j < sections_.size(); j++) {
                            if (sections_[j].name == ".rodata") { rodata_idx = j + 1; break; }
                        }
                        memcpy(sym + 6, &rodata_idx, 2);
                        // Keep as OBJECT type with hidden visibility
                        sym[4] = 1 | (0 << 4);  // STT_OBJECT | STB_LOCAL
                        sym[5] = 2;  // STV_HIDDEN
                        printf("  Patched dynsym %s: val=0x%lx, shndx=%d\n", sym_name, new_val, rodata_idx);
                    }
                    continue;
                }

                patchSymbol(sym, dynsym_strtab->data.data(), dynsym_strtab->data.size());
            }

            // Add ncclDevFunc_* and oneRankReduce symbols to .dynsym for symbol-based relocations
            // Space was pre-allocated in collectSections() to ensure layout is correct
            // We fill in the pre-allocated space here with actual symbol data
            if (!kernel_text_offsets_.empty() || !onerank_text_offsets_.empty()) {
                printf("Filling %zu ncclDevFunc_* + %zu oneRankReduce symbols into pre-allocated .dynsym/.dynstr...\n",
                       kernel_text_offsets_.size(), onerank_text_offsets_.size());
                
                uint16_t text_shndx = 0;
                for (size_t j = 0; j < sections_.size(); j++) {
                    if (sections_[j].name == ".text") { text_shndx = j + 1; break; }
                }

                // Current position in pre-allocated .dynstr space
                size_t dynstr_pos = prealloc_dynstr_orig_size_;
                // Current slot in pre-allocated .dynsym space
                size_t dynsym_slot = prealloc_dynsym_insert_idx_;
                
                // Fill in ncclDevFunc_* symbols
                for (const auto& [kern, text_off] : kernel_text_offsets_) {
                    // Write name to pre-allocated .dynstr space
                    uint32_t name_off = dynstr_pos;
                    memcpy(dynsym_strtab->data.data() + dynstr_pos, kern->name.c_str(), kern->name.size() + 1);
                    dynstr_pos += kern->name.size() + 1;

                    // Create Elf64_Sym entry as LOCAL HIDDEN
                    Elf64_Sym sym = {};
                    sym.st_name = name_off;
                    sym.st_value = text_addr_ + text_off + kern->func_offset;
                    sym.st_size = kern->code.size() - kern->func_offset;
                    sym.st_info = ELF64_ST_INFO(STB_LOCAL, STT_FUNC);
                    sym.st_other = STV_HIDDEN;
                    sym.st_shndx = text_shndx;

                    // Write to pre-allocated .dynsym slot
                    memcpy(dynsym_sec->data.data() + dynsym_slot * 24, &sym, sizeof(sym));
                    dynsym_slot++;
                }
                
                // Fill in oneRankReduce symbols
                for (size_t i = 0; i < onerank_text_offsets_.size(); i++) {
                    // Write name to pre-allocated .dynstr space
                    uint32_t name_off = dynstr_pos;
                    const std::string& name = (i < onerank_names_.size()) ? onerank_names_[i] : "<unknown>";
                    memcpy(dynsym_strtab->data.data() + dynstr_pos, name.c_str(), name.size() + 1);
                    dynstr_pos += name.size() + 1;
                    
                    // Create Elf64_Sym entry as LOCAL HIDDEN
                    Elf64_Sym sym = {};
                    sym.st_name = name_off;
                    sym.st_value = text_addr_ + onerank_text_offsets_[i];
                    sym.st_size = 0;  // Size unknown, but relocations only need the address
                    sym.st_info = ELF64_ST_INFO(STB_LOCAL, STT_FUNC);
                    sym.st_other = STV_HIDDEN;
                    sym.st_shndx = text_shndx;
                    
                    // Write to pre-allocated .dynsym slot
                    memcpy(dynsym_sec->data.data() + dynsym_slot * 24, &sym, sizeof(sym));
                    dynsym_slot++;
                }
                
                printf("  Filled %zu symbols in .dynsym (total %zu), .dynstr used %zu/%zu bytes\n",
                       kernel_text_offsets_.size() + onerank_text_offsets_.size(),
                       dynsym_sec->data.size() / 24, dynstr_pos, dynsym_strtab->data.size());
            }
        }

        // Also patch .symtab
        SectionInfo* symtab_sec = nullptr;
        SectionInfo* strtab_sec = nullptr;
        for (auto& s : sections_) {
            if (s.name == ".symtab") symtab_sec = &s;
            if (s.name == ".strtab") strtab_sec = &s;
        }

        if (symtab_sec && strtab_sec) {
            printf("Patching .symtab symbols...\n");
            for (size_t i = 0; i + 24 <= symtab_sec->data.size(); i += 24) {
                uint8_t* sym = symtab_sec->data.data() + i;

                // Get symbol name
                uint32_t name_idx;
                memcpy(&name_idx, sym, 4);
                const char* sym_name = "";
                if (name_idx < strtab_sec->data.size()) {
                    sym_name = (const char*)strtab_sec->data.data() + name_idx;
                }

                // ncclDevFuncTable_* symbols: change to LOCAL HIDDEN to match IFC
                // IFC has these as _ZL18ncclDevFuncTable_* (LOCAL HIDDEN)
                if (strncmp(sym_name, "ncclDevFuncTable_", 17) == 0) {
                    uint64_t new_val = 0;
                    if (strstr(sym_name, "ncclDevFuncTable_1")) new_val = rodata_addr_ + rodata_table_1_off_;
                    else if (strstr(sym_name, "ncclDevFuncTable_2")) new_val = rodata_addr_ + rodata_table_2_off_;
                    else if (strstr(sym_name, "ncclDevFuncTable_4")) new_val = rodata_addr_ + rodata_table_4_off_;

                    if (new_val != 0) {
                        // Update st_value
                        memcpy(sym + 8, &new_val, 8);
                        // Update st_shndx to .rodata
                        uint16_t rodata_idx = 0;
                        for (size_t j = 0; j < sections_.size(); j++) {
                            if (sections_[j].name == ".rodata") { rodata_idx = j + 1; break; }
                        }
                        memcpy(sym + 6, &rodata_idx, 2);
                        // Change to LOCAL HIDDEN to match IFC
                        sym[4] = ELF64_ST_INFO(STB_LOCAL, STT_OBJECT);
                        sym[5] = STV_HIDDEN;
                        printf("  Patched symtab %s: val=0x%lx, shndx=%d -> LOCAL HIDDEN\n", sym_name, new_val, rodata_idx);
                    }
                    continue;
                }

                patchSymbol(sym, strtab_sec->data.data(), strtab_sec->data.size());
            }
        }

        // Add specialized kernel symbols to .symtab for debugging
        // These are local symbols that don't affect dynamic linking but allow
        // debuggers to show function names in stack traces
        if (symtab_sec && strtab_sec && !kernel_text_offsets_.empty()) {
            printf("Adding %zu specialized kernel symbols to .symtab...\n", kernel_text_offsets_.size());

            uint16_t text_shndx = new_section_indices[".text"];

            for (const auto& [kern, text_off] : kernel_text_offsets_) {
                // Append name to .strtab
                uint32_t name_off = strtab_sec->data.size();
                strtab_sec->data.insert(strtab_sec->data.end(),
                                        kern->name.begin(), kern->name.end());
                strtab_sec->data.push_back('\0');

                // Create Elf64_Sym entry
                // Use STB_GLOBAL (not STB_LOCAL) because ELF requires local symbols
                // to precede global symbols, and sh_info marks the boundary.
                // STV_HIDDEN ensures they don't pollute the dynamic symbol table.
                // Note: text_off is where the code block starts, but func_offset is the offset
                // of the ncclDevFunc_ function within that block (helper functions come before it).
                Elf64_Sym sym = {};
                sym.st_name = name_off;
                sym.st_value = text_addr_ + text_off + kern->func_offset;
                sym.st_size = kern->code.size() - kern->func_offset;  // Function size, not whole block
                sym.st_info = ELF64_ST_INFO(STB_GLOBAL, STT_FUNC);
                sym.st_other = STV_HIDDEN;
                sym.st_shndx = text_shndx;

                // Append to .symtab
                const uint8_t* p = reinterpret_cast<const uint8_t*>(&sym);
                symtab_sec->data.insert(symtab_sec->data.end(), p, p + sizeof(sym));
            }
        }

        // Add __clang_gpu_used_external symbol for oneRankReduce kernels
        // This symbol is generated by lld during RDC linking but we need to create it
        // manually since we use non-RDC mode. It tracks "used" device functions.
        if (symtab_sec && strtab_sec && !onerank_text_offsets_.empty()) {
            uint16_t data_shndx = new_section_indices[".data"];

            // Append name to .strtab
            uint32_t name_off = strtab_sec->data.size();
            const char* sym_name = "__clang_gpu_used_external";
            strtab_sec->data.insert(strtab_sec->data.end(),
                                   sym_name, sym_name + strlen(sym_name) + 1);

            // Create symbol: LOCAL HIDDEN in .data section
            Elf64_Sym sym;
            memset(&sym, 0, sizeof(sym));
            sym.st_name = name_off;
            sym.st_value = data_addr_;  // Start of .data
            sym.st_size = onerank_text_offsets_.size() * 8;  // 8 bytes per entry
            sym.st_info = ELF64_ST_INFO(STB_LOCAL, STT_OBJECT);
            sym.st_other = STV_HIDDEN;
            sym.st_shndx = data_shndx;

            const uint8_t* p = reinterpret_cast<const uint8_t*>(&sym);
            symtab_sec->data.insert(symtab_sec->data.end(), p, p + sizeof(sym));

            printf("Added __clang_gpu_used_external symbol: addr=0x%lx, size=%zu\n",
                   data_addr_, onerank_text_offsets_.size() * 8);
        }

        // Patch has_dyn_sized_stack and has_recursion ABS symbols to 0
        // These are ABS symbols (st_shndx == SHN_ABS == 0xFFF1)
        if (symtab_sec && strtab_sec) {
            int patched = 0;
            for (size_t i = 0; i + 24 <= symtab_sec->data.size(); i += 24) {
                uint8_t* sym = symtab_sec->data.data() + i;
                uint32_t name_idx;
                uint16_t shndx;
                uint64_t val;
                memcpy(&name_idx, sym, 4);
                memcpy(&shndx, sym + 6, 2);
                memcpy(&val, sym + 8, 8);

                // Only patch ABS symbols with value 1
                if (shndx == 0xFFF1 && val == 1 && name_idx < strtab_sec->data.size()) {
                    const char* name = (const char*)strtab_sec->data.data() + name_idx;
                    // Check if this is a Generic kernel's has_dyn_sized_stack or has_recursion
                    if (strstr(name, "Generic_") &&
                        (strstr(name, "has_dyn_sized_stack") || strstr(name, "has_recursion"))) {
                        val = 0;
                        memcpy(sym + 8, &val, 8);
                        patched++;
                    }
                }
            }
            if (patched > 0) {
                printf("Patched %d ABS symbols (has_dyn_sized_stack/has_recursion) to 0\n", patched);
            }
        }

        // Patch ABS symbols for resource maxima (must match KD and .note)
        // These symbols are in .symtab and .strtab with format: kernelname.suffix
        if (symtab_sec && strtab_sec) {
            const char* strtab = (const char*)strtab_sec->data.data();
            for (size_t i = 0; i + 24 <= symtab_sec->data.size(); i += 24) {
                uint8_t* sym = symtab_sec->data.data() + i;
                uint32_t name_idx;
                uint16_t shndx;
                memcpy(&name_idx, sym, 4);
                memcpy(&shndx, sym + 6, 2);

                // Check if this is an ABS symbol (st_shndx == SHN_ABS = 0xFFF1)
                if (shndx == 0xFFF1 && name_idx < strtab_sec->data.size()) {
                    const char* name = strtab + name_idx;
                    size_t len = strlen(name);

                    // Patch .private_seg_size to max_stack_ so ABS matches KD
                    const char* suffix1 = ".private_seg_size";
                    if (len > strlen(suffix1) && strcmp(name + len - strlen(suffix1), suffix1) == 0) {
                        uint64_t new_val = max_stack_;
                        memcpy(sym + 8, &new_val, 8);
                    }

                    // Patch .num_vgpr to max_vgpr_
                    const char* suffix2 = ".num_vgpr";
                    if (len > strlen(suffix2) && strcmp(name + len - strlen(suffix2), suffix2) == 0) {
                        uint64_t new_val = max_vgpr_;
                        memcpy(sym + 8, &new_val, 8);
                    }

                    // Patch .numbered_sgpr to max_sgpr_
                    const char* suffix3 = ".numbered_sgpr";
                    if (len > strlen(suffix3) && strcmp(name + len - strlen(suffix3), suffix3) == 0) {
                        uint64_t new_val = max_sgpr_;
                        memcpy(sym + 8, &new_val, 8);
                    }
                }
            }
        }

        // Patch .debug_line section addresses
        patchDebugLine();

        // Pad .strtab to 8-byte boundary so subsequent sections (including .debug_abbrev) start aligned
        // .symtab is already aligned, so padding .strtab ensures proper alignment for all following sections
        if (strtab_sec) {
            size_t padding = (8 - (strtab_sec->data.size() % 8)) % 8;
            if (padding > 0) {
                strtab_sec->data.insert(strtab_sec->data.end(), padding, 0);
                printf("Padded .strtab: +%zu bytes (total %zu bytes) to 8-byte boundary\n",
                       padding, strtab_sec->data.size());
            }
        }

        // Build .rela.dyn section with R_AMDGPU_RELATIVE64 relocations
        if (!buildRelocations()) return;
    }

    // Helper to read ULEB128 value from buffer
    static uint64_t readULEB128(const uint8_t*& p, const uint8_t* end) {
        uint64_t result = 0;
        unsigned shift = 0;
        while (p < end) {
            uint8_t byte = *p++;
            result |= (uint64_t)(byte & 0x7f) << shift;
            if ((byte & 0x80) == 0) break;
            shift += 7;
        }
        return result;
    }

    // Helper to write ULEB128 value to buffer
    static void writeULEB128(std::vector<uint8_t>& buf, uint64_t value) {
        // Standard ULEB128 encoding: check BEFORE shifting
        while (value >= 0x80) {
            buf.push_back((value & 0x7f) | 0x80);
            value >>= 7;
        }
        buf.push_back(value & 0x7f);
    }


    void mergeDebugInfoWithDWARFLinker() {
        printf("Merging debug info using DWARFLinker...\n");
        
        // Create DWARFLinker with error handlers
        auto error_handler = [](const llvm::Twine& Err, llvm::StringRef Context, const llvm::DWARFDie* DIE) {
            fprintf(stderr, "DWARFLinker error: %s (context: %s)\n", Err.str().c_str(), Context.str().c_str());
        };
        auto warning_handler = [](const llvm::Twine& Warn, llvm::StringRef Context, const llvm::DWARFDie* DIE) {
            fprintf(stderr, "DWARFLinker warning: %s (context: %s)\n", Warn.str().c_str(), Context.str().c_str());
        };
        auto strings_translator = [](llvm::StringRef Str) { return Str; };  // No translation needed
        
        // CustomStreamer: AMDGPU streamer with CPU and feature string from --target (e.g. gfx942:xnack-:sramecc+).
        LLVMInitializeAMDGPUTargetInfo();
        LLVMInitializeAMDGPUTarget();
        LLVMInitializeAMDGPUTargetMC();
        LLVMInitializeAMDGPUAsmPrinter();
        llvm::Triple triple("amdgcn-amd-amdhsa");
        // Split --target at first ':' into CPU (e.g. gfx942) and FeatureStr (e.g. xnack-:sramecc+).
        // LLVM MCSubtargetInfo requires each feature to start with '+' or '-'; ROCm uses suffix (xnack-, sramecc+).
        std::string cpu_str = "gfx942";
        std::string feature_str;
        if (!gpu_target_.empty()) {
            size_t colon = gpu_target_.find(':');
            if (colon != std::string::npos) {
                cpu_str = gpu_target_.substr(0, colon);
                std::string raw = gpu_target_.substr(colon + 1);
                // Convert ROCm suffix style (xnack-, sramecc+) to LLVM prefix style (-xnack, +sramecc)
                for (size_t i = 0; i < raw.size(); ) {
                    size_t end = raw.find(':', i);
                    if (end == std::string::npos) end = raw.size();
                    std::string part(raw.substr(i, end - i));
                    if (!part.empty() && part[0] != '+' && part[0] != '-') {
                        if (part.back() == '+') {
                            feature_str += "+" + part.substr(0, part.size() - 1);
                        } else if (part.back() == '-') {
                            feature_str += "-" + part.substr(0, part.size() - 1);
                        } else {
                            feature_str += "+" + part;
                        }
                        if (end < raw.size()) feature_str += ",";
                    }
                    i = end + (end < raw.size() ? 1 : 0);
                }
            } else {
                cpu_str = gpu_target_;
            }
        }
        llvm::SmallVector<char, 0> dwarf_streamer_buffer;
        llvm::raw_svector_ostream dwarf_stream(dwarf_streamer_buffer);
        auto streamer_or = llvm::dwarf_linker::classic::CustomStreamer::createStreamer(
            triple,
            cpu_str.c_str(),
            feature_str.empty() ? "" : feature_str.c_str(),
            llvm::dwarf_linker::DWARFLinkerBase::OutputFileType::Object,
            dwarf_stream,
            warning_handler);
        if (!streamer_or) {
            fprintf(stderr, "ERROR: CustomStreamer::createStreamer failed\n");
            llvm::handleAllErrors(streamer_or.takeError(), [](const llvm::ErrorInfoBase& ei) {
                ei.log(llvm::errs());
                llvm::errs() << "\n";
            });
            setFatalError("CustomStreamer::createStreamer failed");
            return;
        }
        std::unique_ptr<llvm::dwarf_linker::classic::CustomStreamer> dwarf_streamer = std::move(*streamer_or);
        printf("Using CustomStreamer for DWARF emission (CPU=%s, Features=%s)\n", cpu_str.c_str(), feature_str.c_str());

        llvm::dwarf_linker::classic::DWARFLinker linker(error_handler, warning_handler, strings_translator);
        linker.setOutputDWARFEmitter(dwarf_streamer.get());
        
        // Set options to simplify linking (no ODR, full update not just indexes)
        linker.setNoODR(true);  // Don't unique types - simpler for device code
        linker.setUpdateIndexTablesOnly(false);  // Full DWARF update, not just indexes
        
        // Track max DWARF version from input files (like dsymutil does)
        uint16_t max_dwarf_version = 0;
        std::function<void(const llvm::DWARFUnit&)> on_cu_loaded =
            [&max_dwarf_version](const llvm::DWARFUnit& Unit) {
                max_dwarf_version = std::max(Unit.getVersion(), max_dwarf_version);
            };
        
        // Keep DWARFFile objects alive - they're stored by reference in linker
        std::vector<llvm::dwarf_linker::DWARFFile> dwarf_files;
        dwarf_files.reserve(1 + kernel_text_offsets_.size());
        
        // CRITICAL: Store ObjectFile and MemoryBuffer to keep them alive (DWARFContext keeps references)
        // We need to keep these alive until DWARFContext is destroyed
        struct KernelObjectFiles {
            std::unique_ptr<llvm::object::ObjectFile> obj;
            std::unique_ptr<llvm::MemoryBuffer> buf;
        };
        std::vector<KernelObjectFiles> kernel_object_files;
        kernel_object_files.reserve(kernel_text_offsets_.size());
        
        struct DispatcherObjectFiles {
            std::unique_ptr<llvm::object::ObjectFile> obj;
            std::unique_ptr<llvm::MemoryBuffer> buf;
        };
        std::optional<DispatcherObjectFiles> dispatcher_object_files;
        
        // Store dispatcher info to add it LAST (to test if ordering matters)
        struct DispatcherInfo {
            std::string path;
            std::unique_ptr<llvm::DWARFContext> dwarf;
            std::unique_ptr<DeviceLinkerAddressesMap> addresses;
        };
        std::optional<DispatcherInfo> dispatcher_info;
        
        // Prepare dispatcher (but don't add yet - add it last)
        // Use original ELF file directly instead of creating sections manually
        if (disp_file_ && disp_file_->valid() && !disp_path_.empty()) {
            printf("Preparing dispatcher for DWARFLinker (will add last)...\n");
            
            // Read original dispatcher ELF file directly
            MappedFile dispatcher_file(disp_path_);
            if (!dispatcher_file.valid()) {
                fprintf(stderr, "WARNING: Failed to open dispatcher file %s, skipping dispatcher DWARF\n", disp_path_.c_str());
            } else {
                llvm::StringRef dispatcher_elf_ref(reinterpret_cast<const char*>(dispatcher_file.data()), dispatcher_file.size());
                std::unique_ptr<llvm::MemoryBuffer> dispatcher_buf = llvm::MemoryBuffer::getMemBufferCopy(dispatcher_elf_ref, disp_path_);
                
                llvm::Expected<std::unique_ptr<llvm::object::ObjectFile>> dispatcher_obj_or =
                    llvm::object::ObjectFile::createObjectFile(dispatcher_buf->getMemBufferRef());
                if (!dispatcher_obj_or) {
                    fprintf(stderr, "WARNING: Failed to create ObjectFile from dispatcher %s\n", disp_path_.c_str());
                    llvm::consumeError(dispatcher_obj_or.takeError());
                } else {
                    std::unique_ptr<llvm::object::ObjectFile> dispatcher_obj = std::move(dispatcher_obj_or.get());
                    
                    // Create DWARFContext from ObjectFile (same approach as kernels)
                    std::unique_ptr<llvm::DWARFContext> disp_dwarf =
                        llvm::DWARFContext::create(*dispatcher_obj, llvm::DWARFContext::ProcessDebugRelocations::Process,
                                                    nullptr, "", llvmDwarfErrorHandler,
                                                    llvmDwarfWarningHandler, false);
                    if (!disp_dwarf) {
                        fprintf(stderr, "WARNING: Failed to create DWARFContext from dispatcher - skipping dispatcher DWARF\n");
                    } else {
                        // Get dispatcher .text address
                        auto* disp_text = disp_->find(".text");
                        uint64_t disp_orig_text_addr = disp_text ? disp_text->addr : 0;
                        uint64_t disp_new_text_addr = text_addr_;
                        uint64_t disp_text_size = disp_text ? disp_text->size : 0;
                        
                        // Create address map for dispatcher
                        auto disp_addresses = std::make_unique<DeviceLinkerAddressesMap>(
                            disp_orig_text_addr, disp_new_text_addr, disp_text_size);
                        if (!disp_addresses) {
                            fprintf(stderr, "ERROR: Failed to create AddressesMap for dispatcher\n");
                            setFatalError("Failed to create AddressesMap for dispatcher");
                            return;
                        }
                        
                        // Store ObjectFile and MemoryBuffer to keep them alive (DWARFContext keeps references)
                        dispatcher_object_files = DispatcherObjectFiles{std::move(dispatcher_obj), std::move(dispatcher_buf)};
                        
                        // Store dispatcher info to add it last
                        dispatcher_info = DispatcherInfo{disp_path_, std::move(disp_dwarf), std::move(disp_addresses)};
                    }
                }
            }
        }
        
        // Add each kernel
        // Note: Only kernels in kernel_text_offsets_ have chunks (dispatcher + mapped kernels)
        // Iterate through kernel_text_offsets_ to match chunks correctly
        // For testing: limit number of kernels if MAX_KERNELS_FOR_TEST is set
        size_t max_kernels = kernel_text_offsets_.size();
        const char* max_kernels_env = getenv("MAX_KERNELS_FOR_TEST");
        if (max_kernels_env) {
            max_kernels = std::min(max_kernels, (size_t)atoi(max_kernels_env));
            printf("TEST MODE: Limiting to %zu kernels (out of %zu total)\n", max_kernels, kernel_text_offsets_.size());
        }
        printf("Processing %zu kernels (expecting %zu chunks including dispatcher)...\n", 
               max_kernels, debug_info_chunks_.size());
        
        for (size_t i = 0; i < max_kernels; i++) {
            const KernelInfo* kernel = kernel_text_offsets_[i].first;
            if (!kernel) {
                fprintf(stderr, "  Warning: kernel_text_offsets_[%zu] has null kernel pointer, skipping\n", i);
                continue;
            }
            
            if (kernel->debug_info.empty() || kernel->debug_abbrev.empty()) {
                continue;  // Skip kernels without debug info
            }
            
            printf("Adding kernel %zu (%s) to DWARFLinker...\n", i, kernel->source_file.c_str());
            fflush(stdout);  // Flush to see progress before potential crash
            
            // Chunk index = 1 (skip dispatcher) + kernel index
            size_t chunk_idx = 1 + i;
            if (chunk_idx >= debug_info_chunks_.size()) {
                fprintf(stderr, "  ERROR: Chunk index %zu out of range (chunks size: %zu) for kernel %s\n",
                       chunk_idx, debug_info_chunks_.size(), kernel->source_file.c_str());
                setFatalError("Chunk index out of range");
                return;
            }
            
            const auto& chunk = debug_info_chunks_[chunk_idx];
            
            // Create minimal ELF from kernel's debug sections for DWARFContext
            // DEBUG: Verify input data before creating ELF
            if (kernel->debug_info.size() > 14) {
                fprintf(stderr, "  DEBUG: kernel->debug_info[14] = 0x%02x (size=%zu)\n", 
                       kernel->debug_info[14], kernel->debug_info.size());
            }
            
            // Check if kernel has debug info - if not, skip DWARF processing silently
            if (kernel->debug_info.empty() || kernel->debug_abbrev.empty()) {
                // Kernel has no debug info - this is normal for many kernels, skip silently
                continue;
            }
            
            // Use original ELF file directly instead of creating minimal ELF
            // DWARFLinker can handle full ELF files, and this avoids potential format issues
            if (kernel->source_file.empty()) {
                fprintf(stderr, "  Warning: Kernel %zu has no source_file path, skipping\n", i);
                continue;
            }
            
            // Read original ELF file directly
            MappedFile kernel_file(kernel->source_file);
            if (!kernel_file.valid()) {
                fprintf(stderr, "  Warning: Failed to open kernel file %s, skipping\n", kernel->source_file.c_str());
                continue;
            }
            
            llvm::StringRef kernel_elf_ref(reinterpret_cast<const char*>(kernel_file.data()), kernel_file.size());
            std::unique_ptr<llvm::MemoryBuffer> kernel_buf = llvm::MemoryBuffer::getMemBufferCopy(kernel_elf_ref, kernel->source_file);
            
            llvm::Expected<std::unique_ptr<llvm::object::ObjectFile>> kernel_obj_or =
                llvm::object::ObjectFile::createObjectFile(kernel_buf->getMemBufferRef());
            if (!kernel_obj_or) {
                fprintf(stderr, "  Warning: Failed to create ObjectFile from kernel %s ELF\n", kernel->source_file.c_str());
                llvm::consumeError(kernel_obj_or.takeError());
                // Free kernel_buf before continuing
                kernel_buf.reset();
                continue;
            }
            std::unique_ptr<llvm::object::ObjectFile> kernel_obj = std::move(kernel_obj_or.get());
            
            // Create DWARFContext from ObjectFile (more reliable than section-based API)
            // CRITICAL: kernel_obj and kernel_buf must stay alive - DWARFContext keeps references to them
            // We can't free them until DWARFContext is destroyed. However, the ELF vector was already freed above.
            std::unique_ptr<llvm::DWARFContext> kernel_dwarf =
                llvm::DWARFContext::create(*kernel_obj, llvm::DWARFContext::ProcessDebugRelocations::Process,
                                            nullptr, "", llvmDwarfErrorHandler,
                                            llvmDwarfWarningHandler, false);
            
            // NOTE: We keep kernel_obj and kernel_buf alive because DWARFContext may keep references.
            // They will be freed when DWARFContext is destroyed (when dwarf_files is cleared).
            // The ELF vector was already freed above to save memory.
            if (!kernel_dwarf) {
                fprintf(stderr, "  Warning: Failed to create DWARFContext from kernel %s\n", kernel->source_file.c_str());
                continue;
            }
            
            // Minimal verification - only check if we can get CUs (don't parse them all)
            auto cu_range = kernel_dwarf->compile_units();
            size_t num_cus = std::distance(cu_range.begin(), cu_range.end());
            if (num_cus == 0) {
                fprintf(stderr, "  Warning: No compile units found in kernel %s, skipping\n", kernel->source_file.c_str());
                continue;
            }
            
            // Create address map for kernel
            uint64_t kernel_orig_text_addr = chunk.orig_text_addr;
            uint64_t kernel_new_text_addr = text_addr_ + chunk.new_text_offset;
            uint64_t kernel_text_size = kernel->code.size();
            
            if (kernel_text_size == 0) {
                fprintf(stderr, "  Warning: Kernel %s has zero code size, skipping\n", kernel->source_file.c_str());
                continue;
            }
            
            auto kernel_addresses = std::make_unique<DeviceLinkerAddressesMap>(
                kernel_orig_text_addr, kernel_new_text_addr, kernel_text_size);
            if (!kernel_addresses) {
                fprintf(stderr, "  ERROR: Failed to create AddressesMap for kernel %s, skipping\n", kernel->source_file.c_str());
                continue;
            }
            
            // CRITICAL: Store ObjectFile and MemoryBuffer to keep them alive
            // DWARFContext keeps references to them, so they must stay alive until DWARFContext is destroyed
            kernel_object_files.emplace_back(KernelObjectFiles{std::move(kernel_obj), std::move(kernel_buf)});
            
            // Create DWARFFile and add to linker (use OnCUDieLoaded to track max version)
            // CRITICAL: Store in vector first, then pass reference to ensure stability
            // The vector was reserved, so references won't be invalidated
            dwarf_files.emplace_back(
                kernel->source_file, std::move(kernel_dwarf), std::move(kernel_addresses));
            
            // NOTE: The ELF vector was already freed above to save memory.
            // ObjectFile and MemoryBuffer are stored in kernel_object_files and will be freed
            // when kernel_object_files is cleared (after link() completes).
            
            // Add the file to linker (minimal output to reduce memory pressure)
            if ((i + 1) % 100 == 0) {
                fprintf(stderr, "  Added %zu kernels to DWARFLinker...\n", i + 1);
            }
            linker.addObjectFile(dwarf_files.back(), nullptr, on_cu_loaded);
        }
        
        // Add dispatcher LAST (to test if ordering matters)
        if (dispatcher_info.has_value()) {
            printf("Adding dispatcher to DWARFLinker (last)...\n");
            dwarf_files.emplace_back(
                dispatcher_info->path, 
                std::move(dispatcher_info->dwarf), 
                std::move(dispatcher_info->addresses));
            
            // Add dispatcher to linker
            linker.addObjectFile(dwarf_files.back(), nullptr, on_cu_loaded);
        }
        
        // Set target DWARF version AFTER adding all object files (like dsymutil does)
        // If no CUs were found, default to DWARF5 (specialized kernels use DWARF5)
        if (max_dwarf_version == 0)
            max_dwarf_version = 5;
        printf("Added %zu object file(s) to DWARFLinker, setting target DWARF version to %u\n", 
               dwarf_files.size(), max_dwarf_version);
        llvm::Error version_error = linker.setTargetDWARFVersion(max_dwarf_version);
        if (version_error) {
            fprintf(stderr, "ERROR: Failed to set target DWARF version\n");
            llvm::handleAllErrors(std::move(version_error), [](const llvm::ErrorInfoBase& ei) {
                std::string msg_str;
                llvm::raw_string_ostream msg(msg_str);
                ei.log(msg);
                fprintf(stderr, "%s\n", msg_str.c_str());
            });
            setFatalError("Failed to set target DWARF version");
            return;
        }
        
        // Link all objects together - this will call emitter methods to write merged sections
        size_t num_kernels_added = max_kernels;
        size_t num_objects = num_kernels_added + (dispatcher_info.has_value() ? 1 : 0);
        printf("Linking DWARF info (added %zu objects: %zu kernels + %s dispatcher)...\n",
               num_objects, num_kernels_added, dispatcher_info.has_value() ? "1" : "0");
        fflush(stdout);
        
        // Limit memory: process one file at a time so each DWARFContext is unloaded after clone (see plan §9).
        linker.setNumThreads(1);
        
        // Call link() to merge all DWARF
        printf("Linking DWARF info...\n");
        
        llvm::Error link_error = linker.link();
        
        if (link_error) {
            fprintf(stderr, "ERROR: DWARFLinker.link() failed\n");
            llvm::handleAllErrors(std::move(link_error), [](const llvm::ErrorInfoBase& ei) {
                std::string msg_str;
                llvm::raw_string_ostream msg(msg_str);
                ei.log(msg);
                fprintf(stderr, "%s\n", msg_str.c_str());
            });
            setFatalError("DWARFLinker.link() failed");
            return;
        }
        
        dwarf_streamer->finish();
        
        // CRITICAL: Free all DWARFContext objects immediately after linking to free memory
        printf("Freeing DWARFContext objects to reduce memory usage...\n");
        kernel_object_files.clear();
        kernel_object_files.shrink_to_fit();
        dispatcher_object_files.reset();
        dwarf_files.clear();
        dwarf_files.shrink_to_fit();
        if (dispatcher_info.has_value()) {
            dispatcher_info.reset();
        }
        
        // Parse streamer output ELF and add .debug_* sections
        llvm::StringRef buffer_ref(dwarf_streamer_buffer.data(), dwarf_streamer_buffer.size());
        auto mem_buf = llvm::MemoryBuffer::getMemBufferCopy(buffer_ref, "dwarf_streamer_output");
        auto obj_or = llvm::object::ObjectFile::createObjectFile(mem_buf->getMemBufferRef());
        if (!obj_or) {
            fprintf(stderr, "ERROR: Failed to parse CustomStreamer output as object file\n");
            llvm::handleAllErrors(obj_or.takeError(), [](const llvm::ErrorInfoBase& ei) {
                ei.log(llvm::errs());
                llvm::errs() << "\n";
            });
            setFatalError("CustomStreamer output parse failed");
            return;
        }
        llvm::object::ObjectFile& obj = **obj_or;
        static const char* const debug_section_names[] = {
            ".debug_abbrev", ".debug_info", ".debug_str", ".debug_str_offsets",
            ".debug_addr", ".debug_rnglists", ".debug_ranges", ".debug_line",
            ".debug_line_str"
        };
        for (const char* sec_name : debug_section_names) {
            for (auto it = obj.section_begin(); it != obj.section_end(); ++it) {
                auto name_err = it->getName();
                if (!name_err || *name_err != sec_name) continue;
                auto contents = it->getContents();
                if (!contents) continue;
                SectionInfo sec;
                sec.name = sec_name;
                sec.type = SHT_PROGBITS;
                sec.flags = (strcmp(sec_name, ".debug_str") == 0) ? (SHF_MERGE | SHF_STRINGS) : 0;
                sec.alignment = 1;
                if (strcmp(sec_name, ".debug_str") == 0) sec.entsize = 1;
                sec.data.assign(contents->begin(), contents->end());
                printf("Adding %s: %zu bytes (from CustomStreamer)\n", sec_name, sec.data.size());
                sections_.push_back(std::move(sec));
                break;
            }
        }
        printf("DWARFLinker merge complete: %zu total streamer bytes\n",
               (size_t)dwarf_streamer_buffer.size());
    }


    void patchDebugLine() {
        if (debug_line_chunks_.empty()) return;
        // When using DWARFLinker, .debug_line comes from CustomStreamer output; nothing to patch.
        if (!sections_.empty()) {
            for (const auto& s : sections_) {
                if (s.name == ".debug_line") return;  // Debug sections from streamer
            }
        }
    }

    bool buildRelocations() {
        // Find the existing .rela.dyn section (created as placeholder)
        SectionInfo* rela_sec = nullptr;
        for (auto& s : sections_) {
            if (s.name == ".rela.dyn") { rela_sec = &s; break; }
        }
        if (!rela_sec) {
            setFatalError(".rela.dyn section not found");
            return false;
        }

        // Find .dynsym and .dynstr sections
        SectionInfo* dynsym_sec = nullptr;
        SectionInfo* dynsym_strtab = nullptr;
        for (auto& s : sections_) {
            if (s.name == ".dynsym") dynsym_sec = &s;
            if (s.name == ".dynstr") dynsym_strtab = &s;
        }
        if (!dynsym_sec) {
            setFatalError(".dynsym section not found - required for symbol-based relocations");
            return false;
        }
        if (!dynsym_strtab) {
            setFatalError(".dynstr section not found - required for symbol-based relocations");
            return false;
        }

        // Build R_AMDGPU_RELATIVE64 relocations for function tables (like IFC)
        // These relocations tell the runtime to fill table entries with function addresses
        // The runtime applies these per-GPU during code object loading
        rela_sec->data.clear();

        // R_AMDGPU_RELATIVE64 = 13 (0x0d)
        // Each relocation: offset (8) + info (8) + addend (8) = 24 bytes
        const uint64_t R_AMDGPU_RELATIVE64 = 13;
        const uint64_t R_AMDGPU_ABS64 = 3;

        // Build map from function address to symbol index in .dynsym
        // This allows us to use symbol-based relocations instead of absolute addresses
        std::unordered_map<uint64_t, uint32_t> addr_to_symidx;
        printf("Building address-to-symbol-index map from .dynsym...\n");
        uint16_t text_shndx = 0;
        uint64_t text_size = 0;
        for (size_t j = 0; j < sections_.size(); j++) {
            if (sections_[j].name == ".text") {
                text_shndx = j + 1;
                text_size = sections_[j].size();
                break;
            }
        }
        
        for (size_t sym_idx = 0; sym_idx * 24 < dynsym_sec->data.size(); sym_idx++) {
            uint8_t* sym = dynsym_sec->data.data() + sym_idx * 24;
            uint64_t sym_val;
            uint16_t sym_shndx;
            uint32_t name_idx;
            memcpy(&sym_val, sym + 8, 8);
            memcpy(&sym_shndx, sym + 6, 2);
            memcpy(&name_idx, sym, 4);
            
            // Only map symbols in .text section (functions)
            if (sym_shndx == text_shndx && sym_val >= text_addr_ && sym_val < text_addr_ + text_size) {
                // Get symbol name for debugging
                const char* sym_name = "";
                if (name_idx < dynsym_strtab->data.size()) {
                    sym_name = (const char*)dynsym_strtab->data.data() + name_idx;
                }
                addr_to_symidx[sym_val] = sym_idx;
            }
        }
        printf("  Mapped %zu function addresses to symbol indices\n", addr_to_symidx.size());
        if (addr_to_symidx.empty()) {
            setFatalError("No function symbols found in .dynsym - cannot create symbol-based relocations");
            return false;
        }

        // Verify that all functions in tables have symbols BEFORE creating relocations
        // This gives us better diagnostics upfront rather than failing during relocation creation
        struct MissingSymbol {
            uint64_t addr;
            std::string name;
            std::string source;
        };
        std::vector<MissingSymbol> missing_symbols;
        
        // Check table_1 functions
        for (int i = 0; i < FUNC_COUNT; i++) {
            if (table_1_[i]) {
                uint64_t func_addr = text_addr_ + table_1_[i];
                if (addr_to_symidx.find(func_addr) == addr_to_symidx.end()) {
                    MissingSymbol ms;
                    ms.addr = func_addr;
                    ms.name = names_1_[i].empty() ? "<unknown>" : names_1_[i];
                    ms.source = "table_1";
                    missing_symbols.push_back(ms);
                }
            }
        }
        // Check table_2 functions
        for (int i = 0; i < FUNC_COUNT; i++) {
            if (table_2_[i]) {
                uint64_t func_addr = text_addr_ + table_2_[i];
                if (addr_to_symidx.find(func_addr) == addr_to_symidx.end()) {
                    MissingSymbol ms;
                    ms.addr = func_addr;
                    ms.name = names_2_[i].empty() ? "<unknown>" : names_2_[i];
                    ms.source = "table_2";
                    missing_symbols.push_back(ms);
                }
            }
        }
        // Check table_4 functions
        for (int i = 0; i < FUNC_COUNT; i++) {
            if (table_4_[i]) {
                uint64_t func_addr = text_addr_ + table_4_[i];
                if (addr_to_symidx.find(func_addr) == addr_to_symidx.end()) {
                    MissingSymbol ms;
                    ms.addr = func_addr;
                    ms.name = names_4_[i].empty() ? "<unknown>" : names_4_[i];
                    ms.source = "table_4";
                    missing_symbols.push_back(ms);
                }
            }
        }
        // Check oneRankReduce kernels
        for (size_t i = 0; i < onerank_text_offsets_.size(); i++) {
            uint64_t func_addr = text_addr_ + onerank_text_offsets_[i];
            if (addr_to_symidx.find(func_addr) == addr_to_symidx.end()) {
                MissingSymbol ms;
                ms.addr = func_addr;
                ms.name = (i < onerank_names_.size()) ? onerank_names_[i] : "<unknown>";
                ms.source = "onerank";
                missing_symbols.push_back(ms);
            }
        }

        if (!missing_symbols.empty()) {
            fprintf(stderr, "\nERROR: Missing symbols for %zu function(s) referenced in tables but not in .dynsym:\n\n", missing_symbols.size());
            
            // Group by source
            std::map<std::string, std::vector<MissingSymbol>> by_source;
            for (const auto& ms : missing_symbols) {
                by_source[ms.source].push_back(ms);
            }
            
            for (const auto& [source, syms] : by_source) {
                fprintf(stderr, "  From %s (%zu functions):\n", source.c_str(), syms.size());
                for (const auto& ms : syms) {
                    fprintf(stderr, "    0x%lx: %s\n", ms.addr, ms.name.c_str());
                }
                fprintf(stderr, "\n");
            }
            
            // Also show what IS in .dynsym for comparison
            fprintf(stderr, "Symbols currently in .dynsym (text section functions):\n");
            int shown = 0;
            for (const auto& [addr, symidx] : addr_to_symidx) {
                if (shown++ < 20) {  // Show first 20
                    uint8_t* sym = dynsym_sec->data.data() + symidx * 24;
                    uint32_t name_idx;
                    memcpy(&name_idx, sym, 4);
                    const char* sym_name = "";
                    if (name_idx < dynsym_strtab->data.size()) {
                        sym_name = (const char*)dynsym_strtab->data.data() + name_idx;
                    }
                    fprintf(stderr, "  0x%lx: %s\n", addr, sym_name);
                }
            }
            if (addr_to_symidx.size() > 20) {
                fprintf(stderr, "  ... and %zu more\n", addr_to_symidx.size() - 20);
            }
            fprintf(stderr, "\n");
            
            setFatalError("All functions must have symbols in .dynsym for symbol-based relocations");
            return false;
        }

        auto addRelocWithSymbol = [&](uint64_t offset, uint64_t func_addr) {
            size_t pos = rela_sec->data.size();
            rela_sec->data.resize(pos + 24);
            memcpy(rela_sec->data.data() + pos, &offset, 8);
            
            // Look up symbol index for this function address
            // Try exact match first
            auto it = addr_to_symidx.find(func_addr);
            if (it == addr_to_symidx.end()) {
                // If exact match fails, try to find the closest symbol that contains this address
                // (in case func_addr is within a function, not at its start)
                uint32_t best_symidx = 0;
                uint64_t best_dist = UINT64_MAX;
                for (const auto& [sym_addr, symidx] : addr_to_symidx) {
                    if (func_addr >= sym_addr) {
                        uint64_t dist = func_addr - sym_addr;
                        if (dist < best_dist) {
                            // Check if this symbol's size covers the address
                            // We don't have symbol size easily accessible, so use a reasonable threshold
                            // Most functions are at least a few bytes, so if we're within 64KB, assume it's the right symbol
                            if (dist < 0x10000) {
                                best_dist = dist;
                                best_symidx = symidx;
                            }
                        }
                    }
                }
                if (best_dist < UINT64_MAX) {
                    it = addr_to_symidx.find(func_addr - best_dist);
                }
            }
            
            if (it == addr_to_symidx.end()) {
                fprintf(stderr, "ERROR: No symbol found for function address 0x%lx\n", func_addr);
                setFatalError("Failed to find symbol for function address - all functions must have symbols in .dynsym");
                return false;
            }
            
            // Use R_AMDGPU_ABS64 with symbol index
            // r_info = (symbol_index << 32) | relocation_type
            uint64_t info = ((uint64_t)it->second << 32) | R_AMDGPU_ABS64;
            uint64_t addend = 0;  // No addend needed when using symbol
            memcpy(rela_sec->data.data() + pos + 8, &info, 8);
            memcpy(rela_sec->data.data() + pos + 16, &addend, 8);
            return true;
        };

        // Generate relocations for each function table entry
        // Tables are in .rodata at their original PC-relative offsets
        uint64_t table1_addr = rodata_addr_ + rodata_table_1_off_;
        uint64_t table2_addr = rodata_addr_ + rodata_table_2_off_;
        uint64_t table4_addr = rodata_addr_ + rodata_table_4_off_;

        int reloc_count = 0;
        for (int i = 0; i < FUNC_COUNT; i++) {
            if (table_1_[i]) {
                uint64_t func_addr = text_addr_ + table_1_[i];
                if (!addRelocWithSymbol(table1_addr + i * 8, func_addr)) {
                    return false;  // Fatal error already set
                }
                reloc_count++;
            }
            if (table_2_[i]) {
                uint64_t func_addr = text_addr_ + table_2_[i];
                if (!addRelocWithSymbol(table2_addr + i * 8, func_addr)) {
                    return false;  // Fatal error already set
                }
                reloc_count++;
            }
            if (table_4_[i]) {
                uint64_t func_addr = text_addr_ + table_4_[i];
                if (!addRelocWithSymbol(table4_addr + i * 8, func_addr)) {
                    return false;  // Fatal error already set
                }
                reloc_count++;
            }
        }

        // Add relocations for __clang_gpu_used_external (oneRankReduce kernel addresses)
        // IFC has this symbol in .data with 12 relocations pointing to oneRankReduce kernels
        // These relocations fill the array that tracks "used" device functions
        for (size_t i = 0; i < onerank_text_offsets_.size(); i++) {
            // Each entry in __clang_gpu_used_external is 8 bytes
            uint64_t func_addr = text_addr_ + onerank_text_offsets_[i];
            if (!addRelocWithSymbol(data_addr_ + i * 8, func_addr)) {
                return false;  // Fatal error already set
            }
            reloc_count++;
        }

        printf("Built .rela.dyn with %d relocations (all using symbol indices with R_AMDGPU_ABS64)\n", reloc_count);
        if (!onerank_text_offsets_.empty()) {
            printf("  (includes %zu relocations for __clang_gpu_used_external)\n", onerank_text_offsets_.size());
        }

        // Record address/size
        rela_addr_ = rela_sec->addr;
        rela_size_ = rela_sec->data.size();

        printf("  .rela.dyn @ 0x%06lx  size=0x%06zx\n", rela_addr_, rela_size_);

        // Now populate .dynamic section with final rela addresses
        populateDynamicSection();
        return true;
    }

    void populateDynamicSection() {
        for (auto& s : sections_) {
            if (s.name == ".dynamic") {
                std::vector<uint8_t>& dyn_data = s.data;
                dyn_data.clear();

                auto addDyn = [&](uint64_t tag, uint64_t val) {
                    size_t p = dyn_data.size();
                    dyn_data.resize(p + 16);
                    memcpy(dyn_data.data() + p, &tag, 8);
                    memcpy(dyn_data.data() + p + 8, &val, 8);
                };

                // Find section addresses for dynamic entries
                uint64_t dynsym_addr = 0, dynstr_addr = 0, dynstr_sz = 0, gnuhash_addr = 0, hash_addr = 0;
                for (const auto& sec : sections_) {
                    if (sec.name == ".dynsym") dynsym_addr = sec.addr;
                    else if (sec.name == ".dynstr") { dynstr_addr = sec.addr; dynstr_sz = sec.size(); }
                    else if (sec.name == ".gnu.hash") gnuhash_addr = sec.addr;
                    else if (sec.name == ".hash") hash_addr = sec.addr;
                }

                // Add entries in same order as IFC
                if (rela_addr_ && rela_size_) {
                    addDyn(7, rela_addr_);                   // DT_RELA
                    addDyn(8, rela_size_);                   // DT_RELASZ
                    addDyn(9, 24);                           // DT_RELAENT
                    addDyn(0x6ffffff9, rela_size_ / 24);     // DT_RELACOUNT
                }
                if (dynsym_addr) addDyn(6, dynsym_addr);    // DT_SYMTAB
                addDyn(11, 24);                              // DT_SYMENT
                if (dynstr_addr) addDyn(5, dynstr_addr);    // DT_STRTAB
                if (dynstr_sz) addDyn(10, dynstr_sz);       // DT_STRSZ
                if (gnuhash_addr) addDyn(0x6ffffef5, gnuhash_addr);  // DT_GNU_HASH
                if (hash_addr) addDyn(4, hash_addr);        // DT_HASH
                addDyn(0, 0);  // DT_NULL

                printf("Populated .dynamic section: %zu bytes (%zu entries)\n",
                       dyn_data.size(), dyn_data.size() / 16);
                break;
            }
        }
    }

    // ========== Write Output ==========
    bool writeOutput(const std::string& path) {
        // Build .shstrtab
        std::vector<uint8_t> shstrtab;
        shstrtab.push_back(0);
        std::vector<uint32_t> name_offs;
        name_offs.push_back(0);

        for (const auto& s : sections_) {
            name_offs.push_back(shstrtab.size());
            shstrtab.insert(shstrtab.end(), s.name.begin(), s.name.end());
            shstrtab.push_back(0);
        }

        uint32_t shstr_name_off = shstrtab.size();
        const char* shstr_name = ".shstrtab";
        shstrtab.insert(shstrtab.end(), shstr_name, shstr_name + strlen(shstr_name) + 1);

        // Section offsets - recalculate non-alloc sections since patchSections may have
        // modified their sizes (e.g., adding symbols to .symtab/.strtab)
        std::vector<uint64_t> section_offsets;
        uint64_t offset = 0;

        // First pass: use layout-computed offsets for alloc sections, track max offset
        for (const auto& s : sections_) {
            if (s.isAlloc()) {
                section_offsets.push_back(s.offset);
                if (s.type != SHT_NOBITS) {
                    offset = std::max(offset, s.offset + s.fileSize());
                }
            } else {
                section_offsets.push_back(0);  // Will be computed below
            }
        }

        // Second pass: assign offsets to non-alloc sections sequentially after alloc sections
        for (size_t i = 0; i < sections_.size(); i++) {
            if (!sections_[i].isAlloc() && sections_[i].type != SHT_NOBITS) {
                offset = (offset + sections_[i].alignment - 1) & ~(sections_[i].alignment - 1);
                section_offsets[i] = offset;
                offset += sections_[i].fileSize();
            }
        }

        // .shstrtab
        offset = (offset + 7) & ~7UL;
        uint64_t shstrtab_off = offset;
        offset += shstrtab.size();

        // Section headers
        offset = (offset + 7) & ~7UL;
        uint64_t shdr_off = offset;
        size_t num_shdrs = sections_.size() + 2;  // NULL + sections + .shstrtab

        // Build output buffer
        std::vector<uint8_t> out(shdr_off + num_shdrs * 64);

        // ELF header
        Elf64_Ehdr ehdr = {};
        memcpy(ehdr.e_ident, ELFMAG, SELFMAG);
        ehdr.e_ident[EI_CLASS] = ELFCLASS64;
        ehdr.e_ident[EI_DATA] = ELFDATA2LSB;
        ehdr.e_ident[EI_VERSION] = EV_CURRENT;
        ehdr.e_ident[EI_OSABI] = 64;
        ehdr.e_ident[EI_ABIVERSION] = 4;
        ehdr.e_type = ET_DYN;
        ehdr.e_machine = 224;
        ehdr.e_version = EV_CURRENT;
        ehdr.e_phoff = 64;
        ehdr.e_shoff = shdr_off;
        ehdr.e_flags = elf_flags_;
        ehdr.e_ehsize = 64;
        ehdr.e_phentsize = 56;
        ehdr.e_phnum = 9;  // Match IFC: PHDR, LOAD R, LOAD RX, LOAD RW (RELRO), LOAD RW (data), DYNAMIC, GNU_RELRO, GNU_STACK, NOTE
        ehdr.e_shentsize = 64;
        ehdr.e_shnum = num_shdrs;
        ehdr.e_shstrndx = num_shdrs - 1;
        memcpy(out.data(), &ehdr, sizeof(ehdr));

        // Find section boundaries for proper segment creation (matching IFC)
        uint64_t note_start = 0, note_end = 0;
        uint64_t rodata_end = 0;
        uint64_t text_start = 0, text_end = 0;
        uint64_t data_rel_ro_start = 0, data_rel_ro_end = 0;
        uint64_t dyn_sec_addr = 0, dyn_sec_size = 0;
        uint64_t data_start = 0, bss_end = 0;
        uint64_t relro_pad_addr = 0, relro_pad_size = 0;
        for (const auto& s : sections_) {
            if (s.name == ".note") { note_start = s.addr; note_end = s.addr + s.size(); }
            if (s.name == ".rodata") rodata_end = s.addr + s.size();
            if (s.name == ".text") { text_start = s.addr; text_end = s.addr + s.size(); }
            if (s.name == ".data.rel.ro") { data_rel_ro_start = s.addr; data_rel_ro_end = s.addr + s.size(); }
            if (s.name == ".dynamic") { dyn_sec_addr = s.addr; dyn_sec_size = s.size(); }
            if (s.name == ".data") data_start = s.addr;
            if (s.name == ".bss") bss_end = s.addr + s.size();
            if (s.name == ".relro_padding") { relro_pad_addr = s.addr; relro_pad_size = s.size(); }
        }
        // RELRO segment: .data.rel.ro + .dynamic + .relro_padding
        // If no .data.rel.ro section, start RELRO at .dynamic
        uint64_t relro_start = data_rel_ro_start ? data_rel_ro_start : dyn_sec_addr;
        uint64_t relro_filesz = (dyn_sec_addr + dyn_sec_size) - relro_start;
        uint64_t relro_memsz = relro_end_ - relro_start;

        // Program headers (9 segments to match IFC)
        size_t phdr_off = 64;

        // 1. PHDR - program header table itself
        Elf64_Phdr phdr_phdr = {};
        phdr_phdr.p_type = PT_PHDR;
        phdr_phdr.p_flags = PF_R;
        phdr_phdr.p_offset = phdr_phdr.p_vaddr = phdr_phdr.p_paddr = 64;
        phdr_phdr.p_filesz = phdr_phdr.p_memsz = 9 * 56;  // 9 program headers
        phdr_phdr.p_align = 8;
        memcpy(out.data() + phdr_off, &phdr_phdr, sizeof(phdr_phdr));
        phdr_off += 56;

        // 2. LOAD R - read-only sections (from 0 to end of .rodata)
        Elf64_Phdr phdr_load_r = {};
        phdr_load_r.p_type = PT_LOAD;
        phdr_load_r.p_flags = PF_R;
        phdr_load_r.p_offset = phdr_load_r.p_vaddr = phdr_load_r.p_paddr = 0;
        phdr_load_r.p_filesz = phdr_load_r.p_memsz = rodata_end;
        phdr_load_r.p_align = 0x1000;
        memcpy(out.data() + phdr_off, &phdr_load_r, sizeof(phdr_load_r));
        phdr_off += 56;

        // 3. LOAD R E - executable section (.text)
        Elf64_Phdr phdr_load_rx = {};
        phdr_load_rx.p_type = PT_LOAD;
        phdr_load_rx.p_flags = PF_R | PF_X;
        phdr_load_rx.p_offset = phdr_load_rx.p_vaddr = phdr_load_rx.p_paddr = text_start;
        phdr_load_rx.p_filesz = phdr_load_rx.p_memsz = text_end - text_start;
        phdr_load_rx.p_align = 0x1000;
        memcpy(out.data() + phdr_off, &phdr_load_rx, sizeof(phdr_load_rx));
        phdr_off += 56;

        // 4. LOAD RW (RELRO) - .data.rel.ro + .dynamic + .relro_padding
        Elf64_Phdr phdr_load_relro = {};
        phdr_load_relro.p_type = PT_LOAD;
        phdr_load_relro.p_flags = PF_R | PF_W;
        phdr_load_relro.p_offset = phdr_load_relro.p_vaddr = phdr_load_relro.p_paddr = relro_start;
        phdr_load_relro.p_filesz = relro_filesz;
        phdr_load_relro.p_memsz = relro_memsz;
        phdr_load_relro.p_align = 0x1000;
        memcpy(out.data() + phdr_off, &phdr_load_relro, sizeof(phdr_load_relro));
        phdr_off += 56;

        // 5. LOAD RW - .data + .bss
        Elf64_Phdr phdr_load_data = {};
        phdr_load_data.p_type = PT_LOAD;
        phdr_load_data.p_flags = PF_R | PF_W;
        phdr_load_data.p_offset = phdr_load_data.p_vaddr = phdr_load_data.p_paddr = data_start;
        // Find .data section to get file size
        uint64_t data_filesz = 0;
        for (const auto& s : sections_) {
            if (s.name == ".data") data_filesz = s.fileSize();
        }
        phdr_load_data.p_filesz = data_filesz;
        phdr_load_data.p_memsz = bss_end - data_start;
        phdr_load_data.p_align = 0x1000;
        memcpy(out.data() + phdr_off, &phdr_load_data, sizeof(phdr_load_data));
        phdr_off += 56;

        // 6. DYNAMIC
        Elf64_Phdr phdr_dyn = {};
        phdr_dyn.p_type = PT_DYNAMIC;
        phdr_dyn.p_flags = PF_R | PF_W;
        phdr_dyn.p_offset = phdr_dyn.p_vaddr = phdr_dyn.p_paddr = dyn_sec_addr;
        phdr_dyn.p_filesz = phdr_dyn.p_memsz = dyn_sec_size;
        phdr_dyn.p_align = 8;
        memcpy(out.data() + phdr_off, &phdr_dyn, sizeof(phdr_dyn));
        phdr_off += 56;

        // 7. GNU_RELRO - marks .data.rel.ro + .dynamic as read-only after relocs
        Elf64_Phdr phdr_relro = {};
        phdr_relro.p_type = PT_GNU_RELRO;  // 0x6474e552
        phdr_relro.p_flags = PF_R;
        phdr_relro.p_offset = phdr_relro.p_vaddr = phdr_relro.p_paddr = relro_start;
        phdr_relro.p_filesz = relro_filesz;
        phdr_relro.p_memsz = relro_memsz;
        phdr_relro.p_align = 1;
        memcpy(out.data() + phdr_off, &phdr_relro, sizeof(phdr_relro));
        phdr_off += 56;

        // 8. GNU_STACK (marks stack as RW, no execute)
        Elf64_Phdr phdr_stack = {};
        phdr_stack.p_type = PT_GNU_STACK;  // 0x6474e551
        phdr_stack.p_flags = PF_R | PF_W;
        phdr_stack.p_offset = phdr_stack.p_vaddr = phdr_stack.p_paddr = 0;
        phdr_stack.p_filesz = phdr_stack.p_memsz = 0;
        phdr_stack.p_align = 0;
        memcpy(out.data() + phdr_off, &phdr_stack, sizeof(phdr_stack));
        phdr_off += 56;

        // 9. NOTE
        Elf64_Phdr phdr_note = {};
        phdr_note.p_type = PT_NOTE;
        phdr_note.p_flags = PF_R;
        phdr_note.p_offset = phdr_note.p_vaddr = phdr_note.p_paddr = note_start;
        phdr_note.p_filesz = phdr_note.p_memsz = note_end - note_start;
        phdr_note.p_align = 4;
        memcpy(out.data() + phdr_off, &phdr_note, sizeof(phdr_note));

        // Section data
        for (size_t i = 0; i < sections_.size(); i++) {
            if (sections_[i].type != SHT_NOBITS) {
                memcpy(out.data() + section_offsets[i], sections_[i].data.data(), sections_[i].data.size());
            }
        }
        memcpy(out.data() + shstrtab_off, shstrtab.data(), shstrtab.size());

        // Section headers
        auto writeShdr = [&](size_t idx, uint32_t name, uint32_t type, uint64_t flags,
                            uint64_t addr, uint64_t off, uint64_t size,
                            uint32_t link, uint32_t info, uint64_t align, uint64_t ent) {
            Elf64_Shdr shdr = {};
            shdr.sh_name = name;
            shdr.sh_type = type;
            shdr.sh_flags = flags;
            shdr.sh_addr = addr;
            shdr.sh_offset = off;
            shdr.sh_size = size;
            shdr.sh_link = link;
            shdr.sh_info = info;
            shdr.sh_addralign = align;
            shdr.sh_entsize = ent;
            memcpy(out.data() + shdr_off + idx * 64, &shdr, sizeof(shdr));
        };

        writeShdr(0, 0, SHT_NULL, 0, 0, 0, 0, 0, 0, 0, 0);

        // Find indices for linking
        int dynstr_idx = -1, dynsym_idx = -1, strtab_idx = -1;
        for (size_t i = 0; i < sections_.size(); i++) {
            if (sections_[i].name == ".dynstr") dynstr_idx = i + 1;
            if (sections_[i].name == ".dynsym") dynsym_idx = i + 1;
            if (sections_[i].name == ".strtab") strtab_idx = i + 1;
        }

        // Calculate sh_info for .dynsym (index of first global symbol)
        uint32_t dynsym_sh_info = 1;  // Default: first symbol is global (index 1, after NULL symbol)
        for (auto& s : sections_) {
            if (s.name == ".dynsym") {
                // Count local symbols (STB_LOCAL) - sh_info points to first global symbol
                size_t local_count = 0;
                for (size_t i = 0; i * 24 < s.data.size(); i++) {
                    uint8_t* sym = s.data.data() + i * 24;
                    uint8_t binding = ELF64_ST_BIND(sym[4]);
                    if (binding == STB_LOCAL) {
                        local_count++;
                    } else if (binding == STB_GLOBAL || binding == STB_WEAK) {
                        break;  // Found first global
                    }
                }
                dynsym_sh_info = local_count + 1;  // +1 for NULL symbol at index 0
                break;
            }
        }

        for (size_t i = 0; i < sections_.size(); i++) {
            const auto& s = sections_[i];
            uint32_t link = 0, info = 0;

            if (s.name == ".dynsym" && dynstr_idx >= 0) { link = dynstr_idx; info = dynsym_sh_info; }
            else if ((s.name == ".gnu.hash" || s.name == ".hash") && dynsym_idx >= 0) { link = dynsym_idx; }
            else if (s.name == ".rela.dyn" && dynsym_idx >= 0) { link = dynsym_idx; }
            else if (s.name == ".dynamic" && dynstr_idx >= 0) { link = dynstr_idx; }
            else if (s.name == ".symtab" && strtab_idx >= 0) { link = strtab_idx; info = 26; }

            writeShdr(i + 1, name_offs[i + 1], s.type, s.flags,
                      s.addr, section_offsets[i], s.size(),
                      link, info, s.alignment, s.entsize);
        }

        writeShdr(num_shdrs - 1, shstr_name_off, SHT_STRTAB, 0, 0, shstrtab_off, shstrtab.size(), 0, 0, 1, 0);

        // Write file
        FILE* f = fopen(path.c_str(), "wb");
        if (!f) { fprintf(stderr, "Cannot write %s\n", path.c_str()); return false; }
        size_t written = fwrite(out.data(), 1, out.size(), f);
        fclose(f);
        if (written != out.size()) {
            fprintf(stderr, "Error: Failed to write all data to %s (wrote %zu of %zu bytes)\n", 
                    path.c_str(), written, out.size());
            return false;
        }

        printf("Wrote %s: %zu bytes\n", path.c_str(), out.size());
        return true;
    }
};

// ============================================================================
// Main
// ============================================================================

void printUsage(const char* prog) {
    fprintf(stderr, "Usage: %s -o output.elf --dispatcher dispatcher.elf --host-table table.cpp [--target arch] [--input-dir dir] [--omit-dwarf]\n", prog);
    fprintf(stderr, "  --input-dir: directory containing .o device binaries (not host fat binaries)\n");
    fprintf(stderr, "  --omit-dwarf: do not add DWARF sections to the merged ELF (use if loader rejects code object with debug)\n");
}

int main(int argc, char** argv) {
    printf("device_linker: " __DATE__ " " __TIME__ " (debug merge + .debug_info line_strp patch)\n");
    std::string output, dispatcher, host_table, input_dir;
    std::vector<std::string> inputs;

    bool omit_dwarf = false;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-o" && i + 1 < argc) output = argv[++i];
        else if (arg == "--dispatcher" && i + 1 < argc) dispatcher = argv[++i];
        else if (arg == "--host-table" && i + 1 < argc) host_table = argv[++i];
        else if (arg == "--input-dir" && i + 1 < argc) input_dir = argv[++i];
        else if (arg == "--target" && i + 1 < argc) g_target_arch = argv[++i];
        else if (arg == "--omit-dwarf") omit_dwarf = true;
        else if (arg[0] != '-') inputs.push_back(arg);
    }

    // Detect local GPU if no target specified
    if (g_target_arch.empty()) {
        g_target_arch = detectLocalGpu();
        if (g_target_arch.empty()) {
            fprintf(stderr, "Error: No --target specified and could not detect local GPU\n");
            return 1;
        }
        printf("Device Linker: target=%s (auto-detected)\n", g_target_arch.c_str());
    } else {
        printf("Device Linker: target=%s\n", g_target_arch.c_str());
    }

    if (output.empty() || dispatcher.empty()) { printUsage(argv[0]); return 1; }

    // Collect input files: search for all .o files in the input directory
    if (!input_dir.empty()) {
        std::error_code ec;
        for (const auto& e : fs::directory_iterator(input_dir, ec)) {
            if (!ec && e.is_regular_file() && e.path().extension() == ".o") {
                inputs.push_back(e.path().string());
            }
        }
        if (ec) {
            fprintf(stderr, "Error reading input directory %s: %s\n", input_dir.c_str(), ec.message().c_str());
            return 1;
        }
        std::sort(inputs.begin(), inputs.end());
        if (inputs.empty()) {
            fprintf(stderr, "No .o files found in --input-dir (%s). Build specialized kernels as device-only.\n", input_dir.c_str());
            return 1;
        }
        
        // For testing: limit number of inputs if MAX_KERNELS_FOR_TEST is set
        const char* max_kernels_env = getenv("MAX_KERNELS_FOR_TEST");
        if (max_kernels_env) {
            size_t max_kernels = (size_t)atoi(max_kernels_env);
            if (max_kernels < inputs.size()) {
                inputs.resize(max_kernels);
                printf("TEST MODE: Limited input collection to %zu kernels (out of %zu total)\n", max_kernels, inputs.size() + (inputs.size() == max_kernels ? 0 : inputs.size() - max_kernels));
            }
        }
    }

    if (inputs.empty()) { fprintf(stderr, "No input files\n"); return 1; }
    printf("Processing %zu device binary input(s)\n", inputs.size());

    g_llvm_dwarf_error = false;
    auto funcid_map = parseHostTable(host_table);
    printf("Loaded %zu funcId mappings\n", funcid_map.size());

    // Process kernels in parallel
    std::vector<KernelInfo> kernels(inputs.size());
    std::mutex mtx;
    int done = 0;

    auto worker = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; i++) {
            // Inputs are device binaries (.o = amdgpu ELF); read directly, no extraction from host fat binary
            std::vector<uint8_t> data;
            {
                MappedFile f(inputs[i]);
                if (!f.valid()) { kernels[i] = parseKernel(data); continue; }
                const uint8_t* p = static_cast<const uint8_t*>(f.data());
                data.assign(p, p + f.size());
            }
            kernels[i] = parseKernel(data, inputs[i]);

            std::lock_guard<std::mutex> lock(mtx);
            if (++done % 100 == 0) printf("  %d/%zu...\n", done, inputs.size());
        }
    };

    unsigned int nt = std::thread::hardware_concurrency();
    if (nt == 0) nt = 4;
    std::vector<std::thread> threads;
    size_t chunk = (inputs.size() + nt - 1) / nt;

    for (unsigned int t = 0; t < nt; t++) {
        size_t s = t * chunk, e = std::min(s + chunk, inputs.size());
        if (s < e) threads.emplace_back(worker, s, e);
    }
    for (auto& th : threads) th.join();

    for (size_t i = 0; i < kernels.size(); i++) {
        if (kernels[i].dwarf_version != 0 && kernels[i].dwarf_version != 4 && kernels[i].dwarf_version != 5) {
            fprintf(stderr, "Error: %s has DWARF version %u; device linker requires DWARF4 or DWARF5\n",
                    i < inputs.size() ? inputs[i].c_str() : "(kernel)", (unsigned)kernels[i].dwarf_version);
            return 1;
        }
    }

    if (g_llvm_dwarf_error) {
        fprintf(stderr, "Error: LLVM reported DWARF errors during kernel parsing (see above). Exiting.\n");
        return 1;
    }

    // Link
    DeviceLinker linker(dispatcher);
    if (!linker.load()) { fprintf(stderr, "Cannot load dispatcher\n"); return 1; }

    // Set GPU target for proper LDS calculation
    linker.setGpuTarget(g_target_arch);
    if (omit_dwarf) {
        linker.setOmitDwarf(true);
        printf("Device Linker: omitting DWARF sections (--omit-dwarf)\n");
    }

    // For testing: limit number of kernels if MAX_KERNELS_FOR_TEST is set
    size_t max_kernels_to_add = kernels.size();
    const char* max_kernels_env = getenv("MAX_KERNELS_FOR_TEST");
    if (max_kernels_env) {
        max_kernels_to_add = std::min(max_kernels_to_add, (size_t)atoi(max_kernels_env));
        printf("TEST MODE: Limiting to %zu kernels (out of %zu total)\n", max_kernels_to_add, kernels.size());
    }
    
    for (size_t i = 0; i < max_kernels_to_add; i++) {
        linker.addKernel(kernels[i]);
    }
    linker.setFuncIdMap(funcid_map);

    if (!linker.link(output)) return 1;

    // Generate header file with funcId -> name mapping for host-side tracing
    std::string header_path = output;
    auto dot = header_path.rfind('.');
    if (dot != std::string::npos) header_path = header_path.substr(0, dot);
    header_path += "_funcid_names.h";
    if (!linker.writeFuncIdHeader(header_path)) {
        fprintf(stderr, "Error: Failed to write funcId header file\n");
        return 1;
    }

    return 0;
}
