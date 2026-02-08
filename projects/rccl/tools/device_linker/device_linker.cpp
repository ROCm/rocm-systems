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
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Error.h"
#include "llvm/BinaryFormat/Dwarf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <thread>
#include <mutex>
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
};

// Helper to create a minimal ELF for LLVM parsing
static std::vector<uint8_t> createMinimalElfForDwarf(
    const std::vector<uint8_t>& debug_info,
    const std::vector<uint8_t>& debug_abbrev) {

    constexpr size_t ELF_HDR_SIZE = sizeof(Elf64_Ehdr);
    constexpr size_t SHDR_SIZE = sizeof(Elf64_Shdr);
    constexpr size_t NUM_SECTIONS = 4;

    const char shstrtab[] = "\0.debug_info\0.debug_abbrev\0.shstrtab\0";
    constexpr size_t SHSTRTAB_SIZE = sizeof(shstrtab);
    constexpr size_t SHSTRTAB_DEBUG_INFO = 1;
    constexpr size_t SHSTRTAB_DEBUG_ABBREV = 13;
    constexpr size_t SHSTRTAB_SHSTRTAB = 27;

    size_t shdr_offset = ELF_HDR_SIZE;
    size_t debug_info_offset = shdr_offset + NUM_SECTIONS * SHDR_SIZE;
    size_t debug_abbrev_offset = debug_info_offset + debug_info.size();
    size_t shstrtab_offset = debug_abbrev_offset + debug_abbrev.size();
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
    ehdr->e_shstrndx = 3;

    Elf64_Shdr* shdrs = reinterpret_cast<Elf64_Shdr*>(elf_data.data() + shdr_offset);

    shdrs[1].sh_name = SHSTRTAB_DEBUG_INFO;
    shdrs[1].sh_type = SHT_PROGBITS;
    shdrs[1].sh_offset = debug_info_offset;
    shdrs[1].sh_size = debug_info.size();
    shdrs[1].sh_addralign = 1;

    shdrs[2].sh_name = SHSTRTAB_DEBUG_ABBREV;
    shdrs[2].sh_type = SHT_PROGBITS;
    shdrs[2].sh_offset = debug_abbrev_offset;
    shdrs[2].sh_size = debug_abbrev.size();
    shdrs[2].sh_addralign = 1;

    shdrs[3].sh_name = SHSTRTAB_SHSTRTAB;
    shdrs[3].sh_type = SHT_STRTAB;
    shdrs[3].sh_offset = shstrtab_offset;
    shdrs[3].sh_size = SHSTRTAB_SIZE;
    shdrs[3].sh_addralign = 1;

    memcpy(elf_data.data() + debug_info_offset, debug_info.data(), debug_info.size());
    memcpy(elf_data.data() + debug_abbrev_offset, debug_abbrev.data(), debug_abbrev.size());
    memcpy(elf_data.data() + shstrtab_offset, shstrtab, SHSTRTAB_SIZE);

    return elf_data;
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

// Structure to hold parsed abbreviation info
struct AbbrevAttr {
    uint16_t attr;    // DW_AT_*
    uint16_t form;    // DW_FORM_*
};

struct AbbrevEntry {
    uint64_t code;
    uint64_t tag;
    bool has_children;
    std::vector<AbbrevAttr> attrs;
};

// Parse abbreviation table
static std::unordered_map<uint64_t, AbbrevEntry> parseAbbrevTable(const std::vector<uint8_t>& abbrev) {
    std::unordered_map<uint64_t, AbbrevEntry> table;
    size_t pos = 0;

    while (pos < abbrev.size()) {
        size_t bytes;
        uint64_t code = decodeULEB128(&abbrev[pos], abbrev.size() - pos, bytes);
        pos += bytes;

        if (code == 0) continue;  // End of abbreviation section

        AbbrevEntry entry;
        entry.code = code;
        entry.tag = decodeULEB128(&abbrev[pos], abbrev.size() - pos, bytes);
        pos += bytes;

        entry.has_children = (abbrev[pos++] != 0);

        // Read attribute specs until (0,0)
        while (pos < abbrev.size()) {
            uint64_t attr = decodeULEB128(&abbrev[pos], abbrev.size() - pos, bytes);
            pos += bytes;
            uint64_t form = decodeULEB128(&abbrev[pos], abbrev.size() - pos, bytes);
            pos += bytes;

            if (attr == 0 && form == 0) break;

            entry.attrs.push_back({(uint16_t)attr, (uint16_t)form});
        }

        table[code] = entry;
    }

    return table;
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
            return (dwarf_version >= 4) ? 4 : addr_size;  // DWARF32
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
        default:
            return 0;
    }
}

// Find byte positions of various DWARF attributes that need patching
// This function properly parses the abbreviation table and DIE data
static DwarfAttrPositions findDwarfAttrPositions(
    const std::vector<uint8_t>& debug_info,
    const std::vector<uint8_t>& debug_abbrev) {

    DwarfAttrPositions result;

    if (debug_info.empty() || debug_abbrev.empty()) {
        return result;
    }

    // Parse abbreviation table
    auto abbrev_table = parseAbbrevTable(debug_abbrev);

    // Process each CU in debug_info
    size_t cu_pos = 0;
    while (cu_pos < debug_info.size()) {
        // Parse CU header
        uint32_t unit_length;
        memcpy(&unit_length, &debug_info[cu_pos], 4);
        if (unit_length == 0xffffffff) {
            // DWARF64 - skip for now
            break;
        }

        size_t cu_end = cu_pos + 4 + unit_length;
        if (cu_end > debug_info.size()) break;

        uint16_t version;
        memcpy(&version, &debug_info[cu_pos + 4], 2);

        uint8_t addr_size;
        size_t die_start;

        if (version == 5) {
            // DWARF5: unit_length(4) + version(2) + unit_type(1) + addr_size(1) + abbrev_offset(4) = 12
            addr_size = debug_info[cu_pos + 7];
            die_start = cu_pos + 12;
        } else {
            // DWARF4: unit_length(4) + version(2) + abbrev_offset(4) + addr_size(1) = 11
            addr_size = debug_info[cu_pos + 10];
            die_start = cu_pos + 11;
        }

        // Parse the root DIE (compile_unit)
        size_t pos = die_start;
        size_t bytes;
        uint64_t abbrev_code = decodeULEB128(&debug_info[pos], cu_end - pos, bytes);
        pos += bytes;

        if (abbrev_code == 0) {
            cu_pos = cu_end;
            continue;
        }

        auto it = abbrev_table.find(abbrev_code);
        if (it == abbrev_table.end()) {
            cu_pos = cu_end;
            continue;
        }

        const AbbrevEntry& entry = it->second;

        // Walk through attributes, recording positions of ones we care about
        for (const auto& attr : entry.attrs) {
            size_t attr_pos = pos;  // Position of this attribute's value

            // Get the size of this form
            size_t form_size = getFormFixedSize(attr.form, addr_size, version);
            if (form_size == 0) {
                form_size = skipVariableForm(attr.form, &debug_info[pos], cu_end - pos);
            }

            // Check if this is an attribute we care about with a 4-byte form
            if (attr.form == llvm::dwarf::DW_FORM_sec_offset ||
                attr.form == llvm::dwarf::DW_FORM_data4) {

                uint32_t val;
                memcpy(&val, &debug_info[attr_pos], 4);

                switch (attr.attr) {
                    case llvm::dwarf::DW_AT_ranges:
                        result.ranges.push_back({attr_pos, val});
                        break;
                    case llvm::dwarf::DW_AT_str_offsets_base:
                        result.str_offsets_base.push_back({attr_pos, val});
                        break;
                    case llvm::dwarf::DW_AT_addr_base:
                        result.addr_base.push_back({attr_pos, val});
                        break;
                    case llvm::dwarf::DW_AT_rnglists_base:
                        result.rnglists_base.push_back({attr_pos, val});
                        break;
                    case llvm::dwarf::DW_AT_stmt_list:
                        result.stmt_list.push_back({attr_pos, val});
                        break;
                }
            }

            pos += form_size;
            if (pos >= cu_end) break;
        }

        cu_pos = cu_end;
    }

    return result;
}

// Legacy wrapper for code that only needs ranges
static std::vector<std::pair<size_t, uint32_t>> findRangesOffsets(
    const std::vector<uint8_t>& debug_info,
    const std::vector<uint8_t>& debug_abbrev) {
    return findDwarfAttrPositions(debug_info, debug_abbrev).ranges;
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
    std::vector<uint8_t> debug_ranges;  // DWARF4 ranges (different from DWARF5 rnglists)
    DwarfAttrPositions dwarf_attr_positions;  // Positions of various DWARF attributes that need patching
    uint64_t orig_text_addr = 0;    // Original .text address for debug_line patching
    int vgpr = 0, sgpr = 0, lds = 0, stack = 0;
};

KernelInfo parseKernel(const std::vector<uint8_t>& elf_data) {
    KernelInfo info;
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

    for (size_t i = 0; i < nsyms; i++) {
        if (ELF64_ST_TYPE(syms[i].st_info) == STT_FUNC) {
            const char* n = strings + syms[i].st_name;
            // Look for either ncclDevFunc_ or ncclDevKernel_..._Specialized
            if (strstr(n, "ncclDevFunc_") || strstr(n, "ncclDevKernel_")) {
                info.name = n;
                // Extract from beginning of .text through end of ncclDevFunc_
                // This includes helper functions (like runTreeSplit) that ncclDevFunc_ calls
                // via PC-relative addressing. Without these, the PC-relative offsets would
                // point to garbage after merging.
                uint64_t func_start = syms[i].st_value;
                uint64_t func_end = func_start + syms[i].st_size;
                uint64_t extract_size = func_end - text->addr;

                const uint8_t* p = file.at<uint8_t>(text->offset);
                info.code.assign(p, p + extract_size);
                info.func_offset = func_start - text->addr;  // Offset of ncclDevFunc_ within extracted code
                break;
            }
        }
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
        auto* debug_ranges = elf.find(".debug_ranges");
        if (debug_ranges && debug_ranges->size > 0) {
            info.debug_ranges = elf.getBytes(*debug_ranges);
        }

        // Use LLVM to find DWARF attribute positions for later patching
        if (!info.debug_info.empty() && !info.debug_abbrev.empty()) {
            info.dwarf_attr_positions = findDwarfAttrPositions(info.debug_info, info.debug_abbrev);
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

    bool link(const std::string& output_path) {
        printf("=== Pass 1: Collect and Size ===\n");
        if (!collectSections()) return false;

        printf("\n=== Pass 2: Layout ===\n");
        computeLayout();

        // Merge and patch debug sections after layout (needs text_addr_)
        printf("\n=== Pass 2b: Merge Debug Info ===\n");
        mergeDebugInfo();

        printf("\n=== Pass 3: Patch and Write ===\n");
        patchSections();

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
    std::unique_ptr<MappedFile> disp_file_;
    std::unique_ptr<ElfParser> disp_;
    std::vector<KernelInfo> kernels_;
    std::vector<std::pair<const KernelInfo*, uint64_t>> kernel_text_offsets_;  // kernel -> text offset
    std::vector<std::pair<size_t, uint64_t>> specialized_kd_offsets_;  // KD offset in .rodata -> text offset
    std::vector<uint64_t> onerank_text_offsets_;  // oneRankReduce kernel offsets in dispatcher .text
    std::unordered_map<std::string, FuncIdMapping> funcid_map_;

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
        size_t ranges_base;        // Base offset in merged .debug_ranges (DWARF4)
        size_t line_base;          // Base offset in merged .debug_line
        DwarfAttrPositions dwarf_attr_positions;  // Positions of DWARF attributes in this chunk
    };
    std::vector<DebugInfoChunk> debug_info_chunks_;
    std::vector<uint8_t> merged_debug_abbrev_;
    std::vector<uint8_t> merged_debug_str_;
    std::vector<uint8_t> merged_debug_str_offsets_;
    std::vector<uint8_t> merged_debug_addr_;
    std::vector<uint8_t> merged_debug_rnglists_;
    std::vector<uint8_t> merged_debug_ranges_;  // DWARF4 ranges (different from DWARF5 rnglists)
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

    // Calculate required LDS size based on GPU architecture
    // Must match the host-side calculation in enqueue.cc: rcclShmemDynamicSize()
    int calculateRequiredLDS() const {
        // Constants from RCCL source:
        // NCCL_LL128_SHMEM_ELEMS_PER_THREAD = 8
        // ncclCollUnroll(cudaArch) = cudaArch >= 800 ? 8 : 4  (for non-12xx)
        // ncclNvlsUnrollBytes = 64

        int warpSize = (gpu_arch_ >= 900) ? 64 : 64;  // AMD GFX9+ uses 64-wide warps
        int maxNthreads = (gpu_arch_ == 950) ? 512 : 256;
        int ncclCollUnroll = (gpu_arch_ >= 800) ? 8 : 4;
        int ncclNvlsUnrollBytes = 64;

        // Per-warp scratch size calculation (matches rcclShmemScratchWarpSize)
        int ll = 0;
        int ll128 = (8 * warpSize) * sizeof(uint64_t);  // NCCL_LL128_SHMEM_ELEMS_PER_THREAD * WARP_SIZE * 8
        int simple = (ncclCollUnroll * warpSize + 1) * 16;
        int nvls = (gpu_arch_ >= 900) ? (warpSize * ncclNvlsUnrollBytes + 16) : 16;

        int perWarpScratch = std::max({ll, ll128, simple, nvls});
        perWarpScratch = (perWarpScratch + 15) & ~15;  // Pad to 16 bytes

        // Total dynamic shared memory
        int numWarps = maxNthreads / warpSize;
        int totalDynamicShared = perWarpScratch * numWarps;

        printf("LDS calculation for gfx%d: warpSize=%d, maxNthreads=%d, numWarps=%d\n",
               gpu_arch_, warpSize, maxNthreads, numWarps);
        printf("  Per-warp scratch: LL=%d, LL128=%d, SIMPLE=%d, NVLS=%d -> max=%d\n",
               ll, ll128, simple, nvls, perWarpScratch);
        printf("  Total dynamic shared memory required: %d bytes\n", totalDynamicShared);

        return totalDynamicShared;
    }

    // ========== Pass 1: Collect ==========
    bool collectSections() {
        elf_flags_ = disp_->ehdr()->e_flags;

        // Compute max resources from kernels
        for (const auto& k : kernels_) {
            max_vgpr_ = std::max(max_vgpr_, k.vgpr);
            max_sgpr_ = std::max(max_sgpr_, k.sgpr);
            max_lds_ = std::max(max_lds_, k.lds);
            max_stack_ = std::max(max_stack_, k.stack);
        }
        printf("Max resources from kernels: VGPR=%d, SGPR=%d, LDS=%d, Stack=%d\n",
               max_vgpr_, max_sgpr_, max_lds_, max_stack_);

        // Calculate required LDS based on RCCL formula (must match host-side calculation)
        int requiredLDS = calculateRequiredLDS();
        if (requiredLDS > max_lds_) {
            printf("NOTE: Adjusting LDS from %d to %d (required by RCCL formula)\n",
                   max_lds_, requiredLDS);
            max_lds_ = requiredLDS;
        }
        printf("Final LDS size: %d bytes\n", max_lds_);

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
        auto* disp_note = disp_->find(".note");
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
                        // Store offset within .text
                        onerank_text_offsets_.push_back(syms[i].st_value - disp_text_for_onerank->addr);
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
        // (if any have debug info from -gline-tables-only compilation)
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
                printf("Built .debug_line: %zu bytes (%zu chunks)\n",
                       debug_line.data.size(), debug_line_chunks_.size());
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
                printf("Built .debug_line_str: %zu bytes\n", merged_debug_line_str_.size());
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
                auto* disp_debug_ranges = disp_->find(".debug_ranges");  // DWARF4
                auto* disp_debug_info = disp_->find(".debug_info");
                auto* disp_text = disp_->find(".text");

                if (disp_debug_info && disp_debug_info->size > 0) {
                    auto disp_di = disp_->getBytes(*disp_debug_info);

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
                    chunk.ranges_base = merged_debug_ranges_.size();  // DWARF4 ranges
                    chunk.line_base = 0;  // First debug_line chunk

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
                        merged_debug_str_offsets_.insert(merged_debug_str_offsets_.end(), data.begin(), data.end());
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

                    // Find DWARF attribute positions in dispatcher's debug_info
                    if (disp_debug_abbrev && disp_debug_abbrev->size > 0) {
                        auto abbrev_data = disp_->getBytes(*disp_debug_abbrev);
                        chunk.dwarf_attr_positions = findDwarfAttrPositions(disp_di, abbrev_data);
                    }
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
                        chunk.ranges_base = merged_debug_ranges_.size();  // DWARF4 ranges
                        chunk.line_base = line_offset;

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
                            merged_debug_str_offsets_.insert(merged_debug_str_offsets_.end(),
                                kern->debug_str_offsets.begin(), kern->debug_str_offsets.end());
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

        // Fill tables in .rodata if they're there (const tables case)
        if (rodata_sec && rodata_table_2_off_ > 0) {
            printf("Function tables in .rodata (const) - pre-filling at offsets 0x%lx, 0x%lx, 0x%lx\n",
                   rodata_table_1_off_, rodata_table_2_off_, rodata_table_4_off_);

            uint8_t* rodata = rodata_sec->data.data();
            int pop1 = 0, pop2 = 0, pop4 = 0;
            for (int i = 0; i < FUNC_COUNT; i++) {
                if (table_1_[i] && rodata_table_1_off_) {
                    uint64_t addr = text_addr_ + table_1_[i];
                    memcpy(rodata + rodata_table_1_off_ + i * 8, &addr, 8);
                    pop1++;
                }
                if (table_2_[i] && rodata_table_2_off_) {
                    uint64_t addr = text_addr_ + table_2_[i];
                    memcpy(rodata + rodata_table_2_off_ + i * 8, &addr, 8);
                    pop2++;
                }
                if (table_4_[i] && rodata_table_4_off_) {
                    uint64_t addr = text_addr_ + table_4_[i];
                    memcpy(rodata + rodata_table_4_off_ + i * 8, &addr, 8);
                    pop4++;
                }
            }
            printf("  Function entries pre-filled in .rodata: table_1=%d, table_2=%d, table_4=%d\n", pop1, pop2, pop4);
        }
        // For .data.rel.ro tables, don't pre-fill - R_AMDGPU_RELATIVE64 relocations
        // will fill them at load time (like IFC). This is important for multi-GPU.
        else if (data_rel_ro_sec) {
            printf("Function tables at 0x%lx (will be filled by relocations at load time)\n", data_rel_ro_addr_);
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

                // RSRC1 at offset 0x30: VGPR/SGPR
                uint32_t rsrc1;
                memcpy(&rsrc1, kd + 0x30, 4);
                int vgpr_g = (max_vgpr_ + 3) / 4 - 1;
                int sgpr_g = (max_sgpr_ + 7) / 8 - 1;
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

            // Patch dispatcher KDs (first 3)
            for (size_t kd_off : {0UL, 64UL, 128UL}) {
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
            printf("Patched %zu dispatcher KDs + %zu specialized KDs\n",
                   3UL, specialized_kd_offsets_.size());
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

        // Patch ABS symbols for resource maxima
        // These symbols are in .symtab and .strtab with format:
        // kernelname.suffix
        // strtab_sec is already set above
        //
        // NOTE: We DO NOT patch .private_seg_size - it must match KD and metadata (both 0)
        // The Generic kernels dispatch via function pointers; the called functions
        // handle their own stack requirements.

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

                    // DO NOT patch .private_seg_size - must match KD (0)
                    // const char* suffix1 = ".private_seg_size";
                    // if (len > strlen(suffix1) && strcmp(name + len - strlen(suffix1), suffix1) == 0) {
                    //     // Leave at original value (0) to match KD and metadata
                    // }

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

        // Build .rela.dyn section with R_AMDGPU_RELATIVE64 relocations
        buildRelocations();
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
        do {
            uint8_t byte = value & 0x7f;
            value >>= 7;
            if (value != 0) byte |= 0x80;
            buf.push_back(byte);
        } while (value != 0);
    }

    // Merge and patch debug sections from all kernels
    // Patches addresses based on where code ended up, and adjusts cross-section offsets
    void mergeDebugInfo() {
        if (debug_info_chunks_.empty()) {
            printf("No debug_info chunks, skipping debug section merge\n");
            return;
        }

        printf("Merging debug sections for %zu chunks...\n", debug_info_chunks_.size());

        // Step 1: Patch addresses in .debug_addr
        // DWARF5 .debug_addr format: header (8 bytes) + array of addresses
        // Header: unit_length(4) + version(2) + address_size(1) + segment_selector_size(1)
        if (!merged_debug_addr_.empty()) {
            printf("Patching .debug_addr addresses...\n");
            size_t pos = 0;
            size_t chunk_idx = 0;

            for (const auto& chunk : debug_info_chunks_) {
                if (pos >= merged_debug_addr_.size()) break;

                // Find the end of this chunk's debug_addr contribution
                size_t chunk_end = (chunk_idx + 1 < debug_info_chunks_.size())
                    ? debug_info_chunks_[chunk_idx + 1].addr_base
                    : merged_debug_addr_.size();

                if (chunk_end <= pos) {
                    chunk_idx++;
                    continue;
                }

                // Calculate address delta for this chunk
                int64_t addr_delta = (int64_t)(text_addr_ + chunk.new_text_offset) - (int64_t)chunk.orig_text_addr;

                // Parse header to find where addresses start
                if (pos + 8 > chunk_end) {
                    chunk_idx++;
                    pos = chunk_end;
                    continue;
                }

                uint32_t unit_length;
                memcpy(&unit_length, &merged_debug_addr_[pos], 4);
                uint16_t version;
                memcpy(&version, &merged_debug_addr_[pos + 4], 2);
                uint8_t addr_size = merged_debug_addr_[pos + 6];

                if (version != 5 || addr_size != 8) {
                    printf("  Chunk %zu: unexpected debug_addr format (version=%d, addr_size=%d)\n",
                           chunk_idx, version, addr_size);
                    chunk_idx++;
                    pos = chunk_end;
                    continue;
                }

                // Addresses start after 8-byte header
                size_t addr_start = pos + 8;
                int patched = 0;

                for (size_t p = addr_start; p + 8 <= chunk_end; p += 8) {
                    uint64_t addr;
                    memcpy(&addr, &merged_debug_addr_[p], 8);

                    // Only patch addresses that look like they're in the original text range
                    // (non-zero and within reasonable bounds)
                    if (addr != 0 && addr >= chunk.orig_text_addr) {
                        uint64_t new_addr = addr + addr_delta;
                        memcpy(&merged_debug_addr_[p], &new_addr, 8);
                        patched++;
                    }
                }

                printf("  Chunk %zu: patched %d addresses (delta=0x%lx)\n",
                       chunk_idx, patched, (uint64_t)addr_delta);

                chunk_idx++;
                pos = chunk_end;
            }
        }

        // Step 1.5: Patch offsets in .debug_str_offsets
        // DWARF5 .debug_str_offsets format: header (8 bytes) + array of 4-byte offsets into .debug_str
        // Header: unit_length(4) + version(2) + padding(2)
        // When we merge .debug_str from multiple kernels, we need to adjust these offsets
        if (!merged_debug_str_offsets_.empty()) {
            printf("Patching .debug_str_offsets...\n");
            size_t pos = 0;
            size_t chunk_idx = 0;

            for (const auto& chunk : debug_info_chunks_) {
                if (pos >= merged_debug_str_offsets_.size()) break;

                // Find the end of this chunk's debug_str_offsets contribution
                size_t chunk_end = (chunk_idx + 1 < debug_info_chunks_.size())
                    ? debug_info_chunks_[chunk_idx + 1].str_offsets_base
                    : merged_debug_str_offsets_.size();

                if (chunk_end <= pos) {
                    chunk_idx++;
                    continue;
                }

                // Skip if no adjustment needed (first chunk starts at 0)
                if (chunk.str_base == 0) {
                    printf("  Chunk %zu: str_base=0, no patching needed\n", chunk_idx);
                    chunk_idx++;
                    pos = chunk_end;
                    continue;
                }

                // Parse header to verify format
                if (pos + 8 > chunk_end) {
                    chunk_idx++;
                    pos = chunk_end;
                    continue;
                }

                uint32_t unit_length;
                memcpy(&unit_length, &merged_debug_str_offsets_[pos], 4);
                uint16_t version;
                memcpy(&version, &merged_debug_str_offsets_[pos + 4], 2);

                if (version != 5) {
                    printf("  Chunk %zu: unexpected version %d (expected 5)\n", chunk_idx, version);
                    chunk_idx++;
                    pos = chunk_end;
                    continue;
                }

                // Offsets start after 8-byte header
                size_t offsets_start = pos + 8;
                int patched = 0;

                for (size_t p = offsets_start; p + 4 <= chunk_end; p += 4) {
                    uint32_t old_offset;
                    memcpy(&old_offset, &merged_debug_str_offsets_[p], 4);
                    uint32_t new_offset = old_offset + (uint32_t)chunk.str_base;
                    memcpy(&merged_debug_str_offsets_[p], &new_offset, 4);
                    patched++;
                }

                printf("  Chunk %zu: patched %d string offsets (str_base=0x%zx)\n",
                       chunk_idx, patched, chunk.str_base);

                chunk_idx++;
                pos = chunk_end;
            }
        }

        // Step 2: Patch .debug_info CU headers and attributes
        // DWARF5 CU header: unit_length(4) + version(2) + unit_type(1) + addr_size(1) + debug_abbrev_offset(4)
        // DWARF4 CU header: unit_length(4) + version(2) + debug_abbrev_offset(4) + addr_size(1)
        if (!merged_debug_info_.empty()) {
            printf("Patching .debug_info CU headers...\n");

            for (size_t i = 0; i < debug_info_chunks_.size(); i++) {
                const auto& chunk = debug_info_chunks_[i];
                size_t cu_start = chunk.merged_offset;

                if (cu_start + 12 > merged_debug_info_.size()) continue;

                // Read version to determine header format
                uint16_t version;
                memcpy(&version, &merged_debug_info_[cu_start + 4], 2);

                size_t abbrev_offset_pos;
                if (version == 5) {
                    // DWARF5: unit_length(4) + version(2) + unit_type(1) + addr_size(1) + debug_abbrev_offset(4)
                    abbrev_offset_pos = cu_start + 8;
                } else {
                    // DWARF4: unit_length(4) + version(2) + debug_abbrev_offset(4) + addr_size(1)
                    abbrev_offset_pos = cu_start + 6;
                }

                // Patch debug_abbrev_offset
                uint32_t new_abbrev_offset = (uint32_t)chunk.abbrev_base;
                memcpy(&merged_debug_info_[abbrev_offset_pos], &new_abbrev_offset, 4);

                printf("  CU %zu (DWARF%d): abbrev_offset -> 0x%x\n", i, version, new_abbrev_offset);

                // Step 2.5: Patch DWARF attributes using positions found by LLVM
                const auto& attrs = chunk.dwarf_attr_positions;

                // Helper to patch a list of attributes
                auto patchAttrs = [&](const std::vector<std::pair<size_t, uint32_t>>& positions,
                                     size_t base_offset, const char* name) {
                    int patched = 0;
                    for (const auto& [offset, orig_val] : positions) {
                        size_t pos = chunk.merged_offset + offset;
                        if (pos + 4 <= merged_debug_info_.size()) {
                            uint32_t new_val = orig_val + (uint32_t)base_offset;
                            memcpy(&merged_debug_info_[pos], &new_val, 4);
                            patched++;
                        }
                    }
                    if (patched > 0) {
                        printf("    Patched %d %s (base=0x%zx)\n", patched, name, base_offset);
                    }
                };

                // Patch DW_AT_ranges (points to .debug_ranges for DWARF4)
                if (chunk.ranges_base > 0) {
                    patchAttrs(attrs.ranges, chunk.ranges_base, "DW_AT_ranges");
                }

                // Patch DW_AT_str_offsets_base (points to .debug_str_offsets for DWARF5)
                if (chunk.str_offsets_base > 0) {
                    patchAttrs(attrs.str_offsets_base, chunk.str_offsets_base, "DW_AT_str_offsets_base");
                }

                // Patch DW_AT_addr_base (points to .debug_addr for DWARF5)
                if (chunk.addr_base > 0) {
                    patchAttrs(attrs.addr_base, chunk.addr_base, "DW_AT_addr_base");
                }

                // Patch DW_AT_rnglists_base (points to .debug_rnglists for DWARF5)
                if (chunk.rnglists_base > 0) {
                    patchAttrs(attrs.rnglists_base, chunk.rnglists_base, "DW_AT_rnglists_base");
                }

                // Patch DW_AT_stmt_list (points to .debug_line)
                if (chunk.line_base > 0) {
                    patchAttrs(attrs.stmt_list, chunk.line_base, "DW_AT_stmt_list");
                }
            }
        }

        // Step 3: Add merged debug sections to output
        if (!merged_debug_abbrev_.empty()) {
            SectionInfo sec;
            sec.name = ".debug_abbrev";
            sec.type = SHT_PROGBITS;
            sec.flags = 0;
            sec.alignment = 1;
            sec.data = std::move(merged_debug_abbrev_);
            printf("Adding .debug_abbrev: %zu bytes\n", sec.data.size());
            sections_.push_back(std::move(sec));
        }

        if (!merged_debug_str_.empty()) {
            SectionInfo sec;
            sec.name = ".debug_str";
            sec.type = SHT_PROGBITS;
            sec.flags = SHF_MERGE | SHF_STRINGS;
            sec.alignment = 1;
            sec.entsize = 1;
            sec.data = std::move(merged_debug_str_);
            printf("Adding .debug_str: %zu bytes\n", sec.data.size());
            sections_.push_back(std::move(sec));
        }

        if (!merged_debug_str_offsets_.empty()) {
            SectionInfo sec;
            sec.name = ".debug_str_offsets";
            sec.type = SHT_PROGBITS;
            sec.flags = 0;
            sec.alignment = 1;
            sec.data = std::move(merged_debug_str_offsets_);
            printf("Adding .debug_str_offsets: %zu bytes\n", sec.data.size());
            sections_.push_back(std::move(sec));
        }

        if (!merged_debug_addr_.empty()) {
            SectionInfo sec;
            sec.name = ".debug_addr";
            sec.type = SHT_PROGBITS;
            sec.flags = 0;
            sec.alignment = 1;
            sec.data = std::move(merged_debug_addr_);
            printf("Adding .debug_addr: %zu bytes\n", sec.data.size());
            sections_.push_back(std::move(sec));
        }

        if (!merged_debug_rnglists_.empty()) {
            SectionInfo sec;
            sec.name = ".debug_rnglists";
            sec.type = SHT_PROGBITS;
            sec.flags = 0;
            sec.alignment = 1;
            sec.data = std::move(merged_debug_rnglists_);
            printf("Adding .debug_rnglists: %zu bytes\n", sec.data.size());
            sections_.push_back(std::move(sec));
        }

        if (!merged_debug_ranges_.empty()) {
            SectionInfo sec;
            sec.name = ".debug_ranges";
            sec.type = SHT_PROGBITS;
            sec.flags = 0;
            sec.alignment = 1;
            sec.data = std::move(merged_debug_ranges_);
            printf("Adding .debug_ranges: %zu bytes\n", sec.data.size());
            sections_.push_back(std::move(sec));
        }

        if (!merged_debug_info_.empty()) {
            SectionInfo sec;
            sec.name = ".debug_info";
            sec.type = SHT_PROGBITS;
            sec.flags = 0;
            sec.alignment = 1;
            sec.data = std::move(merged_debug_info_);
            printf("Adding .debug_info: %zu bytes\n", sec.data.size());
            sections_.push_back(std::move(sec));
        }
    }

    // Patch DWARF5 string offsets in .debug_line prologue
    // DWARF5 stores directory/file names as offsets into .debug_line_str using DW_FORM_line_strp
    // When we concatenate multiple .debug_line_str sections, we need to adjust these offsets
    void patchDwarf5StringOffsets() {
        SectionInfo* debug_line_sec = nullptr;
        for (auto& s : sections_) {
            if (s.name == ".debug_line") { debug_line_sec = &s; break; }
        }
        if (!debug_line_sec || debug_line_sec->data.empty()) return;

        int total_patched = 0;

        for (const auto& chunk : debug_line_chunks_) {
            // Skip if no string offset adjustment needed
            if (chunk.str_offset_base == 0) continue;

            uint8_t* data = debug_line_sec->data.data() + chunk.merged_offset;
            size_t size = chunk.size;
            if (size < 12) continue;

            // Parse DWARF line table header
            // Format: unit_length(4) + version(2) + [address_size(1) + seg_sel_size(1) for DWARF5] + header_length(4)
            uint32_t unit_length;
            memcpy(&unit_length, data, 4);
            uint16_t version;
            memcpy(&version, data + 4, 2);

            // Only DWARF5 uses .debug_line_str
            if (version != 5) continue;

            // DWARF5 header: unit_length(4) + version(2) + addr_size(1) + seg_sel_size(1) + header_length(4)
            uint8_t addr_size = data[6];
            // uint8_t seg_sel_size = data[7];
            uint32_t header_length;
            memcpy(&header_length, data + 8, 4);

            // Prologue starts after header_length field (at offset 12)
            const uint8_t* prologue = data + 12;
            const uint8_t* prologue_end = data + 12 + header_length;
            const uint8_t* p = prologue;

            if (prologue_end > data + size) continue;

            // Skip: min_inst_len(1), max_ops(1), default_is_stmt(1), line_base(1), line_range(1), opcode_base(1)
            if (p + 6 > prologue_end) continue;
            uint8_t opcode_base = p[5];
            p += 6;

            // Skip standard_opcode_lengths array (opcode_base - 1 entries)
            if (p + (opcode_base - 1) > prologue_end) continue;
            p += (opcode_base - 1);

            // DW_FORM constants
            const uint8_t DW_FORM_data1 = 0x0b;      // 1 byte
            const uint8_t DW_FORM_data2 = 0x05;      // 2 bytes
            const uint8_t DW_FORM_data4 = 0x06;      // 4 bytes
            const uint8_t DW_FORM_data8 = 0x07;      // 8 bytes
            const uint8_t DW_FORM_data16 = 0x1e;     // 16 bytes (MD5 checksum)
            const uint8_t DW_FORM_udata = 0x0f;      // ULEB128
            const uint8_t DW_FORM_line_strp = 0x1f;  // 4-byte offset into .debug_line_str (DWARF32)
            const uint8_t DW_FORM_string = 0x08;     // null-terminated inline string
            const uint8_t DW_FORM_strp = 0x0e;       // 4-byte offset into .debug_str

            // Lambda to patch offsets in an entry list
            auto patchEntries = [&](int entry_count, const std::vector<std::pair<uint8_t, uint8_t>>& formats) {
                int patched = 0;
                for (int i = 0; i < entry_count && p < prologue_end; i++) {
                    for (const auto& [content_type, form] : formats) {
                        if (p >= prologue_end) break;

                        if (form == DW_FORM_line_strp) {
                            // 4-byte offset for DWARF32
                            if (p + 4 <= prologue_end) {
                                uint32_t old_offset;
                                memcpy(&old_offset, p, 4);
                                uint32_t new_offset = old_offset + (uint32_t)chunk.str_offset_base;
                                memcpy(const_cast<uint8_t*>(p), &new_offset, 4);
                                patched++;
                            }
                            p += 4;
                        } else if (form == DW_FORM_data1) {
                            p += 1;
                        } else if (form == DW_FORM_data2) {
                            p += 2;
                        } else if (form == DW_FORM_data4 || form == DW_FORM_strp) {
                            p += 4;
                        } else if (form == DW_FORM_data8) {
                            p += 8;
                        } else if (form == DW_FORM_data16) {
                            p += 16;  // MD5 checksum
                        } else if (form == DW_FORM_udata) {
                            readULEB128(p, prologue_end);
                        } else if (form == DW_FORM_string) {
                            // Skip inline null-terminated string
                            while (p < prologue_end && *p != 0) p++;
                            if (p < prologue_end) p++;  // Skip null terminator
                        } else {
                            // Unknown form - log and skip 4 bytes as guess
                            // printf("Warning: unknown DWARF form 0x%02x\n", form);
                            p += 4;
                        }
                    }
                }
                return patched;
            };

            // Directory entry format
            if (p >= prologue_end) continue;
            uint8_t dir_format_count = *p++;
            std::vector<std::pair<uint8_t, uint8_t>> dir_formats;
            for (int i = 0; i < dir_format_count && p < prologue_end; i++) {
                uint8_t content_type = (uint8_t)readULEB128(p, prologue_end);
                uint8_t form = (uint8_t)readULEB128(p, prologue_end);
                dir_formats.push_back({content_type, form});
            }

            // Directory count and entries
            uint64_t dir_count = readULEB128(p, prologue_end);
            total_patched += patchEntries((int)dir_count, dir_formats);

            // File name entry format
            if (p >= prologue_end) continue;
            uint8_t file_format_count = *p++;
            std::vector<std::pair<uint8_t, uint8_t>> file_formats;
            for (int i = 0; i < file_format_count && p < prologue_end; i++) {
                uint8_t content_type = (uint8_t)readULEB128(p, prologue_end);
                uint8_t form = (uint8_t)readULEB128(p, prologue_end);
                file_formats.push_back({content_type, form});
            }

            // File count and entries
            uint64_t file_count = readULEB128(p, prologue_end);
            total_patched += patchEntries((int)file_count, file_formats);
        }

        if (total_patched > 0) {
            printf("  Patched %d DWARF5 string offsets in .debug_line\n", total_patched);
        }
    }

    void patchDebugLine() {
        if (debug_line_chunks_.empty()) return;

        SectionInfo* debug_line_sec = nullptr;
        for (auto& s : sections_) {
            if (s.name == ".debug_line") { debug_line_sec = &s; break; }
        }
        if (!debug_line_sec || debug_line_sec->data.empty()) return;

        printf("Patching .debug_line addresses (%zu chunks)...\n", debug_line_chunks_.size());
        int patched = 0;
        for (const auto& chunk : debug_line_chunks_) {
            // Scan this chunk for DW_LNE_set_address opcodes
            // Format: 0x00 (extended opcode), length (usually 0x09 for 8-byte addr),
            //         0x02 (DW_LNE_set_address), 8-byte address
            uint8_t* data = debug_line_sec->data.data() + chunk.merged_offset;
            size_t size = chunk.size;

            for (size_t i = 0; i + 11 <= size; i++) {
                // Look for: 00 09 02 <8-byte-addr>
                // The length byte (09) = 1 + 8 (opcode + address)
                if (data[i] == 0x00 && data[i+1] == 0x09 && data[i+2] == 0x02) {
                    uint64_t old_addr;
                    memcpy(&old_addr, &data[i+3], 8);

                    // Calculate new address:
                    // old_addr is relative to original .text in the kernel's ELF
                    // new_addr = text_addr_ + chunk.new_text_offset + (old_addr - chunk.orig_text_addr)
                    uint64_t new_addr = text_addr_ + chunk.new_text_offset +
                                       (old_addr - chunk.orig_text_addr);

                    memcpy(&data[i+3], &new_addr, 8);
                    patched++;
                }
            }
        }

        printf("  Patched %d DW_LNE_set_address opcodes\n", patched);

        // Patch DWARF5 string offsets in .debug_line
        // DWARF5 uses DW_FORM_line_strp (4-byte offsets into .debug_line_str) for file/dir names
        patchDwarf5StringOffsets();

        // Note: Debug sections are merged and patched in mergeDebugInfo() which is called
        // after layout. It patches .debug_addr addresses and .debug_info CU header offsets.
    }

    void buildRelocations() {
        // Find the existing .rela.dyn section (created as placeholder)
        SectionInfo* rela_sec = nullptr;
        for (auto& s : sections_) {
            if (s.name == ".rela.dyn") { rela_sec = &s; break; }
        }
        if (!rela_sec) {
            fprintf(stderr, "Error: .rela.dyn section not found\n");
            return;
        }

        // Build R_AMDGPU_RELATIVE64 relocations for function tables (like IFC)
        // These relocations tell the runtime to fill table entries with function addresses
        // The runtime applies these per-GPU during code object loading
        rela_sec->data.clear();

        // R_AMDGPU_RELATIVE64 = 13 (0x0d)
        // Each relocation: offset (8) + info (8) + addend (8) = 24 bytes
        const uint64_t R_AMDGPU_RELATIVE64 = 13;

        auto addReloc = [&](uint64_t offset, uint64_t addend) {
            size_t pos = rela_sec->data.size();
            rela_sec->data.resize(pos + 24);
            memcpy(rela_sec->data.data() + pos, &offset, 8);
            uint64_t info = R_AMDGPU_RELATIVE64;
            memcpy(rela_sec->data.data() + pos + 8, &info, 8);
            memcpy(rela_sec->data.data() + pos + 16, &addend, 8);
        };

        // Generate relocations for each function table entry
        // Tables are in .rodata at their original PC-relative offsets
        uint64_t table1_addr = rodata_addr_ + rodata_table_1_off_;
        uint64_t table2_addr = rodata_addr_ + rodata_table_2_off_;
        uint64_t table4_addr = rodata_addr_ + rodata_table_4_off_;

        int reloc_count = 0;
        for (int i = 0; i < FUNC_COUNT; i++) {
            if (table_1_[i]) {
                addReloc(table1_addr + i * 8, text_addr_ + table_1_[i]);
                reloc_count++;
            }
            if (table_2_[i]) {
                addReloc(table2_addr + i * 8, text_addr_ + table_2_[i]);
                reloc_count++;
            }
            if (table_4_[i]) {
                addReloc(table4_addr + i * 8, text_addr_ + table_4_[i]);
                reloc_count++;
            }
        }

        // Add relocations for __clang_gpu_used_external (oneRankReduce kernel addresses)
        // IFC has this symbol in .data with 12 relocations pointing to oneRankReduce kernels
        // These relocations fill the array that tracks "used" device functions
        for (size_t i = 0; i < onerank_text_offsets_.size(); i++) {
            // Each entry in __clang_gpu_used_external is 8 bytes
            addReloc(data_addr_ + i * 8, text_addr_ + onerank_text_offsets_[i]);
            reloc_count++;
        }

        printf("Built .rela.dyn with %d R_AMDGPU_RELATIVE64 relocations\n", reloc_count);
        if (!onerank_text_offsets_.empty()) {
            printf("  (includes %zu relocations for __clang_gpu_used_external)\n", onerank_text_offsets_.size());
        }

        // Record address/size
        rela_addr_ = rela_sec->addr;
        rela_size_ = rela_sec->data.size();

        printf("  .rela.dyn @ 0x%06lx  size=0x%06zx\n", rela_addr_, rela_size_);

        // Now populate .dynamic section with final rela addresses
        populateDynamicSection();
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

        for (size_t i = 0; i < sections_.size(); i++) {
            const auto& s = sections_[i];
            uint32_t link = 0, info = 0;

            if (s.name == ".dynsym" && dynstr_idx >= 0) { link = dynstr_idx; info = 1; }
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
        fwrite(out.data(), 1, out.size(), f);
        fclose(f);

        printf("Wrote %s: %zu bytes\n", path.c_str(), out.size());
        return true;
    }
};

// ============================================================================
// Main
// ============================================================================

void printUsage(const char* prog) {
    fprintf(stderr, "Usage: %s -o output.o --dispatcher disp.o --host-table table.cpp [--target arch] [--input-dir dir | files...]\n", prog);
}

int main(int argc, char** argv) {
    std::string output, dispatcher, host_table, input_dir;
    std::vector<std::string> inputs;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-o" && i + 1 < argc) output = argv[++i];
        else if (arg == "--dispatcher" && i + 1 < argc) dispatcher = argv[++i];
        else if (arg == "--host-table" && i + 1 < argc) host_table = argv[++i];
        else if (arg == "--input-dir" && i + 1 < argc) input_dir = argv[++i];
        else if (arg == "--target" && i + 1 < argc) g_target_arch = argv[++i];
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

    // Collect input files
    bool extract = false;
    if (!input_dir.empty()) {
        for (const auto& e : fs::directory_iterator(input_dir)) {
            if (e.path().extension() == ".o" && e.path().string().find(".device.o") != std::string::npos)
                inputs.push_back(e.path().string());
        }
        if (inputs.empty()) {
            extract = true;
            for (const auto& e : fs::directory_iterator(input_dir)) {
                std::string n = e.path().filename().string();
                if (e.path().extension() == ".o" && n.find("specialized_") == 0 && n.find(".device.o") == std::string::npos)
                    inputs.push_back(e.path().string());
            }
        }
        std::sort(inputs.begin(), inputs.end());
    }

    if (inputs.empty()) { fprintf(stderr, "No input files\n"); return 1; }
    printf("Processing %zu input files\n", inputs.size());

    auto funcid_map = parseHostTable(host_table);
    printf("Loaded %zu funcId mappings\n", funcid_map.size());

    // Process kernels in parallel
    std::vector<KernelInfo> kernels(inputs.size());
    std::mutex mtx;
    int done = 0;

    auto worker = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; i++) {
            auto data = extract ? extractDeviceCode(inputs[i]) : [&]() {
                MappedFile f(inputs[i]);
                if (!f.valid()) return std::vector<uint8_t>();
                const uint8_t* p = static_cast<const uint8_t*>(f.data());
                return std::vector<uint8_t>(p, p + f.size());
            }();
            kernels[i] = parseKernel(data);

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

    // Link
    DeviceLinker linker(dispatcher);
    if (!linker.load()) { fprintf(stderr, "Cannot load dispatcher\n"); return 1; }

    // Set GPU target for proper LDS calculation
    linker.setGpuTarget(g_target_arch);

    for (const auto& k : kernels) linker.addKernel(k);
    linker.setFuncIdMap(funcid_map);

    if (!linker.link(output)) return 1;

    // Generate header file with funcId -> name mapping for host-side tracing
    std::string header_path = output;
    auto dot = header_path.rfind('.');
    if (dot != std::string::npos) header_path = header_path.substr(0, dot);
    header_path += "_funcid_names.h";
    linker.writeFuncIdHeader(header_path);

    return 0;
}
