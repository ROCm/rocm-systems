/**
 * Device Linker - Merges specialized kernel device objects into a single ELF.
 * 
 * Usage:
 *   device_linker -o output.o --dispatcher minimal_device.o --host-table host_table.cpp input1.o input2.o ...
 *   device_linker -o output.o --dispatcher minimal_device.o --host-table host_table.cpp --input-dir <dir>
 */

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
// Constants
// ============================================================================

constexpr int FUNC_COUNT = 859;  // Max funcId is 858
constexpr int FUNC_ALIGNMENT = 256;

// ============================================================================
// Memory-mapped file wrapper
// ============================================================================

class MappedFile {
public:
    MappedFile(const std::string& path) : fd_(-1), data_(nullptr), size_(0) {
        fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            fprintf(stderr, "Error: Cannot open %s\n", path.c_str());
            return;
        }
        
        struct stat st;
        if (fstat(fd_, &st) < 0) {
            close(fd_);
            fd_ = -1;
            return;
        }
        
        size_ = st.st_size;
        data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (data_ == MAP_FAILED) {
            data_ = nullptr;
            close(fd_);
            fd_ = -1;
        }
    }
    
    ~MappedFile() {
        if (data_) munmap(data_, size_);
        if (fd_ >= 0) close(fd_);
    }
    
    // Non-copyable, non-movable (owns raw resources)
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&&) = delete;
    MappedFile& operator=(MappedFile&&) = delete;
    
    bool valid() const { return data_ != nullptr; }
    size_t size() const { return size_; }
    const void* data() const { return data_; }
    
    template<typename T>
    const T* at(size_t offset) const {
        return reinterpret_cast<const T*>(static_cast<const char*>(data_) + offset);
    }
    
    const char* str(size_t offset) const {
        return static_cast<const char*>(data_) + offset;
    }

private:
    int fd_;
    void* data_;
    size_t size_;
};

// ============================================================================
// ELF utilities
// ============================================================================

struct Section {
    std::string name;
    uint32_t type;
    uint64_t flags;
    uint64_t addr;
    uint64_t offset;
    uint64_t size;
};

class ElfFile {
public:
    ElfFile(const MappedFile& file) : file_(file) {
        if (!file.valid()) return;
        
        ehdr_ = file.at<Elf64_Ehdr>(0);
        if (memcmp(ehdr_->e_ident, ELFMAG, SELFMAG) != 0) {
            ehdr_ = nullptr;
            return;
        }
        
        // Parse section headers
        const Elf64_Shdr* shdrs = file.at<Elf64_Shdr>(ehdr_->e_shoff);
        const Elf64_Shdr& shstrtab = shdrs[ehdr_->e_shstrndx];
        const char* strtab = file.str(shstrtab.sh_offset);
        
        for (int i = 0; i < ehdr_->e_shnum; i++) {
            Section sec;
            sec.name = strtab + shdrs[i].sh_name;
            sec.type = shdrs[i].sh_type;
            sec.flags = shdrs[i].sh_flags;
            sec.addr = shdrs[i].sh_addr;
            sec.offset = shdrs[i].sh_offset;
            sec.size = shdrs[i].sh_size;
            sections_.push_back(sec);
        }
    }
    
    bool valid() const { return ehdr_ != nullptr; }
    const Elf64_Ehdr* ehdr() const { return ehdr_; }
    const std::vector<Section>& sections() const { return sections_; }
    
    const Section* findSection(const std::string& name) const {
        for (const auto& sec : sections_) {
            if (sec.name == name) return &sec;
        }
        return nullptr;
    }
    
    const void* sectionData(const Section& sec) const {
        return file_.at<void>(sec.offset);
    }
    
    std::vector<uint8_t> getSectionBytes(const Section& sec) const {
        const uint8_t* data = file_.at<uint8_t>(sec.offset);
        return std::vector<uint8_t>(data, data + sec.size);
    }

private:
    const MappedFile& file_;
    const Elf64_Ehdr* ehdr_ = nullptr;
    std::vector<Section> sections_;
};

// ============================================================================
// Device code extraction from host .o files
// ============================================================================

// Global target arch for device extraction (set from command line)
std::string g_target_arch = "gfx942";

// Extract device code from a host object file using clang-offload-bundler
// Returns empty vector on failure
std::vector<uint8_t> extractDeviceFromHostObject(const std::string& path) {
    const std::string& target_arch = g_target_arch;
    // First check if this is already a device ELF
    {
        MappedFile file(path);
        if (!file.valid()) return {};
        
        const Elf64_Ehdr* ehdr = file.at<Elf64_Ehdr>(0);
        if (memcmp(ehdr->e_ident, ELFMAG, SELFMAG) != 0) return {};
        
        // Device ELF has e_machine == 224 (AMDGPU), host has e_machine == 62 (x86-64)
        if (ehdr->e_machine == 224) {
            const uint8_t* data = static_cast<const uint8_t*>(file.data());
            return std::vector<uint8_t>(data, data + file.size());
        }
    }
    
    // Use clang-offload-bundler to extract device code (handles CCOB compression)
    // First, dump the .hip_fatbin section
    std::string fatbin_path = path + ".fatbin.tmp";
    std::string device_path = path + ".device.tmp";
    
    std::string dump_cmd = "/opt/rocm/llvm/bin/llvm-objcopy --dump-section=.hip_fatbin=\"" + 
                           fatbin_path + "\" \"" + path + "\" 2>/dev/null";
    if (system(dump_cmd.c_str()) != 0) {
        return {};
    }
    
    // Unbundle to get device code
    std::string unbundle_cmd = "/opt/rocm/llvm/bin/clang-offload-bundler --type=o "
                               "--targets=hipv4-amdgcn-amd-amdhsa--" + target_arch + " "
                               "--input=\"" + fatbin_path + "\" "
                               "--output=\"" + device_path + "\" "
                               "--unbundle 2>/dev/null";
    int ret = system(unbundle_cmd.c_str());
    unlink(fatbin_path.c_str());
    
    if (ret != 0) {
        return {};
    }
    
    // Read the extracted device code
    std::vector<uint8_t> result;
    MappedFile device_file(device_path);
    if (device_file.valid()) {
        const uint8_t* data = static_cast<const uint8_t*>(device_file.data());
        result.assign(data, data + device_file.size());
    }
    
    unlink(device_path.c_str());
    return result;
}

// ============================================================================
// Kernel info extracted from each .device.o
// ============================================================================

struct KernelInfo {
    std::string source_file;
    std::string mangled_name;
    uint64_t func_offset;      // Offset within .text
    uint64_t func_size;
    std::vector<uint8_t> code; // The actual machine code
    
    // Resource requirements
    int vgpr_count = 0;
    int sgpr_count = 0;
    int lds_size = 0;
    int stack_size = 0;
};

// Parse symbols from .symtab
std::string findDevFunc(const MappedFile& file, const ElfFile& elf, 
                        uint64_t& offset, uint64_t& size) {
    const Section* symtab = elf.findSection(".symtab");
    const Section* strtab = elf.findSection(".strtab");
    if (!symtab || !strtab) return "";
    
    const char* strings = file.str(strtab->offset);
    const Elf64_Sym* syms = file.at<Elf64_Sym>(symtab->offset);
    size_t nsyms = symtab->size / sizeof(Elf64_Sym);
    
    for (size_t i = 0; i < nsyms; i++) {
        if (ELF64_ST_TYPE(syms[i].st_info) == STT_FUNC) {
            const char* name = strings + syms[i].st_name;
            if (strstr(name, "ncclDevFunc_") != nullptr) {
                offset = syms[i].st_value;
                size = syms[i].st_size;
                return name;
            }
        }
    }
    return "";
}

// Parse resource requirements from .note section
void parseNoteMetadata(const MappedFile& file, const ElfFile& elf, KernelInfo& info) {
    const Section* note = elf.findSection(".note");
    if (!note) return;
    
    // Walk note entries
    size_t pos = 0;
    while (pos < note->size) {
        const uint32_t* hdr = file.at<uint32_t>(note->offset + pos);
        uint32_t namesz = hdr[0];
        uint32_t descsz = hdr[1];
        uint32_t type = hdr[2];
        
        size_t name_off = pos + 12;
        size_t desc_off = ((name_off + namesz + 3) & ~3);
        
        // Look for AMDGPU metadata (type 32)
        if (type == 32) {
            const char* desc = file.str(note->offset + desc_off);
            // Parse MessagePack metadata - look for known keys
            // This is a simplified parser that looks for the string keys
            std::string_view data(desc, descsz);
            
            auto findInt = [&](const char* key) -> int {
                auto pos = data.find(key);
                if (pos == std::string_view::npos) return 0;
                pos += strlen(key);
                if (pos >= data.size()) return 0;
                uint8_t b = data[pos];
                if (b <= 0x7f) return b;  // fixint
                if (b == 0xcc && pos + 1 < data.size()) return (uint8_t)data[pos + 1];  // uint8
                if (b == 0xcd && pos + 2 < data.size()) return ((uint8_t)data[pos + 1] << 8) | (uint8_t)data[pos + 2];  // uint16
                if (b == 0xce && pos + 4 < data.size()) {
                    return ((uint8_t)data[pos + 1] << 24) | ((uint8_t)data[pos + 2] << 16) |
                           ((uint8_t)data[pos + 3] << 8) | (uint8_t)data[pos + 4];  // uint32
                }
                return 0;
            };
            
            info.vgpr_count = findInt(".vgpr_count");
            info.sgpr_count = findInt(".sgpr_count");
            info.lds_size = findInt(".group_segment_fixed_size");
            info.stack_size = findInt(".private_segment_fixed_size");
        }
        
        pos = ((desc_off + descsz + 3) & ~3);
    }
}

// Process device code from bytes (for extracted device code)
KernelInfo processDeviceBytes(const std::string& source_name, const std::vector<uint8_t>& device_data) {
    KernelInfo info;
    info.source_file = source_name;
    
    if (device_data.empty()) return info;
    
    // Create a temporary file to parse with existing ELF utilities
    // (Could optimize this later to parse in-memory)
    std::string tmp_path = "/tmp/device_" + std::to_string(std::hash<std::string>{}(source_name)) + ".o";
    FILE* tmp = fopen(tmp_path.c_str(), "wb");
    if (!tmp) return info;
    fwrite(device_data.data(), 1, device_data.size(), tmp);
    fclose(tmp);
    
    MappedFile file(tmp_path);
    if (!file.valid()) {
        unlink(tmp_path.c_str());
        return info;
    }
    
    ElfFile elf(file);
    if (!elf.valid()) {
        unlink(tmp_path.c_str());
        return info;
    }
    
    // Find ncclDevFunc_* symbol
    uint64_t func_offset, func_size;
    info.mangled_name = findDevFunc(file, elf, func_offset, func_size);
    if (info.mangled_name.empty()) {
        unlink(tmp_path.c_str());
        return info;
    }
    
    info.func_offset = func_offset;
    info.func_size = func_size;
    
    // Get .text section and extract function code
    const Section* text = elf.findSection(".text");
    if (!text) {
        unlink(tmp_path.c_str());
        return info;
    }
    
    // Calculate file offset of function
    uint64_t file_offset = text->offset + (func_offset - text->addr);
    const uint8_t* code_ptr = file.at<uint8_t>(file_offset);
    info.code.assign(code_ptr, code_ptr + func_size);
    
    // Parse metadata
    parseNoteMetadata(file, elf, info);
    
    unlink(tmp_path.c_str());
    return info;
}

// Process a single device object file (or host object with embedded device code)
KernelInfo processDeviceObject(const std::string& path, bool extract_from_host = false) {
    KernelInfo info;
    info.source_file = fs::path(path).filename().string();
    
    // If extracting from host object, try to get device code first
    if (extract_from_host) {
        auto device_data = extractDeviceFromHostObject(path);
        if (!device_data.empty()) {
            return processDeviceBytes(info.source_file, device_data);
        }
        // Fall through to try direct parsing (maybe it's actually a device object)
    }
    
    MappedFile file(path);
    if (!file.valid()) return info;
    
    ElfFile elf(file);
    if (!elf.valid()) return info;
    
    // Find ncclDevFunc_* symbol
    uint64_t func_offset, func_size;
    info.mangled_name = findDevFunc(file, elf, func_offset, func_size);
    if (info.mangled_name.empty()) return info;
    
    info.func_offset = func_offset;
    info.func_size = func_size;
    
    // Get .text section and extract function code
    const Section* text = elf.findSection(".text");
    if (!text) return info;
    
    // Calculate file offset of function
    uint64_t file_offset = text->offset + (func_offset - text->addr);
    const uint8_t* code_ptr = file.at<uint8_t>(file_offset);
    info.code.assign(code_ptr, code_ptr + func_size);
    
    // Parse metadata
    parseNoteMetadata(file, elf, info);
    
    return info;
}

// ============================================================================
// FuncId mapping from host_table.cpp
// ============================================================================

struct FuncIdMapping {
    int func_id;
    int unroll;  // 1, 2, or 4
};

std::unordered_map<std::string, FuncIdMapping> parseHostTable(const std::string& path) {
    std::unordered_map<std::string, FuncIdMapping> mapping;
    
    std::ifstream file(path);
    if (!file) {
        fprintf(stderr, "Warning: Cannot open host_table.cpp: %s\n", path.c_str());
        return mapping;
    }
    
    // Match: {key, id}, // Comment COLL ALGO PROTO REDOP TYPE ACC PIPELINE UNROLL
    std::regex pattern(R"(\{(\d+),\s*(\d+)\},\s*//\s*(.+))");
    std::string line;
    
    while (std::getline(file, line)) {
        std::smatch match;
        if (std::regex_search(line, match, pattern)) {
            int func_id = std::stoi(match[2]);
            std::string comment = match[3];
            
            // Parse comment: "AllReduce RING LL Sum f32 0 0 2"
            std::istringstream iss(comment);
            std::vector<std::string> parts;
            std::string part;
            while (iss >> part) parts.push_back(part);
            
            if (parts.size() >= 8) {
                // Build lookup key: Coll_Algo_Proto_Redop_Type_Acc_Pipeline_Unroll
                // (includes unroll to match the mangled name exactly)
                std::string key = parts[0] + "_" + parts[1] + "_" + parts[2] + "_" +
                                  parts[3] + "_" + parts[4] + "_" + parts[5] + "_" + parts[6];
                int unroll = std::stoi(parts[7]);
                mapping[key] = {func_id, unroll};
            }
        }
    }
    
    return mapping;
}

// Demangle to extract function name components and unroll factor
struct DemangleResult {
    std::string key;  // Coll_Algo_Proto_Redop_Type_Acc_Pipeline (without unroll)
    int unroll;       // 1, 2, or 4
};

DemangleResult demangleToKey(const std::string& mangled) {
    // _Z48ncclDevFunc_AllReduce_RING_LL_Sum_f32_0_0_2v
    // -> key: AllReduce_RING_LL_Sum_f32_0_0, unroll: 2
    
    if (mangled.substr(0, 2) != "_Z") return {"", 0};
    
    size_t i = 2;
    while (i < mangled.size() && isdigit(mangled[i])) i++;
    if (i == 2) return {"", 0};
    
    int len = std::stoi(mangled.substr(2, i - 2));
    std::string name = mangled.substr(i, len);
    
    // Remove "ncclDevFunc_" prefix
    const char* prefix = "ncclDevFunc_";
    if (name.substr(0, strlen(prefix)) == prefix) {
        name = name.substr(strlen(prefix));
    }
    
    // Parse: AllReduce_RING_LL_Sum_f32_0_0_2
    auto parts = std::vector<std::string>();
    std::istringstream iss(name);
    std::string part;
    while (std::getline(iss, part, '_')) parts.push_back(part);
    
    if (parts.size() < 8) return {"", 0};
    
    // Extract unroll (last part)
    int unroll = std::stoi(parts.back());
    
    // Rebuild key without the last part (unroll)
    std::string key;
    for (size_t j = 0; j < parts.size() - 1; j++) {
        if (j > 0) key += "_";
        key += parts[j];
    }
    
    return {key, unroll};
}

// ============================================================================
// ELF Builder
// ============================================================================

class ElfBuilder {
public:
    void setFlags(uint32_t flags) { flags_ = flags; }
    
    void setDynamicInfo(uint64_t symtab_addr, uint64_t strtab_addr, uint64_t strtab_size,
                        uint64_t gnuhash_addr, uint64_t hash_addr) {
        has_dynamic_info_ = true;
        dyn_symtab_ = symtab_addr;
        dyn_strtab_ = strtab_addr;
        dyn_strsz_ = strtab_size;
        dyn_gnuhash_ = gnuhash_addr;
        dyn_hash_ = hash_addr;
    }
    
    void addSection(const std::string& name, uint32_t type, uint64_t flags,
                    uint64_t addr, const std::vector<uint8_t>& data, 
                    uint64_t align = 1, uint64_t entsize = 0) {
        SectionDef sec;
        sec.name = name;
        sec.type = type;
        sec.flags = flags;
        sec.addr = addr;
        sec.data = data;
        sec.align = align;
        sec.entsize = entsize;
        sections_.push_back(std::move(sec));
    }
    
    void updateSection(const std::string& name, const std::vector<uint8_t>& data) {
        for (auto& sec : sections_) {
            if (sec.name == name) {
                sec.data = data;
                return;
            }
        }
    }
    
    std::vector<uint8_t> build() {
        // For DYN output, we need:
        // - ELF header with program headers
        // - Program headers (LOAD segments)
        // - Sections
        // - .dynamic section
        // - Section headers
        
        // Build section name string table
        std::vector<uint8_t> shstrtab;
        shstrtab.push_back(0);  // Empty string at index 0
        std::vector<uint32_t> name_offsets;
        name_offsets.push_back(0);  // NULL section
        
        for (const auto& sec : sections_) {
            name_offsets.push_back(shstrtab.size());
            shstrtab.insert(shstrtab.end(), sec.name.begin(), sec.name.end());
            shstrtab.push_back(0);
        }
        // Add .dynamic name
        uint32_t dynamic_name_off = shstrtab.size();
        const char* dynamic_name = ".dynamic";
        shstrtab.insert(shstrtab.end(), dynamic_name, dynamic_name + strlen(dynamic_name) + 1);
        
        // Add .shstrtab name
        uint32_t shstrtab_name_off = shstrtab.size();
        const char* shstrtab_name = ".shstrtab";
        shstrtab.insert(shstrtab.end(), shstrtab_name, shstrtab_name + strlen(shstrtab_name) + 1);
        
        // Calculate layout with program headers
        // Layout: ELF header (64) + Program headers + Padding + Section data + .dynamic + .shstrtab + Section headers
        // CRITICAL: For LOAD segment with p_offset=0, p_vaddr=0, file offsets MUST equal virtual addresses
        const size_t num_phdrs = 3;  // PHDR, LOAD, DYNAMIC
        const size_t phdr_size = 56;
        size_t header_end = 64 + num_phdrs * phdr_size;  // After ELF header and program headers
        
        std::vector<size_t> section_offsets;
        size_t offset = header_end;
        
        for (const auto& sec : sections_) {
            // Each section's file offset must equal its virtual address
            // This ensures the LOAD segment (p_offset=0, p_vaddr=0) maps correctly
            size_t target_offset = sec.addr;
            if (target_offset < offset) {
                // Virtual address is before current offset - need to adjust
                // This shouldn't happen if sections are sorted by address
                fprintf(stderr, "Warning: Section %s vaddr 0x%lx < current offset 0x%lx\n",
                        sec.name.c_str(), sec.addr, offset);
                target_offset = (offset + sec.align - 1) & ~(sec.align - 1);
            }
            section_offsets.push_back(target_offset);
            offset = target_offset + sec.data.size();
        }
        
        // .dynamic section - populate with symbol table pointers
        offset = (offset + 8 - 1) & ~(8 - 1);
        size_t dynamic_offset = offset;
        std::vector<uint8_t> dynamic_data;
        
        auto addDynEntry = [&](uint64_t tag, uint64_t val) {
            size_t pos = dynamic_data.size();
            dynamic_data.resize(pos + 16);
            memcpy(dynamic_data.data() + pos, &tag, 8);
            memcpy(dynamic_data.data() + pos + 8, &val, 8);
        };
        
        if (has_dynamic_info_) {
            // SYMTAB, SYMENT, STRTAB, STRSZ, GNU_HASH, HASH, NULL
            addDynEntry(6, dyn_symtab_);   // DT_SYMTAB
            addDynEntry(11, 24);            // DT_SYMENT (24 bytes per symbol)
            addDynEntry(5, dyn_strtab_);    // DT_STRTAB
            addDynEntry(10, dyn_strsz_);    // DT_STRSZ
            addDynEntry(0x6ffffef5, dyn_gnuhash_); // DT_GNU_HASH
            addDynEntry(4, dyn_hash_);      // DT_HASH
        }
        addDynEntry(0, 0);  // DT_NULL
        
        offset += dynamic_data.size();
        
        // shstrtab section
        size_t shstrtab_offset = offset;
        offset += shstrtab.size();
        
        // Section headers (8-byte aligned)
        offset = (offset + 8 - 1) & ~(8 - 1);
        size_t shdr_offset = offset;
        size_t num_sections = sections_.size() + 3;  // NULL + sections + .dynamic + .shstrtab
        
        // Build output
        std::vector<uint8_t> out(shdr_offset + num_sections * 64);
        
        // ELF header
        Elf64_Ehdr ehdr = {};
        memcpy(ehdr.e_ident, ELFMAG, SELFMAG);
        ehdr.e_ident[EI_CLASS] = ELFCLASS64;
        ehdr.e_ident[EI_DATA] = ELFDATA2LSB;
        ehdr.e_ident[EI_VERSION] = EV_CURRENT;
        ehdr.e_ident[EI_OSABI] = 64;  // ELFOSABI_AMDGPU_HSA
        ehdr.e_ident[EI_ABIVERSION] = 4;  // AMDGPU ABI version 4
        ehdr.e_type = ET_DYN;  // Shared object (DYN)
        ehdr.e_machine = 224;  // EM_AMDGPU
        ehdr.e_version = EV_CURRENT;
        ehdr.e_phoff = 64;  // Program headers start right after ELF header
        ehdr.e_shoff = shdr_offset;
        ehdr.e_flags = flags_;
        ehdr.e_ehsize = 64;
        ehdr.e_phentsize = phdr_size;
        ehdr.e_phnum = num_phdrs;
        ehdr.e_shentsize = 64;
        ehdr.e_shnum = num_sections;
        ehdr.e_shstrndx = num_sections - 1;
        memcpy(out.data(), &ehdr, sizeof(ehdr));
        
        // Program headers
        size_t phdr_off = 64;
        
        // PHDR segment (points to program header table itself)
        Elf64_Phdr phdr_phdr = {};
        phdr_phdr.p_type = PT_PHDR;
        phdr_phdr.p_flags = PF_R;
        phdr_phdr.p_offset = 64;
        phdr_phdr.p_vaddr = 64;
        phdr_phdr.p_paddr = 64;
        phdr_phdr.p_filesz = num_phdrs * phdr_size;
        phdr_phdr.p_memsz = num_phdrs * phdr_size;
        phdr_phdr.p_align = 8;
        memcpy(out.data() + phdr_off, &phdr_phdr, sizeof(phdr_phdr));
        phdr_off += phdr_size;
        
        // LOAD segment for all sections (simplified: one LOAD covering everything)
        Elf64_Phdr phdr_load = {};
        phdr_load.p_type = PT_LOAD;
        phdr_load.p_flags = PF_R | PF_W | PF_X;
        phdr_load.p_offset = 0;
        phdr_load.p_vaddr = 0;
        phdr_load.p_paddr = 0;
        phdr_load.p_filesz = shdr_offset;  // Everything up to section headers
        phdr_load.p_memsz = shdr_offset;
        phdr_load.p_align = 0x1000;
        memcpy(out.data() + phdr_off, &phdr_load, sizeof(phdr_load));
        phdr_off += phdr_size;
        
        // DYNAMIC segment
        Elf64_Phdr phdr_dyn = {};
        phdr_dyn.p_type = PT_DYNAMIC;
        phdr_dyn.p_flags = PF_R | PF_W;
        phdr_dyn.p_offset = dynamic_offset;
        phdr_dyn.p_vaddr = dynamic_offset;
        phdr_dyn.p_paddr = dynamic_offset;
        phdr_dyn.p_filesz = dynamic_data.size();
        phdr_dyn.p_memsz = dynamic_data.size();
        phdr_dyn.p_align = 8;
        memcpy(out.data() + phdr_off, &phdr_dyn, sizeof(phdr_dyn));
        
        // Section data
        for (size_t i = 0; i < sections_.size(); i++) {
            memcpy(out.data() + section_offsets[i], 
                   sections_[i].data.data(), sections_[i].data.size());
        }
        // .dynamic data
        memcpy(out.data() + dynamic_offset, dynamic_data.data(), dynamic_data.size());
        // .shstrtab data
        memcpy(out.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());
        
        // Section headers
        auto writeShdr = [&](size_t idx, uint32_t name, uint32_t type, uint64_t flags,
                            uint64_t addr, uint64_t offset, uint64_t size,
                            uint32_t link, uint32_t info, uint64_t align, uint64_t entsize) {
            Elf64_Shdr shdr = {};
            shdr.sh_name = name;
            shdr.sh_type = type;
            shdr.sh_flags = flags;
            shdr.sh_addr = addr;
            shdr.sh_offset = offset;
            shdr.sh_size = size;
            shdr.sh_link = link;
            shdr.sh_info = info;
            shdr.sh_addralign = align;
            shdr.sh_entsize = entsize;
            memcpy(out.data() + shdr_offset + idx * 64, &shdr, sizeof(shdr));
        };
        
        // NULL section
        writeShdr(0, 0, SHT_NULL, 0, 0, 0, 0, 0, 0, 0, 0);
        
        // Find section indices for linking
        int dynstr_idx = -1, dynsym_idx = -1;
        for (size_t i = 0; i < sections_.size(); i++) {
            if (sections_[i].name == ".dynstr") dynstr_idx = i + 1;
            if (sections_[i].name == ".dynsym") dynsym_idx = i + 1;
        }
        
        // User sections
        for (size_t i = 0; i < sections_.size(); i++) {
            const auto& sec = sections_[i];
            uint32_t link = 0;
            uint32_t info = 0;
            
            // Set sh_link based on section type
            if (sec.name == ".dynsym" && dynstr_idx >= 0) {
                link = dynstr_idx;  // .dynsym links to .dynstr
                info = 1;           // First non-local symbol
            } else if ((sec.name == ".gnu.hash" || sec.name == ".hash") && dynsym_idx >= 0) {
                link = dynsym_idx;  // Hash tables link to .dynsym
            }
            
            writeShdr(i + 1, name_offsets[i + 1], sec.type, sec.flags,
                      sec.addr, section_offsets[i], sec.data.size(),
                      link, info, sec.align, sec.entsize);
        }
        
        // .dynamic section (links to .dynstr)
        size_t dynamic_idx = sections_.size() + 1;
        writeShdr(dynamic_idx, dynamic_name_off, SHT_DYNAMIC, SHF_ALLOC | SHF_WRITE,
                  dynamic_offset, dynamic_offset, dynamic_data.size(), 
                  dynstr_idx >= 0 ? dynstr_idx : 0, 0, 8, 16);
        
        // .shstrtab
        writeShdr(num_sections - 1, shstrtab_name_off, SHT_STRTAB, 0,
                  0, shstrtab_offset, shstrtab.size(), 0, 0, 1, 0);
        
        return out;
    }

private:
    struct SectionDef {
        std::string name;
        uint32_t type;
        uint64_t flags;
        uint64_t addr;
        std::vector<uint8_t> data;
        uint64_t align;
        uint64_t entsize;
    };
    
    uint32_t flags_ = 0;
    std::vector<SectionDef> sections_;
    
    // Dynamic section info
    bool has_dynamic_info_ = false;
    uint64_t dyn_symtab_ = 0;
    uint64_t dyn_strtab_ = 0;
    uint64_t dyn_strsz_ = 0;
    uint64_t dyn_gnuhash_ = 0;
    uint64_t dyn_hash_ = 0;
};

// ============================================================================
// Main
// ============================================================================

void printUsage(const char* prog) {
    fprintf(stderr, "Usage: %s -o output.o --dispatcher disp.o --host-table host_table.cpp [--target arch] [--input-dir dir | files...]\n", prog);
    fprintf(stderr, "  --target arch  GPU target (e.g., gfx942:sramecc+:xnack+)\n");
}

int main(int argc, char** argv) {
    std::string output_path;
    std::string dispatcher_path;
    std::string host_table_path;
    std::string input_dir;
    std::vector<std::string> input_files;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-o" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--dispatcher" && i + 1 < argc) {
            dispatcher_path = argv[++i];
        } else if (arg == "--host-table" && i + 1 < argc) {
            host_table_path = argv[++i];
        } else if (arg == "--input-dir" && i + 1 < argc) {
            input_dir = argv[++i];
        } else if (arg == "--target" && i + 1 < argc) {
            g_target_arch = argv[++i];
        } else if (arg[0] != '-') {
            input_files.push_back(arg);
        }
    }
    
    printf("Device Linker: target=%s\n", g_target_arch.c_str());
    
    if (output_path.empty() || dispatcher_path.empty()) {
        printUsage(argv[0]);
        return 1;
    }
    
    // Collect input files from directory if specified
    bool use_device_extraction = false;
    if (!input_dir.empty()) {
        // First, look for pre-extracted .device.o files (from hipcc wrapper)
        for (const auto& entry : fs::directory_iterator(input_dir)) {
            if (entry.path().extension() == ".o" && 
                entry.path().string().find(".device.o") != std::string::npos) {
                input_files.push_back(entry.path().string());
            }
        }
        
        // If no .device.o files, look for specialized_*.o files and extract device code
        if (input_files.empty()) {
            printf("No .device.o files found, will extract device code from .o files\n");
            use_device_extraction = true;
            for (const auto& entry : fs::directory_iterator(input_dir)) {
                std::string name = entry.path().filename().string();
                if (entry.path().extension() == ".o" && 
                    name.find("specialized_") == 0 &&
                    name.find(".device.o") == std::string::npos) {
                    input_files.push_back(entry.path().string());
                }
            }
        }
        
        std::sort(input_files.begin(), input_files.end());
    }
    
    if (input_files.empty()) {
        fprintf(stderr, "Error: No input files\n");
        return 1;
    }
    
    printf("Device Linker: Processing %zu input files\n", input_files.size());
    
    // Parse host_table for funcId mapping
    auto funcid_map = parseHostTable(host_table_path);
    printf("Loaded %zu funcId mappings from host_table.cpp\n", funcid_map.size());
    
    // Process input files in parallel
    std::vector<KernelInfo> kernels(input_files.size());
    std::mutex progress_mutex;
    int processed = 0;
    
    auto worker = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; i++) {
            kernels[i] = processDeviceObject(input_files[i], use_device_extraction);
            
            std::lock_guard<std::mutex> lock(progress_mutex);
            processed++;
            if (processed % 100 == 0) {
                printf("  Processed %d/%zu...\n", processed, input_files.size());
            }
        }
    };
    
    // Use thread pool
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    
    std::vector<std::thread> threads;
    size_t chunk_size = (input_files.size() + num_threads - 1) / num_threads;
    
    for (unsigned int t = 0; t < num_threads; t++) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, input_files.size());
        if (start < end) {
            threads.emplace_back(worker, start, end);
        }
    }
    
    for (auto& th : threads) {
        th.join();
    }
    
    // Find max resource requirements
    int max_vgpr = 0, max_sgpr = 0, max_lds = 0, max_stack = 0;
    for (const auto& k : kernels) {
        max_vgpr = std::max(max_vgpr, k.vgpr_count);
        max_sgpr = std::max(max_sgpr, k.sgpr_count);
        max_lds = std::max(max_lds, k.lds_size);
        max_stack = std::max(max_stack, k.stack_size);
    }
    printf("Max resources: VGPR=%d, SGPR=%d, LDS=%d, Stack=%d\n", 
           max_vgpr, max_sgpr, max_lds, max_stack);
    
    // Load dispatcher
    MappedFile disp_file(dispatcher_path);
    if (!disp_file.valid()) {
        fprintf(stderr, "Error: Cannot load dispatcher: %s\n", dispatcher_path.c_str());
        return 1;
    }
    ElfFile dispatcher(disp_file);
    if (!dispatcher.valid()) {
        fprintf(stderr, "Error: Invalid dispatcher ELF\n");
        return 1;
    }
    
    // Get dispatcher sections
    const Section* disp_note = dispatcher.findSection(".note");
    const Section* disp_rodata = dispatcher.findSection(".rodata");
    const Section* disp_text = dispatcher.findSection(".text");
    const Section* disp_bss = dispatcher.findSection(".bss");
    
    if (!disp_text || !disp_bss) {
        fprintf(stderr, "Error: Dispatcher missing required sections\n");
        return 1;
    }
    
    // Build merged code section
    std::vector<uint8_t> disp_text_data = dispatcher.getSectionBytes(*disp_text);
    
    // Align to 256 for function code
    while (disp_text_data.size() % FUNC_ALIGNMENT != 0) {
        disp_text_data.push_back(0);
    }
    
    uint64_t func_code_vaddr = disp_text->addr + disp_text_data.size();
    printf("Function code starts at vaddr 0x%lx\n", func_code_vaddr);
    
    // Build function tables (one per unroll factor) and append code
    // Layout: table_1[FUNC_COUNT], table_2[FUNC_COUNT], table_4[FUNC_COUNT]
    std::vector<uint64_t> func_table_1(FUNC_COUNT, 0);
    std::vector<uint64_t> func_table_2(FUNC_COUNT, 0);
    std::vector<uint64_t> func_table_4(FUNC_COUNT, 0);
    int mapped_count = 0;
    int mapped_unroll_1 = 0, mapped_unroll_2 = 0, mapped_unroll_4 = 0;
    size_t total_code_size = 0;
    
    std::vector<std::string> unmapped;
    int empty_name = 0, empty_code = 0, out_of_range = 0, bad_unroll = 0;
    for (const auto& k : kernels) {
        if (k.mangled_name.empty()) { empty_name++; continue; }
        if (k.code.empty()) { empty_code++; continue; }
        
        auto [key, unroll] = demangleToKey(k.mangled_name);
        if (key.empty()) {
            unmapped.push_back(k.mangled_name + " -> failed to demangle");
            continue;
        }
        
        auto it = funcid_map.find(key);
        if (it == funcid_map.end()) {
            unmapped.push_back(k.mangled_name + " -> key='" + key + "'");
            continue;
        }
        
        int funcid = it->second.func_id;
        int expected_unroll = it->second.unroll;
        
        if (funcid < 0 || funcid >= FUNC_COUNT) {
            out_of_range++;
            continue;
        }
        
        // Verify unroll matches
        if (unroll != expected_unroll) {
            fprintf(stderr, "Warning: unroll mismatch for %s: got %d, expected %d\n",
                    k.mangled_name.c_str(), unroll, expected_unroll);
        }
        
        // Record function address (current end of .text section)
        uint64_t func_vaddr = disp_text->addr + disp_text_data.size();
        
        // Store in the appropriate table based on unroll factor
        switch (unroll) {
            case 1:
                func_table_1[funcid] = func_vaddr;
                mapped_unroll_1++;
                break;
            case 2:
                func_table_2[funcid] = func_vaddr;
                mapped_unroll_2++;
                break;
            case 4:
                func_table_4[funcid] = func_vaddr;
                mapped_unroll_4++;
                break;
            default:
                bad_unroll++;
                continue;
        }
        
        // Append code
        disp_text_data.insert(disp_text_data.end(), k.code.begin(), k.code.end());
        total_code_size += k.code.size();
        
        // Align for next function
        while (disp_text_data.size() % FUNC_ALIGNMENT != 0) {
            disp_text_data.push_back(0);
        }
        
        mapped_count++;
    }
    
    printf("Mapped %d functions (unroll1=%d, unroll2=%d, unroll4=%d), total code size: %zu bytes\n", 
           mapped_count, mapped_unroll_1, mapped_unroll_2, mapped_unroll_4, total_code_size);
    printf("Skipped: %d empty name, %d empty code, %d out of range, %d bad unroll\n", 
           empty_name, empty_code, out_of_range, bad_unroll);
    fflush(stdout);
    
    if (!unmapped.empty()) {
        printf("Unmapped functions (%zu):\n", unmapped.size());
        for (size_t i = 0; i < std::min(unmapped.size(), size_t(10)); i++) {
            printf("  %s\n", unmapped[i].c_str());
        }
        if (unmapped.size() > 10) {
            printf("  ... and %zu more\n", unmapped.size() - 10);
        }
    }
    
    // Build .data section (function tables)
    // Layout: table_1[FUNC_COUNT], table_2[FUNC_COUNT], table_4[FUNC_COUNT]
    size_t table_size = FUNC_COUNT * 8;
    std::vector<uint8_t> data_section(table_size * 3, 0);
    
    // Populate all three tables
    for (int i = 0; i < FUNC_COUNT; i++) {
        memcpy(data_section.data() + i * 8, &func_table_1[i], 8);
        memcpy(data_section.data() + table_size + i * 8, &func_table_2[i], 8);
        memcpy(data_section.data() + table_size * 2 + i * 8, &func_table_4[i], 8);
    }
    
    // Get additional dispatcher sections for symbol tables
    const Section* disp_dynsym = dispatcher.findSection(".dynsym");
    const Section* disp_dynstr = dispatcher.findSection(".dynstr");
    const Section* disp_gnuhash = dispatcher.findSection(".gnu.hash");
    const Section* disp_hash = dispatcher.findSection(".hash");
    
    // Calculate relocated .data address (must be after .text ends)
    // disp_text_data already contains the merged code
    uint64_t text_end = disp_text->addr + disp_text_data.size();
    uint64_t new_data_addr = (text_end + 0xFFF) & ~0xFFFULL;  // Page-align
    uint64_t data_offset = new_data_addr - disp_bss->addr;  // Delta for symbol fixup
    
    printf("Relocating .data: 0x%lx -> 0x%lx (delta=0x%lx)\n", 
           disp_bss->addr, new_data_addr, data_offset);
    
    // Fix .dynsym entries that point to .data section before adding to builder
    std::vector<uint8_t> dynsym_data;
    if (disp_dynsym) {
        dynsym_data = dispatcher.getSectionBytes(*disp_dynsym);
        // Fix symbol values pointing to old .data/.bss range
        for (size_t i = 0; i < dynsym_data.size(); i += 24) {
            if (i + 24 > dynsym_data.size()) break;
            
            uint64_t st_value;
            memcpy(&st_value, dynsym_data.data() + i + 8, 8);
            
            // If symbol value is in the old .data/.bss range, relocate it
            if (st_value >= disp_bss->addr && st_value < disp_bss->addr + 0x10000) {
                st_value += data_offset;
                memcpy(dynsym_data.data() + i + 8, &st_value, 8);
            }
        }
    }
    
    // Build output ELF using our ElfBuilder
    ElfBuilder builder;
    builder.setFlags(dispatcher.ehdr()->e_flags);
    
    // .note
    if (disp_note) {
        auto note_data = dispatcher.getSectionBytes(*disp_note);
        builder.addSection(".note", SHT_NOTE, SHF_ALLOC, disp_note->addr, note_data, 4);
    }
    
    // .dynsym (with fixed symbol values)
    if (disp_dynsym) {
        builder.addSection(".dynsym", SHT_DYNSYM, SHF_ALLOC, disp_dynsym->addr, dynsym_data, 8, 24);
    }
    
    // .gnu.hash (copy from dispatcher)
    if (disp_gnuhash) {
        auto gnuhash_data = dispatcher.getSectionBytes(*disp_gnuhash);
        builder.addSection(".gnu.hash", SHT_GNU_HASH, SHF_ALLOC, disp_gnuhash->addr, gnuhash_data, 8);
    }
    
    // .hash (copy from dispatcher)
    if (disp_hash) {
        auto hash_data = dispatcher.getSectionBytes(*disp_hash);
        builder.addSection(".hash", SHT_HASH, SHF_ALLOC, disp_hash->addr, hash_data, 4, 4);
    }
    
    // .dynstr (copy from dispatcher)
    if (disp_dynstr) {
        auto dynstr_data = dispatcher.getSectionBytes(*disp_dynstr);
        builder.addSection(".dynstr", SHT_STRTAB, SHF_ALLOC, disp_dynstr->addr, dynstr_data, 1);
    }
    
    // .rodata
    if (disp_rodata) {
        auto rodata_data = dispatcher.getSectionBytes(*disp_rodata);
        
        // Update kernel descriptors with max resources
        // KD offsets: 0, 64, 128 (3 kernels)
        auto updateKD = [&](size_t off) {
            if (off + 64 > rodata_data.size()) return;
            
            // LDS (offset 0)
            uint32_t lds = max_lds;
            memcpy(rodata_data.data() + off, &lds, 4);
            
            // Stack (offset 4)
            uint32_t stack = max_stack;
            memcpy(rodata_data.data() + off + 4, &stack, 4);
            
            // RSRC1 (offset 0x30) - VGPR/SGPR encoding
            uint32_t rsrc1;
            memcpy(&rsrc1, rodata_data.data() + off + 0x30, 4);
            int vgpr_granule = (max_vgpr + 3) / 4 - 1;
            int sgpr_granule = (max_sgpr + 7) / 8 - 1;
            rsrc1 = (rsrc1 & ~0x3FF) | (vgpr_granule & 0x3F) | ((sgpr_granule & 0xF) << 6);
            memcpy(rodata_data.data() + off + 0x30, &rsrc1, 4);
        };
        
        updateKD(0);
        updateKD(64);
        updateKD(128);
        
        builder.addSection(".rodata", SHT_PROGBITS, SHF_ALLOC, disp_rodata->addr, rodata_data, 64);
    }
    
    // .text (merged)
    builder.addSection(".text", SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, 
                       disp_text->addr, disp_text_data, 256);
    
    // .data (function tables) at relocated address
    builder.addSection(".data", SHT_PROGBITS, SHF_ALLOC | SHF_WRITE,
                       new_data_addr, data_section, 16);
    
    // Set dynamic section info for proper .dynamic entries
    if (disp_dynsym && disp_dynstr && disp_gnuhash && disp_hash) {
        builder.setDynamicInfo(disp_dynsym->addr, disp_dynstr->addr, disp_dynstr->size,
                               disp_gnuhash->addr, disp_hash->addr);
    }
    
    // Write output
    auto output = builder.build();
    
    FILE* out = fopen(output_path.c_str(), "wb");
    if (!out) {
        fprintf(stderr, "Error: Cannot write output: %s\n", output_path.c_str());
        return 1;
    }
    fwrite(output.data(), 1, output.size(), out);
    fclose(out);
    
    printf("Wrote %s: %zu bytes\n", output_path.c_str(), output.size());
    
    return 0;
}
