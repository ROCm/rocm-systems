/**
 * Device Linker - Merges specialized kernel device objects into a single ELF.
 * 
 * Two-pass design:
 *   Pass 1: Collect all sections, compute final sizes
 *   Pass 2: Layout sections sequentially, compute addresses
 *   Pass 3: Patch sections with final addresses, write output
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
                uint64_t off = text->offset + (syms[i].st_value - text->addr);
                const uint8_t* p = file.at<uint8_t>(off);
                info.code.assign(p, p + syms[i].st_size);
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
        return disp_->valid();
    }
    
    void addKernel(const KernelInfo& k) { kernels_.push_back(k); }
    void setFuncIdMap(const std::unordered_map<std::string, FuncIdMapping>& m) { funcid_map_ = m; }
    
    bool link(const std::string& output_path) {
        printf("=== Pass 1: Collect and Size ===\n");
        if (!collectSections()) return false;
        
        printf("\n=== Pass 2: Layout ===\n");
        computeLayout();
        
        printf("\n=== Pass 3: Patch and Write ===\n");
        patchSections();
        
        return writeOutput(output_path);
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
    };
    std::vector<DebugLineChunk> debug_line_chunks_;
    std::vector<uint8_t> merged_debug_line_str_;  // Merged .debug_line_str data
    
    // Output sections
    std::vector<SectionInfo> sections_;
    uint32_t elf_flags_ = 0;
    
    // Resource maxima
    int max_vgpr_ = 0, max_sgpr_ = 0, max_lds_ = 0, max_stack_ = 0;
    
    // Function tables (text offsets per funcId)
    std::vector<uint64_t> table_1_, table_2_, table_4_;
    
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
        printf("Max resources: VGPR=%d, SGPR=%d, LDS=%d, Stack=%d\n",
               max_vgpr_, max_sgpr_, max_lds_, max_stack_);
        
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
            uint64_t rel_addr = text.data.size();  // Relative to .text start
            
            switch (unroll) {
                case 1: table_1_[funcid] = rel_addr; break;
                case 2: table_2_[funcid] = rel_addr; break;
                case 4: table_4_[funcid] = rel_addr; break;
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
                    orig_str_size            // orig_str_size
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
                        orig_str_size              // orig_str_size
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
            
            // Copy additional debug sections from dispatcher for source display
            // Dispatcher covers the Generic kernel code; specialized kernels have their own debug_line
            auto addDebugSection = [this](const char* name, const std::vector<uint8_t>& data, 
                                         uint64_t flags = 0, uint64_t entsize = 0) {
                if (data.empty()) return;
                SectionInfo sec;
                sec.name = name;
                sec.type = SHT_PROGBITS;
                sec.flags = flags;
                sec.alignment = 1;
                sec.entsize = entsize;
                sec.data = data;
                printf("Copied %s: %zu bytes\n", name, data.size());
                sections_.push_back(std::move(sec));
            };
            
            // Try to get debug sections from dispatcher first
            auto copyDebugFromElf = [&](ElfParser* elf, const char* source) {
                auto getSection = [elf](const char* name) -> std::vector<uint8_t> {
                    auto* sec = elf->find(name);
                    if (sec && sec->size > 0) return elf->getBytes(*sec);
                    return {};
                };
                
                auto abbrev = getSection(".debug_abbrev");
                auto info = getSection(".debug_info");
                auto str = getSection(".debug_str");
                auto str_off = getSection(".debug_str_offsets");
                auto addr = getSection(".debug_addr");
                auto rng = getSection(".debug_rnglists");
                
                if (!abbrev.empty() || !info.empty()) {
                    printf("Using debug sections from %s\n", source);
                    addDebugSection(".debug_abbrev", abbrev);
                    addDebugSection(".debug_info", info);
                    addDebugSection(".debug_str", str, SHF_MERGE | SHF_STRINGS, 1);
                    addDebugSection(".debug_str_offsets", str_off);
                    addDebugSection(".debug_addr", addr);
                    addDebugSection(".debug_rnglists", rng);
                    return true;
                }
                return false;
            };
            
            // Prefer dispatcher debug info (covers Generic kernels)
            if (!copyDebugFromElf(disp_.get(), "dispatcher")) {
                // Fall back to first specialized kernel
                if (!kernel_text_offsets_.empty()) {
                    const KernelInfo* first_kern = kernel_text_offsets_[0].first;
                    addDebugSection(".debug_abbrev", first_kern->debug_abbrev);
                    addDebugSection(".debug_info", first_kern->debug_info);
                    addDebugSection(".debug_str", first_kern->debug_str, SHF_MERGE | SHF_STRINGS, 1);
                    addDebugSection(".debug_str_offsets", first_kern->debug_str_offsets);
                    addDebugSection(".debug_addr", first_kern->debug_addr);
                    addDebugSection(".debug_rnglists", first_kern->debug_rnglists);
                }
            }
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
                       s.name.c_str(), s.addr, s.size(), s.alignment);
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
                Elf64_Sym sym = {};
                sym.st_name = name_off;
                sym.st_value = text_addr_ + text_off;
                sym.st_size = kern->code.size();
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
        
        // Calculate the address delta for dispatcher debug sections
        // Dispatcher's original .text address vs merged position
        auto* disp_text = disp_->find(".text");
        uint64_t disp_orig_text = disp_text ? disp_text->addr : 0;
        int64_t disp_delta = (int64_t)text_addr_ - (int64_t)disp_orig_text;
        
        // Patch .debug_info section - adjust DW_AT_low_pc/high_pc addresses
        // For DWARF4, these are typically 8-byte addresses at fixed positions
        SectionInfo* debug_info_sec = nullptr;
        for (auto& s : sections_) {
            if (s.name == ".debug_info") { debug_info_sec = &s; break; }
        }
        if (debug_info_sec && !debug_info_sec->data.empty() && disp_delta != 0) {
            // DWARF4 compile unit header: unit_length(4) + version(2) + abbr_offset(4) + addr_size(1) = 11 bytes
            // Then DIE entries. The DW_AT_low_pc is typically at offset 0x13 (19) for this format
            // DW_AT_high_pc follows (could be address or offset)
            // We'll scan for addresses in the dispatcher's original range and patch them
            uint8_t* data = debug_info_sec->data.data();
            size_t size = debug_info_sec->data.size();
            int info_patched = 0;
            
            // Scan for 8-byte values that look like addresses in the original dispatcher range
            auto* disp_text_sec = disp_->find(".text");
            uint64_t disp_text_start = disp_text_sec ? disp_text_sec->addr : 0;
            uint64_t disp_text_end = disp_text_start + (disp_text_sec ? disp_text_sec->size : 0);
            
            for (size_t off = 11; off + 8 <= size; off++) {
                uint64_t val;
                memcpy(&val, data + off, 8);
                // Check if this looks like an address in the original dispatcher .text range
                if (val >= disp_text_start && val < disp_text_end + 0x10000) {
                    uint64_t new_val = val + disp_delta;
                    memcpy(data + off, &new_val, 8);
                    info_patched++;
                    off += 7;  // Skip rest of this value
                }
            }
            if (info_patched > 0) {
                printf("  Patched %d addresses in .debug_info (delta=%+ld)\n", info_patched, disp_delta);
            }
        }
        
        // Also patch .debug_addr section if present
        // The addresses are relative to the first kernel's .text, need to adjust to merged .text
        SectionInfo* debug_addr_sec = nullptr;
        for (auto& s : sections_) {
            if (s.name == ".debug_addr") { debug_addr_sec = &s; break; }
        }
        if (debug_addr_sec && !debug_addr_sec->data.empty() && !debug_line_chunks_.empty()) {
            // Use the first kernel's original text address as the base
            uint64_t first_orig_text = debug_line_chunks_[0].orig_text_addr;
            uint64_t first_new_text = text_addr_ + debug_line_chunks_[0].new_text_offset;
            int64_t addr_delta = (int64_t)first_new_text - (int64_t)first_orig_text;
            
            // .debug_addr format: header (12 bytes for DWARF32) + array of 8-byte addresses
            // Header: length (4), version (2), address_size (1), segment_selector_size (1) = 8 bytes
            // Then addresses
            size_t header_size = 8;  // DWARF32 header
            int addr_patched = 0;
            for (size_t off = header_size; off + 8 <= debug_addr_sec->data.size(); off += 8) {
                uint64_t old_addr;
                memcpy(&old_addr, debug_addr_sec->data.data() + off, 8);
                uint64_t new_addr = old_addr + addr_delta;
                memcpy(debug_addr_sec->data.data() + off, &new_addr, 8);
                addr_patched++;
            }
            printf("  Patched %d addresses in .debug_addr (delta=%+ld)\n", addr_patched, addr_delta);
        }
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
        uint64_t relro_start = data_rel_ro_start;
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
    
    for (const auto& k : kernels) linker.addKernel(k);
    linker.setFuncIdMap(funcid_map);
    
    if (!linker.link(output)) return 1;
    
    return 0;
}
