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
    
    // Computed during layout
    uint64_t addr = 0;
    uint64_t offset = 0;
    
    size_t size() const { return data.size(); }
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
                shdrs[i].sh_entsize
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
    };
    
    const ParsedSection* find(const std::string& name) const {
        for (const auto& s : sections_) if (s.name == name) return &s;
        return nullptr;
    }
    
    std::vector<uint8_t> getBytes(const ParsedSection& s) const {
        const uint8_t* p = file_.at<uint8_t>(s.offset);
        return std::vector<uint8_t>(p, p + s.size);
    }

private:
    const MappedFile& file_;
    const Elf64_Ehdr* ehdr_;
    std::vector<ParsedSection> sections_;
};

// ============================================================================
// Kernel extraction utilities
// ============================================================================

std::string g_target_arch = "gfx942";

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
    
    // Parse .note for resources
    auto* note = elf.find(".note");
    if (note) {
        auto data = elf.getBytes(*note);
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
    std::unordered_map<std::string, FuncIdMapping> funcid_map_;
    
    // Output sections
    std::vector<SectionInfo> sections_;
    uint32_t elf_flags_ = 0;
    
    // Resource maxima
    int max_vgpr_ = 0, max_sgpr_ = 0, max_lds_ = 0, max_stack_ = 0;
    
    // Function tables
    std::vector<uint64_t> table_1_, table_2_, table_4_;
    
    // Addresses computed during layout
    uint64_t text_addr_ = 0;
    uint64_t data_addr_ = 0;
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
            
            text.data.insert(text.data.end(), k.code.begin(), k.code.end());
            while (text.data.size() % FUNC_ALIGNMENT != 0) text.data.push_back(0);
            mapped++;
        }
        printf("Mapped %d kernel functions, total .text size: %zu bytes\n", mapped, text.data.size());
        sections_.push_back(std::move(text));
        
        // Build .data (function tables)
        SectionInfo data;
        data.name = ".data";
        data.type = SHT_PROGBITS;
        data.flags = SHF_ALLOC | SHF_WRITE;
        data.alignment = 16;
        data.data.resize(table_spacing_ * 2 + FUNC_COUNT * 8, 0);
        // Will populate with absolute addresses after layout
        sections_.push_back(std::move(data));
        
        // Build .note (patched)
        auto* disp_note = disp_->find(".note");
        if (disp_note) {
            SectionInfo note;
            note.name = ".note";
            note.type = SHT_NOTE;
            note.flags = SHF_ALLOC;
            note.alignment = 4;
            
            auto orig = disp_->getBytes(*disp_note);
            patchNote(orig, note.data);
            sections_.push_back(std::move(note));
        }
        
        // Build .rodata (KDs - will patch later)
        auto* disp_rodata = disp_->find(".rodata");
        if (disp_rodata) {
            SectionInfo rodata;
            rodata.name = ".rodata";
            rodata.type = SHT_PROGBITS;
            rodata.flags = SHF_ALLOC;
            rodata.alignment = 64;
            rodata.data = disp_->getBytes(*disp_rodata);
            sections_.push_back(std::move(rodata));
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
        
        return true;
    }
    
    void patchNote(const std::vector<uint8_t>& orig, std::vector<uint8_t>& out) {
        // Patch private_segment_fixed_size: 24 -> max_stack_
        const char* target = ".private_segment_fixed_size";
        size_t tlen = strlen(target);
        
        std::vector<size_t> patches;
        for (size_t i = 0; i + 1 + tlen + 1 <= orig.size(); i++) {
            if (orig[i] == 0xbb && memcmp(orig.data() + i + 1, target, tlen) == 0 && orig[i + 1 + tlen] == 24) {
                patches.push_back(i + 1 + tlen);
            }
        }
        
        out.reserve(orig.size() + patches.size() * 2 + 4);
        size_t prev = 0;
        for (size_t p : patches) {
            out.insert(out.end(), orig.begin() + prev, orig.begin() + p);
            out.push_back(0xcd);
            out.push_back((max_stack_ >> 8) & 0xff);
            out.push_back(max_stack_ & 0xff);
            prev = p + 1;
            printf("  .note: private_segment_fixed_size 24 -> %d\n", max_stack_);
        }
        out.insert(out.end(), orig.begin() + prev, orig.end());
        
        // Update descsz and pad to alignment
        if (!patches.empty() && out.size() >= 12) {
            uint32_t namesz, old_descsz;
            memcpy(&namesz, out.data(), 4);
            memcpy(&old_descsz, out.data() + 4, 4);
            
            uint32_t new_descsz = old_descsz + patches.size() * 2;
            memcpy(out.data() + 4, &new_descsz, 4);
            
            // Calculate expected total size with alignment
            uint32_t namesz_aligned = (namesz + 3) & ~3;
            uint32_t old_descsz_aligned = (old_descsz + 3) & ~3;
            uint32_t new_descsz_aligned = (new_descsz + 3) & ~3;
            
            // Pad to match aligned size
            size_t expected_size = 12 + namesz_aligned + new_descsz_aligned;
            while (out.size() < expected_size) out.push_back(0);
            
            printf("  .note: descsz %u -> %u, size %zu -> %zu\n", 
                   old_descsz, new_descsz, orig.size(), out.size());
        }
    }
    
    // ========== Pass 2: Layout ==========
    void computeLayout() {
        // Layout order for AMDGPU code object:
        // [ELF header + phdrs] [.note] [.dynsym] [.gnu.hash] [.hash] [.dynstr] [.rodata] [.text] [.data]
        // Then non-alloc: [.symtab] [.strtab] [.dynamic] [.shstrtab] [section headers]
        
        const size_t ehdr_size = 64;
        const size_t phdr_size = 56;
        const size_t num_phdrs = 3;
        uint64_t addr = ehdr_size + num_phdrs * phdr_size;
        
        // Sort allocated sections in desired order
        auto order = [](const std::string& n) -> int {
            if (n == ".note") return 0;
            if (n == ".dynsym") return 1;
            if (n == ".gnu.hash") return 2;
            if (n == ".hash") return 3;
            if (n == ".dynstr") return 4;
            if (n == ".rodata") return 5;
            if (n == ".text") return 6;
            if (n == ".data") return 7;
            return 100;
        };
        
        std::sort(sections_.begin(), sections_.end(), [&](const SectionInfo& a, const SectionInfo& b) {
            bool a_alloc = a.isAlloc(), b_alloc = b.isAlloc();
            if (a_alloc != b_alloc) return a_alloc;  // Alloc sections first
            if (a_alloc) return order(a.name) < order(b.name);
            return false;
        });
        
        // Assign addresses to allocated sections
        for (auto& s : sections_) {
            if (!s.isAlloc()) continue;
            
            addr = (addr + s.alignment - 1) & ~(s.alignment - 1);
            s.addr = addr;
            s.offset = addr;  // For LOAD segment, offset == addr
            addr += s.size();
            
            if (s.name == ".text") text_addr_ = s.addr;
            if (s.name == ".data") data_addr_ = s.addr;
            
            printf("  %-12s @ 0x%06lx  size=0x%06zx  align=%lu\n",
                   s.name.c_str(), s.addr, s.size(), s.alignment);
        }
        
        // Non-allocated sections go at end
        for (auto& s : sections_) {
            if (s.isAlloc()) continue;
            addr = (addr + s.alignment - 1) & ~(s.alignment - 1);
            s.addr = 0;
            s.offset = addr;
            addr += s.size();
            printf("  %-12s @ offset 0x%06lx  size=0x%06zx (non-alloc)\n",
                   s.name.c_str(), s.offset, s.size());
        }
    }
    
    // ========== Pass 3: Patch ==========
    void patchSections() {
        // Patch .data with absolute function addresses
        SectionInfo* data_sec = nullptr;
        for (auto& s : sections_) if (s.name == ".data") { data_sec = &s; break; }
        
        if (data_sec) {
            printf("Populating function tables at 0x%lx\n", data_addr_);
            int populated = 0;
            for (int i = 0; i < FUNC_COUNT; i++) {
                uint64_t addr1 = table_1_[i] ? text_addr_ + table_1_[i] : 0;
                uint64_t addr2 = table_2_[i] ? text_addr_ + table_2_[i] : 0;
                uint64_t addr4 = table_4_[i] ? text_addr_ + table_4_[i] : 0;
                if (addr1 || addr2 || addr4) populated++;
                memcpy(data_sec->data.data() + i * 8, &addr1, 8);
                memcpy(data_sec->data.data() + table_spacing_ + i * 8, &addr2, 8);
                memcpy(data_sec->data.data() + table_spacing_ * 2 + i * 8, &addr4, 8);
            }
            printf("  Populated %d function entries\n", populated);
        }
        
        // Patch .text with PC-relative table references
        SectionInfo* text_sec = nullptr;
        for (auto& s : sections_) if (s.name == ".text") { text_sec = &s; break; }
        
        if (text_sec) {
            uint64_t table_addrs[3] = { data_addr_, data_addr_ + table_spacing_, data_addr_ + table_spacing_ * 2 };
            
            // Find s_getpc_b64 + s_add_u32 patterns
            uint32_t s_getpc = 0xBE801C00;
            uint32_t s_add = 0x8000FF00;
            
            auto* disp_bss = disp_->find(".bss");
            uint64_t old_data_addr = disp_bss ? disp_bss->addr : 0x8000;
            
            int patches = 0;
            for (size_t off = 0; off + 12 <= text_sec->data.size(); off += 4) {
                uint32_t i1, i2;
                memcpy(&i1, text_sec->data.data() + off, 4);
                memcpy(&i2, text_sec->data.data() + off + 4, 4);
                
                if (i1 == s_getpc && i2 == s_add) {
                    uint32_t old_lit;
                    memcpy(&old_lit, text_sec->data.data() + off + 8, 4);
                    
                    uint64_t pc = text_addr_ + off + 4;
                    uint64_t old_target = pc + old_lit;
                    
                    int idx = -1;
                    if (old_target >= old_data_addr && old_target < old_data_addr + 0x6000) {
                        uint64_t rel = old_target - old_data_addr;
                        if (rel < table_spacing_) idx = 0;
                        else if (rel < table_spacing_ * 2) idx = 1;
                        else idx = 2;
                    }
                    
                    if (idx >= 0) {
                        uint32_t new_lit = (uint32_t)(table_addrs[idx] - pc);
                        memcpy(text_sec->data.data() + off + 8, &new_lit, 4);
                        printf("  Patched table ref: PC=0x%lx, table %d -> 0x%lx\n", pc, idx, table_addrs[idx]);
                        patches++;
                    }
                }
            }
            printf("Patched %d PC-relative table references\n", patches);
        }
        
        // Patch .rodata (KDs)
        SectionInfo* rodata_sec = nullptr;
        for (auto& s : sections_) if (s.name == ".rodata") { rodata_sec = &s; break; }
        
        if (rodata_sec) {
            for (size_t kd_off : {0UL, 64UL, 128UL}) {
                if (kd_off + 64 > rodata_sec->data.size()) continue;
                
                uint8_t* kd = rodata_sec->data.data() + kd_off;
                
                // LDS and stack
                uint32_t lds = max_lds_, stack = max_stack_;
                memcpy(kd + 0, &lds, 4);
                memcpy(kd + 4, &stack, 4);
                
                // RSRC1: VGPR/SGPR
                uint32_t rsrc1;
                memcpy(&rsrc1, kd + 0x34, 4);
                int vgpr_g = (max_vgpr_ + 3) / 4 - 1;
                int sgpr_g = (max_sgpr_ + 7) / 8 - 1;
                rsrc1 = (rsrc1 & ~0x3FF) | (vgpr_g & 0x3F) | ((sgpr_g & 0xF) << 6);
                memcpy(kd + 0x34, &rsrc1, 4);
                
                // RSRC2: SCRATCH_EN, USER_SGPR
                uint32_t rsrc2;
                memcpy(&rsrc2, kd + 0x38, 4);
                if (stack > 0) rsrc2 |= 1;
                rsrc2 = (rsrc2 & ~(0x1f << 1)) | (10 << 1);
                memcpy(kd + 0x38, &rsrc2, 4);
                
                // CODE_PROPERTIES
                uint16_t props = 0x001e;
                memcpy(kd + 0x3c, &props, 2);
                
                printf("  KD[%zu]: LDS=%d, stack=%d, RSRC2=0x%08x, props=0x%04x\n",
                       kd_off / 64, lds, stack, rsrc2, props);
            }
        }
        
        // Patch .dynsym symbol values
        SectionInfo* dynsym_sec = nullptr;
        for (auto& s : sections_) if (s.name == ".dynsym") { dynsym_sec = &s; break; }
        
        if (dynsym_sec) {
            auto* disp_bss = disp_->find(".bss");
            uint64_t old_data = disp_bss ? disp_bss->addr : 0x8000;
            uint64_t delta = data_addr_ - old_data;
            
            for (size_t i = 0; i + 24 <= dynsym_sec->data.size(); i += 24) {
                uint64_t val;
                memcpy(&val, dynsym_sec->data.data() + i + 8, 8);
                if (val >= old_data && val < old_data + 0x10000) {
                    val += delta;
                    memcpy(dynsym_sec->data.data() + i + 8, &val, 8);
                }
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
        
        uint32_t dyn_name_off = shstrtab.size();
        const char* dyn_name = ".dynamic";
        shstrtab.insert(shstrtab.end(), dyn_name, dyn_name + strlen(dyn_name) + 1);
        
        uint32_t shstr_name_off = shstrtab.size();
        const char* shstr_name = ".shstrtab";
        shstrtab.insert(shstrtab.end(), shstr_name, shstr_name + strlen(shstr_name) + 1);
        
        // Find max allocated offset for non-alloc placement
        uint64_t offset = 0;
        for (const auto& s : sections_) {
            if (s.isAlloc()) offset = std::max(offset, s.offset + s.size());
        }
        
        // Place non-alloc sections
        std::vector<uint64_t> section_offsets;
        for (const auto& s : sections_) {
            if (s.isAlloc()) {
                section_offsets.push_back(s.offset);
            } else {
                offset = (offset + s.alignment - 1) & ~(s.alignment - 1);
                section_offsets.push_back(offset);
                offset += s.size();
            }
        }
        
        // .dynamic
        offset = (offset + 7) & ~7UL;
        uint64_t dyn_off = offset;
        std::vector<uint8_t> dyn_data;
        auto addDyn = [&](uint64_t tag, uint64_t val) {
            size_t p = dyn_data.size();
            dyn_data.resize(p + 16);
            memcpy(dyn_data.data() + p, &tag, 8);
            memcpy(dyn_data.data() + p + 8, &val, 8);
        };
        
        // Find section addresses for dynamic entries
        uint64_t dynsym_addr = 0, dynstr_addr = 0, dynstr_sz = 0, gnuhash_addr = 0, hash_addr = 0;
        for (const auto& s : sections_) {
            if (s.name == ".dynsym") dynsym_addr = s.addr;
            else if (s.name == ".dynstr") { dynstr_addr = s.addr; dynstr_sz = s.size(); }
            else if (s.name == ".gnu.hash") gnuhash_addr = s.addr;
            else if (s.name == ".hash") hash_addr = s.addr;
        }
        
        if (dynsym_addr) addDyn(6, dynsym_addr);    // DT_SYMTAB
        addDyn(11, 24);                              // DT_SYMENT
        if (dynstr_addr) addDyn(5, dynstr_addr);    // DT_STRTAB
        if (dynstr_sz) addDyn(10, dynstr_sz);       // DT_STRSZ
        if (gnuhash_addr) addDyn(0x6ffffef5, gnuhash_addr);
        if (hash_addr) addDyn(4, hash_addr);
        addDyn(0, 0);  // DT_NULL
        offset += dyn_data.size();
        
        // .shstrtab
        uint64_t shstrtab_off = offset;
        offset += shstrtab.size();
        
        // Section headers
        offset = (offset + 7) & ~7UL;
        uint64_t shdr_off = offset;
        size_t num_shdrs = sections_.size() + 3;  // NULL + sections + .dynamic + .shstrtab
        
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
        ehdr.e_phnum = 3;
        ehdr.e_shentsize = 64;
        ehdr.e_shnum = num_shdrs;
        ehdr.e_shstrndx = num_shdrs - 1;
        memcpy(out.data(), &ehdr, sizeof(ehdr));
        
        // Program headers
        size_t phdr_off = 64;
        
        Elf64_Phdr phdr_phdr = {};
        phdr_phdr.p_type = PT_PHDR;
        phdr_phdr.p_flags = PF_R;
        phdr_phdr.p_offset = phdr_phdr.p_vaddr = phdr_phdr.p_paddr = 64;
        phdr_phdr.p_filesz = phdr_phdr.p_memsz = 3 * 56;
        phdr_phdr.p_align = 8;
        memcpy(out.data() + phdr_off, &phdr_phdr, sizeof(phdr_phdr));
        phdr_off += 56;
        
        // Find max alloc end for LOAD segment - must include .dynamic!
        uint64_t load_end = dyn_off + dyn_data.size();  // Include .dynamic in LOAD
        
        Elf64_Phdr phdr_load = {};
        phdr_load.p_type = PT_LOAD;
        phdr_load.p_flags = PF_R | PF_W | PF_X;
        phdr_load.p_offset = phdr_load.p_vaddr = phdr_load.p_paddr = 0;
        phdr_load.p_filesz = phdr_load.p_memsz = load_end;
        phdr_load.p_align = 0x1000;
        memcpy(out.data() + phdr_off, &phdr_load, sizeof(phdr_load));
        phdr_off += 56;
        
        Elf64_Phdr phdr_dyn = {};
        phdr_dyn.p_type = PT_DYNAMIC;
        phdr_dyn.p_flags = PF_R | PF_W;
        phdr_dyn.p_offset = phdr_dyn.p_vaddr = phdr_dyn.p_paddr = dyn_off;
        phdr_dyn.p_filesz = phdr_dyn.p_memsz = dyn_data.size();
        phdr_dyn.p_align = 8;
        memcpy(out.data() + phdr_off, &phdr_dyn, sizeof(phdr_dyn));
        
        // Section data
        for (size_t i = 0; i < sections_.size(); i++) {
            memcpy(out.data() + section_offsets[i], sections_[i].data.data(), sections_[i].data.size());
        }
        memcpy(out.data() + dyn_off, dyn_data.data(), dyn_data.size());
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
            else if (s.name == ".symtab" && strtab_idx >= 0) { link = strtab_idx; info = 26; }
            
            writeShdr(i + 1, name_offs[i + 1], s.type, s.flags,
                      s.addr, section_offsets[i], s.size(),
                      link, info, s.alignment, s.entsize);
        }
        
        size_t dyn_idx = sections_.size() + 1;
        writeShdr(dyn_idx, dyn_name_off, SHT_DYNAMIC, SHF_ALLOC | SHF_WRITE,
                  dyn_off, dyn_off, dyn_data.size(), dynstr_idx > 0 ? dynstr_idx : 0, 0, 8, 16);
        
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
    
    printf("Device Linker: target=%s\n", g_target_arch.c_str());
    
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
