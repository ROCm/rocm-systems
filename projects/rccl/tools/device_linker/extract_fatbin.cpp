/*
 * Extract and parse .hip_fatbin section from a shared library.
 *
 * Usage: extract_fatbin <librccl.so> [output_dir]
 *
 * The .hip_fatbin section may contain multiple components:
 * - CLANG_OFFLOAD_BUNDLE: Standard offload bundle with embedded device ELFs
 * - CCOB: Compressed Code OBject (used by --offload-compress)
 *
 * Each component is written to a separate file in output_dir.
 *
 * Build: g++ -O2 -std=c++17 -o extract_fatbin extract_fatbin.cpp
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <elf.h>

namespace fs = std::filesystem;

static const char CLANG_BUNDLE_MAGIC[] = "__CLANG_OFFLOAD_BUNDLE__";
static const char CCOB_MAGIC[] = "CCOB";

struct BundleEntry {
    std::string name;
    uint64_t offset;
    uint64_t size;
};

struct Component {
    std::string type;
    uint64_t offset;
    uint64_t size;
    // For CLANG_OFFLOAD_BUNDLE
    std::vector<BundleEntry> entries;
    // For CCOB
    uint16_t version;
    uint16_t flags;
};

// Read file into memory
std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    
    size_t size = file.tellg();
    file.seekg(0);
    
    std::vector<uint8_t> data(size);
    file.read(reinterpret_cast<char*>(data.data()), size);
    
    if (!file) {
        throw std::runtime_error("Failed to read file: " + path);
    }
    
    return data;
}

// Write data to file
void write_file(const std::string& path, const uint8_t* data, size_t size) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot create file: " + path);
    }
    file.write(reinterpret_cast<const char*>(data), size);
}

// Extract .hip_fatbin section from ELF
std::vector<uint8_t> extract_hip_fatbin(const std::vector<uint8_t>& elf_data) {
    if (elf_data.size() < sizeof(Elf64_Ehdr)) {
        throw std::runtime_error("File too small for ELF header");
    }
    
    const auto* ehdr = reinterpret_cast<const Elf64_Ehdr*>(elf_data.data());
    
    if (memcmp(ehdr->e_ident, ELFMAG, SELFMAG) != 0) {
        throw std::runtime_error("Not an ELF file");
    }
    
    if (ehdr->e_ident[EI_CLASS] != ELFCLASS64) {
        throw std::runtime_error("Not a 64-bit ELF");
    }
    
    // Find section header string table
    if (ehdr->e_shstrndx == SHN_UNDEF || ehdr->e_shoff == 0) {
        throw std::runtime_error("No section header string table");
    }
    
    const auto* shdrs = reinterpret_cast<const Elf64_Shdr*>(elf_data.data() + ehdr->e_shoff);
    const auto& shstrtab_hdr = shdrs[ehdr->e_shstrndx];
    const char* shstrtab = reinterpret_cast<const char*>(elf_data.data() + shstrtab_hdr.sh_offset);
    
    // Find .hip_fatbin section
    for (uint16_t i = 0; i < ehdr->e_shnum; i++) {
        const char* name = shstrtab + shdrs[i].sh_name;
        if (strcmp(name, ".hip_fatbin") == 0) {
            uint64_t offset = shdrs[i].sh_offset;
            uint64_t size = shdrs[i].sh_size;
            
            if (offset + size > elf_data.size()) {
                throw std::runtime_error(".hip_fatbin section extends past end of file");
            }
            
            return std::vector<uint8_t>(
                elf_data.data() + offset,
                elf_data.data() + offset + size
            );
        }
    }
    
    throw std::runtime_error("No .hip_fatbin section found");
}

// Parse CLANG_OFFLOAD_BUNDLE at given offset
bool parse_clang_bundle(const std::vector<uint8_t>& data, size_t offset, Component& comp) {
    if (offset + 24 > data.size()) return false;
    if (memcmp(data.data() + offset, CLANG_BUNDLE_MAGIC, 24) != 0) return false;
    
    comp.type = "CLANG_OFFLOAD_BUNDLE";
    comp.offset = offset;
    
    size_t pos = offset + 24;
    
    if (pos + 8 > data.size()) return false;
    uint64_t num_entries;
    memcpy(&num_entries, data.data() + pos, 8);
    pos += 8;
    
    uint64_t max_end = pos;
    
    for (uint64_t i = 0; i < num_entries; i++) {
        if (pos + 24 > data.size()) return false;
        
        BundleEntry entry;
        memcpy(&entry.offset, data.data() + pos, 8);
        pos += 8;
        memcpy(&entry.size, data.data() + pos, 8);
        pos += 8;
        
        uint64_t name_len;
        memcpy(&name_len, data.data() + pos, 8);
        pos += 8;
        
        if (pos + name_len > data.size()) return false;
        entry.name = std::string(reinterpret_cast<const char*>(data.data() + pos), name_len);
        // Remove null terminator if present
        while (!entry.name.empty() && entry.name.back() == '\0') {
            entry.name.pop_back();
        }
        pos += name_len;
        
        comp.entries.push_back(entry);
        
        uint64_t entry_end = entry.offset + entry.size;
        if (entry_end > max_end) max_end = entry_end;
    }
    
    comp.size = max_end;
    return true;
}

// Parse CCOB at given offset
bool parse_ccob(const std::vector<uint8_t>& data, size_t offset, Component& comp) {
    if (offset + 8 > data.size()) return false;
    if (memcmp(data.data() + offset, CCOB_MAGIC, 4) != 0) return false;
    
    comp.type = "CCOB";
    comp.offset = offset;
    
    memcpy(&comp.version, data.data() + offset + 4, 2);
    memcpy(&comp.flags, data.data() + offset + 6, 2);
    
    // Find end by looking for next magic or EOF
    size_t pos = offset + 8;
    size_t end_pos = data.size();
    
    while (pos < data.size() - 4) {
        if (memcmp(data.data() + pos, CCOB_MAGIC, 4) == 0) {
            end_pos = pos;
            break;
        }
        if (pos + 24 <= data.size() && 
            memcmp(data.data() + pos, CLANG_BUNDLE_MAGIC, 24) == 0) {
            end_pos = pos;
            break;
        }
        pos++;
    }
    
    comp.size = end_pos - offset;
    return true;
}

// Parse entire fatbin
std::vector<Component> parse_fatbin(const std::vector<uint8_t>& data) {
    std::vector<Component> components;
    size_t pos = 0;
    
    while (pos < data.size()) {
        // Skip zero padding
        while (pos < data.size() && data[pos] == 0) {
            pos++;
        }
        
        if (pos >= data.size()) break;
        
        Component comp;
        
        // Try CLANG_OFFLOAD_BUNDLE
        if (parse_clang_bundle(data, pos, comp)) {
            components.push_back(comp);
            pos = comp.offset + comp.size;
            continue;
        }
        
        // Try CCOB
        if (parse_ccob(data, pos, comp)) {
            components.push_back(comp);
            pos = comp.offset + comp.size;
            continue;
        }
        
        // Unknown format
        fprintf(stderr, "Error: Unknown format at offset 0x%zx: ", pos);
        for (size_t i = 0; i < 20 && pos + i < data.size(); i++) {
            fprintf(stderr, "%02x", data[pos + i]);
        }
        fprintf(stderr, "...\n");
        throw std::runtime_error("Unknown format in fatbin");
    }
    
    return components;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <librccl.so> [output_dir]\n", argv[0]);
        return 1;
    }
    
    std::string so_path = argv[1];
    std::string output_dir = argc > 2 ? argv[2] : "fatbin_components";
    
    try {
        printf("Extracting .hip_fatbin from %s...\n", so_path.c_str());
        
        // Read ELF file
        auto elf_data = read_file(so_path);
        printf("  Read %zu bytes (%.2f MB)\n", elf_data.size(), elf_data.size() / (1024.0 * 1024.0));
        
        // Extract .hip_fatbin section
        auto fatbin_data = extract_hip_fatbin(elf_data);
        printf("  Extracted .hip_fatbin: %zu bytes (%.2f MB)\n", 
               fatbin_data.size(), fatbin_data.size() / (1024.0 * 1024.0));
        
        // Parse components
        printf("\nParsing .hip_fatbin...\n");
        auto components = parse_fatbin(fatbin_data);
        
        printf("\nFound %zu component(s):\n", components.size());
        
        for (size_t i = 0; i < components.size(); i++) {
            const auto& comp = components[i];
            printf("\n  [%zu] %s\n", i, comp.type.c_str());
            printf("      Offset: 0x%lx\n", (unsigned long)comp.offset);
            printf("      Size: %lu bytes (%.2f MB)\n", 
                   (unsigned long)comp.size, comp.size / (1024.0 * 1024.0));
            
            if (comp.type == "CLANG_OFFLOAD_BUNDLE") {
                printf("      Entries: %zu\n", comp.entries.size());
                for (const auto& entry : comp.entries) {
                    printf("        - %s: offset=0x%lx, size=%lu\n",
                           entry.name.c_str(), (unsigned long)entry.offset, (unsigned long)entry.size);
                }
            } else if (comp.type == "CCOB") {
                printf("      Version: %u\n", comp.version);
                printf("      Flags: 0x%x\n", comp.flags);
            }
        }
        
        // Check for unconsumed data
        if (!components.empty()) {
            uint64_t last_end = 0;
            for (const auto& c : components) {
                uint64_t end = c.offset + c.size;
                if (end > last_end) last_end = end;
            }
            
            if (last_end < fatbin_data.size()) {
                size_t remaining = fatbin_data.size() - last_end;
                size_t non_zero = 0;
                for (size_t i = last_end; i < fatbin_data.size(); i++) {
                    if (fatbin_data[i] != 0) non_zero++;
                }
                
                if (non_zero > 0) {
                    printf("\nWarning: %zu bytes after last component (%zu non-zero)\n", remaining, non_zero);
                } else {
                    printf("\nNote: %zu bytes of zero padding after last component\n", remaining);
                }
            }
        }
        
        // Write components
        printf("\nWriting components to %s/...\n", output_dir.c_str());
        fs::create_directories(output_dir);
        
        for (size_t i = 0; i < components.size(); i++) {
            const auto& comp = components[i];
            std::string filename;
            
            if (comp.type == "CLANG_OFFLOAD_BUNDLE") {
                filename = "bundle_" + std::to_string(i) + ".bin";
            } else if (comp.type == "CCOB") {
                filename = "ccob_" + std::to_string(i) + ".bin";
            } else {
                filename = "unknown_" + std::to_string(i) + ".bin";
            }
            
            std::string filepath = output_dir + "/" + filename;
            write_file(filepath, fatbin_data.data() + comp.offset, comp.size);
            printf("  Wrote %s (%lu bytes)\n", filepath.c_str(), (unsigned long)comp.size);
        }
        
        printf("\nDone!\n");
        return 0;
        
    } catch (const std::exception& e) {
        fprintf(stderr, "\nError: %s\n", e.what());
        return 1;
    }
}
