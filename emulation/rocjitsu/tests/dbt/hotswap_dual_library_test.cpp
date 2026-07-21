// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hotswap_dual_library_test.cpp
/// @brief Loads and exercises the simulator and HotSwap DSOs in one process.

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/vm/rj_vm.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct ComgrData {
  uint64_t handle = 0;
};

constexpr int kComgrSuccess = 0;
constexpr int kComgrExecutable = 0x8;
constexpr std::string_view kGfx1250B0Isa = "amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+";
constexpr std::string_view kGfx1250A0Isa = "amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific-";

class SharedObject {
public:
  explicit SharedObject(const char *path)
      : path_(path), handle_(::dlopen(path, RTLD_NOW | RTLD_LOCAL)) {
    if (handle_ == nullptr)
      throw std::runtime_error("dlopen(" + path_ + ") failed: " + ::dlerror());
  }

  SharedObject(const SharedObject &) = delete;
  SharedObject &operator=(const SharedObject &) = delete;

  ~SharedObject() {
    if (handle_ != nullptr)
      (void)::dlclose(handle_);
  }

  template <typename Function> [[nodiscard]] Function symbol(const char *name) const {
    ::dlerror();
    void *address = ::dlsym(handle_, name);
    const char *error = ::dlerror();
    if (error != nullptr)
      throw std::runtime_error("dlsym(" + path_ + ", " + name + ") failed: " + error);

    static_assert(sizeof(Function) == sizeof(address));
    Function function;
    std::memcpy(&function, &address, sizeof(function));
    return function;
  }

private:
  std::string path_;
  void *handle_ = nullptr;
};

[[nodiscard]] uint64_t align_up(uint64_t value, uint64_t alignment) {
  const uint64_t remainder = value % alignment;
  return remainder == 0 ? value : value + alignment - remainder;
}

[[nodiscard]] uint32_t add_name(std::vector<uint8_t> &table, std::string_view name) {
  const uint32_t offset = static_cast<uint32_t>(table.size());
  table.insert(table.end(), name.begin(), name.end());
  table.push_back('\0');
  return offset;
}

/// Build a tiny, valid gfx1250 HSA code object without requiring a gfx1250
/// assembler in the build environment. The s_clause instruction is a B0
/// erratum site and must be rewritten to s_nop for A0.
[[nodiscard]] std::vector<uint8_t> make_gfx1250_b0_code_object() {
  using namespace rocjitsu;

  constexpr uint64_t kTextOffset = 0x100;
  constexpr uint64_t kTextVaddr = 0x1100;
  constexpr uint64_t kLoadAlign = 0x1000;
  constexpr uint64_t kDescriptorSize = 64;
  constexpr uint64_t kDescriptorEntryOffset = 16;
  constexpr std::array<uint32_t, 2> kText = {
      0xBF850004u, // s_clause 4
      0xBFB00000u, // s_endpgm
  };
  constexpr uint64_t kTextSize = kText.size() * sizeof(uint32_t);

  std::vector<uint8_t> section_names{'\0'};
  const uint32_t text_name = add_name(section_names, ".text");
  const uint32_t rodata_name = add_name(section_names, ".rodata");
  const uint32_t symtab_name = add_name(section_names, ".symtab");
  const uint32_t strtab_name = add_name(section_names, ".strtab");
  const uint32_t shstrtab_name = add_name(section_names, ".shstrtab");

  std::vector<uint8_t> symbol_names{'\0'};
  const uint32_t kernel_symbol_name = add_name(symbol_names, "kernel.kd");

  const uint64_t rodata_offset = kTextOffset + kTextSize;
  const uint64_t rodata_vaddr = kTextVaddr + kTextSize + kLoadAlign;
  const uint64_t strtab_offset = rodata_offset + kDescriptorSize;
  const uint64_t symtab_offset = align_up(strtab_offset + symbol_names.size(), 8);
  constexpr size_t kSymbolCount = 2;
  const uint64_t shstrtab_offset = symtab_offset + kSymbolCount * sizeof(Elf64_Sym);
  const uint64_t section_header_offset = align_up(shstrtab_offset + section_names.size(), 8);
  constexpr uint16_t kSectionCount = 6;
  constexpr uint16_t kProgramHeaderCount = 2;

  std::vector<uint8_t> image(section_header_offset + kSectionCount * sizeof(Elf64_Shdr), 0);

  Elf64_Ehdr header{};
  std::memcpy(header.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  header.e_ident[EI_CLASS] = ELFCLASS64;
  header.e_ident[EI_DATA] = 1;    // ELFDATA2LSB
  header.e_ident[EI_VERSION] = 1; // EV_CURRENT
  header.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  header.e_type = ET_DYN;
  header.e_machine = EM_AMDGPU;
  header.e_version = 1;
  header.e_phoff = sizeof(Elf64_Ehdr);
  header.e_shoff = section_header_offset;
  header.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX1250;
  header.e_ehsize = sizeof(Elf64_Ehdr);
  header.e_phentsize = sizeof(Elf64_Phdr);
  header.e_phnum = kProgramHeaderCount;
  header.e_shentsize = sizeof(Elf64_Shdr);
  header.e_shnum = kSectionCount;
  header.e_shstrndx = 5;
  std::memcpy(image.data(), &header, sizeof(header));

  std::array<Elf64_Phdr, kProgramHeaderCount> program_headers{};
  program_headers[0].p_type = PT_LOAD;
  program_headers[0].p_flags = 0x5; // PF_R | PF_X
  program_headers[0].p_offset = kTextOffset;
  program_headers[0].p_vaddr = kTextVaddr;
  program_headers[0].p_paddr = kTextVaddr;
  program_headers[0].p_filesz = kTextSize;
  program_headers[0].p_memsz = kTextSize;
  program_headers[0].p_align = kLoadAlign;
  program_headers[1].p_type = PT_LOAD;
  program_headers[1].p_flags = 0x4; // PF_R
  program_headers[1].p_offset = rodata_offset;
  program_headers[1].p_vaddr = rodata_vaddr;
  program_headers[1].p_paddr = rodata_vaddr;
  program_headers[1].p_filesz = kDescriptorSize;
  program_headers[1].p_memsz = kDescriptorSize;
  program_headers[1].p_align = kLoadAlign;
  std::memcpy(image.data() + header.e_phoff, program_headers.data(), sizeof(program_headers));

  std::memcpy(image.data() + kTextOffset, kText.data(), kTextSize);
  const int64_t entry_offset =
      static_cast<int64_t>(kTextVaddr) - static_cast<int64_t>(rodata_vaddr);
  std::memcpy(image.data() + rodata_offset + kDescriptorEntryOffset, &entry_offset,
              sizeof(entry_offset));
  std::memcpy(image.data() + strtab_offset, symbol_names.data(), symbol_names.size());

  std::array<Elf64_Sym, kSymbolCount> symbols{};
  symbols[1].st_name = kernel_symbol_name;
  symbols[1].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject);
  symbols[1].st_shndx = 2;
  symbols[1].st_value = rodata_vaddr;
  symbols[1].st_size = kDescriptorSize;
  std::memcpy(image.data() + symtab_offset, symbols.data(), sizeof(symbols));
  std::memcpy(image.data() + shstrtab_offset, section_names.data(), section_names.size());

  std::array<Elf64_Shdr, kSectionCount> section_headers{};
  section_headers[1].sh_name = text_name;
  section_headers[1].sh_type = SHT_PROGBITS;
  section_headers[1].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  section_headers[1].sh_addr = kTextVaddr;
  section_headers[1].sh_offset = kTextOffset;
  section_headers[1].sh_size = kTextSize;
  section_headers[1].sh_addralign = sizeof(uint32_t);
  section_headers[2].sh_name = rodata_name;
  section_headers[2].sh_type = SHT_PROGBITS;
  section_headers[2].sh_flags = SHF_ALLOC;
  section_headers[2].sh_addr = rodata_vaddr;
  section_headers[2].sh_offset = rodata_offset;
  section_headers[2].sh_size = kDescriptorSize;
  section_headers[2].sh_addralign = 64;
  section_headers[3].sh_name = symtab_name;
  section_headers[3].sh_type = SHT_SYMTAB;
  section_headers[3].sh_offset = symtab_offset;
  section_headers[3].sh_size = sizeof(symbols);
  section_headers[3].sh_link = 4;
  section_headers[3].sh_info = 1;
  section_headers[3].sh_addralign = 8;
  section_headers[3].sh_entsize = sizeof(Elf64_Sym);
  section_headers[4].sh_name = strtab_name;
  section_headers[4].sh_type = SHT_STRTAB;
  section_headers[4].sh_offset = strtab_offset;
  section_headers[4].sh_size = symbol_names.size();
  section_headers[4].sh_addralign = 1;
  section_headers[5].sh_name = shstrtab_name;
  section_headers[5].sh_type = SHT_STRTAB;
  section_headers[5].sh_offset = shstrtab_offset;
  section_headers[5].sh_size = section_names.size();
  section_headers[5].sh_addralign = 1;
  std::memcpy(image.data() + section_header_offset, section_headers.data(),
              sizeof(section_headers));
  return image;
}

[[nodiscard]] std::optional<uint32_t> first_executable_word(const std::vector<uint8_t> &image) {
  using namespace rocjitsu;
  if (image.size() < sizeof(Elf64_Ehdr))
    return std::nullopt;
  Elf64_Ehdr header{};
  std::memcpy(&header, image.data(), sizeof(header));
  if (header.e_shentsize != sizeof(Elf64_Shdr) || header.e_shoff > image.size() ||
      header.e_shnum > (image.size() - header.e_shoff) / sizeof(Elf64_Shdr)) {
    return std::nullopt;
  }

  for (uint16_t i = 0; i < header.e_shnum; ++i) {
    Elf64_Shdr section{};
    std::memcpy(&section, image.data() + header.e_shoff + i * sizeof(section), sizeof(section));
    if (section.sh_type != SHT_PROGBITS || (section.sh_flags & SHF_EXECINSTR) == 0 ||
        section.sh_offset > image.size() || sizeof(uint32_t) > image.size() - section.sh_offset) {
      continue;
    }
    uint32_t word = 0;
    std::memcpy(&word, image.data() + section.sh_offset, sizeof(word));
    return word;
  }
  return std::nullopt;
}

void exercise_simulator(const SharedObject &simulator) {
  using Create = rj_status_t (*)(const char *, rj_vm_mode_t, rj_vm_t **);
  using Run = rj_status_t (*)(rj_vm_t *, uint64_t *);
  using Destroy = void (*)(rj_vm_t *);

  const auto create = simulator.symbol<Create>("rj_vm_create_from_string");
  const auto run = simulator.symbol<Run>("rj_vm_run");
  const auto destroy = simulator.symbol<Destroy>("rj_vm_destroy");
  constexpr const char *kConfig = R"({
    "max_ticks": 10000,
    "num_threads": 1,
    "vm": {"arch": "gfx1250"},
    "topology": {
      "root": {"name": "soc", "type": "soc", "children": [
        {"name": "vram", "type": "gpu_memory"},
        {"name": "xcd0", "type": "xcd", "children": [
          {"name": "l2", "type": "l2_cache"},
          {"name": "cp", "type": "command_processor"},
          {"name": "se0", "type": "shader_engine", "children": [
            {"name": "cu[0:1]", "type": "compute_unit", "config": [
              {"key": "num_wf_slots", "value": "10"},
              {"key": "sgprs_per_wf", "value": "104"},
              {"key": "vgprs_per_wf", "value": "256"},
              {"key": "lds_size_kb", "value": "64"}
            ]}
          ]}
        ]}
      ]},
      "links": [
        {"src": "xcd0.cp.req_0", "dst": "xcd0.se0.cu0.cpl", "latency": 1, "weight": 2},
        {"src": "xcd0.se0.cu0.req", "dst": "xcd0.l2.cpl_0", "latency": 1, "weight": 10}
      ]
    }
  })";

  rj_vm_t *vm = nullptr;
  if (create(kConfig, RJ_VM_MODE_DEFAULT, &vm) != ROCJITSU_STATUS_SUCCESS || vm == nullptr)
    throw std::runtime_error("the simulator DSO could not create a gfx1250 VM");
  uint64_t ticks = 0;
  const rj_status_t status = run(vm, &ticks);
  destroy(vm);
  if (status != ROCJITSU_STATUS_SUCCESS)
    throw std::runtime_error("the simulator DSO could not run the gfx1250 VM");
}

void exercise_hotswap(const SharedObject &hotswap) {
  using Create = int (*)(int, ComgrData *);
  using Release = int (*)(ComgrData);
  using Set = int (*)(ComgrData, size_t, const char *);
  using Get = int (*)(ComgrData, size_t *, char *);
  using GetIsa = int (*)(ComgrData, size_t *, char *);
  using Rewrite = int (*)(ComgrData, const char *, const char *, ComgrData *);

  const auto create = hotswap.symbol<Create>("amd_comgr_create_data");
  const auto release = hotswap.symbol<Release>("amd_comgr_release_data");
  const auto set = hotswap.symbol<Set>("amd_comgr_set_data");
  const auto get = hotswap.symbol<Get>("amd_comgr_get_data");
  const auto get_isa = hotswap.symbol<GetIsa>("amd_comgr_get_data_isa_name");
  const auto rewrite = hotswap.symbol<Rewrite>("amd_comgr_hotswap_rewrite");

  const std::vector<uint8_t> input_bytes = make_gfx1250_b0_code_object();
  ComgrData input{};
  if (create(kComgrExecutable, &input) != kComgrSuccess)
    throw std::runtime_error("amd_comgr_create_data failed for the input");

  ComgrData output{};
  const auto release_data = [&](ComgrData data) {
    if (data.handle != 0)
      (void)release(data);
  };

  if (set(input, input_bytes.size(), reinterpret_cast<const char *>(input_bytes.data())) !=
      kComgrSuccess) {
    release_data(input);
    throw std::runtime_error("amd_comgr_set_data failed");
  }
  if (rewrite(input, kGfx1250B0Isa.data(), kGfx1250A0Isa.data(), &output) != kComgrSuccess) {
    release_data(input);
    throw std::runtime_error("the HotSwap DSO could not rewrite gfx1250 B0 to A0");
  }

  size_t output_size = 0;
  if (get(output, &output_size, nullptr) != kComgrSuccess || output_size == 0) {
    release_data(output);
    release_data(input);
    throw std::runtime_error("amd_comgr_get_data could not size the rewritten code object");
  }
  std::vector<uint8_t> output_bytes(output_size);
  if (get(output, &output_size, reinterpret_cast<char *>(output_bytes.data())) != kComgrSuccess) {
    release_data(output);
    release_data(input);
    throw std::runtime_error("amd_comgr_get_data could not read the rewritten code object");
  }

  size_t isa_size = 0;
  if (get_isa(output, &isa_size, nullptr) != kComgrSuccess || isa_size == 0) {
    release_data(output);
    release_data(input);
    throw std::runtime_error("the rewritten object has no gfx1250 ISA name");
  }
  std::string isa_name(isa_size, '\0');
  if (get_isa(output, &isa_size, isa_name.data()) != kComgrSuccess ||
      isa_name.find("gfx1250") == std::string::npos) {
    release_data(output);
    release_data(input);
    throw std::runtime_error("the rewritten object is not identified as gfx1250");
  }

  release_data(output);
  release_data(input);
  if (output_bytes == input_bytes)
    throw std::runtime_error("gfx1250 B0 to A0 rewriting did not change the code object");
  constexpr uint32_t kGfx1250SNop = 0xBF800000u;
  if (first_executable_word(output_bytes) != kGfx1250SNop)
    throw std::runtime_error("gfx1250 B0 s_clause was not rewritten to A0 s_nop");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: " << argv[0]
              << " <librocjitsu.so> <librocjitsu_comgr.so> <simulator-first|hotswap-first>\n";
    return 2;
  }

  try {
    std::unique_ptr<SharedObject> simulator;
    std::unique_ptr<SharedObject> hotswap;
    const std::string_view order = argv[3];
    if (order == "simulator-first") {
      simulator = std::make_unique<SharedObject>(argv[1]);
      hotswap = std::make_unique<SharedObject>(argv[2]);
    } else if (order == "hotswap-first") {
      hotswap = std::make_unique<SharedObject>(argv[2]);
      simulator = std::make_unique<SharedObject>(argv[1]);
    } else {
      throw std::runtime_error("unknown load order: " + std::string(order));
    }

    exercise_simulator(*simulator);
    exercise_hotswap(*hotswap);
    std::cout << "loaded and exercised both DSOs with " << order << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
