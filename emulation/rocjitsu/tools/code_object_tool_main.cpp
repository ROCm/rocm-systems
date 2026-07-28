// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/waitcheck.h"
#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/executable.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "util/except.h"

RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

using namespace rocjitsu;

namespace {

constexpr int kUsageError = 1;
constexpr int kInputError = 2;
constexpr int kOutputError = 3;
constexpr uint8_t kElfSymbolTypeFunc = 2;
using KernelDescriptor = rocr::llvm::amdhsa::kernel_descriptor_t;

static_assert(sizeof(KernelDescriptor) == 64, "AMDHSA kernel descriptor size changed");

constexpr std::array<rj_code_target_id_t, 8> kKnownTargets = {
    ROCJITSU_CODE_TARGET_GFX942,  ROCJITSU_CODE_TARGET_GFX950,  ROCJITSU_CODE_TARGET_GFX1100,
    ROCJITSU_CODE_TARGET_GFX1150, ROCJITSU_CODE_TARGET_GFX1151, ROCJITSU_CODE_TARGET_GFX1200,
    ROCJITSU_CODE_TARGET_GFX1201, ROCJITSU_CODE_TARGET_GFX1250,
};

struct CliOptions {
  std::string input_path;
  std::optional<rj_code_target_id_t> target;
  uint32_t code_object_index = 0;
  bool list_code_objects = false;
  bool list_kernels = false;
  bool run_waitcheck = false;
  std::optional<std::string> extract_path;
  std::optional<std::string> map_location;
  std::optional<std::string> disassemble_location;
  std::optional<size_t> repro_diagnostic;
  uint64_t context_bytes = 128;
  size_t max_diagnostics = 256;
  bool show_help = false;
};

struct SelectedCodeObject {
  std::unique_ptr<Executable> executable;
  const AmdGpuCodeObject *code_object = nullptr;
  rj_code_target_id_t target = ROCJITSU_CODE_TARGET_INVALID;
  uint32_t index = 0;
};

struct TextSectionInfo {
  const Section *section = nullptr;
  std::string name;
  uint64_t vaddr = 0;
  uint64_t file_offset = 0;
  uint64_t size = 0;
};

struct KernelInfo {
  std::string name;
  uint64_t descriptor_vaddr = 0;
  uint64_t entry_offset = 0;
  uint64_t entry_vaddr = 0;
  uint64_t entry_file_offset = 0;
  uint64_t size = 0;
};

struct CodeObjectInfo {
  std::vector<TextSectionInfo> text_sections;
  std::vector<KernelInfo> kernels;
};

struct MappedLocation {
  const TextSectionInfo *text = nullptr;
  const KernelInfo *kernel = nullptr;
  uint64_t section_offset = 0;
  uint64_t vaddr = 0;
  uint64_t file_offset = 0;
};

[[nodiscard]] bool fits_in_image(uint64_t offset, uint64_t size, size_t image_size) {
  return offset <= image_size && size <= image_size - offset;
}

[[nodiscard]] std::string hex_value(uint64_t value) {
  std::ostringstream os;
  os << "0x" << std::hex << value;
  return os.str();
}

[[nodiscard]] std::string count_label(size_t count, bool lower_bound) {
  std::ostringstream os;
  if (lower_bound)
    os << ">=";
  os << count;
  return os.str();
}

[[nodiscard]] std::optional<rj_code_target_id_t> parse_target(std::string_view value) {
  for (rj_code_target_id_t target : kKnownTargets) {
    if (value == rj_code_target_name(target))
      return target;
  }
  return std::nullopt;
}

[[nodiscard]] bool parse_u64(std::string_view text, uint64_t &value) {
  int base = 10;
  if (text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
    text.remove_prefix(2);
    base = 16;
  }
  auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), value, base);
  return ec == std::errc{} && ptr == text.data() + text.size();
}

[[nodiscard]] bool parse_u32(std::string_view text, uint32_t &value) {
  uint64_t parsed = 0;
  if (!parse_u64(text, parsed) || parsed > UINT32_MAX)
    return false;
  value = static_cast<uint32_t>(parsed);
  return true;
}

[[nodiscard]] bool parse_size(std::string_view text, size_t &value) {
  uint64_t parsed = 0;
  if (!parse_u64(text, parsed))
    return false;
  value = static_cast<size_t>(parsed);
  return static_cast<uint64_t>(value) == parsed;
}

[[nodiscard]] bool require_value(int argc, char **argv, int &index, std::string_view flag,
                                 std::string_view &value) {
  if (index + 1 >= argc) {
    std::cerr << "missing value for " << flag << "\n";
    return false;
  }
  value = argv[++index];
  return true;
}

void print_help() {
  std::cout
      << "Usage: rj_co INPUT [options]\n\n"
      << "Options:\n"
      << "  --target TARGET          Select target code object\n"
      << "  --code-object-index N    Code-object index for the selected target (default: 0)\n"
      << "  --list-code-objects      List code-object counts by target\n"
      << "  --extract-code-object P Write the selected raw device ELF to path P\n"
      << "  --list-kernels           List kernels, entry offsets, VAs, file offsets, and sizes\n"
      << "  --map LOC                Map .text+offset, VA, or file offset to section/kernel\n"
      << "  --disassemble-window LOC Decode instructions around LOC (default context: 128 bytes)\n"
      << "  --context-bytes N        Context for --disassemble-window\n"
      << "  --waitcheck              Run waitcheck and group diagnostics by kernel\n"
      << "  --repro-diagnostic N     Emit a markdown repro block for waitcheck diagnostic N\n"
      << "  --max-diagnostics N      Diagnostics retained for --waitcheck/--repro (default: 256)\n"
      << "  --help                   Show this help\n\n"
      << "LOC accepts .text+0x1234, a virtual address, or a code-object file offset.\n";
}

[[nodiscard]] bool parse_args(int argc, char **argv, CliOptions &options) {
  for (int i = 1; i < argc; ++i) {
    std::string_view arg = argv[i];
    std::string_view value;
    if (arg == "--help" || arg == "-h") {
      options.show_help = true;
      return true;
    }
    if (arg == "--list-code-objects") {
      options.list_code_objects = true;
      continue;
    }
    if (arg == "--list-kernels") {
      options.list_kernels = true;
      continue;
    }
    if (arg == "--extract-code-object") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      options.extract_path = std::string(value);
      continue;
    }
    if (arg == "--waitcheck") {
      options.run_waitcheck = true;
      continue;
    }
    if (arg == "--target") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      auto target = parse_target(value);
      if (!target) {
        std::cerr << "invalid target: " << value << "\n";
        return false;
      }
      options.target = *target;
      continue;
    }
    if (arg == "--code-object-index") {
      if (!require_value(argc, argv, i, arg, value) ||
          !parse_u32(value, options.code_object_index)) {
        std::cerr << "invalid code-object index\n";
        return false;
      }
      continue;
    }
    if (arg == "--map") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      options.map_location = std::string(value);
      continue;
    }
    if (arg == "--disassemble-window") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      options.disassemble_location = std::string(value);
      continue;
    }
    if (arg == "--context-bytes") {
      size_t parsed = 0;
      if (!require_value(argc, argv, i, arg, value) || !parse_size(value, parsed)) {
        std::cerr << "invalid context byte count\n";
        return false;
      }
      options.context_bytes = static_cast<uint64_t>(parsed);
      continue;
    }
    if (arg == "--repro-diagnostic") {
      size_t parsed = 0;
      if (!require_value(argc, argv, i, arg, value) || !parse_size(value, parsed)) {
        std::cerr << "invalid diagnostic index\n";
        return false;
      }
      options.repro_diagnostic = parsed;
      continue;
    }
    if (arg == "--max-diagnostics") {
      size_t parsed = 0;
      if (!require_value(argc, argv, i, arg, value) || !parse_size(value, parsed)) {
        std::cerr << "invalid max diagnostics\n";
        return false;
      }
      options.max_diagnostics = parsed;
      continue;
    }
    if (!arg.empty() && arg.front() != '-') {
      if (!options.input_path.empty()) {
        std::cerr << "only one input path is supported\n";
        return false;
      }
      options.input_path = std::string(arg);
      continue;
    }
    std::cerr << "unknown option: " << arg << "\n";
    return false;
  }

  if (options.show_help)
    return true;
  if (options.input_path.empty()) {
    std::cerr << "input path is required\n";
    return false;
  }
  if (!options.list_code_objects && !options.extract_path && !options.list_kernels &&
      !options.map_location && !options.disassemble_location && !options.run_waitcheck &&
      !options.repro_diagnostic) {
    options.list_kernels = true;
  }
  return true;
}

void list_code_objects(const Executable &executable) {
  for (rj_code_target_id_t target : kKnownTargets)
    std::cout << rj_code_target_name(target) << ": " << executable.num_code_objects(target) << "\n";
}

[[nodiscard]] SelectedCodeObject
select_code_object(const CliOptions &options, const std::string &input_path, std::string &error) {
  SelectedCodeObject selected;
  selected.executable = std::make_unique<Executable>(input_path);
  if (!selected.executable->is_valid()) {
    error = "failed to parse input executable or code object";
    selected.executable.reset();
    return selected;
  }

  if (options.target) {
    selected.code_object =
        selected.executable->code_object(*options.target, options.code_object_index);
    if (!selected.code_object) {
      error = "failed to select " + std::string(rj_code_target_name(*options.target)) + "[" +
              std::to_string(options.code_object_index) + "]";
      selected.executable.reset();
      return selected;
    }
    selected.target = *options.target;
    selected.index = options.code_object_index;
    return selected;
  }

  std::optional<rj_code_target_id_t> target_with_objects;
  uint32_t target_count = 0;
  for (rj_code_target_id_t target : kKnownTargets) {
    if (selected.executable->num_code_objects(target) == 0)
      continue;
    target_with_objects = target;
    ++target_count;
  }
  if (target_count == 0) {
    error = "no known AMDGPU code objects found";
    selected.executable.reset();
    return selected;
  }
  if (target_count > 1) {
    error = "multiple targets found; pass --target";
    selected.executable.reset();
    return selected;
  }

  selected.code_object =
      selected.executable->code_object(*target_with_objects, options.code_object_index);
  if (!selected.code_object) {
    error = "failed to select " + std::string(rj_code_target_name(*target_with_objects)) + "[" +
            std::to_string(options.code_object_index) + "]";
    selected.executable.reset();
    return selected;
  }
  selected.target = *target_with_objects;
  selected.index = options.code_object_index;
  return selected;
}

[[nodiscard]] bool extract_code_object(const AmdGpuCodeObject &code_object,
                                       const std::string &output_path) {
  std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
  if (!output)
    return false;
  output.write(code_object.image_data(), static_cast<std::streamsize>(code_object.image_size()));
  return output.good();
}

[[nodiscard]] std::string read_cstr(const char *strtab, size_t strtab_size, uint32_t offset) {
  if (strtab == nullptr || offset >= strtab_size)
    return {};
  const size_t max_len = strtab_size - offset;
  return {strtab + offset, strnlen(strtab + offset, max_len)};
}

[[nodiscard]] std::optional<std::vector<Elf64_Shdr>>
section_headers(const CodeObject &code_object) {
  const auto *image = reinterpret_cast<const uint8_t *>(code_object.image_data());
  const size_t image_size = code_object.image_size();
  if (image == nullptr || image_size < sizeof(Elf64_Ehdr))
    return std::nullopt;

  const auto *ehdr = reinterpret_cast<const Elf64_Ehdr *>(image);
  if (std::memcmp(ehdr->e_ident, EI_MAGIC, EI_MAGIC_SIZE) != 0 ||
      ehdr->e_ident[EI_CLASS] != ELFCLASS64 || ehdr->e_shentsize != sizeof(Elf64_Shdr)) {
    return std::nullopt;
  }
  const uint64_t shdr_bytes = static_cast<uint64_t>(ehdr->e_shnum) * sizeof(Elf64_Shdr);
  if (!fits_in_image(ehdr->e_shoff, shdr_bytes, image_size))
    return std::nullopt;

  std::vector<Elf64_Shdr> shdrs(ehdr->e_shnum);
  std::memcpy(shdrs.data(), image + ehdr->e_shoff, shdr_bytes);
  return shdrs;
}

[[nodiscard]] const TextSectionInfo *text_for_vaddr(const CodeObjectInfo &info, uint64_t vaddr) {
  for (const TextSectionInfo &text : info.text_sections) {
    if (vaddr >= text.vaddr && vaddr < text.vaddr + text.size)
      return &text;
  }
  return nullptr;
}

[[nodiscard]] const TextSectionInfo *text_for_file_offset(const CodeObjectInfo &info,
                                                          uint64_t file_offset) {
  for (const TextSectionInfo &text : info.text_sections) {
    if (file_offset >= text.file_offset && file_offset < text.file_offset + text.size)
      return &text;
  }
  return nullptr;
}

[[nodiscard]] std::optional<KernelInfo>
kernel_from_descriptor_symbol(const CodeObject &code_object, const CodeObjectInfo &base_info,
                              const std::vector<Elf64_Shdr> &shdrs, const Elf64_Sym &sym,
                              std::string name) {
  if (sym.st_shndx >= shdrs.size())
    return std::nullopt;

  const Elf64_Shdr &descriptor_section = shdrs[sym.st_shndx];
  if (sym.st_value < descriptor_section.sh_addr)
    return std::nullopt;
  const uint64_t descriptor_file_offset =
      descriptor_section.sh_offset + (sym.st_value - descriptor_section.sh_addr);
  if (!fits_in_image(descriptor_file_offset, sizeof(KernelDescriptor), code_object.image_size()))
    return std::nullopt;

  KernelDescriptor descriptor{};
  std::memcpy(&descriptor, code_object.image_data() + descriptor_file_offset, sizeof(descriptor));
  const int64_t entry_vaddr_signed =
      static_cast<int64_t>(sym.st_value) + descriptor.kernel_code_entry_byte_offset;
  if (entry_vaddr_signed < 0)
    return std::nullopt;

  const uint64_t entry_vaddr = static_cast<uint64_t>(entry_vaddr_signed);
  const TextSectionInfo *text = text_for_vaddr(base_info, entry_vaddr);
  if (text == nullptr)
    return std::nullopt;

  KernelInfo kernel;
  kernel.name = std::move(name);
  kernel.descriptor_vaddr = sym.st_value;
  kernel.entry_vaddr = entry_vaddr;
  kernel.entry_offset = entry_vaddr - text->vaddr;
  kernel.entry_file_offset = text->file_offset + kernel.entry_offset;
  return kernel;
}

[[nodiscard]] CodeObjectInfo inspect_code_object(const CodeObject &code_object) {
  CodeObjectInfo info;
  for (const Section *section : code_object.text_sections()) {
    info.text_sections.push_back(
        {section, section->name(), section->vaddr(), section->sectionOffset(), section->size()});
  }

  const auto shdrs = section_headers(code_object);
  if (!shdrs)
    return info;

  std::map<uint64_t, uint64_t> function_sizes;
  std::set<std::pair<uint64_t, std::string>> seen_kernels;
  const auto *image = reinterpret_cast<const uint8_t *>(code_object.image_data());
  for (const Elf64_Shdr &symtab_shdr : *shdrs) {
    if (symtab_shdr.sh_type != SHT_SYMTAB && symtab_shdr.sh_type != SHT_DYNSYM)
      continue;
    if (symtab_shdr.sh_entsize < sizeof(Elf64_Sym) ||
        !fits_in_image(symtab_shdr.sh_offset, symtab_shdr.sh_size, code_object.image_size())) {
      continue;
    }
    if (symtab_shdr.sh_link >= shdrs->size())
      continue;
    const Elf64_Shdr &strtab_shdr = (*shdrs)[symtab_shdr.sh_link];
    if (!fits_in_image(strtab_shdr.sh_offset, strtab_shdr.sh_size, code_object.image_size()))
      continue;

    const char *strtab = reinterpret_cast<const char *>(image + strtab_shdr.sh_offset);
    const char *symtab = reinterpret_cast<const char *>(image + symtab_shdr.sh_offset);
    const size_t symbol_count = symtab_shdr.sh_size / symtab_shdr.sh_entsize;
    for (size_t sym_index = 0; sym_index < symbol_count; ++sym_index) {
      Elf64_Sym sym{};
      std::memcpy(&sym, symtab + sym_index * symtab_shdr.sh_entsize, sizeof(sym));
      std::string name = read_cstr(strtab, strtab_shdr.sh_size, sym.st_name);
      if (name.empty())
        continue;

      if (elf_symbol_type(sym.st_info) == kElfSymbolTypeFunc && sym.st_size != 0) {
        if (const TextSectionInfo *text = text_for_vaddr(info, sym.st_value))
          function_sizes[sym.st_value - text->vaddr] = sym.st_size;
      }

      if (sym.st_size != sizeof(KernelDescriptor) ||
          elf_symbol_type(sym.st_info) != kElfSymbolTypeObject || name.size() <= 3 ||
          name.substr(name.size() - 3) != ".kd") {
        continue;
      }
      name.resize(name.size() - 3);
      if (auto kernel = kernel_from_descriptor_symbol(code_object, info, *shdrs, sym, name)) {
        if (!seen_kernels.insert({kernel->entry_offset, kernel->name}).second)
          continue;
        info.kernels.push_back(std::move(*kernel));
      }
    }
  }

  std::sort(info.kernels.begin(), info.kernels.end(),
            [](const KernelInfo &lhs, const KernelInfo &rhs) {
              return std::tie(lhs.entry_offset, lhs.name) < std::tie(rhs.entry_offset, rhs.name);
            });
  for (size_t i = 0; i < info.kernels.size(); ++i) {
    KernelInfo &kernel = info.kernels[i];
    if (auto size_it = function_sizes.find(kernel.entry_offset); size_it != function_sizes.end()) {
      kernel.size = size_it->second;
      continue;
    }
    uint64_t limit = 0;
    if (i + 1 < info.kernels.size()) {
      limit = info.kernels[i + 1].entry_offset;
    } else if (const TextSectionInfo *text = text_for_vaddr(info, kernel.entry_vaddr)) {
      limit = text->size;
    }
    if (limit > kernel.entry_offset)
      kernel.size = limit - kernel.entry_offset;
  }
  return info;
}

[[nodiscard]] const KernelInfo *kernel_for_offset(const CodeObjectInfo &info, uint64_t offset) {
  const KernelInfo *best = nullptr;
  for (const KernelInfo &kernel : info.kernels) {
    if (offset < kernel.entry_offset)
      continue;
    if (kernel.size != 0 && offset >= kernel.entry_offset + kernel.size)
      continue;
    if (best == nullptr || kernel.entry_offset > best->entry_offset)
      best = &kernel;
  }
  return best;
}

[[nodiscard]] std::optional<MappedLocation> map_location(const CodeObjectInfo &info,
                                                         std::string_view location) {
  uint64_t value = 0;
  const TextSectionInfo *text = nullptr;
  uint64_t section_offset = 0;
  if (location.size() > 6 && location.substr(0, 6) == ".text+") {
    if (!parse_u64(location.substr(6), value))
      return std::nullopt;
    section_offset = value;
    if (!info.text_sections.empty())
      text = &info.text_sections.front();
  } else {
    if (!parse_u64(location, value))
      return std::nullopt;
    if ((text = text_for_vaddr(info, value)) != nullptr) {
      section_offset = value - text->vaddr;
    } else if ((text = text_for_file_offset(info, value)) != nullptr) {
      section_offset = value - text->file_offset;
    } else if (!info.text_sections.empty() && value < info.text_sections.front().size) {
      text = &info.text_sections.front();
      section_offset = value;
    } else {
      return std::nullopt;
    }
  }

  if (text == nullptr || section_offset >= text->size)
    return std::nullopt;

  MappedLocation mapped;
  mapped.text = text;
  mapped.kernel = kernel_for_offset(info, section_offset);
  mapped.section_offset = section_offset;
  mapped.vaddr = text->vaddr + section_offset;
  mapped.file_offset = text->file_offset + section_offset;
  return mapped;
}

void print_mapped_location(std::string_view label, const MappedLocation &mapped) {
  std::cout << label << ": section=" << mapped.text->name
            << " section-offset=" << hex_value(mapped.section_offset)
            << " va=" << hex_value(mapped.vaddr)
            << " file-offset=" << hex_value(mapped.file_offset);
  if (mapped.kernel) {
    std::cout << " kernel=" << mapped.kernel->name << "+"
              << hex_value(mapped.section_offset - mapped.kernel->entry_offset);
  }
  std::cout << "\n";
}

void print_kernels(const CodeObjectInfo &info, rj_code_target_id_t target, uint32_t index) {
  std::cout << "target " << rj_code_target_name(target) << "[" << index << "]\n";
  for (const TextSectionInfo &text : info.text_sections) {
    std::cout << "section " << text.name << " size=" << hex_value(text.size)
              << " va=" << hex_value(text.vaddr) << " file-offset=" << hex_value(text.file_offset)
              << "\n";
  }
  for (const KernelInfo &kernel : info.kernels) {
    std::cout << "kernel " << kernel.name << " entry=.text+" << hex_value(kernel.entry_offset)
              << " va=" << hex_value(kernel.entry_vaddr)
              << " file-offset=" << hex_value(kernel.entry_file_offset)
              << " size=" << hex_value(kernel.size) << "\n";
  }
}

[[nodiscard]] std::string mnemonic(std::string_view instruction) {
  size_t begin = instruction.find_first_not_of(' ');
  if (begin == std::string_view::npos)
    return {};
  size_t end = instruction.find_first_of(" \t", begin);
  return std::string(instruction.substr(begin, end - begin));
}

void print_waitcheck_summary(const WaitcheckReport &report, const CodeObjectInfo &info,
                             rj_code_target_id_t target, uint32_t index) {
  std::cout << "waitcheck " << rj_code_target_name(target) << "[" << index
            << "]: instructions=" << report.instructions_analyzed
            << " memory-events=" << report.memory_events_tracked << " diagnostics="
            << count_label(report.diagnostics_observed, report.diagnostics_truncated)
            << " retained=" << report.diagnostics.size() << "\n";

  struct KernelGroup {
    size_t count = 0;
    std::map<std::string, size_t> pairs;
  };
  std::map<std::string, KernelGroup> groups;
  for (const WaitcheckDiagnostic &diag : report.diagnostics) {
    const KernelInfo *kernel = kernel_for_offset(info, diag.section_offset);
    std::string name = kernel ? kernel->name : "<unknown>";
    KernelGroup &group = groups[name];
    ++group.count;
    ++group.pairs[mnemonic(diag.producer_instruction) + " -> " + mnemonic(diag.instruction)];
  }

  for (const auto &[name, group] : groups) {
    std::cout << "kernel " << name << " diagnostics=" << group.count << "\n";
    for (const auto &[pair, count] : group.pairs)
      std::cout << "  " << pair << ": " << count << "\n";
  }
}

[[nodiscard]] std::string relative_command_path(const std::filesystem::path &path) {
  std::error_code ec;
  std::filesystem::path relative =
      std::filesystem::relative(path, std::filesystem::current_path(), ec);
  if (!ec && !relative.empty())
    return relative.string();
  return path.string();
}

void print_repro(const CliOptions &options, const WaitcheckReport &report,
                 const CodeObjectInfo &info, rj_code_target_id_t target, uint32_t index) {
  if (!options.repro_diagnostic || *options.repro_diagnostic >= report.diagnostics.size()) {
    std::cerr << "diagnostic index is out of range; retained diagnostics="
              << report.diagnostics.size() << "\n";
    return;
  }

  const WaitcheckDiagnostic &diag = report.diagnostics[*options.repro_diagnostic];
  const KernelInfo *kernel = kernel_for_offset(info, diag.section_offset);
  std::cout << "### waitcheck diagnostic " << *options.repro_diagnostic << "\n\n";
  std::cout << "Repro:\n\n";
  std::cout << "```sh\n";
  std::cout << "rj_waitcheck " << relative_command_path(options.input_path) << " --target "
            << rj_code_target_name(target) << " --code-object-index " << index
            << " --max-diagnostics " << (*options.repro_diagnostic + 1) << "\n";
  std::cout << "```\n\n";
  std::cout << "Artifact: `" << relative_command_path(options.input_path) << "`\n";
  std::cout << "Target: `" << rj_code_target_name(target) << "[" << index << "]`\n";
  if (kernel) {
    std::cout << "Kernel: `" << kernel->name << "` (`.text+" << hex_value(kernel->entry_offset)
              << ".." << hex_value(kernel->entry_offset + kernel->size) << "`)\n";
  }
  std::cout << "\nDiagnostic:\n\n";
  std::cout << "```text\n" << diag.message << "\n";
  std::cout << "producer .text+" << hex_value(diag.producer_section_offset) << ": "
            << diag.producer_instruction << "\n";
  std::cout << "consumer .text+" << hex_value(diag.section_offset) << ": " << diag.instruction
            << "\n";
  std::cout << "```\n";
}

void disassemble_window(const MappedLocation &mapped, rj_code_arch_t arch, uint64_t context_bytes) {
  auto decoder = Decoder::create(arch);
  if (!decoder) {
    std::cerr << "failed to create decoder for " << static_cast<int>(arch) << "\n";
    return;
  }

  const uint64_t range_begin =
      mapped.section_offset > context_bytes ? mapped.section_offset - context_bytes : 0;
  const uint64_t range_end =
      std::min<uint64_t>(mapped.text->size, mapped.section_offset + context_bytes);
  uint64_t decode_offset = mapped.kernel ? mapped.kernel->entry_offset : range_begin;
  decode_offset &= ~uint64_t{3};

  std::vector<uint32_t> words(reinterpret_cast<const uint32_t *>(mapped.text->section->data()),
                              reinterpret_cast<const uint32_t *>(mapped.text->section->data()) +
                                  mapped.text->size / sizeof(uint32_t));
  words.resize(words.size() + 2);

  while (decode_offset < mapped.text->size && decode_offset <= range_end) {
    std::unique_ptr<Instruction> inst;
    try {
      inst.reset(decoder->decode(&words[decode_offset / sizeof(uint32_t)]));
    } catch (const util::Exception &ex) {
      std::cerr << "decode failed at .text+" << hex_value(decode_offset) << ": " << ex.what()
                << "\n";
      return;
    }
    if (!inst || inst->size() <= 0)
      return;
    if (decode_offset >= range_begin) {
      std::cout << ".text+" << std::setw(8) << std::setfill('0') << std::hex << decode_offset
                << std::setfill(' ') << std::dec << ": " << inst->disassemble() << "\n";
    }
    decode_offset += static_cast<uint64_t>(inst->size());
  }
}

[[nodiscard]] std::optional<WaitcheckReport> run_waitcheck(const AmdGpuCodeObject &code_object,
                                                           rj_code_target_id_t target,
                                                           size_t max_diagnostics) {
  const rj_code_arch_t arch = waitcheck_arch_for_target(target);
  if (arch == ROCJITSU_CODE_ARCH_INVALID) {
    std::cerr << "waitcheck does not support target " << rj_code_target_name(target) << "\n";
    return std::nullopt;
  }
  WaitcheckOptions options;
  options.max_diagnostics = max_diagnostics;
  WaitcheckReport report = analyze_waitcnts(code_object, arch, options);
  if (!report.supported) {
    std::cerr << "waitcheck analysis failed: " << report.analysis_error << "\n";
    return std::nullopt;
  }
  return report;
}

} // namespace

int main(int argc, char **argv) {
  CliOptions options;
  if (!parse_args(argc, argv, options))
    return kUsageError;
  if (options.show_help) {
    print_help();
    return 0;
  }

  Executable executable(options.input_path);
  if (!executable.is_valid()) {
    std::cerr << options.input_path << ": failed to parse input executable or code object\n";
    return kInputError;
  }
  if (options.list_code_objects)
    list_code_objects(executable);

  if (options.list_code_objects && !options.extract_path && !options.list_kernels &&
      !options.map_location && !options.disassemble_location && !options.run_waitcheck &&
      !options.repro_diagnostic) {
    return 0;
  }

  std::string error;
  SelectedCodeObject selected = select_code_object(options, options.input_path, error);
  if (!selected.executable) {
    std::cerr << options.input_path << ": " << error << "\n";
    return kInputError;
  }

  if (options.extract_path && !extract_code_object(*selected.code_object, *options.extract_path)) {
    std::cerr << "failed to write extracted code object: " << *options.extract_path << "\n";
    return kOutputError;
  }

  CodeObjectInfo info = inspect_code_object(*selected.code_object);
  if (options.list_kernels)
    print_kernels(info, selected.target, selected.index);

  if (options.map_location) {
    auto mapped = map_location(info, *options.map_location);
    if (!mapped) {
      std::cerr << "failed to map location: " << *options.map_location << "\n";
      return kInputError;
    }
    print_mapped_location(*options.map_location, *mapped);
  }

  if (options.disassemble_location) {
    auto mapped = map_location(info, *options.disassemble_location);
    if (!mapped) {
      std::cerr << "failed to map location: " << *options.disassemble_location << "\n";
      return kInputError;
    }
    const rj_code_arch_t arch = waitcheck_arch_for_target(selected.target);
    if (arch == ROCJITSU_CODE_ARCH_INVALID) {
      std::cerr << "decoder target is not supported by rj_co: "
                << rj_code_target_name(selected.target) << "\n";
      return kInputError;
    }
    disassemble_window(*mapped, arch, options.context_bytes);
  }

  std::optional<WaitcheckReport> report;
  if (options.run_waitcheck || options.repro_diagnostic) {
    report = run_waitcheck(*selected.code_object, selected.target, options.max_diagnostics);
    if (!report)
      return kInputError;
  }
  if (options.run_waitcheck)
    print_waitcheck_summary(*report, info, selected.target, selected.index);
  if (options.repro_diagnostic)
    print_repro(options, *report, info, selected.target, selected.index);

  return 0;
}
