#include "code_object_image.h"

#include <elf.h>
#include <hip/hip_runtime_api.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <array>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

template <typename T>
std::optional<T> read_struct(std::span<const uint8_t> bytes, uint64_t offset) {
  if (offset > bytes.size() || sizeof(T) > bytes.size() - offset)
    return std::nullopt;
  T value{};
  memcpy(&value, bytes.data() + offset, sizeof(T));
  return value;
}

bool range_in_bounds(std::span<const uint8_t> bytes, uint64_t offset, uint64_t size) {
  return offset <= bytes.size() && size <= bytes.size() - offset;
}

std::optional<std::string_view> string_at(std::span<const uint8_t> bytes, uint64_t offset,
                                          uint64_t max_size) {
  if (!range_in_bounds(bytes, offset, max_size))
    return std::nullopt;
  const char *begin = reinterpret_cast<const char *>(bytes.data() + offset);
  const void *nul = memchr(begin, '\0', static_cast<size_t>(max_size));
  if (nul == nullptr)
    return std::nullopt;
  const char *end = reinterpret_cast<const char *>(nul);
  return std::string_view(begin, static_cast<size_t>(end - begin));
}

std::optional<Elf64_Shdr> section_at(std::span<const uint8_t> elf, const Elf64_Ehdr &ehdr,
                                     uint16_t index) {
  if (index >= ehdr.e_shnum || ehdr.e_shentsize != sizeof(Elf64_Shdr))
    return std::nullopt;
  return read_struct<Elf64_Shdr>(
      elf, ehdr.e_shoff + static_cast<uint64_t>(index) * ehdr.e_shentsize);
}

std::optional<std::string> first_function_symbol(std::span<const uint8_t> elf) {
  std::optional<Elf64_Ehdr> ehdr = read_struct<Elf64_Ehdr>(elf, 0);
  if (!ehdr || memcmp(ehdr->e_ident, ELFMAG, SELFMAG) != 0 ||
      ehdr->e_ident[EI_CLASS] != ELFCLASS64 || ehdr->e_ident[EI_DATA] != ELFDATA2LSB ||
      ehdr->e_shnum == 0 || ehdr->e_shnum == SHN_UNDEF ||
      ehdr->e_shentsize != sizeof(Elf64_Shdr)) {
    return std::nullopt;
  }
  if (!range_in_bounds(elf, ehdr->e_shoff,
                       static_cast<uint64_t>(ehdr->e_shnum) * ehdr->e_shentsize)) {
    return std::nullopt;
  }

  auto scan_symbols = [&](uint32_t table_type) -> std::optional<std::string> {
    for (uint16_t section_index = 0; section_index < ehdr->e_shnum; ++section_index) {
      std::optional<Elf64_Shdr> symtab = section_at(elf, *ehdr, section_index);
      if (!symtab || symtab->sh_type != table_type || symtab->sh_entsize != sizeof(Elf64_Sym) ||
          symtab->sh_link >= ehdr->e_shnum) {
        continue;
      }
      std::optional<Elf64_Shdr> strtab = section_at(elf, *ehdr, symtab->sh_link);
      if (!strtab || !range_in_bounds(elf, symtab->sh_offset, symtab->sh_size) ||
          !range_in_bounds(elf, strtab->sh_offset, strtab->sh_size)) {
        continue;
      }

      const uint64_t symbol_count = symtab->sh_size / symtab->sh_entsize;
      for (uint64_t i = 0; i < symbol_count; ++i) {
        std::optional<Elf64_Sym> sym =
            read_struct<Elf64_Sym>(elf, symtab->sh_offset + i * symtab->sh_entsize);
        if (!sym || ELF64_ST_TYPE(sym->st_info) != STT_FUNC || sym->st_name == 0 ||
            sym->st_shndx == SHN_UNDEF || sym->st_shndx >= ehdr->e_shnum) {
          continue;
        }
        std::optional<Elf64_Shdr> target_section = section_at(elf, *ehdr, sym->st_shndx);
        if (!target_section || (target_section->sh_flags & SHF_EXECINSTR) == 0)
          continue;
        if (sym->st_name >= strtab->sh_size)
          continue;
        std::optional<std::string_view> name =
            string_at(elf, strtab->sh_offset + sym->st_name, strtab->sh_size - sym->st_name);
        if (name && !name->empty())
          return std::string(*name);
      }
    }
    return std::nullopt;
  };

  if (std::optional<std::string> name = scan_symbols(SHT_SYMTAB))
    return name;
  return scan_symbols(SHT_DYNSYM);
}

std::optional<std::string> first_kernel_in_image(const char *path) {
  std::vector<uint8_t> image = rocjitsu::fuzzer::afl_dbi::read_file_bytes(path);
  if (image.empty())
    return std::nullopt;

  std::vector<rocjitsu::fuzzer::afl_dbi::DeviceImage> device_images =
      rocjitsu::fuzzer::afl_dbi::extract_device_images(image);
  if (device_images.empty() && rocjitsu::fuzzer::afl_dbi::is_raw_elf_image(image))
    device_images.push_back({"raw-elf", "", std::move(image)});

  for (const rocjitsu::fuzzer::afl_dbi::DeviceImage &device_image : device_images) {
    if (std::optional<std::string> kernel = first_function_symbol(device_image.bytes))
      return kernel;
  }
  return std::nullopt;
}

void usage(const char *argv0) {
  fprintf(stderr, "usage: %s <module.co|hsaco> [kernel] [--repeat-get] [--concurrent-get]\n",
          argv0);
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    usage(argv[0]);
    return 2;
  }

  const char *module_path = argv[1];
  bool repeat_get = false;
  bool concurrent_get = false;
  std::string kernel_name;
  for (int i = 2; i < argc; ++i) {
    if (strcmp(argv[i], "--repeat-get") == 0) {
      repeat_get = true;
    } else if (strcmp(argv[i], "--concurrent-get") == 0) {
      concurrent_get = true;
    } else if (kernel_name.empty()) {
      kernel_name = argv[i];
    } else {
      usage(argv[0]);
      return 2;
    }
  }

  if (kernel_name.empty()) {
    std::optional<std::string> discovered = first_kernel_in_image(module_path);
    if (!discovered) {
      fprintf(stderr, "ccob_module_probe: failed to discover a kernel symbol in %s\n",
              module_path);
      return 2;
    }
    kernel_name = std::move(*discovered);
  }

  hipModule_t module = nullptr;
  hipError_t err = hipModuleLoad(&module, module_path);
  if (err != hipSuccess) {
    fprintf(stderr, "hipModuleLoad failed: %d %s\n", static_cast<int>(err),
            hipGetErrorString(err));
    return 3;
  }

  hipFunction_t function = nullptr;
  hipFunction_t repeated_function = nullptr;
  if (concurrent_get) {
    std::array<hipFunction_t, 2> functions = {nullptr, nullptr};
    std::array<hipError_t, 2> errors = {hipSuccess, hipSuccess};
    std::array<std::thread, 2> threads = {
        std::thread([&] {
          errors[0] = hipModuleGetFunction(&functions[0], module, kernel_name.c_str());
        }),
        std::thread([&] {
          errors[1] = hipModuleGetFunction(&functions[1], module, kernel_name.c_str());
        }),
    };
    for (std::thread &thread : threads)
      thread.join();
    for (size_t i = 0; i < errors.size(); ++i) {
      if (errors[i] != hipSuccess) {
        fprintf(stderr, "concurrent hipModuleGetFunction[%zu] failed: %d %s kernel=%s\n", i,
                static_cast<int>(errors[i]), hipGetErrorString(errors[i]),
                kernel_name.c_str());
        (void)hipModuleUnload(module);
        return 4;
      }
    }
    function = functions[0];
    repeated_function = functions[1];
  } else {
    err = hipModuleGetFunction(&function, module, kernel_name.c_str());
    if (err != hipSuccess) {
      fprintf(stderr, "hipModuleGetFunction failed: %d %s kernel=%s\n", static_cast<int>(err),
              hipGetErrorString(err), kernel_name.c_str());
      (void)hipModuleUnload(module);
      return 4;
    }
  }

  if (repeat_get) {
    err = hipModuleGetFunction(&repeated_function, module, kernel_name.c_str());
    if (err != hipSuccess) {
      fprintf(stderr, "repeat hipModuleGetFunction failed: %d %s kernel=%s\n",
              static_cast<int>(err), hipGetErrorString(err), kernel_name.c_str());
      (void)hipModuleUnload(module);
      return 5;
    }
  }

  printf("module=%p function=%p repeat_function=%p kernel=%s\n", reinterpret_cast<void *>(module),
         reinterpret_cast<void *>(function), reinterpret_cast<void *>(repeated_function),
         kernel_name.c_str());
  err = hipModuleUnload(module);
  if (err != hipSuccess) {
    fprintf(stderr, "hipModuleUnload failed: %d %s\n", static_cast<int>(err),
            hipGetErrorString(err));
    return 6;
  }
  return 0;
}
