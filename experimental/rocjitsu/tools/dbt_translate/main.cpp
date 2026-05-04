// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file main.cpp
/// @brief CLI tool to translate AMDGPU code objects across supported ISAs.
///
/// This tool uses the existing DBT pipeline (`BinaryTranslator`) to emit a
/// translated AMDGPU code object image.

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/dbt/binary_translator.h"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <exception>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

using namespace rocjitsu;

namespace {

enum class FlagType {
  kInput,
  kOutput,
  kGuestArch,
  kHostArch,
  kHelp,
  kUnknown,
};

struct ToolOptions {
  std::string input_path;
  std::string output_path;
  std::optional<rj_code_arch_t> guest_arch;
  rj_code_arch_t host_arch = ROCJITSU_CODE_ARCH_INVALID;
  bool show_help = false;
};

[[nodiscard]] FlagType parse_flag(std::string_view arg) {
  if (arg == "--help")
    return FlagType::kHelp;
  if (arg == "-i" || arg == "--input")
    return FlagType::kInput;
  if (arg == "-o" || arg == "--output")
    return FlagType::kOutput;
  if (arg == "-g" || arg == "--guest-arch")
    return FlagType::kGuestArch;
  if (arg == "-h" || arg == "--host-arch")
    return FlagType::kHostArch;
  return FlagType::kUnknown;
}

std::string lower_ascii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::optional<rj_code_arch_t> parse_arch(std::string value) {
  const auto normalized = lower_ascii(std::move(value));
  if (normalized == "cdna1")
    return ROCJITSU_CODE_ARCH_CDNA1;
  if (normalized == "cdna2")
    return ROCJITSU_CODE_ARCH_CDNA2;
  if (normalized == "cdna3")
    return ROCJITSU_CODE_ARCH_CDNA3;
  if (normalized == "cdna4")
    return ROCJITSU_CODE_ARCH_CDNA4;
  if (normalized == "rdna1")
    return ROCJITSU_CODE_ARCH_RDNA1;
  if (normalized == "rdna2")
    return ROCJITSU_CODE_ARCH_RDNA2;
  if (normalized == "rdna3")
    return ROCJITSU_CODE_ARCH_RDNA3;
  if (normalized == "rdna3.5" || normalized == "rdna3_5")
    return ROCJITSU_CODE_ARCH_RDNA3_5;
  if (normalized == "rdna4")
    return ROCJITSU_CODE_ARCH_RDNA4;
  if (normalized == "gfx908")
    return ROCJITSU_CODE_ARCH_CDNA1;
  if (normalized == "gfx90a")
    return ROCJITSU_CODE_ARCH_CDNA2;
  if (normalized == "gfx942")
    return ROCJITSU_CODE_ARCH_CDNA3;
  if (normalized == "gfx950")
    return ROCJITSU_CODE_ARCH_CDNA4;
  if (normalized == "gfx1010")
    return ROCJITSU_CODE_ARCH_RDNA1;
  if (normalized == "gfx1030")
    return ROCJITSU_CODE_ARCH_RDNA2;
  if (normalized == "gfx1100")
    return ROCJITSU_CODE_ARCH_RDNA3;
  if (normalized == "gfx1150")
    return ROCJITSU_CODE_ARCH_RDNA3_5;
  if (normalized == "gfx1200" || normalized == "gfx1201")
    return ROCJITSU_CODE_ARCH_RDNA4;
  return std::nullopt;
}

[[nodiscard]] const char *arch_to_string(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return "cdna1";
  case ROCJITSU_CODE_ARCH_CDNA2:
    return "cdna2";
  case ROCJITSU_CODE_ARCH_CDNA3:
    return "cdna3";
  case ROCJITSU_CODE_ARCH_CDNA4:
    return "cdna4";
  case ROCJITSU_CODE_ARCH_RDNA1:
    return "rdna1";
  case ROCJITSU_CODE_ARCH_RDNA2:
    return "rdna2";
  case ROCJITSU_CODE_ARCH_RDNA3:
    return "rdna3";
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return "rdna3_5";
  case ROCJITSU_CODE_ARCH_RDNA4:
    return "rdna4";
  default:
    return "invalid";
  }
}

[[nodiscard]] rj_code_arch_t arch_from_mach(uint32_t mach) {
  switch (mach) {
  case EF_AMDGPU_MACH_AMDGCN_GFX908:
    return ROCJITSU_CODE_ARCH_CDNA1;
  case EF_AMDGPU_MACH_AMDGCN_GFX90A:
    return ROCJITSU_CODE_ARCH_CDNA2;
  case EF_AMDGPU_MACH_AMDGCN_GFX942:
    return ROCJITSU_CODE_ARCH_CDNA3;
  case EF_AMDGPU_MACH_AMDGCN_GFX950:
    return ROCJITSU_CODE_ARCH_CDNA4;
  case EF_AMDGPU_MACH_AMDGCN_GFX1010:
    return ROCJITSU_CODE_ARCH_RDNA1;
  case EF_AMDGPU_MACH_AMDGCN_GFX1030:
    return ROCJITSU_CODE_ARCH_RDNA2;
  case EF_AMDGPU_MACH_AMDGCN_GFX1100:
    return ROCJITSU_CODE_ARCH_RDNA3;
  case EF_AMDGPU_MACH_AMDGCN_GFX1150:
    return ROCJITSU_CODE_ARCH_RDNA3_5;
  case EF_AMDGPU_MACH_AMDGCN_GFX1200:
  case EF_AMDGPU_MACH_AMDGCN_GFX1201:
    return ROCJITSU_CODE_ARCH_RDNA4;
  default:
    return ROCJITSU_CODE_ARCH_INVALID;
  }
}

[[nodiscard]] rj_code_arch_t infer_guest_arch(const AmdGpuCodeObject &obj) {
  if (obj.image_size() < sizeof(Elf64_Ehdr))
    return ROCJITSU_CODE_ARCH_INVALID;

  const auto *ehdr = reinterpret_cast<const Elf64_Ehdr *>(obj.image_data());
  const uint32_t mach = ehdr->e_flags & EF_AMDGPU_MACH;
  return arch_from_mach(mach);
}

[[nodiscard]] bool write_binary(const std::string &path, const std::vector<uint8_t> &bytes) {
  std::ofstream file(path, std::ios::binary);
  if (!file)
    return false;
  file.write(reinterpret_cast<const char *>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  return file.good();
}

void print_usage(const char *prog) {
  std::fprintf(stderr, "Usage: %s --input <code_object> --output <translated_object> \\\n", prog);
  std::fprintf(stderr, "           --host-arch <arch> [--guest-arch <arch>]\n\n");
  std::fprintf(stderr,
               "  <arch> values: cdna1 cdna2 cdna3 cdna4 rdna1 rdna2 rdna3 rdna3_5 rdna4\n");
  std::fprintf(stderr,
               "  guest-arch can be omitted for supported MACH values in the input image.\n\n");
  std::fprintf(stderr, "This tool translates AMDGPU ELF code objects using BinaryTranslator.\n");
}

bool parse_args(int argc, char *argv[], ToolOptions &opts) {
  for (int i = 1; i < argc; ++i) {
    const auto flag = parse_flag(argv[i]);
    if (flag == FlagType::kUnknown) {
      std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
      return false;
    }
    if (flag == FlagType::kHelp) {
      opts.show_help = true;
      print_usage(argv[0]);
      return true;
    }

    if (i + 1 == argc) {
      std::fprintf(stderr, "Missing value for option: %s\n", argv[i]);
      return false;
    }
    const char *value = argv[++i];

    switch (flag) {
    case FlagType::kInput:
      opts.input_path = value;
      break;
    case FlagType::kOutput:
      opts.output_path = value;
      break;
    case FlagType::kGuestArch: {
      const auto parsed = parse_arch(value);
      if (!parsed) {
        std::fprintf(stderr, "Invalid guest architecture: %s\n", value);
        return false;
      }
      opts.guest_arch = parsed;
      break;
    }
    case FlagType::kHostArch: {
      const auto parsed = parse_arch(value);
      if (!parsed) {
        std::fprintf(stderr, "Invalid host architecture: %s\n", value);
        return false;
      }
      opts.host_arch = *parsed;
      break;
    }
    case FlagType::kHelp:
      return true;
    case FlagType::kUnknown:
      return false;
    }
  }

  return true;
}

} // namespace

int main(int argc, char *argv[]) {
  if (argc == 1) {
    print_usage(argv[0]);
    return 1;
  }

  ToolOptions opts;
  if (!parse_args(argc, argv, opts))
    return opts.show_help ? 0 : 1;
  if (opts.show_help)
    return 0;

  if (opts.input_path.empty()) {
    std::fprintf(stderr, "Missing required --input path\n");
    print_usage(argv[0]);
    return 1;
  }
  if (opts.output_path.empty()) {
    std::fprintf(stderr, "Missing required --output path\n");
    print_usage(argv[0]);
    return 1;
  }
  if (opts.host_arch == ROCJITSU_CODE_ARCH_INVALID) {
    std::fprintf(stderr, "Missing required --host-arch\n");
    print_usage(argv[0]);
    return 1;
  }

  AmdGpuCodeObject source(opts.input_path);
  if (!source.is_valid()) {
    std::fprintf(stderr, "Failed to load valid AMDGPU code object from: %s\n",
                 opts.input_path.c_str());
    return 1;
  }

  const auto inferred_guest = infer_guest_arch(source);
  if (!opts.guest_arch) {
    if (inferred_guest == ROCJITSU_CODE_ARCH_INVALID) {
      std::fprintf(stderr, "Could not infer guest architecture from input Mach bits.\n");
      std::fprintf(stderr, "Pass --guest-arch explicitly.\n");
      return 1;
    }
    opts.guest_arch = inferred_guest;
  } else if (inferred_guest != ROCJITSU_CODE_ARCH_INVALID && inferred_guest != *opts.guest_arch) {
    std::fprintf(stderr, "Warning: input MACH implies %s but --guest-arch=%s was passed.\n",
                 arch_to_string(inferred_guest), arch_to_string(*opts.guest_arch));
  }

  const auto guest = *opts.guest_arch;
  if (guest == ROCJITSU_CODE_ARCH_INVALID) {
    std::fprintf(stderr, "Invalid source architecture.\n");
    return 1;
  }

  BinaryTranslator translator(guest, opts.host_arch);
  TranslatedCodeObject translated;
  try {
    translated = translator.translate(source);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "Translation failed: %s\n", e.what());
    return 1;
  } catch (...) {
    std::fprintf(stderr, "Translation failed: unknown error\n");
    return 1;
  }
  if (translated.elf_bytes.empty()) {
    std::fprintf(stderr, "Translation produced an empty output image.\n");
    return 1;
  }

  if (!write_binary(opts.output_path, translated.elf_bytes)) {
    std::fprintf(stderr, "Failed to write output code object: %s\n", opts.output_path.c_str());
    return 1;
  }

  std::fprintf(stdout, "Wrote translated object: %s\n", opts.output_path.c_str());
  std::fprintf(stdout, "Guest: %s, Host: %s\n", arch_to_string(guest),
               arch_to_string(opts.host_arch));
  std::fprintf(stdout, "Output size: %zu bytes\n", translated.elf_bytes.size());

  if (!translated.warnings.empty()) {
    std::fprintf(stdout, "Warnings:\n");
    for (const auto &warning : translated.warnings)
      std::fprintf(stdout, "  %s\n", warning.c_str());
  }

  return 0;
}
