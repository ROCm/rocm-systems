// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/dbt/binary_translator.h"
#include "rocjitsu_translate_common.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

void print_usage(const char *argv0) {
  std::cerr << "usage: " << argv0
            << " <input.hsaco> <output.hsaco> <guest-arch> <host-arch> [host-mach]\n"
            << "example: " << argv0 << " in.hsaco out.hsaco gfx1250 rdna4 gfx1201\n";
}

const char *diagnostic_severity_name(rocjitsu::DiagnosticSeverity severity) {
  switch (severity) {
  case rocjitsu::DiagnosticSeverity::Warning:
    return "warning";
  case rocjitsu::DiagnosticSeverity::Error:
    return "error";
  }
  return "diagnostic";
}

const char *diagnostic_kind_name(rocjitsu::DiagnosticKind kind) {
  switch (kind) {
  case rocjitsu::DiagnosticKind::UnsupportedGuestArch:
    return "unsupported-guest-arch";
  case rocjitsu::DiagnosticKind::KernelDescriptor:
    return "kernel-descriptor";
  case rocjitsu::DiagnosticKind::Legalization:
    return "legalization";
  case rocjitsu::DiagnosticKind::ExpandMissing:
    return "expand-missing";
  case rocjitsu::DiagnosticKind::ExpandFailed:
    return "expand-failed";
  case rocjitsu::DiagnosticKind::ResourceLimit:
    return "resource-limit";
  }
  return "unknown";
}

void print_diagnostics(const std::vector<rocjitsu::TranslationDiagnostic> &diagnostics) {
  for (const auto &diagnostic : diagnostics) {
    std::cerr << diagnostic_severity_name(diagnostic.severity) << ": "
              << diagnostic_kind_name(diagnostic.kind);
    if (diagnostic.guest_offset)
      std::cerr << " .text+0x" << std::hex << *diagnostic.guest_offset << std::dec;
    if (!diagnostic.mnemonic.empty())
      std::cerr << ' ' << diagnostic.mnemonic;
    std::cerr << ": " << diagnostic.message << '\n';
    for (const auto &item : diagnostic.required_work)
      std::cerr << "  required: " << item << '\n';
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 5 && argc != 6) {
    print_usage(argv[0]);
    return 1;
  }

  const auto guest_arch = rocjitsu::tools::parse_arch(argv[3]);
  if (!guest_arch) {
    std::cerr << "unknown guest architecture: " << argv[3] << '\n';
    return 1;
  }

  const auto host_arch = rocjitsu::tools::parse_arch(argv[4]);
  if (!host_arch) {
    std::cerr << "unknown host architecture: " << argv[4] << '\n';
    return 1;
  }

  const auto host_mach =
      argc == 6 ? rocjitsu::tools::parse_mach(argv[5]) : std::optional<uint32_t>(0);
  if (!host_mach) {
    std::cerr << "unknown host MACH value: " << argv[5] << '\n';
    return 1;
  }

  rocjitsu::AmdGpuCodeObject code_object(argv[1]);
  if (!code_object.is_valid()) {
    std::cerr << "failed to load AMDGPU code object: " << argv[1] << '\n';
    return 1;
  }

  rocjitsu::BinaryTranslator translator(*guest_arch, *host_arch, *host_mach);
  auto result = translator.translate(code_object);
  if (result.elf_bytes.empty()) {
    std::cerr << "translation produced an empty code object\n";
    return 1;
  }

  for (const auto &warning : result.warnings)
    std::cerr << "warning: " << warning << '\n';
  print_diagnostics(result.diagnostics);
  if (rocjitsu::has_error_diagnostic(result.diagnostics)) {
    std::cerr << "translation failed\n";
    return 1;
  }

  std::ofstream output(argv[2], std::ios::binary);
  if (!output) {
    std::cerr << "failed to open output file: " << argv[2] << '\n';
    return 1;
  }
  output.write(reinterpret_cast<const char *>(result.elf_bytes.data()),
               static_cast<std::streamsize>(result.elf_bytes.size()));
  if (!output) {
    std::cerr << "failed to write output file: " << argv[2] << '\n';
    return 1;
  }

  return 0;
}
