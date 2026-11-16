// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "target.h"

#include <array>

namespace rocjitsu::isa_diff {
namespace {

struct Row {
  std::string_view name;
  uint32_t mach; ///< EF_AMDGPU_MACH_AMDGCN_* (llvm/BinaryFormat/ELF.h)
  rj_code_arch_t arch;
  unsigned wave_size; ///< Default wave size.
  unsigned major;
  bool unified_vgprs;
  bool wave32_ok, wave64_ok;
};

constexpr std::array kTargets = std::to_array<Row>({
    {"gfx908", 0x30, ROCJITSU_CODE_ARCH_CDNA1, 64, 9, false, false, true},
    {"gfx90a", 0x3f, ROCJITSU_CODE_ARCH_CDNA2, 64, 9, true, false, true},
    {"gfx940", 0x40, ROCJITSU_CODE_ARCH_CDNA3, 64, 9, true, false, true},
    {"gfx941", 0x4b, ROCJITSU_CODE_ARCH_CDNA3, 64, 9, true, false, true},
    {"gfx942", 0x4c, ROCJITSU_CODE_ARCH_CDNA3, 64, 9, true, false, true},
    {"gfx950", 0x4f, ROCJITSU_CODE_ARCH_CDNA4, 64, 9, true, false, true},
    {"gfx1010", 0x33, ROCJITSU_CODE_ARCH_RDNA1, 32, 10, false, true, true},
    {"gfx1011", 0x34, ROCJITSU_CODE_ARCH_RDNA1, 32, 10, false, true, true},
    {"gfx1012", 0x35, ROCJITSU_CODE_ARCH_RDNA1, 32, 10, false, true, true},
    {"gfx1013", 0x42, ROCJITSU_CODE_ARCH_RDNA1, 32, 10, false, true, true},
    {"gfx1030", 0x36, ROCJITSU_CODE_ARCH_RDNA2, 32, 10, false, true, true},
    {"gfx1031", 0x37, ROCJITSU_CODE_ARCH_RDNA2, 32, 10, false, true, true},
    {"gfx1032", 0x38, ROCJITSU_CODE_ARCH_RDNA2, 32, 10, false, true, true},
    {"gfx1033", 0x39, ROCJITSU_CODE_ARCH_RDNA2, 32, 10, false, true, true},
    {"gfx1034", 0x3e, ROCJITSU_CODE_ARCH_RDNA2, 32, 10, false, true, true},
    {"gfx1035", 0x3d, ROCJITSU_CODE_ARCH_RDNA2, 32, 10, false, true, true},
    {"gfx1036", 0x45, ROCJITSU_CODE_ARCH_RDNA2, 32, 10, false, true, true},
    {"gfx1100", 0x41, ROCJITSU_CODE_ARCH_RDNA3, 32, 11, false, true, true},
    {"gfx1101", 0x46, ROCJITSU_CODE_ARCH_RDNA3, 32, 11, false, true, true},
    {"gfx1102", 0x47, ROCJITSU_CODE_ARCH_RDNA3, 32, 11, false, true, true},
    {"gfx1103", 0x44, ROCJITSU_CODE_ARCH_RDNA3, 32, 11, false, true, true},
    {"gfx1150", 0x43, ROCJITSU_CODE_ARCH_RDNA3_5, 32, 11, false, true, true},
    {"gfx1151", 0x4a, ROCJITSU_CODE_ARCH_RDNA3_5, 32, 11, false, true, true},
    {"gfx1152", 0x55, ROCJITSU_CODE_ARCH_RDNA3_5, 32, 11, false, true, true},
    {"gfx1153", 0x58, ROCJITSU_CODE_ARCH_RDNA3_5, 32, 11, false, true, true},
    {"gfx1200", 0x48, ROCJITSU_CODE_ARCH_RDNA4, 32, 12, false, true, true},
    {"gfx1201", 0x4e, ROCJITSU_CODE_ARCH_RDNA4, 32, 12, false, true, true},
    {"gfx1250", 0x49, ROCJITSU_CODE_ARCH_CDNA5, 32, 12, false, true, false},
    {"gfx1251", 0x5a, ROCJITSU_CODE_ARCH_CDNA5, 32, 12, false, true, false},
});

Target make(const Row &r) {
  Target t;
  t.name = std::string(r.name);
  t.arch = r.arch;
  t.wave_size = r.wave_size;
  t.major = r.major;
  t.unified_vgprs = r.unified_vgprs;
  return t;
}

const Row *row(std::string_view name) {
  for (const Row &r : kTargets)
    if (r.name == name)
      return &r;
  return nullptr;
}

} // namespace

std::optional<Target> target_by_name(std::string_view name) {
  const Row *r = row(name);
  return r ? std::optional<Target>(make(*r)) : std::nullopt;
}

std::optional<Target> target_by_elf_mach(uint32_t mach) {
  for (const Row &r : kTargets)
    if (r.mach == mach)
      return make(r);
  return std::nullopt;
}

std::optional<Target> with_wave_size(Target t, unsigned wave_size) {
  const Row *r = row(t.name);
  if (!r || (wave_size == 32 && !r->wave32_ok) || (wave_size == 64 && !r->wave64_ok) ||
      (wave_size != 32 && wave_size != 64))
    return std::nullopt;
  t.wave_size = wave_size;
  return t;
}

std::optional<Target> parse_target(std::string_view text) {
  const size_t colon = text.find(':');
  auto t = target_by_name(text.substr(0, colon));
  if (!t || colon == std::string_view::npos)
    return t;
  const std::string_view w = text.substr(colon + 1);
  if (w == "w32" || w == "wave32")
    return with_wave_size(*t, 32);
  if (w == "w64" || w == "wave64")
    return with_wave_size(*t, 64);
  return std::nullopt;
}

std::string supported_targets() {
  std::string s;
  for (const Row &r : kTargets)
    s += (s.empty() ? "" : ",") + std::string(r.name);
  return s;
}

} // namespace rocjitsu::isa_diff
