// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/consan/target.h"
#include "rocjitsu/code/patch/consan/prepare.h"
#include "rocjitsu/code/patch/entry_prologue.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/target_registry.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

namespace rocjitsu::consan::test {

struct TargetMode {
  const char *target;
  uint32_t wave_size;
};

inline constexpr std::array kTargetModes = {
    TargetMode{"gfx942", 64},  TargetMode{"gfx950", 64},  TargetMode{"gfx1100", 32},
    TargetMode{"gfx1100", 64}, TargetMode{"gfx1201", 32}, TargetMode{"gfx1201", 64},
    TargetMode{"gfx1250", 32},
};

inline std::filesystem::path target_fixture_path(TargetMode mode, std::string_view fixture) {
#ifdef CONSAN_FIXTURE_DIR
  return std::filesystem::path(CONSAN_FIXTURE_DIR) / "targets" / mode.target /
         ("wave" + std::to_string(mode.wave_size)) / (std::string(fixture) + ".hsaco");
#else
  (void)mode;
  (void)fixture;
  return {};
#endif
}

inline std::vector<uint8_t> read_fixture(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

// Isolate the generic bootstrap for the pointer-restore mutation controls.
// Production ConSan delegates planning and placement to Instrumentor.
inline std::optional<DbiEntryProloguePlan> bootstrap_plan(const InputProgram &input) {
  AmdGpuCodeObject object(input.image.data(), input.image.size());
  auto decoder = input.target->public_id == ROCJITSU_CODE_TARGET_INVALID
                     ? Decoder::create(input.target->arch)
                     : Decoder::create(default_isa_target_registry(), input.target->public_id);
  if (!decoder)
    return std::nullopt;
  const std::array entries{input.kernel.entry_text_offset};
  auto blocks = BasicBlock::build_reachable(object, *decoder, input.target->arch, entries);
  if (blocks.failed())
    return std::nullopt;
  std::vector<BasicBlock *> scope;
  for (const auto &block : blocks.value())
    scope.push_back(block.get());
  RegisterSet reserved;
  reserved.expand({RegClass::SGPR, 0, static_cast<uint8_t>(input.original_sgprs)});
  return plan_dbi_entry_prologue(scope, input.kernel.descriptor, input.target->arch, 106, reserved);
}

} // namespace rocjitsu::consan::test
