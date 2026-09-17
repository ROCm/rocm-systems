// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "../patch/consan/consan_final_validation_test_support.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_lowering.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <span>

namespace {

void require(bool condition) {
  if (!condition)
    std::abort();
}

void exercise_transform(std::span<const uint8_t> input, const rocjitsu::consan::Options &options) {
  const rocjitsu::consan::TransformArtifacts result = rocjitsu::consan::lower(input, options);
  require(result.program_inventory.code_object_id() ==
          rocjitsu::consan::make_code_object_id(input));
  if (result.outcome == rocjitsu::consan::TransformOutcome::ModifiedValid) {
    require(!result.replacement.empty());
    require(!result.patches.empty());
    require(rocjitsu::consan::validate_modified_elf(input, result).empty());

    if (std::getenv("RJ_CONSAN_FUZZ_REQUIRE_UNMATCHED_WAIT_ABORT")) {
      const rocjitsu::consan::PatchInfo *abort_patch = nullptr;
      for (const rocjitsu::consan::PatchInfo &patch : result.patches) {
        if (patch.kind != rocjitsu::consan::PatchKind::InlineMalformedBarrierAbort)
          continue;
        require(abort_patch == nullptr);
        abort_patch = &patch;
      }
      require(abort_patch != nullptr);
      const rocjitsu::AmdGpuCodeObject original(input.data(), input.size());
      require(original.is_valid());
      require(original.text_sections().size() == 1);
      const uint64_t file_offset =
          original.text_sections().front()->sectionOffset() + abort_patch->anchor_offset;
      require(file_offset + sizeof(uint32_t) <= input.size());
      require(file_offset + sizeof(uint32_t) <= result.replacement.size());
      uint32_t original_word = 0;
      uint32_t replacement_word = 0;
      std::memcpy(&original_word, input.data() + file_offset, sizeof(original_word));
      std::memcpy(&replacement_word, result.replacement.data() + file_offset,
                  sizeof(replacement_word));
      require(original_word == 0xBF94FFFFu);    // s_barrier_wait -1
      require(replacement_word == 0xBFB00000u); // s_endpgm
    }
  } else {
    require(result.replacement.empty());
    require(result.patches.empty());
  }
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size > (1u << 20u))
    return 0;

  const std::span<const uint8_t> input(data, size);
#if defined(RJ_CONSAN_FUZZ_PROFILE_SUPERCOLLIDER)
  rocjitsu::consan::Options supercollider;
  supercollider.mode = rocjitsu::consan::Mode::SuperCollider;
  supercollider.probe_lds_check_trap = true;
  supercollider.probe_flat_check_trap = true;
  supercollider.max_patches = 8;
  supercollider.abort_unmatched_barrier_wait = true;
  exercise_transform(input, supercollider);
#elif defined(RJ_CONSAN_FUZZ_PROFILE_DEFAULT)
  rocjitsu::consan::Options options;
  options.mode = rocjitsu::consan::Mode::Default;
  options.abort_unmatched_barrier_wait = true;
  exercise_transform(input, options);
#else
#error "A single RJ_CONSAN_FUZZ_PROFILE_* definition is required"
#endif
  return 0;
}
