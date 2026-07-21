// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan.h"

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

void exercise_transform(std::span<const uint8_t> input, const rocjitsu::ConSanOptions &options) {
  const rocjitsu::ConSanResult result = rocjitsu::try_patch_consan(input, options);
  require(result.visited_code_object);
  require(result.input_size == input.size());
  if (result.outcome == rocjitsu::ConSanTransformOutcome::ModifiedValid) {
    require(result.modified);
    require(result.final_validation_passed);
    require(!result.elf_bytes.empty());
    require(!result.patches.empty());
    require(rocjitsu::validate_consan_modified_elf(input, result).empty());
    require(rocjitsu::consan_install_action(result, false) ==
            rocjitsu::ConSanInstallAction::LoadReplacement);

    if (std::getenv("RJ_CONSAN_FUZZ_REQUIRE_UNMATCHED_WAIT_ABORT")) {
      const rocjitsu::ConSanPatchInfo *abort_patch = nullptr;
      for (const rocjitsu::ConSanPatchInfo &patch : result.patches) {
        if (patch.kind != rocjitsu::ConSanPatchKind::InlineMalformedBarrierAbort)
          continue;
        require(abort_patch == nullptr);
        abort_patch = &patch;
      }
      require(abort_patch != nullptr);
      require(result.text_sections.size() == 1);
      const uint64_t file_offset =
          result.text_sections.front().file_offset + abort_patch->anchor_offset;
      require(file_offset + sizeof(uint32_t) <= input.size());
      require(file_offset + sizeof(uint32_t) <= result.elf_bytes.size());
      uint32_t original_word = 0;
      uint32_t replacement_word = 0;
      std::memcpy(&original_word, input.data() + file_offset, sizeof(original_word));
      std::memcpy(&replacement_word, result.elf_bytes.data() + file_offset,
                  sizeof(replacement_word));
      require(original_word == 0xBF94FFFFu);    // s_barrier_wait -1
      require(replacement_word == 0xBFB00000u); // s_endpgm
    }
  } else {
    require(!result.modified);
    require(!result.final_validation_passed);
    require(result.elf_bytes.empty());
    require(result.patches.empty());
    require(rocjitsu::consan_install_action(result, false) ==
            rocjitsu::ConSanInstallAction::LoadOriginal);
  }
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size > (1u << 20u))
    return 0;

  const std::span<const uint8_t> input(data, size);
#if defined(RJ_CONSAN_FUZZ_PROFILE_SUPERCOLLIDER)
  rocjitsu::ConSanOptions supercollider;
  supercollider.flavor = rocjitsu::ConSanFlavor::SuperCollider;
  supercollider.probe_lds_check_trap = true;
  supercollider.probe_flat_check_trap = true;
  supercollider.probe_trampoline_nop = true;
  supercollider.max_patches = 8;
  supercollider.abort_unmatched_barrier_wait = true;
  exercise_transform(input, supercollider);
#elif defined(RJ_CONSAN_FUZZ_PROFILE_RECORD_REPLAY)
  rocjitsu::ConSanOptions moi;
  moi.flavor = rocjitsu::ConSanFlavor::Moi;
  moi.moi_engine = rocjitsu::ConSanMoiEngine::RecordReplay;
  moi.abort_unmatched_barrier_wait = true;
  exercise_transform(input, moi);
#elif defined(RJ_CONSAN_FUZZ_PROFILE_INLINE_SHADOW)
  rocjitsu::ConSanOptions moi;
  moi.flavor = rocjitsu::ConSanFlavor::Moi;
  moi.moi_engine = rocjitsu::ConSanMoiEngine::InlineShadow;
  moi.abort_unmatched_barrier_wait = true;
  exercise_transform(input, moi);
#elif defined(RJ_CONSAN_FUZZ_PROFILE_SAMPLED)
  rocjitsu::ConSanOptions moi;
  moi.flavor = rocjitsu::ConSanFlavor::Moi;
  moi.moi_engine = rocjitsu::ConSanMoiEngine::Sampled;
  moi.abort_unmatched_barrier_wait = true;
  exercise_transform(input, moi);
#else
#error "A single RJ_CONSAN_FUZZ_PROFILE_* definition is required"
#endif
  return 0;
}
