// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/rj_gfx1250_b0_to_a0.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/dbt/binary_translator.h"

#include <cstdlib>
#include <cstring>
#include <new>

rj_status_t rj_gfx1250_b0_to_a0_translate(const void *source_elf, size_t source_size,
                                          uint8_t **translated_elf, size_t *translated_size) {
  if (translated_elf)
    *translated_elf = nullptr;
  if (translated_size)
    *translated_size = 0;

  if (!source_elf || source_size == 0 || !translated_elf || !translated_size)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;

  try {
    const auto *source_bytes = static_cast<const uint8_t *>(source_elf);
    rocjitsu::AmdGpuCodeObject source(source_bytes, source_size);
    if (!source.is_valid() || source.target_id() != ROCJITSU_CODE_TARGET_GFX1250)
      return ROCJITSU_STATUS_INVALID_CODE_OBJECT;

    rocjitsu::BinaryTranslatorOptions options;
    options.input_revision = rocjitsu::ProcessorRevision::Gfx1250B0;
    options.output_revision = rocjitsu::ProcessorRevision::Gfx1250A0;
    rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250, 0,
                                          options);
    auto result = translator.translate(source);
    if (result.elf_bytes.empty() || !result.dispatchable())
      return ROCJITSU_STATUS_ERROR;

    auto *output = static_cast<uint8_t *>(std::malloc(result.elf_bytes.size()));
    if (!output)
      return ROCJITSU_STATUS_OUT_OF_RESOURCES;

    std::memcpy(output, result.elf_bytes.data(), result.elf_bytes.size());
    *translated_elf = output;
    *translated_size = result.elf_bytes.size();
    return ROCJITSU_STATUS_SUCCESS;
  } catch (const std::bad_alloc &) {
    return ROCJITSU_STATUS_OUT_OF_RESOURCES;
  } catch (...) {
    return ROCJITSU_STATUS_ERROR;
  }
}

void rj_gfx1250_b0_to_a0_free(void *translated_elf) { std::free(translated_elf); }
