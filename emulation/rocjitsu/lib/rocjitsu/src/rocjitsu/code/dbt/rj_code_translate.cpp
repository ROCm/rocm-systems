// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/rj_code_internal.h"

#include "rocjitsu/code/dbt/binary_translator.h"

rj_status_t rj_code_translate(const rj_code_object_t *source, const rj_code_dbt_options_t *options,
                              rj_code_object_t **translated) {
  if (!source || !source->co || !options || !translated)
    return ROCJITSU_STATUS_ERROR;

  rocjitsu::BinaryTranslator translator(options->guest_arch, options->host_arch);
  auto result = translator.translate(*source->co);

  if (result.elf_bytes.empty())
    return ROCJITSU_STATUS_ERROR;

  auto owned = std::make_unique<rocjitsu::AmdGpuCodeObject>(result.elf_bytes.data(),
                                                            result.elf_bytes.size());
  if (!owned->is_valid())
    return ROCJITSU_STATUS_ERROR;

  auto *obj = new rj_code_object_t{};
  obj->co = owned.get();
  obj->owned_co = std::move(owned);
  *translated = obj;
  return ROCJITSU_STATUS_SUCCESS;
}
