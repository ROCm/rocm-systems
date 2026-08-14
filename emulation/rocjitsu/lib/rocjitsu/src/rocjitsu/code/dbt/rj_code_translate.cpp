// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/rj_code_internal.h"

#include "rocjitsu/code/dbt/binary_translator.h"

rj_status_t rj_code_translate(const rj_code_object_t *source, const rj_code_dbt_options_t *options,
                              rj_code_object_t **translated) {
  if (!translated)
    return ROCJITSU_STATUS_ERROR;
  *translated = nullptr;
  if (!source || !source->co || !options)
    return ROCJITSU_STATUS_ERROR;

  try {
    // This C entry point carries no silicon revision (see rj_code_dbt_options_t).
    // A same-architecture gfx1250 translation is direction-ambiguous without one, so
    // BinaryTranslator fails it closed internally. Revision-aware B0->A0 translation
    // is reached through the C++ BinaryTranslator directly (DBT hook and CLI).
    rocjitsu::BinaryTranslator translator(options->guest_arch, options->host_arch);
    auto result = translator.translate(*source->co);

    // dispatchable() implies ok(): reject both error-diagnostic translations and
    // non-dispatchable skipped-kernel artifacts (an s_endpgm stub that completes
    // normally without producing the kernel's outputs) rather than handing back a code
    // object that would silently produce wrong results if executed. This matches
    // the CLI and the HSA load hook, the other consumers of translate().
    if (result.elf_bytes.empty() || !result.dispatchable())
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
  } catch (...) {
    return ROCJITSU_STATUS_ERROR;
  }
}
