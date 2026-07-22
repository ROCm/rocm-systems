// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/rj_code_internal.h"

#include "rocjitsu/code/dbt/binary_translator.h"
#include "rocjitsu/code/dbt/processor_revision.h"

namespace {

[[nodiscard]] rocjitsu::ProcessorRevision to_processor_revision(rj_code_revision_t revision) {
  switch (revision) {
  case ROCJITSU_CODE_REVISION_GFX1250_A0:
    return rocjitsu::ProcessorRevision::Gfx1250A0;
  case ROCJITSU_CODE_REVISION_GFX1250_B0:
    return rocjitsu::ProcessorRevision::Gfx1250B0;
  case ROCJITSU_CODE_REVISION_UNSPECIFIED:
  default:
    return rocjitsu::ProcessorRevision::Unspecified;
  }
}

} // namespace

rj_status_t rj_code_translate(const rj_code_object_t *source, const rj_code_dbt_options_t *options,
                              rj_code_object_t **translated) {
  if (!source || !source->co || !options || !translated)
    return ROCJITSU_STATUS_ERROR;

  // gfx1250 A0 and B0 share an ELF machine ID, so a same-architecture gfx1250
  // translation is meaningless without both revisions: fail closed rather than
  // pass the object through unchanged (which would silently skip any required
  // workarounds). Revisions must only be given for gfx1250.
  const bool guest_is_gfx1250 = options->guest_arch == ROCJITSU_CODE_ARCH_GFX1250;
  const bool host_is_gfx1250 = options->host_arch == ROCJITSU_CODE_ARCH_GFX1250;
  if (guest_is_gfx1250 != (options->input_revision != ROCJITSU_CODE_REVISION_UNSPECIFIED))
    return ROCJITSU_STATUS_ERROR;
  if (host_is_gfx1250 != (options->output_revision != ROCJITSU_CODE_REVISION_UNSPECIFIED))
    return ROCJITSU_STATUS_ERROR;

  rocjitsu::BinaryTranslatorOptions translator_options;
  translator_options.input_revision = to_processor_revision(options->input_revision);
  translator_options.output_revision = to_processor_revision(options->output_revision);
  rocjitsu::BinaryTranslator translator(options->guest_arch, options->host_arch, 0,
                                        translator_options);
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
}
