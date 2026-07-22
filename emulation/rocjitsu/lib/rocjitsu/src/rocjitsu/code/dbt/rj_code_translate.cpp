// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/rj_code_internal.h"

#include "rocjitsu/code/dbt/binary_translator.h"
#include "rocjitsu/code/dbt/processor_revision.h"

#include <cstring>
#include <optional>
#include <type_traits>

namespace {

// Map a C revision enum to the internal one. Returns nullopt for any value
// outside the known closed set so an unknown (e.g. future or garbage) value is
// rejected rather than silently treated as Unspecified — which would skip every
// gfx1250 workaround via same-ISA identity copying.
//
// The field is read through its underlying integer type: rj_code_revision_t is a
// plain C enum, so an lvalue-to-rvalue load of an out-of-range value (garbage,
// uninitialized, or from a newer producer) is undefined behavior. memcpy reads
// the bytes without an enum-typed load, then the raw value is validated.
[[nodiscard]] std::optional<rocjitsu::ProcessorRevision>
to_processor_revision(const rj_code_revision_t &field) {
  std::underlying_type_t<rj_code_revision_t> raw;
  std::memcpy(&raw, &field, sizeof(raw));
  switch (raw) {
  case ROCJITSU_CODE_REVISION_UNSPECIFIED:
    return rocjitsu::ProcessorRevision::Unspecified;
  case ROCJITSU_CODE_REVISION_GFX1250_A0:
    return rocjitsu::ProcessorRevision::Gfx1250A0;
  case ROCJITSU_CODE_REVISION_GFX1250_B0:
    return rocjitsu::ProcessorRevision::Gfx1250B0;
  }
  return std::nullopt;
}

} // namespace

rj_status_t rj_code_translate(const rj_code_object_t *source, const rj_code_dbt_options_t *options,
                              rj_code_object_t **translated) {
  if (!source || !source->co || !options || !translated)
    return ROCJITSU_STATUS_ERROR;

  // Reject unknown raw revision values before any policy decision.
  const auto input_revision = to_processor_revision(options->input_revision);
  const auto output_revision = to_processor_revision(options->output_revision);
  if (!input_revision || !output_revision)
    return ROCJITSU_STATUS_ERROR;

  // gfx1250 A0 and B0 share an ELF machine ID, so a same-architecture gfx1250
  // translation is meaningless without both revisions: fail closed rather than
  // pass the object through unchanged (which would silently skip any required
  // workarounds). Revisions must only be given for gfx1250.
  const bool guest_is_gfx1250 = options->guest_arch == ROCJITSU_CODE_ARCH_GFX1250;
  const bool host_is_gfx1250 = options->host_arch == ROCJITSU_CODE_ARCH_GFX1250;
  if (guest_is_gfx1250 != (*input_revision != rocjitsu::ProcessorRevision::Unspecified))
    return ROCJITSU_STATUS_ERROR;
  if (host_is_gfx1250 != (*output_revision != rocjitsu::ProcessorRevision::Unspecified))
    return ROCJITSU_STATUS_ERROR;

  rocjitsu::BinaryTranslatorOptions translator_options;
  translator_options.input_revision = *input_revision;
  translator_options.output_revision = *output_revision;
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
