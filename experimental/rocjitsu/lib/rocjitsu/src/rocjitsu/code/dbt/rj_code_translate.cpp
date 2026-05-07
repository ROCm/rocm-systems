// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/rj_code_internal.h"

#include "rocjitsu/code/dbt/binary_translator.h"

#include <string>
#include <utility>

namespace {

thread_local std::string g_last_error;

void set_last_error(std::string message) { g_last_error = std::move(message); }

} // namespace

const char *rj_code_last_error(void) { return g_last_error.c_str(); }

rj_status_t rj_code_translate(const rj_code_object_t *source, const rj_code_dbt_options_t *options,
                              rj_code_object_t **translated) {
  set_last_error("");
  if (!source || !source->co || !options || !translated) {
    set_last_error("invalid rj_code_translate argument");
    return ROCJITSU_STATUS_ERROR;
  }
  *translated = nullptr;

  rocjitsu::BinaryTranslator translator(options->guest_arch, options->host_arch,
                                        options->target_mach);
  auto result = translator.translate(*source->co);

  if (!result.warnings.empty()) {
    std::string message = "DBT emitted warnings; refusing to export partially translated object";
    for (const auto &warning : result.warnings) {
      message += "\n  ";
      message += warning;
    }
    set_last_error(std::move(message));
    return ROCJITSU_STATUS_ERROR;
  }

  if (result.elf_bytes.empty()) {
    set_last_error("DBT produced an empty code object");
    return ROCJITSU_STATUS_ERROR;
  }

  auto owned = std::make_unique<rocjitsu::AmdGpuCodeObject>(result.elf_bytes.data(),
                                                            result.elf_bytes.size());
  if (!owned->is_valid()) {
    set_last_error("DBT produced an invalid AMDGPU code object");
    return ROCJITSU_STATUS_ERROR;
  }

  auto *obj = new rj_code_object_t{};
  obj->co = owned.get();
  obj->owned_co = std::move(owned);
  *translated = obj;
  return ROCJITSU_STATUS_SUCCESS;
}
