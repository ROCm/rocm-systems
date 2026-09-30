// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>

#include "addc/lib_c.h"

// Library-local test seam: swaps the addc entry points the CPER adapter calls, so unit tests can
// reach the guards real addc never triggers (no context, a misreported count, a throw). Not
// amdsmi_-prefixed, so the linker version script keeps it out of libamd_smi.so (no public ABI);
// tests reach it through the static archive.
struct AddcApi_t {
  auto (*get_context)() -> addc_context_t*;
  auto (*get_summary)(addc_context_t*, const uint8_t*, size_t, const char*,
                      addc_cper_summary_entry_t*, size_t*, addc_error_info_t*) -> addc_status_t;
  auto (*decode_json)(addc_context_t*, const uint8_t*, size_t, const char*, char*, size_t*,
                      addc_error_info_t*) -> addc_status_t;
};

// Passing nullptr restores addc. Not thread-safe: only the single-threaded tests call it.
auto addc_set_api_for_testing(const AddcApi_t* api) -> void;
