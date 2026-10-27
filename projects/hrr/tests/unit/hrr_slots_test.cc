/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Archive Slots
 * @{
 * @ingroup HRRTest
 * GPU-free unit tests for the helpers that keep archive pointers out of HIP: which JIT options
 * carry a pointer, and reading a string out of a fixed-size record field that
 * the archive did not NUL-terminate.
 */

#include "hrr_test_common.hh"
#include "hrr_jit_options.h"
#include "hrr_payload_bounds.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// The table in hrr_jit_options.h is keyed by the enumerator values; pin the
// ones that matter to the real enum so a renumbering fails to compile.
static_assert(hipJitOptionMaxRegisters == 0, "table assumes hipJitOption numbering");
static_assert(hipJitOptionWallTime == 2, "table assumes hipJitOption numbering");
static_assert(hipJitOptionInfoLogBuffer == 3, "table assumes hipJitOption numbering");
static_assert(hipJitOptionInfoLogBufferSizeBytes == 4, "table assumes hipJitOption numbering");
static_assert(hipJitOptionErrorLogBuffer == 5, "table assumes hipJitOption numbering");
static_assert(hipJitOptionErrorLogBufferSizeBytes == 6, "table assumes hipJitOption numbering");
static_assert(hipJitOptionGlobalSymbolNames == 17, "table assumes hipJitOption numbering");
static_assert(hipJitOptionGlobalSymbolAddresses == 18, "table assumes hipJitOption numbering");
static_assert(hipJitOptionGlobalSymbolCount == 19, "table assumes hipJitOption numbering");
static_assert(hipJitOptionOverrideDirectiveValues == 28, "table assumes hipJitOption numbering");
static_assert(hipJitOptionIRtoISAOptExt == 10000, "table assumes hipJitOption numbering");
static_assert(hipJitOptionIRtoISAOptCountExt == 10001, "table assumes hipJitOption numbering");

/**
 * Test Description
 * ----------------
 *   - The options whose value is a pointer (log buffers, symbol tables, the
 *     HIP-only option string) are not integer-valued; the integer ones are; a
 *     number the table does not know is refused.
 */
HRR_TEST_CASE(Unit_HRR_Slots_JitOptionClassification) {
  for (int o : {hipJitOptionWallTime, hipJitOptionInfoLogBuffer, hipJitOptionErrorLogBuffer,
                hipJitOptionGlobalSymbolNames, hipJitOptionGlobalSymbolAddresses,
                hipJitOptionIRtoISAOptExt})
    CHECK_FALSE(hrr::jit_option_value_is_integer(o));
  for (int o : {hipJitOptionMaxRegisters, hipJitOptionInfoLogBufferSizeBytes,
                hipJitOptionErrorLogBufferSizeBytes, hipJitOptionOptimizationLevel,
                hipJitOptionGlobalSymbolCount, hipJitOptionIRtoISAOptCountExt})
    CHECK(hrr::jit_option_value_is_integer(o));
  CHECK_FALSE(hrr::jit_option_value_is_integer(-1));
  CHECK_FALSE(hrr::jit_option_value_is_integer(30));
  CHECK_FALSE(hrr::jit_option_value_is_integer(99999));
}

/**
 * Test Description
 * ----------------
 *   - jit_options_carry_pointer looks at the first n options only, and an empty
 *     list carries none.
 */
HRR_TEST_CASE(Unit_HRR_Slots_JitOptionArray) {
  const hipJitOption ints[] = {hipJitOptionMaxRegisters, hipJitOptionOptimizationLevel};
  CHECK_FALSE(hrr::jit_options_carry_pointer(ints, 2));
  CHECK_FALSE(hrr::jit_options_carry_pointer(ints, 0));

  const hipJitOption mixed[] = {hipJitOptionMaxRegisters, hipJitOptionInfoLogBuffer};
  CHECK(hrr::jit_options_carry_pointer(mixed, 2));
  CHECK_FALSE(hrr::jit_options_carry_pointer(mixed, 1));
}

/**
 * Test Description
 * ----------------
 *   - bounded_cstr stops at the first NUL, takes the whole field when there is
 *     none, and never reads past the field: a buffer with no NUL at all, held
 *     in exactly-sized storage, comes back as exactly the field's bytes.
 */
HRR_TEST_CASE(Unit_HRR_Slots_BoundedCstr) {
  const uint8_t terminated[8] = {'a', 'b', 'c', 0, 'x', 'y', 'z', 0};
  CHECK(hrr::bounded_cstr(terminated, sizeof(terminated)) == "abc");

  std::vector<uint8_t> unterminated(256, 'A');  // exact-size heap block
  const std::string s = hrr::bounded_cstr(unterminated.data(), unterminated.size());
  CHECK(s.size() == 256);
  CHECK(s.find('\0') == std::string::npos);
  CHECK(std::strlen(s.c_str()) == 256);

  const uint8_t empty[4] = {0, 'x', 'y', 'z'};
  CHECK(hrr::bounded_cstr(empty, sizeof(empty)).empty());
  CHECK(hrr::bounded_cstr(empty, 0).empty());
}

/** @} */
