/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * Compiled as C (not C++). hip/library_types.h provides the *_t names in C
 * through a `#ifndef __cplusplus` typedef block; C++ test builds never see
 * that block, so this translation unit checks it compiles and the values match
 * the documented (CUDA-compatible) values.
 */

#include <hip/library_types.h>

_Static_assert(HIP_EMULATION_STRATEGY_DEFAULT == 0, "HIP_EMULATION_STRATEGY_DEFAULT");
_Static_assert(HIP_EMULATION_STRATEGY_PERFORMANT == 1, "HIP_EMULATION_STRATEGY_PERFORMANT");
_Static_assert(HIP_EMULATION_STRATEGY_EAGER == 2, "HIP_EMULATION_STRATEGY_EAGER");
_Static_assert(HIP_EMULATION_MANTISSA_CONTROL_DYNAMIC == 0,
               "HIP_EMULATION_MANTISSA_CONTROL_DYNAMIC");
_Static_assert(HIP_EMULATION_MANTISSA_CONTROL_FIXED == 1, "HIP_EMULATION_MANTISSA_CONTROL_FIXED");
_Static_assert(HIP_EMULATION_SPECIAL_VALUES_SUPPORT_NONE == 0,
               "HIP_EMULATION_SPECIAL_VALUES_SUPPORT_NONE");
_Static_assert(HIP_EMULATION_SPECIAL_VALUES_SUPPORT_INFINITY == 1,
               "HIP_EMULATION_SPECIAL_VALUES_SUPPORT_INFINITY");
_Static_assert(HIP_EMULATION_SPECIAL_VALUES_SUPPORT_NAN == 2,
               "HIP_EMULATION_SPECIAL_VALUES_SUPPORT_NAN");
_Static_assert(HIP_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT == 0xFFFF,
               "HIP_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT");

/* Returns 1 if all types are usable from C and hold the expected values. */
int hipEmulationTypesC(void) {
  /* Both the typedef names and the *_t names must be usable in C. */
  hipEmulationStrategy_t strategy = HIP_EMULATION_STRATEGY_EAGER;
  hipEmulationStrategy strategy2 = HIP_EMULATION_STRATEGY_DEFAULT;
  enum hipEmulationStrategy_t strategy3 = HIP_EMULATION_STRATEGY_PERFORMANT;

  hipEmulationMantissaControl_t mantissa = HIP_EMULATION_MANTISSA_CONTROL_FIXED;
  hipEmulationMantissaControl mantissa2 = HIP_EMULATION_MANTISSA_CONTROL_DYNAMIC;

  hipEmulationSpecialValuesSupport_t special =
      (hipEmulationSpecialValuesSupport_t)(HIP_EMULATION_SPECIAL_VALUES_SUPPORT_INFINITY |
                                           HIP_EMULATION_SPECIAL_VALUES_SUPPORT_NAN);
  hipEmulationSpecialValuesSupport special2 = HIP_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT;

  return strategy == 2 && strategy2 == 0 && strategy3 == 1 && mantissa == 1 && mantissa2 == 0 &&
         special == 3 && special2 == 0xFFFF;
}
