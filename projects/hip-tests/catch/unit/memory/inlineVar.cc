/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// This test verifies the usage of inline constant variable in different translation units
// The inline variable is declared in MemUtils.hh, it is used in memoryCommon.cc where
// set_value and get_value are defined.

#include "MemUtils.hh"
#include "memoryCommon.hh"
using namespace mem_utils;

TEST_CASE(Unit_hipMemcpyToFromSymbol_GlobalConstVar) {
  int const initialValue = 10;
  set_value(initialValue);
  int const finalValue = get_value();
  REQUIRE(finalValue == initialValue);
}
