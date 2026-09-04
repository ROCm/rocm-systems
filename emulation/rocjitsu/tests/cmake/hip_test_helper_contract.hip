// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "hip_test_helper_contract.h"
#include "rocjitsu/base/rj_compiler.h"

static_assert(kHipTestHelperContractValue == 42);

RJ_NOINLINE int read_contract_value() { return kHipTestHelperContractValue; }

int main() { return read_contract_value() == 42 ? 0 : 1; }
