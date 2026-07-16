// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "hip_test_helper_contract.h"

static_assert(kHipTestHelperContractValue == 42);

int main() { return kHipTestHelperContractValue == 42 ? 0 : 1; }
