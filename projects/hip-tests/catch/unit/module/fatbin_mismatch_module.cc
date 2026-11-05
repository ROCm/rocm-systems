/*
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
*/

/*
Fixture for Unit_FatBin_NoCompatibleCodeObject_ReturnsInvalidImage.

This is compiled by CMake into fatbin_mismatch_module.code targeting gfx900 and
gfx906 ONLY (two Vega archs), producing a multi-entry offload bundle. Loaded on
any non-Vega device, no bundle entry matches the device's ISA, which drives
FatBinaryInfo::ExtractFatBinaryUsingCOMGR through its "no compatible code object"
else-branch and must return hipErrorInvalidImage.
*/

#include <hip/hip_runtime.h>

extern "C" __global__ void mismatch_kernel(int* out) { *out = 1; }
