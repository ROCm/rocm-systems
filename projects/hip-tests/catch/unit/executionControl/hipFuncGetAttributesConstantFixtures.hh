/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <hip/hip_runtime_api.h>

namespace ConstantAttributeTest {
const void* NoConstantsKernel();
const void* MultipleConstantsKernel();
const void* MultipleConstantsReaderKernel();
hipError_t GetFirstConstantSize(size_t* size);
hipError_t GetInlineConstantSize(size_t* size);
}  // namespace ConstantAttributeTest
