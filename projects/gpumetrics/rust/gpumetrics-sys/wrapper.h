// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// bindgen entry point: pulls in the whole flat C API surface. capi.h transitively
// includes types.h and plugin_abi.h, which carry gpum_value / gpum_sample.

#include "gpumetrics/capi.h"
