// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

namespace amd::roc::aql_resident {
// Built with ROCclr's existing internal blit program. The backend owns its
// kernel object; it does not need another compiler or HIP module loader.
const char* kernelSource();
constexpr const char* kBindingKernelName = "__amd_rocclr_resident_binding_fixup";
}  // namespace amd::roc::aql_resident
