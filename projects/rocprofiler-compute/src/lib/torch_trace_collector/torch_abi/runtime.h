// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "torch_abi.h"

namespace torch_abi
{

/**
 * Checks whether a pair of native-library build IDs is allowlisted.
 * @param torch_cpu_build_id GNU build ID for the loaded libtorch_cpu provider.
 * @param c10_build_id GNU build ID for the loaded libc10 provider.
 * @return True only for an explicitly validated artifact pair.
 */
[[nodiscard]] bool runtime_pair_is_supported(const GnuBuildId& torch_cpu_build_id,
                                             const GnuBuildId& c10_build_id) noexcept;

/**
 * Checks the loaded libtorch_cpu and libc10 GNU build identities.
 * @return True only for an explicitly validated runtime artifact pair.
 */
[[nodiscard]] bool runtime_is_supported() noexcept;

}  // namespace torch_abi
