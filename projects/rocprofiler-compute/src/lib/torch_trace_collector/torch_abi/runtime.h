// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

namespace torch_abi
{

/**
 * Checks the loaded libtorch_cpu and libc10 identities and stable AOTI ABI.
 * @return True only for an explicitly validated runtime artifact.
 */
bool runtime_is_supported() noexcept;

}  // namespace torch_abi
