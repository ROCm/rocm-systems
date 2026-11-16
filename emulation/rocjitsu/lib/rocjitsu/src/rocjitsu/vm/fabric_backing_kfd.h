// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file fabric_backing_kfd.h
/// @brief Adapter from a retained KFD peer lease to a fabric backing.

#pragma once

#include "rocjitsu/kmd/linux/kfd_process.h"
#include "rocjitsu/vm/fabric_backing.h"

namespace rocjitsu {

[[nodiscard]] FabricBacking make_fabric_backing(KfdProcess::GpuBackingLease lease);

} // namespace rocjitsu
