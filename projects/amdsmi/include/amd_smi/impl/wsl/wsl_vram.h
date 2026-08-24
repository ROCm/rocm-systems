/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 */

#pragma once

#include <cstdint>

#include "amd_smi/impl/wsl/wsl_adapter.h"

namespace wsl {

// Total VRAM capacity in bytes: local heap (visible+invisible) plus, for
// integrated/APU parts, non-local (system) heap.
uint64_t VramTotal(const WslAdapterInfo& info);

// Currently-used VRAM in bytes (local + non-local as applicable). Returns
// false if the underlying D3DKMT queries fail.
bool QueryVramUsage(const WslAdapterInfo& info, uint64_t* usage_bytes);

// Available VRAM in bytes = VramTotal() - QueryVramUsage(), floored at 0.
// Waits on the paging queue's fence via D3DKMTWaitForSynchronizationObjectFromCpu
// before querying usage, matching WDDMDevice::VramAvail()'s behavior.
bool VramAvailable(const WslAdapterInfo& info, uint64_t* available_bytes);

}  // namespace wsl
