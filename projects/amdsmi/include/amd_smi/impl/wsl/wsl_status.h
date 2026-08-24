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

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/wsl/wddm_types.h"

namespace wsl {

// Translates an NTSTATUS (as returned by D3DKMT/DXCore calls) into an
// amdsmi_status_t. Pure translation, no hsakmt dependency.
amdsmi_status_t ToAmdsmiStatus(NTSTATUS status);

}  // namespace wsl
