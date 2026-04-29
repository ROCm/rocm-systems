// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// Declarations for free-standing block table accessors provided by each GFX factory.
// These allow the HardwareArchitecture layer to obtain the block tables without
// including architecture-specific linux headers.

#ifndef SRC_CORE_FACTORY_BLOCK_TABLES_H_
#define SRC_CORE_FACTORY_BLOCK_TABLES_H_

#include <stddef.h>
#include "def/gpu_block_info.h"

namespace aql_profile {

const GpuBlockInfo** GetGfx9BlockTable();
size_t GetGfx9BlockTableSize();

const GpuBlockInfo** GetGfx10BlockTable();
size_t GetGfx10BlockTableSize();

const GpuBlockInfo** GetGfx11BlockTable();
size_t GetGfx11BlockTableSize();

}  // namespace aql_profile

#endif  // SRC_CORE_FACTORY_BLOCK_TABLES_H_
