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

#ifndef SRC_PM4_GFX1250_PRIMITIVES_PROVIDER_HPP_
#define SRC_PM4_GFX1250_PRIMITIVES_PROVIDER_HPP_

// gfx1250_def.h brings in the gfx1250 register set and puts
// gfxip::gfx12::gfx1250::Primitives into scope via using-declarations.
#include "def/gfx1250_def.h"
#include "pm4/gfx12/gfx12_primitives_provider_base.hpp"

namespace pm4_builder {

class Gfx1250PrimitivesProvider
    : public Gfx12PrimitivesProviderBase<Primitives> {};

}  // namespace pm4_builder

#endif  // SRC_PM4_GFX1250_PRIMITIVES_PROVIDER_HPP_
