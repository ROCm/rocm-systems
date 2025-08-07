// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "api.hpp"
#include "core/categories.hpp"
#include "core/config.hpp"
#include "library/components/category_region.hpp"
#include "library/tracing.hpp"

extern "C" void
rocprofsys_progress_hidden(const char* _name)
{
    // mark the progress point
    rocprofsys::component::category_region<rocprofsys::category::causal>::mark<
        rocprofsys::quirk::causal>(_name);
}

extern "C" void
rocprofsys_annotated_progress_hidden(const char*              _name,
                                     rocprofsys_annotation_t* _annotations,
                                     size_t                   _annotation_count)
{
    // mark the progress point
    rocprofsys::component::category_region<rocprofsys::category::causal>::mark<
        rocprofsys::quirk::causal>(_name, _annotations, _annotation_count);
}
