// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCATTACH_FIXTURE_VALUE
#    define ROCATTACH_FIXTURE_VALUE 0
#endif

#ifdef ROCATTACH_FIXTURE_PADDED
// Force this fixture to place the attach symbol at a different offset than the
// normal fixture, while keeping the exported attach ABI the same.
extern "C" __attribute__((visibility("default"), noinline, used)) int
rocattach_layout_padding_function(int value)
{
    volatile int result = value;
    for(int i = 0; i < 1024; ++i)
    {
        result += i;
    }
    return result;
}
#endif

extern "C" __attribute__((visibility("default"))) int
rocprofiler_register_attach()
{
    return ROCATTACH_FIXTURE_VALUE;
}

extern "C" __attribute__((visibility("default"))) int
rocprofiler_register_detach()
{
    return -ROCATTACH_FIXTURE_VALUE;
}
