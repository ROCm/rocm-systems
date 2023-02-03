/* Copyright (c) 2022 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#ifndef SAMPLES_PCSAMPLER_CODE_PRINTING_SAMPLE_PROGRAM_OPTIONS_HPP_
#define SAMPLES_PCSAMPLER_CODE_PRINTING_SAMPLE_PROGRAM_OPTIONS_HPP_

#include <chrono>
#include <cstdint>

struct program_options {
    program_options()
    : device(0)
    , no_gpu(false)
    , hip_memset(false)
    , rands_len(1024 * 1024 * 4)
    , gt(0)
    , seed(std::chrono::steady_clock::now().time_since_epoch().count())
    , disassemble(false)
    , pc_sampling(false)
    {}

    int device;
    bool no_gpu;
    bool hip_memset;
    size_t rands_len;
    uint64_t gt;
    uint64_t seed;
    bool disassemble;
    bool pc_sampling;
};

#endif // SAMPLES_PCSAMPLER_CODE_PRINTING_SAMPLE_PROGRAM_OPTIONS_HPP_
