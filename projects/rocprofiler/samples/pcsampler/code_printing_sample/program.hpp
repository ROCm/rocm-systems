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

#ifndef SAMPLES_PCSAMPLER_CODE_PRINTING_SAMPLE_PROGRAM_HPP_
#define SAMPLES_PCSAMPLER_CODE_PRINTING_SAMPLE_PROGRAM_HPP_

#define PROGNAME "code_printing_sample"

#define HIP_ERROR(code)                                                 \
    do {                                                                \
        fprintf(stderr,                                                 \
                PROGNAME ": Assertion failed at %s:%d, HIP error: %s\n", \
                __FILE__, __LINE__, hipGetErrorString((code)));         \
        fflush(stderr);                                                 \
    } while (false);

#define HIP_CHECK_BREAK(expr, var)                                      \
    if (auto const code = (expr); hipSuccess != code) {                 \
        HIP_ERROR(code);                                                \
        (var) = code;                                                   \
        break;                                                          \
    }

#define ROCPROFILER_ERROR(code)                                           \
    do {                                                                \
        fprintf(stderr,                                                 \
                PROGNAME ": Assertion failed at %s:%d, ROCProfiler error: %s\n", \
                __FILE__, __LINE__, rocprofiler_error_str(code));      \
        fflush(stderr);                                                 \
    } while (false);

#define ROCPROFILER_CHECK(expr, var)                                      \
    if ((var) = (expr); ROCPROFILER_STATUS_SUCCESS != (var)) {            \
        ROCPROFILER_ERROR((var));                                         \
    }

#endif // SAMPLES_PCSAMPLER_CODE_PRINTING_SAMPLE_PROGRAM_HPP_
