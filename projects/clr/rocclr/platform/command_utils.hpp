/* Copyright (c) 2021 - 2021 Advanced Micro Devices, Inc.

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

#ifndef COMMAND_UTILS_HPP_
#define COMMAND_UTILS_HPP_

// Dummy command types for Stream Wait and Write commands.
#define ROCCLR_COMMAND_STREAM_WAIT_VALUE 0x4501
#define ROCCLR_COMMAND_STREAM_WRITE_VALUE 0x4502
#define ROCCLR_COMMAND_BATCH_STREAM 0x4503

// Stream Wait Value Conidtions
#define ROCCLR_STREAM_WAIT_VALUE_GTE 0x0
#define ROCCLR_STREAM_WAIT_VALUE_EQ 0x1
#define ROCCLR_STREAM_WAIT_VALUE_AND 0x2
#define ROCCLR_STREAM_WAIT_VALUE_NOR 0x3

#define ROCCLR_HIP_GL_CONTEXT_KHR 0x2100
#define ROCCLR_HIP_GLX_DISPLAY_KHR 0x2101
#define ROCCLR_HIP_WGL_HDC_KHR 0x2102

#endif  // COMMAND_UTILS_HPP_