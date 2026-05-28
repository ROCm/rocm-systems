/*
Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef HELLORCCL_HPP
#define HELLORCCL_HPP
#include <iostream>

#define HIP_CALL(cmd)                                                 \
  do {                                                                \
    hipError_t error = (cmd);                                         \
    if (error != hipSuccess)                                          \
    {                                                                   \
      std::cerr << "Encountered HIP error (" << hipGetErrorString(error) << ") at line " \
                << __LINE__ << " in file " << __FILE__ << "\n";         \
      exit(-1);                                                         \
    }                                                                   \
  } while (0)

#define NCCL_CALL(cmd) \
  do { \
    ncclResult_t error = (cmd);                 \
    if (error != ncclSuccess)                   \
    {                                           \
      std::cerr << "Encountered NCCL error (" << ncclGetErrorString(error) << ") at line " \
                << __LINE__ << " in file " << __FILE__ << "\n";         \
      exit(-1);                                                         \
    }                                                                   \
  } while (0)

#endif
