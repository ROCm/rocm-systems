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


#ifndef TEST_CTRL_TEST_ASSERT_H_
#define TEST_CTRL_TEST_ASSERT_H_

#define TEST_ASSERT(cond)                                                                    \
  {                                                                                          \
    if (!(cond)) {                                                                           \
      std::cerr << "Assert failed(" << #cond << ") at " << __FILE__ << ", line " << __LINE__ \
                << std::endl;                                                                \
      exit(-1);                                                                              \
    }                                                                                        \
  }

#define TEST_STATUS(cond)                                                              \
  {                                                                                    \
    if (!(cond)) {                                                                     \
      std::cerr << "Test error at " << __FILE__ << ", line " << __LINE__ << std::endl; \
      const char* message;                                                             \
      rocprofiler_error_string(&message);                                              \
      std::cerr << "ERROR: " << message << std::endl;                                  \
      exit(-1);                                                                        \
    }                                                                                  \
  }

#endif  // TEST_CTRL_TEST_ASSERT_H_
