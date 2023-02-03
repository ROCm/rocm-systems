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

#include <gtest/gtest.h>

#include <vector>

#include "utils/helper.h"

TEST(WhenTrucatingLongKernelNames, KernelNameGetsTruncatedProperly) {
  std::string long_kernel_name =
      "void kernel_7r_3d_pml<32, 8, 4>(long long, long long, long long, int, "
      "int, int, long long, long long, long long, long long, long long, long "
      "long, long long, long long, long long, float, float, float, float "
      "const*, float*, float const*, float*, float const*) [clone .kd]";

  std::string trunkated_name = rocmtools::truncate_name(long_kernel_name);

  EXPECT_EQ("kernel_7r_3d_pml", trunkated_name);
}
