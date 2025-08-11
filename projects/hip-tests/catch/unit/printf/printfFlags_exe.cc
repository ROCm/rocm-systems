/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#include <hip_test_context.hh>

__global__ void test_kernel() {
  printf("%08d\n", 42);
  printf("%08i\n", -42);
  printf("%08u\n", 42);
  printf("%08g\n", 123.456);
  printf("%0+8d\n", 42);
  printf("%+d\n", -42);
  printf("%+08d\n", 42);
  printf("%-8s\n", "xyzzy");
  printf("% i\n", -42);
  printf("% i\n", 42);
  printf("%-16.8d\n", 42);
  printf("%16.8d\n", 42);
  printf("%#o\n", 42);
  printf("%#x\n", 42);
  printf("%#X\n", 42);
#if HT_AMD
  printf("%#F\n", 42.);
#else
  printf("%#f\n", 42.);
#endif
  printf("%#e\n", 42.);
  printf("%#E\n", 42.);
  printf("%#g\n", 42.);
  printf("%#G\n", 42.);
  printf("%#a\n", 42.);
  printf("%#A\n", 42.);
}

int main() {
  test_kernel<<<1, 1>>>();
  static_cast<void>(hipDeviceSynchronize());
}