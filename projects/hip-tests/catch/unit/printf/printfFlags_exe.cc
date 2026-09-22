/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
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