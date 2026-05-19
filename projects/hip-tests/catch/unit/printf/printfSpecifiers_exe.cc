/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_context.hh>

__global__ void test_kernel() {
  const char* N = nullptr;
  const char* s = "hello world";
  printf("xyzzy\n");
  printf("%%\n");
  printf("hello %% world\n");
  printf("%%s\n");
  // Two special tests to make sure that the compiler pass correctly
  // skips over a '%%' without affecting the logic for locating
  // string arguments.
  printf("%%s%p\n", (void*)0xf01dab1eca55e77e);
  printf("%%c%s\n", "xyzzy");
  printf("%c%c%c\n", 's', 'e', 'p');
  printf("%d\n", -42);
  printf("%i\n", -42);
  printf("%u\n", 42);
  printf("%o\n", 42);
  printf("%x\n", 42);
  printf("%X\n", 42);
  printf("%f\n", 123.456);
#if HT_AMD
  printf("%F\n", -123.456);
#else
  printf("%f\n", -123.456);
#endif
  printf("%e\n", -123.456);
  printf("%E\n", 123.456);
  printf("%g\n", 123.456);
  printf("%G\n", -123.456);
  printf("%a\n", 123.456);
  printf("%A\n", -123.456);
  printf("%c\n", 'x');
  printf("%s\n", N);
  printf("%p\n", (void*)N);
#if HT_AMD
  printf("%.*f %*.*s %p\n", 8, 3.14159, 8, 5, s, (void*)0xf01dab1eca55e77e);
#else
  // In Cuda, printf doesn't support %.*, %*.*
  printf("%.8f %8.5s %p\n", 3.14159, s, (void*)0xf01dab1eca55e77e);
#endif
}

int main() {
  test_kernel<<<1, 1>>>();
  static_cast<void>(hipDeviceSynchronize());
  return 0;
}
