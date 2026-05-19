/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_context.hh>

#if defined(_WIN32)
#if defined(_WIN64)
typedef __int64 ssize_t;
#else   // !_WIN64
typedef __int32 ssize_t;
#endif  // !_WIN64
#endif  /*_WIN32*/

__global__ void test_kernel() {
  printf("%hd %hi\n", short(-42), short(-42));
  printf("%ld %li\n", -42l, -42l);
  printf("%lld %lli\n", -42ll, -42ll);
  printf("%hu %ho\n", ushort(42), ushort(42));
  printf("%lu %lo\n", 42l, 42l);
  printf("%llu %llo\n", 42ll, 42ll);
  printf("%hx %hX\n", ushort(42), ushort(42));
  printf("%lx %lX\n", 42l, 42l);
  printf("%llx %llX\n", 42ll, 42ll);
  printf("%lf\n", 123.456);
  printf("%lc\n", wint_t('x'));
#if HT_AMD
  const char* N = nullptr;
  printf("%lF\n", 123.456);
  printf("%hhd %hhi\n", char(-42), char(-42));
  printf("%jd %ji\n", intmax_t(-42l), intmax_t(-42l));
  printf("%zd %zi\n", ssize_t(-42l), ssize_t(-42l));
  printf("%td %ti\n", (ptrdiff_t)N, (ptrdiff_t)N);
  printf("%hhu %hho\n", static_cast<unsigned char>(42), static_cast<unsigned char>(42));
  printf("%ju %jo\n", uintmax_t(42l), uintmax_t(42l));
  printf("%zu %zo\n", size_t(42l), size_t(42l));
  printf("%tu %to\n", (ptrdiff_t)N, (ptrdiff_t)N);
#endif
}

int main() {
  test_kernel<<<1, 1>>>();
  static_cast<void>(hipDeviceSynchronize());
}