/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip/hip_vector_types.h>

#include "../vector_types/vector_types_common.hh"

#include <string>
#include <sstream>
#include <type_traits>

using namespace std;  // NOLINT

template <typename V, enable_if_t<!is_integral<decltype(declval<V>().x)>{}>* = nullptr>
bool integer_unary_tests(V&, V&) {
  return true;
}

template <typename V, enable_if_t<!is_integral<decltype(declval<V>().x)>{}>* = nullptr>
bool integer_binary_tests(V&, V&, V&...) {
  return true;
}

template <typename V, enable_if_t<is_integral<decltype(declval<V>().x)>{}>* = nullptr>
void integer_unary_tests(V f1, V f2) {
  f1 %= f2;
  REQUIRE(f1 == MakeVector<V>(0));
  f1 &= f2;
  REQUIRE(f1 == MakeVector<V>(0));
  f1 |= f2;
  REQUIRE(f1 == MakeVector<V>(1));
  f1 ^= f2;
  REQUIRE(f1 == MakeVector<V>(0));
  f1 = MakeVector<V>(1);
  f1 <<= f2;
  REQUIRE(f1 == MakeVector<V>(2));
  f1 >>= f2;
  REQUIRE(f1 == MakeVector<V>(1));
  f2 = ~f1;
  REQUIRE(f2 == MakeVector<V>(~1));
}

template <typename V, enable_if_t<is_integral<decltype(declval<V>().x)>{}>* = nullptr>
void integer_binary_tests(V f1, V f2, V f3) {
  f3 = f1 % f2;
  REQUIRE(f3 == MakeVector<V>(0));
  f1 = f3 & f2;
  REQUIRE(f1 == MakeVector<V>(0));
  f2 = f1 ^ f3;
  REQUIRE(f2 == MakeVector<V>(0));
  f1 = MakeVector<V>(1);
  f2 = MakeVector<V>(2);
  f3 = f1 << f2;
  REQUIRE(f3 == MakeVector<V>(4));
  f2 = f3 >> f1;
  REQUIRE(f2 == MakeVector<V>(2));
}

template <typename V> bool constructor_tests() {
  if (is_constructible<V, unsigned char>{} && is_constructible<V, signed char>{} &&
      is_constructible<V, uint32_t>{} && is_constructible<V, int32_t>{} &&
      is_constructible<V, unsigned int>{} && is_constructible<V, signed int>{} &&
      is_constructible<V, uint64_t>{} && is_constructible<V, int64_t>{} &&
      is_constructible<V, uint64_t>{} && is_constructible<V, int64_t>{} &&
      is_constructible<V, float>{} && is_constructible<V, double>{}) {
    return true;
  }
}

template <typename V> bool TestVectorType() {
  const V v1 = MakeVector<V>(1);
  const V v2 = MakeVector<V>(2);
  const V v3 = MakeVector<V>(3);
  const V v4 = MakeVector<V>(4);

  V f1 = MakeVector<V>(1);
  V f2 = MakeVector<V>(1);
  V f3 = f1 + f2;
  REQUIRE((f3 == v2));
  f2 = f3 - f1;
  REQUIRE((f2 == v1));
  f1 = f2 * f3;
  REQUIRE((f1 == v2));
  f2 = f1 / f3;
  REQUIRE((f2 == v1));
  integer_binary_tests(f1, f2, f3);

  f1 = MakeVector<V>(2);
  f2 = MakeVector<V>(1);
  f1 += f2;
  REQUIRE((f1 == v3));
  f1 -= f2;
  REQUIRE((f1 == v2));
  f1 *= f2;
  REQUIRE((f1 == v2));
  f1 /= f2;
  REQUIRE((f1 == v2));

  integer_unary_tests(f1, f2);

  f1 = v2;
  f2 = f1++;
  REQUIRE((f1 == v3));
  REQUIRE((f2 == v2));
  f2 = f1--;
  REQUIRE((f2 == v3));
  REQUIRE((f1 == v2));
  f2 = ++f1;
  REQUIRE((f1 == v3));
  REQUIRE((f2 == v3));
  f2 = --f1;
  REQUIRE((f1 == v2));
  REQUIRE((f2 == v2));

  REQUIRE(constructor_tests<V>() == true);

  f1 = v3;
  f2 = v4;
  f3 = v3;
  REQUIRE(!(f1 == f2));
  REQUIRE((f1 != f2));

  using T = typename V::value_type;

  const T& x = f1.x;
  T& y = f2.x;
  const volatile T& z = f3.x;
  volatile T& w = f2.x;

  REQUIRE(x == MakeVector<V>(3));
  REQUIRE(y == MakeVector<V>(4));
  REQUIRE(z == MakeVector<V>(3));
  REQUIRE(w == MakeVector<V>(4));

  stringstream str;
  str << f1.x;
  str >> f2.x;

  REQUIRE(f1.x == f2.x);

  return true;
}

template <typename... Ts, enable_if_t<sizeof...(Ts) == 0>* = nullptr> bool TestVectorTypes() {
  return true;
}

template <typename T, typename... Ts> bool TestVectorTypes() {
  if (!TestVectorType<T>()) return false;
  return TestVectorTypes<Ts...>();
}

bool CheckVectorTypes() {
  return TestVectorTypes<char1, char2, char3, char4, uchar1, uchar2, uchar3, uchar4, short1, short2,
                         short3, short4, ushort1, ushort2, ushort3, ushort4, int1, int2, int3, int4,
                         uint1, uint2, uint3, uint4, long1, long2, long3, long4, ulong1, ulong2,
                         ulong3, ulong4, longlong1, longlong2, longlong3, longlong4, ulonglong1,
                         ulonglong2, ulonglong3, ulonglong4, float1, float2, float3, float4,
                         double1, double2, double3, double4>();
}
HIP_TEST_CASE(Unit_hipVectorTypes_test_on_host) {
  REQUIRE(sizeof(float1) == 4);
  REQUIRE(sizeof(float2) >= 8);
  REQUIRE(sizeof(float3) == 12);
  REQUIRE(sizeof(float4) >= 16);

  bool result = false;
  result = CheckVectorTypes();
  REQUIRE(result == true);
}
