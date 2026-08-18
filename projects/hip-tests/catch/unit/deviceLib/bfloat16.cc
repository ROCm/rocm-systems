/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip/hip_bf16.h>

#include <cmath>
#include <memory>
#include <limits>
#include <algorithm>

// Struct used to generate floats from combination of various componenets
union float_holder {
  float fp32;
  struct parts_ {
    unsigned int fp32_mantisa : 16;  // ignored for bf16
    unsigned int bf16_mantisa : 7;   // bf16 mantisa
    unsigned int exponent : 8;
    unsigned int sign : 1;
  } parts;
  unsigned int u32;
};

std::vector<float> getAllBF16() {
  constexpr unsigned char max_mantissa = std::numeric_limits<unsigned char>::max() >> 1;  // 7 bits
  const size_t max_bf16_num =
      2 /*sign*/ * std::pow(2, 8) /*exponent*/ * std::pow(2, 7) /*mantissa*/;

  std::vector<float> f_in;
  f_in.reserve(max_bf16_num);


  for (size_t s = 0; s <= 1; s++) {                                            // sign
    for (size_t e = 0; e <= std::numeric_limits<unsigned char>::max(); e++) {  // expo
      for (size_t m = 0; m <= max_mantissa; m++) {                             // man
        float_holder hold;
        hold.u32 = 0;  // Init - clear all bits

        hold.parts.sign = s;
        hold.parts.exponent = e;
        hold.parts.bf16_mantisa = m;

        f_in.push_back(hold.fp32);
      }
    }
  }
  return f_in;
}

enum MathOp { Add = 0, Sub, Mul, Div, LastOp = Div };

__device__ __hip_bfloat16 bf16_math(__hip_bfloat16 a, __hip_bfloat16 b, MathOp op) {
  switch (op) {
    case Add:
      return __hadd(a, b);
    case Sub:
      return __hsub(a, b);
    case Mul:
      return __hmul(a, b);
    case Div:
      return __hdiv(a, b);
  }
}

__device__ float fp32_math(float a, float b, MathOp op) {
  switch (op) {
    case Add:
      return a + b;
    case Sub:
      return a - b;
    case Mul:
      return a * b;
    case Div:
      return a / b;
  }
}

__device__ void bf16_math_op_kernel(float* a, float* b, float* c) {
  for (int i = Add; i < LastOp; i++) {
    auto op = static_cast<MathOp>(i);
    c[i] = __bfloat162float(bf16_math(__float2bfloat16(a[i]), __float2bfloat16(b[i]), op));
  }
}

__device__ void fp32_math_op_kernel(float* a, float* b, float* c) {
  for (int i = Add; i < LastOp; i++) {
    auto op = static_cast<MathOp>(i);
    c[i] = fp32_math(a[i], b[i], op);
  }
}

// c = a MathOp b in fp32
// conv_res = a MathOp b in bf16 converted back to fp32
__global__ void do_math(float* a, float* b, float* c, float* conv_res) {
  fp32_math_op_kernel(a, b, c);
  bf16_math_op_kernel(a, b, conv_res);
}

// Convert float -> bfloat16 -> float
__global__ void fp32_bf16_fp32(float* a, float* c, size_t size) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    auto b = __float2bfloat16(a[i]);
    c[i] = __bfloat162float(b);
  }
}

__device__ unsigned bool_to_unsigned(bool in) { return in ? 1u : 0u; }

// Test equal compare
__global__ void bf16_is_equal(float* val, unsigned* res, size_t size) {
  auto i = threadIdx.x;
  if (i < size) {
    auto v1 = __float2bfloat16(val[i]);
    auto v2 = __float2bfloat16(val[i]);
    res[i] =
        bool_to_unsigned((__heq(v1, v2) && __hge(v1, v2) &&
                          __hle(v1, v2)));  // Equal, Greater Equal, Less Equal should all have true
  }
}

// Test other compare functions
__global__ void bf16_compare(float* val, unsigned* res, size_t size) {
  auto i = threadIdx.x;
  if (i < size) {
    __hip_bfloat16 v1 = __float2bfloat16(val[i]);
    __hip_bfloat16 v2 = __float2bfloat16(val[i]);

    v1 = __hadd(v1, v2);                          // v1 = v1 + v2
    bool r1 = (__hlt(v2, v1) && __hgt(v1, v2) &&  // v1 > v2
               __hne(v1, v2) &&                   // v1 != v2
               __heq(__hmax(v1, v2), v1) &&       // max(v1,v2) == v1
               __heq(__hmin(v1, v2), v2));        // min(v1,v2) == v2

    v1 = __hsub(v1, v2);      // Back to v1's original value
    bool r2 = __heq(v1, v2);  // v1 == v2

    v1 = __hmul(v1, v2);                  // v1 = v1 * v2, both have same values so square it
    bool r3 = __heq(v1, __hmul(v2, v2));  // v1 == (v2 * v2)

    v1 = hsqrt(v1);           // Back to v1's original value
    bool r4 = __heq(v1, v2);  // v1 == v2

    v1 = __hdiv(v1, v2);                          // v1 = v1/v2, both have same values
    bool r5 = __heq(v1, __float2bfloat16(1.0f));  // v1 == 1.0f

    // Uncomment to debug
    // printf("%u - %u - %u - %u - %u\n", bool_to_unsigned(r1), bool_to_unsigned(r2),
    //       bool_to_unsigned(r3), bool_to_unsigned(r4), bool_to_unsigned(r5));

    res[i] = bool_to_unsigned(r1 && r2 && r3 && r4 && r5);
  }
}

// Order of predicates written by bf16_compare_nan(); the host uses the same
// indices to label failures and look up the expected result.
enum Bf16NanPredicate {
  kHeq, kHne, kHlt, kHle, kHgt, kHge,        // ordered scalar   -> false for NaN
  kHequ, kHneu, kHltu, kHleu, kHgtu, kHgeu,  // unordered scalar -> true for NaN
  kHbeq2, kHbne2,                            // ordered bool2    -> false for NaN
  kBf16NanPredicateCount
};

// Test NaN comparison semantics: for inputs where at least one operand is NaN the
// ordered predicates (no 'u' suffix) must return false and the unordered predicates
// ('u' suffix) must return true. Each thread writes one slot per predicate so the
// host can identify exactly which intrinsic misbehaved. Each a[i]/b[i] pair must
// have at least one NaN.
__global__ void bf16_compare_nan(const float* a, const float* b, unsigned* res, size_t size) {
  auto i = threadIdx.x;
  if (i < size) {
    const __hip_bfloat16 x = __float2bfloat16(a[i]);
    const __hip_bfloat16 y = __float2bfloat16(b[i]);
    const __hip_bfloat162 x2{x, x};
    const __hip_bfloat162 y2{y, y};
    unsigned* out = res + i * kBf16NanPredicateCount;

    out[kHeq] = bool_to_unsigned(__heq(x, y));
    out[kHne] = bool_to_unsigned(__hne(x, y));
    out[kHlt] = bool_to_unsigned(__hlt(x, y));
    out[kHle] = bool_to_unsigned(__hle(x, y));
    out[kHgt] = bool_to_unsigned(__hgt(x, y));
    out[kHge] = bool_to_unsigned(__hge(x, y));

    out[kHequ] = bool_to_unsigned(__hequ(x, y));
    out[kHneu] = bool_to_unsigned(__hneu(x, y));
    out[kHltu] = bool_to_unsigned(__hltu(x, y));
    out[kHleu] = bool_to_unsigned(__hleu(x, y));
    out[kHgtu] = bool_to_unsigned(__hgtu(x, y));
    out[kHgeu] = bool_to_unsigned(__hgeu(x, y));

    out[kHbeq2] = bool_to_unsigned(__hbeq2(x2, y2));
    out[kHbne2] = bool_to_unsigned(__hbne2(x2, y2));
  }
}

// Order of scalar operator results written by bf16_nan_operators().
enum Bf16NanOp {
  kOpNeQnan, kOpNeSnan,  // operator!= : unordered -> true for NaN
  kOpEqQnan, kOpEqSnan,  // operator== : ordered   -> false for NaN
  kOpNeOne, kOpEqOne,    // sanity: normal value
  kBf16NanOpCount
};

__global__ void bf16_nan_operators(unsigned* out) {
  const __hip_bfloat16 qnan = __ushort_as_bfloat16((unsigned short)0x7FC0U);  // quiet NaN
  const __hip_bfloat16 snan = __ushort_as_bfloat16((unsigned short)0x7FA0U);  // signaling NaN
  const __hip_bfloat16 one = __ushort_as_bfloat16((unsigned short)0x3F80U);   // 1.0

  out[kOpNeQnan] = bool_to_unsigned(qnan != qnan);
  out[kOpNeSnan] = bool_to_unsigned(snan != snan);
  out[kOpEqQnan] = bool_to_unsigned(qnan == qnan);
  out[kOpEqSnan] = bool_to_unsigned(snan == snan);
  out[kOpNeOne] = bool_to_unsigned(one != one);
  out[kOpEqOne] = bool_to_unsigned(one == one);
}

// Convert to bits
__global__ void bf16_conv_bits(float* val, unsigned short* res, size_t size) {
  auto i = threadIdx.x;
  if (i < size) {
    __hip_bfloat16 v1 = __float2bfloat16(val[i]);
    res[i] = __bfloat16_as_ushort(v1);
  }
}

__global__ void bf16_neg(float* in, float* out, size_t size) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = __bfloat162float(__hneg(__float2bfloat16(in[i])));
  }
}

__global__ void bf16_to_short(float* in, short* s_res, unsigned short* u_res, size_t size) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    s_res[i] = __bfloat16_as_short(__float2bfloat16(in[i]));
    u_res[i] = __bfloat16_as_ushort(__float2bfloat16(in[i]));
  }
}

__global__ void short_to_bf16(short* in, float* out, size_t size) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = __bfloat162float(__short_as_bfloat16(in[i]));
  }
}

__global__ void ushort_to_bf16(unsigned short* in, float* out, size_t size) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = __bfloat162float(__ushort_as_bfloat16(in[i]));
  }
}

__global__ void bf16_fma(float* in1, float* in2, float plus_y, float* out, size_t size) {
  int i = threadIdx.x;
  auto y_bf = __float2bfloat16(plus_y);
  if (i < size) {
    auto in1_bf = __float2bfloat16(in1[i]);
    auto in2_bf = __float2bfloat16(in2[i]);
    auto res_bf = __hfma(in1_bf, in2_bf, y_bf);
    out[i] = res_bf;  // convert back to float
  }
}

HIP_TEST_CASE(Unit_bf16_basic) {
  auto f_in = getAllBF16();
  auto max_bf16_num = f_in.size();

  SECTION("Conversion float to bfloat16 to float") {
    constexpr size_t size = 256;

    float *d_a, *d_c;
    HIP_CHECK(hipMalloc(&d_a, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&d_c, sizeof(float) * size));

    auto h_a = std::make_unique<float[]>(size);
    auto h_c = std::make_unique<float[]>(size);

    for (size_t i = 0; i < size; i++) {
      h_a[i] = i + 1.25;
    }

    HIP_CHECK(hipMemcpy(d_a, h_a.get(), sizeof(float) * size, hipMemcpyHostToDevice));

    fp32_bf16_fp32<<<1, size>>>(d_a, d_c, size);

    HIP_CHECK(hipMemcpy(h_c.get(), d_c, sizeof(float) * size, hipMemcpyDeviceToHost));

    for (size_t i = 0; i < size; i++) {
      INFO("Initial: " << h_a[i] << " - After Conv: " << h_c[i]);
      // The relative error should be less than 1/(2^7) since bfloat16 has 7 bits mantissa.
      REQUIRE((std::fabs(h_c[i] - h_a[i]) / h_a[i]) < (1.0 / 128.0f));
    }

    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_c));
  }

  SECTION("Math Op Accuracy") {
    constexpr size_t size = static_cast<size_t>(LastOp);

    float f_val1[size], f_val2[size], f_res[size], bf_res[size];
    for (size_t i = 0; i < size; i++) {
      f_val1[i] = i + 1.50;
      f_val2[i] = i + 1.25;
    }

    float *df_val1, *df_val2, *float_res, *bf16_res;
    HIP_CHECK(hipMalloc(&df_val1, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&df_val2, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&float_res, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&bf16_res, sizeof(float) * size));

    HIP_CHECK(hipMemcpy(df_val1, f_val1, sizeof(float) * size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(df_val2, f_val2, sizeof(float) * size, hipMemcpyHostToDevice));

    do_math<<<1, 1>>>(df_val1, df_val2, float_res, bf16_res);

    HIP_CHECK(hipMemcpy(f_res, float_res, sizeof(float) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(bf_res, bf16_res, sizeof(float) * size, hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(df_val1));
    HIP_CHECK(hipFree(df_val2));
    HIP_CHECK(hipFree(float_res));
    HIP_CHECK(hipFree(bf16_res));

    for (size_t i = 0; i < size; i++) {
      INFO("FP res: " << f_res[i] << " - BF16 res: " << bf_res[i]);
      // The relative error should be less than 1/(2^7) since bfloat16 has 7 bits mantissa.
      REQUIRE((std::fabs(bf_res[i] - f_res[i]) / f_res[i]) < (1.0 / 128.0f));
    }
  }

  SECTION("Equal bfloat16") {
    constexpr size_t size = 5;
    float in[size] = {1.0f, 0.5f, -0.33333f, 0.0f, -0.0f}, *d_in;
    unsigned* d_res;
    HIP_CHECK(hipMalloc(&d_in, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&d_res, sizeof(unsigned) * size));

    HIP_CHECK(hipMemcpy(d_in, in, sizeof(float) * size, hipMemcpyHostToDevice));

    bf16_is_equal<<<1, size>>>(d_in, d_res, size);

    std::vector<unsigned> res(size, 0);
    HIP_CHECK(hipMemcpy(res.data(), d_res, sizeof(unsigned) * size, hipMemcpyDeviceToHost));

    for (size_t i = 0; i < res.size(); i++) {
      INFO("Index: " << i << " input: " << in[i] << " output: " << res[i]);
      REQUIRE(res[i] == 1);
    }

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_res));
  }

  SECTION("MathOp Compare") {
    constexpr size_t size = 7;
    float in[size] = {0.5f, 1.0f, 1.5f, 0.33333f, 2.5f, 3.0f, 3.5f}, *d_in;
    unsigned* d_res;
    HIP_CHECK(hipMalloc(&d_in, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&d_res, sizeof(unsigned) * size));

    HIP_CHECK(hipMemcpy(d_in, in, sizeof(float) * size, hipMemcpyHostToDevice));

    bf16_compare<<<1, size>>>(d_in, d_res, size);

    std::vector<unsigned> res(size, 0);
    HIP_CHECK(hipMemcpy(res.data(), d_res, sizeof(unsigned) * size, hipMemcpyDeviceToHost));

    for (size_t i = 0; i < res.size(); i++) {
      INFO("Index: " << i << " input: " << in[i] << " output: " << res[i]);
      REQUIRE(res[i] == 1);
    }

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_res));
  }

  SECTION("NaN comparison semantics") {
    struct PredicateCase {
      const char* name;
      unsigned expected;
    };
    // Indexed by Bf16NanPredicate; drives both the failure label and the
    // expected result so the two cannot drift out of alignment.
    static const PredicateCase kPredicates[kBf16NanPredicateCount] = {
        {"__heq", 0}, {"__hne", 0}, {"__hlt", 0}, {"__hle", 0},
        {"__hgt", 0}, {"__hge", 0},
        {"__hequ", 1}, {"__hneu", 1}, {"__hltu", 1}, {"__hleu", 1},
        {"__hgtu", 1}, {"__hgeu", 1},
        {"__hbeq2", 0}, {"__hbne2", 0}};

    const float qnan = std::numeric_limits<float>::quiet_NaN();
    constexpr size_t size = 6;
    // Each pair has at least one NaN.
    float a[size] = {qnan, qnan, 1.0f, qnan, -2.0f, qnan};
    float b[size] = {qnan, 1.0f, qnan, -3.0f, qnan, 0.0f};
    float *d_a, *d_b;
    unsigned* d_res;
    HIP_CHECK(hipMalloc(&d_a, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&d_b, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&d_res, sizeof(unsigned) * size * kBf16NanPredicateCount));

    HIP_CHECK(hipMemcpy(d_a, a, sizeof(float) * size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, b, sizeof(float) * size, hipMemcpyHostToDevice));

    bf16_compare_nan<<<1, size>>>(d_a, d_b, d_res, size);

    std::vector<unsigned> res(size * kBf16NanPredicateCount, 0);
    HIP_CHECK(hipMemcpy(res.data(), d_res,
                        sizeof(unsigned) * res.size(), hipMemcpyDeviceToHost));

    for (size_t i = 0; i < size; i++) {
      for (int p = 0; p < kBf16NanPredicateCount; p++) {
        CAPTURE(i, a[i], b[i], kPredicates[p].name);
        CHECK(res[i * kBf16NanPredicateCount + p] == kPredicates[p].expected);
      }
    }

    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_res));
  }

  SECTION("NaN operator semantics") {
    struct OpCase {
      const char* name;
      unsigned expected;
    };
    static const OpCase kOps[kBf16NanOpCount] = {
        {"qnan != qnan", 1}, {"snan != snan", 1},
        {"qnan == qnan", 0}, {"snan == snan", 0},
        {"one != one", 0},   {"one == one", 1}};

    unsigned* d_res;
    HIP_CHECK(hipMalloc(&d_res, sizeof(unsigned) * kBf16NanOpCount));

    bf16_nan_operators<<<1, 1>>>(d_res);

    std::vector<unsigned> res(kBf16NanOpCount, 0);
    HIP_CHECK(hipMemcpy(res.data(), d_res, sizeof(unsigned) * res.size(),
                        hipMemcpyDeviceToHost));

    for (int p = 0; p < kBf16NanOpCount; p++) {
      CAPTURE(kOps[p].name);
      CHECK(res[p] == kOps[p].expected);
    }

    HIP_CHECK(hipFree(d_res));
  }

  SECTION("Bits equal") {
    constexpr size_t size = 5;
    float in[size] = {1.0f, 0.5f, 0.33333f, 3.38e38f, 3.40e38f}, *d_in;
    unsigned short* d_res;
    HIP_CHECK(hipMalloc(&d_in, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&d_res, sizeof(unsigned short) * size));

    HIP_CHECK(hipMemcpy(d_in, in, sizeof(float) * size, hipMemcpyHostToDevice));

    bf16_conv_bits<<<1, size>>>(d_in, d_res, size);

    std::vector<unsigned short> res_cmp = {0x3f80, 0x3f00, 0x3eab, 0x7f7e, 0x7f80 /*Inf*/};
    std::vector<unsigned short> res(size, 0);
    HIP_CHECK(hipMemcpy(res.data(), d_res, sizeof(unsigned short) * size, hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_res));
    for (size_t i = 0; i < res.size(); i++) {
      INFO("Index: " << i << " input: " << in[i] << " expected: " << res_cmp[i]
                     << " result: " << res[i]);
      REQUIRE(abs(static_cast<int>(res_cmp[i] - res[i])) <= 2);
    }
  }

  SECTION("Round trip equal") {
    constexpr size_t size = 7;
    float *d_in, *d_out;
    std::vector<float> in = {
        std::numeric_limits<float>::infinity(), -1.0f, -0.5f, -0.0f, 0.0f, 0.5f, 1.0f};
    HIP_CHECK(hipMalloc(&d_in, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&d_out, sizeof(float) * size));

    HIP_CHECK(hipMemcpy(d_in, in.data(), sizeof(float) * size, hipMemcpyHostToDevice));

    fp32_bf16_fp32<<<1, size>>>(d_in, d_out, size);

    std::vector<float> res(size, 0.0f);

    HIP_CHECK(hipMemcpy(res.data(), d_out, sizeof(unsigned) * size, hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    REQUIRE(in == res);
  }

  SECTION("Round trip subsection") {
    float *in, *out;
    HIP_CHECK(hipMalloc(&in, sizeof(float) * max_bf16_num));
    HIP_CHECK(hipMalloc(&out, sizeof(float) * max_bf16_num));
    HIP_CHECK(hipMemcpy(in, f_in.data(), sizeof(float) * max_bf16_num, hipMemcpyHostToDevice));

    fp32_bf16_fp32<<<(max_bf16_num / 256) + 1, 256>>>(in, out, max_bf16_num);  // round-trip

    std::vector<float> f_out(f_in.size(), 0.0f);
    HIP_CHECK(hipMemcpy(f_out.data(), out, sizeof(float) * max_bf16_num, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(in));
    HIP_CHECK(hipFree(out));

    REQUIRE(f_in.size() == f_out.size());  // Size should be equal
    for (size_t i = 0; i < f_in.size(); i++) {
      INFO("Initial: " << f_in[i] << " After Conversion: " << f_out[i]);
      if (std::isnan(f_in[i])) {  // NaNs can't be compared
        REQUIRE(std::isnan(f_out[i]));
      } else {
        REQUIRE(f_in[i] == f_out[i]);
      }
    }
  }

  SECTION("Neg Subsection") {
    float *in, *out;
    HIP_CHECK(hipMalloc(&in, sizeof(float) * max_bf16_num));
    HIP_CHECK(hipMalloc(&out, sizeof(float) * max_bf16_num));
    HIP_CHECK(hipMemcpy(in, f_in.data(), sizeof(float) * max_bf16_num, hipMemcpyHostToDevice));

    bf16_neg<<<(max_bf16_num / 256) + 1, 256>>>(in, out, max_bf16_num);  // round-trip

    std::vector<float> f_out(f_in.size(), 0.0f);
    HIP_CHECK(hipMemcpy(f_out.data(), out, sizeof(float) * max_bf16_num, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(in));
    HIP_CHECK(hipFree(out));

    REQUIRE(f_in.size() == f_out.size());  // Size should be equal
    for (size_t i = 0; i < f_in.size(); i++) {
      INFO("Initial: " << f_in[i] << " After Conversion: " << f_out[i]);
      if (std::isnan(f_in[i])) {  // NaNs can't be compared
        REQUIRE(std::isnan(f_out[i]));
      } else {
        REQUIRE(f_in[i] == -f_out[i]);
      }
    }
  }

  SECTION("fma") {
    std::vector<float> in1, in2;
    constexpr size_t size = 32;
    in1.reserve(size);
    in2.reserve(size);
    for (size_t i = 1; i <= size; i++) {
      in1.push_back(i * 0.5f);
      in2.push_back(i * 0.6f);
    }
    float *d_in1, *d_in2, *d_out;
    HIP_CHECK(hipMalloc(&d_in1, sizeof(float) * in1.size()));
    HIP_CHECK(hipMalloc(&d_in2, sizeof(float) * in2.size()));
    HIP_CHECK(hipMalloc(&d_out, sizeof(float) * in2.size()));

    HIP_CHECK(hipMemcpy(d_in1, in1.data(), sizeof(float) * in1.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_in2, in2.data(), sizeof(float) * in2.size(), hipMemcpyHostToDevice));

    bf16_fma<<<1, size>>>(d_in1, d_in2, 1.0f, d_out, size);
    std::vector<float> gpu_res(size, 0.0f);
    HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(float) * in2.size(), hipMemcpyDeviceToHost));

    for (size_t i = 0; i < size; i++) {
      auto expected = in1[i] * in2[i] + 1.0f;
      INFO("iter: " << i << " Expected: " << expected << " got: " << gpu_res[i]);
      REQUIRE(std::fabs(expected - gpu_res[i]) <= 1.5f);
    }

    HIP_CHECK(hipFree(d_in1))
    HIP_CHECK(hipFree(d_in2));
    HIP_CHECK(hipFree(d_out));
  }

  SECTION("abs") {
    std::vector<float> in = {-1.0f, -0.0f, +0.0f, +1.0f};
    for (const auto i : in) {
      auto bf = __float2bfloat16(i);
      auto bf_abs = __habs(bf);
      float cvt_back = bf_abs;
      INFO("Original: " << i << " expected: " << std::fabs(i) << " got: " << cvt_back);
      REQUIRE(std::fabs(i) == cvt_back);
    }
  }
}

template <typename Type> __global__ void bf16_cvt_to_integral(Type* in, float* out, size_t size) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = __hip_bfloat16(in[i]);
  }
}

HIP_TEMPLATE_TEST_CASE(Unit_bf16_conversion_to_integral_type, unsigned short, short, int,
                   unsigned int) {
  constexpr TestType start = std::is_unsigned<TestType>::value
                                 ? std::numeric_limits<unsigned short>::min()
                                 : std::numeric_limits<short>::min();
  constexpr TestType end = std::is_unsigned<TestType>::value
                               ? std::numeric_limits<unsigned short>::max()
                               : std::numeric_limits<short>::max();
  const size_t size = (start < 0) ? end - start : end + start;

  TestType* d_input;
  float* d_res;
  HIP_CHECK(hipMalloc(&d_input, sizeof(TestType) * size));
  HIP_CHECK(hipMalloc(&d_res, sizeof(float) * size));

  std::vector<float> res, gpu_res;
  std::vector<TestType> input;
  input.reserve(size);
  gpu_res.reserve(size);
  res.reserve(size);
  for (TestType i = start; i < end; i++) {
    input.push_back(i);
    res.push_back(static_cast<float>(i));
    gpu_res.push_back(0.0f);
  }

  HIP_CHECK(
      hipMemcpy(d_input, input.data(), sizeof(TestType) * input.size(), hipMemcpyHostToDevice));
  auto cvt_kernel = bf16_cvt_to_integral<TestType>;
  uint32_t blocks = static_cast<uint32_t>(size / 256) + 1;
  cvt_kernel<<<blocks, 256>>>(d_input, d_res, size);
  HIP_CHECK(hipMemcpy(gpu_res.data(), d_res, sizeof(float) * res.size(), hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(d_res));
  HIP_CHECK(hipFree(d_input));

  for (size_t i = 0; i < size; i++) {
    if (!(std::isnan(res[i]) || std::isnan(gpu_res[i]))) {
      INFO("lhs: " << gpu_res[i] << " rhs: " << res[i]);
      if (gpu_res[i] != res[i]) CHECK((std::fabs(gpu_res[i] - res[i]) / res[i]) < (1.0 / 128.0f));
    }
  }
}

__global__ void bf162_eq(float* in, char* out, size_t size) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    auto temp = __bfloat162bfloat162(__float2bfloat16(in[i]));
    out[i] = __hbequ2(temp, temp) ? 1 : 0;
  }
}

__global__ void bf162_neq(float* in, char* out, size_t size) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    __hip_bfloat16 val = in[i];
    __hip_bfloat16 other_val = __hne(__float2bfloat16(1.0f), val) ? 1.0f : 2.0f;
    __hip_bfloat162 temp1(val, other_val);
    __hip_bfloat162 temp2(other_val, val);
    out[i] = (__hbne2(temp1, temp2)) ? 1 : 0;
  }
}

HIP_TEST_CASE(Unit_bf162_basic) {
  auto f_in = getAllBF16();
  auto max_bf16_num = f_in.size();

  SECTION("Eq Operation") {
    float* in;
    HIP_CHECK(hipMalloc(&in, sizeof(float) * max_bf16_num));
    HIP_CHECK(hipMemcpy(in, f_in.data(), sizeof(float) * max_bf16_num, hipMemcpyHostToDevice));
    char* out;
    HIP_CHECK(hipMalloc(&out, sizeof(char) * max_bf16_num));
    bf162_eq<<<(max_bf16_num / 256) + 1, 256>>>(in, out, max_bf16_num);
    std::vector<char> result(max_bf16_num, 0);
    HIP_CHECK(hipMemcpy(result.data(), out, sizeof(char) * max_bf16_num, hipMemcpyDeviceToHost));
    // Cant use allof, incase of mismatch we need to show which value had a mismatch
    for (size_t i = 0; i < max_bf16_num; i++) {
      if (!std::isnan(f_in[i])) {
        INFO("Comparing: " << f_in[i] << " for iter: " << i);
        REQUIRE(result[i] == 1);
      }
    }
    HIP_CHECK(hipFree(in));
    HIP_CHECK(hipFree(out));
  }

  SECTION("Neq Operation") {
    float* in;
    HIP_CHECK(hipMalloc(&in, sizeof(float) * max_bf16_num));
    HIP_CHECK(hipMemcpy(in, f_in.data(), sizeof(float) * max_bf16_num, hipMemcpyHostToDevice));
    char* out;
    HIP_CHECK(hipMalloc(&out, sizeof(char) * max_bf16_num));
    bf162_neq<<<(max_bf16_num / 256) + 1, 256>>>(in, out, max_bf16_num);
    std::vector<char> result(max_bf16_num, 0);
    HIP_CHECK(hipMemcpy(result.data(), out, sizeof(char) * max_bf16_num, hipMemcpyDeviceToHost));
    // Cant use allof, incase of mismatch we need to show which value had a mismatch
    for (size_t i = 0; i < max_bf16_num; i++) {
      if (!std::isnan(f_in[i])) {
        INFO("Comparing: " << f_in[i] << " for iter: " << i << " result: " << (int)result[i]);
        REQUIRE(result[i] == 1);
      }
    }
    HIP_CHECK(hipFree(in));
    HIP_CHECK(hipFree(out));
  }
}


HIP_TEST_CASE(Unit_bf16_operators_host) {
  SECTION("Sanity with 1 and 0") {
    INFO("1+0 <-> 0+1");
    auto bf16_one = HIPRT_ONE_BF16;
    auto bf16_zero = HIPRT_ZERO_BF16;
    REQUIRE(__heq((bf16_one + bf16_zero), (bf16_zero + bf16_one)));
    REQUIRE((bf16_one + bf16_zero) == (bf16_zero + bf16_one));
  }

  SECTION("Compare") {
    auto l = __float2bfloat16(1.1f), r = __float2bfloat16(2.2f);

    INFO("Comparing 1.1f and 2.2f");
    REQUIRE(l < r);
    REQUIRE(r > l);
    REQUIRE(l != r);
    REQUIRE(l == l);
    REQUIRE(r == r);
    REQUIRE(l <= l);
    REQUIRE(r <= r);
    REQUIRE(l >= l);
    REQUIRE(r >= r);

    REQUIRE_FALSE(l > r);
    REQUIRE_FALSE(r < l);
    REQUIRE_FALSE(l == r);
    REQUIRE_FALSE(l != l);
    REQUIRE_FALSE(r != r);
  }

  SECTION("Math operator") {
    constexpr float fl = 1.5f, fr = 2.9f;
    auto l = __float2bfloat16(fl), r = __float2bfloat16(fr);

    auto approx_equal = [](__hip_bfloat16 a, float b) -> bool {
      // The relative error should be less than 1/(2^7) since bfloat16 has 7 bits mantissa.
      UNSCOPED_INFO("Comparing: " << __bfloat162float(a) << " - " << b);
      return (std::fabs(__bfloat162float(a) - b) / b) < (1.0 / 128.0f);
    };

    REQUIRE(approx_equal(l * r, fl * fr));
    REQUIRE(approx_equal(l / r, fl / fr));
    REQUIRE(approx_equal(l + r, fl + fr));
    REQUIRE(approx_equal(l - r, fl - fr));

    REQUIRE(approx_equal(r * l, fr * fl));
    REQUIRE(approx_equal(r / l, fr / fl));
    REQUIRE(approx_equal(r + l, fr + fl));
    REQUIRE(approx_equal(r - l, fr - fl));
  }

  SECTION("Unary") {
    constexpr float fl = 7.8f, fr = 9.9f;
    auto l = __float2bfloat16(fl), r = __float2bfloat16(fr);

    REQUIRE(-l == -l);
    REQUIRE(-r == -r);
    REQUIRE(r != -r);
    REQUIRE((l + (-l)) == HIPRT_ZERO_BF16);
    REQUIRE(((-l) * (-l)) == (l * l));
    REQUIRE((l * -r) == -(l * r));
    REQUIRE((l + -l) == HIPRT_ZERO_BF16);
    REQUIRE((l / -l) == -HIPRT_ONE_BF16);
  }
}

HIP_TEST_CASE(Unit_bf162_operators_host) {
  SECTION("Sanity with 1 and 0") {
    INFO("1+0 <-> 0+1");
    __hip_bfloat162 bf162_one = {HIPRT_ONE_BF16, HIPRT_ONE_BF16};
    __hip_bfloat162 bf162_zero = {HIPRT_ZERO_BF16, HIPRT_ZERO_BF16};
    __hip_bfloat162 true_val = bf162_one;
    REQUIRE(__heq2((bf162_one + bf162_zero), (bf162_zero + bf162_one)) == true_val);
    REQUIRE((bf162_one + bf162_zero) == (bf162_zero + bf162_one));
  }

  SECTION("Compare") {
    __hip_bfloat162 l = {__float2bfloat16(1.1f), __float2bfloat16(1.1f)},
                    r = {__float2bfloat16(2.2f), __float2bfloat16(2.2f)};

    INFO("Comparing {1.1f, 1.1f} and {2.2f, 2.2f}");
    REQUIRE(l < r);
    REQUIRE(r > l);
    REQUIRE(l != r);
    REQUIRE(l == l);
    REQUIRE(r == r);
    REQUIRE(l <= l);
    REQUIRE(r <= r);
    REQUIRE(l >= l);
    REQUIRE(r >= r);

    REQUIRE_FALSE(l > r);
    REQUIRE_FALSE(r < l);
    REQUIRE_FALSE(l == r);
    REQUIRE_FALSE(l != l);
    REQUIRE_FALSE(r != r);
  }

  SECTION("Unary") {
    constexpr float fl = 7.8f, fr = 9.9f;
    __hip_bfloat162 l = {__float2bfloat16(fl), __float2bfloat16(fr)},
                    r = {__float2bfloat16(fr), __float2bfloat16(fl)};

    REQUIRE(-l == -l);
    REQUIRE(-r == -r);
    REQUIRE(r != -r);
    REQUIRE((l + (-l)) == __hip_bfloat162{HIPRT_ZERO_BF16, HIPRT_ZERO_BF16});
    REQUIRE(((-l) * (-l)) == (l * l));
    REQUIRE((l * -r) == -(l * r));
    REQUIRE((l + -l) == __hip_bfloat162{HIPRT_ZERO_BF16, HIPRT_ZERO_BF16});
    REQUIRE((l / -l) == -__hip_bfloat162{HIPRT_ONE_BF16, HIPRT_ONE_BF16});
  }

  SECTION("Compare mismatched lanes") {
    // Regression: the __hip_bfloat162 relational operators must compare each lane
    // against its counterpart. A prior bug compared l.x against both r.x and r.y
    // (ignoring l.y), so vectors whose two lanes differ were mis-compared.
    __hip_bfloat162 la = {__float2bfloat16(1.0f), __float2bfloat16(4.0f)};
    __hip_bfloat162 ra = {__float2bfloat16(2.0f), __float2bfloat16(3.0f)};
    // lane0: 1<2 (true), lane1: 4<3 (false) -> overall false
    REQUIRE_FALSE(la < ra);
    REQUIRE_FALSE(la <= ra);

    __hip_bfloat162 lc = {__float2bfloat16(4.0f), __float2bfloat16(1.0f)};
    __hip_bfloat162 rc = {__float2bfloat16(3.0f), __float2bfloat16(2.0f)};
    // lane0: 4>3 (true), lane1: 1>2 (false) -> overall false
    REQUIRE_FALSE(lc > rc);
    REQUIRE_FALSE(lc >= rc);

    // sanity: both lanes satisfy the relation -> true
    __hip_bfloat162 lo = {__float2bfloat16(1.0f), __float2bfloat16(2.0f)};
    __hip_bfloat162 hi = {__float2bfloat16(3.0f), __float2bfloat16(4.0f)};
    REQUIRE(lo < hi);
    REQUIRE(lo <= hi);
    REQUIRE(hi > lo);
    REQUIRE(hi >= lo);
  }

  SECTION("__hbneu2 both-lane reduction") {
    // Regression: __hbneu2 (unordered not-equal) must be true only when BOTH lanes
    // are not-equal (&&), not when either lane differs (||).
    __hip_bfloat162 a = {__float2bfloat16(1.0f), __float2bfloat16(2.0f)};
    __hip_bfloat162 eq_x = {__float2bfloat16(1.0f), __float2bfloat16(3.0f)};  // lane0 equal
    __hip_bfloat162 eq_y = {__float2bfloat16(9.0f), __float2bfloat16(2.0f)};  // lane1 equal
    __hip_bfloat162 both = {__float2bfloat16(5.0f), __float2bfloat16(6.0f)};  // both differ
    __hip_bfloat162 same = a;                                                // both equal

    REQUIRE(__hbneu2(a, both));        // both lanes not-equal -> true
    REQUIRE_FALSE(__hbneu2(a, eq_x));  // one lane equal -> false (was true with ||)
    REQUIRE_FALSE(__hbneu2(a, eq_y));  // one lane equal -> false (was true with ||)
    REQUIRE_FALSE(__hbneu2(a, same));  // both equal -> false
  }

  SECTION("__hgt2 / __hisnan2 high-lane mask") {
    // Regression: __hgt2 and __hisnan2 returned 1.0 for the high (.y) lane
    // unconditionally (both ternary branches were HIPRT_ONE_BF16).
    __hip_bfloat162 a = {__float2bfloat16(5.0f), __float2bfloat16(1.0f)};
    __hip_bfloat162 b = {__float2bfloat16(2.0f), __float2bfloat16(3.0f)};
    float2 gt = __hgt2(a, b);  // lane0: 5>2 -> 1.0, lane1: 1>3 -> 0.0
    REQUIRE(gt.x == 1.0f);
    REQUIRE(gt.y == 0.0f);     // was 1.0f with the bug

    __hip_bfloat162 n = {__float2bfloat16(NAN), __float2bfloat16(1.0f)};
    float2 isn = __hisnan2(n);  // lane0: NaN -> 1.0, lane1: not NaN -> 0.0
    REQUIRE(isn.x == 1.0f);
    REQUIRE(isn.y == 0.0f);     // was 1.0f with the bug
  }
}

// Bunch of tests which make sure we are packaging stuff correctly.
// i.e. highs2bfloat lows2bfloat etc and its various combinations
HIP_TEST_CASE(Unit_bf16_bf162_convert_tests) {
  SECTION("float2->bfloat->float2") {
    float2 in = {3.0f, 4.0f};
    auto bf162 = __float22bfloat162_rn(in);
    auto back = __bfloat1622float2(bf162);
    INFO("original x: " << in.x << " y: " << in.y);
    INFO("cvt back x: " << back.x << " y: " << back.y);
    REQUIRE(in == back);
  }

  SECTION("double->bfloat->double") {
    double in = 5.0;
    auto bf16 = __double2bfloat16(in);
    double back = bf16;
    INFO("Original: " << in << " back: " << back);
    REQUIRE(in == back);
  }

  SECTION("bfloat16->bfloat162->bfloat") {
    float in = 4.0f;
    auto bf16 = __float2bfloat16(in);
    auto bf162 = __bfloat162bfloat162(bf16);
    auto high = __high2float(bf162);
    auto low = __low2float(bf162);
    REQUIRE(high == low);
    REQUIRE(high == in);
  }

  SECTION("Half to bfloat") {
    float in1 = 5.0f, in2 = 6.0f;
    auto bf16_1 = __float2bfloat16(in1);
    auto bf16_2 = __float2bfloat16(in2);
    auto bf162 = __halves2bfloat162(bf16_1, bf16_2);
    float high = __high2bfloat16(bf162);  // force conversion from bfloat to float
    float low = __low2bfloat16(bf162);
    REQUIRE(high == in2);
    REQUIRE(low == in1);
  }

  SECTION("high/low to bfloat162") {
    float in1 = 3.0f, in2 = 4.0f;
    auto bf16_1 = __float2bfloat16(in1);
    auto bf16_2 = __float2bfloat16(in2);
    auto bf162_original = __halves2bfloat162(bf16_1, bf16_2);
    auto high_bf16 = __high2bfloat162(bf162_original);
    auto low_bf16 = __low2bfloat162(bf162_original);
    REQUIRE(high_bf16 == __hip_bfloat162(in2, in2));
    REQUIRE(low_bf16 == __hip_bfloat162(in1, in1));
  }

  SECTION("highs/lows to bfloat162") {
    float in1 = 7.0f, in2 = 8.0f;
    auto bf16_1 = __float2bfloat16(in1);
    auto bf16_2 = __float2bfloat16(in2);
    auto bf162_1 = __halves2bfloat162(bf16_1, bf16_2);
    auto bf162_2 = __halves2bfloat162(bf16_2, bf16_1);
    auto high_bf16 = __highs2bfloat162(bf162_1, bf162_2);
    auto low_bf16 = __lows2bfloat162(bf162_1, bf162_2);
    REQUIRE(high_bf16 == __hip_bfloat162(in2, in1));
    REQUIRE(low_bf16 == __hip_bfloat162(in1, in2));
  }

  SECTION("Low high to high low") {
    float in1 = 1.0f, in2 = 2.0f;
    auto bf16_1 = __float2bfloat16(in1);
    auto bf16_2 = __float2bfloat16(in2);
    auto bf162 = __halves2bfloat162(bf16_1, bf16_2);
    auto inverted = __lowhigh2highlow(bf162);
    REQUIRE(inverted == __halves2bfloat162(bf16_2, bf16_1));
  }
}

__global__ void bf16_shfl_down(float* in, float* out, int size) {
  int i = threadIdx.x;
  if (i < size) {
    auto val = __float2bfloat16(in[i]);
    for (int j = size / 2; j > 0; j /= 2) {
      val += __shfl_down_sync(__activemask(), val, j, size);
    }
    out[i] = val;
  }
}

__global__ void bf16_shfl_up(float* in, float* out, int size) {
  int i = threadIdx.x;
  if (i < size) {
    auto val = __float2bfloat16(in[i]);
    for (int j = size / 2; j > 0; j /= 2) {
      val += __shfl_up_sync(__activemask(), val, j, size);
    }
    out[i] = val;
  }
}

__global__ void bf16_shfl_xor(float* in, float* out, int size) {
  int i = threadIdx.x;
  if (i < size) {
    auto val = __float2bfloat16(in[i]);
    for (int j = size / 2; j > 0; j /= 2) {
      val += __shfl_xor_sync(__activemask(), val, j, size);
    }
    out[i] = val;
  }
}

__global__ void bf16_shfl_sync(float* in, float* out, int size) {
  int i = threadIdx.x;
  if (i < size) {
    auto val = __float2bfloat16(in[i]);
    val += __shfl_sync(__activemask(), val, size - 1, size);
    out[i] = val;
  }
}

HIP_TEST_CASE(Unit_bf16_shfl) {
  auto warp_size = getWarpSize();
  std::vector<float> in;
  for (size_t i = 1; i <= warp_size; i++) {
    in.push_back(i);
  }

  float *d_in, *d_out;
  HIP_CHECK(hipMalloc(&d_in, sizeof(float) * in.size()));
  HIP_CHECK(hipMalloc(&d_out, sizeof(float) * in.size()));

  HIP_CHECK(hipMemcpy(d_in, in.data(), sizeof(float) * in.size(), hipMemcpyHostToDevice));

  std::vector<float> out(warp_size, 0.0f);

  SECTION("shfl_down") {
    bf16_shfl_down<<<1, warp_size>>>(d_in, d_out, warp_size);
    HIP_CHECK(hipMemcpy(out.data(), d_out, sizeof(float) * out.size(), hipMemcpyDeviceToHost));
    REQUIRE(out[0] == (warp_size * (warp_size + 1) / 2));
  }

  SECTION("shfl_up") {
    bf16_shfl_up<<<1, warp_size>>>(d_in, d_out, warp_size);
    HIP_CHECK(hipMemcpy(out.data(), d_out, sizeof(float) * out.size(), hipMemcpyDeviceToHost));
    REQUIRE(out[warp_size - 1] == (warp_size * (warp_size + 1) / 2));
  }

  SECTION("shfl_xor") {
    bf16_shfl_xor<<<1, warp_size>>>(d_in, d_out, warp_size);
    HIP_CHECK(hipMemcpy(out.data(), d_out, sizeof(float) * out.size(), hipMemcpyDeviceToHost));
    REQUIRE(out[0] == (warp_size * (warp_size + 1) / 2));
  }

  SECTION("shfl_sync") {
    bf16_shfl_sync<<<1, warp_size>>>(d_in, d_out, warp_size);
    HIP_CHECK(hipMemcpy(out.data(), d_out, sizeof(float) * out.size(), hipMemcpyDeviceToHost));
    REQUIRE(out[0] == (warp_size + 1));
  }

  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));
}

__global__ void bf162_shfl_down(float2* in, float2* out, int size) {
  int i = threadIdx.x;
  if (i < size) {
    auto val = __float22bfloat162_rn(in[i]);
    for (int j = size / 2; j > 0; j /= 2) {
      val += __shfl_down_sync(__activemask(), val, j, size);
    }
    out[i] = val;
  }
}

__global__ void bf162_shfl_up(float2* in, float2* out, int size) {
  int i = threadIdx.x;
  if (i < size) {
    auto val = __float22bfloat162_rn(in[i]);
    for (int j = size / 2; j > 0; j /= 2) {
      val += __shfl_up_sync(__activemask(), val, j, size);
    }
    out[i] = val;
  }
}

__global__ void bf162_shfl_xor(float2* in, float2* out, int size) {
  int i = threadIdx.x;
  if (i < size) {
    auto val = __float22bfloat162_rn(in[i]);
    for (int j = size / 2; j > 0; j /= 2) {
      val += __shfl_xor_sync(__activemask(), val, j, size);
    }
    out[i] = val;
  }
}

__global__ void bf162_shfl_sync(float2* in, float2* out, int size) {
  int i = threadIdx.x;
  if (i < size) {
    auto val = __float22bfloat162_rn(in[i]);
    val += __shfl_sync(__activemask(), val, size - 1, size);
    out[i] = val;
  }
}

HIP_TEST_CASE(Unit_bf162_shfl) {
  auto warp_size = getWarpSize();
  std::vector<float2> in;
  for (size_t i = 1; i <= warp_size; i++) {
    in.push_back(float2{i, i * 2});
  }

  float2 *d_in, *d_out;
  HIP_CHECK(hipMalloc(&d_in, sizeof(float2) * in.size()));
  HIP_CHECK(hipMalloc(&d_out, sizeof(float2) * in.size()));

  HIP_CHECK(hipMemcpy(d_in, in.data(), sizeof(float2) * in.size(), hipMemcpyHostToDevice));

  std::vector<float2> out(warp_size, float2{0.0f, 0.0f});

  SECTION("shfl_down") {
    bf162_shfl_down<<<1, warp_size>>>(d_in, d_out, warp_size);
    HIP_CHECK(hipMemcpy(out.data(), d_out, sizeof(float2) * out.size(), hipMemcpyDeviceToHost));
    auto res = (warp_size * (warp_size + 1) / 2);
    INFO("Expected: x: " << res << " y: " << (res * 2));
    INFO("Got:      x: " << out[0].x << " y: " << out[0].y);
    REQUIRE(out[0] == float2{res, res * 2});
  }

  SECTION("shfl_up") {
    bf162_shfl_up<<<1, warp_size>>>(d_in, d_out, warp_size);
    HIP_CHECK(hipMemcpy(out.data(), d_out, sizeof(float2) * out.size(), hipMemcpyDeviceToHost));
    auto res = (warp_size * (warp_size + 1) / 2);
    INFO("Expected: x: " << res << " y: " << (res * 2));
    INFO("Got:      x: " << out[warp_size - 1].x << " y: " << out[warp_size - 1].y);
    REQUIRE(out[warp_size - 1] == float2{res, res * 2});
  }

  SECTION("shfl_xor") {
    bf162_shfl_xor<<<1, warp_size>>>(d_in, d_out, warp_size);
    HIP_CHECK(hipMemcpy(out.data(), d_out, sizeof(float2) * out.size(), hipMemcpyDeviceToHost));
    auto res = (warp_size * (warp_size + 1) / 2);
    INFO("Expected: x: " << res << " y: " << (res * 2));
    INFO("Got:      x: " << out[0].x << " y: " << out[0].y);
    REQUIRE(out[0] == float2{res, res * 2});
  }

  SECTION("shfl_sync") {
    bf162_shfl_sync<<<1, warp_size>>>(d_in, d_out, warp_size);
    HIP_CHECK(hipMemcpy(out.data(), d_out, sizeof(float2) * out.size(), hipMemcpyDeviceToHost));
    auto res = warp_size + 1;
    INFO("Expected: x: " << res << " y: " << (res * 2));
    INFO("Got:      x: " << out[warp_size - 1].x << " y: " << out[warp_size - 1].y);
    REQUIRE(out[0] == float2{res, res * 2});
  }

  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));
}

__global__ void bf16_hrcp(float* in, float* out) {
  int i = threadIdx.x;
  __hip_bfloat16 bf{in[i]};
  out[i] = hrcp(bf);
}

__global__ void bf16_hlog2(float* in, float* out) {
  int i = threadIdx.x;
  __hip_bfloat16 bf{in[i]};
  out[i] = hlog2(bf);
}

__global__ void bf16_hlog10(float* in, float* out) {
  int i = threadIdx.x;
  __hip_bfloat16 bf{in[i]};
  out[i] = hlog10(bf);
}

__global__ void bf16_hlog(float* in, float* out) {
  int i = threadIdx.x;
  __hip_bfloat16 bf{in[i]};
  out[i] = hlog(bf);
}

__global__ void bf16_hcos(float* in, float* out) {
  int i = threadIdx.x;
  __hip_bfloat16 bf{in[i]};
  out[i] = hcos(bf);
}

__global__ void bf16_hsin(float* in, float* out) {
  int i = threadIdx.x;
  __hip_bfloat16 bf{in[i]};
  out[i] = hsin(bf);
}

__global__ void bf16_hexp(float* in, float* out) {
  int i = threadIdx.x;
  __hip_bfloat16 bf{in[i]};
  out[i] = hexp(bf);
}

__global__ void bf16_hexp2(float* in, float* out) {
  int i = threadIdx.x;
  __hip_bfloat16 bf{in[i]};
  out[i] = hexp2(bf);
}

__global__ void bf16_hsqrt(float* in, float* out) {
  int i = threadIdx.x;
  __hip_bfloat16 bf{in[i]};
  out[i] = hsqrt(bf);
}

__global__ void bf16_hrsqrt(float* in, float* out) {
  int i = threadIdx.x;
  __hip_bfloat16 bf{in[i]};
  out[i] = hrsqrt(bf);
}

HIP_TEST_CASE(Unit_bf16_value_ops) {
  constexpr size_t size = 32;
  float *d_in, *d_out;
  HIP_CHECK(hipMalloc(&d_in, sizeof(float) * size));
  HIP_CHECK(hipMalloc(&d_out, sizeof(float) * size));

  std::vector<float> in;
  in.reserve(size);
  for (size_t i = 1; i <= size; i++) {
    in.push_back(static_cast<float>(i));
  }

  HIP_CHECK(hipMemcpy(d_in, in.data(), sizeof(float) * size, hipMemcpyHostToDevice));

  SECTION("hrcp") {
    bf16_hrcp<<<1, size>>>(d_in, d_out);
    std::vector<float> res(size, 0.0f);
    HIP_CHECK(hipMemcpy(res.data(), d_out, sizeof(float) * size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      float rcp_res = 1.0f / in[i];
      INFO("From GPU : " << res[i] << " from cpu: " << rcp_res);
      REQUIRE(std::fabs(res[i] - rcp_res) <= 0.02f);
    }
  }

  SECTION("hlog2") {
    bf16_hlog2<<<1, size>>>(d_in, d_out);
    std::vector<float> res(size, 0.0f);
    HIP_CHECK(hipMemcpy(res.data(), d_out, sizeof(float) * size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      float rcp_res = std::log2f(in[i]);
      INFO("From GPU : " << res[i] << " from cpu: " << rcp_res);
      REQUIRE(std::fabs(res[i] - rcp_res) <= 0.02f);
    }
  }

  SECTION("hlog10") {
    bf16_hlog10<<<1, size>>>(d_in, d_out);
    std::vector<float> res(size, 0.0f);
    HIP_CHECK(hipMemcpy(res.data(), d_out, sizeof(float) * size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      float rcp_res = std::log10f(in[i]);
      INFO("From GPU : " << res[i] << " from cpu: " << rcp_res);
      REQUIRE(std::fabs(res[i] - rcp_res) <= 0.02f);
    }
  }

  SECTION("hlog") {
    bf16_hlog<<<1, size>>>(d_in, d_out);
    std::vector<float> res(size, 0.0f);
    HIP_CHECK(hipMemcpy(res.data(), d_out, sizeof(float) * size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      float rcp_res = std::logf(in[i]);
      INFO("From GPU : " << res[i] << " from cpu: " << rcp_res);
      REQUIRE(std::fabs(res[i] - rcp_res) <= 0.02f);
    }
  }

  SECTION("hcos") {
    bf16_hcos<<<1, size>>>(d_in, d_out);
    std::vector<float> res(size, 0.0f);
    HIP_CHECK(hipMemcpy(res.data(), d_out, sizeof(float) * size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      float rcp_res = std::cos(in[i]);
      INFO("From GPU : " << res[i] << " from cpu: " << rcp_res);
      REQUIRE(std::fabs(res[i] - rcp_res) <= 0.02f);
    }
  }

  SECTION("hsin") {
    bf16_hsin<<<1, size>>>(d_in, d_out);
    std::vector<float> res(size, 0.0f);
    HIP_CHECK(hipMemcpy(res.data(), d_out, sizeof(float) * size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      float rcp_res = std::sin(in[i]);
      INFO("From GPU : " << res[i] << " from cpu: " << rcp_res);
      REQUIRE(std::fabs(res[i] - rcp_res) <= 0.02f);
    }
  }

  SECTION("hexp") {
    constexpr size_t size_override = 7;  // the exp values goes too high and hence we limit it to 7
    bf16_hexp<<<1, size_override>>>(d_in, d_out);
    std::vector<float> res(size_override, 0.0f);
    HIP_CHECK(hipMemcpy(res.data(), d_out, sizeof(float) * size_override, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size_override; i++) {
      float rcp_res = std::exp(in[i]);
      INFO("Input: " << in[i] << " From GPU : " << res[i] << " from cpu: " << rcp_res);
      REQUIRE(std::fabs(res[i] - rcp_res) <= 1.0f);
    }
  }

  SECTION("hexp2") {
    bf16_hexp2<<<1, size>>>(d_in, d_out);
    std::vector<float> res(size, 0.0f);
    HIP_CHECK(hipMemcpy(res.data(), d_out, sizeof(float) * size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      float rcp_res = std::exp2f(in[i]);
      INFO("Input: " << in[i] << " From GPU : " << res[i] << " from cpu: " << rcp_res);
      REQUIRE(std::fabs(res[i] - rcp_res) <= 1.0f);
    }
  }

  SECTION("hsqrt") {
    bf16_hsqrt<<<1, size>>>(d_in, d_out);
    std::vector<float> res(size, 0.0f);
    HIP_CHECK(hipMemcpy(res.data(), d_out, sizeof(float) * size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      float rcp_res = std::sqrt(in[i]);
      INFO("Input: " << in[i] << " from GPU : " << res[i] << " from cpu: " << rcp_res);
      REQUIRE(std::fabs(res[i] - rcp_res) <= 0.02f);
    }
  }

  SECTION("hrsqrt") {
    bf16_hrsqrt<<<1, size>>>(d_in, d_out);
    std::vector<float> res(size, 0.0f);
    HIP_CHECK(hipMemcpy(res.data(), d_out, sizeof(float) * size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      float rcp_res = 1.0f / std::sqrt(in[i]);
      INFO("Input: " << in[i] << " from GPU : " << res[i] << " from cpu: " << rcp_res);
      REQUIRE(std::fabs(res[i] - rcp_res) <= 0.02f);
    }
  }

  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));
}

__global__ void bf16_hfloor(float* in, float* out) {
  int i = threadIdx.x;
  __hip_bfloat16 bf{in[i]};
  out[i] = hfloor(bf);
}

__global__ void bf16_hceil(float* in, float* out) {
  int i = threadIdx.x;
  __hip_bfloat16 bf{in[i]};
  out[i] = hceil(bf);
}

__global__ void bf16_hrint(float* in, float* out) {
  int i = threadIdx.x;
  __hip_bfloat16 bf{in[i]};
  out[i] = hrint(bf);
}

__global__ void bf16_htrunc(float* in, float* out) {
  int i = threadIdx.x;
  __hip_bfloat16 bf{in[i]};
  out[i] = htrunc(bf);
}

HIP_TEST_CASE(Unit_bf16_floor_ceil) {
  constexpr size_t size = 32;
  float *d_in, *d_out;
  HIP_CHECK(hipMalloc(&d_in, sizeof(float) * size));
  HIP_CHECK(hipMalloc(&d_out, sizeof(float) * size));

  std::vector<float> in;
  in.reserve(size);
  for (size_t i = 1; i <= size; i++) {
    float tmp = static_cast<float>(i);
    if (i % 2 == 0)
      in.push_back(tmp - 0.1f);
    else
      in.push_back(tmp + 0.1f);
  }

  HIP_CHECK(hipMemcpy(d_in, in.data(), sizeof(float) * size, hipMemcpyHostToDevice));

  SECTION("hceil") {
    bf16_hceil<<<1, size>>>(d_in, d_out);
    std::vector<float> res(size, 0.0f);
    HIP_CHECK(hipMemcpy(res.data(), d_out, sizeof(float) * size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      float rcp_res = std::ceil(in[i]);
      INFO("Input: " << in[i] << "from GPU : " << res[i] << " from cpu: " << rcp_res);
      REQUIRE(res[i] == rcp_res);
    }
  }

  SECTION("hfloor") {
    bf16_hfloor<<<1, size>>>(d_in, d_out);
    std::vector<float> res(size, 0.0f);
    HIP_CHECK(hipMemcpy(res.data(), d_out, sizeof(float) * size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      float rcp_res = std::floor(in[i]);
      INFO("Input: " << in[i] << "from GPU : " << res[i] << " from cpu: " << rcp_res);
      REQUIRE(res[i] == rcp_res);
    }
  }

  SECTION("hrint") {
    bf16_hrint<<<1, size>>>(d_in, d_out);
    std::vector<float> res(size, 0.0f);
    HIP_CHECK(hipMemcpy(res.data(), d_out, sizeof(float) * size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      float rcp_res = std::round(in[i]);
      INFO("Input: " << in[i] << "from GPU : " << res[i] << " from cpu: " << rcp_res);
      REQUIRE(res[i] == rcp_res);
    }
  }

  SECTION("htrunc") {
    bf16_htrunc<<<1, size>>>(d_in, d_out);
    std::vector<float> res(size, 0.0f);
    HIP_CHECK(hipMemcpy(res.data(), d_out, sizeof(float) * size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      float rcp_res = std::trunc(in[i]);
      INFO("Input: " << in[i] << "from GPU : " << res[i] << " from cpu: " << rcp_res);
      REQUIRE(res[i] == rcp_res);
    }
  }

  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));
}

__global__ void bf162_hfloor(float2* in, float2* out) {
  int i = threadIdx.x;
  __hip_bfloat162 bf{in[i].x, in[i].y};
  out[i] = h2floor(bf);
}

__global__ void bf162_hceil(float2* in, float2* out) {
  int i = threadIdx.x;
  __hip_bfloat162 bf{in[i].x, in[i].y};
  out[i] = h2ceil(bf);
}

__global__ void bf162_hrint(float2* in, float2* out) {
  int i = threadIdx.x;
  __hip_bfloat162 bf{in[i].x, in[i].y};
  out[i] = h2rint(bf);
}

__global__ void bf162_htrunc(float2* in, float2* out) {
  int i = threadIdx.x;
  __hip_bfloat162 bf{in[i].x, in[i].y};
  out[i] = h2trunc(bf);
}

HIP_TEST_CASE(Unit_bf162_floor_ceil) {
  constexpr size_t size = 32;
  float2 *d_in, *d_out;
  HIP_CHECK(hipMalloc(&d_in, sizeof(float2) * size));
  HIP_CHECK(hipMalloc(&d_out, sizeof(float2) * size));

  std::vector<float2> in;
  in.reserve(size);
  for (size_t i = 0; i < size; i++) {
    float tmp = static_cast<float>(i);
    in.push_back(float2{tmp - 0.1f, tmp + 0.1f});
  }

  HIP_CHECK(hipMemcpy(d_in, in.data(), sizeof(float2) * size, hipMemcpyHostToDevice));

  SECTION("hceil") {
    bf162_hceil<<<1, size>>>(d_in, d_out);
    std::vector<float2> res(size);
    HIP_CHECK(hipMemcpy(res.data(), d_out, sizeof(float2) * size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      float2 rcp_res{std::ceil(in[i].x), std::ceil(in[i].y)};
      INFO("Input: " << in[i].x << ", " << in[i].y << " from GPU : " << res[i].x << ", " << res[i].y
                     << " from cpu: " << rcp_res.x << ", " << rcp_res.y);
      REQUIRE(res[i] == rcp_res);
    }
  }

  SECTION("hfloor") {
    bf162_hfloor<<<1, size>>>(d_in, d_out);
    std::vector<float2> res(size);
    HIP_CHECK(hipMemcpy(res.data(), d_out, sizeof(float2) * size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      float2 rcp_res{std::floor(in[i].x), std::floor(in[i].y)};
      INFO("Input: " << in[i].x << ", " << in[i].y << " from GPU : " << res[i].x << ", " << res[i].y
                     << " from cpu: " << rcp_res.x << ", " << rcp_res.y);
      REQUIRE(res[i] == rcp_res);
    }
  }

  SECTION("hrint") {
    bf162_hrint<<<1, size>>>(d_in, d_out);
    std::vector<float2> res(size);
    HIP_CHECK(hipMemcpy(res.data(), d_out, sizeof(float2) * size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      float2 rcp_res{std::round(in[i].x), std::round(in[i].y)};
      INFO("Input: " << in[i].x << ", " << in[i].y << " from GPU : " << res[i].x << ", " << res[i].y
                     << " from cpu: " << rcp_res.x << ", " << rcp_res.y);
      REQUIRE(res[i] == rcp_res);
    }
  }

  SECTION("htrunc") {
    bf162_htrunc<<<1, size>>>(d_in, d_out);
    std::vector<float2> res(size);
    HIP_CHECK(hipMemcpy(res.data(), d_out, sizeof(float2) * size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      float2 rcp_res{std::trunc(in[i].x), std::trunc(in[i].y)};
      INFO("Input: " << in[i].x << ", " << in[i].y << " from GPU : " << res[i].x << ", " << res[i].y
                     << " from cpu: " << rcp_res.x << ", " << rcp_res.y);
      REQUIRE(res[i] == rcp_res);
    }
  }

  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));
}
