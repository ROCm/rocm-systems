#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include <iostream>

// no extern C, coz we want to use bool
bool test_fp16() {
  {
    __half h = 1.0f;
    float r = h;
    auto res = (r == 1.0f);
    if (!res) {
      std::cout << "Failed in to convert 1.0 back to float, we got: " << r << std::endl;
      return false;
    }
  }

  {
    __half h = 10.0f;
    float r = h;
    auto res = (r == 10.0f);
    if (!res) {
      std::cout << "Failed in to convert 1.0 back to float, we got: " << r << std::endl;
      return false;
    }
  }

  return true;
}

bool test_half22float2() {
  __half2 h2(__float2half(1.5f), __float2half(-2.0f));
  float2 f2 = __half22float2(h2);
  if (f2.x != 1.5f || f2.y != -2.0f) {
    std::cout << "__half22float2 lanes: got " << f2.x << ", " << f2.y << std::endl;
    return false;
  }
  return true;
}

bool test_bf16() {
  {
    __hip_bfloat16 h = 1.0f;
    float r = h;
    auto res = (r == 1.0f);
    if (!res) {
      std::cout << "Failed in to convert 1.0 back to float, we got: " << r << std::endl;
      return false;
    }
  }

  {
    __hip_bfloat16 h = 10.0f;
    float r = h;
    auto res = (r == 10.0f);
    if (!res) {
      std::cout << "Failed in to convert 1.0 back to float, we got: " << r << std::endl;
      return false;
    }
  }

  return true;
}
