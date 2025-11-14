#include <iostream>
#include "Multiplication.h"
#include "Addition.h"

int main() {
  constexpr int N = 5;

  float x[N] = {1.0f, 2.0f, 8.0f, 4.0f, 16.0f};
  float y[N] = {2.0f, 2.0f, 1.0f, 4.0f, 2.0f};
  float z[N] = {10.0f, 11.0f, 12.0f, 13.0f, 14.0f};
  float out[N] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  std::cout << "Vector x :" << std::endl;
  for (int i = 0; i < N; i++) {
      std::cout << x[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "Vector y :" << std::endl;
  for (int i = 0; i < N; i++) {
      std::cout << y[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "Vector z :" << std::endl;
  for (int i = 0; i < N; i++) {
      std::cout << z[i] << " ";
  }
  std::cout << std::endl;

  Multiplication mul;
  mul.LaunchKernelForMultiplication(x, y, out, N);

  std::cout << "Result Vector x*y :" << std::endl;
  for (int i = 0; i < N; i++) {
      std::cout << out[i] << " ";
  }
  std::cout << std::endl;

  hipStream_t refStream = nullptr;
  CHECK_HIP_ERROR(hipStreamCreate(&refStream));

  mul.getStream(refStream);
  Addition add;
  add.customizeStreamAttributes(refStream);
  add.LaunchKernelForAddition(out, z, out, N);

  std::cout << "Result Vector x*y+z :" << std::endl;
  for (int i = 0; i < N; i++) {
      std::cout << out[i] << " ";
	  REQUIRE(out[i] == (x[i] * y[i] + z[i]));
  }
  std::cout << std::endl;

  CHECK_HIP_ERROR(hipStreamDestroy(refStream));

  return 0;
}
