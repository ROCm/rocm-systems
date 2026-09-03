// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

namespace {

constexpr int kM = 128;
constexpr int kN = 128;
constexpr int kK = 128;
constexpr float kAlpha = 1.0F;
constexpr float kBeta = 0.0F;
constexpr long double kUnitRoundoff = 0x1p-24L;
constexpr long double kSafetyFactor = 4.0L;

float input_a(int row, int column) {
  const int numerator = (row * 17 + column * 13 + 1) % 33 - 16;
  return static_cast<float>(numerator) / 16.0F;
}

float input_b(int row, int column) {
  const int numerator = (row * 29 + column * 7 + 3) % 33 - 16;
  return static_cast<float>(numerator) / 16.0F;
}

std::size_t column_major_index(int row, int column, int leading_dimension) {
  return static_cast<std::size_t>(row) + static_cast<std::size_t>(column) * leading_dimension;
}

bool check_hip(hipError_t status, const char *operation) {
  if (status == hipSuccess) {
    return true;
  }
  std::cerr << "m2-sgemm: " << operation << " failed: " << hipGetErrorString(status) << '\n';
  return false;
}

bool check_rocblas(rocblas_status status, const char *operation) {
  if (status == rocblas_status_success) {
    return true;
  }
  std::cerr << "m2-sgemm: " << operation << " failed: status " << static_cast<int>(status) << '\n';
  return false;
}

} // namespace

int main(int argument_count, char **arguments) {
  if (argument_count == 2 && std::string_view(arguments[1]) == "--describe") {
    std::cout
        << R"json({"alpha":1.0,"api":"rocblas_sgemm","beta":0.0,"data_type":"fp32","error_bound":{"formula":"4 * gamma_k * sum(abs(A[i,k] * B[k,j]))","gamma_k":"K * 2^-24 / (1 - K * 2^-24)","safety_factor":4.0,"unit_roundoff":"2^-24"},"guest_executable":"opt/rocjitsu/bin/m2-rocblas-sgemm","input_a":"((row * 17 + column * 13 + 1) % 33 - 16) / 16","input_b":"((row * 29 + column * 7 + 3) % 33 - 16) / 16","k":128,"layout":"column-major","m":128,"n":128,"reference_accumulator":"long double","trans_a":"none","trans_b":"none"})json"
        << '\n';
    return 0;
  }
  if (argument_count != 1) {
    std::cerr << "usage: " << arguments[0] << " [--describe]\n";
    return 2;
  }

  const std::size_t elements_a = static_cast<std::size_t>(kM) * kK;
  const std::size_t elements_b = static_cast<std::size_t>(kK) * kN;
  const std::size_t elements_c = static_cast<std::size_t>(kM) * kN;
  std::vector<float> host_a(elements_a);
  std::vector<float> host_b(elements_b);
  std::vector<float> host_c(elements_c, std::numeric_limits<float>::quiet_NaN());
  std::vector<long double> reference(elements_c);
  std::vector<long double> error_bounds(elements_c);

  for (int column = 0; column < kK; ++column) {
    for (int row = 0; row < kM; ++row) {
      host_a[column_major_index(row, column, kM)] = input_a(row, column);
    }
  }
  for (int column = 0; column < kN; ++column) {
    for (int row = 0; row < kK; ++row) {
      host_b[column_major_index(row, column, kK)] = input_b(row, column);
    }
  }

  const long double gamma_k = (static_cast<long double>(kK) * kUnitRoundoff) /
                              (1.0L - static_cast<long double>(kK) * kUnitRoundoff);
  for (int column = 0; column < kN; ++column) {
    for (int row = 0; row < kM; ++row) {
      long double sum = 0.0L;
      long double sum_of_absolute_products = 0.0L;
      for (int inner = 0; inner < kK; ++inner) {
        const long double product =
            static_cast<long double>(host_a[column_major_index(row, inner, kM)]) *
            static_cast<long double>(host_b[column_major_index(inner, column, kK)]);
        sum += product;
        sum_of_absolute_products += std::abs(product);
      }
      const std::size_t index = column_major_index(row, column, kM);
      reference[index] = sum;
      error_bounds[index] = kSafetyFactor * gamma_k * sum_of_absolute_products;
    }
  }

  float *device_a = nullptr;
  float *device_b = nullptr;
  float *device_c = nullptr;
  rocblas_handle handle = nullptr;
  int result = 1;

  if (!check_hip(hipMalloc(&device_a, elements_a * sizeof(float)), "hipMalloc(A)") ||
      !check_hip(hipMalloc(&device_b, elements_b * sizeof(float)), "hipMalloc(B)") ||
      !check_hip(hipMalloc(&device_c, elements_c * sizeof(float)), "hipMalloc(C)") ||
      !check_hip(
          hipMemcpy(device_a, host_a.data(), elements_a * sizeof(float), hipMemcpyHostToDevice),
          "hipMemcpy(A)") ||
      !check_hip(
          hipMemcpy(device_b, host_b.data(), elements_b * sizeof(float), hipMemcpyHostToDevice),
          "hipMemcpy(B)") ||
      !check_hip(
          hipMemcpy(device_c, host_c.data(), elements_c * sizeof(float), hipMemcpyHostToDevice),
          "hipMemcpy(C)") ||
      !check_rocblas(rocblas_create_handle(&handle), "rocblas_create_handle") ||
      !check_rocblas(rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, kM, kN,
                                   kK, &kAlpha, device_a, kM, device_b, kK, &kBeta, device_c, kM),
                     "rocblas_sgemm") ||
      !check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize") ||
      !check_hip(
          hipMemcpy(host_c.data(), device_c, elements_c * sizeof(float), hipMemcpyDeviceToHost),
          "hipMemcpy(C result)")) {
    goto cleanup;
  }

  {
    long double maximum_absolute_error = 0.0L;
    long double maximum_allowed_error = 0.0L;
    std::size_t failures = 0;
    for (std::size_t index = 0; index < elements_c; ++index) {
      const long double observed = static_cast<long double>(host_c[index]);
      const long double error = std::abs(observed - reference[index]);
      maximum_absolute_error = std::max(maximum_absolute_error, error);
      maximum_allowed_error = std::max(maximum_allowed_error, error_bounds[index]);
      if (!std::isfinite(host_c[index]) || error > error_bounds[index]) {
        ++failures;
        if (failures <= 8) {
          std::cerr << "m2-sgemm: mismatch index=" << index << " observed=" << host_c[index]
                    << " reference=" << reference[index] << " error=" << error
                    << " bound=" << error_bounds[index] << '\n';
        }
      }
    }

    std::cout << std::setprecision(18)
              << "m2-sgemm: api=rocblas_sgemm type=fp32 layout=column-major "
                 "trans_a=none trans_b=none m=128 n=128 k=128 alpha=1 beta=0\n"
              << "m2-sgemm: unit_roundoff=" << kUnitRoundoff << " gamma_k=" << gamma_k
              << " safety_factor=" << kSafetyFactor << '\n'
              << "m2-sgemm: max_absolute_error=" << maximum_absolute_error
              << " max_allowed_error=" << maximum_allowed_error << " failures=" << failures << '\n';
    if (failures != 0) {
      goto cleanup;
    }
  }

  std::cout << "m2-sgemm: PASS\n";
  result = 0;

cleanup:
  if (handle != nullptr) {
    check_rocblas(rocblas_destroy_handle(handle), "rocblas_destroy_handle");
  }
  if (device_c != nullptr) {
    check_hip(hipFree(device_c), "hipFree(C)");
  }
  if (device_b != nullptr) {
    check_hip(hipFree(device_b), "hipFree(B)");
  }
  if (device_a != nullptr) {
    check_hip(hipFree(device_a), "hipFree(A)");
  }
  return result;
}
