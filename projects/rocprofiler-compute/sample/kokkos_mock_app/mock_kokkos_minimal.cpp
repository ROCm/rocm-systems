#include "mock_kokkos.hpp"

int main() {
  // Initialize Kokkos
  Kokkos::initialize();

  const int n = 100;

  // Create a View - make sure it's not const and properly sized
  Kokkos::View<int> data("test_data", n);

  // Get device pointer for kernel access
  int *data_ptr = data.data(); // Get raw device pointer

  // Fill the view with data using parallel_for
  Kokkos::parallel_for<Kokkos::HIP>(
      Kokkos::RangePolicy<int>(0, n),
      [=] __device__(
          const int i) { // Add __device__ and use [=] with device pointer
        data_ptr[i] = i * i;
      });

  // Debug: Print first few values to verify data was written
  std::cout << "First few values: ";
  for (int i = 0; i < 5 && i < n; ++i) {
    std::cout << data(i) << " "; // This uses host access
  }
  std::cout << std::endl;

  // Sum all values using parallel_reduce with device pointer
  // CHANGED: Use Kokkos::HIP instead of Kokkos::Serial to launch GPU kernels
  int sum = 0;
  Kokkos::parallel_reduce<Kokkos::HIP>(
      Kokkos::RangePolicy<int>(0, n),
      [=] __device__(const int i,
                     int &result) { // Add __device__ and use device pointer
        result += data_ptr[i];
      },
      sum);

  std::cout << "Sum of squares from 0 to " << (n - 1) << " = " << sum
            << std::endl;

  // Expected sum should be: 0² + 1² + 2² + ... + 99² = 328350
  int expected = 0;
  for (int i = 0; i < n; i++) {
    expected += i * i;
  }
  std::cout << "Expected sum: " << expected << std::endl;

  // Clean up
  Kokkos::finalize();

  return 0;
}