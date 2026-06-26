// Global memory race example
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

__global__ void sum_with_race(int *result, int *data, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    *result += data[idx];  // RACE: multiple blocks update same location
  }
}

int main() {
  const int N = 1000;
  std::vector<int> h_data(N, 1);

  int *d_data, *d_result;
  hipMalloc(&d_data, N * sizeof(int));
  hipMalloc(&d_result, sizeof(int));

  hipMemcpy(d_data, h_data.data(), N * sizeof(int), hipMemcpyHostToDevice);
  hipMemset(d_result, 0, sizeof(int));

  sum_with_race<<<10, 100>>>(d_result, d_data, N);
  hipDeviceSynchronize();

  int result = 0;
  hipMemcpy(&result, d_result, sizeof(int), hipMemcpyDeviceToHost);

  std::cout << "Expected sum: " << N << std::endl;
  std::cout << "Actual sum: " << result << std::endl;
  std::cout << "Lost updates: " << (N - result) << std::endl;

  hipFree(d_data);
  hipFree(d_result);
  return 0;
}