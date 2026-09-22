#include <stdio.h>

#include "vector_lib.h"

int main() {
  const int N = 1024;
  float sum = 0.0f;

  printf("Client: Calling vector_add_and_sum with N=%d...\n", N);

  int result = vector_add_and_sum(N, &sum);

  if (result == 0) {
    printf("Client: Success! Sum of results = %f\n", sum);

    // Verify the sum is correct
    // Expected: sum of (i + i*2) = sum of (3*i) for i=0 to N-1
    // = 3 * (N-1) * N / 2
    float expected_sum = 3.0f * (N - 1) * N / 2.0f;
    printf("Client: Expected sum = %f\n", expected_sum);

    if (sum == expected_sum) {
      printf("Client: Verification passed!\n");
      return 0;
    } else {
      printf("Client: Verification failed (sum mismatch)\n");
      return 1;
    }
  } else {
    printf("Client: Failed to compute vector addition\n");
    return 1;
  }
}
