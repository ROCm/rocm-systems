#ifndef VECTOR_LIB_H
#define VECTOR_LIB_H

#ifdef __cplusplus
extern "C" {
#endif

// Performs vector addition on the GPU and returns the sum of all elements
// Returns 0 on success, -1 on failure
int vector_add_and_sum(int n, float* result_sum);

#ifdef __cplusplus
}
#endif

#endif  // VECTOR_LIB_H
