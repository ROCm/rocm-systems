# Example 8: Matrix Multiplication (GEMM) Debugging

## Objective

Debug rocBLAS GEMM operations and verify numerical accuracy.

## Prerequisites

Requires rocBLAS library (part of ROCm installation).

## Files

- `src/gemm_test.cpp` - rocBLAS GEMM debugging
- `Makefile`

## Quick Start

```bash
make
make run
```

## Key Debugging Points

- Matrix dimension validation
- Leading dimension (lda, ldb, ldc) correctness
- Transpose flags
- Alpha/beta parameters
- Numerical accuracy

## Common Issues

### Wrong Dimensions
```cpp
// Ensure: C (M x N) = A (M x K) * B (K x N)
rocblas_sgemm(handle, transA, transB,
              M, N, K,  // Dimensions must match!
              &alpha, A, lda, B, ldb,
              &beta, C, ldc);
```

### Leading Dimension Errors
```cpp
// lda must be >= M for column-major
int lda = M;  // Correct
int lda = K;  // Wrong!
```