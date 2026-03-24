#include <hip/hip_runtime.h>
extern "C" __global__ void f64_math(double* out, const double* in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double x = in[i];
    double r = sqrt(x);       // v_sqrt_f64 + Newton-Raphson
    double s = 1.0 / x;       // v_rcp_f64 + refinement
    out[i] = r + s;
  }
}
