#include "Common.h"

class Multiplication {
  public:
  Multiplication();
  ~Multiplication();

  void LaunchKernelForMultiplication(float *x, float *y, float *z, int N);
  
  void getStream(hipStream_t &strm);
  void customizeStreamAttributes(hipStream_t &strm);

  private:
  hipStream_t stream;
};
