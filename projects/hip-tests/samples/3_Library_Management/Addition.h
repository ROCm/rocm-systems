#include "Common.h"

void LaunchKernelForAddition(float *x, float *y, float *z, int N);

class Addition {
public:
  Addition();
  ~Addition();

  void LaunchKernelForAddition(float *x, float *y, float *z, int N);

  void getStream(hipStream_t &strm);
  void customizeStreamAttributes(hipStream_t &strm);

private:
  hipStream_t stream;
};
