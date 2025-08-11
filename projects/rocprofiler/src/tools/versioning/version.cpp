#include <rocm-core/rocm_version.h>
#include <iostream>
#include <sstream>
#include <cstdint>

using std::uint32_t;


int main() {
  const char* envvar = getenv("ROCPROFILER_LIBRARY_VERSION");
  uint32_t mj = 0, mn = 0, p = 0;
  int ret = 0;
  ret = getROCmVersion(&mj, &mn, &p);
  if (ret != VerSuccess) {
    std::cerr << "Error occured while retreiving rocm version!\n";
  } else {
    std::cout << "ROCm version: " << mj << "." << mn << "." << p << "\n";
    if (envvar) std::cout << "ROCProfiler Version: " << envvar << "\n";
    std::cout << "\nFull Build information for ROCm:\t";
    printBuildInfo();
  }
  return 1;
}