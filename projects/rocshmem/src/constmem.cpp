#include "constmem.hpp"
#include "envvar.hpp"

namespace rocshmem {

__constant__ constmem_t constmem;

void init_constant_memory(void) {
  std::string envstr;
  constmem_t constmem_values;

  memset(&constmem_values, 0, sizeof(constmem_t));

  envstr = envvar::gda::alltoallv_algo;

  if (envstr.empty() || envstr.find("HT") != std::string::npos) {
    constmem_values.alltoall_wg_algo = ALLTOALLV_ALGO_HT;
  } else {
    constmem_values.alltoall_wg_algo = ALLTOALLV_ALGO_LL;
  }

  CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(constmem), &constmem_values, sizeof(constmem_t)));
}

}
