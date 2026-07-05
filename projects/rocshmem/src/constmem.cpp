#include "constmem.hpp"
#include "backend_bc.hpp"
#include "envvar.hpp"
#if defined(USE_GDA)
#include "gda/backend_gda.hpp"
#endif

namespace rocshmem {

extern Backend *backend;

void init_constant_memory(void) {
  std::string envstr;
  constmem_t constmem_values;

  memset(&constmem_values, 0, sizeof(constmem_t));

  envstr = envvar::gda::alltoallv_wg_algo;

  if (envstr.empty() || envstr.find("GET") != std::string::npos) {
    constmem_values.alltoall_wg_algo = gda::ALLTOALLV_WG_ALGO_GET;
  } else {
    constmem_values.alltoall_wg_algo = gda::ALLTOALLV_WG_ALGO_COPY;
  }

  constmem_values.my_pe = backend->getMyPE();
  constmem_values.num_pes = backend->getNumPEs();

  constmem_values.ipc_first_pe = backend->ipcImpl.ipc_first_pe;
  constmem_values.ipc_stride = backend->ipcImpl.ipc_stride;
  // ipc_shm_size == 0 means IPC disabled (fast early return on device).
  // Non-zero when IPC is available, regardless of stride pattern.
  constmem_values.ipc_shm_size = (backend->ipcImpl.pes_with_ipc_avail != nullptr)
                                 ? backend->ipcImpl.shm_size : 0;

  constmem_values.backend_type = backend->get_type();
#if defined(USE_GDA)
  if (constmem_values.backend_type == BackendType::GDA_BACKEND) {
    constmem_values.gda_provider = static_cast<GDABackend*>(backend)->get_gda_provider();
  }
#endif

  CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(constmem), &constmem_values, sizeof(constmem_t)));
}

}
