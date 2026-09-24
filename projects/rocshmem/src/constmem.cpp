#include "constmem.hpp"
#include "backend_bc.hpp"
#include "envvar.hpp"
#if defined(USE_GDA)
#include "gda/backend_gda.hpp"
#endif

/**
 * @file constmem.cpp
 */

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
  constmem_values.heap_base =
      reinterpret_cast<uintptr_t>(backend->heap.get_local_heap_base());
  constmem_values.heap_size = backend->heap.get_size();

  constmem_values.backend_type = backend->get_type();
#if defined(USE_GDA)
  if (constmem_values.backend_type == BackendType::GDA_BACKEND) {
    constmem_values.gda_provider = static_cast<GDABackend*>(backend)->get_gda_provider();
  }
#endif

  CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(constmem), &constmem_values, sizeof(constmem_t)));
}

}  // namespace rocshmem

/**
 * @brief Exported C function for GIN QP factory to initialize __constant__ constmem.
 *
 * Lives in librocshmem.a so HIP_SYMBOL(constmem) resolves via device linking.
 * Callable from librccl.so via -rdynamic symbol export.
 *
 * @param[in] provider GDA provider enumerator from rocshmem::gda::provider / rocshmem::GDAProvider.
 * @param[in] rank Rank of this PE.
 */
extern "C" void rocshmem_gin_init_constmem(int provider, int rank) {
  using namespace rocshmem;

  // Initialize constmem.gda_provider for QP device dispatch
  GDAProvider gda_prov = static_cast<GDAProvider>(provider);
  constmem_t* cm_addr{nullptr};
  if (hipGetSymbolAddress(reinterpret_cast<void**>(&cm_addr),
                          HIP_SYMBOL(constmem)) == hipSuccess) {
    CHECK_HIP(hipMemcpy(&cm_addr->gda_provider, &gda_prov, sizeof(gda_prov), hipMemcpyDefault));
  }

  // Initialize logd_constants for device-side error reporting
  log_pe_number = rank;
  uint32_t log_flags = 0;
  if (envvar::log_flags.show_error) log_flags |= logd_constants::SHOW_ERROR;
  if (envvar::log_flags.show_warn)  log_flags |= logd_constants::SHOW_WARN;
  if (envvar::log_flags.show_info)  log_flags |= logd_constants::SHOW_INFO;
  if (envvar::log_flags.show_trace) log_flags |= logd_constants::SHOW_TRACE;
  if (envvar::log_flags.show_color) log_flags |= logd_constants::SHOW_COLOR;
  struct logd_constants host_logd{rank, log_flags};
  struct logd_constants* logd_addr{nullptr};
  if (hipGetSymbolAddress(reinterpret_cast<void**>(&logd_addr),
                          HIP_SYMBOL(logd_constants)) == hipSuccess) {
    CHECK_HIP(hipMemcpy(logd_addr, &host_logd, sizeof(host_logd), hipMemcpyDefault));
  }
}
