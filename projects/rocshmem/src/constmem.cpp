#include "constmem.hpp"
#include "backend_bc.hpp"
#include "envvar.hpp"
#include "tdm.hpp"
#if defined(USE_GDA)
#include "gda/backend_gda.hpp"
#endif
#if defined(USE_IPC)
#include "ipc/backend_ipc.hpp"
#endif

/**
 * @file constmem.cpp
 */

namespace rocshmem {

extern Backend *backend;

uint32_t tdm_resolved_tile_bytes = 0;

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

  constmem_values.tdm_tile_bytes = envvar::tdm::tile_bytes;

#if defined(USE_TDM)
  {
    int device_id = 0;
    CHECK_HIP(hipGetDevice(&device_id));
    int max_shared_mem_per_block = 0;
    CHECK_HIP(hipDeviceGetAttribute(&max_shared_mem_per_block,
                                     hipDeviceAttributeMaxSharedMemoryPerBlock,
                                     device_id));

    // tile_dim0 is a 16-bit field (see tdm.hpp GROUP1), so the tile can
    // never encode more than 65535 elements regardless of available LDS.
    constexpr uint32_t element_bytes = 1u << tdm::FlatCopyElementLog2;
    constexpr uint32_t max_encodable_tile_bytes = 65535u * element_bytes;

    if (constmem_values.tdm_tile_bytes == 0) {
      // Auto-size: reserve 16KB for rocSHMEM's own (small) static
      // __shared__ usage plus headroom for the caller's, then halve the
      // rest for double buffering.
      constexpr size_t reserve_bytes = 16 * 1024;
      const size_t avail_bytes =
          (static_cast<size_t>(max_shared_mem_per_block) > reserve_bytes)
              ? static_cast<size_t>(max_shared_mem_per_block) - reserve_bytes
              : 0;
      uint32_t auto_tile_bytes =
          static_cast<uint32_t>((avail_bytes / 2) / element_bytes) * element_bytes;
      constmem_values.tdm_tile_bytes =
          (auto_tile_bytes < max_encodable_tile_bytes) ? auto_tile_bytes
                                                        : max_encodable_tile_bytes;
      if (constmem_values.tdm_tile_bytes == 0) {
        LOG_WARN(
            "Not enough LDS on this device (%d bytes/block) to auto-size a TDM "
            "tile after reserving %zu bytes; TDM disabled. Set "
            "ROCSHMEM_TDM_TILE_BYTES explicitly to override.",
            max_shared_mem_per_block, reserve_bytes);
      } else {
        LOG_INFO("TDM tile size auto-sized to %u bytes (device max shared mem/block: %d)",
                 constmem_values.tdm_tile_bytes, max_shared_mem_per_block);
      }
    } else {
      const size_t tdm_lds_bytes = tdm::lds_bytes_for_tile(constmem_values.tdm_tile_bytes);
      if (tdm_lds_bytes > static_cast<size_t>(max_shared_mem_per_block)) {
        LOG_ERROR_ABORT(
            "ROCSHMEM_TDM_TILE_BYTES=%u needs %zu bytes of LDS (double-buffered) "
            "but this device only has %d bytes of shared memory per block. "
            "Lower ROCSHMEM_TDM_TILE_BYTES (or set it to 0 to auto-size).",
            constmem_values.tdm_tile_bytes, tdm_lds_bytes, max_shared_mem_per_block);
      }
    }
  }
  tdm_resolved_tile_bytes = constmem_values.tdm_tile_bytes;
#endif

  constmem_values.backend_type = backend->get_type();
#if defined(USE_GDA)
  if (constmem_values.backend_type == BackendType::GDA_BACKEND) {
    constmem_values.gda_provider = static_cast<GDABackend*>(backend)->get_gda_provider();
  }
#endif
#if defined(USE_IPC)
  if (constmem_values.backend_type == BackendType::IPC_BACKEND) {
    // Mirrors the value IPCBackend::setup_wrk_sync_buffers() already
    // validated and used to size/stride the pWrk pool -- read from there
    // instead of re-deriving it, so device and host code can never disagree.
    constmem_values.reduce_ring_wrkdata_bytes = static_cast<uint32_t>(
        static_cast<IPCBackend*>(backend)->reduce_ring_wrkdata_bytes_);
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
