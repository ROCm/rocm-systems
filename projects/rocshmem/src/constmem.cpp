#include "constmem.hpp"
#include "backend_bc.hpp"
#include "envvar.hpp"
#include "lds_stage.hpp"
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

#if defined(USE_SDMA)
  // Follow the node-local agreement rather than the environment: after an agreed fallback the env
  // still enables SDMA but the queue handles are null, and every rank must pick the same alltoall.
  const SdmaImpl &sdma = backend->ipcImpl.sdmaImpl_;
  constmem_values.ipc_sdma_threshold =
      sdma.sdmaEnabled ? sdma.sdmaThreshold : SDMA_THRESHOLD_DISABLED;
#else
  constmem_values.ipc_sdma_threshold = SDMA_THRESHOLD_DISABLED;
#endif

  constmem_values.heap_base =
      reinterpret_cast<uintptr_t>(backend->heap.get_local_heap_base());
  constmem_values.heap_size = backend->heap.get_size();

  constmem_values.lds_stage_tile_bytes = envvar::lds_stage::tile_bytes;
  constmem_values.lds_stage_put_threshold = envvar::lds_stage::put_threshold;
  constmem_values.lds_stage_get_threshold = envvar::lds_stage::get_threshold;

#if defined(USE_LDS_STAGE)
  {
    int device_id = 0;
    CHECK_HIP(hipGetDevice(&device_id));
    int max_shared_mem_per_block = 0;
    CHECK_HIP(hipDeviceGetAttribute(&max_shared_mem_per_block,
                                     hipDeviceAttributeMaxSharedMemoryPerBlock,
                                     device_id));

    constexpr uint32_t element_bytes = 1u << lds_stage::FlatCopyElementLog2;
    // TDM's tile_dim0 is a 16-bit field (see lds_stage.hpp GROUP1), so the
    // tile can never encode more than 65535 elements regardless of available
    // LDS. Both backends are gfx1250-only, so apply the same cap uniformly.
    constexpr uint32_t max_encodable_tile_bytes = 65535u * element_bytes;
    // Reserve 16KB for rocSHMEM's own (small) static __shared__ usage plus
    // headroom for the caller's, then halve the rest for double buffering.
    constexpr size_t reserve_bytes = 16 * 1024;
    const size_t avail_bytes =
        (static_cast<size_t>(max_shared_mem_per_block) > reserve_bytes)
            ? static_cast<size_t>(max_shared_mem_per_block) - reserve_bytes
            : 0;
    const uint32_t avail_tile_bytes =
        static_cast<uint32_t>((avail_bytes / 2) / element_bytes) * element_bytes;
    const uint32_t fittable_tile_bytes =
        (avail_tile_bytes < max_encodable_tile_bytes) ? avail_tile_bytes
                                                      : max_encodable_tile_bytes;

    if (constmem_values.lds_stage_tile_bytes == 0) {
      constmem_values.lds_stage_tile_bytes = fittable_tile_bytes;
      if (constmem_values.lds_stage_tile_bytes == 0) {
        LOG_WARN(
            "Not enough LDS on this device (%d bytes/block) to auto-size an "
            "LDS-staged copy tile after reserving %zu bytes; the fast path is "
            "disabled. Set ROCSHMEM_LDS_STAGE_TILE_BYTES explicitly to override.",
            max_shared_mem_per_block, reserve_bytes);
      } else {
        LOG_INFO("LDS-staged copy tile size auto-sized to %u bytes (device max shared mem/block: %d)",
                 constmem_values.lds_stage_tile_bytes, max_shared_mem_per_block);
      }
    } else {
      const uint32_t aligned_tile_bytes = constmem_values.lds_stage_tile_bytes -
                                           (constmem_values.lds_stage_tile_bytes % element_bytes);
      if (aligned_tile_bytes != constmem_values.lds_stage_tile_bytes) {
        LOG_WARN(
            "ROCSHMEM_LDS_STAGE_TILE_BYTES=%u is not a multiple of the %u-byte "
            "element size; rounding down to %u bytes.",
            constmem_values.lds_stage_tile_bytes, element_bytes, aligned_tile_bytes);
        constmem_values.lds_stage_tile_bytes = aligned_tile_bytes;
      }
      if (constmem_values.lds_stage_tile_bytes > max_encodable_tile_bytes) {
        LOG_WARN(
            "ROCSHMEM_LDS_STAGE_TILE_BYTES=%u exceeds the 65535-element "
            "tile_dim0 encoding cap (%u bytes); falling back to %u bytes.",
            constmem_values.lds_stage_tile_bytes, max_encodable_tile_bytes, fittable_tile_bytes);
        constmem_values.lds_stage_tile_bytes = fittable_tile_bytes;
      } else {
        // fittable_tile_bytes is always <= avail_bytes/2, so it's guaranteed
        // to pass this check -- only worth checking when we didn't just fall
        // back to it above.
        const size_t lds_stage_bytes =
            lds_stage::lds_bytes_for_tile(constmem_values.lds_stage_tile_bytes);
        if (lds_stage_bytes > static_cast<size_t>(max_shared_mem_per_block)) {
          // Don't abort: fall back to whatever tile size does fit (possibly 0,
          // which disables the fast path for this run) rather than taking the
          // job down.
          LOG_WARN(
              "ROCSHMEM_LDS_STAGE_TILE_BYTES=%u needs %zu bytes of LDS (double-buffered) "
              "but this device only has %d bytes of shared memory per block; "
              "falling back to %u bytes.",
              constmem_values.lds_stage_tile_bytes, lds_stage_bytes, max_shared_mem_per_block,
              fittable_tile_bytes);
          constmem_values.lds_stage_tile_bytes = fittable_tile_bytes;
        }
      }
    }
  }
  backend->lds_stage_resolved_tile_bytes = constmem_values.lds_stage_tile_bytes;
#endif

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
