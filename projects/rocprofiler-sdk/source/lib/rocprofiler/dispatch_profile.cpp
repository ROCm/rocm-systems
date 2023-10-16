#include <rocprofiler/rocprofiler.h>

#include "lib/rocprofiler/aql/helpers.hpp"
#include "lib/rocprofiler/counters/core.hpp"
#include "lib/rocprofiler/counters/evaluate_ast.hpp"
#include "lib/rocprofiler/counters/metrics.hpp"
#include "lib/rocprofiler/hsa/agent_cache.hpp"

extern "C" {
/**
 * @brief Configure Dispatch Profile Counting Service.
 *
 * @param [in] context_id
 * @param [in] agent_id
 * @param [in] buffer_id
 * @param [in] callback
 * @param [in] callback_data_args
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_configure_dispatch_profile_counting_service(
    rocprofiler_context_id_t                         context_id,
    rocprofiler_profile_config_id_t                  profile,
    rocprofiler_profile_counting_dispatch_callback_t callback,
    void*                                            callback_data_args)
{
    return rocprofiler::counters::configure_dispatch(
               context_id, profile.handle, callback, callback_data_args)
               ? ROCPROFILER_STATUS_SUCCESS
               : ROCPROFILER_STATUS_ERROR;
}
}
