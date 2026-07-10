/* Tracepoint definition TU for the rocm_hsa provider, compiled into
 * hsa-runtime64 itself.
 *
 * hsa-runtime64 both defines and creates the tracepoint probes here, and
 * links liblttng-ust directly. The whole body is guarded on
 * HSA_ENABLE_LTTNG_UST so that an `-DROCR_ENABLE_LTTNG_UST=OFF` build
 * compiles this TU to nothing without needing to touch the source-list in
 * CMakeLists.txt.
 */
#if defined(HSA_ENABLE_LTTNG_UST) && HSA_ENABLE_LTTNG_UST

#define LTTNG_UST_TRACEPOINT_CREATE_PROBES
#define LTTNG_UST_TRACEPOINT_DEFINE

#include "rocm_hsa_tp.h"
#endif
