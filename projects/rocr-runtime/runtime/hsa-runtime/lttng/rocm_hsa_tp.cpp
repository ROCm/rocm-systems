/* Tracepoint provider package (TPP) for the rocm_hsa provider. The whole
 * body is guarded on HSA_ENABLE_LTTNG_UST so an
 * `-DROCR_ENABLE_LTTNG_UST=OFF` build compiles this TU to nothing without
 * touching the source-list in CMakeLists.txt. */
#if defined(HSA_ENABLE_LTTNG_UST) && HSA_ENABLE_LTTNG_UST
#define LTTNG_UST_TRACEPOINT_CREATE_PROBES
#define LTTNG_UST_TRACEPOINT_DEFINE
#include "rocm_hsa_tp.h"
#endif
