/* Tracepoint definition TU for the rocm_hsa provider, compiled into
 * hsa-runtime64 itself.
 *
 * Two link modes (see ROCR_LTTNG_UST_LINK_MODE in CMakeLists.txt):
 *
 *  - dynamic (default): this TU only DEFINEs the tracepoint registration
 *    data and declares dynamic linkage to the probe-registration symbols
 *    (LTTNG_UST_TRACEPOINT_PROBE_DYNAMIC_LINKAGE) -- it does NOT create
 *    the probes and hsa-runtime64 does NOT link liblttng-ust. Probe
 *    creation lives in the separate rocm_hsa_lttng_provider DSO (see
 *    lttng/rocm_hsa_tp_provider.c), which is loaded at runtime (e.g. via
 *    LD_PRELOAD) to activate tracing. This is LTTng-UST's documented
 *    "dynamic loading" pattern -- see
 *    shared/lttng/lttng-ust/README.md ("Dynamic loading").
 *
 *  - direct (legacy fallback, kept only for A/B comparison during
 *    development): this TU both defines and creates the tracepoint
 *    probes, and hsa-runtime64 links liblttng-ust directly, exactly as
 *    before this split.
 *
 * The whole body is guarded on HSA_ENABLE_LTTNG_UST so that an
 * `-DROCR_ENABLE_LTTNG_UST=OFF` build compiles this TU to nothing without
 * needing to touch the source-list in CMakeLists.txt.
 */
#if defined(HSA_ENABLE_LTTNG_UST) && HSA_ENABLE_LTTNG_UST

#if defined(HSA_LTTNG_UST_DYNAMIC_LINKAGE) && HSA_LTTNG_UST_DYNAMIC_LINKAGE
#define LTTNG_UST_TRACEPOINT_DEFINE
#define LTTNG_UST_TRACEPOINT_PROBE_DYNAMIC_LINKAGE
#else
#define LTTNG_UST_TRACEPOINT_CREATE_PROBES
#define LTTNG_UST_TRACEPOINT_DEFINE
#endif

#include "rocm_hsa_tp.h"
#endif
