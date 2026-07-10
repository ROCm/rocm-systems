/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * Tracepoint definition TU for the rocm_hip provider, compiled into
 * amdhip64 itself.
 *
 * amdhip64 both defines and creates the tracepoint probes here, and links
 * liblttng-ust directly. The whole body is guarded on HIP_ENABLE_LTTNG_UST
 * so that an `-DHIP_ENABLE_LTTNG_UST=OFF` build compiles this TU to nothing
 * without needing to touch the source-list in CMakeLists.txt.
 */
#if defined(HIP_ENABLE_LTTNG_UST) && HIP_ENABLE_LTTNG_UST

#define LTTNG_UST_TRACEPOINT_CREATE_PROBES
#define LTTNG_UST_TRACEPOINT_DEFINE

#include "rocm_hip_tp.h"
#endif
