/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * Tracepoint provider package (TPP) for the rocm_hip provider. This is the
 * single translation unit per .so that defines the LTTng tracepoint
 * registration symbols. Including rocm_hip_tp.h with both
 * LTTNG_UST_TRACEPOINT_CREATE_PROBES and LTTNG_UST_TRACEPOINT_DEFINE
 * defined causes the LTTng macros to expand into the registration data.
 *
 * The whole body is guarded on HIP_ENABLE_LTTNG_UST so that an
 * `-DHIP_ENABLE_LTTNG_UST=OFF` build compiles this TU to nothing without
 * needing to touch the source-list in CMakeLists.txt.
 */
#if defined(HIP_ENABLE_LTTNG_UST) && HIP_ENABLE_LTTNG_UST
#define LTTNG_UST_TRACEPOINT_CREATE_PROBES
#define LTTNG_UST_TRACEPOINT_DEFINE
#include "rocm_hip_tp.h"
#endif
