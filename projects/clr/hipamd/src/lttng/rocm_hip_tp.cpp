/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * Tracepoint provider package (TPP) for the rocm_hip provider. This is the
 * single translation unit per .so that defines the LTTng tracepoint
 * registration symbols. Including rocm_hip_tp.h with both
 * LTTNG_UST_TRACEPOINT_CREATE_PROBES and LTTNG_UST_TRACEPOINT_DEFINE
 * defined causes the LTTng macros to expand into the registration data.
 */
#define LTTNG_UST_TRACEPOINT_CREATE_PROBES
#define LTTNG_UST_TRACEPOINT_DEFINE
#include "rocm_hip_tp.h"
