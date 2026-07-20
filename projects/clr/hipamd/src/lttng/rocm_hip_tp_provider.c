/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * Tracepoint provider package (TPP) for the rocm_hip provider, built as a
 * separate DSO (rocm_hip_lttng_provider) so amdhip64 does not need to link
 * liblttng-ust directly (see HIP_LTTNG_UST_LINK_MODE=dynamic in
 * CMakeLists.txt). Load this .so at runtime (e.g. `LD_PRELOAD=`) to
 * activate rocm_hip:* tracepoint emission -- LTTng-UST's documented
 * "dynamic loading" pattern. See shared/lttng/lttng-ust/README.md
 * ("Dynamic loading") and doc/examples/demo/tp3.c for the upstream
 * reference example this file mirrors.
 *
 * Compiled as plain C (per LTTng's own recommendation for probe TUs); the
 * provider header and the curated per-API event header it includes are
 * both C-compatible.
 */
#define LTTNG_UST_TRACEPOINT_CREATE_PROBES
#include "rocm_hip_tp.h"
