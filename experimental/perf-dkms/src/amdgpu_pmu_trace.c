/*
 * amdgpu_pmu_trace.c - Tracepoint definitions for AMDGPU PMU
 *
 * This file creates the actual tracepoint code by defining CREATE_TRACE_POINTS
 * before including the tracepoint header. This must be done exactly once in
 * the entire kernel module.
 *
 * The CREATE_TRACE_POINTS macro causes the TRACE_EVENT() macros to generate
 * the actual tracepoint functions (e.g., trace_counter_sample()) instead of
 * just declaring them.
 */

#define CREATE_TRACE_POINTS
#include "amdgpu_pmu_trace.h"
