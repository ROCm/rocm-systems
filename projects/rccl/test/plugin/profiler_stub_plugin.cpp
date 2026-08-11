/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Profiler plugin stub for ProfilerPluginCloseTests. It loads, but exports no
// ncclProfiler_v* symbol, which is what makes ncclProfilerPluginInit() reject it.

extern "C" {

// Only here so the library has a symbol of its own.
int rcclTestProfilerStubMarker(void) { return 1; }

} // extern "C"
