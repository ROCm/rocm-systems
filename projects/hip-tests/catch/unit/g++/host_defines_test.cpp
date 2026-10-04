/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Compiled with g++ to check the HIP headers as seen by a host compiler.
#include <hip/hip_runtime_api.h>

// Reserved identifiers that system headers use as attribute names must not be
// macros: libstdc++, libc++ and glibc spell noinline as
// __attribute__((__noinline__)) or [[__gnu__::__noinline__]].
#ifdef __noinline__
#error "__noinline__ must not be defined for host compilers"
#endif

bool test_host_defines() { return true; }
