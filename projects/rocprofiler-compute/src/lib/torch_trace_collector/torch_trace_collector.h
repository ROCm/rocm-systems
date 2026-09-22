// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

__attribute__((visibility("default"))) int torch_trace_collector_install(void);
__attribute__((visibility("default"))) int torch_trace_collector_push_launcher_tid(uint64_t launcher_tid);
__attribute__((visibility("default"))) int torch_trace_collector_pop_launcher_tid(void);

#ifdef __cplusplus
}
#endif
