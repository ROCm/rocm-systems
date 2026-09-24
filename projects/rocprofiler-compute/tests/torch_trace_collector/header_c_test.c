// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "torch_trace_collector.h"

_Static_assert(TORCH_TRACE_COLLECTOR_ABI_REVISION == 1U, "unexpected ABI revision");
_Static_assert(sizeof(struct torch_trace_collector_stats) ==
                   TORCH_TRACE_COLLECTOR_STATS_ABI_SIZE,
               "unexpected statistics ABI size");

void
torch_trace_collector_check_c_declarations(void)
{
    (void) &torch_trace_collector_abi_revision;
    (void) &torch_trace_collector_install;
    (void) &torch_trace_collector_uninstall;
    (void) &torch_trace_collector_is_installed;
    (void) &torch_trace_collector_push_user_scope;
    (void) &torch_trace_collector_pop_user_scope;
    (void) &torch_trace_collector_get_stats;
}
