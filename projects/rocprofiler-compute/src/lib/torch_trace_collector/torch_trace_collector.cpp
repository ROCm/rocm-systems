// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "process_state.h"

namespace torch_trace_collector::detail
{

ProcessState& process_state()
{
    static ProcessState state;
    return state;
}

}  // namespace torch_trace_collector::detail
