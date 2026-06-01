// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace rocprofiler_compute_tool
{
struct pc_sampling_record_t
{
    uint64_t code_object_id     = 0;
    uint64_t code_object_offset = 0;
    uint64_t dispatch_id        = 0;
    // Only stochastic samples carry an issued/not-issued bit. Host-trap records
    // have no such field; leaving it unset omits "wave_issued" from the JSON so
    // the output matches what the rocprofiler-sdk tool emits for host-trap.
    std::optional<bool>        wave_issued;
    bool                       is_stochastic = false;
    std::optional<std::string> stall_reason;  // SDK string WITH full prefix; absent when issued
};
}  // namespace rocprofiler_compute_tool
