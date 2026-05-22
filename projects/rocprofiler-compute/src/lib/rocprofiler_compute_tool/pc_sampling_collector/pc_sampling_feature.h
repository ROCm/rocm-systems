// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include "pc_sampling_collector.h"
#include "synchronized.hpp"

#include <cstdint>
#include <filesystem>
#include <utility>

namespace rocm_compute
{
enum class PcSamplingMode : uint8_t
{
    Disabled,
    Stochastic,
    HostTrap
};

struct pc_sampling_feature_t
{
    pc_sampling_feature_t(PcSamplingMode               feature_mode,
                          std::filesystem::path        code_obj_output_filename,
                          pc_sampling_collector_t::ptr feature_collector)
        : mode(feature_mode)
        , output_filename(std::move(code_obj_output_filename))
        , collector(std::move(feature_collector))
    {
    }

    PcSamplingMode                               mode;
    std::filesystem::path                        output_filename;
    synchronized_t<pc_sampling_collector_t::ptr> collector;
};
}  // namespace rocm_compute
