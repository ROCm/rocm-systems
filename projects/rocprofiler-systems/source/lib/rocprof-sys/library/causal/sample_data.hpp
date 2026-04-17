// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/defines.hpp"
#include "core/timemory.hpp"

#include <chrono>
#include <cstdint>

namespace rocprofsys
{
namespace causal
{
struct sample_data
{
    uintptr_t        address = 0x0;
    mutable uint64_t count   = 0;

    bool operator==(sample_data _v) const { return (address == _v.address); }
    bool operator!=(sample_data _v) const { return !(*this == _v); }
    bool operator<(sample_data _v) const { return (address < _v.address); }
    bool operator>(sample_data _v) const { return (address > _v.address); }
    bool operator<=(sample_data _v) const { return (address <= _v.address); }
    bool operator>=(sample_data _v) const { return (address >= _v.address); }

    template <typename ArchiveT>
    void serialize(ArchiveT& ar, const unsigned)
    {
        ar(cereal::make_nvp("address", address), cereal::make_nvp("count", count));
    }
};

std::map<uint32_t, std::vector<sample_data>>
get_samples();

void
add_samples(uint32_t, const std::vector<uintptr_t>&);

std::vector<sample_data> get_samples(uint32_t);

void add_sample(uint32_t, uintptr_t, uint64_t = 1);

void
add_samples(uint32_t, const std::map<uintptr_t, uint64_t>&);
}  // namespace causal
}  // namespace rocprofsys
