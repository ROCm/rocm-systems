// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include "common/defines.h"
#include "core/trace_cache/cacheable.hpp"
#include "core/trace_cache/sample_processor_interface.hpp"
#include "core/trace_cache/sample_type.hpp"

#include <memory>
#include <utility>
#include <vector>

namespace rocprofsys::trace_cache
{

struct sample_processor_t
{
    void clear_handlers() { m_processors.clear(); }

    void add_handler(std::shared_ptr<sample_processor_interface> handler)
    {
        m_processors.push_back(std::move(handler));
    }

    template <typename SampleType>
    ROCPROFSYS_INLINE void handle_sample(const SampleType& sample) const
    {
        for(const auto& processor : m_processors)
        {
            processor->handle(sample);
        }
    }

    ROCPROFSYS_INLINE void prepare_for_processing() const noexcept
    {
        for(const auto& processor : m_processors)
        {
            processor->prepare_for_processing();
        }
    }

    ROCPROFSYS_INLINE void finalize_processing() const noexcept
    {
        for(const auto& processor : m_processors)
        {
            processor->finalize_processing();
        }
    }

    [[nodiscard]] ROCPROFSYS_INLINE bool is_empty() const noexcept
    {
        return m_processors.empty();
    }

private:
    std::vector<std::shared_ptr<sample_processor_interface>> m_processors;
};

}  // namespace rocprofsys::trace_cache
