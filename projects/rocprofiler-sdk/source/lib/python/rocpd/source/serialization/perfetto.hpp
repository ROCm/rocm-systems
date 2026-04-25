// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "lib/common/logging.hpp"
#include "lib/common/mpl.hpp"

#include <rocprofiler-sdk/cxx/perfetto.hpp>

#include <cereal/cereal.hpp>
#include <cereal/external/base64.hpp>

#include <utility>

namespace cereal
{
class PerfettoAnnotationOutputArchive
: public OutputArchive<PerfettoAnnotationOutputArchive, AllowEmptyClassElision>
{
public:
    using perfetto_event_context_t = ::rocprofiler::sdk::perfetto_event_context_t;

    PerfettoAnnotationOutputArchive(perfetto_event_context_t& ctx)
    : OutputArchive<PerfettoAnnotationOutputArchive, AllowEmptyClassElision>(this)
    , m_ctx{ctx}
    {}

    ~PerfettoAnnotationOutputArchive() CEREAL_NOEXCEPT override = default;

    //! Annotates the data in the perfetto event
    template <typename Tp, typename Up>
    void save(Tp&& name, Up&& value)
    {
        rocprofiler::sdk::add_perfetto_annotation(
            m_ctx, std::forward<Tp>(name), std::forward<Up>(value));
    }

private:
    perfetto_event_context_t& m_ctx;
};

//! Saving for POD types to binary
template <typename Tp>
inline void
CEREAL_SAVE_FUNCTION_NAME(PerfettoAnnotationOutputArchive& ar, const NameValuePair<Tp>& t)
{
    ar.save(t.name, t.value);
}
}  // namespace cereal

// register archives for polymorphic support
CEREAL_REGISTER_ARCHIVE(cereal::PerfettoAnnotationOutputArchive)
