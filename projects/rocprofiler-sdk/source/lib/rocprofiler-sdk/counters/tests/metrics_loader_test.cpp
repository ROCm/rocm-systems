// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/rocprofiler-sdk/counters/metrics.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <string_view>

namespace
{
namespace counters = ::rocprofiler::counters;
}

TEST(metrics_loader, duplicate_architecture_definitions)
{
    const auto test_yaml = std::string{R"(
rocprofiler-sdk:
  counters:
    - name: OVERLAP
      description: Identical partial overlap
      definitions:
        - architectures: [gfx906, gfx942]
          expression: simd_count
        - architectures: [gfx942, gfx950]
          expression: simd_count
    - name: ARCH_SPECIFIC
      description: Architecture-specific definitions
      definitions:
        - architectures: [gfx906]
          expression: simd_count
        - architectures: [gfx942]
          expression: simd_per_cu
    - name: EVENT_EQUIVALENT
      description: Equivalent event spellings
      definitions:
        - architectures: [gfx942]
          block: SQ
          event: 1
        - architectures: [gfx942]
          block: SQ
          event: "001"
)"};

    auto custom_definition = counters::CustomCounterDefinition{};
    custom_definition.data = test_yaml;
    ASSERT_EQ(counters::setCustomCounterDefinition(custom_definition), ROCPROFILER_STATUS_SUCCESS);

    auto loaded_metrics = counters::loadMetrics();
    auto expect_metric  = [&loaded_metrics](std::string_view arch,
                                           std::string_view name,
                                           std::string_view expression,
                                           std::string_view event) {
        const auto& metrics = loaded_metrics->arch_to_metric.at(std::string{arch});
        auto matches = [&](const auto& metric) { return std::string_view{metric.name()} == name; };
        auto metric  = std::find_if(metrics.begin(), metrics.end(), matches);

        ASSERT_EQ(std::count_if(metrics.begin(), metrics.end(), matches), 1)
            << name << " on " << arch;
        ASSERT_NE(metric, metrics.end());
        EXPECT_EQ(std::string_view{metric->expression()}, expression);
        EXPECT_EQ(std::string_view{metric->event()}, event);
    };

    expect_metric("gfx906", "OVERLAP", "simd_count", "");
    expect_metric("gfx942", "OVERLAP", "simd_count", "");
    expect_metric("gfx950", "OVERLAP", "simd_count", "");
    expect_metric("gfx906", "ARCH_SPECIFIC", "simd_count", "");
    expect_metric("gfx942", "ARCH_SPECIFIC", "simd_per_cu", "");
    expect_metric("gfx942", "EVENT_EQUIVALENT", "", "1");
}
