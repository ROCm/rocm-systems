/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "configuration.h"
#include "context.h"
#include "hipfile-warnings.h"
#include "hipfile.h"

#include "io-test.h"
#include "io-verify.h"
#include "test-common.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <optional>
#include <string>
#include <vector>

namespace hipFileTest {

// ---------------------------------------------------------------------------
// Kernel launch shape.
// ---------------------------------------------------------------------------
enum class GridMode { Default, OneWorkgroup, ManyWorkgroups };

constexpr unsigned kManyWorkgroups = 300;

inline dim3
gridFor(GridMode mode, size_t n)
{
    switch (mode) {
        case GridMode::OneWorkgroup:
            return dim3(1);
        case GridMode::ManyWorkgroups:
            return dim3(kManyWorkgroups);
        case GridMode::Default:
        default:
            return defaultGrid(n);
    }
}

// ---------------------------------------------------------------------------
// One parameter type for every data-modification suite.
// ---------------------------------------------------------------------------
struct IoTestScenario {
    std::string               name{};                             // the full gtest case name
    IoTestBackend             backend  = IoTestBackend::Fallback; // backend fulfilling the I/O request
    size_t                    io_bytes = 4_KiB;                   // bytes transferred through the hipFile API
    hoff_t                    file_off = 0;                       // where the data sits in the file
    hoff_t                    buf_off  = 0;                       // where the data sits in the device buffer
    size_t                    stride   = 1;                       // 1 == modify every element
    GridMode                  grid     = GridMode::Default;       // workgroup count the kernel launches with
    std::optional<ExtendCase> ext      = std::nullopt;            // set only for extending writes
};

// A value of one of the parameters of an `IoTestScenario` and the string to reflect
// it with in the test name.
template <class T> struct Axis {
    T           value;
    std::string name;
};

// ---------------------------------------------------------------------------
// This class is responsible for building a series of IoTestScenarios. Allowing you
// to use a single class/object to represent all of the parameters of a test instance,
// as opposed to many small classes.
//
// This class contains functions that mimic functionality of GTest's `param_generator`s
// (e.g., `over` for `testing::Combine`, `add` for `testing::Values`).
// ---------------------------------------------------------------------------
class IoTestScenarioSet {
public:
    explicit IoTestScenarioSet(const IoTestScenario &base) : rows_{base}
    {
    }

    // This function is like GTest's `testing::Combine`. You specify a `field` of `IoTestScenario`
    // that you would like to vary over a range of values defined by `axis`, and `over`
    // multiplies each existing `IoTestScenario` in `rows_` to have an instance that takes on
    // each of the values in `axis`.
    template <class F, class Range> IoTestScenarioSet &over(F IoTestScenario::*field, const Range &axis)
    {
        std::vector<IoTestScenario> expanded;
        for (const IoTestScenario &row : rows_) {
            for (const auto &a : axis) {
                IoTestScenario next = row;
                next.*field         = a.value;
                next.name           = next.name.empty() ? a.name : next.name + "_" + a.name;
                expanded.push_back(next);
            }
        }
        rows_ = expanded;
        return *this;
    }

    // If you were to repeatedly use this function, this function is like GTest's `testing::Values`,
    // except you can append to an existing set of `IoTestScenario`.
    // This function should we used for adding bespoke test parameter scenarios that cannot be
    // expressed by a matrix of individual parameters that are combined by `over`.
    IoTestScenarioSet &add(const IoTestScenario &one)
    {
        rows_.push_back(one);
        return *this;
    }

    // This function combines two sets of `IoTestScenario`, which typically isn't possible in GTest,
    // unless you used multiple `INSTANTIATE_TEST_SUITE_P` calls.
    IoTestScenarioSet &add(const IoTestScenarioSet &other)
    {
        rows_.insert(rows_.end(), other.rows_.begin(), other.rows_.end());
        return *this;
    }

    std::vector<IoTestScenario> build() const
    {
        return rows_;
    }

private:
    std::vector<IoTestScenario> rows_;
};

inline std::string
ioTestScenarioName(const testing::TestParamInfo<IoTestScenario> &info)
{
    return info.param.name;
}

// ---------------------------------------------------------------------------
// Axis values that are common between many test suites.
// ---------------------------------------------------------------------------
HIPFILE_WARN_NO_EXIT_DTOR_OFF
inline const std::array<Axis<IoTestBackend>, 2> kBackends{{
    {IoTestBackend::Fastpath, "Fastpath"},
    {IoTestBackend::Fallback, "Fallback"},
}};

inline const std::array<Axis<size_t>, 3> kCombinedSizes{{
    {4_KiB, "sub_chunk"},
    {kChunkBytes + 4_KiB, "cross_chunk"},
    {2 * kChunkBytes, "multi_chunk"},
}};
HIPFILE_WARN_NO_EXIT_DTOR_ON

} // namespace hipFileTest
