// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
//

#pragma once

#include <rocprofiler-sdk/fwd.h>

#include <cstdint>

namespace rocprofiler
{
namespace sdk
{
namespace version
{
/**
 * @brief This function extracts the `<major>.<minor>.<patch>` from a single integer which encodes
 * this triplet by a factor of `N`. E.g. version 3.2.1 using N=100 is represented by the value of
 * 30201; version 3.2.1 using N=1000 is represented by the value of 3002001.
 *
 * For a given factor `N`, the major version can be extracted by
 * dividing the version value by (N * N); the minor version can be extracted via
 * computing the modulus of (N * N) and then divided by N; the patch version is simply the modulus
 * of N.
 */
template <uint64_t FactorV = 100>
constexpr rocprofiler_version_triplet_t
compute_version_triplet(uint64_t version)
{
    constexpr auto factor = FactorV;

    return rocprofiler_version_triplet_t{
        .major = static_cast<uint32_t>(version / (factor * factor)),
        .minor = static_cast<uint32_t>((version % (factor * factor)) / factor),
        .patch = static_cast<uint32_t>(version % factor)};
}
}  // namespace version
}  // namespace sdk
}  // namespace rocprofiler
