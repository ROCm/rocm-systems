// Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// with the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimers in the
// documentation and/or other materials provided with the distribution.
//
// * Neither the names of Advanced Micro Devices, Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this Software without specific prior written permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.

#pragma once

#include <cstdint>
#include <set>
#include <string>

namespace rocprofsys
{
namespace pmc
{

/**
 * @brief Version information for device providers and drivers.
 */
struct version
{
    struct
    {
        uint32_t major   = 0;
        uint32_t minor   = 0;
        uint32_t release = 0;
    } numeric_representation;
    std::string string_representation;
};

/**
 * @brief Device selection mode for filtering devices.
 */
enum class device_selection_mode : uint8_t
{
    ALL,      ///< Include all devices
    NONE,     ///< Exclude all devices
    SPECIFIC  ///< Include only specific devices by index
};

/**
 * @brief Device filter configuration.
 */
struct device_filter
{
    device_selection_mode mode = device_selection_mode::ALL;
    std::set<size_t>      indices;  ///< Device indices when mode is SPECIFIC
};

}  // namespace pmc
}  // namespace rocprofsys
