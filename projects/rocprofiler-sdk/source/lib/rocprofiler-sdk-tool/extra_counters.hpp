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

#pragma once

#include <string>
#include <string_view>

namespace rocprofiler
{
namespace tool
{
// The initial configure call owns the loaded definition snapshot. A reattach with no
// definition variables, or with the same nonempty FILE path retained by an older register,
// reuses that snapshot. New helper-injected contents are distinguishable because the helper
// explicitly supplies FILE="" alongside CONTENTS.
inline bool
reuse_extra_counters_snapshot(bool             refreshed_file_set,
                              std::string&     refreshed_file,
                              bool             refreshed_contents_set,
                              std::string&     refreshed_contents,
                              std::string_view original_file,
                              std::string_view original_contents)
{
    const auto no_new_definition = !refreshed_file_set && !refreshed_contents_set;
    const auto same_file_path =
        refreshed_file_set && !refreshed_file.empty() && refreshed_file == original_file;
    if(!no_new_definition && !same_file_path) return false;

    refreshed_file     = original_file;
    refreshed_contents = original_contents;
    return true;
}
}  // namespace tool
}  // namespace rocprofiler
