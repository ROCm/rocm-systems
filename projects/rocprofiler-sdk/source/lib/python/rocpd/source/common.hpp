// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/output/agent_info.hpp"
#include "lib/output/kernel_symbol_info.hpp"
#include "lib/output/metadata.hpp"

#include <sqlite3.h>
#include <cereal/archives/json.hpp>

#include <cstdint>
#include <deque>
#include <sstream>
#include <string>
#include <utility>

namespace rocpd
{
namespace common
{
template <typename FuncT, typename... Args>
void
read_json_string(const std::string& inp, FuncT&& _func, Args&&... _args)
{
    using json_archive = cereal::JSONInputArchive;

    {
        auto json_ss = std::stringstream{inp};
        auto ar      = json_archive{json_ss};
        std::forward<FuncT>(_func)(ar, std::forward<Args>(_args)...);
    }
}
}  // namespace common
}  // namespace rocpd
