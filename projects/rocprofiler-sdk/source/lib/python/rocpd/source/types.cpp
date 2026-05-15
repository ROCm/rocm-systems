// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/python/rocpd/source/types.hpp"
#include "lib/python/rocpd/source/common.hpp"

#include "lib/output/node_info.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/cxx/serialization.hpp>

#include <cereal/cereal.hpp>

#include <cstdint>

namespace rocpd
{
namespace types
{
// std::string
// blob::hexdigest() const
// {
//     auto _oss = std::ostringstream{};
//     for(auto itr : (*this))
//         _oss << std::hex << std::setw(2) << std::setfill('0') <<
//         static_cast<int>(itr);
//     return _oss.str();
// }

// std::string
// blob::hexliteral() const
// {
//     auto _oss = std::ostringstream{};
//     _oss << "X'";
//     for(auto itr : (*this))
//         _oss << std::hex << std::setw(2) << std::setfill('0') <<
//         static_cast<int>(itr);
//     _oss << "'";
//     return _oss.str();
// }

void
agent::load_extdata()
{
    if(has_extdata())
        common::read_json_string(
            extdata, [](auto& _ar, base_type& _base) { cereal::load(_ar, _base); }, base());
}

region::decoded_extdata
region::get_extdata() const
{
    auto _msg = decoded_extdata{};
    if(has_extdata())
        common::read_json_string(
            extdata, [](auto& ar, auto& msg) { cereal::load(ar, msg); }, _msg);
    return _msg;
}

sample::decoded_extdata
sample::get_extdata() const
{
    auto _msg = decoded_extdata{};
    if(has_extdata())
        common::read_json_string(
            extdata, [](auto& ar, auto& msg) { cereal::load(ar, msg); }, _msg);
    return _msg;
}
}  // namespace types
}  // namespace rocpd
