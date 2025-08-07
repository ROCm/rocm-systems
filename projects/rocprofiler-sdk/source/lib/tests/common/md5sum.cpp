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

#include "lib/common/md5sum.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>

TEST(common, md5sum)
{
    auto _val = rocprofiler::common::md5sum{};

    _val.update(234234);
    _val.finalize();

    auto _hex_digest = _val.hexdigest();
    EXPECT_EQ(_hex_digest, "5cdf897625070d805f3fb2469b08eebb");

    auto read_hex = [](std::string&& _inp) {
        int _ret = 0;
        std::stringstream{_inp} >> std::hex >> _ret;
        return _ret;
    };

    auto _raw_digest = _val.rawdigest();
    for(size_t i = 0; i < _raw_digest.size(); ++i)
    {
        auto _sub     = _hex_digest.substr(i * 2, 2);
        auto _extract = read_hex(fmt::format("0x{}", _sub));
        int  _raw     = _raw_digest.at(i);
        EXPECT_EQ(_raw, _extract) << fmt::format(
            "i={}, raw[i]={}, hex={}, hexdigest={}", i, _raw, _sub, _hex_digest);
    }
}
