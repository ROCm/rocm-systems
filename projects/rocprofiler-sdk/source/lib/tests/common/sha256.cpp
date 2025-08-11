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

#include "lib/common/sha256.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>

TEST(common, sha256)
{
    auto _val = rocprofiler::common::sha256{};

    _val.update("rocprofiler-sdk|rocprofiler-sdk-roctx|rocprofiler-sdk-rocpd");

    auto _hex_digest = _val.hexdigest();
    EXPECT_EQ(_hex_digest, "e58b701acc3c524f881e49fff2833879eb17975b6a072d9ecc27de5a5344aefc");

    auto read_hex = [](std::string&& _inp) {
        uint32_t _ret = 0;
        std::stringstream{_inp} >> std::hex >> _ret;
        return _ret;
    };

    auto _raw_digest = _val.rawdigest();
    for(size_t i = 0; i < _raw_digest.size(); ++i)
    {
        auto _sub     = _hex_digest.substr(i * 8, 8);
        auto _extract = read_hex(fmt::format("0x{}", _sub));
        int  _raw     = _raw_digest.at(i);
        EXPECT_EQ(_raw, _extract) << fmt::format(
            "i={}, raw[i]={}, hex={}, hexdigest={}", i, _raw, _sub, _hex_digest);
    }
}
