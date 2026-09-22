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
#include <cstring>
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

TEST(common, sha256_digest_bytes)
{
    // digest() must be the big-endian serialisation of the same state that
    // hexdigest() prints, so the two can never disagree.
    auto _val = rocprofiler::common::sha256{};
    _val.update("rocprofiler-sdk|rocprofiler-sdk-roctx|rocprofiler-sdk-rocpd");

    auto _hex_digest = _val.hexdigest();

    auto _bytes = _val.digest();

    auto _from_bytes = std::string{};
    for(auto _b : _bytes)
        _from_bytes += fmt::format("{:02x}", _b);

    EXPECT_EQ(_from_bytes, _hex_digest);

    // Calling it twice must not change the answer.
    EXPECT_EQ(_bytes, _val.digest());
}

TEST(common, sha256_fips_vectors)
{
    auto hex_of = [](const std::string& _msg) {
        auto _h = rocprofiler::common::sha256{};
        if(!_msg.empty()) _h.update(_msg);
        return _h.hexdigest();
    };

    // FIPS 180-4 published examples, including the empty message and the
    // 56-byte case that forces a second padding block.
    EXPECT_EQ(hex_of(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    EXPECT_EQ(hex_of("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    EXPECT_EQ(hex_of("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");

    // Long message fed in chunks: exercises update() across block boundaries.
    auto _h     = rocprofiler::common::sha256{};
    auto _chunk = std::string(1000, 'a');
    for(int i = 0; i < 1000; ++i)
        _h.update(_chunk);
    EXPECT_EQ(_h.hexdigest(), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}
