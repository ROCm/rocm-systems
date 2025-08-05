// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#include "lib/common/sha256.hpp"
#include "lib/common/defines.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/mpl.hpp"

#include <unistd.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace rocprofiler
{
namespace common
{
sha256::sha256() { reset(); }

sha256::sha256(const std::string& data)
{
    reset();
    update(data);
    finalize();
}

void
sha256::update(const uint8_t* data, size_t len)
{
    ROCP_CI_LOG_IF(INFO, m_finalized) << "attempt to update sha256 after finalized";
    if(m_finalized) return;
    for(size_t i = 0; i < len; ++i)
    {
        m_data[m_datalen++] = data[i];
        if(m_datalen == 64)
        {
            transform();
            m_bitlen += 512;
            m_datalen = 0;
        }
    }
}

void
sha256::update(const std::string& data)
{
    ROCP_CI_LOG_IF(INFO, m_finalized) << "attempt to update sha256 after finalized";
    if(m_finalized) return;
    update(reinterpret_cast<const uint8_t*>(data.data()), data.size());
}

void
sha256::finalize()
{
    if(m_finalized) return;

    uint32_t idx = m_datalen;
    if(m_datalen < 56)
    {
        m_data[idx++] = 0x80;
        while(idx < 56)
            m_data[idx++] = 0x00;
    }
    else
    {
        m_data[idx++] = 0x80;
        while(idx < 64)
            m_data[idx++] = 0x00;
        transform();
        std::memset(m_data.data(), 0, 56);
    }

    m_bitlen += m_datalen * 8;
    for(int j = 0; j < 8; ++j)
        m_data[63 - j] = static_cast<uint8_t>((m_bitlen >> (8 * j)) & 0xFF);

    transform();

    m_finalized = true;
}

std::string
sha256::hexdigest()
{
    finalize();

    auto oss = std::ostringstream{};
    for(int j = 0; j < 8; ++j)
        oss << std::hex << std::setfill('0') << std::setw(8) << m_state[j];
    return oss.str();
}

std::array<uint32_t, 8>
sha256::rawdigest()
{
    finalize();

    return m_state;
}

uint32_t
sha256::rotr(uint32_t x, uint32_t n)
{
    return (x >> n) | (x << (32 - n));
}

uint32_t
sha256::ch(uint32_t x, uint32_t y, uint32_t z)
{
    return (x & y) ^ (~x & z);
}

uint32_t
sha256::maj(uint32_t x, uint32_t y, uint32_t z)
{
    return (x & y) ^ (x & z) ^ (y & z);
}

uint32_t
sha256::sig0(uint32_t x)
{
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

uint32_t
sha256::sig1(uint32_t x)
{
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

uint32_t
sha256::theta0(uint32_t x)
{
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

uint32_t
sha256::theta1(uint32_t x)
{
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

void
sha256::transform()
{
    uint32_t m[64];
    for(int i = 0; i < 16; ++i)
    {
        m[i] = (m_data[i * 4] << 24) | (m_data[i * 4 + 1] << 16) | (m_data[i * 4 + 2] << 8) |
               (m_data[i * 4 + 3]);
    }
    for(int i = 16; i < 64; ++i)
    {
        m[i] = theta1(m[i - 2]) + m[i - 7] + theta0(m[i - 15]) + m[i - 16];
    }

    uint32_t a = m_state[0];
    uint32_t b = m_state[1];
    uint32_t c = m_state[2];
    uint32_t d = m_state[3];
    uint32_t e = m_state[4];
    uint32_t f = m_state[5];
    uint32_t g = m_state[6];
    uint32_t h = m_state[7];

    for(int i = 0; i < 64; ++i)
    {
        uint32_t t1 = h + sig1(e) + ch(e, f, g) + m_k[i] + m[i];
        uint32_t t2 = sig0(a) + maj(a, b, c);
        h           = g;
        g           = f;
        f           = e;
        e           = d + t1;
        d           = c;
        c           = b;
        b           = a;
        a           = t1 + t2;
    }

    m_state[0] += a;
    m_state[1] += b;
    m_state[2] += c;
    m_state[3] += d;
    m_state[4] += e;
    m_state[5] += f;
    m_state[6] += g;
    m_state[7] += h;
}

void
sha256::reset()
{
    m_state   = {0x6a09e667,
               0xbb67ae85,
               0x3c6ef372,
               0xa54ff53a,
               0x510e527f,
               0x9b05688c,
               0x1f83d9ab,
               0x5be0cd19};
    m_datalen = 0;
    m_bitlen  = 0;
}
}  // namespace common
}  // namespace rocprofiler
