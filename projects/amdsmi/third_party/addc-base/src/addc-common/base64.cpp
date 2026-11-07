// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "base64.hpp"

#include <limits>
#include <stdexcept>

namespace addc::detail
{

namespace
{

constexpr char kAlphabet[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

[[nodiscard]] constexpr int decodeValue(unsigned char value) noexcept
{
    if (value >= 'A' && value <= 'Z')
    {
        return value - 'A';
    }
    if (value >= 'a' && value <= 'z')
    {
        return value - 'a' + 26;
    }
    if (value >= '0' && value <= '9')
    {
        return value - '0' + 52;
    }
    if (value == '+')
    {
        return 62;
    }
    if (value == '/')
    {
        return 63;
    }
    return -1;
}

} // anonymous namespace

std::string base64_encode(std::span<const uint8_t> data)
{
    if (data.size() > (std::numeric_limits<std::size_t>::max() - 2U) / 4U * 3U)
    {
        throw std::length_error("base64 input is too large");
    }

    const std::size_t output_size = ((data.size() + 2U) / 3U) * 4U;
    std::string output(output_size, '\0');

    std::size_t input_pos = 0U;
    std::size_t output_pos = 0U;
    while (input_pos + 3U <= data.size())
    {
        const uint32_t group =
            (static_cast<uint32_t>(data[input_pos]) << 16U) |
            (static_cast<uint32_t>(data[input_pos + 1U]) << 8U) |
            static_cast<uint32_t>(data[input_pos + 2U]);
        output[output_pos++] = kAlphabet[(group >> 18U) & 0x3fU];
        output[output_pos++] = kAlphabet[(group >> 12U) & 0x3fU];
        output[output_pos++] = kAlphabet[(group >> 6U) & 0x3fU];
        output[output_pos++] = kAlphabet[group & 0x3fU];
        input_pos += 3U;
    }

    const std::size_t remaining = data.size() - input_pos;
    if (remaining == 1U)
    {
        const uint32_t group = static_cast<uint32_t>(data[input_pos]) << 16U;
        output[output_pos++] = kAlphabet[(group >> 18U) & 0x3fU];
        output[output_pos++] = kAlphabet[(group >> 12U) & 0x3fU];
        output[output_pos++] = '=';
        output[output_pos++] = '=';
    }
    else if (remaining == 2U)
    {
        const uint32_t group =
            (static_cast<uint32_t>(data[input_pos]) << 16U) |
            (static_cast<uint32_t>(data[input_pos + 1U]) << 8U);
        output[output_pos++] = kAlphabet[(group >> 18U) & 0x3fU];
        output[output_pos++] = kAlphabet[(group >> 12U) & 0x3fU];
        output[output_pos++] = kAlphabet[(group >> 6U) & 0x3fU];
        output[output_pos++] = '=';
    }

    return output;
}

std::optional<std::vector<uint8_t>> base64_decode(std::string_view text)
{
    if (text.empty())
    {
        return std::vector<uint8_t>{};
    }
    if ((text.size() % 4U) != 0U)
    {
        return std::nullopt;
    }

    const std::size_t quartet_count = text.size() / 4U;
    std::size_t padding = 0U;
    if (text.back() == '=')
    {
        padding = 1U;
        if (text[text.size() - 2U] == '=')
        {
            padding = 2U;
        }
    }

    std::vector<uint8_t> output;
    output.reserve(quartet_count * 3U - padding);

    for (std::size_t quartet = 0U; quartet < quartet_count; ++quartet)
    {
        const std::size_t pos = quartet * 4U;
        const bool final_quartet = quartet + 1U == quartet_count;
        const char c0 = text[pos];
        const char c1 = text[pos + 1U];
        const char c2 = text[pos + 2U];
        const char c3 = text[pos + 3U];

        const int v0 = decodeValue(static_cast<unsigned char>(c0));
        const int v1 = decodeValue(static_cast<unsigned char>(c1));
        if (v0 < 0 || v1 < 0)
        {
            return std::nullopt;
        }

        if (c2 == '=')
        {
            if (!final_quartet || c3 != '=' || (v1 & 0x0f) != 0)
            {
                return std::nullopt;
            }
            output.push_back(static_cast<uint8_t>((v0 << 2) | (v1 >> 4)));
            continue;
        }

        const int v2 = decodeValue(static_cast<unsigned char>(c2));
        if (v2 < 0)
        {
            return std::nullopt;
        }

        output.push_back(static_cast<uint8_t>((v0 << 2) | (v1 >> 4)));
        if (c3 == '=')
        {
            if (!final_quartet || (v2 & 0x03) != 0)
            {
                return std::nullopt;
            }
            output.push_back(static_cast<uint8_t>((v1 << 4) | (v2 >> 2)));
            continue;
        }

        const int v3 = decodeValue(static_cast<unsigned char>(c3));
        if (v3 < 0)
        {
            return std::nullopt;
        }
        output.push_back(static_cast<uint8_t>((v1 << 4) | (v2 >> 2)));
        output.push_back(static_cast<uint8_t>((v2 << 6) | v3));
    }

    return output;
}

} // namespace addc::detail
