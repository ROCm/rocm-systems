#pragma once

/**
 * @file format_hex.hpp
 * @brief Lowercase "0x"-prefixed, unpadded hex rendering of an address.
 */

#include <array>
#include <charconv>
#include <cstdint>
#include <string>

namespace cpu_snap::core
{

/**
 * @brief Render @p value as "0x<lowercase hex>" with no leading zeros
 *        (identical to printf "0x%lx"); 0 renders as "0x0".
 */
[[nodiscard]] inline std::string format_hex(std::uint64_t value)
{
    constexpr int k_hex_base = 16;  // NOLINT(readability-identifier-naming)

    std::array<char, sizeof("0x") - 1 + (sizeof(value) * 2)> buf{ '0', 'x' };
    const auto [end, ec] =
        std::to_chars(buf.data() + 2, buf.data() + buf.size(), value, k_hex_base);
    return { buf.data(), end };
}

}  // namespace cpu_snap::core
