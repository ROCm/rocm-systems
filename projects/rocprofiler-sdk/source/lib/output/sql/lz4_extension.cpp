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

#include "lib/output/sql/lz4_extension.hpp"

#include "lib/common/logging.hpp"

#include <fmt/format.h>

#include <lz4frame.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>

namespace rocprofiler
{
namespace tool
{
namespace sql
{
namespace
{
constexpr auto magic       = std::array<char, 4>{'L', 'Z', '4', 'F'};
constexpr auto header_size = size_t{8};

void
throw_lz4_error(std::string_view operation, size_t code)
{
    if(LZ4F_isError(code) != 0)
        throw std::runtime_error{fmt::format("{} failed: {}", operation, LZ4F_getErrorName(code))};
}

void
write_u32_le(std::vector<std::byte>& data, uint32_t value)
{
    data.emplace_back(static_cast<std::byte>(value & 0xffU));
    data.emplace_back(static_cast<std::byte>((value >> 8U) & 0xffU));
    data.emplace_back(static_cast<std::byte>((value >> 16U) & 0xffU));
    data.emplace_back(static_cast<std::byte>((value >> 24U) & 0xffU));
}

uint32_t
read_u32_le(std::string_view data)
{
    const auto* ptr = reinterpret_cast<const unsigned char*>(data.data());
    return (static_cast<uint32_t>(ptr[0]) | (static_cast<uint32_t>(ptr[1]) << 8U) |
            (static_cast<uint32_t>(ptr[2]) << 16U) | (static_cast<uint32_t>(ptr[3]) << 24U));
}

std::string_view
sqlite_value_bytes(sqlite3_value* value)
{
    if(value == nullptr || sqlite3_value_type(value) == SQLITE_NULL) return {};

    const auto size = sqlite3_value_bytes(value);
    if(size <= 0) return {};

    const auto* data = sqlite3_value_blob(value);
    if(data == nullptr) return {};

    return {static_cast<const char*>(data), static_cast<size_t>(size)};
}

int
sqlite_value_level(sqlite3_value* value)
{
    if(value == nullptr || sqlite3_value_type(value) == SQLITE_NULL) return 1;
    return std::clamp(sqlite3_value_int(value), 1, 12);
}

void
sqlite_lz4_compress(sqlite3_context* ctx, int argc, sqlite3_value** argv)
{
    try
    {
        const auto input = sqlite_value_bytes(argv[0]);
        const auto level = (argc > 1) ? sqlite_value_level(argv[1]) : 1;
        auto       data  = lz4_compress_buffer(input, level);
        sqlite3_result_blob(ctx, data.data(), static_cast<int>(data.size()), SQLITE_TRANSIENT);
    } catch(const std::exception& e)
    {
        sqlite3_result_error(ctx, e.what(), -1);
    }
}

void
sqlite_lz4_decompress(sqlite3_context* ctx, int /*argc*/, sqlite3_value** argv)
{
    try
    {
        const auto input = sqlite_value_bytes(argv[0]);
        auto       data  = lz4_decompress_buffer(input);
        sqlite3_result_blob(ctx, data.data(), static_cast<int>(data.size()), SQLITE_TRANSIENT);
    } catch(const std::exception& e)
    {
        sqlite3_result_error(ctx, e.what(), -1);
    }
}
}  // namespace

std::vector<std::byte>
lz4_compress_buffer(std::string_view input, int level)
{
    if(input.empty()) return {};

    if(input.size() > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error{"lz4 payload exceeds 4 GiB wire-format limit"};

    auto preferences                          = LZ4F_preferences_t{};
    preferences.frameInfo.blockMode           = LZ4F_blockIndependent;
    preferences.frameInfo.contentChecksumFlag = LZ4F_contentChecksumEnabled;
    preferences.compressionLevel              = level;

    const auto bound = LZ4F_compressFrameBound(input.size(), &preferences);
    throw_lz4_error("LZ4F_compressFrameBound", bound);

    auto output = std::vector<std::byte>{};
    output.reserve(header_size + bound);
    output.insert(output.end(),
                  reinterpret_cast<const std::byte*>(magic.data()),
                  reinterpret_cast<const std::byte*>(magic.data() + magic.size()));
    write_u32_le(output, static_cast<uint32_t>(input.size()));

    const auto compressed_offset = output.size();
    output.resize(output.size() + bound);

    const auto compressed_size = LZ4F_compressFrame(
        output.data() + compressed_offset, bound, input.data(), input.size(), &preferences);
    throw_lz4_error("LZ4F_compressFrame", compressed_size);
    output.resize(compressed_offset + compressed_size);
    return output;
}

std::string
lz4_decompress_buffer(std::string_view input)
{
    if(input.empty()) return {};

    if(input.size() < header_size || std::memcmp(input.data(), magic.data(), magic.size()) != 0)
        throw std::runtime_error{"invalid lz4 blob magic"};

    const auto uncompressed_size = read_u32_le(input.substr(magic.size(), sizeof(uint32_t)));
    auto       output            = std::string(uncompressed_size, '\0');
    auto       payload           = input.substr(header_size);

    LZ4F_dctx* ctx = nullptr;
    throw_lz4_error("LZ4F_createDecompressionContext",
                    LZ4F_createDecompressionContext(&ctx, LZ4F_VERSION));

    auto cleanup = std::unique_ptr<LZ4F_dctx, decltype(&LZ4F_freeDecompressionContext)>{
        ctx, LZ4F_freeDecompressionContext};

    size_t     dst_size = output.size();
    size_t     src_size = payload.size();
    const auto rc =
        LZ4F_decompress(ctx, output.data(), &dst_size, payload.data(), &src_size, nullptr);
    throw_lz4_error("LZ4F_decompress", rc);

    if(rc != 0 || src_size != payload.size() || dst_size != output.size())
        throw std::runtime_error{"truncated lz4 blob"};

    return output;
}

void
register_lz4_functions(sqlite3* conn)
{
    ROCP_FATAL_IF(conn == nullptr) << "Cannot register LZ4 SQLite functions on null connection";

    constexpr auto flags = SQLITE_UTF8 | SQLITE_DETERMINISTIC;

    auto rc = sqlite3_create_function_v2(
        conn, "lz4_compress", 1, flags, nullptr, sqlite_lz4_compress, nullptr, nullptr, nullptr);
    ROCP_FATAL_IF(rc != SQLITE_OK)
        << "sqlite3_create_function_v2(lz4_compress) failed: " << sqlite3_errmsg(conn);

    rc = sqlite3_create_function_v2(
        conn, "lz4_compress", 2, flags, nullptr, sqlite_lz4_compress, nullptr, nullptr, nullptr);
    ROCP_FATAL_IF(rc != SQLITE_OK)
        << "sqlite3_create_function_v2(lz4_compress) failed: " << sqlite3_errmsg(conn);

    rc = sqlite3_create_function_v2(conn,
                                    "lz4_decompress",
                                    1,
                                    flags,
                                    nullptr,
                                    sqlite_lz4_decompress,
                                    nullptr,
                                    nullptr,
                                    nullptr);
    ROCP_FATAL_IF(rc != SQLITE_OK)
        << "sqlite3_create_function_v2(lz4_decompress) failed: " << sqlite3_errmsg(conn);
}
}  // namespace sql
}  // namespace tool
}  // namespace rocprofiler
