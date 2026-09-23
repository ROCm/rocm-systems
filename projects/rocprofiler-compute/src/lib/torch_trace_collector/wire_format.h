// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string_view>

namespace torch_trace_collector::detail
{

inline constexpr const char* kEncodedPercent = "%25";
inline constexpr const char* kEncodedSlash   = "%2F";
inline constexpr const char* kUnavailable    = "n/a";

class BoundedTextOutput
{
public:
    explicit BoundedTextOutput(std::span<char> output) noexcept
        : output_{output}
    {
    }

    BoundedTextOutput(char* output, std::size_t capacity) noexcept
        : output_{output == nullptr ? std::span<char>{} : std::span<char>{output, capacity}}
    {
    }

    void append(std::string_view value) noexcept
    {
        if (!output_.empty() && size_ < output_.size() - 1)
        {
            const std::size_t copied = std::min(value.size(), output_.size() - 1 - size_);
            if (copied > 0)
            {
                std::copy_n(value.data(), copied, output_.data() + size_);
            }
        }
        size_ += value.size();
    }

    void append(char value) noexcept { append(std::string_view{&value, 1}); }

    [[nodiscard]] std::size_t finish() noexcept
    {
        if (!output_.empty())
        {
            output_[std::min(size_, output_.size() - 1)] = '\0';
        }
        return size_ + 1;
    }

private:
    std::span<char> output_;
    std::size_t     size_ = 0;
};

inline void append_encoded_marker_name(BoundedTextOutput& output, std::string_view name) noexcept
{
    for (const char character : name)
    {
        if (character == '%')
        {
            output.append(kEncodedPercent);
        }
        else if (character == '/')
        {
            output.append(kEncodedSlash);
        }
        else
        {
            output.append(character);
        }
    }
}

template<typename IntegerT>
void append_decimal(BoundedTextOutput& output, IntegerT value) noexcept
{
    std::array<char, std::numeric_limits<IntegerT>::digits10 + 3> buffer;
    const auto converted = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    if (converted.ec == std::errc{})
    {
        output.append(std::string_view{buffer.data(),
                                       static_cast<std::size_t>(converted.ptr - buffer.data())});
    }
}

struct RangeNameFields
{
    std::string_view             name;
    std::string_view             context;
    std::int64_t                 sequence_number;
    std::uint64_t                thread_id;
    std::uint64_t                forward_thread_id;
    std::optional<std::uint64_t> launcher_thread_id;
    std::string_view             scope;
    std::string_view             arguments;
    std::string_view             backend;
};

inline std::size_t format_range_name(std::span<char> destination, const RangeNameFields& fields) noexcept
{
    BoundedTextOutput output{destination};
    append_encoded_marker_name(output, fields.name);
    output.append(':');
    output.append(fields.context);
    output.append("|seqNr=");
    if (fields.sequence_number < 0)
    {
        output.append(kUnavailable);
    }
    else
    {
        append_decimal(output, fields.sequence_number);
    }
    output.append("|tid=");
    append_decimal(output, fields.thread_id);
    output.append("|ftid=");
    append_decimal(output, fields.forward_thread_id);
    output.append("|ltid=");
    if (fields.launcher_thread_id.has_value())
    {
        append_decimal(output, *fields.launcher_thread_id);
    }
    else
    {
        output.append(kUnavailable);
    }
    output.append("|scope=");
    output.append(fields.scope);
    output.append("|args=");
    output.append(fields.arguments);
    if (!fields.backend.empty())
    {
        output.append('|');
        output.append(fields.backend);
    }
    return output.finish();
}

}  // namespace torch_trace_collector::detail
