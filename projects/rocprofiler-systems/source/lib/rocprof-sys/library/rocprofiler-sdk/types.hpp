// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/units/data_size.hpp"

#include <compare>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/base.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

namespace rocprofsys::domains
{
using namespace units::literals;

enum class collection_mode : std::uint8_t
{
    callback,
    buffered
};

using domain_id_t    = std::size_t;
using operation_id_t = std::size_t;

struct operation_info
{
    operation_id_t id;
    std::string    name;
};

struct buffer_properties
{
    units::kibibytes buffer_size;
    units::kibibytes buffer_watermark;
};

struct domain_key
{
    collection_mode mode;
    domain_id_t     value;

    auto operator<=>(const domain_key&) const = default;
};

struct domain_info
{
    domain_key                      key;
    std::string_view                name;
    std::vector<operation_info>     operations;
    std::optional<std::string_view> group;
};

struct domain_group
{
    std::string_view name;
};

struct domain_descriptor
{
    std::string_view            name;
    std::size_t                 id;
    collection_mode             mode;
    std::optional<domain_group> group;
};

/// Matches the buffer sizing used by the rocprof-sys rocprofiler-sdk backend.
inline constexpr buffer_properties k_default_buffer_properties{ .buffer_size = 64_kib,
                                                                .buffer_watermark =
                                                                    63_kib };

template <typename SdkBackend>
using buffer_tracing_cb_t = void (*)(typename SdkBackend::context_id_t      context,
                                     typename SdkBackend::buffer_id_t       buffer_id,
                                     typename SdkBackend::record_header_t** headers,
                                     std::size_t num_headers, void* data,
                                     std::uint64_t drop_count);

template <typename SdkBackend>
using callback_tracing_cb_t =
    void (*)(typename SdkBackend::callback_tracing_record_t record,
             typename SdkBackend::user_data_t* user_data, void* callback_data);

using configure_cb_t = void (*)();

template <typename SdkBackend>
struct buffered_domain_definition
{
    domain_descriptor               meta;
    buffer_tracing_cb_t<SdkBackend> on_records;
    buffer_properties               buffer = k_default_buffer_properties;
    configure_cb_t                  on_configure;
};

template <typename SdkBackend>
struct callback_domain_definition
{
    domain_descriptor                 meta;
    callback_tracing_cb_t<SdkBackend> on_record;
    configure_cb_t                    on_configure;
};

struct domain_configuration
{
    domain_key                  key;
    std::vector<operation_id_t> operations;
};

template <typename SdkBackend, typename RecordT, void (*Callback)(RecordT*, void*)>
struct buffered_callback_dispatcher
{
    // NOLINTNEXTLINE (readability-function-size)
    static void callback(SdkBackend::context_id_t /*context*/,
                         SdkBackend::buffer_id_t /*buffer_id*/,
                         SdkBackend::record_header_t** headers, std::size_t num_headers,
                         void* data, std::uint64_t /*drop_count*/)
    {
        if(headers == nullptr)
        {
            return;
        }

        for(std::size_t i = 0; i < num_headers; i++)
        {
            if(headers[i] == nullptr)
            {
                continue;
            }

            Callback(static_cast<RecordT*>(headers[i]->payload), data);
        }
    }
};

}  // namespace rocprofsys::domains

template <>
struct fmt::formatter<rocprofsys::domains::collection_mode>
: fmt::formatter<std::string_view>
{
    template <typename FormatContext>
    auto format(rocprofsys::domains::collection_mode mode, FormatContext& ctx) const
    {
        std::string_view str = "unknown";
        switch(mode)
        {
            case rocprofsys::domains::collection_mode::callback: str = "callback"; break;
            case rocprofsys::domains::collection_mode::buffered: str = "buffered"; break;
        }
        return fmt::formatter<std::string_view>::format(str, ctx);
    }
};

template <>
struct fmt::formatter<rocprofsys::domains::operation_info>
: fmt::formatter<std::string_view>
{
    template <typename FormatContext>
    auto format(const rocprofsys::domains::operation_info& op_info,
                FormatContext&                             ctx) const
    {
        return fmt::format_to(ctx.out(), "operation_info [id: {} name: {}]", op_info.id,
                              op_info.name);
    }
};

template <>
struct fmt::formatter<rocprofsys::domains::buffer_properties>
: fmt::formatter<std::string_view>
{
    template <typename FormatContext>
    auto format(const rocprofsys::domains::buffer_properties& buffer_props,
                FormatContext&                                ctx) const
    {
        return fmt::format_to(ctx.out(), "buffer [size: {} watermark: {}]",
                              buffer_props.buffer_size, buffer_props.buffer_watermark);
    }
};

template <>
struct fmt::formatter<rocprofsys::domains::domain_key> : fmt::formatter<std::string_view>
{
    template <typename FormatContext>
    auto format(const rocprofsys::domains::domain_key& key, FormatContext& ctx) const
    {
        return fmt::format_to(ctx.out(), "domain_key [mode: {} value: {}]", key.mode,
                              key.value);
    }
};

template <>
struct fmt::formatter<rocprofsys::domains::domain_group>
: fmt::formatter<std::string_view>
{
    template <typename FormatContext>
    auto format(const rocprofsys::domains::domain_group& group, FormatContext& ctx) const
    {
        return fmt::format_to(ctx.out(), "domain_group [name: {}]", group.name);
    }
};

template <>
struct fmt::formatter<rocprofsys::domains::domain_descriptor>
: fmt::formatter<std::string_view>
{
    template <typename FormatContext>
    auto format(const rocprofsys::domains::domain_descriptor& descriptor,
                FormatContext&                                ctx) const
    {
        fmt::format_to(ctx.out(), "domain_descriptor [name: {} id: {} mode: {} group: ",
                       descriptor.name, descriptor.id, descriptor.mode);
        if(descriptor.group)
        {
            fmt::format_to(ctx.out(), "{}", *descriptor.group);
        }
        else
        {
            fmt::format_to(ctx.out(), "none");
        }
        return fmt::format_to(ctx.out(), "]");
    }
};

template <>
struct fmt::formatter<rocprofsys::domains::domain_info> : fmt::formatter<std::string_view>
{
    template <typename FormatContext>
    auto format(const rocprofsys::domains::domain_info& info, FormatContext& ctx) const
    {
        fmt::format_to(ctx.out(),
                       "domain_info [key: {} name: {} operations: [{}] group: ", info.key,
                       info.name, fmt::join(info.operations, ", "));
        if(info.group)
        {
            fmt::format_to(ctx.out(), "{}", *info.group);
        }
        else
        {
            fmt::format_to(ctx.out(), "none");
        }
        return fmt::format_to(ctx.out(), "]");
    }
};

template <>
struct fmt::formatter<rocprofsys::domains::domain_configuration>
: fmt::formatter<std::string_view>
{
    template <typename FormatContext>
    auto format(const rocprofsys::domains::domain_configuration& config,
                FormatContext&                                   ctx) const
    {
        return fmt::format_to(ctx.out(),
                              "domain_configuration [key: {} operations: [{}]]",
                              config.key, fmt::join(config.operations, ", "));
    }
};
