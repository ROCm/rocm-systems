// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::domains
{

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
    std::size_t buffer_size;
    std::size_t buffer_watermark;
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
inline constexpr buffer_properties k_default_buffer_properties{
    .buffer_size      = static_cast<std::size_t>(16 * 4096),
    .buffer_watermark = static_cast<std::size_t>(15 * 4096)
};

template <typename SdkBackend>
struct buffered_domain_definition
{
    domain_descriptor               meta;
    buffer_tracing_cb_t<SdkBackend> on_records;
    buffer_properties               buffer = k_default_buffer_properties;
};

template <typename SdkBackend>
struct callback_domain_definition
{
    domain_descriptor                 meta;
    callback_tracing_cb_t<SdkBackend> on_record;
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
