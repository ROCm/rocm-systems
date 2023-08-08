
#pragma once

#include <rocprofiler/config.h>
#include <rocprofiler/rocprofiler.h>

#include <array>
#include <atomic>
#include <bitset>
#include <cstddef>
#include <cstdint>

namespace rocprofiler
{
namespace internal
{
// number of bits to reserve all op codes
constexpr size_t domain_ops_offset    = ROCPROFILER_DOMAIN_OPS_MAX;
constexpr size_t reserved_domain_size = ROCPROFILER_DOMAIN_OPS_RESERVED * 8;
constexpr size_t max_configs_count    = 8;

struct correlation_config
{
    uint64_t                        id                   = 0;
    uint64_t                        external_id          = 0;
    ::rocprofiler_external_cid_cb_t external_id_callback = nullptr;

    static uint64_t get_unique_record_id();
};

struct domain_config
{
    ::rocprofiler_sync_callback_t     user_sync_callback = nullptr;
    int64_t                           domains            = 0;
    std::bitset<reserved_domain_size> opcodes            = {};

    /// check if domain is enabled
    bool operator()(::rocprofiler_tracer_activity_domain_t) const;

    /// check if op in a domain is enabled
    bool operator()(::rocprofiler_tracer_activity_domain_t, uint32_t) const;
};

struct buffer_config
{
    ::rocprofiler_buffer_callback_t callback = nullptr;
    uint64_t                        buffer_size;
    // Memory::GenericBuffer*          buffer     = nullptr;
    uint64_t buffer_idx = 0;
};

using filter_config = ::rocprofiler_filter_config;

struct config
{
    // size is used to ensure that we never read past the end of the version
    size_t              size           = 0;        // = sizeof(rocprofiler_config)
    uint32_t            compat_version = 0;        // set by user
    uint32_t            api_version    = 0;        // set by rocprofiler
    uint64_t            session_idx    = 0;        // session id index
    void*               user_data      = nullptr;  // user data passed to callbacks
    correlation_config* correlation_id = nullptr;  // &my_cid_config (optional)
    buffer_config*      buffer         = nullptr;  // = &my_buffer_config (required)
    domain_config*      domain         = nullptr;  // = &my_domain_config (required)
    filter_config*      filter         = nullptr;  // = &my_filter_config (optional)
};

std::array<rocprofiler::internal::config*, max_configs_count>&
get_registered_configs();

std::array<std::atomic<rocprofiler::internal::config*>, max_configs_count>&
get_active_configs();
}  // namespace internal
}  // namespace rocprofiler
