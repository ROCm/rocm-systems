// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::avail
{
/// Identifies which backend produced a diagnostic so a partial result can name
/// the degraded source instead of failing the whole query.
enum class source_id : std::uint8_t
{
    rocprofiler_sdk,
    amd_smi,
};

[[nodiscard]] std::string_view
to_string(source_id source) noexcept;

struct diagnostic
{
    source_id   source = source_id::rocprofiler_sdk;
    std::string message = {};
};

/// A GPU agent reported by rocprofiler-sdk. This is the authoritative device
/// list; AMD SMI only adds detail to it.
struct sdk_agent_entry
{
    std::uint64_t handle          = 0;
    std::size_t   index           = 0;
    std::string   name            = {};
    std::string   product_name    = {};
    std::string   vendor_name     = {};
    std::string   pci_bdf         = {};
    bool          runtime_visible = true;
};

/// A GPU as seen by AMD SMI, keyed by PCI BDF for correlation with an agent.
struct smi_device_entry
{
    std::string pci_bdf      = {};
    std::string market_name  = {};
    std::string vendor_name  = {};
};

/// One GPU, from rocprofiler-sdk, with AMD SMI fields left empty when SMI has
/// no device at the same PCI BDF.
struct device_record
{
    std::uint64_t agent_handle    = 0;
    std::size_t   index           = 0;
    std::string   name            = {};
    std::string   product_name    = {};
    std::string   vendor_name     = {};
    std::string   pci_bdf         = {};
    bool          runtime_visible = true;

    std::optional<std::string> smi_market_name = {};
    std::optional<std::string> smi_vendor_name = {};

    [[nodiscard]] bool has_smi_data() const noexcept
    {
        return smi_market_name.has_value() || smi_vendor_name.has_value();
    }
};

struct counter_dimension
{
    std::string name     = {};
    std::size_t position = 0;
};

struct counter_record
{
    std::uint64_t                  id          = 0;
    std::string                    name        = {};
    std::string                    description = {};
    std::string                    block       = {};
    std::string                    expression  = {};
    bool                           is_constant = false;
    bool                           is_derived  = false;
    std::vector<counter_dimension> dimensions  = {};
};

/// Counters supported by one GPU. Counter names repeat across devices, so the
/// owning device is part of the record rather than a lookup the caller makes.
struct device_counters
{
    std::uint64_t               agent_handle = 0;
    std::size_t                 index        = 0;
    std::string                 device_name  = {};
    std::vector<counter_record> counters     = {};
};

struct device_query_result
{
    std::vector<device_record> devices     = {};
    std::vector<diagnostic>    diagnostics = {};
};

struct counter_query_result
{
    std::vector<device_counters> devices     = {};
    std::vector<diagnostic>      diagnostics = {};
};

/// An SDK tracing domain. Names match ROCPROFSYS_ROCM_DOMAINS / --trace tokens.
/// Alias rows list the SDK kinds this project maps the token onto.
struct trace_domain_record
{
    std::string              name          = {};
    bool                     callback      = false;
    bool                     buffered      = false;
    bool                     alias         = false;
    bool                     is_default    = false;
    std::vector<std::string> alias_members = {};
    std::vector<std::string> operations    = {};
};

struct trace_query_result
{
    std::vector<trace_domain_record> traces      = {};
    std::vector<diagnostic>          diagnostics = {};
};
}  // namespace rocprofsys::avail
