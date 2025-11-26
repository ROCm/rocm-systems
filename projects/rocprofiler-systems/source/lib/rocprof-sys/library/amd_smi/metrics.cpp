// Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// with the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimers in the
// documentation and/or other materials provided with the distribution.
//
// * Neither the names of Advanced Micro Devices, Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this Software without specific prior written permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.

#include "library/amd_smi/metrics.hpp"
#include "core/gpu.hpp"
#include "core/trace_cache/sample_type.hpp"

#include <bitset>
#include <climits>
#include <cstdint>

namespace rocprofsys
{
namespace amd_smi
{
namespace metrics
{
#if ROCPROFSYS_USE_ROCM > 0

ProcessedMetrics
process_vcn_metrics(const amdsmi_gpu_metrics_t& raw_metrics, bool is_device_level_only)
{
    ProcessedMetrics result{};

    if(is_device_level_only)
    {
        copy_valid_metrics(result.metrics.vcn_activity, raw_metrics.vcn_activity,
                           UINT16_MAX);
        result.has_data = !result.metrics.vcn_activity.empty();
    }
    else
    {
        // XCP-level VCN metrics
        for(const auto& xcp : raw_metrics.xcp_stats)
        {
            std::vector<uint16_t> xcp_vcn_data;
            copy_valid_metrics(xcp_vcn_data, xcp.vcn_busy, UINT16_MAX);
            if(!xcp_vcn_data.empty())
            {
                result.metrics.vcn_busy.push_back(std::move(xcp_vcn_data));
            }
        }
        result.has_data = !result.metrics.vcn_busy.empty();
    }

    return result;
}

ProcessedMetrics
process_jpeg_metrics(const amdsmi_gpu_metrics_t& raw_metrics, bool is_device_level_only)
{
    ProcessedMetrics result{};

    if(is_device_level_only)
    {
        copy_valid_metrics(result.metrics.jpeg_activity, raw_metrics.jpeg_activity,
                           UINT16_MAX);
        result.has_data = !result.metrics.jpeg_activity.empty();
    }
    else
    {
        // XCP-level JPEG metrics
        for(const auto& xcp : raw_metrics.xcp_stats)
        {
            std::vector<uint16_t> xcp_jpeg_data;
            copy_valid_metrics(xcp_jpeg_data, xcp.jpeg_busy, UINT16_MAX);
            if(!xcp_jpeg_data.empty())
            {
                result.metrics.jpeg_busy.push_back(std::move(xcp_jpeg_data));
            }
        }
        result.has_data = !result.metrics.jpeg_busy.empty();
    }

    return result;
}

ProcessedMetrics
process_xgmi_metrics(const amdsmi_gpu_metrics_t& raw_metrics)
{
    ProcessedMetrics result{};

    result.metrics.xgmi_link_width = filter_unsupported_value(raw_metrics.xgmi_link_width);
    result.metrics.xgmi_link_speed = filter_unsupported_value(raw_metrics.xgmi_link_speed);

    copy_valid_metrics(result.metrics.xgmi_read_data_acc, raw_metrics.xgmi_read_data_acc,
                       UINT64_MAX);
    copy_valid_metrics(result.metrics.xgmi_write_data_acc, raw_metrics.xgmi_write_data_acc,
                       UINT64_MAX);

    result.has_data = result.metrics.xgmi_link_width != 0 ||
                      result.metrics.xgmi_link_speed != 0 ||
                      !result.metrics.xgmi_read_data_acc.empty() ||
                      !result.metrics.xgmi_write_data_acc.empty();

    return result;
}

ProcessedMetrics
process_pcie_metrics(const amdsmi_gpu_metrics_t& raw_metrics)
{
    ProcessedMetrics result{};

    result.metrics.pcie_link_width = filter_unsupported_value(raw_metrics.pcie_link_width);
    result.metrics.pcie_link_speed = filter_unsupported_value(raw_metrics.pcie_link_speed);
    result.metrics.pcie_bandwidth_acc =
        filter_unsupported_value(raw_metrics.pcie_bandwidth_acc);
    result.metrics.pcie_bandwidth_inst =
        filter_unsupported_value(raw_metrics.pcie_bandwidth_inst);

    result.has_data = result.metrics.pcie_link_width != 0 ||
                      result.metrics.pcie_link_speed != 0 ||
                      result.metrics.pcie_bandwidth_acc != 0 ||
                      result.metrics.pcie_bandwidth_inst != 0;

    return result;
}

ProcessedMetrics
merge_processed_metrics(const ProcessedMetrics& vcn, const ProcessedMetrics& jpeg,
                        const ProcessedMetrics& xgmi, const ProcessedMetrics& pcie)
{
    ProcessedMetrics result{};

    // Merge VCN data
    result.metrics.vcn_activity = vcn.metrics.vcn_activity;
    result.metrics.vcn_busy     = vcn.metrics.vcn_busy;

    // Merge JPEG data
    result.metrics.jpeg_activity = jpeg.metrics.jpeg_activity;
    result.metrics.jpeg_busy     = jpeg.metrics.jpeg_busy;

    // Merge XGMI data
    result.metrics.xgmi_link_width    = xgmi.metrics.xgmi_link_width;
    result.metrics.xgmi_link_speed    = xgmi.metrics.xgmi_link_speed;
    result.metrics.xgmi_read_data_acc = xgmi.metrics.xgmi_read_data_acc;
    result.metrics.xgmi_write_data_acc = xgmi.metrics.xgmi_write_data_acc;

    // Merge PCIe data
    result.metrics.pcie_link_width    = pcie.metrics.pcie_link_width;
    result.metrics.pcie_link_speed    = pcie.metrics.pcie_link_speed;
    result.metrics.pcie_bandwidth_acc  = pcie.metrics.pcie_bandwidth_acc;
    result.metrics.pcie_bandwidth_inst = pcie.metrics.pcie_bandwidth_inst;

    result.has_data = vcn.has_data || jpeg.has_data || xgmi.has_data || pcie.has_data;

    return result;
}

#endif  // ROCPROFSYS_USE_ROCM > 0

size_t
serialize_settings(const settings& s)
{
    std::bitset<8> settings_bits;
    settings_bits.reset();

    settings_bits.set(
        static_cast<int>(trace_cache::amd_smi_sample::settings_positions::busy), s.busy);
    settings_bits.set(
        static_cast<int>(trace_cache::amd_smi_sample::settings_positions::temp), s.temp);
    settings_bits.set(
        static_cast<int>(trace_cache::amd_smi_sample::settings_positions::power),
        s.power);
    settings_bits.set(
        static_cast<int>(trace_cache::amd_smi_sample::settings_positions::mem_usage),
        s.mem_usage);
    settings_bits.set(
        static_cast<int>(trace_cache::amd_smi_sample::settings_positions::vcn_activity),
        s.vcn_activity);
    settings_bits.set(
        static_cast<int>(trace_cache::amd_smi_sample::settings_positions::jpeg_activity),
        s.jpeg_activity);
    settings_bits.set(
        static_cast<int>(trace_cache::amd_smi_sample::settings_positions::xgmi), s.xgmi);
    settings_bits.set(
        static_cast<int>(trace_cache::amd_smi_sample::settings_positions::pcie), s.pcie);

    return settings_bits.to_ulong();
}

std::vector<uint8_t>
serialize_gpu_metrics(const gpu::gpu_metrics_t&              metrics,
                      const gpu::gpu_metrics_capabilities_t& capabilities,
                      const settings&                        s)
{
    // Convert amd_smi::settings to gpu::gpu_metrics_settings_t
    gpu::gpu_metrics_settings_t gpu_settings;
    gpu_settings.vcn_activity  = s.vcn_activity;
    gpu_settings.jpeg_activity = s.jpeg_activity;
    gpu_settings.xgmi          = s.xgmi;
    gpu_settings.pcie          = s.pcie;

    // Use the shared serialization function
    return gpu::serialize_gpu_metrics(metrics, capabilities, gpu_settings);
}

}  // namespace metrics
}  // namespace amd_smi
}  // namespace rocprofsys
