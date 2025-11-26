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

#pragma once

#include "core/gpu_metrics.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "library/amd_smi.hpp"

#if ROCPROFSYS_USE_ROCM > 0
#    include <amd_smi/amdsmi.h>
#endif

#include <bitset>
#include <cstdint>
#include <limits>
#include <vector>

namespace rocprofsys
{
namespace amd_smi
{
namespace metrics
{
/// @brief Filters out unsupported values indicated by max uint
/// @tparam T Numeric type
/// @param value The value to filter
/// @return 0 if value equals max for type T, otherwise the original value
template <typename T>
constexpr T
filter_unsupported_value(T value)
{
    return (value == std::numeric_limits<T>::max()) ? T{ 0 } : value;
}

/// @brief Copies valid metrics from source to destination, filtering out max values
/// @tparam DestT Destination container type (must support push_back)
/// @tparam SrcT Source container type (must be iterable)
/// @tparam MaxT Type of the max value to filter
/// @param dest Destination container
/// @param src Source container
/// @param max_val Maximum value indicating "unsupported"
template <typename DestT, typename SrcT, typename MaxT>
void
copy_valid_metrics(DestT& dest, const SrcT& src, MaxT max_val)
{
    for(const auto& val : src)
    {
        if(val != max_val) dest.push_back(val);
    }
}

#if ROCPROFSYS_USE_ROCM > 0

/// @brief Result of processing advanced GPU metrics
struct ProcessedMetrics
{
    gpu::gpu_metrics_t metrics{};
    bool               has_data = false;
};

/// @brief Process VCN activity metrics from raw GPU metrics
/// @param raw_metrics Raw metrics from AMD SMI
/// @param is_device_level_only Whether VCN is device-level only (vs per-XCP)
/// @return Processed metrics and flag indicating if data was found
ProcessedMetrics
process_vcn_metrics(const amdsmi_gpu_metrics_t& raw_metrics, bool is_device_level_only);

/// @brief Process JPEG activity metrics from raw GPU metrics
/// @param raw_metrics Raw metrics from AMD SMI
/// @param is_device_level_only Whether JPEG is device-level only (vs per-XCP)
/// @return Processed metrics and flag indicating if data was found
ProcessedMetrics
process_jpeg_metrics(const amdsmi_gpu_metrics_t& raw_metrics, bool is_device_level_only);

/// @brief Process XGMI metrics from raw GPU metrics
/// @param raw_metrics Raw metrics from AMD SMI
/// @return Processed metrics and flag indicating if data was found
ProcessedMetrics
process_xgmi_metrics(const amdsmi_gpu_metrics_t& raw_metrics);

/// @brief Process PCIe metrics from raw GPU metrics
/// @param raw_metrics Raw metrics from AMD SMI
/// @return Processed metrics and flag indicating if data was found
ProcessedMetrics
process_pcie_metrics(const amdsmi_gpu_metrics_t& raw_metrics);

/// @brief Merge multiple ProcessedMetrics into a single gpu_metrics_t
/// @param vcn VCN metrics result
/// @param jpeg JPEG metrics result
/// @param xgmi XGMI metrics result
/// @param pcie PCIe metrics result
/// @return Combined metrics with has_data flag
ProcessedMetrics
merge_processed_metrics(const ProcessedMetrics& vcn, const ProcessedMetrics& jpeg,
                        const ProcessedMetrics& xgmi, const ProcessedMetrics& pcie);

#endif  // ROCPROFSYS_USE_ROCM > 0

/// @brief Serialize settings to a bitset representation
/// @param s Settings to serialize
/// @return Serialized settings as size_t
size_t
serialize_settings(const settings& s);

/// @brief Serialize GPU metrics for storage
/// @param metrics GPU metrics to serialize
/// @param capabilities GPU capabilities flags
/// @param s Settings indicating which metrics are enabled
/// @return Serialized bytes
std::vector<uint8_t>
serialize_gpu_metrics(const gpu::gpu_metrics_t&               metrics,
                      const gpu::gpu_metrics_capabilities_t&  capabilities,
                      const settings&                         s);

}  // namespace metrics
}  // namespace amd_smi
}  // namespace rocprofsys
