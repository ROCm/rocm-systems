/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_main.h"
#include "rocm_smi/rocm_smi_dyn_gpu_metrics.h"
#include "rocm_smi/rocm_smi_logger.h"
#include "rocm_smi/rocm_smi_utils.h"
#include <cstddef>
#include <cstring>
#include <shared_mutex>
#include <optional>

namespace amd::smi
{

using namespace details;

struct Cursor {
  const std::byte* byte_ptr;
  std::size_t      remainder;
};

// Used when mismatch in schema to safely skip value
static inline bool skip_payload(Cursor& cur,
                                AMDGpuMetricAttributeType_t t,
                                uint64_t instances) {

  const std::size_t elem = get_metric_bytes(t);
  if (elem == 0 || instances > std::numeric_limits<size_t>::max() / elem) {
    return false;
  }

  const std::size_t bytes = static_cast<std::size_t>(instances) * elem;
  if (cur.remainder < bytes) {
    return false;
  }

  cur.byte_ptr  += bytes;
  cur.remainder -= bytes;
  return true;
}

// Lookup a schema instance for (attr_id, attr_type)
static inline rsmi_status_t schema_lookup_instance( AMDGpuMetricAttributeId_t attr_id,
                                                    AMDGpuMetricAttributeType_t attr_type,
                                                    AMDGpuMetricAttributeInstance_t& schema_inst) {

  if (const auto attr_id_itr = AMDGpuMetricsBaseSchema.find(attr_id); attr_id_itr != AMDGpuMetricsBaseSchema.end()) {
      const auto& inst = attr_id_itr->second.m_instance;
      if (inst.m_attribute_type == attr_type) {
          schema_inst = inst;
          return RSMI_STATUS_SUCCESS;
      }
      return RSMI_STATUS_NOT_SUPPORTED;
  }
  return RSMI_STATUS_NOT_FOUND;
}

template <class T>
static inline std::optional<T> read_scalar(Cursor& c) {
  // Ensure we can read safely
  if (c.remainder < sizeof(T)) {
    return std::nullopt;
  }
  T v{};
  std::memcpy(&v, c.byte_ptr, sizeof(T));
  c.byte_ptr   += sizeof(T);
  c.remainder -= sizeof(T);
  return v;
}

template <class T>
static inline std::optional<std::vector<T>> read_vector(Cursor& c, std::size_t count) {

  static_assert(std::is_integral_v<T> && std::is_trivially_copyable_v<T>,
              "metrics expect integral element types");

  // Prevent size_t overflow
  if (count > SIZE_MAX / sizeof(T) || count == 0) {
      return std::nullopt;
  }

  // Ensure we can read entire array safely
  const std::size_t bytes = count * sizeof(T);
  if (c.remainder < bytes) {
    return std::nullopt;
  }

  std::vector<T> out;
  out.resize(count);
  std::memcpy(out.data(), c.byte_ptr, bytes);
  c.byte_ptr   += bytes;
  c.remainder -= bytes;
  return out;
}

// Template to fill AMDGpuMetricAttributeValue_t with either a scalar<T> or vector<T>
template <typename T>
static inline std::optional<AMDGpuMetricAttributeValue_t> read_metric_value(Cursor& c,
                                                                            uint64_t instances) {

  if (instances == 1) {
    if (auto v = read_scalar<T>(c)) {
      return AMDGpuMetricAttributeValue_t{*v};
    }
    return std::nullopt;
  }
  if (auto vv = read_vector<T>(c, static_cast<std::size_t>(instances))) {
    return AMDGpuMetricAttributeValue_t{std::move(*vv)};
  }
  return std::nullopt;
}

auto AMDGpuDynamicMetrics_t::parse_from_buffer(const std::byte* data,
                                              std::size_t size) noexcept -> rsmi_status_t {
  std::ostringstream ss;
  rsmi_status_t status = RSMI_STATUS_SUCCESS;
  if (!data || (size < (sizeof(AMDGpuDynamicMetricsHeader_v1_t) + sizeof(uint32_t)))) {
      return RSMI_STATUS_INSUFFICIENT_SIZE;
  }

  // Grab header
  details::AMDGpuDynamicMetricsHeader_v1_t hdr{};
  std::memcpy(&hdr, data, sizeof(hdr));

  // Advance metrics pointer past header and keep track of remaining file size
  Cursor cur{ (data + sizeof(hdr)), (size - sizeof(hdr)) };

  // Grab attribute count, directly after header and increment
  auto attr_count_opt = read_scalar<uint32_t>(cur);
  if (!attr_count_opt) {
    return RSMI_STATUS_UNEXPECTED_SIZE;
  }
  uint32_t attr_count = *attr_count_opt;
  if (attr_count == 0 || attr_count > size){
    return RSMI_STATUS_UNEXPECTED_SIZE;
  }
  std::string m_header_version_str = std::to_string(static_cast<uint32_t>(hdr.m_format_revision))
                                     + "." +
                                     std::to_string(static_cast<uint32_t>(hdr.m_content_revision));
  ss << __PRETTY_FUNCTION__
     << " | Info: Dynamic GPU Metrics"
     << " | Attr Count: " << attr_count
     << " | Header Version: " << m_header_version_str
     << " | Header Size: " << hdr.get_size()
     << " | Total Size: " << size
     << " |";
  LOG_TRACE(ss);

  details::AMDGpuMetricSchemaType_t metrics_data;
  metrics_data.reserve(attr_count);
  AMDGpuDynamicMetricsOffsetMap_t offsets;
  for (uint32_t i = 0; i < attr_count; ++i) {

    if (cur.remainder < sizeof(uint64_t)) {
      return RSMI_STATUS_UNEXPECTED_SIZE;
    }

    // Absolute offset for attribute start in file
    const std::size_t entry_start = static_cast<std::size_t>(cur.byte_ptr - data);

    // Read attribute instance and increment
    auto enc_opt = read_scalar<uint64_t>(cur);
    if (!enc_opt) {
      return RSMI_STATUS_UNEXPECTED_SIZE;
    }
    const uint64_t enc = *enc_opt;

    const auto dec = amdgpu_metrics_decode_attr(enc);

    const auto attr_type     = static_cast<AMDGpuMetricAttributeType_t>(dec.m_attr_type);
    const auto attr_id       = static_cast<AMDGpuMetricAttributeId_t>(dec.m_attr_id);
    const auto instances    = static_cast<uint64_t>(dec.m_attr_instance);

    if (instances == 0) {
      return RSMI_STATUS_UNEXPECTED_SIZE;
    }

    // Schema lookup
    AMDGpuMetricAttributeInstance_t inst{};
    status = schema_lookup_instance(attr_id, attr_type, inst);
    if (status != RSMI_STATUS_SUCCESS){
      ss << __PRETTY_FUNCTION__
        << " | Warn: schema lookup miss"
        << " | Attr ID: "   << static_cast<std::underlying_type_t<AMDGpuMetricAttributeId_t>>(attr_id)
        << " | Attr Type: " << static_cast<std::underlying_type_t<AMDGpuMetricAttributeType_t>>(attr_type)
        << " | Returning = " << getRSMIStatusString(status)
        << " |";
      LOG_TRACE(ss);

      if (!skip_payload(cur, attr_type, instances)){
        return status;
      }
      continue; // Do not emit row, go to next attribute
    }

    // Read scalar or all vector values after attribute instance
    AMDGpuMetricAttributeValue_t val{};

    std::optional<AMDGpuMetricAttributeValue_t> mv;
    switch (attr_type) {
        case AMDGpuMetricAttributeType_t::TYPE_UINT8: {
          mv = read_metric_value<std::uint8_t>(cur, instances);
          break;
        }
        case AMDGpuMetricAttributeType_t::TYPE_INT8: {
          mv = read_metric_value<std::int8_t>(cur, instances);
          break;
        }
        case AMDGpuMetricAttributeType_t::TYPE_UINT16: {
          mv = read_metric_value<std::uint16_t>(cur, instances);
          break;
        }
        case AMDGpuMetricAttributeType_t::TYPE_INT16: {
          mv = read_metric_value<std::int16_t>(cur, instances);
          break;
        }
        case AMDGpuMetricAttributeType_t::TYPE_UINT32: {
          mv = read_metric_value<std::uint32_t>(cur, instances);
          break;
        }
        case AMDGpuMetricAttributeType_t::TYPE_INT32: {
          mv = read_metric_value<std::int32_t>(cur, instances);
          break;
        }
        case AMDGpuMetricAttributeType_t::TYPE_UINT64: {
          mv = read_metric_value<std::uint64_t>(cur, instances);
          break;
        }
        case AMDGpuMetricAttributeType_t::TYPE_INT64: {
          mv = read_metric_value<std::int64_t>(cur, instances);
          break;
        }
        default: return RSMI_STATUS_INSUFFICIENT_SIZE;
    }

    if (!mv) {
      return RSMI_STATUS_UNEXPECTED_SIZE;
    }

    val = std::move(*mv); // safely set val
    const uint32_t row_index = static_cast<uint32_t>(metrics_data.size());
    metrics_data.emplace_back(inst, val);
    offsets.try_emplace(entry_start, row_index);
  }

  {
    std::unique_lock<std::shared_mutex> lk(m_mutex);
    m_header = hdr;
    m_attr_count = attr_count;
    m_dynamic_metrics_data.swap(metrics_data);
    m_dynamic_metrics_data_offsets.swap(offsets);
  }
  return RSMI_STATUS_SUCCESS;
}

auto AMDGpuDynamicMetrics_t::parse_from_file(const std::string& metrics_file_path,
                                            std::size_t read_size) -> rsmi_status_t {
  AMDGPUMetricsDynDataBuffer_t buf;

  auto st = read_dynamic_gpu_metrics_file(metrics_file_path, read_size, buf);
  if (st != RSMI_STATUS_SUCCESS) {
    return st;
  }

  return parse_from_buffer(reinterpret_cast<const std::byte*>(buf.data()), buf.size());
}

rsmi_status_t read_dynamic_gpu_metrics_file(const std::string& metrics_file_path,
                                              const size_t read_size,
                                              AMDGPUMetricsDynDataBuffer_t& out) {

  // Clear output buffer and open file stream
  out.clear();
  std::ifstream gpu_metrics_file(metrics_file_path, std::ios::binary);
  if (!gpu_metrics_file.is_open()) {
        return RSMI_STATUS_NOT_FOUND;
  }

  if ((read_size <= 0)) {
      return RSMI_STATUS_UNEXPECTED_SIZE;
  }

  out.resize(read_size);
  gpu_metrics_file.read(reinterpret_cast<char*>(out.data()),
                    static_cast<std::streamsize>(read_size));

  const std::streamsize gpu_metrics_filesize = gpu_metrics_file.gcount();

  if(gpu_metrics_filesize <= 0){
    out.clear();
    return RSMI_STATUS_NO_DATA;
  }

  out.resize(static_cast<std::size_t>(gpu_metrics_filesize));
  return RSMI_STATUS_SUCCESS;

}

}   // namespace amd::smi
