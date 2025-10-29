/*
Copyright (c) 2025 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "rdc_lib/impl/RdcMetricControl.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include "rdc_lib/RdcLogger.h"

namespace amd {
namespace rdc {

RdcMetricControl::RdcMetricControl() {
  // Load configuration: first from file (if specified), then from environment
  // Environment variables will override file settings
  const char* config_file = std::getenv("RDC_METRIC_CONFIG_FILE");
  if (config_file) {
    setFromConfigFile(config_file);
  }
  setFromEnvironment();
}

RdcMetricControl& RdcMetricControl::getInstance() {
  static RdcMetricControl instance;
  return instance;
}

bool RdcMetricControl::parseBoolValue(const std::string& value) const {
  std::string lower_value = value;
  std::transform(lower_value.begin(), lower_value.end(), lower_value.begin(), ::tolower);
  return (lower_value == "1" || lower_value == "true" || lower_value == "yes");
}

std::string RdcMetricControl::trim(const std::string& str) const {
  size_t first = str.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return "";
  size_t last = str.find_last_not_of(" \t\r\n");
  return str.substr(first, (last - first + 1));
}

void RdcMetricControl::setFromEnvironment() {
  const char* env_val = nullptr;

  // Level 1: Data sources
  if ((env_val = std::getenv("RDC_ENABLE_AMDSMI"))) {
    enable_amdsmi_queries = parseBoolValue(env_val);
  }
  if ((env_val = std::getenv("RDC_ENABLE_ROCPROFILER"))) {
    enable_rocprofiler_queries = parseBoolValue(env_val);
  }

  // Level 2: AMD-SMI Categories
  if ((env_val = std::getenv("RDC_ENABLE_POWER_METRICS"))) {
    enable_power_metrics = parseBoolValue(env_val);
  }
  if ((env_val = std::getenv("RDC_ENABLE_THERMAL_METRICS"))) {
    enable_thermal_metrics = parseBoolValue(env_val);
  }
  if ((env_val = std::getenv("RDC_ENABLE_CLOCK_METRICS"))) {
    enable_clock_metrics = parseBoolValue(env_val);
  }
  if ((env_val = std::getenv("RDC_ENABLE_MEMORY_METRICS"))) {
    enable_memory_metrics = parseBoolValue(env_val);
  }
  if ((env_val = std::getenv("RDC_ENABLE_UTILIZATION_METRICS"))) {
    enable_utilization_metrics = parseBoolValue(env_val);
  }
  if ((env_val = std::getenv("RDC_ENABLE_ECC_METRICS"))) {
    enable_ecc_metrics = parseBoolValue(env_val);
  }
  if ((env_val = std::getenv("RDC_ENABLE_PCIE_METRICS"))) {
    enable_pcie_metrics = parseBoolValue(env_val);
  }
  if ((env_val = std::getenv("RDC_ENABLE_XGMI_METRICS"))) {
    enable_xgmi_metrics = parseBoolValue(env_val);
  }
  if ((env_val = std::getenv("RDC_ENABLE_DEVICE_INFO"))) {
    enable_device_info = parseBoolValue(env_val);
  }
  if ((env_val = std::getenv("RDC_ENABLE_CPU_METRICS"))) {
    enable_cpu_metrics = parseBoolValue(env_val);
  }

  // Level 2: ROCProfiler Categories
  if ((env_val = std::getenv("RDC_ENABLE_PROF_OCCUPANCY"))) {
    enable_prof_occupancy = parseBoolValue(env_val);
  }
  if ((env_val = std::getenv("RDC_ENABLE_PROF_FLOPS"))) {
    enable_prof_flops = parseBoolValue(env_val);
  }
  if ((env_val = std::getenv("RDC_ENABLE_PROF_BANDWIDTH"))) {
    enable_prof_bandwidth = parseBoolValue(env_val);
  }
  if ((env_val = std::getenv("RDC_ENABLE_PROF_PIPELINE"))) {
    enable_prof_pipeline = parseBoolValue(env_val);
  }

  // Tier 3A: Individual API controls
  if ((env_val = std::getenv("RDC_ENABLE_AMDSMI_GET_GPU_PCI_THROUGHPUT"))) {
    enable_amdsmi_get_gpu_pci_throughput = parseBoolValue(env_val);
  }
  if ((env_val = std::getenv("RDC_ENABLE_AMDSMI_GET_GPU_METRICS_INFO"))) {
    enable_amdsmi_get_gpu_metrics_info = parseBoolValue(env_val);
  }
  if ((env_val = std::getenv("RDC_ENABLE_AMDSMI_GPU_VALIDATE_RAS_EEPROM"))) {
    enable_amdsmi_gpu_validate_ras_eeprom = parseBoolValue(env_val);
  }
  if ((env_val = std::getenv("RDC_ENABLE_AMDSMI_GET_GPU_COMPUTE_PROCESS_INFO"))) {
    enable_amdsmi_get_gpu_compute_process_info = parseBoolValue(env_val);
  }
  if ((env_val = std::getenv("RDC_ENABLE_AMDSMI_GET_MINMAX_BANDWIDTH_BETWEEN_PROCESSORS"))) {
    enable_amdsmi_get_minmax_bandwidth_between_processors = parseBoolValue(env_val);
  }
  if ((env_val = std::getenv("RDC_ENABLE_AMDSMI_TOPO_GET_LINK_WEIGHT"))) {
    enable_amdsmi_topo_get_link_weight = parseBoolValue(env_val);
  }
  if ((env_val = std::getenv("RDC_ENABLE_AMDSMI_TOPO_GET_LINK_TYPE"))) {
    enable_amdsmi_topo_get_link_type = parseBoolValue(env_val);
  }
  if ((env_val = std::getenv("RDC_ENABLE_AMDSMI_GET_GPU_ECC_STATUS"))) {
    enable_amdsmi_get_gpu_ecc_status = parseBoolValue(env_val);
  }
  if ((env_val = std::getenv("RDC_ENABLE_AMDSMI_GET_GPU_ECC_COUNT"))) {
    enable_amdsmi_get_gpu_ecc_count = parseBoolValue(env_val);
  }
  if ((env_val = std::getenv("RDC_ENABLE_AMDSMI_GET_GPU_BAD_PAGE_INFO"))) {
    enable_amdsmi_get_gpu_bad_page_info = parseBoolValue(env_val);
  }
  if ((env_val = std::getenv("RDC_ENABLE_AMDSMI_GET_TEMP_METRIC"))) {
    enable_amdsmi_get_temp_metric = parseBoolValue(env_val);
  }
  if ((env_val = std::getenv("RDC_ENABLE_AMDSMI_GET_POWER_INFO"))) {
    enable_amdsmi_get_power_info = parseBoolValue(env_val);
  }
  if ((env_val = std::getenv("RDC_ENABLE_AMDSMI_GET_CPU_SOCKET_ENERGY"))) {
    enable_amdsmi_get_cpu_socket_energy = parseBoolValue(env_val);
  }
  if ((env_val = std::getenv("RDC_ENABLE_AMDSMI_GET_CPU_FCLK_MCLK"))) {
    enable_amdsmi_get_cpu_fclk_mclk = parseBoolValue(env_val);
  }
  if ((env_val = std::getenv("RDC_ENABLE_ROCPROF_SAMPLING"))) {
    enable_rocprof_sampling = parseBoolValue(env_val);
  }

  // Optimizations
  if ((env_val = std::getenv("RDC_ENABLE_BULK_FETCH"))) {
    enable_bulk_fetch = parseBoolValue(env_val);
  }

  // Config file and reload settings
  if ((env_val = std::getenv("RDC_METRIC_CONFIG_FILE"))) {
    config_file_path_ = env_val;
    setFromConfigFile(config_file_path_);
  }
  if ((env_val = std::getenv("RDC_CONFIG_RELOAD_INTERVAL"))) {
    config_reload_interval_ = std::stoull(env_val);
  }
}

void RdcMetricControl::parseConfigLine(const std::string& section, const std::string& key,
                                        const std::string& value) {
  bool bool_value = parseBoolValue(value);

  // Data sources section
  if (section == "data_sources") {
    if (key == "amdsmi") {
      enable_amdsmi_queries = bool_value;
    } else if (key == "rocprofiler") {
      enable_rocprofiler_queries = bool_value;
    }
  }
  // AMD-SMI categories section
  else if (section == "amdsmi_categories") {
    if (key == "power") {
      enable_power_metrics = bool_value;
    } else if (key == "thermal") {
      enable_thermal_metrics = bool_value;
    } else if (key == "clocks") {
      enable_clock_metrics = bool_value;
    } else if (key == "memory") {
      enable_memory_metrics = bool_value;
    } else if (key == "utilization") {
      enable_utilization_metrics = bool_value;
    } else if (key == "ecc") {
      enable_ecc_metrics = bool_value;
    } else if (key == "pcie") {
      enable_pcie_metrics = bool_value;
    } else if (key == "xgmi") {
      enable_xgmi_metrics = bool_value;
    } else if (key == "device_info") {
      enable_device_info = bool_value;
    } else if (key == "cpu") {
      enable_cpu_metrics = bool_value;
    }
  }
  // ROCProfiler categories section
  else if (section == "rocprofiler_categories") {
    if (key == "occupancy") {
      enable_prof_occupancy = bool_value;
    } else if (key == "flops") {
      enable_prof_flops = bool_value;
    } else if (key == "bandwidth") {
      enable_prof_bandwidth = bool_value;
    } else if (key == "pipeline") {
      enable_prof_pipeline = bool_value;
    }
  }
  // Tier 3A APIs section
  else if (section == "tier3a_apis") {
    if (key == "amdsmi_get_gpu_pci_throughput") {
      enable_amdsmi_get_gpu_pci_throughput = bool_value;
    } else if (key == "amdsmi_get_gpu_metrics_info") {
      enable_amdsmi_get_gpu_metrics_info = bool_value;
    } else if (key == "amdsmi_gpu_validate_ras_eeprom") {
      enable_amdsmi_gpu_validate_ras_eeprom = bool_value;
    } else if (key == "amdsmi_get_gpu_compute_process_info") {
      enable_amdsmi_get_gpu_compute_process_info = bool_value;
    } else if (key == "amdsmi_get_minmax_bandwidth_between_processors") {
      enable_amdsmi_get_minmax_bandwidth_between_processors = bool_value;
    } else if (key == "amdsmi_topo_get_link_weight") {
      enable_amdsmi_topo_get_link_weight = bool_value;
    } else if (key == "amdsmi_topo_get_link_type") {
      enable_amdsmi_topo_get_link_type = bool_value;
    } else if (key == "amdsmi_get_gpu_ecc_status") {
      enable_amdsmi_get_gpu_ecc_status = bool_value;
    } else if (key == "amdsmi_get_gpu_ecc_count") {
      enable_amdsmi_get_gpu_ecc_count = bool_value;
    } else if (key == "amdsmi_get_gpu_bad_page_info") {
      enable_amdsmi_get_gpu_bad_page_info = bool_value;
    } else if (key == "amdsmi_get_temp_metric") {
      enable_amdsmi_get_temp_metric = bool_value;
    } else if (key == "amdsmi_get_power_info") {
      enable_amdsmi_get_power_info = bool_value;
    } else if (key == "amdsmi_get_cpu_socket_energy") {
      enable_amdsmi_get_cpu_socket_energy = bool_value;
    } else if (key == "amdsmi_get_cpu_fclk_mclk") {
      enable_amdsmi_get_cpu_fclk_mclk = bool_value;
    } else if (key == "rocprof_sampling") {
      enable_rocprof_sampling = bool_value;
    }
  }
  // Optimizations section
  else if (section == "optimizations") {
    if (key == "bulk_fetch") {
      enable_bulk_fetch = bool_value;
    }
  }
  // Reload section
  else if (section == "reload") {
    if (key == "interval") {
      config_reload_interval_ = std::stoull(value);
    }
  }
}

void RdcMetricControl::setFromConfigFile(const std::string& path) {
  std::ifstream config_file(path);
  if (!config_file.is_open()) {
    RDC_LOG(RDC_DEBUG, "Unable to open config file: " << path);
    return;
  }

  std::string line;
  std::string current_section;

  while (std::getline(config_file, line)) {
    line = trim(line);

    // Skip empty lines and comments
    if (line.empty() || line[0] == '#' || line[0] == ';') {
      continue;
    }

    // Check for section header
    if (line[0] == '[' && line[line.length() - 1] == ']') {
      current_section = line.substr(1, line.length() - 2);
      current_section = trim(current_section);
      continue;
    }

    // Parse key=value pairs
    size_t eq_pos = line.find('=');
    if (eq_pos != std::string::npos) {
      std::string key = trim(line.substr(0, eq_pos));
      std::string value = trim(line.substr(eq_pos + 1));

      // Remove inline comments
      size_t comment_pos = value.find('#');
      if (comment_pos != std::string::npos) {
        value = trim(value.substr(0, comment_pos));
      }

      parseConfigLine(current_section, key, value);
    }
  }

  config_file.close();
  RDC_LOG(RDC_INFO, "Loaded metric control config from: " << path);
}

void RdcMetricControl::reloadConfigIfNeeded() {
  uint64_t reload_interval = config_reload_interval_.load();

  // If reload is disabled or no config file is set, skip
  if (reload_interval == 0 || config_file_path_.empty()) {
    return;
  }

  // Increment iteration count
  uint64_t current_iteration = iteration_count_.fetch_add(1) + 1;

  // Check if we should reload
  if (current_iteration % reload_interval == 0) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    setFromConfigFile(config_file_path_);
    RDC_LOG(RDC_DEBUG, "Reloaded metric control config at iteration " << current_iteration);
  }
}

bool RdcMetricControl::shouldCollectField(rdc_field_t field_id) const {
  // Check Level 1: Data source
  // Determine if field is from AMD-SMI or ROCProfiler
  // ROCProfiler fields are defined in the range [RDC_FI_PROF_OCCUPANCY_PERCENT (800), RDC_EVNT_XGMI_0_NOP_TX).
  // RDC_EVNT_XGMI_0_NOP_TX marks the start of XGMI event fields, so this range check isolates ROCProfiler field IDs.
  bool is_rocprof_field = (field_id >= RDC_FI_PROF_OCCUPANCY_PERCENT && field_id < RDC_EVNT_XGMI_0_NOP_TX);

  if (is_rocprof_field) {
    if (!enable_rocprofiler_queries) {
      return false;
    }
    // Check ROCProfiler subcategories if needed
    // Note: More granular categorization can be added here
    return true;
  }

  // AMD-SMI field
  if (!enable_amdsmi_queries) {
    return false;
  }

  // Check Level 2: Category
  // Power metrics
  if (field_id == RDC_FI_POWER_USAGE) {
    return enable_power_metrics;
  }

  // Thermal metrics
  if (field_id == RDC_FI_GPU_TEMP || field_id == RDC_FI_MEMORY_TEMP) {
    return enable_thermal_metrics;
  }

  // Clock metrics
  if (field_id == RDC_FI_GPU_CLOCK) {
    return enable_clock_metrics;
  }

  // Memory metrics
  if (field_id == RDC_FI_GPU_MEMORY_USAGE || field_id == RDC_FI_GPU_MEMORY_TOTAL ||
      field_id == RDC_FI_GPU_MEMORY_ACTIVITY || field_id == RDC_FI_GPU_MEMORY_MAX_BANDWIDTH ||
      field_id == RDC_FI_GPU_MEMORY_CUR_BANDWIDTH) {
    return enable_memory_metrics;
  }

  // Utilization metrics
  if (field_id == RDC_FI_GPU_UTIL || field_id == RDC_FI_GPU_BUSY_PERCENT ||
      field_id == RDC_FI_GPU_MM_ENC_UTIL || field_id == RDC_FI_GPU_MM_DEC_UTIL) {
    return enable_utilization_metrics;
  }

  // ECC metrics
  if ((field_id >= RDC_FI_ECC_FIRST && field_id <= RDC_FI_ECC_LAST) ||
      field_id == RDC_FI_ECC_CORRECT_TOTAL || field_id == RDC_FI_ECC_UNCORRECT_TOTAL ||
      field_id == RDC_FI_GPU_PAGE_RETRIED) {
    return enable_ecc_metrics;
  }

  // PCIe metrics
  if (field_id == RDC_FI_PCIE_TX || field_id == RDC_FI_PCIE_RX ||
      field_id == RDC_FI_PCIE_BANDWIDTH) {
    return enable_pcie_metrics;
  }

  // XGMI metrics (both events and KB fields)
  if ((field_id >= RDC_FI_XGMI_0_READ_KB && field_id <= RDC_FI_XGMI_TOTAL_WRITE_KB) ||
      (field_id >= RDC_EVNT_XGMI_0_NOP_TX && field_id < RDC_EVNT_NOTIF_VMFAULT)) {
    return enable_xgmi_metrics;
  }

  // CPU metrics
  if (field_id >= RDC_FI_CPU_FIRST && field_id <= RDC_FI_CPU_LAST) {
    return enable_cpu_metrics;
  }

  // Device info (IDs, names, etc.) - typically fast, controlled by enable_device_info
  // Default to enabled for unlisted fields
  return true;
}

std::string RdcMetricControl::getConfigString() const {
  std::ostringstream oss;
  oss << "[Data Sources]\n";
  oss << "  AMD-SMI: " << (enable_amdsmi_queries ? "enabled" : "disabled") << "\n";
  oss << "  ROCProfiler: " << (enable_rocprofiler_queries ? "enabled" : "disabled") << "\n";

  oss << "\n[AMD-SMI Categories]\n";
  oss << "  Power: " << (enable_power_metrics ? "enabled" : "disabled") << "\n";
  oss << "  Thermal: " << (enable_thermal_metrics ? "enabled" : "disabled") << "\n";
  oss << "  Clocks: " << (enable_clock_metrics ? "enabled" : "disabled") << "\n";
  oss << "  Memory: " << (enable_memory_metrics ? "enabled" : "disabled") << "\n";
  oss << "  Utilization: " << (enable_utilization_metrics ? "enabled" : "disabled") << "\n";
  oss << "  ECC: " << (enable_ecc_metrics ? "enabled" : "disabled") << "\n";
  oss << "  PCIe: " << (enable_pcie_metrics ? "enabled" : "disabled") << "\n";
  oss << "  XGMI: " << (enable_xgmi_metrics ? "enabled" : "disabled") << "\n";
  oss << "  Device Info: " << (enable_device_info ? "enabled" : "disabled") << "\n";
  oss << "  CPU: " << (enable_cpu_metrics ? "enabled" : "disabled") << "\n";

  oss << "\n[ROCProfiler Categories]\n";
  oss << "  Occupancy: " << (enable_prof_occupancy ? "enabled" : "disabled") << "\n";
  oss << "  FLOPS: " << (enable_prof_flops ? "enabled" : "disabled") << "\n";
  oss << "  Bandwidth: " << (enable_prof_bandwidth ? "enabled" : "disabled") << "\n";
  oss << "  Pipeline: " << (enable_prof_pipeline ? "enabled" : "disabled") << "\n";

  oss << "\n[Tier 3A APIs]\n";
  oss << "  PCI Throughput: " << (enable_amdsmi_get_gpu_pci_throughput ? "enabled" : "disabled")
      << "\n";
  oss << "  GPU Metrics Info: " << (enable_amdsmi_get_gpu_metrics_info ? "enabled" : "disabled")
      << "\n";
  oss << "  RAS EEPROM: " << (enable_amdsmi_gpu_validate_ras_eeprom ? "enabled" : "disabled")
      << "\n";
  oss << "  Process Info: "
      << (enable_amdsmi_get_gpu_compute_process_info ? "enabled" : "disabled") << "\n";

  oss << "\n[Optimizations]\n";
  oss << "  Bulk Fetch: " << (enable_bulk_fetch ? "enabled" : "disabled") << "\n";
  oss << "  Config Reload Interval: " << config_reload_interval_ << "\n";

  return oss.str();
}

void RdcMetricControl::enableCategory(const std::string& category) {
  std::string lower_cat = category;
  std::transform(lower_cat.begin(), lower_cat.end(), lower_cat.begin(), ::tolower);

  if (lower_cat == "amdsmi") {
    enable_amdsmi_queries = true;
  } else if (lower_cat == "rocprofiler") {
    enable_rocprofiler_queries = true;
  } else if (lower_cat == "power") {
    enable_power_metrics = true;
  } else if (lower_cat == "thermal") {
    enable_thermal_metrics = true;
  } else if (lower_cat == "clocks") {
    enable_clock_metrics = true;
  } else if (lower_cat == "memory") {
    enable_memory_metrics = true;
  } else if (lower_cat == "utilization") {
    enable_utilization_metrics = true;
  } else if (lower_cat == "ecc") {
    enable_ecc_metrics = true;
  } else if (lower_cat == "pcie") {
    enable_pcie_metrics = true;
  } else if (lower_cat == "xgmi") {
    enable_xgmi_metrics = true;
  } else if (lower_cat == "device_info") {
    enable_device_info = true;
  } else if (lower_cat == "cpu") {
    enable_cpu_metrics = true;
  }
}

void RdcMetricControl::disableCategory(const std::string& category) {
  std::string lower_cat = category;
  std::transform(lower_cat.begin(), lower_cat.end(), lower_cat.begin(), ::tolower);

  if (lower_cat == "amdsmi") {
    enable_amdsmi_queries = false;
  } else if (lower_cat == "rocprofiler") {
    enable_rocprofiler_queries = false;
  } else if (lower_cat == "power") {
    enable_power_metrics = false;
  } else if (lower_cat == "thermal") {
    enable_thermal_metrics = false;
  } else if (lower_cat == "clocks") {
    enable_clock_metrics = false;
  } else if (lower_cat == "memory") {
    enable_memory_metrics = false;
  } else if (lower_cat == "utilization") {
    enable_utilization_metrics = false;
  } else if (lower_cat == "ecc") {
    enable_ecc_metrics = false;
  } else if (lower_cat == "pcie") {
    enable_pcie_metrics = false;
  } else if (lower_cat == "xgmi") {
    enable_xgmi_metrics = false;
  } else if (lower_cat == "device_info") {
    enable_device_info = false;
  } else if (lower_cat == "cpu") {
    enable_cpu_metrics = false;
  }
}

void RdcMetricControl::enableAll() {
  // Level 1
  enable_amdsmi_queries = true;
  enable_rocprofiler_queries = true;

  // Level 2: AMD-SMI
  enable_power_metrics = true;
  enable_thermal_metrics = true;
  enable_clock_metrics = true;
  enable_memory_metrics = true;
  enable_utilization_metrics = true;
  enable_ecc_metrics = true;
  enable_pcie_metrics = true;
  enable_xgmi_metrics = true;
  enable_device_info = true;
  enable_cpu_metrics = true;

  // Level 2: ROCProfiler
  enable_prof_occupancy = true;
  enable_prof_flops = true;
  enable_prof_bandwidth = true;
  enable_prof_pipeline = true;

  // Tier 3A
  enable_amdsmi_get_gpu_pci_throughput = true;
  enable_amdsmi_get_gpu_metrics_info = true;
  enable_amdsmi_gpu_validate_ras_eeprom = true;
  enable_amdsmi_get_gpu_compute_process_info = true;
  enable_amdsmi_get_minmax_bandwidth_between_processors = true;
  enable_amdsmi_topo_get_link_weight = true;
  enable_amdsmi_topo_get_link_type = true;
  enable_amdsmi_get_gpu_ecc_status = true;
  enable_amdsmi_get_gpu_ecc_count = true;
  enable_amdsmi_get_gpu_bad_page_info = true;
  enable_amdsmi_get_temp_metric = true;
  enable_amdsmi_get_power_info = true;
  enable_amdsmi_get_cpu_socket_energy = true;
  enable_amdsmi_get_cpu_fclk_mclk = true;
  enable_rocprof_sampling = true;

  // Optimizations
  enable_bulk_fetch = true;
}

void RdcMetricControl::disableAll() {
  // Level 1
  enable_amdsmi_queries = false;
  enable_rocprofiler_queries = false;

  // Level 2: AMD-SMI
  enable_power_metrics = false;
  enable_thermal_metrics = false;
  enable_clock_metrics = false;
  enable_memory_metrics = false;
  enable_utilization_metrics = false;
  enable_ecc_metrics = false;
  enable_pcie_metrics = false;
  enable_xgmi_metrics = false;
  enable_device_info = false;
  enable_cpu_metrics = false;

  // Level 2: ROCProfiler
  enable_prof_occupancy = false;
  enable_prof_flops = false;
  enable_prof_bandwidth = false;
  enable_prof_pipeline = false;

  // Tier 3A
  enable_amdsmi_get_gpu_pci_throughput = false;
  enable_amdsmi_get_gpu_metrics_info = false;
  enable_amdsmi_gpu_validate_ras_eeprom = false;
  enable_amdsmi_get_gpu_compute_process_info = false;
  enable_amdsmi_get_minmax_bandwidth_between_processors = false;
  enable_amdsmi_topo_get_link_weight = false;
  enable_amdsmi_topo_get_link_type = false;
  enable_amdsmi_get_gpu_ecc_status = false;
  enable_amdsmi_get_gpu_ecc_count = false;
  enable_amdsmi_get_gpu_bad_page_info = false;
  enable_amdsmi_get_temp_metric = false;
  enable_amdsmi_get_power_info = false;
  enable_amdsmi_get_cpu_socket_energy = false;
  enable_amdsmi_get_cpu_fclk_mclk = false;
  enable_rocprof_sampling = false;

  // Optimizations
  enable_bulk_fetch = false;
}

}  // namespace rdc
}  // namespace amd
