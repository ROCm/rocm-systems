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

#ifndef INCLUDE_RDC_LIB_IMPL_RDCMETRICCONTROL_H_
#define INCLUDE_RDC_LIB_IMPL_RDCMETRICCONTROL_H_

#include <atomic>
#include <mutex>
#include <string>

#include "rdc/rdc.h"

namespace amd {
namespace rdc {

/**
 * @brief Singleton class for controlling metric collection enable/disable
 *
 * This class provides a hierarchical control system for RDC metric collection
 * to help identify performance bottlenecks:
 *
 * Level 1: Data source controls (AMD-SMI vs ROCProfiler)
 * Level 2: Category controls (power, thermal, clocks, etc.)
 * Tier 3A: Individual API controls for known performance hotspots
 *
 * All controls default to enabled (true) for backward compatibility.
 * When a metric is disabled, collection returns 0 with RDC_ST_OK status.
 */
class RdcMetricControl {
 public:
  /**
   * @brief Get the singleton instance
   * @return Reference to the singleton instance
   */
  static RdcMetricControl& getInstance();

  /**
   * @brief Check if a specific field should be collected
   * @param field_id The RDC field ID to check
   * @return true if the field should be collected, false otherwise
   */
  bool shouldCollectField(rdc_field_t field_id) const;

  /**
   * @brief Load configuration from environment variables
   *
   * Parses environment variables with pattern RDC_ENABLE_* and RDC_*
   * Values "1", "true", "TRUE", "yes" = enabled, anything else = disabled
   */
  void setFromEnvironment();

  /**
   * @brief Load configuration from a config file
   * @param path Path to the INI-style configuration file
   */
  void setFromConfigFile(const std::string& path);

  /**
   * @brief Reload configuration if the reload interval has been reached
   *
   * Increments iteration counter and reloads config file if needed.
   * Thread-safe.
   */
  void reloadConfigIfNeeded();

  /**
   * @brief Get current configuration as a string
   * @return String representation of current configuration
   */
  std::string getConfigString() const;

  /**
   * @brief Enable a category at runtime
   * @param category Category name (e.g., "power", "thermal", "amdsmi")
   */
  void enableCategory(const std::string& category);

  /**
   * @brief Disable a category at runtime
   * @param category Category name (e.g., "power", "thermal", "amdsmi")
   */
  void disableCategory(const std::string& category);

  /**
   * @brief Enable all metric collection
   */
  void enableAll();

  /**
   * @brief Disable all metric collection
   */
  void disableAll();

  // Level 1 Controls: Data Sources
  std::atomic<bool> enable_amdsmi_queries{true};
  std::atomic<bool> enable_rocprofiler_queries{true};

  // Level 2 Controls: AMD-SMI Categories
  std::atomic<bool> enable_power_metrics{true};
  std::atomic<bool> enable_thermal_metrics{true};
  std::atomic<bool> enable_clock_metrics{true};
  std::atomic<bool> enable_memory_metrics{true};
  std::atomic<bool> enable_utilization_metrics{true};
  std::atomic<bool> enable_ecc_metrics{true};
  std::atomic<bool> enable_pcie_metrics{true};
  std::atomic<bool> enable_xgmi_metrics{true};
  std::atomic<bool> enable_device_info{true};
  std::atomic<bool> enable_cpu_metrics{true};

  // Level 2 Controls: ROCProfiler Categories
  std::atomic<bool> enable_prof_occupancy{true};
  std::atomic<bool> enable_prof_flops{true};
  std::atomic<bool> enable_prof_bandwidth{true};
  std::atomic<bool> enable_prof_pipeline{true};

  // Tier 3A Controls: Known Performance Hotspots
  // PCIe throughput - VERY SLOW (resets counters, cached for 30s in RDC)
  std::atomic<bool> enable_amdsmi_get_gpu_pci_throughput{true};

  // Bulk fetch optimization (fetches many metrics at once)
  std::atomic<bool> enable_amdsmi_get_gpu_metrics_info{true};

  // EEPROM validation - VERY SLOW (hardware access)
  std::atomic<bool> enable_amdsmi_gpu_validate_ras_eeprom{true};

  // Process information (system-wide scan)
  std::atomic<bool> enable_amdsmi_get_gpu_compute_process_info{true};

  // Topology operations (O(n²) with GPU count)
  std::atomic<bool> enable_amdsmi_get_minmax_bandwidth_between_processors{true};
  std::atomic<bool> enable_amdsmi_topo_get_link_weight{true};
  std::atomic<bool> enable_amdsmi_topo_get_link_type{true};

  // ECC status/count (iterates over GPU blocks)
  std::atomic<bool> enable_amdsmi_get_gpu_ecc_status{true};
  std::atomic<bool> enable_amdsmi_get_gpu_ecc_count{true};
  std::atomic<bool> enable_amdsmi_get_gpu_bad_page_info{true};

  // Sensor reads
  std::atomic<bool> enable_amdsmi_get_temp_metric{true};
  std::atomic<bool> enable_amdsmi_get_power_info{true};

  // CPU HSMP operations (mailbox interface)
  std::atomic<bool> enable_amdsmi_get_cpu_socket_energy{true};
  std::atomic<bool> enable_amdsmi_get_cpu_fclk_mclk{true};

  // ROCProfiler sampling
  std::atomic<bool> enable_rocprof_sampling{true};

  // Optimization Controls
  std::atomic<bool> enable_bulk_fetch{true};

 private:
  // Private constructor for singleton
  RdcMetricControl();

  // Delete copy/move constructors and assignment operators
  RdcMetricControl(const RdcMetricControl&) = delete;
  RdcMetricControl& operator=(const RdcMetricControl&) = delete;
  RdcMetricControl(RdcMetricControl&&) = delete;
  RdcMetricControl& operator=(RdcMetricControl&&) = delete;

  // Iteration tracking for config reload
  std::atomic<uint64_t> iteration_count_{0};
  std::atomic<uint64_t> config_reload_interval_{0};
  std::string config_file_path_;
  std::mutex config_mutex_;

  // Helper methods
  bool parseBoolValue(const std::string& value) const;
  void parseConfigLine(const std::string& section, const std::string& key,
                       const std::string& value);
  std::string trim(const std::string& str) const;
};

}  // namespace rdc
}  // namespace amd

#endif  // INCLUDE_RDC_LIB_IMPL_RDCMETRICCONTROL_H_
