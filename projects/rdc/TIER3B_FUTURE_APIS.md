# Tier 3B: Future AMD-SMI API Controls

This document lists all remaining AMD-SMI APIs that could potentially be added as individual control flags in future iterations of the RDC Metric Control system. Currently, Tier 3A includes the 15 most performance-critical APIs. The APIs below are categorized for potential future implementation.

## Current Implementation Status

**Tier 3A (Implemented):** 15 APIs covering the highest-impact performance hotspots
- PCIe throughput
- Bulk GPU metrics fetch
- RAS EEPROM validation
- Process information
- Topology operations (3 APIs)
- ECC status and count (3 APIs)
- Temperature and power sensors
- CPU HSMP operations (2 APIs)
- ROCProfiler sampling

**Tier 3B (This Document):** Remaining AMD-SMI APIs for potential future fine-grained control

---

## Discovery & Initialization APIs

### Low Performance Impact (Fast)
- `amdsmi_init()` - One-time initialization
- `amdsmi_shut_down()` - One-time cleanup
- `amdsmi_get_socket_handles()` - Fast, cached internally
- `amdsmi_get_processor_handles()` - Fast, cached internally
- `amdsmi_get_processor_type()` - Fast, cached
- `amdsmi_get_processor_handle_from_bdf()` - Fast lookup

**Recommendation:** Low priority for individual controls. These are called infrequently and are already fast.

---

## GPU Core Metrics APIs

### Already Covered by Category Controls
These are covered by Level 2 category controls (power, thermal, clocks, memory, utilization):

#### Power Category
- `amdsmi_get_power_cap_info()` - Get power cap limits
- `amdsmi_set_power_cap()` - Set power cap (write operation)
- `amdsmi_get_gpu_power_profile_presets()` - Get power profiles

#### Clock Category
- `amdsmi_get_clock_info()` - Get all clock frequencies
- `amdsmi_set_clk_freq()` - Set clock frequency (write operation)
- `amdsmi_get_od_volt_info()` - Get overdrive voltage info
- `amdsmi_set_od_clk_info()` - Set overdrive settings (write operation)
- `amdsmi_reset_gpu_clocks()` - Reset clocks (write operation)

#### Memory Category
- `amdsmi_get_gpu_vram_info()` - Get VRAM information
- `amdsmi_get_gpu_vram_usage()` - VRAM usage (called by RDC)
- `amdsmi_get_gpu_memory_total()` - Total memory (called by RDC)
- `amdsmi_get_gpu_memory_partition()` - Memory partition info
- `amdsmi_get_gpu_ras_feature_info()` - RAS feature support

#### Utilization Category
- `amdsmi_get_gpu_activity()` - GPU activity percentages (called by RDC)

**Recommendation:** Low priority. Already controlled via Level 2 categories.

---

## Voltage APIs

### Potentially Slow (Hardware Reads)
- `amdsmi_get_gpu_volt_metric()` - Read voltage sensor
- `amdsmi_get_gpu_overdrive_level()` - Read overdrive settings
- `amdsmi_set_gpu_overdrive_level()` - Write overdrive (write operation)
- `amdsmi_get_gpu_metrics_header_info()` - Metrics header

**Performance Characteristics:**
- Voltage sensor reads can be as slow as temperature/power reads
- Usually sub-millisecond but can vary by hardware

**Recommendation:** Medium priority for Tier 3B
- Add control for `amdsmi_get_gpu_volt_metric` if voltage monitoring shows performance impact

---

## PCIe APIs

### Mixed Performance Impact

#### Already in Tier 3A
- `amdsmi_get_gpu_pci_throughput()` - VERY SLOW, already controlled

#### Fast Operations
- `amdsmi_get_gpu_bdf_id()` - Fast, BDF lookup
- `amdsmi_get_pcie_info()` - Fast, reads static PCIe info
- `amdsmi_get_gpu_pci_bandwidth()` - Fast, static limits

#### Slow Operations
- `amdsmi_get_pcie_link_metrics()` - May read hardware counters
- `amdsmi_set_pcie_link_width_control()` - Write operation

**Recommendation:** Low-Medium priority
- Most PCIe APIs are already fast
- `amdsmi_get_pcie_link_metrics` could be added to Tier 3B if needed

---

## Topology APIs

### Already in Tier 3A
- `amdsmi_get_minmax_bandwidth_between_processors()` - O(n²)
- `amdsmi_topo_get_link_weight()` - O(n²)
- `amdsmi_topo_get_link_type()` - O(n²)

### Fast Topology APIs
- `amdsmi_is_P2P_accessible()` - Fast check
- `amdsmi_get_gpu_numa_node_number()` - Fast, static
- `amdsmi_topo_get_numa_node_number()` - Fast, static

**Recommendation:** Low priority. Critical topology APIs already controlled.

---

## XGMI/Link APIs

### Potentially Slow (Iterative)
- `amdsmi_get_gpu_xgmi_error_status()` - Checks XGMI errors
- `amdsmi_reset_gpu_xgmi_error()` - Reset errors (write operation)
- `amdsmi_topo_get_link_metrics()` - Link performance metrics

**Performance Characteristics:**
- These may iterate over multiple XGMI links
- On systems with 8+ GPUs, this becomes O(n²)

**Recommendation:** Medium priority for Tier 3B
- Add controls if XGMI metric collection shows overhead

---

## Health & Diagnostics APIs

### Already Partially Covered

#### In Tier 3A
- `amdsmi_get_gpu_bad_page_info()` - Retired pages
- `amdsmi_gpu_validate_ras_eeprom()` - EEPROM validation

#### Fast Health Checks
- `amdsmi_get_gpu_ras_block_features_enabled()` - Check RAS blocks
- `amdsmi_get_gpu_bad_page_threshold()` - Get threshold value

#### Potentially Slow
- `amdsmi_status_code_to_string()` - String conversion (fast)
- `amdsmi_get_violation_status()` - Throttling status (may be slow)

**Recommendation:** Low-Medium priority
- `amdsmi_get_violation_status` could be added if throttle checking is slow

---

## Fan & Thermal Control APIs

### Write Operations (Low Priority)
- `amdsmi_set_fan_speed()` - Write fan speed
- `amdsmi_reset_fan()` - Reset fan control
- `amdsmi_set_gpu_fan_speed()` - Set fan RPM

### Read Operations (Already Controlled)
- Temperature reads already controlled via `enable_amdsmi_get_temp_metric`

**Recommendation:** Low priority. Write operations don't affect read performance.

---

## Event & Notification APIs

### Potentially Slow (Blocking)
- `amdsmi_init_gpu_event_notification()` - Initialize events
- `amdsmi_get_gpu_event_notification()` - Wait for events (blocking)
- `amdsmi_stop_gpu_event_notification()` - Stop events

**Performance Characteristics:**
- Event notification can block if waiting for events
- Not typically used in high-frequency polling scenarios

**Recommendation:** Medium priority for Tier 3B
- Add control if RDC uses event notifications in performance-sensitive paths

---

## Firmware & Version APIs

### Fast (Cached/Static)
- `amdsmi_get_fw_info()` - Firmware versions (static)
- `amdsmi_get_gpu_asic_info()` - ASIC info (already called by RDC)
- `amdsmi_get_gpu_vbios_info()` - VBIOS version (static)
- `amdsmi_get_gpu_driver_info()` - Driver info (static)
- `amdsmi_get_lib_version()` - Library version (static)

**Recommendation:** Very low priority. These are static/cached and very fast.

---

## Performance Counter APIs

### ROCProfiler Integration (Already Controlled)
- Performance counters are handled through ROCProfiler
- Already controlled via `enable_rocprof_sampling`

**Recommendation:** No additional controls needed.

---

## Administrative/Configuration APIs

### Write Operations
- `amdsmi_set_gpu_clk_limit()` - Set clock limits
- `amdsmi_set_gpu_od_clk_info()` - Set overdrive
- `amdsmi_set_gpu_pci_bandwidth()` - Set PCIe bandwidth
- `amdsmi_set_gpu_perf_level()` - Set performance level
- `amdsmi_set_power_profile()` - Set power profile

**Recommendation:** Very low priority. Write operations don't impact metric read performance.

---

## CPU-Specific APIs

### Already in Tier 3A
- `amdsmi_get_cpu_socket_energy()` - HSMP energy
- `amdsmi_get_cpu_fclk_mclk()` - HSMP clocks

### Fast CPU APIs
- `amdsmi_get_cpu_model_name()` - Static string
- `amdsmi_get_cpu_family()` - Static value
- `amdsmi_get_cpu_smu_fw_version()` - Firmware version (cached)
- `amdsmi_get_cpu_hsmp_driver_version()` - Driver version (cached)
- `amdsmi_get_cpu_hsmp_proto_ver()` - Protocol version (static)

### Potentially Slow CPU APIs
- `amdsmi_get_cpu_core_energy()` - Per-core energy (HSMP)
- `amdsmi_get_cpu_socket_power()` - Socket power (HSMP)
- `amdsmi_get_cpu_socket_power_cap()` - Power cap (HSMP)
- `amdsmi_get_cpu_pwr_svi_telemetry_all_rails()` - SVI telemetry
- `amdsmi_get_cpu_current_io_bandwidth()` - I/O bandwidth
- `amdsmi_get_cpu_current_xgmi_bw()` - XGMI bandwidth
- `amdsmi_set_cpu_*()` - Write operations

**Recommendation:** Medium priority for Tier 3B
- CPU HSMP operations can have variable latency
- If CPU monitoring is slow, add individual controls for CPU power/energy APIs

---

## Summary: Recommended Tier 3B Additions

### High Priority (Add Next)
1. **`amdsmi_get_pcie_link_metrics()`** - PCIe link counters, may be slow
2. **`amdsmi_get_violation_status()`** - Throttling status, may iterate blocks
3. **`amdsmi_get_cpu_core_energy()`** - Per-core HSMP energy reads

### Medium Priority
4. **`amdsmi_get_cpu_socket_power()`** - HSMP socket power
5. **`amdsmi_get_gpu_volt_metric()`** - Voltage sensor reads
6. **`amdsmi_topo_get_link_metrics()`** - XGMI link metrics
7. **`amdsmi_get_gpu_xgmi_error_status()`** - XGMI error checking

### Low Priority (Only if Needed)
8. Event notification APIs (if used in hot paths)
9. Additional CPU HSMP APIs (SVI telemetry, bandwidth)

---

## Implementation Template for Tier 3B

When adding a Tier 3B API control:

1. **Add to RdcMetricControl.h:**
   ```cpp
   std::atomic<bool> enable_amdsmi_<api_name>{true};
   ```

2. **Add to RdcMetricControl.cc:**
   - Environment variable parsing
   - Config file parsing
   - Enable/disable methods

3. **Wrap API call in source:**
   ```cpp
   if (!control.enable_amdsmi_<api_name>) {
     // Return appropriate zero/default value
     return RDC_ST_OK;
   }
   // Call actual AMD-SMI API
   ```

4. **Update configuration file example**

---

## Performance Testing Recommendations

Before adding Tier 3B controls:
1. Profile RDC with all Tier 3A controls enabled
2. Identify remaining bottlenecks
3. Add controls only for APIs showing >1ms average latency
4. Test on multiple hardware platforms (MI300X, MI250X, etc.)

---

## Notes

- This document is for future reference only
- Current implementation (Tier 3A) covers the highest-impact APIs
- Tier 3B should be added incrementally based on real-world performance data
- Not all APIs need individual controls - category controls are sufficient for most cases
