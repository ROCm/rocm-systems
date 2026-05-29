# AMD SMI / ROCm-SMI Hardware Compatibility

> Source: (mirror of the internal Confluence "Hardware compatibility" page).

## Legend & Notes


### Hardware compatibility: Legend

- SUPPORTED = API working with correct output (or intentionally skipped for risky operations such as GPU reset, partition changes, or driver reload)
- UNSUPPORTED = API exists but returns an error on this hardware/driver
- API DOES NOT EXIST = amdsmi module has no such attribute on this build
- Default filter on "Unsupported" features (select TBD and -Empty- to get a complete list of all features)
- Filter items to be shown are or'd when selected (old TableFilter function)

### Notes on the table

- Tables apply to Linux BM and Linux Guest
- BM = Bare Metal
- Guest columns
  - 1VF - available in 1VF
  - MVF - available in MVF
- ROCm-SMI(Dep) - ROCm-SMI Deprecation is an ongoing effort
- * = prelimiary

### Automated API Test Script

- The API status data can be reproduced on any host using the automated test script

- test_amdsmi_all_apis.py  (v1.1.0)
- Covers 250 known amdsmi Python APIs across all categories (GPU, CPU/HSMP, topology, RAS, partitioning, etc.)
- Auto-detects a Python interpreter that has amdsmi installed and re-executes itself with it
- Classifies each API as supported, unsupported, skipped, or api does not exist
- Write/setter APIs are gated behind root — run with sudo for full coverage
- Safe by default: skips risky operations and reverts any mutations made during testing

### Quick usage

```bash
python3 test_amdsmi_all_apis.py                   # read-only sweep
sudo python3 test_amdsmi_all_apis.py              # include write APIs
sudo python3 test_amdsmi_all_apis.py --all-gpus   # also sweep all GPUs
```
- Produces a timestamped markdown report. See the script header for full CLI options (--gpu-index, --output, --skip-writes, --no-revert).

## AMDSMI Linux Hardware Compatibility

| Category | CLI | API | MI450 BM* | MI450 PF* | MI455 Guest [mvf or 1vf]* | MI300x BM | MI300X-O PT | MI300X Guest[VF] | MI300X-O Guest [mvf or 1vf] | MI300A BM | MI300A Guest | MI210BM | MI200 Guest | MI350X BM | NV4x BM | NV4x Guest | NV32 BM | NV32 Guest | NV31 BM | NV31 Guest | NV21 BM | NV21 Guest |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Library Init |  | amdsmi_init | supported |  | supported | supported |  | supported | supported | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Library Init |  | amdsmi_shut_down | supported |  | supported | supported |  | supported | supported | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Device |  | amdsmi_get_processor_type | supported |  | supported | supported |  | supported | supported | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Device | amd-smi version --gpu_version | amdsmi_get_processor_handles | supported |  | supported | supported |  | supported | supported | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Device |  | amdsmi_get_socket_handles | supported |  | supported | supported |  | supported | supported | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Device |  | amdsmi_get_socket_info | supported |  | supported | supported |  | supported | supported | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Device | amd-smi list | amdsmi_get_gpu_device_bdf | supported |  | supported | supported |  | supported | supported | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Device | amd-smi list | amdsmi_get_gpu_device_uuid | supported |  | supported | supported |  | supported | Unsupported | supported | APU does not have FRU EEPROM<br>AMD SMI fields currently uses serial (not unique_id) + other fields | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Device |  | amdsmi_get_processor_handle_from_bdf | Unsupported |  | supported | supported |  | supported | Unsupported | supported | Unsupported | supported |  | supported | supported | Unsupported | supported | Unsupported | supported |  |  |  |
| Info | amd-smi version --gpu_version or dkms status | amdsmi_get_gpu_driver_version | API does not exist |  | API does not exist | API does not exist |  | API does not exist | Unsupported | API does not exist | Unsupported | API does not exist |  | API does not exist | API does not exist | Unsupported | API does not exist | Unsupported | API does not exist |  |  |  |
| Static | amd-smi static --gpu [] --asic | amdsmi_get_gpu_asic_info | supported |  | supported | supported |  | supported | Unsupported | supported | Unsupported | supported |  | supported | supported | Unsupported | supported | Unsupported | supported |  |  |  |
| Static | amd-smi static --vram | amdsmi_get_gpu_vram_info | supported |  | supported | supported |  | supported | Unsupported | supported | Unsupported | supported |  | supported | supported |  | supported | Unsupported | supported |  |  |  |
| Static | amd-smi static --limit | amdsmi_get_power_cap_info | supported |  | supported | supported |  | supported | Unsupported | supported | Unsupported | supported |  | supported | supported | Unsupported | supported | Unsupported | supported |  |  |  |
| Static | amd-smi monitor --pcie | amdsmi_get_pcie_info | Unsupported |  | supported | supported |  | supported | pcie_link_speed pcie_link_width supported with PMFW 85.99 and amdgpu 1744392 | supported | pcie_link_speed pcie_link_width supported with PMFW 85.99 and amdgpu 1744392 | supported |  | supported | supported | Unsupported | supported | Unsupported | supported |  |  |  |
| Info | amd-smi static --vbios | amdsmi_get_gpu_vbios_info | supported |  | supported | supported |  | supported | Unsupported | supported | Unsupported | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Info | amd-smi firmware | amdsmi_get_fw_info | supported | SDMA0,1 only | supported | supported | SDMA0,1 only | supported | Unsupported | supported | Unsupported | supported |  | supported | supported | Unsupported | supported | Unsupported | supported |  |  |  |
| Monitoring | amd-smi metric --usage | amdsmi_get_gpu_activity | Unsupported |  | supported | supported |  | supported | Unsupported | supported | Unsupported | supported |  | supported | supported | Unsupported | supported | Unsupported | supported |  | supported<br>NV/MI100/MI200: use MM_ACTIVITY/VCN_ACTIVITY<br>MI300+: use VCN_BUSY & JPEG_BUSY |  |
| Monitoring | amd-smi monitor --vram-usage | amdsmi_get_gpu_vram_usage | Unsupported |  | supported | supported |  | supported |  | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Monitoring | amd-smi monitor -p | amdsmi_get_power_info | Unsupported | instantaneous | supported | supported | instantaneous | supported | Unsupported | supported | Unsupported | supported |  | supported | supported | 1VF<br>average | supported | 1VF<br>average | supported | 1VF<br>average | average | 1VF<br>average |
|  | amd-smi monitor -w 1 -W 10 -p |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  | amd-smi metric --power |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Monitoring |  | amdsmi_get_clock_info | Unsupported | S: indicates DeepSleep | supported | supported | S: indicates DeepSleep | supported | Unsupported | supported | Unsupported | supported |  | supported | supported | Unsupported | supported | Unsupported | supported |  |  |  |
| Monitoring |  | amdsmi_get_pcie_link_status | API does not exist |  | API does not exist | API does not exist |  | API does not exist | Unsupported | API does not exist | Unsupported | API does not exist |  | API does not exist | API does not exist | Unsupported | API does not exist | Unsupported | API does not exist |  |  |  |
| Monitoring |  | amdsmi_get_pcie_link_caps | API does not exist |  | API does not exist | API does not exist |  | API does not exist | api does not exist yet | API does not exist |  | API does not exist |  | API does not exist | API does not exist |  | API does not exist |  | API does not exist |  |  |  |
| Monitoring |  | amdsmi_get_gpu_bad_page_info | supported |  | unsupported | supported |  | supported | Unsupported | unsupported | Unsupported | supported |  | supported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported |  |  |  |
| Power Management |  | amdsmi_get_gpu_target_frequency_range | API does not exist |  | API does not exist | API does not exist |  | API does not exist | Unsupported | API does not exist | Unsupported | API does not exist |  | API does not exist | API does not exist | Unsupported | API does not exist | Unsupported | API does not exist |  |  |  |
| Process |  | amdsmi_get_gpu_process_list | supported |  | supported | supported |  | supported |  | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Process |  | amdsmi_get_gpu_process_info | API does not exist |  | API does not exist | API does not exist |  | API does not exist |  | API does not exist |  | API does not exist |  | API does not exist | API does not exist |  | API does not exist |  | API does not exist |  |  |  |
| ECC Error |  | amdsmi_get_gpu_ecc_error_count | API does not exist |  | API does not exist | API does not exist |  | API does not exist | Unsupported | API does not exist | Unsupported | API does not exist |  | API does not exist | API does not exist | Unsupported | API does not exist | Unsupported | API does not exist |  |  |  |
| Info |  | amdsmi_get_gpu_board_info | supported |  | supported | supported |  | supported | Unsupported | supported | Unsupported | supported |  | supported | supported | Unsupported | supported | Unsupported | supported |  |  |  |
| RAS |  | amdsmi_get_gpu_ras_block_features_enabled | Unsupported | GFX/SDMA/MMHUB | unsupported | supported | GFX/SDMA/MMHUB | supported | Unsupported | supported | Unsupported | supported |  | supported | unsupported | Unsupported | unsupported | Unsupported | unsupported |  |  |  |
| Functions |  | amdsmi_open_supported_func_iterator | API does not exist |  | API does not exist | API does not exist |  | API does not exist | Unsupported | API does not exist | Unsupported | API does not exist |  | API does not exist | API does not exist | Unsupported | API does not exist | Unsupported | API does not exist |  |  |  |
| Functions |  | amdsmi_open_supported_variant_iterator | API does not exist |  | API does not exist | API does not exist |  | API does not exist | Unsupported | API does not exist | Unsupported | API does not exist |  | API does not exist | API does not exist | Unsupported | API does not exist | Unsupported | API does not exist |  |  |  |
| Functions |  | amdsmi_close_supported_func_iterator | API does not exist |  | API does not exist | API does not exist |  | API does not exist | Unsupported | API does not exist | Unsupported | API does not exist |  | API does not exist | API does not exist | Unsupported | API does not exist | Unsupported | API does not exist |  |  |  |
| Functions |  | amdsmi_next_func_iter | API does not exist |  | API does not exist | API does not exist |  | API does not exist | Unsupported | API does not exist | Unsupported | API does not exist |  | API does not exist | API does not exist | Unsupported | API does not exist | Unsupported | API does not exist |  |  |  |
| Functions |  | amdsmi_get_func_iter_value | API does not exist |  | API does not exist | API does not exist |  | API does not exist | Unsupported | API does not exist | Unsupported | API does not exist |  | API does not exist | API does not exist | Unsupported | API does not exist | Unsupported | API does not exist |  |  |  |
| Functions |  | amdsmi_set_gpu_pci_bandwidth | supported |  | unsupported | supported |  | supported | Unsupported | unsupported | Unsupported | unsupported |  | unsupported | supported | Unsupported | supported | Unsupported | supported |  |  |  |
| Functions | amd-smi set -g all --power-cap | amdsmi_set_power_cap | supported |  | unsupported | supported |  | unsupported | Unsupported | supported | Unsupported | supported |  | supported | supported | Unsupported | supported | Unsupported | supported |  |  |  |
|  | amd-smi reset -g all --power-cap |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Functions | sudo amd-smi reset --profile | amdsmi_set_gpu_power_profile | supported | Unsupported | unsupported | unsupported | Unsupported | unsupported | Unsupported | unsupported | Unsupported | unsupported |  | unsupported | supported | Unsupported | supported | Unsupported | supported |  |  |  |
| Functions |  | amdsmi_set_gpu_clk_range | supported |  | unsupported | unsupported |  | unsupported | Unsupported | unsupported | Unsupported | unsupported |  | unsupported | unsupported | Unsupported | supported | Unsupported | supported |  |  |  |
| Functions |  | amdsmi_set_gpu_od_clk_info | supported | Unsupported | supported | supported | Unsupported | supported | Unsupported | supported | Unsupported | supported |  | supported | unsupported | Unsupported | supported | Unsupported | supported |  |  |  |
| Functions |  | amdsmi_set_gpu_od_volt_info | supported | Unsupported | unsupported | unsupported | Unsupported | unsupported | Unsupported | unsupported | Unsupported | unsupported |  | unsupported | unsupported | Unsupported | unsupported | Unsupported | unsupported |  |  |  |
| Functions |  | amdsmi_set_gpu_perf_level_v1 | API does not exist |  | API does not exist | API does not exist |  | API does not exist | Unsupported | API does not exist | Unsupported | API does not exist |  | API does not exist | API does not exist | Unsupported | API does not exist | Unsupported | API does not exist |  |  |  |
| Functions |  | amdsmi_set_gpu_perf_level | supported |  | supported | supported |  | supported | Unsupported | supported | Unsupported | supported |  | supported | supported | Unsupported | supported | Unsupported | supported |  |  |  |
| Functions |  | amdsmi_set_xgmi_plpd | supported |  | unsupported | supported |  | supported |  | unsupported | Unsupported | supported | Unsupported | unsupported | unsupported |  | unsupported |  | supported | Unsupported | Unsupported | Unsupported |
| Functions |  | amdsmi_get_gpu_power_profile_presets | unsupported | Unsupported | unsupported | unsupported | Unsupported | unsupported | Unsupported | unsupported | Unsupported | unsupported |  | unsupported | unsupported | Unsupported | unsupported | Unsupported | unsupported |  |  |  |
| Functions | sudo amd-smi reset --gpureset | amdsmi_reset_gpu | supported | Individual GPU reset supported including XGMI configs<br>SWDEV-496777 | supported | supported | Individual GPU reset supported including XGMI configs<br>SWDEV-496777 | supported | Unsupported | supported | Unsupported | supported |  | supported | supported | Unsupported | supported | Unsupported | supported |  |  |  |
| Functions |  | amdsmi_set_gpu_perf_determinism_mode | supported |  | unsupported | supported |  | supported | Unsupported | supported | Unsupported | supported |  | unsupported | unsupported | Unsupported | unsupported | Unsupported | unsupported |  |  |  |
| Functions |  | amdsmi_set_gpu_fan_speed | supported | Unsupported | unsupported | unsupported | Unsupported | unsupported | Unsupported | unsupported | Unsupported | unsupported |  | unsupported | supported | Unsupported | unsupported | Unsupported | unsupported |  |  |  |
| Functions |  | amdsmi_reset_gpu_fan | supported | Unsupported | unsupported | unsupported | Unsupported | unsupported | Unsupported | unsupported | Unsupported | unsupported |  | unsupported | supported | Unsupported | supported | Unsupported | supported |  |  |  |
| Functions |  | amdsmi_set_clk_freq | supported | SCLK only | unsupported | supported | SCLK only | supported | 1VF | unsupported | 1VF | unsupported |  | unsupported | supported | Unsupported | unsupported | Unsupported | supported |  |  |  |
| Functions |  | amdsmi_set_gpu_overdrive_level_v1 | API does not exist | Unsupported | API does not exist | API does not exist | Unsupported | API does not exist | Unsupported | API does not exist | Unsupported | API does not exist |  | API does not exist | API does not exist | Unsupported | API does not exist | Unsupported | API does not exist |  |  |  |
| Functions |  | amdsmi_set_gpu_overdrive_level | supported | Unsupported | unsupported | unsupported | Unsupported | unsupported | Unsupported | unsupported | Unsupported | unsupported |  | unsupported | unsupported | Unsupported | unsupported | Unsupported | unsupported |  |  |  |
| State |  | amdsmi_get_gpu_fan_rpms | unsupported | Unsupported | unsupported | unsupported | Unsupported | unsupported | Unsupported | unsupported | Unsupported | unsupported |  | unsupported | supported | Unsupported | supported | Unsupported | supported |  |  |  |
| State |  | amdsmi_get_gpu_fan_speed | unsupported | Unsupported | unsupported | unsupported | Unsupported | unsupported | Unsupported | unsupported | Unsupported | unsupported |  | unsupported | supported | Unsupported | supported | Unsupported | supported |  |  |  |
| State |  | amdsmi_get_gpu_fan_speed_max | unsupported | Unsupported | unsupported | unsupported | Unsupported | unsupported | Unsupported | unsupported | Unsupported | unsupported |  | unsupported | supported | Unsupported | supported | Unsupported | supported |  |  |  |
| State |  | amdsmi_get_temp_metric | Unsupported | hotspot/memory | unsupported | Unsupported | hotspot/memory | unsupported | Unsupported | unsupported | Unsupported | supported |  | unsupported | supported | Unsupported | supported | Unsupported | supported |  |  |  |
| State |  | amdsmi_get_gpu_volt_metric | Unsupported | Unsupported | unsupported | Unsupported | Unsupported | unsupported | Unsupported | unsupported | Unsupported | supported |  | unsupported | supported | Unsupported | supported | Unsupported | supported |  |  |  |
| Performance |  | amdsmi_get_busy_percent | API does not exist |  | API does not exist | API does not exist |  | API does not exist | Unsupported | API does not exist | Unsupported | API does not exist |  | API does not exist | API does not exist | Unsupported | API does not exist | Unsupported | API does not exist |  |  |  |
| Performance |  | amdsmi_get_utilization_count | unsupported |  | unsupported | supported |  | supported | Unsupported | unsupported | Unsupported | unsupported |  | unsupported | unsupported | Unsupported | unsupported | Unsupported | unsupported |  |  |  |
| Performance | amd-smi metric --perf-level | amdsmi_get_gpu_perf_level | supported |  | supported | supported |  | supported | Unsupported | supported | Unsupported | supported |  | supported | supported | Unsupported | supported | Unsupported | supported |  |  |  |
| Performance |  | amdsmi_get_gpu_overdrive_level | unsupported | Unsupported | unsupported | unsupported | Unsupported | unsupported | Unsupported | unsupported | Unsupported | unsupported |  | unsupported | unsupported | Unsupported | unsupported | Unsupported | unsupported |  |  |  |
| Performance |  | amdsmi_get_clk_freq | supported |  | supported | supported |  | supported | Unsupported | supported | Unsupported | supported |  | supported | supported | Unsupported | supported | Unsupported | supported |  |  |  |
| Performance |  | amdsmi_get_gpu_od_volt_info | supported | Unsupported | supported | supported | Unsupported | supported | Unsupported | supported | Unsupported | supported |  | supported | unsupported | Unsupported | supported | Unsupported | supported |  |  |  |
| Performance | amd-smi monitor | amdsmi_get_gpu_metrics_info | Unsupported |  | supported | supported |  | supported | Unsupported | supported | Unsupported | supported |  | supported | supported | Unsupported | supported | Unsupported | supported |  |  |  |
| Performance |  | amdsmi_get_gpu_od_volt_curve_regions | unsupported | Unsupported | unsupported | unsupported | Unsupported | unsupported | Unsupported | unsupported | Unsupported | unsupported |  | unsupported | unsupported | Unsupported | unsupported | Unsupported | unsupported |  |  |  |
| Performance |  | amdsmi_get_gpu_power_profile_presets | unsupported | Unsupported | unsupported | unsupported | Unsupported | unsupported | Unsupported | unsupported | Unsupported | unsupported |  | unsupported | unsupported | Unsupported | unsupported | Unsupported | unsupported |  |  |  |
| Performance |  | amdsmi_gpu_counter_group_supported | unsupported |  | unsupported | unsupported |  | unsupported | supported | unsupported |  | unsupported |  | unsupported | unsupported |  | unsupported |  | unsupported |  |  |  |
| Performance |  | amdsmi_gpu_create_counter | unsupported |  | supported | supported |  | supported | supported | supported |  | supported |  | unsupported | unsupported |  | supported |  | unsupported |  |  |  |
| Performance |  | amdsmi_gpu_destroy_counter | unsupported |  | supported | supported |  | supported | supported | supported |  | supported |  | unsupported | unsupported |  | supported |  | unsupported |  |  |  |
| Performance |  | amdsmi_gpu_control_counter | unsupported |  | supported | supported |  | supported | supported | supported |  | supported |  | unsupported | unsupported |  | supported |  | unsupported |  |  |  |
| Performance |  | amdsmi_gpu_read_counter | unsupported |  | supported | supported |  | supported | supported | supported |  | supported |  | unsupported | unsupported |  | supported |  | unsupported |  |  |  |
| Performance |  | amdsmi_get_gpu_available_counters | Unsupported |  | unsupported | Unsupported |  | unsupported | supported | unsupported | Unsupported | supported |  | unsupported | unsupported | Unsupported | unsupported | Unsupported | unsupported |  |  |  |
| Error |  | amdsmi_get_gpu_ecc_count | Unsupported |  | unsupported | supported |  | supported | supported | unsupported | Unsupported | supported |  | supported | unsupported | Unsupported | unsupported | Unsupported | unsupported |  |  |  |
| Error |  | amdsmi_get_gpu_ecc_enabled | supported |  | unsupported | supported |  | supported | supported | supported | Unsupported | supported |  | supported | unsupported | Unsupported | unsupported | Unsupported | unsupported |  |  |  |
| Error |  | amdsmi_get_gpu_ecc_status | supported |  | unsupported | supported |  | supported | supported | supported | Unsupported | supported |  | supported | unsupported | Unsupported | unsupported | Unsupported | unsupported |  |  |  |
| Error |  | amdsmi_status_string | API does not exist |  | API does not exist | API does not exist |  | API does not exist | api does not exist yet | API does not exist | Unsupported | API does not exist |  | API does not exist | API does not exist | Unsupported | API does not exist | Unsupported | API does not exist |  |  |  |
| Info |  | amdsmi_get_gpu_compute_process_info | supported |  | supported | supported |  | supported | supported | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Info |  | amdsmi_get_gpu_compute_process_info_by_pid | supported |  | unsupported | unsupported |  | unsupported | supported | supported |  | unsupported |  | supported | supported |  | unsupported |  | supported |  |  |  |
| Info |  | amdsmi_get_gpu_compute_process_gpus | supported |  | unsupported | unsupported |  | unsupported | supported | unsupported |  | unsupported |  | supported | supported |  | unsupported |  | supported |  |  |  |
| Info |  | amdsmi_gpu_xgmi_error_status | unsupported |  | unsupported | unsupported |  | unsupported | Unsupported | unsupported | Unsupported | supported |  | unsupported | unsupported | Unsupported | unsupported | Unsupported | unsupported |  |  |  |
| Info |  | amdsmi_reset_gpu_xgmi_error | supported |  | unsupported | unsupported |  | unsupported | Unsupported | unsupported | Unsupported | supported |  | API does not exist | unsupported | Unsupported | unsupported | Unsupported | unsupported |  |  |  |
| PCIE |  | amdsmi_get_gpu_pci_id | API does not exist |  | API does not exist | API does not exist |  | API does not exist | Unsupported | API does not exist | Unsupported | API does not exist |  | API does not exist | API does not exist | Unsupported | API does not exist | Unsupported | API does not exist |  |  |  |
| PCIE |  | amdsmi_get_gpu_pci_bandwidth | Unsupported | Rx/Tx Not Supported | unsupported | supported | Rx/Tx Not Supported | unsupported | Unsupported | unsupported | Unsupported | supported |  | unsupported | supported | Unsupported | supported | Unsupported | supported |  | Supported |  |
| PCIE | amd-smi metric -P | amdsmi_get_gpu_pci_throughput | Unsupported | Unsupported<br>Rx/Tx/max packet size Not Supported | unsupported | Unsupported | Unsupported<br>Rx/Tx/max packet size Not Supported | unsupported | Unsupported<br>Rx/Tx/max packet size Not Supported | unsupported | Unsupported<br>Rx/Tx/max packet size Not Supported | supported |  | unsupported | unsupported | Unsupported<br>Rx/Tx/max packet size Not Supported | unsupported | Unsupported<br>Rx/Tx/max packet size Not Supported | unsupported |  | Supported<br>amdsmitstReadWrite.TestPciReadWrite |  |
| PCIE |  | amdsmi_get_gpu_pci_replay_counter | unsupported |  | unsupported | unsupported |  | unsupported | Unsupported | supported | Unsupported | unsupported |  | unsupported | unsupported | Unsupported | unsupported | Unsupported | unsupported |  |  |  |
| PCIE |  | amdsmi_get_gpu_topo_numa_affinity | supported |  | supported | supported |  | supported | Unsupported | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Power |  | amdsmi_get_energy_count | Unsupported |  | supported | supported |  | supported | Unsupported | supported | Unsupported | supported |  | supported | supported | Unsupported | supported | Unsupported | supported |  |  |  |
| Memory |  | amdsmi_get_gpu_memory_total | supported |  | supported | supported |  | supported | supported | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Memory |  | amdsmi_get_gpu_memory_usage | supported |  | supported | supported |  | supported | supported | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Memory |  | amdsmi_get_gpu_memory_busy_percent | API does not exist |  | API does not exist | API does not exist |  | API does not exist | Unsupported | API does not exist | Unsupported | API does not exist |  | API does not exist | API does not exist | Unsupported | API does not exist | Unsupported | API does not exist |  |  |  |
| Memory |  | amdsmi_get_gpu_memory_reserved_pages | supported |  | unsupported | supported |  | supported | supported | unsupported |  | supported |  | supported | unsupported |  | unsupported |  | unsupported |  |  |  |
| Events |  | AmdSmiEventReader | supported |  | supported | supported |  | supported | Unsupported | supported | Unsupported | supported |  | supported | supported | Unsupported | supported | Unsupported | supported |  |  |  |
| Info |  | amdsmi_get_gpu_vendor_name | supported |  | supported | supported |  | supported | Unsupported | supported | Unsupported | supported |  | supported | supported | Unsupported | supported | Unsupported | supported |  |  |  |
| Info |  | amdsmi_get_gpu_id | supported |  | supported | supported |  | supported | Unsupported | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Info |  | amdsmi_get_gpu_vram_vendor | Unsupported | Unsupported | unsupported | supported | Unsupported | unsupported | Unsupported | unsupported | Unsupported | supported |  | supported | supported | Unsupported | supported | Unsupported | supported |  |  |  |
| Info |  | amdsmi_get_gpu_drm_render_minor | API does not exist |  | API does not exist | API does not exist |  | API does not exist | api does not exist yet | API does not exist | Unsupported | API does not exist |  | API does not exist | API does not exist | Unsupported | API does not exist | Unsupported | API does not exist |  |  |  |
| Info | amd-smi static --asic | amdsmi_get_gpu_subsystem_id | supported |  | supported | supported |  | supported | Unsupported | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Info |  | amdsmi_get_gpu_subsystem_name | supported |  | supported | supported |  | supported | Unsupported | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Info |  | amdsmi_get_version | API does not exist |  | API does not exist | API does not exist |  | API does not exist | api does not exist yet | API does not exist | Unsupported | API does not exist |  | API does not exist | API does not exist | Unsupported | API does not exist | Unsupported | API does not exist |  |  |  |
| Info |  | amdsmi_get_version_str | API does not exist |  | API does not exist | API does not exist |  | API does not exist | api does not exist yet | API does not exist | Unsupported | API does not exist |  | API does not exist | API does not exist | Unsupported | API does not exist | Unsupported | API does not exist |  |  |  |
| Topology |  | amdsmi_topo_get_numa_node_number | supported |  | supported | supported |  | supported | supported | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Topology |  | amdsmi_topo_get_link_weight | supported |  | supported | supported |  | supported | supported | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Topology | amd-smi topology --numa-bw | amdsmi_get_minmax_bandwidth | API does not exist |  | API does not exist | API does not exist |  | API does not exist | Unsupported | API does not exist | Unsupported | API does not exist |  | API does not exist | API does not exist | Unsupported | API does not exist | Unsupported | API does not exist |  |  |  |
| Topology |  | amdsmi_topo_get_link_type | supported |  | supported | supported |  | supported | supported | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Topology |  | amdsmi_is_P2P_accessible | supported |  | supported | supported |  | supported | Unsupported | supported | Unsupported | supported |  | supported | supported | Unsupported | supported | Unsupported | supported |  |  |  |
| Topology |  | amdsmi_get_xgmi_info | supported |  | supported | supported |  | supported | Unsupported | supported | Unsupported | supported |  | supported | supported | Unsupported | supported | Unsupported | supported |  |  |  |
| Topology |  | amdsmi_get_xgmi_plpd | API does not exist |  | unsupported | supported |  | supported |  | unsupported | Unsupported | API does not exist | Unsupported | unsupported | unsupported |  | unsupported |  | unsupported | Unsupported | Unsupported | Unsupported |
| NPS |  | amdmi_dev_nps_mode_get | API does not exist | TBD | API does not exist | API does not exist | TBD | API does not exist | TBD | API does not exist | TBD | API does not exist |  | API does not exist | API does not exist | TBD | API does not exist | TBD | API does not exist |  |  |  |
| NPS |  | amdsmi_dev_nps_mode_set | API does not exist | TBD | API does not exist | API does not exist | TBD | API does not exist | TBD | API does not exist | TBD | API does not exist |  | API does not exist | API does not exist | TBD | API does not exist | TBD | API does not exist |  |  |  |
| NPS |  | amdsmi_dev_nps_mode_reset | API does not exist | TBD | API does not exist | API does not exist | TBD | API does not exist | TBD | API does not exist | TBD | API does not exist |  | API does not exist | API does not exist | TBD | API does not exist | TBD | API does not exist |  |  |  |
| AINIC |  | amdsmi_get_nic_driver_info |  |  |  |  |  |  |  |  |  |  |  |  |  |  | API does not exist |  | API does not exist |  |  |  |
| AINIC |  | amdsmi_get_nic_asic_info |  |  |  |  |  |  |  |  |  |  |  |  |  |  | API does not exist |  | API does not exist |  |  |  |
| AINIC |  | amdsmi_get_nic_bus_info |  |  |  |  |  |  |  |  |  |  |  |  |  |  | API does not exist |  | API does not exist |  |  |  |
| AINIC |  | amdsmi_get_nic_numa_info |  |  |  |  |  |  |  |  |  |  |  |  |  |  | API does not exist |  | API does not exist |  |  |  |
| AINIC |  | amdsmi_get_nic_port_info |  |  |  |  |  |  |  |  |  |  |  |  |  |  | API does not exist |  | API does not exist |  |  |  |
| AINIC |  | amdsmi_get_nic_rdma_dev_info |  |  |  |  |  |  |  |  |  |  |  |  |  |  | API does not exist |  | API does not exist |  |  |  |
| AINIC |  | amdsmi_get_nic_rdma_port_statistics |  |  |  |  |  |  |  |  |  |  |  |  |  |  | API does not exist |  | API does not exist |  |  |  |
| Processor/Device Discovery | amd-smi list | amdsmi_get_processor_handles_by_type |  |  | unsupported | supported |  | supported |  | unsupported |  | API does not exist |  | API does not exist | unsupported |  | unsupported |  | unsupported |  |  |  |
| Processor/Device Discovery | amd-smi list | amdsmi_get_processor_count_from_handles |  |  | supported | supported |  | supported |  | supported |  | API does not exist |  | API does not exist | supported |  | supported |  | supported |  |  |  |
| Processor/Device Discovery | amd-smi list | amdsmi_get_processor_info |  |  | supported | supported |  | supported |  | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Processor/Device Discovery | amd-smi node | amdsmi_get_node_handle |  |  | unsupported | API does not exist |  | API does not exist |  | API does not exist |  | API does not exist |  | API does not exist | unsupported |  | unsupported |  | unsupported |  |  |  |
| Processor/Device Discovery | amd-smi list | amdsmi_get_gpu_bdf_id |  |  | supported | supported |  | supported |  | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Processor/Device Discovery | amd-smi list -e | amdsmi_get_gpu_enumeration_info |  |  | supported | supported |  | supported |  | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Processor/Device Discovery | amd-smi list | amdsmi_get_gpu_kfd_info |  |  | supported | supported |  | supported |  | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Driver & Version info | amd-smi version or amd-smi static -d | amdsmi_get_gpu_driver_info |  |  | supported | supported |  | supported |  | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Driver & Version info | amd-smi version | amdsmi_get_lib_version |  |  | supported | supported |  | supported |  | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Driver & Version info | amd-smi version | amdsmi_get_rocm_version |  |  | supported | supported |  | supported |  | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Driver & Version info | Internal - error handling | amdsmi_status_code_to_string |  |  | unsupported | unsupported |  | unsupported |  | unsupported |  | unsupported |  | unsupported | unsupported |  | unsupported |  | unsupported |  |  |  |
| GPU Hardware info | amd-smi static -c | amdsmi_get_gpu_cache_info |  |  | unsupported | supported |  | supported |  | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| GPU Hardware info | amd-smi static -a | amdsmi_get_gpu_revision |  |  | supported | supported |  | supported |  | supported |  | supported |  | API does not exist | supported |  | supported |  | supported |  |  |  |
| Firmware info | amd-smi firmware | amdsmi_get_npm_info |  |  | unsupported | API does not exist |  | API does not exist |  | API does not exist |  | API does not exist |  | API does not exist | unsupported |  | unsupported |  | unsupported |  |  |  |
| Power management | amd-smi static -l | amdsmi_get_supported_power_cap |  |  | supported | supported |  | supported |  | supported |  | API does not exist |  | API does not exist | supported |  | supported |  | supported |  |  |  |
| Power management | amd-smi static | amdsmi_is_gpu_power_management_enabled |  |  | supported | supported |  | supported |  | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Power management | amd-smi static | amdsmi_get_violation_status |  |  | supported | supported |  | supported |  | supported |  | unsupported |  | unsupported | unsupported |  | unsupported |  | unsupported |  |  |  |
| Power management | amd-smi reset (driver reload) | amdsmi_gpu_driver_reload |  |  | supported | supported |  | supported |  | supported |  | supported |  | API does not exist | supported |  | supported |  | supported |  |  |  |
| PCIe info | amd-smi metric -P | amdsmi_get_gpu_ptl_formats |  |  | unsupported | unsupported |  | unsupported |  | unsupported |  | API does not exist |  | API does not exist | unsupported |  | unsupported |  | unsupported |  |  |  |
| PCIe info | amd-smi metric -P | amdsmi_get_gpu_ptl_state |  |  | unsupported | unsupported |  | unsupported |  | unsupported |  | API does not exist |  | API does not exist | unsupported |  | unsupported |  | unsupported |  |  |  |
| PCIe info | amd-smi set (PTL formats) | amdsmi_set_gpu_ptl_formats |  |  | supported | supported |  | supported |  | supported |  | API does not exist |  | API does not exist | supported |  | supported |  | supported |  |  |  |
| PCIe info | amd-smi set (PTL state) | amdsmi_set_gpu_ptl_state |  |  | supported | supported |  | supported |  | supported |  | API does not exist |  | API does not exist | supported |  | supported |  | supported |  |  |  |
| Activity & Utilization | amd-smi metric -u | amdsmi_get_gpu_busy_percent |  |  | supported | supported |  | supported |  | supported |  | API does not exist |  | API does not exist | supported |  | supported |  | supported |  |  |  |
| Activity & Utilization | amd-smi metric | amdsmi_get_gpu_xcd_counter |  |  | supported | supported |  | supported |  | supported |  | supported |  | unsupported | supported |  | supported |  | supported |  |  |  |
| Activity & Utilization | amd-smi reset (clean local data) | amdsmi_clean_gpu_local_data |  |  | supported | supported |  | supported |  | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Clock & Frequency | amd-smi set (clock limit) | amdsmi_set_gpu_clk_limit |  |  | unsupported | ERROR |  | ERROR |  | unsupported |  | unsupported |  | unsupported | unsupported |  | unsupported |  | unsupported |  |  |  |
| Clock & Frequency | amd-smi static -C | amdsmi_get_gpu_od_clk_info |  |  | API does not exist | API does not exist |  | API does not exist |  | API does not exist |  | API does not exist |  | API does not exist | API does not exist |  | API does not exist |  | API does not exist |  |  |  |
| Clock & Frequency | amd-smi static | amdsmi_get_dfc_ctrl |  |  | unsupported | API does not exist |  | API does not exist |  | API does not exist |  | API does not exist |  | API does not exist | unsupported |  | unsupported |  | unsupported |  |  |  |
| Clock & Frequency | amd-smi set (DFC control) | amdsmi_set_dfc_ctrl |  |  | supported | API does not exist |  | API does not exist |  | API does not exist |  | API does not exist |  | API does not exist | supported |  | supported |  | supported |  |  |  |
| Clock & Frequency | amd-smi metric | amdsmi_get_soc_pstate |  |  | unsupported | supported |  | supported |  | unsupported |  | API does not exist |  | API does not exist | unsupported |  | unsupported |  | unsupported |  |  |  |
| Clock & Frequency | amd-smi set (SOC P-state) | amdsmi_set_soc_pstate |  |  | unsupported | supported |  | supported |  | unsupported |  | unsupported |  | unsupported | unsupported |  | unsupported |  | unsupported |  |  |  |
| Performance level & Overdrive | amd-smi metric -o | amdsmi_get_gpu_mem_overdrive_level |  |  | unsupported | unsupported |  | unsupported |  | unsupported |  | unsupported |  | unsupported | unsupported |  | unsupported |  | unsupported |  |  |  |
| Temperature & Voltage & Metrics | amd-smi metric | amdsmi_get_gpu_metrics_header_info |  |  | supported | supported |  | supported |  | supported |  | supported |  | unsupported | supported |  | supported |  | supported |  |  |  |
| Temperature & Voltage & Metrics | amd-smi metric | amdsmi_get_gpu_pm_metrics_info |  |  | unsupported | unsupported |  | unsupported |  | unsupported |  | unsupported |  | unsupported | unsupported |  | unsupported |  | unsupported |  |  |  |
| Temperature & Voltage & Metrics | amd-smi partition | amdsmi_get_gpu_partition_metrics_info |  |  | supported | supported |  | unsupported |  | unsupported |  | API does not exist |  | API does not exist | unsupported |  | unsupported |  | unsupported |  |  |  |
| Temperature & Voltage & Metrics | amd-smi static | amdsmi_get_gpu_reg_table_info |  |  | unsupported | supported |  | supported |  | supported |  | unsupported |  | supported | unsupported |  | unsupported |  | unsupported |  |  |  |
| Memory | amd-smi bad-pages | amdsmi_get_gpu_bad_page_threshold |  |  | unsupported | supported |  | supported |  | supported |  | supported |  | supported | unsupported |  | unsupported |  | unsupported |  |  |  |
| RAS & ECC | amd-smi ras or amd-smi static -r | amdsmi_get_gpu_ras_feature_info |  |  | unsupported | supported |  | supported |  | supported |  | supported |  | supported | unsupported |  | unsupported |  | unsupported |  |  |  |
| RAS & ECC | amd-smi metric -e | amdsmi_get_gpu_total_ecc_count |  |  | supported | supported |  | supported |  | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| RAS & ECC | amd-smi ras --cper | amdsmi_get_gpu_cper_entries |  |  | unsupported | supported |  | supported |  | supported |  | unsupported |  | supported | unsupported |  | unsupported |  | unsupported |  |  |  |
| RAS & ECC | amd-smi ras | amdsmi_get_afids_from_cper |  |  | API does not exist | API does not exist |  | API does not exist |  | API does not exist |  | API does not exist |  | API does not exist | API does not exist |  | API does not exist |  | API does not exist |  |  |  |
| RAS & ECC | amd-smi ras | amdsmi_gpu_validate_ras_eeprom |  |  | unsupported | unsupported |  | unsupported |  | unsupported |  | API does not exist |  | API does not exist | unsupported |  | unsupported |  | unsupported |  |  |  |
| Process Management | amd-smi process | amdsmi_get_gpu_process_isolation |  |  | supported | supported |  | supported |  | supported |  | API does not exist |  | API does not exist | supported |  | supported |  | supported |  |  |  |
| Process Management | amd-smi set (process isolation) | amdsmi_set_gpu_process_isolation |  |  | supported | supported |  | supported |  | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Topology & NUMA | amd-smi topology -a | amdsmi_topo_get_p2p_status |  |  | unsupported | supported |  | unsupported |  | supported |  | supported |  | supported | supported |  | unsupported |  | supported |  |  |  |
| Topology & NUMA | amd-smi topology -a | amdsmi_get_P2P_status |  |  | API does not exist | API does not exist |  | API does not exist |  | API does not exist |  | API does not exist |  | API does not exist | API does not exist |  | API does not exist |  | API does not exist |  |  |  |
| Topology & NUMA | amd-smi topology -b | amdsmi_get_minmax_bandwidth_between_processors |  |  | unsupported | unsupported |  | unsupported |  | supported |  | supported |  | supported | unsupported |  | unsupported |  | unsupported |  |  |  |
| Topology & NUMA | amd-smi topology | amdsmi_get_link_metrics |  |  | supported | supported |  | supported |  | supported |  | supported |  | API does not exist | supported |  | supported |  | supported |  |  |  |
| Topology & NUMA | amd-smi topology | amdsmi_get_link_topology_nearest |  |  | unsupported | supported |  | supported |  | unsupported |  | unsupported |  | unsupported | unsupported |  | unsupported |  | unsupported |  |  |  |
| XGMI | amd-smi xgmi -m | amdsmi_get_gpu_xgmi_link_status |  |  | unsupported | supported |  | supported |  | supported |  | supported |  | unsupported | unsupported |  | unsupported |  | unsupported |  |  |  |
| Partitioning | amd-smi partition -c | amdsmi_get_gpu_compute_partition |  |  | supported | supported |  | supported |  | supported |  | unsupported |  | supported | unsupported |  | unsupported |  | unsupported |  |  |  |
| Partitioning | amd-smi set -C <PARTITION> | amdsmi_set_gpu_compute_partition |  |  | supported | supported |  | supported |  | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Partitioning | amd-smi partition -m | amdsmi_get_gpu_memory_partition |  |  | unsupported | supported |  | supported |  | supported |  | unsupported |  | supported | unsupported |  | unsupported |  | unsupported |  |  |  |
| Partitioning | amd-smi set -M <PARTITION> | amdsmi_set_gpu_memory_partition |  |  | supported | supported |  | supported |  | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Partitioning | amd-smi set -M <MODE> | amdsmi_set_gpu_memory_partition_mode |  |  | supported | supported |  | supported |  | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Partitioning | amd-smi partition | amdsmi_get_gpu_memory_partition_config |  |  | unsupported | supported |  | supported |  | supported |  | unsupported |  | supported | unsupported |  | unsupported |  | unsupported |  |  |  |
| Partitioning | amd-smi partition -a | amdsmi_get_gpu_accelerator_partition_profile |  |  | supported | supported |  | supported |  | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Partitioning | amd-smi set (accelerator partition) | amdsmi_set_gpu_accelerator_partition_profile |  |  | supported | supported |  | supported |  | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Partitioning | amd-smi partition | amdsmi_get_gpu_accelerator_partition_profile_config |  |  | supported | supported |  | supported |  | supported |  | unsupported |  | supported | unsupported |  | unsupported |  | unsupported |  |  |  |
| Virtualization | amd-smi static | amdsmi_get_gpu_virtualization_mode |  |  | supported | supported |  | supported |  | supported |  | supported |  | supported | supported |  | supported |  | supported |  |  |  |
| Virtualization | amd-smi topology | amdsmi_get_cpu_affinity_with_scope |  |  | unsupported | supported |  | supported |  | unsupported |  | unsupported |  | unsupported | unsupported |  | unsupported |  | unsupported |  |  |  |
| Event notification | amd-smi event | amdsmi_init_gpu_event_notification |  |  | supported | supported |  | supported |  | supported |  | API does not exist |  | API does not exist | supported |  | supported |  | supported |  |  |  |
| Event notification | amd-smi event | amdsmi_set_gpu_event_notification_mask |  |  | supported | supported |  | supported |  | supported |  | API does not exist |  | API does not exist | supported |  | supported |  | supported |  |  |  |
| Event notification | amd-smi event | amdsmi_get_gpu_event_notification |  |  | unsupported | supported |  | supported |  | unsupported |  | API does not exist |  | API does not exist | unsupported |  | unsupported |  | unsupported |  |  |  |
| Event notification | amd-smi event | amdsmi_stop_gpu_event_notification |  |  | supported | supported |  | supported |  | supported |  | API does not exist |  | API does not exist | supported |  | supported |  | supported |  |  |  |

## ROCm-SMI Linux Hardware Compatibility

| Category | CLI | API | MI300X-O BM | MI300X-O | MI300X-O Guest | MI300A BM | MI300A Guest | MI200 BM | MI200 Guest | MI100 BM | MI100 Guest | NV4x BM | NV4x Guest | NV32 BM | NV32 Guest | NV31 BM | NV31 Guest | NV21 BM | NV21 Guest |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Display | rocm-smi --alldevices |  |  |  | Unverified |  | Unverified |  | Unverified |  | Unverified |  | Unverified | Yes | Unverified | Yes | Unverified |  | Unverified |
| Display | rocm-smi --showhw |  |  |  | Unsupported |  | Unsupported |  | 1VF |  | 1VF |  | Unsupported | Yes | Unsupported | Yes | Unsupported |  | 1VF |
| Display | rocm-smi -a, --showallinfo |  |  |  | Unsupported |  | Unsupported |  | 1VF |  | 1VF |  | Unsupported | Yes | Unsupported | Yes | Unsupported |  | 1VF |
| Display | rocm-smi -e [EVENT [EVENT ...]]; --showevents [EVENT [EVENT ...]] |  |  |  | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported | Yes | Unsupported |  | Unsupported |  | Unsupported |
| Topology | rocm-smi -i, --showed |  |  |  | Unsupported |  | Unsupported |  | 1VF |  | 1VF |  | Unsupported | Yes | Unsupported | Yes | Unsupported |  | 1VF |
| Topology | rocm-smi -v, --showvbios |  |  |  | Unsupported | Unsupported | Unsupported |  | 1VF |  | 1VF |  | Unsupported | Yes | Unsupported | Yes | Unsupported |  | 1VF |
| Topology | rocm-smi --showdriverversion |  |  |  | Unsupported |  | Unsupported |  | 1VF |  | 1VF |  | Unsupported | Yes | Unsupported | Yes | Unsupported |  | 1VF |
| Topology | rocm-smi --showfwinfo [BLOCK [BLOCK …]] |  | SDMA0,1 only | SDMA0,1 only | Unsupported | SDMA0,1 only | Unsupported | SDMA0,1 only | 1VF | SDMA0,1 only | 1VF | SDMA0,1 only | Unsupported | Yes | Unsupported | Yes | Unsupported | SDMA0,1 only | 1VF |
| Topology | rocm-smi --showmclkrange |  |  |  | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported | Yes | Unsupported | Yes | Unsupported |  | Unsupported |
| Topology | rocm-smi --showmemvendor |  |  | Unsupported | Unsupported | Unsupported | Unsupported |  | 1VF |  | 1VF |  | Unsupported | Yes | Unsupported | Yes | Unsupported |  | 1VF |
| Topology | rocm-smi --showsclkrange |  |  |  | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported | Yes | Unsupported | Yes | Unsupported |  | Unsupported |
| Topology | rocm-smi --showproductname |  |  |  | Unsupported |  | Unsupported |  | 1VF |  | 1VF |  | Unsupported | Yes | Unsupported | Yes | Unsupported |  | 1VF |
| Topology | rocm-smi --showserial |  |  |  | Unsupported | APU does not have FRU EEPROM | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported | Yes | Unsupported | Yes | Unsupported |  | Unsupported |
| Topology | rocm-smi --showuniqueid |  |  |  | Unsupported |  | Unsupported |  | 1VF |  | 1VF |  | 1VF | Yes | 1VF | Yes | 1VF |  | 1VF |
| Topology | rocm-smi --showvoltagerange |  | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Yes | Unsupported | Yes | Unsupported |  | Unsupported |
| Topology | rocm-smi --showbus |  |  |  | Unsupported |  | Unsupported |  | 1VF |  | 1VF |  | Unsupported | Yes | Unsupported | Yes | Unsupported |  | 1VF |
| Page | rocm-smi --showpagesinfo |  |  | Unsupported | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported | Yes | Unsupported | Yes | Unsupported |  | Unsupported |
| Page | rocm-smi --showpendingpages |  |  | Unsupported | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported | Yes | Unsupported | Yes | Unsupported |  | Unsupported |
| Page | rocm-smi --showretiredpages |  |  | Unsupported | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported | Yes | Unsupported | Yes | Unsupported |  | Unsupported |
| Page | rocm-smi --showunreservablepages |  |  | Unsupported | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported | Yes | Unsupported | Yes | Unsupported |  | Unsupported |
| Hardware | rocm-smi -f, --showfan |  | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | 1VF | Unsupported | 1VF | Unsupported | Unsupported | Yes | Unsupported | Yes | Unsupported |  | 1VF |
| Hardware | rocm-smi -P, --showpower | rsmi_dev_power_get | instantaneous | instantaneous | Unsupported | instantaneous | Unsupported | average | 1VF | average | 1VF | average | 1VF | Yes | 1VF | Yes | 1VF<br>average | average | 1VF<br>average |
| Hardware | rocm-smi -t, --showtemp |  | hotspot/memory | hotspot/memory | Unsupported |  | Unsupported | junction | 1VF |  | 1VF |  | Unsupported | Yes | Unsupported | Yes | Unsupported |  | 1VF |
| Hardware | rocm-smi --showtempgraph |  | hotspot | hotspot | Unsupported |  | Unsupported | junction | 1VF |  | 1VF |  | Unsupported | cannot test | Unsupported | cannot test | Unsupported |  | 1VF |
| Hardware | rocm-smi -u, --showuse |  |  |  | Unsupported |  | Unsupported |  | 1VF |  | 1VF |  | Unsupported | Yes | Unsupported | Yes | Unsupported |  | 1VF |
| Hardware | rocm-smi --showmemuse |  |  |  | Unsupported |  | Unsupported |  | 1VF |  | 1VF |  | Unsupported | Yes | Unsupported | Yes | Unsupported |  | 1VF |
| Hardware | rocm-smi --showvoltage |  | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported |  | 1VF |  | 1VF |  | Unsupported | Yes | Unsupported | Yes | Unsupported |  | 1VF |
| Software | rocm-smi -b, --showbw |  |  |  | Unsupported | Unsupported | Unsupported |  | Unsupported |  | Unsupported | Unsupported | Unsupported | Yes | Unsupported | Yes | Unsupported |  | Unsupported |
| Software | rocm-smi -c, --showclocks |  | S: indicates DeepSleep | S: indicates DeepSleep | PCIe not supported |  | PCIe not supported |  | PCIe not supported |  | PCIe not supported |  | PCIe not supported | Yes | PCIe not supported | Yes | PCIe not supported |  | PCIe not supported |
| Software | rocm-smi -g, --showgpuclocks |  | S: indicates DeepSleep | S: indicates DeepSleep | Unsupported |  | Unsupported |  | 1VF |  | 1VF |  | Unsupported | Yes | Unsupported | Yes | Unsupported |  | 1VF |
| Software | rocm-smi -l, --showprofile |  | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported |  | Unsupported |  | Unsupported | Yes | Unsupported | Yes | Unsupported |  | Unsupported |
| Software | rocm-smi -M, --showmaxpower | rsmi_dev_power_cap_get |  |  | Unsupported |  | Unsupported |  | 1VF |  | 1VF |  | Unsupported | Yes | Unsupported | Yes | Unsupported |  | 1VF |
| Software | rocm-smi -m, --showmemoverdrive |  | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Yes | Unsupported | Yes | Unsupported |  | Unsupported |
| Software | rocm-smi -o, --showoverdrive |  | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Yes | Unsupported | Yes | Unsupported |  | Unsupported |
| Software | rocm-smi -p, --showperflevel |  |  |  | Unsupported |  | Unsupported |  | 1VF |  | 1VF |  | Unsupported | Yes | Unsupported | Yes | Unsupported |  | 1VF |
| Software | rocm-smi -S, --showclkvolt |  | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Yes | Unsupported | Yes | Unsupported |  | Unsupported |
| Software | rocm-smi -s, --showclkfrq |  |  |  | PCIe not supported |  | PCIe not supported |  | PCIe not supported |  | PCIe not supported |  | PCIe not supported | Yes | PCIe not supported | Yes | PCIe not supported |  | PCIe not supported |
| Software | rocm-smi --showenergycounter |  |  |  | Unsupported |  | Unsupported |  | 1VF |  | 1VF |  | Unsupported | Yes | Unsupported | Yes | Unsupported |  | 1VF |
| Software | rocm-smi --showmeminfo TYPE [TYPE …] |  |  |  | Unsupported |  | Unsupported |  | 1VF |  | 1VF |  | Unsupported | Yes | Unsupported | Yes | Unsupported |  | 1VF |
| Software | rocm-smi --showpids |  |  |  | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported | CU OCCUPANCY Unsupported<br>SDMA Unsupported | Unsupported | Yes | Unsupported | Yes | Unsupported |  | Unsupported |
| Software | rocm-smi --showpidgpus [SHOWPIDGPUS [SHOWPIDGPUS …]] |  |  |  | Unsupported |  | Unsupported |  | 1VF |  | 1VF |  | PCIe not supported | Yes | PCIe not supported | Yes | PCIe not supported |  | 1VF |
| Software | rocm-smi --showreplaycount |  |  |  | Unsupported |  | Unsupported |  | 1VF |  | 1VF |  | Unsupported | Yes | Unsupported | Yes | Unsupported |  | 1VF |
| Software | rocm-smi --showrasinfo [SHOWRASINFO [SHOWRASINFO …]] |  | GFX/SDMA/MMHUB | GFX/SDMA/MMHUB | Unsupported | GFX/SDMA/MMHUB | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported | Yes | Unsupported | Yes | Unsupported |  | Unsupported |
| Software | rocm-smi --showvc |  | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Yes | Unsupported | Yes | Unsupported |  | Unsupported |
| Software | rocm-smi --showxgmierr |  |  |  | Unsupported |  | Unsupported |  | 1VF |  | 1VF |  | Unsupported | Yes | Unsupported | Yes | Unsupported |  | 1VF |
| Software | rocm-smi --showtopo |  |  | Unsupported | Unsupported |  | Unsupported |  | 1VF |  | 1VF |  | Unsupported | Yes | Unsupported | Yes | Unsupported |  | 1VF |
| Software | rocm-smi --showtopoweight |  |  | Unsupported | Unsupported |  | Unsupported |  | 1VF |  | 1VF |  | Unsupported | Yes | Unsupported | Yes | Unsupported |  | 1VF |
| Software | rocm-smi --showtopohops |  |  | Unsupported | Unsupported |  | Unsupported |  | 1VF |  | 1VF |  | Unsupported | Yes | Unsupported | Yes | Unsupported |  | 1VF |
| Software | rocm-smi --showtopotype |  |  | Unsupported | Unsupported |  | Unsupported |  | 1VF |  | 1VF |  | PCIe not supported | Yes | PCIe not supported | Yes | PCIe not supported |  | 1VF |
| Software | rocm-smi --showtoponuma |  |  | Unsupported | Unverified |  | Unverified |  | Unverified |  | Unverified |  | Unsupported | Yes | Unsupported | Yes | Unsupported |  | Unverified |
| Software | rocm-smi --showtopoaccess |  |  | Unsupported | Unsupported |  | Unsupported |  | 1VF |  | 1VF |  | Unsupported | Yes | Unsupported | Yes | Unsupported |  | 1VF |
| Software | rocm-smi --shownodesbw |  |  | Unsupported | Unverified |  | Unverified |  | Unverified |  | Unverified |  | Unsupported | Yes | Unsupported | Yes | Unsupported |  | Unverified |
| Set | rocm-smi --setsclk LEVEL [LEVEL …] |  |  |  | 1VF |  | 1VF |  | Unsupported |  | Unsupported |  | Unsupported | Write operation needed | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Set | rocm-smi --setmclk LEVEL [LEVEL …] |  | MAX only | MAX only | Unsupported | MAX only | Unsupported | Unsupported | Unsupported |  | Unsupported |  | Unsupported | Write operation needed | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Set | rocm-smi --setperfdeterminism SCLK |  | Support |  | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported | Write operation needed | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Set | rocm-smi --setpcie LEVEL [LEVEL …] |  | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported | Write operation needed | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Set | rocm-smi --setslevel SCLKLEVEL SCLK SVOLT |  |  |  | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported | Write operation needed | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Set | rocm-smi --setmlevel MCLKLEVEL MCLK MVOLT |  | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported | Write operation needed | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Set | rocm-smi --setvc POINT SCLK SVOLT |  | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported | Write operation needed | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Set | rocm-smi --setsrange MINMAX SCLK |  |  |  | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported | Write operation needed | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Set | rocm-smi --setmrange MINMAX SCLK |  | MAX only | MAX only | Unsupported | MAX only | Unsupported | Unsupported | Unsupported |  | Unsupported |  | Unsupported | Write operation needed | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Set | rocm-smi --setfan LEVEL |  | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Write operation needed | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Set | rocm-smi --setperflevel LEVEL |  | Auto and manual only | Auto and manual only | Unsupported | Auto and manual only | Unsupported | Auto, manual, high, low | Unsupported |  | Unsupported |  | Unsupported | Write operation needed | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Set | rocm-smi --setoverdrive % |  | Unsupported | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported |  | Write operation needed |  | Write operation needed |  |  |  |
| Set | rocm-smi --setmemoverdrive % |  | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported |  | Unsupported |  | Unsupported | Unsupported | Unsupported | Write operation needed | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Set | rocm-smi --setpoweroverdrive WATTS | rsmi_dev_power_cap_set |  |  | Unsupported |  | Unsupported |  | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Write operation needed | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Set | rocm-smi --setprofile SETPROFILE |  | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported |  | Unsupported |  | Unsupported | Write operation needed | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Set | rocm-smi --setclock LEVEL LEVEL |  | sclk only | sclk only | sclk, 1VF only | sclk only | sclk, 1VF only | Unsupported | Unsupported |  | Unsupported | PCIE not supported | Unsupported | Write operation needed | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Set | rocm-smi --rasenable BLOCK ERRTYPE |  |  |  | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported | Write operation needed | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Set | rocm-smi --rasdisable BLOCK ERRTYPE |  |  |  | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported | Write operation needed | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Set | rocm-smi --rasinject BLOCK |  |  |  | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported | Write operation needed | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Set | rocm-smi --setextremum min\|max sclk\|mclk CLK |  |  |  | 1VF only |  | 1VF only | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Write operation needed | Unsupported | Write operation needed | Unsupported | Unsupported | Unsupported |
| Reset | rocm-smi -r, --resetclocks |  |  |  | Unsupported |  | Unsupported | SCLK only | Unsupported |  | Unsupported |  | Unsupported | Write operation needed | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Reset | rocm-smi --resetfans |  | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Write operation needed | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Reset | rocm-smi --resetprofile |  | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported |  | Unsupported |  | Unsupported | Write operation needed | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Reset | rocm-smi --resetpoweroverdrive | rsmi_dev_power_cap_set | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Write operation needed | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Reset | rocm-smi --resetxgmierr |  |  |  | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported |  | Unsupported | Write operation needed | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Reset | rocm-smi --resetperfdeterminism |  |  |  | Unsupported |  | Unsupported |  | Unverified |  | Unverified |  | Unverified | Write operation needed | Unverified | Write operation needed | Unverified |  | Unverified |
| Reset | rocm-smi --resetgpu |  | Individual GPU reset supported including XGMI configs | Individual GPU reset supported including XGMI configs | Unsupported | Individual GPU reset supported including XGMI configs | Unsupported | If XGMI system: Mode1 Unsupported | Unsupported |  | Unsupported |  | Unsupported | Write operation needed | Unsupported | Write operation needed | Unsupported |  | Unsupported |
|  | amdsmi_reset_gpu |  | SWDEV-496777 | SWDEV-496777 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Output | rocm-smi --loglevel LEVEL |  |  |  | N/A |  | N/A |  | N/A |  | N/A |  | N/A | Write operation needed | N/A | Write operation needed | N/A |  | N/A |
| Output | rocm-smi --json |  |  |  | N/A |  | N/A |  | N/A |  | N/A |  | N/A | cannot test | N/A | Yes | N/A |  | N/A |
| Output | rocm-smi --csv |  |  |  | N/A |  | N/A |  | N/A |  | N/A |  | N/A | cannot test | N/A | Yes | N/A |  | N/A |
| Partition | rocm-smi --showcomputepartition |  |  |  | TBD |  | TBD | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Yes | Unsupported | Yes | Unsupported | Unsupported | Unsupported |
| Partition | rocm-smi --showmemorypartition |  | TBD | TBD | TBD | TBD | TBD | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Yes | Unsupported | Yes | Unsupported | Unsupported | Unsupported |
| Partition | rocm-smi --setcomputepartition |  |  |  | TBD |  | TBD | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Write operation needed | Unsupported | Write operation needed | Unsupported | Unsupported | Unsupported |
| NPS | rocm-smi --setmemorypartition |  | TBD | TBD | TBD | TBD | TBD | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Write operation needed | Unsupported | Write operation needed | Unsupported | Unsupported | Unsupported |
| NPS | rocm-smi --resetcomputepartition |  |  |  | TBD |  | TBD | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Write operation needed | Unsupported | Write operation needed | Unsupported | Unsupported | Unsupported |
| NPS | rocm-smi --resetmemorypartition |  | TBD | TBD | TBD | TBD | TBD | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Write operation needed | Unsupported | Write operation needed | Unsupported | Unsupported | Unsupported |

## New ROCm-SMI Table

| Category | CLI | API | MI300X-O BM | MI300X-O | MI300X-O Guest | MI300A BM | MI300A Guest | MI200 BM | MI210 BM | MI200 Guest | MI100 BM | MI100 Guest | NV4x BM | NV4x Guest | NV32 BM | NV32 Guest | NV31 BM | NV31 Guest | NV21 BM | NV21 Guest |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Initialization & Shutdown | Internal - called automatically on rocm-smi startup | rsmi_init |  |  | Unverified |  | Unverified |  | SUPPORTED | Unverified |  | Unverified |  | Unverified | SUPPORTED | Unverified | Yes | Unverified |  | Unverified |
| Initialization & Shutdown | Internal - called automatically on rocm-smi exit | rsmi_shut_down |  |  | Unsupported |  | Unsupported |  | SUPPORTED | 1VF |  | 1VF |  | Unsupported | SUPPORTED | Unsupported | Yes | Unsupported |  | 1VF |
| Device Discovery | rocm-smi (device enumeration) | rsmi_num_monitor_devices |  |  | Unsupported |  | Unsupported |  | SUPPORTED | 1VF |  | 1VF |  | Unsupported | SUPPORTED | Unsupported | Yes | Unsupported |  | 1VF |
| Device Discovery | rocm-smi --showcomputepartition | rsmi_dev_partition_id_get |  |  | Unsupported |  | Unsupported |  | SUPPORTED | Unsupported |  | Unsupported |  | Unsupported | SUPPORTED | Unsupported |  | Unsupported |  | Unsupported |
| Device Identification | rocm-smi --showid or rocm-smi -i | rsmi_dev_id_get |  |  | Unsupported |  | Unsupported |  | SUPPORTED | 1VF |  | 1VF |  | Unsupported | SUPPORTED | Unsupported | Yes | Unsupported |  | 1VF |
| Device Identification | rocm-smi --showproductname | rsmi_dev_revision_get |  |  | Unsupported | Unsupported | Unsupported |  | SUPPORTED | 1VF |  | 1VF |  | Unsupported | SUPPORTED | Unsupported | Yes | Unsupported |  | 1VF |
| Device Identification | rocm-smi --showproductname | rsmi_dev_sku_get |  |  | Unsupported |  | Unsupported |  | SUPPORTED | 1VF |  | 1VF |  | Unsupported | UNSUPPORTED | Unsupported | Yes | Unsupported |  | 1VF |
| Device Identification | rocm-smi --showproductname | rsmi_dev_vendor_id_get | SDMA0,1 only | SDMA0,1 only | Unsupported | SDMA0,1 only | Unsupported | SDMA0,1 only | SUPPORTED | 1VF | SDMA0,1 only | 1VF | SDMA0,1 only | Unsupported | SUPPORTED | Unsupported | Yes | Unsupported | SDMA0,1 only | 1VF |
| Device Identification | rocm-smi --showproductname | rsmi_dev_name_get |  |  | Unsupported |  | Unsupported |  | SUPPORTED | Unsupported |  | Unsupported |  | Unsupported | SUPPORTED | Unsupported | Yes | Unsupported |  | Unsupported |
| Device Identification | rocm-smi --showproductname | rsmi_dev_brand_get |  | Unsupported | Unsupported | Unsupported | Unsupported |  | SUPPORTED | 1VF |  | 1VF |  | Unsupported | SUPPORTED | Unsupported | Yes | Unsupported |  | 1VF |
| Device Identification | rocm-smi --showproductname | rsmi_dev_vendor_name_get |  |  | Unsupported |  | Unsupported |  | SUPPORTED | Unsupported |  | Unsupported |  | Unsupported | SUPPORTED | Unsupported | Yes | Unsupported |  | Unsupported |
| Device Identification | rocm-smi --showproductname | rsmi_dev_market_name_get |  |  | Unsupported |  | Unsupported |  | SUPPORTED | 1VF |  | 1VF |  | Unsupported | UNSUPPORTED | Unsupported | Yes | Unsupported |  | 1VF |
| Device Identification | rocm-smi --showmemvendor | rsmi_dev_vram_vendor_get |  |  | Unsupported | APU does not have FRU EEPROM | Unsupported |  | SUPPORTED | Unsupported |  | Unsupported |  | Unsupported | SUPPORTED | Unsupported | Yes | Unsupported |  | Unsupported |
| Device Identification | rocm-smi --showserial | rsmi_dev_serial_number_get |  |  | Unsupported |  | Unsupported |  | SUPPORTED | 1VF |  | 1VF |  | 1VF | UNSUPPORTED | 1VF | Yes | 1VF |  | 1VF |
| Device Identification | rocm-smi --showproductname | rsmi_dev_subsystem_id_get | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | SUPPORTED | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | SUPPORTED | Unsupported | Yes | Unsupported |  | Unsupported |
| Device Identification | rocm-smi --showproductname | rsmi_dev_subsystem_name_get |  |  | Unsupported |  | Unsupported |  | SUPPORTED | 1VF |  | 1VF |  | Unsupported | SUPPORTED | Unsupported | Yes | Unsupported |  | 1VF |
| Device Identification | rocm-smi --showid or rocm-smi -i | rsmi_dev_drm_render_minor_get |  | Unsupported | Unsupported |  | Unsupported |  | SUPPORTED | Unsupported |  | Unsupported |  | Unsupported | SUPPORTED | Unsupported | Yes | Unsupported |  | Unsupported |
| Device IdentificationDevice Identification | rocm-smi --showproductname | rsmi_dev_subsystem_vendor_id_get |  | Unsupported | Unsupported |  | Unsupported |  | SUPPORTED | Unsupported |  | Unsupported |  | Unsupported | SUPPORTED | Unsupported | Yes | Unsupported |  | Unsupported |
| Device Identification | rocm-smi --showuniqueid | rsmi_dev_unique_id_get |  | Unsupported | Unsupported |  | Unsupported |  | SUPPORTED | Unsupported |  | Unsupported |  | Unsupported | SUPPORTED | Unsupported | Yes | Unsupported |  | Unsupported |
| Device Identification | rocm-smi --showtopo | rsmi_dev_xgmi_physical_id_get |  | Unsupported | Unsupported |  | Unsupported |  | SUPPORTED | Unsupported |  | Unsupported |  | Unsupported | UNSUPPORTED | Unsupported | Yes | Unsupported |  | Unsupported |
| Device Identification | rocm-smi --showuniqueid | rsmi_dev_guid_get | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | SUPPORTED | 1VF | Unsupported | 1VF | Unsupported | Unsupported | SUPPORTED | Unsupported | Yes | Unsupported |  | 1VF |
| Device Identification | rocm-smi --showtopo | rsmi_dev_node_id_get | instantaneous | instantaneous | Unsupported | instantaneous | Unsupported | average | SUPPORTED | 1VF | average | 1VF | average | 1VF | SUPPORTED | 1VF | Yes | 1VF<br>average | average | 1VF<br>average |
| Device Identification | No direct CLI mapping | rsmi_dev_device_identifiers_get | hotspot/memory | hotspot/memory | Unsupported |  | Unsupported | junction | UNSUPPORTED | 1VF |  | 1VF |  | Unsupported | SUPPORTED | Unsupported | Yes | Unsupported |  | 1VF |
| Device Identification | rocm-smi --showproductname | rsmi_dev_target_graphics_version_get | hotspot | hotspot | Unsupported |  | Unsupported | junction | SUPPORTED | 1VF |  | 1VF |  | Unsupported | SUPPORTED | Unsupported | cannot test | Unsupported |  | 1VF |
| PCIe Operations | rocm-smi --showbw or rocm-smi -b | rsmi_dev_pci_bandwidth_get |  |  | Unsupported |  | Unsupported |  | SUPPORTED | 1VF |  | 1VF |  | Unsupported | SUPPORTED | Unsupported | Yes | Unsupported |  | 1VF |
| PCIe Operations | rocm-smi --setpcie LEVEL [LEVEL ...] | rsmi_dev_pci_bandwidth_set |  |  | Unsupported |  | Unsupported |  | UNSUPPORTED | 1VF |  | 1VF |  | Unsupported | SUPPORTED | Unsupported | Yes | Unsupported |  | 1VF |
| PCIe Operations | rocm-smi --showbus | rsmi_dev_pci_id_get | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported |  | SUPPORTED | 1VF |  | 1VF |  | Unsupported | SUPPORTED | Unsupported | Yes | Unsupported |  | 1VF |
| PCIe Operations | rocm-smi --showbw | rsmi_dev_pci_throughput_get |  |  | Unsupported | Unsupported | Unsupported |  | SUPPORTED | Unsupported |  | Unsupported | Unsupported | Unsupported | UNSUPPORTED | Unsupported | Yes | Unsupported |  | Unsupported |
| PCIe Operations | rocm-smi --showreplaycount | rsmi_dev_pci_replay_counter_get | S: indicates DeepSleep | S: indicates DeepSleep | PCIe not supported |  | PCIe not supported |  | UNSUPPORTED | PCIe not supported |  | PCIe not supported |  | PCIe not supported | UNSUPPORTED | PCIe not supported | Yes | PCIe not supported |  | PCIe not supported |
| PCIe Operations | rocm-smi --showtoponuma | rsmi_topo_numa_affinity_get | S: indicates DeepSleep | S: indicates DeepSleep | Unsupported |  | Unsupported |  | SUPPORTED | 1VF |  | 1VF |  | Unsupported | SUPPORTED | Unsupported | Yes | Unsupported |  | 1VF |
| Power Management | rocm-smi --showpower or rocm-smi -P | rsmi_dev_power_ave_get | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | SUPPORTED | Unsupported |  | Unsupported |  | Unsupported | SUPPORTED | Unsupported | Yes | Unsupported |  | Unsupported |
| Power Management | rocm-smi --showpower or rocm-smi -P | rsmi_dev_power_get |  |  | Unsupported |  | Unsupported |  | SUPPORTED | 1VF |  | 1VF |  | Unsupported | SUPPORTED | Unsupported | Yes | Unsupported |  | 1VF |
| Power Management | rocm-smi --showpower | rsmi_dev_current_socket_power_get | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | UNSUPPORTED | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | UNSUPPORTED | Unsupported | Yes | Unsupported |  | Unsupported |
| Power Management | rocm-smi --showenergycounter | rsmi_dev_energy_count_get | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | SUPPORTED | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | SUPPORTED | Unsupported | Yes | Unsupported |  | Unsupported |
| Power Management | rocm-smi --showmaxpower or rocm-smi -M | rsmi_dev_power_cap_get |  |  | Unsupported |  | Unsupported |  | SUPPORTED | 1VF |  | 1VF |  | Unsupported | SUPPORTED | Unsupported | Yes | Unsupported |  | 1VF |
| Power Management | rocm-smi --setpoweroverdrive WATTS | rsmi_dev_power_cap_set | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | SUPPORTED | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | SUPPORTED | Unsupported | Yes | Unsupported |  | Unsupported |
| Power Management | rocm-smi --showmaxpower | rsmi_dev_power_cap_default_get |  |  | PCIe not supported |  | PCIe not supported |  | SUPPORTED | PCIe not supported |  | PCIe not supported |  | PCIe not supported | SUPPORTED | PCIe not supported | Yes | PCIe not supported |  | PCIe not supported |
| Power Management | rocm-smi --showmaxpower | rsmi_dev_power_cap_range_get |  |  | Unsupported |  | Unsupported |  | SUPPORTED | 1VF |  | 1VF |  | Unsupported | SUPPORTED | Unsupported | Yes | Unsupported |  | 1VF |
| Power Management | rocm-smi --showprofile or rocm-smi -l | rsmi_dev_power_profile_presets_get |  |  | Unsupported |  | Unsupported |  | UNSUPPORTED | 1VF |  | 1VF |  | Unsupported | SUPPORTED | Unsupported | Yes | Unsupported |  | 1VF |
| Power Management | rocm-smi --setprofile PROFILE | rsmi_dev_power_profile_set |  |  | Unsupported |  | Unsupported |  | UNSUPPORTED | Unsupported |  | Unsupported | CU OCCUPANCY Unsupported<br>SDMA Unsupported | Unsupported | UNSUPPORTED | UNSUPPORTED | Yes | Unsupported |  | Unsupported |
| Memory Operations | rocm-smi --showmeminfo vram | rsmi_dev_memory_total_get |  |  | Unsupported |  | Unsupported |  | SUPPORTED | 1VF |  | 1VF |  | PCIe not supported | SUPPORTED | PCIe not supported | Yes | PCIe not supported |  | 1VF |
| Memory Operations | rocm-smi --showmeminfo vram or rocm-smi --showmemuse | rsmi_dev_memory_usage_get |  |  | Unsupported |  | Unsupported |  | SUPPORTED | 1VF |  | 1VF |  | Unsupported | SUPPORTED | Unsupported | Yes | Unsupported |  | 1VF |
| Memory Operations | rocm-smi --showmemuse | rsmi_dev_memory_busy_percent_get | GFX/SDMA/MMHUB | GFX/SDMA/MMHUB | Unsupported | GFX/SDMA/MMHUB | Unsupported |  | SUPPORTED | Unsupported |  | Unsupported |  | Unsupported | SUPPORTED | Unsupported | Yes | Unsupported |  | Unsupported |
| Memory Operations | rocm-smi --showpagesinfo or rocm-smi --showretiredpages | rsmi_dev_memory_reserved_pages_get | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | SUPPORTED | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | UNSUPPORTED | Unsupported | Yes | Unsupported |  | Unsupported |
| Memory Operations | rocm-smi --showmemorypartition | rsmi_dev_memory_partition_get |  |  | Unsupported |  | Unsupported |  | UNSUPPORTED | 1VF |  | 1VF |  | Unsupported | UNSUPPORTED | Unsupported | Yes | Unsupported |  | 1VF |
| Memory Operations | rocm-smi --setmemorypartition {NPS1,NPS2,NPS4,NPS8} | rsmi_dev_memory_partition_set |  | Unsupported | Unsupported |  | Unsupported |  | SUPPORTED | 1VF |  | 1VF |  | Unsupported | SUPPORTED | Unsupported | Yes | Unsupported |  | 1VF |
| Memory Operations | rocm-smi --showmemorypartition | rsmi_dev_memory_partition_capabilities_get |  | Unsupported | Unsupported |  | Unsupported |  | UNSUPPORTED | 1VF |  | 1VF |  | Unsupported | UNSUPPORTED | Unsupported | Yes | Unsupported |  | 1VF |
| Fan control | rocm-smi --showfan or rocm-smi -f | rsmi_dev_fan_rpms_get |  | Unsupported | Unsupported |  | Unsupported |  | UNSUPPORTED | 1VF |  | 1VF |  | Unsupported | SUPPORTED | Unsupported | Yes | Unsupported |  | 1VF |
| Fan control | rocm-smi --showfan or rocm-smi -f | rsmi_dev_fan_speed_get |  | Unsupported | Unsupported |  | Unsupported |  | UNSUPPORTED | 1VF |  | 1VF |  | PCIe not supported | SUPPORTED | PCIe not supported | Yes | PCIe not supported |  | 1VF |
| Fan control | rocm-smi --showfan | rsmi_dev_fan_speed_max_get |  | Unsupported | Unverified |  | Unverified |  | UNSUPPORTED | Unverified |  | Unverified |  | Unsupported | SUPPORTED | Unsupported | Yes | Unsupported |  | Unverified |
| Fan control | rocm-smi --setfan LEVEL | rsmi_dev_fan_speed_set |  | Unsupported | Unsupported |  | Unsupported |  | UNSUPPORTED | 1VF |  | 1VF |  | Unsupported | UNSUPPORTED | Unsupported | Yes | Unsupported |  | 1VF |
| Fan control | rocm-smi --resetfans | rsmi_dev_fan_reset |  | Unsupported | Unverified |  | Unverified |  | UNSUPPORTED | Unverified |  | Unverified |  | Unsupported | SUPPORTED | Unsupported | Yes | Unsupported |  | Unverified |
| Temperature & Voltage | rocm-smi --showtemp or rocm-smi -t | rsmi_dev_temp_metric_get |  |  | 1VF |  | 1VF |  | SUPPORTED | Unsupported |  | Unsupported |  | Unsupported | SUPPORTED | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Temperature & Voltage | rocm-smi --showvoltage or rocm-smi --showvoltagerange | rsmi_dev_volt_metric_get | MAX only | MAX only | Unsupported | MAX only | Unsupported | Unsupported | SUPPORTED | Unsupported |  | Unsupported |  | Unsupported | SUPPORTED | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Performance & Clocking | rocm-smi --showuse or rocm-smi -u | rsmi_dev_busy_percent_get | Support |  | Unsupported |  | Unsupported |  | SUPPORTED | Unsupported |  | Unsupported |  | Unsupported | SUPPORTED | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Performance & Clocking | rocm-smi --showuse | rsmi_utilization_count_get | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported |  | UNSUPPORTED | Unsupported |  | Unsupported |  | Unsupported | UNSUPPORTED | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Performance & Clocking | rocm-smi --showuse | rsmi_dev_activity_metric_get |  |  | Unsupported |  | Unsupported |  | UNSUPPORTED | Unsupported |  | Unsupported |  | Unsupported | SUPPORTED | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Performance & Clocking | rocm-smi --showuse | rsmi_dev_activity_avg_mm_get | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported |  | SUPPORTED | Unsupported |  | Unsupported |  | Unsupported | SUPPORTED | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Performance & Clocking | rocm-smi --showperflevel or rocm-smi -p | rsmi_dev_perf_level_get | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported |  | SUPPORTED | Unsupported |  | Unsupported |  | Unsupported | SUPPORTED | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Performance & Clocking | rocm-smi --setperflevel LEVEL | rsmi_dev_perf_level_set |  |  | Unsupported |  | Unsupported |  | SUPPORTED | Unsupported |  | Unsupported |  | Unsupported | SUPPORTED | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Performance & Clocking | rocm-smi --setperflevel LEVEL | rsmi_dev_perf_level_set_v1 | MAX only | MAX only | Unsupported | MAX only | Unsupported | Unsupported | SUPPORTED | Unsupported |  | Unsupported |  | Unsupported | SUPPORTED | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Performance & Clocking | rocm-smi --setperfdeterminism SCLK | rsmi_perf_determinism_mode_set | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | SUPPORTED | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | UNSUPPORTED | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Performance & Clocking | rocm-smi --showoverdrive or rocm-smi -o | rsmi_dev_overdrive_level_get | Auto and manual only | Auto and manual only | Unsupported | Auto and manual only | Unsupported | Auto, manual, high, low | UNSUPPORTED | Unsupported |  | Unsupported |  | Unsupported | UNSUPPORTED | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Performance & Clocking | rocm-smi --setoverdrive % | rsmi_dev_overdrive_level_set | Unsupported | Unsupported |  | Unsupported |  | Unsupported | UNSUPPORTED |  | Unsupported |  | Unsupported |  | UNSUPPORTED |  | Write operation needed |  |  |  |
| Performance & Clocking | rocm-smi --setoverdrive % | rsmi_dev_overdrive_level_set_v1 | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported |  | UNSUPPORTED | Unsupported |  | Unsupported | Unsupported | Unsupported | UNSUPPORTED | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Performance & Clocking | rocm-smi --showmemoverdrive or rocm-smi -m | rsmi_dev_mem_overdrive_level_get |  |  | Unsupported |  | Unsupported |  | UNSUPPORTED | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | UNSUPPORTED | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Performance & Clocking | rocm-smi --showclocks or rocm-smi -c or rocm-smi --showclkfrq | rsmi_dev_gpu_clk_freq_get | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | SUPPORTED | Unsupported |  | Unsupported |  | Unsupported | SUPPORTED | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Performance & Clocking | rocm-smi --setsclk LEVEL or rocm-smi --setmclk LEVEL | rsmi_dev_gpu_clk_freq_set | sclk only | sclk only | sclk, 1VF only | sclk only | sclk, 1VF only | Unsupported | UNSUPPORTED | Unsupported |  | Unsupported | PCIE not supported | Unsupported | SUPPORTED | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Performance & Clocking | rocm-smi --setsrange MIN MAX or rocm-smi --setmrange MIN MAX | rsmi_dev_clk_range_set |  |  | Unsupported |  | Unsupported |  | UNSUPPORTED | Unsupported |  | Unsupported |  | Unsupported | SUPPORTED | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Performance & Clocking | rocm-smi --setextremum min\|max sclk\|mclk CLK | rsmi_dev_clk_extremum_set |  |  | Unsupported |  | Unsupported |  | UNSUPPORTED | Unsupported |  | Unsupported |  | Unsupported | UNSUPPORTED | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Performance & Clocking | rocm-smi --setslevel or rocm-smi --setmlevel | rsmi_dev_od_clk_info_set |  |  | Unsupported |  | Unsupported |  | UNSUPPORTED | Unsupported |  | Unsupported |  | Unsupported | SUPPORTED | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Performance & Clocking | rocm-smi --gpureset | rsmi_dev_gpu_reset |  |  | 1VF only |  | 1VF only | Unsupported | SUPPORTED | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | SUPPORTED | Unsupported | Write operation needed | Unsupported | Unsupported | Unsupported |
| Overdrive & Voltage Curve | rocm-smi --showvc or rocm-smi --showclkvolt | rsmi_dev_od_volt_info_get |  |  | Unsupported |  | Unsupported | SCLK only | SUPPORTED | Unsupported |  | Unsupported |  | Unsupported | SUPPORTED | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Overdrive & Voltage Curve | rocm-smi --setvc POINT SCLK SVOLT | rsmi_dev_od_volt_info_set | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | UNSUPPORTED | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | UNSUPPORTED | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Overdrive & Voltage Curve | rocm-smi --showvc | rsmi_dev_od_volt_curve_regions_get | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | UNSUPPORTED | Unsupported |  | Unsupported |  | Unsupported | UNSUPPORTED | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| GPU Metrics | rocm-smi --showmetrics | rsmi_dev_gpu_metrics_info_get | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | SUPPORTED | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | SUPPORTED | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| GPU Metrics | rocm-smi --showmetrics | rsmi_dev_metrics_header_info_get |  |  | Unsupported |  | Unsupported |  | SUPPORTED | Unsupported |  | Unsupported |  | Unsupported | SUPPORTED | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| GPU Metrics | rocm-smi --showmetrics | rsmi_dev_metrics_xcd_counter_get |  |  | Unsupported |  | Unsupported |  | SUPPORTED | Unverified |  | Unverified |  | Unverified | SUPPORTED | Unverified | Write operation needed | Unverified |  | Unverified |
| GPU Metrics | rocm-smi --showmetrics | rsmi_dev_metrics_log_get | Individual GPU reset supported including XGMI configs<br>SWDEV-496777 | Individual GPU reset supported including XGMI configs<br>SWDEV-496777 | Unsupported | Individual GPU reset supported including XGMI configs | Unsupported | If XGMI system: Mode1 Unsupported | SUPPORTED | Unsupported |  | Unsupported |  | Unsupported | SUPPORTED | Unsupported | Write operation needed | Unsupported |  | Unsupported |
| Version & Firmware | rocm-smi --version | rsmi_version_get |  |  | N/A |  | N/A |  | SUPPORTED | N/A |  | N/A |  | N/A | SUPPORTED | N/A | Write operation needed | N/A |  | N/A |
| Version & Firmware | rocm-smi --version or rocm-smi --showdriverversion | rsmi_version_str_get |  |  | N/A |  | N/A |  | SUPPORTED | N/A |  | N/A |  | N/A | SUPPORTED | N/A | Yes | N/A |  | N/A |
| Version & Firmware | rocm-smi --showvbios or rocm-smi -v | rsmi_dev_vbios_version_get |  |  | N/A |  | N/A |  | SUPPORTED | N/A |  | N/A |  | N/A | SUPPORTED | N/A | Yes | N/A |  | N/A |
| Version & Firmware | rocm-smi --showfwinfo [BLOCK] | rsmi_dev_firmware_version_get |  |  | TBD |  | TBD | Unsupported | UNSUPPORTED | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | UNSUPPORTED | Unsupported | Yes | Unsupported | Unsupported | Unsupported |
| ECC & Error Management | rocm-smi --showrasinfo [BLOCK] | rsmi_dev_ecc_count_get | TBD | TBD | TBD | TBD | TBD | Unsupported | UNSUPPORTED | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | UNSUPPORTED | Unsupported | Yes | Unsupported | Unsupported | Unsupported |
| ECC & Error Management | rocm-smi --showrasinfo | rsmi_dev_ecc_enabled_get |  |  | TBD |  | TBD | Unsupported | SUPPORTED | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | UNSUPPORTED | Unsupported | Write operation needed | Unsupported | Unsupported | Unsupported |
| ECC & Error Management | rocm-smi --showrasinfo | rsmi_dev_ecc_status_get | TBD | TBD | TBD | TBD | TBD | Unsupported | UNSUPPORTED | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | UNSUPPORTED | Unsupported | Write operation needed | Unsupported | Unsupported | Unsupported |
| ECC & Error Management | Internal - error message handling | rsmi_status_string |  |  | TBD |  | TBD | Unsupported | SUPPORTED | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | SUPPORTED | Unsupported | Write operation needed | Unsupported | Unsupported | Unsupported |
| Performance Counters | No direct CLI mapping (Internal) | rsmi_dev_counter_group_supported |  |  |  |  |  |  | UNSUPPORTED |  |  |  |  |  | UNSUPPORTED |  |  |  |  |  |
| Performance Counters | No direct CLI mapping (Internal) | rsmi_dev_counter_create |  |  |  |  |  |  | SUPPORTED |  |  |  |  |  | SUPPORTED |  |  |  |  |  |
| Performance Counters | No direct CLI mapping (Internal) | rsmi_dev_counter_destroy |  |  |  |  |  |  | SUPPORTED |  |  |  |  |  | SUPPORTED |  |  |  |  |  |
| Performance Counters | No direct CLI mapping (Internal) | rsmi_counter_control |  |  |  |  |  |  | SUPPORTED |  |  |  |  |  | SUPPORTED |  |  |  |  |  |
| Performance Counters | No direct CLI mapping (Internal) | rsmi_counter_read |  |  |  |  |  |  | SUPPORTED |  |  |  |  |  | SUPPORTED |  |  |  |  |  |
| Performance Counters | No direct CLI mapping (Internal) | rsmi_counter_available_counters_get |  |  |  |  |  |  | SUPPORTED |  |  |  |  |  | UNSUPPORTED |  |  |  |  |  |
| Process Information | rocm-smi --showpids | rsmi_compute_process_info_get |  |  |  |  |  |  | SUPPORTED |  |  |  |  |  | SUPPORTED |  |  |  |  |  |
| Process Information | rocm-smi --showpids | rsmi_compute_process_info_by_pid_get |  |  |  |  |  |  | UNSUPPORTED |  |  |  |  |  | UNSUPPORTED |  |  |  |  |  |
| Process Information | rocm-smi --showpidgpus | rsmi_compute_process_info_by_device_get |  |  |  |  |  |  | UNSUPPORTED |  |  |  |  |  | UNSUPPORTED |  |  |  |  |  |
| Process Information | rocm-smi --showpidgpus PID | rsmi_compute_process_gpus_get |  |  |  |  |  |  | UNSUPPORTED |  |  |  |  |  | UNSUPPORTED |  |  |  |  |  |
| XGMI Operations | rocm-smi --showxgmierr | rsmi_dev_xgmi_error_status |  |  |  |  |  |  | SUPPORTED |  |  |  |  |  | UNSUPPORTED |  |  |  |  |  |
| XGMI Operations | rocm-smi --resetxgmierr | rsmi_dev_xgmi_error_reset |  |  |  |  |  |  | SUPPORTED |  |  |  |  |  | UNSUPPORTED |  |  |  |  |  |
| XGMI Operations | rocm-smi --showtopo | rsmi_dev_xgmi_hive_id_get |  |  |  |  |  |  | SUPPORTED |  |  |  |  |  | SUPPORTED |  |  |  |  |  |
| Topology | rocm-smi --showtoponuma | rsmi_topo_get_numa_node_number |  |  |  |  |  |  | SUPPORTED |  |  |  |  |  | SUPPORTED |  |  |  |  |  |
| Topology | rocm-smi --showtopoweight | rsmi_topo_get_link_weight |  |  |  |  |  |  | SUPPORTED |  |  |  |  |  | SUPPORTED |  |  |  |  |  |
| Topology | rocm-smi --showtopotype or rocm-smi --showtopohops | rsmi_topo_get_link_type |  |  |  |  |  |  | SUPPORTED |  |  |  |  |  | SUPPORTED |  |  |  |  |  |
| Topology | rocm-smi --showbw | rsmi_minmax_bandwidth_get |  |  |  |  |  |  | UNSUPPORTED |  |  |  |  |  | UNSUPPORTED |  |  |  |  |  |
| Topology | rocm-smi --showtopoaccess | rsmi_is_P2P_accessible |  |  |  |  |  |  | SUPPORTED |  |  |  |  |  | SUPPORTED |  |  |  |  |  |
| Compute Partitioning | rocm-smi --showcomputepartition | rsmi_dev_compute_partition_get |  |  |  |  |  |  | UNSUPPORTED |  |  |  |  |  | UNSUPPORTED |  |  |  |  |  |
| Compute Partitioning | rocm-smi --setcomputepartition {CPX,SPX,DPX,TPX,QPX} | rsmi_dev_compute_partition_set |  |  |  |  |  |  | SUPPORTED |  |  |  |  |  | SUPPORTED |  |  |  |  |  |
| Iterator APIs | Internal - function discovery | rsmi_dev_supported_func_iterator_open |  |  |  |  |  |  | SUPPORTED |  |  |  |  |  | SUPPORTED |  |  |  |  |  |
| Iterator APIs | Internal - function discovery | rsmi_dev_supported_variant_iterator_open |  |  |  |  |  |  | SUPPORTED |  |  |  |  |  | SUPPORTED |  |  |  |  |  |
| Iterator APIs | Internal - function discovery | rsmi_func_iter_value_get |  |  |  |  |  |  | SUPPORTED |  |  |  |  |  | SUPPORTED |  |  |  |  |  |
| Iterator APIs | Internal - function discovery | rsmi_func_iter_next |  |  |  |  |  |  | SUPPORTED |  |  |  |  |  | SUPPORTED |  |  |  |  |  |
| Iterator APIs | Internal - function discovery | rsmi_dev_supported_func_iterator_close |  |  |  |  |  |  | SUPPORTED |  |  |  |  |  | SUPPORTED |  |  |  |  |  |
| Event Notification | No direct CLI mapping (library internal) | rsmi_event_notification_init |  |  |  |  |  |  | SUPPORTED |  |  |  |  |  | SUPPORTED |  |  |  |  |  |
| Event Notification | No direct CLI mapping (library internal) | rsmi_event_notification_mask_set |  |  |  |  |  |  | SUPPORTED |  |  |  |  |  | SUPPORTED |  |  |  |  |  |
| Event Notification | No direct CLI mapping (library internal) | rsmi_event_notification_get |  |  |  |  |  |  | UNSUPPORTED |  |  |  |  |  | UNSUPPORTED |  |  |  |  |  |
| Event Notification | No direct CLI mapping (library internal) | rsmi_event_notification_stop |  |  |  |  |  |  | SUPPORTED |  |  |  |  |  | SUPPORTED |  |  |  |  |  |

## Hypervisor / KVM Compatibility

| Category | CLI | API | MI350 KVM | MI300HyperV | MI300KVM | Nv21/Nv32 HyperV | Nv21/Nv32 KVM | Nv32 BM | Nv21/Nv32 Guest |
|---|---|---|---|---|---|---|---|---|---|
| Library Init | amd-smi list | amdsmi_init |  |  |  |  |  |  |  |
| Library Init | amd-smi list | amdsmi_shut_down |  |  |  |  |  |  |  |
| Device | amd-smi list | amdsmi_get_processor_type |  |  |  |  |  |  |  |
| Device | amd-smi list | amdsmi_get_processor_handles |  |  |  |  |  |  |  |
| Device | amd-smi list | amdsmi_get_socket_handles |  | N/A | N/A | N/A | N/A | N/A | N/A |
| Device | amd-smi list | amdsmi_get_socket_info |  | N/A | N/A | N/A | N/A | N/A | N/A |
| Device | amd-smi list | amdsmi_get_gpu_device_bdf |  |  |  |  |  |  |  |
| Device | amd-smi list | amdsmi_get_gpu_device_uuid |  |  |  |  |  |  |  |
| Device | N/A | amdsmi_get_processor_handle_from_bdf |  |  |  |  |  |  |  |
| Device | N/A | amdsmi_get_index_from_processor_handle |  |  |  |  |  | Unsupported | Unsupported |
| Device | N/A | amdsmi_get_vf_handle_from_bdf |  |  |  |  |  | N/A | N/A |
| Device | amd-smi list | amdsmi_get_vf_bdf |  |  |  |  |  | N/A | N/A |
| Device | amd-smi list | amdsmi_get_vf_uuid |  |  |  |  |  | N/A | N/A |
| Device | N/A | amdsmi_get_processor_handle_from_index |  |  |  |  |  | Unsupported | Unsupported |
| Device | N/A | amdsmi_get_vf_handle_from_uuid |  |  |  |  |  | N/A | N/A |
| Device | amd-smi static | amdsmi_get_gpu_virtualization_mode |  |  |  |  |  | Unsupported | Unsupported |
| Device | N/A | amdsmi_get_processor_handle_from_uuid |  |  |  |  |  | Unsupported | Unsupported |
| Device | amd-smi static | amdsmi_get_cpu_affinity_with_scope |  |  |  |  |  |  |  |
| Device | amd-smi list | amdsmi_get_vf_handle_from_index |  |  |  |  |  | N/A | N/A |
| SW Versioning | amd-smi static --driver | amdsmi_get_gpu_driver_info |  |  |  |  |  |  |  |
| SW Versioning | amd-smi static --driver | amdsmi_get_gpu_driver_model |  |  | N/A |  | N/A | N/A | N/A |
| SW Versioning | amd-smi version | amdsmi_get_lib_version |  |  |  |  |  |  |  |
| Static | amd-smi static --asic | amdsmi_get_gpu_asic_info |  |  |  |  |  |  |  |
| Static | amd-smi static --vram | amdsmi_get_gpu_vram_info |  |  |  |  |  | Unsupported | Unsupported |
| Static | amd-smi static --limit | amdsmi_get_power_cap_info |  |  |  |  |  |  | Unsupported |
| Static | amd-smi static --bus | amdsmi_get_pcie_info |  |  |  |  |  |  | Partially (only static) |
|  | amd-smi metric --pcie |  |  |  |  |  |  |  |  |
| Static | amd-smi static --fb-info | amdsmi_get_fb_layout |  |  |  |  |  | Unsupported | Unsupported |
| Static | amd-smi static --vbios | amdsmi_get_gpu_vbios_info |  |  |  |  |  |  |  |
| Static | amd-smi firmware --fw-list | amdsmi_get_fw_info |  |  |  |  |  |  | Unsupported |
| Static | amd-smi firmware --error-records | amdsmi_get_fw_error_records |  |  |  |  |  | Unsupported | Unsupported |
| Static | amd-smi static --dfc-fw | amdsmi_get_dfc_fw_table |  |  | N/A |  | N/A | N/A | N/A |
| Static | amd-smi static --board | amdsmi_get_gpu_board_info |  |  |  |  |  | Unsupported | Unsupported |
| Static | amd-smi static --cache | amdsmi_get_gpu_cache_Info |  |  |  | Unsupported | Unsupported | Unsupported | Unsupported |
| Static | N/A | amdsmi_status_code_to_string |  |  |  |  |  |  |  |
| Monitoring | amd-smi metric --usage | amdsmi_get_gpu_activity |  |  |  |  |  |  |  |
| Monitoring | amd-smi metric --power | amdsmi_get_power_info |  |  |  |  |  |  | Unsupported |
| Monitoring | amd-smi set --power-cap | amdsmi_set_power_cap |  |  |  |  |  | N/A | N/A |
| Monitoring | amd-smi metric --clock | amdsmi_get_clock_info |  |  |  |  |  |  | Unsupported |
| Monitoring | amd-smi metric --temperature | amdsmi_get_temp_metric |  |  |  |  |  |  | Unsupported |
| Monitoring | amd-smi metric --power | amdsmi_is_gpu_power_management_enabled |  |  |  | (only Nv32) | (only Nv32) | Unsupported | Unsupported |
| Monitoring | amd-smi metric | amdsmi_get_gpu_metrics |  |  |  | N/A | N/A | N/A | N/A |
| Monitoring | amd-smi metric --fb-usage | amdsmi_get_gpu_vram_usage |  | Unsupported | Unsupported | Unsupported | Unsupported |  |  |
| Monitoring | amd-smi set --power-cap | amdsmi_set_power_cap |  |  |  | Unsupported | Unsupported | Unsupported | Unsupported |
| Monitoring | amd-smi metric | amdsmi_get_gpu_throttling_status |  | Unsupported | Unsupported | Unsupported | Unsupported |  | Unsupported |
| Monitoring | amd-smi static --soc-pstate | amdsmi_get_soc_pstate | Unsupported |  |  | Unsupported | Unsupported | N/A | N/A |
| Monitoring | amd-smi set --soc-pstate | amdsmi_set_soc_pstate | Unsupported |  |  | Unsupported | Unsupported | N/A | N/A |
| ECC/RAS | amd-smi bad-pages | amdsmi_get_gpu_bad_page_info |  |  |  |  |  | Unsupported | N/A |
| ECC/RAS | amd-smi metric --ecc-block | amdsmi_get_gpu_ecc_count |  |  |  |  |  | Unsupported | N/A |
| ECC/RAS | amd-smi metric --ecc | amdsmi_get_gpu_total_ecc_count |  |  |  |  |  |  | N/A |
| ECC/RAS | amd-smi static --ras | amdsmi_get_gpu_ras_feature_info |  |  |  | (only Nv32) | (only Nv32) | Unsupported | N/A |
| ECC/RAS | amd-smi static --ras | amdsmi_get_gpu_ecc_enabled |  |  |  |  |  | Unsupported | N/A |
| ECC/RAS | amd-smi static --ras | amdsmi_get_bad_page_threshold |  |  |  |  |  | Unsupported | N/A |
| ECC/RAS | amd-smi ras --cper | amdsmi_get_gpu_cper_entries |  |  |  | N/A | N/A | N/A | N/A |
| ECC/RAS | amd-smi ras --cper | amdsmi_get_afids_from_cper |  |  |  | N/A | N/A | N/A | N/A |
| Topology | amd-smi xgmi | amdsmi_get_link_metrics |  |  |  | N/A | N/A | N/A | N/A |
| Topology | amd-smi xgmi | amdsmi_get_xgmi_plpd |  |  |  | N/A | N/A | N/A | N/A |
| Topology | amd-smi set --xgmi-plpd | amdsmi_set_xgmi_plpd |  |  |  | N/A | N/A | N/A | N/A |
| Topology | amd-smi topology | amdsmi_get_link_topology |  |  |  | N/A | N/A | N/A | N/A |
| Topology | amd-smi xgmi | amdsmi_get_xgmi_fb_sharing_caps |  |  |  | N/A | N/A | N/A | N/A |
| Topology | amd-smi xgmi | amdsmi_get_xgmi_fb_sharing_mode_info |  |  |  | N/A | N/A | N/A | N/A |
| Topology | N/A | amdsmi_get_link_topology_nearest |  |  |  | N/A | N/A | N/A | N/A |
| Topology | amd-smi topology | amdsmi_topo_get_p2p_status |  |  |  | N/A | N/A | N/A | N/A |
| Topology | amd-smi set --xgmi --fb-sharing-mode | amdsmi_set_xgmi_fb_sharing_mode |  |  |  | N/A | N/A | N/A | N/A |
| Topology | amd-smi set --xgmi --fb-sharing-mode | amdsmi_set_xgmi_fb_sharing_mode_v2 |  |  |  | N/A | N/A | N/A | N/A |
| Topology | N/A | amdsmi_get_link_topology_nearest |  |  |  | N/A | N/A | N/A | N/A |
| Topology | amd-smi topology | amdsmi_topo_get_p2p_status |  |  |  | N/A | N/A | N/A | N/A |
| Topology | N/A | amdsmi_topo_get_numa_node_number |  |  |  | N/A | N/A | N/A | N/A |
| Event monitoring | amd-smi event | amdsmi_event_create |  |  |  |  |  | Unsupported | Unsupported |
| Event monitoring | amd-smi event | amdsmi_event_read |  |  |  |  |  | Unsupported | Unsupported |
| Event monitoring | amd-smi event | amdsmi_event_destroy |  |  |  |  |  | Unsupported | Unsupported |
| VF monitoring | amd-smi static --num-vf | amdsmi_get_num_vf |  |  |  |  |  | N/A | N/A |
| VF monitoring | N/A | amdsmi_get_vf_partition_info |  |  |  |  |  | N/A | N/A |
| VF monitoring | amd-smi static --vf | amdsmi_get_vf_info |  |  |  |  |  | N/A | N/A |
| VF monitoring | amd-smi metric --vf --sched --guard | amdsmi_get_vf_data |  |  |  |  |  | N/A | N/A |
| VF monitoring | amd-smi metric --vf --guest-data | amdsmi_get_guest_data |  |  |  |  |  | N/A | N/A |
| VF monitoring | amd-smi firmware --vf | amdsmi_get_vf_fw_info |  |  |  |  |  | N/A | N/A |
| VF monitoring | amd-smi profile | amdsmi_get_partition_profile_info |  |  | N/A |  | N/A | N/A | N/A |
| VF management | amd-smi set --num-vf | amdsmi_set_num_vf |  | N/A |  | N/A |  | N/A | N/A |
| VF management | amd-smi reset --vf-fb | amdsmi_clear_vf_fb |  | N/A |  | N/A |  | N/A | N/A |
| Process information | amd-smi process | amdsmi_get_gpu_process_list |  | N/A | N/A | N/A | N/A |  |  |
| Process information | amd-smi process | amdsmi_get_gpu_process_info |  | N/A | N/A | N/A | N/A |  |  |
| Process Information | amd-smi static --process-isolation | amdsmi_get_gpu_process_isolation |  | N/A | N/A | N/A | N/A | N/A |  |
| Process Information | amd-smi set --process-isolation=1 | amdsmi_set_gpu_process_isolation |  | N/A | N/A | N/A | N/A | N/A |  |
| GPU Management | amd-smi reset --gpu-reset | amdsmi_reset_gpu |  | Unsupported |  | Unsupported | Unsupported | N/A | N/A |
| GPU Management | amd-smi reset --clean-local-data | amdsmi_clean_gpu_local_data |  | N/A | N/A | N/A | N/A | N/A |  |
| Partitioning | amd-smi partition | amdsmi_get_gpu_memory_partition_config |  |  |  | N/A | N/A | N/A | N/A |
| Partitioning | amd-smi partition | amdsmi_get_gpu_accelerator_partition_profile_config |  |  |  | N/A | N/A | N/A | N/A |
| Partitioning | amd-smi partition | amdsmi_get_gpu_accelerator_partition_profile |  |  |  | N/A | N/A | N/A | N/A |
| Partitioning | amd-smi set --memory-partition | amdsmi_set_gpu_memory_partition_mode |  |  |  | N/A | N/A | N/A | N/A |
| Partitioning | amd-smi set --accelerator-partition | amdsmi_set_gpu_accelerator_partition_profile |  |  |  | N/A | N/A | N/A | N/A |
| Partitioning | amd-smi partition --global | amdsmi_get_gpu_accelerator_partition_profile_config_global |  |  |  | N/A | N/A | N/A | N/A |
 
## AMDSMI CPU Hardware Compatibility

| Category | CLI | API | Venice | Turin |
|---|---|---|---|---|
| Library Init |  |  |  |  |
| Device |  |  |  |  |
| Info |  |  |  |  |
| Static |  |  |  |  |
| Monitoring |  |  |  |  |
| Power Management |  |  |  |  |
| Process |  |  |  |  |
| RAS |  |  |  |  |
| FunctionsState |  |  |  |  |
| State |  |  |  |  |
| Performance |  |  |  |  |
| Error |  |  |  |  |
| Memory |  |  |  |  |
| Events |  |  |  |  |
| Topology |  |  |  |  |
| Etc |  |  |  |  |
