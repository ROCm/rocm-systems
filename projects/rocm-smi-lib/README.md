# rocm_smi_lib

C++ Library interface for ROCm-SMI to allow you to monitor/trace GPU system atributes 

For developer Familer with NVML here is our key to understand our API 

| NVML API	| Type |	Sub-type	| Rocm-smi equivalent API  |
|-------------------|--------|---------------|-----------------------------------|
| **Power State**  | | | |	
| GetEnforcedPowerLimit|	Device	|Power|	rsmi_dev_power_cap_set() |
| GetPowerManagementLimit|	Device	|Power	|rsmi_dev_power_cap_get() |
| GetPowerManagementLimitConstraints|	Device|	Power|	rsmi_dev_power_cap_range_get()|
| GetPowerUsage|	Device| Power | rsmi_dev_power_ave_get()|
| SetPowerManagementLimit |	Device	| Power| rsmi_dev_power_cap_set()|
| **Performance State**  | | | |	
| GetApplicationsClock	| Device |Clocks |	rsmi_dev_get_gpu_sys_freq() rsmi_dev_get_mem_sys_freq()|
| GetAutoBoostedClocksEnabled |	Device	|Clocks| rsmi_dev_perf_level_get()|
| GetClock	|Device	|Clocks	 |rsmi_dev_get_gpu_sys_freq(). rsmi_dev_get_mem_sys_freq()|
| GetClockInfo | 	Device	| Clocks	| rsmi_dev_get_gpu_sys_freq(),  rsmi_dev_get_mem_sys_freq()|
| GetMaxClockInfo | Device	| Clocks	| rsmi_dev_get_gpu_sys_freq(),  rsmi_dev_get_mem_sys_freq()|
| GetSupportedGraphicsClocks|	Device |	Clocks|	rsmi_dev_get_gpu_sys_freq(), rsmi_dev_get_mem_sys_freq()|
| GetSupportedMemoryClocks	|Device| Clocks |rsmi_dev_get_gpu_sys_freq() rsmi_dev_get_mem_sys_freq()|
| SetAutoBoostedClocksEnabled|	Device|	Clocks | rsmi_dev_perf_level_set() |
| GetFanSpeed	| Device | Physcial | rsmi_dev_mon_get_fan_speed() rsmi_dev_mon_get_max_fan_speed()  rsmi_dev_fan_rpms_get() |
| GetTemperature |	Device	|Physcial |	rsmi_dev_temp_metric_get() |
| GetTemperatureThreshold. |	Device	| Physcial | rsmi_dev_temp_metric_get()|
| **Initialization and Cleanup**	| | | |	
| Init	| | | |
| InitWithFlags | Mngt | Admin | rsmi_init() |
| Shutdown| Mngt | 	Admin	 |rsmi_shut_down() |
|**Error Reporting** | | | |		
| ErrorString |	ErrRpt	 |Err. Rep  | rsmi_status_string()  |
| **Unit Queries**	| | | |		
| UnitGetCount  |Unit | Admin | rsmi_num_monitor_devices()  |
| UnitGetFanSpeedInfo | Unit | Physcial | rsmi_dev_fan_rpms_get(), rsmi_dev_van_speed_get() |
| UnitGetTemperature |. Device | Physical | rsmi_dev_temp_metric_get() |
| UnitGetUnitInfo| 	Unit |ID. | rsmi_dev_id_get() (device ID) | 
| **Unit Commands**	| | | |		
| DeviceSetApplicationsClocks |	Device | Clocks| rsmi_dev_gpu_clk_freq_set() |
| DeviceSetComputeMode |	Device | Performance | rsmi_dev_power_profile_set() rsmi_dev_gpu_clk_freq_set() rsmi_dev_perf_level_set() |
| DeviceSetGpuOperationMode|	Device |	Power | rsmi_dev_power_profile_set()|
| DeviceSetPowerManagementLimit	| Device |	Power | rsmi_dev_power_cap_set() rsmi_dev_power_profile_set()|
| **ROCm Only Clock Management API**	| | | 
|	| Device |	Clocks	| rsmi_dev_perf_level_get() |
|	| Device |	Clocks| rsmi_dev_perf_level_set() |
|	| Device. |	Clocks	| rsmi_dev_overdrive_level_get() |
|	| Device. |	Clocks |	rsmi_dev_overdrive_level_set() |

Example application is logging performance data like Kerenl execution time  vs GPU Temprature,  GPU and Memory Clocks 

### To build library and example:
mkdir -p build

cd build

cmake ..

make

cd ..

The above commands will result in building the library librocm_smi.so and
an example, rocm_smi_ex, which links with this library.


