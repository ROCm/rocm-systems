// SPDX-License-Identifier: MIT
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

package goamdsmi

/*
#cgo CFLAGS: -Wall -I/opt/rocm/include
#cgo LDFLAGS: -L/opt/rocm/lib -L/opt/rocm/lib64 -lgoamdsmi_shim64 -Wl,--unresolved-symbols=ignore-in-object-files
#include <cstdint>
#include <amdsmi_go_shim.h>
*/
import "C"
import "unsafe"

// ``GO_gpu_init`` initializes the GPU and reports whether the initialization was
// successful. This function must be called before using other AMD SMI
// functions.
//
// Output: ``bool``, returns true on success or false on fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_gpu_init() {
//       GPU initialization is successful...
//   }
func GO_gpu_init() (bool) {
	return bool(C.goamdsmi_gpu_init())
}

// ``GO_gpu_shutdown`` shuts down the GPU and reports whether the shutdown was successful.
//
// Output: ``bool``, returns true on success or false on fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_gpu_shutdown() {
//       GPU shutdown is successful...
//   }
func GO_gpu_shutdown() (bool) {
	return bool(C.goamdsmi_gpu_shutdown())
}

// ``GO_gpu_num_monitor_devices`` returns the number of GPU monitor devices
// available.
//
// Output: ``uint``, returns the number of GPU monitor devices on success or 0 on
// fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_gpu_shutdown() {
//       GPU shutdown is successful...
//   }
func GO_gpu_num_monitor_devices() (uint) {
	return uint(C.goamdsmi_gpu_num_monitor_devices())
}

// ``GO_gpu_dev_name_get`` returns the name of the GPU device at the specified GPU
// index.
//
// Input parameter: ``int``, GPU index.
//
// Output: ``char*``, returns GPU device name on success or "NA" on fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_gpu_init() {
//       num_gpus := int(goamdsmi.GO_gpu_num_monitor_devices())
//       for i := 0; i < num_gpus; i++ {
//           goamdsmi.GO_gpu_dev_name_get(i)
//       }
//   }
func GO_gpu_dev_name_get(i int) (*C.char) {
	return C.goamdsmi_gpu_dev_name_get(C.uint(i))
}

// ``GO_gpu_dev_id_get`` returns the device ID of the GPU device at the specified GPU
// index.
//
// Input parameter: ``int``, GPU index.
//
// Output: ``uint16``, returns GPU device ID on success or ``0xFFFF`` on fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_gpu_init() {
//       num_gpus := int(goamdsmi.GO_gpu_num_monitor_devices())
//       for i := 0; i < num_gpus; i++ {
//           value16 := goamdsmi.GO_gpu_dev_id_get(i)
//       }
//   }
func GO_gpu_dev_id_get(i int) (C.uint16_t) {
	return C.uint16_t(C.goamdsmi_gpu_dev_id_get(C.uint(i)))
}

// ``GO_gpu_dev_pci_id_get`` returns the device PCI ID of the device at the
// specified GPU index.
//
// Input parameter: ``int``, GPU index.
//
// Output: ``uint64``, returns GPU devices PCI ID on success or ``0xFFFFFFFFFFFFFFFF``
// on fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_gpu_init() {
//       dev_pci_id := int(goamdsmi.GO_gpu_dev_pci_id_get())
//   }
func GO_gpu_dev_pci_id_get(i int) (C.uint64_t) {
	return C.goamdsmi_gpu_dev_pci_id_get(C.uint(i))
}

// ``GO_gpu_dev_vbios_version_get`` returns the VBIOS version of the GPU device at the
// specified GPU index.
//
// Input parameter: ``int``, GPU index.
//
// Output: ``char*``, returns VBIOS version on success or "NA" on fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_gpu_init() {
//       dev_pci_id := int(goamdsmi.GO_gpu_dev_pci_id_get())
//   }
func GO_gpu_dev_vbios_version_get(i int) (*C.char) {
	return C.goamdsmi_gpu_dev_vbios_version_get(C.uint(i))
}

// ``GO_gpu_dev_vendor_name_get`` returns the vendor name of the GPU device at the
// specified GPU index.
//
// Input parameter: ``int``, GPU index.
//
// Output: ``char*``, returns the GPU device name on success or "NA" on fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_gpu_init() {
//       num_gpus := int(goamdsmi.GO_gpu_num_monitor_devices())
//       for i := 0; i < num_gpus; i++ {
//           goamdsmi.GO_gpu_dev_vendor_name_get()
//       }
//   }
func GO_gpu_dev_vendor_name_get(i int) (*C.char) {
	return C.goamdsmi_gpu_dev_vendor_name_get(C.uint(i))
}

// ``GO_gpu_dev_power_cap_get`` returns the power cap of the GPU at the specified
// GPU index.
//
// Input parameter: ``int``, GPU index.
//
// Output: ``uint64``, returns GPU power cap on success or ``0xFFFFFFFFFFFFFFFF`` on
// fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_gpu_init() {
//       num_gpus := int(goamdsmi.GO_gpu_num_monitor_devices())
//       for i := 0; i < num_gpus; i++ {
//           dev_power_cap := int(goamdsmi.GO_gpu_dev_power_cap_get(i))
//       }
//   }
func GO_gpu_dev_power_cap_get(i int) (C.uint64_t) {
    return C.goamdsmi_gpu_dev_power_cap_get(C.uint(i))
}

// ``GO_gpu_dev_power_get`` returns the power of the GPU at the specified GPU index.
//
// Input parameter: ``int``, GPU index.
//
// Output: ``uint64``, returns GPU power on success or ``0xFFFFFFFFFFFFFFFF`` on fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_gpu_init() {
//       num_gpus := int(goamdsmi.GO_gpu_num_monitor_devices())
//       for i := 0; i < num_gpus; i++ {
//           dev_power := int(goamdsmi.GO_gpu_dev_power_get(i))
//       }
//   }
func GO_gpu_dev_power_get(i int) (C.uint64_t) {
    return C.goamdsmi_gpu_dev_power_get(C.uint(i))
}

// ``GO_gpu_dev_temp_metric_get`` returns the temperature of the GPU at the
// specified GPU index, sensor, and metric number.
//
// Input parameters:
//   - int, GPU index.
//   - int, sensor number.
//   - int, metric number.
//
// Output: ``uint64``, returns GPU temperature on success or ``0xFFFFFFFFFFFFFFFF`` on
// fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_gpu_init() {
//       num_gpus := int(goamdsmi.GO_gpu_num_monitor_devices())
//       for i := 0; i < num_gpus; i++ {
//           temp := int(goamdsmi.GO_gpu_dev_temp_metric_get(i, 1, 0))
//       }
//   }
func GO_gpu_dev_temp_metric_get(i int, sensor int, metric int) (C.uint64_t) {
    return C.goamdsmi_gpu_dev_temp_metric_get(C.uint(i), C.uint(sensor), C.uint(metric))
}

// ``GO_gpu_dev_perf_level_get`` returns the perf level of the GPU at the
// specified GPU index.
//
// Input parameter: ``int``, GPU index.
//
// Output: ``uint32``, returns GPU perf level on success or ``0xFFFFFFFF`` on fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_gpu_init() {
//       num_gpus := int(goamdsmi.GO_gpu_num_monitor_devices())
//       for i := 0; i < num_gpus; i++ {
//           dev_perf_level := int(goamdsmi.GO_gpu_dev_perf_level_get(i))
//       }
//   }
func GO_gpu_dev_perf_level_get(i int) (C.uint32_t) {
	return C.goamdsmi_gpu_dev_perf_level_get(C.uint(i))
}

// ``GO_gpu_dev_overdrive_level_get`` returns the overdrive level of the GPU at the
// specified GPU index.
//
// Input parameter: ``int``, GPU index.
//
// Output: ``uint32``, returns GPU perf level on success or ``0xFFFFFFFF`` on fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_gpu_init() {
//       num_gpus := int(goamdsmi.GO_gpu_num_monitor_devices())
//       for i := 0; i < num_gpus; i++ {
//           dev_overdrive_level := int(goamdsmi.GO_gpu_dev_overdrive_level_get(i))
//       }
//   }
func GO_gpu_dev_overdrive_level_get(i int) (C.uint32_t) {
	return C.goamdsmi_gpu_dev_perf_level_get(C.uint(i))
}

// ``GO_gpu_dev_mem_overdrive_level_get`` returns the mem overdrive level of the GPU at the
// specified GPU index.
//
// Input parameter: ``int``, GPU index.
//
// Output: ``uint32``, returns GPU perf level on success or ``0xFFFFFFFF`` on fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_gpu_init() {
//       num_gpus := int(goamdsmi.GO_gpu_num_monitor_devices())
//       for i := 0; i < num_gpus; i++ {
//           mem_overdrive_level := int(goamdsmi.GO_gpu_dev_mem_overdrive_level_get(i))
//       }
//   }
func GO_gpu_dev_mem_overdrive_level_get(i int) (C.uint32_t) {
	return C.goamdsmi_gpu_dev_overdrive_level_get(C.uint(i))
}

// ``GO_gpu_dev_gpu_clk_freq_get_sclk`` returns the system clock (SCLK) frequency of
// the GPU at the specified GPU index.
//
// Input parameter: ``int``, GPU index.
//
// Output: ``uint64``, returns GPU SCLK frequency level on success or
// ``0xFFFFFFFFFFFFFFFF`` on fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_gpu_init() {
//       num_gpus := int(goamdsmi.GO_gpu_num_monitor_devices())
//       for i := 0; i < num_gpus; i++ {
//           dev_sclk_freq := int(goamdsmi.GO_gpu_dev_gpu_clk_freq_get_sclk(i))
//       }
//   }
func GO_gpu_dev_gpu_clk_freq_get_sclk(i int) (C.uint64_t) {
	return C.goamdsmi_gpu_dev_gpu_clk_freq_get_sclk(C.uint(i))
}

// ``GO_gpu_dev_gpu_clk_freq_get_mclk`` returns the memory clock (MCLK) frequency of
// the GPU at the specified GPU index.
//
// Input parameter: ``int``, GPU index.
//
// Output: ``uint64``, returns GPU MCLK frequency level on success or
// ``0xFFFFFFFFFFFFFFFF`` on fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_gpu_init() {
//       num_gpus := int(goamdsmi.GO_gpu_num_monitor_devices())
//       for i := 0; i < num_gpus; i++ {
//           dev_sclk_freq := int(goamdsmi.GO_gpu_dev_gpu_clk_freq_get_mclk(i))
//       }
//   }
func GO_gpu_dev_gpu_clk_freq_get_mclk(i int) (C.uint64_t) {
    return C.goamdsmi_gpu_dev_gpu_clk_freq_get_mclk(C.uint(i))
}

// ``GO_gpu_od_volt_freq_range_min_get_sclk`` returns the minimum system clock
// (SCLK) frequency of the GPU at the specified GPU index.
//
// Input parameter: ``int``, GPU index.
//
// Output: ``uint64``, returns GPU minimum SCLK frequency level on success or
// ``0xFFFFFFFFFFFFFFFF`` on fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_gpu_init() {
//       num_gpus := int(goamdsmi.GO_gpu_num_monitor_devices())
//       for i := 0; i < num_gpus; i++ {
//           dev_min_sclk := int(goamdsmi.GO_gpu_od_volt_freq_range_min_get_sclk(i))
//       }
//   }
func GO_gpu_od_volt_freq_range_min_get_sclk(i int) (C.uint64_t) {
	return C.goamdsmi_gpu_od_volt_freq_range_min_get_sclk(C.uint(i))
}

// ``GO_gpu_od_volt_freq_range_min_get_mclk`` returns the minimum memory clock
// (MCLK) frequency of the GPU at the specified GPU index.
//
// Input parameter: ``int``, GPU index.
//
// Output: ``uint64``, returns GPU minimum MCLK frequency level on success or
// ``0xFFFFFFFFFFFFFFFF`` on fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_gpu_init() {
//       num_gpus := int(goamdsmi.GO_gpu_num_monitor_devices())
//       for i := 0; i < num_gpus; i++ {
//           dev_min_mclk := int(goamdsmi.GO_gpu_od_volt_freq_range_min_get_mclk(i))
//       }
//   }
func GO_gpu_od_volt_freq_range_min_get_mclk(i int) (C.uint64_t) {
	return C.goamdsmi_gpu_od_volt_freq_range_min_get_mclk(C.uint(i))
}

// ``GO_gpu_od_volt_freq_range_max_get_sclk`` returns the maximum system clock
// (SCLK) frequency of the GPU at the specified GPU index.
//
// Input parameter: ``int``, GPU index.
//
// Output: ``uint64``, returns GPU maximum SCLK frequency level on success or
// ``0xFFFFFFFFFFFFFFFF`` on fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_gpu_init() {
//       num_gpus := int(goamdsmi.GO_gpu_num_monitor_devices())
//       for i := 0; i < num_gpus; i++ {
//           dev_max_sclk := int(goamdsmi.GO_gpu_od_volt_freq_range_max_get_sclk(i))
//       }
//   }
func GO_gpu_od_volt_freq_range_max_get_sclk(i int) (C.uint64_t) {
	return C.goamdsmi_gpu_od_volt_freq_range_max_get_sclk(C.uint(i))
}

// ``GO_gpu_od_volt_freq_range_max_get_mclk`` returns the maximum memory clock
// (MCLK) frequency of the GPU at the specified GPU index.
//
// Input parameter: ``int``, GPU index.
//
// Output: ``uint64``, returns GPU maximum MCLK frequency level on success or
// ``0xFFFFFFFFFFFFFFFF`` on fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_gpu_init() {
//       num_gpus := int(goamdsmi.GO_gpu_num_monitor_devices())
//       for i := 0; i < num_gpus; i++ {
//           dev_max_mclk := int(goamdsmi.GO_gpu_od_volt_freq_range_max_get_mclk(i))
//       }
//   }
func GO_gpu_od_volt_freq_range_max_get_mclk(i int) (C.uint64_t) {
	return C.goamdsmi_gpu_od_volt_freq_range_max_get_mclk(C.uint(i))
}

// ``GO_gpu_dev_gpu_busy_percent_get`` returns the busy percentage of the GPU at the
// specified GPU index.
//
// Input parameter: ``int``, GPU index.
//
// Output: ``uint32``, returns GPU busy percentage on success or ``0xFFFFFFFF`` on fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_gpu_init() {
//       num_gpus := int(goamdsmi.GO_gpu_num_monitor_devices())
//       for i := 0; i < num_gpus; i++ {
//           dev_busy_perc := int(goamdsmi.GO_gpu_dev_gpu_busy_percent_get(i))
//       }
//   }
func GO_gpu_dev_gpu_busy_percent_get(i int) (C.uint32_t) {
	return C.goamdsmi_gpu_dev_gpu_busy_percent_get(C.uint(i))
}

// ``GO_gpu_dev_gpu_memory_busy_percent_get`` returns the memory busy percentage of
// the GPU at the specified GPU index.
//
// Input parameter: ``int``, GPU index.
//
// Output: ``uint64``, returns GPU memory busy percentage on success or
// ``0xFFFFFFFFFFFFFFFF`` on fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_gpu_init() {
//       num_gpus := int(goamdsmi.GO_gpu_num_monitor_devices())
//       for i := 0; i < num_gpus; i++ {
//           mem_busy_perc := int(goamdsmi.GO_gpu_dev_gpu_memory_busy_percent_get(i))
//       }
//   }
func GO_gpu_dev_gpu_memory_busy_percent_get(i int) (C.uint64_t) {
	return C.goamdsmi_gpu_dev_gpu_memory_busy_percent_get(C.uint(i))
}

// ``GO_gpu_dev_gpu_memory_usage_get`` returns the memory usage of the GPU at the
// specified GPU index.
//
// Input parameter: ``int``, GPU index.
//
// Output: ``uint64``, returns GPU memory usage on success or ``0xFFFFFFFFFFFFFFFF`` on
// fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_gpu_init() {
//       num_gpus := int(goamdsmi.GO_gpu_num_monitor_devices())
//       for i := 0; i < num_gpus; i++ {
//           mem_usage := int(goamdsmi.GO_gpu_dev_gpu_memory_usage_get(i))
//       }
//   }
func GO_gpu_dev_gpu_memory_usage_get (i int) (C.uint64_t) {
	return C.goamdsmi_gpu_dev_gpu_memory_usage_get(C.uint(i))
}

// ``GO_gpu_dev_gpu_memory_total_get`` returns the total memory of the GPU at the
// specified GPU index.
//
// Input parameter: ``int``, GPU index.
//
// Output: ``uint64``, returns GPU memory usage on success or ``0xFFFFFFFFFFFFFFFF`` on
// fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_gpu_init() {
//       num_gpus := int(goamdsmi.GO_gpu_num_monitor_devices())
//       for i := 0; i < num_gpus; i++ {
//           mem_total := int(goamdsmi.GO_gpu_dev_gpu_memory_total_get(i))
//       }
//   }
func GO_gpu_dev_gpu_memory_total_get (i int) (C.uint64_t) {
	return C.goamdsmi_gpu_dev_gpu_memory_total_get(C.uint(i))
}

//CPU ESMI or AMDSMI calls

// ``GO_cpu_init`` initializes the CPU and reports whether the initialization was
// successful.
//
// Output: ``bool``, returns true on success or false on fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_cpu_init() {
//       CPU initialization is successful...
//   }
func GO_cpu_init() (bool) {
	return bool(C.goamdsmi_cpu_init())
}

// ``GO_cpu_number_of_sockets_get`` returns the number of available CPU sockets.
//
// Output: ``uint``, returns the number of CPU sockets on success or 0 on fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_cpu_init() {
//       num_sockets := int(goamdsmi.GO_cpu_number_of_sockets_get())
//   }
func GO_cpu_number_of_sockets_get() (uint) {
	return uint(C.goamdsmi_cpu_number_of_sockets_get())
}

// ``GO_cpu_number_of_threads_get`` returns the number of available CPU sockets.
//
// Output: ``uint``, returns the number of CPU threads on success or 0 on fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_cpu_init() {
//       num_threads := int(goamdsmi.GO_cpu_number_of_threads_get())
//   }
func GO_cpu_number_of_threads_get() (uint) {
	return uint(C.goamdsmi_cpu_number_of_threads_get())
}

// ``GO_cpu_threads_per_core_get`` returns the thread count per available CPU core.
//
// Output: ``uint``, returns the CPU thread count on success or 0 on fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_cpu_init() {
//       num_threads_per_core := int(goamdsmi.GO_cpu_threads_per_core_get())
//   }
func GO_cpu_threads_per_core_get() (uint) {
	return uint(C.goamdsmi_cpu_threads_per_core_get())
}

// ``GO_cpu_core_energy_get`` returns the CPU core energy for the specified thread
// index.
//
// Input parameter: ``int``, thread index.
//
// Output: ``uint64``, returns CPU core energy on success or ``0xFFFFFFFFFFFFFFFF`` on
// fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_cpu_init() {
//       num_threads := int(goamdsmi.GO_cpu_number_of_threads_get())
//       for i := 0; i < num_threads; i++ {
//           core_energy := int(goamdsmi.GO_cpu_core_energy_get(i))
//       }
//   }
func GO_cpu_core_energy_get(i int) (C.uint64_t) {
	return C.goamdsmi_cpu_core_energy_get(C.uint(i))
}

// ``GO_cpu_core_boostlimit_get`` returns the CPU core boost limit for the specified
// thread index.
//
// Input parameter: ``int``, thread index.
//
// Output: ``uint32``, returns CPU core boost limit on success or ``0xFFFFFFFF`` on
// fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_cpu_init() {
//       num_threads := int(goamdsmi.GO_cpu_number_of_threads_get())
//       for i := 0; i < num_threads; i++ {
//           core_boost_limit := int(goamdsmi.GO_cpu_core_boostlimit_get(i))
//       }
//   }
func GO_cpu_core_boostlimit_get(i int) (C.uint32_t) {
	return C.goamdsmi_cpu_core_boostlimit_get(C.uint(i))
}

// ``GO_cpu_socket_energy_get`` returns the CPU socket energy for the specified
// socket index.
//
// Input parameter: ``int``, socket index.
//
// Output: ``uint64``, returns socket energy level on success or ``0xFFFFFFFFFFFFFFFF``
// on fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_cpu_init() {
//       num_sockets := int(goamdsmi.GO_cpu_number_of_sockets_get())
//       for i := 0; i < num_sockets; i++ {
//           socket_energy := int(goamdsmi.GO_cpu_socket_energy_get(i))
//       }
//   }
func GO_cpu_socket_energy_get(i int) (C.uint64_t) {
	return C.goamdsmi_cpu_socket_energy_get(C.uint(i))
}

// ``GO_cpu_socket_power_get`` returns the socket power for the specified socket
// index.
//
// Input parameter: ``int``, socket index.
//
// Output: ``uint32``, returns socket energy level on success or ``0xFFFFFFFF``
// on fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_cpu_init() {
//       num_sockets := int(goamdsmi.GO_cpu_number_of_sockets_get())
//       for i := 0; i < num_sockets; i++ {
//           socket_power := int(goamdsmi.GO_cpu_socket_power_get(i))
//       }
//   }
func GO_cpu_socket_power_get(i int) (C.uint32_t) {
	return C.goamdsmi_cpu_socket_power_get(C.uint(i))
}

// ``GO_cpu_socket_power_cap_get`` returns the socket power cap for the specified
// socket index.
//
// Input parameter: ``int``, socket index.
//
// Output: ``uint32``, returns socket power cap on success or ``0xFFFFFFFF``
// on fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_cpu_init() {
//       num_sockets := int(goamdsmi.GO_cpu_number_of_sockets_get())
//       for i := 0; i < num_sockets; i++ {
//           socket_power_cap := int(goamdsmi.GO_cpu_socket_power_cap_get(i))
//       }
//   }
func GO_cpu_socket_power_cap_get(i int) (C.uint32_t) {
	return C.goamdsmi_cpu_socket_power_cap_get(C.uint(i))
}

// ``GO_cpu_socket_power_cap_get`` returns the PROCHOT status for the specified
// socket index.
//
// Input parameter: ``int``, socket index.
//
// Output: ``uint32``, returns PROCHOT status on success or ``0xFFFFFFFF`` on fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_cpu_init() {
//       num_sockets := int(goamdsmi.GO_cpu_number_of_sockets_get())
//       for i := 0; i < num_sockets; i++ {
//           prochot_status := int(goamdsmi.GO_cpu_prochot_status_get(i))
//       }
//   }
func GO_cpu_prochot_status_get(i int) (C.uint32_t) {
	return C.goamdsmi_cpu_prochot_status_get(C.uint(i))
}

// ``GO_gpu_uma_carveout_info_get`` retrieves the UMA carveout configuration
// information for the specified GPU device.
//
// Note: This is a kernel UAPI feature (sysfs), not libdrm.
//
// Input parameters:
//   - ``device_index int``: GPU device index
//   - ``current_index *uint32``: pointer to store current carveout option index
//   - ``num_options *uint32``: pointer to store number of available options
//   - ``options *[16][256]byte``: pointer to array for option descriptions
//     (16 == AMDSMI_MAX_CARVEOUT_OPTIONS, 256 == AMDSMI_MAX_STRING_LENGTH from amdsmi.h)
//
// Output: ``int32``, returns ``0`` on success or ``-1`` on fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_gpu_init() {
//       var currentIdx, numOpts uint32
//       var opts [16][256]byte
//       ret := goamdsmi.GO_gpu_uma_carveout_info_get(0, &currentIdx, &numOpts, &opts)
//       if ret == 0 {
//           // Process UMA carveout info...
//       }
//   }
func GO_gpu_uma_carveout_info_get(device_index int, current_index *uint32, num_options *uint32, options *[16][256]byte) int32 {
	return int32(C.goamdsmi_gpu_uma_carveout_info_get(
		C.uint32_t(device_index),
		(*C.uint32_t)(current_index),
		(*C.uint32_t)(num_options),
		(*[16][256]C.char)(unsafe.Pointer(options))))
}

// ``GO_gpu_uma_carveout_set`` sets the UMA carveout size for the specified GPU
// device by option index. Requires system reboot to take effect.
//
// Note: This is a kernel UAPI feature (sysfs), not libdrm.
//
// Input parameters:
//   - ``device_index int``: GPU device index
//   - ``option_index uint32``: carveout option index to set
//
// Output: ``int32``, returns ``0`` on success or ``-1`` on fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   if true == goamdsmi.GO_gpu_init() {
//       ret := goamdsmi.GO_gpu_uma_carveout_set(0, 3)
//       if ret == 0 {
//           // UMA carveout set successfully, reboot required
//       }
//   }
func GO_gpu_uma_carveout_set(device_index int, option_index uint32) int32 {
	return int32(C.goamdsmi_gpu_uma_carveout_set(C.uint32_t(device_index), C.uint32_t(option_index)))
}

// ``GO_ttm_info_get`` retrieves the TTM (shared GPU memory) pages limit.
// This is a system-wide setting, not per-GPU.
//
// Note: This is a kernel UAPI feature (modprobe.d), not libdrm.
//
// Input parameter:
//   - ``current_pages *uint64``: pointer to store current TTM pages limit
//
// Output: ``int32``, returns ``0`` on success or ``-1`` on fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   var pages uint64
//   ret := goamdsmi.GO_ttm_info_get(&pages)
//   if ret == 0 {
//       // Process TTM pages info...
//   }
func GO_ttm_info_get(current_pages *uint64) int32 {
	return int32(C.goamdsmi_ttm_info_get((*C.uint64_t)(current_pages)))
}

// ``GO_ttm_pages_limit_set`` sets the TTM (shared GPU memory) pages limit.
// This is a system-wide setting, not per-GPU. Requires system reboot to take effect.
//
// Note: This is a kernel UAPI feature (modprobe.d), not libdrm.
//
// Input parameter:
//   - ``pages uint64``: TTM pages limit to set
//
// Output: ``int32``, returns ``0`` on success or ``-1`` on fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   ret := goamdsmi.GO_ttm_pages_limit_set(3104239)
//   if ret == 0 {
//       // TTM pages limit set successfully, reboot required
//   }
func GO_ttm_pages_limit_set(pages uint64) int32 {
	return int32(C.goamdsmi_ttm_pages_limit_set(C.uint64_t(pages)))
}

// ``GO_ttm_pages_limit_reset`` resets the TTM (shared GPU memory) pages limit
// to its default value.
//
// This is a system-wide setting, not per-GPU. A system reboot may be required
// for the change to take effect.
//
// Note: This is a kernel UAPI feature (modprobe.d), not libdrm.
//
// Output: ``int32``, returns ``0`` on success or ``-1`` on fail.
//
// Example:
//
//   import "github.com/ROCm/amdsmi"
//
//   ret := goamdsmi.GO_ttm_pages_limit_reset()
//   if ret == 0 {
//       // TTM pages limit reset successfully, reboot may be required
//   }
func GO_ttm_pages_limit_reset() int32 {
	return int32(C.goamdsmi_ttm_pages_limit_reset())
}
