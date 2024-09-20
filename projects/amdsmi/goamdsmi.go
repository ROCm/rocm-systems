// SPDX-License-Identifier: MIT
/*
 * Copyright (c) 2022, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sellcopies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *  - The above copyright notice and this permission notice shall be included in
 *    all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * Except as contained in this notice, the name of the Advanced Micro Devices,
 * Inc. shall not be used in advertising or otherwise to promote the sale, use
 * or other dealings in this Software without prior written authorization from
 * the Advanced Micro Devices, Inc.
 *
 */

package goamdsmi

/*
#cgo CFLAGS: -Wall -I/opt/rocm/include
#cgo LDFLAGS: -L/opt/rocm/lib -L/opt/rocm/lib64 -lgoamdsmi_shim64 -Wl,--unresolved-symbols=ignore-in-object-files
#include <stdint.h>
#include <amdsmi_go_shim.h>
*/
import "C"

//GPU ROCM or AMDSMI calls
func GO_gpu_init() (bool) {
	return bool(C.goamdsmi_gpu_init())
}

func GO_gpu_shutdown() (bool) {
	return bool(C.goamdsmi_gpu_shutdown())
}

func GO_gpu_num_monitor_devices() (uint) {
	return uint(C.goamdsmi_gpu_num_monitor_devices())
}

func GO_gpu_dev_name_get(i int) (*C.char) {
	return C.goamdsmi_gpu_dev_name_get(C.uint(i))
}

func GO_gpu_dev_id_get(i int) (C.uint16_t) {
	return C.uint16_t(C.goamdsmi_gpu_dev_id_get(C.uint(i)))
}

func GO_gpu_dev_pci_id_get(i int) (C.uint64_t) {
	return C.goamdsmi_gpu_dev_pci_id_get(C.uint(i))
}

func GO_gpu_dev_vbios_version_get(i int) (*C.char) {
	return C.goamdsmi_gpu_dev_vbios_version_get(C.uint(i))
}

func GO_gpu_dev_vendor_name_get(i int) (*C.char) {
	return C.goamdsmi_gpu_dev_vendor_name_get(C.uint(i))
}

func GO_gpu_dev_power_cap_get(i int) (C.uint64_t) {
    return C.goamdsmi_gpu_dev_power_cap_get(C.uint(i))
}

func GO_gpu_dev_power_get(i int) (C.uint64_t) {
    return C.goamdsmi_gpu_dev_power_get(C.uint(i))
}

func GO_gpu_dev_temp_metric_get(i int, sensor int, metric int) (C.uint64_t) {
    return C.goamdsmi_gpu_dev_temp_metric_get(C.uint(i), C.uint(sensor), C.uint(metric))
}

func GO_gpu_dev_perf_level_get(i int) (C.uint32_t) {
	return C.goamdsmi_gpu_dev_perf_level_get(C.uint(i))
}

func GO_gpu_dev_overdrive_level_get(i int) (C.uint32_t) {
	return C.goamdsmi_gpu_dev_perf_level_get(C.uint(i))
}

func GO_gpu_dev_mem_overdrive_level_get(i int) (C.uint32_t) {
	return C.goamdsmi_gpu_dev_overdrive_level_get(C.uint(i))
}

func GO_gpu_dev_gpu_clk_freq_get_sclk(i int) (C.uint64_t) {
	return C.goamdsmi_gpu_dev_gpu_clk_freq_get_sclk(C.uint(i))
}

func GO_gpu_dev_gpu_clk_freq_get_mclk(i int) (C.uint64_t) {
    return C.goamdsmi_gpu_dev_gpu_clk_freq_get_mclk(C.uint(i))
}

func GO_gpu_od_volt_freq_range_min_get_sclk(i int) (C.uint64_t) {
	return C.goamdsmi_gpu_od_volt_freq_range_min_get_sclk(C.uint(i))
}

func GO_gpu_od_volt_freq_range_min_get_mclk(i int) (C.uint64_t) {
	return C.goamdsmi_gpu_od_volt_freq_range_min_get_mclk(C.uint(i))
}

func GO_gpu_od_volt_freq_range_max_get_sclk(i int) (C.uint64_t) {
	return C.goamdsmi_gpu_od_volt_freq_range_max_get_sclk(C.uint(i))
}

func GO_gpu_od_volt_freq_range_max_get_mclk(i int) (C.uint64_t) {
	return C.goamdsmi_gpu_od_volt_freq_range_max_get_mclk(C.uint(i))
}

func GO_gpu_dev_gpu_busy_percent_get(i int) (C.uint32_t) {
	return C.goamdsmi_gpu_dev_gpu_busy_percent_get(C.uint(i))
}

func GO_gpu_dev_gpu_memory_busy_percent_get(i int) (C.uint64_t) {
	return C.goamdsmi_gpu_dev_gpu_memory_busy_percent_get(C.uint(i))
}

func GO_gpu_dev_gpu_memory_usage_get (i int) (C.uint64_t) {
	return C.goamdsmi_gpu_dev_gpu_memory_usage_get(C.uint(i))
}

func GO_gpu_dev_gpu_memory_total_get (i int) (C.uint64_t) {
	return C.goamdsmi_gpu_dev_gpu_memory_total_get(C.uint(i))
}

//CPU ESMI or AMDSMI calls
func GO_cpu_init() (bool) {
	return bool(C.goamdsmi_cpu_init())
}

func GO_cpu_number_of_sockets_get() (uint) {
	return uint(C.goamdsmi_cpu_number_of_sockets_get())
}

func GO_cpu_number_of_threads_get() (uint) {
	return uint(C.goamdsmi_cpu_number_of_threads_get())
}

func GO_cpu_threads_per_core_get() (uint) {
	return uint(C.goamdsmi_cpu_threads_per_core_get())
}

func GO_cpu_core_energy_get(i int) (C.uint64_t) {
	return C.goamdsmi_cpu_core_energy_get(C.uint(i))
}

func GO_cpu_core_boostlimit_get(i int) (C.uint32_t) {
	return C.goamdsmi_cpu_core_boostlimit_get(C.uint(i))
}

func GO_cpu_socket_energy_get(i int) (C.uint64_t) {
	return C.goamdsmi_cpu_socket_energy_get(C.uint(i))
}

func GO_cpu_socket_power_get(i int) (C.uint32_t) {
	return C.goamdsmi_cpu_socket_power_get(C.uint(i))
}

func GO_cpu_socket_power_cap_get(i int) (C.uint32_t) {
	return C.goamdsmi_cpu_socket_power_cap_get(C.uint(i))
}

func GO_cpu_prochot_status_get(i int) (C.uint32_t) {
	return C.goamdsmi_cpu_prochot_status_get(C.uint(i))
}
