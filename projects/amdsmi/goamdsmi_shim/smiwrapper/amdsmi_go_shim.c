// SPDX-License-Identifier: MIT
/*
 * Copyright (c) 2024, Advanced Micro Devices, Inc.
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

#include <stdint.h>
#include <string.h>
#include "amdsmi_go_shim.h"
#ifdef AMDSMI_BUILD
#include <amd_smi/amdsmi.h>
#endif
#include <unistd.h>
#define nullptr ((void*)0)

#ifdef AMDSMI_BUILD
#define MAX_SOCKET_ACROSS_SYSTEM         4
#define CPU_0                            0
#define GPU_SENSOR_0                     0
#define MAX_CPU_PER_SOCKET               4
#define MAX_PHYSICALCORE_ACROSS_SYSTEM 384
#define MAX_LOGICALCORE_ACROSS_SYSTEM  768
#define MAX_GPU_DEVICE_ACROSS_SYSTEM    24
#define MAX_GPU_POWER_FROM_DRIVER      0xFFFF

#define AMDSMI_DRIVER_NAME     "AMDSMI"
#define AMDSMI_LIB_FILE        "/opt/rocm/lib/libamd_smi.so"
#define AMDSMI_LIB64_FILE      "/opt/rocm/lib64/libamd_smi.so"

#define AMDGPU_DRIVER_NAME     "AMDGPUDriver"
#define AMDGPU_INITSTATE_FILE  "/sys/module/amdgpu/initstate"

#define AMDHSMP_DRIVER_NAME    "AMDHSMPDriver"
#define AMDHSMP_INITSTATE_FILE "/sys/module/amd_hsmp/initstate"

static uint32_t num_apuSockets              = GOAMDSMI_VALUE_0;
static uint32_t num_cpuSockets              = GOAMDSMI_VALUE_0;
static uint32_t num_gpuSockets              = GOAMDSMI_VALUE_0;
static uint32_t cpuInitCompleted            = false;
static uint32_t gpuInitCompleted            = false;
static uint32_t apuInitCompleted            = false;

static uint32_t num_cpu_inAllSocket                = GOAMDSMI_VALUE_0;
static uint32_t num_cpu_physicalCore_inAllSocket   = GOAMDSMI_VALUE_0;
static uint32_t num_gpu_devices_inAllSocket        = GOAMDSMI_VALUE_0;

static amdsmi_socket_handle     amdsmi_apusocket_handle_all_socket[MAX_SOCKET_ACROSS_SYSTEM+MAX_GPU_DEVICE_ACROSS_SYSTEM]    = {0};
static amdsmi_socket_handle     amdsmi_cpusocket_handle_all_socket[MAX_SOCKET_ACROSS_SYSTEM]                                 = {0};
static amdsmi_socket_handle     amdsmi_gpusocket_handle_all_socket[MAX_GPU_DEVICE_ACROSS_SYSTEM]                             = {0};
static amdsmi_processor_handle  amdsmi_processor_handle_all_cpu_across_socket[MAX_SOCKET_ACROSS_SYSTEM*MAX_CPU_PER_SOCKET]   = {0};
static amdsmi_processor_handle  amdsmi_processor_handle_all_cpu_physicalCore_across_socket[MAX_PHYSICALCORE_ACROSS_SYSTEM]   = {0};
static amdsmi_processor_handle  amdsmi_processor_handle_all_gpu_device_across_socket[MAX_GPU_DEVICE_ACROSS_SYSTEM]           = {0};

goamdsmi_status_t is_file_present(const char* driver_name, const char* file_name)
{
    if(0 == access(file_name, F_OK)) 
    {
        if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) {printf("AMDSMI, Success, %s found \"%s\" and returns:%d\n", driver_name, file_name, GOAMDSMI_STATUS_SUCCESS);}
        return GOAMDSMI_STATUS_SUCCESS;
    }
    if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_2)) {printf("AMDSMI, Status, %s not found, missing \"%s\" and returns:%d\n", driver_name, file_name, GOAMDSMI_STATUS_FAILURE);}
    return GOAMDSMI_STATUS_FAILURE;
}

goamdsmi_status_t go_shim_amdsmi_present()
{
    if((GOAMDSMI_STATUS_SUCCESS == is_file_present(AMDSMI_DRIVER_NAME, AMDSMI_LIB_FILE)) || (GOAMDSMI_STATUS_SUCCESS == is_file_present(AMDSMI_DRIVER_NAME, AMDSMI_LIB64_FILE)))
    {
        return GOAMDSMI_STATUS_SUCCESS;
    }
    return GOAMDSMI_STATUS_FAILURE;
}

goamdsmi_status_t check_amdgpu_driver()
{
    return is_file_present(AMDGPU_DRIVER_NAME, AMDGPU_INITSTATE_FILE);
}

goamdsmi_status_t check_hsmp_driver()
{
    return  is_file_present(AMDHSMP_DRIVER_NAME, AMDHSMP_INITSTATE_FILE);
}

goamdsmi_status_t go_shim_amdsmiapu_init(goamdsmi_Init_t goamdsmi_Init)
{
    if((GOAMDSMI_CPU_INIT == goamdsmi_Init) && (true == cpuInitCompleted))
    {
        if((GOAMDSMI_VALUE_0 == num_cpuSockets)||(GOAMDSMI_VALUE_0 == num_cpu_inAllSocket)||(GOAMDSMI_VALUE_0 == num_cpu_physicalCore_inAllSocket))
        {
            if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) {printf("AMDSMI, Failed, Returns previous enumurated AMDSMICPUInit:%d, CpuSocketCount:%d, CpuCount:%d, CpuPhysicalCoreCount:%d\n", GOAMDSMI_STATUS_FAILURE, num_cpuSockets, num_cpu_inAllSocket, num_cpu_physicalCore_inAllSocket);}
            return GOAMDSMI_STATUS_FAILURE;
        }
        else 
        {
            if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) {printf("AMDSMI, Success, Returns previous enumurated AMDSMICPUInit:%d, CpuSocketCount:%d, CpuCount:%d, CpuPhysicalCoreCount:%d\n", GOAMDSMI_STATUS_SUCCESS, num_cpuSockets, num_cpu_inAllSocket, num_cpu_physicalCore_inAllSocket);}
            return GOAMDSMI_STATUS_SUCCESS;
        }
    }
    
    if((GOAMDSMI_GPU_INIT == goamdsmi_Init) && (true == gpuInitCompleted))
    {
        if((GOAMDSMI_VALUE_0 == num_gpuSockets)||(GOAMDSMI_VALUE_0 == num_gpu_devices_inAllSocket))
        {
            if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) {printf("AMDSMI, Failed, Returns previous enumurated AMDSMIGPUInit:%d, GpuSocketCount:%d, GpuCount:%d\n", GOAMDSMI_STATUS_FAILURE, num_gpuSockets, num_gpu_devices_inAllSocket);}
            return GOAMDSMI_STATUS_FAILURE;
        }
        else
        {
            if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) {printf("AMDSMI, Success, Returns previous enumurated AMDSMIGPUInit:%d, GpuSocketCount:%d, GpuCount:%d\n", GOAMDSMI_STATUS_SUCCESS, num_gpuSockets, num_gpu_devices_inAllSocket);}
            return GOAMDSMI_STATUS_SUCCESS;
        }
    }

#if 0
    if(GOAMDSMI_STATUS_FAILURE == go_shim_amdsmi_present())
    {
        if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) {printf("AMDSMI, Failed, AMDSMI not present in the System, missing \"%s\" (or) \"%s\"\n", AMDSMI_LIB_FILE, AMDSMI_LIB64_FILE);}
        return GOAMDSMI_STATUS_FAILURE;
    }
#endif

    if ((GOAMDSMI_STATUS_SUCCESS == check_amdgpu_driver()) && (GOAMDSMI_STATUS_SUCCESS == check_hsmp_driver())) 
    {
        if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_2)) {printf("AMDSMI, Status, Identified APU machine and going to enumurate APU\n");}

        if( (AMDSMI_STATUS_SUCCESS == amdsmi_init(AMDSMI_INIT_AMD_APUS)) &&
            (AMDSMI_STATUS_SUCCESS == amdsmi_get_socket_handles(&num_apuSockets, nullptr)) &&
            (AMDSMI_STATUS_SUCCESS == amdsmi_get_socket_handles(&num_apuSockets, &amdsmi_apusocket_handle_all_socket[0])) &&
            (GOAMDSMI_VALUE_0 != num_apuSockets))
        {
            cpuInitCompleted = true;
            gpuInitCompleted = true;
            apuInitCompleted = true;
            if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) {printf("AMDSMI, Success, Identified APU machine ApuNumSockets=%d\n",num_apuSockets);}
            for(uint32_t socket_counter = 0; socket_counter < num_apuSockets; socket_counter++)
            {
                uint32_t num_cpu               = GOAMDSMI_VALUE_0;
                uint32_t num_cpu_physicalCores = GOAMDSMI_VALUE_0;
                uint32_t num_gpu_devices       = GOAMDSMI_VALUE_0;

                //CPU
                processor_type_t cpu_processor_type         = AMDSMI_PROCESSOR_TYPE_AMD_CPU;
                processor_type_t cpu_core_processor_type    = AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE;
                if( (AMDSMI_STATUS_SUCCESS == amdsmi_get_processor_handles_by_type(amdsmi_apusocket_handle_all_socket[socket_counter], cpu_processor_type, nullptr, &num_cpu)) &&
                    (GOAMDSMI_VALUE_0 != num_cpu) &&
                    (AMDSMI_STATUS_SUCCESS == amdsmi_get_processor_handles_by_type(amdsmi_apusocket_handle_all_socket[socket_counter], cpu_processor_type, &amdsmi_processor_handle_all_cpu_across_socket[num_cpu_inAllSocket], &num_cpu)))
                {
                    if( (AMDSMI_STATUS_SUCCESS == amdsmi_get_processor_handles_by_type(amdsmi_apusocket_handle_all_socket[socket_counter], cpu_core_processor_type, nullptr, &num_cpu_physicalCores)) &&
                        (GOAMDSMI_VALUE_0 != num_cpu_physicalCores) &&
                        (AMDSMI_STATUS_SUCCESS == amdsmi_get_processor_handles_by_type(amdsmi_apusocket_handle_all_socket[socket_counter], cpu_core_processor_type, &amdsmi_processor_handle_all_cpu_physicalCore_across_socket[num_cpu_physicalCore_inAllSocket], &num_cpu_physicalCores)))
                    {
                        num_cpu_physicalCore_inAllSocket = num_cpu_physicalCore_inAllSocket+num_cpu_physicalCores;
                    }
                    num_cpu_inAllSocket = num_cpu_inAllSocket+num_cpu;
                    num_cpuSockets = num_cpuSockets+1;
                }

                //GPU
                processor_type_t gpu_device_processor_type    = AMDSMI_PROCESSOR_TYPE_AMD_GPU;
                if( (AMDSMI_STATUS_SUCCESS == amdsmi_get_processor_handles_by_type(amdsmi_apusocket_handle_all_socket[socket_counter], gpu_device_processor_type, nullptr, &num_gpu_devices)) &&
                    (GOAMDSMI_VALUE_0 != num_gpu_devices) &&
                    (AMDSMI_STATUS_SUCCESS == amdsmi_get_processor_handles_by_type(amdsmi_apusocket_handle_all_socket[socket_counter], gpu_device_processor_type, &amdsmi_processor_handle_all_gpu_device_across_socket[num_gpu_devices_inAllSocket], &num_gpu_devices)))
                {
                     num_gpu_devices_inAllSocket = num_gpu_devices_inAllSocket+num_gpu_devices;
                     num_gpuSockets = num_gpuSockets+1;
                }
            }
        }
    }
    else if(GOAMDSMI_CPU_INIT == goamdsmi_Init)
    {
        if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_2)) {printf("AMDSMI, Status, Going to enumurate only CPU\n");}
        cpuInitCompleted = true;
        
        if (GOAMDSMI_STATUS_SUCCESS == check_hsmp_driver()) 
        {
            if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_2)) {printf("AMDSMI, Status, Identified CPU Driver and going to enumurate only CPU\n");}

            if( (AMDSMI_STATUS_SUCCESS != amdsmi_init(AMDSMI_INIT_AMD_CPUS)) ||
                (AMDSMI_STATUS_SUCCESS != amdsmi_get_socket_handles(&num_cpuSockets, nullptr)) || 
                (AMDSMI_STATUS_SUCCESS != amdsmi_get_socket_handles(&num_cpuSockets, &amdsmi_cpusocket_handle_all_socket[0])) ||
                (GOAMDSMI_VALUE_0 == num_cpuSockets))
            {
                if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) {printf("AMDSMI, Failed, AMDSMICPUInit:0, CpuNumSockets=0\n");}
                return GOAMDSMI_STATUS_FAILURE;
            }        
        }
        else
        {
            if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_3)) {printf("AMDSMI, Status, Missing CPU Driver and not going to enumurate only CPU\n");}
        }
        //CPU
        for(uint32_t cpu_socket_counter = 0; cpu_socket_counter < num_cpuSockets; cpu_socket_counter++)
        {
            uint32_t num_cpu               = GOAMDSMI_VALUE_0;
            uint32_t num_cpu_physicalCores = GOAMDSMI_VALUE_0;

            processor_type_t cpu_processor_type         = AMDSMI_PROCESSOR_TYPE_AMD_CPU;
            processor_type_t cpu_core_processor_type    = AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE;
            if( (AMDSMI_STATUS_SUCCESS == amdsmi_get_processor_handles_by_type(amdsmi_cpusocket_handle_all_socket[cpu_socket_counter], cpu_processor_type, nullptr, &num_cpu)) &&
                (GOAMDSMI_VALUE_0 != num_cpu) &&
                (AMDSMI_STATUS_SUCCESS == amdsmi_get_processor_handles_by_type(amdsmi_cpusocket_handle_all_socket[cpu_socket_counter], cpu_processor_type, &amdsmi_processor_handle_all_cpu_across_socket[num_cpu_inAllSocket], &num_cpu)))
            {
                if( (AMDSMI_STATUS_SUCCESS == amdsmi_get_processor_handles_by_type(amdsmi_cpusocket_handle_all_socket[cpu_socket_counter], cpu_core_processor_type, nullptr, &num_cpu_physicalCores)) &&
                    (GOAMDSMI_VALUE_0 != num_cpu_physicalCores) &&
                    (AMDSMI_STATUS_SUCCESS == amdsmi_get_processor_handles_by_type(amdsmi_cpusocket_handle_all_socket[cpu_socket_counter], cpu_core_processor_type, &amdsmi_processor_handle_all_cpu_physicalCore_across_socket[num_cpu_physicalCore_inAllSocket], &num_cpu_physicalCores)))
                {
                    num_cpu_physicalCore_inAllSocket = num_cpu_physicalCore_inAllSocket+num_cpu_physicalCores;
                }
                num_cpu_inAllSocket = num_cpu_inAllSocket+num_cpu;
            }
        }
    }
    else if(GOAMDSMI_GPU_INIT == goamdsmi_Init)    
    {
        if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_2)) {printf("AMDSMI, Status, Going to enumurate only GPU\n");}
        gpuInitCompleted = true;
        
        if (GOAMDSMI_STATUS_SUCCESS == check_amdgpu_driver()) 
        {
            if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_2)) {printf("AMDSMI, Status, Identified GPU Driver and going to enumurate only GPU\n");}

            if( (AMDSMI_STATUS_SUCCESS != amdsmi_init(AMDSMI_INIT_AMD_GPUS)) ||
                (AMDSMI_STATUS_SUCCESS != amdsmi_get_socket_handles(&num_gpuSockets, nullptr)) || 
                (AMDSMI_STATUS_SUCCESS != amdsmi_get_socket_handles(&num_gpuSockets, &amdsmi_gpusocket_handle_all_socket[0])) ||
                (GOAMDSMI_VALUE_0 == num_gpuSockets))
            {
                if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) {printf("AMDSMI, Failed, AMDSMIGPUInit:0, GpuNumSockets=0\n");}
                return GOAMDSMI_STATUS_FAILURE;
            }
        }
        else
        {
            if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_3)) {printf("AMDSMI, Status, Missing GPU Driver and not going to enumurate only GPU\n");}
        }
        
        //GPU
        for(uint32_t gpu_socket_counter = 0; gpu_socket_counter < num_gpuSockets; gpu_socket_counter++)
        {
            uint32_t num_gpu_devices       = GOAMDSMI_VALUE_0;
            
            processor_type_t gpu_device_processor_type    = AMDSMI_PROCESSOR_TYPE_AMD_GPU;
            if( (AMDSMI_STATUS_SUCCESS == amdsmi_get_processor_handles_by_type(amdsmi_gpusocket_handle_all_socket[gpu_socket_counter], gpu_device_processor_type, nullptr, &num_gpu_devices)) &&
                (GOAMDSMI_VALUE_0 != num_gpu_devices) &&
                (AMDSMI_STATUS_SUCCESS == amdsmi_get_processor_handles_by_type(amdsmi_gpusocket_handle_all_socket[gpu_socket_counter], gpu_device_processor_type, &amdsmi_processor_handle_all_gpu_device_across_socket[num_gpu_devices_inAllSocket], &num_gpu_devices)))
            {
                num_gpu_devices_inAllSocket = num_gpu_devices_inAllSocket+num_gpu_devices;
            }
        }
    }
    
    //CPU
    if((GOAMDSMI_CPU_INIT == goamdsmi_Init) && ((GOAMDSMI_VALUE_0 == num_cpuSockets)||(GOAMDSMI_VALUE_0 == num_cpu_inAllSocket)||(GOAMDSMI_VALUE_0 == num_cpu_physicalCore_inAllSocket)))
    {
        if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) {printf("AMDSMI, Failed, CPU Enumuration Failed AMDSMICPUInit:%d, CpuSocketCount:%d, CpuCount:%d, CpuPhysicalCoreCount:%d,\n", GOAMDSMI_STATUS_FAILURE, num_cpuSockets, num_cpu_inAllSocket, num_cpu_physicalCore_inAllSocket);}
        return GOAMDSMI_STATUS_FAILURE;
    }
    
    //GPU
    if((GOAMDSMI_GPU_INIT == goamdsmi_Init) && ((GOAMDSMI_VALUE_0 == num_gpuSockets)||(GOAMDSMI_VALUE_0 == num_gpu_devices_inAllSocket)))
    {
        if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) {printf("AMDSMI, Failed, GPU Enumuration Failed AMDSMIGPUInit:%d, GpuSocketCount:%d, GpuCount:%d\n", GOAMDSMI_STATUS_FAILURE, num_gpuSockets, num_gpu_devices_inAllSocket);}
        return GOAMDSMI_STATUS_FAILURE;
    }

    if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) 
    {
        if((GOAMDSMI_CPU_INIT == goamdsmi_Init) || apuInitCompleted)    printf("AMDSMI, Status, AMDSMICPUInit:%d, CpuSocketCount:%d, CpuCount:%d, CpuPhysicalCoreCount:%d,\n", GOAMDSMI_STATUS_SUCCESS, num_cpuSockets, num_cpu_inAllSocket, num_cpu_physicalCore_inAllSocket);
        if((GOAMDSMI_GPU_INIT == goamdsmi_Init) || apuInitCompleted)    printf("AMDSMI, Status, AMDSMIGPUInit:%d, GpuSocketCount:%d, GpuCount:%d\n", GOAMDSMI_STATUS_SUCCESS, num_gpuSockets, num_gpu_devices_inAllSocket);
    }
    
    return GOAMDSMI_STATUS_SUCCESS;
}
////////////////////////////////////////////////------------CPU------------////////////////////////////////////////////////
bool goamdsmi_cpu_init()    
{
	bool cpu_init_success = false;
    if(GOAMDSMI_STATUS_SUCCESS == go_shim_amdsmiapu_init(GOAMDSMI_CPU_INIT))
    {
        if((num_cpu_inAllSocket) && (num_cpu_physicalCore_inAllSocket)) cpu_init_success = true;
    }
	if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) {printf("AMDSMI, %s, InitAMDSMICPUInit:%d, CpuSocketCount:%d, CpuCount:%d, CpuPhysicalCoreCount:%d,\n", cpu_init_success?"Success":"Failed", cpu_init_success?1:0, num_cpuSockets, num_cpu_inAllSocket, num_cpu_physicalCore_inAllSocket);}
    return cpu_init_success;
}

uint32_t goamdsmi_cpu_threads_per_core_get()
{
	bool readSuccess                = false;
    uint32_t threads_per_core_temp  = GOAMDSMI_VALUE_0;

    if((AMDSMI_STATUS_SUCCESS == amdsmi_get_threads_per_core(&threads_per_core_temp))) readSuccess = true;
    if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) {printf("AMDSMI, %s, CpuThreadsPerCore:%lu\n", readSuccess?"Success":"Failed", (unsigned long)(threads_per_core_temp));}

    return threads_per_core_temp;
}

uint32_t goamdsmi_cpu_number_of_threads_get()
{
	bool readSuccess              = false;
    uint32_t number_of_threads    = GOAMDSMI_VALUE_0;
    uint32_t num_threads_per_core = goamdsmi_cpu_threads_per_core_get();
    if(0 != num_threads_per_core)
    {
		readSuccess = true;
        number_of_threads = num_cpu_physicalCore_inAllSocket*num_threads_per_core;
    }
    if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) {printf("AMDSMI, %s, CpuNumThreads:%lu\n", readSuccess?"Success":"Failed", (unsigned long)(number_of_threads));}
    return number_of_threads;
}

uint32_t goamdsmi_cpu_number_of_sockets_get()
{
	uint32_t number_of_sockets = num_cpuSockets;
    if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) {printf("AMDSMI, Success, CpuNumSockets:%lu\n", (unsigned long)(number_of_sockets));}
    return number_of_sockets;
}

uint64_t goamdsmi_cpu_core_energy_get(uint32_t thread_index)
{
	bool readSuccess = false;
    uint64_t core_energy_temp    = GOAMDSMI_UINT64_MAX;
    uint32_t physicalCore_index  = thread_index%num_cpu_physicalCore_inAllSocket;

    if (AMDSMI_STATUS_SUCCESS == amdsmi_get_cpu_core_energy(amdsmi_processor_handle_all_cpu_physicalCore_across_socket[physicalCore_index], &core_energy_temp)) readSuccess = true;
    if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) {printf("AMDSMI, %s for Thread:%d PC:%d, CpuCoreEnergy:%llu, CpuCoreEnergyJoules:%.6f, CpuCoreEnergyKJoules:%.9f\n", readSuccess?"Success":"Failed", thread_index, physicalCore_index, (unsigned long long)(core_energy_temp), ((double)(core_energy_temp))/1000000, ((double)(core_energy_temp))/1000000000);}

    return core_energy_temp;
}

uint64_t goamdsmi_cpu_socket_energy_get(uint32_t socket_index)
{
	bool readSuccess = false;
    uint64_t socket_energy_temp = GOAMDSMI_UINT64_MAX;
    if ((AMDSMI_STATUS_SUCCESS == amdsmi_get_cpu_socket_energy(amdsmi_processor_handle_all_cpu_across_socket[socket_index], &socket_energy_temp))) readSuccess = true;
    if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) {printf("AMDSMI, %s for Socket:%d, CpuSocketEnergy:%llu, CpuSocketEnergyJoules:%.6f, CpuSocketEnergyKJoules:%.9f\n", readSuccess?"Success":"Failed", socket_index, (unsigned long long)(socket_energy_temp), ((double)(socket_energy_temp))/1000000, ((double)(socket_energy_temp))/1000000000);}

    return socket_energy_temp;
}

uint32_t goamdsmi_cpu_prochot_status_get(uint32_t socket_index)
{
	bool readSuccess = false;
    uint32_t prochot_temp  = GOAMDSMI_UINT32_MAX;
    if ((AMDSMI_STATUS_SUCCESS == amdsmi_get_cpu_prochot_status(amdsmi_processor_handle_all_cpu_across_socket[socket_index], &prochot_temp))) readSuccess = true;
    if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) {printf("AMDSMI, %s for Socket:%d, CpuProchotStatus:%lu\n", readSuccess?"Success":"Failed", socket_index, (unsigned long)(prochot_temp));}

    return prochot_temp;
}

uint32_t goamdsmi_cpu_socket_power_get(uint32_t socket_index)
{
	bool readSuccess = false;
    uint32_t socket_power_temp = GOAMDSMI_UINT32_MAX;
    if ((AMDSMI_STATUS_SUCCESS == amdsmi_get_cpu_socket_power(amdsmi_processor_handle_all_cpu_across_socket[socket_index], &socket_power_temp))) readSuccess = true;
    if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) {printf("AMDSMI, %s for Socket:%d, CpuSocketPower:%lu, CpuSocketPowerWatt:%.3f\n", readSuccess?"Success":"Failed", socket_index, (unsigned long)(socket_power_temp), ((double)(socket_power_temp))/1000);}

    return socket_power_temp;
}

uint32_t goamdsmi_cpu_socket_power_cap_get(uint32_t socket_index)
{
	bool readSuccess = false;
    uint32_t socket_power_cap_temp = GOAMDSMI_UINT32_MAX;
    if ((AMDSMI_STATUS_SUCCESS == amdsmi_get_cpu_socket_power_cap(amdsmi_processor_handle_all_cpu_across_socket[socket_index], &socket_power_cap_temp))) readSuccess = true;
    if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) {printf("AMDSMI, %s for Socket:%d, CpuSocketPowerCap:%lu, CpuSocketPowerCapWatt:%.3f\n", readSuccess?"Success":"Failed", socket_index, (unsigned long)(socket_power_cap_temp), ((double)(socket_power_cap_temp))/1000);}

    return socket_power_cap_temp;
}

uint32_t goamdsmi_cpu_core_boostlimit_get(uint32_t thread_index)
{
	bool readSuccess = false;
    uint32_t core_boostlimit_temp  = GOAMDSMI_UINT32_MAX;
    uint32_t physicalCore_index    = thread_index%num_cpu_physicalCore_inAllSocket;

    if (AMDSMI_STATUS_SUCCESS == amdsmi_get_cpu_core_boostlimit(amdsmi_processor_handle_all_cpu_physicalCore_across_socket[physicalCore_index], &core_boostlimit_temp)) readSuccess = true;
    if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) {printf("AMDSMI, %s for Thread:%d PC:%d, CpuCoreBoostLimit:%lu\n", readSuccess?"Success":"Failed", thread_index, physicalCore_index, (unsigned long)(core_boostlimit_temp));}

    return core_boostlimit_temp;
}

////////////////////////////////////////////////------------GPU------------////////////////////////////////////////////////
bool goamdsmi_gpu_init()
{
	bool gpu_init_success = false;
    if(GOAMDSMI_STATUS_SUCCESS == go_shim_amdsmiapu_init(GOAMDSMI_GPU_INIT))
    {
        if((num_gpu_devices_inAllSocket)) gpu_init_success = true;
    }
    if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) {printf("AMDSMI, %s, InitAMDSMIGPUInit:%d, GpuSocketCount:%d, GpuCount:%d\n", gpu_init_success?"Success":"Failed", gpu_init_success?1:0, num_gpuSockets, num_gpu_devices_inAllSocket);}
	
    return gpu_init_success;
}

bool goamdsmi_gpu_shutdown()
{
    return false;
}

uint32_t goamdsmi_gpu_num_monitor_devices()
{
    uint32_t gpu_num_monitor_devices = num_gpu_devices_inAllSocket;
    if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) {printf("AMDSMI, Success, GpuNumMonitorDevices:%lu\n", (unsigned long)(gpu_num_monitor_devices));}
    return gpu_num_monitor_devices;
}

char* goamdsmi_gpu_dev_name_get(uint32_t dv_ind)
{
	uint32_t len = 256;
    char* dev_name = (char*)malloc(sizeof(char)*len);dev_name[0] = '\0';
    strcpy(dev_name, GOAMDSMI_STRING_NA);
	
    return dev_name;
}

uint16_t goamdsmi_gpu_dev_id_get(uint32_t dv_ind)
{
    bool readSuccess         = false;
    uint16_t gpu_dev_id_temp = GOAMDSMI_UINT16_MAX;
    
    if((dv_ind < num_gpu_devices_inAllSocket) && (AMDSMI_STATUS_SUCCESS == amdsmi_get_gpu_id(amdsmi_processor_handle_all_gpu_device_across_socket[dv_ind], &gpu_dev_id_temp))) readSuccess = true;
    if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) {printf("AMDSMI, %s for Gpu:%d, GpuDevId:%d\n", readSuccess?"Success":"Failed", dv_ind, gpu_dev_id_temp);}

    return gpu_dev_id_temp;
}

uint64_t goamdsmi_gpu_dev_pci_id_get(uint32_t dv_ind)
{
	uint64_t gpu_pci_id = GOAMDSMI_UINT64_MAX;
    return gpu_pci_id;
}

char* goamdsmi_gpu_dev_vendor_name_get(uint32_t dv_ind)
{
    uint32_t len = 256;
    char* gpu_vendor_name = (char*)malloc(sizeof(char)*len);gpu_vendor_name[0] = '\0';
    strcpy(gpu_vendor_name, GOAMDSMI_STRING_NA);
    
    return gpu_vendor_name;
}

char* goamdsmi_gpu_dev_vbios_version_get(uint32_t dv_ind)
{
    uint32_t len = 256;
    char* vbios_version = (char*)malloc(sizeof(char)*len);vbios_version[0] = '\0';
    strcpy(vbios_version, GOAMDSMI_STRING_NA);
    
    return vbios_version;
}

uint64_t goamdsmi_gpu_dev_power_cap_get(uint32_t dv_ind)
{
    bool readSuccess                                    = false;
    uint64_t gpu_power_cap                              = GOAMDSMI_UINT64_MAX;
    amdsmi_power_cap_info_t amdsmi_power_cap_info_temp  = {0};

    if((dv_ind < num_gpu_devices_inAllSocket) && (AMDSMI_STATUS_SUCCESS == amdsmi_get_power_cap_info(amdsmi_processor_handle_all_gpu_device_across_socket[dv_ind], GPU_SENSOR_0, &amdsmi_power_cap_info_temp)))
    {
        readSuccess = true;
        gpu_power_cap = amdsmi_power_cap_info_temp.power_cap;
    }
	if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) {printf("AMDSMI, %s for Gpu:%d, GpuPowerCap:%llu, GpuPowerCapInWatt:%.6f\n", readSuccess?"Success":"Failed", dv_ind, (unsigned long long)(gpu_power_cap), ((double)(gpu_power_cap))/1000000);}
    return gpu_power_cap;
}

uint64_t goamdsmi_gpu_dev_power_get(uint32_t dv_ind)
{
    uint64_t gpu_power                          = GOAMDSMI_UINT64_MAX;
    uint64_t gpu_power_temp                     = GOAMDSMI_UINT64_MAX;
    amdsmi_power_info_t amdsmi_power_info_temp  = {0};

    if((dv_ind < num_gpu_devices_inAllSocket) && (AMDSMI_STATUS_SUCCESS == amdsmi_get_power_info(amdsmi_processor_handle_all_gpu_device_across_socket[dv_ind], &amdsmi_power_info_temp)))
    {
        gpu_power_temp = amdsmi_power_info_temp.average_socket_power;
        if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_2)) {printf("AMDSMI, Success for Gpu:%d, GpuPowerAverage:%llu, GpuPowerAverageinWatt:%.6f\n", dv_ind, (unsigned long long)(gpu_power_temp), ((double)(gpu_power_temp))/1000000);}

        if(MAX_GPU_POWER_FROM_DRIVER == gpu_power_temp)
        {
            gpu_power_temp = amdsmi_power_info_temp.current_socket_power;
            if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_2)) {printf("AMDSMI, Success for Gpu:%d, GpuPowerCurrent:%llu, GpuPowerCurrentinWatt:%.6f\n", dv_ind, (unsigned long long)(gpu_power_temp), ((double)(gpu_power_temp))/1000000);}
        }
        gpu_power = gpu_power_temp;
        gpu_power = (gpu_power)*1000000;//to maintain backward compatibity with old ROCM SMI
        if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) {printf("AMDSMI, Success for Gpu:%d, GpuPower:%llu, GpuPowerinWatt:%.6f\n", dv_ind, (unsigned long long)(gpu_power), ((double)(gpu_power))/1000000);}
        return gpu_power;
    }

    amdsmi_gpu_metrics_t metrics = {0};
    if((dv_ind < num_gpu_devices_inAllSocket) && (AMDSMI_STATUS_SUCCESS == amdsmi_get_gpu_metrics_info(amdsmi_processor_handle_all_gpu_device_across_socket[dv_ind], &metrics)))
    {
        gpu_power_temp = metrics.average_socket_power;
        if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_2)) {printf("AMDSMI, Success for Gpu:%d, GpuPowerAverageFromMetrics:%llu, GpuPowerAverageFromMetricsinWatt:%.6f\n", dv_ind, (unsigned long long)gpu_power_temp, ((double)(gpu_power_temp))/1000000);}

        if(MAX_GPU_POWER_FROM_DRIVER == gpu_power_temp)
        {
            gpu_power_temp = metrics.current_socket_power;
            if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_2)) {printf("AMDSMI, Success for Gpu:%d, GpuPowerCurrentFromMetrics:%llu, GpuPowerCurrentFromMetricsinWatt:%.6f\n", dv_ind, (unsigned long long)gpu_power_temp, ((double)(gpu_power_temp))/1000000);}
        }
        gpu_power = gpu_power_temp;
        gpu_power = (gpu_power)*1000000;//to maintain backward compatibity with old ROCM SMI
        if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) {printf("AMDSMI, Success for Gpu:%d, GpuPowerFromMetrics:%llu, GpuPowerFromMetricsinWatt:%.6f\n", dv_ind, (unsigned long long)(gpu_power), ((double)(gpu_power))/1000000);}
        return gpu_power;
    }

    if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) {printf("AMDSMI, Failed for Gpu:%d, GpuPower:%llu, GpuPowerinWatt:%.6f\n", dv_ind, (unsigned long long)(gpu_power), ((double)(gpu_power))/1000000);}
    return gpu_power;
}

uint64_t goamdsmi_gpu_dev_temp_metric_get(uint32_t dv_ind, uint32_t sensor, uint32_t metric)
{
    bool readSuccess             = false;
    uint64_t gpu_temperature      = GOAMDSMI_UINT64_MAX;
    uint64_t gpu_temperature_temp = GOAMDSMI_UINT64_MAX;

    if((dv_ind < num_gpu_devices_inAllSocket) && (AMDSMI_STATUS_SUCCESS == amdsmi_get_temp_metric(amdsmi_processor_handle_all_gpu_device_across_socket[dv_ind], sensor, metric, &gpu_temperature_temp)))
    {
        readSuccess = true;
        gpu_temperature = gpu_temperature_temp;
        gpu_temperature = (gpu_temperature)*1000;//to maintain backward compatibity with old ROCM SMI
        if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) {printf("AMDSMI, %s for Gpu:%d Sensor:%d Metric:%d, GpuTemperature:%llu, GpuTemperatureInDegree:%.3f\n", readSuccess?"Success":"Failed", dv_ind, sensor, metric, (unsigned long long)(gpu_temperature), ((double)(gpu_temperature))/1000);}
    }
    return gpu_temperature;
}

uint32_t goamdsmi_gpu_dev_overdrive_level_get(uint32_t dv_ind)
{
	uint32_t gpu_overdrive_level = GOAMDSMI_UINT32_MAX;
    return gpu_overdrive_level;
}

uint32_t goamdsmi_gpu_dev_mem_overdrive_level_get(uint32_t dv_ind)
{
	uint32_t gpu_mem_overdrive_level = GOAMDSMI_UINT32_MAX;
    return gpu_mem_overdrive_level;
}

uint32_t goamdsmi_gpu_dev_perf_level_get(uint32_t dv_ind)
{
	uint32_t gpu_perf = GOAMDSMI_UINT32_MAX;
    return gpu_perf;
}

uint64_t goamdsmi_gpu_dev_gpu_clk_freq_get_sclk(uint32_t dv_ind)
{
    bool readSuccess          = false;
    uint64_t gpu_sclk_freq    = GOAMDSMI_UINT64_MAX;
    amdsmi_frequencies_t freq = {0};

    if((dv_ind < num_gpu_devices_inAllSocket) && (AMDSMI_STATUS_SUCCESS == amdsmi_get_clk_freq(amdsmi_processor_handle_all_gpu_device_across_socket[dv_ind], AMDSMI_CLK_TYPE_SYS, &freq)))
    {
        readSuccess = true;
        gpu_sclk_freq = freq.frequency[freq.current];
    }
    if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) {printf("AMDSMI, %s for Gpu:%d, GpuSclkFreq:%llu, GpuSclkFreqMhz:%.6f\n", readSuccess?"Success":"Failed", dv_ind, (unsigned long long)(gpu_sclk_freq), ((double)(gpu_sclk_freq))/1000000);}

    return gpu_sclk_freq;
}

uint64_t goamdsmi_gpu_dev_gpu_clk_freq_get_mclk(uint32_t dv_ind)
{
    bool readSuccess          = false;
    uint64_t gpu_memclk_freq  = GOAMDSMI_UINT64_MAX;
    amdsmi_frequencies_t freq = {0};

    if((dv_ind < num_gpu_devices_inAllSocket) && (AMDSMI_STATUS_SUCCESS == amdsmi_get_clk_freq(amdsmi_processor_handle_all_gpu_device_across_socket[dv_ind], AMDSMI_CLK_TYPE_MEM, &freq)))
    {
        readSuccess = true;
        gpu_memclk_freq = freq.frequency[freq.current];
    }
    if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) {printf("AMDSMI, %s for Gpu:%d, GpuMclkFreq:%llu, GpuMclkFreqMhz:%.6f\n", readSuccess?"Success":"Failed", dv_ind, (unsigned long long)(gpu_memclk_freq), ((double)(gpu_memclk_freq))/1000000);}

    return gpu_memclk_freq;
}

uint64_t goamdsmi_gpu_od_volt_freq_range_min_get_sclk(uint32_t dv_ind)
{
	uint64_t gpu_min_sclk = GOAMDSMI_UINT64_MAX;
    return gpu_min_sclk;
}

uint64_t goamdsmi_gpu_od_volt_freq_range_min_get_mclk(uint32_t dv_ind)
{
    uint64_t gpu_min_memclk = GOAMDSMI_UINT64_MAX;
    return gpu_min_memclk;
}

uint64_t goamdsmi_gpu_od_volt_freq_range_max_get_sclk(uint32_t dv_ind)
{
    uint64_t gpu_max_sclk = GOAMDSMI_UINT64_MAX;
    return gpu_max_sclk;
}

uint64_t goamdsmi_gpu_od_volt_freq_range_max_get_mclk(uint32_t dv_ind)
{
    uint64_t gpu_max_memclk = GOAMDSMI_UINT64_MAX;
    return gpu_max_memclk;
}

uint32_t goamdsmi_gpu_dev_gpu_busy_percent_get(uint32_t dv_ind)
{
    bool readSuccess                = false;
    uint32_t gpu_busy_percent       = GOAMDSMI_UINT32_MAX;
    amdsmi_engine_usage_t amdsmi_engine_usage_temp;

    if(AMDSMI_STATUS_SUCCESS == amdsmi_get_gpu_activity(amdsmi_processor_handle_all_gpu_device_across_socket[dv_ind], &amdsmi_engine_usage_temp))
    {
        readSuccess = true;
        gpu_busy_percent = amdsmi_engine_usage_temp.gfx_activity;
    }
    if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) {printf("AMDSMI, %s for Gpu:%d, GpuBusyPerc:%lu\n", readSuccess?"Success":"Failed", dv_ind, (unsigned long)(gpu_busy_percent));}

    return gpu_busy_percent;
}

uint64_t goamdsmi_gpu_dev_gpu_memory_busy_percent_get(uint32_t dv_ind)
{
    bool readSuccess                 = false;
    uint64_t gpu_memory_busy_percent = GOAMDSMI_UINT64_MAX;
    uint64_t gpu_memory_usage_temp   = GOAMDSMI_UINT64_MAX;
    uint64_t gpu_memory_total_temp   = GOAMDSMI_UINT64_MAX;

    if( (AMDSMI_STATUS_SUCCESS == amdsmi_get_gpu_memory_usage(amdsmi_processor_handle_all_gpu_device_across_socket[dv_ind], AMDSMI_MEM_TYPE_VRAM, &gpu_memory_usage_temp))&&
        (AMDSMI_STATUS_SUCCESS == amdsmi_get_gpu_memory_total(amdsmi_processor_handle_all_gpu_device_across_socket[dv_ind], AMDSMI_MEM_TYPE_VRAM, &gpu_memory_total_temp)))
    {
        readSuccess = true;
        gpu_memory_busy_percent = (uint64_t)(gpu_memory_usage_temp*100)/gpu_memory_total_temp;
    }
	if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) {printf("AMDSMI, %s for Gpu:%d, GpuMemoryBusyPerc:%llu\n", readSuccess?"Success":"Failed", dv_ind, (unsigned long long)(gpu_memory_busy_percent));}

    return gpu_memory_busy_percent;
}

uint64_t goamdsmi_gpu_dev_gpu_memory_usage_get(uint32_t dv_ind)
{
    bool readSuccess               = false;
    uint64_t gpu_memory_usage      = GOAMDSMI_UINT64_MAX;
    uint64_t gpu_memory_usage_temp = GOAMDSMI_UINT64_MAX;

    if(AMDSMI_STATUS_SUCCESS == amdsmi_get_gpu_memory_usage(amdsmi_processor_handle_all_gpu_device_across_socket[dv_ind], AMDSMI_MEM_TYPE_VRAM, &gpu_memory_usage_temp))
    {
        readSuccess = true;
        gpu_memory_usage = (uint64_t)gpu_memory_usage_temp;
    }
    if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) {printf("AMDSMI, %s for Gpu:%d, GpuMemoryUsage:%llu\n", readSuccess?"Success":"Failed", dv_ind, (unsigned long long)(gpu_memory_usage));}
        
    return gpu_memory_usage;
}

uint64_t goamdsmi_gpu_dev_gpu_memory_total_get(uint32_t dv_ind)
{
    bool readSuccess               = false;
    uint64_t gpu_memory_total      = GOAMDSMI_UINT64_MAX;
    uint64_t gpu_memory_total_temp = GOAMDSMI_UINT64_MAX;

    if(AMDSMI_STATUS_SUCCESS == amdsmi_get_gpu_memory_total(amdsmi_processor_handle_all_gpu_device_across_socket[dv_ind], AMDSMI_MEM_TYPE_VRAM, &gpu_memory_total_temp))
    {
        readSuccess = true;
        gpu_memory_total = (uint64_t)gpu_memory_total_temp;
    }
    if (enable_debug_level(GOAMDSMI_DEBUG_LEVEL_1)) {printf("AMDSMI, %s for Gpu:%d, GpuMemoryTotal:%llu\n", readSuccess?"Success":"Failed", dv_ind, (unsigned long long)(gpu_memory_total));}
        
    return gpu_memory_total;
}
#else
////////////////////////////////////////////////------------CPU------------////////////////////////////////////////////////
bool goamdsmi_cpu_init()                                           {return false;}
uint32_t goamdsmi_cpu_threads_per_core_get()                       {return GOAMDSMI_VALUE_0;}
uint32_t goamdsmi_cpu_number_of_threads_get()                      {return GOAMDSMI_VALUE_0;}
uint32_t goamdsmi_cpu_number_of_sockets_get()                      {return GOAMDSMI_VALUE_0;}
uint64_t goamdsmi_cpu_core_energy_get(uint32_t thread_index)       {return GOAMDSMI_UINT64_MAX;}
uint64_t goamdsmi_cpu_socket_energy_get(uint32_t socket_index)     {return GOAMDSMI_UINT64_MAX;}
uint32_t goamdsmi_cpu_prochot_status_get(uint32_t socket_index)    {return GOAMDSMI_UINT32_MAX;}
uint32_t goamdsmi_cpu_socket_power_get(uint32_t socket_index)      {return GOAMDSMI_UINT32_MAX;}
uint32_t goamdsmi_cpu_socket_power_cap_get(uint32_t socket_index)  {return GOAMDSMI_UINT32_MAX;}
uint32_t goamdsmi_cpu_core_boostlimit_get(uint32_t thread_index)   {return GOAMDSMI_UINT32_MAX;}

////////////////////////////////////////////////------------GPU------------////////////////////////////////////////////////
bool goamdsmi_gpu_init()                                           {return false;}
bool goamdsmi_gpu_shutdown()                                       {return false;}
uint32_t goamdsmi_gpu_num_monitor_devices()                        {return GOAMDSMI_VALUE_0;}
char* goamdsmi_gpu_dev_name_get(uint32_t dv_ind)                   {return NULL;}
uint16_t goamdsmi_gpu_dev_id_get(uint32_t dv_ind)                  {return GOAMDSMI_UINT16_MAX;}
uint64_t goamdsmi_gpu_dev_pci_id_get(uint32_t dv_ind)              {return GOAMDSMI_UINT64_MAX;}
char* goamdsmi_gpu_dev_vendor_name_get(uint32_t dv_ind)            {return NULL;}
char* goamdsmi_gpu_dev_vbios_version_get(uint32_t dv_ind)          {return NULL;}
uint64_t goamdsmi_gpu_dev_power_cap_get(uint32_t dv_ind)           {return GOAMDSMI_UINT64_MAX;}
uint64_t goamdsmi_gpu_dev_power_get(uint32_t dv_ind)               {return GOAMDSMI_UINT64_MAX;}
uint64_t goamdsmi_gpu_dev_temp_metric_get(uint32_t dv_ind, uint32_t sensor, uint32_t metric)    {return GOAMDSMI_UINT64_MAX;}
uint32_t goamdsmi_gpu_dev_overdrive_level_get(uint32_t dv_ind)     {return GOAMDSMI_UINT32_MAX;}
uint32_t goamdsmi_gpu_dev_mem_overdrive_level_get(uint32_t dv_ind) {return GOAMDSMI_UINT32_MAX;}
uint32_t goamdsmi_gpu_dev_perf_level_get(uint32_t dv_ind)          {return GOAMDSMI_UINT32_MAX;}
uint64_t goamdsmi_gpu_dev_gpu_clk_freq_get_sclk(uint32_t dv_ind)           {return GOAMDSMI_UINT64_MAX;}
uint64_t goamdsmi_gpu_dev_gpu_clk_freq_get_mclk(uint32_t dv_ind)           {return GOAMDSMI_UINT64_MAX;}
uint64_t goamdsmi_gpu_od_volt_freq_range_min_get_sclk(uint32_t dv_ind)     {return GOAMDSMI_UINT64_MAX;}
uint64_t goamdsmi_gpu_od_volt_freq_range_min_get_mclk(uint32_t dv_ind)     {return GOAMDSMI_UINT64_MAX;}
uint64_t goamdsmi_gpu_od_volt_freq_range_max_get_sclk(uint32_t dv_ind)     {return GOAMDSMI_UINT64_MAX;}
uint64_t goamdsmi_gpu_od_volt_freq_range_max_get_mclk(uint32_t dv_ind)     {return GOAMDSMI_UINT64_MAX;}
uint32_t goamdsmi_gpu_dev_gpu_busy_percent_get(uint32_t dv_ind)            {return GOAMDSMI_UINT32_MAX;}
uint64_t goamdsmi_gpu_dev_gpu_memory_busy_percent_get(uint32_t dv_ind)     {return GOAMDSMI_UINT64_MAX;}
uint64_t goamdsmi_gpu_dev_gpu_memory_usage_get(uint32_t dv_ind)            {return GOAMDSMI_UINT64_MAX;}
uint64_t goamdsmi_gpu_dev_gpu_memory_total_get(uint32_t dv_ind)            {return GOAMDSMI_UINT64_MAX;}
#endif
