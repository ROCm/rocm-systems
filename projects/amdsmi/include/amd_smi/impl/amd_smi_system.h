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

#ifndef AMD_SMI_INCLUDE_AMD_SMI_SYSTEM_H_
#define AMD_SMI_INCLUDE_AMD_SMI_SYSTEM_H_

#include <vector>
#include <set>
#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_socket.h"
#include "amd_smi/impl/amd_smi_processor.h"
#include "amd_smi/impl/amd_smi_drm.h"

namespace amd::smi {

// Singleton: Only one system in an application
class AMDSmiSystem {
 public:
    static AMDSmiSystem& getInstance() {
        static AMDSmiSystem instance;
        return instance;
    }
    amdsmi_status_t init(uint64_t flags);
    amdsmi_status_t cleanup();

    std::vector<AMDSmiSocket*>& get_sockets() {return sockets_;}

    amdsmi_status_t handle_to_socket(amdsmi_socket_handle socket_handle,
            AMDSmiSocket** socket);

    amdsmi_status_t handle_to_processor(amdsmi_processor_handle processor_handle,
            AMDSmiProcessor** device);

    amdsmi_status_t gpu_index_to_handle(uint32_t gpu_index,
                    amdsmi_processor_handle* processor_handle);

    amdsmi_status_t get_cpu_family(uint32_t *cpu_family);

    amdsmi_status_t get_cpu_model(uint32_t *cpu_model);

    amdsmi_status_t get_cpu_model_name(uint32_t socket_id, std::string *model_name);

    amdsmi_status_t get_sys_cpu_cores_per_socket(uint32_t *core_num) ;

    amdsmi_status_t get_sys_num_of_cpu_sockets(uint32_t *sock_num);

    std::vector<uint32_t> get_cpu_sockets_from_numa_node(int32_t numa_node);
 private:
    AMDSmiSystem() : init_flag_(AMDSMI_INIT_AMD_GPUS) {}

    /* The GPU socket id is used to identify the socket, so that the XCDs
    on the same physical device will be collected under the same socket.
    The BD part of the BDF is used as GPU socket to represent a phyiscal device.
    */
    amdsmi_status_t get_gpu_socket_id(uint32_t index, std::string& socketid);
    amdsmi_status_t populate_amd_gpu_devices();
    amdsmi_status_t populate_amd_cpus();
    uint64_t init_flag_;
    AMDSmiDrm drm_;
    std::vector<AMDSmiSocket*> sockets_;
    std::set<AMDSmiProcessor*> processors_;     // Track valid processors
};
} // namespace amd::smi

#endif  // AMD_SMI_INCLUDE_AMD_SMI_SYSTEM_H_
