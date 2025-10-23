/*
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */
#ifndef INCLUDE_ROCM_SMI_ROCM_SMI_KFD_H_
#define INCLUDE_ROCM_SMI_ROCM_SMI_KFD_H_

#include <string>
#include <vector>
#include <unordered_set>
#include <memory>
#include <map>

#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_device.h"
#include "rocm_smi/rocm_smi_io_link.h"

namespace amd {
namespace smi {

class KFDNode {
 public:
    explicit KFDNode(uint32_t node_ind) : node_indx_(node_ind) {}
    ~KFDNode();

    int Initialize();
    int ReadProperties(void);
    int get_property_value(std::string property, uint64_t *value);
    uint64_t gpu_id(void) const {return gpu_id_;}
    std::string name(void) const {return name_;}
    uint32_t node_index(void) const {return node_indx_;}
    uint32_t numa_node_number(void) const {return numa_node_number_;}
    uint64_t numa_node_weight(void) const {return numa_node_weight_;}
    uint64_t xgmi_hive_id(void) const {return xgmi_hive_id_;}
    uint32_t cu_count(void) const {return cu_count_;}
    IO_LINK_TYPE numa_node_type(void) const {return numa_node_type_;}
    int get_io_link_type(uint32_t node_to, IO_LINK_TYPE *type);
    int get_io_link_weight(uint32_t node_to, uint64_t *weight);
    int get_io_link_bandwidth(uint32_t node_to, uint64_t *max_bandwidth,
                                                       uint64_t *min_bandwidth);
    std::shared_ptr<Device> amdgpu_device(void) const {return amdgpu_device_;}
    uint32_t amdgpu_dev_index(void) const {return amdgpu_dev_index_;}
    void set_amdgpu_dev_index(uint32_t val) {amdgpu_dev_index_ = val;}

    // Get memory from kfd
    int get_total_memory(uint64_t* total);
    int get_used_memory(uint64_t* used);

    // Get gfx target version from kfd
    int get_gfx_target_version(uint64_t* gfx_target_version);
    // Get gpu_id (AKA GUID) version from kfd
    int get_gpu_id(uint64_t *gpu_id);
    // Get node id from kfd
    int get_node_id(uint32_t *node_id);

 private:
    uint32_t node_indx_;
    uint32_t amdgpu_dev_index_;
    uint64_t gpu_id_;
    std::string name_;
    uint32_t numa_node_number_;
    uint64_t numa_node_weight_;
    IO_LINK_TYPE numa_node_type_;
    uint64_t xgmi_hive_id_;
    uint32_t cu_count_;
    std::map<uint32_t, IO_LINK_TYPE> io_link_type_;
    std::map<uint32_t, uint64_t> io_link_weight_;
    std::map<uint32_t, uint64_t> io_link_max_bandwidth_;
    std::map<uint32_t, uint64_t> io_link_min_bandwidth_;
    std::map<uint32_t, std::shared_ptr<IOLink>> io_link_map_;
    std::map<std::string, uint64_t> properties_;
    std::shared_ptr<Device> amdgpu_device_;
};

int
DiscoverKFDNodes(std::map<uint64_t, std::shared_ptr<KFDNode>> *nodes);

int
GetProcessInfo(rsmi_process_info_t *procs, uint32_t num_allocated,
                                                   uint32_t *num_procs_found);
int
GetProcessInfoForPID(uint32_t pid, rsmi_process_info_t *proc,
                     std::unordered_set<uint64_t> *gpu_set);

int
GetProcessGPUs(uint32_t pid, std::unordered_set<uint64_t> *gpu_count);
int
ReadKFDDeviceProperties(uint32_t dev_id, std::vector<std::string> *retVec);

int read_node_properties(uint32_t node, std::string property_name,
                         uint64_t *val);
int get_gpu_id(uint32_t node, uint64_t *gpu_id);

}  // namespace smi
}  // namespace amd

#endif  // INCLUDE_ROCM_SMI_ROCM_SMI_KFD_H_
