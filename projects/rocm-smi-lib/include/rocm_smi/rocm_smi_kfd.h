/*
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2019, Advanced Micro Devices, Inc.
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
    std::shared_ptr<Device> amdgpu_device(void) const {return amdgpu_device_;}
    uint32_t amdgpu_dev_index(void) const {return amdgpu_dev_index_;}
    void set_amdgpu_dev_index(uint32_t val) {amdgpu_dev_index_ = val;}

 private:
    uint32_t node_indx_;
    uint32_t amdgpu_dev_index_;
    uint64_t gpu_id_;
    std::string name_;
    std::map<std::string, uint64_t> properties_;
    std::shared_ptr<Device> amdgpu_device_;
};

int
DiscoverKFDNodes(std::map<uint64_t, std::shared_ptr<KFDNode>> *nodes);

int
GetProcessInfo(rsmi_process_info_t *procs, uint32_t num_allocated,
                                                   uint32_t *num_procs_found);
int
GetProcessInfoForPID(uint32_t pid, rsmi_process_info_t *proc);

int
GetProcessGPUs(uint32_t pid, std::unordered_set<uint64_t> *gpu_count);
int
ReadKFDDeviceProperties(uint32_t dev_id, std::vector<std::string> *retVec);

}  // namespace smi
}  // namespace amd

#endif  // INCLUDE_ROCM_SMI_ROCM_SMI_KFD_H_
