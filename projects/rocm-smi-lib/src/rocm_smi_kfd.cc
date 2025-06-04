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

#include <dirent.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <regex>

#include "rocm_smi/rocm_smi_io_link.h"
#include "rocm_smi/rocm_smi_kfd.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_exception.h"
#include "rocm_smi/rocm_smi_utils.h"
#include "rocm_smi/rocm_smi_main.h"
#include "rocm_smi/rocm_smi_logger.h"

namespace amd {
namespace smi {

static const char *kKFDProcPathRoot = "/sys/class/kfd/kfd/proc";
static const char *kKFDNodesPathRoot = "/sys/class/kfd/kfd/topology/nodes";
// Sysfs file names
static const char *kKFDPasidFName = "pasid";



// KFD Node Property strings
// static const char *kKFDNodePropCPU_CORES_COUNTStr =    "cpu_cores_count";
// static const char *kKFDNodePropSIMD_COUNTStr =         "simd_count";
// static const char *kKFDNodePropMEM_BANKS_COUNTStr =    "mem_banks_count";
// static const char *kKFDNodePropCACHES_COUNTStr =       "caches_count";
// static const char *kKFDNodePropIO_LINKS_COUNTStr =     "io_links_count";
// static const char *kKFDNodePropCPU_CORE_ID_BASEStr =   "cpu_core_id_base";
// static const char *kKFDNodePropSIMD_ID_BASEStr =       "simd_id_base";
// static const char *kKFDNodePropMAX_WAVES_PER_SIMDStr = "max_waves_per_simd";
// static const char *kKFDNodePropLDS_SIZE_IN_KBStr =     "lds_size_in_kb";
// static const char *kKFDNodePropGDS_SIZE_IN_KBStr =     "gds_size_in_kb";
// static const char *kKFDNodePropNUM_GWSStr =            "num_gws";
// static const char *kKFDNodePropWAVE_FRONT_SIZEStr =    "wave_front_size";

static const char *kKFDNodePropARRAY_COUNTStr = "array_count";
static const char *kKFDNodePropSIMD_ARRAYS_PER_ENGINEStr =
                                                     "simd_arrays_per_engine";
static const char *kKFDNodePropCU_PER_SIMD_ARRAYStr = "cu_per_simd_array";
// static const char *kKFDNodePropSIMD_PER_CUStr = "simd_per_cu";
// static const char *kKFDNodePropMAX_SLOTS_SCRATCH_CUStr =
//                                                     "max_slots_scratch_cu";

// static const char *kKFDNodePropVENDOR_IDStr =          "vendor_id";
// static const char *kKFDNodePropDEVICE_IDStr =          "device_id";
static const char *kKFDNodePropLOCATION_IDStr =          "location_id";
static const char *kKFDNodePropDOMAINStr =               "domain";
// static const char *kKFDNodePropDRM_RENDER_MINORStr =   "drm_render_minor";
static const char *kKFDNodePropHIVE_IDStr =            "hive_id";
// static const char *kKFDNodePropNUM_SDMA_ENGINESStr =   "num_sdma_engines";
// static const char *kKFDNodePropNUM_SDMA_XGMI_ENGINESStr =
//                                                   "num_sdma_xgmi_engines";
// static const char *kKFDNodePropNUM_SDMA_QUEUES_PER_ENGINEStr =
//                                              "num_sdma_queues_per_engine";
// static const char *kKFDNodePropNUM_CP_QUEUESStr =      "num_cp_queues";
// static const char *kKFDNodePropMAX_ENGINE_CLK_FCOMPUTEStr =
//                                                 "max_engine_clk_fcompute";
// static const char *kKFDNodePropLOCAL_MEM_SIZEStr =     "local_mem_size";
// static const char *kKFDNodePropFW_VERSIONStr =         "fw_version";
// static const char *kKFDNodePropCAPABILITYStr =         "capability";
// static const char *kKFDNodePropDEBUG_PROPStr =         "debug_prop";
// static const char *kKFDNodePropSDMA_FW_VERSIOStr =     "sdma_fw_versio";
// static const char *kKFDNodePropMAX_ENGINE_CLK_CCOMPUTEStr =
//                                                "max_engine_clk_ccompute";

static bool is_number(const std::string &s) {
  return !s.empty() && std::all_of(s.begin(), s.end(), ::isdigit);
}

static std::string KFDDevicePath(uint32_t dev_id) {
  std::string node_path = kKFDNodesPathRoot;
  node_path += '/';
  node_path += std::to_string(dev_id);
  return node_path;
}

static int OpenKFDNodeFile(uint32_t dev_id, std::string node_file,
                                                          std::ifstream *fs) {
  std::string line;
  int ret;
  std::string f_path;
  bool reg_file;

  assert(fs != nullptr);

  f_path = KFDDevicePath(dev_id);
  f_path += "/";
  f_path += node_file;

  ret = isRegularFile(f_path, &reg_file);

  if (ret != 0) {
    return ret;
  }
  if (!reg_file) {
    return ENOENT;
  }

  fs->open(f_path);

  if (!fs->is_open()) {
      return errno;
  }

  return 0;
}

bool KFDNodeSupported(uint32_t node_indx) {
  std::ifstream fs;
  bool ret = true;
  int err;
  err = OpenKFDNodeFile(node_indx, "properties", &fs);

  if (err == ENOENT) {
    return false;
  }
  if (fs.peek() == std::ifstream::traits_type::eof()) {
    ret = false;
  }
  fs.close();
  return ret;
}

int ReadKFDDeviceProperties(uint32_t kfd_node_id,
                                           std::vector<std::string> *retVec) {
  std::string line;
  int ret;
  std::ifstream fs;
  std::string properties_path;
  std::ostringstream ss;

  assert(retVec != nullptr);

  ret = OpenKFDNodeFile(kfd_node_id, "properties", &fs);

  if (ret) {
    return ret;
  }

  ss << __PRETTY_FUNCTION__ << " | properties file contains = {";
  while (std::getline(fs, line)) {
    retVec->push_back(line);
    ss << line << ",\n";
  }
  ss << "}";
  // Leaving below to debug any future properties file changes
  // LOG_DEBUG(ss);

  if (retVec->empty()) {
    fs.close();
    return ENOENT;
  }
  // Remove any *trailing* empty (whitespace) lines
  while (retVec->back().find_first_not_of(" \t\n\v\f\r") == std::string::npos) {
    retVec->pop_back();
  }

  fs.close();
  return 0;
}

static int ReadKFDGpuId(uint32_t kfd_node_id, uint64_t *gpu_id) {
  std::string line;
  int ret;
  std::ifstream fs;
  std::string gpu_id_str;

  assert(gpu_id != nullptr);

  ret = OpenKFDNodeFile(kfd_node_id, "gpu_id", &fs);

  if (ret) {
    fs.close();
    return ret;
  }

  std::stringstream ss;
  ss << fs.rdbuf();
  fs.close();

  gpu_id_str = ss.str();

  gpu_id_str.erase(std::remove(gpu_id_str.begin(), gpu_id_str.end(), '\n'),
                                                            gpu_id_str.end());

  if (!is_number(gpu_id_str)) {
    return ENXIO;
  }

  *gpu_id = static_cast<uint64_t>(std::stoi(gpu_id_str));
  return 0;
}

static int ReadKFDGpuName(uint32_t kfd_node_id, std::string *gpu_name) {
  std::string line;
  int ret;
  std::ifstream fs;

  assert(gpu_name != nullptr);

  ret = OpenKFDNodeFile(kfd_node_id, "name", &fs);

  if (ret) {
    fs.close();
    return ret;
  }

  std::stringstream ss;
  ss << fs.rdbuf();
  fs.close();

  *gpu_name = ss.str();

  gpu_name->erase(std::remove(gpu_name->begin(), gpu_name->end(), '\n'),
                                                             gpu_name->end());

  return 0;
}

int GetProcessInfo(rsmi_process_info_t *procs, uint32_t num_allocated,
                                                  uint32_t *num_procs_found) {
  assert(num_procs_found != nullptr);

  *num_procs_found = 0;
  errno = 0;
  auto proc_dir = opendir(kKFDProcPathRoot);

  if (proc_dir == nullptr) {
    perror("Unable to open process directory");
    return errno;
  }
  auto dentry = readdir(proc_dir);

  std::string proc_id_str;
  std::string tmp;

  while (dentry != nullptr) {
    if (dentry->d_name[0] == '.') {
      dentry = readdir(proc_dir);
      continue;
    }

    proc_id_str = dentry->d_name;
    assert(is_number(proc_id_str) && "Unexpected file name in kfd/proc dir");
    if (!is_number(proc_id_str)) {
      dentry = readdir(proc_dir);
      continue;
    }
    if (procs && *num_procs_found < num_allocated) {
      int err;
      std::string tmp;

      procs[*num_procs_found].process_id =
                                static_cast<uint32_t>(std::stoi(proc_id_str));

      std::string pasid_str_path = kKFDProcPathRoot;
      pasid_str_path += "/";
      pasid_str_path += proc_id_str;
      pasid_str_path += "/";
      pasid_str_path += kKFDPasidFName;

      err = ReadSysfsStr(pasid_str_path, &tmp);
      if (err) {
        dentry = readdir(proc_dir);
        continue;
      }
      assert(is_number(tmp) && "Unexpected value in pasid file");
      if (!is_number(tmp)) {
        closedir(proc_dir);
        return EINVAL;
      }
      procs[*num_procs_found].pasid = static_cast<uint32_t>(std::stoi(tmp));
    }
    ++(*num_procs_found);

    dentry = readdir(proc_dir);
  }

  errno = 0;
  if (closedir(proc_dir)) {
    return errno;
  }
  return 0;
}

// Read the gpuid files found in all the <queue id> dirs and put them in
// gpus_found.
// Directory structure:
//     /sys/class/kfd/kfd/proc/<pid>/queues/<queue id>/gpuid

int GetProcessGPUs(uint32_t pid, std::unordered_set<uint64_t> *gpu_set) {
  int err;

  assert(gpu_set != nullptr);
  if (gpu_set == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }
  errno = 0;

  std::string queues_dir = kKFDProcPathRoot;
  queues_dir += "/";
  queues_dir += std::to_string(pid);
  queues_dir += "/queues";

  auto queues_dir_hd = opendir(queues_dir.c_str());

  if (queues_dir_hd == nullptr) {
    std::string err_str = "Unable to open queues directory for process ";
    err_str += std::to_string(pid);
    perror(err_str.c_str());
    return ESRCH;
  }

  auto q_dentry = readdir(queues_dir_hd);

  std::string q_gpu_id_str;
  std::string q_dir;

  std::string tmp;

  while (q_dentry != nullptr) {
    if (q_dentry->d_name[0] == '.') {
      q_dentry = readdir(queues_dir_hd);
      continue;
    }

    if (!is_number(q_dentry->d_name)) {
      q_dentry = readdir(queues_dir_hd);
      continue;
    }

    q_gpu_id_str = queues_dir + '/' + q_dentry->d_name + "/gpuid";

    err = ReadSysfsStr(q_gpu_id_str, &tmp);
    if (err) {
      q_dentry = readdir(queues_dir_hd);
      continue;
    }

    uint64_t val;
    try {
      val = static_cast<uint64_t>(std::stoi(tmp));
    } catch (...) {
      std::cerr << "Error; read invalid data: " << tmp << " from " <<
                                                    q_gpu_id_str << std::endl;
      closedir(queues_dir_hd);
      return ENXIO;  // Return "no such device" if we read an invalid gpu id
    }
    gpu_set->insert(val);

    q_dentry = readdir(queues_dir_hd);
  }

  errno = 0;
  if (closedir(queues_dir_hd)) {
    return errno;
  }
  return 0;
}

static int CheckValidProcessInfoData(const std::string& s, int sysfs_ret){
  if(sysfs_ret==0 && !is_number(s)){
    return EINVAL;
  }
  return sysfs_ret;
}

int GetProcessInfoForPID(uint32_t pid, rsmi_process_info_t *proc,
                         std::unordered_set<uint64_t> *gpu_set) {
  assert(proc != nullptr);
  assert(gpu_set != nullptr);
  int err;
  std::string tmp;
  std::unordered_set<uint64_t>::iterator itr;

  std::string proc_str_path = kKFDProcPathRoot;
  proc_str_path += "/";
  proc_str_path +=  std::to_string(pid);

  if (!FileExists(proc_str_path.c_str())) {
    return ESRCH;
  }
  proc->process_id = pid;

  std::string pasid_str_path = proc_str_path;
  pasid_str_path += "/";
  pasid_str_path += kKFDPasidFName;

  err = ReadSysfsStr(pasid_str_path, &tmp);
  if (err) {
    return err;
  }
  assert(is_number(tmp) && "Unexpected value in pasid file");

  if (!is_number(tmp)) {
    return EINVAL;
  }
  proc->pasid = static_cast<uint32_t>(std::stoi(tmp));

  proc->vram_usage = 0;
  proc->sdma_usage = 0;
  proc->cu_occupancy = 0;

  uint32_t cu_count = 0;
  static amd::smi::RocmSMI& smi = amd::smi::RocmSMI::getInstance();
  static std::map<uint64_t, std::shared_ptr<KFDNode>>& kfd_node_map =
                                                           smi.kfd_node_map();

  for (itr = gpu_set->begin(); itr != gpu_set->end(); itr++) {
    uint64_t gpu_id = (*itr);

    std::string vram_str_path = proc_str_path;
    vram_str_path += "/vram_";
    vram_str_path += std::to_string(gpu_id);

    err = ReadSysfsStr(vram_str_path, &tmp);
    auto sysfs_data_errcode = CheckValidProcessInfoData(tmp, err);

    // Report all errors, except ENOENT (2), which should be ignored
    // and the proc->vram_usage should be unmodified
    if (!(sysfs_data_errcode == 0 || sysfs_data_errcode == ENOENT)){
      return sysfs_data_errcode;
    }
    // Do not store any invalid values
    else if (sysfs_data_errcode == 0) {
      proc->vram_usage += std::stoull(tmp);
    }

    std::string sdma_str_path = proc_str_path;
    sdma_str_path += "/sdma_";
    sdma_str_path += std::to_string(gpu_id);

    err = ReadSysfsStr(sdma_str_path, &tmp);
    sysfs_data_errcode = CheckValidProcessInfoData(tmp, err);

    if (!(sysfs_data_errcode == 0 || sysfs_data_errcode == ENOENT)){
      return sysfs_data_errcode;
    }
    else if (sysfs_data_errcode == 0) {
      proc->sdma_usage += std::stoull(tmp);
    }

    // Build the path and read from Sysfs file, info that
    // encodes Compute Unit usage by a process of interest
    std::string cu_occupancy_path = proc_str_path;
    cu_occupancy_path += "/stats_";
    cu_occupancy_path += std::to_string(gpu_id);
    cu_occupancy_path += "/cu_occupancy";

    err = ReadSysfsStr(cu_occupancy_path, &tmp);
    sysfs_data_errcode = CheckValidProcessInfoData(tmp, err);

    if (!(sysfs_data_errcode == 0 || sysfs_data_errcode == ENOENT)){
      return sysfs_data_errcode;
    }
    else if(sysfs_data_errcode==0){
      // Update CU usage by the process
      proc->cu_occupancy += std::stoi(tmp);
      // Collect count of compute units
      cu_count += kfd_node_map[gpu_id]->cu_count();
    }
    else {
      // Some GFX revisions do not provide cu_occupancy debugfs method
      // which may cause ENOENT
      proc->cu_occupancy = CU_OCCUPANCY_INVALID;
      cu_count = 0;
    }
  }

  // Adjust CU occupancy to percent.
  if (cu_count > 0) {
    proc->cu_occupancy = ((proc->cu_occupancy * 100) / cu_count);
  }

  return 0;
}

int DiscoverKFDNodes(std::map<uint64_t, std::shared_ptr<KFDNode>> *nodes) {
  assert(nodes != nullptr);

  if (nodes == nullptr) {
    return EINVAL;
  }
  assert(nodes->empty());

  nodes->clear();

  std::shared_ptr<KFDNode> node;
  uint32_t node_indx;

  auto kfd_node_dir = opendir(kKFDNodesPathRoot);
  if (kfd_node_dir == nullptr) {
    return errno;
  }

  auto dentry = readdir(kfd_node_dir);
  while (dentry != nullptr) {
    if (dentry->d_name[0] == '.') {
      dentry = readdir(kfd_node_dir);
      continue;
    }

    if (!is_number(dentry->d_name)) {
      dentry = readdir(kfd_node_dir);
      continue;
    }

    node_indx = static_cast<uint32_t>(std::stoi(dentry->d_name));

    if (!KFDNodeSupported(node_indx)) {
      dentry = readdir(kfd_node_dir);
      continue;
    }

    node = std::make_shared<KFDNode>(node_indx);

    node->Initialize();

    if (node->gpu_id() == 0) {
      // Don't add; this is a cpu node.
      dentry = readdir(kfd_node_dir);
      continue;
    }

    uint64_t kfd_gpu_node_bus_fn;
    uint64_t kfd_gpu_node_domain;
    int ret;
    ret =
      node->get_property_value(kKFDNodePropLOCATION_IDStr,
                                                        &kfd_gpu_node_bus_fn);
    if (ret != 0) {
      std:: cerr << "Failed to open properties file for kfd node " <<
                                       node->node_index() << "." << std::endl;
      closedir(kfd_node_dir);
      return ret;
    }
    ret =
        node->get_property_value(kKFDNodePropDOMAINStr, &kfd_gpu_node_domain);
    if (ret != 0) {
      std::cerr << "Failed to get \"domain\" properity from properties "
              "files for kfd node " << node->node_index() << "." << std::endl;
      closedir(kfd_node_dir);
      return ret;
    }

    uint64_t kfd_bdfid =
                       (kfd_gpu_node_domain << 32) | (kfd_gpu_node_bus_fn);
    (*nodes)[kfd_bdfid] = node;

    dentry = readdir(kfd_node_dir);
  }

  if (closedir(kfd_node_dir)) {
    std::string err_str = "Failed to close KFD node directory ";
    err_str += kKFDNodesPathRoot;
    err_str += ".";
    perror(err_str.c_str());
    return 1;
  }
  return 0;
}

KFDNode::~KFDNode() = default;

int KFDNode::ReadProperties(void) {
  int ret;

  std::vector<std::string> propVec;

  assert(properties_.empty());
  if (!properties_.empty()) {
    return 0;
  }

  ret = ReadKFDDeviceProperties(node_indx_, &propVec);

  if (ret) {
    return ret;
  }

  std::string key_str;
  std::string val_str;
  uint64_t val_int;  // Assume all properties are unsigned integers for now
  std::istringstream fs;
  std::ostringstream ss;

  for (const auto & i : propVec) {
    fs.str(i);
    fs >> key_str;
    fs >> val_str;
    // Leaving below to debug any new properties file changes
    // ss << __PRETTY_FUNCTION__ << " | key = " << key_str
    //    << "; val = " << val_str;
    // LOG_TRACE(ss);
    val_int = std::stoull(val_str);
    properties_[key_str] = val_int;

    fs.str("");
    fs.clear();
  }

  return 0;
}

int
KFDNode::Initialize(void) {
  int ret = 0;
  ret = ReadProperties();
  if (ret) {return ret;}

  ret = ReadKFDGpuId(node_indx_, &gpu_id_);
  if (ret || (gpu_id_ == 0)) {return ret;}

  ret = ReadKFDGpuName(node_indx_, &name_);

  ret = get_property_value(kKFDNodePropHIVE_IDStr, &xgmi_hive_id_);
  if (ret != 0) {
    throw amd::smi::rsmi_exception(RSMI_INITIALIZATION_ERROR,
    "Failed to initialize rocm_smi library (get xgmi hive id).");
  }

  std::map<uint32_t, std::shared_ptr<IOLink>> io_link_map_tmp;
  ret = DiscoverIOLinksPerNode(node_indx_, &io_link_map_tmp);
  if (ret != 0) {
    throw amd::smi::rsmi_exception(RSMI_INITIALIZATION_ERROR,
    "Failed to initialize rocm_smi library (IO Links discovery per node).");
  }

  std::map<uint32_t, std::shared_ptr<IOLink>>::iterator it;
  uint32_t node_to;
  uint64_t node_to_gpu_id;
  std::shared_ptr<IOLink> link;
  bool numa_node_found = false;
  for (it = io_link_map_tmp.begin(); it != io_link_map_tmp.end(); it++) {
    io_link_map_[it->first] = it->second;
    node_to = it->first;
    link = it->second;
    ret = ReadKFDGpuId(node_to, &node_to_gpu_id);
    if (ret) {return ret;}
    if (node_to_gpu_id == 0) {  //  CPU node
      if (numa_node_found) {
        if (numa_node_weight_ > link->weight()) {
          numa_node_number_ = node_to;
          numa_node_weight_ = link->weight();
          numa_node_type_ = link->type();
        }
      } else {
        numa_node_number_ = node_to;
        numa_node_weight_ = link->weight();
        numa_node_type_ = link->type();
        numa_node_found = true;
      }
    } else {
      io_link_type_[node_to] = link->type();
      io_link_weight_[node_to] = link->weight();
      io_link_max_bandwidth_[node_to] = link->max_bandwidth();
      io_link_min_bandwidth_[node_to] = link->min_bandwidth();

    }
  }

  // Pre-compute the total number of compute units a device has
  uint64_t tmp_val;
  ret = get_property_value(kKFDNodePropSIMD_ARRAYS_PER_ENGINEStr, &tmp_val);
  if (ret != 0) {
    throw amd::smi::rsmi_exception(RSMI_INITIALIZATION_ERROR,
    "Failed to initialize rocm_smi library "
                                 "(get number of shader arrays per engine).");
  }
  cu_count_ = uint32_t(tmp_val);
  ret = get_property_value(kKFDNodePropARRAY_COUNTStr, &tmp_val);
  if (ret != 0) {
    throw amd::smi::rsmi_exception(RSMI_INITIALIZATION_ERROR,
    "Failed to initialize rocm_smi library (get number of shader arrays).");
  }
  cu_count_ = cu_count_ * uint32_t(tmp_val);
  ret = get_property_value(kKFDNodePropCU_PER_SIMD_ARRAYStr, &tmp_val);
  if (ret != 0) {
    throw amd::smi::rsmi_exception(RSMI_INITIALIZATION_ERROR,
    "Failed to initialize rocm_smi library (get number of CU's per array).");
  }
  cu_count_ = cu_count_ * uint32_t(tmp_val);

  return ret;
}

int
KFDNode::get_property_value(std::string property, uint64_t *value) {
  assert(value != nullptr);
  if (value == nullptr) {
    return EINVAL;
  }
  if (properties_.find(property) == properties_.end()) {
    return EINVAL;
  }
  *value = properties_[property];
  return 0;
}

int
KFDNode::get_io_link_type(uint32_t node_to, IO_LINK_TYPE *type) {
  assert(type != nullptr);
  if (type == nullptr) {
    return EINVAL;
  }
  if (io_link_type_.find(node_to) == io_link_type_.end()) {
    return EINVAL;
  }
  *type = io_link_type_[node_to];
  return 0;
}

int
KFDNode::get_io_link_weight(uint32_t node_to, uint64_t *weight) {
  assert(weight != nullptr);
  if (weight == nullptr) {
    return EINVAL;
  }
  if (io_link_weight_.find(node_to) == io_link_weight_.end()) {
    return EINVAL;
  }
  *weight = io_link_weight_[node_to];
  return 0;
}

int
KFDNode::get_io_link_bandwidth(uint32_t node_to, uint64_t *max_bandwidth,
                                                      uint64_t *min_bandwidth){
  assert (max_bandwidth != nullptr && min_bandwidth != nullptr);
  if (max_bandwidth == nullptr || min_bandwidth == nullptr ){
    return EINVAL;
  }

  if (io_link_max_bandwidth_.find(node_to) == io_link_max_bandwidth_.end() ||
      io_link_min_bandwidth_.find(node_to) == io_link_min_bandwidth_.end()){
        return EINVAL;
      }

  *max_bandwidth = io_link_max_bandwidth_[node_to];
  *min_bandwidth = io_link_min_bandwidth_[node_to];

  return 0;
}
// /sys/class/kfd/kfd/topology/nodes/*/mem_banks/*/properties
// size_in_bytes 68702699520
int KFDNode::get_total_memory(uint64_t* total) {
  std::ostringstream ss;
  if (total == nullptr) {
    return EINVAL;
  }
  *total = 0;

  std::string f_path  = kKFDNodesPathRoot;
  f_path += "/";
  f_path += std::to_string(node_indx_);
  f_path += "/mem_banks";
  int subDirCount = subDirectoryCountInPath(f_path);
  ss << __PRETTY_FUNCTION__ << " | [before loop] Within " << f_path
     << " has subdirectory count = " << std::to_string(subDirCount);
  LOG_DEBUG(ss);

  auto kfd_node_dir = opendir(f_path.c_str());
  if (kfd_node_dir == nullptr) {
    return errno;
  }
  auto dentry = readdir(kfd_node_dir);
  while (dentry != nullptr && subDirCount > 0) {
    ss << __PRETTY_FUNCTION__ << " | [inside loop] Within " << f_path
       << " has subdirectory count = " << std::to_string(subDirCount);
    LOG_DEBUG(ss);
    if (dentry->d_name[0] == '.') {
      dentry = readdir(kfd_node_dir);
      continue;
    }

    if (!is_number(dentry->d_name)) {
      dentry = readdir(kfd_node_dir);
      continue;
    }

    // read "size_in_bytes 68702699520" line
    const std::string size_in_bytes_property = "size_in_bytes ";
    std::string memory_bank_file = f_path + "/"
                  + dentry->d_name + "/properties";
    std::ifstream fs(memory_bank_file);
    if (!fs) {
      dentry = readdir(kfd_node_dir);
      continue;
    }
    std::string line;
    while (std::getline(fs, line)) {
      if (line.substr(0, size_in_bytes_property.length())
           == size_in_bytes_property) {
          auto bytes = line.substr(size_in_bytes_property.length());
          try {
            *total += std::stol(bytes);
            break;
          } catch(...) {
            dentry = readdir(kfd_node_dir);
            continue;
          }
      }
    }  // end loop for lines in property file
    subDirCount--;
  }  // end loop for mem_bank directory

  if (closedir(kfd_node_dir)) {
    std::string err_str = "Failed to close KFD node directory ";
    err_str += f_path;
    err_str += ".";
    perror(err_str.c_str());
    return 1;
  }
  return 0;
}

// ioctl on kfd node device
int KFDNode::get_used_memory(uint64_t* used) {
  if (used == nullptr) return EINVAL;
  static const char *kPathKFDIoctl = "/dev/kfd";

  int kfd_fd = open(kPathKFDIoctl, O_RDWR | O_CLOEXEC);
  if (kfd_fd <= 0) {
      return 1;
  }
  struct kfd_ioctl_get_available_memory_args mem = {0, 0, 0};
  mem.gpu_id = static_cast<uint32_t>(gpu_id_);
  if (ioctl(kfd_fd, AMDKFD_IOC_AVAILABLE_MEMORY , &mem) != 0) {
    close(kfd_fd);
    return 1;
  }
  close(kfd_fd);

  // used = total - available
  uint64_t total = 0;
  int ret = get_total_memory(&total);
  if (ret == 0 && total > 0 && mem.available < total) {
    *used = total - mem.available;
    return 0;
  }

  return 1;
}

// /sys/class/kfd/kfd/topology/nodes/*/properties
int read_node_properties(uint32_t node, std::string property_name,
                         uint64_t *val) {
  std::ostringstream ss;
  std::string propertiesFullPath = "/sys/class/kfd/kfd/topology/nodes/"
    + std::to_string(node) + "/properties";
  int retVal = EINVAL;
  if (property_name.empty() || val == nullptr) {
    ss << __PRETTY_FUNCTION__
       << " | File: " << propertiesFullPath
       << " | Issue: Could not read node #" << std::to_string(node)
       << ", property_name is empty or *val is nullptr "
       << " | return = " << std::to_string(retVal)
       << " | ";
    LOG_DEBUG(ss);
    return retVal;
  }
  std::shared_ptr<KFDNode> myNode = std::shared_ptr<KFDNode>(new KFDNode(node));
  myNode->Initialize();
  if (KFDNodeSupported(node)) {
    retVal = myNode->get_property_value(property_name, val);
    ss << __PRETTY_FUNCTION__
       << " | File: " << propertiesFullPath
       << " | Successfully read node #" << std::to_string(node)
       << " for property_name = " << property_name
       << " | Data (" << property_name << ") * val = "
       << std::to_string(*val)
       << " | return = " << std::to_string(retVal)
       << " | ";
    LOG_DEBUG(ss);
  } else {
    retVal = 1;
    ss << __PRETTY_FUNCTION__
       << " | File: " << propertiesFullPath
       << " | Issue: Could not read node #" << std::to_string(node)
       << ", KFD node was an unsupported node."
       << " | return = " << std::to_string(retVal)
       << " | ";
    LOG_ERROR(ss);
  }
  return retVal;
}

// /sys/class/kfd/kfd/topology/nodes/*/gpu_id
int get_gpu_id(uint32_t node, uint64_t *gpu_id) {
  std::ostringstream ss;
  std::string gpu_id_FullPath = "/sys/class/kfd/kfd/topology/nodes/"
    + std::to_string(node) + "/gpu_id";
  int retVal = EINVAL;
  if (gpu_id == nullptr) {
    ss << __PRETTY_FUNCTION__
       << " | File: " << gpu_id_FullPath
       << " | Issue: Could not read node #" << std::to_string(node)
       << ", gpu_id is a nullptr "
       << " | return = " << std::to_string(retVal)
       << " | ";
    LOG_DEBUG(ss);
    return retVal;
  }
  std::shared_ptr<KFDNode> myNode = std::shared_ptr<KFDNode>(new KFDNode(node));
  myNode->Initialize();
  if (KFDNodeSupported(node)) {
    retVal = ReadKFDGpuId(node, gpu_id);
    ss << __PRETTY_FUNCTION__
       << " | File: " << gpu_id_FullPath
       << " | Successfully read node #" << std::to_string(node)
       << " for gpu_id"
       << " | Data (gpu_id) *gpu_id = "
       << std::to_string(*gpu_id)
       << " | return = " << std::to_string(retVal)
       << " | ";
    LOG_DEBUG(ss);
  } else {
    retVal = 1;
    ss << __PRETTY_FUNCTION__
       << " | File: " << gpu_id_FullPath
       << " | Issue: Could not read node #" << std::to_string(node)
       << ", KFD node was an unsupported node."
       << " | return = " << std::to_string(retVal)
       << " | ";
    LOG_ERROR(ss);
  }
  return retVal;
}

// /sys/class/kfd/kfd/topology/nodes/*/properties | grep gfx_target_version
int KFDNode::get_gfx_target_version(uint64_t *gfx_target_version) {
  std::ostringstream ss;
  std::string properties_path = "/sys/class/kfd/kfd/topology/nodes/"
    + std::to_string(this->node_indx_) + "/properties";
  uint64_t gfx_version = 0;
  int ret = read_node_properties(this->node_indx_, "gfx_target_version",
                                 &gfx_version);
  *gfx_target_version = gfx_version;
  ss << __PRETTY_FUNCTION__
     << " | File: " << properties_path
     << " | Read node: " << std::to_string(this->node_indx_)
     << " for gfx_target_version"
     << " | Data (*gfx_target_version): "
     << std::to_string(*gfx_target_version)
     << " | Return: "
     << getRSMIStatusString(amd::smi::ErrnoToRsmiStatus(ret), false)
     << " | ";
  LOG_DEBUG(ss);
  return ret;
}

// Public interface for device
// /sys/class/kfd/kfd/topology/nodes/*/gpu_id
int KFDNode::get_gpu_id(uint64_t *gpu_id) {
  std::ostringstream ss;
  std::string gpuid_path = "/sys/class/kfd/kfd/topology/nodes/"
    + std::to_string(this->node_indx_) + "/gpu_id";
  const uint64_t undefined_gpu_id = std::numeric_limits<uint64_t>::max();
  std::string gpu_id_string = "";
  *gpu_id = undefined_gpu_id;
  int ret = ReadSysfsStr(gpuid_path, &gpu_id_string);
  if (ret != 0 || gpu_id_string.empty()) {
    ss << __PRETTY_FUNCTION__
       << " | File: " << gpuid_path
       << " | Data (*gpu_id): empty or nullptr"
       << " | Issue: Could not read node #" << std::to_string(this->node_indx_)
       << ". KFD node was an unsupported node or value read was empty."
       << " | Return: "
       << getRSMIStatusString(amd::smi::ErrnoToRsmiStatus(ret), false)
       << " | ";
    LOG_ERROR(ss);
    return ret;
  }
  *gpu_id = std::stoull(gpu_id_string);
  if (*gpu_id == 0) {  // CPU node - return not supported
    *gpu_id = undefined_gpu_id;
    ret = ENOENT;  // map to RSMI_STATUS_NOT_SUPPORTED
  }
  ss << __PRETTY_FUNCTION__
     << " | File: " << gpuid_path
     << " | Read node #: " << std::to_string(this->node_indx_)
     << " | Data (*gpu_id): " << std::to_string(*gpu_id)
     << " | Return: "
     << getRSMIStatusString(amd::smi::ErrnoToRsmiStatus(ret), false)
     << " | ";
  LOG_DEBUG(ss);
  return ret;
}

// Public interface for device
// /sys/class/kfd/kfd/topology/nodes/<node_id>
int KFDNode::get_node_id(uint32_t *node_id) {
  std::ostringstream ss;
  int ret = 0;
  std::string nodeid_path = "/sys/class/kfd/kfd/topology/nodes/"
    + std::to_string(this->node_indx_);
  *node_id = this->node_indx_;
  ss << __PRETTY_FUNCTION__
     << " | File: " << nodeid_path
     << " | Read node #: " << std::to_string(this->node_indx_)
     << " | Data (*node_id): " << std::to_string(*node_id)
     << " | Return: "
     << getRSMIStatusString(amd::smi::ErrnoToRsmiStatus(ret), false)
     << " | ";
  LOG_DEBUG(ss);
  return ret;
}

}  // namespace smi
}  // namespace amd
