// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <dirent.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <fstream>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_utils.h"
#include "rocm_smi/rocm_smi_kfd.h"

extern "C" {

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc99-designator"
#endif
// Using known extension, array designators
static const char* container_type_name[AMDSMI_MAX_CONTAINER_TYPE] = {
    [AMDSMI_CONTAINER_LXC] = "lxc",
    [AMDSMI_CONTAINER_DOCKER] = "docker",
};
#ifdef __clang__
#pragma clang diagnostic pop
#endif

amdsmi_status_t gpuvsmi_pid_is_gpu(const std::string& path, const char* bdf) {
  DIR* d;
  struct dirent* dir;

  d = opendir(path.c_str());
  if (!d) return AMDSMI_STATUS_NO_PERM;

  /* iterate through all the fds, try to find
   * a match for the GPU bdf
   */
  while ((dir = readdir(d)) != NULL) {
    std::string file = path + dir->d_name;
    std::ifstream fdinfo(file.c_str());
    for (std::string line; std::getline(fdinfo, line);) {
      if (line.find(bdf) != std::string::npos) {
        closedir(d);
        return AMDSMI_STATUS_SUCCESS;
      }
    }
  }

  closedir(d);

  return AMDSMI_STATUS_NOT_FOUND;
}

// See fdinfo.h for the full contract. The fast path uses `known_kfd_gpu_id`;
// the sentinels 0 and UINT64_MAX force the topology-discovery fallback below.
amdsmi_status_t gpu_is_in_kfd_pid(const amdsmi_bdf_t& bdf, long pid, uint64_t known_kfd_gpu_id) {
  uint64_t target_gid = known_kfd_gpu_id;

  if (target_gid == 0 || target_gid == UINT64_MAX) {
    // pack (domain,bus,device,function) to the same 64-bit key
    // (DOMAIN << 32) | (BUS << 8) | (DEVICE << 3) | FUNCTION
    auto pack_bdf_to_kfd_bdfid = [](const amdsmi_bdf_t& b) -> uint64_t {
      const uint64_t domain = static_cast<uint64_t>(b.domain_number & 0xffffu);
      const uint64_t bus = static_cast<uint64_t>(b.bus_number & 0xffu);
      const uint64_t dev = static_cast<uint64_t>(b.device_number & 0x1fu);
      const uint64_t func = static_cast<uint64_t>(b.function_number & 0x7u);
      const uint64_t loc = (bus << 8) | (dev << 3) | func;
      return (domain << 32) | loc;
    };

    // Build map of KFD nodes
    std::map<uint64_t, std::shared_ptr<amd::smi::KFDNode>> nodes;
    int ret = DiscoverKFDNodes(&nodes);
    if (ret != 0) {
      return AMDSMI_STATUS_API_FAILED;
    }

    // Convert bdf and find node
    const uint64_t key = pack_bdf_to_kfd_bdfid(bdf);
    auto it = nodes.find(key);
    if (it == nodes.end()) {
      return AMDSMI_STATUS_NOT_FOUND;
    }

    // Grab gpu id and ensure not cpu
    target_gid = it->second->gpu_id();
    if (target_gid == 0) {
      return AMDSMI_STATUS_NOT_FOUND;
    }
  }

  // Get all KFD GPU ids for pid
  std::unordered_set<uint64_t> pid_gids;
  int ret = amd::smi::GetKfdGpuIdsForPid(pid, &pid_gids);
  if (ret != 0) {
    if (ret == EACCES) {
      return AMDSMI_STATUS_NO_PERM;
    }
    return AMDSMI_STATUS_NOT_FOUND;
  }

  // Return success if gpu id is in pid gpu ids
  return (pid_gids.count(target_gid) ? AMDSMI_STATUS_SUCCESS : AMDSMI_STATUS_NOT_FOUND);
}

amdsmi_status_t gpuvsmi_get_pid_info(const amdsmi_bdf_t& bdf, long int pid,
                                     amdsmi_proc_info_t& info, uint64_t kfd_gpu_id) {
  char bdf_str[13];
  DIR* d;
  struct dirent* dir;

  /* 0000:00:00.0 */
  snprintf(bdf_str, 13, "%04" PRIx32 ":%02" PRIx32 ":%02" PRIx32 ".%" PRIu32,
           static_cast<uint32_t>(bdf.domain_number & 0xffff),
           static_cast<uint32_t>(bdf.bus_number & 0xff),
           static_cast<uint32_t>(bdf.device_number & 0x1f),
           static_cast<uint32_t>(bdf.function_number & 0x7));

  std::string path = "/proc/" + std::to_string(pid) + "/fdinfo/";
  std::string name_path = "/proc/" + std::to_string(pid) + "/exe";
  std::string cgroup_path = "/proc/" + std::to_string(pid) + "/cgroup";

  amdsmi_status_t ret = gpu_is_in_kfd_pid(bdf, pid, kfd_gpu_id);

  if (ret != AMDSMI_STATUS_SUCCESS) {
    // If kfd process detection fails, fallback on old bdf code
    ret = gpuvsmi_pid_is_gpu(path.c_str(), bdf_str);
    if (ret != AMDSMI_STATUS_SUCCESS) {
      return ret;
    }
  }

  d = opendir(path.c_str());
  if (!d) return AMDSMI_STATUS_NO_PERM;

  memset(&info, 0, sizeof(info));
  /* Iterate through all fdinfos */
  while ((dir = readdir(d)) != NULL) {
    std::string file = path + dir->d_name;
    std::ifstream fdinfo(file.c_str());

    for (std::string bdfline; getline(fdinfo, bdfline);) {
      if (bdfline.find("drm-pdev:") != std::string::npos) {
        char fd_bdf_str[13];

        /* Only check against fdinfo files that contain a bdf */
        if (sscanf(bdfline.c_str(), "drm-pdev:       %12s", &fd_bdf_str[0]) != 1) continue;

        /* Populate amdsmi_proc_info_t struct only if the bdf in
         * the fdinfo file matches the passed bdf */
        if (strncmp(bdf_str, fd_bdf_str, 13) == 0) {
          std::ifstream fdinfo(file.c_str());

          for (std::string line; getline(fdinfo, line);) {
            if (line.find("drm-memory-gtt:") != std::string::npos) {
              unsigned long mem;
              if (sscanf(line.c_str(), "drm-memory-gtt:  %" PRIu64, &mem) != 1) continue;
              info.mem += mem * 1000;
              info.memory_usage.gtt_mem += mem * 1000;
            } else if (line.find("drm-memory-cpu:") != std::string::npos) {
              unsigned long mem;
              if (sscanf(line.c_str(), "drm-memory-cpu:  %" PRIu64, &mem) != 1) continue;
              info.mem += mem * 1000;
              info.memory_usage.cpu_mem += mem * 1000;
            } else if (line.find("drm-memory-vram:") != std::string::npos) {
              unsigned long mem;
              if (sscanf(line.c_str(), "drm-memory-vram:  %" PRIu64, &mem) != 1) continue;
              info.mem += mem * 1000;
              info.memory_usage.vram_mem += mem * 1000;
            } else if (line.find("drm-engine-gfx") != std::string::npos) {
              uint64_t engine_gfx;
              if (sscanf(line.c_str(), "drm-engine-gfx:  %" PRIu64, &engine_gfx) != 1) continue;
              info.engine_usage.gfx = engine_gfx;
            } else if (line.find("drm-engine-enc") != std::string::npos) {
              uint64_t engine_enc;
              if (sscanf(line.c_str(), "drm-engine-enc:  %" PRIu64, &engine_enc) != 1) continue;
              info.engine_usage.enc = engine_enc;
            }
          }
        }
      }
    }
  }

  closedir(d);

  //  Note: If possible at all, try to get the name of the process/container.
  //        In case the other info fail, get at least something.
  char exe_realpath[PATH_MAX] = {0};
  ssize_t len = readlink(name_path.c_str(), exe_realpath, sizeof(exe_realpath) - 1);
  std::string name = (len > 0) ? std::string(exe_realpath, static_cast<size_t>(len)) : "N/A";

  if (name.empty()) return AMDSMI_STATUS_API_FAILED;

  snprintf(info.name, sizeof(info.name), "%s", name.c_str());

  for (int i = 0; i < AMDSMI_MAX_CONTAINER_TYPE; i++) {
    std::ifstream cgroup_info(cgroup_path.c_str());
    std::string container_id;
    for (std::string line; getline(cgroup_info, line);) {
      if (line.find(container_type_name[i]) != std::string::npos) {
        container_id =
            line.substr(line.find(container_type_name[i]) + strlen(container_type_name[i]) + 1, 16);
        snprintf(info.container_name, sizeof(info.container_name), "%s", container_id.c_str());
        break;
      }
    }
    if (strlen(info.container_name) > 0) break;
  }
  info.pid = (uint32_t)pid;

  return AMDSMI_STATUS_SUCCESS;
}

}  // extern "C"
