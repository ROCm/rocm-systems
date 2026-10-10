// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <dirent.h>
#include <fcntl.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_container_id_parser.h"
#include "amd_smi/impl/amd_smi_utils.h"
#include "rocm_smi/rocm_smi_kfd.h"

auto smi_amdgpu_parse_drm_memory(std::string_view line, std::string_view key, uint64_t* bytes)
    -> bool {
  if (line.substr(0, key.size()) != key) return false;
  const std::string_view text = trim(line.substr(key.size()));
  // The value ends at the first blank; the unit, if any, follows it.
  const size_t value_end = std::min(text.size(), text.find_first_of(" \t"));
  const auto value = parse_number_from_string<uint64_t>(text.substr(0, value_end));
  if (!value) return false;
  const std::string_view unit = trim(text.substr(value_end));
  constexpr std::pair<std::string_view, uint64_t> kUnits[] = {
      {"", 1}, {"KiB", uint64_t{1} << 10}, {"MiB", uint64_t{1} << 20}};
  for (const auto& [name, scale] : kUnits) {
    if (unit == name && *value <= std::numeric_limits<uint64_t>::max() / scale) {
      *bytes = *value * scale;
      return true;
    }
  }
  return false;
}

extern "C" {

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
  // Every fd that refers to a DRM client (dup, fork, fd passing) lists that
  // client's usage, so count each drm-client-id once.
  std::unordered_set<uint64_t> clients;
  while ((dir = readdir(d)) != NULL) {
    if (dir->d_name[0] == '.') continue;
    // Opened through the directory already open, not by its path again, so a
    // PID reused during the scan cannot switch these reads to another process.
    const int fd = openat(dirfd(d), dir->d_name, O_RDONLY | O_CLOEXEC);
    if (fd < 0) continue;
    FILE* fdinfo = fdopen(fd, "r");
    if (fdinfo == nullptr) {
      close(fd);
      continue;
    }
    char fd_bdf_str[13] = "";
    bool has_client = false;
    uint64_t client = 0, vram = 0, gtt = 0, cpu = 0, gfx = 0, enc = 0;
    char* l = nullptr;
    size_t l_size = 0;
    while (::getline(&l, &l_size, fdinfo) > 0) {
      if (sscanf(l, "drm-pdev: %12s", fd_bdf_str) == 1) continue;
      if (sscanf(l, "drm-client-id: %" SCNu64, &client) == 1) {
        has_client = true;
        continue;
      }
      if (smi_amdgpu_parse_drm_memory(l, "drm-memory-vram:", &vram)) continue;
      if (smi_amdgpu_parse_drm_memory(l, "drm-memory-gtt:", &gtt)) continue;
      if (smi_amdgpu_parse_drm_memory(l, "drm-memory-cpu:", &cpu)) continue;
      if (sscanf(l, "drm-engine-gfx: %" SCNu64, &gfx) == 1) continue;
      if (sscanf(l, "drm-engine-enc: %" SCNu64, &enc) == 1) continue;
    }
    free(l);
    fclose(fdinfo);
    if (strncmp(bdf_str, fd_bdf_str, sizeof(fd_bdf_str)) != 0) continue;
    if (has_client && !clients.insert(client).second) continue;
    info.memory_usage.vram_mem += vram;
    info.memory_usage.gtt_mem += gtt;
    info.memory_usage.cpu_mem += cpu;
    info.engine_usage.gfx += gfx;
    info.engine_usage.enc += enc;
  }
  info.mem = info.memory_usage.vram_mem + info.memory_usage.gtt_mem + info.memory_usage.cpu_mem;

  closedir(d);

  //  Note: If possible at all, try to get the name of the process/container.
  //        In case the other info fail, get at least something.
  char exe_realpath[PATH_MAX] = {0};
  ssize_t len = readlink(name_path.c_str(), exe_realpath, sizeof(exe_realpath) - 1);
  std::string name = (len > 0) ? std::string(exe_realpath, static_cast<size_t>(len)) : "N/A";

  if (name.empty()) return AMDSMI_STATUS_API_FAILED;

  // strncpy(dst, src, min(CAP, len)) leaves info.name unterminated when
  // name.length() >= AMDSMI_MAX_STRING_LENGTH; readlink() of /proc/<pid>/exe
  // can produce up to PATH_MAX bytes.
  amd::smi::CopyBounded(info.name, sizeof(info.name), name);

  std::vector<std::string> cgroup_lines;
  {
    std::ifstream cgroup_info(cgroup_path.c_str());
    for (std::string line; getline(cgroup_info, line);) cgroup_lines.push_back(line);
  }
  amd::smi::ResolveContainerId(cgroup_lines, info.container_name, sizeof(info.container_name));

  info.pid = (uint32_t)pid;

  return AMDSMI_STATUS_SUCCESS;
}

}  // extern "C"
