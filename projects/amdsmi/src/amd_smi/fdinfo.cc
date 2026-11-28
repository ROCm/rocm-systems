// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "amd_smi/impl/fdinfo.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_container_id_parser.h"
#include "amd_smi/impl/amd_smi_drm_fdinfo_parser.h"
#include "amd_smi/impl/amd_smi_self_fdinfo.h"
#include "amd_smi/impl/amd_smi_utils.h"
#include "rocm_smi/rocm_smi_kfd.h"

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

namespace {

bool IsNumericFdName(const char* name) {
  if (name == nullptr || name[0] == '\0') return false;
  for (const char* p = name; *p != '\0'; ++p) {
    if (*p < '0' || *p > '9') return false;
  }
  return true;
}

SelfFdinfoReader g_self_fdinfo_reader = nullptr;

// procfs fails both the open and later reads with ENOENT once the descriptor
// is closed.
amdsmi_status_t ReadProcSelfFdinfo(int fd, std::string* text) {
  const std::string path = "/proc/self/fdinfo/" + std::to_string(fd);
  const int fdinfo_fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fdinfo_fd < 0) return errno == ENOENT ? AMDSMI_STATUS_NOT_FOUND : AMDSMI_STATUS_FILE_ERROR;
  const amdsmi_status_t status = amd::smi::ReadFdinfoFile(fdinfo_fd, text);
  close(fdinfo_fd);
  return status;
}

// Adds descriptor `fd` to `clients` if it is an amdgpu client of `render_dev`.
// The descriptor is not duplicated: closing a duplicate would release every
// POSIX record lock the process holds on that file. Instead fdinfo is read
// before and after the fstat, and both reads must name the same client and, on
// kernels that print it, the fstat'ed inode. A descriptor number closed and
// reused in between is skipped rather than paired with another file's device.
// Only amdgpu clients are fstat'ed, so an unrelated descriptor on a stale
// network mount cannot fail the query.
amdsmi_status_t CollectSelfFd(int fd, dev_t render_dev, SelfFdinfoReader read_fdinfo,
                              std::map<std::string, amd::smi::DrmClientUsage>* clients) {
  std::string text;
  amdsmi_status_t status = read_fdinfo(fd, &text);
  if (status == AMDSMI_STATUS_NOT_FOUND) return AMDSMI_STATUS_SUCCESS;
  if (status != AMDSMI_STATUS_SUCCESS) return status;
  const auto before = amd::smi::ParseDrmFdinfo(text);
  if (before.driver != "amdgpu") return AMDSMI_STATUS_SUCCESS;

  struct stat st{};
  const bool stat_ok = fstat(fd, &st) == 0;
  if (stat_ok && (!S_ISCHR(st.st_mode) || st.st_rdev != render_dev)) return AMDSMI_STATUS_SUCCESS;

  status = read_fdinfo(fd, &text);
  if (status == AMDSMI_STATUS_NOT_FOUND) return AMDSMI_STATUS_SUCCESS;
  if (status != AMDSMI_STATUS_SUCCESS) return status;
  auto usage = amd::smi::ParseDrmFdinfo(text);
  if (usage.driver != "amdgpu" || usage.client != before.client || usage.ino != before.ino) {
    return AMDSMI_STATUS_SUCCESS;
  }
  if (!stat_ok) return AMDSMI_STATUS_FILE_ERROR;
  if (!usage.ino.empty() && usage.ino != std::to_string(st.st_ino)) return AMDSMI_STATUS_SUCCESS;

  if (!usage.valid) return AMDSMI_STATUS_UNEXPECTED_DATA;
  (*clients)[usage.client] = usage;
  return AMDSMI_STATUS_SUCCESS;
}

}  // namespace

namespace amd::smi {

amdsmi_status_t ReadFdinfoFile(int fdinfo_fd, std::string* text) {
  text->clear();
  char buf[4096];
  for (;;) {
    const ssize_t n = read(fdinfo_fd, buf, sizeof(buf));
    if (n > 0) {
      text->append(buf, static_cast<size_t>(n));
    } else if (n == 0) {
      return AMDSMI_STATUS_SUCCESS;
    } else if (errno != EINTR) {
      return errno == ENOENT ? AMDSMI_STATUS_NOT_FOUND : AMDSMI_STATUS_FILE_ERROR;
    }
  }
}

}  // namespace amd::smi

namespace amd::smi::test_hooks {

const SelfFdinfoReader& SelfFdinfoReaderHook() { return g_self_fdinfo_reader; }

AMDSMI_TEST_HOOK void SetSelfFdinfoReaderHook(const SelfFdinfoReader& reader) {
  g_self_fdinfo_reader = reader;
}

}  // namespace amd::smi::test_hooks

amdsmi_status_t gpuvsmi_get_self_info(dev_t render_dev, amdsmi_proc_info_t& info,
                                      SelfFdinfoReader read_fdinfo) {
  if (read_fdinfo == nullptr) read_fdinfo = ReadProcSelfFdinfo;

  DIR* d = opendir("/proc/self/fd");
  if (!d) {
    return (errno == EACCES || errno == EPERM) ? AMDSMI_STATUS_NO_PERM : AMDSMI_STATUS_FILE_ERROR;
  }

  std::map<std::string, amd::smi::DrmClientUsage> clients;
  amdsmi_status_t status = AMDSMI_STATUS_SUCCESS;
  for (struct dirent* dir; (dir = readdir(d)) != NULL;) {
    if (!IsNumericFdName(dir->d_name)) continue;
    const int fd = atoi(dir->d_name);
    if (fd == dirfd(d)) continue;
    status = CollectSelfFd(fd, render_dev, read_fdinfo, &clients);
    if (status != AMDSMI_STATUS_SUCCESS) break;
  }
  closedir(d);
  if (status != AMDSMI_STATUS_SUCCESS) return status;

  if (clients.empty()) return AMDSMI_STATUS_NOT_FOUND;

  memset(&info, 0, sizeof(info));
  for (const auto& [key, usage] : clients) {
    info.memory_usage.vram_mem += usage.vram;
    info.memory_usage.gtt_mem += usage.gtt;
    info.memory_usage.cpu_mem += usage.cpu;
    info.engine_usage.gfx += usage.gfx_ns;
    info.engine_usage.enc += usage.enc_ns;
  }
  info.mem = info.memory_usage.vram_mem + info.memory_usage.gtt_mem + info.memory_usage.cpu_mem;
  info.pid = static_cast<uint32_t>(getpid());

  char exe_realpath[PATH_MAX] = {0};
  ssize_t len = readlink("/proc/self/exe", exe_realpath, sizeof(exe_realpath) - 1);
  amd::smi::CopyBounded(info.name, sizeof(info.name),
                        len > 0 ? std::string(exe_realpath, static_cast<size_t>(len)) : "N/A");

  info.cu_occupancy = std::numeric_limits<uint32_t>::max();
  info.evicted_time = std::numeric_limits<uint32_t>::max();
  info.sdma_usage = std::numeric_limits<uint64_t>::max();
  return AMDSMI_STATUS_SUCCESS;
}
