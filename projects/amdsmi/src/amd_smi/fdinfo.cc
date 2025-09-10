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

#include <dirent.h>
#include <cinttypes>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_utils.h"

extern "C" {

static const char *container_type_name[AMDSMI_MAX_CONTAINER_TYPE] = {
    [AMDSMI_CONTAINER_LXC] = "lxc",
    [AMDSMI_CONTAINER_DOCKER] = "docker",
};

amdsmi_status_t gpuvsmi_pid_is_gpu(const std::string &path, const char *bdf) {
  DIR *d;
  struct dirent *dir;

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

amdsmi_status_t gpuvsmi_get_pids(const amdsmi_bdf_t &bdf, std::vector<long int> &pids,
                                 uint64_t *size) {
  char bdf_str[13];
  DIR *d;
  struct dirent *dir;

  /* 0000:00:00.0 */
  snprintf(bdf_str, 13, "%04" PRIx32 ":%02" PRIx32 ":%02" PRIx32 ".%" PRIu32,
           static_cast<uint32_t>(bdf.domain_number & 0xffff),
           static_cast<uint32_t>(bdf.bus_number & 0xff),
           static_cast<uint32_t>(bdf.device_number & 0x1f),
           static_cast<uint32_t>(bdf.function_number & 0x7));

  d = opendir("/proc");
  if (!d) return AMDSMI_STATUS_NO_PERM;

  pids.clear();
  /* Find the pid folders in /proc/ that we have access to */
  while ((dir = readdir(d)) != NULL) {
    if (dir->d_type == DT_DIR) {
      /* Try to cast the name of the folder to a
       * number, if it fails, it is not */
      char *p;
      long int pid;

      pid = strtol(dir->d_name, &p, 10);
      if (*p != 0) continue;

      /* Check if fdinfo is accesible */
      std::string path = "/proc/" + std::string(dir->d_name) + "/fdinfo/";

      if (access(path.c_str(), R_OK)) continue;

      /* check if GPU is present */
      if (gpuvsmi_pid_is_gpu(path, bdf_str)) continue;
      pids.push_back(pid);
    }
  }
  closedir(d);

  *size = pids.size();
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t gpuvsmi_get_pid_info(const amdsmi_bdf_t &bdf, long int pid,
                                     amdsmi_proc_info_t &info) {
  char bdf_str[13];
  DIR *d;
  struct dirent *dir;

  /* 0000:00:00.0 */
  snprintf(bdf_str, 13, "%04" PRIx32 ":%02" PRIx32 ":%02" PRIx32 ".%" PRIu32,
           static_cast<uint32_t>(bdf.domain_number & 0xffff),
           static_cast<uint32_t>(bdf.bus_number & 0xff),
           static_cast<uint32_t>(bdf.device_number & 0x1f),
           static_cast<uint32_t>(bdf.function_number & 0x7));

  std::string path = "/proc/" + std::to_string(pid) + "/fdinfo/";
  std::string name_path = "/proc/" + std::to_string(pid) + "/exe";
  std::string cgroup_path = "/proc/" + std::to_string(pid) + "/cgroup";

  if (gpuvsmi_pid_is_gpu(path.c_str(), bdf_str)) {
    return AMDSMI_STATUS_INVAL;
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
        if (sscanf(bdfline.c_str(), "drm-pdev:       %s", &fd_bdf_str[0]) != 1) continue;

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
  std::string name = (len > 0) ? std::string(exe_realpath, len) : "N/A";

  if (name.empty()) return AMDSMI_STATUS_API_FAILED;

  strncpy(info.name, name.c_str(),
          std::min((unsigned long)AMDSMI_MAX_STRING_LENGTH, name.length()));

  for (int i = 0; i < AMDSMI_MAX_CONTAINER_TYPE; i++) {
    std::ifstream cgroup_info(cgroup_path.c_str());
    std::string container_id;
    for (std::string line; getline(cgroup_info, line);) {
      if (line.find(container_type_name[i]) != std::string::npos) {
        container_id = line.substr(line.find(container_type_name[i]) +
                                   strlen(container_type_name[i]) + 1, 16);
        strcpy(info.container_name, container_id.c_str());
        break;
      }
    }
    if (strlen(info.container_name) > 0) break;
  }
  info.pid = (uint32_t)pid;

  return AMDSMI_STATUS_SUCCESS;
}

}  // extern "C"
