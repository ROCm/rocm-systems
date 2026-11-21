/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "core/inc/amd_dmem_cgroup.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <fstream>
#include <string>

namespace rocr {
namespace AMD {
namespace dmem_cgroup {

namespace {

constexpr const char kCgroupMount[] = "/sys/fs/cgroup";

// The key the dmem controller uses for a GPU's framebuffer region:
// "drm/<domain>:<bus>:<device>.<function>/vram".
std::string VramRegionKey(HSAuint32 domain, HSAuint32 location_id) {
  char key[64];
  snprintf(key, sizeof(key), "drm/%04x:%02x:%02x.%x/vram", domain, (location_id >> 8) & 0xFF,
           (location_id >> 3) & 0x1F, location_id & 0x7);
  return std::string(key);
}

// The calling process's cgroup as a path relative to the cgroup2 mount, from the "0::<path>" line
// of /proc/self/cgroup. Inside a cgroup namespace this is "/", so the mount root is the process's
// own cgroup.
bool SelfCgroupPath(std::string* path) {
  std::ifstream file("/proc/self/cgroup");
  std::string line;
  while (std::getline(file, line)) {
    if (line.compare(0, 3, "0::") != 0) continue;
    *path = line.substr(3);
    while (!path->empty() && path->back() == '/') path->pop_back();
    return true;
  }
  return false;
}

// Reads one region's value from a dmem interface file, where each line is "<region key> <value>"
// and the value is a byte count or "max" for uncapped. Returns false if the file is unreadable,
// the region is absent, or it is uncapped.
bool ReadRegionValue(const std::string& path, const std::string& key, HSAuint64* value) {
  FILE* file = fopen(path.c_str(), "r");
  if (file == nullptr) return false;

  bool found = false;
  char line[256];
  while (fgets(line, sizeof(line), file) != nullptr) {
    // Require the separating space so a key cannot match another key's prefix.
    if (strncmp(line, key.c_str(), key.size()) != 0 || line[key.size()] != ' ') continue;

    const char* field = line + key.size() + 1;
    if (strncmp(field, "max", 3) == 0) break;

    char* end = nullptr;
    unsigned long long parsed = strtoull(field, &end, 10);
    if (end != field) {
      *value = static_cast<HSAuint64>(parsed);
      found = true;
    }
    break;
  }

  fclose(file);
  return found;
}

}  // namespace

bool QueryVramLimit(HSAuint32 domain, HSAuint32 location_id, VramLimit* limit) {
  std::string cgroup;
  if (!SelfCgroupPath(&cgroup)) return false;
  const std::string key = VramRegionKey(domain, location_id);

  // Any level from the process's cgroup up to the root can refuse an allocation,
  // so the cap is the tightest dmem.max on that path and the headroom the least
  // dmem.max minus dmem.current.
  bool found = false;
  HSAuint64 max = 0;
  HSAuint64 available = 0;
  for (std::string dir = std::string(kCgroupMount) + cgroup;; dir.erase(dir.rfind('/'))) {
    HSAuint64 level_max = 0;
    if (ReadRegionValue(dir + "/dmem.max", key, &level_max)) {
      HSAuint64 level_current = 0;
      ReadRegionValue(dir + "/dmem.current", key, &level_current);
      const HSAuint64 level_available = level_max - std::min(level_max, level_current);
      if (!found || level_max < max) max = level_max;
      if (!found || level_available < available) available = level_available;
      found = true;
    }
    if (dir == kCgroupMount) break;
  }

  if (!found) return false;
  limit->max = max;
  limit->available = available;
  return true;
}

}  // namespace dmem_cgroup
}  // namespace AMD
}  // namespace rocr
