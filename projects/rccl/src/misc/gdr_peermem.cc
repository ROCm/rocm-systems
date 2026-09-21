/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "gdr_peermem.h"
#include "debug.h"

#include <dirent.h>
#include <limits.h>
#include <stdio.h>
#include <unistd.h>

int ncclIbScanPeerMemClients(const char* const* basePaths) {
  int found = 0;
  for (int i = 0; basePaths[i] && found == 0; ++i) {
    DIR* dir = opendir(basePaths[i]);
    if (dir == NULL) continue;
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
      if (entry->d_name[0] == '.') continue;
      // Registration only completes once the client's sysfs group, and so its `version`
      // attribute, is in place, so probing for `version` avoids trusting readdir's d_type.
      char versionPath[PATH_MAX];
      int len = snprintf(versionPath, sizeof(versionPath), "%s/%s/version", basePaths[i], entry->d_name);
      if (len < 0 || len >= (int)sizeof(versionPath)) continue;
      if (access(versionPath, F_OK) != 0) continue;
      found = 1;
      INFO(NCCL_INIT, "Found peer memory client %s/%s", basePaths[i], entry->d_name);
      break;
    }
    closedir(dir);
  }
  if (found == 0) INFO(NCCL_INIT, "No peer memory client found, GDR via peermem disabled");
  return found;
}

int ncclIbScanDefaultPeerMemClients(void) {
  // `memory_peers` lives under `/sys/kernel/mm/` on Linux 5.15 (e.g. Ubuntu 22.04); newer kernels
  // may omit it or place it under `/sys/kernel/` or `/sys/`, depending on the ib_peer_mem module.
  static const char* const memoryPeersPaths[] = {"/sys/kernel/mm/memory_peers", "/sys/kernel/memory_peers",
                                                 "/sys/memory_peers", NULL};
  return ncclIbScanPeerMemClients(memoryPeersPaths);
}
