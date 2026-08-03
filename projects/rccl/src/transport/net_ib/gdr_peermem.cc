/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "gdr_peermem.h"
#include "debug.h"

#include <dirent.h>

int ncclIbScanPeerMemClients(const char* const* basePaths) {
  int found = 0;
  for (int i = 0; basePaths[i] && found == 0; ++i) {
    DIR* dir = opendir(basePaths[i]);
    if (dir == NULL) continue;
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
      if (entry->d_name[0] == '.') continue;
      if (entry->d_type != DT_DIR && entry->d_type != DT_UNKNOWN) continue;
      found = 1;
      INFO(NCCL_INIT, "Found peer memory client %s/%s", basePaths[i], entry->d_name);
      break;
    }
    closedir(dir);
  }
  return found;
}
