/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "kernel_config.h"

#include <stdio.h>
#include <string.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <fstream>
#include <iterator>

namespace {

const char* kKernelConfigPathTemplates[] = {
  "/proc/config.gz",
  "/boot/config-%s",
  "/usr/src/linux-%s/.config",
  "/usr/src/linux/.config",
  "/usr/lib/modules/%s/config",
  "/usr/lib/ostree-boot/config-%s",
  "/usr/lib/kernel/config-%s",
  "/usr/src/linux-headers-%s/.config",
  "/lib/modules/%s/build/.config",
};

bool isConfigGzPath(const char* pathTemplate) {
  return strstr(pathTemplate, "/proc/config.gz") != NULL;
}

void formatKernelConfigPath(const char* pathTemplate, const char* release,
                            char* pathOut, size_t pathOutLen) {
  snprintf(pathOut, pathOutLen, pathTemplate, release);
}

}  // namespace

bool ncclKernelConfigContentHasOption(const std::string& content, const char* option) {
  if (option == NULL || option[0] == '\0') return false;
  return content.find(option) != std::string::npos;
}

bool ncclKernelConfigReadFile(const char* path, std::string* content) {
  if (path == NULL || content == NULL) return false;

  std::ifstream in(path);
  if (!in.is_open()) return false;

  content->assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  return true;
}

bool ncclKernelConfigReadGzip(const char* path, std::string* content) {
  if (path == NULL || content == NULL) return false;
  if (access(path, R_OK) != 0) return false;

  char cmd[256];
  snprintf(cmd, sizeof(cmd), "zcat %s 2>/dev/null", path);

  FILE* fp = popen(cmd, "r");
  if (fp == NULL) return false;

  content->clear();
  char buf[512];
  while (fgets(buf, sizeof(buf), fp) != NULL) {
    content->append(buf);
  }

  const int status = pclose(fp);
  return status != -1 && !content->empty();
}

bool ncclKernelConfigReadFirstAvailable(std::string* content, char* pathOut,
                                        size_t pathOutLen) {
  if (content == NULL) return false;

  struct utsname utsname;
  if (uname(&utsname) == -1) return false;

  char resolvedPath[128];
  for (const char* pathTemplate : kKernelConfigPathTemplates) {
    formatKernelConfigPath(pathTemplate, utsname.release, resolvedPath, sizeof(resolvedPath));

    bool readOk = false;
    if (isConfigGzPath(pathTemplate)) {
      readOk = ncclKernelConfigReadGzip(resolvedPath, content);
    } else {
      readOk = ncclKernelConfigReadFile(resolvedPath, content);
    }

    if (!readOk) continue;

    if (pathOut != NULL && pathOutLen > 0) {
      snprintf(pathOut, pathOutLen, "%s", resolvedPath);
    }
    return true;
  }

  return false;
}

bool ncclKernelHasConfigOption(const char* option) {
  std::string content;
  char path[128];

  struct utsname utsname;
  if (uname(&utsname) == -1) return false;

  for (const char* pathTemplate : kKernelConfigPathTemplates) {
    formatKernelConfigPath(pathTemplate, utsname.release, path, sizeof(path));

    bool readOk = false;
    if (isConfigGzPath(pathTemplate)) {
      readOk = ncclKernelConfigReadGzip(path, &content);
    } else {
      readOk = ncclKernelConfigReadFile(path, &content);
    }

    if (!readOk) continue;
    if (ncclKernelConfigContentHasOption(content, option)) return true;
  }

  return false;
}

bool ncclIommuPassthroughOk(const char* cmdline) {
  if (cmdline != NULL && strstr(cmdline, "iommu=pt") != NULL) return true;
  return ncclKernelHasConfigOption("CONFIG_IOMMU_DEFAULT_PASSTHROUGH=y");
}
