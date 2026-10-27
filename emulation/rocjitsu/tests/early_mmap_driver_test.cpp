// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file early_mmap_driver_test.cpp
/// @brief Minimal executable used by the early-mmap preload regression test.

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <link.h>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char **argv) {
  if (argc == 1)
    return 0;
  void *handle = dlopen("libc.so.6", RTLD_NOW | RTLD_NOLOAD);
  link_map *map = nullptr;
  if (!handle || dlinfo(handle, RTLD_DI_LINKMAP, &map) != 0 || !map)
    return 1;
  if (argc == 3 && std::strcmp(argv[1], "--verify-libc-alias") == 0)
    return std::strcmp(map->l_name, argv[2]) == 0 ? 0 : 1;
  if (argc != 2 || std::strcmp(argv[1], "--libc-alias") != 0)
    return 1;

  char directory[] = "/tmp/rocjitsu-libc-alias-XXXXXX";
  if (!mkdtemp(directory))
    return 1;
  const std::string alias = std::string(directory) + "/libc-alias.so";
  int result = 1;
  if (symlink(map->l_name, alias.c_str()) == 0) {
    // Keep any sanitizer first in the preload list. The loader maps preloads
    // before their dependencies, so this alias supplies libc to every DSO.
    const char *previous = std::getenv("LD_PRELOAD");
    const std::string preload = previous ? std::string(previous) + ":" + alias : alias;
    if (setenv("LD_PRELOAD", preload.c_str(), 1) == 0) {
      const pid_t child = fork();
      if (child == 0) {
        execl("/proc/self/exe", argv[0], "--verify-libc-alias", alias.c_str(), nullptr);
        _exit(127);
      }
      if (child > 0) {
        int status = 0;
        pid_t waited;
        do {
          waited = waitpid(child, &status, 0);
        } while (waited < 0 && errno == EINTR);
        if (waited == child && WIFEXITED(status))
          result = WEXITSTATUS(status);
      }
    }
    unlink(alias.c_str());
  }
  rmdir(directory);
  dlclose(handle);
  return result;
}
