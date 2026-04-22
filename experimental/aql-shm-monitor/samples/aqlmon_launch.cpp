// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <errno.h>
#include <limits.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

namespace {

[[noreturn]] void fail(const char* msg) {
  std::fprintf(stderr, "aqlmon_launch: %s\n", msg);
  std::exit(EXIT_FAILURE);
}

[[noreturn]] void failf(const char* prefix, const std::string& value) {
  std::fprintf(stderr, "aqlmon_launch: %s%s\n", prefix, value.c_str());
  std::exit(EXIT_FAILURE);
}

std::string read_self_exe() {
  char buffer[PATH_MAX] = {};
  const auto length = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
  if(length < 0) fail(std::strerror(errno));
  buffer[length] = '\0';
  return std::string{buffer};
}

bool file_exists(const fs::path& path) { return fs::exists(path) && fs::is_regular_file(path); }

fs::path find_library_dir() {
  if(const char* env = std::getenv("AQLMON_LAUNCH_LIBDIR"); env != nullptr && *env != '\0') {
    return fs::path{env};
  }

  const auto self = fs::path{read_self_exe()};
  const auto self_dir = self.parent_path();
  const auto prefix_lib = self_dir.parent_path() / "lib";

  if(file_exists(self_dir / "libaql_shm_monitor.so")) return self_dir;
  if(file_exists(prefix_lib / "libaql_shm_monitor.so")) return prefix_lib;

  fail("could not find libaql_shm_monitor.so next to the launcher or under ../lib");
}

std::string join_preload(const std::vector<std::string>& entries) {
  std::string value = {};
  for(size_t i = 0; i < entries.size(); ++i) {
    if(i > 0) value += ':';
    value += entries[i];
  }
  return value;
}

void set_env(const char* name, const std::string& value, bool overwrite = true) {
  if(setenv(name, value.c_str(), overwrite ? 1 : 0) != 0) fail(std::strerror(errno));
}

void prepend_env(const char* name, const std::string& value) {
  const char* current = std::getenv(name);
  if(current == nullptr || *current == '\0') {
    set_env(name, value);
    return;
  }

  set_env(name, value + ":" + current);
}

void usage() {
  std::fprintf(stderr,
               "usage: aqlmon_launch [--shm-name NAME] -- <app> [args...]\n");
}

}  // namespace

int main(int argc, char** argv) {
  if(argc < 3) {
    usage();
    return EXIT_FAILURE;
  }

  const char* shm_name = nullptr;
  int app_index = -1;

  for(int i = 1; i < argc; ++i) {
    const auto arg = std::string_view{argv[i]};
    if(arg == "--") {
      app_index = i + 1;
      break;
    }
    if(arg == "--shm-name") {
      if((i + 1) >= argc) fail("missing value after --shm-name");
      shm_name = argv[++i];
      continue;
    }

    failf("unknown argument: ", std::string{arg});
  }

  if(app_index <= 0 || app_index >= argc) fail("missing application after --");

  const auto libdir = find_library_dir();
  const auto monitor_lib = libdir / "libaql_shm_monitor.so";
  if(!file_exists(monitor_lib)) failf("missing monitor library: ", monitor_lib.string());

  prepend_env("LD_LIBRARY_PATH", libdir.string());
  prepend_env("LD_PRELOAD", join_preload({monitor_lib.string()}));
  if(shm_name != nullptr) set_env("AQLMONITOR_SHM_NAME", shm_name);

  execvp(argv[app_index], argv + app_index);
  fail(std::strerror(errno));
}
