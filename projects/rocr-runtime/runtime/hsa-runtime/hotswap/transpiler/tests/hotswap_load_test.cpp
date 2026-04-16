// Head-to-head test: load AITER .co files directly through the HSA runtime's
// LoadCodeObject path, bypassing HIP's ISA compatibility check.
//
// Usage:
//   ./hotswap_load_test <co_dir> [--recursive]
//
// Required env vars:
//   HSA_HOTSWAP_ISA_OVERRIDE=gfx942
//   HSA_HOTSWAP_RULES=/dev/null
// For IR raiser:
//   HSA_HOTSWAP_IR_RAISER=1

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static std::vector<uint8_t> readFile(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return {};
  auto sz = f.tellg();
  f.seekg(0);
  std::vector<uint8_t> buf(sz);
  f.read(reinterpret_cast<char *>(buf.data()), sz);
  return buf;
}

static void collectCoFiles(const std::string &dir,
                           std::vector<std::string> &out, bool recursive) {
  DIR *d = opendir(dir.c_str());
  if (!d) return;
  while (struct dirent *e = readdir(d)) {
    if (e->d_name[0] == '.') continue;
    std::string full = dir + "/" + e->d_name;
    struct stat st;
    if (stat(full.c_str(), &st) != 0) continue;
    if (S_ISDIR(st.st_mode) && recursive) {
      collectCoFiles(full, out, recursive);
    } else if (S_ISREG(st.st_mode)) {
      std::string name = e->d_name;
      if (name.size() >= 3 && name.substr(name.size() - 3) == ".co")
        out.push_back(full);
      else if (name.size() >= 6 && name.substr(name.size() - 6) == ".hsaco")
        out.push_back(full);
    }
  }
  closedir(d);
}

static hsa_agent_t g_gpu_agent = {0};

static hsa_status_t findGpuAgent(hsa_agent_t agent, void *) {
  hsa_device_type_t type;
  hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
  if (type == HSA_DEVICE_TYPE_GPU) {
    g_gpu_agent = agent;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

static bool tryLoadCodeObject(hsa_agent_t agent, const std::vector<uint8_t> &data) {
  hsa_code_object_reader_t reader;
  hsa_status_t st = hsa_code_object_reader_create_from_memory(
      data.data(), data.size(), &reader);
  if (st != HSA_STATUS_SUCCESS) return false;

  hsa_executable_t exec;
  st = hsa_executable_create_alt(HSA_PROFILE_FULL,
                                  HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                  nullptr, &exec);
  if (st != HSA_STATUS_SUCCESS) {
    hsa_code_object_reader_destroy(reader);
    return false;
  }

  st = hsa_executable_load_agent_code_object(exec, agent, reader,
                                              nullptr, nullptr);
  bool ok = (st == HSA_STATUS_SUCCESS);

  if (ok) {
    st = hsa_executable_freeze(exec, nullptr);
    ok = (st == HSA_STATUS_SUCCESS);
  }

  hsa_executable_destroy(exec);
  hsa_code_object_reader_destroy(reader);
  return ok;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <co_dir> [--recursive]\n", argv[0]);
    return 1;
  }
  const char *dir = argv[1];
  bool recursive = false;
  for (int i = 2; i < argc; i++)
    if (strcmp(argv[i], "--recursive") == 0) recursive = true;

  const char *raiser = getenv("HSA_HOTSWAP_IR_RAISER");
  const char *override_isa = getenv("HSA_HOTSWAP_ISA_OVERRIDE");
  printf("=== HotSwap Load Test (HSA API) ===\n");
  printf("  HSA_HOTSWAP_ISA_OVERRIDE = %s\n", override_isa ? override_isa : "(unset)");
  printf("  HSA_HOTSWAP_IR_RAISER    = %s\n", raiser ? raiser : "(unset)");
  printf("  Mode: %s\n\n",
         (raiser && raiser[0] == '1') ? "IR RAISER" : "LEGACY TRANSPILER");

  hsa_status_t st = hsa_init();
  if (st != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "hsa_init failed: %d\n", st);
    return 1;
  }

  hsa_iterate_agents(findGpuAgent, nullptr);
  if (!g_gpu_agent.handle) {
    fprintf(stderr, "No GPU agent found\n");
    hsa_shut_down();
    return 1;
  }

  char agentName[64] = {};
  hsa_agent_get_info(g_gpu_agent, HSA_AGENT_INFO_NAME, agentName);
  printf("  GPU: %s\n", agentName);
  printf("  Directory: %s%s\n\n", dir, recursive ? " (recursive)" : "");

  std::vector<std::string> files;
  collectCoFiles(dir, files, recursive);
  std::sort(files.begin(), files.end());

  if (files.empty()) {
    printf("No .co/.hsaco files found in %s\n", dir);
    hsa_shut_down();
    return 1;
  }
  printf("Found %zu code object files\n\n", files.size());

  int passed = 0, failed = 0;
  double totalMs = 0;

  for (auto &f : files) {
    auto data = readFile(f);
    if (data.empty()) {
      printf("  SKIP  %-60s (cannot read)\n", f.c_str());
      continue;
    }

    auto t0 = std::chrono::steady_clock::now();
    bool ok = tryLoadCodeObject(g_gpu_agent, data);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    totalMs += ms;

    // Strip path prefix for readability
    std::string shortName = f;
    size_t pos = f.rfind("aiter_gfx950/");
    if (pos != std::string::npos) shortName = f.substr(pos + 13);

    if (ok) {
      printf("  OK    %-55s %8.1f ms\n", shortName.c_str(), ms);
      passed++;
    } else {
      printf("  FAIL  %-55s %8.1f ms\n", shortName.c_str(), ms);
      failed++;
    }
  }

  printf("\n=== Summary ===\n");
  printf("  Passed: %d / %zu\n", passed, files.size());
  printf("  Failed: %d / %zu\n", failed, files.size());
  printf("  Total time: %.1f ms (avg %.1f ms/file)\n",
         totalMs, files.size() > 0 ? totalMs / files.size() : 0.0);
  printf("  Mode: %s\n",
         (raiser && raiser[0] == '1') ? "IR RAISER" : "LEGACY TRANSPILER");

  hsa_shut_down();
  return failed > 0 ? 1 : 0;
}
