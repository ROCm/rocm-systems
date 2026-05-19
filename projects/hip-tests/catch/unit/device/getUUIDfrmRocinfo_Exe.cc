/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>
#include <cstring>
#ifdef __linux__
#include <unistd.h>
#endif
#include <cstdio>
#include <map>
#include <sstream>
#include <vector>
#define COMMAND_LEN 256
#define BUFFER_LEN 512

#ifdef __linux__
int main(int argc, char** argv) {
  if (argc < 0) {
    return -1;
  }
  int testPassed = 0;
  FILE* fpipe;
  char command[COMMAND_LEN] = "";
  const char* rocmInfo = "rocminfo";

  snprintf(command, COMMAND_LEN, "%s", rocmInfo);
  strncat(command, " | grep -i Uuid:", COMMAND_LEN);
  // Execute the rocminfo command and extract the UUID info
  fpipe = popen(command, "r");
  if (fpipe == nullptr) {
    printf("Unable to create command file\n");
    return -1;
  }
  char command_op[BUFFER_LEN];
  int j = 0;
  std::map<int, std::vector<char>> uuid_map;
  while (fgets(command_op, BUFFER_LEN, fpipe)) {
    std::string rocminfo_line(command_op);
    if (std::string::npos != rocminfo_line.find("CPU-")) {
      continue;
    } else if (auto loc = rocminfo_line.find("GPU-"); loc != std::string::npos) {
      if (std::string::npos == rocminfo_line.find("GPU-XX")) {
        std::vector<char> t_uuid(20, 0);
        std::memcpy(t_uuid.data(), &rocminfo_line[loc], 20);
        uuid_map[j] = t_uuid;
      }
    }
    j++;
  }
  std::string s = argv[1];
  std::string delimiter = ",";
  size_t pos = 0;
  std::vector<std::string> token;
  while ((pos = s.find(delimiter)) != std::string::npos) {
    token.push_back(s.substr(4, 16));
    s.erase(0, pos + delimiter.length());
  }
  token.push_back(s.substr(4, 16));
  int devCount = 0;
  hipError_t localError;
  localError = hipGetDeviceCount(&devCount);
  if (localError == hipSuccess) {
    printf("HIP Api returned hipSuccess");
  }
  for (int i = 0; i < devCount; i++) {
    std::string uuid = token[i];
    std::string mapVal(uuid_map[i].begin(), uuid_map[i].begin() + uuid_map[i].size());
    if (memcmp(mapVal.substr(4, 16).c_str(), uuid.c_str(), 16) == 0) {
      testPassed += 1;
    }
  }
  if (testPassed == devCount) {
    return 1;
  }
  return 0;
}
#endif
