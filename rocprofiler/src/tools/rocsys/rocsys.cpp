/* Copyright (c) 2022 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <dlfcn.h>
#include <fcntl.h>

#include <iostream>
#include <algorithm>
#include <vector>

#include "src/utils/filesystem.hpp"

/*
mpiexec -n 16 rocsys --session-new test launch python app.py
rocsys
*/

namespace {
struct shmd_t {
  int command;
};
struct shmd_t* shmd = nullptr;
}  // namespace

namespace fs = rocprofiler::common::filesystem;

void report(const char* msg, int terminate) {
  std::cerr << msg << ": " << errno << std::endl;
  if (terminate) exit(-1); /* failure */
}

pid_t getpid(void);
void exit(int);

int toint(const char* chararr) {
  int num = 0;
  std::string s;
  if (chararr != nullptr) {
    s = chararr;
  }
  if (s.empty()) {
    return 0;
  }
  transform(s.begin(), s.end(), s.begin(), ::tolower);
  for (auto itr : s) {
    if (itr >= 'a' && itr <= 'z') {
      num += (itr - 'a' + 1) * 1 + (itr - 'a' + 1) * 10 + (itr - 'a' + 1) * 100;
    }
  }
  if (std::to_string(num).length() > 16) {
    perror("generated number is more than permissable limit");
    exit(1);
  }
  return num;
}

int main(int argc, char* argv[]) {
  fs::path bin_path;

  if (argc < 4) {
    perror(
        "rocsys: launch must be preceeded by --session <name>\n"
        "e.g. rocsys --session <SESSION_NAME> launch <MPI_COMMAND> <MPI_ARGUMENTS> rocprofv2\n\t "
        "<ROCPROFV2_OPTIONS> <APP_EXEC>\n"
        "where all mpiexec options must come before rocsys\n"
        "rocsys: start must be preceeded by --session <name>\n\t"
        "rocsys --session <name> start \n"
        "rocsys: stop must be preceeded by --session <name>\n \t"
        "rocsys --session <name> stop \n"
        "rocsys: exit must be preceeded by --session <name>\n \t"
        "rocsys --session <name> exit \n");
    exit(1);
  }

  if (Dl_info dl_info; dladdr(reinterpret_cast<void*>(main), &dl_info) != 0) {
    bin_path = fs::path(dl_info.dli_fname).remove_filename();
  } else {
    bin_path = "/opt/rocm";
  }

  std::string session_name = "default_session_roctracer";
  bool session_name_found{false};
  std::string message;
  bool message_flag{true};
  bool args_flag{true};
  int i = 0;
  int session_id = 0;
  int* session_id_shm;


  int sys_type = 0;

  while (std::string(argv[i]).find("rocsys") != std::string::npos) {
    i++;
    break;
  }

  const char* shared_memory_key = "ROC_SYS_KEY";

  const int SIZE = sizeof(int);

  for (; i < argc; i++) {
    if (args_flag && strcmp(argv[i], "--session") == 0) {
      i++;
      if (argv[i])
        session_name = argv[i];
      else
        std::cerr << "Error: missing session name" << std::endl;
      session_name_found = true;
      session_id = toint(argv[i]);
    } else if (args_flag && message_flag && (strcmp(argv[i], "launch") == 0)) {
      sys_type = 3;
      message_flag = false;
      args_flag = false;
    } else if (args_flag && message_flag && (strcmp(argv[i], "start") == 0)) {
      sys_type = 4;
      message_flag = false;
      args_flag = false;
      break;
    } else if (args_flag && message_flag && (strcmp(argv[i], "stop") == 0)) {
      sys_type = 5;
      message_flag = false;
      args_flag = false;
      break;
    } else if (args_flag && message_flag && (strcmp(argv[i], "exit") == 0)) {
      sys_type = 6;
      message_flag = false;
      args_flag = false;
    } else if (!args_flag) {
      break;
    } else {
      report(
          "rocsys: launch must be preceeded by --session <name>\n"
          "e.g. rocsys --session <SESSION_NAME> launch <MPI_COMMAND> <MPI_ARGUMENTS> rocprofv2\n\t "
          "<ROCPROFV2_OPTIONS> <APP_EXEC>\n"
          "where all mpiexec options must come before rocsys\n"
          "rocsys: start must be preceeded by --session <name>\n\t"
          "rocsys --session <name> start \n"
          "rocsys: stop must be preceeded by --session <name>\n \t"
          "rocsys --session <name> stop \n"
          "rocsys: exit must be preceeded by --session <name>\n \t"
          "rocsys --session <name> exit \n",
          1);
    }
  }

  if (!session_name_found) session_id = toint(session_name.c_str());

  if (sys_type != 3) {
    static int shm_fd_sn = shm_open(std::to_string(session_id).c_str(), O_CREAT | O_RDWR, 0666);
    int status = ftruncate(shm_fd_sn, 1024);
    if (status < 0) std::cerr << "Invalid Command!" << std::endl;
    shmd = reinterpret_cast<struct shmd_t*>(mmap(0, 1024, PROT_WRITE, MAP_SHARED, shm_fd_sn, 0));
  }


  switch (sys_type) {
    case 3: {
      static int shm_fd = shm_open(shared_memory_key, O_CREAT | O_RDWR, 0666);
      int status = ftruncate(shm_fd, SIZE);
      if (status < 0) std::cerr << "Invalid Command!" << std::endl;
      session_id_shm = reinterpret_cast<int*>(mmap(0, SIZE, PROT_WRITE, MAP_SHARED, shm_fd, 0));
      *session_id_shm = session_id;
      printf("ROCSYS:: Session ID: %d\n", session_id);

      int argindex = i;
      std::vector<std::string> env_vars;
      env_vars.emplace_back("");
      if (std::string(argv[i]).find("rocprofv2") != std::string::npos) {
        env_vars.emplace_back("--roc-sys");
        env_vars.emplace_back(std::to_string(session_id));
      }
      for (argindex++; argindex < argc; argindex++) {
       if (std::string(argv[argindex]).find("rocprofv2") != std::string::npos) {
          fs::path command = bin_path;
          command.append("rocprofv2");
          env_vars.emplace_back(command.c_str());
          env_vars.emplace_back("--roc-sys");
          env_vars.emplace_back(std::to_string(session_id));
        } else {
          env_vars.emplace_back(argv[argindex]);
        }
      }
      char* exec_args[env_vars.size() + 1];
      for (uint32_t i = 0; i < env_vars.size(); i++) {
        exec_args[i] = const_cast<char*>(env_vars.at(i).c_str());
        std::cout << exec_args[i] << " ";
      }
      exec_args[env_vars.size()] = NULL;
      std::cout << std::endl;
      if (strncmp(argv[i], "rocprofv2", 8) != 0) {
        execvp(argv[i], exec_args);
      } else {
        fs::path command = bin_path;
        command.append("rocprofv2");
        execvp(command.c_str(), exec_args);
      }
      std::cout << "Error: can't launch server (" << errno << "):" << std::endl;
      exit(EXIT_FAILURE);
    }
    case 4: {
      shmd->command = 4;
      break;
    }
    case 5: {
      shmd->command = 5;
      break;
    }
    case 6: {
      shmd->command = 6;
      break;
    }
    default: {
      report("ROCSYS:: Error: Not possible to reach here, please report(invalid sys_type)!\n", 1);
    }
  }
  msync(shmd, sizeof(shmd->command), MS_SYNC | MS_INVALIDATE);
  return 1;
}
