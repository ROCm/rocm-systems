/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hip_test_common.hh"
#include "hip_test_filesystem.hh"

#include <string>
#include <array>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <random>
#include <fstream>
#include <streambuf>
#include <thread>
#include <future>
#include <vector>
#include <sstream>
#include <algorithm>
#include <map>

#if HT_WIN
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <tchar.h>
#include <tlhelp32.h>
#else
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <errno.h>
#include <sys/un.h>
#include <sys/socket.h>
#endif

#if HT_WIN
typedef PROCESS_INFORMATION Process;
#else
typedef pid_t Process;
#endif

inline std::string getSelfExePath() {
#if HT_WIN
  char path[MAX_PATH];
  DWORD len = GetModuleFileName(NULL, path, MAX_PATH);
  return std::string(path, len);
#else
  char path[4096];
  ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
  if (len < 0) return "";
  path[len] = '\0';
  return std::string(path);
#endif
}

inline unsigned long getParentProcessId() {
#if HT_WIN
  DWORD pid = GetCurrentProcessId();
  HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
  if (snapshot == INVALID_HANDLE_VALUE) return 0;
  PROCESSENTRY32 pe = {};
  pe.dwSize = sizeof(pe);
  if (Process32First(snapshot, &pe)) {
    do {
      if (pe.th32ProcessID == pid) {
        CloseHandle(snapshot);
        return pe.th32ParentProcessID;
      }
    } while (Process32Next(snapshot, &pe));
  }
  CloseHandle(snapshot);
  return 0;
#else
  return static_cast<unsigned long>(getppid());
#endif
}

namespace hip {
/*
Class to spawn a process in isolation and test its standard output and return status.

How to use:
  hip::SpawnProc proc("ExeName", true);  // true = capture stdout
  proc.setEnv("MY_VAR", "value");        // optional: set env vars for child
  int exitCode = proc.run("optional args");
  std::string output = proc.getOutput();

For non-blocking spawn (e.g. IPC patterns):
  hip::SpawnProc proc(getSelfExePath());
  proc.spawn("arg1 arg2");
  // ... do work while child runs ...
  int exitCode = proc.wait();
*/
class SpawnProc {
  std::string exeName_;
  std::string resultStr_;
  std::string tmpFileName_;
  bool captureOutput_;
  Process process_{};
  bool spawned_ = false;
  std::future<int> asyncFuture_;
  std::map<std::string, std::string> envVars_;

  std::string getRandomString(size_t len = 6) {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, 25);

    std::string res;
    for (size_t i = 0; i < len; i++) {
      res += 'a' + dist(rng);
    }
    return res;
  }

  static std::vector<std::string> splitArgs(const std::string& s) {
    std::vector<std::string> tokens;
    std::istringstream iss(s);
    std::string tok;
    while (iss >> tok) tokens.push_back(tok);
    return tokens;
  }

  int doSpawn(const std::string& commandLineArgs) {
    std::vector<std::string> argStorage;
    argStorage.push_back(exeName_);
    if (!commandLineArgs.empty()) {
      auto parsed = splitArgs(commandLineArgs);
      argStorage.insert(argStorage.end(), parsed.begin(), parsed.end());
    }

#if HT_WIN
    std::string cmdLine;
    for (const auto& arg : argStorage) {
      if (!cmdLine.empty()) cmdLine += ' ';
      cmdLine += arg;
    }
    std::vector<char> cmdLineInput(cmdLine.begin(), cmdLine.end());
    cmdLineInput.push_back('\0');
    STARTUPINFO si = {};
    si.cb = sizeof(si);
    HANDLE hFile = INVALID_HANDLE_VALUE;
    BOOL inheritHandles = FALSE;

    if (captureOutput_) {
      SECURITY_ATTRIBUTES sa = {};
      sa.nLength = sizeof(sa);
      sa.bInheritHandle = TRUE;
      hFile = CreateFile(tmpFileName_.c_str(), GENERIC_WRITE, 0, &sa,
                         CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
      if (hFile != INVALID_HANDLE_VALUE) {
        si.dwFlags |= STARTF_USESTDHANDLES;
        si.hStdOutput = hFile;
        si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
        si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
        inheritHandles = TRUE;
      }
    }

    std::vector<char> envBlock;
    LPVOID lpEnvironment = NULL;
    if (!envVars_.empty()) {
      auto icaseFind = [&](const std::string& key) {
        std::string lowerKey = key;
        std::transform(lowerKey.begin(), lowerKey.end(), lowerKey.begin(), ::tolower);
        for (const auto& kv : envVars_) {
          std::string lowerMapKey = kv.first;
          std::transform(lowerMapKey.begin(), lowerMapKey.end(), lowerMapKey.begin(), ::tolower);
          if (lowerMapKey == lowerKey) return true;
        }
        return false;
      };
      LPCH currentEnv = GetEnvironmentStrings();
      if (currentEnv) {
        LPSTR var = currentEnv;
        while (*var) {
          std::string entry(var);
          auto eq = entry.find('=', 1);
          std::string key = (eq != std::string::npos) ? entry.substr(0, eq) : entry;
          if (!icaseFind(key)) {
            envBlock.insert(envBlock.end(), entry.begin(), entry.end());
            envBlock.push_back('\0');
          }
          var += strlen(var) + 1;
        }
        FreeEnvironmentStrings(currentEnv);
      }
      for (const auto& kv : envVars_) {
        std::string entry = kv.first + "=" + kv.second;
        envBlock.insert(envBlock.end(), entry.begin(), entry.end());
        envBlock.push_back('\0');
      }
      envBlock.push_back('\0');
      lpEnvironment = envBlock.data();
    }

    memset(&process_, 0, sizeof(process_));
    BOOL ok = CreateProcess(exeName_.c_str(), cmdLineInput.data(), NULL, NULL,
                            inheritHandles, 0, lpEnvironment, NULL, &si, &process_);

    if (hFile != INVALID_HANDLE_VALUE) CloseHandle(hFile);
    if (!ok) return GetLastError();
    spawned_ = true;
    return 0;
#else
    std::vector<char*> argv;
    for (auto& arg : argStorage) {
      argv.push_back(&arg[0]);
    }
    argv.push_back(nullptr);

    process_ = fork();
    if (process_ == 0) {
      for (const auto& kv : envVars_) {
        setenv(kv.first.c_str(), kv.second.c_str(), 1);
      }
      if (captureOutput_) {
        int fd = open(tmpFileName_.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd >= 0) {
          dup2(fd, STDOUT_FILENO);
          close(fd);
        }
      }
      execvp(argv[0], argv.data());
      _exit(127);
    } else if (process_ < 0) {
      return -1;
    }
    spawned_ = true;
    return 0;
#endif
  }

  int doWait() {
    if (!spawned_) return -1;

#if HT_WIN
    WaitForSingleObject(process_.hProcess, INFINITE);
    DWORD ec;
    GetExitCodeProcess(process_.hProcess, &ec);
    CloseHandle(process_.hProcess);
    CloseHandle(process_.hThread);
    int exitCode = static_cast<int>(ec);
#else
    int status = 0;
    int exitCode = -1;
    for (;;) {
      pid_t ret = waitpid(process_, &status, 0);
      if (ret < 0) {
        if (errno == EINTR) {
          continue;  // Retry if interrupted by a signal
        }
        return errno;
      }
      if (WIFEXITED(status)) {
        exitCode = WEXITSTATUS(status);
        break;
      }
      if (WIFSIGNALED(status)) {
        // Use a conventional encoding for signal termination
        exitCode = 128 + WTERMSIG(status);
        break;
      }
    }
#endif

    if (captureOutput_) {
      std::ifstream t(tmpFileName_.c_str());
      resultStr_ = std::string((std::istreambuf_iterator<char>(t)),
                                std::istreambuf_iterator<char>());
      t.close();
    }
    spawned_ = false;
    return exitCode;
  }

 public:
  SpawnProc(std::string exeName, bool captureOutput = false)
      : exeName_(std::move(exeName)), captureOutput_(captureOutput) {
    if (!fs::path(exeName_).is_absolute()) {
      auto dir = fs::path(TestContext::get().currentPath());
      dir /= exeName_;
      exeName_ = dir.string();
    }
    if (TestContext::get().isWindows()) {
      if (fs::path(exeName_).extension().empty()) {
        exeName_ += ".exe";
      }
    }
    INFO("Testing that exe exists: " << exeName_);
    REQUIRE(fs::exists(exeName_));

    if (captureOutput_) {
      auto path = fs::temp_directory_path();
      path /= getRandomString();
      tmpFileName_ = path.string();
      INFO("Testing that capture file does not exist already: " << tmpFileName_);
      REQUIRE(!fs::exists(tmpFileName_));
    }
  }

  void setEnv(const std::string& key, const std::string& value) {
    envVars_[key] = value;
  }

  /**
   * Non-blocking: start the child process and return immediately.
   * Returns 0 on success, non-zero on failure.
   * Use wait() afterwards to collect the exit code.
   */
  int spawn(std::string commandLineArgs = "") {
    return doSpawn(commandLineArgs);
  }

  /**
   * Wait for a previously spawned or async-launched process.
   * Returns the child exit code.
   */
  int wait() {
    if (asyncFuture_.valid()) {
      asyncFuture_.wait();
      return asyncFuture_.get();
    }
    return doWait();
  }

  /**
   * Blocking convenience: spawn + wait.
   * Returns the child exit code.
   */
  int run(std::string commandLineArgs = "") {
    int err = doSpawn(commandLineArgs);
    if (err != 0) return err;
    return doWait();
  }

  void run_async(std::string commandLineArgs = "") {
    asyncFuture_ = std::async(std::launch::async, &hip::SpawnProc::run, this, commandLineArgs);
  }

  Process getProcess() const { return process_; }

  std::string getOutput() { return resultStr_; }
};
}  // namespace hip
