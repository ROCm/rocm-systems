/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <algorithm>
#include <windows.h>
#else
#include <csignal>
#include <unistd.h>
#include <sys/wait.h>
#endif

namespace hrr::test {

// Shell-free subprocess helper. POSIX splits the argument string on whitespace and keeps quotes literal
// (Catch2's test-spec parser strips them). Environment overrides apply in the child only.
// run() returns the exit code, 128 + signal number if killed by a signal, or 127 if the launch failed.
class SpawnProc {
 public:
  // capture_stderr merges stderr into getOutput(); it has no effect without capture_stdout.
  explicit SpawnProc(std::string exe, bool capture_stdout = false,
                     bool capture_stderr = false)
      : exe_(std::move(exe)),
        capture_stdout_(capture_stdout),
        capture_stderr_(capture_stderr) {}

  void setEnv(const std::string& key, const std::string& value) {
    env_.push_back({key, value});
  }

  // Prefer this over setEnv(key, ""): CLR's flag parser turns an empty HIP_HRR_CAPTURE_OUTPUT into a space, which still enables capture.
  void unsetEnv(const std::string& key) { unset_env_.push_back(key); }

  static constexpr int kKilledOnTimeout = 128 + 9;

  int run(const std::string& args) { return run_impl(args, 0); }

  // Returns kKilledOnTimeout if the deadline fires; timeout_seconds <= 0 means no deadline.
  int runWithTimeout(const std::string& args, int timeout_seconds) {
    return run_impl(args, timeout_seconds);
  }

  const std::string& getOutput() const { return output_; }

 private:
  int run_impl(const std::string& args, int timeout_seconds) {
    output_.clear();
    timeout_seconds_ = timeout_seconds;
    std::vector<std::string> tokens = split_args(args);

#if defined(_WIN32)
    return run_windows(tokens);
#else
    return run_posix(tokens);
#endif
  }

  static std::vector<std::string> split_args(const std::string& args) {
    std::vector<std::string> out;
    std::string cur;
    bool in_token = false;
    for (char c : args) {
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        if (in_token) {
          out.push_back(cur);
          cur.clear();
          in_token = false;
        }
      } else {
        cur.push_back(c);
        in_token = true;
      }
    }
    if (in_token) out.push_back(cur);
    return out;
  }

#if !defined(_WIN32)
  int run_posix(const std::vector<std::string>& tokens) {
    // execvp does the PATH lookup, using the child's environment, when exe_ is not a path.
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>(exe_.c_str()));
    for (const auto& t : tokens) argv.push_back(const_cast<char*>(t.c_str()));
    argv.push_back(nullptr);

    int pipefd[2] = {-1, -1};
    if (capture_stdout_ && ::pipe(pipefd) != 0) return 127;

    pid_t pid = ::fork();
    if (pid < 0) {
      if (capture_stdout_) {
        ::close(pipefd[0]);
        ::close(pipefd[1]);
      }
      return 127;
    }

    if (pid == 0) {
      for (const auto& key : unset_env_) {
        ::unsetenv(key.c_str());
      }
      for (const auto& kv : env_) {
        ::setenv(kv.first.c_str(), kv.second.c_str(), 1);
      }
      if (capture_stdout_) {
        ::close(pipefd[0]);
        ::dup2(pipefd[1], STDOUT_FILENO);
        if (capture_stderr_) ::dup2(pipefd[1], STDERR_FILENO);
        ::close(pipefd[1]);
      }
      ::execvp(exe_.c_str(), argv.data());
      // Without a message, exit 127 is indistinguishable from a missing dynamic library.
      // The loader writes its own line; this covers an execve failure (ENOENT, EACCES on a noexec mount).
      dprintf(STDERR_FILENO, "hrr spawn: exec '%s': %s\n", exe_.c_str(),
              strerror(errno));
      ::_exit(127);
    }

    // Arm the deadline before draining the pipe: the read blocks until the child exits,
    // so the kill has to come from another thread.
    std::atomic<bool> finished{false};
    std::thread watchdog;
    if (timeout_seconds_ > 0) {
      watchdog = std::thread([&finished, pid, secs = timeout_seconds_]() {
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(secs);
        while (!finished.load() && std::chrono::steady_clock::now() < deadline)
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
        if (!finished.load()) ::kill(pid, SIGKILL);
      });
    }

    if (capture_stdout_) {
      ::close(pipefd[1]);
      char buffer[4096];
      ssize_t n;
      while (true) {
        n = ::read(pipefd[0], buffer, sizeof(buffer));
        if (n > 0) {
          output_.append(buffer, static_cast<size_t>(n));
        } else if (n == 0) {
          break;
        } else if (errno != EINTR) {
          break;
        }
      }
      ::close(pipefd[0]);
    }

    int status = 0;
    int rc = 0;
    while (::waitpid(pid, &status, 0) < 0) {
      if (errno != EINTR) {
        rc = 127;
        break;
      }
    }
    finished.store(true);
    if (watchdog.joinable()) watchdog.join();
    if (rc == 127) return 127;
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return 127;
  }
#endif

#if defined(_WIN32)
  int run_windows(const std::vector<std::string>& tokens) {
    // Callers quote paths with spaces: CreateProcess uses no shell, so rejoining with single
    // spaces preserves their quoting.
    std::string cmdline = exe_;
    for (const auto& t : tokens) {
      cmdline += " ";
      cmdline += t;
    }

    std::string env_block = build_env_block();

    HANDLE read_h = nullptr;
    HANDLE write_h = nullptr;
    SECURITY_ATTRIBUTES sa;
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    sa.lpSecurityDescriptor = nullptr;

    if (capture_stdout_) {
      if (!CreatePipe(&read_h, &write_h, &sa, 0)) return 127;
      SetHandleInformation(read_h, HANDLE_FLAG_INHERIT, 0);
    }

    STARTUPINFOA si;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    if (capture_stdout_) {
      si.dwFlags |= STARTF_USESTDHANDLES;
      si.hStdOutput = write_h;
      si.hStdError = capture_stderr_ ? write_h : GetStdHandle(STD_ERROR_HANDLE);
      si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    }

    PROCESS_INFORMATION pi;
    ZeroMemory(&pi, sizeof(pi));

    std::vector<char> cmd_mutable(cmdline.begin(), cmdline.end());
    cmd_mutable.push_back('\0');

    BOOL ok = CreateProcessA(
        nullptr, cmd_mutable.data(), nullptr, nullptr,
        /*bInheritHandles=*/capture_stdout_ ? TRUE : FALSE, 0,
        env_block.empty() ? nullptr : env_block.data(), nullptr, &si, &pi);

    if (!ok) {
      if (capture_stdout_) {
        CloseHandle(read_h);
        CloseHandle(write_h);
      }
      return 127;
    }

    std::atomic<bool> finished{false};
    std::thread watchdog;
    if (timeout_seconds_ > 0) {
      HANDLE proc_h = pi.hProcess;
      watchdog = std::thread([&finished, proc_h, secs = timeout_seconds_]() {
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(secs);
        while (!finished.load() && std::chrono::steady_clock::now() < deadline)
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
        if (!finished.load())
          TerminateProcess(proc_h, static_cast<UINT>(kKilledOnTimeout));
      });
    }

    if (capture_stdout_) {
      CloseHandle(write_h);
      char buffer[4096];
      DWORD n = 0;
      while (ReadFile(read_h, buffer, sizeof(buffer), &n, nullptr) && n > 0) {
        output_.append(buffer, n);
      }
      CloseHandle(read_h);
    }

    WaitForSingleObject(pi.hProcess, INFINITE);
    finished.store(true);
    if (watchdog.joinable()) watchdog.join();
    DWORD exit_code = 127;
    GetExitCodeProcess(pi.hProcess, &exit_code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return static_cast<int>(exit_code);
  }

  std::string build_env_block() const {
    // Start from the parent's environment, apply overrides (case-insensitive names, per Windows),
    // and emit a double-null-terminated block for CreateProcess.
    std::vector<std::pair<std::string, std::string>> vars;
    LPCH env = GetEnvironmentStringsA();
    if (env) {
      for (LPCH p = env; *p != '\0';) {
        std::string entry(p);
        p += entry.size() + 1;
        size_t eq = entry.find('=');
        if (eq == std::string::npos || eq == 0) continue;  // skip drive entries
        vars.push_back({entry.substr(0, eq), entry.substr(eq + 1)});
      }
      FreeEnvironmentStringsA(env);
    }
    auto ieq = [](const std::string& a, const std::string& b) {
      return a.size() == b.size() && _stricmp(a.c_str(), b.c_str()) == 0;
    };
    for (const auto& key : unset_env_) {
      vars.erase(std::remove_if(vars.begin(), vars.end(),
                                [&](const auto& v) { return ieq(v.first, key); }),
                 vars.end());
    }
    for (const auto& kv : env_) {
      bool replaced = false;
      for (auto& v : vars) {
        if (ieq(v.first, kv.first)) {
          v.second = kv.second;
          replaced = true;
          break;
        }
      }
      if (!replaced) vars.push_back(kv);
    }
    std::string block;
    for (const auto& v : vars) {
      block += v.first;
      block += '=';
      block += v.second;
      block.push_back('\0');
    }
    block.push_back('\0');
    return block;
  }
#endif

  std::string exe_;
  int timeout_seconds_ = 0;
  bool capture_stdout_ = false;
  bool capture_stderr_ = false;
  std::string output_;
  std::vector<std::pair<std::string, std::string>> env_;
  std::vector<std::string> unset_env_;
};

}  // namespace hrr::test
