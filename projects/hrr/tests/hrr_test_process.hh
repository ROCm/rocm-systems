/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <cerrno>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#include <sys/wait.h>
#endif

namespace hrr::test {

// Minimal, shell-free subprocess helper for HRR behavior tests.
//
// It implements only what the migrated HRR tests need: overriding environment
// variables for the child, launching a child process without going through a
// shell, and optionally capturing its stdout. It replaces the hip-tests
// SpawnProc helper. Extend only this file if Windows ever needs richer process
// semantics rather than re-importing the hip-tests process helpers.
//
// POSIX: the child is launched with fork()/execvp(). The argument string is
// split on whitespace into argv tokens; no shell is involved, so quote
// characters are literal parts of a token (a token like "name" is passed to the
// child including the quotes — Catch2's own test-spec parser then strips them).
// Environment overrides are applied in the child only (via setenv after fork),
// so the parent's environment is never mutated; from the child's point of view
// an overridden variable such as PATH is replaced entirely with the given value.
//
// run() returns the child's exit code, or 128 + signal number if it was killed
// by a signal (127 if the process could not be launched). Both the capturing
// and non-capturing paths use the same normalization.
class SpawnProc {
 public:
  explicit SpawnProc(std::string exe, bool capture_stdout = false)
      : exe_(std::move(exe)), capture_stdout_(capture_stdout) {}

  void setEnv(const std::string& key, const std::string& value) {
    env_.push_back({key, value});
  }

  int run(const std::string& args) {
    output_.clear();
    std::vector<std::string> tokens = split_args(args);

#if defined(_WIN32)
    return run_windows(tokens);
#else
    return run_posix(tokens);
#endif
  }

  const std::string& getOutput() const { return output_; }

 private:
  // Split on whitespace into argv tokens, keeping every other character
  // (including quotes) literal. No shell-style quote or escape processing.
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
    // argv[0] is the executable itself; execvp performs the PATH lookup (using
    // the child's environment) when exe_ is not an absolute/relative path.
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
      // Child: apply environment overrides (affects this process only), wire up
      // stdout capture if requested, then exec. On failure exit with 127.
      for (const auto& kv : env_) {
        ::setenv(kv.first.c_str(), kv.second.c_str(), 1);
      }
      if (capture_stdout_) {
        ::close(pipefd[0]);
        ::dup2(pipefd[1], STDOUT_FILENO);
        ::close(pipefd[1]);
      }
      ::execvp(exe_.c_str(), argv.data());
      ::_exit(127);
    }

    // Parent.
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
    while (::waitpid(pid, &status, 0) < 0) {
      if (errno != EINTR) return 127;
    }
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return 127;
  }
#endif

#if defined(_WIN32)
  int run_windows(const std::vector<std::string>& tokens) {
    // Build a command line: the executable followed by the raw tokens. Callers
    // are responsible for quoting paths that contain spaces (CreateProcess does
    // not use a shell). Rejoining with single spaces preserves the caller's
    // quoting.
    std::string cmdline = exe_;
    for (const auto& t : tokens) {
      cmdline += " ";
      cmdline += t;
    }

    // Build the child environment block: inherit the parent's environment, then
    // apply overrides, without mutating the parent's environment.
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
      si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
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
    DWORD exit_code = 127;
    GetExitCodeProcess(pi.hProcess, &exit_code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return static_cast<int>(exit_code);
  }

  std::string build_env_block() const {
    // Start from the parent's environment, apply overrides (case-insensitive on
    // the variable name, per Windows semantics), and emit a double-null
    // terminated block for CreateProcess.
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
  bool capture_stdout_ = false;
  std::string output_;
  std::vector<std::pair<std::string, std::string>> env_;
};

}  // namespace hrr::test
