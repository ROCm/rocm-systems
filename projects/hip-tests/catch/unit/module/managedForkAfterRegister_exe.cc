/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
// Declares a __managed__ variable and forks before calling any HIP API, so the child
// drives the first managed-symbol touch of the process.
//
// Registering a managed variable runs from a static initializer. Allocating it eagerly
// there also brings up the runtime, which opens the KFD; libhsakmt then latches
// hsakmt_forked on the pid change and CHECK_KFD_OPEN fails every subsequent hsaKmt call
// for the lifetime of the child (libhsakmt/src/openclose.c). Deferring the allocation
// leaves that connection unopened until the application makes its first HIP call.
//
// The child has to come from a plain fork(), not hip::SpawnProc: that helper execs, and
// /dev/kfd is opened O_CLOEXEC, so an exec'd child drops the descriptor and passes
// whether or not the allocation was deferred.
//
// Needs to be its own executable: the KFD check must run before any HIP API in the
// process, which a test case sharing the Catch binary cannot guarantee.

#include <hip/hip_runtime.h>

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>

#include <dirent.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

__managed__ int g_managedVal = 7;

// Reaches the variable through the symbol rather than through a kernel argument, so the
// launch depends on the device-side pointer slot that deferred initialization fills.
__global__ void writeManaged(int value) { g_managedVal = value; }

namespace {

constexpr int kSentinel = 99;
constexpr char kKfdDevice[] = "/dev/kfd";
constexpr char kProcSelfFd[] = "/proc/self/fd";
constexpr char kDeferredLoadingEnv[] = "HIP_ENABLE_DEFERRED_LOADING";

// Walking /proc/self/fd is the only in-process evidence that the runtime has reached the
// driver: libhsakmt holds the connection as a descriptor and exposes no query for it.
// Returns nullopt when the directory cannot be walked, so an unreadable /proc is not
// reported as an absent connection.
std::optional<bool> isKfdOpen() {
  const std::unique_ptr<DIR, int (*)(DIR*)> dir(opendir(kProcSelfFd), &closedir);
  if (dir == nullptr) {
    return std::nullopt;
  }

  for (const dirent* entry = readdir(dir.get()); entry != nullptr; entry = readdir(dir.get())) {
    const std::string link = std::string(kProcSelfFd) + "/" + entry->d_name;
    // Read into a full-length buffer so a longer target cannot truncate into a match.
    char target[PATH_MAX];
    const ssize_t length = readlink(link.c_str(), target, sizeof(target) - 1);
    if (length < 0) {
      continue;  // "." and "..", and descriptors closed during the walk.
    }
    target[length] = '\0';
    if (std::strcmp(target, kKfdDevice) == 0) {
      return true;
    }
  }

  return false;
}

// Mirrors how `__hipRegisterManagedVar` reads the variable
// (clr/hipamd/src/hip_platform.cpp). The caller always sets it explicitly, so the
// runtime's default for an unset variable is deliberately not reproduced here.
std::optional<bool> deferredLoadingEnabled() {
  const char* value = std::getenv(kDeferredLoadingEnv);
  if (value == nullptr) {
    return std::nullopt;
  }
  return std::atoi(value) != 0;
}

// Returns the child's exit status, or -1 when it did not exit on its own.
int runManagedTouchInChild() {
  // Empty the shared stdout buffer first, so the child does not re-emit what the parent
  // has already queued.
  fflush(stdout);

  const pid_t pid = fork();
  if (pid < 0) {
    printf("fork failed: %s\n", std::strerror(errno));
    return -1;
  }

  if (pid == 0) {
    writeManaged<<<1, 1>>>(kSentinel);
    const hipError_t launch = hipGetLastError();
    const hipError_t sync = hipDeviceSynchronize();
    const int observed = g_managedVal;
    printf("child: launch=%s sync=%s g_managedVal=%d\n", hipGetErrorString(launch),
           hipGetErrorString(sync), observed);
    fflush(stdout);
    // Leaves through _exit. The sibling helper executables check HIP calls with a local
    // macro that returns on failure, which here would let the child fall through into the
    // parent's remaining work and run on as a second parent.
    const bool ok = (launch == hipSuccess && sync == hipSuccess && observed == kSentinel);
    _exit(ok ? 0 : 1);
  }

  int status = 0;
  while (waitpid(pid, &status, 0) < 0) {
    if (errno != EINTR) {
      printf("waitpid failed: %s\n", std::strerror(errno));
      return -1;
    }
  }
  if (WIFSIGNALED(status)) {
    printf("child terminated by signal %d\n", WTERMSIG(status));
    return -1;
  }
  if (!WIFEXITED(status)) {
    printf("child did not exit normally (status 0x%x)\n", status);
    return -1;
  }
  return WEXITSTATUS(status);
}

}  // namespace

int main() {
  const std::optional<bool> deferred = deferredLoadingEnabled();
  if (!deferred.has_value()) {
    printf("%s must be set to 0 or 1\n", kDeferredLoadingEnv);
    return -1;
  }

  const std::optional<bool> kfdOpen = isKfdOpen();
  if (!kfdOpen.has_value()) {
    printf("cannot walk %s to determine whether the KFD is open\n", kProcSelfFd);
    return -1;
  }

  const bool expectKfdOpen = !*deferred;
  if (*kfdOpen != expectKfdOpen) {
    printf("KFD is %s before the first HIP API call, expected %s with %s=%d\n",
           *kfdOpen ? "open" : "not open", expectKfdOpen ? "open" : "not open", kDeferredLoadingEnv,
           *deferred ? 1 : 0);
    return -1;
  }

  if (!*deferred) {
    // Stop before the fork. Eager registration has already attached the runtime to the
    // KFD, so a child forked here is the defect the deferral avoids; forking anyway would
    // leave a crashed child and a GPU core dump behind on every run.
    printf("OK: eager registration opened the KFD before main\n");
    return 0;
  }

  const int result = runManagedTouchInChild();
  if (result == 0) {
    printf("OK: forked child initialized and wrote the managed variable\n");
  }
  return result;
}
