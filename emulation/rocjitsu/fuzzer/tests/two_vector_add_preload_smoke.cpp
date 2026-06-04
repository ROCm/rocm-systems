#include "rocjitsu_fuzzer/afl_runtime.h"

#include <hip/hip_runtime_api.h>
#include <sys/shm.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

using rocjitsu::fuzzer::afl::kDeviceStart;
using rocjitsu::fuzzer::afl::kEntryCounterSlot;
using rocjitsu::fuzzer::afl::kMapSize;

bool have_hip_device() {
  int device_count = 0;
  return hipGetDeviceCount(&device_count) == hipSuccess && device_count > 0;
}

void set_env_or_die(const char *name, const std::string &value) {
  if (setenv(name, value.c_str(), 1) != 0) {
    std::fprintf(stderr, "setenv(%s) failed: %s\n", name, std::strerror(errno));
    std::_Exit(127);
  }
}

void prepend_env_or_die(const char *name, const std::string &prefix) {
  const char *existing = std::getenv(name);
  if (existing == nullptr || existing[0] == '\0') {
    set_env_or_die(name, prefix);
    return;
  }
  set_env_or_die(name, prefix + ":" + existing);
}

class SharedAflMap {
public:
  SharedAflMap() {
    id_ = shmget(IPC_PRIVATE, kMapSize, IPC_CREAT | 0600);
    if (id_ < 0) {
      std::fprintf(stderr, "shmget failed: %s\n", std::strerror(errno));
      return;
    }

    void *attached = shmat(id_, nullptr, 0);
    if (attached == reinterpret_cast<void *>(-1)) {
      std::fprintf(stderr, "shmat failed: %s\n", std::strerror(errno));
      shmctl(id_, IPC_RMID, nullptr);
      id_ = -1;
      return;
    }

    data_ = static_cast<uint8_t *>(attached);
    std::memset(data_, 0, kMapSize);
  }

  SharedAflMap(const SharedAflMap &) = delete;
  SharedAflMap &operator=(const SharedAflMap &) = delete;

  ~SharedAflMap() {
    if (data_ != nullptr)
      shmdt(data_);
    if (id_ >= 0)
      shmctl(id_, IPC_RMID, nullptr);
  }

  bool valid() const { return id_ >= 0 && data_ != nullptr; }
  int id() const { return id_; }
  const uint8_t *data() const { return data_; }

private:
  int id_ = -1;
  uint8_t *data_ = nullptr;
};

[[noreturn]] void exec_target(const char *target, const char *seed, const char *preload,
                              const char *library_path, int shm_id) {
  set_env_or_die("__AFL_SHM_ID", std::to_string(shm_id));
  set_env_or_die("AFL_MAP_SIZE", std::to_string(kMapSize));
  set_env_or_die("AFL_QUIET", "1");
  set_env_or_die("ROCJITSU_AFL_REQUIRE_PERSISTENT_HOOKS", "1");
  prepend_env_or_die("LD_PRELOAD", preload);
  prepend_env_or_die("LD_LIBRARY_PATH", library_path);

  char *const argv[] = {const_cast<char *>(target), const_cast<char *>(seed), nullptr};
  execv(target, argv);
  std::fprintf(stderr, "execv(%s) failed: %s\n", target, std::strerror(errno));
  std::_Exit(127);
}

bool child_succeeded(int status) {
  if (WIFEXITED(status) && WEXITSTATUS(status) == 0)
    return true;

  if (WIFEXITED(status)) {
    std::fprintf(stderr, "two-vector-add preload smoke exited with %d\n", WEXITSTATUS(status));
  } else if (WIFSIGNALED(status)) {
    std::fprintf(stderr, "two-vector-add preload smoke died on signal %d\n", WTERMSIG(status));
  } else {
    std::fprintf(stderr, "two-vector-add preload smoke ended unexpectedly\n");
  }
  return false;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: %s <target> <preload> <seed> <ld-library-path>\n", argv[0]);
    return 2;
  }

  if (!have_hip_device())
    return 0;

  SharedAflMap shm;
  if (!shm.valid())
    return 1;

  const pid_t pid = fork();
  if (pid < 0) {
    std::fprintf(stderr, "fork failed: %s\n", std::strerror(errno));
    return 1;
  }

  if (pid == 0)
    exec_target(argv[1], argv[3], argv[2], argv[4], shm.id());

  int status = 0;
  if (waitpid(pid, &status, 0) < 0) {
    std::fprintf(stderr, "waitpid failed: %s\n", std::strerror(errno));
    return 1;
  }
  if (!child_succeeded(status))
    return 1;

  const uint8_t entry_counter = shm.data()[kDeviceStart + kEntryCounterSlot];
  if (entry_counter == 0) {
    std::fprintf(stderr, "preload smoke did not observe device-half AFL feedback\n");
    return 1;
  }

  return 0;
}
