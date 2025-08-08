#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstdlib>
#include <sstream>
#include <sstream>
#include <cstring>
#include "profiling_lock.h"
#include <stdexcept>

#define ROCPROFILER_LOCK_FILE "/tmp/rocprofiler_process.lock"
#define ROCPROFILER_PID_FILE "/tmp/rocprofiler.pid"

int acquire_lock(char const* lockName) {
  // umask to set permissions on file creation.
  // base permissions (rw) given
  mode_t m = umask(0);
  int fd = open(lockName, O_RDWR | O_CREAT, 0666);
  umask(m);
  if (fd >= 0 && flock(fd, LOCK_EX | LOCK_NB) < 0) {
    close(fd);
    fd = -1;
  }
  return fd;
}

void release_lock(int fd, char const* lockName) {
  if (fd < 0) return;
  remove(lockName);
  close(fd);
}

bool file_exists(const char* file_name) {
  struct stat buffer;
  return stat(file_name, &buffer) == 0;
}

int create_pid_file(const char* pid_file) {
  FILE* file = fopen(pid_file, "w");
  if (!file) return -1;
  fprintf(file, "%d", (int)getpid());
  fclose(file);
  return 0;
}

int read_pid_file(const char* pid_file) {
  FILE* file = fopen(pid_file, "r");
  if (!file) return -1;
  int pid_value = -1;
  fscanf(file, "%d", &pid_value);
  fclose(file);
  return pid_value > 0 ? pid_value : -1;
}

bool check_process_exists(int pid) {
  struct stat sts;
  std::stringstream ss;
  ss << "/proc/" << pid;
  if (stat(ss.str().c_str(), &sts) == -1) {
    return false;
  }
  return true;
}

void terminate_current_profiler_instance() {
  std::stringstream oss;
  oss << "\nA profiling instance already exists! Multiple profiling instances are not "
      << "allowed.\nCheck " << ROCPROFILER_PID_FILE
      << " and kill the process, delete this .pid file and try again.\nTerminating "
         "...\n";
  throw std::runtime_error(oss.str()); 
}


bool check_standalone_mode() {
  static bool is_standalone_mode = [] {
    // Checking environment variable to see if interception is enabled.
    // value of zero indicates standalone mode
    const char* intercept_env = getenv("ROCP_HSA_INTERCEPT");
    int intercept_env_value = 0;
    if (intercept_env != NULL) {
      intercept_env_value = atoi(intercept_env);
    }
    return intercept_env_value == 0;
  }();
  return is_standalone_mode;
}

void ProfilingLock::Lock(LockMode mode) {
  // check if the profiler v1 is running in standalone mode
  bool is_standalone_mode_v1 = check_standalone_mode() && (mode == PROFILER_V1_LOCK);

  ProfilingLock* profiling_lock = Instance();
  // Check if we have already locked in this process
  if (profiling_lock->already_locked.exchange(true)) return;
  if (file_exists(profiling_lock->pid_file)) {
    profiling_lock->lock = acquire_lock(profiling_lock->lock_file);
    if (profiling_lock->lock < 1) {
      release_lock(profiling_lock->lock, profiling_lock->lock_file);
      terminate_current_profiler_instance();
    }
    int pid = read_pid_file(profiling_lock->pid_file);
    if (check_process_exists(pid)) {
      release_lock(profiling_lock->lock, profiling_lock->lock_file);
      terminate_current_profiler_instance();
    }
    if (is_standalone_mode_v1) create_pid_file(profiling_lock->pid_file);
    release_lock(profiling_lock->lock, profiling_lock->lock_file);
  } else {
    profiling_lock->lock = acquire_lock(profiling_lock->lock_file);
    if (profiling_lock->lock < 1) terminate_current_profiler_instance();
    if (is_standalone_mode_v1) create_pid_file(profiling_lock->pid_file);
    release_lock(profiling_lock->lock, profiling_lock->lock_file);
  }
  return;
}

ProfilingLock::ProfilingLock() {
  lock_file = ROCPROFILER_LOCK_FILE;
  pid_file = ROCPROFILER_PID_FILE;
  lock = -1;
}

ProfilingLock::~ProfilingLock(){
  this->lock = acquire_lock(this->lock_file);
  if (this->lock < 1) return; // lock couldn't be acquired
  remove(this->pid_file); // remove the pid file
  release_lock(this->lock, this->lock_file);
  return;
}

ProfilingLock* ProfilingLock::Instance() {
  static ProfilingLock instance;
  return &instance;
}
