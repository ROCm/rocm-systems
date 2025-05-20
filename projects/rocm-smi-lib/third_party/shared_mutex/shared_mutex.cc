/*
Modifications Copyright 2019 - 2025 Advanced Micro Devices, Inc. All Rights
Reserved.
Copyright (c) 2018 Oleg Yamnikov

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include "shared_mutex.h"  // NOLINT(build/include)
#include <errno.h>  // errno, ENOENT
#include <fcntl.h>  // O_RDWR, O_CREATE
#include <linux/limits.h>  // NAME_MAX
#include <sys/mman.h>  // shm_open, shm_unlink, mmap, munmap,
                      // PROT_READ, PROT_WRITE, MAP_SHARED, MAP_FAILED
#include <unistd.h>  // ftruncate, close
#include <stdio.h>  // perror
#include <stdlib.h>  // malloc, free
#include <string.h>  // strcpy
#include <time.h>   // clock_gettime
#include <assert.h>

#include <sys/types.h>
#include <dirent.h>
#include <algorithm>
#include <string>
#include <vector>

#include "rocm_smi/rocm_smi_exception.h"
#include "rocm_smi/rocm_smi_main.h"

#define THREAD_ONLY_ENV_VAR "RSMI_MUTEX_THREAD_ONLY"
#define MUTEX_TIME_OUT_ENV_VAR "RSMI_MUTEX_TIMEOUT"
#define DEFAULT_MUTEX_TIMEOUT_SECONDS 5

static int GetEnvVarUInteger(const char *ev_str) {
  ev_str = getenv(ev_str);

  if (ev_str) {
    const int ret = atoi(ev_str);
    return ret;
  }

  return -1;
}

// find which processes are using the file by searching /proc/*/fd
static std::vector<std::string> lsof(const char* filename) {
  struct dirent *entry = nullptr;
  DIR *dp = nullptr;
  std::vector<std::string> process_id;

  pid_t cur_pid = getpid();
  dp = opendir("/proc");
  if (dp != nullptr) {
    while ((entry = readdir(dp))) {
      std::string id(entry->d_name);
      // ignore current process
      if (id == std::to_string(cur_pid)) continue;
      // the process id should be a number
      if (std::all_of(id.begin(), id.end(), ::isdigit)) {
        process_id.push_back(entry->d_name);
      }
    }
    closedir(dp);
  }

  std::vector<std::string> matched_process;
  for (unsigned int i=0; i < process_id.size(); i++) {
      std::string folder_name("/proc/");
      folder_name += process_id[i]+"/fd/";
      dp = opendir(folder_name.c_str());
      if (dp == nullptr) continue;
      while ((entry = readdir(dp))) {
         std::string p(folder_name+entry->d_name);
         char buf[512];
         memset(buf, 0, 512);
         if (readlink(p.c_str(), buf, sizeof(buf)-1) < 0) continue;
         if (!strcmp(filename, buf)) matched_process.push_back(process_id[i]);
      }
      closedir(dp);
  }
  return matched_process;
}

// RSMI_MUTEX_THREAD_ONLY = 1 to enable thread safe mutex
shared_mutex_t init_thread_safe_only(const char *name) {
  shared_mutex_t mutex = {nullptr, 0, nullptr, 0};
  errno = 0;
  mutex.shm_fd = -1;
  mutex.created = 0;
  pthread_mutex_t *mutex_ptr =  new pthread_mutex_t();

  pthread_mutexattr_t attr;
  if (pthread_mutexattr_init(&attr)) {
    perror("pthread_mutexattr_init");
    return mutex;
  }
  if (pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED)) {
    perror("pthread_mutexattr_setpshared");
    return mutex;
  }

  if (pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE)) {
    perror("pthread_mutexattr_settype");
    return mutex;
  }
  if (pthread_mutexattr_setrobust(&attr, PTHREAD_MUTEX_ROBUST)) {
    perror("pthread_mutexattr_setrobust");
    return mutex;
  }
  if (pthread_mutex_init(mutex_ptr, &attr)) {
    perror("pthread_mutex_init");
    return mutex;
  }

  mutex.ptr = mutex_ptr;
  mutex.name = reinterpret_cast<char *>(malloc(NAME_MAX+1));
  (void)snprintf(mutex.name, NAME_MAX + 1, "%s", name);
  return mutex;
}

shared_mutex_t shared_mutex_init(const char *name, mode_t mode, bool retried) {
  shared_mutex_t mutex = {nullptr, 0, nullptr, 0};
  errno = 0;

  amd::smi::RocmSMI& smi = amd::smi::RocmSMI::getInstance();

  if (GetEnvVarUInteger(THREAD_ONLY_ENV_VAR) == 1 || smi.is_thread_only_mutex()) {
    return init_thread_safe_only(name);
  }

  // Open existing shared memory object, or create one.
  // Two separate calls are needed here, to mark fact of creation
  // for later initialization of pthread mutex.
  mutex.shm_fd = shm_open(name, O_RDWR, mode);
  if (errno == ENOENT) {
    mutex.shm_fd = shm_open(name, O_RDWR|O_CREAT, mode);
    mutex.created = 1;
    // Change permissions of shared memory, so everybody can access it.
    // Avoiding the umask of shm_open
    if (fchmod(mutex.shm_fd, mode) != 0) {
      perror("fchmod");
    }
  }
  if (mutex.shm_fd == -1) {
    perror("shm_open");
    return mutex;
  }

  // Truncate shared memory segment so it would contain
  // pthread_mutex_t AND the ref. count
  if (ftruncate(mutex.shm_fd, sizeof(pthread_mutex_t)) != 0) {
    perror("ftruncate");
    return mutex;
  }

  // Map pthread mutex into the shared memory.
  void *addr = mmap(
    nullptr,
    sizeof(pthread_mutex_t),
    PROT_READ|PROT_WRITE,
    MAP_SHARED,
    mutex.shm_fd,
    0);

  if (addr == MAP_FAILED) {
    perror("mmap");
    return mutex;
  }

  pthread_mutex_t *mutex_ptr =  reinterpret_cast<pthread_mutex_t *>(addr);

  // Make sure the mutex wasn't left in a locked state. If we can't
  // acquire it in 5 sec., re-do everything.
  struct timespec expireTime;
  clock_gettime(CLOCK_REALTIME, &expireTime);
  int time_out = GetEnvVarUInteger(MUTEX_TIME_OUT_ENV_VAR);
  time_out = time_out < DEFAULT_MUTEX_TIMEOUT_SECONDS ? DEFAULT_MUTEX_TIMEOUT_SECONDS: time_out;
  expireTime.tv_sec += time_out;

  pid_t cur_pid = getpid();
  int ret;

  // also need to set the attribute when retried as the mutex is re-initialized.
  if (mutex.created || retried) {
    pthread_mutexattr_t attr;
    if (pthread_mutexattr_init(&attr)) {
      perror("pthread_mutexattr_init");
      return mutex;
    }
    if (pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED)) {
      perror("pthread_mutexattr_setpshared");
      return mutex;
    }

    if (pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE)) {
      perror("pthread_mutexattr_settype");
      return mutex;
    }
    if (pthread_mutexattr_setrobust(&attr, PTHREAD_MUTEX_ROBUST)) {
      perror("pthread_mutexattr_setrobust");
      return mutex;
    }
    if (pthread_mutex_init(mutex_ptr, &attr)) {
      perror("pthread_mutex_init");
      return mutex;
    }
  }

  ret = pthread_mutex_timedlock(mutex_ptr, &expireTime);


  if (ret == EOWNERDEAD) {
    ret = pthread_mutex_consistent(mutex_ptr);
    // This function should not fail unless mutex_ptr is not robust
    // or mutex_ptr is not in an inconsistent state. Neither scenario
    // should ever be true at this point in the code.
    assert(!ret);

    // ...but if there are undocumented failure cases for
    // pthread_mutex_consistent() handle them for release builds.
    if (ret) {
      fprintf(stderr, "pthread_mutex_consistent() returned %d\n", ret);
      free(mutex.name);

      throw amd::smi::rsmi_exception(RSMI_STATUS_BUSY, __FUNCTION__);
      return mutex;
    }

    fprintf(stderr, "%d detected dead process, and making mutex %s consistent.\n",
              cur_pid, name);
    // The mutex is locked even if EOWNERDEAD was returned,and need to unlock it.
    if (pthread_mutex_unlock(mutex_ptr)) {
      perror("pthread_mutex_unlock");
    }
  } else if (ret || (mutex.created == 0 &&
                     reinterpret_cast<shared_mutex_t *>(addr)->ptr == nullptr)) {
    // Something is out of sync.

    // When process crash before unlock the mutex, the mutex is in bad status.
    // reset the mutex if no process is using it, and then retry lock
    if (!retried) {
      std::vector<std::string> ids = lsof(name);
      if (ids.size() == 0) {  // no process is using it
        fprintf(stderr, "%d re-init the mutex %s since no one use it. ret:%d ptr:%p\n",
              cur_pid, name, ret, reinterpret_cast<shared_mutex_t *>(addr)->ptr);
        memset(mutex_ptr, 0, sizeof(pthread_mutex_t));
        // Set mutex.created == 1 so that it can be initialized latter.
        mutex.created = 1;
        free(mutex.name);
        return shared_mutex_init(name, mode, true);
      }
    }

    fprintf(stderr, "pthread_mutex_timedlock() returned %d\n", ret);
    perror("Failed to initialize RSMI device mutex after 5 seconds. Previous "
     "execution may not have shutdown cleanly. To fix problem, stop all "
     "rocm_smi programs, and then delete the rocm_smi* shared memory files in"
                                                                " /dev/shm.");
    free(mutex.name);

    throw amd::smi::rsmi_exception(RSMI_STATUS_BUSY, __FUNCTION__);
    return mutex;
  } else {
    const int ret = pthread_mutex_unlock(mutex_ptr);
    if (ret) {
      perror("pthread_mutex_unlock");
      fprintf(stderr, "%d init_mutex %s: unlock timed lock, ret: %d\n", cur_pid, name, ret);
    }
  }



  mutex.ptr = mutex_ptr;
  mutex.name = reinterpret_cast<char *>(malloc(NAME_MAX+1));
  (void)snprintf(mutex.name, NAME_MAX + 1, "%s", name);
  return mutex;
}

int shared_mutex_close(shared_mutex_t mutex) {
  amd::smi::RocmSMI& smi = amd::smi::RocmSMI::getInstance();
  const bool is_thread_only = GetEnvVarUInteger(THREAD_ONLY_ENV_VAR) == 1 ||
          smi.is_thread_only_mutex();
  if (is_thread_only) {
    delete mutex.ptr;
  } else if (munmap(reinterpret_cast<void *>(mutex.ptr), sizeof(pthread_mutex_t))) {
    perror("munmap");
    return -1;
  }
  mutex.ptr = nullptr;
  if (!is_thread_only && close(mutex.shm_fd)) {
    perror("close");
    return -1;
  }
  mutex.shm_fd = 0;
  free(mutex.name);

  return 0;
}

int shared_mutex_destroy(shared_mutex_t mutex) {
  amd::smi::RocmSMI& smi = amd::smi::RocmSMI::getInstance();
  const bool is_thread_only = GetEnvVarUInteger(THREAD_ONLY_ENV_VAR) == 1 ||
          smi.is_thread_only_mutex();
  if ((errno = pthread_mutex_destroy(mutex.ptr))) {
    perror("pthread_mutex_destroy");
    return -1;
  }
  if (is_thread_only) {
    delete mutex.ptr;
  } else if (munmap(reinterpret_cast<void *>(mutex.ptr), sizeof(pthread_mutex_t))) {
    perror("munmap");
    return -1;
  }
  mutex.ptr = nullptr;
  if (!is_thread_only && close(mutex.shm_fd)) {
    perror("close");
    return -1;
  }
  mutex.shm_fd = 0;
  if (!is_thread_only && shm_unlink(mutex.name)) {
    perror("shm_unlink");
    return -1;
  }
  free(mutex.name);
  return 0;
}
