/*
Modifications Copyright © 2019 – 2020 Advanced Micro Devices, Inc. All Rights
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

#include "rocm_smi/rocm_smi_exception.h"

shared_mutex_t shared_mutex_init(const char *name, mode_t mode) {
  shared_mutex_t mutex = {NULL, 0, NULL, 0};
  errno = 0;

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
    NULL,
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
  expireTime.tv_sec += 5;

  int ret = pthread_mutex_timedlock(mutex_ptr, &expireTime);

  if (ret || (mutex.created == 0 &&
                     reinterpret_cast<shared_mutex_t *>(addr)->ptr == NULL)) {
    // Something is out of sync.
    fprintf(stderr, "pthread_mutex_timedlock() returned %d\n", ret);
    perror("Failed to initialize RSMI device mutex after 5 seconds. Previous "
     "execution may not have shutdown cleanly. To fix problem, stop all "
     "rocm_smi programs, and then delete the rocm_smi* shared memory files in"
                                                                " /dev/shm.");
    free(mutex.name);

    throw amd::smi::rsmi_exception(RSMI_STATUS_BUSY, __FUNCTION__);
    return mutex;
  } else {
    if (pthread_mutex_unlock(mutex_ptr)) {
      perror("pthread_mutex_unlock");
    }
  }

  if (mutex.created) {
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
    if (pthread_mutex_init(mutex_ptr, &attr)) {
      perror("pthread_mutex_init");
      return mutex;
    }
  }

  mutex.ptr = mutex_ptr;
  mutex.name = reinterpret_cast<char *>(malloc(NAME_MAX+1));
  (void)snprintf(mutex.name, NAME_MAX + 1, "%s", name);
  return mutex;
}

int shared_mutex_close(shared_mutex_t mutex) {
  if (munmap(reinterpret_cast<void *>(mutex.ptr), sizeof(pthread_mutex_t))) {
    perror("munmap");
    return -1;
  }
  mutex.ptr = NULL;
  if (close(mutex.shm_fd)) {
    perror("close");
    return -1;
  }
  mutex.shm_fd = 0;
  free(mutex.name);

  return 0;
}

int shared_mutex_destroy(shared_mutex_t mutex) {
  if ((errno = pthread_mutex_destroy(mutex.ptr))) {
    perror("pthread_mutex_destroy");
    return -1;
  }
  if (munmap(reinterpret_cast<void *>(mutex.ptr), sizeof(pthread_mutex_t))) {
    perror("munmap");
    return -1;
  }
  mutex.ptr = NULL;
  if (close(mutex.shm_fd)) {
    perror("close");
    return -1;
  }
  mutex.shm_fd = 0;
  if (shm_unlink(mutex.name)) {
    perror("shm_unlink");
    return -1;
  }
  free(mutex.name);
  return 0;
}
