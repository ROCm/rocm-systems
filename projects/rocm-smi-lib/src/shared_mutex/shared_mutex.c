#include "shared_mutex.h"
#include <errno.h> // errno, ENOENT
#include <fcntl.h> // O_RDWR, O_CREATE
#include <linux/limits.h> // NAME_MAX
#include <sys/mman.h> // shm_open, shm_unlink, mmap, munmap,
                      // PROT_READ, PROT_WRITE, MAP_SHARED, MAP_FAILED
#include <unistd.h> // ftruncate, close
#include <stdio.h> // perror
#include <stdlib.h> // malloc, free
#include <string.h> // strcpy

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
    // Change permissions of shared memory, so every body can access it. Avoiding the umask of shm_open
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
    0
  );
  if (addr == MAP_FAILED) {
    perror("mmap");
    return mutex;
  }

  if (mutex.created == 0 && ((shared_mutex_t *)addr)->ptr == NULL) {
    // Something is out of sync. Unlink shm and start over.
    if (shm_unlink(name)) {
      mutex.shm_fd = 0;
      perror("shm_unlink");
    }
    free(mutex.name);

    return shared_mutex_init(name, mode);
  }

  pthread_mutex_t *mutex_ptr =  (pthread_mutex_t *)addr;

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
  mutex.name = (char *)malloc(NAME_MAX+1);
  strcpy(mutex.name, name);
  return mutex;
}

int shared_mutex_close(shared_mutex_t mutex) {  
  if (munmap((void *)mutex.ptr, sizeof(pthread_mutex_t))) {
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
  if (munmap((void *)mutex.ptr, sizeof(pthread_mutex_t))) {
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
