// NOLINT(legal/copyright)
// See LICENSE file

#ifndef SRC_SHARED_MUTEX_SHARED_MUTEX_H_
#define SRC_SHARED_MUTEX_SHARED_MUTEX_H_

#include <sys/stat.h>

#include <pthread.h>  // pthread_mutex_t, pthread_mutexattr_t,
                      // pthread_mutexattr_init, pthread_mutexattr_setpshared,
                      // pthread_mutex_init, pthread_mutex_destroy

// Structure of a shared mutex.
typedef struct shared_mutex_t {
  pthread_mutex_t *ptr;  // Pointer to the pthread mutex and
                         // shared memory segment.
  int shm_fd;            // Descriptor of shared memory object.
  char* name;            // Name of the mutex and associated
                         // shared memory object.
  int created;           // Equals 1 (true) if initialization
                         // of this structure caused creation
                         // of a new shared mutex.
                         // Equals 0 (false) if this mutex was
                         // just retrieved from shared memory.
} shared_mutex_t;

// Initialize a new shared mutex with given `name`. If a mutex
// with such name exists in the system, it will be loaded.
// Otherwise a new mutes will by created.
//
// In case of any error, it will be printed into the standard output
// and the returned structure will have `ptr` equal `NULL`.
// `errno` wil not be reset in such case, so you may used it.
//
// **NOTE:** In case when the mutex appears to be uncreated,
// this function becomes *non-thread-safe*. If multiple threads
// call it at one moment, there occur several race conditions,
// in which one call might recreate another's shared memory
// object or rewrite another's pthread mutex in the shared memory.
// There is no workaround currently, except to run first
// initialization only before multi-threaded or multi-process
// functionality.
shared_mutex_t shared_mutex_init(const char *name, mode_t mode);

// Close access to the shared mutex and free all the resources,
// used by the structure.
//
// Returns 0 in case of success. If any error occurs, it will be
// printed into the standard output and the function will return -1.
// `errno` wil not be reset in such case, so you may used it.
//
// **NOTE:** It will not destroy the mutex. The mutex would not
// only be available to other processes using it right now,
// but also to any process which might want to use it later on.
// For complete desctruction use `shared_mutex_destroy` instead.
//
// **NOTE:** It will not unlock locked mutex.
int shared_mutex_close(shared_mutex_t mutex);

// Close and destroy shared mutex.
// Any open pointers to it will be invalidated.
//
// Returns 0 in case of success. If any error occurs, it will be
// printed into the standard output and the function will return -1.
// `errno` wil not be reset in such case, so you may used it.
//
// **NOTE:** It will not unlock locked mutex.
int shared_mutex_destroy(shared_mutex_t mutex);

#endif  // SRC_SHARED_MUTEX_SHARED_MUTEX_H_
