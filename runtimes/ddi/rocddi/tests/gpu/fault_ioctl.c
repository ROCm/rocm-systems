// SPDX-License-Identifier: MIT

#define _GNU_SOURCE
#include "fault_ioctl.h"
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/syscall.h>
#include <uapi/linux/kfd_ioctl.h>
#include <unistd.h>

#define CREATE_QUEUE AMDKFD_IOC_CREATE_QUEUE
#define DESTROY_QUEUE AMDKFD_IOC_DESTROY_QUEUE
#define SET_CU_MASK AMDKFD_IOC_SET_CU_MASK
#define ALLOC_MEMORY AMDKFD_IOC_ALLOC_MEMORY_OF_GPU
#define FREE_MEMORY AMDKFD_IOC_FREE_MEMORY_OF_GPU
#define MAX_ALLOCATIONS 32

struct aql_control {
  unsigned char before_read[128];
  uint64_t read_index;
  uint32_t read_base_offset, tmpring_size;
  uint32_t scratch_resource[4];
  uint64_t scratch_address, scratch_size;
  uint32_t wave_bytes, properties;
  uint64_t scratch_max_use, inactive_signal;
};
_Static_assert(offsetof(struct aql_control, scratch_address) == 160,
               "AQL scratch layout");
_Static_assert(offsetof(struct aql_control, inactive_signal) == 192,
               "AQL signal layout");

struct allocation {
  uint64_t va, size, handle;
  _Atomic int freed;
};
static struct allocation recorded[MAX_ALLOCATIONS];
static unsigned recorded_count;
static pthread_mutex_t allocation_lock = PTHREAD_MUTEX_INITIALIZER;
static _Atomic int mode;
static _Atomic unsigned creates, create_faults, mask_faults, destroy_faults;
static _Atomic unsigned allocations, frees;
static _Atomic uint64_t signal_address, scratch_address;

void rocddi_fault_arm(int selected) {
  atomic_store(&mode, 0);
  atomic_store(&creates, 0);
  atomic_store(&create_faults, 0);
  atomic_store(&mask_faults, 0);
  atomic_store(&destroy_faults, 0);
  atomic_store(&allocations, 0);
  atomic_store(&frees, 0);
  pthread_mutex_lock(&allocation_lock);
  recorded_count = 0;
  pthread_mutex_unlock(&allocation_lock);
  atomic_store(&signal_address, 0);
  atomic_store(&scratch_address, 0);
  atomic_store(&mode, selected);
}

static void observe_address(uint64_t address, unsigned *seen, unsigned *freed,
                            uint64_t *handle) {
  if (!address)
    return;
  for (unsigned i = 0; i < recorded_count; ++i) {
    if (address >= recorded[i].va &&
        address - recorded[i].va < recorded[i].size) {
      *seen = 1;
      *freed = atomic_load(&recorded[i].freed) != 0;
      *handle = recorded[i].handle;
      return;
    }
  }
}
void rocddi_fault_stats(struct fault_stats *out) {
  out->creates = atomic_load(&creates);
  out->create_faults = atomic_load(&create_faults);
  out->mask_faults = atomic_load(&mask_faults);
  out->destroy_faults = atomic_load(&destroy_faults);
  out->allocations = atomic_load(&allocations);
  out->frees = atomic_load(&frees);
  out->signal_seen = out->signal_freed = out->scratch_seen =
      out->scratch_freed = 0;
  out->signal_handle = out->scratch_handle = 0;
  pthread_mutex_lock(&allocation_lock);
  observe_address(atomic_load(&signal_address), &out->signal_seen,
                  &out->signal_freed, &out->signal_handle);
  observe_address(atomic_load(&scratch_address), &out->scratch_seen,
                  &out->scratch_freed, &out->scratch_handle);
  pthread_mutex_unlock(&allocation_lock);
}

/* These opt-in Linux test processes call ioctl with a third argument. */
int ioctl(int fd, unsigned long request, void *arg) {
  int selected = atomic_load(&mode);
  if (selected && request == DESTROY_QUEUE &&
      selected != ROCDDI_FAULT_CREATE_EFAULT) {
    atomic_fetch_add(&destroy_faults, 1);
    errno = EIO;
    return -1;
  }
  if (selected == ROCDDI_FAULT_CU_MASK_ROLLBACK && request == SET_CU_MASK) {
    atomic_fetch_add(&mask_faults, 1);
    errno = EIO;
    return -1;
  }
  int result = (int)syscall(SYS_ioctl, fd, request, arg);
  if (!selected)
    return result;
  if (request == ALLOC_MEMORY && result == 0) {
    struct kfd_ioctl_alloc_memory_of_gpu_args *allocation = arg;
    atomic_fetch_add(&allocations, 1);
    pthread_mutex_lock(&allocation_lock);
    if (recorded_count < MAX_ALLOCATIONS) {
      struct allocation *entry = &recorded[recorded_count++];
      entry->va = allocation->va_addr;
      entry->size = allocation->size;
      entry->handle = allocation->handle;
      atomic_store(&entry->freed, 0);
    }
    pthread_mutex_unlock(&allocation_lock);
  }
  if (request == FREE_MEMORY && result == 0) {
    uint64_t handle = *(uint64_t *)arg;
    atomic_fetch_add(&frees, 1);
    pthread_mutex_lock(&allocation_lock);
    for (unsigned i = 0; i < recorded_count; ++i)
      if (recorded[i].handle == handle)
        atomic_store(&recorded[i].freed, 1);
    pthread_mutex_unlock(&allocation_lock);
  }
  if (request == CREATE_QUEUE && result == 0) {
    struct kfd_ioctl_create_queue_args *queue = arg;
    struct aql_control control;
    atomic_fetch_add(&creates, 1);
    if (queue->queue_type == KFD_IOC_QUEUE_TYPE_COMPUTE_AQL &&
        queue->read_pointer_address >= 128) {
      memcpy(&control, (void *)(uintptr_t)(queue->read_pointer_address - 128),
             sizeof(control));
      atomic_store(&signal_address, control.inactive_signal);
      atomic_store(&scratch_address, control.scratch_address);
    }
    if (selected == ROCDDI_FAULT_CREATE_EFAULT) {
      atomic_fetch_add(&create_faults, 1);
      errno = EFAULT;
      return -1;
    }
    if (selected == ROCDDI_FAULT_DOORBELL_ROLLBACK)
      queue->doorbell_offset = UINT64_MAX;
  }
  return result;
}
