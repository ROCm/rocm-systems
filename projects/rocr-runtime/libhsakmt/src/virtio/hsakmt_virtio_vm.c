/*
 * Copyright 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include <sys/mman.h>
#include <unistd.h>

#include "hsakmt_virtio_device.h"

void* vhsakmt_vm_start(void) {
  void* vm_start = malloc(getpagesize());
  if (!vm_start) return NULL;

  free(vm_start);
  return vm_start;
}

int vhsakmt_reserve_va(uint64_t start, uint64_t size) {
  int32_t protFlags = PROT_NONE;
  int32_t mapFlags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED;
  void* va = mmap((void*)start, size, protFlags, mapFlags, -1, 0);
  if (va == MAP_FAILED) return -ENOMEM;

  if (va != (void*)start) return -ENOMEM;

  madvise(va, size, MADV_DONTFORK);

  return 0;
}

void vhsakmt_dereserve_va(uint64_t start, uint64_t size) { munmap((void*)start, size); }

void vhsakmt_set_scratch_area(vhsakmt_device_handle dev, uint32_t node, uint64_t start,
                              uint64_t size) {
  if (!dev->vhsakmt_nodes || !dev->sys_props) return;
  if (node >= dev->sys_props->NumNodes) return;

  pthread_mutex_lock(&dev->vhsakmt_mutex);

  if (dev->vhsakmt_nodes[node].scratch_start && dev->vhsakmt_nodes[node].scratch_size) goto out;

  dev->vhsakmt_nodes[node].scratch_start = start;
  dev->vhsakmt_nodes[node].scratch_size = size;

out:
  pthread_mutex_unlock(&dev->vhsakmt_mutex);
}

bool vhsakmt_is_scratch_mem(vhsakmt_device_handle dev, void* addr) {
  uint32_t i;
  if (!dev->vhsakmt_nodes || !dev->sys_props) return false;

  for (i = 0; i < dev->sys_props->NumNodes; i++) {
    if ((uint64_t)addr >= dev->vhsakmt_nodes[i].scratch_start &&
        (uint64_t)addr <= dev->vhsakmt_nodes[i].scratch_start + dev->vhsakmt_nodes[i].scratch_size)
      return true;
  }

  return false;
}

void vhsakmt_set_vm_area(vhsakmt_device_handle dev, uint64_t start, uint64_t size) {
  pthread_mutex_lock(&dev->vhsakmt_mutex);
  if (dev->vm_start && dev->vm_size) goto out;

  dev->vm_start = start;
  dev->vm_size = size;

out:
  pthread_mutex_unlock(&dev->vhsakmt_mutex);
}

bool vhsakmt_is_userptr(vhsakmt_device_handle dev, void* addr) {
  return !((uint64_t)addr >= dev->vm_start && (uint64_t)addr <= dev->vm_start + dev->vm_size);
}

int vhsakmt_set_node_doorbell(vhsakmt_device_handle dev, uint32_t node, void* doorbell) {
  if (!dev->vhsakmt_nodes || !dev->sys_props) return -EINVAL;
  if (node >= dev->sys_props->NumNodes) return -EINVAL;

  pthread_mutex_lock(&dev->vhsakmt_mutex);

  dev->vhsakmt_nodes[node].doorbell_base = doorbell;

  pthread_mutex_unlock(&dev->vhsakmt_mutex);

  return 0;
}

void* vhsakmt_node_doorbell(vhsakmt_device_handle dev, uint32_t node) {
  if (!dev->vhsakmt_nodes || !dev->sys_props) return NULL;
  if (node >= dev->sys_props->NumNodes) return NULL;

  return dev->vhsakmt_nodes[node].doorbell_base;
}
