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

#include "hsakmt/hsakmt_virtio.h"
#include "hsakmt_virtio_device.h"

pthread_mutex_t dev_mutex = PTHREAD_MUTEX_INITIALIZER;
vhsakmt_device_handle dev_list = NULL;

vhsakmt_device_handle vhsakmt_dev(void) { return dev_list; }

static HSAKMT_STATUS vhsakmt_openKFD_cmd(vhsakmt_device_handle dev) {
  void* vm_start = vhsakmt_vm_start();
  if (!vm_start) return -HSAKMT_STATUS_NO_MEMORY;
  struct vhsakmt_ccmd_query_info_rsp* rsp;
  struct vhsakmt_ccmd_query_info_req req = {
      .hdr = VHSAKMT_CCMD(QUERY_INFO, sizeof(struct vhsakmt_ccmd_query_info_req)),
      .type = VHSAKMT_CCMD_QUERY_OPEN_KFD,
      .open_kfd_args =
          {
              .cur_vm_start = VHSA_VPTR_TO_UINT64(vm_start),
          },
  };

  if (!req.open_kfd_args.cur_vm_start) {
    vhsa_err("%s: failed to get current heap start address\n", __FUNCTION__);
    return -HSAKMT_STATUS_ERROR;
  }

  rsp = vhsakmt_alloc_rsp(dev, &req.hdr, sizeof(struct vhsakmt_ccmd_query_info_rsp));
  if (!rsp) return -HSAKMT_STATUS_NO_MEMORY;

  vhsakmt_execbuf_cpu(dev, &req.hdr, __FUNCTION__);
  if (!rsp->open_kfd_rsp.vm_start || !rsp->open_kfd_rsp.vm_size) {
    vhsa_err("%s: failed to get KFD VM area\n", __FUNCTION__);
    return -HSAKMT_STATUS_ERROR;
  }

  vhsakmt_set_vm_area(dev, rsp->open_kfd_rsp.vm_start, rsp->open_kfd_rsp.vm_size);
  if (vhsakmt_reserve_va(dev->vm_start, dev->vm_size)) {
    vhsa_err("%s: failed to reserve VM area: [%lx-%lx]-0x%lx\n", __FUNCTION__, dev->vm_start,
             dev->vm_start + dev->vm_size, dev->vm_size);
    return -HSAKMT_STATUS_NO_MEMORY;
  }

  vhsa_debug("%s: kfd vm range: [%lx-%lx]-0x%lx\n", __FUNCTION__, dev->vm_start,
             dev->vm_start + dev->vm_size, dev->vm_size);
  return rsp->ret;
}

static vhsakmt_device_handle vhsakmt_device_init(void) {
  int fd;
  vhsakmt_device_handle dev = NULL;

  if (vhsakmt_dev()) return vhsakmt_dev();

  pthread_mutex_lock(&dev_mutex);

  fd = virtio_gpu_kfd_open();
  if (fd < 0) goto open_failed;

  dev = calloc(1, sizeof(struct vhsakmt_device));
  if (!dev) goto open_failed;

  dev->vgdev = virtio_gpu_init(fd, 0);
  if (!dev->vgdev) goto malloc_failed;

  rbtree_init(&dev->bo_rbt);
  atomic_store(&dev->next_blob_id, 1);
  atomic_store(&dev->refcount, 1);
  pthread_mutex_init(&dev->bo_handles_mutex, NULL);
  pthread_mutex_init(&dev->vhsakmt_mutex, NULL);
  dev_list = dev;

  pthread_mutex_unlock(&dev_mutex);
  return dev;

malloc_failed:
  free(dev);
  dev = NULL;
open_failed:
  pthread_mutex_unlock(&dev_mutex);
  return dev;
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtOpenKFD(void) {
  vhsakmt_device_handle dev;
  char* d = getenv("VHSAKMT_DEBUG_LEVEL");
  if (d) vhsakmt_debug_level = atoi(d);

  dev = vhsakmt_device_init();
  if (!dev) return HSAKMT_STATUS_ERROR;

  return vhsakmt_openKFD_cmd(vhsakmt_dev());
}

static void vhsakmt_device_destroy(struct vhsakmt_device* dev) {
  pthread_mutex_destroy(&dev->bo_handles_mutex);
  vhsakmt_dereserve_va(dev->vm_start, dev->vm_size);

  if (dev->sys_props) free(dev->sys_props);
  if (dev->vhsakmt_nodes) free(dev->vhsakmt_nodes);

  virtio_gpu_close(dev->vgdev);
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtCloseKFD(void) {
  vhsakmt_device_handle dev = vhsakmt_dev();
  pthread_mutex_lock(&dev_mutex);
  if (vhsakmt_atomic_dec_return(&dev->refcount) <= 0) vhsakmt_device_destroy(dev);
  pthread_mutex_unlock(&dev_mutex);
  return 0;
}
