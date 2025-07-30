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

static int vhsakmt_set_sys_props(vhsakmt_device_handle dev, HsaSystemProperties* sys_props) {
  int r = 0;

  pthread_mutex_lock(&dev->vhsakmt_mutex);
  if (dev->sys_props) {
    r = 0;
    goto out;
  }

  dev->sys_props = calloc(1, sizeof(HsaSystemProperties));
  if (!dev->sys_props) {
    r = -ENOMEM;
    goto out;
  }

  memcpy(dev->sys_props, sys_props, sizeof(HsaSystemProperties));

out:
  pthread_mutex_unlock(&dev->vhsakmt_mutex);
  return r;
}

static int vhsakmt_set_node_props(vhsakmt_device_handle dev, uint32_t node,
                                  HsaNodeProperties* node_props) {
  int r = 0;
  if (!dev->sys_props) return -EINVAL;
  if (node >= dev->sys_props->NumNodes) return -EINVAL;

  pthread_mutex_lock(&dev->vhsakmt_mutex);

  if (!dev->vhsakmt_nodes) {
    dev->vhsakmt_nodes = calloc(dev->sys_props->NumNodes, sizeof(struct vhsakmt_node));
    if (!dev->vhsakmt_nodes) {
      r = -ENOMEM;
      goto out;
    }
  }

  memcpy(&dev->vhsakmt_nodes[node].node_props, node_props, sizeof(HsaNodeProperties));

out:
  pthread_mutex_unlock(&dev->vhsakmt_mutex);
  return r;
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtGetVersion(HsaVersionInfo* v) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle dev = vhsakmt_dev();
  struct vhsakmt_ccmd_query_info_rsp* rsp;
  struct vhsakmt_ccmd_query_info_req req = {
      .hdr = VHSAKMT_CCMD(QUERY_INFO, sizeof(struct vhsakmt_ccmd_query_info_req)),
      .type = VHSAKMT_CCMD_QUERY_GET_VER,
  };

  rsp = vhsakmt_alloc_rsp(dev, &req.hdr, sizeof(struct vhsakmt_ccmd_query_info_rsp));
  if (!rsp) return -ENOMEM;

  vhsakmt_execbuf_cpu(dev, &req.hdr, __FUNCTION__);
  memcpy(v, &rsp->kfd_version, sizeof(HsaVersionInfo));

  return rsp->ret;
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtAcquireSystemProperties(HsaSystemProperties* SystemProperties) {
  CHECK_VIRTIO_KFD_OPEN();

  int r;
  vhsakmt_device_handle dev = vhsakmt_dev();
  struct vhsakmt_ccmd_query_info_rsp* rsp;
  struct vhsakmt_ccmd_query_info_req req = {
      .hdr = VHSAKMT_CCMD(QUERY_INFO, sizeof(struct vhsakmt_ccmd_query_info_req)),
      .type = VHSAKMT_CCMD_QUERY_GET_SYS_PROP,
  };

  rsp = vhsakmt_alloc_rsp(dev, &req.hdr, sizeof(struct vhsakmt_ccmd_query_info_rsp));
  if (!rsp) return -ENOMEM;

  vhsakmt_execbuf_cpu(dev, &req.hdr, __FUNCTION__);
  if (!rsp) return -ENOMEM;

  memcpy(SystemProperties, &rsp->sys_props, sizeof(HsaSystemProperties));

  r = vhsakmt_set_sys_props(dev, SystemProperties);
  if (r) return r;

  return rsp->ret;
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtReleaseSystemProperties(void) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle dev = vhsakmt_dev();
  struct vhsakmt_ccmd_query_info_rsp* rsp;
  struct vhsakmt_ccmd_query_info_req req = {
      .hdr = VHSAKMT_CCMD(QUERY_INFO, sizeof(struct vhsakmt_ccmd_query_info_req)),
      .type = VHSAKMT_CCMD_QUERY_REL_SYS_PROP,
  };

  rsp = vhsakmt_alloc_rsp(dev, &req.hdr, sizeof(struct vhsakmt_ccmd_query_info_rsp));
  if (!rsp) return -ENOMEM;

  vhsakmt_execbuf_cpu(dev, &req.hdr, __FUNCTION__);
  if (!rsp) return -ENOMEM;

  if (dev->sys_props) {
    free(dev->sys_props);
    dev->sys_props = NULL;
  }

  return rsp->ret;
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtGetNodeProperties(HSAuint32 NodeId,
                                                 HsaNodeProperties* NodeProperties) {
  CHECK_VIRTIO_KFD_OPEN();

  int r;
  vhsakmt_device_handle dev = vhsakmt_dev();
  struct vhsakmt_ccmd_query_info_rsp* rsp;
  struct vhsakmt_ccmd_query_info_req req = {
      .hdr = VHSAKMT_CCMD(QUERY_INFO, sizeof(struct vhsakmt_ccmd_query_info_req)),
      .NodeID = NodeId,
      .type = VHSAKMT_CCMD_QUERY_GET_NODE_PROP,
  };

  rsp = vhsakmt_alloc_rsp(dev, &req.hdr, sizeof(struct vhsakmt_ccmd_query_info_rsp));
  if (!rsp) return -ENOMEM;

  vhsakmt_execbuf_cpu(dev, &req.hdr, __FUNCTION__);
  if (!rsp) return -ENOMEM;

  memcpy(NodeProperties, &rsp->node_props, sizeof(HsaNodeProperties));

  r = vhsakmt_set_node_props(dev, NodeId, NodeProperties);
  if (r) return r;

  return rsp->ret;
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtGetXNACKMode(HSAint32* enable) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle dev = vhsakmt_dev();
  struct vhsakmt_ccmd_query_info_rsp* rsp;
  struct vhsakmt_ccmd_query_info_req req = {
      .hdr = VHSAKMT_CCMD(QUERY_INFO, sizeof(struct vhsakmt_ccmd_query_info_req)),
      .type = VHSAKMT_CCMD_QUERY_GET_XNACK_MODE,
  };

  rsp = vhsakmt_alloc_rsp(dev, &req.hdr, sizeof(struct vhsakmt_ccmd_query_info_rsp));
  if (!rsp) return -ENOMEM;

  vhsakmt_execbuf_cpu(dev, &req.hdr, __FUNCTION__);
  if (!rsp) return -ENOMEM;

  memcpy(enable, &rsp->xnack_mode, sizeof(HSAint32));

  return rsp->ret;
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtRuntimeEnable(void* rDebug, bool setupTtmp) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle dev = vhsakmt_dev();
  struct vhsakmt_ccmd_query_info_rsp* rsp;
  struct vhsakmt_ccmd_query_info_req req = {
      .hdr = VHSAKMT_CCMD(QUERY_INFO, sizeof(struct vhsakmt_ccmd_query_info_req)),
      .run_time_enable_args.setupTtmp = setupTtmp,
      .type = VHSAKMT_CCMD_QUERY_RUN_TIME_ENABLE,
  };

  rsp = vhsakmt_alloc_rsp(dev, &req.hdr, sizeof(struct vhsakmt_ccmd_query_info_rsp));
  if (!rsp) return -ENOMEM;

  vhsakmt_execbuf_cpu(dev, &req.hdr, __FUNCTION__);
  if (!rsp) return -ENOMEM;

  return rsp->ret;
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtRuntimeDisable(void) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle dev = vhsakmt_dev();
  struct vhsakmt_ccmd_query_info_rsp* rsp;
  struct vhsakmt_ccmd_query_info_req req = {
      .hdr = VHSAKMT_CCMD(QUERY_INFO, sizeof(struct vhsakmt_ccmd_query_info_req)),
      .type = VHSAKMT_CCMD_QUERY_RUN_TIME_DISABLE,
  };

  rsp = vhsakmt_alloc_rsp(dev, &req.hdr, sizeof(struct vhsakmt_ccmd_query_info_rsp));
  if (!rsp) return -ENOMEM;

  vhsakmt_execbuf_cpu(dev, &req.hdr, __FUNCTION__);
  if (!rsp) return -ENOMEM;

  return rsp->ret;
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtGetNodeMemoryProperties(HSAuint32 NodeId, HSAuint32 NumBanks,
                                                       HsaMemoryProperties* MemoryProperties) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle dev = vhsakmt_dev();
  struct vhsakmt_ccmd_query_info_rsp* rsp;
  struct vhsakmt_ccmd_query_info_req req = {
      .hdr = VHSAKMT_CCMD(QUERY_INFO, sizeof(struct vhsakmt_ccmd_query_info_req)),
      .type = VHSAKMT_CCMD_QUERY_GET_NOD_MEM_PROP,
      .node_mem_prop_args.NodeId = NodeId,
      .node_mem_prop_args.NumBanks = NumBanks,
  };

  rsp = vhsakmt_alloc_rsp(
      dev, &req.hdr,
      sizeof(struct vhsakmt_ccmd_query_info_rsp) + NumBanks * sizeof(HsaMemoryProperties));
  if (!rsp) return -ENOMEM;

  vhsakmt_execbuf_cpu(dev, &req.hdr, __FUNCTION__);

  memcpy(MemoryProperties, rsp->payload, NumBanks * sizeof(HsaMemoryProperties));

  return rsp->ret;
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtGetNodeCacheProperties(HSAuint32 NodeId, HSAuint32 ProcessorId,
                                                      HSAuint32 NumCaches,
                                                      HsaCacheProperties* CacheProperties) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle dev = vhsakmt_dev();
  struct vhsakmt_ccmd_query_info_rsp* rsp;
  struct vhsakmt_ccmd_query_info_req req = {
      .hdr = VHSAKMT_CCMD(QUERY_INFO, sizeof(struct vhsakmt_ccmd_query_info_req)),
      .type = VHSAKMT_CCMD_QUERY_GET_NOD_CACHE_PROP,
      .node_cache_prop_args.NodeId = NodeId,
      .node_cache_prop_args.ProcessorId = ProcessorId,
      .node_cache_prop_args.NumCaches = NumCaches,
  };

  rsp = vhsakmt_alloc_rsp(
      dev, &req.hdr,
      sizeof(struct vhsakmt_ccmd_query_info_rsp) + NumCaches * sizeof(HsaCacheProperties));
  if (!rsp) return -ENOMEM;

  vhsakmt_execbuf_cpu(dev, &req.hdr, __FUNCTION__);

  memcpy(CacheProperties, rsp->payload, NumCaches * sizeof(HsaCacheProperties));

  return rsp->ret;
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtGetNodeIoLinkProperties(HSAuint32 NodeId, HSAuint32 NumIoLinks,
                                                       HsaIoLinkProperties* IoLinkProperties) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle dev = vhsakmt_dev();
  struct vhsakmt_ccmd_query_info_rsp* rsp;
  struct vhsakmt_ccmd_query_info_req req = {
      .hdr = VHSAKMT_CCMD(QUERY_INFO, sizeof(struct vhsakmt_ccmd_query_info_req)),
      .type = VHSAKMT_CCMD_QUERY_GET_NOD_IO_LINK_PROP,
      .node_io_link_args.NodeId = NodeId,
      .node_io_link_args.NumIoLinks = NumIoLinks,
  };

  rsp = vhsakmt_alloc_rsp(
      dev, &req.hdr,
      sizeof(struct vhsakmt_ccmd_query_info_rsp) + NumIoLinks * sizeof(HsaIoLinkProperties));
  if (!rsp) return -ENOMEM;

  vhsakmt_execbuf_cpu(dev, &req.hdr, __FUNCTION__);

  memcpy(IoLinkProperties, rsp->payload, NumIoLinks * sizeof(HsaIoLinkProperties));

  return rsp->ret;
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtGetClockCounters(HSAuint32 NodeId, HsaClockCounters* Counters) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle dev = vhsakmt_dev();
  struct vhsakmt_ccmd_query_info_rsp* rsp;
  struct vhsakmt_ccmd_query_info_req req = {
      .hdr = VHSAKMT_CCMD(QUERY_INFO, sizeof(struct vhsakmt_ccmd_query_info_req)),
      .type = VHSAKMT_CCMD_QUERY_GET_CLOCK_COUNTERS,
      .NodeID = NodeId,
  };

  rsp = vhsakmt_alloc_rsp(dev, &req.hdr, sizeof(struct vhsakmt_ccmd_query_info_rsp));
  if (!rsp) return -ENOMEM;

  vhsakmt_execbuf_cpu(dev, &req.hdr, __FUNCTION__);

  memcpy(Counters, &rsp->clock_counters, sizeof(HsaClockCounters));

  return rsp->ret;
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtGetRuntimeCapabilities(HSAuint32* caps_mask) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle dev = vhsakmt_dev();
  struct vhsakmt_ccmd_query_info_rsp* rsp;
  struct vhsakmt_ccmd_query_info_req req = {
      .hdr = VHSAKMT_CCMD(QUERY_INFO, sizeof(struct vhsakmt_ccmd_query_info_req)),
      .type = VHSAKMT_CCMD_QUERY_GET_RUNTIME_CAPS,
  };

  rsp = vhsakmt_alloc_rsp(dev, &req.hdr, sizeof(struct vhsakmt_ccmd_query_info_rsp));
  if (!rsp) return -ENOMEM;

  vhsakmt_execbuf_cpu(dev, &req.hdr, __FUNCTION__);

  *caps_mask = rsp->caps;

  return rsp->ret;
}
