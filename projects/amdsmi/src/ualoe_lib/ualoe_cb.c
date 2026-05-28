/*
 * Copyright (c) 2025 Advanced Micro Devices, Inc. (AMD)
 *
 * This file contains confidential and proprietary information
 * of Advanced Micro Devices, Inc. and is protected under U.S.
 * and international copyright and other intellectual property
 * laws.
 *
 * DISCLAIMER
 * This disclaimer is not a license and does not grant any
 * rights to the materials distributed herewith. Except as
 * otherwise provided in a valid license issued to you by
 * AMD, and to the maximum extent permitted by applicable
 * law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
 * WITH ALL FAULTS, AND AMD HEREBY DISCLAIMS ALL WARRANTIES
 * AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
 * BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
 * INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
 * (2) AMD shall not be liable (whether in contract or tort,
 * including negligence, or under any other theory of
 * liability) for any loss or damage of any kind or nature
 * related to, arising under or in connection with these
 * materials, including for any direct, or any indirect,
 * special, incidental, or consequential loss or damage
 * (including loss of data, profits, goodwill, or any type of
 * loss or damage suffered as a result of any action brought
 * by a third party) even if such damage or loss was
 * reasonably foreseeable or AMD had been advised of the
 * possibility of the same.
 *
 * CRITICAL APPLICATIONS
 * AMD products are not designed or intended to be fail-
 * safe, or for use in any application requiring fail-safe
 * performance, such as life-support or safety devices or
 * systems, Class III medical devices, nuclear facilities,
 * applications related to the deployment of airbags, or any
 * other applications that could lead to death, personal
 * injury, or severe property or environmental damage
 * (individually and collectively, "Critical
 * Applications"). Customer assumes the sole risk and
 * liability of any use of AMD products in Critical
 * Applications, subject only to applicable laws and
 * regulations governing limitations on product liability.
 *
 * THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
 * PART OF THIS FILE AT ALL TIMES
 */

#include "ualoe_cb.h"

#include <cbl_cfg/uapi.h>
#include <errno.h>
#include <netlink/genl/ctrl.h>
#include <netlink/genl/family.h>
#include <netlink/genl/genl.h>
#include <netlink/netlink.h>
#include <netlink/socket.h>
#include <pthread.h>
#include <stdbool.h>
#include <sys/eventfd.h>
#include <sys/queue.h>
#include <unistd.h>

#include "ualoe_log.h"

struct cfg_cb_ctx {
  struct nl_sock* sk;
  ualoe_event_callback_t callback;
  ualoe_handle_t handle;
  int dev_id;
  void* user_context;
  uint32_t seq;
  int family_id;
  int mcast_grp;
  LIST_ENTRY(cfg_cb_ctx) lentry;
  pthread_t tid;
  int shutdown_fd;
};

LIST_HEAD(cb_threads, cfg_cb_ctx);
static struct cb_threads registered_cbs = LIST_HEAD_INITIALIZER(cb_threads);
static pthread_mutex_t cb_lock = PTHREAD_MUTEX_INITIALIZER;

static const struct nla_policy cfg_timestamp_policy[] = {
    [CFG_EVT_ATTR_TS_SECS] = {.type = NLA_U64},
    [CFG_EVT_ATTR_TS_NSECS] = {.type = NLA_U64},
};

static struct nla_policy cfg_phase_event_policy[] = {
    [CFG_EVT_ATTR_TS] = {.type = NLA_NESTED},
    [CFG_EVT_ATTR_PHASE_EVENT] = {.type = NLA_U64},
    [CFG_EVT_ATTR_DEV_ID] = {.type = NLA_S32},
};

static struct nla_policy cfg_ifoe_link_event_policy[] = {
    [CFG_EVT_ATTR_TS] = {.type = NLA_NESTED},
    [CFG_EVT_ATTR_LABEL] = {.type = NLA_STRING, .maxlen = UALOE_LABEL_SIZE},
    [CFG_EVT_ATTR_LOGICAL_IDX] = {.type = NLA_U32},
    [CFG_EVT_ATTR_LINK_DOWN] = {.type = NLA_U8},
    [CFG_EVT_ATTR_DX_ISOLATED] = {.type = NLA_U8},
    [CFG_EVT_ATTR_NETPORT_COUNT] = {.type = NLA_U32},
    [CFG_EVT_ATTR_NETPORT0] = {.type = NLA_NESTED},
    [CFG_EVT_ATTR_NETPORT1] = {.type = NLA_NESTED},
    [CFG_EVT_ATTR_NETPORT2] = {.type = NLA_NESTED},
    [CFG_EVT_ATTR_NETPORT3] = {.type = NLA_NESTED},
    [CFG_EVT_ATTR_DEV_ID] = {.type = NLA_S32},
};

static struct nla_policy cfg_netport_policy[] = {
    [CFG_EVT_ATTR_NETPORT_LOGICAL_IDX] = {.type = NLA_U32},
    [CFG_EVT_ATTR_NETPORT_FAULT] = {.type = NLA_U8},
    [CFG_EVT_ATTR_NETPORT_STREAMS_FAILOVER] = {.type = NLA_U32},
    [CFG_EVT_ATTR_NETPORT_STREAMS_PAUSED] = {.type = NLA_U32},
};

static struct nla_policy cfg_netport_link_event_policy[] = {
    [CFG_EVT_ATTR_TS] = {.type = NLA_NESTED},
    [CFG_EVT_ATTR_LABEL] = {.type = NLA_STRING, .maxlen = UALOE_LABEL_SIZE},
    [CFG_EVT_ATTR_LOGICAL_IDX] = {.type = NLA_U32},
    [CFG_EVT_ATTR_LINK_DOWN] = {.type = NLA_U8},
    [CFG_EVT_ATTR_DEV_ID] = {.type = NLA_S32},
};

static int ualoe_cb_parse_ts(struct nlattr* attr, struct timespec* ts) {
  struct nlattr* attrs[CFG_EVT_ATTR_TS_MAX + 1];
  int rc;

  rc = nla_parse_nested(attrs, CFG_EVT_ATTR_TS_MAX, attr, cfg_timestamp_policy);
  if (rc) return rc;

  if (!attrs[CFG_EVT_ATTR_TS_SECS] || !attrs[CFG_EVT_ATTR_TS_NSECS]) {
    ualoe_log_error("Invalid TS attribute\n");
    return EINVAL;
  }

  ts->tv_sec = nla_get_u64(attrs[CFG_EVT_ATTR_TS_SECS]);
  ts->tv_nsec = nla_get_u64(attrs[CFG_EVT_ATTR_TS_NSECS]);

  return 0;
}

static int ualoe_cb_process_phase_event(struct nl_msg* msg, void* arg) {
  /* Callers must never hold cb_lock */
  struct nlattr* attrs[CFG_EVT_ATTR_MAX + 1];
  struct nlmsghdr* nlh = nlmsg_hdr(msg);
  struct genlmsghdr* genlh = nlmsg_data(nlh);
  struct cfg_cb_ctx* cb_ctx = arg;
  ualoe_event_t event;
  int dev_id;
  int rc;

  rc = nla_parse(attrs, CFG_EVT_ATTR_MAX, genlmsg_attrdata(genlh, 0), genlmsg_attrlen(genlh, 0),
                 cfg_phase_event_policy);
  if (rc) return NL_OK;

  if (!attrs[CFG_EVT_ATTR_DEV_ID]) {
    ualoe_log_error("No DEV_ID attribute found in phase event\n");
    return NL_OK;
  }

  dev_id = nla_get_s32(attrs[CFG_EVT_ATTR_DEV_ID]);
  if (dev_id != cb_ctx->dev_id) {
    return NL_OK;
  }

  if (!attrs[CFG_EVT_ATTR_TS]) {
    ualoe_log_error("No TS attribute found in phase event\n");
    return NL_OK;
  }

  if (!attrs[CFG_EVT_ATTR_PHASE_EVENT]) {
    ualoe_log_error("No phase change event attribute found\n");
    return NL_OK;
  }

  event.id = UALOE_EVENT_CONFIG_PHASE_CHANGE;
  event.u.config_phase_change.phase = nla_get_u64(attrs[CFG_EVT_ATTR_PHASE_EVENT]);

  rc = ualoe_cb_parse_ts(attrs[CFG_EVT_ATTR_TS], &event.timestamp);
  if (rc) return NL_OK;

  if (cb_ctx->callback) cb_ctx->callback(cb_ctx->user_context, cb_ctx->handle, event);

  return NL_OK;
}

static int ualoe_cb_parse_netport(struct nlattr* attr, int attrtype, ualoe_event_t* event,
                                  unsigned int idx) {
  struct nlattr* nested_attrs[CFG_EVT_ATTR_NETPORT_MAX + 1];
  int rc;

  rc = nla_parse_nested(nested_attrs, attrtype, attr, cfg_netport_policy);
  if (rc) return rc;

  if (!nested_attrs[CFG_EVT_ATTR_NETPORT_LOGICAL_IDX] ||
      !nested_attrs[CFG_EVT_ATTR_NETPORT_FAULT] ||
      !nested_attrs[CFG_EVT_ATTR_NETPORT_STREAMS_FAILOVER] ||
      !nested_attrs[CFG_EVT_ATTR_NETPORT_STREAMS_PAUSED]) {
    ualoe_log_error("Missing mandatory attributes in netport \n");
    return EINVAL;
  }

  event->u.ifoe_link_change.netports[idx].logical_idx =
      nla_get_u32(nested_attrs[CFG_EVT_ATTR_NETPORT_LOGICAL_IDX]);
  event->u.ifoe_link_change.netports[idx].fault =
      nla_get_u8(nested_attrs[CFG_EVT_ATTR_NETPORT_FAULT]) ? true : false;
  event->u.ifoe_link_change.netports[idx].streams_failover =
      nla_get_u32(nested_attrs[CFG_EVT_ATTR_NETPORT_STREAMS_FAILOVER]);
  event->u.ifoe_link_change.netports[idx].streams_paused =
      nla_get_u32(nested_attrs[CFG_EVT_ATTR_NETPORT_STREAMS_PAUSED]);

  return 0;
}

static int ualoe_cb_parse_ifoe_link_event(struct nlattr** attrs, ualoe_event_t* event) {
  int netport_attr, rc;

  if (!attrs[CFG_EVT_ATTR_LABEL] || !attrs[CFG_EVT_ATTR_LOGICAL_IDX] ||
      !attrs[CFG_EVT_ATTR_LINK_DOWN] || !attrs[CFG_EVT_ATTR_DX_ISOLATED] ||
      !attrs[CFG_EVT_ATTR_NETPORT_COUNT]) {
    ualoe_log_error("Missing mandatory attributes in IFoE link event\n");
    return EINVAL;
  }

  nla_strlcpy(event->u.ifoe_link_change.name.text, attrs[CFG_EVT_ATTR_LABEL], UALOE_LABEL_SIZE);
  event->u.ifoe_link_change.logical_idx = nla_get_u32(attrs[CFG_EVT_ATTR_LOGICAL_IDX]);
  event->u.ifoe_link_change.link_down = nla_get_u8(attrs[CFG_EVT_ATTR_LINK_DOWN]) ? true : false;
  event->u.ifoe_link_change.dx_isolated =
      nla_get_u8(attrs[CFG_EVT_ATTR_DX_ISOLATED]) ? true : false;
  event->u.ifoe_link_change.netport_count = nla_get_u32(attrs[CFG_EVT_ATTR_NETPORT_COUNT]);

  if (event->u.ifoe_link_change.netport_count > UALOE_MAX_NETPORTS_PER_IFOE_STATION) {
    ualoe_log_error("Invalid netport count %u in IFoE link event\n",
                    event->u.ifoe_link_change.netport_count);
    return EINVAL;
  }

  for (uint32_t i = 0; i < event->u.ifoe_link_change.netport_count; i++) {
    switch (i) {
      case 0:
        netport_attr = CFG_EVT_ATTR_NETPORT0;
        break;
      case 1:
        netport_attr = CFG_EVT_ATTR_NETPORT1;
        break;
      case 2:
        netport_attr = CFG_EVT_ATTR_NETPORT2;
        break;
      case 3:
        netport_attr = CFG_EVT_ATTR_NETPORT3;
        break;
      default:
        ualoe_log_error("Unexpected netport index %u in IFoE link event\n", i);
        return EINVAL;
    }

    rc = ualoe_cb_parse_netport(attrs[netport_attr], CFG_EVT_ATTR_NETPORT_MAX, event, i);
    if (rc) return rc;
  }
  return 0;
}

static int ualoe_cb_process_ifoe_link_event(struct nl_msg* msg, void* arg) {
  struct nlattr* attrs[CFG_EVT_ATTR_MAX + 1];
  struct nlmsghdr* nlh = nlmsg_hdr(msg);
  struct genlmsghdr* genlh = nlmsg_data(nlh);
  struct cfg_cb_ctx* cb_ctx = arg;
  ualoe_event_t event;
  int dev_id;
  int rc;

  rc = nla_parse(attrs, CFG_EVT_ATTR_MAX, genlmsg_attrdata(genlh, 0), genlmsg_attrlen(genlh, 0),
                 cfg_ifoe_link_event_policy);
  if (rc) return NL_OK;

  if (!attrs[CFG_EVT_ATTR_DEV_ID]) {
    ualoe_log_error("No DEV_ID attribute found in IFoE link event\n");
    return NL_OK;
  }

  dev_id = nla_get_s32(attrs[CFG_EVT_ATTR_DEV_ID]);
  if (dev_id != cb_ctx->dev_id) {
    return NL_OK;
  }

  if (!attrs[CFG_EVT_ATTR_TS]) {
    ualoe_log_error("No TS attribute found in IFoE link event\n");
    return NL_OK;
  }

  event.id = UALOE_EVENT_IFOE_LINK_CHANGE;
  rc = ualoe_cb_parse_ts(attrs[CFG_EVT_ATTR_TS], &event.timestamp);
  if (rc) return NL_OK;

  rc = ualoe_cb_parse_ifoe_link_event(attrs, &event);
  if (rc) return NL_OK;

  if (cb_ctx->callback) cb_ctx->callback(cb_ctx->user_context, cb_ctx->handle, event);

  return NL_OK;
}

static int ualoe_cb_process_netport_link_event(struct nl_msg* msg, void* arg) {
  struct nlattr* attrs[CFG_EVT_ATTR_MAX + 1];
  struct nlmsghdr* nlh = nlmsg_hdr(msg);
  struct genlmsghdr* genlh = nlmsg_data(nlh);
  struct cfg_cb_ctx* cb_ctx = arg;
  ualoe_event_t event;
  int dev_id;
  int rc;

  rc = nla_parse(attrs, CFG_EVT_ATTR_MAX, genlmsg_attrdata(genlh, 0), genlmsg_attrlen(genlh, 0),
                 cfg_netport_link_event_policy);
  if (rc) return NL_OK;

  if (!attrs[CFG_EVT_ATTR_DEV_ID]) {
    ualoe_log_error("No DEV_ID attribute found in netport link event\n");
    return NL_OK;
  }

  dev_id = nla_get_s32(attrs[CFG_EVT_ATTR_DEV_ID]);
  if (dev_id != cb_ctx->dev_id) {
    return NL_OK;
  }

  if (!attrs[CFG_EVT_ATTR_TS] || !attrs[CFG_EVT_ATTR_LABEL] || !attrs[CFG_EVT_ATTR_LOGICAL_IDX] ||
      !attrs[CFG_EVT_ATTR_LINK_DOWN]) {
    ualoe_log_error("Missing attribute in netport link change event\n");
    return NL_OK;
  }

  event.id = UALOE_EVENT_NETPORT_LINK_CHANGE;

  rc = ualoe_cb_parse_ts(attrs[CFG_EVT_ATTR_TS], &event.timestamp);
  if (rc) return NL_OK;

  nla_strlcpy(event.u.netport_link_change.name.text, attrs[CFG_EVT_ATTR_LABEL], UALOE_LABEL_SIZE);
  event.u.netport_link_change.logical_idx = nla_get_u32(attrs[CFG_EVT_ATTR_LOGICAL_IDX]);
  event.u.netport_link_change.link_down = nla_get_u8(attrs[CFG_EVT_ATTR_LINK_DOWN]) ? true : false;

  if (cb_ctx->callback) cb_ctx->callback(cb_ctx->user_context, cb_ctx->handle, event);
  return NL_OK;
}

static int ualoe_cb_process_event(struct nl_msg* msg, void* arg) {
  struct nlmsghdr* nlh = nlmsg_hdr(msg);
  struct genlmsghdr* genlh = nlmsg_data(nlh);

  switch (genlh->cmd) {
    case CFG_EVT_CMD_PHASE_EVENT:
      return ualoe_cb_process_phase_event(msg, arg);
    case CFG_EVT_CMD_IFOE_LINK_EVENT:
      return ualoe_cb_process_ifoe_link_event(msg, arg);
    case CFG_EVT_CMD_NETPORT_LINK_EVENT:
      return ualoe_cb_process_netport_link_event(msg, arg);
    default:
      ualoe_log_error("Unknown cmd in event message: %d\n", genlh->cmd);
      return NL_SKIP;
  }
}

static void* ualoe_cb_event_poller(void* arg) {
  /* Callers must never hold cb_lock */
  struct cfg_cb_ctx* cb_ctx = arg;
  struct pollfd fds[2];
  int rc;

  fds[0].fd = nl_socket_get_fd(cb_ctx->sk);
  fds[0].events = POLLIN;
  fds[1].fd = cb_ctx->shutdown_fd;
  fds[1].events = POLLIN;

  while (1) {
    rc = poll(fds, 2, -1);
    if (rc < 0) break;

    /* Check if shutdown was requested */
    if (fds[1].revents & POLLIN) break;

    /* Check for error conditions on netlink socket */
    if (fds[0].revents & (POLLERR | POLLHUP | POLLNVAL)) break;

    /* Data available on netlink socket, process it */
    if (fds[0].revents & POLLIN) {
      rc = nl_recvmsgs_default(cb_ctx->sk);
      if (rc < 0) {
        ualoe_log_error("nl_recvmsgs_default failed: %d\n", rc);
        break;
      }
    }
  }

  return NULL;
}

static void ualoe_cb_ctx_init(struct cfg_cb_ctx* cb_ctx, ualoe_handle_t handle, int dev_id,
                              ualoe_event_callback_t callback, void* user_context) {
  cb_ctx->seq = 0;
  cb_ctx->family_id = 0;
  cb_ctx->mcast_grp = 0;
  cb_ctx->dev_id = dev_id;
  cb_ctx->callback = callback;
  cb_ctx->handle = handle;
  cb_ctx->user_context = user_context;
  cb_ctx->tid = 0;
  cb_ctx->shutdown_fd = -1;
}

int ualoe_cb_init(ualoe_handle_t handle, int dev_id, ualoe_event_callback_t callback,
                  void* user_context) {
  struct cfg_cb_ctx *cb_ctx, *iter;
  pthread_t tid;
  int rc;

  cb_ctx = malloc(sizeof(*cb_ctx));
  if (!cb_ctx) return ENOMEM;

  ualoe_cb_ctx_init(cb_ctx, handle, dev_id, callback, user_context);

  cb_ctx->shutdown_fd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  if (cb_ctx->shutdown_fd < 0) {
    rc = errno;
    goto free_cb_ctx;
  }

  cb_ctx->sk = nl_socket_alloc();
  if (!cb_ctx->sk) {
    rc = ENOMEM;
    goto close_eventfd;
  }

  rc = genl_connect(cb_ctx->sk);
  if (rc < 0) {
    rc = -rc;
    goto free_nl_sock;
  }

  cb_ctx->family_id = genl_ctrl_resolve(cb_ctx->sk, CFG_FAMILY_NAME);
  if (cb_ctx->family_id < 0) {
    rc = -cb_ctx->family_id;
    goto close_sock;
  }

  /* Disable checking for sequence checks for async multicast messages. */
  nl_socket_disable_seq_check(cb_ctx->sk);

  cb_ctx->mcast_grp = genl_ctrl_resolve_grp(cb_ctx->sk, CFG_FAMILY_NAME, CFG_MC_GRP_NAME);
  if (cb_ctx->mcast_grp < 0) {
    rc = -cb_ctx->mcast_grp;
    goto close_sock;
  }

  rc = nl_socket_add_membership(cb_ctx->sk, cb_ctx->mcast_grp);
  if (rc < 0) {
    rc = -rc;
    goto close_sock;
  }

  rc = nl_socket_modify_cb(cb_ctx->sk, NL_CB_VALID, NL_CB_CUSTOM, ualoe_cb_process_event, cb_ctx);
  if (rc) goto close_sock;

  pthread_mutex_lock(&cb_lock);
  LIST_FOREACH(iter, &registered_cbs, lentry) {
    if (iter->handle == handle) {
      rc = EALREADY;
      pthread_mutex_unlock(&cb_lock);
      goto close_sock;
    }
  }

  LIST_INSERT_HEAD(&registered_cbs, cb_ctx, lentry);
  pthread_mutex_unlock(&cb_lock);

  if (pthread_create(&tid, NULL, ualoe_cb_event_poller, cb_ctx)) {
    rc = errno;
    pthread_mutex_lock(&cb_lock);
    LIST_REMOVE(cb_ctx, lentry);
    pthread_mutex_unlock(&cb_lock);
    goto close_sock;
  }

  cb_ctx->tid = tid;

  return 0;

close_sock:
  nl_close(cb_ctx->sk);
free_nl_sock:
  nl_socket_free(cb_ctx->sk);
close_eventfd:
  close(cb_ctx->shutdown_fd);
free_cb_ctx:
  free(cb_ctx);
  return rc;
}

static void ualoe_cb_unreg(struct cfg_cb_ctx* cb_ctx) {
  /* Callers must hold cb_lock */
  void* res;
  int rc;

  if (!cb_ctx) return;

  /*
   * Write to eventfd to signal the thread to exit gracefully.
   * The thread is blocked on poll() waiting for either netlink events
   * or shutdown signal, and will exit cleanly without interrupting
   * nl_recv() mid-allocation.
   */
  if (eventfd_write(cb_ctx->shutdown_fd, 1) < 0)
    ualoe_log_error("Failed to write to shutdown eventfd: %d\n", errno);

  rc = pthread_join(cb_ctx->tid, &res);
  if (rc) ualoe_log_error("Failed to join cb thread rc=%d\n", rc);

  if (res != PTHREAD_CANCELED)
    ualoe_log_error("Handle %d's cb thread not canceled\n", cb_ctx->handle);

  LIST_REMOVE(cb_ctx, lentry);
  nl_close(cb_ctx->sk);
  nl_socket_free(cb_ctx->sk);
  close(cb_ctx->shutdown_fd);
  free(cb_ctx);
}

void ualoe_cb_fini(ualoe_handle_t handle) {
  struct cfg_cb_ctx* cb_ctx;

  pthread_mutex_lock(&cb_lock);
  LIST_FOREACH(cb_ctx, &registered_cbs, lentry) {
    if (cb_ctx->handle == handle) {
      ualoe_cb_unreg(cb_ctx);
      break;
    }
  }
  pthread_mutex_unlock(&cb_lock);
}
