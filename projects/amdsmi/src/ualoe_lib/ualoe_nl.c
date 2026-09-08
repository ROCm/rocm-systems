// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "ualoe_nl.h"

#include <cbl_cfg/uapi.h>
#include <errno.h>
#include <libmnl/libmnl.h>
#include <netlink/genl/ctrl.h>
#include <netlink/genl/family.h>
#include <netlink/genl/genl.h>
#include <netlink/netlink.h>
#include <netlink/socket.h>
#include <pthread.h>
#include <stdbool.h>
#include <sys/queue.h>

#include "ualoe_cb.h"
#include "ualoe_cdev.h"
#include "ualoe_lib.h"
#include "ualoe_log.h"

struct ualoe_nl_handle {
  struct mnl_socket* sk;
  int dev_id;
  int family_id;
  int fd;
  int port_id;
  int seq;
  int cdev_fd;
  pthread_mutex_t request_lock;
  LIST_ENTRY(ualoe_nl_handle) lentry;
};

LIST_HEAD(handles, ualoe_nl_handle);
static struct handles open_handles = LIST_HEAD_INITIALIZER(handles);
static pthread_mutex_t handle_lock = PTHREAD_MUTEX_INITIALIZER;

static void ualoe_nl_init_msg(struct ualoe_nl_handle* handle, struct nlmsghdr** nlh,
                              struct genlmsghdr** genlh, char* buf, int cmd, int type, int vers) {
  *nlh = mnl_nlmsg_put_header(buf);
  (*nlh)->nlmsg_type = type;
  (*nlh)->nlmsg_flags = NLM_F_REQUEST | NLM_F_ACK;
  (*nlh)->nlmsg_seq = 0;
  (*nlh)->nlmsg_pid = handle->port_id;
  *genlh = mnl_nlmsg_put_extra_header(*nlh, sizeof(**genlh));
  (*genlh)->cmd = cmd;
  (*genlh)->version = vers;
}

static void ualoe_nl_request_lock(struct ualoe_nl_handle* nl_handle, struct nlmsghdr* nlh) {
  pthread_mutex_lock(&nl_handle->request_lock);
  nlh->nlmsg_seq = ++nl_handle->seq;
}

static void ualoe_nl_request_unlock(struct ualoe_nl_handle* nl_handle) {
  pthread_mutex_unlock(&nl_handle->request_lock);
}

struct ualoe_nl_response_ctx {
  mnl_cb_t cb;
  void* data;
  int callback_error;
  bool skip_payloads;
};

static int ualoe_nl_response_cb(const struct nlmsghdr* nlh, void* data) {
  struct ualoe_nl_response_ctx* ctx = data;
  int rc;

  if (ctx->skip_payloads) return MNL_CB_OK;

  errno = 0;
  rc = ctx->cb ? ctx->cb(nlh, ctx->data) : MNL_CB_OK;
  if (rc == MNL_CB_ERROR) {
    ctx->callback_error = errno ? errno : EPROTO;
    ctx->skip_payloads = true;
  } else if (rc == MNL_CB_STOP) {
    ctx->skip_payloads = true;
  }

  /* Continue until mnl_cb_run consumes the terminal ACK or NLMSG_DONE. */
  return MNL_CB_OK;
}

static int ualoe_nl_send_recv(struct ualoe_nl_handle* nl_handle, char* buf, size_t buf_size,
                              struct nlmsghdr* nlh, mnl_cb_t cb, void* cb_data) {
  struct ualoe_nl_response_ctx ctx = {
      .cb = cb,
      .data = cb_data,
  };
  unsigned int exp_seq, exp_pid;
  int rc;

  ualoe_nl_request_lock(nl_handle, nlh);

  if (mnl_socket_sendto(nl_handle->sk, buf, nlh->nlmsg_len) < 0) {
    rc = errno;
    ualoe_log_error("Failed to send message rc=%d\n", rc);
    goto unlock;
  }

  exp_seq = nlh->nlmsg_seq;
  exp_pid = nlh->nlmsg_pid;

  rc = mnl_socket_recvfrom(nl_handle->sk, buf, buf_size);
  while (rc > 0) {
    rc = mnl_cb_run(buf, rc, exp_seq, exp_pid, ualoe_nl_response_cb, &ctx);
    if (rc <= MNL_CB_STOP) break;
    rc = mnl_socket_recvfrom(nl_handle->sk, buf, buf_size);
  }

  if (ctx.callback_error) {
    rc = ctx.callback_error;
  } else if (rc == -1) {
    rc = errno;
  } else {
    rc = 0;
  }

unlock:
  ualoe_nl_request_unlock(nl_handle);
  return rc;
}

static int ualoe_nl_get_family_attrs(const struct nlattr* attr, void* data) {
  struct ualoe_nl_handle* nl_handle = data;
  int type = mnl_attr_get_type(attr);

  if (type != CTRL_ATTR_FAMILY_ID) return MNL_CB_OK;

  nl_handle->family_id = mnl_attr_get_u16(attr);
  return MNL_CB_OK;
}

static int ualoe_nl_get_family_cb(const struct nlmsghdr* nlh, void* data) {
  return mnl_attr_parse(nlh, sizeof(struct genlmsghdr), ualoe_nl_get_family_attrs, data);
}

static int ualoe_nl_get_family_id(struct ualoe_nl_handle* nl_handle) {
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CTRL_CMD_GETFAMILY, GENL_ID_CTRL, 2);

  mnl_attr_put_strz(nlh, CTRL_ATTR_FAMILY_NAME, CFG_FAMILY_NAME);

  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, ualoe_nl_get_family_cb, nl_handle);
}

static int ualoe_nl_connect_attrs(const struct nlattr* attr, void* data) {
  struct ualoe_nl_handle* nl_handle = data;
  int type = mnl_attr_get_type(attr);

  switch (type) {
    case CFG_ATTR_DEV_ID:
      nl_handle->dev_id = mnl_attr_get_u32(attr);
      break;
    default:
      break;
  }
  return MNL_CB_OK;
}

static int ualoe_nl_connect_cb(const struct nlmsghdr* nlh, void* data) {
  return mnl_attr_parse(nlh, sizeof(struct genlmsghdr), ualoe_nl_connect_attrs, data);
}

static int ualoe_nl_connect(struct ualoe_nl_handle* nl_handle, const char* addr) {
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_CONNECT, nl_handle->family_id, 1);

  mnl_attr_put_strz(nlh, CFG_ATTR_PCI_ADDR, addr);

  rc = ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, ualoe_nl_connect_cb, nl_handle);
  if (rc) return rc;

  if (nl_handle->dev_id < 0) {
    ualoe_log_error("Device %s did not return a device ID\n", addr);
    return ENODEV;
  }

  return 0;
}

int ualoe_nl_open(const char* name, ualoe_handle_t* handle) {
  struct ualoe_nl_handle* nl_handle;
  int rc;

  nl_handle = malloc(sizeof(*nl_handle));
  if (!nl_handle) return ENOMEM;

  rc = pthread_mutex_init(&nl_handle->request_lock, NULL);
  if (rc) goto free_handle;

  nl_handle->seq = 0;
  nl_handle->sk = mnl_socket_open(NETLINK_GENERIC);
  if (!nl_handle->sk) {
    rc = errno;
    goto destroy_mutex;
  }

  if (mnl_socket_bind(nl_handle->sk, 0, MNL_SOCKET_AUTOPID) < 0) {
    rc = errno;
    goto free_socket;
  }

  nl_handle->fd = mnl_socket_get_fd(nl_handle->sk);
  nl_handle->port_id = mnl_socket_get_portid(nl_handle->sk);
  nl_handle->dev_id = -1; /* Initialize to invalid before connect */

  rc = ualoe_nl_get_family_id(nl_handle);
  if (rc) goto free_socket;

  rc = ualoe_nl_connect(nl_handle, name);
  if (rc) goto free_socket;

  /**
   * For now, keep cdev fd for ioctl calls until netlink support is
   * added for all operations.
   */
  rc = ualoe_cdev_open(name, &nl_handle->cdev_fd);
  if (rc) goto free_socket;

  pthread_mutex_lock(&handle_lock);
  LIST_INSERT_HEAD(&open_handles, nl_handle, lentry);
  pthread_mutex_unlock(&handle_lock);

  *handle = nl_handle->fd;

  return 0;

free_socket:
  mnl_socket_close(nl_handle->sk);
destroy_mutex:
  pthread_mutex_destroy(&nl_handle->request_lock);
free_handle:
  free(nl_handle);
  return rc;
}

int ualoe_nl_close(ualoe_handle_t handle) {
  struct ualoe_nl_handle* nl_handle;

  pthread_mutex_lock(&handle_lock);
  LIST_FOREACH(nl_handle, &open_handles, lentry) {
    if (nl_handle->fd == handle) {
      LIST_REMOVE(nl_handle, lentry);
      break;
    }
  }
  pthread_mutex_unlock(&handle_lock);

  if (!nl_handle) return ENOENT;

  /**
   * For now, close cdev fd until netlink support is added for all
   * operations.
   */
  ualoe_cb_fini(handle);

  pthread_mutex_lock(&nl_handle->request_lock);
  ualoe_cdev_close(nl_handle->cdev_fd);
  mnl_socket_close(nl_handle->sk);
  pthread_mutex_unlock(&nl_handle->request_lock);
  pthread_mutex_destroy(&nl_handle->request_lock);
  free(nl_handle);

  return 0;
}

static int ualoe_nl_find_handle(ualoe_handle_t handle, struct ualoe_nl_handle** nl_handle) {
  struct ualoe_nl_handle* iter;

  *nl_handle = NULL;
  pthread_mutex_lock(&handle_lock);
  LIST_FOREACH(iter, &open_handles, lentry) {
    if (iter->fd == handle) {
      *nl_handle = iter;
      break;
    }
  }
  pthread_mutex_unlock(&handle_lock);

  return *nl_handle ? 0 : ENOENT;
}

static int ualoe_nl_parse_version_nested(const struct nlattr* attr, void* arg) {
  ualoe_version_t* version = arg;

  if (!version) return MNL_CB_OK;

  switch (mnl_attr_get_type(attr)) {
    case CFG_ATTR_VERSION_MAJOR:
      version->major = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_VERSION_MINOR:
      version->minor = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_VERSION_PATCH:
      version->patch = mnl_attr_get_u32(attr);
      break;
    default:
      errno = EINVAL;
      return MNL_CB_ERROR;
  }
  return MNL_CB_OK;
}

struct ualoe_nl_version_ctx {
  ualoe_version_t* fw_version;
  ualoe_version_t* telemetry_version;
};

static int ualoe_nl_parse_get_version(const struct nlattr* attr, void* arg) {
  struct ualoe_nl_version_ctx* ctx = arg;

  switch (mnl_attr_get_type(attr)) {
    case CFG_ATTR_FW_VERSION:
      return mnl_attr_parse_nested(attr, ualoe_nl_parse_version_nested, ctx->fw_version);
    case CFG_ATTR_TELEMETRY_VERSION:
      return mnl_attr_parse_nested(attr, ualoe_nl_parse_version_nested, ctx->telemetry_version);
    default:
      return MNL_CB_OK;
  }
}

static int ualoe_nl_get_version_handler(const struct nlmsghdr* nlh, void* arg) {
  return mnl_attr_parse(nlh, sizeof(struct genlmsghdr), ualoe_nl_parse_get_version, arg);
}

int ualoe_nl_get_version(ualoe_handle_t handle, ualoe_version_t* lib_version,
                         ualoe_version_t* fw_version, ualoe_version_t* telemetry_version) {
  struct ualoe_nl_handle* nl_handle;
  struct ualoe_nl_version_ctx ctx;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  /* Populate library version (static) */
  if (lib_version) {
    lib_version->major = UALOE_LIB_VERSION_MAJOR;
    lib_version->minor = UALOE_LIB_VERSION_MINOR;
    lib_version->patch = UALOE_LIB_VERSION_PATCH;
  }

  /* Prepare context for parsing */
  ctx.fw_version = fw_version;
  ctx.telemetry_version = telemetry_version;

  /* Send request to kernel */
  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_GET_VERSION, nl_handle->family_id, 1);

  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);

  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, ualoe_nl_get_version_handler, &ctx);
}

int ualoe_nl_reset(ualoe_handle_t handle) {
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_RESET, nl_handle->family_id, 1);

  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);
  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, NULL, NULL);
}

static int ualoe_nl_parse_get_caps_nested(const struct nlattr* attr, void* arg) {
  ualoe_capabilities_t* caps = arg;

  switch (mnl_attr_get_type(attr)) {
    case CFG_ATTR_CAPS_IFOE_STATION_COUNT:
      caps->num_configured_stations = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_CAPS_ACCELERATOR_COUNT:
      caps->max_accelerators = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_CAPS_NETPORTS_PER_STATION:
      caps->num_netports_per_station = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_CAPS_PATHS_PER_STATION:
      caps->num_paths_per_station = mnl_attr_get_u32(attr);
      break;
    default:
      errno = EINVAL;
      return MNL_CB_ERROR;
  }
  return MNL_CB_OK;
}

static int ualoe_nl_parse_get_caps(const struct nlattr* attr, void* arg) {
  switch (mnl_attr_get_type(attr)) {
    case CFG_ATTR_CAPABILITIES:
      return mnl_attr_parse_nested(attr, ualoe_nl_parse_get_caps_nested, arg);
    default:
      return MNL_CB_OK;
  }
}

static int ualoe_nl_get_caps_handler(const struct nlmsghdr* nlh, void* arg) {
  return mnl_attr_parse(nlh, sizeof(struct genlmsghdr), ualoe_nl_parse_get_caps, arg);
}

int ualoe_nl_get_capabilities(ualoe_handle_t handle, ualoe_capabilities_t* caps) {
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_GET_CAPABILITIES, nl_handle->family_id,
                    1);

  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);

  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, ualoe_nl_get_caps_handler, caps);
}

int ualoe_nl_set_identity(ualoe_handle_t handle, uint32_t accelerator_id) {
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_SET_ACCELERATOR_ID, nl_handle->family_id,
                    1);

  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);
  mnl_attr_put_u32(nlh, CFG_ATTR_ACCELERATOR_ID, accelerator_id);

  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, NULL, NULL);
}

int ualoe_nl_set_accelerator_config(ualoe_handle_t handle, unsigned bitmask_size,
                                    uint32_t active_accelerator_bitmask[],
                                    uint32_t local_accelerator_bitmask[]) {
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  struct nlattr* nest;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_SET_ACCELERATOR_CONFIG,
                    nl_handle->family_id, 1);

  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);

  nest = mnl_attr_nest_start(nlh, CFG_ATTR_ACCEL_CONFIG);
  mnl_attr_put_u32(nlh, CFG_ATTR_ACCEL_CONFIG_BITMASK_SIZE, bitmask_size);
  if (bitmask_size > 0 && active_accelerator_bitmask) {
    if (!mnl_attr_put_check(nlh, sizeof(buf), CFG_ATTR_ACCEL_CONFIG_ACTIVE_BITMASK, bitmask_size,
                            active_accelerator_bitmask)) {
      return ENOMEM;
    }
  }

  if (bitmask_size > 0 && local_accelerator_bitmask) {
    if (!mnl_attr_put_check(nlh, sizeof(buf), CFG_ATTR_ACCEL_CONFIG_LOCAL_BITMASK, bitmask_size,
                            local_accelerator_bitmask)) {
      return ENOMEM;
    }
  }
  mnl_attr_nest_end(nlh, nest);

  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, NULL, NULL);
}

int ualoe_nl_set_config_phase(ualoe_handle_t handle, ualoe_config_phase_e next_phase) {
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc = MNL_CB_OK;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_SET_CONFIG_PHASE, nl_handle->family_id,
                    1);

  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);
  mnl_attr_put_u32(nlh, CFG_ATTR_CONFIG_PHASE, next_phase);

  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, NULL, NULL);
}

static int ualoe_nl_parse_get_current_config_phase_nested(const struct nlattr* attr, void* arg) {
  ualoe_config_phase_e* phase = arg;

  switch (mnl_attr_get_type(attr)) {
    case CFG_ATTR_IFOE_CONFIG_CURR_PHASE:
      *phase = mnl_attr_get_u32(attr);
      break;
    default:
      errno = EINVAL;
      return MNL_CB_ERROR;
  }
  return MNL_CB_OK;
}

static int ualoe_nl_parse_get_current_config_phase(const struct nlattr* attr, void* arg) {
  switch (mnl_attr_get_type(attr)) {
    case CFG_ATTR_CONFIG_PHASE:
      return mnl_attr_parse_nested(attr, ualoe_nl_parse_get_current_config_phase_nested, arg);
    default:
      return MNL_CB_OK;
  }
}

static int ualoe_nl_get_current_config_phase_handler(const struct nlmsghdr* nlh, void* arg) {
  return mnl_attr_parse(nlh, sizeof(struct genlmsghdr), ualoe_nl_parse_get_current_config_phase,
                        arg);
}

int ualoe_nl_get_current_config_phase(ualoe_handle_t handle, ualoe_config_phase_e* phase) {
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_GET_CURRENT_CONFIG_PHASE,
                    nl_handle->family_id, 1);

  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);

  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh,
                            ualoe_nl_get_current_config_phase_handler, phase);
}

int ualoe_nl_set_ifoe_config(ualoe_handle_t handle, ifoe_virt_mode_e virt_mode,
                             ifoe_encap_type_e encap_type, ifoe_failover_mode_e failover_mode,
                             ifoe_loopback_mode_e loopback_mode) {
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_SET_IFOE_CONFIG, nl_handle->family_id, 1);

  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);

  mnl_attr_put_u32(nlh, CFG_ATTR_IFOE_CONFIG_VIRT_MODE, virt_mode);
  mnl_attr_put_u32(nlh, CFG_ATTR_IFOE_CONFIG_ENCAP_MODE, encap_type);
  mnl_attr_put_u32(nlh, CFG_ATTR_IFOE_CONFIG_FAILOVER_MODE, failover_mode);
  mnl_attr_put_u32(nlh, CFG_ATTR_IFOE_CONFIG_LOOPBACK_MODE, loopback_mode);

  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, NULL, NULL);
}

struct ualoe_nl_ifoe_cfg {
  ifoe_config_t* config;
  uint32_t* active_accelerator_bitmask;
  uint32_t* local_accelerator_bitmask;
  uint32_t* enabled_accelerator_bitmask;
  uint32_t bitmask_size;
};

/*
 * Copy bitmask payload from netlink attribute to destination buffer,
 * truncating if the payload exceeds the requested size.
 *
 * Logs a warning if truncated bits are non-zero (information loss).
 *
 * @param attr Netlink attribute containing the bitmask payload
 * @param dest Destination buffer for the bitmask
 * @param dest_size Size of the destination buffer in bytes
 * @param name Name of the bitmask for logging purposes
 */
static void copy_bitmask_with_validation(const struct nlattr* attr, void* dest, unsigned dest_size,
                                         const char* name) {
  unsigned payload_size = mnl_attr_get_payload_len(attr);
  const uint8_t* payload = mnl_attr_get_payload(attr);

  if (payload_size > dest_size) {
    /* Check if truncated bits contain non-zero data */
    unsigned i;
    bool has_nonzero = false;

    for (i = dest_size; i < payload_size; i++) {
      if (payload[i] != 0) {
        has_nonzero = true;
        break;
      }
    }

    if (has_nonzero) {
      ualoe_log_warning(
          "Warning: %s payload length (%u) exceeds requested bitmask size (%u) and truncated bits "
          "are non-zero\n",
          name, payload_size, dest_size);
    }

    payload_size = dest_size;
  }

  memcpy(dest, payload, payload_size);

  if (payload_size < dest_size) memset(dest + payload_size, 0, dest_size - payload_size);
}

static int ualoe_nl_parse_ifoe_config(const struct nlattr* attr, void* arg) {
  struct ualoe_nl_ifoe_cfg* cfg = arg;
  ifoe_config_t* config = cfg->config;

  switch (mnl_attr_get_type(attr)) {
    case CFG_ATTR_IFOE_CONFIG_FLAGS:
      config->configured_flags = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_IFOE_CONFIG_ACCEL_ID:
      config->accelerator_id = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_IFOE_CONFIG_CURR_PHASE:
      config->current_phase = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_IFOE_CONFIG_VIRT_MODE:
      config->virt_mode = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_IFOE_CONFIG_ENCAP_MODE:
      config->encap_mode = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_IFOE_CONFIG_FAILOVER_MODE:
      config->failover_mode = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_IFOE_CONFIG_LOOPBACK_MODE:
      config->loopback_mode = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_IFOE_CONFIG_CRYPTO_MODE:
      config->crypto_mode = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_IFOE_CONFIG_ENABLED_ACCEL:
      if (cfg->bitmask_size > 0 && cfg->enabled_accelerator_bitmask) {
        copy_bitmask_with_validation(attr, cfg->enabled_accelerator_bitmask, cfg->bitmask_size,
                                     "enabled_accelerator_bitmask");
      }
      break;
    default:
      errno = EINVAL;
      return MNL_CB_ERROR;
  }
  return MNL_CB_OK;
}

static int ualoe_nl_parse_accel_config(const struct nlattr* attr, void* arg) {
  struct ualoe_nl_ifoe_cfg* cfg = arg;

  switch (mnl_attr_get_type(attr)) {
    case CFG_ATTR_ACCEL_CONFIG_BITMASK_SIZE:
      if (cfg->bitmask_size > 0) {
        if (mnl_attr_get_u32(attr) > cfg->bitmask_size) {
          ualoe_log_warning(
              "Warning: accel_config_bitmask_size attribute value (%u) exceeds requested bitmask "
              "size (%u)\n",
              mnl_attr_get_u32(attr), cfg->bitmask_size);
        }
      }
      break;
    case CFG_ATTR_ACCEL_CONFIG_ACTIVE_BITMASK:
      if (cfg->bitmask_size > 0 && cfg->active_accelerator_bitmask) {
        copy_bitmask_with_validation(attr, cfg->active_accelerator_bitmask, cfg->bitmask_size,
                                     "active_accelerator_bitmask");
      }
      break;
    case CFG_ATTR_ACCEL_CONFIG_LOCAL_BITMASK:
      if (cfg->bitmask_size > 0 && cfg->local_accelerator_bitmask) {
        copy_bitmask_with_validation(attr, cfg->local_accelerator_bitmask, cfg->bitmask_size,
                                     "local_accelerator_bitmask");
      }
      break;
    default:
      errno = EINVAL;
      return MNL_CB_ERROR;
  }
  return MNL_CB_OK;
}

static int ualoe_nl_parse_get_ifoe_config(const struct nlattr* attr, void* arg) {
  switch (mnl_attr_get_type(attr)) {
    case CFG_ATTR_IFOE_CONFIG:
      return mnl_attr_parse_nested(attr, ualoe_nl_parse_ifoe_config, arg);
    case CFG_ATTR_ACCEL_CONFIG:
      return mnl_attr_parse_nested(attr, ualoe_nl_parse_accel_config, arg);
    default:
      return MNL_CB_OK;
  }
}

static int ualoe_nl_get_ifoe_config_handler(const struct nlmsghdr* nlh, void* arg) {
  return mnl_attr_parse(nlh, sizeof(struct genlmsghdr), ualoe_nl_parse_get_ifoe_config, arg);
}

int ualoe_nl_get_ifoe_config(ualoe_handle_t handle, ifoe_config_t* config, unsigned bitmask_size,
                             uint32_t active_accelerator_bitmask[],
                             uint32_t local_accelerator_bitmask[],
                             uint32_t enabled_accelerator_bitmask[]) {
  struct ualoe_nl_handle* nl_handle;
  struct ualoe_nl_ifoe_cfg cb_ctx;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_GET_IFOE_CONFIG, nl_handle->family_id, 1);

  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);

  cb_ctx = (struct ualoe_nl_ifoe_cfg){
      .config = config,
      .active_accelerator_bitmask = active_accelerator_bitmask,
      .local_accelerator_bitmask = local_accelerator_bitmask,
      .enabled_accelerator_bitmask = enabled_accelerator_bitmask,
      .bitmask_size = bitmask_size,
  };

  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, ualoe_nl_get_ifoe_config_handler,
                            &cb_ctx);
}

int ualoe_nl_enable_accelerators(ualoe_handle_t handle, unsigned bitmask_size,
                                 uint32_t enabled_accelerator_bitmask[]) {
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_SET_ENABLED_ACCELERATOR,
                    nl_handle->family_id, 1);

  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);

  mnl_attr_put_u32(nlh, CFG_ATTR_ENABLED_ACCEL_SIZE, bitmask_size);
  if (bitmask_size > 0 && enabled_accelerator_bitmask)
    mnl_attr_put(nlh, CFG_ATTR_ENABLED_ACCEL, bitmask_size, enabled_accelerator_bitmask);

  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, NULL, NULL);
}

int ualoe_nl_config_crypto(ualoe_handle_t handle, ualoe_crypto_mode_e mode) {
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_IFOE_CONFIG_CRYPTO, nl_handle->family_id,
                    1);
  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);
  mnl_attr_put_u32(nlh, CFG_ATTR_CRYPTO_MODE, mode);

  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, NULL, NULL);
}

int ualoe_nl_set_tx_crypto_key(ualoe_handle_t handle, ualoe_crypto_key_id_e key_id,
                               ualoe_crypto_key_t* key) {
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_IFOE_SET_TX_CRYPTO_KEY,
                    nl_handle->family_id, 1);

  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);
  mnl_attr_put_u32(nlh, CFG_ATTR_CRYPTO_KEY_ID, key_id);
  mnl_attr_put_u32(nlh, CFG_ATTR_CRYPTO_KEY_LEN, UALOE_CRYPTO_KEY_SIZE);
  mnl_attr_put(nlh, CFG_ATTR_CRYPTO_KEY, UALOE_CRYPTO_KEY_SIZE * sizeof(uint32_t), key);

  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, NULL, NULL);
}

int ualoe_nl_disable_rx_crypto_key(ualoe_handle_t handle, ualoe_crypto_key_id_e key_id) {
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_IFOE_DISABLE_RX_CRYPTO_KEY,
                    nl_handle->family_id, 1);

  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);
  mnl_attr_put_u32(nlh, CFG_ATTR_CRYPTO_KEY_ID, key_id);

  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, NULL, NULL);
}

int ualoe_nl_set_rx_crypto_key(ualoe_handle_t handle, ualoe_crypto_key_id_e key_id,
                               ualoe_crypto_key_t* key) {
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_IFOE_SET_RX_CRYPTO_KEY,
                    nl_handle->family_id, 1);

  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);
  mnl_attr_put_u32(nlh, CFG_ATTR_CRYPTO_KEY_ID, key_id);
  mnl_attr_put_u32(nlh, CFG_ATTR_CRYPTO_KEY_LEN, UALOE_CRYPTO_KEY_SIZE);
  mnl_attr_put(nlh, CFG_ATTR_CRYPTO_KEY, UALOE_CRYPTO_KEY_SIZE * sizeof(uint32_t), key);

  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, NULL, NULL);
}

static int ifoe_nl_parse_station_desc(const struct nlattr* attr, void* data) {
  ifoe_station_desc_t* desc = data;

  if (desc == NULL) {
    errno = ERANGE;
    return MNL_CB_ERROR;
  }

  switch (mnl_attr_get_type(attr)) {
    case CFG_ATTR_STATION_LABEL:
      if (mnl_attr_validate(attr, MNL_TYPE_STRING) < 0) return MNL_CB_ERROR;
      snprintf(desc->name.text, UALOE_LABEL_SIZE, "%s", (char*)mnl_attr_get_str(attr));
      break;
    case CFG_ATTR_STATION_LOGICAL_IDX:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      desc->logical_idx = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_STATION_PHYSICAL_IDX:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      desc->physical_idx = mnl_attr_get_u32(attr);
      break;
    default:
      errno = EINVAL;
      return MNL_CB_ERROR;
  }
  return MNL_CB_OK;
}

struct ifoe_nl_station_list_ctx {
  ifoe_station_desc_t* descs;
  unsigned desc_count;
  unsigned received;
};

static int ifoe_nl_get_station_list_handler(const struct nlmsghdr* nlh, void* data) {
  struct ifoe_nl_station_list_ctx* ctx = data;
  ifoe_station_desc_t* desc = ctx->received < ctx->desc_count ? &ctx->descs[ctx->received] : NULL;
  int rc;

  rc = mnl_attr_parse(nlh, sizeof(struct genlmsghdr), ifoe_nl_parse_station_desc, desc);
  if (rc > MNL_CB_STOP) ctx->received++;
  return rc;
}

int ifoe_nl_get_station_list(ualoe_handle_t handle, unsigned desc_count,
                             ifoe_station_desc_t descs[]) {
  struct ifoe_nl_station_list_ctx ctx = {
      .descs = descs,
      .desc_count = desc_count,
  };
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_GET_STATION_LIST, nl_handle->family_id,
                    1);
  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);
  mnl_attr_put_u32(nlh, CFG_ATTR_DESC_COUNT, desc_count);

  rc = ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, ifoe_nl_get_station_list_handler, &ctx);
  if (rc) return rc;
  return ctx.received == desc_count ? 0 : ERANGE;
}

int ifoe_nl_station_ctrl(ualoe_handle_t handle, unsigned station_idx, ifoe_station_state_e state) {
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_STATION_CTRL, nl_handle->family_id, 1);
  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);
  mnl_attr_put_u32(nlh, CFG_ATTR_STATION_LOGICAL_IDX, station_idx);
  mnl_attr_put_u32(nlh, CFG_ATTR_STATION_STATE, state);
  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, NULL, NULL);
}

struct ualoe_nl_netport_cb_ctx {
  unsigned logical_idx;
  bool fault;
  unsigned streams_failover;
  unsigned streams_paused;
};

static int ifoe_nl_parse_station_netport_attr(const struct nlattr* attr, void* data) {
  struct ualoe_nl_netport_cb_ctx* netport = data;

  switch (mnl_attr_get_type(attr)) {
    case CFG_ATTR_NETPORT_STATE_IDX:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      netport->logical_idx = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_NETPORT_STATE_FAULT:
      if (mnl_attr_validate(attr, MNL_TYPE_U8) < 0) return MNL_CB_ERROR;
      netport->fault = mnl_attr_get_u8(attr);
      break;
    case CFG_ATTR_NETPORT_STATE_STREAMS_FAILOVER:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      netport->streams_failover = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_NETPORT_STATE_STREAMS_PAUSED:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      netport->streams_paused = mnl_attr_get_u32(attr);
      break;
    default:
      errno = EINVAL;
      return MNL_CB_ERROR;
  }
  return MNL_CB_OK;
}

static int ifoe_nl_parse_station_netport(const struct nlattr* attr, ifoe_station_state_t* state,
                                         unsigned netport_idx) {
  struct ualoe_nl_netport_cb_ctx netport;
  int rc;

  if (netport_idx >= UALOE_MAX_NETPORTS_PER_IFOE_STATION || netport_idx >= state->netport_count) {
    errno = ERANGE;
    return MNL_CB_ERROR;
  }

  rc = mnl_attr_parse_nested(attr, ifoe_nl_parse_station_netport_attr, &netport);
  if (rc != MNL_CB_OK) return rc;

  state->netports[netport_idx].logical_idx = netport.logical_idx;
  state->netports[netport_idx].fault = netport.fault;
  state->netports[netport_idx].streams_failover = netport.streams_failover;
  state->netports[netport_idx].streams_paused = netport.streams_paused;
  return MNL_CB_OK;
}

static int ifoe_nl_parse_station_state(const struct nlattr* attr, void* data) {
  ifoe_station_state_t* state = data;
  int rc;

  switch (mnl_attr_get_type(attr)) {
    case CFG_ATTR_STATION_STATE:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      state->state = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_STATION_LINK_DOWN:
      if (mnl_attr_validate(attr, MNL_TYPE_U8) < 0) return MNL_CB_ERROR;
      state->link_down = mnl_attr_get_u8(attr);
      break;
    case CFG_ATTR_STATION_DX_ISOLATED:
      if (mnl_attr_validate(attr, MNL_TYPE_U8) < 0) return MNL_CB_ERROR;
      state->dx_isolated = mnl_attr_get_u8(attr);
      break;
    case CFG_ATTR_BANDWIDTH:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      state->bandwidth = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_STATION_LOGICAL_IDX:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      state->logical_idx = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_STATION_PHYSICAL_IDX:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      state->physical_idx = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_NETPORT_COUNT:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      state->netport_count = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_NETPORT0:
      if (mnl_attr_validate(attr, MNL_TYPE_NESTED) < 0) return MNL_CB_ERROR;
      rc = ifoe_nl_parse_station_netport(attr, state, 0);
      if (rc != MNL_CB_OK) return rc;
      break;
    case CFG_ATTR_NETPORT1:
      if (mnl_attr_validate(attr, MNL_TYPE_NESTED) < 0) return MNL_CB_ERROR;
      rc = ifoe_nl_parse_station_netport(attr, state, 1);
      if (rc != MNL_CB_OK) return rc;
      break;
    case CFG_ATTR_NETPORT2:
      if (mnl_attr_validate(attr, MNL_TYPE_NESTED) < 0) return MNL_CB_ERROR;
      rc = ifoe_nl_parse_station_netport(attr, state, 2);
      if (rc != MNL_CB_OK) return rc;
      break;
    case CFG_ATTR_NETPORT3:
      if (mnl_attr_validate(attr, MNL_TYPE_NESTED) < 0) return MNL_CB_ERROR;
      rc = ifoe_nl_parse_station_netport(attr, state, 3);
      if (rc != MNL_CB_OK) return rc;
      break;
    default:
      errno = EINVAL;
      return MNL_CB_ERROR;
  }
  return MNL_CB_OK;
}

static int ifoe_nl_station_get_state_handler(const struct nlmsghdr* nlh, void* data) {
  return mnl_attr_parse(nlh, sizeof(struct genlmsghdr), ifoe_nl_parse_station_state, data);
}

int ifoe_nl_station_get_state(ualoe_handle_t handle, unsigned station_idx,
                              ifoe_station_state_t* state) {
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_STATION_GET_STATE, nl_handle->family_id,
                    1);
  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);
  mnl_attr_put_u32(nlh, CFG_ATTR_STATION_LOGICAL_IDX, station_idx);
  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, ifoe_nl_station_get_state_handler,
                            state);
}

int ifoe_nl_set_path_to_port_map(ualoe_handle_t handle, bool specify_station,
                                 bool specify_accelerator, bool reenable_streams,
                                 unsigned station_idx, unsigned accelerator_id, unsigned path_count,
                                 unsigned map[]) {
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  if (path_count == 0) {
    ualoe_log_error("%s: path_count cannot be zero\n", __func__);
    return EINVAL;
  }

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_SET_PATH_PORT_MAP, nl_handle->family_id,
                    1);
  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);
  mnl_attr_put_u32(nlh, CFG_ATTR_DESC_COUNT, path_count);
  mnl_attr_put_u8(nlh, CFG_ATTR_REENABLE_STREAMS, reenable_streams);
  mnl_attr_put(nlh, CFG_ATTR_PATH_TO_PORT_MAP, path_count * sizeof(unsigned), map);

  if (specify_station) mnl_attr_put_u32(nlh, CFG_ATTR_STATION_LOGICAL_IDX, station_idx);
  if (specify_accelerator) mnl_attr_put_u32(nlh, CFG_ATTR_ACCELERATOR_ID, accelerator_id);

  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, NULL, NULL);
}

struct ifoe_nl_path_to_port_map_cb_ctx {
  unsigned path_count;
  unsigned* map;
};

static int ifoe_nl_get_path_to_port_map_parse(const struct nlattr* attr, void* data) {
  struct ifoe_nl_path_to_port_map_cb_ctx* cb_ctx = data;

  switch (mnl_attr_get_type(attr)) {
    case CFG_ATTR_PATH_TO_PORT_MAP:
      if (mnl_attr_get_payload_len(attr) != cb_ctx->path_count * sizeof(unsigned)) {
        errno = EINVAL;
        return MNL_CB_ERROR;
      }
      memcpy(cb_ctx->map, mnl_attr_get_payload(attr), cb_ctx->path_count * sizeof(unsigned));
      break;
    case CFG_ATTR_DESC_COUNT:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      if (mnl_attr_get_u32(attr) != cb_ctx->path_count) {
        errno = ERANGE;
        return MNL_CB_ERROR;
      }
      break;
    default:
      errno = EINVAL;
      return MNL_CB_ERROR;
  }
  return MNL_CB_OK;
}

static int ifoe_nl_get_path_to_port_map_cb(const struct nlmsghdr* nlh, void* data) {
  return mnl_attr_parse(nlh, sizeof(struct genlmsghdr), ifoe_nl_get_path_to_port_map_parse, data);
}

int ifoe_nl_get_path_to_port_map(ualoe_handle_t handle, unsigned station_idx,
                                 unsigned accelerator_id, unsigned path_count, unsigned map[]) {
  struct ifoe_nl_path_to_port_map_cb_ctx cb_ctx;
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_GET_PATH_PORT_MAP, nl_handle->family_id,
                    1);
  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);
  mnl_attr_put_u32(nlh, CFG_ATTR_STATION_LOGICAL_IDX, station_idx);
  mnl_attr_put_u32(nlh, CFG_ATTR_ACCELERATOR_ID, accelerator_id);

  cb_ctx = (struct ifoe_nl_path_to_port_map_cb_ctx){
      .path_count = path_count,
      .map = map,
  };

  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, ifoe_nl_get_path_to_port_map_cb,
                            &cb_ctx);
}

static int ifoe_nl_parse_netport_properties(const struct nlattr* attr, void* data) {
  ualoe_netport_properties_t* properties = data;

  switch (mnl_attr_get_type(attr)) {
    case CFG_ATTR_NETPORT_ETH_TECH_MASK:
      if (mnl_attr_get_payload_len(attr) != sizeof(properties->eth_tech_mask)) {
        errno = EINVAL;
        return MNL_CB_ERROR;
      }
      memcpy(&properties->eth_tech_mask, mnl_attr_get_payload(attr),
             sizeof(properties->eth_tech_mask));
      break;
    case CFG_ATTR_NETPORT_FEC_MODE:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      properties->fec_modes = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_NETPORT_NUM_LANES:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      properties->num_lanes = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_NETPORT_LOOPBACK_MODE:
      if (mnl_attr_validate(attr, MNL_TYPE_U64) < 0) return MNL_CB_ERROR;
      properties->loopback_modes = mnl_attr_get_u64(attr);
      break;
    case CFG_ATTR_NETPORT_MAX_FRAME_LEN:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      properties->max_frame_len = mnl_attr_get_u32(attr);
      break;
    default:
      errno = EINVAL;
      return MNL_CB_ERROR;
  }
  return MNL_CB_OK;
}

static int ifoe_nl_get_netport_properties_handler(const struct nlmsghdr* nlh, void* data) {
  return mnl_attr_parse(nlh, sizeof(struct genlmsghdr), ifoe_nl_parse_netport_properties, data);
}

int ifoe_nl_get_netport_properties(ualoe_handle_t handle, ualoe_netport_properties_t* properties) {
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_NETPORT_GET_PROPERTIES,
                    nl_handle->family_id, 1);
  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);

  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh,
                            ifoe_nl_get_netport_properties_handler, properties);
}

static int ifoe_nl_parse_netport_desc(const struct nlattr* attr, void* data) {
  ifoe_netport_desc_t* desc = data;

  if (desc == NULL) {
    errno = ERANGE;
    return MNL_CB_ERROR;
  }

  switch (mnl_attr_get_type(attr)) {
    case CFG_ATTR_NETPORT_LOGICAL_IDX:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      desc->logical_idx = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_NETPORT_REL_IDX:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      desc->station_rel_netport_idx = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_NETPORT_LABEL:
      if (mnl_attr_validate(attr, MNL_TYPE_STRING) < 0) return MNL_CB_ERROR;
      snprintf(desc->name.text, UALOE_LABEL_SIZE, "%s", (char*)mnl_attr_get_str(attr));
      break;
    case CFG_ATTR_STATION_LOGICAL_IDX:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      desc->station_idx = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_STATION_LABEL:
      if (mnl_attr_validate(attr, MNL_TYPE_STRING) < 0) return MNL_CB_ERROR;
      snprintf(desc->station_name.text, UALOE_LABEL_SIZE, "%s", (char*)mnl_attr_get_str(attr));
      break;
    default:
      errno = EINVAL;
      return MNL_CB_ERROR;
  }
  return MNL_CB_OK;
}

struct ifoe_nl_netport_list_ctx {
  ifoe_netport_desc_t* descs;
  unsigned desc_count;
  unsigned received;
};

static int ifoe_nl_get_netport_list_handler(const struct nlmsghdr* nlh, void* data) {
  struct ifoe_nl_netport_list_ctx* ctx = data;
  ifoe_netport_desc_t* desc = ctx->received < ctx->desc_count ? &ctx->descs[ctx->received] : NULL;
  int rc;

  rc = mnl_attr_parse(nlh, sizeof(struct genlmsghdr), ifoe_nl_parse_netport_desc, desc);
  if (rc > MNL_CB_STOP) ctx->received++;
  return rc;
}
int ifoe_nl_get_netport_list(ualoe_handle_t handle, unsigned desc_count,
                             ifoe_netport_desc_t descs[]) {
  struct ifoe_nl_netport_list_ctx ctx = {
      .descs = descs,
      .desc_count = desc_count,
  };
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_GET_NETPORT_LIST, nl_handle->family_id,
                    1);
  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);
  mnl_attr_put_u32(nlh, CFG_ATTR_DESC_COUNT, desc_count);

  rc = ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, ifoe_nl_get_netport_list_handler, &ctx);
  if (rc == ENOSPC) /* Preserve the legacy v1 behavior when descs is too small. */
    return EINVAL;
  if (rc) return rc;
  return ctx.received == desc_count ? 0 : ERANGE;
}

/** Context for parsing netport list responses which include the
 *  total netport count alongside each descriptor.
 */
struct netport_list_v2_ctx {
  ifoe_netport_desc_t* descs;
  ifoe_netport_desc_t* desc;
  unsigned max_count;
  unsigned received;
  unsigned total_count;
};

static int ifoe_nl_parse_netport_desc_v2(const struct nlattr* attr, void* data) {
  struct netport_list_v2_ctx* ctx = data;

  switch (mnl_attr_get_type(attr)) {
    case CFG_ATTR_NETPORT_LOGICAL_IDX:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      if (ctx->desc) ctx->desc->logical_idx = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_NETPORT_REL_IDX:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      if (ctx->desc) ctx->desc->station_rel_netport_idx = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_NETPORT_LABEL:
      if (mnl_attr_validate(attr, MNL_TYPE_STRING) < 0) return MNL_CB_ERROR;
      if (ctx->desc)
        snprintf(ctx->desc->name.text, UALOE_LABEL_SIZE, "%s", (char*)mnl_attr_get_str(attr));
      break;
    case CFG_ATTR_STATION_LOGICAL_IDX:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      if (ctx->desc) ctx->desc->station_idx = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_STATION_LABEL:
      if (mnl_attr_validate(attr, MNL_TYPE_STRING) < 0) return MNL_CB_ERROR;
      if (ctx->desc)
        snprintf(ctx->desc->station_name.text, UALOE_LABEL_SIZE, "%s",
                 (char*)mnl_attr_get_str(attr));
      break;
    case CFG_ATTR_NETPORT_TOTAL_COUNT:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      ctx->total_count = mnl_attr_get_u32(attr);
      break;
    default:
      break;
  }
  return MNL_CB_OK;
}

static int ifoe_nl_get_netport_list_v2_handler(const struct nlmsghdr* nlh, void* data) {
  struct netport_list_v2_ctx* ctx = data;
  int rc;

  ctx->desc = ctx->received < ctx->max_count ? &ctx->descs[ctx->received] : NULL;
  rc = mnl_attr_parse(nlh, sizeof(struct genlmsghdr), ifoe_nl_parse_netport_desc_v2, ctx);
  if (rc > MNL_CB_STOP) ctx->received++;
  return rc;
}

int ifoe_nl_get_netport_list_v2(ualoe_handle_t handle, unsigned* desc_count,
                                ifoe_netport_desc_t descs[]) {
  struct netport_list_v2_ctx ctx = {
      .descs = descs,
      .max_count = *desc_count,
  };
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  /* Omit CFG_ATTR_DESC_COUNT so the driver returns all configured
   * netports and includes CFG_ATTR_NETPORT_TOTAL_COUNT in responses.
   */
  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_GET_NETPORT_LIST, nl_handle->family_id,
                    1);
  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);

  rc = ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, ifoe_nl_get_netport_list_v2_handler,
                          &ctx);
  if (rc) return rc;

  if (ctx.total_count && ctx.received != ctx.total_count) {
    ualoe_log_error("Expected %u netports but received %u\n", ctx.total_count, ctx.received);
    return EIO;
  }

  *desc_count = ctx.received;

  return ctx.received > ctx.max_count ? ENOSPC : 0;
}

int ifoe_nl_netport_ctrl(ualoe_handle_t handle, unsigned netport_idx, ifoe_netport_state_e state) {
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_NETPORT_CTRL, nl_handle->family_id, 1);
  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);
  mnl_attr_put_u32(nlh, CFG_ATTR_NETPORT_LOGICAL_IDX, netport_idx);
  mnl_attr_put_u32(nlh, CFG_ATTR_NETPORT_STATE, state);
  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, NULL, NULL);
}

int ifoe_nl_netport_config_link_auto(ualoe_handle_t handle, unsigned netport_idx,
                                     bool parallel_detect_enable, __uint128_t advertised_eth_techs,
                                     ualoe_fec_mode_e requested_fec_mode) {
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_NETPORT_CONFIG_LINK_AUTO,
                    nl_handle->family_id, 1);
  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);
  mnl_attr_put_u32(nlh, CFG_ATTR_NETPORT_LOGICAL_IDX, netport_idx);
  mnl_attr_put_u8(nlh, CFG_ATTR_NETPORT_PARALLEL_DETECT_ENABLED, parallel_detect_enable);
  mnl_attr_put_u8(nlh, CFG_ATTR_NETPORT_FEC_MODE, requested_fec_mode);
  if (!mnl_attr_put_check(nlh, sizeof(buf), CFG_ATTR_NETPORT_ETH_TECH_MASK,
                          sizeof(advertised_eth_techs), &advertised_eth_techs)) {
    ualoe_log_error("%s: Failed to put advertised_eth_techs to buffer\n", __func__);
    return ENOMEM;
  }

  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, NULL, NULL);
}

int ifoe_nl_netport_config_link_manual(ualoe_handle_t handle, unsigned netport_idx,
                                       ualoe_eth_tech_e eth_tech, ualoe_fec_mode_e fec_mode,
                                       ualoe_netport_loopback_mode_e loopback_mode) {
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_NETPORT_CONFIG_LINK_MANUAL,
                    nl_handle->family_id, 1);
  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);
  mnl_attr_put_u32(nlh, CFG_ATTR_NETPORT_LOGICAL_IDX, netport_idx);
  mnl_attr_put_u16(nlh, CFG_ATTR_NETPORT_LINK_TECHNOLOGY, eth_tech);
  mnl_attr_put_u8(nlh, CFG_ATTR_NETPORT_FEC_MODE, fec_mode);
  mnl_attr_put_u8(nlh, CFG_ATTR_NETPORT_LOOPBACK_MODE, loopback_mode);

  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, NULL, NULL);
}

_Static_assert((sizeof(ifoe_accelerator_addr_map_t) ==
                sizeof(struct cfg_ifoe_accelerator_addr_map)),
               "mac_addr_map types are incompatible; explicit copying is required");

#ifndef UALOE_MAX_ACCELERATORS
/* FIXME: IFOESW-804: The value 1024 is a temporary placeholder for the maximum
 * number of accelerators. Update this value to match hardware or system
 * constraints when available.
 */
#define UALOE_MAX_ACCELERATORS 1024
#endif

/* Number of uint32_t attributes in the netlink message header for
 * ifoe_nl_netport_set_accelerator_addr_map. This includes:
 *   - CFG_ATTR_DEV_ID
 *   - CFG_ATTR_NETPORT_LOGICAL_IDX
 *   - CFG_ATTR_DESC_COUNT
 *   - (other attributes as required by the protocol, up to 25 total)
 * Update this value if the message structure changes.
 */
#define NETPORT_SET_ACCELERATOR_ADDR_MAP_U32_ATTRS 25

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define NETPORT_ACCELERATOR_ADDR_MAP_BUF_SIZE                                                     \
  MAX(MNL_SOCKET_BUFFER_SIZE, MNL_NLMSG_HDRLEN +                                                  \
                                  NETPORT_SET_ACCELERATOR_ADDR_MAP_U32_ATTRS * sizeof(uint32_t) + \
                                  UALOE_MAX_ACCELERATORS * sizeof(ifoe_accelerator_addr_map_t))

static bool convert_network_addr_type_ifoe_to_cfg(ifoe_network_addr_type_e from,
                                                  enum cfg_network_addr_type* to) {
  switch (from) {
    case IFOE_NETWORK_ADDR_TYPE_MAC:
      *to = CFG_NETWORK_ADDR_TYPE_MAC;
      return true;
    case IFOE_NETWORK_ADDR_TYPE_IP:
      *to = CFG_NETWORK_ADDR_TYPE_IP;
      return true;
    case IFOE_NETWORK_ADDR_TYPE_MAC_IP:
      *to = CFG_NETWORK_ADDR_TYPE_MAC_IP;
      return true;
  }
  return false;
}

int ifoe_nl_netport_set_accelerator_addr_map(ualoe_handle_t handle, unsigned netport_idx,
                                             ifoe_network_addr_type_e map_addr_type,
                                             unsigned map_count,
                                             ifoe_accelerator_addr_map_t map[]) {
  struct ualoe_nl_handle* nl_handle;
  enum cfg_network_addr_type cfg_map_addr_type;
  char buf[NETPORT_ACCELERATOR_ADDR_MAP_BUF_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  if (!convert_network_addr_type_ifoe_to_cfg(map_addr_type, &cfg_map_addr_type)) {
    ualoe_log_error("%s: Invalid address type: %d\n", __func__, map_addr_type);
    return EINVAL;
  }

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_NETPORT_SET_ACCELERATOR_ADDR_MAP,
                    nl_handle->family_id, 1);
  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);
  mnl_attr_put_u32(nlh, CFG_ATTR_NETPORT_LOGICAL_IDX, netport_idx);
  mnl_attr_put_u32(nlh, CFG_ATTR_NETPORT_IFOE_ADDR_TYPE, cfg_map_addr_type);
  mnl_attr_put_u32(nlh, CFG_ATTR_DESC_COUNT, map_count);
  if (!mnl_attr_put_check(nlh, sizeof(buf), CFG_ATTR_NETPORT_ACCELERATOR_ADDR_MAP,
                          map_count * sizeof(ifoe_accelerator_addr_map_t), map)) {
    ualoe_log_error("%s: Failed to put map to buffer\n", __func__);
    return ENOMEM;
  }

  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, NULL, NULL);
}

struct get_accelerator_addr_map_cb_ctx {
  unsigned map_count;
  ifoe_accelerator_addr_map_t* map;
};

static int ifoe_nl_parse_netport_accel_addr_map(const struct nlattr* attr, void* data) {
  struct get_accelerator_addr_map_cb_ctx* cb_ctx = data;

  switch (mnl_attr_get_type(attr)) {
    case CFG_ATTR_NETPORT_ACCELERATOR_ADDR_MAP:
      if (mnl_attr_get_payload_len(attr) !=
          cb_ctx->map_count * sizeof(ifoe_accelerator_addr_map_t)) {
        errno = ERANGE;
        return MNL_CB_ERROR;
      }
      memcpy(cb_ctx->map, mnl_attr_get_payload(attr),
             cb_ctx->map_count * sizeof(ifoe_accelerator_addr_map_t));
      break;
    case CFG_ATTR_DESC_COUNT:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      if (mnl_attr_get_u32(attr) != cb_ctx->map_count) {
        errno = ERANGE;
        return MNL_CB_ERROR;
      }
      break;
    default:
      errno = EINVAL;
      return MNL_CB_ERROR;
  }
  return MNL_CB_OK;
}

static int ifoe_nl_get_accel_addr_map_cb(const struct nlmsghdr* nlh, void* data) {
  return mnl_attr_parse(nlh, sizeof(struct genlmsghdr), ifoe_nl_parse_netport_accel_addr_map, data);
}

int ifoe_nl_netport_get_accelerator_addr_map(ualoe_handle_t handle, unsigned netport_idx,
                                             unsigned map_count,
                                             ifoe_accelerator_addr_map_t map[]) {
  char buf[NETPORT_ACCELERATOR_ADDR_MAP_BUF_SIZE];
  struct ualoe_nl_handle* nl_handle;
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_NETPORT_GET_ACCELERATOR_ADDR_MAP,
                    nl_handle->family_id, 1);
  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);
  mnl_attr_put_u32(nlh, CFG_ATTR_NETPORT_LOGICAL_IDX, netport_idx);
  mnl_attr_put_u32(nlh, CFG_ATTR_DESC_COUNT, map_count);

  struct get_accelerator_addr_map_cb_ctx cb_ctx = {
      .map_count = map_count,
      .map = map,
  };

  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, ifoe_nl_get_accel_addr_map_cb,
                            &cb_ctx);
}

int ifoe_nl_netport_set_addr(ualoe_handle_t handle, unsigned netport_idx,
                             ifoe_network_addr_type_e addr_type, uint8_t mac_addr[],
                             uint32_t ip_addr) {
  struct ualoe_nl_handle* nl_handle;
  enum cfg_network_addr_type cfg_addr_type;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  if (!convert_network_addr_type_ifoe_to_cfg(addr_type, &cfg_addr_type)) {
    ualoe_log_error("%s: Invalid address type: %d\n", __func__, addr_type);
    return EINVAL;
  }

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_NETPORT_SET_ADDR, nl_handle->family_id,
                    1);
  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);
  mnl_attr_put_u32(nlh, CFG_ATTR_NETPORT_LOGICAL_IDX, netport_idx);
  mnl_attr_put_u32(nlh, CFG_ATTR_NETPORT_IFOE_ADDR_TYPE, cfg_addr_type);
  if (mac_addr) {
    mnl_attr_put(nlh, CFG_ATTR_NETPORT_IFOE_MAC_ADDR, UALOE_MAC_ADDRESS_SIZE, mac_addr);
  }
  mnl_attr_put_u32(nlh, CFG_ATTR_NETPORT_IFOE_IP_ADDR, ip_addr);
  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, NULL, NULL);
}

static int ifoe_nl_parse_netport_state(const struct nlattr* attr, void* data) {
  ifoe_netport_state_t* state = data;

  switch (mnl_attr_get_type(attr)) {
    case CFG_ATTR_NETPORT_STATE:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      state->state = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_NETPORT_AUTONEG_ENABLED:
      if (mnl_attr_validate(attr, MNL_TYPE_U8) < 0) return MNL_CB_ERROR;
      state->autoneg_enabled = mnl_attr_get_u8(attr);
      break;
    case CFG_ATTR_NETPORT_PARALLEL_DETECT_ENABLED:
      if (mnl_attr_validate(attr, MNL_TYPE_U8) < 0) return MNL_CB_ERROR;
      state->parallel_detect_enabled = mnl_attr_get_u8(attr);
      break;
    case CFG_ATTR_NETPORT_LINK_FLAGS:
      if (mnl_attr_validate(attr, MNL_TYPE_U64) < 0) return MNL_CB_ERROR;
      state->link_flags = mnl_attr_get_u64(attr);
      break;
    case CFG_ATTR_NETPORT_LINK_TECHNOLOGY:
      if (mnl_attr_validate(attr, MNL_TYPE_U16) < 0) return MNL_CB_ERROR;
      state->link_technology = mnl_attr_get_u16(attr);
      break;
    case CFG_ATTR_NETPORT_FEC_MODE:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      state->fec_mode = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_NETPORT_LOOPBACK_MODE:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      state->loopback_mode = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_NETPORT_IFOE_MAC_ADDR:
      memcpy(state->ifoe_mac_addr, mnl_attr_get_payload(attr), UALOE_MAC_ADDRESS_SIZE);
      break;
    case CFG_ATTR_NETPORT_PERM_ADDR:
      memcpy(state->permanent_mac_addr, mnl_attr_get_payload(attr), UALOE_MAC_ADDRESS_SIZE);
      break;
    default:
      errno = EINVAL;
      return MNL_CB_ERROR;
  }
  return MNL_CB_OK;
}

static int ualoe_nl_netport_state_handler(const struct nlmsghdr* nlh, void* data) {
  return mnl_attr_parse(nlh, sizeof(struct genlmsghdr), ifoe_nl_parse_netport_state, data);
}

int ifoe_nl_netport_get_state(ualoe_handle_t handle, unsigned netport_idx,
                              ifoe_netport_state_t* state) {
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_NETPORT_GET_STATE, nl_handle->family_id,
                    1);
  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);
  mnl_attr_put_u32(nlh, CFG_ATTR_NETPORT_LOGICAL_IDX, netport_idx);
  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, ualoe_nl_netport_state_handler,
                            state);
}

static int ifoe_nl_parse_ifcp_netport_state(const struct nlattr* attr, void* data) {
  ifoe_ifcp_netport_state_t* state = data;

  switch (mnl_attr_get_type(attr)) {
    case CFG_ATTR_IFCP_LOCAL_PORT_ID:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      state->local_port_id = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_IFCP_LOCAL_LINK_UP:
      if (mnl_attr_validate(attr, MNL_TYPE_U8) < 0) return MNL_CB_ERROR;
      state->local_link_up = mnl_attr_get_u8(attr);
      break;
    case CFG_ATTR_IFCP_LOCAL_IN_ERROR:
      if (mnl_attr_validate(attr, MNL_TYPE_U8) < 0) return MNL_CB_ERROR;
      state->local_in_error = mnl_attr_get_u8(attr);
      break;
    case CFG_ATTR_IFCP_LOCAL_LL_UP:
      if (mnl_attr_validate(attr, MNL_TYPE_U8) < 0) return MNL_CB_ERROR;
      state->local_ll_up = mnl_attr_get_u8(attr);
      break;
    case CFG_ATTR_IFCP_LOCAL_MAC_ADDR:
      if (mnl_attr_get_payload_len(attr) != UALOE_MAC_ADDRESS_SIZE) {
        errno = ERANGE;
        return MNL_CB_ERROR;
      }
      memcpy(state->local_mac_addr, mnl_attr_get_payload(attr), UALOE_MAC_ADDRESS_SIZE);
      break;
    case CFG_ATTR_IFCP_PEER_DEVICE_ID:
      if (mnl_attr_validate(attr, MNL_TYPE_U64) < 0) return MNL_CB_ERROR;
      state->peer_device_id = mnl_attr_get_u64(attr);
      break;
    case CFG_ATTR_IFCP_PEER_PORT_ID:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      state->peer_port_id = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_IFCP_PEER_ENABLED:
      if (mnl_attr_validate(attr, MNL_TYPE_U8) < 0) return MNL_CB_ERROR;
      state->peer_enabled = mnl_attr_get_u8(attr);
      break;
    case CFG_ATTR_IFCP_PEER_IN_ERROR:
      if (mnl_attr_validate(attr, MNL_TYPE_U8) < 0) return MNL_CB_ERROR;
      state->peer_in_error = mnl_attr_get_u8(attr);
      break;
    case CFG_ATTR_IFCP_PEER_ENCRYPT_MODE:
      if (mnl_attr_validate(attr, MNL_TYPE_U8) < 0) return MNL_CB_ERROR;
      state->peer_encrypt_mode = (ualoe_crypto_mode_e)mnl_attr_get_u8(attr);
      break;
    case CFG_ATTR_IFCP_PEER_MAC_ADDR:
      if (mnl_attr_get_payload_len(attr) != UALOE_MAC_ADDRESS_SIZE) {
        errno = ERANGE;
        return MNL_CB_ERROR;
      }
      memcpy(state->peer_mac_addr, mnl_attr_get_payload(attr), UALOE_MAC_ADDRESS_SIZE);
      break;
    case CFG_ATTR_IFCP_PEER_MTU:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      state->peer_mtu = mnl_attr_get_u32(attr);
      break;
    default:
      errno = EINVAL;
      return MNL_CB_ERROR;
  }
  return MNL_CB_OK;
}

static int ualoe_nl_ifcp_netport_state_handler(const struct nlmsghdr* nlh, void* data) {
  return mnl_attr_parse(nlh, sizeof(struct genlmsghdr), ifoe_nl_parse_ifcp_netport_state, data);
}

int ifoe_nl_ifcp_netport_get_state(ualoe_handle_t handle, unsigned netport_idx,
                                   ifoe_ifcp_netport_state_t* state) {
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_IFCP_GET_NETPORT_STATE,
                    nl_handle->family_id, 1);
  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);
  mnl_attr_put_u32(nlh, CFG_ATTR_NETPORT_LOGICAL_IDX, netport_idx);
  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, ualoe_nl_ifcp_netport_state_handler,
                            state);
}

static int ifoe_nl_parse_ifcp_netport_stats(const struct nlattr* attr, void* data) {
  ifoe_ifcp_netport_stats_t* stats = data;

  switch (mnl_attr_get_type(attr)) {
    case CFG_ATTR_IFCP_STATS_RX_COUNT:
      if (mnl_attr_validate(attr, MNL_TYPE_U64) < 0) return MNL_CB_ERROR;
      stats->rx_count = mnl_attr_get_u64(attr);
      break;
    case CFG_ATTR_IFCP_STATS_TX_COUNT:
      if (mnl_attr_validate(attr, MNL_TYPE_U64) < 0) return MNL_CB_ERROR;
      stats->tx_count = mnl_attr_get_u64(attr);
      break;
    case CFG_ATTR_IFCP_STATS_TX_ERRORS:
      if (mnl_attr_validate(attr, MNL_TYPE_U64) < 0) return MNL_CB_ERROR;
      stats->tx_errors = mnl_attr_get_u64(attr);
      break;
    case CFG_ATTR_IFCP_STATS_RX_DROPPED:
      if (mnl_attr_validate(attr, MNL_TYPE_U64) < 0) return MNL_CB_ERROR;
      stats->rx_dropped = mnl_attr_get_u64(attr);
      break;
    default:
      errno = EINVAL;
      return MNL_CB_ERROR;
  }
  return MNL_CB_OK;
}

static int ualoe_nl_ifcp_netport_stats_handler(const struct nlmsghdr* nlh, void* data) {
  return mnl_attr_parse(nlh, sizeof(struct genlmsghdr), ifoe_nl_parse_ifcp_netport_stats, data);
}

int ifoe_nl_ifcp_netport_get_stats(ualoe_handle_t handle, unsigned netport_idx,
                                   ifoe_ifcp_netport_stats_t* stats) {
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_IFCP_GET_NETPORT_STATS,
                    nl_handle->family_id, 1);
  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);
  mnl_attr_put_u32(nlh, CFG_ATTR_NETPORT_LOGICAL_IDX, netport_idx);
  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, ualoe_nl_ifcp_netport_stats_handler,
                            stats);
}

static int ualoe_nl_parse_scaleup_fabric_config(const struct nlattr* attr, void* arg) {
  ualoe_scaleup_fabric_config_t* config = arg;
  unsigned payload_len, count;

  switch (mnl_attr_get_type(attr)) {
    case CFG_ATTR_SCALEUP_PHYSICAL_POD_ID:
      if (mnl_attr_get_payload_len(attr) != UALOE_SCALEUP_FABRIC_POD_ID_SIZE) {
        errno = ERANGE;
        return MNL_CB_ERROR;
      }
      memcpy(config->physical_pod_id, mnl_attr_get_payload(attr), UALOE_SCALEUP_FABRIC_POD_ID_SIZE);
      break;
    case CFG_ATTR_SCALEUP_PHYSICAL_POD_SIZE:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      config->physical_pod_size = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_SCALEUP_LOCAL_ACCELERATORS:
      payload_len = mnl_attr_get_payload_len(attr);
      if (payload_len % sizeof(uint32_t) != 0) {
        errno = ERANGE;
        return MNL_CB_ERROR;
      }
      count = payload_len / sizeof(uint32_t);
      if (count > UALOE_SCALEUP_FABRIC_MAX_LOCAL_ACCELERATORS) {
        errno = ERANGE;
        return MNL_CB_ERROR;
      }
      memcpy(config->local_accelerators, mnl_attr_get_payload(attr), payload_len);
      config->num_local_accelerators = count;
      break;
    default:
      break;
  }
  return MNL_CB_OK;
}

static int ualoe_nl_get_scaleup_fabric_config_handler(const struct nlmsghdr* nlh, void* arg) {
  return mnl_attr_parse(nlh, sizeof(struct genlmsghdr), ualoe_nl_parse_scaleup_fabric_config, arg);
}

int ualoe_nl_get_scaleup_fabric_config(ualoe_handle_t handle,
                                       ualoe_scaleup_fabric_config_t* config) {
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_GET_SCALEUP_FABRIC_CONFIG,
                    nl_handle->family_id, 1);
  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);

  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh,
                            ualoe_nl_get_scaleup_fabric_config_handler, config);
}

int ualoe_nl_set_scaleup_fabric_config(ualoe_handle_t handle,
                                       const ualoe_scaleup_fabric_config_t* config) {
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_SET_SCALEUP_FABRIC_CONFIG,
                    nl_handle->family_id, 1);
  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);
  if (!mnl_attr_put_check(nlh, sizeof(buf), CFG_ATTR_SCALEUP_PHYSICAL_POD_ID,
                          sizeof(config->physical_pod_id), config->physical_pod_id)) {
    ualoe_log_error("%s: Failed to put physical_pod_id to buffer\n", __func__);
    return ENOMEM;
  }
  mnl_attr_put_u32(nlh, CFG_ATTR_SCALEUP_PHYSICAL_POD_SIZE, config->physical_pod_size);
  if (!mnl_attr_put_check(nlh, sizeof(buf), CFG_ATTR_SCALEUP_LOCAL_ACCELERATORS,
                          config->num_local_accelerators * sizeof(uint32_t),
                          config->local_accelerators)) {
    ualoe_log_error("%s: Failed to put local_accelerators to buffer\n", __func__);
    return ENOMEM;
  }

  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, NULL, NULL);
}

int ualoe_nl_set_scaleup_fabric_vpod_config(ualoe_handle_t handle,
                                            const ualoe_scaleup_fabric_vpod_config_t* config) {
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_SET_SCALEUP_FABRIC_VPOD_CONFIG,
                    nl_handle->family_id, 1);
  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);
  mnl_attr_put_u32(nlh, CFG_ATTR_SCALEUP_VIRTUAL_POD_ID, config->virtual_pod_id);
  mnl_attr_put_u32(nlh, CFG_ATTR_SCALEUP_NPA_ADDRESS_MODE, config->npa_address_mode);

  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, NULL, NULL);
}

int ualoe_nl_set_scaleup_fabric_station_info(ualoe_handle_t handle,
                                             const ualoe_scaleup_fabric_station_info_t* info) {
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_SET_SCALEUP_FABRIC_STATION_INFO,
                    nl_handle->family_id, 1);
  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);
  mnl_attr_put_u32(nlh, CFG_ATTR_SCALEUP_STATION_COUNT, info->num_stations);
  if (!mnl_attr_put_check(nlh, sizeof(buf), CFG_ATTR_SCALEUP_STATION_BANDWIDTH, info->num_stations,
                          info->station_bandwidth)) {
    ualoe_log_error("%s: Failed to put station_bandwidth to buffer\n", __func__);
    return ENOMEM;
  }

  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, NULL, NULL);
}

static int ualoe_nl_parse_scaleup_fabric_vpod_config(const struct nlattr* attr, void* arg) {
  ualoe_scaleup_fabric_vpod_config_t* config = arg;

  switch (mnl_attr_get_type(attr)) {
    case CFG_ATTR_SCALEUP_VIRTUAL_POD_ID:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      config->virtual_pod_id = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_SCALEUP_NPA_ADDRESS_MODE:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      config->npa_address_mode = mnl_attr_get_u32(attr);
      break;
    default:
      break;
  }
  return MNL_CB_OK;
}

static int ualoe_nl_get_scaleup_fabric_vpod_config_handler(const struct nlmsghdr* nlh, void* arg) {
  return mnl_attr_parse(nlh, sizeof(struct genlmsghdr), ualoe_nl_parse_scaleup_fabric_vpod_config,
                        arg);
}

int ualoe_nl_get_scaleup_fabric_vpod_config(ualoe_handle_t handle,
                                            ualoe_scaleup_fabric_vpod_config_t* config) {
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_GET_SCALEUP_FABRIC_VPOD_CONFIG,
                    nl_handle->family_id, 1);
  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);

  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh,
                            ualoe_nl_get_scaleup_fabric_vpod_config_handler, config);
}

struct ualoe_nl_scaleup_station_ctx {
  ualoe_scaleup_fabric_station_info_t* info;
  uint32_t reported_count;
  unsigned bandwidth_len;
};

static int ualoe_nl_parse_scaleup_fabric_station_info(const struct nlattr* attr, void* arg) {
  struct ualoe_nl_scaleup_station_ctx* ctx = arg;

  switch (mnl_attr_get_type(attr)) {
    case CFG_ATTR_SCALEUP_STATION_COUNT:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      ctx->reported_count = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_SCALEUP_STATION_BANDWIDTH:
      ctx->bandwidth_len = mnl_attr_get_payload_len(attr);
      if (ctx->bandwidth_len > sizeof(ctx->info->station_bandwidth)) {
        errno = ERANGE;
        return MNL_CB_ERROR;
      }
      memcpy(ctx->info->station_bandwidth, mnl_attr_get_payload(attr), ctx->bandwidth_len);
      break;
    default:
      break;
  }
  return MNL_CB_OK;
}

static int ualoe_nl_get_scaleup_fabric_station_info_handler(const struct nlmsghdr* nlh, void* arg) {
  struct ualoe_nl_scaleup_station_ctx* ctx = arg;

  return mnl_attr_parse(nlh, sizeof(struct genlmsghdr), ualoe_nl_parse_scaleup_fabric_station_info,
                        ctx);
}

int ualoe_nl_get_scaleup_fabric_station_info(ualoe_handle_t handle,
                                             ualoe_scaleup_fabric_station_info_t* info) {
  struct ualoe_nl_scaleup_station_ctx ctx = {
      .info = info,
      .reported_count = 0,
      .bandwidth_len = 0,
  };
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_GET_SCALEUP_FABRIC_STATION_INFO,
                    nl_handle->family_id, 1);
  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);

  rc = ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh,
                          ualoe_nl_get_scaleup_fabric_station_info_handler, &ctx);
  if (rc) return rc;

  /* Enforce the invariant that the reported station count equals the
   * bandwidth payload length in bytes (one byte per station); a response
   * that violates it is rejected. An all-zero result (no stations) is a
   * valid empty set.
   */
  if (ctx.reported_count != ctx.bandwidth_len) return EINVAL;

  info->num_stations = ctx.reported_count;
  return 0;
}

static int ualoe_nl_parse_gpu_identity(const struct nlattr* attr, void* arg) {
  ualoe_gpu_identity_t* identity = arg;

  switch (mnl_attr_get_type(attr)) {
    case CFG_ATTR_GPU_PHYS_ID:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      identity->phys_id = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_GPU_NUM_GPUS:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      identity->num_gpus = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_GPU_TRAY_TYPE:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      identity->tray_type = mnl_attr_get_u32(attr);
      break;
    case CFG_ATTR_GPU_OAM_ID:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      identity->oam_id = mnl_attr_get_u32(attr);
      break;
    default:
      break;
  }
  return MNL_CB_OK;
}

static int ualoe_nl_get_gpu_identity_handler(const struct nlmsghdr* nlh, void* arg) {
  return mnl_attr_parse(nlh, sizeof(struct genlmsghdr), ualoe_nl_parse_gpu_identity, arg);
}

int ualoe_nl_get_gpu_identity(ualoe_handle_t handle, ualoe_gpu_identity_t* identity) {
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_GET_GPU_IDENTITY, nl_handle->family_id,
                    1);
  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);

  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, ualoe_nl_get_gpu_identity_handler,
                            identity);
}

static int ualoe_nl_parse_telemetry_category_mask(const struct nlattr* attr, void* arg) {
  unsigned* category_mask = arg;

  switch (mnl_attr_get_type(attr)) {
    case CFG_ATTR_CATEGORY_MASK:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) return MNL_CB_ERROR;
      *category_mask = mnl_attr_get_u32(attr);
      break;
    default:
      break;
  }
  return MNL_CB_OK;
}

static int ualoe_nl_telemetry_get_category_mask_handler(const struct nlmsghdr* nlh, void* arg) {
  return mnl_attr_parse(nlh, sizeof(struct genlmsghdr), ualoe_nl_parse_telemetry_category_mask,
                        arg);
}

int ualoe_nl_telemetry_get_category_mask(ualoe_handle_t handle, unsigned* category_mask) {
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  *category_mask = 0;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_GET_TELEMETRY_CATEGORY_MASK,
                    nl_handle->family_id, 1);
  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);

  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh,
                            ualoe_nl_telemetry_get_category_mask_handler, category_mask);
}

int ualoe_nl_telemetry_alloc(ualoe_handle_t handle, unsigned category_mask,
                             ualoe_telemetry_t** telemetry) {
  struct ualoe_nl_handle* nl_handle;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  pthread_mutex_lock(&nl_handle->request_lock);
  rc = ualoe_cdev_telemetry_alloc(nl_handle->cdev_fd, category_mask, telemetry);
  pthread_mutex_unlock(&nl_handle->request_lock);
  return rc;
}

int ualoe_nl_telemetry_get(ualoe_handle_t handle, ualoe_telemetry_t* telemetry) {
  struct ualoe_nl_handle* nl_handle;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  pthread_mutex_lock(&nl_handle->request_lock);
  rc = ualoe_cdev_telemetry_get(nl_handle->cdev_fd, telemetry);
  pthread_mutex_unlock(&nl_handle->request_lock);
  return rc;
}

int ualoe_nl_telemetry_free(ualoe_handle_t handle, ualoe_telemetry_t* telemetry) {
  struct ualoe_nl_handle* nl_handle;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  pthread_mutex_lock(&nl_handle->request_lock);
  rc = ualoe_cdev_telemetry_free(nl_handle->cdev_fd, telemetry);
  pthread_mutex_unlock(&nl_handle->request_lock);
  return rc;
}

int ualoe_nl_l2ping_start(ualoe_handle_t handle, ualoe_ping_spec_t* spec, ualoe_ping_t** ping) {
  struct ualoe_nl_handle* nl_handle;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  pthread_mutex_lock(&nl_handle->request_lock);
  rc = ualoe_cdev_l2ping_start(nl_handle->cdev_fd, spec, ping);
  pthread_mutex_unlock(&nl_handle->request_lock);
  return rc;
}

int ualoe_nl_l2ping_update(ualoe_handle_t handle, ualoe_ping_t* ping) {
  struct ualoe_nl_handle* nl_handle;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  pthread_mutex_lock(&nl_handle->request_lock);
  rc = ualoe_cdev_l2ping_update(nl_handle->cdev_fd, ping);
  pthread_mutex_unlock(&nl_handle->request_lock);
  return rc;
}

int ualoe_nl_l2ping_fini(ualoe_handle_t handle, ualoe_ping_t* ping) {
  struct ualoe_nl_handle* nl_handle;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  pthread_mutex_lock(&nl_handle->request_lock);
  rc = ualoe_cdev_l2ping_fini(nl_handle->cdev_fd, ping);
  pthread_mutex_unlock(&nl_handle->request_lock);
  return rc;
}

int ualoe_nl_register_event_callback(ualoe_handle_t handle, ualoe_event_callback_t callback,
                                     void* user_context) {
  struct ualoe_nl_handle* nl_handle;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  return ualoe_cb_init(handle, nl_handle->dev_id, nl_handle->cdev_fd, callback, user_context);
}

/*---------- PMA lane and PRBS diagnostic netlink functions ----------*/

int ualoe_nl_diag_config_pma_lane(ualoe_handle_t handle, unsigned netport_idx, unsigned lane_idx,
                                  bool enable, ualoe_pma_rate_e pma_rate,
                                  ualoe_netport_loopback_mode_e loopback_mode,
                                  ualoe_pma_polarity_e tx_polarity,
                                  ualoe_pma_polarity_e rx_polarity) {
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_DIAG_CONFIG_PMA_LANE,
                    nl_handle->family_id, 1);

  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);
  mnl_attr_put_u32(nlh, CFG_ATTR_NETPORT_LOGICAL_IDX, netport_idx);
  mnl_attr_put_u32(nlh, CFG_ATTR_LANE_IDX, lane_idx);
  mnl_attr_put_u8(nlh, CFG_ATTR_ENABLE, (uint8_t)enable);
  mnl_attr_put_u32(nlh, CFG_ATTR_PMA_RATE, (uint32_t)pma_rate);
  mnl_attr_put_u32(nlh, CFG_ATTR_NETPORT_LOOPBACK_MODE, (uint32_t)loopback_mode);
  mnl_attr_put_u32(nlh, CFG_ATTR_TX_POLARITY, (uint32_t)tx_polarity);
  mnl_attr_put_u32(nlh, CFG_ATTR_RX_POLARITY, (uint32_t)rx_polarity);

  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, NULL, NULL);
}

int ualoe_nl_diag_config_prbs_tx(ualoe_handle_t handle, unsigned netport_idx, unsigned lane_idx,
                                 bool enable, ualoe_prbs_pattern_e pattern,
                                 __uint128_t user_pattern) {
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_DIAG_CONFIG_PRBS_TX, nl_handle->family_id,
                    1);

  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);
  mnl_attr_put_u32(nlh, CFG_ATTR_NETPORT_LOGICAL_IDX, netport_idx);
  mnl_attr_put_u32(nlh, CFG_ATTR_LANE_IDX, lane_idx);
  mnl_attr_put_u8(nlh, CFG_ATTR_ENABLE, (uint8_t)enable);
  mnl_attr_put_u32(nlh, CFG_ATTR_PRBS_PATTERN, (uint32_t)pattern);
  if (!mnl_attr_put_check(nlh, sizeof(buf), CFG_ATTR_USER_PATTERN, sizeof(user_pattern),
                          &user_pattern)) {
    ualoe_log_error("%s: Failed to put user_pattern to buffer\n", __func__);
    return ENOMEM;
  }

  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, NULL, NULL);
}

int ualoe_nl_diag_config_prbs_rx(ualoe_handle_t handle, unsigned netport_idx, unsigned lane_idx,
                                 bool enable, bool resync, ualoe_prbs_pattern_e pattern,
                                 __uint128_t user_pattern) {
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_DIAG_CONFIG_PRBS_RX, nl_handle->family_id,
                    1);

  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);
  mnl_attr_put_u32(nlh, CFG_ATTR_NETPORT_LOGICAL_IDX, netport_idx);
  mnl_attr_put_u32(nlh, CFG_ATTR_LANE_IDX, lane_idx);
  mnl_attr_put_u8(nlh, CFG_ATTR_ENABLE, (uint8_t)enable);
  mnl_attr_put_u8(nlh, CFG_ATTR_RESYNC, (uint8_t)resync);
  mnl_attr_put_u32(nlh, CFG_ATTR_PRBS_PATTERN, (uint32_t)pattern);
  if (!mnl_attr_put_check(nlh, sizeof(buf), CFG_ATTR_USER_PATTERN, sizeof(user_pattern),
                          &user_pattern)) {
    ualoe_log_error("%s: Failed to put user_pattern to buffer\n", __func__);
    return ENOMEM;
  }

  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, NULL, NULL);
}

static int ualoe_nl_parse_prbs_results_nested(const struct nlattr* attr, void* arg) {
  ualoe_prbs_results_t* results = arg;
  uint64_t elapsed_us;

  switch (mnl_attr_get_type(attr)) {
    case CFG_ATTR_PRBS_RESULTS_VALID:
      if (mnl_attr_validate(attr, MNL_TYPE_U8) < 0) return MNL_CB_ERROR;
      results->valid = !!mnl_attr_get_u8(attr);
      break;
    case CFG_ATTR_PRBS_RESULTS_RXEQ_SUCCESS:
      if (mnl_attr_validate(attr, MNL_TYPE_U8) < 0) return MNL_CB_ERROR;
      results->rxeq_success = !!mnl_attr_get_u8(attr);
      break;
    case CFG_ATTR_PRBS_RESULTS_CDR_LOCK:
      if (mnl_attr_validate(attr, MNL_TYPE_U8) < 0) return MNL_CB_ERROR;
      results->cdr_lock = !!mnl_attr_get_u8(attr);
      break;
    case CFG_ATTR_PRBS_RESULTS_PATTERN_LOCK:
      if (mnl_attr_validate(attr, MNL_TYPE_U8) < 0) return MNL_CB_ERROR;
      results->pattern_lock = !!mnl_attr_get_u8(attr);
      break;
    case CFG_ATTR_PRBS_RESULTS_OVERFLOW:
      if (mnl_attr_validate(attr, MNL_TYPE_U8) < 0) return MNL_CB_ERROR;
      results->overflow = !!mnl_attr_get_u8(attr);
      break;
    case CFG_ATTR_PRBS_RESULTS_ELAPSED_US:
      if (mnl_attr_validate(attr, MNL_TYPE_U64) < 0) return MNL_CB_ERROR;
      elapsed_us = mnl_attr_get_u64(attr);
      results->interval.tv_sec = (time_t)(elapsed_us / 1000000ULL);
      results->interval.tv_nsec = (long)((elapsed_us % 1000000ULL) * 1000ULL);
      break;
    case CFG_ATTR_PRBS_RESULTS_ERROR_COUNT:
      if (mnl_attr_validate(attr, MNL_TYPE_U64) < 0) return MNL_CB_ERROR;
      results->error_count = mnl_attr_get_u64(attr);
      break;
    default:
      break;
  }
  return MNL_CB_OK;
}

static int ualoe_nl_parse_prbs_results(const struct nlattr* attr, void* arg) {
  if (mnl_attr_get_type(attr) == CFG_ATTR_PRBS_RESULTS)
    return mnl_attr_parse_nested(attr, ualoe_nl_parse_prbs_results_nested, arg);
  return MNL_CB_OK;
}

static int ualoe_nl_get_prbs_results_handler(const struct nlmsghdr* nlh, void* arg) {
  return mnl_attr_parse(nlh, sizeof(struct genlmsghdr), ualoe_nl_parse_prbs_results, arg);
}

int ualoe_nl_diag_get_prbs_results(ualoe_handle_t handle, unsigned netport_idx, unsigned lane_idx,
                                   ualoe_prbs_results_t* results) {
  struct ualoe_nl_handle* nl_handle;
  char buf[MNL_SOCKET_BUFFER_SIZE];
  struct genlmsghdr* genlh;
  struct nlmsghdr* nlh;
  int rc;

  rc = ualoe_nl_find_handle(handle, &nl_handle);
  if (rc) return rc;

  ualoe_nl_init_msg(nl_handle, &nlh, &genlh, buf, CFG_CMD_DIAG_GET_PRBS_RESULTS,
                    nl_handle->family_id, 1);

  mnl_attr_put_u32(nlh, CFG_ATTR_DEV_ID, nl_handle->dev_id);
  mnl_attr_put_u32(nlh, CFG_ATTR_NETPORT_LOGICAL_IDX, netport_idx);
  mnl_attr_put_u32(nlh, CFG_ATTR_LANE_IDX, lane_idx);

  return ualoe_nl_send_recv(nl_handle, buf, sizeof(buf), nlh, ualoe_nl_get_prbs_results_handler,
                            results);
}
