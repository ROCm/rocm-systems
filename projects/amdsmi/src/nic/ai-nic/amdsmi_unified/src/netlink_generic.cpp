// SPDX-License-Identifier: MIT
/*
 * Copyright Advanced Micro Devices, Inc.
 *
 * Implementation of Generic Netlink Client - Layer 2
 */

#include "netlink_generic.h"

#include <netlink/errno.h>
#include <netlink/genl/ctrl.h>
#include <netlink/socket.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <optional>
#include <string>
#include <utility>

#include "smi_nic_log.h"

namespace amd::nic::netlink {

namespace {
// "family=<int> cmd=<u8> flags=0x<u16>" fits well within this.
constexpr size_t kQueryHeadTextLen = 64;
constexpr int kNlBufferBytes = 32768;
}  // namespace

int nle_to_errno(int nl_error) {
  switch (nl_error) {
    case NLE_OPNOTSUPP:
      return EOPNOTSUPP;
    case NLE_OBJ_NOTFOUND:
      return ENOENT;
    case NLE_NOMEM:
      return ENOMEM;
    case NLE_PERM:
      return EPERM;
    case NLE_AGAIN:
      return EAGAIN;
    case NLE_BAD_SOCK:
      return EBADF;
    case NLE_NODEV:
      return ENODEV;
    case NLE_INVAL:
      return EINVAL;
    case NLE_BUSY:
      return EBUSY;
    case NLE_NOACCESS:
      return EACCES;
    default:
      return EIO;
  }
}

GenericNetlinkClient::GenericNetlinkClient()
    : socket_(), is_connected_(false), is_reconnect_pending_(false) {}

int GenericNetlinkClient::open_socket() {
  int ret = socket_.connect(NETLINK_GENERIC);
  if (ret < 0) {
    return ret;
  }

  socket_.set_buffer_size(kNlBufferBytes, kNlBufferBytes);
  socket_.disable_auto_ack();
  ret = socket_.set_recv_timeout(kNlRecvTimeoutSec);
  if (ret < 0) {
    // Leave no open descriptor behind, so a later attempt starts from a closed socket.
    nl_close(socket_.get());
    return ret;
  }
  return 0;
}

int GenericNetlinkClient::connect() {
  if (is_connected_) {
    return 0;  // Already connected
  }

  const int ret = open_socket();
  if (ret < 0) {
    return ret;
  }

  is_connected_ = true;
  return 0;
}

std::optional<int> GenericNetlinkClient::resolve_family_id(const std::string& family_name) {
  if (!is_connected_) {
    return std::nullopt;
  }

  int family_id = genl_ctrl_resolve(socket_.get(), family_name.c_str());
  if (family_id < 0) {
    return std::nullopt;
  }

  return family_id;
}

namespace {
// C-style callback wrapper to bridge to C++ std::function
int callback_wrapper(struct nl_msg* msg, void* arg) {
  if (!arg) {
    return NL_SKIP;
  }

  auto* pair = static_cast<std::pair<GenericNetlinkClient::MessageHandler*, void*>*>(arg);
  GenericNetlinkClient::MessageHandler* handler = pair->first;
  void* user_arg = pair->second;

  if (!handler) {
    return NL_SKIP;
  }

  return (*handler)(msg, user_arg);
}
}  // namespace

int GenericNetlinkClient::query(int family_id, uint8_t cmd, uint8_t version,
                                std::function<int(NLMessage&)> build_fn, MessageHandler handler,
                                void* arg, uint16_t flags) {
  const int ret =
      query_impl(family_id, cmd, version, std::move(build_fn), std::move(handler), arg, flags);
  if (amd::smi::nic::log::is_enabled()) {
    char head[kQueryHeadTextLen];
    std::snprintf(head, sizeof(head), "family=%d cmd=%u flags=0x%x", family_id,
                  static_cast<unsigned>(cmd), static_cast<unsigned>(flags));
    NIC_LOG_DEBUG(std::string("netlink query ") + head + " -> " +
                  ((ret < 0) ? (std::string("FAIL ") + nl_geterror(-ret)) : "SUCCESS"));
  }
  return ret;
}

int GenericNetlinkClient::query_impl(int family_id, uint8_t cmd, uint8_t version,
                                     std::function<int(NLMessage&)> build_fn,
                                     MessageHandler handler, void* arg, uint16_t flags) {
  if (!is_connected_) {
    return -NLE_BAD_SOCK;
  }

  // A timed-out query leaves its socket unusable; the replacement is opened here so a failed
  // attempt (for example no free descriptors) is retried by the next query.
  if (is_reconnect_pending_) {
    socket_ = NLSocket();
    const int reconnect_ret = open_socket();
    if (reconnect_ret < 0) {
      return reconnect_ret;
    }
    is_reconnect_pending_ = false;
  }

  NLMessage msg;

  void* hdr = msg.put_genl_header(0, 0, family_id, 0, flags, cmd, version);
  if (!hdr) {
    return -NLE_NOMEM;
  }

  if (build_fn) {
    int ret = build_fn(msg);
    if (ret < 0) {
      return ret;
    }
  }

  NLCallback cb;

  std::pair<MessageHandler*, void*> cb_arg(&handler, arg);
  int ret = cb.set(NL_CB_VALID, NL_CB_CUSTOM, callback_wrapper, &cb_arg);
  if (ret < 0) {
    return ret;
  }

  ret = nl_send_auto(socket_.get(), msg.get());
  if (ret < 0) {
    return ret;
  }

  ret = nl_recvmsgs(socket_.get(), cb.get());
  if (ret == -NLE_AGAIN) {
    // A late reply must not reach the next query, and a fresh socket has a new port id. The same
    // code can come from a kernel NLMSG_ERROR of -EAGAIN, where the replacement is harmless.
    is_reconnect_pending_ = true;
    return ret;
  }
  if (ret < 0) {
    return ret;
  }

  return 0;
}

}  // namespace amd::nic::netlink
