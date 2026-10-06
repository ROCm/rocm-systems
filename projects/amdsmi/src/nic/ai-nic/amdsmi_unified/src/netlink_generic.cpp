// SPDX-License-Identifier: MIT
/*
 * Copyright Advanced Micro Devices, Inc.
 *
 * Implementation of Generic Netlink Client - Layer 2
 */

#include "netlink_generic.h"

#include <netlink/errno.h>
#include <netlink/genl/ctrl.h>

#include <cstdio>
#include <cstring>

#include "smi_nic_log.h"

namespace amd::nic::netlink {

namespace {
// "family=<int> cmd=<u8> flags=0x<u16>" fits well within this.
constexpr size_t kQueryHeadTextLen = 64;
}  // namespace

GenericNetlinkClient::GenericNetlinkClient() : socket_(), connected_(false) {}

int GenericNetlinkClient::connect() {
  if (connected_) {
    return 0;  // Already connected
  }

  int ret = socket_.connect(NETLINK_GENERIC);
  if (ret < 0) {
    return ret;
  }

  socket_.set_buffer_size(32768, 32768);
  socket_.disable_auto_ack();

  connected_ = true;
  return 0;
}

std::optional<int> GenericNetlinkClient::resolve_family_id(const std::string& family_name) {
  if (!connected_) {
    return std::nullopt;
  }

  int family_id = genl_ctrl_resolve(socket_.get(), family_name.c_str());
  if (family_id < 0) {
    return std::nullopt;
  }

  return family_id;
}

// C-style callback wrapper to bridge to C++ std::function
static int callback_wrapper(struct nl_msg* msg, void* arg) {
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
  if (!connected_) {
    return -NLE_BAD_SOCK;
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
  cb.set(NL_CB_VALID, NL_CB_CUSTOM, callback_wrapper, &cb_arg);

  int ret = nl_send_auto(socket_.get(), msg.get());
  if (ret < 0) {
    return ret;
  }

  ret = nl_recvmsgs(socket_.get(), cb.get());
  if (ret < 0) {
    return ret;
  }

  return 0;
}

}  // namespace amd::nic::netlink
