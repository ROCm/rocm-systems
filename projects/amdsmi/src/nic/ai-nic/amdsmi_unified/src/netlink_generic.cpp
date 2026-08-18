// SPDX-License-Identifier: MIT
/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Implementation of Generic Netlink Client - Layer 2
 */

#include "netlink_generic.h"

#include <netlink/errno.h>
#include <netlink/genl/ctrl.h>

#include <cstring>

namespace amd::nic::netlink {

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
