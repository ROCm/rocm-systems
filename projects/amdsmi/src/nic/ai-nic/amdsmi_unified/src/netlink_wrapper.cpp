// SPDX-License-Identifier: MIT
/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Implementation of RAII wrappers for libnl-3 netlink library
 */

#include "netlink_wrapper.h"

#include <netlink/errno.h>

#include <cstring>
#include <stdexcept>
#include <utility>

namespace amd::nic::netlink {

NLSocket::NLSocket() : sock_(nl_socket_alloc()) {
  if (!sock_) {
    throw std::runtime_error("Failed to allocate netlink socket");
  }
}

NLSocket::~NLSocket() {
  if (sock_) {
    nl_socket_free(sock_);
    sock_ = nullptr;
  }
}

NLSocket::NLSocket(NLSocket&& other) noexcept : sock_(other.sock_) { other.sock_ = nullptr; }

NLSocket& NLSocket::operator=(NLSocket&& other) noexcept {
  if (this != &other) {
    if (sock_) {
      nl_socket_free(sock_);
    }
    sock_ = other.sock_;
    other.sock_ = nullptr;
  }
  return *this;
}

int NLSocket::connect(int /* protocol */) {
  if (!sock_) {
    return -NLE_BAD_SOCK;
  }
  // Note: protocol parameter ignored - genl_connect always uses NETLINK_GENERIC
  return genl_connect(sock_);
}

void NLSocket::set_buffer_size(int rx_size, int tx_size) {
  if (!sock_) {
    return;
  }
  nl_socket_set_buffer_size(sock_, rx_size, tx_size);
}

NLMessage::NLMessage() : msg_(nlmsg_alloc()) {
  if (!msg_) {
    throw std::runtime_error("Failed to allocate netlink message");
  }
}

NLMessage::NLMessage(struct nl_msg* msg) : msg_(msg) {
  if (!msg_) {
    throw std::runtime_error("Cannot wrap null netlink message");
  }
}

NLMessage::~NLMessage() {
  if (msg_) {
    nlmsg_free(msg_);
    msg_ = nullptr;
  }
}

NLMessage::NLMessage(NLMessage&& other) noexcept : msg_(other.msg_) { other.msg_ = nullptr; }

NLMessage& NLMessage::operator=(NLMessage&& other) noexcept {
  if (this != &other) {
    if (msg_) {
      nlmsg_free(msg_);
    }
    msg_ = other.msg_;
    other.msg_ = nullptr;
  }
  return *this;
}

void* NLMessage::put_genl_header(uint32_t port, uint32_t seq, int family, int hdrlen, int flags,
                                 uint8_t cmd, uint8_t version) {
  if (!msg_) {
    return nullptr;
  }
  return genlmsg_put(msg_, port, seq, family, hdrlen, flags, cmd, version);
}

struct nl_msg* NLMessage::release() {
  struct nl_msg* tmp = msg_;
  msg_ = nullptr;
  return tmp;
}

NLCallback::NLCallback() : cb_(nl_cb_alloc(NL_CB_DEFAULT)) {
  if (!cb_) {
    throw std::runtime_error("Failed to allocate netlink callback");
  }
}

NLCallback::~NLCallback() {
  if (cb_) {
    nl_cb_put(cb_);
    cb_ = nullptr;
  }
}

NLCallback::NLCallback(NLCallback&& other) noexcept : cb_(other.cb_) { other.cb_ = nullptr; }

NLCallback& NLCallback::operator=(NLCallback&& other) noexcept {
  if (this != &other) {
    if (cb_) {
      nl_cb_put(cb_);
    }
    cb_ = other.cb_;
    other.cb_ = nullptr;
  }
  return *this;
}

int NLCallback::set(int type, int kind, nl_recvmsg_msg_cb_t func, void* arg) {
  if (!cb_) {
    return -NLE_BAD_SOCK;
  }
  return nl_cb_set(cb_, static_cast<nl_cb_type>(type), static_cast<nl_cb_kind>(kind), func, arg);
}

NLAttributes::NLAttributes(NLMessage& msg) : msg_(msg) {}

int NLAttributes::put_u32(int type, uint32_t value) {
  if (!msg_.is_valid()) {
    return -NLE_NOMEM;
  }
  return nla_put_u32(msg_.get(), type, value);
}

int NLAttributes::put_u16(int type, uint16_t value) {
  if (!msg_.is_valid()) {
    return -NLE_NOMEM;
  }
  return nla_put_u16(msg_.get(), type, value);
}

int NLAttributes::put_u8(int type, uint8_t value) {
  if (!msg_.is_valid()) {
    return -NLE_NOMEM;
  }
  return nla_put_u8(msg_.get(), type, value);
}

int NLAttributes::put_string(int type, const char* value) {
  if (!msg_.is_valid()) {
    return -NLE_NOMEM;
  }
  return nla_put_string(msg_.get(), type, value);
}

int NLAttributes::put_string(int type, const std::string& value) {
  return put_string(type, value.c_str());
}

int NLAttributes::put_flag(int type) {
  if (!msg_.is_valid()) {
    return -NLE_NOMEM;
  }
  return nla_put_flag(msg_.get(), type);
}

struct nlattr* NLAttributes::nest_start(int type) {
  if (!msg_.is_valid()) {
    return nullptr;
  }
  return nla_nest_start(msg_.get(), type);
}

void NLAttributes::nest_end(struct nlattr* nested) {
  if (msg_.is_valid() && nested) {
    nla_nest_end(msg_.get(), nested);
  }
}

int NLAttributes::parse(struct nlmsghdr* nlh, int hdrlen, struct nlattr** tb, int maxtype,
                        struct nla_policy* policy) {
  return nlmsg_parse(nlh, hdrlen, tb, maxtype, policy);
}

uint32_t NLAttributes::get_u32(struct nlattr* attr) { return attr ? nla_get_u32(attr) : 0; }

uint16_t NLAttributes::get_u16(struct nlattr* attr) { return attr ? nla_get_u16(attr) : 0; }

uint8_t NLAttributes::get_u8(struct nlattr* attr) { return attr ? nla_get_u8(attr) : 0; }

const char* NLAttributes::get_string(struct nlattr* attr) {
  return attr ? nla_get_string(attr) : "";
}

bool NLAttributes::exists(struct nlattr* attr) { return attr != nullptr; }

}  // namespace amd::nic::netlink
