/* Copyright (c) 2020 - 2022 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "utils/debug.hpp"
#include "top.hpp"
#include "utils/flags.hpp"

#include "devhcmessages.hpp"

#include <vector>
namespace amd {

enum {
  DESCRIPTOR_OFFSET_FLAG_BEGIN = 0,
  DESCRIPTOR_OFFSET_FLAG_END = 1,
  DESCRIPTOR_OFFSET_RESERVED0 = 2,
  DESCRIPTOR_OFFSET_LEN = 5,
  DESCRIPTOR_OFFSET_ID = 8
};

enum {
  DESCRIPTOR_WIDTH_FLAG_BEGIN = 1,
  DESCRIPTOR_WIDTH_FLAG_END = 1,
  DESCRIPTOR_WIDTH_RESERVED0 = 3,
  DESCRIPTOR_WIDTH_LEN = 3,
  DESCRIPTOR_WIDTH_ID = 56
};

struct Message {
  std::vector<uint64_t> data_;
  bool live_;
  uint64_t messageId_;

  void clear() {
    live_ = false;
    data_.clear();
  };

  void append(uint64_t* payload, uint8_t len) { data_.insert(data_.end(), payload, payload + len); }

  Message(uint64_t c) : live_(true), messageId_(c) {};
};

static uint64_t getField(uint64_t desc, uint8_t offset, uint8_t width) {
  return (desc >> offset) & ((1ULL << width) - 1);
}

static uint64_t setField(uint64_t desc, uint64_t value, uint8_t offset, uint8_t width) {
  uint64_t resetMask = ~(((1ULL << width) - 1) << offset);
  return (desc & resetMask) | (value << offset);
}

static uint64_t getLen(uint64_t desc) {
  return getField(desc, DESCRIPTOR_OFFSET_LEN, DESCRIPTOR_WIDTH_LEN);
}

static uint64_t getBeginFlag(uint64_t desc) {
  return getField(desc, DESCRIPTOR_OFFSET_FLAG_BEGIN, DESCRIPTOR_WIDTH_FLAG_BEGIN);
}

static uint64_t getEndFlag(uint64_t desc) {
  return getField(desc, DESCRIPTOR_OFFSET_FLAG_END, DESCRIPTOR_WIDTH_FLAG_END);
}

static uint64_t resetBeginFlag(uint64_t desc) {
  uint64_t resetMask = ~(1 << DESCRIPTOR_OFFSET_FLAG_BEGIN);
  return desc & resetMask;
}

static uint64_t getMessageId(uint64_t desc) {
  return getField(desc, DESCRIPTOR_OFFSET_ID, DESCRIPTOR_WIDTH_ID);
}

static uint64_t setMessageId(uint64_t desc, uint64_t id) {
  return setField(desc, id, DESCRIPTOR_OFFSET_ID, DESCRIPTOR_WIDTH_ID);
}

Message* MessageHandler::newMessage() {
  if (!freeSlots_.empty()) {
    auto c = freeSlots_.back();
    freeSlots_.pop_back();
    assert(c <= messageSlots_.size());
    Message* m = messageSlots_[c];
    assert(m);
    assert(m->messageId_ == c);
    assert(m->data_.empty());
    m->live_ = true;
    return m;
  }

  Message* m = new Message(messageSlots_.size());
  messageSlots_.push_back(m);
  return m;
}

MessageHandler::~MessageHandler() {
  for (auto M : messageSlots_) {
    delete M;
  }
}

Message* MessageHandler::getMessage(uint64_t messageId) {
  if (messageSlots_.size() <= messageId) {
    return nullptr;
  }

  auto m = messageSlots_[messageId];

  return m->live_ ? m : nullptr;
}

void MessageHandler::discardMessage(Message* message) {
  message->clear();
  freeSlots_.push_back(message->messageId_);
  // TODO: Consider reducing the number of message slots based on
  // some busy-ness heuristic.
}

// Defined in devhcprintf.cpp

void handlePrintf(uint64_t* output, const uint64_t* input, uint64_t len);


bool MessageHandler::handlePayload(uint32_t service, uint64_t* payload) {
  Message* message = nullptr;

  auto desc = payload[0];
  auto begin = getBeginFlag(desc);
  auto end = getEndFlag(desc);

  if (begin) {
    message = newMessage();
    desc = resetBeginFlag(desc);
    desc = setMessageId(desc, message->messageId_);
    payload[0] = desc;
  } else {
    auto messageId = getMessageId(desc);
    message = getMessage(messageId);
  }

  if (!message) {
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "Hostcall: No message found");
    return false;
  }

  auto len = getLen(desc);
  message->append(payload + 1, len);

  if (!end) {
    return true;
  }

  switch (service) {
    case SERVICE_PRINTF:
      amd::handlePrintf(payload, message->data_.data(), message->data_.size());
      break;
    default:
      ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "Hostcall: Messages not supported for service %d",
              service);
      return false;
  }
  discardMessage(message);
  return true;
}
}  // namespace amd