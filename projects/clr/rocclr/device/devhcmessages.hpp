/* Copyright (c) 2020 - 2021 Advanced Micro Devices, Inc.

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

/** \file Concatenating hostcalls into a message
 *
 *  A message is a stream of 64-bit integers transmitted as a series
 *  of hostcall invocations by the device code. Although the hostcall
 *  is "warp-wide", the message for each workitem is distinct.
 *
 *  Of the eight uint64_t arguments in hostcall, the first argument is
 *  used as the message descriptor, while the rest are used for
 *  message contents. The descriptor consists of the following fields:
 *
 *  - Bit  0     is the BEGIN flag.
 *  - Bit  1     is the END flag.
 *  - Bits 2-4   are reserved and must be zero.
 *  - Bits 5-7   indicate the number of elements being transmitted.
 *  - Bits 8-63  contain a 56-bit message ID.
 *
 *  A hostcall with the BEGIN flag set in the descriptor indicates the
 *  start of a new message. A hostcall with the END flag set indicates
 *  the end of a message. A single hostcall can have both flags set if
 *  the message fits in the payload of a single hostcall.  Each
 *  hostcall indicates the number of uint64_t elements in the payload
 *  that contain data to be appended to the message.
 *
 *  When the message receives a hostcall with the BEGIN flag set, it allocates a
 *  new message ID, which is transmitted to the device via the first return
 *  value in the hostcall. Every subsequent hostcall containing the same message
 *  ID appends its payload to that message. The message is said to be "active"
 *  until a corresponding END hostcall is received.
 *
 *  When the message handler receives a hostcall with the END flag set, it
 *  invokes the corresponding message handler on the contents of the
 *  accumulated message, and then discards the message. The handler
 *  may return up to two uint64_t values, that are transmitted to the
 *  device via the return value of the last hostcall.
 *
 *  Behaviour is undefined in each of the following cases:
 *  - An END packet is received with a non-existent message ID, or with
 *    the ID of an inactive message (previously terminated by an END packet).
 *  - No END packet is received for an active message.
 *  - Any of the reserved bits are non-zero.
 *  - Different hostcalls indicate the same active message ID but a
 *    different service.
 */

#pragma once

#include <vector>

namespace amd {

enum ServiceID {
  SERVICE_RESERVED = 0,
  SERVICE_FUNCTION_CALL = 1,
  SERVICE_PRINTF = 2,
  SERVICE_DEVMEM = 3
#if defined(__clang__)
#if __has_feature(address_sanitizer)
  ,
  SERVICE_SANITIZER = 4
#endif
#endif
};

struct Message;

class MessageHandler {
  std::vector<uint64_t> freeSlots_;
  std::vector<Message*> messageSlots_;

  Message* newMessage();
  Message* getMessage(uint64_t messageId);
  void discardMessage(Message* message);

 public:
  ~MessageHandler();
  bool handlePayload(uint32_t service, uint64_t* payload);
};
}  // namespace amd