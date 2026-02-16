/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "sync/abql_block_mutex.hpp"

#include "util.hpp"

namespace rocshmem {

__device__ ABQLBlockMutex::TicketT ABQLBlockMutex::lock() {
  TicketT my_ticket{grab_ticket_()};
  wait_turn_(my_ticket);
  return my_ticket;
}

__device__ void ABQLBlockMutex::unlock(TicketT my_ticket) {
  TicketT next_ticket{my_ticket + 1};
  signal_next_(next_ticket);
}

__device__ ABQLBlockMutex::TicketT ABQLBlockMutex::grab_ticket_() {
  TicketT ticket{atomicAdd(&ticket_, 1)};
  return ticket;
}

__device__ bool ABQLBlockMutex::is_turn_(TicketT ticket) {
  size_t index{ticket % memory_channels_};
  return turns_[index].vol_ == ticket;
}

__device__ void ABQLBlockMutex::wait_turn_(TicketT ticket) {
  size_t index{ticket % memory_channels_};
  while (turns_[index].vol_ != ticket) {
  }
}

__device__ void ABQLBlockMutex::signal_next_(TicketT ticket) {
  size_t index{ticket % memory_channels_};
  turns_[index].vol_ = ticket;
  __threadfence();
}

}  // namespace rocshmem
