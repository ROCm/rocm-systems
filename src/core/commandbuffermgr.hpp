// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <cstdint>
#include <future>
#include <map>
#include <string>
#include <vector>

#include "core/aql_profile_exception.h"
#include "core/aql_profile_exception.h"
#include "core/aql_profile.hpp"

namespace aql_profile {
class CommandBufferMgr {
 public:
  struct info_t {
    uint32_t prefix_size;
    uint32_t rdcmds_size;
    uint32_t rd2cmds_size;
    uint32_t is_rd_fetch2;
    uint32_t precmds_size;
    uint32_t postcmds_size;
  };

  CommandBufferMgr(void* ptr, const uint32_t& size) { Init(descriptor_t{ptr, size}, false); }
  explicit CommandBufferMgr(const profile_t* profile) { Init(profile->command_buffer, true); }

  char* GetPrefix() { return reinterpret_cast<char*>(buffer_.ptr); }
  char* GetPrefix1() { return reinterpret_cast<char*>(buffer_.ptr) + sizeof(info_t); }
  char* AddPrefix(const uint32_t& delta) {
    const uint32_t size = Align(delta);
    char* ptr = (buffer_.ptr != NULL) ? GetPrefix() + info_.prefix_size : NULL;
    info_.prefix_size += delta;
    buffer_.size -= (size < buffer_.size) ? size : buffer_.size;
    if (buffer_.size == 0)
      throw aql_profile_exc_msg("CommandBufferMgr::AddPrefix(): buffer size set to zero");
    return (buffer_.size != 0) ? ptr : NULL;
  }

  bool SetRdSize(const uint32_t& rd_data_size) {
    const uint32_t size = Align(rd_data_size);
    const bool suc = (size <= buffer_.size);
    if (suc) {
      info_.rdcmds_size = rd_data_size;
      buffer_.size -= size;
    }
    if (!suc)
      throw aql_profile_exc_msg("CommandBufferMgr::SetRdSize(): size set out of the buffer");
    return suc;
  }

  bool SetRd2Size(const uint32_t& rd_data_size) {
    const uint32_t size = Align(rd_data_size);
    const bool suc = SetRdSize(Align(size));
    if (suc) {
      info_.rd2cmds_size = rd_data_size;
      info_.rdcmds_size = 2 * size;
    }
    if (!suc)
      throw aql_profile_exc_msg("CommandBufferMgr::SetRd2Size(): size set out of the buffer");
    return suc;
  }

  bool SetPreSize(const uint32_t& pre_data_size) {
    const uint32_t size = Align(pre_data_size);
    const bool suc = (size <= buffer_.size);
    if (suc) {
      info_.precmds_size = pre_data_size;
      buffer_.size -= size;
    }
    if (!suc)
      throw aql_profile_exc_msg("CommandBufferMgr::SetPreSize(): size set out of the buffer");
    return suc;
  }

  bool Finalize(const uint32_t& data_size) {
    bool suc = (data_size > info_.precmds_size);
    if (suc) {
      const uint32_t post_data_size = data_size - info_.precmds_size;
      const uint32_t size = Align(post_data_size);
      suc = (size <= buffer_.size);
      if (suc) {
        info_.postcmds_size = post_data_size;
        buffer_.size -= size;
      }
      if (!suc)
        throw aql_profile_exc_msg("CommandBufferMgr::Finalize(): postcmd size is out of cmdbuffer");
    }
    if (!suc) throw aql_profile_exc_msg("CommandBufferMgr::Finalize(): postcmd size is zero");

    if (info_slot_) *info_slot_ = info_;

    return suc;
  }

  uint32_t GetSize() const { return GetEndOffset(); }

  descriptor_t GetRdDescr() const {
    descriptor_t descr;
    descr.ptr = reinterpret_cast<char*>(buffer_.ptr) + GetRdOffset();
    descr.size = info_.rdcmds_size;
    return descr;
  }

  descriptor_t FetchRdDescr() {
    descriptor_t descr;
    if (info_.is_rd_fetch2 == 0) {
      info_.is_rd_fetch2 = 1;
      descr.ptr = reinterpret_cast<char*>(buffer_.ptr) + GetRdOffset();
    } else {
      descr.ptr = reinterpret_cast<char*>(buffer_.ptr) + GetRdOffset() + (info_.rdcmds_size / 2);
    }
    descr.size = info_.rd2cmds_size;
    return descr;
  }

  descriptor_t GetPreDescr() const {
    descriptor_t descr;
    descr.ptr = reinterpret_cast<char*>(buffer_.ptr) + GetPreOffset();
    descr.size = info_.precmds_size;
    return descr;
  }

  descriptor_t GetPostDescr() const {
    descriptor_t descr;
    descr.ptr = reinterpret_cast<char*>(buffer_.ptr) + GetPostOffset();
    descr.size = info_.postcmds_size;
    return descr;
  }

  static uint32_t Align(const uint32_t& size) { return (size + align_mask_) & ~align_mask_; }

 private:
  void Init(const descriptor_t& buffer, const bool& import) {
    buffer_ = buffer;
    info_ = {};
    info_slot_ = NULL;

    uint32_t prefix_size = sizeof(info_t);
    if (buffer_.ptr != NULL) {
      info_slot_ = reinterpret_cast<info_t*>(GetPrefix());
      if (import) {
        prefix_size = info_slot_->prefix_size;
        info_ = *info_slot_;
        info_.prefix_size = 0;
      }
    } else {
      buffer_.size = UINT_MAX;
    }
    AddPrefix(prefix_size);
  }

  uint32_t GetRdOffset() const { return Align(info_.prefix_size); }
  uint32_t GetPreOffset() const { return GetRdOffset() + Align(info_.rdcmds_size); }
  uint32_t GetPostOffset() const { return GetPreOffset() + Align(info_.precmds_size); }
  uint32_t GetEndOffset() const { return GetPostOffset() + Align(info_.postcmds_size); }

  static const uint32_t align_size_ = 0x100;
  static const uint32_t align_mask_ = align_size_ - 1;

  descriptor_t buffer_;
  info_t info_;
  info_t* info_slot_;
};
}  // namespace aql_profile