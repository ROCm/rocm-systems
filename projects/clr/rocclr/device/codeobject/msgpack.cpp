/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "device/codeobject/msgpack.hpp"

namespace amd::codeobject {

namespace {

// MessagePack type tags (https://github.com/msgpack/msgpack/blob/master/spec.md)
constexpr uint8_t kNil = 0xc0;
constexpr uint8_t kFalse = 0xc2;
constexpr uint8_t kTrue = 0xc3;
constexpr uint8_t kBin8 = 0xc4;
constexpr uint8_t kBin16 = 0xc5;
constexpr uint8_t kBin32 = 0xc6;
constexpr uint8_t kExt8 = 0xc7;
constexpr uint8_t kExt16 = 0xc8;
constexpr uint8_t kExt32 = 0xc9;
constexpr uint8_t kFloat32 = 0xca;
constexpr uint8_t kFloat64 = 0xcb;
constexpr uint8_t kUint8 = 0xcc;
constexpr uint8_t kUint64 = 0xcf;
constexpr uint8_t kInt8 = 0xd0;
constexpr uint8_t kInt64 = 0xd3;
constexpr uint8_t kFixExt1 = 0xd4;
constexpr uint8_t kFixExt16 = 0xd8;
constexpr uint8_t kStr8 = 0xd9;
constexpr uint8_t kStr16 = 0xda;
constexpr uint8_t kStr32 = 0xdb;
constexpr uint8_t kArray16 = 0xdc;
constexpr uint8_t kArray32 = 0xdd;
constexpr uint8_t kMap16 = 0xde;
constexpr uint8_t kMap32 = 0xdf;

}  // namespace

// ================================================================================================
bool MsgPackReader::advance(uint64_t bytes) {
  if (bytes > static_cast<uint64_t>(end_ - cur_)) return false;
  cur_ += bytes;
  return true;
}

// ================================================================================================
bool MsgPackReader::readBigEndian(size_t bytes, uint64_t* value) {
  if (bytes > static_cast<size_t>(end_ - cur_)) return false;
  uint64_t v = 0;
  for (size_t i = 0; i < bytes; ++i) {
    v = (v << 8) | static_cast<uint8_t>(cur_[i]);
  }
  cur_ += bytes;
  *value = v;
  return true;
}

// ================================================================================================
bool MsgPackReader::readContainerSize(uint8_t fixBase, uint8_t fixMask, uint8_t tag16,
                                      uint8_t tag32, uint32_t* count) {
  if (cur_ >= end_) return false;
  const uint8_t tag = static_cast<uint8_t>(*cur_);
  uint64_t value = 0;
  if ((tag & ~fixMask) == fixBase) {
    ++cur_;
    *count = tag & fixMask;
    return true;
  }
  if (tag == tag16 || tag == tag32) {
    ++cur_;
    if (!readBigEndian(tag == tag16 ? 2 : 4, &value)) return false;
    *count = static_cast<uint32_t>(value);
    return true;
  }
  return false;
}

// ================================================================================================
bool MsgPackReader::readMapSize(uint32_t* count) {
  return readContainerSize(0x80, 0x0f, kMap16, kMap32, count);
}

// ================================================================================================
bool MsgPackReader::readArraySize(uint32_t* count) {
  return readContainerSize(0x90, 0x0f, kArray16, kArray32, count);
}

// ================================================================================================
bool MsgPackReader::readString(std::string_view* value) {
  if (cur_ >= end_) return false;
  const uint8_t tag = static_cast<uint8_t>(*cur_);
  uint64_t size = 0;
  if ((tag & 0xe0) == 0xa0) {
    ++cur_;
    size = tag & 0x1f;
  } else if (tag == kStr8 || tag == kStr16 || tag == kStr32) {
    ++cur_;
    const size_t bytes = (tag == kStr8) ? 1 : (tag == kStr16) ? 2 : 4;
    if (!readBigEndian(bytes, &size)) return false;
  } else {
    return false;
  }
  const char* start = cur_;
  if (!advance(size)) return false;
  *value = std::string_view(start, static_cast<size_t>(size));
  return true;
}

// ================================================================================================
bool MsgPackReader::readUint(uint64_t* value) {
  if (cur_ >= end_) return false;
  const uint8_t tag = static_cast<uint8_t>(*cur_);
  if (tag <= 0x7f) {  // positive fixint
    ++cur_;
    *value = tag;
    return true;
  }
  if (tag >= kUint8 && tag <= kUint64) {
    ++cur_;
    return readBigEndian(size_t{1} << (tag - kUint8), value);
  }
  if (tag >= kInt8 && tag <= kInt64) {
    const char* start = cur_++;
    const size_t bytes = size_t{1} << (tag - kInt8);
    uint64_t raw = 0;
    if (!readBigEndian(bytes, &raw)) return false;
    // Negative if the sign bit of the stored width is set
    if ((raw >> (bytes * 8 - 1)) & 1) {
      cur_ = start;
      return false;
    }
    *value = raw;
    return true;
  }
  return false;
}

// ================================================================================================
bool MsgPackReader::readBool(bool* value) {
  if (cur_ >= end_) return false;
  const uint8_t tag = static_cast<uint8_t>(*cur_);
  if (tag == kTrue || tag == kFalse) {
    ++cur_;
    *value = (tag == kTrue);
    return true;
  }
  uint64_t number = 0;
  if (!readUint(&number)) return false;
  *value = (number != 0);
  return true;
}

// ================================================================================================
bool MsgPackReader::skip() {
  // Values still to skip; containers add their elements instead of recursing
  uint64_t pending = 1;
  while (pending > 0) {
    if (cur_ >= end_) return false;
    const uint8_t tag = static_cast<uint8_t>(*cur_++);
    --pending;
    uint64_t size = 0;
    if (tag <= 0x7f || tag >= 0xe0 || tag == kNil || tag == kFalse || tag == kTrue) {
      continue;  // fixints, nil, booleans
    } else if ((tag & 0xf0) == 0x80) {  // fixmap
      pending += 2u * (tag & 0x0f);
    } else if ((tag & 0xf0) == 0x90) {  // fixarray
      pending += tag & 0x0f;
    } else if ((tag & 0xe0) == 0xa0) {  // fixstr
      if (!advance(tag & 0x1f)) return false;
    } else if (tag == kBin8 || tag == kStr8) {
      if (!readBigEndian(1, &size) || !advance(size)) return false;
    } else if (tag == kBin16 || tag == kStr16) {
      if (!readBigEndian(2, &size) || !advance(size)) return false;
    } else if (tag == kBin32 || tag == kStr32) {
      if (!readBigEndian(4, &size) || !advance(size)) return false;
    } else if (tag == kExt8 || tag == kExt16 || tag == kExt32) {
      const size_t bytes = (tag == kExt8) ? 1 : (tag == kExt16) ? 2 : 4;
      if (!readBigEndian(bytes, &size) || !advance(size + 1)) return false;  // + type byte
    } else if (tag == kFloat32) {
      if (!advance(4)) return false;
    } else if (tag == kFloat64) {
      if (!advance(8)) return false;
    } else if (tag >= kUint8 && tag <= kUint64) {
      if (!advance(uint64_t{1} << (tag - kUint8))) return false;
    } else if (tag >= kInt8 && tag <= kInt64) {
      if (!advance(uint64_t{1} << (tag - kInt8))) return false;
    } else if (tag >= kFixExt1 && tag <= kFixExt16) {
      if (!advance((uint64_t{1} << (tag - kFixExt1)) + 1)) return false;  // + type byte
    } else if (tag == kArray16 || tag == kArray32) {
      if (!readBigEndian(tag == kArray16 ? 2 : 4, &size)) return false;
      pending += size;
    } else if (tag == kMap16 || tag == kMap32) {
      if (!readBigEndian(tag == kMap16 ? 2 : 4, &size)) return false;
      pending += 2 * size;
    } else {
      return false;  // 0xc1 is never used
    }
  }
  return true;
}

}  // namespace amd::codeobject
