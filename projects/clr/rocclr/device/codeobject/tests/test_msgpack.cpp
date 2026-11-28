/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// MessagePack reader used for NT_AMDGPU_METADATA notes: every encoding width of the types the
// metadata uses, skipping of everything else, and bounds on truncated or hostile input.

#include "device/codeobject/msgpack.hpp"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <string_view>
#include <vector>

using amd::codeobject::MsgPackReader;

namespace {

using Bytes = std::vector<unsigned char>;

MsgPackReader Reader(const Bytes& bytes) {
  return MsgPackReader(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}
//! The reader keeps a pointer into the bytes, so they must outlive it
MsgPackReader Reader(Bytes&& bytes) = delete;

//! A string of @p size 'x' with the header @p header
Bytes String(Bytes header, size_t size) {
  header.insert(header.end(), size, 'x');
  return header;
}

}  // namespace

TEST_CASE("Unsigned integers in every width", "[msgpack]") {
  struct Case {
    Bytes bytes;
    uint64_t value;
  };
  const Case cases[] = {
      {{0x00}, 0},
      {{0x7f}, 127},
      {{0xcc, 0xff}, 255},
      {{0xcd, 0x12, 0x34}, 0x1234},
      {{0xce, 0x12, 0x34, 0x56, 0x78}, 0x12345678},
      {{0xcf, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08}, 0x0102030405060708ull},
      {{0xd0, 0x7f}, 127},           // int8, not negative
      {{0xd1, 0x01, 0x00}, 256},     // int16
      {{0xd2, 0x00, 0x01, 0x00, 0x00}, 65536},
      {{0xd3, 0, 0, 0, 1, 0, 0, 0, 0}, 0x100000000ull},
  };
  for (const Case& c : cases) {
    MsgPackReader reader = Reader(c.bytes);
    uint64_t value = ~0ull;
    REQUIRE(reader.readUint(&value));
    CHECK(value == c.value);
    CHECK(reader.position() == reinterpret_cast<const char*>(c.bytes.data()) + c.bytes.size());
  }
}

TEST_CASE("Negative and non-integer values aren't unsigned", "[msgpack]") {
  for (const Bytes& bytes : {Bytes{0xff}, Bytes{0xd0, 0x80}, Bytes{0xd1, 0x80, 0x00},
                             Bytes{0xd3, 0x80, 0, 0, 0, 0, 0, 0, 0}, Bytes{0xc0},
                             Bytes{0xa1, 'x'}, Bytes{0xca, 0, 0, 0, 0}}) {
    MsgPackReader reader = Reader(bytes);
    uint64_t value = 0;
    CHECK_FALSE(reader.readUint(&value));
    // A failed read leaves the value to be skipped
    CHECK(reader.skip());
  }
}

TEST_CASE("Booleans and integers as booleans", "[msgpack]") {
  struct Case {
    Bytes bytes;
    bool ok;
    bool value;
  };
  const Case cases[] = {{{0xc3}, true, true},
                        {{0xc2}, true, false},
                        {{0x01}, true, true},
                        {{0x00}, true, false},
                        {{0xa4, 't', 'r', 'u', 'e'}, false, false}};
  for (const Case& c : cases) {
    MsgPackReader reader = Reader(c.bytes);
    bool value = !c.value;
    REQUIRE(reader.readBool(&value) == c.ok);
    if (c.ok) CHECK(value == c.value);
  }
}

TEST_CASE("Strings in every width", "[msgpack]") {
  for (const Bytes& bytes : {String({0xa0}, 0), String({0xbf}, 31), String({0xd9, 200}, 200),
                             String({0xda, 0x01, 0x00}, 256),
                             String({0xdb, 0x00, 0x01, 0x00, 0x00}, 65536)}) {
    MsgPackReader reader = Reader(bytes);
    std::string_view value;
    REQUIRE(reader.readString(&value));
    CHECK(value.size() == bytes.size() - (bytes[0] < 0xc0 ? 1 : bytes[0] == 0xd9 ? 2
                                          : bytes[0] == 0xda ? 3 : 5));
    CHECK(value.find_first_not_of('x') == std::string_view::npos);
    // The view points into the note: no copy
    CHECK(value.data() + value.size() ==
          reinterpret_cast<const char*>(bytes.data()) + bytes.size());
  }
}

TEST_CASE("Map and array headers in every width", "[msgpack]") {
  struct Case {
    Bytes bytes;
    bool map;
    uint32_t count;
  };
  const Case cases[] = {{{0x80}, true, 0},
                        {{0x8f}, true, 15},
                        {{0xde, 0x01, 0x00}, true, 256},
                        {{0xdf, 0x00, 0x01, 0x00, 0x00}, true, 65536},
                        {{0x93}, false, 3},
                        {{0xdc, 0x00, 0x10}, false, 16},
                        {{0xdd, 0x00, 0x00, 0x01, 0x00}, false, 256}};
  for (const Case& c : cases) {
    MsgPackReader reader = Reader(c.bytes);
    uint32_t count = ~0u;
    REQUIRE((c.map ? reader.readMapSize(&count) : reader.readArraySize(&count)));
    CHECK(count == c.count);
    // A map isn't an array and the other way round
    MsgPackReader other = Reader(c.bytes);
    CHECK_FALSE((c.map ? other.readArraySize(&count) : other.readMapSize(&count)));
  }
}

TEST_CASE("Skip steps over any value, including unknown metadata", "[msgpack]") {
  // {"a": [1, -1, nil, 1.5f, 2.5d, bin8 "xy", fixext1, ext8, {"b": true}], "c": 7}
  const Bytes bytes = {0x82, 0xa1, 'a', 0x99, 0x01, 0xff, 0xc0,
                       0xca, 0x3f, 0xc0, 0x00, 0x00,
                       0xcb, 0x40, 0x04, 0, 0, 0, 0, 0, 0,
                       0xc4, 0x02, 'x', 'y',
                       0xd4, 0x01, 0x00,
                       0xc7, 0x02, 0x05, 'a', 'b',
                       0x81, 0xa1, 'b', 0xc3,
                       0xa1, 'c', 0x07};
  MsgPackReader reader = Reader(bytes);
  uint32_t count = 0;
  REQUIRE(reader.readMapSize(&count));
  REQUIRE(count == 2);
  std::string_view key;
  REQUIRE(reader.readString(&key));
  CHECK(key == "a");
  REQUIRE(reader.skip());
  REQUIRE(reader.readString(&key));
  CHECK(key == "c");
  uint64_t value = 0;
  REQUIRE(reader.readUint(&value));
  CHECK(value == 7);
}

TEST_CASE("Truncated input never reads past the end", "[msgpack]") {
  const Bytes full = {0x82, 0xa1, 'a', 0x92, 0xcd, 0x12, 0x34, 0xd9, 0x03, 'x', 'y', 'z',
                      0xa1, 'c', 0xcf, 1, 2, 3, 4, 5, 6, 7, 8};
  for (size_t size = 0; size < full.size(); ++size) {
    INFO("size " << size);
    // A copy of exactly that size, so any overread is outside the buffer
    const Bytes bytes(full.begin(), full.begin() + static_cast<std::ptrdiff_t>(size));
    MsgPackReader reader = Reader(bytes);
    CHECK_FALSE(reader.skip());
  }
  MsgPackReader whole = Reader(full);
  CHECK(whole.skip());
}

TEST_CASE("Container counts larger than the input fail", "[msgpack]") {
  // An array32 claiming 4 billion elements, a str32 claiming 4 GB, a bin32 likewise
  for (const Bytes& bytes : {Bytes{0xdd, 0xff, 0xff, 0xff, 0xff, 0x01},
                             Bytes{0xdf, 0xff, 0xff, 0xff, 0xff, 0x01, 0x01},
                             Bytes{0xdb, 0xff, 0xff, 0xff, 0xff, 'x'},
                             Bytes{0xc6, 0xff, 0xff, 0xff, 0xff, 'x'}}) {
    MsgPackReader reader = Reader(bytes);
    CHECK_FALSE(reader.skip());
  }
  const Bytes hugeStr = {0xdb, 0xff, 0xff, 0xff, 0xff, 'x'};
  MsgPackReader str = Reader(hugeStr);
  std::string_view value;
  CHECK_FALSE(str.readString(&value));
}

TEST_CASE("The never-used tag 0xc1 is rejected", "[msgpack]") {
  const Bytes bytes = {0xc1};
  MsgPackReader reader = Reader(bytes);
  CHECK_FALSE(reader.skip());
}
