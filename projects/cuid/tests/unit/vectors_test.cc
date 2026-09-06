/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

// The cross-layer conformance vectors: the CUID wire format stated as values
// rather than prose. The kernel driver and this library must both reproduce
// every one of them bit for bit, or a fleet sees two different names for the
// same component.
//
// They are read from tests/vectors/cuid_vectors.txt rather than transcribed
// here, because a transcribed vector can decay in the same direction as the bug
// it was meant to catch, as the component-type check in reverse_lookup_test.cc
// did.

#include "unit/vectors_test.h"

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "src/cuid_util.h"
#include "src/hmac.h"

namespace {

struct Vector {
  std::string kind;     // primary | derived | aux-input
  std::string name;     // P-1, D-2, A-1, ...
  std::string payload;  // hex
  std::string hmac;     // hex, empty for primaries
  std::string uuid;     // rendered, empty for aux-input
};

std::string to_hex(const uint8_t* data, size_t len) {
  static const char kDigits[] = "0123456789abcdef";
  std::string out;
  out.reserve(len * 2);
  for (size_t i = 0; i < len; ++i) {
    out.push_back(kDigits[data[i] >> 4]);
    out.push_back(kDigits[data[i] & 0x0F]);
  }
  return out;
}

// Rows the file must contain. Asserted exactly, not as a lower bound: a
// truncated file would otherwise pass by not containing the row that fails.
// Bump this when a vector is added to tests/vectors/cuid_vectors.py.
constexpr size_t kExpectedVectorCount = 14;

// Where cuid_vectors.txt is. $AMDCUID_VECTORS_PATH wins; then the copy
// install() puts beside the binary, which is the only one an installed test
// has; then the compiled-in source path, which names the build machine.
std::string executable_dir() {
  char buf[4096];
  const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n <= 0) return std::string();
  buf[n] = '\0';
  const std::string exe(buf);
  const size_t slash = exe.find_last_of('/');
  if (slash == std::string::npos) return std::string();
  return exe.substr(0, slash);
}

std::vector<std::string> vectors_candidates() {
  std::vector<std::string> paths;
  // NOLINTNEXTLINE(concurrency-mt-unsafe) - nothing in this library calls setenv
  const char* env = std::getenv("AMDCUID_VECTORS_PATH");
  if (env && env[0]) {
    paths.emplace_back(env);
    return paths;
  }
  const std::string dir = executable_dir();
  if (!dir.empty()) paths.push_back(dir + "/cuid_vectors.txt");
  paths.emplace_back(AMDCUID_VECTORS_FILE);
  return paths;
}

// Split on tabs without dropping the empty fields that carry meaning: a
// primary's hmac column is empty and is still a column.
std::vector<std::string> split_tabs(const std::string& line) {
  std::vector<std::string> fields;
  size_t start = 0;
  for (;;) {
    const size_t tab = line.find('\t', start);
    if (tab == std::string::npos) {
      fields.push_back(line.substr(start));
      return fields;
    }
    fields.push_back(line.substr(start, tab - start));
    start = tab + 1;
  }
}

bool load_vectors(std::vector<Vector>& out, std::string& why) {
  const std::vector<std::string> candidates = vectors_candidates();
  std::ifstream in;
  std::string path;
  for (const auto& candidate : candidates) {
    in.open(candidate);
    if (in.is_open()) {
      path = candidate;
      break;
    }
    in.clear();
  }
  if (!in.is_open()) {
    why = "cannot open the conformance vectors; tried:";
    for (const auto& candidate : candidates) why += " " + candidate;
    return false;
  }

  std::string line;
  size_t lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    // A CRLF checkout otherwise leaves \r on the last column of every row, and
    // every comparison against it fails with two strings that print
    // identically.
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
    if (line.empty() || line[0] == '#') continue;

    const std::vector<std::string> fields = split_tabs(line);
    // aux-input has no uuid and stops at four columns; everything else has five.
    if (fields.size() < 4 || fields.size() > 5) {
      why = path + ":" + std::to_string(lineno) + ": expected 4 or 5 tab-separated fields, got " +
            std::to_string(fields.size());
      return false;
    }

    Vector v;
    v.kind = fields[0];
    v.name = fields[1];
    v.payload = fields[2];
    v.hmac = fields[3];
    v.uuid = (fields.size() == 5) ? fields[4] : std::string();

    // An unrecognised kind used to be read into the list and then never
    // matched, so a misspelled kind removed a vector's coverage silently.
    if (v.kind != "primary" && v.kind != "derived" && v.kind != "aux-input") {
      why = path + ":" + std::to_string(lineno) + ": unknown vector kind '" + v.kind + "'";
      return false;
    }
    if (v.name.empty() || v.payload.empty()) {
      why = path + ":" + std::to_string(lineno) + ": empty name or payload";
      return false;
    }
    if (v.kind == "aux-input") {
      if (!v.uuid.empty()) {
        why = path + ":" + std::to_string(lineno) + ": aux-input carries no uuid";
        return false;
      }
    } else if (v.uuid.empty()) {
      why = path + ":" + std::to_string(lineno) + ": " + v.kind + " has no uuid";
      return false;
    }

    out.push_back(v);
  }

  if (out.size() != kExpectedVectorCount) {
    why = path + ": expected " + std::to_string(kExpectedVectorCount) + " vectors, parsed " +
          std::to_string(out.size());
    return false;
  }
  return true;
}

// Hex without std::stoul, which throws on a malformed digit and takes the whole
// process down instead of failing one assertion.
bool from_hex(const std::string& hex, std::vector<uint8_t>& out) {
  if (hex.empty() || (hex.size() % 2) != 0) return false;
  out.clear();
  out.reserve(hex.size() / 2);
  for (size_t i = 0; i < hex.size(); i += 2) {
    int byte = 0;
    for (size_t nibble = 0; nibble < 2; ++nibble) {
      const char c = hex[i + nibble];
      int value;
      if (c >= '0' && c <= '9') {
        value = c - '0';
      } else if (c >= 'a' && c <= 'f') {
        value = (c - 'a') + 10;
      } else if (c >= 'A' && c <= 'F') {
        value = (c - 'A') + 10;
      } else {
        return false;
      }
      byte = (byte << 4) | value;
    }
    out.push_back(static_cast<uint8_t>(byte));
  }
  return true;
}

const Vector* find(const std::vector<Vector>& v, const std::string& kind, const std::string& name) {
  for (const auto& e : v) {
    if (e.kind == kind && e.name == name) return &e;
  }
  return nullptr;
}

// Every primary and derived vector must render to its stated UUID and survive
// the round-trip back to its stated payload.
void check_framing(const std::vector<Vector>& vectors) {
  for (const auto& v : vectors) {
    if (v.kind != "primary" && v.kind != "derived") continue;
    ASSERT_EQ(v.payload.size(), 32u) << v.name;

    std::vector<uint8_t> bytes;
    ASSERT_TRUE(from_hex(v.payload, bytes)) << "malformed payload hex: " << v.name;
    ASSERT_EQ(bytes.size(), 16u) << v.name;
    uint8_t raw[16] = {0};
    std::memcpy(raw, bytes.data(), sizeof(raw));

    amdcuid_id_t id = {{0}};
    CuidUtilities::add_UUIDv8_bits(raw, &id);
    EXPECT_EQ(CuidUtilities::get_cuid_as_string(&id), v.uuid) << "framing: " << v.name;

    // Version 8, variant 10b, per RFC 9562.
    EXPECT_EQ(id.bytes[6] >> 4, 0x8) << "version nibble: " << v.name;
    EXPECT_EQ(id.bytes[8] >> 6, 0x2) << "variant bits: " << v.name;

    uint8_t back[16] = {0};
    CuidUtilities::remove_UUIDv8_bits(&id, back);
    EXPECT_EQ(to_hex(back, sizeof(back)), v.payload) << "round-trip: " << v.name;
  }
}

// The six bits the framing discards must be exactly the padding, payload
// 122:127, and not payload 120:121, which carry the Component Type's high bits.
void check_framing_drops_only_padding() {
  amdcuid_id_t base = {{0}};
  const uint8_t zero[16] = {0};
  CuidUtilities::add_UUIDv8_bits(zero, &base);

  for (int bit = 0; bit < 128; ++bit) {
    uint8_t raw[16] = {0};
    raw[bit >> 3] = static_cast<uint8_t>(1u << (bit & 7));
    amdcuid_id_t id = {{0}};
    CuidUtilities::add_UUIDv8_bits(raw, &id);

    const bool dropped = std::memcmp(id.bytes, base.bytes, 16) == 0;
    const bool is_padding = bit >= 122;
    EXPECT_EQ(dropped, is_padding)
        << "payload bit " << bit << (is_padding ? " should be discarded" : " must be preserved");
  }
}

// 0x0-0xa and 0xf are the named Component Types; 0xb-0xe are reserved, NONE is
// not a Component Type, and nothing above 0xff can be one.
void check_device_type_validity() {
  for (unsigned value = 0x0; value <= 0xa; ++value) {
    EXPECT_TRUE(amdcuid_device_type_is_valid(static_cast<amdcuid_device_type_t>(value)))
        << "0x" << std::hex << value << " is a named Component Type";
  }
  EXPECT_TRUE(amdcuid_device_type_is_valid(AMDCUID_DEVICE_TYPE_OTHER));

  for (unsigned value = 0xb; value <= 0xe; ++value) {
    EXPECT_FALSE(amdcuid_device_type_is_valid(static_cast<amdcuid_device_type_t>(value)))
        << "0x" << std::hex << value << " is reserved, not a Component Type";
  }

  EXPECT_FALSE(amdcuid_device_type_is_valid(AMDCUID_DEVICE_TYPE_NONE));
  EXPECT_FALSE(amdcuid_device_type_is_valid(static_cast<amdcuid_device_type_t>(0x10)));
  EXPECT_FALSE(amdcuid_device_type_is_valid(static_cast<amdcuid_device_type_t>(0xFE)));
  EXPECT_FALSE(amdcuid_device_type_is_valid(static_cast<amdcuid_device_type_t>(0x1000)));
}

}  // namespace

TestVectors::TestVectors() {
  SetTitle("Conformance Vectors");
  SetDescription(
      "Reproduce the shared cross-layer CUID conformance vectors: primary "
      "packing, UUIDv8 framing, the derived fold and the auxiliary "
      "construction.");
}

void TestVectors::SetUp() {}

void TestVectors::Run() {
  std::vector<Vector> vectors;
  std::string why;
  ASSERT_TRUE(load_vectors(vectors, why)) << why;

  check_framing(vectors);
  check_framing_drops_only_padding();

  // ---- primary packing -------------------------------------------------
  struct PrimaryCase {
    const char* name;
    uint64_t serial;
    uint16_t unit_id;
    uint8_t revision_id;
    uint16_t device_id;
    uint16_t vendor_id;
    amdcuid_device_type_t type;
    bool aux;
  };
  // Every primary row in the file, and the same rows the kernel KAT asserts in
  // tools/testing/selftests/amdgpu/cuid_kat.c.
  //
  // P-1 and T-GPU are deliberately the same inputs: P-1 is the primary the
  // derived vectors D-1/D-2 are folded from, T-GPU is the GPU row of the
  // Component Type sweep. Their equality is asserted below.
  static const PrimaryCase kPrimaries[] = {
      {"P-1", 0x06C5349BD3AAABD4ULL, 0, 0x00, 0x73A3, 0x1002, AMDCUID_DEVICE_TYPE_GPU, false},
      {"P-2", 0x8E8C71777252EBFFULL, 0, 0x00, 0x73A3, 0x1002, AMDCUID_DEVICE_TYPE_GPU, false},
      {"U-1", 0x06C5349BD3AAABD4ULL, 0x0123, 0x00, 0x73A3, 0x1002, AMDCUID_DEVICE_TYPE_GPU, false},
      {"T-PLATFORM", 0x06C5349BD3AAABD4ULL, 0, 0x00, 0x73A3, 0x1002, AMDCUID_DEVICE_TYPE_PLATFORM,
       false},
      {"T-CPU", 0x06C5349BD3AAABD4ULL, 0, 0x00, 0x73A3, 0x1002, AMDCUID_DEVICE_TYPE_CPU, false},
      {"T-GPU", 0x06C5349BD3AAABD4ULL, 0, 0x00, 0x73A3, 0x1002, AMDCUID_DEVICE_TYPE_GPU, false},
      {"T-NIC", 0x06C5349BD3AAABD4ULL, 0, 0x00, 0x73A3, 0x1002, AMDCUID_DEVICE_TYPE_NIC, false},
      {"T-NPU", 0x06C5349BD3AAABD4ULL, 0, 0x00, 0x73A3, 0x1002, AMDCUID_DEVICE_TYPE_NPU, false},
      {"T-OTHER", 0x06C5349BD3AAABD4ULL, 0, 0x00, 0x73A3, 0x1002, AMDCUID_DEVICE_TYPE_OTHER, false},
  };

  for (const auto& c : kPrimaries) {
    const Vector* v = find(vectors, "primary", c.name);
    ASSERT_NE(v, nullptr) << "missing vector " << c.name;

    amdcuid_primary_id id = {};
    ASSERT_EQ(CuidUtilities::generate_primary_cuid(c.serial, c.unit_id, c.revision_id, c.device_id,
                                                   c.vendor_id, c.type, &id, c.aux),
              AMDCUID_STATUS_SUCCESS);
    EXPECT_EQ(to_hex(id.raw_bits, sizeof(id.raw_bits)), v->payload) << "payload: " << c.name;
    EXPECT_EQ(CuidUtilities::get_cuid_as_string(&id.UUIDv8_representation), v->uuid)
        << "uuid: " << c.name;
  }

  // Adding a vector and forgetting the case above must fail rather than pass
  // quietly.
  for (const auto& v : vectors) {
    if (v.kind != "primary") continue;
    bool covered = false;
    for (const auto& c : kPrimaries) {
      if (v.name == c.name) covered = true;
    }
    // A-1 is the auxiliary primary and is built from an auxiliary serial, not
    // from a fixed one; it is asserted in the auxiliary block below.
    if (v.name == "A-1") covered = true;
    EXPECT_TRUE(covered) << "primary vector " << v.name << " is in the file but not asserted";
  }

  // The same component described twice; if the generator makes them differ, one
  // of the two roles has changed meaning.
  {
    const Vector* p_1 = find(vectors, "primary", "P-1");
    const Vector* t_gpu = find(vectors, "primary", "T-GPU");
    ASSERT_NE(p_1, nullptr);
    ASSERT_NE(t_gpu, nullptr);
    EXPECT_EQ(p_1->payload, t_gpu->payload);
    EXPECT_EQ(p_1->uuid, t_gpu->uuid);
  }

  // An NPU must not render identically to a Platform. It did, for as long as
  // payload bits 120:121 were written to 126:127 and then discarded.
  {
    const Vector* npu = find(vectors, "primary", "T-NPU");
    const Vector* platform = find(vectors, "primary", "T-PLATFORM");
    ASSERT_NE(npu, nullptr);
    ASSERT_NE(platform, nullptr);
    EXPECT_NE(npu->uuid, platform->uuid);
  }

  // Every Component Type must render distinctly.
  {
    std::vector<std::string> type_uuids;
    for (const auto& c : kPrimaries) {
      if (std::strncmp(c.name, "T-", 2) != 0) continue;
      const Vector* v = find(vectors, "primary", c.name);
      ASSERT_NE(v, nullptr) << c.name;
      for (const auto& seen : type_uuids) {
        EXPECT_NE(seen, v->uuid) << "two Component Types render alike: " << c.name;
      }
      type_uuids.push_back(v->uuid);
    }
  }

  check_device_type_validity();

  // ---- derived fold ----------------------------------------------------
  const Vector* p1 = find(vectors, "primary", "P-1");
  ASSERT_NE(p1, nullptr);

  amdcuid_primary_id primary = {};
  ASSERT_EQ(CuidUtilities::generate_primary_cuid(0x06C5349BD3AAABD4ULL, 0, 0x00, 0x73A3, 0x1002,
                                                 AMDCUID_DEVICE_TYPE_GPU, &primary, false),
            AMDCUID_STATUS_SUCCESS);

  {
    // D-1: the canonical fallback seed. Its exact bytes are what make an
    // unprovisioned machine agree with the kernel.
    const Vector* v = find(vectors, "derived", "D-1");
    ASSERT_NE(v, nullptr);
    cuid_hmac h(kDefaultSeed, kDefaultSeedLen);
    ASSERT_TRUE(h.is_valid());
    amdcuid_derived_id derived = {};
    ASSERT_EQ(CuidUtilities::generate_derived_cuid(&primary, &derived, &h), AMDCUID_STATUS_SUCCESS);
    EXPECT_EQ(to_hex(derived.raw_bits, sizeof(derived.raw_bits)), v->payload) << "D-1 payload";
    EXPECT_EQ(CuidUtilities::get_cuid_as_string(&derived.UUIDv8_representation), v->uuid)
        << "D-1 uuid";
  }

  {
    // D-2: a key that is not a placeholder, pinning the fold, the framing and
    // the operand order independently of any constant that might later move.
    const Vector* v = find(vectors, "derived", "D-2");
    ASSERT_NE(v, nullptr);
    uint8_t key[key_length];
    for (size_t i = 0; i < sizeof(key); ++i) key[i] = static_cast<uint8_t>(i);
    cuid_hmac h(key);
    ASSERT_TRUE(h.is_valid());
    amdcuid_derived_id derived = {};
    ASSERT_EQ(CuidUtilities::generate_derived_cuid(&primary, &derived, &h), AMDCUID_STATUS_SUCCESS);
    EXPECT_EQ(to_hex(derived.raw_bits, sizeof(derived.raw_bits)), v->payload) << "D-2 payload";
    EXPECT_EQ(CuidUtilities::get_cuid_as_string(&derived.UUIDv8_representation), v->uuid)
        << "D-2 uuid";
    // Payload octet 14 is digest octet 13 masked to five bits: the derived slot
    // is 45 bits, not 46, because bit 117 belongs to the auxiliary marker.
    EXPECT_EQ(derived.raw_bits[14], 0x1A);
  }

  // ---- auxiliary -------------------------------------------------------
  {
    const Vector* input = find(vectors, "aux-input", "A-1");
    const Vector* aux_primary = find(vectors, "primary", "A-1");
    const Vector* aux_derived = find(vectors, "derived", "A-2");
    ASSERT_NE(input, nullptr);
    ASSERT_NE(aux_primary, nullptr);
    ASSERT_NE(aux_derived, nullptr);

    // The 32-octet auxiliary input structure, built by the implementation and
    // compared against the file, not read out of the file and fed back in. The
    // kernel has no auxiliary path, so no KAT pins these field offsets.
    //
    // The machine identity is injected: read_machine_id() reads the host's,
    // and the vector needs the generator's fixed value on every host.
    static const uint8_t kVectorMachineId[16] = {0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
                                                 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef};
    CuidUtilities::AuxiliaryInput aux_input;
    aux_input.format = CuidUtilities::kAuxFormatPcie;
    aux_input.routing_id = CuidUtilities::routing_id_from_bdf("0000:63:00.0");
    aux_input.revision_id = 0x00;
    aux_input.device_id = 0x73A3;
    aux_input.vendor_id = 0x1002;
    aux_input.component_type = static_cast<uint8_t>(AMDCUID_DEVICE_TYPE_GPU);

    uint8_t structure[32] = {0};
    CuidUtilities::pack_auxiliary_input(aux_input, kVectorMachineId, structure);
    EXPECT_EQ(to_hex(structure, sizeof(structure)), input->payload)
        << "A-1 auxiliary input structure";

    // ... and the digest of it, so the serial below is the implementation's own
    // answer rather than a number copied out of the file.
    ASSERT_EQ(input->hmac.size(), 64u);
    uint64_t serial = 0;
    ASSERT_EQ(CuidUtilities::make_fallback_fingerprint(aux_input, kVectorMachineId, serial),
              AMDCUID_STATUS_SUCCESS);

    std::vector<uint8_t> digest;
    ASSERT_TRUE(from_hex(input->hmac, digest));
    ASSERT_EQ(digest.size(), 32u);
    uint64_t expected_serial = 0;
    for (size_t i = 0; i < 8; ++i) {
      expected_serial |= static_cast<uint64_t>(digest[i]) << (8 * i);
    }
    EXPECT_EQ(serial, expected_serial) << "A-1 auxiliary serial";

    amdcuid_primary_id aux = {};
    ASSERT_EQ(CuidUtilities::generate_primary_cuid(serial, 0, 0x00, 0x73A3, 0x1002,
                                                   AMDCUID_DEVICE_TYPE_GPU, &aux, true),
              AMDCUID_STATUS_SUCCESS);
    EXPECT_EQ(to_hex(aux.raw_bits, sizeof(aux.raw_bits)), aux_primary->payload) << "A-1 payload";
    EXPECT_EQ(CuidUtilities::get_cuid_as_string(&aux.UUIDv8_representation), aux_primary->uuid)
        << "A-1 uuid";
    EXPECT_NE(aux.raw_bits[14] & 0x20, 0) << "A-1 auxiliary bit not set";

    cuid_hmac h(kTemporaryKey, kTemporaryKeyLen);
    ASSERT_TRUE(h.is_valid());
    amdcuid_derived_id derived = {};
    ASSERT_EQ(CuidUtilities::generate_derived_cuid(&aux, &derived, &h), AMDCUID_STATUS_SUCCESS);
    EXPECT_EQ(to_hex(derived.raw_bits, sizeof(derived.raw_bits)), aux_derived->payload)
        << "A-2 payload";
    EXPECT_EQ(CuidUtilities::get_cuid_as_string(&derived.UUIDv8_representation), aux_derived->uuid)
        << "A-2 uuid";
    // The marker must survive derivation, or an auxiliary value cannot be
    // recognised as one without its primary.
    EXPECT_NE(derived.raw_bits[14] & 0x20, 0) << "A-2 auxiliary bit not carried";
  }
}

void TestVectors::DisplayTestInfo() { TestBase::DisplayTestInfo(); }
void TestVectors::DisplayResults() const { TestBase::DisplayResults(); }
void TestVectors::Close() {}
