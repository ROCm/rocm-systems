// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef HMAC_H
#define HMAC_H

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>

#include "include/amd_cuid.h"

// Length of a provisioned secret, in bytes. Not a maximum: the specification
// defines a 256-bit shared secret, and any other size is rejected as corrupt.
#define key_length 32
#define hash_length 32

// The node key lives in the UEFI variable AmdCuidKey. The library reads it,
// as root only, through an amdgpu device's cuid_seed or else straight from
// efivarfs. It writes it only in amdcuid_set_hash_key(), through the driver
// when amdgpu is loaded and to efivarfs otherwise.
constexpr char kEfivarsDir[] = "/sys/firmware/efi/efivars";
constexpr char kKeyVariablePath[] =
    "/sys/firmware/efi/efivars/AmdCuidKey-e41c1f7f-63cb-46b9-bf27-36a55f92a06d";
constexpr uint32_t kKeyVariableAttributes = 0x00000007;
constexpr size_t kKeyVariablePayloadLen = 36;
constexpr uint8_t kKeyVariableVersion = 1;
constexpr uint8_t kKeyVariableProvisioned = 0x01;

std::recursive_mutex& cuid_operation_mutex();

// HMAC-SHA-256 over the in-tree SHA-256 (see sha256.h), keyed with the node
// key or a caller-supplied one.
class cuid_hmac {
 private:
  uint8_t* key;
  size_t key_len;
  bool valid;
  bool provisioned_;
  amdcuid_status_t key_store_status_;
  // Guards key, key_len, valid, provisioned_ and key_store_status_ against
  // concurrent readers (generate_hmac_sha256, get_key_info) and writers
  // (set_hmac_key, reload_key).
  mutable std::mutex key_mutex_;

 public:
  cuid_hmac();
  cuid_hmac(uint8_t key_data[key_length]);
  // Key of an explicit length; used by tests. Does not read cuid_seed or
  // efivarfs.
  cuid_hmac(const char* key_data, size_t len);
  ~cuid_hmac();
  bool is_valid() const {
    std::lock_guard<std::mutex> lock(key_mutex_);
    return valid;
  }

  amdcuid_status_t generate_hmac_sha256(const uint8_t* data, size_t data_len, uint8_t* out_hash,
                                        size_t* out_len);
  amdcuid_status_t set_hmac_algorithm(const char* digest_name);

  // Replace the in-memory key without touching cuid_seed or efivarfs. Used by
  // tests; the library obtains the node key through reload_key().
  amdcuid_status_t set_hmac_key(const uint8_t key_data[key_length]);

  // Rediscover the node key: any amdgpu device's cuid_seed, else the efivarfs
  // variable, else none. A non-root caller never has a key, so it gets
  // temporary CUIDs for key-gated components. KEY_ERROR when the variable
  // exists but is malformed; nobody overwrites it automatically.
  amdcuid_status_t reload_key();

  // Store status, provisioned state and the first 8 octets of the unkeyed
  // SHA-256 of the key, read under one lock so a concurrent reload cannot mix
  // pre- and post-rekey state.
  amdcuid_status_t get_key_info(amdcuid_key_info_t* info) const;

  amdcuid_status_t generate_key(uint8_t key[key_length]);
};

// Unkeyed SHA-256 digest of data into a 32-byte output buffer.
//
// Namespaced for the same reason cuid::get_hash_from_raw is: this archive is
// linked into libamd_smi.so, where a name this generic at global scope invites
// a collision. Declared here rather than in cuid_util.h to keep hmac.h
// self-contained; the definition is in hmac.cc, beside the rocm::sha2 calls.
namespace CuidUtilities {
amdcuid_status_t sha256_unkeyed(const uint8_t* data, size_t data_len, uint8_t out[32]);

// Parse the efivarfs representation of AmdCuidKey: a 4-octet attribute word,
// then version (1), flags (only bit 0 defined), two reserved zero octets and
// the 32-octet key. KEY_ERROR for any other size or content.
amdcuid_status_t parse_key_variable(const uint8_t* data, size_t len, uint8_t key[key_length],
                                    bool& provisioned);

// Build the efivarfs representation of AmdCuidKey for an administrator-set key.
void build_key_variable(const uint8_t key[key_length], uint8_t out[4 + kKeyVariablePayloadLen]);

// A key that must not be provisioned: all 32 octets equal, or a public
// constant (AMD-CUID-DEFAULT-SEED-v1, AMD-CUID-TEMP-KEY-v1, the conformance
// vectors' test seed 00..1f) zero-padded to 32 octets.
bool is_rejected_key(const uint8_t key[key_length]);
}  // namespace CuidUtilities

#endif  // HMAC_H
