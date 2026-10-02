// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMD_SMI_CUID_SEED_H_
#define AMD_SMI_CUID_SEED_H_

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace amd {
namespace smi {
namespace detail {

// Whether a failed amdsmi_set_cuid_seed() left the node key changed to the
// submitted seed, from the key fingerprints read before and after the call.
// It takes a key known before the call, different after it and equal to the
// seed: an unknown prior key could already have been the key read after, and
// another administrator can change the key between the reads.
// seed_fingerprint is the seed's own fingerprint, or nullptr in a build that
// cannot compute one, where nothing shows the new key is the seed.
inline bool cuid_seed_committed(bool before_known, bool before_provisioned,
                                const uint8_t* before_fingerprint, const uint8_t* after_fingerprint,
                                const uint8_t* seed_fingerprint, size_t size) {
  if (!before_known || seed_fingerprint == nullptr) return false;
  const bool changed =
      !before_provisioned || std::memcmp(before_fingerprint, after_fingerprint, size) != 0;
  return changed && std::memcmp(after_fingerprint, seed_fingerprint, size) == 0;
}

}  // namespace detail
}  // namespace smi
}  // namespace amd

#endif  // AMD_SMI_CUID_SEED_H_
