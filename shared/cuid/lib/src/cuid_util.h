// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef CUID_UTIL_H
#define CUID_UTIL_H

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "hmac.h"
#include "include/amd_cuid.h"
#include "src/cuid_internal.h"

enum LogLevel { DEBUG, INFO, WARN, ERROR };

class Logger {
 public:
  static Logger& instance() {
    static Logger logger_;
    return logger_;
  }

  void set_level(LogLevel level) { level_ = level; }
  LogLevel level() const { return level_; }

  const char* LogLevelName(LogLevel level) const;

  void log(LogLevel level, const std::string& msg) const;

 private:
  Logger() : level_(INFO) {}
  LogLevel level_;
};

// NOLINTBEGIN(bugprone-macro-parentheses)
// `msg` is deliberately left unparenthesised. Callers pass a stream-
// continuation fragment such as `"failed: " << path`, which is only valid
// glued onto the left of `_log_stream_ <<`. Wrapping it would evaluate
// `const char[] << std::string` as an expression of its own, which does not
// compile. clang-tidy cannot see that, so the check is suppressed here rather
// than obeyed.
#define LOG(level, msg)                                  \
  do {                                                   \
    std::ostringstream _log_stream_;                     \
    _log_stream_ << msg;                                 \
    Logger::instance().log((level), _log_stream_.str()); \
  } while (0)
// NOLINTEND(bugprone-macro-parentheses)

namespace CuidUtilities {
// Thread-safe replacement for strerror(). strerror() returns a pointer into a
// static buffer, so two threads reporting errors at once can read a torn or
// wrong message. libamdcuid is linked into multithreaded hosts -- amd_smi and
// libhsa-runtime64.so among them -- so it must not use it.
std::string errno_string(int err);

// A zero hardware fingerprint is the absence of an identity, not an identity.
// Unprogrammed DSN capabilities and unconfigured MAC addresses both read back
// as all-zero, and reporting that as a successful fingerprint gives every such
// device on every machine the same primary CUID. Callers use this to convert
// "read succeeded, value is meaningless" into HW_FINGERPRINT_NOT_FOUND, which
// routes the device onto the temporary-CUID path it should have been on.
inline amdcuid_status_t validate_fingerprint(uint64_t fingerprint) {
  return (fingerprint == 0) ? AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND : AMDCUID_STATUS_SUCCESS;
}

// The CUID attributes amdgpu publishes under /sys/bus/pci/devices/<bdf>/.
// cuid_primary is 0400 and gated on CAP_SYS_ADMIN because its payload embeds
// the raw serial; cuid_derived is 0444 and is what unprivileged tools
// consume. cuid_seed is deliberately absent here: it is the secret.
constexpr const char kDriverPrimaryAttribute[] = "cuid_primary";
constexpr const char kDriverDerivedAttribute[] = "cuid_derived";

// Read a driver-published CUID attribute and parse its RFC 9562 UUID string
// into `id`. `path` is the full attribute file, so a test can point it at a
// fake sysfs root; read_driver_cuid() below is the BDF-based wrapper.
//
// The three failure modes are distinct because the caller acts on each
// differently:
//   FILE_NOT_FOUND    no attribute (pre-CUID driver, or no serial found), so
//                     there is no kernel CUID to have. Logged at DEBUG.
//   PERMISSION_DENIED present but not readable by us. The kernel still holds
//                     the authoritative value, so do not compute a rival one.
//   INVALID_FORMAT    readable, but not a UUID.
//
// Never throws: sysfs is read with open()/read() rather than an ifstream, so
// errno is meaningful and nothing escapes into hosts built without exceptions.
amdcuid_status_t read_driver_cuid_from_path(const std::string& path, amdcuid_id_t* id);

// Read `attribute` (kDriverPrimaryAttribute or kDriverDerivedAttribute) for
// the device at `bdf`, in the standard "dddd:bb:dd.f" form.
amdcuid_status_t read_driver_cuid(const std::string& bdf, const std::string& attribute,
                                  amdcuid_id_t* id);

// Write a 4-bit field into payload bits 118:121 of a 16-octet payload, leaving
// every other bit alone. The field straddles an octet boundary: payload 118:119
// are bits 6:7 of raw[14], payload 120:121 are bits 0:1 of raw[15]. Payload
// 122:127 are padding and stay zero; add_UUIDv8_bits() and remove_UUIDv8_bits()
// frame the last octet on that basis.
//
// One helper because both packers write this field (Component Type in the
// primary, reserved in the derived); do not open-code it in either. Putting the
// high half in the padding instead renders the Component Type modulo 4, making
// an NPU (0x4) indistinguishable from a Platform (0x0), and the derived field
// is reserved and so always zero today, which is exactly why a second copy of
// this packing can drift without any test noticing.
inline void pack_component_type_bits(uint8_t value, uint8_t raw_bits[16]) {
  raw_bits[14] = static_cast<uint8_t>((raw_bits[14] & 0x3F) | ((value & 0x3) << 6));
  raw_bits[15] = static_cast<uint8_t>((raw_bits[15] & 0xFC) | ((value & 0xC) >> 2));
}

// Whether this process holds CAP_SYS_ADMIN in its effective set. amdgpu gates
// reading cuid_primary and cuid_seed on it, so root without it (a default
// container) is unprivileged as far as the driver is concerned.
bool has_cap_sys_admin();

std::string read_sysfs_file(const std::string& path);
std::string readlink_bdf(const std::string& device_path);
std::string bdf_to_device_path(const std::string& bdf, amdcuid_device_type_t device_type);
std::string real_dev_path_from_fd(int fd);
std::string get_real_path(const std::string& path);
amdcuid_status_t generate_derived_cuid(const amdcuid_primary_id* primary_id,
                                       amdcuid_derived_id* derived_id, cuid_hmac* hmac);
// device_type is the enumeration, not an integer: it is written straight into
// the Component Type field, so a raw value must not be passable here.
// UnitID > 0x1FFF returns INVALID_ARGUMENT without modifying primary_id.
amdcuid_status_t generate_primary_cuid(uint64_t serial_number, uint16_t unit_id,
                                       uint8_t revision_id, uint16_t device_id, uint16_t vendor_id,
                                       amdcuid_device_type_t device_type,
                                       amdcuid_primary_id* primary_id, bool temp = false);
void remove_UUIDv8_bits(amdcuid_id_t* id, uint8_t out_raw_bits[16]);

// True when `id` was constructed from the 122-bit CUID payload, and so its
// fields can be decoded; false when it was adopted verbatim from firmware.
//
// The Platform CUID is the exception: where firmware supplies a system UUID,
// that UUID is the identifier, carrying whatever version bits firmware wrote
// (1, 3 or 4 in practice, never 8). So the version nibble is the discriminator
// and must be checked before any field is read.
bool is_constructed(const amdcuid_id_t* id);
void add_UUIDv8_bits(const uint8_t raw_bits[16], amdcuid_id_t* id);
std::string get_cuid_as_string(const amdcuid_id_t* id);
amdcuid_status_t uuid_string_to_uint8(const std::string& uuid_str, uint8_t* uuid);
std::string device_type_to_string(amdcuid_device_type_t type);

bool is_valid_bdf(const std::string& bdf);

// Format field of the auxiliary input structure (bits 0:15).
constexpr uint16_t kAuxFormatPcie = 1;
constexpr uint16_t kAuxFormatCpu = 2;

// The auxiliary CUID's 256-bit input structure, per the CUID specification but
// with two boundaries repaired so the widths sum to 256: the published table
// gives Format 0-16 and Machine ID 17-143, which overlap and total 257.
//
//   bits   0:15   Format          1 = PCIe device, 2 = CPU
//   bits  16:143  Machine ID      zero; the machine-id keys the HMAC instead
//   bits 144:175  PCIe Routing ID (segment<<16)|(bus<<8)|(device<<3)|function
//   bits 176:183  RevisionID      CPU: stepping
//   bits 184:199  DeviceID        CPU: family and model
//   bits 200:215  VendorID
//   bits 216:219  Component Type  on-wire numbering
//   bits 220:255  Reserved, zero
//
// Fixed-width binary, not a string: stripping non-hex characters out of a
// rendered string erases the separators, so "0000:65:00.0" and "0000:65:0:00.0"
// collapse to the same input.
struct AuxiliaryInput {
  uint16_t format = 0;
  uint32_t routing_id = 0;
  uint8_t revision_id = 0;
  uint16_t device_id = 0;
  uint16_t vendor_id = 0;
  uint8_t component_type = 0;
};

// Pack "dddd:bb:dd.f" into the 32-bit Routing ID. Returns 0 for a malformed
// BDF, which is_valid_bdf() should have rejected already.
uint32_t routing_id_from_bdf(const std::string& bdf);

// Pack the 32-octet auxiliary input structure from `input`, at the field
// positions documented on AuxiliaryInput above. Those positions are wire
// format: the kernel implements no auxiliary path, so nothing else pins them.
void pack_auxiliary_input(const AuxiliaryInput& input, uint8_t out[32]);

// Label of the temporary-CUID application key, 16 ASCII octets, no NUL.
constexpr char kTemporaryKeyLabel[] = "AMD-CUID-TEMP-v2";

// K_app = HMAC-SHA256(key = machine-id as 16 octets, msg = kTemporaryKeyLabel).
// Keys both the auxiliary serial and the derived temporary CUID, so neither
// exposes the machine-id itself (machine-id(5)). HW_FINGERPRINT_NOT_FOUND when
// this host has no machine-id.
amdcuid_status_t temporary_key(uint8_t out[32]);
void temporary_key(const uint8_t machine_id[16], uint8_t out[32]);

// The auxiliary serial: the first 8 octets of HMAC-SHA256(K_app, the 32-octet
// input structure), little-endian. Placed in payload bits 0:63 of an otherwise
// normal primary that has bit 117 set.
amdcuid_status_t make_fallback_fingerprint(const AuxiliaryInput& input, uint64_t& fingerprint);

// The same, with the machine identity supplied rather than read from this host,
// for the conformance vectors, which need a fixed one to reproduce on any host.
amdcuid_status_t make_fallback_fingerprint(const AuxiliaryInput& input,
                                           const uint8_t machine_id[16], uint64_t& fingerprint);

// The largest value the 13-bit UnitID field holds. A larger index must be
// refused rather than masked: masking sends 0x2000 to 0, which means "the
// component as a whole", so a sub-unit would answer with its parent's
// identifier.
constexpr uint16_t kMaxUnitId = 0x1FFF;

// GPU VF (SR-IOV Virtual Function) utilities
int extract_render_minor(const std::string& path);

// What a device turned out to be, as far as SR-IOV is concerned. The two
// questions are separate because the answers are used differently.
//
// On a host a VF's physfn link is visible, and its index says which share of the
// card this is, so the VF is a sub-unit: the card's serial plus that index names
// it, exactly as a partition index does.
//
// In a guest there is no physfn link and the index is unknowable. What is
// reachable there -- the PCIe Device Serial Number, if the VF exposes one at all
// -- belongs to the physical card, and nothing distinguishes it from the card's
// own identity, so every guest sharing that card would publish the same one and
// it would be the host's. The kernel declines to publish a CUID on a VF for this
// reason; this structure is how the library declines.
struct VfIdentity {
  bool is_vf = false;        // the device reports itself as a virtual function
  bool index_known = false;  // ... and we could determine which one
  uint16_t unit_id = 0;      // 1-based index, or 0 when not a VF / unknown
};

VfIdentity get_gpu_vf_identity(const std::string& device_path);
}  // namespace CuidUtilities

#endif
