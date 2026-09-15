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
// the raw serial; cuid_secondary is 0444 and is what unprivileged tools
// consume. cuid_seed is deliberately absent here: it is the secret.
constexpr const char kDriverPrimaryAttribute[] = "cuid_primary";
constexpr const char kDriverSecondaryAttribute[] = "cuid_secondary";

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

// Read `attribute` (kDriverPrimaryAttribute or kDriverSecondaryAttribute) for
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

std::string read_sysfs_file(const std::string& path);
std::string readlink_bdf(const std::string& device_path);
std::string bdf_to_device_path(const std::string& bdf, amdcuid_device_type_t device_type);
std::string real_dev_path_from_fd(int fd);
std::string get_real_path(const std::string& path);
amdcuid_status_t generate_derived_cuid(const amdcuid_primary_id* primary_id,
                                       amdcuid_derived_id* derived_id, cuid_hmac* hmac);
// device_type is the enumeration, not an integer: it is written straight into
// the Component Type field, so a raw value must not be passable here.
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
//   bits  16:143  Machine ID      128 bits, from /etc/machine-id
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

// Pack the 32-octet auxiliary input structure from `input` and a machine
// identity, at the field positions documented on AuxiliaryInput above. Those
// positions are wire format: the kernel implements no auxiliary path, so
// nothing else pins them. The machine identity is an argument so the
// conformance vectors can assert the structure octet for octet against the
// aux-input row of cuid_vectors.txt.
void pack_auxiliary_input(const AuxiliaryInput& input, const uint8_t machine_id[16],
                          uint8_t out[32]);

// The auxiliary serial: the first 8 octets of the unkeyed SHA-256 of the
// 32-octet input structure, little-endian. Placed in payload bits 0:63 of an
// otherwise normal primary that has bit 117 set.
amdcuid_status_t make_fallback_fingerprint(const AuxiliaryInput& input, uint64_t& fingerprint);

// The same, with the machine identity supplied rather than read from this host
// (the overload above is this one behind read_machine_id()). For the
// conformance vectors, which need a fixed one to reproduce on any host.
amdcuid_status_t make_fallback_fingerprint(const AuxiliaryInput& input,
                                           const uint8_t machine_id[16], uint64_t& fingerprint);

// GPU VF (SR-IOV Virtual Function) utilities
int extract_render_minor(const std::string& path);
uint16_t get_gpu_vf_id(const std::string& device_path);

// Where the CUID records live: /var/lib/amdcuid, root-owned and 0755, with the
// unprivileged record 0644 and the privileged one 0600. Machine state, not
// configuration, so beside the key store's AMDCUID_CONFIG_DIR (/etc/amdcuid)
// rather than in it, and not in /tmp: the lock and temp names this library
// derives are predictable, so in a world-writable directory a local user can
// pre-create one as a symlink for the next root-privileged refresh to follow.
// Overridable at build time; an unprivileged process may also point it
// elsewhere with $AMDCUID_RECORD_DIR, ignored outright when euid is 0.
const std::string& record_dir();
const std::string& cuid_file();
const std::string& priv_cuid_file();
}  // namespace CuidUtilities

#endif
