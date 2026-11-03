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

// amdgpu publishes a component's UnitID, decimal, in cuid_unit_id under
// /sys/bus/pci/devices/<bdf>/ and under each partition's xcp directory. It is
// 0444, so any caller can read it.
constexpr const char kDriverUnitIdAttribute[] = "cuid_unit_id";

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
// reading cuid_primary on it, so root without it (a default
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

// The temporary CUID's 256-bit input structure S, per the CUID specification
// but with two boundaries repaired so the widths sum to 256: the published
// table gives Format 0-16 and Machine ID 17-143, which overlap and total 257.
// Packed LSB-first, each field little-endian.
//
//   bits   0:15   Format          1 = PCIe device, 2 = CPU or platform
//   bits  16:143  Machine ID      the 16 machine-id octets in file order
//   bits 144:175  PCIe Routing ID (segment<<16)|(bus<<8)|(device<<3)|function
//   bits 176:183  RevisionID      CPU: stepping
//   bits 184:199  DeviceID        CPU: family and model
//   bits 200:215  VendorID
//   bits 216:219  Component Type  on-wire numbering
//   bits 220:223  Reserved, zero
//   bits 224:239  UnitID          13 bits, 237:239 zero; the partition or
//                                 function, which a shared BDF cannot tell apart
//   bits 240:255  Segment high    PCI segment >> 16, zero below 0x10000 (VMD
//                                 domains start at 0x10000)
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
  uint16_t unit_id = 0;
  uint16_t segment_high = 0;
};

// Pack "dddd:bb:dd.f" into the 32-bit Routing ID, the segment's low 16 bits
// included. Returns 0 for a malformed BDF, which is_valid_bdf() should have
// rejected already.
uint32_t routing_id_from_bdf(const std::string& bdf);

// The segment's bits 16:31, which the Routing ID has no room for.
uint16_t segment_high_from_bdf(const std::string& bdf);

// Routing ID and segment high of `bdf` into `input`.
void set_pcie_routing(AuxiliaryInput& input, const std::string& bdf);

// Pack S from `input` and `machine_id`, at the field positions documented on
// AuxiliaryInput above. Those positions are wire format: the kernel implements
// no temporary path, so nothing else pins them.
void pack_auxiliary_input(const AuxiliaryInput& input, const uint8_t machine_id[16],
                          uint8_t out[32]);

// The CUID application id, UUIDv5(UUIDv5(NAMESPACE_DNS, "amd.com"),
// "com.amd.cuid.v1") = deeda863-4023-5bac-865a-9923e61370d7, as 16 octets.
constexpr uint8_t kCuidApplicationId[16] = {0xde, 0xed, 0xa8, 0x63, 0x40, 0x23, 0x5b, 0xac,
                                            0x86, 0x5a, 0x99, 0x23, 0xe6, 0x13, 0x70, 0xd7};

// One line of a machine-id file, newline removed, as 16 octets. False unless it
// is exactly 32 hex digits and not all zero, which machine-id(5) forbids, so a
// host with systemd's "uninitialized" placeholder or a zeroed image takes the
// no-machine-id error path instead of sharing temporary CUIDs with its clones.
bool parse_machine_id(const std::string& line, uint8_t out[16]);

// The machine-id from `path`, or from `fallback` only when `path` does not
// exist. False when the file read holds no valid machine-id.
bool read_machine_id(const std::string& path, const std::string& fallback, uint8_t out[16]);

// The machine-id from /etc/machine-id, else /var/lib/dbus/machine-id, or from
// `path`/`fallback`. HW_FINGERPRINT_NOT_FOUND, warned once per process, when
// there is none: a temporary CUID without it would be the same on every
// identically-configured host.
amdcuid_status_t host_machine_id(uint8_t out[16]);
amdcuid_status_t host_machine_id(const std::string& path, const std::string& fallback,
                                 uint8_t out[16]);

// The temporary derived CUID: HMAC-SHA256(key = S, msg = kCuidApplicationId)
// truncated to 16 octets with the RFC 9562 version 8 and variant 10b bits
// overwritten and payload bit 117, the Auxiliary Value Identifier (rendered
// octet 15 bit 7), set. There is no temporary primary.
void temporary_cuid(const AuxiliaryInput& input, const uint8_t machine_id[16],
                    amdcuid_derived_id* out);
amdcuid_status_t temporary_cuid(const AuxiliaryInput& input, amdcuid_derived_id* out);

// The largest value the 13-bit UnitID field holds. A larger index must be
// refused rather than masked: masking sends 0x2000 to 0, which means "the
// component as a whole", so a sub-unit would answer with its parent's
// identifier.
constexpr uint16_t kMaxUnitId = 0x1FFF;

// The UnitID of partition k of a GPU split into n >= 2 partitions:
// n*(n-1)/2 + 1 + k. UnitID 0 is the whole device and 1 a whole device in
// SLC mode, so a partition's is at least 2; n is at most 127.
constexpr uint16_t partition_unit_id(unsigned n, unsigned k) {
  return static_cast<uint16_t>(n * (n - 1) / 2 + 1 + k);
}

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
