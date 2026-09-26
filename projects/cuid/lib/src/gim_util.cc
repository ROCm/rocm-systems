// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gim_util.h"

#include "cuid_util.h"

#include <cctype>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace cuid {
namespace gim {

namespace {

// GIM SMI ABI mirror types. These mirror the binary layout of the host
// AMD-SMI / GIM smi_cmd_def.h and smi_cmd.h types, duplicated here to avoid a
// build dependency on the GIM driver source. The static_asserts guard against
// ABI drift.

constexpr size_t kSmiMaxStringLength = 256;
constexpr size_t kSmiMaxPayload = 1024;          // uint32_t entries
constexpr size_t kSmiMaxPayloadBytes = kSmiMaxPayload * sizeof(uint32_t);
constexpr size_t kSmiMaxDevices = 32;

// ioctl base type for SMI commands: gim_ioctl_type::SMI_IOCTL = (1 << 24).
constexpr uint32_t kSmiIoctlBase = 0x01000000u;

// Command codes (subset). Values come from enum smi_cmd_code.
constexpr uint32_t kSmiCmdCodeHandshake = kSmiIoctlBase | 0x01u;
constexpr uint32_t kSmiCmdCodeGetServerStaticInfo = kSmiIoctlBase | 0x02u;
constexpr uint32_t kSmiCmdCodeGetAsicInfo = kSmiIoctlBase | 0x2Du;

// SMI handshake versions; we negotiate the highest version we know about and
// fall back through earlier versions on rejection.
constexpr uint32_t kSmiVersionBeta4 = 0x00000007u;
constexpr uint32_t kSmiVersionBeta3 = 0x00000006u;
constexpr uint32_t kSmiVersionBeta2 = 0x00000005u;
constexpr uint32_t kSmiVersionBeta1 = 0x00000004u;
constexpr uint32_t kSmiVersionBeta0 = 0x00000003u;
constexpr uint32_t kSmiVersionAlpha0 = 0x00000002u;

constexpr uint32_t kHandshakeVersions[] = {
    kSmiVersionBeta4, kSmiVersionBeta3, kSmiVersionBeta2,
    kSmiVersionBeta1, kSmiVersionBeta0, kSmiVersionAlpha0};

// _IOWR('S', 0, struct smi_ioctl_cmd). The struct size is encoded in the
// ioctl number, so we use the same total wire size (4108 bytes) the GIM
// driver expects for its smi_ioctl_cmd transport.
constexpr size_t kSmiIoctlCmdSize = 4 + 2 + 2 + 4 + kSmiMaxPayloadBytes;
constexpr unsigned long kSmiIoctlCmd =
    _IOC(_IOC_READ | _IOC_WRITE, 'S', 0, kSmiIoctlCmdSize);

#pragma pack(push, 1)
struct SmiInHdrWire {
  uint32_t code;
  int16_t in_len;
  int16_t out_len;
};
struct SmiOutHdrWire {
  int32_t status;
};
#pragma pack(pop)
static_assert(sizeof(SmiInHdrWire) == 8, "SmiInHdrWire size mismatch");
static_assert(sizeof(SmiOutHdrWire) == 4, "SmiOutHdrWire size mismatch");

// SMI ioctl wire frame: header + payload buffer.
struct SmiIoctlCmdWire {
  SmiInHdrWire in_hdr;
  SmiOutHdrWire out_hdr;
  uint8_t payload[kSmiMaxPayloadBytes];
};
static_assert(sizeof(SmiIoctlCmdWire) == kSmiIoctlCmdSize,
              "SmiIoctlCmdWire size mismatch");

// SMI handshake input/output payload (struct smi_handshake).
struct SmiHandshakeWire {
  uint32_t version;
};
static_assert(sizeof(SmiHandshakeWire) == 4, "SmiHandshakeWire size mismatch");

// struct smi_device_handle_t (smi_device_handle.h). The GIM SMI ABI identifies
// a device by a 16-byte handle: an opaque 64-bit handle plus the 64-bit PCI
// device id. Modeling this as a bare uint64_t (as an earlier revision did)
// under-sizes every consumer of the handle and corrupts the device array.
struct SmiDeviceHandleWire {
  uint64_t handle;
  uint64_t device_id;
};
static_assert(sizeof(SmiDeviceHandleWire) == 16,
              "SmiDeviceHandleWire size mismatch");

// One entry of struct smi_server_static_info::devices[] (Linux, natural
// alignment). dev_id is a 16-byte smi_device_handle_t, so each entry is 40
// bytes.
struct SmiServerDeviceWire {
  SmiDeviceHandleWire dev_id;
  uint64_t bdf;     // union smi_bdf packed 64-bit value
  uint8_t failed;
  uint8_t padding[3];
  uint32_t reserved[3];
};
static_assert(sizeof(SmiServerDeviceWire) == 40,
              "SmiServerDeviceWire size mismatch");

// struct smi_server_static_info (Linux, natural alignment). The trailing
// reserved area expands the struct to the full 4096-byte SMI payload; the GIM
// driver validates out_len against this exact size, so it must be 4096.
struct SmiServerStaticInfoWire {
  uint32_t debug_level;
  uint32_t num_devices;
  SmiServerDeviceWire devices[kSmiMaxDevices];
  uint64_t reserved[351];
};
static_assert(sizeof(SmiServerStaticInfoWire) == 4096,
              "SmiServerStaticInfoWire size mismatch");
static_assert(sizeof(SmiServerStaticInfoWire) <= kSmiMaxPayloadBytes,
              "SmiServerStaticInfoWire larger than SMI payload");

// struct smi_device_info (input for GET_ASIC_INFO). Holds a full 16-byte
// smi_device_handle_t.
struct SmiDeviceInfoWire {
  SmiDeviceHandleWire dev_id;
};
static_assert(sizeof(SmiDeviceInfoWire) == 16,
              "SmiDeviceInfoWire size mismatch");

// struct smi_asic_info (Linux, natural alignment).
struct SmiAsicInfoWire {
  char market_name[kSmiMaxStringLength];
  uint32_t vendor_id;
  char vendor_name[kSmiMaxStringLength];
  uint32_t subvendor_id;
  // device_id is naturally 8-aligned at offset 520, so no padding is needed
  // here. (vendor_name ends at 516, subvendor_id occupies 516..520.)
  uint64_t device_id;
  uint32_t rev_id;
  char asic_serial[kSmiMaxStringLength];
  uint32_t oam_id;
  uint32_t num_of_compute_units;
  // target_graphics_version is 8-byte aligned. num_of_compute_units ends at
  // offset 796, so the compiler inserts 4 bytes of padding before the next
  // uint64_t. Modeling that pad explicitly keeps the layout deterministic.
  uint32_t _pad1;
  uint64_t target_graphics_version;
  uint32_t subsystem_id;
  uint32_t reserved[21];
};
static_assert(offsetof(SmiAsicInfoWire, vendor_id) == 256,
              "SmiAsicInfoWire.vendor_id offset mismatch");
static_assert(offsetof(SmiAsicInfoWire, device_id) == 520,
              "SmiAsicInfoWire.device_id offset mismatch");
static_assert(offsetof(SmiAsicInfoWire, asic_serial) == 532,
              "SmiAsicInfoWire.asic_serial offset mismatch");
static_assert(offsetof(SmiAsicInfoWire, target_graphics_version) == 800,
              "SmiAsicInfoWire.target_graphics_version offset mismatch");
static_assert(sizeof(SmiAsicInfoWire) <= kSmiMaxPayloadBytes,
              "SmiAsicInfoWire larger than SMI payload");

// Copy a C string field bounded to its array length, ensuring NUL-termination.
std::string fixed_string_to_std(const char *src, size_t cap) {
  size_t n = 0;
  while (n < cap && src[n] != '\0') {
    ++n;
  }
  return std::string(src, n);
}

bool is_canonical_bdf(const std::string &bdf) {
  return bdf.size() == 12 && bdf[4] == ':' && bdf[7] == ':' && bdf[10] == '.';
}

} // namespace

// ----------------------------------------------------------------------------
// GimClient
// ----------------------------------------------------------------------------

GimClient::GimClient() = default;

GimClient::~GimClient() {
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

bool GimClient::is_available() {
  struct stat st {};
  if (::stat(kGimSmiDevicePath, &st) != 0) {
    return false;
  }
  return S_ISCHR(st.st_mode);
}

amdcuid_status_t GimClient::init() {
  if (handshake_done_) {
    return AMDCUID_STATUS_SUCCESS;
  }
  if (!is_available()) {
    return AMDCUID_STATUS_UNSUPPORTED;
  }

  if (fd_ < 0) {
    int fd = ::open(kGimSmiDevicePath, O_RDWR | O_CLOEXEC);
    if (fd < 0) {
      const int err = errno;
      LOG(DEBUG, "GIM: open(" << kGimSmiDevicePath
                              << ") failed: " << std::strerror(err));
      if (err == EACCES || err == EPERM) {
        return AMDCUID_STATUS_PERMISSION_DENIED;
      }
      return AMDCUID_STATUS_UNSUPPORTED;
    }
    fd_ = fd;
  }

  // Negotiate the highest handshake version the driver accepts.
  for (uint32_t version : kHandshakeVersions) {
    SmiHandshakeWire hs{version};
    SmiHandshakeWire resp{};
    amdcuid_status_t st = do_ioctl(kSmiCmdCodeHandshake, &hs, sizeof(hs), &resp,
                                   sizeof(resp));
    if (st == AMDCUID_STATUS_SUCCESS) {
      handshake_done_ = true;
      LOG(DEBUG, "GIM: handshake succeeded with version 0x" << std::hex
                                                            << version);
      return AMDCUID_STATUS_SUCCESS;
    }
  }

  LOG(WARN, "GIM: handshake failed for all known SMI versions");
  ::close(fd_);
  fd_ = -1;
  return AMDCUID_STATUS_UNSUPPORTED;
}

amdcuid_status_t GimClient::do_ioctl(uint32_t cmd_code, const void *in,
                                     size_t in_len, void *out, size_t out_len) {
  if (fd_ < 0) {
    return AMDCUID_STATUS_UNSUPPORTED;
  }
  if (in_len > kSmiMaxPayloadBytes || out_len > kSmiMaxPayloadBytes) {
    return AMDCUID_STATUS_INVALID_ARGUMENT;
  }

  SmiIoctlCmdWire cmd;
  std::memset(&cmd, 0, sizeof(cmd));
  cmd.in_hdr.code = cmd_code;
  cmd.in_hdr.in_len = static_cast<int16_t>(in_len);
  cmd.in_hdr.out_len = static_cast<int16_t>(out_len);
  if (in != nullptr && in_len > 0) {
    std::memcpy(cmd.payload, in, in_len);
  }

  if (::ioctl(fd_, kSmiIoctlCmd, &cmd) != 0) {
    const int err = errno;
    LOG(DEBUG, "GIM: ioctl(cmd=0x" << std::hex << cmd_code
                                   << ") failed: " << std::strerror(err));
    if (err == EACCES || err == EPERM) {
      return AMDCUID_STATUS_PERMISSION_DENIED;
    }
    return AMDCUID_STATUS_UNSUPPORTED;
  }
  if (cmd.out_hdr.status != 0) {
    LOG(DEBUG, "GIM: SMI cmd 0x" << std::hex << cmd_code
                                 << " returned status " << std::dec
                                 << cmd.out_hdr.status);
    return AMDCUID_STATUS_UNSUPPORTED;
  }
  if (out != nullptr && out_len > 0) {
    std::memcpy(out, cmd.payload, out_len);
  }
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t GimClient::get_devices(std::vector<GimDeviceEntry> &out) {
  out.clear();
  amdcuid_status_t st = init();
  if (st != AMDCUID_STATUS_SUCCESS) {
    return st;
  }

  SmiServerStaticInfoWire info;
  std::memset(&info, 0, sizeof(info));
  st = do_ioctl(kSmiCmdCodeGetServerStaticInfo, nullptr, 0, &info,
                sizeof(info));
  if (st != AMDCUID_STATUS_SUCCESS) {
    return st;
  }

  uint32_t n = info.num_devices;
  if (n > kSmiMaxDevices) {
    n = kSmiMaxDevices;
  }
  out.reserve(n);
  for (uint32_t i = 0; i < n; ++i) {
    GimDeviceEntry e;
    e.dev_id = info.devices[i].dev_id.handle;
    e.bdf = format_bdf(info.devices[i].bdf);
    e.failed = info.devices[i].failed != 0;
    out.push_back(std::move(e));
  }
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t GimClient::lookup_dev_id(const std::string &bdf,
                                          uint64_t &dev_id) {
  if (!is_canonical_bdf(bdf)) {
    return AMDCUID_STATUS_INVALID_ARGUMENT;
  }
  std::vector<GimDeviceEntry> devices;
  amdcuid_status_t st = get_devices(devices);
  if (st != AMDCUID_STATUS_SUCCESS) {
    return st;
  }
  for (const auto &e : devices) {
    if (e.bdf == bdf) {
      dev_id = e.dev_id;
      return AMDCUID_STATUS_SUCCESS;
    }
  }
  return AMDCUID_STATUS_DEVICE_NOT_FOUND;
}

amdcuid_status_t GimClient::get_asic_info(uint64_t dev_id, GimAsicInfo &info) {
  amdcuid_status_t st = init();
  if (st != AMDCUID_STATUS_SUCCESS) {
    return st;
  }

  SmiDeviceInfoWire req{{dev_id, 0}};
  SmiAsicInfoWire raw;
  std::memset(&raw, 0, sizeof(raw));
  st = do_ioctl(kSmiCmdCodeGetAsicInfo, &req, sizeof(req), &raw, sizeof(raw));
  if (st != AMDCUID_STATUS_SUCCESS) {
    return st;
  }

  info.market_name = fixed_string_to_std(raw.market_name, kSmiMaxStringLength);
  info.vendor_id = raw.vendor_id;
  info.vendor_name = fixed_string_to_std(raw.vendor_name, kSmiMaxStringLength);
  info.subvendor_id = raw.subvendor_id;
  info.device_id = raw.device_id;
  info.rev_id = raw.rev_id;
  info.asic_serial =
      fixed_string_to_std(raw.asic_serial, kSmiMaxStringLength);
  info.oam_id = raw.oam_id;
  info.num_of_compute_units = raw.num_of_compute_units;
  info.target_graphics_version = raw.target_graphics_version;
  info.subsystem_id = raw.subsystem_id;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t GimClient::get_asic_info_for_bdf(const std::string &bdf,
                                                  GimAsicInfo &info) {
  uint64_t dev_id = 0;
  amdcuid_status_t st = lookup_dev_id(bdf, dev_id);
  if (st != AMDCUID_STATUS_SUCCESS) {
    return st;
  }
  return get_asic_info(dev_id, info);
}

bool GimClient::parse_asic_serial(const std::string &serial, uint64_t &out) {
  if (serial.empty()) {
    return false;
  }
  // Accept optional 0x/0X prefix.
  size_t pos = 0;
  if (serial.size() > 2 && serial[0] == '0' &&
      (serial[1] == 'x' || serial[1] == 'X')) {
    pos = 2;
  }
  if (pos >= serial.size()) {
    return false;
  }
  // Validate hex digits and bound length to 16 hex chars (64 bits). Reject
  // anything longer to avoid silently truncating real data.
  if (serial.size() - pos > 16) {
    return false;
  }
  for (size_t i = pos; i < serial.size(); ++i) {
    if (!std::isxdigit(static_cast<unsigned char>(serial[i]))) {
      return false;
    }
  }
  try {
    out = std::stoull(serial.substr(pos), nullptr, 16);
  } catch (...) {
    return false;
  }
  return true;
}

std::string GimClient::format_bdf(uint64_t packed_bdf) {
  // union smi_bdf bit layout (LSB first):
  //   function_number : 3
  //   device_number   : 5
  //   bus_number      : 8
  //   domain_number   : 48
  const uint32_t function = static_cast<uint32_t>(packed_bdf & 0x7u);
  const uint32_t device = static_cast<uint32_t>((packed_bdf >> 3) & 0x1Fu);
  const uint32_t bus = static_cast<uint32_t>((packed_bdf >> 8) & 0xFFu);
  const uint32_t domain = static_cast<uint32_t>((packed_bdf >> 16) & 0xFFFFu);
  char buf[16];
  std::snprintf(buf, sizeof(buf), "%04x:%02x:%02x.%u", domain, bus, device,
                function);
  return std::string(buf);
}

} // namespace gim
} // namespace cuid
