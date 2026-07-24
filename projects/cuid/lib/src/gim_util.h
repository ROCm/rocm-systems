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

#ifndef CUID_GIM_UTIL_H
#define CUID_GIM_UTIL_H

#include "include/amd_cuid.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cuid {
namespace gim {

// GIM SMI character device exposed by the GIM kernel driver.
constexpr const char *kGimSmiDevicePath = "/dev/gim-smi0";

// Mirror of struct smi_asic_info subset returned by SMI_CMD_CODE_GET_ASIC_INFO
// in the host AMD-SMI / GIM ABI. Strings are NUL-terminated.
struct GimAsicInfo {
  std::string market_name;
  uint32_t vendor_id = 0;
  std::string vendor_name;
  uint32_t subvendor_id = 0;
  uint64_t device_id = 0;
  uint32_t rev_id = 0;
  std::string asic_serial; // hexadecimal string (e.g. "0x1234ABCD")
  uint32_t oam_id = 0;
  uint32_t num_of_compute_units = 0;
  uint64_t target_graphics_version = 0;
  uint32_t subsystem_id = 0;
};

struct GimDeviceEntry {
  uint64_t dev_id = 0;     // Opaque GIM device handle.
  std::string bdf;         // Canonical "dddd:bb:dd.f" PCI BDF.
  bool failed = false;     // Reported by GIM as failed.
};

// RAII wrapper around the GIM SMI ioctl interface, used to query device info
// when sysfs is not populated (GIM SR-IOV hosts). All methods are safe to call
// when the device node is absent and return AMDCUID_STATUS_UNSUPPORTED then.
class GimClient {
public:
  GimClient();
  ~GimClient();

  GimClient(const GimClient &) = delete;
  GimClient &operator=(const GimClient &) = delete;

  // Returns true when the GIM SMI character device exists in /dev. Does not
  // require any special privilege.
  static bool is_available();

  // Open the GIM device and complete the version handshake. Subsequent calls
  // are no-ops once the connection is established. Returns:
  //   AMDCUID_STATUS_SUCCESS           - ready to issue commands
  //   AMDCUID_STATUS_UNSUPPORTED       - GIM device node not present, ioctl
  //                                      transport failed, or no negotiated
  //                                      handshake version is supported
  //   AMDCUID_STATUS_PERMISSION_DENIED - device exists but cannot be opened
  amdcuid_status_t init();

  // Returns true once init() has succeeded.
  bool is_connected() const { return fd_ >= 0 && handshake_done_; }

  // Enumerate all GPUs visible to the GIM driver. Calls init() if not already
  // connected.
  amdcuid_status_t get_devices(std::vector<GimDeviceEntry> &out);

  // Look up the GIM device handle for a given canonical PCI BDF string.
  amdcuid_status_t lookup_dev_id(const std::string &bdf, uint64_t &dev_id);

  // Query ASIC info (vendor/device/rev IDs, serial, etc.) for a GIM device
  // handle previously returned by get_devices().
  amdcuid_status_t get_asic_info(uint64_t dev_id, GimAsicInfo &info);

  // Convenience wrapper: lookup then query.
  amdcuid_status_t get_asic_info_for_bdf(const std::string &bdf,
                                         GimAsicInfo &info);

  // Parse an ASIC serial number reported by GIM as a hexadecimal string into
  // a 64-bit value. Accepts an optional "0x"/"0X" prefix and rejects empty
  // or non-hex input. Returns true on success.
  static bool parse_asic_serial(const std::string &serial, uint64_t &out);

  // Convert a GIM smi_bdf 64-bit packed value into the canonical
  // "dddd:bb:dd.f" string representation. Exposed for testing.
  static std::string format_bdf(uint64_t packed_bdf);

private:
  // Issue a single SMI ioctl. payload buffers are clamped to the SMI payload
  // size; returns AMDCUID_STATUS_INVALID_ARGUMENT if the request is too large.
  amdcuid_status_t do_ioctl(uint32_t cmd_code, const void *in, size_t in_len,
                            void *out, size_t out_len);

  int fd_ = -1;
  bool handshake_done_ = false;
};

} // namespace gim
} // namespace cuid

#endif // CUID_GIM_UTIL_H
