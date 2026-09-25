// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef CUID_DEVICE_H
#define CUID_DEVICE_H

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <tuple>

#include "include/amd_cuid.h"
#include "src/cuid_internal.h"
#include "src/hmac.h"

namespace cuid {

// Unpack the 109-bit hash out of a derived CUID's 16 raw payload octets.
// Shared by every path that reconstitutes a derived ID it did not compute
// itself (the driver interface, the index-building pass below), so all of
// them recover the same hash from the same value.
//
// Namespaced because this archive is linked into libamd_smi.so, where a name
// this generic at global scope invites a collision.
void get_hash_from_raw(uint8_t raw_bytes[16], uint8_t out_hash[14]);

}  // namespace cuid

class CuidDevice;

// A device's identity as far as duplicate/alias detection needs: enough to
// route it (device_type, bdf, device_node) and to tell two entries for the
// same physical NIC apart from two that merely collided. CuidDeviceManager
// builds these from live devices on every index rebuild.
struct CuidDeviceEntry {
  amdcuid_device_type_t device_type = AMDCUID_DEVICE_TYPE_NONE;

  amdcuid_id_t primary_cuid{};
  amdcuid_id_t derived_cuid{};

  uint16_t vendor_id = 0;
  uint16_t device_id = 0;
  uint8_t revision_id = 0;
  uint16_t unit_id = 0xFFFF;
  std::string device_node;
  std::string bdf;

  bool is_temporary = false;

  CuidDeviceEntry() {
    std::memset(primary_cuid.bytes, 0, sizeof(primary_cuid.bytes));
    std::memset(derived_cuid.bytes, 0, sizeof(derived_cuid.bytes));
  }
};

namespace cuid {
using DeviceRoute = std::tuple<amdcuid_device_type_t, std::string, std::string>;
DeviceRoute device_route(const CuidDeviceEntry& entry);
// Whether `a` and `b` are two functions of the same physical NIC rather than a
// real collision: same BDF prefix, same PCI identity, same non-auxiliary
// derived CUID and, when `require_primary`, the same whole-NIC primary CUID
// too. `require_primary` is set for a root caller, which can see primaries;
// a non-root one is judged on the derived value alone.
bool nic_component_alias(const CuidDeviceEntry& a, const CuidDeviceEntry& b, bool require_primary);
}  // namespace cuid

class CuidDevice {
 public:
  virtual ~CuidDevice() = default;
  virtual amdcuid_device_type_t type() const = 0;
  virtual amdcuid_status_t get_primary_cuid(amdcuid_primary_id& id) const = 0;
  virtual amdcuid_status_t get_hardware_fingerprint(uint64_t& fingerprint) const = 0;

  // Whether this component's derived CUID needs the node key: true for CPU,
  // NIC, NPU and Platform. A GPU is named by its driver or is temporary.
  virtual bool key_gated_identity() const { return false; }

  // The temporary primary a key-gated component takes when there is no key,
  // the same one it takes without a hardware fingerprint.
  virtual amdcuid_status_t get_auxiliary_primary_cuid(amdcuid_primary_id& id) const {
    (void)id;
    return AMDCUID_STATUS_UNSUPPORTED;
  }

  // The driver's cuid_derived, else this library's computation. Where the
  // kernel answers, it wins. Virtual: a spatial GPU partition answers from the
  // driver alone.
  virtual amdcuid_status_t get_derived_cuid(amdcuid_derived_id& id,
                                            cuid_hmac* hmac = nullptr) const;

  // Whether this component's identity is auxiliary. The marker is carried by
  // the primary and copied into the derived value unchanged, so reading it
  // needs no HMAC; `hmac` is only consulted to decide, for a key-gated
  // component, whether get_primary_cuid() or get_auxiliary_primary_cuid()
  // is the one whose primary answers -- the same choice get_derived_cuid()
  // makes, so the two agree on the same component.
  amdcuid_status_t is_temporary_cuid(bool* is_temporary, cuid_hmac* hmac = nullptr) const;

  // The stage that produced the last successful derivation;
  // AMDCUID_SOURCE_UNKNOWN before one and after a failed one.
  amdcuid_source_t derived_source() const { return last_source_; }

  // Stage 1 of the staged lookup: read `attribute` (one of
  // CuidUtilities::kDriverPrimaryAttribute / kDriverDerivedAttribute) for
  // this device's BDF, where it has one.
  //
  // On the base class rather than on CuidGpu because the interface is a
  // property of the PCI device, not of amdgpu: a NIC or NPU whose driver grows
  // the same attributes is then answered by the kernel with no change here.
  //
  // AMDCUID_STATUS_UNSUPPORTED when the driver publishes nothing, or the device
  // has no BDF, and the caller should move on to the later stages. A
  // present-but-unreadable attribute is PERMISSION_DENIED rather than an
  // absence: the kernel holds the authoritative value, so computing a local one
  // manufactures the divergence this ordering exists to prevent.
  amdcuid_status_t read_driver_published(const std::string& attribute, amdcuid_id_t& out,
                                         uint8_t raw_bits[16]) const;

  // Where this component's driver-published attributes live. The default is
  // /sys/bus/pci/devices/<bdf>/<attribute>; a spatial partition overrides it,
  // since its BDF is its parent's. UNSUPPORTED when there is nothing to read.
  virtual amdcuid_status_t driver_attribute_path(const std::string& attribute,
                                                 std::string& path) const;

  // Stage 1 for the primary specifically. Returns AMDCUID_STATUS_UNSUPPORTED
  // when the caller should carry on to the later stages, including when the
  // driver publishes no cuid_derived beside cuid_primary, and any other failure
  // verbatim, notably PERMISSION_DENIED on cuid_primary, which is
  // CAP_SYS_ADMIN-gated because its payload embeds the raw serial.
  amdcuid_status_t driver_primary_cuid(amdcuid_primary_id& id) const;

  // Virtual accessors for common device properties with default wrong device
  // type implementations
  virtual amdcuid_status_t get_vendor_id(uint16_t& vendor_id) const {
    vendor_id = 0;
    return AMDCUID_STATUS_WRONG_DEVICE_TYPE;
  }
  virtual amdcuid_status_t get_family(uint16_t& family) const {
    family = 0;
    return AMDCUID_STATUS_WRONG_DEVICE_TYPE;
  }
  virtual amdcuid_status_t get_model(uint16_t& model) const {
    model = 0;
    return AMDCUID_STATUS_WRONG_DEVICE_TYPE;
  }
  virtual amdcuid_status_t get_device_id(uint16_t& device_id) const {
    device_id = 0;
    return AMDCUID_STATUS_WRONG_DEVICE_TYPE;
  }
  virtual amdcuid_status_t get_revision_id(uint8_t& revision_id) const {
    revision_id = 0;
    return AMDCUID_STATUS_WRONG_DEVICE_TYPE;
  }
  virtual amdcuid_status_t get_unit_id(uint16_t& unit_id) const {
    unit_id = 0;
    return AMDCUID_STATUS_WRONG_DEVICE_TYPE;
  }
  virtual amdcuid_status_t get_pci_class(uint16_t& pci_class) const {
    pci_class = 0;
    return AMDCUID_STATUS_WRONG_DEVICE_TYPE;
  }
  virtual amdcuid_status_t get_core(uint16_t& core) const {
    core = 0;
    return AMDCUID_STATUS_WRONG_DEVICE_TYPE;
  }
  virtual amdcuid_status_t get_physical_id(uint16_t& physical_id) const {
    physical_id = 0;
    return AMDCUID_STATUS_WRONG_DEVICE_TYPE;
  }
  virtual amdcuid_status_t get_bdf(std::string& bdf) const {
    bdf.clear();
    return AMDCUID_STATUS_WRONG_DEVICE_TYPE;
  }
  virtual amdcuid_status_t get_device_path(std::string& path) const {
    path.clear();
    return AMDCUID_STATUS_WRONG_DEVICE_TYPE;
  }

 protected:
  // Stage 1: the driver's cuid_derived. UNSUPPORTED when it holds no value
  // for this component.
  amdcuid_status_t driver_derived_cuid(amdcuid_derived_id& id) const;

  // Record the derivation's source: a later sysfs lookup could observe
  // provisioning changes and attribute the returned value to the wrong stage.
  mutable amdcuid_source_t last_source_ = AMDCUID_SOURCE_UNKNOWN;
};

typedef std::shared_ptr<CuidDevice> DevicePtr;

#endif  // CUID_DEVICE_H
