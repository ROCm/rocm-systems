// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef CUID_DEVICE_H
#define CUID_DEVICE_H

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>

#include "include/amd_cuid.h"
#include "src/cuid_internal.h"
#include "src/cuid_util.h"
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

// A device's route (device_type, bdf, device_node) and the derived CUID it
// had when CuidDeviceManager last rebuilt its index.
struct CuidDeviceEntry {
  amdcuid_device_type_t device_type = AMDCUID_DEVICE_TYPE_NONE;
  amdcuid_id_t derived_cuid{};
  std::string device_node;
  std::string bdf;
};

namespace cuid {
using DeviceRoute = std::tuple<amdcuid_device_type_t, std::string, std::string>;
DeviceRoute device_route(const CuidDeviceEntry& entry);
}  // namespace cuid

class CuidDevice {
 public:
  virtual ~CuidDevice() = default;
  virtual amdcuid_device_type_t type() const = 0;
  virtual amdcuid_status_t get_primary_cuid(amdcuid_primary_id& id) const = 0;
  virtual amdcuid_status_t get_hardware_fingerprint(uint64_t& fingerprint) const = 0;

  // Whether this component's derived CUID needs the node key: true for CPU,
  // NIC, NPU, Platform and GPU.
  virtual bool key_gated_identity() const { return false; }

  // The input structure of this component's temporary CUID, without the
  // machine-id. Taken by a key-gated component when there is no key, and by
  // any component whose get_primary_cuid() reports HW_FINGERPRINT_NOT_FOUND,
  // that is, which has no permanent primary.
  virtual amdcuid_status_t get_temporary_input(CuidUtilities::AuxiliaryInput& input) const {
    (void)input;
    return AMDCUID_STATUS_UNSUPPORTED;
  }

  // The temporary derived CUID from get_temporary_input() and this host's
  // machine-id. There is no temporary primary.
  amdcuid_status_t get_temporary_cuid(amdcuid_derived_id& id) const;

  // This library's computation: keyed with `hmac` where it holds a key,
  // otherwise temporary for a key-gated component.
  virtual amdcuid_status_t get_derived_cuid(amdcuid_derived_id& id,
                                            cuid_hmac* hmac = nullptr) const;

  // Whether this component's identity is temporary: whether the derivation
  // takes the temporary path. `hmac` decides that for a key-gated component,
  // as in get_derived_cuid(), so the two agree on the same component.
  amdcuid_status_t is_temporary_cuid(bool* is_temporary, cuid_hmac* hmac = nullptr) const;

  // The stage that produced the last successful derivation;
  // AMDCUID_SOURCE_UNKNOWN before one and after a failed one.
  amdcuid_source_t derived_source() const { return last_source_; }

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
  // Record the derivation's source: a later sysfs lookup could observe
  // provisioning changes and attribute the returned value to the wrong stage.
  mutable amdcuid_source_t last_source_ = AMDCUID_SOURCE_UNKNOWN;
};

typedef std::shared_ptr<CuidDevice> DevicePtr;

#endif  // CUID_DEVICE_H
