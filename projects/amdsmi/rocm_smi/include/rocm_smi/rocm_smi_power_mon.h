// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_ROCM_SMI_ROCM_SMI_POWER_MON_H_
#define INCLUDE_ROCM_SMI_ROCM_SMI_POWER_MON_H_

#include <cstdint>
#include <string>

#include "rocm_smi/rocm_smi_common.h"

namespace amd::smi {

enum PowerMonTypes {
  kPowerMaxGPUPower,
};

class PowerMon {
 public:
  explicit PowerMon(std::string path, RocmSMI_env_vars const* e);
  ~PowerMon(void);
  const std::string path(void) const { return path_; }

  uint32_t dev_index(void) const { return dev_index_; }
  void set_dev_index(uint32_t ind) { dev_index_ = ind; }
  int readPowerValue(PowerMonTypes type, uint64_t* power);

 private:
  std::string path_;
  const RocmSMI_env_vars* env_;
  uint32_t dev_index_;
};

}  // namespace amd::smi

#endif  // INCLUDE_ROCM_SMI_ROCM_SMI_POWER_MON_H_
