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

#ifndef INCLUDE_ROCM_SMI_ROCM_SMI_POWER_MON_H_
#define INCLUDE_ROCM_SMI_ROCM_SMI_POWER_MON_H_

#include <string>
#include <cstdint>

#include "rocm_smi/rocm_smi_common.h"

namespace amd::smi {

enum PowerMonTypes {
  kPowerMaxGPUPower,
};


class PowerMon {
 public:
    explicit PowerMon(std::string path, RocmSMI_env_vars const *e);
    ~PowerMon(void);
    const std::string path(void) const {return path_;}

    uint32_t dev_index(void) const {return dev_index_;}
    void set_dev_index(uint32_t ind) {dev_index_ = ind;}
    int readPowerValue(PowerMonTypes type, uint64_t *power);

 private:
    std::string path_;
    const RocmSMI_env_vars *env_;
    uint32_t dev_index_;
};

} // namespace amd::smi

#endif  // INCLUDE_ROCM_SMI_ROCM_SMI_POWER_MON_H_
