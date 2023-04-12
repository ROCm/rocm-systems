/* Copyright (c) 2022 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#ifndef SRC_PCSAMPLER_GFXIP_GFXIP_H_
#define SRC_PCSAMPLER_GFXIP_GFXIP_H_

#include <cstddef>
#include <cstdint>
#include <unordered_map>

#include <linux/types.h>
#include <linux/ioctl.h>

#include <dirent.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <unistd.h>

#include <hsa/hsa.h>

#include "rocprofiler.h"
#include "src/core/hardware/hsa_info.h"
#include "src/core/hsa/hsa_support.h"
#include "src/utils/handle.h"

#include <pciaccess.h>

#define __maybe_unused

namespace rocprofiler::pc_sampler {

struct PCSampler;

namespace gfxip {

namespace util {

struct dir_closer {
  void operator()(DIR* dir) {
    if (dir != nullptr) closedir(dir);
  }
};

struct fd_closer {
  void operator()(int fd) {
    if (fd >= 0) close(fd);
  }
};

}  // namespace util

struct amdgpu_debugfs_regs2_iocdata {
  __u32 use_srbm, use_grbm, pg_lock;
  struct {
    __u32 se, sh, instance;
  } grbm;
  struct {
    __u32 me, pipe, queue, vmid;
  } srbm;
};

enum AMDGPU_DEBUGFS_REGS2_CMDS { AMDGPU_DEBUGFS_REGS2_CMD_SET_STATE = 0 };

#define AMDGPU_DEBUGFS_REGS2_IOC_SET_STATE                                                         \
  _IOWR(0x20, AMDGPU_DEBUGFS_REGS2_CMD_SET_STATE, struct amdgpu_debugfs_regs2_iocdata)

enum {
  GC_HWIP = 1,  // Graphics Core IP
  OSSSYS_HWIP,
  MAX_HWIP
};
static constexpr int HWIP_MAX_INSTANCE = 11;

#define REG_OFFSET(ip, inst, reg) (dev.reg_offset_[ip##_HWIP][inst][reg##_BASE_IDX] + reg)

#define REG_FIELD_SHIFT(reg, field) reg##__##field##__SHIFT
#define REG_FIELD_MASK(reg, field) reg##__##field##_MASK

#define REG_GET_FIELD(value, reg, field)                                                           \
  (((value)&REG_FIELD_MASK(reg, field)) >> REG_FIELD_SHIFT(reg, field))
#define REG_SET_FIELD(orig_val, reg, field, field_val)                                             \
  (((orig_val) & ~REG_FIELD_MASK(reg, field)) |                                                    \
   (REG_FIELD_MASK(reg, field) & ((field_val) << REG_FIELD_SHIFT(reg, field))))

struct device_t {
  device_t(const bool pci_inited, const HSAAgentInfo& agent_info);
  ~device_t();

  device_t(const device_t&) = delete;
  device_t& operator=(const device_t&) = delete;
  device_t(device_t&&) = default;

  const HSAAgentInfo& agent_info_;

  struct pci_device* pci_device_;
  size_t pci_memory_size_;
  uint32_t* pci_memory_;

  uint32_t* reg_offset_[MAX_HWIP][HWIP_MAX_INSTANCE];

  struct debugfs_fds {
    debugfs_fds() : mmio2(-1) {}

    rocprofiler::handle_t<int, util::fd_closer> mmio2;
  } fd_;
};

uint32_t pasid();

int debugfs_ioctl_set_state(const device_t& dev, const struct amdgpu_debugfs_regs2_iocdata& ioc);
int debugfs_ioctl_write_register(const device_t& dev,
                                 const struct amdgpu_debugfs_regs2_iocdata& ioc,
                                 const uint64_t addr, const uint32_t value);
uint32_t debugfs_ioctl_read_register(const device_t& dev,
                                     const struct amdgpu_debugfs_regs2_iocdata& ioc,
                                     const uint64_t addr);

void vega10_reg_offset_init(device_t& dev);
void vega20_reg_offset_init(device_t& dev);
void arct_reg_offset_init(device_t& dev);
void aldebaran_reg_offset_init(device_t& dev);

void read_pc_samples_v9(const device_t& dev, PCSampler* sampler);
void read_pc_samples_v9_ioctl(const device_t& dev, PCSampler* sampler);

}  // namespace gfxip

}  // namespace rocprofiler::pc_sampler
#endif  // SRC_PCSAMPLER_GFXIP_GFXIP_H_
