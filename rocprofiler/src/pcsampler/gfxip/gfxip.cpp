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

#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <optional>
#include <vector>
#include <cstring>

#include <fcntl.h>

#include "rocprofiler.h"

#include "gfxip.h"
#include "src/utils/helper.h"
#include "src/utils/libpci_helper.h"

static const char DEBUG_DRI_PATH[] = "/sys/kernel/debug/dri/";
static const char DEV_PFX[] = "dev=";

namespace rocprofiler::pc_sampler::gfxip {

namespace {

static int find_pci_instance(const std::string& pci_string) {
  rocprofiler::handle_t<DIR*, util::dir_closer> dir(opendir(DEBUG_DRI_PATH));
  if (dir.get() == nullptr) {
    char* errstr = strerror(errno);
    warning("Can't open debugfs dri directory: %s\n", errstr);
    goto fail;
  }

  struct dirent* dent;
  while ((dent = readdir(dir.get())) != nullptr) {
    if (strcmp(dent->d_name, ".") == 0 || strcmp(dent->d_name, "..") == 0) continue;

    std::string name(DEBUG_DRI_PATH);
    name += dent->d_name;
    name += "/name";

    std::string device;
    {
      std::ifstream ifs(name);
      if (!ifs.is_open()) return -1;
      ifs.ignore(std::numeric_limits<std::streamsize>::max(), ' ');
      ifs >> device;
    }
    if (device.empty()) continue;
    if (auto p = device.find(DEV_PFX); p != device.npos) device.erase(p, strlen(DEV_PFX));
    if (pci_string == device) return std::stoi(dent->d_name);
  }

fail:
  return -1;
}

}  // namespace

uint32_t pasid() {
  static std::optional<uint32_t> pasid;

  if (!pasid) {
    std::ifstream ifs("/sys/class/kfd/kfd/proc/" + std::to_string(getpid()) + "/pasid");
    if (!ifs.is_open()) return 0;
    ifs >> *pasid;
  }

  return *pasid;
}

int debugfs_ioctl_set_state(const device_t& dev, const struct amdgpu_debugfs_regs2_iocdata& ioc) {
  int ret = ioctl(dev.fd_.mmio2.get(), AMDGPU_DEBUGFS_REGS2_IOC_SET_STATE, &ioc);
  if (ret < 0) {
    fatal("Couldn't set register ioctl state\n");
  }
  return ret;
}

int debugfs_ioctl_write_register(const device_t& dev,
                                 const struct amdgpu_debugfs_regs2_iocdata& ioc,
                                 const uint64_t addr, const uint32_t value) {
  debugfs_ioctl_set_state(dev, ioc);
  if (lseek(dev.fd_.mmio2.get(), addr * 4, SEEK_SET) < 0) {
    fatal("Cannot seek to MMIO address for write\n");
  }
  int r = -1;
  if ((r = write(dev.fd_.mmio2.get(), &value, sizeof(value))) != sizeof(value)) {
    fatal("Cannot write to MMIO register\n");
  }
  return r;
}

uint32_t debugfs_ioctl_read_register(const device_t& dev,
                                     const struct amdgpu_debugfs_regs2_iocdata& ioc,
                                     const uint64_t addr) {
  // Select the SE, SH, and CU.
  debugfs_ioctl_set_state(dev, ioc);

  if (lseek(dev.fd_.mmio2.get(), addr * 4, SEEK_SET) < 0) {
    fatal("Cannot seek to MMIO address for read\n");
  }

  uint32_t value = -1U;
  if (read(dev.fd_.mmio2.get(), &value, sizeof(value)) != sizeof(value)) {
    fatal("Cannot read from MMIO register\n");
  }

  return value;
}

device_t::device_t(const bool pci_inited, const HSAAgentInfo& info)
    : agent_info_(info), pci_memory_(nullptr) {
  const auto pci_domain = agent_info_.GetDeviceInfo().getPCIDomain();
  const auto pci_location_id = agent_info_.GetDeviceInfo().getPCILocationID();

  std::string name([pci_domain, pci_location_id]() {
    std::ostringstream out;
    out.fill('0');
    out << std::hex << std::setw(4) << pci_domain << ':' << std::hex << std::setw(2)
        << (pci_location_id >> 8) << ':' << std::hex << std::setw(2) << (pci_location_id & 0xFF)
        << '.' << 0;
    return out.str();
  }());

  int instance = find_pci_instance(name);
  bool use_pciaccess_library = false;
  {
    std::string dri_debug_path_pfx(DEBUG_DRI_PATH);
    dri_debug_path_pfx += std::to_string(instance);
    dri_debug_path_pfx += '/';

    std::string path(dri_debug_path_pfx + "amdgpu_regs2");
    fd_.mmio2.reset(open(path.c_str(), O_RDWR));
    if (fd_.mmio2.get() < 0) {
      warning("Couldn't open amdgpu_regs2 debugfs file\n");
      if (!pci_inited) {
        constexpr char msg[] = "PCI system uninitialized; no PC sampling methods available\n";
        fatal(msg);
      }
    } else {
      use_pciaccess_library = true;
    }
  }
  if (use_pciaccess_library) {

    pci_device_ =
        GetPciAccessLibApi()->pci_device_find_by_slot(pci_domain, pci_location_id >> 8, pci_location_id & 0xFF, 0);
    if (!pci_device_ || GetPciAccessLibApi()->pci_device_probe(pci_device_)) fatal("failed to probe the GPU device\n");

    // Look for a region between 256KB and 4096KB, 32-bit, non IO, and non prefetchable.
    for (size_t region = 0; region < sizeof(pci_device::regions) / sizeof(pci_device::regions[0]);
        ++region) {
      if (pci_device_->regions[region].is_64 == 0 &&
          pci_device_->regions[region].is_prefetchable == 0 &&
          pci_device_->regions[region].is_IO == 0 &&
          pci_device_->regions[region].size >= (256UL * 1024) &&
          pci_device_->regions[region].size <= (4096UL * 1024)) {
        pci_memory_size_ = pci_device_->regions[region].size;
        if (GetPciAccessLibApi()->pci_device_map_range(pci_device_, pci_device_->regions[region].base_addr,
                                pci_device_->regions[region].size, PCI_DEV_MAP_FLAG_WRITABLE,
                                (void**)&pci_memory_))
          fatal("failed to map the registers\n");
      }
    }

    if (pci_memory_ == nullptr) fatal("could not find the pci memory address\n");
  }

device_specific_init:
  // FIXME: detect the gfxip and call the correct routine.
  vega20_reg_offset_init(*this);
}

device_t::~device_t() {
  if (pci_memory_ && GetPciAccessLibApi()->pci_device_unmap_range(pci_device_, pci_memory_, pci_memory_size_)) {
    warning("failed to unmap the pci memory\n");
  }
}

}  // namespace rocprofiler::pc_sampler::gfxip
