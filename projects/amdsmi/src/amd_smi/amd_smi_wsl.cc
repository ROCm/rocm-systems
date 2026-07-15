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

#include "amd_smi/impl/amd_smi_wsl.h"

#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>  // NOLINT(build/c++11)
#include <string>
#include <utility>
#include <vector>

#include "amd_smi/impl/amd_smi_lib_loader.h"
#include "amd_smi/impl/amd_smi_test_internal.h"
#include "amd_smi/impl/amd_smi_uuid.h"

// When amdsmi happens to be compiled in a HIP-aware environment (a HIP platform
// macro is predefined AND the runtime headers are reachable) pull in the HIP
// attribute enum so the hardcoded fallback below can be cross-checked at build
// time. In the normal amdsmi build (plain g++, no HIP headers on the include
// path) neither guard holds, nothing is included, and the build is unaffected.
#if defined(__HIP_PLATFORM_AMD__) && defined(__has_include)
#if __has_include(<hip/hip_runtime_api.h>)
#include <hip/hip_runtime_api.h>
#define AMDSMI_WSL_HIP_HEADERS_AVAILABLE 1
#endif
#endif

namespace amd::smi {

namespace {

// AMD/ATI PCI vendor id.
constexpr uint32_t kAmdPciVendorId = 0x1002;
constexpr const char* kAmdVendorName = "Advanced Micro Devices, Inc. [AMD/ATI]";

// hipError_t success value (the HIP error enum is ABI-stable; hipSuccess == 0).
constexpr int kHipSuccess = 0;

// HIP runtime entry points used by the WSL2 fallback. Only forward-ABI-stable
// signatures are bound here -- in particular hipDeviceProp_t (whose layout is
// version dependent) is intentionally avoided.
using pfn_hipGetDeviceCount = int (*)(int*);
using pfn_hipDeviceGetName = int (*)(char*, int, int);
using pfn_hipDeviceGetPCIBusId = int (*)(char*, int, int);
using pfn_hipGetDevice = int (*)(int*);
using pfn_hipSetDevice = int (*)(int);
using pfn_hipMemGetInfo = int (*)(size_t*, size_t*);
using pfn_hipDeviceGetAttribute = int (*)(int*, int, int);

// hipDeviceAttributeMultiprocessorCount: ABI-stable enumerator value (== CU
// count on AMD GPUs). Binding the value directly avoids pulling in the
// version-dependent hipDeviceProp_t. The value is verified against ROCm's
// hip_runtime_api.h; the static_assert below catches a future renumbering
// whenever the HIP headers are visible at build time.
constexpr int kHipDeviceAttributeMultiprocessorCount = 63;
#ifdef AMDSMI_WSL_HIP_HEADERS_AVAILABLE
static_assert(static_cast<int>(hipDeviceAttributeMultiprocessorCount) ==
                  kHipDeviceAttributeMultiprocessorCount,
              "HIP renumbered hipDeviceAttributeMultiprocessorCount; update "
              "kHipDeviceAttributeMultiprocessorCount to match.");
#endif

// Lazily-loaded HIP runtime. amdsmi does not link against HIP; the library is
// opened on demand via dlopen and is optional.
struct HipRuntime {
  AMDSmiLibraryLoader loader;
  bool available = false;
  pfn_hipGetDeviceCount GetDeviceCount = nullptr;
  pfn_hipDeviceGetName DeviceGetName = nullptr;
  pfn_hipDeviceGetPCIBusId DeviceGetPCIBusId = nullptr;
  pfn_hipGetDevice GetDevice = nullptr;
  pfn_hipSetDevice SetDevice = nullptr;
  pfn_hipMemGetInfo MemGetInfo = nullptr;
  pfn_hipDeviceGetAttribute DeviceGetAttribute = nullptr;  // optional
};

HipRuntime& hip_runtime() {
  static HipRuntime rt;
  static std::once_flag once;
  std::call_once(once, [&]() {
    // Prefer the unversioned dev symlink, then common SONAME versions.
    const char* candidates[] = {
        "libamdhip64.so",
        "libamdhip64.so.7",
        "libamdhip64.so.6",
        "libamdhip64.so.5",
    };
    bool loaded = false;
    for (const char* name : candidates) {
      // Quietly probe each SONAME candidate: a typical WSL2 install ships only
      // one of them, so a failed dlopen here is expected and must not emit
      // stderr noise from a monitoring library.
      if (rt.loader.load(name, /*log_errors=*/false) == AMDSMI_STATUS_SUCCESS) {
        loaded = true;
        break;
      }
    }
    if (!loaded) {
      return;
    }

    // All of these must resolve for the fallback to be usable.
    bool ok = true;
    ok &= rt.loader.load_symbol(&rt.GetDeviceCount, "hipGetDeviceCount") == AMDSMI_STATUS_SUCCESS;
    ok &= rt.loader.load_symbol(&rt.DeviceGetName, "hipDeviceGetName") == AMDSMI_STATUS_SUCCESS;
    ok &= rt.loader.load_symbol(&rt.DeviceGetPCIBusId, "hipDeviceGetPCIBusId") ==
          AMDSMI_STATUS_SUCCESS;
    ok &= rt.loader.load_symbol(&rt.GetDevice, "hipGetDevice") == AMDSMI_STATUS_SUCCESS;
    ok &= rt.loader.load_symbol(&rt.SetDevice, "hipSetDevice") == AMDSMI_STATUS_SUCCESS;
    ok &= rt.loader.load_symbol(&rt.MemGetInfo, "hipMemGetInfo") == AMDSMI_STATUS_SUCCESS;
    // Optional, best effort: absence is acceptable, so load quietly.
    rt.loader.load_symbol(&rt.DeviceGetAttribute, "hipDeviceGetAttribute",
                          /*log_errors=*/false);
    rt.available = ok;
  });
  return rt;
}

// Copy a std::string into a fixed-size, always-null-terminated char buffer.
void set_fixed_string(char* dst, size_t cap, const std::string& src) {
  if (dst == nullptr || cap == 0) {
    return;
  }
  const size_t n = std::min(src.size(), cap - 1);
  std::memcpy(dst, src.data(), n);
  dst[n] = '\0';
}

// RAII guard that saves the current HIP device on construction and restores it
// on destruction. amdsmi is a monitoring library and must not leave the host
// application's per-thread current device changed after a query.
class ScopedHipDevice {
 public:
  ScopedHipDevice(const HipRuntime& hip, int target) : hip_(hip) {
    have_prev_ = hip_.GetDevice != nullptr && hip_.GetDevice(&prev_) == kHipSuccess;
    if (hip_.SetDevice != nullptr && hip_.SetDevice(target) == kHipSuccess) {
      set_ok_ = true;
      if (!have_prev_) {
        // HIP defaults a thread without a current context to device 0.
        prev_ = 0;
        have_prev_ = true;
      }
    }
  }
  ~ScopedHipDevice() {
    if (set_ok_ && have_prev_ && hip_.SetDevice != nullptr) {
      hip_.SetDevice(prev_);
    }
  }
  bool set_ok() const { return set_ok_; }

  ScopedHipDevice(const ScopedHipDevice&) = delete;
  ScopedHipDevice& operator=(const ScopedHipDevice&) = delete;

 private:
  const HipRuntime& hip_;
  int prev_ = 0;
  bool have_prev_ = false;
  bool set_ok_ = false;
};

amdsmi_status_t get_vram_usage_bytes(const HipRuntime& hip, int hip_index,
                                     WslVramUsageBytes* usage) {
  if (usage == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }
  *usage = {};
  if (!hip.available) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }

  ScopedHipDevice dev_guard(hip, hip_index);
  if (!dev_guard.set_ok()) {
    return AMDSMI_STATUS_API_FAILED;
  }
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  if (hip.MemGetInfo == nullptr || hip.MemGetInfo(&free_bytes, &total_bytes) != kHipSuccess) {
    return AMDSMI_STATUS_API_FAILED;
  }
  usage->total = static_cast<uint64_t>(total_bytes);
  usage->used = total_bytes >= free_bytes ? static_cast<uint64_t>(total_bytes - free_bytes) : 0;
  return AMDSMI_STATUS_SUCCESS;
}

// Parse a HIP "domain:bus:device.function" string (e.g. "0000:c5:00.0") into an
// amdsmi_bdf_t. Returns false if the string is not in the expected form.
bool parse_pci_bus_id(const std::string& bus_id, amdsmi_bdf_t* out) {
  unsigned int domain = 0, bus = 0, dev = 0, func = 0;
  if (std::sscanf(bus_id.c_str(), "%x:%x:%x.%x", &domain, &bus, &dev, &func) != 4) {
    return false;
  }
  amdsmi_bdf_t bdf = {};
  // Mask the bus/device/function components to their BDF bit-field widths so the
  // narrowing store is explicit (a well-formed PCI bus id is always in range;
  // this also keeps the build clean under -Wconversion). domain_number is wide
  // enough that no mask is needed.
  bdf.bdf.domain_number = domain;
  bdf.bdf.bus_number = bus & 0xFFu;
  bdf.bdf.device_number = dev & 0x1Fu;
  bdf.bdf.function_number = func & 0x7u;
  *out = bdf;
  return true;
}

// Best-effort read of the PCI device id from sysfs. Usually absent on WSL2 (the
// GPU is a paravirtual /dev/dxg device), in which case 0 is returned.
uint64_t read_pci_device_id(const std::string& bus_id) {
  if (bus_id.empty()) {
    return 0;
  }
  // sysfs uses lowercase BDF strings.
  std::string lower = bus_id;
  for (char& c : lower) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  const std::string path = "/sys/bus/pci/devices/" + lower + "/device";
  std::ifstream f(path);
  if (!f) {
    return 0;
  }
  std::string value;
  f >> value;  // e.g. "0x1586"
  if (value.empty()) {
    return 0;
  }
  try {
    return static_cast<uint64_t>(std::stoul(value, nullptr, 16));
  } catch (...) {
    return 0;
  }
}

}  // namespace

bool is_wsl2_environment() {
  // 1. Kernel must advertise Microsoft/WSL.
  bool kernel_is_wsl = false;
  {
    std::ifstream f("/proc/version");
    std::string line;
    if (f && std::getline(f, line)) {
      auto contains = [&line](const char* needle) {
        return line.find(needle) != std::string::npos;
      };
      kernel_is_wsl = contains("microsoft") || contains("Microsoft") || contains("WSL");
    }
  }
  if (!kernel_is_wsl) {
    return false;
  }

  // 2. The WDDM paravirtual GPU device must be present.
  if (::access("/dev/dxg", F_OK) != 0) {
    return false;
  }

  // 3. The native amdgpu compute path must be absent; if /dev/kfd exists the
  //    normal amdsmi backend works and the fallback must not engage.
  if (::access("/dev/kfd", F_OK) == 0) {
    return false;
  }

  return true;
}

amdsmi_status_t wsl_discover_gpus(std::vector<WslGpuInfo>* out) {
  if (out == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }
  out->clear();

  HipRuntime& hip = hip_runtime();
  if (!hip.available) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }

  int device_count = 0;
  if (hip.GetDeviceCount(&device_count) != kHipSuccess || device_count <= 0) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }

  out->reserve(static_cast<size_t>(device_count));
  for (int i = 0; i < device_count; ++i) {
    WslGpuInfo gpu;
    gpu.hip_index = i;

    char name_buf[AMDSMI_MAX_STRING_LENGTH] = {0};
    if (hip.DeviceGetName(name_buf, static_cast<int>(sizeof(name_buf)), i) == kHipSuccess) {
      name_buf[sizeof(name_buf) - 1] = '\0';
      gpu.market_name = name_buf;
    }

    char bdf_buf[64] = {0};
    if (hip.DeviceGetPCIBusId(bdf_buf, static_cast<int>(sizeof(bdf_buf)), i) == kHipSuccess) {
      bdf_buf[sizeof(bdf_buf) - 1] = '\0';
      gpu.bdf_string = bdf_buf;
      parse_pci_bus_id(gpu.bdf_string, &gpu.bdf);
      gpu.device_id = read_pci_device_id(gpu.bdf_string);
    }

    // VRAM total is intentionally NOT queried here. hipMemGetInfo requires a
    // current device, and calling hipSetDevice on every GPU during enumeration
    // would force a primary HIP context to be created on each one just to list
    // devices -- an undesirable side effect for a monitoring library. The size
    // is fetched lazily in wsl_fill_vram_info() instead.

    // Compute-unit count via the ABI-stable attribute query (best effort).
    if (hip.DeviceGetAttribute != nullptr) {
      int cu = 0;
      if (hip.DeviceGetAttribute(&cu, kHipDeviceAttributeMultiprocessorCount, i) == kHipSuccess &&
          cu > 0) {
        gpu.num_compute_units = static_cast<uint32_t>(cu);
      }
    }

    out->push_back(std::move(gpu));
  }

  if (out->empty()) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t wsl_fill_asic_info(const WslGpuInfo& gpu, amdsmi_asic_info_t* info) {
  if (info == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }
  *info = {};
  // Default every field to its documented "not supported" sentinel.
  info->market_name[0] = '\0';
  info->vendor_id = std::numeric_limits<uint32_t>::max();
  info->vendor_name[0] = '\0';
  info->subvendor_id = std::numeric_limits<uint32_t>::max();
  info->device_id = std::numeric_limits<uint64_t>::max();
  info->rev_id = std::numeric_limits<uint32_t>::max();
  info->asic_serial[0] = '\0';
  info->oam_id = std::numeric_limits<uint32_t>::max();
  info->num_of_compute_units = std::numeric_limits<uint32_t>::max();
  info->target_graphics_version = std::numeric_limits<uint64_t>::max();
  info->subsystem_id = std::numeric_limits<uint32_t>::max();
  info->flags = 0;

  if (!gpu.market_name.empty()) {
    set_fixed_string(info->market_name, AMDSMI_MAX_STRING_LENGTH, gpu.market_name);
  }
  info->vendor_id = kAmdPciVendorId;
  set_fixed_string(info->vendor_name, AMDSMI_MAX_STRING_LENGTH, kAmdVendorName);
  if (gpu.device_id != 0) {
    info->device_id = gpu.device_id;
  }
  if (gpu.num_compute_units != 0) {
    info->num_of_compute_units = gpu.num_compute_units;
  }
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t wsl_fill_board_info(const WslGpuInfo& gpu, amdsmi_board_info_t* info) {
  if (info == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }
  *info = {};
  info->model_number[0] = '\0';
  info->product_serial[0] = '\0';
  info->fru_id[0] = '\0';
  info->product_name[0] = '\0';
  info->manufacturer_name[0] = '\0';

  if (!gpu.market_name.empty()) {
    set_fixed_string(info->product_name, AMDSMI_MAX_STRING_LENGTH, gpu.market_name);
  }
  set_fixed_string(info->manufacturer_name, AMDSMI_MAX_STRING_LENGTH, kAmdVendorName);
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t wsl_fill_vram_info(const WslGpuInfo& gpu, amdsmi_vram_info_t* info) {
  if (info == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }
  *info = {};
  info->vram_type = AMDSMI_VRAM_TYPE_UNKNOWN;
  info->vram_vendor[0] = '\0';
  info->vram_bit_width = std::numeric_limits<uint32_t>::max();
  info->vram_max_bandwidth = std::numeric_limits<uint64_t>::max();

  WslVramUsageBytes usage;
  amdsmi_status_t status = wsl_get_vram_usage_bytes(gpu, &usage);
  if (status != AMDSMI_STATUS_SUCCESS) {
    return status;
  }
  // hipMemGetInfo reports bytes; amdsmi_vram_info_t.vram_size is in MB.
  info->vram_size = usage.total / (1024ULL * 1024ULL);
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t wsl_fill_vram_usage(const WslGpuInfo& gpu, amdsmi_vram_usage_t* info) {
  if (info == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }
  *info = {};

  WslVramUsageBytes usage;
  amdsmi_status_t status = wsl_get_vram_usage_bytes(gpu, &usage);
  if (status != AMDSMI_STATUS_SUCCESS) {
    return status;
  }
  info->vram_total = static_cast<uint32_t>(usage.total / (1024ULL * 1024ULL));
  info->vram_used = static_cast<uint32_t>(usage.used / (1024ULL * 1024ULL));
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t wsl_get_vram_usage_bytes(const WslGpuInfo& gpu, WslVramUsageBytes* usage) {
  return get_vram_usage_bytes(hip_runtime(), gpu.hip_index, usage);
}

amdsmi_status_t wsl_generate_device_uuid(const WslGpuInfo& gpu, char* uuid) {
  if (uuid == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }
  const uint16_t device_id = gpu.device_id != 0 ? static_cast<uint16_t>(gpu.device_id & 0xFFFF)
                                                : std::numeric_limits<uint16_t>::max();
  const uint64_t serial = (static_cast<uint64_t>(gpu.bdf.bdf.domain_number) << 24) |
                          (static_cast<uint64_t>(gpu.bdf.bdf.bus_number) << 16) |
                          (static_cast<uint64_t>(gpu.bdf.bdf.device_number) << 8) |
                          static_cast<uint64_t>(gpu.bdf.bdf.function_number);
  return amdsmi_uuid_gen(uuid, serial, device_id, /*idx=*/0xff);
}

bool amdsmi_test_wsl_scoped_device(amdsmi_test_hip_get_device_fn get_device,
                                   amdsmi_test_hip_set_device_fn set_device, int target) {
  HipRuntime hip;
  hip.GetDevice = get_device;
  hip.SetDevice = set_device;
  ScopedHipDevice device_guard(hip, target);
  return device_guard.set_ok();
}

amdsmi_status_t amdsmi_test_wsl_get_vram_usage_bytes(amdsmi_test_hip_get_device_fn get_device,
                                                     amdsmi_test_hip_set_device_fn set_device,
                                                     amdsmi_test_hip_mem_get_info_fn mem_get_info,
                                                     int target, WslVramUsageBytes* usage) {
  HipRuntime hip;
  hip.available = true;
  hip.GetDevice = get_device;
  hip.SetDevice = set_device;
  hip.MemGetInfo = mem_get_info;
  return get_vram_usage_bytes(hip, target, usage);
}

}  // namespace amd::smi
