/*
 * Copyright © 2014 Advanced Micro Devices, Inc.
 * Copyright 2016-2018 Raptor Engineering, LLC. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including
 * the next paragraph) shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <assert.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/sysinfo.h>

#include "libhsakmt.h"
#include "impl/wddm/types.h"
#include "impl/wddm/device.h"
#include "util/utils.h"

/* Number of memory banks added by thunk on top of topology
 * This only includes static heaps like LDS, scratch and SVM,
 * not for MMIO_REMAP heap. MMIO_REMAP memory bank is reported
 * dynamically based on whether mmio aperture was mapped
 * successfully on this node.
 */
#define NUM_OF_IGPU_HEAPS 3
#define NUM_OF_DGPU_HEAPS 3

typedef struct {
  HsaNodeProperties node;
  std::vector<HsaMemoryProperties> mem; /* node->NumBanks elements */
  std::vector<HsaCacheProperties> cache;
  std::vector<HsaIoLinkProperties> link;
} node_props_t;

struct _topology_props {
  HsaSystemProperties *g_system = nullptr;
  std::vector<node_props_t> g_props;
  std::vector<wsl::thunk::WDDMDevice *> wdevices_;
  uint32_t wdevice_num_ = 0;
  uint32_t num_sysfs_nodes = 0;
  int processor_vendor = -1;
  double freq_max_ = 0.0;
};

static _topology_props* dxg_topology = new _topology_props();

/* Supported System Vendors */
enum SUPPORTED_PROCESSOR_VENDORS {
  GENUINE_INTEL = 0,
  AUTHENTIC_AMD,
  IBM_POWER
};
/* Adding newline to make the search easier */
static const char *supported_processor_vendor_name[] = {
  "GenuineIntel",
  "AuthenticAMD",
  "" // POWER requires a different search method
};

static HSAKMT_STATUS topology_take_snapshot(void);
static void topology_drop_snapshot(void);

/* information from /proc/cpuinfo */
struct proc_cpuinfo {
  uint32_t proc_num;                     /* processor */
  uint32_t apicid;                       /* apicid */
  char model_name[HSA_PUBLIC_NAME_SIZE]; /* model name */
};

/* CPU cache table for all CPUs on the system. Each entry has the relative CPU
 * info and caches connected to that CPU.
 */
typedef struct cpu_cacheinfo {
  int32_t proc_num;    /* this cpu's processor number */
  uint32_t num_caches; /* number of caches reported by this cpu */
} cpu_cacheinfo_t;

/* num_subdirs - find the number of sub-directories in the specified path
 *	@dirpath - directory path to find sub-directories underneath
 *	@prefix - only count sub-directory names starting with prefix.
 *		Use blank string, "", to count all.
 *	Return - number of sub-directories
 */
static int num_subdirs(char *dirpath, const char *prefix) {
  int count = 0;
  DIR *dirp;
  struct dirent *dir;
  int prefix_len = strlen(prefix);

  dirp = opendir(dirpath);
  if (dirp) {
    while ((dir = readdir(dirp)) != 0) {
      if ((strcmp(dir->d_name, ".") == 0) || (strcmp(dir->d_name, "..") == 0))
        continue;
      if (prefix_len && strncmp(dir->d_name, prefix, prefix_len))
        continue;
      count++;
    }
    closedir(dirp);
  }

  return count;
}

/* fscanf_dec - read a file whose content is a decimal number
 *      @file [IN ] file to read
 *      @num [OUT] number in the file
 */
static HSAKMT_STATUS fscanf_dec(char *file, uint32_t *num) {
  FILE *fd;
  HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;

  fd = fopen(file, "r");
  if (!fd) {
    pr_err("Failed to open %s\n", file);
    return HSAKMT_STATUS_INVALID_PARAMETER;
  }
  if (fscanf(fd, "%u", num) != 1) {
    pr_err("Failed to parse %s as a decimal.\n", file);
    ret = HSAKMT_STATUS_ERROR;
  }

  fclose(fd);
  return ret;
}

/* fscanf_str - read a file whose content is a string
 *      @file [IN ] file to read
 *      @str [OUT] string in the file
 */
static HSAKMT_STATUS fscanf_str(char *file, char *str) {
  FILE *fd;
  HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;

  fd = fopen(file, "r");
  if (!fd) {
    pr_err("Failed to open %s\n", file);
    return HSAKMT_STATUS_INVALID_PARAMETER;
  }
  if (fscanf(fd, "%s", str) != 1) {
    pr_err("Failed to parse %s as a string.\n", file);
    ret = HSAKMT_STATUS_ERROR;
  }

  fclose(fd);
  return ret;
}

/* fscanf_size - read a file whose content represents size as a string
 *      @file [IN ] file to read
 *      @bytes [OUT] sizes in bytes
 */
static HSAKMT_STATUS fscanf_size(char *file, uint32_t *bytes) {
  FILE *fd;
  HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;
  char unit;
  int n;

  fd = fopen(file, "r");
  if (!fd) {
    pr_err("Failed to open %s\n", file);
    return HSAKMT_STATUS_INVALID_PARAMETER;
  }

  n = fscanf(fd, "%u%c", bytes, &unit);
  if (n < 1) {
    pr_err("Failed to parse %s\n", file);
    ret = HSAKMT_STATUS_ERROR;
  }

  if (n == 2) {
    switch (unit) {
    case 'K':
      *bytes <<= 10;
      break;
    case 'M':
      *bytes <<= 20;
      break;
    case 'G':
      *bytes <<= 30;
      break;
    default:
      ret = HSAKMT_STATUS_ERROR;
      break;
    }
  }

  fclose(fd);
  return ret;
}

/* cpumap_to_cpu_ci - translate shared_cpu_map string + cpuinfo->apicid into
 *		      SiblingMap in cache
 *	@shared_cpu_map [IN ] shared_cpu_map string
 *	@cpuinfo [IN ] cpuinfo to get apicid
 *	@this_cache [OUT] CPU cache to fill in SiblingMap
 */
static void cpumap_to_cpu_ci(char *shared_cpu_map,
                             const std::vector<struct proc_cpuinfo>& cpuinfo,
                             HsaCacheProperties *this_cache) {
  int num_hexs, bit;
  uint32_t proc, apicid, mask;
  char *ch_ptr;

  /* shared_cpu_map is shown as ...X3,X2,X1 Each X is a hex without 0x
   * and it's up to 8 characters(32 bits). For the first 32 CPUs(actually
   * procs), it's presented in X1. The next 32 is in X2, and so on.
   */
  num_hexs = (strlen(shared_cpu_map) + 8) / 9; /* 8 characters + "," */
  ch_ptr = strtok(shared_cpu_map, ",");
  while (num_hexs-- > 0) {
    mask = strtol(ch_ptr, NULL, 16); /* each X */
    for (bit = 0; bit < 32; bit++) {
      if (!((1 << bit) & mask))
        continue;
      proc = num_hexs * 32 + bit;
      apicid = cpuinfo[proc].apicid;
      if (apicid >= HSA_CPU_SIBLINGS) {
        pr_warn("SiblingMap buffer %d is too small\n", HSA_CPU_SIBLINGS);
        continue;
      }
      this_cache->SiblingMap[apicid] = 1;
    }
    ch_ptr = strtok(NULL, ",");
  }
}

/* get_cpu_cache_info - get specified CPU's cache information from sysfs
 *     @prefix [IN] sysfs path for target cpu cache,
 *                  /sys/devices/system/node/nodeX/cpuY/cache
 *     @cpuinfo [IN] /proc/cpuinfo data to get apicid
 *     @cpu_ci: CPU specified. This parameter is an input and also an output.
 *             [IN] cpu_ci->num_caches: number of index dirs
 *             [OUT] cpu_ci->cache_info: to store cache info collected
 *             [OUT] cpu_ci->num_caches: reduces when shared with other cpu(s)
 * Return: number of cache reported from this cpu
 */
static int get_cpu_cache_info(const char *prefix,
                              const std::vector<struct proc_cpuinfo>& cpuinfo,
                              std::vector<HsaCacheProperties>& cache,
                              cpu_cacheinfo_t& cpu_ci) {
  int n;
  char path[256], str[256];
  bool is_power9 = false;

  if (dxg_topology->processor_vendor == IBM_POWER) {
    if (strcmp(cpuinfo[0].model_name, "POWER9") == 0) {
      is_power9 = true;
    }
  }

  HsaCacheProperties this_cache;
  int num_idx = cpu_ci.num_caches;
  for (int idx = 0; idx < num_idx; idx++) {
    memset(&this_cache, 0, sizeof(this_cache));
    /* If this cache is shared by multiple CPUs, we only need
     * to list it in the first CPU.
     */
    if (is_power9) {
      // POWER9 has SMT4
      if (cpu_ci.proc_num & 0x3) {
        /* proc is not 0,4,8,etc.  Skip and reduce the cache count. */
        --cpu_ci.num_caches;
        continue;
      }
    } else {
      snprintf(path, 256, "%s/index%d/shared_cpu_list", prefix, idx);
      /* shared_cpu_list is shown as n1,n2... or n1-n2,n3-n4...
       * For both cases, this cache is listed to proc n1 only.
       */
      fscanf_dec(path, (uint32_t *)&n);
      if (cpu_ci.proc_num != n) {
        /* proc is not n1. Skip and reduce the cache count. */
        --cpu_ci.num_caches;
        continue;
      }
      this_cache.ProcessorIdLow = cpuinfo[cpu_ci.proc_num].apicid;
    }

    /* CacheLevel */
    snprintf(path, 256, "%s/index%d/level", prefix, idx);
    fscanf_dec(path, &this_cache.CacheLevel);
    /* CacheType */
    snprintf(path, 256, "%s/index%d/type", prefix, idx);

    memset(str, 0, sizeof(str));
    fscanf_str(path, str);
    if (!strcmp(str, "Data"))
      this_cache.CacheType.ui32.Data = 1;
    if (!strcmp(str, "Instruction"))
      this_cache.CacheType.ui32.Instruction = 1;
    if (!strcmp(str, "Unified")) {
      this_cache.CacheType.ui32.Data = 1;
      this_cache.CacheType.ui32.Instruction = 1;
    }
    this_cache.CacheType.ui32.CPU = 1;
    /* CacheSize */
    snprintf(path, 256, "%s/index%d/size", prefix, idx);
    fscanf_size(path, &this_cache.CacheSize);
    /* CacheLineSize */
    snprintf(path, 256, "%s/index%d/coherency_line_size", prefix, idx);
    fscanf_dec(path, &this_cache.CacheLineSize);
    /* CacheAssociativity */
    snprintf(path, 256, "%s/index%d/ways_of_associativity", prefix, idx);
    fscanf_dec(path, &this_cache.CacheAssociativity);
    /* CacheLinesPerTag */
    snprintf(path, 256, "%s/index%d/physical_line_partition", prefix, idx);
    fscanf_dec(path, &this_cache.CacheLinesPerTag);
    /* CacheSiblings */
    snprintf(path, 256, "%s/index%d/shared_cpu_map", prefix, idx);
    fscanf_str(path, str);
    cpumap_to_cpu_ci(str, cpuinfo, &this_cache);

    cache.push_back(this_cache);
  }

  return cpu_ci.num_caches;
}

static HSAKMT_STATUS topology_map_node_id(uint32_t node_id,
                                          wsl::thunk::WDDMDevice *&device) {
  uint32_t idx = node_id;
  if ((!dxg_topology->wdevices_.size()) || (!node_id) || (node_id >= dxg_topology->num_sysfs_nodes)) {
    device = nullptr;
    return HSAKMT_STATUS_ERROR;
  }

  device = dxg_topology->wdevices_[node_id - 1];
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS topology_sysfs_get_system_props(HsaSystemProperties& props) {
  HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;
  bool is_node_supported = true;
  uint32_t num_supported_nodes = 0;

  std::memset(&props, 0, sizeof(props));

  D3DKMT_ADAPTERINFO *adapters;
  int num_adapters;
  if (wsl::thunk::WDDMGetAdapters(adapters, num_adapters) != STATUS_SUCCESS) {
    pr_err("Failed to get adapters\n");
    ret = HSAKMT_STATUS_ERROR;
    goto err;
  }

  dxg_topology->num_sysfs_nodes = num_adapters + 1;

  for (auto device : dxg_topology->wdevices_)
    delete device;
  dxg_topology->wdevices_.clear();

  for (uint32_t i = 0; i < num_adapters; i++) {
    wsl::thunk::WDDMDevice *device = new wsl::thunk::WDDMDevice(
        adapters[i].hAdapter, adapters[i].AdapterLuid, i+1);
    assert(device && "Create WDDM Device fail");
    dxg_topology->wdevices_.push_back(device);
  }
  props.NumNodes = dxg_topology->num_sysfs_nodes;

  delete[] adapters;
  return ret;
err:
  return ret;
}

void topology_setup_is_dgpu_param(HsaNodeProperties *props) {
  /* if we found a dGPU node, then treat the whole system as dGPU */
  /* noted that some APUs are also treated as dGPU in runtime */
  if (!props->NumCPUCores && props->NumFComputeCores)
    dxg_runtime->hsakmt_is_dgpu = true;
}

static HSAKMT_STATUS topology_get_cpu_model_name(HsaNodeProperties& props,
                                                 const std::vector<proc_cpuinfo>& cpuinfo) {
  for (int i = 0; i < cpuinfo.size(); i++) {
    if (props.CComputeIdLo == cpuinfo[i].apicid) {
      if (!props.DeviceId) /* CPU-only node */
        strncpy((char *)props.AMDName, cpuinfo[i].model_name,
                sizeof(props.AMDName));
      /* Convert from UTF8 to UTF16 */
      int j;
      for (j = 0;
           cpuinfo[i].model_name[j] != '\0' && j < HSA_PUBLIC_NAME_SIZE - 1; j++)
        props.MarketingName[j] = cpuinfo[i].model_name[j];
      props.MarketingName[j] = '\0';
      return HSAKMT_STATUS_SUCCESS;
    }
  }

  return HSAKMT_STATUS_ERROR;
}

static int topology_search_processor_vendor(const std::string& processor_name) {
  for (unsigned int i = 0; i < ARRAY_LEN(supported_processor_vendor_name); i++) {
    if (processor_name == supported_processor_vendor_name[i])
      return i;
    if (processor_name == "POWER9, altivec supported")
      return IBM_POWER;
  }
  return -1;
}

/* topology_parse_cpuinfo - Parse /proc/cpuinfo and fill up required
 *			topology information
 * cpuinfo [OUT]: output buffer to hold cpu information
 * num_procs: number of processors the output buffer can hold
 */
static HSAKMT_STATUS topology_parse_cpuinfo(std::vector<proc_cpuinfo>& cpuinfo) {
  HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;
  uint32_t num_procs = cpuinfo.size();

  std::ifstream cpuinfo_max_freq(
      "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq");
  if (cpuinfo_max_freq) {
    std::string line;
    std::getline(cpuinfo_max_freq, line);
    dxg_topology->freq_max_ = static_cast<uint32_t>(std::stod(line) / 1000);
  }

  std::ifstream cpuinfo_file("/proc/cpuinfo");
  if (!cpuinfo_file) {
    pr_err("Failed to open /proc/cpuinfo. Unable to get CPU information");
    return HSAKMT_STATUS_ERROR;
  }

  std::string line;
  uint32_t proc = 0;
  while (std::getline(cpuinfo_file, line)) {
    if (line.substr(0, 9) == "processor") {
      proc = std::stoi(line.substr(line.find(':') + 2));
      if (proc >= num_procs) {
        pr_err("cpuinfo contains processor %d larger than %u\n", proc, num_procs);
        return HSAKMT_STATUS_NO_MEMORY;
      }
      continue;
    }

    if (line.substr(0, 9) == "vendor_id" && dxg_topology->processor_vendor == -1) {
      std::string vendor = line.substr(line.find(':') + 2);
      dxg_topology->processor_vendor = topology_search_processor_vendor(vendor.c_str());
      continue;
    }

    if (line.substr(0, 10) == "model name") {
      std::string model_name = line.substr(line.find(':') + 2);
      if (model_name.size() > HSA_PUBLIC_NAME_SIZE)
      model_name.resize(HSA_PUBLIC_NAME_SIZE);
      std::strncpy(cpuinfo[proc].model_name, model_name.c_str(), HSA_PUBLIC_NAME_SIZE);
      continue;
    }

    if (line.substr(0, 6) == "apicid") {
      cpuinfo[proc].apicid = std::stoi(line.substr(line.find(':') + 2));
      continue;
    }

    if (!cpuinfo_max_freq) {
      if (line.substr(0, 7) == "cpu MHz") {
        double freq = std::stod(line.substr(line.find(':') + 2));
        if (freq > dxg_topology->freq_max_) {
          dxg_topology->freq_max_ = freq;
        }
        continue;
      }
    }
  }

  if (dxg_topology->processor_vendor < 0) {
    pr_err("Failed to get Processor Vendor. Setting to %s", supported_processor_vendor_name[GENUINE_INTEL]);
    dxg_topology->processor_vendor = GENUINE_INTEL;
  }

  return ret;
}

static HSAKMT_STATUS topology_sysfs_get_node_props(uint32_t node_id,
                                                   HsaNodeProperties& props,
                                                   bool& p2p_links,
                                                   uint32_t& num_p2pLinks) {
  HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;

  memset(&props, 0, sizeof(props));
  p2p_links = false;
  num_p2pLinks = 0;

  props.MaxEngineClockMhzCCompute = dxg_topology->freq_max_;

  if (node_id == 0) {
    /* CPU node */
    props.NumCPUCores = sysconf(_SC_NPROCESSORS_ONLN);
    props.NumMemoryBanks = 1;
    props.KFDGpuID = 0;
    return HSAKMT_STATUS_SUCCESS;
  }

  /* gpu node */
  wsl::thunk::WDDMDevice *device;
  ret = topology_map_node_id(node_id, device);
  if (ret != HSAKMT_STATUS_SUCCESS)
    return ret;

  props.NumCPUCores = 0;
  props.NumFComputeCores = device->SimdPerCu() * device->ComputeUnitCount();
  props.NumMemoryBanks = 1;
  props.NumCaches = 3;
  props.NumIOLinks = 1;
  props.CComputeIdLo = 0;
  props.FComputeIdLo = 0;
  props.Capability.ui32.ASICRevision = device->AsicRevision();
  props.Capability.ui32.WatchPointsTotalBits =
      std::log2(device->WatchPointsNum());
  props.MaxWavesPerSIMD = device->WavePerCu() / device->SimdPerCu();
  props.LDSSizeInKB = device->LdsSize() / 1024;
  props.GDSSizeInKB = 0;
  props.WaveFrontSize = device->WavefrontSize();
  props.NumShaderBanks = device->NumShaderEngine();
  props.NumArrays = device->ShaderArrayPerShaderEngine();
  props.NumCUPerArray = device->ComputeUnitCount() / props.NumArrays;
  props.NumSIMDPerCU = device->SimdPerCu();
  props.MaxSlotsScratchCU = device->MaxScratchSlotsPerCu();
  props.VendorId = 0x1002;
  props.DeviceId = device->DeviceId();
  props.LocationId = device->PciBusAddr();
  props.LocalMemSize = 0;
  props.MaxEngineClockMhzFCompute = device->MaxEngineClockMhz();
  props.DrmRenderMinor = node_id;

  {
    int i;
    const char *name = device->ProductName();
    for (i = 0; name[i] != 0 && i < HSA_PUBLIC_NAME_SIZE - 1; i++)
      props.MarketingName[i] = name[i];
    props.MarketingName[i] = '\0';
  }
  props.uCodeEngineVersions.uCodeSDMA = device->GetSdmaFwVersion();
  props.DebugProperties.Value = 0;
  props.HiveID = 0;
  props.NumSdmaEngines = device->NumSdmaEngine();
  props.NumSdmaXgmiEngines = 0;
  props.NumSdmaQueuesPerEngine = 6; // TODO
  props.NumCpQueues = device->GetNumCpQueues();
  props.NumGws = 0;
  /*
   * In Native Linux, if the asic is APU, this value will be set to 1,
   * if the asic is dGPU, this value will be set to 0. clr use this info
   * to set hostUnifiedMemory_, but for now wsl does not support this feature.
   * Therefore, fore vaule to 0 temporarily.
   */
  props.Integrated = 0;
  props.Domain = device->Domain();
  props.UniqueID = device->Uuid();
  props.NumXcc = 1;
  props.KFDGpuID = device->DeviceId(); // TODO
  props.FamilyID = device->GfxFamily();

  props.EngineId.ui32.uCode = device->GetMecFwVersion();
  char *envvar = getenv("HSA_OVERRIDE_GFX_VERSION");
  if (envvar) {
    char dummy = '\0';
    uint32_t major = 0, minor = 0, step = 0;
    /* HSA_OVERRIDE_GFX_VERSION=major.minor.stepping */
    if ((sscanf(envvar, "%u.%u.%u%c", &major, &minor, &step, &dummy) != 3) ||
        (major > 63 || minor > 255 || step > 255)) {
      pr_err("HSA_OVERRIDE_GFX_VERSION %s is invalid\n", envvar);
      return HSAKMT_STATUS_ERROR;
    }
    props.OverrideEngineId.ui32.Major = major & 0x3f;
    props.OverrideEngineId.ui32.Minor = minor & 0xff;
    props.OverrideEngineId.ui32.Stepping = step & 0xff;
  } else {
    props.EngineId.ui32.Major = device->Major();
    props.EngineId.ui32.Minor = device->Minor();
    props.EngineId.ui32.Stepping = device->Stepping();
  }

  snprintf((char *)props.AMDName, sizeof(props.AMDName) - 1, "GFX%06x",
           HSA_GET_GFX_VERSION_FULL(props.EngineId.ui32));

  if (!dxg_runtime->is_svm_api_supported)
    props.Capability.ui32.SVMAPISupported = 0;
  props.Capability.ui32.DoorbellType = 2;

  /* Get VGPR/SGPR size in byte per CU */
  props.SGPRSizePerCU = SGPR_SIZE_PER_CU;
  props.VGPRSizePerCU = get_vgpr_size_per_cu(props.EngineId);

  if (props.NumFComputeCores)
    assert(props.EngineId.ui32.Major &&
           "HSA_OVERRIDE_GFX_VERSION may be needed");

  return ret;
}

static HSAKMT_STATUS topology_sysfs_get_mem_props(uint32_t node_id,
                                                  uint32_t mem_id,
                                                  HsaMemoryProperties& props) {
  HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;

  std::memset(&props, 0, sizeof(props));
  if (node_id == 0) {
    /* CPU node */
    props.HeapType = HSA_HEAPTYPE_SYSTEM;

    struct sysinfo info;
    sysinfo(&info);
    props.SizeInBytes = info.totalram;

    /* props.SizeInBytes is the actual physical system
     * memory size. Reserve 1/16th for WSL system usage.
     */
    dxg_runtime->max_single_alloc_size = info.totalram - (info.totalram >> 4);

    props.Flags.MemoryProperty = 0;
    /* TODO: sudo dmidecode --type memory doesn't work on wsl */
    props.Width = 64;
    props.MemoryClockMax = 2133;
    return HSAKMT_STATUS_SUCCESS;
  }

  wsl::thunk::WDDMDevice *device;
  ret = topology_map_node_id(node_id, device);
  if (ret != HSAKMT_STATUS_SUCCESS)
    return ret;

  props.HeapType = HSA_HEAPTYPE_FRAME_BUFFER_PRIVATE;
  props.SizeInBytes = device->LocalHeapSize();
  props.Width = device->MemoryBusWidth();
  props.MemoryClockMax = device->MaxMemoryClockMhz();

  return ret;
}

/* topology_get_cpu_cache_props - Read CPU cache information from sysfs
 *	@node [IN] CPU node number
 *	@cpuinfo [IN] /proc/cpuinfo data
 *	@tbl [OUT] the node table to fill up
 * Return: HSAKMT_STATUS_SUCCESS in success or error number in failure
 */
static HSAKMT_STATUS topology_get_cpu_cache_props(int node,
                                                  const std::vector<proc_cpuinfo>& cpuinfo,
                                                  node_props_t& tbl) {
  HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;

  /* Get max path size from /sys/devices/system/node/node%d/%s/cache
   * below, which will max out according to the largest filename,
   * which can be present twice in the string above. 29 is for the prefix
   * and the +6 is for the cache suffix
   */
#ifndef MAXNAMLEN
/* MAXNAMLEN is the BSD name for NAME_MAX. glibc aliases this as NAME_MAX, but
 * not musl */
#define MAXNAMLEN NAME_MAX
#endif
  constexpr uint32_t MAXPATHSIZE = 29 + MAXNAMLEN + (MAXNAMLEN + 6);
  char path[MAXPATHSIZE], node_dir[MAXPATHSIZE];
  int max_cpus;
  int cache_cnt = 0;
  DIR *dirp = NULL;
  struct dirent *dir;
  char *p;

  /* Get info from /sys/devices/system/node/nodeX/cpuY/cache */
  int node_real = node;
  if (dxg_topology->processor_vendor == IBM_POWER) {
    if (!strcmp(cpuinfo[0].model_name, "POWER9")) {
      node_real = node * 8;
    }
  }
  snprintf(node_dir, MAXPATHSIZE, "/sys/devices/system/node/node%d", node_real);
  /* Other than cpuY folders, this dir also has cpulist and cpumap */
  max_cpus = num_subdirs(node_dir, "cpu");
  if (max_cpus <= 0) {
    /* If CONFIG_NUMA is not enabled in the kernel,
     * /sys/devices/system/node doesn't exist.
     */
    if (node) { /* CPU node must be 0 or something is wrong */
      pr_err("Fail to get cpu* dirs under %s.", node_dir);
      ret = HSAKMT_STATUS_ERROR;
      goto exit;
    }
    /* Fall back to use /sys/devices/system/cpu */
    snprintf(node_dir, MAXPATHSIZE, "/sys/devices/system/cpu");
    max_cpus = num_subdirs(node_dir, "cpu");
    if (max_cpus <= 0) {
      pr_err("Fail to get cpu* dirs under %s\n", node_dir);
      ret = HSAKMT_STATUS_ERROR;
      goto exit;
    }
  }

  dirp = opendir(node_dir);
  while ((dir = readdir(dirp)) != 0) {
    if (strncmp(dir->d_name, "cpu", 3))
      continue;
    if (!isdigit(dir->d_name[3])) /* ignore files like cpulist */
      continue;
    if (strlen(node_dir) + strlen(dir->d_name) + strlen("/cache") + 2 < MAXPATHSIZE) {
      std::string path_str = std::string(node_dir) + "/" + dir->d_name + "/cache";
      strncpy(path, path_str.c_str(), MAXPATHSIZE);
      path[MAXPATHSIZE - 1] = '\0';
    } else {
      pr_err("Path is too long and was truncated.\n");
      goto exit;
    }

    cpu_cacheinfo_t cpu_ci;
    cpu_ci.num_caches = num_subdirs(path, "index");
    cpu_ci.proc_num= atoi(dir->d_name+3);

    cache_cnt += get_cpu_cache_info(path, cpuinfo, tbl.cache, cpu_ci);
  }
  assert(cache_cnt == tbl.cache.size());
  tbl.node.NumCaches = cache_cnt;

exit:
  if (dirp)
    closedir(dirp);
  return ret;
}

/* For a give Node @node_id the function gets @iolink_id information i.e. parses
 * sysfs the following sysfs entry
 * ./nodes/@node_id/io_links/@iolink_id/properties. @node_id has to be valid
 * accessible node.
 *
 * If node_to specified by the @iolink_id is not accessible the function returns
 * HSAKMT_STATUS_NOT_SUPPORTED. If node_to is accessible, then node_to is mapped
 * from sysfs_node to user_node and returns HSAKMT_STATUS_SUCCESS.
 */
static HSAKMT_STATUS topology_sysfs_get_iolink_props(uint32_t node_id,
                                                     uint32_t iolink_id,
                                                     HsaIoLinkProperties& props,
                                                     bool p2pLink) {
  wsl::thunk::WDDMDevice *device;
  topology_map_node_id(node_id, device);

  std::memset(&props, 0, sizeof(props));
  props.IoLinkType = HSA_IOLINKTYPE_PCIEXPRESS;
  props.VersionMajor = props.VersionMinor = 0;
  props.NodeFrom = node_id;
  props.NodeTo = 0;
  props.Weight = 20;
  props.Flags.ui32.Override = 1;
  props.Flags.ui32.NonCoherent = 1;
  props.Flags.ui32.NoAtomics32bit = !(device->SupportPlatformAtomic());
  props.Flags.ui32.NoAtomics64bit = !(device->SupportPlatformAtomic());
  props.RecSdmaEngIdMask = 0;

  return HSAKMT_STATUS_SUCCESS;
}

/* topology_get_free_io_link_slot_for_node - For the given node_id, find the
 * next available free slot to add an io_link
 */
static HsaIoLinkProperties *
topology_get_free_io_link_slot_for_node(uint32_t node_id,
                                        const HsaSystemProperties& sys_props,
                                        std::vector<node_props_t>& node_props) {
  std::vector<HsaIoLinkProperties>& props = node_props[node_id].link;

  if (node_id >= sys_props.NumNodes) {
    pr_err("Invalid node [%d]\n", node_id);
    return NULL;
  }

  if (!props.size()) {
    pr_err("No io_link reported for Node [%d]\n", node_id);
    return NULL;
  }

  if (node_props[node_id].node.NumIOLinks >= sys_props.NumNodes - 1) {
    pr_err("No more space for io_link for Node [%d]\n", node_id);
    return NULL;
  }

  return &props[node_props[node_id].node.NumIOLinks];
}

/* topology_add_io_link_for_node - If a free slot is available,
 * add io_link for the given Node.
 * TODO: Add other members of HsaIoLinkProperties
 */
static HSAKMT_STATUS topology_add_io_link_for_node(
    uint32_t node_from, const HsaSystemProperties& sys_props,
    std::vector<node_props_t>& node_props, HSA_IOLINKTYPE IoLinkType, uint32_t node_to,
    uint32_t Weight) {
  HsaIoLinkProperties *props;

  props =
      topology_get_free_io_link_slot_for_node(node_from, sys_props, node_props);
  if (!props)
    return HSAKMT_STATUS_NO_MEMORY;

  props->IoLinkType = IoLinkType;
  props->NodeFrom = node_from;
  props->NodeTo = node_to;
  props->Weight = Weight;
  node_props[node_from].node.NumIOLinks++;

  return HSAKMT_STATUS_SUCCESS;
}

/* Find the CPU that this GPU (gpu_node) directly connects to */
static int32_t gpu_get_direct_link_cpu(uint32_t gpu_node,
                                       const std::vector<node_props_t>& node_props) {
  const std::vector<HsaIoLinkProperties>& props = node_props[gpu_node].link;
  uint32_t i;

  if (!node_props[gpu_node].node.KFDGpuID || props.empty() ||
      node_props[gpu_node].node.NumIOLinks == 0)
    return -1;

  for (i = 0; i < node_props[gpu_node].node.NumIOLinks; i++)
    if (props[i].IoLinkType == HSA_IOLINKTYPE_PCIEXPRESS &&
        props[i].Weight <= 20) /* >20 is GPU->CPU->GPU */
      return props[i].NodeTo;

  return -1;
}

/* Get node1->node2 IO link information. This should be a direct link that has
 * been created in the kernel.
 */
static HSAKMT_STATUS get_direct_iolink_info(uint32_t node1, uint32_t node2,
                                            const std::vector<node_props_t>& node_props,
                                            HSAuint32 *weight,
                                            HSA_IOLINKTYPE *type) {
  const std::vector<HsaIoLinkProperties>& props = node_props[node1].link;
  uint32_t i;

  if (!props.size())
    return HSAKMT_STATUS_INVALID_NODE_UNIT;

  for (i = 0; i < node_props[node1].node.NumIOLinks; i++)
    if (props[i].NodeTo == node2) {
      if (weight)
        *weight = props[i].Weight;
      if (type)
        *type = props[i].IoLinkType;
      return HSAKMT_STATUS_SUCCESS;
    }

  return HSAKMT_STATUS_INVALID_PARAMETER;
}

static HSAKMT_STATUS get_indirect_iolink_info(uint32_t node1, uint32_t node2,
                                              const std::vector<node_props_t>& node_props,
                                              HSAuint32 *weight,
                                              HSA_IOLINKTYPE *type) {
  int32_t dir_cpu1 = -1, dir_cpu2 = -1;
  HSAKMT_STATUS ret;
  uint32_t i;

  *weight = 0;
  *type = HSA_IOLINKTYPE_UNDEFINED;

  if (node1 == node2)
    return HSAKMT_STATUS_INVALID_PARAMETER;

  /* CPU->CPU is not an indirect link */
  if (!node_props[node1].node.KFDGpuID && !node_props[node2].node.KFDGpuID)
    return HSAKMT_STATUS_INVALID_NODE_UNIT;

  if (node_props[node1].node.HiveID && node_props[node2].node.HiveID &&
      node_props[node1].node.HiveID == node_props[node2].node.HiveID)
    return HSAKMT_STATUS_INVALID_PARAMETER;

  if (node_props[node1].node.KFDGpuID)
    dir_cpu1 = gpu_get_direct_link_cpu(node1, node_props);
  if (node_props[node2].node.KFDGpuID)
    dir_cpu2 = gpu_get_direct_link_cpu(node2, node_props);

  if (dir_cpu1 < 0 && dir_cpu2 < 0)
    return HSAKMT_STATUS_ERROR;

  /* if the node2(dst) is GPU , it need to be large bar for host access*/
  if (node_props[node2].node.KFDGpuID) {
    for (i = 0; i < node_props[node2].node.NumMemoryBanks; ++i)
      if (node_props[node2].mem[i].HeapType == HSA_HEAPTYPE_FRAME_BUFFER_PUBLIC)
        break;
    if (i >= node_props[node2].node.NumMemoryBanks)
      return HSAKMT_STATUS_ERROR;
  }
  /* Possible topology:
   *   GPU --(weight1) -- CPU -- (weight2) -- GPU
   *   GPU --(weight1) -- CPU -- (weight2) -- CPU -- (weight3) -- GPU
   *   GPU --(weight1) -- CPU -- (weight2) -- CPU
   *   CPU -- (weight2) -- CPU -- (weight3) -- GPU
   */
  HSAuint32 weight1 = 0, weight2 = 0, weight3 = 0;
  if (dir_cpu1 >= 0) { /* GPU->CPU ... */
    if (dir_cpu2 >= 0) {
      if (dir_cpu1 == dir_cpu2) /* GPU->CPU->GPU*/ {
        ret =
            get_direct_iolink_info(node1, dir_cpu1, node_props, &weight1, NULL);
        if (ret != HSAKMT_STATUS_SUCCESS)
          return ret;
        ret =
            get_direct_iolink_info(dir_cpu1, node2, node_props, &weight2, type);
      } else /* GPU->CPU->CPU->GPU*/ {
        ret =
            get_direct_iolink_info(node1, dir_cpu1, node_props, &weight1, NULL);
        if (ret != HSAKMT_STATUS_SUCCESS)
          return ret;
        ret = get_direct_iolink_info(dir_cpu1, dir_cpu2, node_props, &weight2,
                                     type);
        if (ret != HSAKMT_STATUS_SUCCESS)
          return ret;
        /* On QPI interconnection, GPUs can't access
         * each other if they are attached to different
         * CPU sockets. CPU<->CPU weight larger than 20
         * means the two CPUs are in different sockets.
         */
        if (*type == HSA_IOLINK_TYPE_QPI_1_1 && weight2 > 20)
          return HSAKMT_STATUS_NOT_SUPPORTED;
        ret =
            get_direct_iolink_info(dir_cpu2, node2, node_props, &weight3, NULL);
      }
    } else /* GPU->CPU->CPU */ {
      ret = get_direct_iolink_info(node1, dir_cpu1, node_props, &weight1, NULL);
      if (ret != HSAKMT_STATUS_SUCCESS)
        return ret;
      ret = get_direct_iolink_info(dir_cpu1, node2, node_props, &weight2, type);
    }
  } else { /* CPU->CPU->GPU */
    ret = get_direct_iolink_info(node1, dir_cpu2, node_props, &weight2, type);
    if (ret != HSAKMT_STATUS_SUCCESS)
      return ret;
    ret = get_direct_iolink_info(dir_cpu2, node2, node_props, &weight3, NULL);
  }

  if (ret != HSAKMT_STATUS_SUCCESS)
    return ret;

  *weight = weight1 + weight2 + weight3;
  return HSAKMT_STATUS_SUCCESS;
}

static void
topology_create_indirect_gpu_links(const HsaSystemProperties& sys_props,
                                   std::vector<node_props_t>& node_props) {

  uint32_t i, j;
  HSAuint32 weight;
  HSA_IOLINKTYPE type;

  for (i = 0; i < sys_props.NumNodes - 1; i++) {
    for (j = i + 1; j < sys_props.NumNodes; j++) {
      get_indirect_iolink_info(i, j, node_props, &weight, &type);
      if (!weight)
        goto try_alt_dir;
      if (topology_add_io_link_for_node(i, sys_props, node_props, type, j,
                                        weight) != HSAKMT_STATUS_SUCCESS)
        pr_err("Fail to add IO link %d->%d\n", i, j);
    try_alt_dir:
      get_indirect_iolink_info(j, i, node_props, &weight, &type);
      if (!weight)
        continue;
      if (topology_add_io_link_for_node(j, sys_props, node_props, type, i,
                                        weight) != HSAKMT_STATUS_SUCCESS)
        pr_err("Fail to add IO link %d->%d\n", j, i);
    }
  }
}

HSAKMT_STATUS topology_take_snapshot(void) {
  uint32_t i, mem_id, cache_id;
  HsaSystemProperties sys_props;
  std::vector<node_props_t>& temp_props = dxg_topology->g_props;
  HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;
  const uint32_t num_procs = sysconf(_SC_NPROCESSORS_ONLN);
  std::vector<proc_cpuinfo> cpuinfo(num_procs);
  uint32_t num_ioLinks;
  bool p2p_links = false;
  uint32_t num_p2pLinks = 0;

  topology_parse_cpuinfo(cpuinfo);

  ret = topology_sysfs_get_system_props(sys_props);
  if (ret != HSAKMT_STATUS_SUCCESS)
    goto err;
  if (sys_props.NumNodes > 0) {
    temp_props.resize(sys_props.NumNodes);

    for (i = 0; i < sys_props.NumNodes; i++) {
      wsl::thunk::WDDMDevice *device_;
      topology_map_node_id(i, device_);

      ret = topology_sysfs_get_node_props(i, temp_props[i].node, p2p_links,
                                          num_p2pLinks);
      if (ret != HSAKMT_STATUS_SUCCESS) {
        goto err;
      }

      topology_setup_is_dgpu_param(&temp_props[i].node);

      if (temp_props[i].node.NumCPUCores)
        topology_get_cpu_model_name(temp_props[i].node, cpuinfo);

      if (temp_props[i].node.NumMemoryBanks) {
        temp_props[i].mem.resize(temp_props[i].node.NumMemoryBanks);

        for (mem_id = 0; mem_id < temp_props[i].node.NumMemoryBanks; mem_id++) {
          ret = topology_sysfs_get_mem_props(i, mem_id,
                                             temp_props[i].mem[mem_id]);
          if (ret != HSAKMT_STATUS_SUCCESS) {
            goto err;
          }
        }
      }

      if (temp_props[i].node.NumCaches) {
        temp_props[i].cache.resize(temp_props[i].node.NumCaches);
        for (int j = 0; j < 3; j++) {
          temp_props[i].cache[j].CacheType.ui32.Data = 1;
          temp_props[i].cache[j].CacheType.ui32.HSACU = 1;
          temp_props[i].cache[j].CacheLevel = j + 1;
        }
        temp_props[i].cache[0].CacheSize = device_->GetL1CacheSize() / 1024;
        temp_props[i].cache[1].CacheSize = device_->GetL2CacheSize() / 1024;
        temp_props[i].cache[2].CacheSize = device_->GetL3CacheSize() / 1024;
      } else if (!temp_props[i].node.KFDGpuID) { /* a CPU node */
        ret = topology_get_cpu_cache_props(i, cpuinfo, temp_props[i]);
        if (ret != HSAKMT_STATUS_SUCCESS) {
          goto err;
        }
      }

      /* To simplify, allocate maximum needed memory for io_links for each node.
       * This removes the need for realloc when indirect and QPI links are added
       * later
       */
      temp_props[i].link.resize(sys_props.NumNodes - 1);
      num_ioLinks = temp_props[i].node.NumIOLinks - num_p2pLinks;
      uint32_t link_id = 0;

      if (num_ioLinks) {
        uint32_t sys_link_id = 0;

        /* Parse all the sysfs specified io links. Skip the ones where the
         * remote node (node_to) is not accessible
         */
        while (sys_link_id < num_ioLinks && link_id < sys_props.NumNodes - 1) {
          ret = topology_sysfs_get_iolink_props(
              i, sys_link_id++, temp_props[i].link[link_id], false);
          if (ret == HSAKMT_STATUS_NOT_SUPPORTED) {
            ret = HSAKMT_STATUS_SUCCESS;
            continue;
          } else if (ret != HSAKMT_STATUS_SUCCESS) {
            goto err;
          }
          link_id++;
        }
        /* sysfs specifies all the io links. Limit the number to valid ones */
        temp_props[i].node.NumIOLinks = link_id;
      }

      if (num_p2pLinks) {
        uint32_t sys_link_id = 0;

        /* Parse all the sysfs specified p2p links.
         */
        while (sys_link_id < num_p2pLinks && link_id < sys_props.NumNodes - 1) {
          ret = topology_sysfs_get_iolink_props(
              i, sys_link_id++, temp_props[i].link[link_id], true);
          if (ret == HSAKMT_STATUS_NOT_SUPPORTED) {
            ret = HSAKMT_STATUS_SUCCESS;
            continue;
          } else if (ret != HSAKMT_STATUS_SUCCESS) {
            goto err;
          }
          link_id++;
        }
        temp_props[i].node.NumIOLinks = link_id;
      }
    }
  }

  if (!p2p_links) {
    /* All direct IO links are created in the kernel. Here we need to
     * connect GPU<->GPU or GPU<->CPU indirect IO links.
     */
    topology_create_indirect_gpu_links(sys_props, temp_props);
  }

  if (!dxg_topology->g_system) {
    dxg_topology->g_system = (HsaSystemProperties *)malloc(sizeof(HsaSystemProperties));
    if (!dxg_topology->g_system) {
      ret = HSAKMT_STATUS_NO_MEMORY;
      goto err;
    }
  }

  *dxg_topology->g_system = sys_props;
err:
  return ret;
}

/* Drop the Snashot of the HSA topology information. Assume lock is held. */
void topology_drop_snapshot(void) {
  if (!!dxg_topology->g_system != !!dxg_topology->g_props.size())
    pr_warn("Probably inconsistency?\n");

  dxg_topology->g_props.clear();

  free(dxg_topology->g_system);
  dxg_topology->g_system = NULL;

  trim_suballocator();
  for (auto device : dxg_topology->wdevices_)
    delete device;
  dxg_topology->wdevices_.clear();
}

HSAKMT_STATUS validate_nodeid(uint32_t nodeid, uint32_t *gpu_id) {
  if (dxg_topology->g_props.empty() || !dxg_topology->g_system || dxg_topology->g_system->NumNodes <= nodeid)
    return HSAKMT_STATUS_INVALID_NODE_UNIT;
  if (gpu_id)
    *gpu_id = dxg_topology->g_props[nodeid].node.KFDGpuID;

  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS gpuid_to_nodeid(uint32_t gpu_id, uint32_t *node_id) {
  uint64_t node_idx;

  for (node_idx = 0; node_idx < dxg_topology->g_system->NumNodes; node_idx++) {
    if (dxg_topology->g_props[node_idx].node.KFDGpuID == gpu_id) {
      *node_id = node_idx;
      return HSAKMT_STATUS_SUCCESS;
    }
  }

  return HSAKMT_STATUS_INVALID_NODE_UNIT;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtAcquireSystemProperties(HsaSystemProperties *SystemProperties) {
  HSAKMT_STATUS err = HSAKMT_STATUS_SUCCESS;

  CHECK_DXG_OPEN();

  if (!SystemProperties)
    return HSAKMT_STATUS_INVALID_PARAMETER;

  pthread_mutex_lock(&dxg_runtime->hsakmt_mutex);

  /* We already have a valid snapshot. Avoid double initialization that
   * would leak memory.
   */
  if (dxg_topology->g_system) {
    *SystemProperties = *dxg_topology->g_system;
    goto out;
  }

  err = topology_take_snapshot();
  if (err != HSAKMT_STATUS_SUCCESS)
    goto out;

  assert(dxg_topology->g_system);

  // err = fmm_init_process_apertures(dxg_topology->g_system->NumNodes);
  if (err != HSAKMT_STATUS_SUCCESS)
    goto init_process_apertures_failed;

  // err = init_process_doorbells(dxg_topology->g_system->NumNodes);
  if (err != HSAKMT_STATUS_SUCCESS)
    goto init_doorbells_failed;

  *SystemProperties = *dxg_topology->g_system;

  goto out;

init_doorbells_failed:
  // fmm_destroy_process_apertures();
init_process_apertures_failed:
  topology_drop_snapshot();

out:
  pthread_mutex_unlock(&dxg_runtime->hsakmt_mutex);
  return err;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtReleaseSystemProperties(void) {
  pthread_mutex_lock(&dxg_runtime->hsakmt_mutex);

  topology_drop_snapshot();

  pthread_mutex_unlock(&dxg_runtime->hsakmt_mutex);

  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS topology_get_node_props(HSAuint32 NodeId,
                                      HsaNodeProperties *NodeProperties) {
  if (!dxg_topology->g_system || dxg_topology->g_props.empty() || NodeId >= dxg_topology->g_system->NumNodes)
    return HSAKMT_STATUS_ERROR;

  *NodeProperties = dxg_topology->g_props[NodeId].node;
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtGetNodeProperties(HSAuint32 NodeId, HsaNodeProperties *NodeProperties) {
  HSAKMT_STATUS err;
  uint32_t gpu_id;

  if (!NodeProperties)
    return HSAKMT_STATUS_INVALID_PARAMETER;

  CHECK_DXG_OPEN();
  pthread_mutex_lock(&dxg_runtime->hsakmt_mutex);

  err = validate_nodeid(NodeId, &gpu_id);
  if (err != HSAKMT_STATUS_SUCCESS)
    goto out;

  err = topology_get_node_props(NodeId, NodeProperties);
  if (err != HSAKMT_STATUS_SUCCESS)
    goto out;
  /* For CPU only node don't add any additional GPU memory banks. */
  if (gpu_id) {
    uint64_t base, limit;
    if (!(NodeProperties->Integrated))
      NodeProperties->NumMemoryBanks += NUM_OF_DGPU_HEAPS;
    else
      NodeProperties->NumMemoryBanks += NUM_OF_IGPU_HEAPS;
    // TODO: for apu
    /*if (fmm_get_aperture_base_and_limit(FMM_MMIO, gpu_id, &base,
                    &limit) == HSAKMT_STATUS_SUCCESS)
            NodeProperties->NumMemoryBanks += 1;*/
  }

out:
  pthread_mutex_unlock(&dxg_runtime->hsakmt_mutex);
  return err;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtGetNodeMemoryProperties(HSAuint32 NodeId, HSAuint32 NumBanks,
                              HsaMemoryProperties *MemoryProperties) {
  HSAKMT_STATUS err = HSAKMT_STATUS_SUCCESS;
  uint32_t i;

  if (!MemoryProperties)
    return HSAKMT_STATUS_INVALID_PARAMETER;

  CHECK_DXG_OPEN();
  pthread_mutex_lock(&dxg_runtime->hsakmt_mutex);

  memset(MemoryProperties, 0, NumBanks * sizeof(HsaMemoryProperties));
  for (i = 0; i < wsl::Min(dxg_topology->g_props[NodeId].node.NumMemoryBanks, NumBanks); i++) {
    assert(dxg_topology->g_props[NodeId].mem.size());
    MemoryProperties[i] = dxg_topology->g_props[NodeId].mem[i];
  }

  /* The following memory banks does not apply to CPU only node */
  wsl::thunk::WDDMDevice *device_ = get_wddmdev(NodeId);
  if (device_ == nullptr)
    goto out;

  /*Add LDS*/
  if (i < NumBanks) {
    MemoryProperties[i].HeapType = HSA_HEAPTYPE_GPU_LDS;
    MemoryProperties[i].VirtualBaseAddress = device_->SharedApertureBase();
    MemoryProperties[i].SizeInBytes = dxg_topology->g_props[NodeId].node.LDSSizeInKB * 1024;
    i++;
  }

  /* Add SCRATCH */
  if (i < NumBanks) {
    MemoryProperties[i].HeapType = HSA_HEAPTYPE_GPU_SCRATCH;
    MemoryProperties[i].VirtualBaseAddress = device_->PrivateApertureBase();
    MemoryProperties[i].SizeInBytes = device_->PrivateApertureSize();
    i++;
  }

out:
  pthread_mutex_unlock(&dxg_runtime->hsakmt_mutex);
  return err;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtGetNodeCacheProperties(
    HSAuint32 NodeId, HSAuint32 ProcessorId, HSAuint32 NumCaches,
    HsaCacheProperties *CacheProperties) {
  HSAKMT_STATUS err;
  uint32_t i;

  if (!CacheProperties)
    return HSAKMT_STATUS_INVALID_PARAMETER;

  CHECK_DXG_OPEN();
  pthread_mutex_lock(&dxg_runtime->hsakmt_mutex);

  /* KFD ADD page 18, snapshot protocol violation */
  if (!dxg_topology->g_system || NodeId >= dxg_topology->g_system->NumNodes) {
    err = HSAKMT_STATUS_INVALID_NODE_UNIT;
    goto out;
  }

  if (NumCaches > dxg_topology->g_props[NodeId].node.NumCaches) {
    err = HSAKMT_STATUS_INVALID_PARAMETER;
    goto out;
  }

  for (i = 0; i < wsl::Min(dxg_topology->g_props[NodeId].node.NumCaches, NumCaches); i++) {
    assert(dxg_topology->g_props[NodeId].cache.size());
    CacheProperties[i] = dxg_topology->g_props[NodeId].cache[i];
  }

  err = HSAKMT_STATUS_SUCCESS;

out:
  pthread_mutex_unlock(&dxg_runtime->hsakmt_mutex);
  return err;
}

HSAKMT_STATUS topology_get_iolink_props(HSAuint32 NodeId, HSAuint32 NumIoLinks,
                                        HsaIoLinkProperties *IoLinkProperties) {
  if (!dxg_topology->g_system || dxg_topology->g_props.empty() || NodeId >= dxg_topology->g_system->NumNodes)
    return HSAKMT_STATUS_ERROR;

  memcpy(IoLinkProperties, dxg_topology->g_props[NodeId].link.data(),
         NumIoLinks * sizeof(*IoLinkProperties));

  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtGetNodeIoLinkProperties(HSAuint32 NodeId, HSAuint32 NumIoLinks,
                              HsaIoLinkProperties *IoLinkProperties) {
  HSAKMT_STATUS err;

  if (!IoLinkProperties)
    return HSAKMT_STATUS_INVALID_PARAMETER;

  CHECK_DXG_OPEN();

  pthread_mutex_lock(&dxg_runtime->hsakmt_mutex);

  /* KFD ADD page 18, snapshot protocol violation */
  if (!dxg_topology->g_system || NodeId >= dxg_topology->g_system->NumNodes) {
    err = HSAKMT_STATUS_INVALID_NODE_UNIT;
    goto out;
  }

  if (NumIoLinks > dxg_topology->g_props[NodeId].node.NumIOLinks) {
    err = HSAKMT_STATUS_INVALID_PARAMETER;
    goto out;
  }

  assert(dxg_topology->g_props[NodeId].link.size());
  err = topology_get_iolink_props(NodeId, NumIoLinks, IoLinkProperties);

out:
  pthread_mutex_unlock(&dxg_runtime->hsakmt_mutex);
  return err;
}

uint16_t get_device_id_by_node_id(HSAuint32 node_id) {
  if (dxg_topology->g_props.empty() || !dxg_topology->g_system || dxg_topology->g_system->NumNodes <= node_id)
    return 0;

  return dxg_topology->g_props[node_id].node.DeviceId;
}

bool prefer_ats(HSAuint32 node_id) {
  return dxg_topology->g_props[node_id].node.Capability.ui32.HSAMMUPresent &&
         dxg_topology->g_props[node_id].node.NumCPUCores &&
         dxg_topology->g_props[node_id].node.NumFComputeCores;
}

uint16_t get_device_id_by_gpu_id(HSAuint32 gpu_id) {
  unsigned int i;

  if (dxg_topology->g_props.empty() || !dxg_topology->g_system)
    return 0;

  for (i = 0; i < dxg_topology->g_system->NumNodes; i++) {
    if (dxg_topology->g_props[i].node.KFDGpuID == gpu_id)
      return dxg_topology->g_props[i].node.DeviceId;
  }

  return 0;
}

uint32_t get_direct_link_cpu(uint32_t gpu_node) {
  HSAuint64 size = 0;
  int32_t cpu_id;
  HSAuint32 i;

  cpu_id = gpu_get_direct_link_cpu(gpu_node, dxg_topology->g_props);
  if (cpu_id == -1)
    return INVALID_NODEID;

  assert(dxg_topology->g_props[cpu_id].mem.size());

  for (i = 0; i < dxg_topology->g_props[cpu_id].node.NumMemoryBanks; i++)
    size += dxg_topology->g_props[cpu_id].mem[i].SizeInBytes;

  return size ? (uint32_t)cpu_id : INVALID_NODEID;
}

HSAKMT_STATUS validate_nodeid_array(uint32_t **gpu_id_array,
                                    uint32_t NumberOfNodes,
                                    uint32_t *NodeArray) {
  HSAKMT_STATUS ret;
  unsigned int i;

  if (NumberOfNodes == 0 || !NodeArray || !gpu_id_array)
    return HSAKMT_STATUS_INVALID_PARAMETER;

  /* Translate Node IDs to gpu_ids */
  *gpu_id_array = (uint32_t *)malloc(NumberOfNodes * sizeof(uint32_t));
  if (!(*gpu_id_array))
    return HSAKMT_STATUS_NO_MEMORY;
  for (i = 0; i < NumberOfNodes; i++) {
    ret = validate_nodeid(NodeArray[i], *gpu_id_array + i);
    if (ret != HSAKMT_STATUS_SUCCESS) {
      free(*gpu_id_array);
      break;
    }
  }

  return ret;
}

uint32_t get_num_sysfs_nodes(void) { return dxg_topology->num_sysfs_nodes; }

wsl::thunk::WDDMDevice *get_wddmdev(uint32_t node_id) {
  if ((!dxg_topology->wdevices_.size()) || (!node_id) || (node_id >= dxg_topology->num_sysfs_nodes))
    return nullptr;

  return dxg_topology->wdevices_[node_id - 1];
}

uint32_t get_num_wddmdev() {
  return dxg_topology->wdevices_.size();
}
