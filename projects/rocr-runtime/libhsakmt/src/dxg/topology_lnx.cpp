/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
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
#include "impl/wddm/types.h"
#include "impl/wddm/device.h"
#include "util/utils.h"
#include "util/os.h"
#include "topology.hpp"

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
#if defined(__linux__)
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
#endif
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

HSAKMT_STATUS topology_map_node_id(uint32_t node_id,
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

  dxg_runtime->HeapFini();
  for (auto device : dxg_topology->wdevices_)
    delete device;
  dxg_topology->wdevices_.clear();

  WDDMCreateDevices(dxg_topology->wdevices_);
  int num_adapters = dxg_topology->wdevices_.size();
  if (num_adapters == 0) {
    pr_err("No WDDM adapters found.\n");
    return HSAKMT_STATUS_ERROR;
  }

  dxg_topology->num_sysfs_nodes = num_adapters + 1;
  dxg_runtime->HeapInit();
  props.NumNodes = dxg_topology->num_sysfs_nodes;
  if (dxg_runtime->default_node > num_adapters)
    dxg_runtime->default_node = num_adapters;

  return ret;
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

/* topology_parse_cpuinfo - Parse /proc/cpuinfo and fill up required
 *			topology information
 * cpuinfo [OUT]: output buffer to hold cpu information
 * num_procs: number of processors the output buffer can hold
 */
static HSAKMT_STATUS topology_parse_cpuinfo(std::vector<proc_cpuinfo>& cpuinfo) {
  HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;
  uint32_t num_procs = cpuinfo.size();

#if defined(__linux__)
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
#else
  rocr::os::cpuid_t cpuid{};
  rocr::os::ParseCpuID(&cpuid);
  dxg_topology->processor_vendor = topology_search_processor_vendor(cpuid.ManufacturerID);
  dxg_topology->freq_max_ = static_cast<double>(rocr::os::SystemClockFrequency());
#endif
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
#if defined(__linux__)
    props.NumCPUCores = sysconf(_SC_NPROCESSORS_ONLN);
#else
    SYSTEM_INFO si;
    ::GetSystemInfo(&si);
    props.NumCPUCores = si.dwNumberOfProcessors;
#endif
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
  props.Capability2.ui32.AqlEmulationPm4_ = device->IsAqlSupported() ? 0 : 1;

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


    props.SizeInBytes = rocr::os::HostTotalPhysicalMemory();

    /* props.SizeInBytes is the actual physical system
     * memory size. Reserve 1/16th for WSL system usage.
     */
    dxg_runtime->max_single_alloc_size = props.SizeInBytes - (props.SizeInBytes >> 4);

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
#if defined(__linux__)
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
#endif
  return ret;
}

HSAKMT_STATUS topology_take_snapshot(void) {
  uint32_t i, mem_id, cache_id;
  HsaSystemProperties sys_props;
  std::vector<node_props_t>& temp_props = dxg_topology->g_props;
  HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;
#if defined(__linux__)
  const uint32_t num_procs = sysconf(_SC_NPROCESSORS_ONLN);
#else
  SYSTEM_INFO si;
  ::GetSystemInfo(&si);
  const uint32_t num_procs = si.dwNumberOfProcessors;
#endif
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
