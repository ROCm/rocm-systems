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
                             const std::vector<struct proc_cpu_info>& cpuinfo,
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
                              const std::vector<struct proc_cpu_info>& cpuinfo,
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

/* topology_parse_cpuinfo - Parse /proc/cpuinfo and fill up required topology information
 * cpuinfo [OUT]: output buffer to hold cpu information
 * num_procs: number of processors the output buffer can hold
 */
HSAKMT_STATUS topology_parse_cpu_info(std::vector<proc_cpu_info>& cpuinfo) {
  HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;
  const uint32_t num_procs = sysconf(_SC_NPROCESSORS_ONLN);
  cpuinfo.resize(num_procs);

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
    cpuinfo[proc].group = 0;      // Not used for Linux
    cpuinfo[proc].numa_node = 0;  // Only 1 numa node for Wsl2
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

/* topology_get_cpu_cache_props - Read CPU cache information from sysfs
 *	@node [IN] CPU node number
 *	@cpuinfo [IN] /proc/cpuinfo data
 *	@tbl [OUT] the node table to fill up
 * Return: HSAKMT_STATUS_SUCCESS in success or error number in failure
 */
static HSAKMT_STATUS topology_get_cpu_cache_props(int node,
                                                  const std::vector<proc_cpu_info>& cpuinfo,
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

HSAKMT_STATUS topology_parse_numa_node_info(std::vector<proc_numa_node_info>& numa_node_info,
  const std::vector<proc_cpu_info>& cpu_info) {
  proc_numa_node_info node_info = {0, 0, 0};
  // WSL2 exposes exactly one NUMA node to the Linux guest, regardless of how many
  // NUMA nodes the Windows host actually has. 
  node_info.ccompute_id_low = cpu_info[0].apicid;  // apicid of the first logical cpu 
  node_info.count = cpu_info.size();  // All logical cpus belong to 1 numa node.
  numa_node_info.push_back(node_info);
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS topology_parse_cpu_cache_props(node_props_t* tbl,
                                             const std::vector<proc_cpu_info>& cpu_info) {
  HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;
  for (uint32_t node_id = 0; node_id < dxg_topology->numa_node_count_; node_id++) {
    ret = topology_get_cpu_cache_props(node_id, cpu_info, tbl[node_id]);
    if (ret != HSAKMT_STATUS_SUCCESS) {
      break;
    }
  }
  return ret;
}