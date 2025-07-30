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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <malloc.h>
#include <string.h>
#include <unistd.h>
#include <ctype.h>
#include <limits.h>

#include <errno.h>
#include <sys/sysinfo.h>
#include <xf86drm.h>
#include <amdgpu.h>
#include <amdgpu_drm.h>

#include "libhsakmt.h"
#include "hsakmt/hsakmtmodel.h"
#include "fmm.h"

/* Number of memory banks added by thunk on top of topology
 * This only includes static heaps like LDS, scratch and SVM,
 * not for MMIO_REMAP heap. MMIO_REMAP memory bank is reported
 * dynamically based on whether mmio aperture was mapped
 * successfully on this node.
 */
#define NUM_OF_IGPU_HEAPS 3
#define NUM_OF_DGPU_HEAPS 3
/* SYSFS related */
#define KFD_SYSFS_PATH "/sys/devices/virtual/kfd/kfd/topology"
#define KFD_SYSFS_PATH_GENERATION_ID "%s/generation_id"
#define KFD_SYSFS_PATH_SYSTEM_PROPERTIES "%s/system_properties"
#define KFD_SYSFS_PATH_NODES "%s/nodes"

static const char *get_topology_dir(void)
{
	if (hsakmt_use_model)
		return hsakmt_model_topology;
	return KFD_SYSFS_PATH;
}

typedef struct {
	HsaNodeProperties node;
	HsaMemoryProperties *mem;     /* node->NumBanks elements */
	HsaCacheProperties *cache;
	HsaIoLinkProperties *link;
} node_props_t;

static HsaSystemProperties *g_system;
static node_props_t *g_props;

/* This array caches sysfs based node IDs of CPU nodes + all supported GPU nodes.
 * It will be used to map user-node IDs to sysfs-node IDs.
 */
static uint32_t *map_user_to_sysfs_node_id;
static uint32_t map_user_to_sysfs_node_id_size;
static uint32_t num_sysfs_nodes;

static int processor_vendor = -1;
/* Supported System Vendors */
enum SUPPORTED_PROCESSOR_VENDORS {
	GENUINE_INTEL = 0,
	AUTHENTIC_AMD,
	IBM_POWER
};
/* Adding newline to make the search easier */
static const char *supported_processor_vendor_name[] = {
	"GenuineIntel\n",
	"AuthenticAMD\n",
	"\n"			// POWER requires a different search method
};

static HSAKMT_STATUS topology_take_snapshot(void);
static void topology_drop_snapshot(void);

static const struct hsa_gfxip_table gfxip_lookup_table[] = {
	/* Kaveri Family */
	{ 0x1304, 7, 0, 0, "Spectre" },
	{ 0x1305, 7, 0, 0, "Spectre" },
	{ 0x1306, 7, 0, 0, "Spectre" },
	{ 0x1307, 7, 0, 0, "Spectre" },
	{ 0x1309, 7, 0, 0, "Spectre" },
	{ 0x130A, 7, 0, 0, "Spectre" },
	{ 0x130B, 7, 0, 0, "Spectre" },
	{ 0x130C, 7, 0, 0, "Spectre" },
	{ 0x130D, 7, 0, 0, "Spectre" },
	{ 0x130E, 7, 0, 0, "Spectre" },
	{ 0x130F, 7, 0, 0, "Spectre" },
	{ 0x1310, 7, 0, 0, "Spectre" },
	{ 0x1311, 7, 0, 0, "Spectre" },
	{ 0x1312, 7, 0, 0, "Spooky" },
	{ 0x1313, 7, 0, 0, "Spectre" },
	{ 0x1315, 7, 0, 0, "Spectre" },
	{ 0x1316, 7, 0, 0, "Spooky" },
	{ 0x1317, 7, 0, 0, "Spooky" },
	{ 0x1318, 7, 0, 0, "Spectre" },
	{ 0x131B, 7, 0, 0, "Spectre" },
	{ 0x131C, 7, 0, 0, "Spectre" },
	{ 0x131D, 7, 0, 0, "Spectre" },
	/* Hawaii Family */
	{ 0x67A0, 7, 0, 1, "Hawaii" },
	{ 0x67A1, 7, 0, 1, "Hawaii" },
	{ 0x67A2, 7, 0, 1, "Hawaii" },
	{ 0x67A8, 7, 0, 1, "Hawaii" },
	{ 0x67A9, 7, 0, 1, "Hawaii" },
	{ 0x67AA, 7, 0, 1, "Hawaii" },
	{ 0x67B0, 7, 0, 1, "Hawaii" },
	{ 0x67B1, 7, 0, 1, "Hawaii" },
	{ 0x67B8, 7, 0, 1, "Hawaii" },
	{ 0x67B9, 7, 0, 1, "Hawaii" },
	{ 0x67BA, 7, 0, 1, "Hawaii" },
	{ 0x67BE, 7, 0, 1, "Hawaii" },
	/* Carrizo Family */
	{ 0x9870, 8, 0, 1, "Carrizo" },
	{ 0x9874, 8, 0, 1, "Carrizo" },
	{ 0x9875, 8, 0, 1, "Carrizo" },
	{ 0x9876, 8, 0, 1, "Carrizo" },
	{ 0x9877, 8, 0, 1, "Carrizo" },
	/* Tonga Family */
	{ 0x6920, 8, 0, 2, "Tonga" },
	{ 0x6921, 8, 0, 2, "Tonga" },
	{ 0x6928, 8, 0, 2, "Tonga" },
	{ 0x6929, 8, 0, 2, "Tonga" },
	{ 0x692B, 8, 0, 2, "Tonga" },
	{ 0x692F, 8, 0, 2, "Tonga" },
	{ 0x6930, 8, 0, 2, "Tonga" },
	{ 0x6938, 8, 0, 2, "Tonga" },
	{ 0x6939, 8, 0, 2, "Tonga" },
	/* Fiji */
	{ 0x7300, 8, 0, 3, "Fiji" },
	{ 0x730F, 8, 0, 3, "Fiji" },
	/* Polaris10 */
	{ 0x67C0, 8, 0, 3, "Polaris10" },
	{ 0x67C1, 8, 0, 3, "Polaris10" },
	{ 0x67C2, 8, 0, 3, "Polaris10" },
	{ 0x67C4, 8, 0, 3, "Polaris10" },
	{ 0x67C7, 8, 0, 3, "Polaris10" },
	{ 0x67C8, 8, 0, 3, "Polaris10" },
	{ 0x67C9, 8, 0, 3, "Polaris10" },
	{ 0x67CA, 8, 0, 3, "Polaris10" },
	{ 0x67CC, 8, 0, 3, "Polaris10" },
	{ 0x67CF, 8, 0, 3, "Polaris10" },
	{ 0x67D0, 8, 0, 3, "Polaris10" },
	{ 0x67DF, 8, 0, 3, "Polaris10" },
	{ 0x6FDF, 8, 0, 3, "Polaris10" },
	/* Polaris11 */
	{ 0x67E0, 8, 0, 3, "Polaris11" },
	{ 0x67E1, 8, 0, 3, "Polaris11" },
	{ 0x67E3, 8, 0, 3, "Polaris11" },
	{ 0x67E7, 8, 0, 3, "Polaris11" },
	{ 0x67E8, 8, 0, 3, "Polaris11" },
	{ 0x67E9, 8, 0, 3, "Polaris11" },
	{ 0x67EB, 8, 0, 3, "Polaris11" },
	{ 0x67EF, 8, 0, 3, "Polaris11" },
	{ 0x67FF, 8, 0, 3, "Polaris11" },
	/* Polaris12 */
	{ 0x6980, 8, 0, 3, "Polaris12" },
	{ 0x6981, 8, 0, 3, "Polaris12" },
	{ 0x6985, 8, 0, 3, "Polaris12" },
	{ 0x6986, 8, 0, 3, "Polaris12" },
	{ 0x6987, 8, 0, 3, "Polaris12" },
	{ 0x6995, 8, 0, 3, "Polaris12" },
	{ 0x6997, 8, 0, 3, "Polaris12" },
	{ 0x699F, 8, 0, 3, "Polaris12" },
	/* VegaM */
	{ 0x694C, 8, 0, 3, "VegaM" },
	{ 0x694E, 8, 0, 3, "VegaM" },
	{ 0x694F, 8, 0, 3, "VegaM" },
	/* Vega10 */
	{ 0x6860, 9, 0, 0, "Vega10" },
	{ 0x6861, 9, 0, 0, "Vega10" },
	{ 0x6862, 9, 0, 0, "Vega10" },
	{ 0x6863, 9, 0, 0, "Vega10" },
	{ 0x6864, 9, 0, 0, "Vega10" },
	{ 0x6867, 9, 0, 0, "Vega10" },
	{ 0x6868, 9, 0, 0, "Vega10" },
	{ 0x6869, 9, 0, 0, "Vega10" },
	{ 0x686A, 9, 0, 0, "Vega10" },
	{ 0x686B, 9, 0, 0, "Vega10" },
	{ 0x686C, 9, 0, 0, "Vega10" },
	{ 0x686D, 9, 0, 0, "Vega10" },
	{ 0x686E, 9, 0, 0, "Vega10" },
	{ 0x687F, 9, 0, 0, "Vega10" },
	/* Vega12 */
	{ 0x69A0, 9, 0, 4, "Vega12" },
	{ 0x69A1, 9, 0, 4, "Vega12" },
	{ 0x69A2, 9, 0, 4, "Vega12" },
	{ 0x69A3, 9, 0, 4, "Vega12" },
	{ 0x69Af, 9, 0, 4, "Vega12" },
	/* Raven */
	{ 0x15DD, 9, 0, 2, "Raven" },
	{ 0x15D8, 9, 0, 2, "Raven" },
	/* Vega20 */
	{ 0x66A0, 9, 0, 6, "Vega20" },
	{ 0x66A1, 9, 0, 6, "Vega20" },
	{ 0x66A2, 9, 0, 6, "Vega20" },
	{ 0x66A3, 9, 0, 6, "Vega20" },
	{ 0x66A4, 9, 0, 6, "Vega20" },
	{ 0x66A7, 9, 0, 6, "Vega20" },
	{ 0x66AF, 9, 0, 6, "Vega20" },
	/* Arcturus */
	{ 0x7388, 9, 0, 8, "Arcturus" },
	{ 0x738C, 9, 0, 8, "Arcturus" },
	{ 0x738E, 9, 0, 8, "Arcturus" },
	{ 0x7390, 9, 0, 8, "Arcturus" },
	/* Aldebaran */
	{ 0x7408, 9, 0, 10, "Aldebaran" },
	{ 0x740C, 9, 0, 10, "Aldebaran" },
	{ 0x740F, 9, 0, 10, "Aldebaran" },
	{ 0x7410, 9, 0, 10, "Aldebaran" },
	/* Renoir */
	{ 0x15E7, 9, 0, 12, "Renoir" },
	{ 0x1636, 9, 0, 12, "Renoir" },
	{ 0x1638, 9, 0, 12, "Renoir" },
	{ 0x164C, 9, 0, 12, "Renoir" },
	/* Navi10 */
	{ 0x7310, 10, 1, 0, "Navi10" },
	{ 0x7312, 10, 1, 0, "Navi10" },
	{ 0x7318, 10, 1, 0, "Navi10" },
	{ 0x731A, 10, 1, 0, "Navi10" },
	{ 0x731E, 10, 1, 0, "Navi10" },
	{ 0x731F, 10, 1, 0, "Navi10" },
	/* cyan_skillfish */
	{ 0x13F9, 10, 1, 3, "cyan_skillfish" },
	{ 0x13FA, 10, 1, 3, "cyan_skillfish" },
	{ 0x13FB, 10, 1, 3, "cyan_skillfish" },
	{ 0x13FC, 10, 1, 3, "cyan_skillfish" },
	{ 0x13FE, 10, 1, 3, "cyan_skillfish" },
	{ 0x143F, 10, 1, 3, "cyan_skillfish" },
	/* Navi14 */
	{ 0x7340, 10, 1, 2, "Navi14" },
	{ 0x7341, 10, 1, 2, "Navi14" },
	{ 0x7347, 10, 1, 2, "Navi14" },
	/* Navi12 */
	{ 0x7360, 10, 1, 1, "Navi12" },
	{ 0x7362, 10, 1, 1, "Navi12" },
	/* SIENNA_CICHLID */
	{ 0x73A0, 10, 3, 0, "SIENNA_CICHLID" },
	{ 0x73A1, 10, 3, 0, "SIENNA_CICHLID" },
	{ 0x73A2, 10, 3, 0, "SIENNA_CICHLID" },
	{ 0x73A3, 10, 3, 0, "SIENNA_CICHLID" },
	{ 0x73A5, 10, 3, 0, "SIENNA_CICHLID" },
	{ 0x73A8, 10, 3, 0, "SIENNA_CICHLID" },
	{ 0x73A9, 10, 3, 0, "SIENNA_CICHLID" },
	{ 0x73AC, 10, 3, 0, "SIENNA_CICHLID" },
	{ 0x73AD, 10, 3, 0, "SIENNA_CICHLID" },
	{ 0x73AB, 10, 3, 0, "SIENNA_CICHLID" },
	{ 0x73AE, 10, 3, 0, "SIENNA_CICHLID" },
	{ 0x73BF, 10, 3, 0, "SIENNA_CICHLID" },
	/* NAVY_FLOUNDER */
	{ 0x73C0, 10, 3, 1, "NAVY_FLOUNDER" },
	{ 0x73C1, 10, 3, 1, "NAVY_FLOUNDER" },
	{ 0x73C3, 10, 3, 1, "NAVY_FLOUNDER" },
	{ 0x73DA, 10, 3, 1, "NAVY_FLOUNDER" },
	{ 0x73DB, 10, 3, 1, "NAVY_FLOUNDER" },
	{ 0x73DC, 10, 3, 1, "NAVY_FLOUNDER" },
	{ 0x73DD, 10, 3, 1, "NAVY_FLOUNDER" },
	{ 0x73DE, 10, 3, 1, "NAVY_FLOUNDER" },
	{ 0x73DF, 10, 3, 1, "NAVY_FLOUNDER" },
	/* DIMGREY_CAVEFISH */
	{ 0x73E0, 10, 3, 2, "DIMGREY_CAVEFISH" },
	{ 0x73E1, 10, 3, 2, "DIMGREY_CAVEFISH" },
	{ 0x73E2, 10, 3, 2, "DIMGREY_CAVEFISH" },
	{ 0x73E8, 10, 3, 2, "DIMGREY_CAVEFISH" },
	{ 0x73E9, 10, 3, 2, "DIMGREY_CAVEFISH" },
	{ 0x73EA, 10, 3, 2, "DIMGREY_CAVEFISH" },
	{ 0x73EB, 10, 3, 2, "DIMGREY_CAVEFISH" },
	{ 0x73EC, 10, 3, 2, "DIMGREY_CAVEFISH" },
	{ 0x73ED, 10, 3, 2, "DIMGREY_CAVEFISH" },
	{ 0x73EF, 10, 3, 2, "DIMGREY_CAVEFISH" },
	{ 0x73FF, 10, 3, 2, "DIMGREY_CAVEFISH" },
	/* VanGogh */
	{ 0x163F, 10, 3, 3, "VanGogh" },
	/* BEIGE_GOBY */
	{ 0x7420, 10, 3, 4, "BEIGE_GOBY" },
	{ 0x7421, 10, 3, 4, "BEIGE_GOBY" },
	{ 0x7422, 10, 3, 4, "BEIGE_GOBY" },
	{ 0x7423, 10, 3, 4, "BEIGE_GOBY" },
	{ 0x743F, 10, 3, 4, "BEIGE_GOBY" },
	/* Yellow_Carp */
	{ 0x164D, 10, 3, 5, "YELLOW_CARP" },
	{ 0x1681, 10, 3, 5, "YELLOW_CARP" },
};

/* information from /proc/cpuinfo */
struct proc_cpuinfo {
	uint32_t proc_num; /* processor */
	uint32_t apicid; /* apicid */
	char model_name[HSA_PUBLIC_NAME_SIZE]; /* model name */
};

/* CPU cache table for all CPUs on the system. Each entry has the relative CPU
 * info and caches connected to that CPU.
 */
typedef struct cpu_cacheinfo {
	uint32_t len; /* length of the table = number of online procs */
	int32_t proc_num; /* this cpu's processor number */
	uint32_t num_caches; /* number of caches reported by this cpu */
	HsaCacheProperties *cache_prop; /* a list of cache properties */
} cpu_cacheinfo_t;

static void free_properties(node_props_t *props, int size)
{
	if (props) {
		int i;
		for (i = 0; i < size; i++) {
			free(props[i].mem);
			free(props[i].cache);
			free(props[i].link);
		}

		free(props);
	}
}

/* num_subdirs - find the number of sub-directories in the specified path
 *	@dirpath - directory path to find sub-directories underneath
 *	@prefix - only count sub-directory names starting with prefix.
 *		Use blank string, "", to count all.
 *	Return - number of sub-directories
 */
static int num_subdirs(char *dirpath, char *prefix)
{
	int count = 0;
	DIR *dirp;
	struct dirent *dir;
	int prefix_len = strlen(prefix);

	dirp = opendir(dirpath);
	if (dirp) {
		while ((dir = readdir(dirp)) != 0) {
			if ((strcmp(dir->d_name, ".") == 0) ||
				(strcmp(dir->d_name, "..") == 0))
				continue;
			if (prefix_len &&
				strncmp(dir->d_name, prefix, prefix_len))
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
static HSAKMT_STATUS fscanf_dec(char *file, uint32_t *num)
{
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
static HSAKMT_STATUS fscanf_str(const char *file, char *str, size_t str_size)
{
	FILE *fd;
	HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;

	fd = fopen(file, "r");
	if (!fd) {
		pr_err("Failed to open %s\n", file);
		return HSAKMT_STATUS_INVALID_PARAMETER;
	}

	if (!fgets(str, (int)str_size, fd)) {
		pr_err("Failed to read from %s.\n", file);
		ret = HSAKMT_STATUS_ERROR;
	} else {
		// Remove possible newline characters at the end, due to using fgets function
		str[strcspn(str, "\r\n")] = '\0';
	}

	fclose(fd);
	return ret;
}

/* fscanf_size - read a file whose content represents size as a string
 *      @file [IN ] file to read
 *      @bytes [OUT] sizes in bytes
 */
static HSAKMT_STATUS fscanf_size(char *file, uint32_t *bytes)
{
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
			*bytes <<= 10; break;
		case 'M':
			*bytes <<= 20; break;
		case 'G':
			*bytes <<= 30; break;
		default:
			ret = HSAKMT_STATUS_ERROR; break;
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
			     struct proc_cpuinfo *cpuinfo,
			     HsaCacheProperties *this_cache)
{
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
				pr_warn("SiblingMap buffer %d is too small\n",
					HSA_CPU_SIBLINGS);
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
static int get_cpu_cache_info(const char *prefix, struct proc_cpuinfo *cpuinfo,
			      cpu_cacheinfo_t *cpu_ci)
{
	int idx, num_idx, n;
	HsaCacheProperties *this_cache;
	char path[256], str[256];
	bool is_power9 = false;

	if (processor_vendor == IBM_POWER) {
		if (strcmp(cpuinfo[0].model_name, "POWER9") == 0) {
			is_power9 = true;
		}
	}

	this_cache = cpu_ci->cache_prop;
	num_idx = cpu_ci->num_caches;
	for (idx = 0; idx < num_idx; idx++) {
		/* If this cache is shared by multiple CPUs, we only need
		 * to list it in the first CPU.
		 */
		if (is_power9) {
			// POWER9 has SMT4
			if (cpu_ci->proc_num & 0x3) {
				/* proc is not 0,4,8,etc.  Skip and reduce the cache count. */
				--cpu_ci->num_caches;
				continue;
			}
		} else {
			snprintf(path, 256, "%s/index%d/shared_cpu_list", prefix, idx);
			/* shared_cpu_list is shown as n1,n2... or n1-n2,n3-n4...
			 * For both cases, this cache is listed to proc n1 only.
			 */
			fscanf_dec(path, (uint32_t *)&n);
			if (cpu_ci->proc_num != n) {
				/* proc is not n1. Skip and reduce the cache count. */
				--cpu_ci->num_caches;
				continue;
			}
			this_cache->ProcessorIdLow = cpuinfo[cpu_ci->proc_num].apicid;
		}

		/* CacheLevel */
		snprintf(path, 256, "%s/index%d/level", prefix, idx);
		fscanf_dec(path, &this_cache->CacheLevel);
		/* CacheType */
		snprintf(path, 256, "%s/index%d/type", prefix, idx);

		memset(str, 0, sizeof(str));
		fscanf_str(path, str, sizeof(str));
		if (!strcmp(str, "Data"))
			this_cache->CacheType.ui32.Data = 1;
		if (!strcmp(str, "Instruction"))
			this_cache->CacheType.ui32.Instruction = 1;
		if (!strcmp(str, "Unified")) {
			this_cache->CacheType.ui32.Data = 1;
			this_cache->CacheType.ui32.Instruction = 1;
		}
		this_cache->CacheType.ui32.CPU = 1;
		/* CacheSize */
		snprintf(path, 256, "%s/index%d/size", prefix, idx);
		fscanf_size(path, &this_cache->CacheSize);
		/* CacheLineSize */
		snprintf(path, 256, "%s/index%d/coherency_line_size", prefix, idx);
		fscanf_dec(path, &this_cache->CacheLineSize);
		/* CacheAssociativity */
		snprintf(path, 256, "%s/index%d/ways_of_associativity", prefix, idx);
		fscanf_dec(path, &this_cache->CacheAssociativity);
		/* CacheLinesPerTag */
		snprintf(path, 256, "%s/index%d/physical_line_partition", prefix, idx);
		fscanf_dec(path, &this_cache->CacheLinesPerTag);
		/* CacheSiblings */
		snprintf(path, 256, "%s/index%d/shared_cpu_map", prefix, idx);
		fscanf_str(path, str, sizeof(str));
		cpumap_to_cpu_ci(str, cpuinfo, this_cache);

		++this_cache;
	}

	return cpu_ci->num_caches;
}

static HSAKMT_STATUS topology_sysfs_get_generation(uint32_t *gen)
{
	FILE *fd;
	HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;

	char path[256];
	snprintf(path, sizeof(path), KFD_SYSFS_PATH_GENERATION_ID, get_topology_dir());

	assert(gen);
	fd = fopen(path, "r");
	if (!fd)
		return HSAKMT_STATUS_ERROR;
	if (fscanf(fd, "%ul", gen) != 1) {
		ret = HSAKMT_STATUS_ERROR;
		goto err;
	}

err:
	fclose(fd);
	return ret;
}

static HSAKMT_STATUS topology_sysfs_map_node_id(uint32_t node_id, uint32_t *sys_node_id)
{
	if ((!map_user_to_sysfs_node_id) || (node_id >= map_user_to_sysfs_node_id_size))
		return HSAKMT_STATUS_NOT_SUPPORTED;

	*sys_node_id = map_user_to_sysfs_node_id[node_id];
	return HSAKMT_STATUS_SUCCESS;
}

static HSAKMT_STATUS topology_sysfs_get_gpu_id(uint32_t sysfs_node_id, uint32_t *gpu_id)
{
	FILE *fd;
	char path[256];
	HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;

	assert(gpu_id);
	snprintf(path, sizeof(path), KFD_SYSFS_PATH_NODES "/%d/gpu_id", get_topology_dir(), sysfs_node_id);
	fd = fopen(path, "r");
	if (!fd)
		return HSAKMT_STATUS_ERROR;
	if (fscanf(fd, "%ul", gpu_id) != 1)
		ret = (errno == EPERM) ? HSAKMT_STATUS_NOT_SUPPORTED :
					 HSAKMT_STATUS_ERROR;
	fclose(fd);

	return ret;
}

/* Check if the @sysfs_node_id is supported. This function will be passed with sysfs node id.
 * This function can not use topology_* help functions, because those functions are
 * using user node id.
 * A sysfs node is not supported
 *	- if corresponding drm render node is not available.
 *	- if node information is not accessible (EPERM)
 */
static HSAKMT_STATUS topology_sysfs_check_node_supported(uint32_t sysfs_node_id, bool *is_node_supported)
{
	uint32_t gpu_id;
	FILE *fd;
	char *read_buf, *p;
	int read_size;
	char prop_name[256];
	char path[256];
	unsigned long long prop_val;
	uint32_t prog;
	uint32_t drm_render_minor = 0;
	int ret_value;
	HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;

	*is_node_supported = false;

	/* Retrieve the GPU ID */
	ret = topology_sysfs_get_gpu_id(sysfs_node_id, &gpu_id);
	if (ret == HSAKMT_STATUS_NOT_SUPPORTED)
		return HSAKMT_STATUS_SUCCESS;
	if (ret != HSAKMT_STATUS_SUCCESS)
		return ret;

	if (gpu_id == 0) {
		*is_node_supported = true;
		return HSAKMT_STATUS_SUCCESS;
	}

	read_buf = malloc(PAGE_SIZE);
	if (!read_buf)
		return HSAKMT_STATUS_NO_MEMORY;

	/* Retrieve the node properties */
	snprintf(path, 256, KFD_SYSFS_PATH_NODES "/%d/properties", get_topology_dir(), sysfs_node_id);
	fd = fopen(path, "r");
	if (!fd) {
		free(read_buf);
		return HSAKMT_STATUS_ERROR;
	}

	read_size = fread(read_buf, 1, PAGE_SIZE, fd);
	if (read_size <= 0) {
		ret = HSAKMT_STATUS_ERROR;
		goto err;
	}

	/* Since we're using the buffer as a string, we make sure the string terminates */
	if (read_size >= PAGE_SIZE)
		read_size = PAGE_SIZE - 1;
	read_buf[read_size] = 0;

	/* Read the node properties */
	prog = 0;
	p = read_buf;
	while (sscanf(p += prog, "%s %llu\n%n", prop_name, &prop_val, &prog) == 2) {
		if (strcmp(prop_name, "drm_render_minor") == 0) {
			drm_render_minor = (int32_t)prop_val;
			break;
		}
	}
	if (!drm_render_minor) {
		ret = HSAKMT_STATUS_ERROR;
		goto err;
	}

	/* Open DRM Render device */
	ret_value = hsakmt_open_drm_render_device(drm_render_minor);
	if (ret_value > 0)
		*is_node_supported = true;
	else if (ret_value != -ENOENT && ret_value != -EPERM)
		ret = HSAKMT_STATUS_ERROR;

err:
	free(read_buf);
	fclose(fd);
	return ret;
}

HSAKMT_STATUS hsakmt_topology_sysfs_get_system_props(HsaSystemProperties *props)
{
	FILE *fd;
	char *read_buf, *p;
	char path[256];
	char prop_name[256];
	unsigned long long prop_val;
	uint32_t prog;
	int read_size;
	HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;
	bool is_node_supported = true;
	uint32_t num_supported_nodes = 0;

	assert(props);
	snprintf(path, sizeof(path), KFD_SYSFS_PATH_SYSTEM_PROPERTIES, get_topology_dir());
	fd = fopen(path, "r");
	if (!fd)
		return HSAKMT_STATUS_ERROR;

	read_buf = malloc(PAGE_SIZE);
	if (!read_buf) {
		ret = HSAKMT_STATUS_NO_MEMORY;
		goto err1;
	}

	read_size = fread(read_buf, 1, PAGE_SIZE, fd);
	if (read_size <= 0) {
		ret = HSAKMT_STATUS_ERROR;
		goto err2;
	}

	/* Since we're using the buffer as a string, we make sure the string terminates */
	if (read_size >= PAGE_SIZE)
		read_size = PAGE_SIZE - 1;
	read_buf[read_size] = 0;

	/* Read the system properties */
	prog = 0;
	p = read_buf;
	while (sscanf(p += prog, "%s %llu\n%n", prop_name, &prop_val, &prog) == 2) {
		if (strcmp(prop_name, "platform_oem") == 0)
			props->PlatformOem = (uint32_t)prop_val;
		else if (strcmp(prop_name, "platform_id") == 0)
			props->PlatformId = (uint32_t)prop_val;
		else if (strcmp(prop_name, "platform_rev") == 0)
			props->PlatformRev = (uint32_t)prop_val;
	}

	/*
	 * Discover the number of sysfs nodes:
	 * Assuming that inside nodes folder there are only folders
	 * which represent the node numbers
	 */
	snprintf(path, sizeof(path), KFD_SYSFS_PATH_NODES, get_topology_dir());
	num_sysfs_nodes = num_subdirs(path, "");

	if (map_user_to_sysfs_node_id == NULL) {
		/* Trade off - num_sysfs_nodes includes all CPU and GPU nodes.
		 * Slightly more memory is allocated than necessary.
		 */
		map_user_to_sysfs_node_id = calloc(num_sysfs_nodes, sizeof(uint32_t));
		if (map_user_to_sysfs_node_id == NULL) {
			ret = HSAKMT_STATUS_NO_MEMORY;
			goto err2;
		}
		map_user_to_sysfs_node_id_size = num_sysfs_nodes;
	} else if (num_sysfs_nodes > map_user_to_sysfs_node_id_size) {
		free(map_user_to_sysfs_node_id);
		map_user_to_sysfs_node_id = calloc(num_sysfs_nodes, sizeof(uint32_t));
		if (map_user_to_sysfs_node_id == NULL) {
			ret = HSAKMT_STATUS_NO_MEMORY;
			goto err2;
		}
		map_user_to_sysfs_node_id_size = num_sysfs_nodes;
	}

	for (uint32_t i = 0; i < num_sysfs_nodes; i++) {
		ret = topology_sysfs_check_node_supported(i, &is_node_supported);
		if (ret != HSAKMT_STATUS_SUCCESS)
			goto sysfs_parse_failed;
		if (is_node_supported)
			map_user_to_sysfs_node_id[num_supported_nodes++] = i;
	}
	props->NumNodes = num_supported_nodes;

	free(read_buf);
	fclose(fd);
	return ret;

sysfs_parse_failed:
	free(map_user_to_sysfs_node_id);
	map_user_to_sysfs_node_id = NULL;
err2:
	free(read_buf);
err1:
	fclose(fd);
	return ret;
}

static const struct hsa_gfxip_table *find_hsa_gfxip_device(uint16_t device_id, uint8_t gfxv_major)
{
	if (gfxv_major > 10)
		return NULL;

	uint32_t i, table_size;

	table_size = sizeof(gfxip_lookup_table)/sizeof(struct hsa_gfxip_table);
	for (i = 0; i < table_size; i++) {
		if (gfxip_lookup_table[i].device_id == device_id)
			return &gfxip_lookup_table[i];
	}
	return NULL;
}

void hsakmt_topology_setup_is_dgpu_param(HsaNodeProperties *props)
{
	/* if we found a dGPU node, then treat the whole system as dGPU */
	if (!props->NumCPUCores && props->NumFComputeCores)
		hsakmt_is_dgpu = true;
}

bool hsakmt_topology_is_svm_needed(HSA_ENGINE_ID EngineId)
{
	if (hsakmt_is_dgpu)
		return true;

	if (HSA_GET_GFX_VERSION_FULL(EngineId.ui32) >= GFX_VERSION_VEGA10)
		return true;

	return false;
}

static HSAKMT_STATUS topology_get_cpu_model_name(HsaNodeProperties *props,
				struct proc_cpuinfo *cpuinfo, int num_procs)
{
	int i, j;

	if (!props) {
		pr_err("Invalid props to get cpu model name\n");
		return HSAKMT_STATUS_INVALID_PARAMETER;
	}

	for (i = 0; i < num_procs; i++, cpuinfo++) {
		if (props->CComputeIdLo == cpuinfo->apicid) {
			if (!props->DeviceId) /* CPU-only node */
				strncpy((char *)props->AMDName, cpuinfo->model_name, sizeof(props->AMDName));
			/* Convert from UTF8 to UTF16 */
			for (j = 0; cpuinfo->model_name[j] != '\0' && j < HSA_PUBLIC_NAME_SIZE - 1; j++)
				props->MarketingName[j] = cpuinfo->model_name[j];
			props->MarketingName[j] = '\0';
			return HSAKMT_STATUS_SUCCESS;
		}
	}

	return HSAKMT_STATUS_ERROR;
}

static int topology_search_processor_vendor(const char *processor_name)
{
	unsigned int i;

	for (i = 0; i < ARRAY_LEN(supported_processor_vendor_name); i++) {
		if (!strcmp(processor_name, supported_processor_vendor_name[i]))
			return i;
		if (!strcmp(processor_name, "POWER9, altivec supported\n"))
			return IBM_POWER;
	}
	return -1;
}

/* topology_parse_cpuinfo - Parse /proc/cpuinfo and fill up required
 *			topology information
 * cpuinfo [OUT]: output buffer to hold cpu information
 * num_procs: number of processors the output buffer can hold
 */
static HSAKMT_STATUS topology_parse_cpuinfo(struct proc_cpuinfo *cpuinfo,
					    uint32_t num_procs)
{
	HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;
	FILE *fd;
	char read_buf[256];
	char *p;
	uint32_t proc = 0;
	size_t p_len;
	const char *proc_cpuinfo_path = "/proc/cpuinfo";

	if (!cpuinfo) {
		pr_err("CPU information will be missing\n");
		return HSAKMT_STATUS_INVALID_PARAMETER;
	}

	fd = fopen(proc_cpuinfo_path, "r");
	if (!fd) {
		pr_err("Failed to open [%s]. Unable to get CPU information",
			proc_cpuinfo_path);
		return HSAKMT_STATUS_ERROR;
	}

#ifdef __PPC64__
	char *p2;

	/* Each line in /proc/cpuinfo that read_buf is constructed, the format
	 * is like this:
	 * "token       : value\n"
	 * where token is our target like vendor_id, model name, apicid ...
	 * and value is the answer
	 */
	while (fgets(read_buf, sizeof(read_buf), fd)) {
		/* processor number */
		if (!strncmp("processor	", read_buf, sizeof("processor	") - 1)) {
			p = strchr(read_buf, ':');
			p += 2; /* remove ": " */
			proc = atoi(p);
			if (proc >= num_procs) {
				pr_warn("cpuinfo contains processor %d larger than %u\n",
					proc, num_procs);
				ret = HSAKMT_STATUS_NO_MEMORY;
				goto exit;
			}
			continue;
		}

		/* vendor name / model name */
		if (!strncmp("cpu	", read_buf, sizeof("cpu	") - 1) &&
			(processor_vendor == -1)) {
			p = strchr(read_buf, ':');
			p += 2; /* remove ": " */
			processor_vendor = topology_search_processor_vendor(p);

			p2 = strchr(p, ',');
			if (p2 != NULL) {
				p2++;
				*p2 = 0;
			}
			if (strlen(p) < HSA_PUBLIC_NAME_SIZE) {
				/* -1 to remove \n from p */
				strncpy(cpuinfo[proc].model_name, p, strlen(p) - 1);
				cpuinfo[proc].model_name[strlen(p) - 1] = '\0';
			} else
				strncpy(cpuinfo[proc].model_name, p, HSA_PUBLIC_NAME_SIZE);
			continue;
		}
	}
#else
	/* Each line in /proc/cpuinfo that read_buf is constructed, the format
	 * is like this:
	 * "token       : value\n"
	 * where token is our target like vendor_id, model name, apicid ...
	 * and value is the answer
	 */
	while (fgets(read_buf, sizeof(read_buf), fd)) {
		/* processor number */
		if (!strncmp("processor", read_buf, sizeof("processor") - 1)) {
			p = strchr(read_buf, ':');
			p += 2; /* remove ": " */
			proc = atoi(p);
			if (proc >= num_procs) {
				pr_warn("cpuinfo contains processor %d larger than %u\n",
					proc, num_procs);
				ret = HSAKMT_STATUS_NO_MEMORY;
				goto exit;
			}
			continue;
		}

		/* vendor name */
		if (!strncmp("vendor_id", read_buf, sizeof("vendor_id") - 1) &&
			(processor_vendor == -1)) {
			p = strchr(read_buf, ':');
			p += 2; /* remove ": " */
			processor_vendor = topology_search_processor_vendor(p);
			continue;
		}

		/* model name */
		if (!strncmp("model name", read_buf, sizeof("model name") - 1)) {
			p = strchr(read_buf, ':');
			p += 2; /* remove ": " */
			p_len = strlen(p);
			if (p_len > HSA_PUBLIC_NAME_SIZE)
				p_len = HSA_PUBLIC_NAME_SIZE;
			memcpy(cpuinfo[proc].model_name, p, p_len);
			cpuinfo[proc].model_name[p_len - 1] = '\0';
			continue;
		}

		/* apicid */
		if (!strncmp("apicid", read_buf, sizeof("apicid") - 1)) {
			p = strchr(read_buf, ':');
			p += 2; /* remove ": " */
			cpuinfo[proc].apicid = atoi(p);
		}
	}
#endif

	if (processor_vendor < 0) {
		pr_err("Failed to get Processor Vendor. Setting to %s",
			supported_processor_vendor_name[GENUINE_INTEL]);
		processor_vendor = GENUINE_INTEL;
	}

exit:
	fclose(fd);
	return ret;
}

static int topology_get_node_props_from_drm(HsaNodeProperties *props)
{
	int drm_fd;
	uint32_t major_version;
	uint32_t minor_version;
	amdgpu_device_handle device_handle;
	struct amdgpu_gpu_info gpu_info;
	const char *name;
	int i, ret = 0;

	if (props == NULL)
		return -1;

	drm_fd = drmOpenRender(props->DrmRenderMinor);
	if (drm_fd < 0)
		return -1;

	if (amdgpu_device_initialize(drm_fd,
		&major_version, &minor_version, &device_handle) < 0) {
		ret = -1;
		goto err_device_initialize;
	}

	name = amdgpu_get_marketing_name(device_handle);
	if (name != NULL) {
		for (i = 0; name[i] != 0 && i < HSA_PUBLIC_NAME_SIZE - 1; i++)
			props->MarketingName[i] = name[i];
		props->MarketingName[i] = '\0';
	}

	if (amdgpu_query_gpu_info(device_handle, &gpu_info)) {
		ret = -1;
		goto err_query_gpu_info;
	}

	props->FamilyID = gpu_info.family_id;
	props->Integrated = !!(gpu_info.ids_flags & AMDGPU_IDS_FLAGS_FUSION);

err_query_gpu_info:
	amdgpu_device_deinitialize(device_handle);
err_device_initialize:
	drmClose(drm_fd);
	return ret;
}

static HSAKMT_STATUS topology_sysfs_get_node_props(uint32_t node_id,
						   HsaNodeProperties *props,
						   bool *p2p_links,
						   uint32_t *num_p2pLinks)
{
	FILE *fd;
	char *read_buf, *p, *envvar, dummy = '\0';
	char prop_name[256];
	char path[256];
	char per_node_override[32];
	unsigned long long prop_val = 0;
	uint32_t prog, major = 0, minor = 0, step = 0;
	int read_size;
	const struct hsa_gfxip_table *hsa_gfxip;
	uint32_t sys_node_id;
	uint32_t gfxv = 0;
	uint8_t gfxv_major, gfxv_minor, gfxv_stepping;
	uint32_t simd_arrays_count = 0;

	HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;

	assert(props);
	ret = topology_sysfs_map_node_id(node_id, &sys_node_id);
	if (ret != HSAKMT_STATUS_SUCCESS)
		return ret;

	/* Retrieve the GPU ID */
	ret = topology_sysfs_get_gpu_id(sys_node_id, &props->KFDGpuID);
	if (ret != HSAKMT_STATUS_SUCCESS)
		return ret;

	read_buf = malloc(PAGE_SIZE);
	if (!read_buf)
		return HSAKMT_STATUS_NO_MEMORY;

	/* Retrieve the node properties */
	snprintf(path, 256, KFD_SYSFS_PATH_NODES "/%d/properties", get_topology_dir(), sys_node_id);
	fd = fopen(path, "r");
	if (!fd) {
		free(read_buf);
		return HSAKMT_STATUS_ERROR;
	}

	read_size = fread(read_buf, 1, PAGE_SIZE, fd);
	if (read_size <= 0) {
		ret = HSAKMT_STATUS_ERROR;
		goto out;
	}

	/* Since we're using the buffer as a string, we make sure the string terminates */
	if (read_size >= PAGE_SIZE)
		read_size = PAGE_SIZE - 1;
	read_buf[read_size] = 0;

	/* Read the node properties */
	prog = 0;
	p = read_buf;
	while (sscanf(p += prog, "%s %llu\n%n", prop_name, &prop_val, &prog) == 2) {
		if (strcmp(prop_name, "cpu_cores_count") == 0)
			props->NumCPUCores = (uint32_t)prop_val;
		else if (strcmp(prop_name, "simd_count") == 0)
			props->NumFComputeCores = (uint32_t)prop_val;
		else if (strcmp(prop_name, "mem_banks_count") == 0)
			props->NumMemoryBanks = (uint32_t)prop_val;
		else if (strcmp(prop_name, "caches_count") == 0)
			props->NumCaches = (uint32_t)prop_val;
		else if (strcmp(prop_name, "io_links_count") == 0)
			props->NumIOLinks = (uint32_t)prop_val;
		else if (strcmp(prop_name, "p2p_links_count") == 0) {
			props->NumIOLinks += (uint32_t)prop_val;
			if (num_p2pLinks)
				*num_p2pLinks = (uint32_t)prop_val;
			if (p2p_links)
				*p2p_links = true;
		} else if (strcmp(prop_name, "cpu_core_id_base") == 0)
			props->CComputeIdLo = (uint32_t)prop_val;
		else if (strcmp(prop_name, "simd_id_base") == 0)
			props->FComputeIdLo = (uint32_t)prop_val;
		else if (strcmp(prop_name, "capability") == 0)
			props->Capability.Value = (uint32_t)prop_val;
		else if (strcmp(prop_name, "capability2") == 0)
			props->Capability2.Value = (uint32_t)prop_val;
		else if (strcmp(prop_name, "debug_prop") == 0)
			props->DebugProperties.Value = (uint64_t)prop_val;
		else if (strcmp(prop_name, "max_waves_per_simd") == 0)
			props->MaxWavesPerSIMD = (uint32_t)prop_val;
		else if (strcmp(prop_name, "lds_size_in_kb") == 0)
			props->LDSSizeInKB = (uint32_t)prop_val;
		else if (strcmp(prop_name, "gds_size_in_kb") == 0)
			props->GDSSizeInKB = (uint32_t)prop_val;
		else if (strcmp(prop_name, "wave_front_size") == 0)
			props->WaveFrontSize = (uint32_t)prop_val;
		else if (strcmp(prop_name, "array_count") == 0)
			simd_arrays_count = (uint32_t)prop_val;
		else if (strcmp(prop_name, "simd_arrays_per_engine") == 0)
			props->NumArrays = (uint32_t)prop_val;
		else if (strcmp(prop_name, "cu_per_simd_array") == 0)
			props->NumCUPerArray = (uint32_t)prop_val;
		else if (strcmp(prop_name, "simd_per_cu") == 0)
			props->NumSIMDPerCU = (uint32_t)prop_val;
		else if (strcmp(prop_name, "max_slots_scratch_cu") == 0)
			props->MaxSlotsScratchCU = (uint32_t)prop_val;
		else if (strcmp(prop_name, "fw_version") == 0)
			props->EngineId.Value = (uint32_t)prop_val & 0x3ff;
		else if (strcmp(prop_name, "vendor_id") == 0)
			props->VendorId = (uint32_t)prop_val;
		else if (strcmp(prop_name, "device_id") == 0)
			props->DeviceId = (uint32_t)prop_val;
		else if (strcmp(prop_name, "location_id") == 0)
			props->LocationId = (uint32_t)prop_val;
		else if (strcmp(prop_name, "domain") == 0)
			props->Domain = (uint32_t)prop_val;
		else if (strcmp(prop_name, "max_engine_clk_fcompute") == 0)
			props->MaxEngineClockMhzFCompute = (uint32_t)prop_val;
		else if (strcmp(prop_name, "max_engine_clk_ccompute") == 0)
			props->MaxEngineClockMhzCCompute = (uint32_t)prop_val;
		else if (strcmp(prop_name, "local_mem_size") == 0)
			props->LocalMemSize = prop_val;
		else if (strcmp(prop_name, "drm_render_minor") == 0)
			props->DrmRenderMinor = (int32_t)prop_val;
		else if (strcmp(prop_name, "sdma_fw_version") == 0)
			props->uCodeEngineVersions.Value = (uint32_t)prop_val & 0x3ff;
		else if (strcmp(prop_name, "hive_id") == 0)
			props->HiveID = prop_val;
		else if (strcmp(prop_name, "unique_id") == 0)
			props->UniqueID = prop_val;
		else if (strcmp(prop_name, "num_sdma_engines") == 0)
			props->NumSdmaEngines = prop_val;
		else if (strcmp(prop_name, "num_sdma_xgmi_engines") == 0)
			props->NumSdmaXgmiEngines = prop_val;
		else if (strcmp(prop_name, "num_gws") == 0)
			props->NumGws = prop_val;
		else if (strcmp(prop_name, "num_sdma_queues_per_engine") == 0)
			props->NumSdmaQueuesPerEngine = prop_val;
		else if (strcmp(prop_name, "num_cp_queues") == 0)
			props->NumCpQueues = prop_val;
		else if (strcmp(prop_name, "num_xcc") == 0)
			props->NumXcc = prop_val;
		else if (strcmp(prop_name, "family_id") == 0)
			props->FamilyID = prop_val;
		else if (strcmp(prop_name, "gfx_target_version") == 0)
			gfxv = (uint32_t)prop_val;
	}

	if (!hsakmt_is_svm_api_supported)
		props->Capability.ui32.SVMAPISupported = 0;

	/* Bail out early, if a CPU node */
	if (!props->NumFComputeCores)
		goto out;

	if (props->NumArrays != 0)
		props->NumShaderBanks = simd_arrays_count/props->NumArrays;

	gfxv_major = HSA_GET_GFX_VERSION_MAJOR(gfxv);
	gfxv_minor = HSA_GET_GFX_VERSION_MINOR(gfxv);
	gfxv_stepping = HSA_GET_GFX_VERSION_STEP(gfxv);

	hsa_gfxip = find_hsa_gfxip_device(props->DeviceId, gfxv_major);
	if (hsa_gfxip || gfxv) {
		snprintf(per_node_override, sizeof(per_node_override), "HSA_OVERRIDE_GFX_VERSION_%d", node_id);
		if ((envvar = getenv(per_node_override)) || (envvar = getenv("HSA_OVERRIDE_GFX_VERSION"))) {
			/* HSA_OVERRIDE_GFX_VERSION=major.minor.stepping */
			if ((sscanf(envvar, "%u.%u.%u%c",
					&major, &minor, &step, &dummy) != 3) ||
				(major > 63 || minor > 255 || step > 255)) {
				pr_err("HSA_OVERRIDE_GFX_VERSION %s is invalid\n",
					envvar);
				ret = HSAKMT_STATUS_ERROR;
				goto out;
			}
			props->OverrideEngineId.ui32.Major = major & 0x3f;
			props->OverrideEngineId.ui32.Minor = minor & 0xff;
			props->OverrideEngineId.ui32.Stepping = step & 0xff;
		}

		if (hsa_gfxip) {
			props->EngineId.ui32.Major = hsa_gfxip->major & 0x3f;
			props->EngineId.ui32.Minor = hsa_gfxip->minor & 0xff;
			props->EngineId.ui32.Stepping = hsa_gfxip->stepping & 0xff;
		} else {
			props->EngineId.ui32.Major = gfxv_major & 0x3f;
			props->EngineId.ui32.Minor = gfxv_minor & 0xff;
			props->EngineId.ui32.Stepping = gfxv_stepping & 0xff;
		}

		/* Set the CAL name of the node. If DID-based hsa_gfxip lookup was
		 * successful, use that name. Otherwise, set to GFX<GFX_VERSION>.
		 */
		if (hsa_gfxip && hsa_gfxip->amd_name)
			strncpy((char *)props->AMDName, hsa_gfxip->amd_name,
					sizeof(props->AMDName)-1);
		else
			snprintf((char *)props->AMDName, sizeof(props->AMDName)-1, "GFX%06x",
					HSA_GET_GFX_VERSION_FULL(props->EngineId.ui32));

		/* Is dGPU Node, not APU
		 * Retrieve the marketing name of the node.
		 */
		if (topology_get_node_props_from_drm(props))
			pr_info("failed to get marketing name for device ID 0x%x\n", props->DeviceId);

		/* Get VGPR/SGPR size in byte per CU */
		props->SGPRSizePerCU = SGPR_SIZE_PER_CU;
		props->VGPRSizePerCU = hsakmt_get_vgpr_size_per_cu(HSA_GET_GFX_VERSION_FULL(props->EngineId.ui32));

	} else if (props->DeviceId)
		/* still return success */
		pr_err("device ID 0x%x is not supported in libhsakmt\n",
				props->DeviceId);

	if (props->NumFComputeCores)
		assert(props->EngineId.ui32.Major && "HSA_OVERRIDE_GFX_VERSION may be needed");

	/* On Older kernels, num_xcc may not be present in system properties.
	 * Set it to 1 if system properties do not report num_xcc.
	 */
	if (!props->NumXcc)
		props->NumXcc = 1;

out:
	free(read_buf);
	fclose(fd);
	return ret;
}

static HSAKMT_STATUS topology_sysfs_get_mem_props(uint32_t node_id,
						  uint32_t mem_id,
						  HsaMemoryProperties *props)
{
	FILE *fd;
	char *read_buf, *p;
	char prop_name[256];
	char path[256];
	unsigned long long prop_val;
	uint32_t prog;
	int read_size;
	HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;
	uint32_t sys_node_id;

	assert(props);
	ret = topology_sysfs_map_node_id(node_id, &sys_node_id);
	if (ret != HSAKMT_STATUS_SUCCESS)
		return ret;

	snprintf(path, 256, KFD_SYSFS_PATH_NODES "/%d/mem_banks/%d/properties", get_topology_dir(), sys_node_id, mem_id);
	fd = fopen(path, "r");
	if (!fd)
		return HSAKMT_STATUS_ERROR;
	read_buf = malloc(PAGE_SIZE);
	if (!read_buf) {
		ret = HSAKMT_STATUS_NO_MEMORY;
		goto err1;
	}

	read_size = fread(read_buf, 1, PAGE_SIZE, fd);
	if (read_size <= 0) {
		ret = HSAKMT_STATUS_ERROR;
		goto err2;
	}

	/* Since we're using the buffer as a string, we make sure the string terminates */
	if (read_size >= PAGE_SIZE)
		read_size = PAGE_SIZE - 1;
	read_buf[read_size] = 0;

	prog = 0;
	p = read_buf;
	while (sscanf(p += prog, "%s %llu\n%n", prop_name, &prop_val, &prog) == 2) {
		if (strcmp(prop_name, "heap_type") == 0)
			props->HeapType = (uint32_t)prop_val;
		else if (strcmp(prop_name, "size_in_bytes") == 0)
			props->SizeInBytes = (uint64_t)prop_val;
		else if (strcmp(prop_name, "flags") == 0)
			props->Flags.MemoryProperty = (uint32_t)prop_val;
		else if (strcmp(prop_name, "width") == 0)
			props->Width = (uint32_t)prop_val;
		else if (strcmp(prop_name, "mem_clk_max") == 0)
			props->MemoryClockMax = (uint32_t)prop_val;
	}

err2:
	free(read_buf);
err1:
	fclose(fd);
	return ret;
}

/* topology_destroy_temp_cpu_cache_list -
 *	Free the memory allocated in topology_create_temp_cpu_cache_list().
 */
static void topology_destroy_temp_cpu_cache_list(
					cpu_cacheinfo_t *temp_cpu_ci_list)
{
	uint32_t n;
	cpu_cacheinfo_t *p_temp_cpu_ci_list = temp_cpu_ci_list;
	cpu_cacheinfo_t *cpu_ci = p_temp_cpu_ci_list;

	if (p_temp_cpu_ci_list) {
		for (n = 0; n < p_temp_cpu_ci_list->len; n++, cpu_ci++)
			free(cpu_ci->cache_prop);
		free(p_temp_cpu_ci_list);
	}

}

/* topology_create_temp_cpu_cache_list - Create a temporary cpu-cache list to
 *		store cpu cache information. This list will be used to copy
 *		HsaCacheProperties in the CPU node. Two buffers are allocated
 *		inside this function: cpu_ci list and cache_prop under each
 *		cpu_ci. Must call topology_destroy_temp_cpu_cache_list to free
 *		the memory after the information is copied.
 *	@node [IN] CPU node number
 *	@cpuinfo [IN] /proc/cpuinfo data
 *	@temp_cpu_ci_list [OUT] cpu-cache-info list with data filled
 * Return: total number of caches under this CPU node
 */
static int topology_create_temp_cpu_cache_list(int node,
	struct proc_cpuinfo *cpuinfo, cpu_cacheinfo_t **temp_cpu_ci_list)
{
	/* Get max path size from /sys/devices/system/node/node%d/%s/cache
	 * below, which will max out according to the largest filename,
	 * which can be present twice in the string above. 29 is for the prefix
	 * and the +6 is for the cache suffix
	 */
#ifndef MAXNAMLEN
/* MAXNAMLEN is the BSD name for NAME_MAX. glibc aliases this as NAME_MAX, but not musl */
#define MAXNAMLEN NAME_MAX
#endif
	const uint32_t MAXPATHSIZE = 29 + MAXNAMLEN + (MAXNAMLEN + 6);
	cpu_cacheinfo_t *p_temp_cpu_ci_list; /* a list of cpu_ci */
	char path[MAXPATHSIZE], node_dir[MAXPATHSIZE];
	int max_cpus;
	cpu_cacheinfo_t *this_cpu; /* one cpu_ci in cpu_ci_list */
	int cache_cnt = 0;
	DIR *dirp = NULL;
	struct dirent *dir;
	char *p;

	if (!temp_cpu_ci_list) {
		pr_err("Invalid temp_cpu_ci_list\n");
		goto exit;
	}
	*temp_cpu_ci_list = NULL;

	/* Get info from /sys/devices/system/node/nodeX/cpuY/cache */
	int node_real = node;
	if (processor_vendor == IBM_POWER) {
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
			goto exit;
		}
		/* Fall back to use /sys/devices/system/cpu */
		snprintf(node_dir, MAXPATHSIZE, "/sys/devices/system/cpu");
		max_cpus = num_subdirs(node_dir, "cpu");
		if (max_cpus <= 0) {
			pr_err("Fail to get cpu* dirs under %s\n", node_dir);
			goto exit;
		}
	}

	p_temp_cpu_ci_list = calloc(max_cpus, sizeof(cpu_cacheinfo_t));
	if (!p_temp_cpu_ci_list) {
		pr_err("Fail to allocate p_temp_cpu_ci_list\n");
		goto exit;
	}
	p_temp_cpu_ci_list->len = 0;

	this_cpu = p_temp_cpu_ci_list;
	dirp = opendir(node_dir);
	while ((dir = readdir(dirp)) != 0) {
		if (strncmp(dir->d_name, "cpu", 3))
			continue;
		if (!isdigit(dir->d_name[3])) /* ignore files like cpulist */
			continue;
		snprintf(path, MAXPATHSIZE, "%s/%s/cache", node_dir, dir->d_name);
		this_cpu->num_caches = num_subdirs(path, "index");
		this_cpu->cache_prop = calloc(this_cpu->num_caches,
					sizeof(HsaCacheProperties));
		if (!this_cpu->cache_prop) {
			pr_err("Fail to allocate cache_info\n");
			goto exit;
		}
		p = &dir->d_name[3];
		this_cpu->proc_num = atoi(p);
		cache_cnt += get_cpu_cache_info(path, cpuinfo, this_cpu);
		++p_temp_cpu_ci_list->len;
		++this_cpu;
	}
	*temp_cpu_ci_list = p_temp_cpu_ci_list;

exit:
	if (dirp)
		closedir(dirp);
	return cache_cnt;
}

/* topology_get_cpu_cache_props - Read CPU cache information from sysfs
 *	@node [IN] CPU node number
 *	@cpuinfo [IN] /proc/cpuinfo data
 *	@tbl [OUT] the node table to fill up
 * Return: HSAKMT_STATUS_SUCCESS in success or error number in failure
 */
static HSAKMT_STATUS topology_get_cpu_cache_props(int node,
			struct proc_cpuinfo *cpuinfo, node_props_t *tbl)
{
	HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;
	cpu_cacheinfo_t *cpu_ci_list = NULL;
	uint32_t n, cache_cnt, i;
	cpu_cacheinfo_t *cpu_ci;
	HsaCacheProperties *this_cache;

	tbl->node.NumCaches = topology_create_temp_cpu_cache_list(
					node, cpuinfo, &cpu_ci_list);
	if (!tbl->node.NumCaches) {
		/* For "Intel Meteor lake Mobile", the cache info is not in sysfs,
		 * That means /sys/devices/system/node/node%d/%s/cache is not exist.
		 * here AMD will not black this issue.
		 */
		pr_debug("CPU cache info is not available for node %d \n", node);
		goto exit;
	}

	tbl->cache = calloc(tbl->node.NumCaches, sizeof(HsaCacheProperties));
	if (!tbl->cache) {
		ret = HSAKMT_STATUS_NO_MEMORY;
		goto exit;
	}

	/* Now fill in the information to cache properties. */
	cache_cnt = 0;
	cpu_ci = cpu_ci_list;
	for (n = 0; n < cpu_ci_list->len; n++, cpu_ci++) {
		this_cache = cpu_ci->cache_prop;
		for (i = 0; i < cpu_ci->num_caches; i++, this_cache++) {
			memcpy(&tbl->cache[cache_cnt++],
			       this_cache,
			       sizeof(HsaCacheProperties));
			if (cache_cnt >= tbl->node.NumCaches)
				goto exit;
		}
	}

exit:
	topology_destroy_temp_cpu_cache_list(cpu_ci_list);

	return ret;
}

static HSAKMT_STATUS topology_sysfs_get_cache_props(uint32_t node_id,
						    uint32_t cache_id,
						    HsaCacheProperties *props)
{
	FILE *fd;
	char *read_buf, *p;
	char prop_name[256];
	char path[256];
	unsigned long long prop_val;
	uint32_t i, prog;
	int read_size;
	HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;
	uint32_t sys_node_id;

	assert(props);
	ret = topology_sysfs_map_node_id(node_id, &sys_node_id);
	if (ret != HSAKMT_STATUS_SUCCESS)
		return ret;

	snprintf(path, 256, KFD_SYSFS_PATH_NODES "/%d/caches/%d/properties", get_topology_dir(), sys_node_id, cache_id);
	fd = fopen(path, "r");
	if (!fd)
		return HSAKMT_STATUS_ERROR;
	read_buf = malloc(PAGE_SIZE);
	if (!read_buf) {
		ret = HSAKMT_STATUS_NO_MEMORY;
		goto err1;
	}

	read_size = fread(read_buf, 1, PAGE_SIZE, fd);
	if (read_size <= 0) {
		ret = HSAKMT_STATUS_ERROR;
		goto err2;
	}

	/* Since we're using the buffer as a string, we make sure the string terminates */
	if (read_size >= PAGE_SIZE)
		read_size = PAGE_SIZE - 1;
	read_buf[read_size] = 0;

	prog = 0;
	p = read_buf;
	while (sscanf(p += prog, "%s %llu\n%n", prop_name, &prop_val, &prog) == 2) {
		if (strcmp(prop_name, "processor_id_low") == 0)
			props->ProcessorIdLow = (uint32_t)prop_val;
		else if (strcmp(prop_name, "level") == 0)
			props->CacheLevel = (uint32_t)prop_val;
		else if (strcmp(prop_name, "size") == 0)
			props->CacheSize = (uint32_t)prop_val;
		else if (strcmp(prop_name, "cache_line_size") == 0)
			props->CacheLineSize = (uint32_t)prop_val;
		else if (strcmp(prop_name, "cache_lines_per_tag") == 0)
			props->CacheLinesPerTag = (uint32_t)prop_val;
		else if (strcmp(prop_name, "association") == 0)
			props->CacheAssociativity = (uint32_t)prop_val;
		else if (strcmp(prop_name, "latency") == 0)
			props->CacheLatency = (uint32_t)prop_val;
		else if (strcmp(prop_name, "type") == 0)
			props->CacheType.Value = (uint32_t)prop_val;
		else if (strcmp(prop_name, "sibling_map") == 0)
			break;
	}

	prog = 0;
	if ((sscanf(p, "sibling_map %n", &prog)) == 0 && prog) {
		i = 0;
		while ((i < HSA_CPU_SIBLINGS) &&
			(sscanf(p += prog, "%u%*[,\n]%n", &props->SiblingMap[i++], &prog) == 1))
			continue;
	}

err2:
	free(read_buf);
err1:
	fclose(fd);
	return ret;
}

static HSAKMT_STATUS topology_map_sysfs_to_user_node_id(uint32_t sys_node_id, uint32_t *user_node_id)
{
	uint32_t node_id;

	for (node_id = 0; node_id < map_user_to_sysfs_node_id_size; node_id++)
		if (map_user_to_sysfs_node_id[node_id] == sys_node_id) {
			*user_node_id = node_id;
			return HSAKMT_STATUS_SUCCESS;
		}
	return HSAKMT_STATUS_INVALID_NODE_UNIT;
}


/* For a give Node @node_id the function gets @iolink_id information i.e. parses sysfs the following sysfs entry
 * ./nodes/@node_id/io_links/@iolink_id/properties. @node_id has to be valid accessible node.
 *
 * If node_to specified by the @iolink_id is not accessible the function returns HSAKMT_STATUS_NOT_SUPPORTED.
 * If node_to is accessible, then node_to is mapped from sysfs_node to user_node and returns HSAKMT_STATUS_SUCCESS.
 */
static HSAKMT_STATUS topology_sysfs_get_iolink_props(uint32_t node_id,
						     uint32_t iolink_id,
						     HsaIoLinkProperties *props, bool p2pLink)
{
	FILE *fd;
	char *read_buf, *p;
	char prop_name[256];
	char path[256];
	unsigned long long prop_val;
	uint32_t prog;
	int read_size;
	HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;
	uint32_t sys_node_id;

	assert(props);
	ret = topology_sysfs_map_node_id(node_id, &sys_node_id);
	if (ret != HSAKMT_STATUS_SUCCESS)
		return ret;

	snprintf(path, 256, KFD_SYSFS_PATH_NODES "/%d/%s/%d/properties", get_topology_dir(), sys_node_id, p2pLink ? "p2p_links" : "io_links", iolink_id);

	fd = fopen(path, "r");
	if (!fd)
		return HSAKMT_STATUS_ERROR;
	read_buf = malloc(PAGE_SIZE);
	if (!read_buf) {
		ret = HSAKMT_STATUS_NO_MEMORY;
		goto err1;
	}

	read_size = fread(read_buf, 1, PAGE_SIZE, fd);
	if (read_size <= 0) {
		ret = (errno == EPERM) ? HSAKMT_STATUS_NOT_SUPPORTED :
					 HSAKMT_STATUS_ERROR;
		goto err2;
	}

	/* Since we're using the buffer as a string, we make sure the string terminates */
	if (read_size >= PAGE_SIZE)
		read_size = PAGE_SIZE - 1;
	read_buf[read_size] = 0;

	prog = 0;
	p = read_buf;
	while (sscanf(p += prog, "%s %llu\n%n", prop_name, &prop_val, &prog) == 2) {
		if (strcmp(prop_name, "type") == 0)
			props->IoLinkType = (uint32_t)prop_val;
		else if (strcmp(prop_name, "version_major") == 0)
			props->VersionMajor = (uint32_t)prop_val;
		else if (strcmp(prop_name, "version_minor") == 0)
			props->VersionMinor = (uint32_t)prop_val;
		else if (strcmp(prop_name, "node_from") == 0) {
			if (sys_node_id != (uint32_t)prop_val) {
				ret = HSAKMT_STATUS_INVALID_NODE_UNIT;
				goto err2;
			}
			props->NodeFrom = node_id;
		} else if (strcmp(prop_name, "node_to") == 0) {
			bool is_node_supported;
			uint32_t sysfs_node_id;

			sysfs_node_id = (uint32_t)prop_val;
			ret = topology_sysfs_check_node_supported(sysfs_node_id, &is_node_supported);
			if (!is_node_supported) {
				ret = HSAKMT_STATUS_NOT_SUPPORTED;
				memset(props, 0, sizeof(*props));
				goto err2;
			}
			ret = topology_map_sysfs_to_user_node_id(sysfs_node_id, &props->NodeTo);
			if (ret != HSAKMT_STATUS_SUCCESS)
				goto err2;
		} else if (strcmp(prop_name, "weight") == 0)
			props->Weight = (uint32_t)prop_val;
		else if (strcmp(prop_name, "min_latency") == 0)
			props->MinimumLatency = (uint32_t)prop_val;
		else if (strcmp(prop_name, "max_latency") == 0)
			props->MaximumLatency = (uint32_t)prop_val;
		else if (strcmp(prop_name, "min_bandwidth") == 0)
			props->MinimumBandwidth = (uint32_t)prop_val;
		else if (strcmp(prop_name, "max_bandwidth") == 0)
			props->MaximumBandwidth = (uint32_t)prop_val;
		else if (strcmp(prop_name, "recommended_transfer_size") == 0)
			props->RecTransferSize = (uint32_t)prop_val;
		else if (strcmp(prop_name, "recommended_sdma_engine_id_mask") == 0)
			props->RecSdmaEngIdMask = (uint32_t)prop_val;
		else if (strcmp(prop_name, "flags") == 0)
			props->Flags.LinkProperty = (uint32_t)prop_val;
	}


err2:
	free(read_buf);
err1:
	fclose(fd);
	return ret;
}

/* topology_get_free_io_link_slot_for_node - For the given node_id, find the
 * next available free slot to add an io_link
 */
static HsaIoLinkProperties *topology_get_free_io_link_slot_for_node(uint32_t node_id,
								    const HsaSystemProperties *sys_props,
								    node_props_t *node_props)
{
	HsaIoLinkProperties *props;

	if (node_id >= sys_props->NumNodes) {
		pr_err("Invalid node [%d]\n", node_id);
		return NULL;
	}

	props = node_props[node_id].link;
	if (!props) {
		pr_err("No io_link reported for Node [%d]\n", node_id);
		return NULL;
	}

	if (node_props[node_id].node.NumIOLinks >= sys_props->NumNodes - 1) {
		pr_err("No more space for io_link for Node [%d]\n", node_id);
		return NULL;
	}

	return &props[node_props[node_id].node.NumIOLinks];
}

/* topology_add_io_link_for_node - If a free slot is available,
 * add io_link for the given Node.
 * TODO: Add other members of HsaIoLinkProperties
 */
static HSAKMT_STATUS topology_add_io_link_for_node(uint32_t node_from,
						   const HsaSystemProperties *sys_props,
						   node_props_t *node_props,
						   HSA_IOLINKTYPE IoLinkType,
						   uint32_t node_to,
						   uint32_t Weight)
{
	HsaIoLinkProperties *props;

	props = topology_get_free_io_link_slot_for_node(node_from,
			sys_props, node_props);
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
static int32_t gpu_get_direct_link_cpu(uint32_t gpu_node, node_props_t *node_props)
{
	HsaIoLinkProperties *props = node_props[gpu_node].link;
	uint32_t i;

	if (!node_props[gpu_node].node.KFDGpuID || !props ||
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
					    node_props_t *node_props, HSAuint32 *weight,
					    HSA_IOLINKTYPE *type)
{
	HsaIoLinkProperties *props = node_props[node1].link;
	uint32_t i;

	if (!props)
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
					      node_props_t *node_props, HSAuint32 *weight,
					      HSA_IOLINKTYPE *type)
{
	int32_t dir_cpu1 = -1, dir_cpu2 = -1;
	HSAuint32 weight1 = 0, weight2 = 0, weight3 = 0;
	HSAKMT_STATUS ret;
	uint32_t i;

	*weight = 0;
	*type = HSA_IOLINKTYPE_UNDEFINED;

	if (node1 == node2)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	/* CPU->CPU is not an indirect link */
	if (!node_props[node1].node.KFDGpuID && !node_props[node2].node.KFDGpuID)
		return HSAKMT_STATUS_INVALID_NODE_UNIT;

	if (node_props[node1].node.HiveID &&
	    node_props[node2].node.HiveID &&
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
			if (node_props[node2].mem[i].HeapType ==
				HSA_HEAPTYPE_FRAME_BUFFER_PUBLIC)
				break;
		if (i >=  node_props[node2].node.NumMemoryBanks)
			return HSAKMT_STATUS_ERROR;
	}
	/* Possible topology:
	 *   GPU --(weight1) -- CPU -- (weight2) -- GPU
	 *   GPU --(weight1) -- CPU -- (weight2) -- CPU -- (weight3) -- GPU
	 *   GPU --(weight1) -- CPU -- (weight2) -- CPU
	 *   CPU -- (weight2) -- CPU -- (weight3) -- GPU
	 */
	if (dir_cpu1 >= 0) { /* GPU->CPU ... */
		if (dir_cpu2 >= 0) {
			if (dir_cpu1 == dir_cpu2) /* GPU->CPU->GPU*/ {
				ret = get_direct_iolink_info(node1, dir_cpu1,
						node_props, &weight1, NULL);
				if (ret != HSAKMT_STATUS_SUCCESS)
					return ret;
				ret = get_direct_iolink_info(dir_cpu1, node2,
						node_props, &weight2, type);
			} else /* GPU->CPU->CPU->GPU*/ {
				ret = get_direct_iolink_info(node1, dir_cpu1,
						node_props, &weight1, NULL);
				if (ret != HSAKMT_STATUS_SUCCESS)
					return ret;
				ret = get_direct_iolink_info(dir_cpu1, dir_cpu2,
						node_props, &weight2, type);
				if (ret != HSAKMT_STATUS_SUCCESS)
					return ret;
				/* On QPI interconnection, GPUs can't access
				 * each other if they are attached to different
				 * CPU sockets. CPU<->CPU weight larger than 20
				 * means the two CPUs are in different sockets.
				 */
				if (*type == HSA_IOLINK_TYPE_QPI_1_1
					&& weight2 > 20)
					return HSAKMT_STATUS_NOT_SUPPORTED;
				ret = get_direct_iolink_info(dir_cpu2, node2,
						node_props, &weight3, NULL);
			}
		} else /* GPU->CPU->CPU */ {
			ret = get_direct_iolink_info(node1, dir_cpu1, node_props,
							&weight1, NULL);
			if (ret != HSAKMT_STATUS_SUCCESS)
				return ret;
			ret = get_direct_iolink_info(dir_cpu1, node2, node_props,
							&weight2, type);
		}
	} else { /* CPU->CPU->GPU */
		ret = get_direct_iolink_info(node1, dir_cpu2, node_props, &weight2,
					type);
		if (ret != HSAKMT_STATUS_SUCCESS)
			return ret;
		ret = get_direct_iolink_info(dir_cpu2, node2, node_props, &weight3,
						NULL);
	}

	if (ret != HSAKMT_STATUS_SUCCESS)
		return ret;

	*weight = weight1 + weight2 + weight3;
	return HSAKMT_STATUS_SUCCESS;
}

static void topology_create_indirect_gpu_links(const HsaSystemProperties *sys_props,
					       node_props_t *node_props)
{

	uint32_t i, j;
	HSAuint32 weight;
	HSA_IOLINKTYPE type;

	for (i = 0; i < sys_props->NumNodes - 1; i++) {
		for (j = i + 1; j < sys_props->NumNodes; j++) {
			get_indirect_iolink_info(i, j, node_props, &weight, &type);
			if (!weight)
				goto try_alt_dir;
			if (topology_add_io_link_for_node(i, sys_props, node_props,
				type, j, weight) != HSAKMT_STATUS_SUCCESS)
				pr_err("Fail to add IO link %d->%d\n", i, j);
try_alt_dir:
			get_indirect_iolink_info(j, i, node_props, &weight, &type);
			if (!weight)
				continue;
			if (topology_add_io_link_for_node(j, sys_props, node_props,
				type, i, weight) != HSAKMT_STATUS_SUCCESS)
				pr_err("Fail to add IO link %d->%d\n", j, i);
		}
	}
}

HSAKMT_STATUS topology_take_snapshot(void)
{
	uint32_t gen_start, gen_end, i, mem_id, cache_id;
	HsaSystemProperties sys_props;
	node_props_t *temp_props = 0;
	HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;
	struct proc_cpuinfo *cpuinfo;
	const uint32_t num_procs = get_nprocs();
	uint32_t num_ioLinks;
	bool p2p_links = false;
	uint32_t num_p2pLinks = 0;

	cpuinfo = calloc(num_procs, sizeof(struct proc_cpuinfo));
	if (!cpuinfo) {
		pr_err("Fail to allocate memory for CPU info\n");
		return HSAKMT_STATUS_NO_MEMORY;
	}
	topology_parse_cpuinfo(cpuinfo, num_procs);

retry:
	ret = topology_sysfs_get_generation(&gen_start);
	if (ret != HSAKMT_STATUS_SUCCESS)
		goto err;
	ret = hsakmt_topology_sysfs_get_system_props(&sys_props);
	if (ret != HSAKMT_STATUS_SUCCESS)
		goto err;
	if (sys_props.NumNodes > 0) {
		temp_props = calloc(sys_props.NumNodes * sizeof(node_props_t), 1);
		if (!temp_props) {
			ret = HSAKMT_STATUS_NO_MEMORY;
			goto err;
		}
		for (i = 0; i < sys_props.NumNodes; i++) {
			ret = topology_sysfs_get_node_props(i,
					&temp_props[i].node,
					&p2p_links, &num_p2pLinks);
			if (ret != HSAKMT_STATUS_SUCCESS) {
				free_properties(temp_props, i);
				goto err;
			}

			if (temp_props[i].node.NumCPUCores)
				topology_get_cpu_model_name(&temp_props[i].node,
							cpuinfo, num_procs);

			if (temp_props[i].node.NumMemoryBanks) {
				temp_props[i].mem = calloc(temp_props[i].node.NumMemoryBanks * sizeof(HsaMemoryProperties), 1);
				if (!temp_props[i].mem) {
					ret = HSAKMT_STATUS_NO_MEMORY;
					free_properties(temp_props, i + 1);
					goto err;
				}
				for (mem_id = 0; mem_id < temp_props[i].node.NumMemoryBanks; mem_id++) {
					ret = topology_sysfs_get_mem_props(i, mem_id, &temp_props[i].mem[mem_id]);
					if (ret != HSAKMT_STATUS_SUCCESS) {
						free_properties(temp_props, i + 1);
						goto err;
					}
				}
			}

			if (temp_props[i].node.NumCaches) {
				temp_props[i].cache = calloc(temp_props[i].node.NumCaches * sizeof(HsaCacheProperties), 1);
				if (!temp_props[i].cache) {
					ret = HSAKMT_STATUS_NO_MEMORY;
					free_properties(temp_props, i + 1);
					goto err;
				}
				for (cache_id = 0; cache_id < temp_props[i].node.NumCaches; cache_id++) {
					ret = topology_sysfs_get_cache_props(i, cache_id, &temp_props[i].cache[cache_id]);
					if (ret != HSAKMT_STATUS_SUCCESS) {
						free_properties(temp_props, i + 1);
						goto err;
					}
				}
			} else if (!temp_props[i].node.KFDGpuID) { /* a CPU node */
				ret = topology_get_cpu_cache_props(
						i, cpuinfo, &temp_props[i]);
				if (ret != HSAKMT_STATUS_SUCCESS) {
					free_properties(temp_props, i + 1);
					goto err;
				}
			}

			/* To simplify, allocate maximum needed memory for io_links for each node. This
			 * removes the need for realloc when indirect and QPI links are added later
			 */
			temp_props[i].link = calloc(sys_props.NumNodes - 1, sizeof(HsaIoLinkProperties));
			if (!temp_props[i].link) {
				ret = HSAKMT_STATUS_NO_MEMORY;
				free_properties(temp_props, i + 1);
				goto err;
			}
			num_ioLinks = temp_props[i].node.NumIOLinks - num_p2pLinks;
			uint32_t link_id = 0;

			if (num_ioLinks) {
				uint32_t sys_link_id = 0;

				/* Parse all the sysfs specified io links. Skip the ones where the
				 * remote node (node_to) is not accessible
				 */
				while (sys_link_id < num_ioLinks &&
					link_id < sys_props.NumNodes - 1) {
					ret = topology_sysfs_get_iolink_props(i, sys_link_id++,
								&temp_props[i].link[link_id], false);
					if (ret == HSAKMT_STATUS_NOT_SUPPORTED) {
						continue;
					} else if (ret != HSAKMT_STATUS_SUCCESS) {
						free_properties(temp_props, i + 1);
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
				while (sys_link_id < num_p2pLinks &&
					link_id < sys_props.NumNodes - 1) {
					ret = topology_sysfs_get_iolink_props(i, sys_link_id++,
								&temp_props[i].link[link_id], true);
					if (ret == HSAKMT_STATUS_NOT_SUPPORTED) {
						continue;
					} else if (ret != HSAKMT_STATUS_SUCCESS) {
						free_properties(temp_props, i + 1);
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
		topology_create_indirect_gpu_links(&sys_props, temp_props);
	}

	ret = topology_sysfs_get_generation(&gen_end);
	if (ret != HSAKMT_STATUS_SUCCESS) {
		free_properties(temp_props, sys_props.NumNodes);
		goto err;
	}

	if (gen_start != gen_end) {
		free_properties(temp_props, sys_props.NumNodes);
		temp_props = 0;
		goto retry;
	}

	if (!g_system) {
		g_system = malloc(sizeof(HsaSystemProperties));
		if (!g_system) {
			free_properties(temp_props, sys_props.NumNodes);
			ret = HSAKMT_STATUS_NO_MEMORY;
			goto err;
		}
	}

	*g_system = sys_props;
	if (g_props)
		free(g_props);
	g_props = temp_props;
err:
	free(cpuinfo);
	return ret;
}

/* Drop the Snapshot of the HSA topology information. Assume lock is held. */
void topology_drop_snapshot(void)
{
	if (!!g_system != !!g_props)
		pr_warn("Probably inconsistency?\n");

	if (g_props) {
		/* Remove state */
		free_properties(g_props, g_system->NumNodes);
		g_props = NULL;
	}

	free(g_system);
	g_system = NULL;

	if (map_user_to_sysfs_node_id) {
		free(map_user_to_sysfs_node_id);
		map_user_to_sysfs_node_id = NULL;
		map_user_to_sysfs_node_id_size = 0;
	}
}

HSAKMT_STATUS hsakmt_validate_nodeid(uint32_t nodeid, uint32_t *gpu_id)
{
	if (!g_props || !g_system || g_system->NumNodes <= nodeid)
		return HSAKMT_STATUS_INVALID_NODE_UNIT;
	if (gpu_id)
		*gpu_id = g_props[nodeid].node.KFDGpuID;

	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS hsakmt_gpuid_to_nodeid(uint32_t gpu_id, uint32_t *node_id)
{
	uint64_t node_idx;

	for (node_idx = 0; node_idx < g_system->NumNodes; node_idx++) {
		if (g_props[node_idx].node.KFDGpuID == gpu_id) {
			*node_id = node_idx;
			return HSAKMT_STATUS_SUCCESS;
		}
	}

	return HSAKMT_STATUS_INVALID_NODE_UNIT;

}

HSAKMT_STATUS HSAKMTAPI hsaKmtAcquireSystemProperties(HsaSystemProperties *SystemProperties)
{
	HSAKMT_STATUS err = HSAKMT_STATUS_SUCCESS;

	CHECK_KFD_OPEN();

	if (!SystemProperties)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	pthread_mutex_lock(&hsakmt_mutex);

	/* We already have a valid snapshot. Avoid double initialization that
	 * would leak memory.
	 */
	if (g_system) {
		*SystemProperties = *g_system;
		goto out;
	}

	err = topology_take_snapshot();
	if (err != HSAKMT_STATUS_SUCCESS)
		goto out;

	assert(g_system);

	if (hsakmt_use_model)
		model_init();

	err = hsakmt_fmm_init_process_apertures(g_system->NumNodes);
	if (err != HSAKMT_STATUS_SUCCESS)
		goto init_process_apertures_failed;

	err = hsakmt_init_process_doorbells(g_system->NumNodes);
	if (err != HSAKMT_STATUS_SUCCESS)
		goto init_doorbells_failed;

	*SystemProperties = *g_system;

	goto out;

init_doorbells_failed:
	hsakmt_fmm_destroy_process_apertures();
init_process_apertures_failed:
	topology_drop_snapshot();

out:
	pthread_mutex_unlock(&hsakmt_mutex);
	return err;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtReleaseSystemProperties(void)
{
	pthread_mutex_lock(&hsakmt_mutex);

	hsakmt_destroy_process_doorbells();
	hsakmt_fmm_destroy_process_apertures();
	topology_drop_snapshot();

	pthread_mutex_unlock(&hsakmt_mutex);

	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS hsakmt_topology_get_node_props(HSAuint32 NodeId,
				      HsaNodeProperties *NodeProperties)
{
	if (!g_system || !g_props || NodeId >= g_system->NumNodes)
		return HSAKMT_STATUS_ERROR;

	*NodeProperties = g_props[NodeId].node;
	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtGetNodeProperties(HSAuint32 NodeId,
						HsaNodeProperties *NodeProperties)
{
	HSAKMT_STATUS err;
	uint32_t gpu_id;

	if (!NodeProperties)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	CHECK_KFD_OPEN();
	pthread_mutex_lock(&hsakmt_mutex);

	err = hsakmt_validate_nodeid(NodeId, &gpu_id);
	if (err != HSAKMT_STATUS_SUCCESS)
		goto out;

	err = hsakmt_topology_get_node_props(NodeId, NodeProperties);
	if (err != HSAKMT_STATUS_SUCCESS)
		goto out;
	/* For CPU only node don't add any additional GPU memory banks. */
	if (gpu_id) {
		uint64_t base, limit;
		if (hsakmt_is_dgpu)
			NodeProperties->NumMemoryBanks += NUM_OF_DGPU_HEAPS;
		else
			NodeProperties->NumMemoryBanks += NUM_OF_IGPU_HEAPS;
		if (hsakmt_fmm_get_aperture_base_and_limit(FMM_MMIO, gpu_id, &base,
				&limit) == HSAKMT_STATUS_SUCCESS)
			NodeProperties->NumMemoryBanks += 1;
	}

out:
	pthread_mutex_unlock(&hsakmt_mutex);
	return err;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtGetNodeMemoryProperties(HSAuint32 NodeId,
						      HSAuint32 NumBanks,
						      HsaMemoryProperties *MemoryProperties)
{
	HSAKMT_STATUS err = HSAKMT_STATUS_SUCCESS;
	uint32_t i, gpu_id;
	HSAuint64 aperture_limit;

	if (!MemoryProperties)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	CHECK_KFD_OPEN();
	pthread_mutex_lock(&hsakmt_mutex);

	err = hsakmt_validate_nodeid(NodeId, &gpu_id);
	if (err != HSAKMT_STATUS_SUCCESS)
		goto out;

	memset(MemoryProperties, 0, NumBanks * sizeof(HsaMemoryProperties));

	for (i = 0; i < MIN(g_props[NodeId].node.NumMemoryBanks, NumBanks); i++) {
		assert(g_props[NodeId].mem);
		MemoryProperties[i] = g_props[NodeId].mem[i];
	}

	/* The following memory banks does not apply to CPU only node */
	if (gpu_id == 0)
		goto out;

	/*Add LDS*/
	if (i < NumBanks &&
		hsakmt_fmm_get_aperture_base_and_limit(FMM_LDS, gpu_id,
				&MemoryProperties[i].VirtualBaseAddress, &aperture_limit) == HSAKMT_STATUS_SUCCESS) {
		MemoryProperties[i].HeapType = HSA_HEAPTYPE_GPU_LDS;
		MemoryProperties[i].SizeInBytes = g_props[NodeId].node.LDSSizeInKB * 1024;
		i++;
	}

	/* Add Local memory - HSA_HEAPTYPE_FRAME_BUFFER_PRIVATE.
	 * For dGPU the topology node contains Local Memory and it is added by
	 * the for loop above
	 */
	if (hsakmt_get_gfxv_by_node_id(NodeId) == GFX_VERSION_KAVERI && i < NumBanks &&
		g_props[NodeId].node.LocalMemSize > 0 &&
		hsakmt_fmm_get_aperture_base_and_limit(FMM_GPUVM, gpu_id,
				&MemoryProperties[i].VirtualBaseAddress, &aperture_limit) == HSAKMT_STATUS_SUCCESS) {
		MemoryProperties[i].HeapType = HSA_HEAPTYPE_FRAME_BUFFER_PRIVATE;
		MemoryProperties[i].SizeInBytes = g_props[NodeId].node.LocalMemSize;
		i++;
	}

	/* Add SCRATCH */
	if (i < NumBanks &&
		hsakmt_fmm_get_aperture_base_and_limit(FMM_SCRATCH, gpu_id,
				&MemoryProperties[i].VirtualBaseAddress, &aperture_limit) == HSAKMT_STATUS_SUCCESS) {
		MemoryProperties[i].HeapType = HSA_HEAPTYPE_GPU_SCRATCH;
		MemoryProperties[i].SizeInBytes = (aperture_limit - MemoryProperties[i].VirtualBaseAddress) + 1;
		i++;
	}

	/* Add SVM aperture */
	if (hsakmt_topology_is_svm_needed(g_props[NodeId].node.EngineId) && i < NumBanks &&
	    hsakmt_fmm_get_aperture_base_and_limit(
		    FMM_SVM, gpu_id, &MemoryProperties[i].VirtualBaseAddress,
		    &aperture_limit) == HSAKMT_STATUS_SUCCESS) {
		MemoryProperties[i].HeapType = HSA_HEAPTYPE_DEVICE_SVM;
		MemoryProperties[i].SizeInBytes = (aperture_limit - MemoryProperties[i].VirtualBaseAddress) + 1;
		i++;
	}

	/* Add mmio aperture */
	if (i < NumBanks &&
		hsakmt_fmm_get_aperture_base_and_limit(FMM_MMIO, gpu_id,
				&MemoryProperties[i].VirtualBaseAddress, &aperture_limit) == HSAKMT_STATUS_SUCCESS) {
		MemoryProperties[i].HeapType = HSA_HEAPTYPE_MMIO_REMAP;
		MemoryProperties[i].SizeInBytes = (aperture_limit - MemoryProperties[i].VirtualBaseAddress) + 1;
		i++;
	}

out:
	pthread_mutex_unlock(&hsakmt_mutex);
	return err;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtGetNodeCacheProperties(HSAuint32 NodeId,
						     HSAuint32 ProcessorId,
						     HSAuint32 NumCaches,
						     HsaCacheProperties *CacheProperties)
{
	HSAKMT_STATUS err;
	uint32_t i;

	if (!CacheProperties)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	CHECK_KFD_OPEN();
	pthread_mutex_lock(&hsakmt_mutex);

	/* KFD ADD page 18, snapshot protocol violation */
	if (!g_system || NodeId >= g_system->NumNodes) {
		err = HSAKMT_STATUS_INVALID_NODE_UNIT;
		goto out;
	}

	if (NumCaches > g_props[NodeId].node.NumCaches) {
		err = HSAKMT_STATUS_INVALID_PARAMETER;
		goto out;
	}

	for (i = 0; i < MIN(g_props[NodeId].node.NumCaches, NumCaches); i++) {
		assert(g_props[NodeId].cache);
		CacheProperties[i] = g_props[NodeId].cache[i];
	}

	err = HSAKMT_STATUS_SUCCESS;

out:
	pthread_mutex_unlock(&hsakmt_mutex);
	return err;
}

HSAKMT_STATUS hsakmt_topology_get_iolink_props(HSAuint32 NodeId,
					HSAuint32 NumIoLinks,
					HsaIoLinkProperties *IoLinkProperties)
{
	if (!g_system || !g_props || NodeId >= g_system->NumNodes)
		return HSAKMT_STATUS_ERROR;

	memcpy(IoLinkProperties, g_props[NodeId].link,
	       NumIoLinks * sizeof(*IoLinkProperties));

	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtGetNodeIoLinkProperties(HSAuint32 NodeId,
						      HSAuint32 NumIoLinks,
						      HsaIoLinkProperties *IoLinkProperties)
{
	HSAKMT_STATUS err;

	if (!IoLinkProperties)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	CHECK_KFD_OPEN();

	pthread_mutex_lock(&hsakmt_mutex);

	/* KFD ADD page 18, snapshot protocol violation */
	if (!g_system || NodeId >= g_system->NumNodes ) {
		err = HSAKMT_STATUS_INVALID_NODE_UNIT;
		goto out;
	}

	if (NumIoLinks > g_props[NodeId].node.NumIOLinks) {
		err = HSAKMT_STATUS_INVALID_PARAMETER;
		goto out;
	}

	assert(g_props[NodeId].link);
	err = hsakmt_topology_get_iolink_props(NodeId, NumIoLinks, IoLinkProperties);

out:
	pthread_mutex_unlock(&hsakmt_mutex);
	return err;
}

uint32_t hsakmt_get_gfxv_by_node_id(HSAuint32 node_id)
{
	return HSA_GET_GFX_VERSION_FULL(g_props[node_id].node.EngineId.ui32);
}

uint16_t hsakmt_get_device_id_by_node_id(HSAuint32 node_id)
{
	if (!g_props || !g_system || g_system->NumNodes <= node_id)
		return 0;

	return g_props[node_id].node.DeviceId;
}

bool hsakmt_prefer_ats(HSAuint32 node_id)
{
	return g_props[node_id].node.Capability.ui32.HSAMMUPresent
			&& g_props[node_id].node.NumCPUCores
			&& g_props[node_id].node.NumFComputeCores;
}

uint16_t hsakmt_get_device_id_by_gpu_id(HSAuint32 gpu_id)
{
	unsigned int i;

	if (!g_props || !g_system)
		return 0;

	for (i = 0; i < g_system->NumNodes; i++) {
		if (g_props[i].node.KFDGpuID == gpu_id)
			return g_props[i].node.DeviceId;
	}

	return 0;
}

uint32_t hsakmt_get_direct_link_cpu(uint32_t gpu_node)
{
	HSAuint64 size = 0;
	int32_t cpu_id;
	HSAuint32 i;

	cpu_id = gpu_get_direct_link_cpu(gpu_node, g_props);
	if (cpu_id == -1)
		return INVALID_NODEID;

	assert(g_props[cpu_id].mem);

	for (i = 0; i < g_props[cpu_id].node.NumMemoryBanks; i++)
		size += g_props[cpu_id].mem[i].SizeInBytes;

	return size ? (uint32_t)cpu_id : INVALID_NODEID;
}


HSAKMT_STATUS hsakmt_validate_nodeid_array(uint32_t **gpu_id_array,
		uint32_t NumberOfNodes, uint32_t *NodeArray)
{
	HSAKMT_STATUS ret;
	unsigned int i;

	if (NumberOfNodes == 0 || !NodeArray || !gpu_id_array)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	/* Translate Node IDs to gpu_ids */
	*gpu_id_array = malloc(NumberOfNodes * sizeof(uint32_t));
	if (!(*gpu_id_array))
		return HSAKMT_STATUS_NO_MEMORY;
	for (i = 0; i < NumberOfNodes; i++) {
		ret = hsakmt_validate_nodeid(NodeArray[i], *gpu_id_array + i);
		if (ret != HSAKMT_STATUS_SUCCESS) {
			free(*gpu_id_array);
			break;
		}
	}

	return ret;
}

inline uint32_t hsakmt_get_num_sysfs_nodes(void)
{
	return num_sysfs_nodes;
}
