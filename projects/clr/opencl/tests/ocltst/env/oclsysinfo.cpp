/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "oclsysinfo.h"

#include <CL/cl.h>
#include <CL/cl_ext.h>

#include <cstdio>

#ifndef MAX_DEVICES
#define MAX_DEVICES 16
#endif  // MAX_DEVICES

int oclSysInfo(std::string& info_string, bool use_cpu, unsigned dev_id,
               unsigned int platformIndex) {
  /*
   * Have a look at the available platforms and pick the one
   * in the platforms vector in index "platformIndex".
   */

  cl_uint numPlatforms;
  cl_platform_id platform = NULL;
  cl_uint num_devices = 0;
  cl_device_id* devices = NULL;
  cl_device_id device = NULL;

  int error = clGetPlatformIDs(0, NULL, &numPlatforms);
  if (CL_SUCCESS != error) {
    fprintf(stderr, "clGetPlatformIDs() failed");
    return 0;
  }
  if (0 < numPlatforms) {
    cl_platform_id* platforms = new cl_platform_id[numPlatforms];
    error = clGetPlatformIDs(numPlatforms, platforms, NULL);
    if (CL_SUCCESS != error) {
      fprintf(stderr, "clGetPlatformIDs() failed");
      return 0;
    }
#if 0
        for (unsigned i = 0; i < numPlatforms; ++i) {
            /* Get the number of requested devices */
            error = clGetDeviceIDs(platforms[i],  (use_cpu) ? CL_DEVICE_TYPE_CPU : CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices );
#if 0
            /* clGetDeviceIDs fails when no GPU devices are present */
            if (error) {
              fprintf(stderr, "clGetDeviceIDs failed: %d\n", error );
              return 0;
            }
#endif
#if 0
            char pbuf[100];

            error = clGetPlatformInfo(
                         platforms[i],
                         CL_PLATFORM_VENDOR,
                         sizeof(pbuf),
                         pbuf,
                         NULL);
            if (!strcmp(pbuf, "Advanced Micro Devices, Inc.")) {
                platform = platforms[i];
                break;
            }
#else
            /* Select platform with GPU devices  present */
            if (num_devices > 0) {
                platform = platforms[i];
                break;
            }
#endif
		}
#endif
    error =
        clGetDeviceIDs(platforms[platformIndex],
                       (use_cpu) ? CL_DEVICE_TYPE_CPU : CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    if (error) {
      fprintf(stderr, "clGetDeviceIDs failed: %d\n", error);
      return 0;
    }
    platform = platforms[platformIndex];
    delete[] platforms;
  }
  if (dev_id >= num_devices) {
    fprintf(stderr, "Device selected does not exist.\n");
    return 0;
  }
  if (NULL == platform) {
    fprintf(stderr, "Couldn't find platform with GPU devices, cannot proceed.\n");
    return 0;
  }

  devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
  if (!devices) {
    fprintf(stderr, "no devices\n");
    return 0;
  }

  /* Get the requested device */
  error = clGetDeviceIDs(platform, (use_cpu) ? CL_DEVICE_TYPE_CPU : CL_DEVICE_TYPE_GPU, num_devices,
                         devices, NULL);
  if (error) {
    fprintf(stderr, "clGetDeviceIDs failed: %d\n", error);
    return 0;
  }

  device = devices[dev_id];

  char c[1024];
  char tmpString[256];
  static const char* no_yes[] = {"NO", "YES"};
  sprintf(tmpString, "\nCompute Device info:\n");
  info_string.append(tmpString);
  clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(c), &c, NULL);
  sprintf(tmpString, "\tPlatform Version: %s\n", c);
  info_string.append(tmpString);
  clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(c), &c, NULL);
  sprintf(tmpString, "\tDevice Name: %s\n", c);
  info_string.append(tmpString);
  clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(c), &c, NULL);
  sprintf(tmpString, "\tVendor: %s\n", c);
  info_string.append(tmpString);
  clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(c), &c, NULL);
  sprintf(tmpString, "\tDevice Version: %s\n", c);
  info_string.append(tmpString);
  clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(c), &c, NULL);
  sprintf(tmpString, "\tDriver Version: %s\n", c);
  info_string.append(tmpString);
  clGetDeviceInfo(device, CL_DEVICE_BOARD_NAME_AMD, sizeof(c), &c, NULL);
  sprintf(tmpString, "\tBoard Name: %s\n", c);
  info_string.append(tmpString);
#if defined(__linux__)
  cl_device_topology_amd topology;
  clGetDeviceInfo(device, CL_DEVICE_TOPOLOGY_AMD, sizeof(topology), &topology, NULL);
  if (topology.raw.type == CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD) {
    sprintf(tmpString, "\tDevice Topology: PCI[ B#%d, D#%d, F#%d]\n", topology.pcie.bus,
            topology.pcie.device, topology.pcie.function);
    info_string.append(tmpString);
  }
#endif
  free(devices);
  return 1;
}
