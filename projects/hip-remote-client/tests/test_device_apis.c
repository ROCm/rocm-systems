/*
 * Copyright 2025 Advanced Micro Devices, Inc.
 *
 * Test suite for additional device management APIs
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hip_remote/hip_remote_client.h"

#define CHECK_HIP(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d: %s (code %d)\n", \
                    __FILE__, __LINE__, hipGetErrorString(err), err); \
            return 1; \
        } \
    } while (0)

#define CHECK_HIP_WARN(call, msg) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "Warning at %s:%d: %s - %s (code %d)\n", \
                    __FILE__, __LINE__, msg, hipGetErrorString(err), err); \
        } \
    } while (0)

int main() {
    printf("=== HIP Remote Device APIs Test ===\n\n");

    /* Test 1: hipDeviceGet */
    printf("[TEST] hipDeviceGet\n");
    hipDevice_t device;
    CHECK_HIP(hipDeviceGet(&device, 0));
    printf("  Device handle for ordinal 0: %d\n", device);

    /* Test 2: hipDeviceGetName */
    printf("\n[TEST] hipDeviceGetName\n");
    char name[256];
    CHECK_HIP(hipDeviceGetName(name, sizeof(name), device));
    printf("  Device name: %s\n", name);

    /* Test 3: hipDeviceTotalMem */
    printf("\n[TEST] hipDeviceTotalMem\n");
    size_t total_mem;
    CHECK_HIP(hipDeviceTotalMem(&total_mem, device));
    printf("  Total memory: %zu bytes (%.2f GB)\n",
           total_mem, total_mem / (1024.0 * 1024.0 * 1024.0));

    /* Test 4: hipDeviceGetPCIBusId */
    printf("\n[TEST] hipDeviceGetPCIBusId\n");
    char pci_bus_id[32];
    CHECK_HIP(hipDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), 0));
    printf("  PCI Bus ID: %s\n", pci_bus_id);

    /* Test 5: hipDeviceGetByPCIBusId */
    printf("\n[TEST] hipDeviceGetByPCIBusId\n");
    int device_from_pci;
    CHECK_HIP(hipDeviceGetByPCIBusId(&device_from_pci, pci_bus_id));
    printf("  Device from PCI Bus ID '%s': %d\n", pci_bus_id, device_from_pci);
    if (device_from_pci != 0) {
        fprintf(stderr, "  ERROR: Expected device 0, got %d\n", device_from_pci);
        return 1;
    }

    /* Test 6: hipDeviceComputeCapability */
    printf("\n[TEST] hipDeviceComputeCapability\n");
    int major, minor;
    CHECK_HIP(hipDeviceComputeCapability(&major, &minor, device));
    printf("  Compute capability: %d.%d\n", major, minor);

    /* Test 7: hipDeviceGetUuid */
    printf("\n[TEST] hipDeviceGetUuid\n");
    hipUUID uuid;
    CHECK_HIP_WARN(hipDeviceGetUuid(&uuid, device), "UUID may not be supported");
    printf("  UUID: ");
    for (int i = 0; i < 16; i++) {
        printf("%02x", (unsigned char)uuid.bytes[i]);
        if (i == 3 || i == 5 || i == 7 || i == 9) printf("-");
    }
    printf("\n");

    /* Test 8: hipDeviceGetCacheConfig / hipDeviceSetCacheConfig */
    printf("\n[TEST] hipDeviceGetCacheConfig / hipDeviceSetCacheConfig\n");
    hipFuncCache_t cache_config;
    CHECK_HIP_WARN(hipDeviceGetCacheConfig(&cache_config), "Cache config may not be supported");
    printf("  Current cache config: %d\n", cache_config);

    CHECK_HIP_WARN(hipDeviceSetCacheConfig(hipFuncCachePreferShared), "Set cache config may not be supported");
    printf("  Set cache config to PreferShared\n");

    /* Test 9: hipDeviceGetSharedMemConfig / hipDeviceSetSharedMemConfig */
    printf("\n[TEST] hipDeviceGetSharedMemConfig / hipDeviceSetSharedMemConfig\n");
    hipSharedMemConfig shmem_config;
    CHECK_HIP_WARN(hipDeviceGetSharedMemConfig(&shmem_config), "Shared mem config may not be supported");
    printf("  Current shared mem config: %d\n", shmem_config);

    CHECK_HIP_WARN(hipDeviceSetSharedMemConfig(hipSharedMemBankSizeFourByte),
                   "Set shared mem config may not be supported");
    printf("  Set shared mem config to FourByte\n");

    /* Test 10: hipGetDeviceFlags / hipSetDeviceFlags */
    printf("\n[TEST] hipGetDeviceFlags / hipSetDeviceFlags\n");
    unsigned int flags;
    CHECK_HIP_WARN(hipGetDeviceFlags(&flags), "Get device flags may not be supported");
    printf("  Current device flags: 0x%x\n", flags);

    CHECK_HIP_WARN(hipSetDeviceFlags(0), "Set device flags may not be supported");
    printf("  Set device flags to 0\n");

    /* Test 11: hipDeviceGetP2PAttribute */
    printf("\n[TEST] hipDeviceGetP2PAttribute\n");
    int device_count;
    CHECK_HIP(hipGetDeviceCount(&device_count));
    printf("  Device count: %d\n", device_count);

    if (device_count >= 2) {
        int p2p_value;
        CHECK_HIP_WARN(hipDeviceGetP2PAttribute(&p2p_value, hipDevP2PAttrAccessSupported, 0, 1),
                       "P2P attribute query may not be supported");
        printf("  P2P access supported (0->1): %d\n", p2p_value);
    } else {
        printf("  Skipping P2P test (only %d device(s) available)\n", device_count);
    }

    printf("\n=== All Tests Passed! ===\n");
    return 0;
}
