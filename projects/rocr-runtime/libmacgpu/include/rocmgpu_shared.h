/*
 * rocmgpu_shared.h — DEXT ABI definitions.
 *
 * This header mirrors the userspace-visible ABI published by
 * ROCmGPU.dext (TheRock/userspace_driver/macos_driver/ROCmGPU/
 * ROCmGPUDriver/ROCmGPUShared.h). Keep in sync with the DEXT source;
 * the ABI is versioned implicitly by selector values — changes
 * require a coordinated update on both sides.
 */

#ifndef MACGPU_ROCMGPU_SHARED_H_
#define MACGPU_ROCMGPU_SHARED_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// IOUserClient external method selectors.
enum ROCmGPUSelector {
    kROCmGPU_GetInfo       = 0,
    kROCmGPU_Reset         = 1,
    kROCmGPU_CfgRead       = 2,
    kROCmGPU_CfgWrite      = 3,
    kROCmGPU_MMIORead32    = 4,
    kROCmGPU_MMIOWrite32   = 5,
    kROCmGPU_MapBAR        = 6,
    kROCmGPU_UnmapBAR      = 7,
    kROCmGPU_AllocDMA      = 8,
    kROCmGPU_FreeDMA       = 9,
    kROCmGPU_MapDMA        = 10,
    kROCmGPU_EnableMSI     = 11,
    kROCmGPU_WaitInterrupt = 12,
    kROCmGPU_SelectorCount = 13,
};

// Memory-type constants for IOConnectMapMemory64().
enum ROCmGPUMemoryType {
    kROCmGPU_MemType_BAR0    = 0,
    kROCmGPU_MemType_BAR1    = 1,
    kROCmGPU_MemType_BAR2    = 2,
    kROCmGPU_MemType_BAR3    = 3,
    kROCmGPU_MemType_BAR4    = 4,
    kROCmGPU_MemType_BAR5    = 5,
    kROCmGPU_MemType_DMABase = 0x100,
};

struct ROCmGPUDeviceInfo {
    uint16_t vendorID;
    uint16_t deviceID;
    uint16_t subsystemVendorID;
    uint16_t subsystemDeviceID;
    uint8_t  revisionID;
    uint8_t  _pad[3];
    struct {
        uint64_t size;
        uint8_t  memoryIndex;
        uint8_t  type;
        uint8_t  is64bit;
        uint8_t  prefetchable;
        uint8_t  _pad2[4];
    } bars[6];
    uint64_t vramSize;
};

enum ROCmGPUDMAFlags {
    kROCmGPU_DMA_Contiguous = (1u << 0),
    kROCmGPU_DMA_Uncached   = (1u << 1),
    kROCmGPU_DMA_ReadOnly   = (1u << 2),
    kROCmGPU_DMA_WriteOnly  = (1u << 3),
};

struct ROCmGPUDMAInfo {
    uint64_t bufferID;
    uint64_t size;
    uint32_t segmentCount;
    uint32_t _pad;
    struct {
        uint64_t address;
        uint64_t length;
    } segments[64];
};

enum ROCmGPUInterruptStatus {
    kROCmGPU_IntStatus_OK      = 0,
    kROCmGPU_IntStatus_Timeout = 1,
    kROCmGPU_IntStatus_Error   = 2,
};

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MACGPU_ROCMGPU_SHARED_H_
