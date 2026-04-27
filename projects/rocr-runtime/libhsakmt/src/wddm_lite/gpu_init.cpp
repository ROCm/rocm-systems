/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
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

/*
 * GPU hardware initialization for the wddm_lite backend.
 *
 * Implements IP discovery (reading the binary table from VRAM) and
 * GMC initialization (programming MMHUB/GFXHUB registers) through
 * the WDDM escape interface.
 *
 * Ported from the Python reference implementations:
 *   - ip_discovery.py (binary table parser)
 *   - gmc_init.py (register programming)
 *
 * Reference: Linux amdgpu discovery.h, gmc_v12_0.c, mmhub_v4_1_0.c,
 *            gfxhub_v12_0.c
 */

#include "gpu_init.h"
#include "wddm_lite_device.h"
#include "wddm_lite_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ======================================================================
 * IP Discovery constants (from Linux amdgpu discovery.h)
 * ====================================================================== */

#define BINARY_SIGNATURE            0x28211407
#define DISCOVERY_TABLE_SIGNATURE   0x53445049  /* "IPDS" */
#define GC_TABLE_ID                 0x4347      /* "GC" */
#define HARVEST_TABLE_SIGNATURE     0x56524148  /* "HARV" */
#define PSP_HEADER_SIZE             256
#define DISCOVERY_TABLE_SIZE        (64 * 1024)

/* Table indices in binary_header.table_list */
#define TABLE_IP_DISCOVERY  0
#define TABLE_GC            1
#define TABLE_HARVEST_INFO  2
#define TOTAL_TABLES        6

/* Hardware IDs from soc15_hw_ip.h (full set) */
#define HWID_MP1      1
#define HWID_MP2      2
#define HWID_GC       11
#define HWID_VCN      12
#define HWID_MMHUB    34
#define HWID_ATHUB    35
#define HWID_OSSSYS   40   /* IH */
#define HWID_HDP      41
#define HWID_SDMA0    42
#define HWID_SDMA1    43
#define HWID_NBIF     108
#define HWID_MP0      255  /* PSP */


/* ======================================================================
 * MMHUB v4.1.0 register offsets (DWORD offsets from MMHUB base_index 0)
 * ====================================================================== */

/* FB location */
#define regMMMC_VM_FB_OFFSET                            0x04C7
#define regMMMC_VM_FB_LOCATION_BASE                     0x0554
#define regMMMC_VM_FB_LOCATION_TOP                      0x0555

/* AGP aperture */
#define regMMMC_VM_AGP_TOP                              0x0556
#define regMMMC_VM_AGP_BOT                              0x0557
#define regMMMC_VM_AGP_BASE                             0x0558

/* System aperture */
#define regMMMC_VM_SYSTEM_APERTURE_LOW_ADDR             0x0559
#define regMMMC_VM_SYSTEM_APERTURE_HIGH_ADDR            0x055A
#define regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB     0x04C8
#define regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB     0x04C9

/* L1 TLB */
#define regMMMC_VM_MX_L1_TLB_CNTL                      0x055B

/* L2 cache */
#define regMMVM_L2_CNTL                                 0x04E4
#define regMMVM_L2_CNTL2                                0x04E5
#define regMMVM_L2_CNTL3                                0x04E6
#define regMMVM_L2_CNTL4                                0x04FD
#define regMMVM_L2_CNTL5                                0x0503

/* Protection fault */
#define regMMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32   0x04F4
#define regMMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32   0x04F5
#define regMMVM_L2_PROTECTION_FAULT_STATUS_LO32         0x04F0
#define regMMVM_L2_PROTECTION_FAULT_CNTL                0x04EC
#define regMMVM_L2_PROTECTION_FAULT_CNTL2               0x04ED

/* Identity aperture */
#define regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_LO32   0x04F7
#define regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_HI32   0x04F8
#define regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_LO32  0x04F9
#define regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_HI32  0x04FA
#define regMMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_LO32      0x04FB
#define regMMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_HI32      0x04FC

/* Context control */
#define regMMVM_CONTEXT0_CNTL                           0x0564
#define regMMVM_CONTEXT1_CNTL                           0x0565

/* Page table base */
#define regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32      0x05CF
#define regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32      0x05D0
#define regMMVM_CONTEXT1_PAGE_TABLE_BASE_ADDR_LO32      0x05D1

/* Page table start/end */
#define regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32     0x05EF
#define regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32     0x05F0
#define regMMVM_CONTEXT1_PAGE_TABLE_START_ADDR_LO32     0x05F1
#define regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32       0x060F
#define regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32       0x0610
#define regMMVM_CONTEXT1_PAGE_TABLE_END_ADDR_LO32       0x0611

/* Invalidation engines */
#define regMMVM_INVALIDATE_ENG0_SEM                     0x0575
#define regMMVM_INVALIDATE_ENG0_REQ                     0x0587
#define regMMVM_INVALIDATE_ENG0_ACK                     0x0599
#define regMMVM_INVALIDATE_ENG0_ADDR_RANGE_LO32         0x05AB
#define regMMVM_INVALIDATE_ENG0_ADDR_RANGE_HI32         0x05AC

#define MMHUB_CTX_DISTANCE          1
#define MMHUB_CTX_ADDR_DISTANCE     2
#define MMHUB_ENG_DISTANCE          1
#define MMHUB_ENG_ADDR_DISTANCE     2


/* ======================================================================
 * GFXHUB v12.0 register offsets (DWORD offsets from GC base_index 0)
 * ====================================================================== */

#define regGCMC_VM_FB_OFFSET                            0x15A7
#define regGCMC_VM_FB_LOCATION_BASE                     0x1614
#define regGCMC_VM_FB_LOCATION_TOP                      0x1615

#define regGCMC_VM_AGP_TOP                              0x1616
#define regGCMC_VM_AGP_BOT                              0x1617
#define regGCMC_VM_AGP_BASE                             0x1618

#define regGCMC_VM_SYSTEM_APERTURE_LOW_ADDR             0x1619
#define regGCMC_VM_SYSTEM_APERTURE_HIGH_ADDR            0x161A
#define regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB     0x15A8
#define regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB     0x15A9

#define regGCMC_VM_MX_L1_TLB_CNTL                      0x161B

#define regGCVM_L2_CNTL                                 0x15C4
#define regGCVM_L2_CNTL2                                0x15C5
#define regGCVM_L2_CNTL3                                0x15C6
#define regGCVM_L2_CNTL4                                0x15DD
#define regGCVM_L2_CNTL5                                0x15E3

#define regGCVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32   0x15D4
#define regGCVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32   0x15D5

/* Protection fault status and control (GFX12: 64-bit status split LO32/HI32) */
#define regGCVM_L2_PROTECTION_FAULT_STATUS_LO32         0x15D0
#define regGCVM_L2_PROTECTION_FAULT_STATUS_HI32         0x15D1
#define regGCVM_L2_PROTECTION_FAULT_CNTL                0x15CC
#define regGCVM_L2_PROTECTION_FAULT_CNTL2               0x15CD
#define regGCVM_L2_PROTECTION_FAULT_ADDR_LO32           0x15D2
#define regGCVM_L2_PROTECTION_FAULT_ADDR_HI32           0x15D3

#define regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_LO32   0x15D7
#define regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_HI32   0x15D8
#define regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_LO32  0x15D9
#define regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_HI32  0x15DA
#define regGCVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_LO32      0x15DB
#define regGCVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_HI32      0x15DC

#define regGCVM_CONTEXT0_CNTL                           0x1624
#define regGCVM_CONTEXT1_CNTL                           0x1625

#define regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32      0x168F
#define regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32      0x1690
#define regGCVM_CONTEXT1_PAGE_TABLE_BASE_ADDR_LO32      0x1691
#define regGCVM_CONTEXT1_PAGE_TABLE_BASE_ADDR_HI32      0x1692

#define regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32     0x16AF
#define regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32     0x16B0
#define regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32       0x16CF
#define regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32       0x16D0
#define regGCVM_CONTEXT1_PAGE_TABLE_START_ADDR_LO32     0x16B1
#define regGCVM_CONTEXT1_PAGE_TABLE_START_ADDR_HI32     0x16B2
#define regGCVM_CONTEXT1_PAGE_TABLE_END_ADDR_LO32       0x16D1
#define regGCVM_CONTEXT1_PAGE_TABLE_END_ADDR_HI32       0x16D2

#define regGCVM_INVALIDATE_ENG0_SEM                     0x1635
#define regGCVM_INVALIDATE_ENG0_REQ                     0x1647
#define regGCVM_INVALIDATE_ENG0_ACK                     0x1659
#define regGCVM_INVALIDATE_ENG0_ADDR_RANGE_LO32         0x166B
#define regGCVM_INVALIDATE_ENG0_ADDR_RANGE_HI32         0x166C

#define GFXHUB_CTX_DISTANCE         1
#define GFXHUB_CTX_ADDR_DISTANCE    2
#define GFXHUB_ENG_DISTANCE         1
#define GFXHUB_ENG_ADDR_DISTANCE    2

/* CP debug (disable UTCL1 error halt for GFXHUB) */
#define regCP_DEBUG                                     0x1E1F


/* ======================================================================
 * GFX12 PTE flag definitions
 * ====================================================================== */

#define AMDGPU_PTE_VALID        (1ULL << 0)
#define AMDGPU_PTE_SYSTEM       (1ULL << 1)
#define AMDGPU_PTE_SNOOPED      (1ULL << 2)
#define AMDGPU_PTE_EXECUTABLE   (1ULL << 4)
#define AMDGPU_PTE_READABLE     (1ULL << 5)
#define AMDGPU_PTE_WRITEABLE    (1ULL << 6)
#define AMDGPU_PTE_IS_PTE       (1ULL << 63)

/* MTYPE for GFX12 (bits 55:54) */
#define MTYPE_UC                3
#define GFX12_PTE_MTYPE(m)      (((ULONGLONG)(m) & 0x3) << 54)

/* VM control bits */
#define L1_TLB_ENABLE                           (1 << 0)
#define L1_TLB_SYSTEM_ACCESS_MODE_MASK          (0x3 << 3)
#define L1_TLB_ENABLE_ADV_DRIVER_MODEL          (1 << 6)
#define L1_TLB_SYSTEM_APERTURE_UNMAPPED_ACCESS  (1 << 5)
#define VM_CONTEXT_ENABLE_CONTEXT               (1 << 0)

/* Default GART size: 512 MB */
#define DEFAULT_GART_SIZE       (512ULL * 1024 * 1024)

/* GART page table: one 8-byte PTE per 4KB page → 1MB table for 512MB */
#define GART_TABLE_SIZE(gart_sz) (((gart_sz) / 4096) * 8)


/* ======================================================================
 * Helper: read/write registers through SOC15 base addressing
 * ====================================================================== */

static ULONG mmhub_rreg(struct WddmLiteDevice *dev, ULONG reg)
{
    ULONG offset = (dev->hw.ip.mmhub_base + reg) * 4;
    return wddm_lite_read_reg32(dev, offset);
}

static void mmhub_wreg(struct WddmLiteDevice *dev, ULONG reg, ULONG val)
{
    ULONG offset = (dev->hw.ip.mmhub_base + reg) * 4;
    wddm_lite_write_reg32(dev, offset, val);
}

static ULONG gfxhub_rreg(struct WddmLiteDevice *dev, ULONG reg)
{
    return gpu_smn_rreg(dev, dev->hw.ip.gc_base + reg);
}

static void gfxhub_wreg(struct WddmLiteDevice *dev, ULONG reg, ULONG val)
{
    gpu_smn_wreg(dev, dev->hw.ip.gc_base + reg, val);
}

/* PSP (MP0) register offsets (DWORD, from mp_14_0_2_offset.h, BASE_IDX=0) */
#define regMPASP_SMN_C2PMSG_35  0x0063  /* Bootloader command/status */
#define regMPASP_SMN_C2PMSG_36  0x0064  /* FW address (>> 20 for 1MB align) */
#define regMPASP_SMN_C2PMSG_81  0x0091  /* SOS alive indicator */

static ULONG mp0_rreg(struct WddmLiteDevice *dev, ULONG reg)
{
    ULONG offset = (dev->hw.ip.mp0_base + reg) * 4;
    return wddm_lite_read_reg32(dev, offset);
}

static void mp0_wreg(struct WddmLiteDevice *dev, ULONG reg, ULONG val)
{
    ULONG offset = (dev->hw.ip.mp0_base + reg) * 4;
    wddm_lite_write_reg32(dev, offset, val);
}

/* Check if PSP SOS (Secure OS) is alive. */
static BOOLEAN psp_is_sos_alive(struct WddmLiteDevice *dev)
{
    if (dev->hw.ip.mp0_base == 0)
        return FALSE;
    ULONG val = mp0_rreg(dev, regMPASP_SMN_C2PMSG_81);
    return (val != 0);
}

/* MP1 C2PMSG register offsets (DWORD, from mp_11_0_offset.h, BASE_IDX=0) */
#define regMP1_SMN_C2PMSG_66    0x0282  /* Message register */
#define regMP1_SMN_C2PMSG_82    0x0292  /* Parameter register */
#define regMP1_SMN_C2PMSG_90    0x029A  /* Response register */

/* SMU v14_0_2 message IDs (from smu_v14_0_2_ppsmc.h in Linux kernel).
 * VERIFIED against codebrowser.dev/linux source — DO NOT guess IDs. */
#define PPSMC_MSG_TestMessage                0x01
#define PPSMC_MSG_GetSmuVersion              0x02
#define PPSMC_MSG_GetDriverIfVersion         0x03
#define PPSMC_MSG_SetAllowedFeaturesMaskLow  0x04
#define PPSMC_MSG_SetAllowedFeaturesMaskHigh 0x05
#define PPSMC_MSG_EnableAllSmuFeatures       0x06
#define PPSMC_MSG_DisableAllSmuFeatures      0x07
#define PPSMC_MSG_EnableSmuFeaturesLow       0x08
#define PPSMC_MSG_EnableSmuFeaturesHigh      0x09
#define PPSMC_MSG_DisableSmuFeaturesLow      0x0A
#define PPSMC_MSG_DisableSmuFeaturesHigh     0x0B
#define PPSMC_MSG_GetRunningSmuFeaturesLow   0x0C
#define PPSMC_MSG_GetRunningSmuFeaturesHigh  0x0D
#define PPSMC_MSG_SetDriverDramAddrHigh      0x0E
#define PPSMC_MSG_SetDriverDramAddrLow       0x0F
#define PPSMC_MSG_SetToolsDramAddrHigh       0x10
#define PPSMC_MSG_SetToolsDramAddrLow        0x11
#define PPSMC_MSG_TransferTableSmu2Dram      0x12
#define PPSMC_MSG_TransferTableDram2Smu      0x13
#define PPSMC_MSG_UseDefaultPPTable          0x14
#define PPSMC_MSG_RunDcBtc                   0x36
#define PPSMC_MSG_AllowGfxOff                0x28
#define PPSMC_MSG_DisallowGfxOff             0x29
/* NOTE: Old defines GetEnabledSmuFeaturesLow=0x09/High=0x0A were WRONG.
 * 0x09 = EnableSmuFeaturesHigh, 0x0A = DisableSmuFeaturesLow.
 * Sending those "queries" was actually enabling/disabling features!
 * Correct getter is GetRunningSmuFeaturesLow=0x0C/High=0x0D. */
/* NOTE: Old defines SetPptableAddrHigh=0x0C/Low=0x0D were WRONG.
 * 0x0C/0x0D are GetRunningSmuFeaturesLow/High.
 * There is no SetPptableAddr in v14_0_2 — use TransferTable instead. */

static ULONG mp1_rreg(struct WddmLiteDevice *dev, ULONG reg)
{
    /* Use SMN indirect if available — direct MMIO doesn't reach SMU on VFIO */
    if (dev->hw.ip.nbio_base1 != 0)
        return gpu_smn_rreg(dev, dev->hw.ip.mp1_base + reg);
    ULONG offset = (dev->hw.ip.mp1_base + reg) * 4;
    return wddm_lite_read_reg32(dev, offset);
}

static void mp1_wreg(struct WddmLiteDevice *dev, ULONG reg, ULONG val)
{
    /* Use SMN indirect if available — direct MMIO doesn't reach SMU on VFIO */
    if (dev->hw.ip.nbio_base1 != 0) {
        gpu_smn_wreg(dev, dev->hw.ip.mp1_base + reg, val);
        return;
    }
    ULONG offset = (dev->hw.ip.mp1_base + reg) * 4;
    wddm_lite_write_reg32(dev, offset, val);
}

/* Direct MMIO write to MP1 register (bypasses SMN indirect).
 *
 * The SMU's internal interrupt that triggers message processing may only
 * fire on direct MMIO writes to C2PMSG_66, not on SMN bus writes.
 * The Linux driver uses WREG32_SOC15(MP1, ...) which is direct MMIO.
 * Our mp1_wreg uses SMN indirect (RSMU_INDEX/DATA) which updates the
 * register value but might not trigger the SMU's interrupt.
 *
 * Use this for C2PMSG_66 writes (the message trigger register). */
static void mp1_wreg_direct(struct WddmLiteDevice *dev, ULONG reg, ULONG val)
{
    ULONG offset = (dev->hw.ip.mp1_base + reg) * 4;
    wddm_lite_write_reg32(dev, offset, val);
}

static ULONG mp1_rreg_direct(struct WddmLiteDevice *dev, ULONG reg)
{
    ULONG offset = (dev->hw.ip.mp1_base + reg) * 4;
    return wddm_lite_read_reg32(dev, offset);
}


/* ======================================================================
 * IP Discovery: parse the binary table from VRAM
 * ====================================================================== */

/* Read a little-endian uint32 from a byte buffer */
static inline ULONG rd32(const UCHAR *buf, ULONG off)
{
    return (ULONG)buf[off]
         | ((ULONG)buf[off+1] << 8)
         | ((ULONG)buf[off+2] << 16)
         | ((ULONG)buf[off+3] << 24);
}

static inline USHORT rd16(const UCHAR *buf, ULONG off)
{
    return (USHORT)buf[off] | ((USHORT)buf[off+1] << 8);
}

/*
 * Parse a single ip_v3/v4 block from the discovery table.
 * Returns bytes consumed, or 0 on error.
 */
static ULONG parse_ip_block(const UCHAR *data, ULONG offset, ULONG data_len,
                             int ip_version, struct GpuIpBlock *out)
{
    if (offset + 8 > data_len)
        return 0;

    USHORT hw_id = rd16(data, offset);
    UCHAR instance = data[offset + 2];
    UCHAR num_bases = data[offset + 3];
    UCHAR major = data[offset + 4];
    UCHAR minor = data[offset + 5];
    UCHAR revision = data[offset + 6];

    memset(out, 0, sizeof(*out));
    out->hw_id = hw_id;
    out->instance = instance;
    out->major = major;
    out->minor = minor;
    out->revision = revision;
    out->num_base_addr = (num_bases > 6) ? 6 : num_bases;

    ULONG base_off = offset + 8;
    ULONG entry_size;

    /* v4 with 64-bit addresses not handled (GFX12 uses 32-bit) */
    for (UCHAR i = 0; i < out->num_base_addr; i++) {
        if (base_off + (i + 1) * 4 > data_len)
            break;
        out->base_addr[i] = rd32(data, base_off + i * 4);
    }
    entry_size = 8 + num_bases * 4;

    return entry_size;
}

/*
 * Extract well-known base addresses from parsed IP blocks.
 */
static void resolve_bases(struct GpuIpDiscovery *ip)
{
    for (ULONG i = 0; i < ip->num_blocks; i++) {
        struct GpuIpBlock *b = &ip->blocks[i];
        if (b->instance != 0)
            continue;

        switch (b->hw_id) {
        case HWID_MMHUB:
            if (b->num_base_addr > 0) ip->mmhub_base = b->base_addr[0];
            break;
        case HWID_GC:
            if (b->num_base_addr > 0) ip->gc_base = b->base_addr[0];
            if (b->num_base_addr > 1) ip->gc_base1 = b->base_addr[1];
            break;
        case HWID_SDMA0:
            if (b->num_base_addr > 0) ip->sdma0_base = b->base_addr[0];
            break;
        case HWID_NBIF:
            if (b->num_base_addr > 0) ip->nbio_base = b->base_addr[0];
            if (b->num_base_addr > 1) ip->nbio_base1 = b->base_addr[1];
            if (b->num_base_addr > 2) ip->nbio_base2 = b->base_addr[2];
            break;
        case HWID_OSSSYS:
            if (b->num_base_addr > 0) ip->ih_base = b->base_addr[0];
            break;
        case HWID_MP0:
            if (b->num_base_addr > 0) ip->mp0_base = b->base_addr[0];
            break;
        case HWID_MP1:
            if (b->num_base_addr > 0) ip->mp1_base = b->base_addr[0];
            if (b->num_base_addr > 1) ip->mp1_base1 = b->base_addr[1];
            break;
        }
    }
}

int gpu_ip_discovery(struct WddmLiteDevice *dev)
{
    struct GpuIpDiscovery *ip = &dev->hw.ip;
    UCHAR *buf = NULL;
    ULONGLONG map_offset;
    ULONGLONG map_size = DISCOVERY_TABLE_SIZE;
    int ret = -1;

    if (dev->hw.ip_discovery_done)
        return 0;

    memset(ip, 0, sizeof(*ip));

    /* Discovery table is at VRAM_SIZE - 64KB */
    if (dev->vram_size < DISCOVERY_TABLE_SIZE) {
        pr_err("gpu_init: VRAM too small for IP discovery (%llu)\n",
               (unsigned long long)dev->vram_size);
        return -1;
    }
    map_offset = dev->vram_size - DISCOVERY_TABLE_SIZE;

    /* Map the last 64KB of VRAM BAR into userspace */
    AMDGPU_ESCAPE_MAP_BAR_DATA map;
    memset(&map, 0, sizeof(map));
    map.Header.Command = AMDGPU_ESCAPE_MAP_BAR;
    map.Header.Size = sizeof(map);
    map.BarIndex = dev->info.VramBarIndex;
    map.Offset = map_offset;
    map.Length = map_size;

    if (wddm_lite_escape(dev, &map, sizeof(map)) != 0 ||
        map.MappedAddress == NULL) {
        pr_err("gpu_init: failed to map VRAM for IP discovery "
               "(bar=%u, offset=0x%llx, len=0x%llx)\n",
               map.BarIndex,
               (unsigned long long)map_offset,
               (unsigned long long)map_size);
        return -1;
    }

    buf = (UCHAR *)map.MappedAddress;

    /* Find the binary signature — may be after PSP header (256 bytes) */
    ULONG base = 0;
    if (map_size > PSP_HEADER_SIZE) {
        ULONG sig_at_psp = rd32(buf, PSP_HEADER_SIZE);
        if (sig_at_psp == BINARY_SIGNATURE)
            base = PSP_HEADER_SIZE;
    }

    USHORT ver_major, ver_minor;

    ULONG sig = rd32(buf, base);
    if (sig != BINARY_SIGNATURE) {
        pr_err("gpu_init: bad IP discovery signature 0x%08x "
               "(expected 0x%08x) at offset %u\n",
               sig, BINARY_SIGNATURE, base);
        goto out_unmap;
    }

    ver_major = rd16(buf, base + 4);
    ver_minor = rd16(buf, base + 6);

    pr_info("gpu_init: IP discovery binary v%u.%u\n", ver_major, ver_minor);

    /* Parse table_list[TOTAL_TABLES] — each entry is (offset:16, checksum:16, size:16, pad:16) */
    USHORT tbl_offsets[TOTAL_TABLES];
    USHORT tbl_sizes[TOTAL_TABLES];
    for (int t = 0; t < TOTAL_TABLES; t++) {
        ULONG tl_off = base + 12 + t * 8;
        tbl_offsets[t] = rd16(buf, tl_off);
        tbl_sizes[t] = rd16(buf, tl_off + 4);
    }

    /* Parse IP discovery table */
    if (tbl_offsets[TABLE_IP_DISCOVERY] > 0 && tbl_sizes[TABLE_IP_DISCOVERY] > 0) {
        ULONG abs_off = base + tbl_offsets[TABLE_IP_DISCOVERY];
        if (abs_off + 14 > map_size)
            goto out_unmap;

        ULONG ip_sig = rd32(buf, abs_off);
        USHORT ip_version = rd16(buf, abs_off + 4);
        USHORT num_dies = rd16(buf, abs_off + 12);

        pr_info("gpu_init: discovery table v%u, %u die(s), sig=0x%08x\n",
                ip_version, num_dies, ip_sig);

        for (USHORT die_idx = 0; die_idx < num_dies && die_idx < 16; die_idx++) {
            ULONG die_info_off = abs_off + 14 + die_idx * 4;
            if (die_info_off + 4 > map_size)
                break;

            USHORT die_id = rd16(buf, die_info_off);
            USHORT die_offset = rd16(buf, die_info_off + 2);

            /* die_offset is relative to binary header start */
            ULONG die_abs = base + die_offset;
            if (die_abs + 4 > map_size)
                break;

            USHORT d_num_ips = rd16(buf, die_abs + 2);
            ULONG ip_off = die_abs + 4;

            for (USHORT j = 0; j < d_num_ips; j++) {
                if (ip->num_blocks >= GPU_MAX_IP_BLOCKS)
                    break;

                struct GpuIpBlock block;
                ULONG consumed = parse_ip_block(buf, ip_off, (ULONG)map_size,
                                                ip_version, &block);
                if (consumed == 0)
                    break;

                ip->blocks[ip->num_blocks++] = block;
                ip_off += consumed;
            }

            (void)die_id;
        }
    }

    resolve_bases(ip);

    pr_info("gpu_init: discovered %u IP blocks\n", ip->num_blocks);
    pr_info("gpu_init:   MMHUB base = 0x%04x\n", ip->mmhub_base);
    pr_info("gpu_init:   GC base    = 0x%04x (base1 = 0x%04x)\n",
            ip->gc_base, ip->gc_base1);
    pr_info("gpu_init:   SDMA0 base = 0x%04x\n", ip->sdma0_base);
    pr_info("gpu_init:   NBIO base  = 0x%04x (base1=0x%04x, base2=0x%04x)\n",
            ip->nbio_base, ip->nbio_base1, ip->nbio_base2);
    pr_info("gpu_init:   IH base    = 0x%04x\n", ip->ih_base);
    pr_info("gpu_init:   MP0 base   = 0x%04x\n", ip->mp0_base);
    pr_info("gpu_init:   MP1 base   = 0x%04x (base1=0x%04x)\n",
            ip->mp1_base, ip->mp1_base1);

    /* Log all discovered blocks */
    for (ULONG i = 0; i < ip->num_blocks; i++) {
        struct GpuIpBlock *b = &ip->blocks[i];
        pr_debug("gpu_init:   [%3u] hw_id=%3u v%u.%u.%u inst=%u bases=%u",
                 i, b->hw_id, b->major, b->minor, b->revision,
                 b->instance, b->num_base_addr);
        for (UCHAR j = 0; j < b->num_base_addr && j < 3; j++)
            pr_debug(" 0x%04x", b->base_addr[j]);
        pr_debug("\n");
    }

    dev->hw.ip_discovery_done = TRUE;
    ret = 0;

out_unmap:
    /* Unmap the VRAM region */
    {
        AMDGPU_ESCAPE_MAP_BAR_DATA unmap;
        memset(&unmap, 0, sizeof(unmap));
        unmap.Header.Command = AMDGPU_ESCAPE_UNMAP_BAR;
        unmap.Header.Size = sizeof(unmap);
        unmap.MappedAddress = map.MappedAddress;
        unmap.MappingHandle = map.MappingHandle;
        wddm_lite_escape(dev, &unmap, sizeof(unmap));
    }

    return ret;
}


/* ======================================================================
 * PSP Firmware Loading
 *
 * Loads PSP SOS (Secure OS) firmware via the bootloader command
 * interface. This wakes up the PSP, which then enables the SMU,
 * which allows disabling GFXOFF, which makes GC registers accessible.
 *
 * The firmware file (psp_14_0_3_sos.bin) contains multiple components
 * (KDB, SPL, SYS_DRV, SOC_DRV, INTF_DRV, DBG_DRV, RAS_DRV, SOS)
 * packed with a v2.0 header describing their layout.
 *
 * Protocol (from tinygrad AM_PSP and Linux amdgpu psp_v14_0.c):
 *   For each component:
 *     1. Wait for bootloader ready (C2PMSG_35 bit 31)
 *     2. Copy firmware data to DMA buffer
 *     3. Write DMA bus address >> 20 to C2PMSG_36
 *     4. Write bootloader command to C2PMSG_35
 *   After all components:
 *     Poll C2PMSG_81 for SOS alive
 *
 * Register layout (mp_14_0_2_offset.h, BASE_IDX=0 from MP0 base):
 *   regMPASP_SMN_C2PMSG_35 = 0x0063  (BL command / status)
 *   regMPASP_SMN_C2PMSG_36 = 0x0064  (MSG1 address >> 20)
 *   regMPASP_SMN_C2PMSG_81 = 0x0091  (SOS alive indicator)
 * ====================================================================== */

/* PSP firmware type enum (matches struct_psp_fw_bin_desc.fw_type) */
#define PSP_FW_TYPE_UNKOWN      0
#define PSP_FW_TYPE_SOS         1
#define PSP_FW_TYPE_SYS_DRV     2
#define PSP_FW_TYPE_KDB         3
#define PSP_FW_TYPE_TOC         4
#define PSP_FW_TYPE_SPL         5
#define PSP_FW_TYPE_RL          6
#define PSP_FW_TYPE_SOC_DRV     7
#define PSP_FW_TYPE_INTF_DRV    8
#define PSP_FW_TYPE_DBG_DRV     9
#define PSP_FW_TYPE_RAS_DRV     10
#define PSP_FW_TYPE_IPKEYMGR    11

/* Forward declarations — defined later but needed by PSP code */
void gpu_hdp_flush(struct WddmLiteDevice *dev);
ULONG gpu_smn_rreg(struct WddmLiteDevice *dev, ULONG reg);
void gpu_smn_wreg(struct WddmLiteDevice *dev, ULONG reg, ULONG val);

/* PSP bootloader command IDs */
#define PSP_BL_LOAD_KEY_DATABASE        0x80000
#define PSP_BL_LOAD_TOS_SPL_TABLE       0x10000000
#define PSP_BL_LOAD_SYSDRV             0x10000
#define PSP_BL_LOAD_SOCDRV             0xB0000
#define PSP_BL_LOAD_INTFDRV            0xD0000
#define PSP_BL_LOAD_DBGDRV             0xC0000
#define PSP_BL_LOAD_RASDRV             0xE0000
#define PSP_BL_LOAD_SOSDRV             0x20000

/* Bootloader command for fw_type → BL command mapping */
static ULONG psp_fw_type_to_bl_cmd(ULONG fw_type)
{
    switch (fw_type) {
    case PSP_FW_TYPE_KDB:      return PSP_BL_LOAD_KEY_DATABASE;
    case PSP_FW_TYPE_SPL:      return PSP_BL_LOAD_TOS_SPL_TABLE;
    case PSP_FW_TYPE_SYS_DRV:  return PSP_BL_LOAD_SYSDRV;
    case PSP_FW_TYPE_SOC_DRV:  return PSP_BL_LOAD_SOCDRV;
    case PSP_FW_TYPE_INTF_DRV: return PSP_BL_LOAD_INTFDRV;
    case PSP_FW_TYPE_DBG_DRV:  return PSP_BL_LOAD_DBGDRV;
    case PSP_FW_TYPE_RAS_DRV:  return PSP_BL_LOAD_RASDRV;
    case PSP_FW_TYPE_SOS:      return PSP_BL_LOAD_SOSDRV;
    default: return 0;
    }
}

static const char *psp_fw_type_name(ULONG fw_type)
{
    switch (fw_type) {
    case PSP_FW_TYPE_SOS:      return "SOS";
    case PSP_FW_TYPE_SYS_DRV:  return "SYS_DRV";
    case PSP_FW_TYPE_KDB:      return "KDB";
    case PSP_FW_TYPE_TOC:      return "TOC";
    case PSP_FW_TYPE_SPL:      return "SPL";
    case PSP_FW_TYPE_RL:       return "RL";
    case PSP_FW_TYPE_SOC_DRV:  return "SOC_DRV";
    case PSP_FW_TYPE_INTF_DRV: return "INTF_DRV";
    case PSP_FW_TYPE_DBG_DRV:  return "DBG_DRV";
    case PSP_FW_TYPE_RAS_DRV:  return "RAS_DRV";
    case PSP_FW_TYPE_IPKEYMGR: return "IPKEYMGR";
    default: return "UNKNOWN";
    }
}

/* SOS component loading order (same as tinygrad AM_PSP.init_hw) */
static const ULONG sos_load_order[] = {
    PSP_FW_TYPE_KDB,
    PSP_FW_TYPE_SPL,      /* For MP0 >= v14, use SPL instead of 2nd KDB */
    PSP_FW_TYPE_SYS_DRV,
    PSP_FW_TYPE_SOC_DRV,
    PSP_FW_TYPE_INTF_DRV,
    PSP_FW_TYPE_DBG_DRV,
    PSP_FW_TYPE_RAS_DRV,
    PSP_FW_TYPE_SOS,
};
#define SOS_LOAD_ORDER_COUNT  (sizeof(sos_load_order) / sizeof(sos_load_order[0]))

/* Firmware binary descriptor (16 bytes each, from firmware file) */
struct PspFwBinDesc {
    ULONG fw_type;
    ULONG fw_version;
    ULONG offset_bytes;
    ULONG size_bytes;
};

/* Wait for PSP bootloader ready (C2PMSG_35 bit 31 set) */
static int psp_wait_for_bootloader(struct WddmLiteDevice *dev)
{
    for (int i = 0; i < 10000; i++) {  /* 10 seconds max */
        ULONG val = mp0_rreg(dev, regMPASP_SMN_C2PMSG_35);
        if (val & 0x80000000)
            return 0;
        Sleep(1);
    }
    pr_err("gpu_psp: bootloader not ready (C2PMSG_35 timeout)\n");
    return -1;
}

/* regMPASP_SMN_C2PMSG_36 offset (for msg1 address) */
#define regMPASP_SMN_C2PMSG_36  0x0064

/* Load one SOS component via bootloader command */
static int psp_bootloader_load_component(struct WddmLiteDevice *dev,
                                          PVOID dma_cpu, ULONGLONG dma_bus,
                                          const UCHAR *fw_data, ULONG fw_size,
                                          ULONG fw_type)
{
    ULONG bl_cmd = psp_fw_type_to_bl_cmd(fw_type);
    if (bl_cmd == 0) {
        pr_info("gpu_psp: skipping component type %u (%s) — no BL command\n",
                fw_type, psp_fw_type_name(fw_type));
        return 0;
    }

    pr_info("gpu_psp: loading %s (%u bytes) via BL cmd 0x%x...\n",
            psp_fw_type_name(fw_type), fw_size, bl_cmd);

    /* Wait for bootloader ready */
    if (psp_wait_for_bootloader(dev) != 0)
        return -1;

    /* Copy firmware data to VRAM buffer */
    memcpy(dma_cpu, fw_data, fw_size);

    /* Ensure writes are flushed through write-combine buffers */
    MemoryBarrier();

    /* Flush HDP to ensure CPU writes to VRAM are visible to PSP DMA */
    gpu_hdp_flush(dev);

    /* Write VRAM MC address >> 20 to C2PMSG_36 */
    ULONG addr_hi = (ULONG)(dma_bus >> 20);
    mp0_rreg(dev, regMPASP_SMN_C2PMSG_36);  /* Dummy read for sync */
    wddm_lite_write_reg32(dev, (dev->hw.ip.mp0_base + regMPASP_SMN_C2PMSG_36) * 4,
                           addr_hi);

    /* Write bootloader command to C2PMSG_35 */
    wddm_lite_write_reg32(dev, (dev->hw.ip.mp0_base + regMPASP_SMN_C2PMSG_35) * 4,
                           bl_cmd);

    /* For SOS component, don't wait — it takes longer and we poll C2PMSG_81 */
    if (fw_type == PSP_FW_TYPE_SOS)
        return 0;

    /* Wait for bootloader to finish loading this component */
    return psp_wait_for_bootloader(dev);
}

int gpu_psp_load_sos(struct WddmLiteDevice *dev, const char *fw_path)
{
    UCHAR *fw_buf = NULL;
    ULONG fw_len = 0;
    int ret = -1;

    if (!dev->hw.ip_discovery_done) {
        pr_err("gpu_psp: IP discovery must be run first\n");
        return -1;
    }

    if (dev->hw.ip.mp0_base == 0) {
        pr_err("gpu_psp: MP0 base not found\n");
        return -1;
    }

    /* Read PSP status registers (always, for diagnostics) */
    ULONG c2pmsg_81 = mp0_rreg(dev, regMPASP_SMN_C2PMSG_81);
    ULONG c2pmsg_35 = mp0_rreg(dev, regMPASP_SMN_C2PMSG_35);

    /* Read additional MP0 registers for diagnostics */
    #define regMPASP_SMN_C2PMSG_0   0x0040
    #define regMPASP_SMN_C2PMSG_33  0x0061
    #define regMPASP_SMN_C2PMSG_64  0x0080
    ULONG c2pmsg_0  = mp0_rreg(dev, regMPASP_SMN_C2PMSG_0);
    ULONG c2pmsg_33 = mp0_rreg(dev, regMPASP_SMN_C2PMSG_33);
    ULONG c2pmsg_64 = mp0_rreg(dev, regMPASP_SMN_C2PMSG_64);

    /* Read VRAM MC base for diagnostics */
    ULONG fb_base_reg_diag = mmhub_rreg(dev, regMMMC_VM_FB_LOCATION_BASE);
    ULONGLONG vram_mc_base_diag = (ULONGLONG)fb_base_reg_diag << 24;

    pr_info("gpu_psp: === PSP Diagnostics ===\n");
    pr_info("gpu_psp: MP0 base = 0x%x\n", dev->hw.ip.mp0_base);
    pr_info("gpu_psp: C2PMSG_81 = 0x%08x (SOS alive = %s)\n",
            c2pmsg_81, c2pmsg_81 ? "YES" : "NO");
    pr_info("gpu_psp: C2PMSG_35 = 0x%08x (BL ready = %s)\n",
            c2pmsg_35, (c2pmsg_35 & 0x80000000) ? "YES" : "NO");
    pr_info("gpu_psp: C2PMSG_0  = 0x%08x\n", c2pmsg_0);
    pr_info("gpu_psp: C2PMSG_33 = 0x%08x\n", c2pmsg_33);
    pr_info("gpu_psp: C2PMSG_64 = 0x%08x\n", c2pmsg_64);
    pr_info("gpu_psp: VRAM MC base = 0x%llx (FB_LOCATION_BASE = 0x%08x)\n",
            (unsigned long long)vram_mc_base_diag, fb_base_reg_diag);
    pr_info("gpu_psp: ========================\n");

    /* Check if SOS is already alive (VBIOS POST may have loaded it) */
    if (c2pmsg_81 != 0) {
        pr_info("gpu_psp: SOS already alive — skipping firmware load\n");
        dev->hw.psp_sos_alive = TRUE;
        return 0;
    }

    /* Check if diagnostic-only mode is set */
    {
        char diag_only[32] = {};
        GetEnvironmentVariableA("HSAKMT_PSP_DIAG_ONLY", diag_only, sizeof(diag_only));
        if (diag_only[0] == '1') {
            pr_info("gpu_psp: diagnostic-only mode — skipping firmware load\n");
            return -1;
        }
    }

    /* Check bootloader readiness */
    if (!(c2pmsg_35 & 0x80000000)) {
        pr_err("gpu_psp: bootloader not ready — cannot load firmware\n");
        return -1;
    }

    /* Read firmware file */
    {
        FILE *fp = fopen(fw_path, "rb");
        if (!fp) {
            pr_err("gpu_psp: cannot open firmware file: %s\n", fw_path);
            return -1;
        }
        fseek(fp, 0, SEEK_END);
        fw_len = (ULONG)ftell(fp);
        fseek(fp, 0, SEEK_SET);

        fw_buf = (UCHAR *)malloc(fw_len);
        if (!fw_buf) {
            pr_err("gpu_psp: malloc(%u) failed\n", fw_len);
            fclose(fp);
            return -1;
        }
        if (fread(fw_buf, 1, fw_len, fp) != fw_len) {
            pr_err("gpu_psp: read error on %s\n", fw_path);
            fclose(fp);
            free(fw_buf);
            return -1;
        }
        fclose(fp);
        pr_info("gpu_psp: loaded firmware file: %s (%u bytes)\n", fw_path, fw_len);
    }

    USHORT hdr_major, hdr_minor;
    ULONG ucode_offset, psp_fw_count;
    ULONGLONG vram_offset, vram_map_size;
    ULONG fb_base_reg;
    ULONGLONG vram_mc_base, fw_mc_addr;

    /* Parse common firmware header */
    if (fw_len < 36) {
        pr_err("gpu_psp: firmware too small (%u bytes)\n", fw_len);
        goto fail;
    }

    hdr_major = rd16(fw_buf, 8);
    hdr_minor = rd16(fw_buf, 10);
    ucode_offset = rd32(fw_buf, 24);
    psp_fw_count = rd32(fw_buf, 32);

    pr_info("gpu_psp: header v%u.%u, ucode_offset=0x%x, %u components\n",
            hdr_major, hdr_minor, ucode_offset, psp_fw_count);

    if (hdr_major != 2 || psp_fw_count > 32 || psp_fw_count == 0) {
        pr_err("gpu_psp: unexpected header version %u.%u or count %u\n",
               hdr_major, hdr_minor, psp_fw_count);
        goto fail;
    }

    /* Parse firmware component descriptors (at offset 36, 16 bytes each) */
    struct PspFwBinDesc descs[32];
    for (ULONG i = 0; i < psp_fw_count; i++) {
        ULONG off = 36 + i * 16;
        if (off + 16 > fw_len) break;
        descs[i].fw_type = rd32(fw_buf, off);
        descs[i].fw_version = rd32(fw_buf, off + 4);
        descs[i].offset_bytes = rd32(fw_buf, off + 8);
        descs[i].size_bytes = rd32(fw_buf, off + 12);

        pr_info("gpu_psp:   [%u] type=%u (%s) size=%u offset=0x%x\n",
                i, descs[i].fw_type, psp_fw_type_name(descs[i].fw_type),
                descs[i].size_bytes, descs[i].offset_bytes);
    }

    /*
     * Allocate firmware transfer buffer in VRAM.
     * The PSP bootloader DMA engine reads from VRAM MC addresses,
     * NOT system memory bus addresses. We map a 1MB region of VRAM
     * at offset 4MB (avoid offset 0 which may overlap with VBIOS
     * framebuffer or other early-boot data).
     *
     * MC address = vram_start + vram_offset
     * We read vram_start from MMHUB FB_LOCATION_BASE register.
     */
    vram_offset = 4 * 1024 * 1024;  /* 4MB into VRAM */
    vram_map_size = 1024 * 1024;  /* 1MB */

    /* Read VRAM MC base from MMHUB (always accessible, even before GMC init) */
    fb_base_reg = mmhub_rreg(dev, regMMMC_VM_FB_LOCATION_BASE);
    vram_mc_base = (ULONGLONG)fb_base_reg << 24;
    fw_mc_addr = vram_mc_base + vram_offset;

    pr_info("gpu_psp: VRAM MC base = 0x%llx, fw buffer MC addr = 0x%llx\n",
            (unsigned long long)vram_mc_base,
            (unsigned long long)fw_mc_addr);

    /* Verify MC address alignment (needs >> 20, so must be 1MB aligned) */
    if (fw_mc_addr & 0xFFFFF) {
        pr_warn("gpu_psp: MC address 0x%llx not 1MB aligned\n",
                (unsigned long long)fw_mc_addr);
    }

    /* Map VRAM into userspace */
    AMDGPU_ESCAPE_MAP_VRAM_DATA vram_map;
    memset(&vram_map, 0, sizeof(vram_map));
    vram_map.Header.Command = AMDGPU_ESCAPE_MAP_VRAM;
    vram_map.Header.Size = sizeof(vram_map);
    vram_map.Offset = vram_offset;
    vram_map.Length = vram_map_size;

    if (wddm_lite_escape(dev, &vram_map, sizeof(vram_map)) != 0 ||
        vram_map.MappedAddress == NULL) {
        pr_err("gpu_psp: VRAM map failed (offset=0x%llx, len=0x%llx)\n",
               (unsigned long long)vram_offset,
               (unsigned long long)vram_map_size);
        goto fail;
    }

    pr_info("gpu_psp: VRAM mapped: cpu=%p, offset=0x%llx\n",
            vram_map.MappedAddress, (unsigned long long)vram_offset);

    /* Load SOS components in order */
    for (ULONG order = 0; order < SOS_LOAD_ORDER_COUNT; order++) {
        ULONG target_type = sos_load_order[order];

        /* Find this component in the firmware descriptors */
        BOOLEAN found = FALSE;
        for (ULONG i = 0; i < psp_fw_count; i++) {
            if (descs[i].fw_type != target_type)
                continue;
            if (descs[i].size_bytes == 0) {
                pr_info("gpu_psp: skipping %s (0 bytes)\n",
                        psp_fw_type_name(target_type));
                found = TRUE;
                break;
            }

            /* Compute absolute offset in firmware file */
            ULONG abs_off = descs[i].offset_bytes + ucode_offset;
            if (abs_off + descs[i].size_bytes > fw_len) {
                pr_err("gpu_psp: component %s overflows firmware "
                       "(off=0x%x + size=0x%x > 0x%x)\n",
                       psp_fw_type_name(target_type),
                       abs_off, descs[i].size_bytes, fw_len);
                goto fail_dma;
            }

            /* Check size fits in VRAM buffer */
            if (descs[i].size_bytes > vram_map_size) {
                pr_err("gpu_psp: component %s too large (%u > %llu)\n",
                       psp_fw_type_name(target_type),
                       descs[i].size_bytes,
                       (unsigned long long)vram_map_size);
                goto fail_dma;
            }

            int load_ret = psp_bootloader_load_component(
                dev, vram_map.MappedAddress, fw_mc_addr,
                fw_buf + abs_off, descs[i].size_bytes,
                target_type);

            if (load_ret != 0) {
                pr_err("gpu_psp: failed to load %s\n",
                       psp_fw_type_name(target_type));
                goto fail_dma;
            }

            found = TRUE;
            break;
        }

        if (!found) {
            pr_info("gpu_psp: component %s not found in firmware — skipping\n",
                    psp_fw_type_name(target_type));
        }
    }

    /* Wait for SOS to come alive (poll C2PMSG_81) */
    pr_info("gpu_psp: waiting for SOS to start...\n");
    for (int i = 0; i < 10000; i++) {  /* 10 seconds */
        if (psp_is_sos_alive(dev)) {
            ULONG sos_val = mp0_rreg(dev, regMPASP_SMN_C2PMSG_81);
            pr_info("gpu_psp: SOS alive! (C2PMSG_81 = 0x%08x)\n", sos_val);
            dev->hw.psp_sos_alive = TRUE;
            ret = 0;
            goto done_dma;
        }
        Sleep(1);
    }
    pr_err("gpu_psp: SOS failed to start (timeout)\n");

done_dma:
fail_dma:
    /* Unmap VRAM buffer (UNMAP_BAR expects MAP_BAR_DATA struct) */
    {
        AMDGPU_ESCAPE_MAP_BAR_DATA unmap;
        memset(&unmap, 0, sizeof(unmap));
        unmap.Header.Command = AMDGPU_ESCAPE_UNMAP_BAR;
        unmap.Header.Size = sizeof(unmap);
        unmap.MappedAddress = vram_map.MappedAddress;
        unmap.MappingHandle = vram_map.MappingHandle;
        wddm_lite_escape(dev, &unmap, sizeof(unmap));
    }

fail:
    free(fw_buf);
    return ret;
}


/* ======================================================================
 * PSP GPCOM Ring — Load IP Firmware
 *
 * After SOS is alive, we create the PSP GPCOM ring and submit
 * LOAD_IP_FW commands to load all GPU firmware (SMU, SDMA, PFP, ME,
 * MEC, RLC), then trigger AUTOLOAD_RLC to bring the GC block online.
 *
 * Ring protocol (from Linux amdgpu psp_v14_0.c):
 *   1. Destroy any existing ring (write 3 to C2PMSG_64)
 *   2. Wait for TOS ready (C2PMSG_64 bit 31 set)
 *   3. Write ring address/size to C2PMSG_69/70/71
 *   4. Create ring (write PSP_RING_TYPE_KM<<16 to C2PMSG_64)
 *   5. Submit 64-byte frames pointing to 1024-byte command buffers
 *   6. Advance write pointer via C2PMSG_67
 *   7. Wait for fence in VRAM
 *
 * Register offsets (MP0 base, DWORD offsets):
 *   C2PMSG_64 = 0x0080 (ring cmd/status)
 *   C2PMSG_67 = 0x0083 (ring write pointer)
 *   C2PMSG_69 = 0x0085 (ring addr low)
 *   C2PMSG_70 = 0x0086 (ring addr high)
 *   C2PMSG_71 = 0x0087 (ring size)
 * ====================================================================== */

/* PSP ring register offsets (DWORD, relative to MP0 base) */
#define regMPASP_SMN_C2PMSG_67  0x0083
#define regMPASP_SMN_C2PMSG_69  0x0085
#define regMPASP_SMN_C2PMSG_70  0x0086
#define regMPASP_SMN_C2PMSG_71  0x0087

/* PSP ring constants */
#define PSP_RING_TYPE_KM        2
#define PSP_RING_SIZE           0x10000  /* 64KB */
#define GFX_CTRL_CMD_ID_DESTROY_RINGS  0x00030000  /* 3 << 16 */

/* PSP ring commands (from psp_gfx_if.h) */
#define GFX_CMD_ID_SETUP_TMR    0x00000005
#define GFX_CMD_ID_LOAD_IP_FW   0x00000006
#define GFX_CMD_ID_LOAD_TOC     0x00000020
#define GFX_CMD_ID_AUTOLOAD_RLC 0x00000021

/* Response flags */
#define GFX_FLAG_RESPONSE       0x80000000
#define GFX_CMD_RESPONSE_MASK   0x8000FFFF

/*
 * PSP GFX firmware types (enum psp_gfx_fw_type from psp_gfx_if.h).
 * NOTE: These are NOT the same as tinygrad's UCODEType enum!
 */
#define GFX_FW_TYPE_CP_ME               1
#define GFX_FW_TYPE_CP_PFP              2
#define GFX_FW_TYPE_CP_MEC              4
#define GFX_FW_TYPE_RLC_G               8
#define GFX_FW_TYPE_SDMA0               9
#define GFX_FW_TYPE_SMU                 18
#define GFX_FW_TYPE_IMU_I               68
#define GFX_FW_TYPE_IMU_D               69
#define GFX_FW_TYPE_SDMA_UCODE_TH0     71
#define GFX_FW_TYPE_SDMA_UCODE_TH1     72
#define GFX_FW_TYPE_RS64_MES            76
#define GFX_FW_TYPE_RS64_MES_STACK      77
#define GFX_FW_TYPE_RS64_KIQ            78
#define GFX_FW_TYPE_RS64_KIQ_STACK      79
#define GFX_FW_TYPE_CP_MES_KIQ          81
#define GFX_FW_TYPE_MES_KIQ_STACK       82
#define GFX_FW_TYPE_RS64_PFP            87
#define GFX_FW_TYPE_RS64_ME             88
#define GFX_FW_TYPE_RS64_MEC            89
#define GFX_FW_TYPE_RS64_PFP_P0_STACK   90
#define GFX_FW_TYPE_RS64_PFP_P1_STACK   91
#define GFX_FW_TYPE_RS64_ME_P0_STACK    92
#define GFX_FW_TYPE_RS64_ME_P1_STACK    93
#define GFX_FW_TYPE_RS64_MEC_P0_STACK   94
#define GFX_FW_TYPE_RS64_MEC_P1_STACK   95
#define GFX_FW_TYPE_RS64_MEC_P2_STACK   96
#define GFX_FW_TYPE_RS64_MEC_P3_STACK   97

/* PSP command buffer version */
#define PSP_GFX_CMD_BUF_VERSION 1

/*
 * PSP GPCOM ring buffer frame (64 bytes = 16 DWORDs).
 * Each entry points to a separate 1024-byte command buffer in VRAM.
 * Completion is signaled by writing fence_value to fence_addr.
 *
 * Reference: psp_gfx_if.h struct psp_gfx_rb_frame
 */
#pragma pack(push, 1)
struct PspGfxRbFrame {
    ULONG   cmd_buf_addr_lo;    /* +0   MC address of cmd buffer (low) */
    ULONG   cmd_buf_addr_hi;    /* +4   MC address of cmd buffer (high) */
    ULONG   cmd_buf_size;       /* +8   command buffer size in bytes */
    ULONG   fence_addr_lo;      /* +12  MC address of fence (low) */
    ULONG   fence_addr_hi;      /* +16  MC address of fence (high) */
    ULONG   fence_value;        /* +20  value to write on completion */
    ULONG   sid_lo;             /* +24  reserved */
    ULONG   sid_hi;             /* +28  reserved */
    UCHAR   vmid;               /* +32  VMID */
    UCHAR   frame_type;         /* +33  0=normal, 1=destroy context */
    UCHAR   reserved1[2];       /* +34  */
    ULONG   reserved2[7];       /* +36  */
};
#pragma pack(pop)

/* Ring frame size: 64 bytes = 16 DWORDs */
#define PSP_RING_FRAME_DWORDS   (sizeof(struct PspGfxRbFrame) / 4)

/*
 * PSP GFX command buffer (1024 bytes).
 * Written to VRAM, ring frame points to it.
 * Reference: psp_gfx_if.h struct psp_gfx_cmd_resp
 */
#pragma pack(push, 1)
struct PspGfxCmdLoadIpFw {
    ULONG   fw_phy_addr_lo;     /* MC address of FW (low) */
    ULONG   fw_phy_addr_hi;     /* MC address of FW (high) */
    ULONG   fw_size;            /* FW buffer size in bytes */
    ULONG   fw_type;            /* psp_gfx_fw_type enum */
};

struct PspGfxCmdLoadToc {
    ULONG   toc_phy_addr_lo;    /* MC address of TOC (low) */
    ULONG   toc_phy_addr_hi;    /* MC address of TOC (high) */
    ULONG   toc_size;           /* TOC size in bytes */
};

struct PspGfxCmdSetupTmr {
    ULONG   buf_phy_addr_lo;    /* MC address of TMR buffer (low) */
    ULONG   buf_phy_addr_hi;    /* MC address of TMR buffer (high) */
    ULONG   buf_size;           /* TMR buffer size in bytes */
    ULONG   tmr_flags;          /* bit0: sriov, bit1: virt_phy_addr */
    ULONG   system_phy_addr_lo; /* System physical address (low) */
    ULONG   system_phy_addr_hi; /* System physical address (high) */
};

struct PspGfxCmdResp {
    ULONG   buf_size;           /* +0   total size (1024) */
    ULONG   buf_version;        /* +4   version (1) */
    ULONG   cmd_id;             /* +8   command ID */
    ULONG   resp_buf_addr_lo;   /* +12  0 for GPCOM */
    ULONG   resp_buf_addr_hi;   /* +16  0 for GPCOM */
    ULONG   resp_offset;        /* +20  0 for GPCOM */
    ULONG   resp_buf_size;      /* +24  0 for GPCOM */
    /* +28: command union (largest member determines padding) */
    union {
        struct PspGfxCmdLoadIpFw load_ip_fw;
        struct PspGfxCmdLoadToc  load_toc;
        struct PspGfxCmdSetupTmr setup_tmr;
    };
    UCHAR   cmd_padding[760];   /* pad union+padding to 784 bytes → offset 812 */
    UCHAR   reserved_1[52];     /* +812 */
    ULONG   resp_status;        /* +864 response status (0 = success) */
    UCHAR   resp_reserved[92];  /* +868 */
    UCHAR   reserved_2[64];     /* +960 */
    /* Total: 1024 bytes */
};
#pragma pack(pop)

/* Firmware headers */
struct CommonFirmwareHeader {
    ULONG   size_bytes;
    ULONG   header_size_bytes;
    USHORT  header_version_major;
    USHORT  header_version_minor;
    USHORT  ip_version_major;
    USHORT  ip_version_minor;
    ULONG   ucode_version;
    ULONG   ucode_size_bytes;
    ULONG   ucode_array_offset_bytes;
    LONG    crc32;
};

/*
 * GFX firmware header v2.0 — used by RS64 firmware (PFP, ME, MEC).
 * Contains both instruction (ucode) and data (stack) sections.
 *
 * Layout matches Linux struct gfx_firmware_header_v2_0:
 *   0x00: common_firmware_header (32 bytes)
 *   0x20: ucode_feature_version (4 bytes)
 *   0x24: ucode_size_bytes      (code section size)
 *   0x28: ucode_offset_bytes    (code section offset from file start)
 *   0x2C: data_size_bytes       (data/stack section size)
 *   0x30: data_offset_bytes     (data/stack offset from file start)
 *   0x34: ucode_start_addr_lo
 *   0x38: ucode_start_addr_hi
 *   Total: 60 bytes (0x3C)
 */
struct GfxFirmwareHeaderV2 {
    /* common_firmware_header (0x00-0x1F, 32 bytes) */
    ULONG   size_bytes;                 /* +0x00 */
    ULONG   header_size_bytes;          /* +0x04 */
    USHORT  header_version_major;       /* +0x08 */
    USHORT  header_version_minor;       /* +0x0A */
    USHORT  ip_version_major;           /* +0x0C */
    USHORT  ip_version_minor;           /* +0x0E */
    ULONG   ucode_version;             /* +0x10 */
    ULONG   common_ucode_size_bytes;   /* +0x14 (common header field) */
    ULONG   ucode_array_offset_bytes;  /* +0x18 */
    LONG    crc32;                      /* +0x1C */
    /* v2.0 extension (0x20-0x3B, 28 bytes) */
    ULONG   ucode_feature_version;     /* +0x20 */
    ULONG   ucode_size_bytes;          /* +0x24 code section size */
    ULONG   ucode_offset_bytes;        /* +0x28 code section offset */
    ULONG   data_size_bytes;           /* +0x2C data/stack section size */
    ULONG   data_offset_bytes;         /* +0x30 data/stack section offset */
    ULONG   ucode_start_addr_lo;       /* +0x34 */
    ULONG   ucode_start_addr_hi;       /* +0x38 */
};

/*
 * SDMA firmware header v3.0.
 * Layout:
 *   0x00: common_firmware_header (32 bytes)
 *   0x20: ucode_feature_version
 *   0x24: ucode_offset_bytes
 *   0x28: ucode_size_bytes
 */
struct SdmaFirmwareHeaderV3 {
    ULONG   size_bytes;
    ULONG   header_size_bytes;
    USHORT  header_version_major;
    USHORT  header_version_minor;
    USHORT  ip_version_major;
    USHORT  ip_version_minor;
    ULONG   ucode_version;
    ULONG   common_ucode_size_bytes;
    ULONG   ucode_array_offset_bytes;
    LONG    crc32;
    /* v3.0 extension */
    ULONG   ucode_feature_version;
    ULONG   ucode_offset_bytes;
    ULONG   ucode_size_bytes;
};

/* Wait for C2PMSG_64 to indicate ready: (val & mask) == expected */
static int psp_ring_wait(struct WddmLiteDevice *dev, ULONG expected,
                          ULONG mask, int timeout_ms)
{
    for (int i = 0; i < timeout_ms; i++) {
        ULONG val = mp0_rreg(dev, regMPASP_SMN_C2PMSG_64);
        if ((val & mask) == expected)
            return 0;
        Sleep(1);
    }
    ULONG final_val = mp0_rreg(dev, regMPASP_SMN_C2PMSG_64);
    pr_err("gpu_psp_ring: timeout waiting for C2PMSG_64 "
           "(expect 0x%08x mask 0x%08x, got 0x%08x)\n",
           expected, mask, final_val);
    return -1;
}

/* ---- PSP Ring Context ---- */

/*
 * Holds the VRAM mappings and state for the PSP GPCOM ring.
 * Created once, used for all firmware loads, then destroyed.
 *
 * VRAM layout (all offsets from VRAM start):
 *   5MB:     Ring buffer (64KB)
 *   6MB:     FW staging area (2MB — largest FW is ~300KB)
 *   8MB:     Command buffer (4KB)
 *   8MB+4K:  Fence (4KB)
 */
struct PspRingContext {
    struct WddmLiteDevice *dev;

    /* MC addresses */
    ULONGLONG vram_mc_base;
    ULONGLONG ring_mc_addr;
    ULONGLONG fw_mc_addr;
    ULONGLONG cmd_mc_addr;
    ULONGLONG fence_mc_addr;

    /* CPU mappings */
    PVOID ring_cpu;
    PVOID fw_cpu;
    PVOID cmd_cpu;
    PVOID fence_cpu;

    /* Mapping handles for cleanup */
    PVOID ring_handle;
    PVOID fw_handle;
    PVOID cmd_handle;
    PVOID fence_handle;

    /* Ring state */
    ULONG wptr;         /* current write pointer (in DWORDs) */
    ULONG fence_seq;    /* monotonically increasing fence value */
    BOOLEAN valid;
};

static void psp_ring_unmap(struct WddmLiteDevice *dev,
                            PVOID cpu_addr, PVOID handle)
{
    if (cpu_addr) {
        AMDGPU_ESCAPE_MAP_BAR_DATA unmap = {};
        unmap.Header.Command = AMDGPU_ESCAPE_UNMAP_BAR;
        unmap.Header.Size = sizeof(unmap);
        unmap.MappedAddress = cpu_addr;
        unmap.MappingHandle = handle;
        wddm_lite_escape(dev, &unmap, sizeof(unmap));
    }
}

static PVOID psp_ring_map_vram(struct WddmLiteDevice *dev,
                                ULONGLONG offset, ULONG length,
                                PVOID *out_handle)
{
    AMDGPU_ESCAPE_MAP_VRAM_DATA map = {};
    map.Header.Command = AMDGPU_ESCAPE_MAP_VRAM;
    map.Header.Size = sizeof(map);
    map.Offset = offset;
    map.Length = length;

    if (wddm_lite_escape(dev, &map, sizeof(map)) != 0 ||
        map.MappedAddress == NULL) {
        return NULL;
    }
    *out_handle = map.MappingHandle;
    return map.MappedAddress;
}

static int psp_ring_init(struct PspRingContext *ctx,
                          struct WddmLiteDevice *dev)
{
    memset(ctx, 0, sizeof(*ctx));
    ctx->dev = dev;
    ctx->fence_seq = 1;

    ULONG fb_base_reg = mmhub_rreg(dev, regMMMC_VM_FB_LOCATION_BASE);
    ULONG fb_top_reg  = mmhub_rreg(dev, regMMMC_VM_FB_LOCATION_TOP);
    ctx->vram_mc_base = (ULONGLONG)fb_base_reg << 24;
    ULONGLONG vram_size = (((ULONGLONG)fb_top_reg << 24) + 0xFFFFFF) - ctx->vram_mc_base + 1;

    /* Place PSP buffers near END of VRAM to avoid boot_time_tmr.
     * The TMR (Trusted Memory Region) occupies the first ~20MB of VRAM.
     * Our old offsets (5-8MB) were INSIDE the TMR, causing the PSP to
     * load stale firmware from TMR instead of our staging data.
     * amdgpu allocates near VRAM end via kernel BO (0x87D6AF8000 etc.).
     * NOTE: VRAM-end buffers don't work (PSP can't access them).
     * Reverting to VRAM-start. TMR overlap theory is disproven. */
    ULONGLONG ring_off  = 5 * 1024 * 1024;
    ULONGLONG fw_off    = 6 * 1024 * 1024;
    ULONGLONG cmd_off   = 8 * 1024 * 1024;
    ULONGLONG fence_off = 8 * 1024 * 1024 + 4096;

    ctx->ring_mc_addr  = ctx->vram_mc_base + ring_off;
    ctx->fw_mc_addr    = ctx->vram_mc_base + fw_off;
    ctx->cmd_mc_addr   = ctx->vram_mc_base + cmd_off;
    ctx->fence_mc_addr = ctx->vram_mc_base + fence_off;

    pr_info("psp_ring: VRAM MC base=0x%llx\n",
            (unsigned long long)ctx->vram_mc_base);
    pr_info("psp_ring: ring=0x%llx fw=0x%llx cmd=0x%llx fence=0x%llx\n",
            (unsigned long long)ctx->ring_mc_addr,
            (unsigned long long)ctx->fw_mc_addr,
            (unsigned long long)ctx->cmd_mc_addr,
            (unsigned long long)ctx->fence_mc_addr);

    /* Map VRAM regions */
    ctx->ring_cpu = psp_ring_map_vram(dev, ring_off, PSP_RING_SIZE,
                                       &ctx->ring_handle);
    if (!ctx->ring_cpu) {
        pr_err("psp_ring: ring VRAM map failed\n");
        return -1;
    }

    ctx->fw_cpu = psp_ring_map_vram(dev, fw_off, 2 * 1024 * 1024,
                                     &ctx->fw_handle);
    if (!ctx->fw_cpu) {
        pr_err("psp_ring: fw staging VRAM map failed\n");
        goto fail;
    }

    ctx->cmd_cpu = psp_ring_map_vram(dev, cmd_off, 4096, &ctx->cmd_handle);
    if (!ctx->cmd_cpu) {
        pr_err("psp_ring: cmd buffer VRAM map failed\n");
        goto fail;
    }

    ctx->fence_cpu = psp_ring_map_vram(dev, fence_off, 4096,
                                        &ctx->fence_handle);
    if (!ctx->fence_cpu) {
        pr_err("psp_ring: fence VRAM map failed\n");
        goto fail;
    }

    /* Zero all buffers at new VRAM locations (end of VRAM may have stale data) */
    memset(ctx->ring_cpu, 0, PSP_RING_SIZE);
    memset(ctx->cmd_cpu, 0, 4096);
    memset(ctx->fence_cpu, 0, 4096);
    MemoryBarrier();

    /* Verify fence is actually zero */
    {
        ULONG fence_check = *(volatile ULONG *)ctx->fence_cpu;
        pr_info("psp_ring: fence after zero: 0x%08x (should be 0)\n", fence_check);
    }

    /* ---- Destroy existing ring + Create fresh ---- */
    {
        ULONG c64 = mp0_rreg(dev, regMPASP_SMN_C2PMSG_64);
        ULONG ring_size = mp0_rreg(dev, regMPASP_SMN_C2PMSG_71);
        pr_info("psp_ring: before destroy: C2PMSG_64=0x%08x size=0x%x\n",
                c64, ring_size);

        if (ring_size != 0) {
            mp0_wreg(dev, regMPASP_SMN_C2PMSG_64,
                     GFX_CTRL_CMD_ID_DESTROY_RINGS);
            Sleep(20);
            for (int i = 0; i < 5000; i++) {
                ULONG v = mp0_rreg(dev, regMPASP_SMN_C2PMSG_64);
                if (v & GFX_FLAG_RESPONSE) {
                    pr_info("psp_ring: destroy done after %d ms: 0x%08x\n",
                            i + 20, v);
                    break;
                }
                Sleep(1);
            }
        }

        /* Wait for TOS ready */
        ULONG v = mp0_rreg(dev, regMPASP_SMN_C2PMSG_64);
        if (!(v & GFX_FLAG_RESPONSE)) {
            if (psp_ring_wait(dev, GFX_FLAG_RESPONSE,
                              GFX_FLAG_RESPONSE, 20000) != 0) {
                pr_err("psp_ring: TOS not ready\n");
                goto fail;
            }
        }

        /* Zero ring and create */
        memset(ctx->ring_cpu, 0, PSP_RING_SIZE);
        MemoryBarrier();
        gpu_hdp_flush(dev);

        mp0_wreg(dev, regMPASP_SMN_C2PMSG_69,
                 (ULONG)(ctx->ring_mc_addr & 0xFFFFFFFF));
        mp0_wreg(dev, regMPASP_SMN_C2PMSG_70,
                 (ULONG)((ctx->ring_mc_addr >> 32) & 0xFFFFFFFF));
        mp0_wreg(dev, regMPASP_SMN_C2PMSG_71, PSP_RING_SIZE);
        mp0_wreg(dev, regMPASP_SMN_C2PMSG_64, PSP_RING_TYPE_KM << 16);
        Sleep(20);

        /* Wait for ring creation: bit 31 set AND low 16 bits == 0 */
        if (psp_ring_wait(dev, GFX_FLAG_RESPONSE, 0x8000FFFF,
                          20000) != 0) {
            /* If strict check fails, try just waiting for response bit */
            c64 = mp0_rreg(dev, regMPASP_SMN_C2PMSG_64);
            ULONG low_status = c64 & 0x0000FFFF;
            pr_warn("psp_ring: ring creation returned status 0x%04x "
                    "(C2PMSG_64=0x%08x)\n", low_status, c64);
            if (!(c64 & GFX_FLAG_RESPONSE)) {
                pr_err("psp_ring: ring creation timed out\n");
                goto fail;
            }
        }

        c64 = mp0_rreg(dev, regMPASP_SMN_C2PMSG_64);
        ULONG status = c64 & 0x0000FFFF;
        pr_info("psp_ring: created: C2PMSG_64=0x%08x (status=0x%04x)\n",
                c64, status);

        ctx->wptr = mp0_rreg(dev, regMPASP_SMN_C2PMSG_67);
        pr_info("psp_ring: ready, wptr=%u\n", ctx->wptr);
    }

    ctx->valid = TRUE;
    return 0;

fail:
    psp_ring_unmap(dev, ctx->fence_cpu, ctx->fence_handle);
    psp_ring_unmap(dev, ctx->cmd_cpu, ctx->cmd_handle);
    psp_ring_unmap(dev, ctx->fw_cpu, ctx->fw_handle);
    psp_ring_unmap(dev, ctx->ring_cpu, ctx->ring_handle);
    memset(ctx, 0, sizeof(*ctx));
    return -1;
}

static void psp_ring_destroy(struct PspRingContext *ctx)
{
    if (!ctx->valid)
        return;

    psp_ring_unmap(ctx->dev, ctx->fence_cpu, ctx->fence_handle);
    psp_ring_unmap(ctx->dev, ctx->cmd_cpu, ctx->cmd_handle);
    psp_ring_unmap(ctx->dev, ctx->fw_cpu, ctx->fw_handle);
    psp_ring_unmap(ctx->dev, ctx->ring_cpu, ctx->ring_handle);
    ctx->valid = FALSE;
}

/*
 * Submit a PSP ring command and wait for completion via fence.
 * The command buffer should already be populated in ctx->cmd_cpu.
 * Returns 0 on success, -1 on failure.
 */
static int psp_ring_submit(struct PspRingContext *ctx, const char *desc)
{
    ULONG fence_val = ctx->fence_seq++;

    /* Clear fence */
    *(volatile ULONG *)ctx->fence_cpu = 0;
    MemoryBarrier();
    gpu_hdp_flush(ctx->dev);

    /* Write 64-byte frame at wptr position */
    ULONG byte_offset = (ctx->wptr * 4) % PSP_RING_SIZE;
    struct PspGfxRbFrame *frame =
        (struct PspGfxRbFrame *)((UCHAR *)ctx->ring_cpu + byte_offset);

    memset(frame, 0, sizeof(*frame));
    frame->cmd_buf_addr_lo = (ULONG)(ctx->cmd_mc_addr & 0xFFFFFFFF);
    frame->cmd_buf_addr_hi = (ULONG)((ctx->cmd_mc_addr >> 32) & 0xFFFFFFFF);
    /* amdgpu leaves cmd_buf_size=0 in the ring frame (confirmed via VRAM capture) */
    frame->cmd_buf_size = 0;
    frame->fence_addr_lo = (ULONG)(ctx->fence_mc_addr & 0xFFFFFFFF);
    frame->fence_addr_hi = (ULONG)((ctx->fence_mc_addr >> 32) & 0xFFFFFFFF);
    frame->fence_value = fence_val;

    MemoryBarrier();
    gpu_hdp_flush(ctx->dev);

    /* Advance wptr */
    ULONG new_wptr = ctx->wptr + PSP_RING_FRAME_DWORDS;
    mp0_wreg(ctx->dev, regMPASP_SMN_C2PMSG_67, new_wptr);
    ctx->wptr = new_wptr;

    /* Wait for fence */
    volatile ULONG *fence_ptr = (volatile ULONG *)ctx->fence_cpu;
    for (int i = 0; i < 30000; i++) {
        if (*fence_ptr == fence_val) {
            struct PspGfxCmdResp *cmd = (struct PspGfxCmdResp *)ctx->cmd_cpu;
            if (cmd->resp_status != 0) {
                pr_err("psp_ring: %s: resp_status=0x%08x (fence ok after %d ms)\n",
                       desc, cmd->resp_status, i);
                return -1;
            }
            pr_info("psp_ring: %s: ok (%d ms)\n", desc, i);
            return 0;
        }
        if (i == 1000 || i == 5000 || i == 15000) {
            pr_info("psp_ring: %s: waiting... (%d ms, fence=%u expect=%u)\n",
                    desc, i, *fence_ptr, fence_val);
        }
        Sleep(1);
    }

    struct PspGfxCmdResp *cmd = (struct PspGfxCmdResp *)ctx->cmd_cpu;
    pr_err("psp_ring: %s: fence timeout (val=%u expect=%u, "
           "resp_status=0x%08x)\n",
           desc, *fence_ptr, fence_val, cmd->resp_status);
    return -1;
}

/*
 * Submit a LOAD_IP_FW command for firmware already staged in fw_cpu.
 */
static int psp_ring_load_fw(struct PspRingContext *ctx,
                             ULONG fw_type, const char *fw_name,
                             const UCHAR *fw_data, ULONG fw_size)
{
    if (fw_size > 2 * 1024 * 1024) {
        pr_err("psp_ring: %s: too large (%u bytes)\n", fw_name, fw_size);
        return -1;
    }

    /* Copy firmware to staging VRAM */
    memcpy(ctx->fw_cpu, fw_data, fw_size);
    MemoryBarrier();
    gpu_hdp_flush(ctx->dev);

    /* Build command buffer — amdgpu VRAM capture shows buf_size=0, buf_ver=0 */
    struct PspGfxCmdResp *cmd = (struct PspGfxCmdResp *)ctx->cmd_cpu;
    memset(cmd, 0, sizeof(*cmd));
    /* Leave buf_size=0, buf_version=0 (matching amdgpu VRAM capture) */
    cmd->cmd_id = GFX_CMD_ID_LOAD_IP_FW;
    cmd->load_ip_fw.fw_phy_addr_lo =
        (ULONG)(ctx->fw_mc_addr & 0xFFFFFFFF);
    cmd->load_ip_fw.fw_phy_addr_hi =
        (ULONG)((ctx->fw_mc_addr >> 32) & 0xFFFFFFFF);
    cmd->load_ip_fw.fw_size = fw_size;
    cmd->load_ip_fw.fw_type = fw_type;

    MemoryBarrier();
    gpu_hdp_flush(ctx->dev);

    /* Dump first 64 bytes of command buffer for SMU firmware load */
    if (fw_type == GFX_FW_TYPE_SMU) {
        UCHAR *p = (UCHAR *)cmd;
        pr_info("psp_ring: SMU LOAD_IP_FW cmd buffer (first 48 bytes):\n");
        for (int row = 0; row < 3; row++) {
            pr_info("  +0x%02x: %02x%02x%02x%02x %02x%02x%02x%02x "
                    "%02x%02x%02x%02x %02x%02x%02x%02x\n",
                    row*16,
                    p[row*16+0], p[row*16+1], p[row*16+2], p[row*16+3],
                    p[row*16+4], p[row*16+5], p[row*16+6], p[row*16+7],
                    p[row*16+8], p[row*16+9], p[row*16+10], p[row*16+11],
                    p[row*16+12], p[row*16+13], p[row*16+14], p[row*16+15]);
        }
        pr_info("psp_ring: fw_mc=0x%llx fw_size=%u fw_type=%u\n",
                ctx->fw_mc_addr, fw_size, fw_type);
        /* Also verify first 16 bytes of firmware data in VRAM */
        UCHAR *fw = (UCHAR *)ctx->fw_cpu;
        pr_info("psp_ring: FW data first 16 bytes in VRAM: "
                "%02x%02x%02x%02x %02x%02x%02x%02x "
                "%02x%02x%02x%02x %02x%02x%02x%02x\n",
                fw[0], fw[1], fw[2], fw[3], fw[4], fw[5], fw[6], fw[7],
                fw[8], fw[9], fw[10], fw[11], fw[12], fw[13], fw[14], fw[15]);
    }

    char desc[64];
    snprintf(desc, sizeof(desc), "LOAD_IP_FW(%s, type=%u, %u bytes)",
             fw_name, fw_type, fw_size);
    return psp_ring_submit(ctx, desc);
}

/*
 * Submit the AUTOLOAD_RLC command to trigger RLC autoload.
 */
static int psp_ring_autoload_rlc(struct PspRingContext *ctx)
{
    struct PspGfxCmdResp *cmd = (struct PspGfxCmdResp *)ctx->cmd_cpu;
    memset(cmd, 0, sizeof(*cmd));
    cmd->cmd_id = GFX_CMD_ID_AUTOLOAD_RLC;

    MemoryBarrier();
    gpu_hdp_flush(ctx->dev);

    return psp_ring_submit(ctx, "AUTOLOAD_RLC");
}

/*
 * Load TOC (Table of Contents) and get required TMR size.
 * The TOC firmware is loaded to VRAM and submitted via LOAD_TOC.
 * PSP responds with the required TMR size in resp_buf_size.
 */
static int psp_ring_load_toc(struct PspRingContext *ctx,
                              const UCHAR *toc_data, ULONG toc_size,
                              ULONG *out_tmr_size)
{
    if (toc_size > 2 * 1024 * 1024) {
        pr_err("psp_ring: TOC too large (%u bytes)\n", toc_size);
        return -1;
    }

    /* Copy TOC to staging VRAM */
    memcpy(ctx->fw_cpu, toc_data, toc_size);
    MemoryBarrier();
    gpu_hdp_flush(ctx->dev);

    /* Build LOAD_TOC command */
    struct PspGfxCmdResp *cmd = (struct PspGfxCmdResp *)ctx->cmd_cpu;
    memset(cmd, 0, sizeof(*cmd));
    cmd->cmd_id = GFX_CMD_ID_LOAD_TOC;
    cmd->load_toc.toc_phy_addr_lo =
        (ULONG)(ctx->fw_mc_addr & 0xFFFFFFFF);
    cmd->load_toc.toc_phy_addr_hi =
        (ULONG)((ctx->fw_mc_addr >> 32) & 0xFFFFFFFF);
    cmd->load_toc.toc_size = toc_size;

    MemoryBarrier();
    gpu_hdp_flush(ctx->dev);

    int ret = psp_ring_submit(ctx, "LOAD_TOC");
    if (ret != 0)
        return ret;

    /* TMR size is returned in resp_buf_size field */
    *out_tmr_size = cmd->resp_buf_size;
    pr_info("psp_ring: TOC loaded, TMR size needed = %u bytes (0x%x)\n",
            *out_tmr_size, *out_tmr_size);
    return 0;
}

/*
 * Set up TMR (Trusted Memory Region).
 * TMR must be set up before loading GFX firmware.
 *
 * tmr_mc_addr: MC address of TMR buffer (must be 1MB aligned)
 * tmr_size: size from LOAD_TOC response (or default 4MB)
 */
static int psp_ring_setup_tmr(struct PspRingContext *ctx,
                               ULONGLONG tmr_mc_addr, ULONG tmr_size)
{
    struct PspGfxCmdResp *cmd = (struct PspGfxCmdResp *)ctx->cmd_cpu;
    memset(cmd, 0, sizeof(*cmd));
    cmd->cmd_id = GFX_CMD_ID_SETUP_TMR;
    cmd->setup_tmr.buf_phy_addr_lo =
        (ULONG)(tmr_mc_addr & 0xFFFFFFFF);
    cmd->setup_tmr.buf_phy_addr_hi =
        (ULONG)((tmr_mc_addr >> 32) & 0xFFFFFFFF);
    cmd->setup_tmr.buf_size = tmr_size;
    /* virt_phy_addr = 1: we pass both virtual (MC) and physical addresses.
     * For VRAM, MC addr and physical addr are the same. */
    cmd->setup_tmr.tmr_flags = 0x2;  /* bit 1 = virt_phy_addr */
    cmd->setup_tmr.system_phy_addr_lo =
        (ULONG)(tmr_mc_addr & 0xFFFFFFFF);
    cmd->setup_tmr.system_phy_addr_hi =
        (ULONG)((tmr_mc_addr >> 32) & 0xFFFFFFFF);

    MemoryBarrier();
    gpu_hdp_flush(ctx->dev);

    pr_info("psp_ring: SETUP_TMR at MC 0x%llx, size=%u\n",
            (unsigned long long)tmr_mc_addr, tmr_size);
    return psp_ring_submit(ctx, "SETUP_TMR");
}

/* ---- Firmware file reading ---- */

static UCHAR *read_firmware_file(const char *path, ULONG *out_len)
{
    FILE *fp = fopen(path, "rb");
    if (!fp)
        return NULL;

    fseek(fp, 0, SEEK_END);
    ULONG len = (ULONG)ftell(fp);
    fseek(fp, 0, SEEK_SET);

    UCHAR *buf = (UCHAR *)malloc(len);
    if (!buf) {
        fclose(fp);
        return NULL;
    }
    if (fread(buf, 1, len, fp) != len) {
        fclose(fp);
        free(buf);
        return NULL;
    }
    fclose(fp);
    *out_len = len;
    return buf;
}

/*
 * Try to find a firmware file by name in the given directory.
 * Tries: dir/name, dir\name (Windows path separators).
 */
static UCHAR *find_and_read_fw(const char *fw_dir, const char *name,
                                 ULONG *out_len)
{
    char path[512];
    UCHAR *buf;

    snprintf(path, sizeof(path), "%s\\%s", fw_dir, name);
    buf = read_firmware_file(path, out_len);
    if (buf) {
        pr_info("psp_ring: loaded %s (%u bytes)\n", path, *out_len);
        return buf;
    }

    snprintf(path, sizeof(path), "%s/%s", fw_dir, name);
    buf = read_firmware_file(path, out_len);
    if (buf) {
        pr_info("psp_ring: loaded %s (%u bytes)\n", path, *out_len);
        return buf;
    }

    return NULL;
}

/*
 * Load a firmware with common_firmware_header (v1.x).
 * Extracts ucode from ucode_array_offset_bytes.
 */
static int load_fw_v1(struct PspRingContext *ctx, const char *fw_dir,
                       const char *filename, ULONG fw_type,
                       const char *alt_filename)
{
    ULONG file_len = 0;
    UCHAR *buf = find_and_read_fw(fw_dir, filename, &file_len);
    if (!buf && alt_filename)
        buf = find_and_read_fw(fw_dir, alt_filename, &file_len);
    if (!buf) {
        pr_warn("psp_ring: firmware not found: %s\n", filename);
        return -1;
    }

    if (file_len < sizeof(struct CommonFirmwareHeader)) {
        pr_err("psp_ring: %s too small (%u bytes)\n", filename, file_len);
        free(buf);
        return -1;
    }

    struct CommonFirmwareHeader *hdr = (struct CommonFirmwareHeader *)buf;
    ULONG offset = hdr->ucode_array_offset_bytes;
    ULONG size = hdr->ucode_size_bytes;

    pr_info("psp_ring: %s: header v%u.%u, ucode_offset=0x%x, ucode_size=%u\n",
            filename, hdr->header_version_major, hdr->header_version_minor,
            offset, size);

    if (offset + size > file_len) {
        pr_err("psp_ring: %s: ucode overflows (0x%x + 0x%x > 0x%x)\n",
               filename, offset, size, file_len);
        free(buf);
        return -1;
    }

    int ret = psp_ring_load_fw(ctx, fw_type, filename, buf + offset, size);
    free(buf);
    return ret;
}

/*
 * Load an RS64 firmware (v2.0 header) with separate code and data sections.
 * Loads the code section with fw_type_code, and the data section with
 * each of the stack fw_types (P0, P1, ...).
 */
static int load_fw_rs64(struct PspRingContext *ctx, const char *fw_dir,
                         const char *filename,
                         ULONG fw_type_code,
                         const ULONG *fw_type_stacks,
                         int num_stacks,
                         ULONGLONG *out_ucode_start)
{
    if (out_ucode_start)
        *out_ucode_start = 0;

    ULONG file_len = 0;
    UCHAR *buf = find_and_read_fw(fw_dir, filename, &file_len);
    if (!buf) {
        pr_warn("psp_ring: RS64 firmware not found: %s\n", filename);
        return -1;
    }

    if (file_len < sizeof(struct GfxFirmwareHeaderV2)) {
        pr_err("psp_ring: %s too small for v2 header (%u bytes)\n",
               filename, file_len);
        free(buf);
        return -1;
    }

    struct GfxFirmwareHeaderV2 *hdr = (struct GfxFirmwareHeaderV2 *)buf;
    pr_info("psp_ring: %s: header v%u.%u, hdr_size=%u\n",
            filename,
            hdr->header_version_major, hdr->header_version_minor,
            hdr->header_size_bytes);

    /*
     * For v2.0 headers, we have separate code and data sections.
     * For v1.x (MES) headers, the layout differs: an extra data_version
     * field at +0x2C shifts data_size/data_offset/ucode_start by 4 bytes.
     *
     * v2 layout: +0x2C=data_size  +0x30=data_off  +0x34=start_lo +0x38=start_hi
     * v1 layout: +0x2C=data_ver   +0x30=data_size +0x34=data_off +0x38=start_lo +0x3C=start_hi
     */
    ULONG code_offset, code_size, data_offset, data_size;

    if (hdr->header_version_major >= 2) {
        code_offset = hdr->ucode_offset_bytes;
        code_size = hdr->ucode_size_bytes;
        data_offset = hdr->data_offset_bytes;
        data_size = hdr->data_size_bytes;

        /* v2: ucode_start at DWORDs 13/14 (offsets 0x34/0x38) */
        if (out_ucode_start && file_len >= 0x3C) {
            ULONG *raw = (ULONG *)buf;
            ULONGLONG start = ((ULONGLONG)raw[14] << 32) | raw[13];
            if (start != 0) {
                *out_ucode_start = start;
                pr_info("psp_ring: %s: ucode_start = 0x%08x_%08x\n",
                        filename, raw[14], raw[13]);
            }
        }
    } else {
        /* v1.x MES header (mes_firmware_header_v1_0):
         *   +0x24: mes_ucode_size_bytes (code size)
         *   +0x28: mes_ucode_offset_bytes (code offset)
         *   +0x2C: mes_ucode_data_version (extra field!)
         *   +0x30: mes_ucode_data_size_bytes (data/stack size)
         *   +0x34: mes_ucode_data_offset_bytes (data/stack offset)
         *   +0x38: mes_uc_start_addr_lo
         *   +0x3C: mes_uc_start_addr_hi
         */
        ULONG *raw = (ULONG *)buf;
        code_offset = raw[10]; /* +0x28 */
        code_size = raw[9];    /* +0x24 */
        data_size = raw[12];   /* +0x30 */
        data_offset = raw[13]; /* +0x34 */

        pr_info("psp_ring: %s: v1 MES header: code=%u@0x%x data=%u@0x%x\n",
                filename, code_size, code_offset, data_size, data_offset);

        /* v1: ucode_start at DWORDs 14/15 (offsets 0x38/0x3C) */
        if (out_ucode_start && file_len >= 0x40) {
            ULONGLONG start = ((ULONGLONG)raw[15] << 32) | raw[14];
            if (start != 0 && start != 0xFFFFFFFFFFFFFFFFULL) {
                *out_ucode_start = start;
                pr_info("psp_ring: %s: ucode_start = 0x%08x_%08x\n",
                        filename, raw[15], raw[14]);
            }
        }
    }

    /* Load code section */
    if (code_offset + code_size > file_len) {
        pr_err("psp_ring: %s: code overflows (0x%x + 0x%x > 0x%x)\n",
               filename, code_offset, code_size, file_len);
        free(buf);
        return -1;
    }

    int ret = psp_ring_load_fw(ctx, fw_type_code, filename,
                                buf + code_offset, code_size);
    if (ret != 0) {
        free(buf);
        return ret;
    }

    /* Load data/stack sections — same data for each stack partition */
    if (data_size > 0 && data_offset + data_size <= file_len) {
        for (int i = 0; i < num_stacks; i++) {
            char stack_desc[64];
            snprintf(stack_desc, sizeof(stack_desc), "%s_P%d_STACK",
                     filename, i);
            ret = psp_ring_load_fw(ctx, fw_type_stacks[i], stack_desc,
                                    buf + data_offset, data_size);
            if (ret != 0) {
                pr_warn("psp_ring: %s stack P%d load failed (continuing)\n",
                        filename, i);
            }
        }
    }

    free(buf);
    return 0;
}

/*
 * Load SMU firmware and verify SMU comes alive.
 * This is called first since SMU is needed for GFXOFF control.
 */
static int psp_load_smu(struct PspRingContext *ctx, const char *fw_dir)
{
    int ret = load_fw_v1(ctx, fw_dir, "smu_14_0_3.bin", GFX_FW_TYPE_SMU,
                          "smu_14_0_2.bin");
    if (ret != 0)
        return ret;

    /* Verify SMU alive */
    Sleep(100);
    ULONG resp = mp1_rreg(ctx->dev, regMP1_SMN_C2PMSG_90);
    mp1_wreg(ctx->dev, regMP1_SMN_C2PMSG_90, 0);
    mp1_wreg(ctx->dev, regMP1_SMN_C2PMSG_82, 0);
    mp1_wreg(ctx->dev, regMP1_SMN_C2PMSG_66, PPSMC_MSG_GetSmuVersion);

    for (int i = 0; i < 500; i++) {
        resp = mp1_rreg(ctx->dev, regMP1_SMN_C2PMSG_90);
        if (resp != 0) {
            ULONG version = mp1_rreg(ctx->dev, regMP1_SMN_C2PMSG_82);
            pr_info("psp_ring: SMU alive, version=0x%08x\n", version);
            return 0;
        }
        Sleep(1);
    }
    pr_warn("psp_ring: SMU not responding after load\n");
    return 0;  /* Non-fatal: SMU loaded but may take time to respond */
}

/*
 * Load all GPU firmware via PSP ring and trigger RLC autoload.
 *
 * Loading order (matching tinygrad AM_PSP.init_hw for PSP v14.0.3):
 *   1. SMU (needed for power management / GFXOFF)
 *   2. Skip TOC/TMR (boot_time_tmr=true for v14.0.3 — VBIOS handles it)
 *   3. SDMA (thread-based TH0 for GFX12)
 *   4. RS64 PFP (code + P0 stack)
 *   5. RS64 ME  (code + P0 stack)
 *   6. RS64 MEC (code + P0 stack)
 *   7. IMU (IRAM + DRAM)
 *   8. RLC sub-components (IRAM + DRAM_BOOT for v2.2)
 *   9. AUTOLOAD_RLC → triggers RLC to bring GC block online
 *
 * After autoload completes, GC registers become accessible and
 * we can proceed with GFXHUB init, SH_MEM config, MEC enable.
 */

/* IMU firmware header (extends common header) */
struct ImuFirmwareHeaderV1 {
    /* common_firmware_header (0x00-0x1F, 32 bytes) */
    ULONG   size_bytes;
    ULONG   header_size_bytes;
    USHORT  header_version_major;
    USHORT  header_version_minor;
    USHORT  ip_version_major;
    USHORT  ip_version_minor;
    ULONG   ucode_version;
    ULONG   ucode_size_bytes;       /* total size */
    ULONG   ucode_array_offset_bytes;
    LONG    crc32;
    /* IMU-specific fields */
    ULONG   imu_iram_ucode_size_bytes;   /* +0x20 */
    ULONG   imu_iram_ucode_offset_bytes; /* +0x24 (usually 0) */
    ULONG   imu_dram_ucode_size_bytes;   /* +0x28 */
    ULONG   imu_dram_ucode_size_bytes2;  /* +0x2C (padding/duplicate) */
};

/* FW types for RLC sub-components */
#define GFX_FW_TYPE_RLC_IRAM                  26
#define GFX_FW_TYPE_RLC_DRAM_BOOT             48

/* ======================================================================
 * GC VRAM staging buffer for backdoor AUTOLOAD_RLC
 *
 * GFX12 uses AMDGPU_FW_LOAD_RLC_BACKDOOR_AUTO: instead of individual
 * LOAD_IP_FW calls for each GFX firmware, all encrypted firmware binaries
 * are placed in a VRAM staging buffer at TOC-defined offsets.  PSP reads
 * GFX_IMU_RLC_BOOTLOADER_ADDR to locate the staging buffer, decrypts the
 * content during AUTOLOAD_RLC, and creates a valid TOC for RLC to consume.
 * RLC then distributes firmware to MEC, RLCP, RLCV engines.
 *
 * Without the staging buffer, AUTOLOAD_RLC creates an empty/invalid TOC,
 * RLC finds no firmware pointers → ID_STATUS stays 0 → MEC never starts.
 * ====================================================================== */

/*
 * SOC24_FIRMWARE_ID values (from Linux amdgpu_rlc.h SOC24_FIRMWARE_ID enum).
 * These identify firmware component slots in the VRAM staging buffer.
 */
#define SOC24_FW_RLC_G_UCODE        1
#define SOC24_FW_RLC_TOC            2
#define SOC24_FW_SDMA_UCODE_TH0     11
#define SOC24_FW_RS64_PFP           18
#define SOC24_FW_RS64_ME            19
#define SOC24_FW_RS64_MEC           20
#define SOC24_FW_RS64_PFP_P0_STACK  23
#define SOC24_FW_RS64_PFP_P1_STACK  24
#define SOC24_FW_RS64_ME_P0_STACK   25
#define SOC24_FW_RS64_ME_P1_STACK   26
#define SOC24_FW_RS64_MEC_P0_STACK  27
#define SOC24_FW_RS64_MEC_P1_STACK  28
#define SOC24_FW_RS64_MEC_P2_STACK  29
#define SOC24_FW_RS64_MEC_P3_STACK  30
#define SOC24_FW_ID_MAX             48  /* accommodate future IDs beyond MAX=43 */

/*
 * VRAM layout for GC staging buffer.
 * Placed at 32MB to clear all existing allocations (PSP ring at 5-8MB,
 * MEC fallback at 10-11MB).  Total size 24MB matches the TOC maximum.
 */
#define GC_STAGING_VRAM_OFFSET  (32ULL * 1024 * 1024)   /* 32 MB */
#define GC_STAGING_BUF_SIZE     (24UL  * 1024 * 1024)   /* 24 MB */

/* GC TOC firmware registers (BASE_IDX=1, PSP-protected, write via MMIO) */
#define regGFX_IMU_RLC_BOOTLOADER_ADDR_HI   0x5f81
#define regGFX_IMU_RLC_BOOTLOADER_ADDR_LO   0x5f82
#define regGFX_IMU_RLC_BOOTLOADER_SIZE      0x5f83

/* Per-firmware slot parsed from the GC TOC binary */
struct GcStagingSlot {
    ULONG offset;   /* byte offset within staging buffer */
    ULONG size;     /* slot size in bytes */
};

/*
 * Parse the GC TOC binary payload (after common firmware header) to
 * fill the slot table.
 *
 * RLC_TABLE_OF_CONTENT_V2 entry format (8 bytes):
 *   DW0: bits[31:25] = fw_id (7 bits), bits[24:0] = offset field
 *   DW1: bits[31:14] = size field (18 bits), bit[12] = size_x16 flag
 *
 *   offset_bytes = offset_field × 32
 *   size_bytes   = size_x16 ? size_field × 4096 : size_field × 4
 *
 * Terminated by an entry with fw_id == 0.
 */
static void gc_toc_parse(const UCHAR *payload, ULONG payload_size,
                          struct GcStagingSlot *slots, int max_slots)
{
    for (ULONG i = 0; i + 8 <= payload_size; i += 8) {
        ULONG dw0      = *(const ULONG *)(payload + i);
        ULONG dw1      = *(const ULONG *)(payload + i + 4);
        ULONG fw_id    = (dw0 >> 25) & 0x7F;
        ULONG off_fld  = dw0 & 0x1FFFFFF;
        ULONG sz_x16   = (dw1 >> 12) & 1;
        ULONG sz_fld   = (dw1 >> 14) & 0x3FFFF;

        if (fw_id == 0)
            break;
        if ((int)fw_id >= max_slots)
            continue;

        slots[fw_id].offset = off_fld * 32;
        slots[fw_id].size   = sz_x16 ? sz_fld * 4096 : sz_fld * 4;
    }
}

/*
 * Copy a firmware payload into the staging buffer at the TOC-defined slot.
 * Copies min(fw_size, slot_size) bytes; zeroes any remainder.
 */
/*
 * Copy firmware payload to the correct partition of the staging buffer.
 * buf_base_off: staging-area offset where this buffer starts (0 or P1_SIZE).
 * Silently skips if the slot is outside this partition's range.
 */
static void gc_staging_copy(UCHAR *buf, ULONG buf_size, ULONG buf_base_off,
                              const struct GcStagingSlot *slots, ULONG fw_id,
                              const UCHAR *fw_data, ULONG fw_size,
                              const char *name)
{
    ULONG slot_off = slots[fw_id].offset;
    ULONG slot_sz  = slots[fw_id].size;

    if (slot_sz == 0) {
        pr_warn("gc_staging: no slot for fw_id=%u (%s)\n", fw_id, name);
        return;
    }
    /* Skip silently if this slot lives in a different partition */
    if (slot_off < buf_base_off ||
        slot_off + slot_sz > buf_base_off + buf_size) {
        return;
    }

    ULONG local_off = slot_off - buf_base_off;
    ULONG copy_sz = (fw_size < slot_sz) ? fw_size : slot_sz;
    memcpy(buf + local_off, fw_data, copy_sz);
    if (copy_sz < slot_sz)
        memset(buf + local_off + copy_sz, 0, slot_sz - copy_sz);

    pr_info("gc_staging: id=%u (%s): slot off=%u sz=%u, copied %u bytes\n",
            fw_id, name, slot_off, slot_sz, copy_sz);
}

/*
 * Set up the GC VRAM staging buffer for backdoor AUTOLOAD_RLC.
 *
 * Steps:
 *   1. Parse gc_12_0_1_toc.bin to get slot offsets/sizes.
 *   2. Allocate 24 MB VRAM buffer at GC_STAGING_VRAM_OFFSET.
 *   3. Copy each firmware's encrypted payload to its TOC-defined slot.
 *   4. Write GFX_IMU_RLC_BOOTLOADER_ADDR_HI/LO/SIZE via direct MMIO BAR
 *      (bypasses PSP protection that blocks SMN indirect access).
 *
 * PSP reads GFX_IMU_RLC_BOOTLOADER_ADDR, decrypts firmware from staging
 * buffer, and creates a valid TOC at RLC_IMU_BOOTLOAD_ADDR.  RLC then
 * distributes firmware to MEC, RLCP, RLCV → BOOTLOAD_STATUS bit 31 set.
 *
 * Returns 0 on success.  Caller should skip LOAD_IP_FW for GFX firmware
 * if this succeeds.
 */
static int gpu_gc_staging_setup(struct WddmLiteDevice *dev,
                                  const char *fw_dir)
{
    struct GcStagingSlot slots[SOC24_FW_ID_MAX];
    memset(slots, 0, sizeof(slots));

    /* ---- Step 1: Parse GC TOC binary ---- */
    ULONG toc_file_len = 0;
    UCHAR *toc_buf = find_and_read_fw(fw_dir, "gc_12_0_1_toc.bin",
                                        &toc_file_len);
    if (!toc_buf)
        toc_buf = find_and_read_fw(fw_dir, "gc_12_0_0_toc.bin",
                                    &toc_file_len);
    if (!toc_buf) {
        pr_warn("gc_staging: GC TOC binary not found in %s\n", fw_dir);
        return -1;
    }

    if (toc_file_len < 0x20) {
        pr_warn("gc_staging: GC TOC too small (%u bytes)\n", toc_file_len);
        free(toc_buf);
        return -1;
    }

    /* Common firmware header: ucode_size_bytes at +0x14, offset at +0x18 */
    ULONG payload_size = *(ULONG *)(toc_buf + 0x14);
    ULONG payload_off  = *(ULONG *)(toc_buf + 0x18);
    if (payload_off + payload_size > toc_file_len || payload_size == 0) {
        pr_warn("gc_staging: GC TOC payload out of bounds "
                "(off=0x%x size=%u file=%u)\n",
                payload_off, payload_size, toc_file_len);
        free(toc_buf);
        return -1;
    }

    gc_toc_parse(toc_buf + payload_off, payload_size, slots, SOC24_FW_ID_MAX);

    pr_info("gc_staging: TOC parsed: RLC_G slot off=%u sz=%u, "
            "MEC slot off=%u sz=%u, MEC_P0 slot off=%u sz=%u\n",
            slots[SOC24_FW_RLC_G_UCODE].offset,
            slots[SOC24_FW_RLC_G_UCODE].size,
            slots[SOC24_FW_RS64_MEC].offset,
            slots[SOC24_FW_RS64_MEC].size,
            slots[SOC24_FW_RS64_MEC_P0_STACK].offset,
            slots[SOC24_FW_RS64_MEC_P0_STACK].size);

    if (slots[SOC24_FW_RLC_G_UCODE].size == 0 ||
        slots[SOC24_FW_RS64_MEC].size == 0) {
        pr_warn("gc_staging: required slots missing from TOC — "
                "firmware may not support backdoor AUTOLOAD\n");
        free(toc_buf);
        return -1;
    }

    /* ---- Step 2: Allocate VRAM staging buffer ----
     * The MAP_VRAM escape has a 16 MB per-mapping limit, so we split
     * the 24 MB staging area into two parts.
     */
#define GC_STAGING_P1_SIZE  (16UL * 1024 * 1024)  /* 0 – 16 MB  */
#define GC_STAGING_P2_SIZE  (GC_STAGING_BUF_SIZE - GC_STAGING_P1_SIZE) /* 16 – 24 MB */

    PVOID staging_handle1 = NULL, staging_handle2 = NULL;
    UCHAR *staging1 = (UCHAR *)psp_ring_map_vram(dev,
                                  GC_STAGING_VRAM_OFFSET,
                                  GC_STAGING_P1_SIZE,
                                  &staging_handle1);
    UCHAR *staging2 = (UCHAR *)psp_ring_map_vram(dev,
                                  GC_STAGING_VRAM_OFFSET + GC_STAGING_P1_SIZE,
                                  GC_STAGING_P2_SIZE,
                                  &staging_handle2);
    if (!staging1 || !staging2) {
        pr_warn("gc_staging: failed to map VRAM staging parts "
                "(part1=%p part2=%p)\n", staging1, staging2);
        if (staging1) psp_ring_unmap(dev, staging1, staging_handle1);
        if (staging2) psp_ring_unmap(dev, staging2, staging_handle2);
        free(toc_buf);
        return -1;
    }

    pr_info("gc_staging: mapped staging buffer at VRAM+32 MB "
            "(MC=0x%llx), p1=%p p2=%p\n",
            (unsigned long long)(dev->hw.gmc.vram_start +
                                  GC_STAGING_VRAM_OFFSET),
            staging1, staging2);

    memset(staging1, 0, GC_STAGING_P1_SIZE);
    memset(staging2, 0, GC_STAGING_P2_SIZE);
    MemoryBarrier();

    /* ---- Step 3: Copy firmware payloads ---- */

    /* id=2 (RLC_TOC): the GC TOC payload itself — RLC reads this to find
     * firmware locations within the staging buffer */
    gc_staging_copy(staging1, GC_STAGING_P1_SIZE, 0,
                     slots, SOC24_FW_RLC_TOC,
                     toc_buf + payload_off, payload_size, "RLC_TOC");
    gc_staging_copy(staging2, GC_STAGING_P2_SIZE, GC_STAGING_P1_SIZE,
                     slots, SOC24_FW_RLC_TOC,
                     toc_buf + payload_off, payload_size, "RLC_TOC");
    free(toc_buf);
    toc_buf = NULL;

    /* id=1 (RLC_G_UCODE): RLC IRAM content (from gc_12_0_1_rlc.bin v2.2).
     * GFX_IMU_RLC_BOOTLOADER_ADDR points here; IMU boots RLC from this. */
    {
        ULONG len = 0;
        UCHAR *buf = find_and_read_fw(fw_dir, "gc_12_0_1_rlc.bin", &len);
        if (!buf)
            buf = find_and_read_fw(fw_dir, "gc_12_0_0_rlc.bin", &len);

        if (buf && len >= 0xAC) {
            ULONG iram_off  = *(ULONG *)(buf + 0xA0); /* rlc_iram_ucode_offset_bytes */
            ULONG iram_size = *(ULONG *)(buf + 0x9C); /* rlc_iram_ucode_size_bytes */
            pr_info("gc_staging: RLC IRAM: off=0x%x size=%u\n",
                    iram_off, iram_size);
            if (iram_size > 0 && iram_off + iram_size <= len) {
                gc_staging_copy(staging1, GC_STAGING_P1_SIZE, 0,
                                 slots, SOC24_FW_RLC_G_UCODE,
                                 buf + iram_off, iram_size, "RLC_G_IRAM");
                gc_staging_copy(staging2, GC_STAGING_P2_SIZE, GC_STAGING_P1_SIZE,
                                 slots, SOC24_FW_RLC_G_UCODE,
                                 buf + iram_off, iram_size, "RLC_G_IRAM");
            } else {
                pr_warn("gc_staging: RLC IRAM out of bounds\n");
            }
        } else {
            pr_warn("gc_staging: gc_12_0_1_rlc.bin not found or too small\n");
        }
        if (buf) free(buf);
    }

    /* RS64 firmwares: PFP, ME, MEC — code section + all pipe stack sections.
     * The data section (stack) content is identical for all pipe slots. */
    struct {
        const char *filename;
        ULONG       code_id;
        ULONG       stack_ids[4];
        int         num_stacks;
        const char *name;
        ULONGLONG  *ucode_start_out;
    } rs64_fw[] = {
        { "gc_12_0_1_pfp.bin", SOC24_FW_RS64_PFP,
          { SOC24_FW_RS64_PFP_P0_STACK, SOC24_FW_RS64_PFP_P1_STACK, 0, 0 }, 2,
          "PFP", &dev->hw.gfx.pfp_ucode_start },
        { "gc_12_0_1_me.bin",  SOC24_FW_RS64_ME,
          { SOC24_FW_RS64_ME_P0_STACK, SOC24_FW_RS64_ME_P1_STACK, 0, 0 }, 2,
          "ME",  &dev->hw.gfx.me_ucode_start },
        { "gc_12_0_1_mec.bin", SOC24_FW_RS64_MEC,
          { SOC24_FW_RS64_MEC_P0_STACK, SOC24_FW_RS64_MEC_P1_STACK,
            SOC24_FW_RS64_MEC_P2_STACK, SOC24_FW_RS64_MEC_P3_STACK }, 4,
          "MEC", &dev->hw.gfx.mec_ucode_start },
    };

    for (int fi = 0; fi < 3; fi++) {
        ULONG len = 0;
        UCHAR *buf = find_and_read_fw(fw_dir, rs64_fw[fi].filename, &len);
        if (!buf) {
            pr_warn("gc_staging: %s not found\n", rs64_fw[fi].filename);
            continue;
        }
        if (len < sizeof(struct GfxFirmwareHeaderV2)) {
            pr_warn("gc_staging: %s too small for v2 header\n",
                    rs64_fw[fi].filename);
            free(buf);
            continue;
        }

        struct GfxFirmwareHeaderV2 *hdr = (struct GfxFirmwareHeaderV2 *)buf;
        if (hdr->header_version_major < 2) {
            pr_warn("gc_staging: %s: header v%u not v2.x, skipping\n",
                    rs64_fw[fi].filename, hdr->header_version_major);
            free(buf);
            continue;
        }

        ULONG code_off  = hdr->ucode_offset_bytes;
        ULONG code_size = hdr->ucode_size_bytes;
        ULONG data_off  = hdr->data_offset_bytes;
        ULONG data_size = hdr->data_size_bytes;

        pr_info("gc_staging: %s: code off=0x%x sz=%u, data off=0x%x sz=%u\n",
                rs64_fw[fi].name, code_off, code_size, data_off, data_size);

        /* Code section → code slot */
        if (code_size > 0 && code_off + code_size <= len) {
            char nbuf[32];
            snprintf(nbuf, sizeof(nbuf), "%s_code", rs64_fw[fi].name);
            gc_staging_copy(staging1, GC_STAGING_P1_SIZE, 0,
                             slots, rs64_fw[fi].code_id,
                             buf + code_off, code_size, nbuf);
            gc_staging_copy(staging2, GC_STAGING_P2_SIZE, GC_STAGING_P1_SIZE,
                             slots, rs64_fw[fi].code_id,
                             buf + code_off, code_size, nbuf);
        } else {
            pr_warn("gc_staging: %s code section out of bounds\n",
                    rs64_fw[fi].filename);
        }

        /* Data/stack section → all pipe stack slots (same content) */
        if (data_size > 0 && data_off + data_size <= len) {
            for (int si = 0; si < rs64_fw[fi].num_stacks; si++) {
                char nbuf[32];
                snprintf(nbuf, sizeof(nbuf), "%s_P%d_stack",
                         rs64_fw[fi].name, si);
                gc_staging_copy(staging1, GC_STAGING_P1_SIZE, 0,
                                 slots, rs64_fw[fi].stack_ids[si],
                                 buf + data_off, data_size, nbuf);
                gc_staging_copy(staging2, GC_STAGING_P2_SIZE, GC_STAGING_P1_SIZE,
                                 slots, rs64_fw[fi].stack_ids[si],
                                 buf + data_off, data_size, nbuf);
            }
        }

        /* Save ucode_start address from firmware header */
        if (rs64_fw[fi].ucode_start_out) {
            *rs64_fw[fi].ucode_start_out =
                ((ULONGLONG)hdr->ucode_start_addr_hi << 32) |
                hdr->ucode_start_addr_lo;
            pr_info("gc_staging: %s ucode_start = 0x%llx\n",
                    rs64_fw[fi].name,
                    (unsigned long long)*rs64_fw[fi].ucode_start_out);
        }

        /* Save MEC code/data bytes for fallback VRAM copy path */
        if (fi == 2 && dev->hw.gfx.mec_fw_code == NULL) {
            if (code_size > 0 && code_off + code_size <= len) {
                dev->hw.gfx.mec_fw_code = (UCHAR *)malloc(code_size);
                if (dev->hw.gfx.mec_fw_code) {
                    memcpy(dev->hw.gfx.mec_fw_code, buf + code_off, code_size);
                    dev->hw.gfx.mec_fw_code_size = code_size;
                }
            }
            if (data_size > 0 && data_off + data_size <= len) {
                dev->hw.gfx.mec_fw_data = (UCHAR *)malloc(data_size);
                if (dev->hw.gfx.mec_fw_data) {
                    memcpy(dev->hw.gfx.mec_fw_data, buf + data_off, data_size);
                    dev->hw.gfx.mec_fw_data_size = data_size;
                }
            }
        }

        free(buf);
    }

    /* Flush staging buffer writes to VRAM */
    MemoryBarrier();
    gpu_hdp_flush(dev);

    /* ---- Step 4: Write GFX_IMU_RLC_BOOTLOADER_ADDR via direct MMIO ----
     *
     * These registers (gc_base1 + 0x5f81/82/83) are in the pspdec block
     * which PSP protects against SMN indirect reads/writes (returns 0xffffffff
     * via gpu_smn_rreg/wreg).  However, wddm_lite_write_reg32 uses direct
     * MMIO BAR access (BarIndex=0, same as Linux WREG32_SOC15), which bypasses
     * the SMN protection.
     *
     * Value = full GPU MC address of the RLC_G firmware slot:
     *   addr = VRAM_MC_BASE + GC_STAGING_VRAM_OFFSET + slots[RLC_G].offset
     *        = 0x8000000000 + 32MB + 0 = 0x8002000000
     * PSP needs the MC address to locate the staging buffer, not just the
     * VRAM-relative offset.
     */
    ULONGLONG rlc_g_addr = dev->hw.gmc.vram_start +
                            GC_STAGING_VRAM_OFFSET +
                            (ULONGLONG)slots[SOC24_FW_RLC_G_UCODE].offset;
    ULONG rlc_g_size = slots[SOC24_FW_RLC_G_UCODE].size;

    pr_info("gc_staging: writing GFX_IMU_RLC_BOOTLOADER_ADDR = 0x%llx "
            "SIZE = %u via direct MMIO\n",
            (unsigned long long)rlc_g_addr, rlc_g_size);

    /* Write LO first, then HI, then SIZE */
    wddm_lite_write_reg32(dev,
        (dev->hw.ip.gc_base1 + regGFX_IMU_RLC_BOOTLOADER_ADDR_LO) * 4,
        (ULONG)(rlc_g_addr & 0xFFFFFFFF));
    wddm_lite_write_reg32(dev,
        (dev->hw.ip.gc_base1 + regGFX_IMU_RLC_BOOTLOADER_ADDR_HI) * 4,
        (ULONG)((rlc_g_addr >> 32) & 0xFFFFFFFF));
    wddm_lite_write_reg32(dev,
        (dev->hw.ip.gc_base1 + regGFX_IMU_RLC_BOOTLOADER_SIZE) * 4,
        rlc_g_size);

    /* Readback via SMN to verify (PSP-gated so likely 0xffffffff, but log it) */
    ULONG rb_lo = gpu_smn_rreg(dev, dev->hw.ip.gc_base1 +
                                 regGFX_IMU_RLC_BOOTLOADER_ADDR_LO);
    ULONG rb_hi = gpu_smn_rreg(dev, dev->hw.ip.gc_base1 +
                                 regGFX_IMU_RLC_BOOTLOADER_ADDR_HI);
    ULONG rb_sz = gpu_smn_rreg(dev, dev->hw.ip.gc_base1 +
                                 regGFX_IMU_RLC_BOOTLOADER_SIZE);
    pr_info("gc_staging: GFX_IMU_RLC_BOOTLOADER readback (SMN): "
            "ADDR=0x%08x_%08x SIZE=0x%08x%s\n",
            rb_hi, rb_lo, rb_sz,
            (rb_lo == 0xffffffff) ?
                " (PSP-gated, MMIO write may still work)" : " (readable!)");

    /* Unmap — PSP/GPU accesses staging buffer directly via VRAM MC address */
    psp_ring_unmap(dev, staging1, staging_handle1);
    psp_ring_unmap(dev, staging2, staging_handle2);

    pr_info("gc_staging: setup complete — staging buf MC=0x%llx, "
            "GFX_IMU_RLC_BOOTLOADER=0x%llx size=%u\n",
            (unsigned long long)(dev->hw.gmc.vram_start +
                                  GC_STAGING_VRAM_OFFSET),
            (unsigned long long)rlc_g_addr, rlc_g_size);
    return 0;
}

int gpu_psp_load_all_fw(struct WddmLiteDevice *dev, const char *fw_dir)
{
    struct PspRingContext ctx;
    int ret;
    int load_failures = 0;

    if (!dev->hw.psp_sos_alive) {
        pr_err("psp_ring: SOS must be alive before ring init\n");
        return -1;
    }

    if (dev->hw.ip.mp0_base == 0) {
        pr_err("psp_ring: MP0 base not found\n");
        return -1;
    }

    /* === SMU liveness probe: BEFORE we touch the PSP ring ===
     * Test if the VBIOS SMU is still monitoring the mailbox.
     * If it fails here, the VBIOS SMU went to sleep after VBIOS completed
     * and NOTHING we do (ring destroy, TOC, etc.) caused it. */
    if (dev->hw.ip.mp1_base != 0) {
        pr_info("psp_ring: === SMU PROBE: before PSP ring init ===\n");
        ULONG p_msg = mp1_rreg_direct(dev, regMP1_SMN_C2PMSG_66);
        ULONG p_resp = mp1_rreg_direct(dev, regMP1_SMN_C2PMSG_90);
        pr_info("psp_ring: SMU state: C2PMSG_66=0x%08x C2PMSG_90=0x%08x\n",
                p_msg, p_resp);

        /* Clear and send DisallowGfxOff probe */
        mp1_wreg_direct(dev, regMP1_SMN_C2PMSG_66, 0);
        Sleep(1);
        mp1_wreg_direct(dev, regMP1_SMN_C2PMSG_90, 0);
        mp1_wreg_direct(dev, regMP1_SMN_C2PMSG_82, 0);
        mp1_wreg_direct(dev, regMP1_SMN_C2PMSG_66, 0x29);  /* DisallowGfxOff */

        DWORD probe_start = GetTickCount();
        BOOLEAN probe_ok = FALSE;
        for (;;) {
            ULONG pr = mp1_rreg_direct(dev, regMP1_SMN_C2PMSG_90);
            if (pr != 0) {
                pr_info("psp_ring: SMU PROBE RESPONDED! resp=0x%x (%u ms) "
                        "— SMU is ALIVE before PSP ring init\n",
                        pr, (unsigned)(GetTickCount() - probe_start));
                probe_ok = TRUE;
                break;
            }
            if (GetTickCount() - probe_start >= 2000) break;
            Sleep(10);
        }
        if (!probe_ok) {
            pr_warn("psp_ring: SMU PROBE FAILED — SMU not responding BEFORE "
                    "PSP ring init. VBIOS SMU went to sleep after VBIOS.\n");
        }
    }

    /* Initialize ring */
    ret = psp_ring_init(&ctx, dev);
    if (ret != 0)
        return ret;

    /*
     * Loading order (matches tinygrad/amdgpu kernel):
     *   1. PSP TOC — establishes firmware attestation table in PSP
     *   2. SMU firmware — must be loaded before TMR setup
     *   3. TMR setup — skipped for boot_time_tmr (PSP v14.0.3)
     *   4. All other firmware (SDMA, PFP, ME, MEC, IMU, RLC)
     *   5. AUTOLOAD_RLC
     *
     * Without the TOC, all non-SMU firmware loads fail with
     * TEE_ERROR_BAD_PARAMETERS (0xFFFF0006).
     *
     * The PSP TOC can be:
     *   1. Pre-extracted as psp_14_0_3_toc.bin, OR
     *   2. Parsed from psp_14_0_3_sos.bin (component type 4 = PSP_TOC)
     */

    /* === Step 1: Load PSP TOC === */
    pr_info("psp_ring: === Step 1: Loading PSP TOC ===\n");
    {
        ULONG toc_data_size = 0;
        UCHAR *toc_data = NULL;

        /* Try pre-extracted TOC file first */
        toc_data = find_and_read_fw(fw_dir, "psp_14_0_3_toc.bin",
                                      &toc_data_size);

        /* If not found, parse from SOS binary */
        if (!toc_data) {
            ULONG sos_len = 0;
            UCHAR *sos_buf = find_and_read_fw(fw_dir,
                                                "psp_14_0_3_sos.bin",
                                                &sos_len);
            if (sos_buf && sos_len >= 52) {
                /* Parse psp_firmware_header_v2_0:
                 *   +0x18: ucode_array_offset_bytes
                 *   +0x20: psp_fw_bin_count
                 *   +0x24: psp_fw_bin[] (16 bytes each)
                 */
                ULONG ucode_off = *(ULONG *)(sos_buf + 0x18);
                ULONG fw_bin_count = *(ULONG *)(sos_buf + 0x20);

                pr_info("psp_ring: SOS binary: %u bytes, "
                        "ucode_off=0x%x, %u fw entries\n",
                        sos_len, ucode_off, fw_bin_count);

                for (ULONG i = 0; i < fw_bin_count && i < 32; i++) {
                    ULONG desc_off = 36 + i * 16;
                    if (desc_off + 16 > sos_len) break;
                    ULONG fw_type = *(ULONG *)(sos_buf + desc_off);
                    ULONG fw_off = *(ULONG *)(sos_buf + desc_off + 8);
                    ULONG fw_sz = *(ULONG *)(sos_buf + desc_off + 12);

                    if (fw_type == 4) { /* PSP_FW_TYPE_PSP_TOC */
                        ULONG abs_off = fw_off + ucode_off;
                        if (abs_off + fw_sz <= sos_len) {
                            toc_data = (UCHAR *)malloc(fw_sz);
                            if (toc_data) {
                                memcpy(toc_data, sos_buf + abs_off, fw_sz);
                                toc_data_size = fw_sz;
                                pr_info("psp_ring: extracted PSP TOC from "
                                        "SOS: off=0x%x size=%u\n",
                                        abs_off, fw_sz);
                            }
                        }
                        break;
                    }
                }
                free(sos_buf);
            } else {
                pr_warn("psp_ring: psp_14_0_3_sos.bin not found in %s\n",
                        fw_dir);
            }
        }

        if (toc_data && toc_data_size > 0) {
            /* Copy TOC to staging VRAM and submit LOAD_TOC */
            ULONG tmr_size = 0;
            ret = psp_ring_load_toc(&ctx, toc_data, toc_data_size,
                                      &tmr_size);
            if (ret != 0) {
                pr_warn("psp_ring: LOAD_TOC failed (continuing)\n");
                load_failures++;
            } else {
                pr_info("psp_ring: TOC loaded, TMR size=%u (0x%x)\n",
                        tmr_size, tmr_size);
            }
            free(toc_data);
        } else {
            pr_err("psp_ring: PSP TOC not found — firmware loading will "
                   "likely fail with BAD_PARAMETERS (0xFFFF0006)\n");
            pr_err("psp_ring: ensure psp_14_0_3_sos.bin or "
                   "psp_14_0_3_toc.bin is in %s\n", fw_dir);
        }
    }

    /* === Step 2: Load SMU firmware (before TMR) === */
    /* ALWAYS load SMU firmware, even in VBIOS-preserve mode.
     *
     * Key finding: the VBIOS SMU goes to sleep after VBIOS completes and
     * doesn't respond to the mailbox. We MUST load fresh SMU firmware to
     * get a responsive mailbox. The fresh firmware has the combo pptable
     * embedded (board-specific PMIC/I2C addresses).
     *
     * In VBIOS-preserve mode (skip Mode1 reset), the I2C/PMIC hardware
     * stays in the state VBIOS left it (working). Loading fresh SMU on top
     * of this should give us: active mailbox + working PMIC access. */
    {
        pr_info("psp_ring: === Step 2: Loading SMU firmware ===\n");
        ret = psp_load_smu(&ctx, fw_dir);
        if (ret != 0) {
            pr_warn("psp_ring: SMU firmware load failed\n");
            load_failures++;
        }
    }

    /* === Step 3: TMR setup — skipped for boot_time_tmr === */
    pr_info("psp_ring: === Step 3: TMR setup skipped (boot_time_tmr) ===\n");

    /* === Step 4: Load remaining firmware === */

    /* SDMA firmware (GFX12 uses thread-based SDMA_UCODE_TH0).
     * Loaded via LOAD_IP_FW regardless of staging path. */
    pr_info("psp_ring: === Loading SDMA firmware ===\n");
    ret = load_fw_v1(&ctx, fw_dir, "sdma_7_0_1.bin",
                      GFX_FW_TYPE_SDMA_UCODE_TH0, "sdma_7_0_0.bin");
    if (ret != 0) {
        pr_warn("psp_ring: SDMA load failed (continuing)\n");
        load_failures++;
    }

    /* RS64 PFP firmware */
    pr_info("psp_ring: === Loading PFP firmware (RS64) ===\n");
    {
        ULONG pfp_stacks[] = { GFX_FW_TYPE_RS64_PFP_P0_STACK };
        ret = load_fw_rs64(&ctx, fw_dir, "gc_12_0_1_pfp.bin",
                            GFX_FW_TYPE_RS64_PFP, pfp_stacks, 1,
                            &dev->hw.gfx.pfp_ucode_start);
        if (ret != 0) {
            pr_warn("psp_ring: PFP load failed (continuing)\n");
            load_failures++;
        }
    }

    /* RS64 ME firmware */
    pr_info("psp_ring: === Loading ME firmware (RS64) ===\n");
    {
        ULONG me_stacks[] = { GFX_FW_TYPE_RS64_ME_P0_STACK };
        ret = load_fw_rs64(&ctx, fw_dir, "gc_12_0_1_me.bin",
                            GFX_FW_TYPE_RS64_ME, me_stacks, 1,
                            &dev->hw.gfx.me_ucode_start);
        if (ret != 0) {
            pr_warn("psp_ring: ME load failed (continuing)\n");
            load_failures++;
        }
    }

    /* RS64 MEC firmware — also saves code/data for VRAM fallback copy path */
    pr_info("psp_ring: === Loading MEC firmware (RS64) ===\n");
    {
        ULONG mec_stacks[] = { GFX_FW_TYPE_RS64_MEC_P0_STACK };
        ret = load_fw_rs64(&ctx, fw_dir, "gc_12_0_1_mec.bin",
                            GFX_FW_TYPE_RS64_MEC, mec_stacks, 1,
                            &dev->hw.gfx.mec_ucode_start);
        if (ret != 0) {
            pr_warn("psp_ring: MEC load failed (continuing)\n");
            load_failures++;
        }

        /* Save MEC firmware code+data bytes for GART-based loading fallback. */
        {
            ULONG mec_len = 0;
            UCHAR *mec_buf = find_and_read_fw(fw_dir,
                                "gc_12_0_1_mec.bin", &mec_len);
            if (mec_buf && mec_len >= sizeof(struct GfxFirmwareHeaderV2)) {
                struct GfxFirmwareHeaderV2 *hdr =
                    (struct GfxFirmwareHeaderV2 *)mec_buf;
                if (hdr->header_version_major >= 2) {
                    ULONG co = hdr->ucode_offset_bytes;
                    ULONG cs = hdr->ucode_size_bytes;
                    ULONG do_ = hdr->data_offset_bytes;
                    ULONG ds = hdr->data_size_bytes;
                    if (co + cs <= mec_len) {
                        dev->hw.gfx.mec_fw_code = (UCHAR *)malloc(cs);
                        if (dev->hw.gfx.mec_fw_code) {
                            memcpy(dev->hw.gfx.mec_fw_code,
                                   mec_buf + co, cs);
                            dev->hw.gfx.mec_fw_code_size = cs;
                            pr_info("psp_ring: saved MEC code: %u bytes\n",
                                    cs);
                        }
                    }
                    if (ds > 0 && do_ + ds <= mec_len) {
                        dev->hw.gfx.mec_fw_data = (UCHAR *)malloc(ds);
                        if (dev->hw.gfx.mec_fw_data) {
                            memcpy(dev->hw.gfx.mec_fw_data,
                                   mec_buf + do_, ds);
                            dev->hw.gfx.mec_fw_data_size = ds;
                            pr_info("psp_ring: saved MEC data: %u bytes\n",
                                    ds);
                        }
                    }
                }
                free(mec_buf);
            }
        }
    }

    /* IMU firmware (IRAM + DRAM) */
    pr_info("psp_ring: === Loading IMU firmware ===\n");
    {
        ULONG imu_len = 0;
        UCHAR *imu_buf = find_and_read_fw(fw_dir, "gc_12_0_1_imu.bin",
                                            &imu_len);
        if (imu_buf && imu_len >= sizeof(struct ImuFirmwareHeaderV1)) {
            struct ImuFirmwareHeaderV1 *imu =
                (struct ImuFirmwareHeaderV1 *)imu_buf;
            ULONG imu_off = imu->ucode_array_offset_bytes;
            ULONG iram_size = imu->imu_iram_ucode_size_bytes;
            ULONG dram_size = imu->imu_dram_ucode_size_bytes;

            pr_info("psp_ring: IMU: iram_size=%u, dram_size=%u, offset=0x%x\n",
                    iram_size, dram_size, imu_off);

            if (imu_off + iram_size + dram_size <= imu_len) {
                /* Load IRAM */
                ret = psp_ring_load_fw(&ctx, GFX_FW_TYPE_IMU_I,
                                        "imu_iram", imu_buf + imu_off,
                                        iram_size);
                if (ret != 0) {
                    pr_warn("psp_ring: IMU IRAM load failed (continuing)\n");
                    load_failures++;
                }

                /* Load DRAM */
                ret = psp_ring_load_fw(&ctx, GFX_FW_TYPE_IMU_D,
                                        "imu_dram",
                                        imu_buf + imu_off + iram_size,
                                        dram_size);
                if (ret != 0) {
                    pr_warn("psp_ring: IMU DRAM load failed (continuing)\n");
                    load_failures++;
                }
            } else {
                pr_warn("psp_ring: IMU data overflows file\n");
                load_failures++;
            }
            free(imu_buf);
        } else {
            pr_warn("psp_ring: IMU firmware not found (continuing)\n");
            load_failures++;
        }
    }

    /* RLC firmware */
    pr_info("psp_ring: === Loading RLC firmware ===\n");
    {
        ULONG rlc_len = 0;
        UCHAR *rlc_buf = find_and_read_fw(fw_dir, "gc_12_0_1_rlc.bin",
                                            &rlc_len);
        if (rlc_buf && rlc_len >= sizeof(struct CommonFirmwareHeader)) {
            struct CommonFirmwareHeader *rlc_hdr =
                (struct CommonFirmwareHeader *)rlc_buf;
            USHORT minor = rlc_hdr->header_version_minor;

            pr_info("psp_ring: RLC header v%u.%u, hdr_size=%u\n",
                    rlc_hdr->header_version_major, minor,
                    rlc_hdr->header_size_bytes);

            if (minor >= 2) {
                /*
                 * v2.2 RLC: load IRAM and DRAM_BOOT sub-components.
                 * struct rlc_firmware_header_v2_2 offsets (from kernel):
                 *   +0x9C: rlc_iram_ucode_size_bytes
                 *   +0xA0: rlc_iram_ucode_offset_bytes
                 *   +0xA4: rlc_dram_ucode_size_bytes
                 *   +0xA8: rlc_dram_ucode_offset_bytes
                 */
                ULONG rlc_iram_size = 0, rlc_iram_off = 0;
                ULONG rlc_dram_size = 0, rlc_dram_off = 0;

                if (rlc_len >= 0xAC) {
                    memcpy(&rlc_iram_size, rlc_buf + 0x9C, 4);
                    memcpy(&rlc_iram_off, rlc_buf + 0xA0, 4);
                    memcpy(&rlc_dram_size, rlc_buf + 0xA4, 4);
                    memcpy(&rlc_dram_off, rlc_buf + 0xA8, 4);
                }

                pr_info("psp_ring: RLC IRAM: off=0x%x size=%u, "
                        "DRAM: off=0x%x size=%u\n",
                        rlc_iram_off, rlc_iram_size,
                        rlc_dram_off, rlc_dram_size);

                if (rlc_iram_size > 0 && rlc_iram_off + rlc_iram_size <= rlc_len) {
                    ret = psp_ring_load_fw(&ctx, GFX_FW_TYPE_RLC_IRAM,
                                            "rlc_iram",
                                            rlc_buf + rlc_iram_off,
                                            rlc_iram_size);
                    if (ret != 0) {
                        pr_warn("psp_ring: RLC IRAM load failed\n");
                        load_failures++;
                    }
                }

                if (rlc_dram_size > 0 && rlc_dram_off + rlc_dram_size <= rlc_len) {
                    ret = psp_ring_load_fw(&ctx, GFX_FW_TYPE_RLC_DRAM_BOOT,
                                            "rlc_dram_boot",
                                            rlc_buf + rlc_dram_off,
                                            rlc_dram_size);
                    if (ret != 0) {
                        pr_warn("psp_ring: RLC DRAM load failed\n");
                        load_failures++;
                    }
                }
            }

            /* ALWAYS load RLC_G (type 8) from the common header — tinygrad does this
             * unconditionally for ALL header versions. RLC_G is the main RLC firmware.
             * Without it, RLC can't execute, AUTOLOAD can't distribute firmware, and
             * EnableAllSmuFeatures can't power up GFX.
             * Our old code only loaded RLC_G as a fallback for non-v2.2 headers,
             * SKIPPING it for v2.2. This was the root cause of the EnableAll hang! */
            {
                ULONG ucode_off = rlc_hdr->ucode_array_offset_bytes;
                ULONG ucode_sz = rlc_hdr->ucode_size_bytes;
                pr_info("psp_ring: loading RLC_G (type=8, %u bytes at 0x%x)\n",
                        ucode_sz, ucode_off);
                if (ucode_off + ucode_sz <= rlc_len) {
                    ret = psp_ring_load_fw(&ctx, GFX_FW_TYPE_RLC_G,
                                            "rlc_g",
                                            rlc_buf + ucode_off,
                                            ucode_sz);
                    if (ret != 0) {
                        pr_warn("psp_ring: RLC_G load failed\n");
                        load_failures++;
                    }
                }
            }
            free(rlc_buf);
        } else {
            pr_warn("psp_ring: RLC firmware not found\n");
            load_failures++;
        }
    }

    /* MES firmware is loaded AFTER AUTOLOAD_RLC (see below).
     * Critical: loading MES before AUTOLOAD causes AUTOLOAD to distribute
     * MES to TMR and lock IC_BASE. Loading after AUTOLOAD keeps IC_BASE
     * writable so we can point it to our own VRAM/GART copy. */

    /* PSP Trusted Applications (TA — RAS, HDCP, DTM) */
    pr_info("psp_ring: === Loading TA firmware ===\n");
    {
        /* TA uses a v1 header with fw_type 0 (generic PSP TA) */
        ret = load_fw_v1(&ctx, fw_dir, "psp_14_0_3_ta.bin",
                          2 /* GFX_FW_TYPE_PSP_TA */, NULL);
        if (ret != 0) {
            pr_warn("psp_ring: TA load failed (continuing)\n");
            load_failures++;
        }
    }

    /* AUTOLOAD_RLC — triggers asynchronous firmware distribution by RLC.
     * CRITICAL: MES must be loaded AFTER this, not before. Loading MES
     * before AUTOLOAD causes AUTOLOAD to distribute MES to TMR and lock
     * IC_BASE/MDBASE. Loading after keeps IC_BASE writable. */
    pr_info("psp_ring: === Sending AUTOLOAD_RLC ===\n");
    ret = psp_ring_autoload_rlc(&ctx);
    if (ret != 0) {
        pr_warn("psp_ring: AUTOLOAD_RLC failed\n");
        load_failures++;
    } else {
        pr_info("psp_ring: AUTOLOAD_RLC sent OK\n");
    }

    /* === MES firmware: loaded AFTER AUTOLOAD_RLC ===
     * This matches the amdgpu kernel driver and tinygrad sequence.
     * With MES loaded post-AUTOLOAD, IC_BASE is not set by AUTOLOAD
     * and remains writable for us to point at our own VRAM/GART copy. */
#define GFX_FW_TYPE_CP_MES       33
#define GFX_FW_TYPE_MES_STACK_V1 34

    pr_info("psp_ring: === Loading MES KIQ firmware (post-AUTOLOAD) ===\n");
    {
        ULONG mes1_stacks[] = { GFX_FW_TYPE_MES_KIQ_STACK };
        ret = load_fw_rs64(&ctx, fw_dir, "gc_12_0_1_mes1.bin",
                            GFX_FW_TYPE_CP_MES_KIQ, mes1_stacks, 1,
                            &dev->hw.gfx.mes_kiq_ucode_start);
        if (ret != 0) {
            pr_warn("psp_ring: MES KIQ load failed (continuing)\n");
            load_failures++;
        }
    }

    pr_info("psp_ring: === Loading MES firmware (post-AUTOLOAD) ===\n");
    {
        ULONG mes_stacks[] = { GFX_FW_TYPE_MES_STACK_V1 };
        ret = load_fw_rs64(&ctx, fw_dir, "gc_12_0_1_uni_mes.bin",
                            GFX_FW_TYPE_CP_MES, mes_stacks, 1,
                            &dev->hw.gfx.mes_ucode_start);
        if (ret != 0) {
            pr_info("psp_ring: trying legacy gc_12_0_1_mes.bin\n");
            ret = load_fw_rs64(&ctx, fw_dir, "gc_12_0_1_mes.bin",
                                GFX_FW_TYPE_CP_MES, mes_stacks, 1, NULL);
        }
        if (ret != 0) {
            pr_warn("psp_ring: MES load failed (continuing)\n");
            load_failures++;
        }
    }

    pr_info("psp_ring: firmware staging complete "
            "(%d failures)\n", load_failures);

    pr_info("psp_ring: ucode_start: PFP=0x%llx ME=0x%llx MEC=0x%llx\n",
            (unsigned long long)dev->hw.gfx.pfp_ucode_start,
            (unsigned long long)dev->hw.gfx.me_ucode_start,
            (unsigned long long)dev->hw.gfx.mec_ucode_start);

    psp_ring_destroy(&ctx);
    return 0;
}

/*
 * Trigger RLC autoload after firmware staging and GFXOFF disable.
 * Creates a fresh PSP ring, sends AUTOLOAD_RLC, polls bootload status.
 * Must be called AFTER gpu_psp_load_all_fw() and gpu_disable_gfxoff().
 */
int gpu_psp_trigger_autoload(struct WddmLiteDevice *dev)
{
    if (!dev->hw.ip_discovery_done || dev->hw.ip.mp0_base == 0) {
        pr_err("psp_autoload: IP discovery not done\n");
        return -1;
    }

    /* Check bootload status before we do anything */
    ULONG bl_before = gpu_smn_rreg(dev, dev->hw.ip.gc_base1 + 0x4e7c);
    pr_info("psp_autoload: RLC_BOOTLOAD_STATUS before = 0x%08x\n", bl_before);
    if (bl_before & 0x80000000) {
        pr_info("psp_autoload: bootload already complete!\n");
        return 0;
    }

    /* Create a fresh PSP ring for the AUTOLOAD command */
    struct PspRingContext ctx;
    int ret = psp_ring_init(&ctx, dev);
    if (ret != 0) {
        pr_err("psp_autoload: failed to create PSP ring\n");
        return ret;
    }

    /* Disable RLC clock gating and power gating before AUTOLOAD.
     * Linux driver does this in non-PSP path. On VFIO, these may
     * prevent RLC from completing firmware distribution. */
    {
        ULONG cgcg = gpu_smn_rreg(dev,
            dev->hw.ip.gc_base1 + 0x4c49); /* RLC_CGCG_CGLS_CTRL */
        ULONG pg = gpu_smn_rreg(dev,
            dev->hw.ip.gc_base1 + 0x4c43); /* RLC_PG_CNTL */
        pr_info("psp_autoload: RLC_CGCG_CGLS_CTRL = 0x%08x, "
                "RLC_PG_CNTL = 0x%08x\n", cgcg, pg);
        {
            /* Preserve VBIOS upper bits (observed: bit 16 set = GFXOFF prevention
             * by VBIOS) and assert bit 20 (GFXIP_FGCG_OVERRIDE) to force GFXOFF
             * exit via RLC register instead of SMU mailbox.  Linux gfx_v12_0 uses
             * this path in gfx_v12_0_gfxclk_fgcg_override().  Clear lower 16 bits
             * to disable active CGCG/CGLS clock gating that would block RLC. */
            ULONG new_cgcg = (cgcg & 0xFFFF0000) | 0x00100000;
            if (new_cgcg != cgcg) {
                pr_info("psp_autoload: RLC_CGCG_CGLS_CTRL 0x%08x → 0x%08x "
                        "(preserve upper + GFXIP_FGCG_OVERRIDE, clear lower gating)\n",
                        cgcg, new_cgcg);
                gpu_smn_wreg(dev, dev->hw.ip.gc_base1 + 0x4c49, new_cgcg);
            }
            if (pg != 0) {
                pr_info("psp_autoload: clearing RLC_PG_CNTL 0x%08x → 0\n", pg);
                gpu_smn_wreg(dev, dev->hw.ip.gc_base1 + 0x4c43, 0); /* PG=0 */
            }
        }
    }

    /* Enable RLC GPM threads before AUTOLOAD.
     * Linux driver's gfx_v12_0_rlc_backdoor_autoload_enable() enables
     * thread0 and thread1 in RLC_GPM_THREAD_ENABLE (0x4c45) before
     * triggering autoload. Without this, RLC may not fully execute. */
    {
        ULONG gpm_thread = gpu_smn_rreg(dev,
            dev->hw.ip.gc_base1 + 0x4c45);
        pr_info("psp_autoload: RLC_GPM_THREAD_ENABLE (before) = 0x%08x\n",
                gpm_thread);
        /* Set THREAD0_ENABLE (bit 0) and THREAD1_ENABLE (bit 1) */
        gpm_thread |= 0x3;
        gpu_smn_wreg(dev, dev->hw.ip.gc_base1 + 0x4c45, gpm_thread);
        gpm_thread = gpu_smn_rreg(dev,
            dev->hw.ip.gc_base1 + 0x4c45);
        pr_info("psp_autoload: RLC_GPM_THREAD_ENABLE (after)  = 0x%08x\n",
                gpm_thread);
    }

    /* Also ensure RLC_CNTL has RLC_ENABLE */
    {
        ULONG rlc_cntl = gpu_smn_rreg(dev,
            dev->hw.ip.gc_base1 + 0x4c00);
        if ((rlc_cntl & 0x1) == 0) {
            pr_info("psp_autoload: enabling RLC before AUTOLOAD\n");
            gpu_smn_wreg(dev, dev->hw.ip.gc_base1 + 0x4c00, 0x1);
        }
    }

    /* Dump GFXHUB state before AUTOLOAD. */
    {
        ULONG ctx0 = gfxhub_rreg(dev, regGCVM_CONTEXT0_CNTL);
        ULONG pt_lo = gfxhub_rreg(dev,
            regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32);
        ULONG pt_hi = gfxhub_rreg(dev,
            regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32);
        pr_info("psp_autoload: GFXHUB before AUTOLOAD: "
                "CONTEXT0_CNTL=0x%08x PT_BASE=0x%08x_%08x\n",
                ctx0, pt_hi, pt_lo);
    }

    /* IMU state and unhalt.
     * Linux sequence (gfx_v12_0_rlc_backdoor_autoload_enable):
     *   1. Set GFX_IMU_RLC_BOOTLOADER_ADDR_HI/LO/SIZE (PSP may set this for us)
     *   2. Set C2PMSG access control (allow all C2PMSG registers)
     *   3. Unhalt IMU (clear bit 0 of GFX_IMU_CORE_CTRL)
     *   4. Wait for GFX_IMU_GFX_RESET_CTRL bits [4:0] = 0x1f (GFX powered up)
     *   5. Disable GPA mode (CPC/CPG_PSP_DEBUG GPA_OVERRIDE)
     *
     * Without IMU completing power sequence:
     *   - CPC_PSP_DEBUG remains 0xffffffff (CPC domain off)
     *   - RLC stalls at BOOTLOAD_STATUS=0x3F — cannot distribute to MEC
     *
     * Register offsets (gc_base1):
     *   GFX_IMU_CORE_CTRL = 0x40b6, GFX_IMU_GFX_RESET_CTRL = 0x40bc
     *   GFX_IMU_C2PMSG_ACCESS_CTRL0 = 0x4040, _CTRL1 = 0x4041
     *   GFX_IMU_RLC_BOOTLOADER_ADDR_HI = 0x5f81, _LO = 0x5f82, SIZE = 0x5f83
     *   GFX_IMU_I_RAM_ADDR = 0x5f90, GFX_IMU_I_RAM_DATA = 0x5f91 */
    {
        /* Diagnose GFX_IMU_RLC_BOOTLOADER_ADDR — kernel sets this to the
         * VRAM-relative offset of the RLC_G firmware staging buffer before
         * unhalting. PSP LOAD_IP_FW for RLC may have set it already. */
        ULONG bl_hi   = gpu_smn_rreg(dev, dev->hw.ip.gc_base1 + 0x5f81);
        ULONG bl_lo   = gpu_smn_rreg(dev, dev->hw.ip.gc_base1 + 0x5f82);
        ULONG bl_size = gpu_smn_rreg(dev, dev->hw.ip.gc_base1 + 0x5f83);
        pr_info("psp_autoload: GFX_IMU_RLC_BOOTLOADER: "
                "ADDR=0x%08x_%08x SIZE=0x%08x\n", bl_hi, bl_lo, bl_size);

        /* Check if IMU IRAM is loaded by PSP (peek first 4 words). */
        gpu_smn_wreg(dev, dev->hw.ip.gc_base1 + 0x5f90, 0);  /* IRAM addr = 0 */
        ULONG iram0 = gpu_smn_rreg(dev, dev->hw.ip.gc_base1 + 0x5f91);
        ULONG iram1 = gpu_smn_rreg(dev, dev->hw.ip.gc_base1 + 0x5f91);
        ULONG iram2 = gpu_smn_rreg(dev, dev->hw.ip.gc_base1 + 0x5f91);
        ULONG iram3 = gpu_smn_rreg(dev, dev->hw.ip.gc_base1 + 0x5f91);
        pr_info("psp_autoload: GFX_IMU_IRAM[0..3] = 0x%08x 0x%08x 0x%08x 0x%08x%s\n",
                iram0, iram1, iram2, iram3,
                (iram0 == 0 && iram1 == 0) ? " (EMPTY - IRAM not loaded!)" : " (loaded)");

        /* Set C2PMSG access control — allow all C2PMSG registers (kernel does
         * this in imu_v12_0_setup before unhalting the IMU). */
        gpu_smn_wreg(dev, dev->hw.ip.gc_base1 + 0x4040, 0x00ffffff);
        gpu_smn_wreg(dev, dev->hw.ip.gc_base1 + 0x4041, 0x0000ffff);
        pr_info("psp_autoload: C2PMSG_ACCESS_CTRL0/1 set\n");

        ULONG imu_ctrl = gpu_smn_rreg(dev, dev->hw.ip.gc_base1 + 0x40b6);
        ULONG imu_reset = gpu_smn_rreg(dev, dev->hw.ip.gc_base1 + 0x40bc);
        pr_info("psp_autoload: GFX_IMU_CORE_CTRL=0x%08x "
                "GFX_IMU_GFX_RESET_CTRL=0x%08x (halted=%d gc_ready=%d)\n",
                imu_ctrl, imu_reset,
                (int)(imu_ctrl & 0x1),
                (int)((imu_reset & 0x1f) == 0x1f));

        if (imu_ctrl != 0xFFFFFFFF && (imu_ctrl & 0x1)) {
            /* IMU is accessible and halted — unhalt it */
            pr_info("psp_autoload: unhalting IMU "
                    "(clearing GFX_IMU_CORE_CTRL bit 0)...\n");
            gpu_smn_wreg(dev,
                dev->hw.ip.gc_base1 + 0x40b6, imu_ctrl & ~0x1U);

            /* Wait up to 30s for IMU to signal GFX reset complete
             * (GFX_IMU_GFX_RESET_CTRL bits [4:0] all set = 0x1f).
             * Kernel allows 10s; we extend to 30s on VFIO.
             * Log every 5s including CPC_PSP_DEBUG to track power-up. */
            for (int i = 0; i < 30000; i++) {
                imu_reset = gpu_smn_rreg(dev,
                    dev->hw.ip.gc_base1 + 0x40bc);
                if ((imu_reset & 0x1f) == 0x1f) {
                    pr_info("psp_autoload: IMU GFX reset complete "
                            "at %d ms (RESET_CTRL=0x%08x)\n",
                            i, imu_reset);
                    break;
                }
                if (i > 0 && (i % 5000) == 0) {
                    ULONG cpc = gpu_smn_rreg(dev,
                        dev->hw.ip.gc_base1 + 0x5c11);
                    pr_info("psp_autoload: IMU wait [%ds]: "
                            "GFX_RESET_CTRL=0x%08x CPC_PSP_DEBUG=0x%08x\n",
                            i / 1000, imu_reset, cpc);
                }
                Sleep(1);
            }
            pr_info("psp_autoload: GFX_IMU_GFX_RESET_CTRL=0x%08x "
                    "(ready=%s)\n", imu_reset,
                    ((imu_reset & 0x1f) == 0x1f) ? "YES" : "NO");
            /* Re-read CPC status after IMU sequence */
            {
                ULONG cpc = gpu_smn_rreg(dev, dev->hw.ip.gc_base1 + 0x5c11);
                pr_info("psp_autoload: CPC_PSP_DEBUG after IMU wait = 0x%08x%s\n",
                        cpc, (cpc == 0xffffffff) ? " (CPC STILL OFF)" : " (CPC ON!)");
            }
        } else if (imu_ctrl == 0xFFFFFFFF) {
            pr_warn("psp_autoload: IMU registers inaccessible "
                    "(0xffffffff) — GFXOFF still active?\n");
        } else {
            pr_info("psp_autoload: IMU already unhalted "
                    "(GFX_IMU_CORE_CTRL bit 0 = 0)\n");
        }

        /* Disable GPA mode: bypass PSP security on CPC/CPG.
         * Linux: gfx_v12_0_disable_gpa_mode() sets GPA_OVERRIDE (bit 3)
         * in CPC_PSP_DEBUG (0x5c11) and CPG_PSP_DEBUG (0x5c10).
         * Called after IMU start to allow RLC to distribute firmware. */
        ULONG cpc_dbg = gpu_smn_rreg(dev,
            dev->hw.ip.gc_base1 + 0x5c11);
        ULONG cpg_dbg = gpu_smn_rreg(dev,
            dev->hw.ip.gc_base1 + 0x5c10);
        pr_info("psp_autoload: CPC_PSP_DEBUG=0x%08x "
                "CPG_PSP_DEBUG=0x%08x\n", cpc_dbg, cpg_dbg);
        if (cpc_dbg != 0xFFFFFFFF && !(cpc_dbg & 0x8)) {
            gpu_smn_wreg(dev, dev->hw.ip.gc_base1 + 0x5c11,
                         cpc_dbg | 0x8);
            pr_info("psp_autoload: GPA_OVERRIDE set in CPC_PSP_DEBUG\n");
        }
        if (cpg_dbg != 0xFFFFFFFF && !(cpg_dbg & 0x8)) {
            gpu_smn_wreg(dev, dev->hw.ip.gc_base1 + 0x5c10,
                         cpg_dbg | 0x8);
            pr_info("psp_autoload: GPA_OVERRIDE set in CPG_PSP_DEBUG\n");
        }
    }

    pr_info("psp_autoload: === Triggering AUTOLOAD_RLC ===\n");
    ret = psp_ring_autoload_rlc(&ctx);
    if (ret != 0) {
        pr_err("psp_autoload: AUTOLOAD_RLC command failed\n");
        psp_ring_destroy(&ctx);
        return ret;
    }

    /* Read RLC_IMU_BOOTLOAD_ADDR after AUTOLOAD command — this is what RLC
     * uses as a pointer to the firmware staging buffer.
     * regRLC_IMU_BOOTLOAD_ADDR_HI = 0x4e10, _LO = 0x4e11, SIZE = 0x4e12
     * (BASE_IDX=1, gc_base1, NOT PSP-protected unlike the 0x5f81/82 registers)
     * Also read CP_MEC_CNTL (0x0802) to check if MEC is halted. */
    {
        ULONG rlc_buf_hi   = gpu_smn_rreg(dev, dev->hw.ip.gc_base1 + 0x4e10);
        ULONG rlc_buf_lo   = gpu_smn_rreg(dev, dev->hw.ip.gc_base1 + 0x4e11);
        ULONG rlc_buf_size = gpu_smn_rreg(dev, dev->hw.ip.gc_base1 + 0x4e12);
        ULONG mec_cntl     = gpu_smn_rreg(dev, dev->hw.ip.gc_base1 + 0x0802);
        pr_info("psp_autoload: post-AUTOLOAD: "
                "RLC_IMU_BOOTLOAD_ADDR=0x%08x_%08x SIZE=0x%08x "
                "CP_MEC_CNTL=0x%08x (me1_halt=%d)\n",
                rlc_buf_hi, rlc_buf_lo, rlc_buf_size,
                mec_cntl, (int)(mec_cntl & 0x1));
    }

    /* Poll RLC bootload status (gc_base1 + 0x4e7c, bit 31 = complete) */
    pr_info("psp_autoload: polling RLC_RLCS_BOOTLOAD_STATUS (10s timeout)...\n");
    {
        ULONG bootload_status = 0;
        ULONG last_status = 0xFFFFFFFF;
        BOOLEAN bootload_done = FALSE;
        {
            DWORD bl_start = GetTickCount();
            for (;;) {
                bootload_status = gpu_smn_rreg(dev,
                    dev->hw.ip.gc_base1 + 0x4e7c);
                if (bootload_status & 0x80000000) {
                    bootload_done = TRUE;
                    pr_info("psp_autoload: BOOTLOAD COMPLETE at %u ms! "
                            "status=0x%08x\n",
                            (unsigned)(GetTickCount() - bl_start),
                            bootload_status);
                    break;
                }
                DWORD e = GetTickCount() - bl_start;
                if (e >= 10000) break;
                if (bootload_status != last_status) {
                    pr_info("psp_autoload: bootload status changed: "
                            "0x%08x → 0x%08x at %u ms\n",
                            last_status, bootload_status, (unsigned)e);
                    last_status = bootload_status;
                }
                Sleep(50);
            }
        }
        pr_info("psp_autoload: RLC_RLCS_BOOTLOAD_STATUS = 0x%08x "
                "(complete=%s, sec_policy=%d,%d, "
                "iram_loaded=%d, iram_done=%d, "
                "fuse_dist=%d, init_done=%d)\n",
                bootload_status,
                bootload_done ? "YES" : "NO",
                (bootload_status >> 2) & 1,
                (bootload_status >> 3) & 1,
                (bootload_status >> 4) & 1,
                (bootload_status >> 5) & 1,
                (bootload_status >> 0) & 1,
                (bootload_status >> 1) & 1);

        if (!bootload_done) {
            ULONG rlc_cntl = gpu_smn_rreg(dev,
                dev->hw.ip.gc_base1 + 0x4c00);
            ULONG rlc_stat = gpu_smn_rreg(dev,
                dev->hw.ip.gc_base1 + 0x4c04);
            ULONG id_s1 = gpu_smn_rreg(dev,
                dev->hw.ip.gc_base1 + 0x4ec3);
            ULONG id_s2 = gpu_smn_rreg(dev,
                dev->hw.ip.gc_base1 + 0x4ec4);
            pr_warn("psp_autoload: RLC bootload did not complete "
                    "(RLC_CNTL=0x%08x RLC_STAT=0x%08x "
                    "ID_STATUS=0x%08x_0x%08x)\n",
                    rlc_cntl, rlc_stat, id_s2, id_s1);

            /* If RLC is not started, enable it.
             * Linux driver gfx_v12_0_rlc_resume() writes RLC_CNTL
             * after AUTOLOAD to start RLC execution. */
            if ((rlc_cntl & 0x1) == 0) {
                pr_info("psp_autoload: RLC not enabled, writing "
                        "RLC_CNTL=1 to start RLC...\n");
                gpu_smn_wreg(dev, dev->hw.ip.gc_base1 + 0x4c00, 0x1);
                rlc_cntl = gpu_smn_rreg(dev,
                    dev->hw.ip.gc_base1 + 0x4c00);
                pr_info("psp_autoload: RLC_CNTL readback = 0x%08x\n",
                        rlc_cntl);
            }

            /* Re-poll bootload status (120s).
             * EnableAllSmuFeatures on VFIO times out at 120s — GFX
             * power-up may take time after that. Poll unconditionally
             * regardless of whether we just enabled RLC.
             * Every 5s: check GCVM fault status to detect firmware
             * VA page faults. Non-zero fault = PT_BASE issue. */
            pr_info("psp_autoload: re-polling bootload_status (120s)...\n");
            last_status = 0xFFFFFFFF;
            ULONG last_fault = 0xFFFFFFFF;
            {
                DWORD bl2_start = GetTickCount();
                DWORD last_diag_s = 0;
                for (;;) {
                    bootload_status = gpu_smn_rreg(dev,
                        dev->hw.ip.gc_base1 + 0x4e7c);
                    if (bootload_status & 0x80000000) {
                        bootload_done = TRUE;
                        pr_info("psp_autoload: BOOTLOAD COMPLETE "
                                "at %u ms (after RLC enable)! "
                                "status=0x%08x\n",
                                (unsigned)(GetTickCount() - bl2_start),
                                bootload_status);
                        break;
                    }
                    DWORD e = GetTickCount() - bl2_start;
                    if (e >= 120000) break;
                    if (bootload_status != last_status) {
                        pr_info("psp_autoload: bootload status: "
                                "0x%08x → 0x%08x at %u ms\n",
                                last_status, bootload_status, (unsigned)e);
                        last_status = bootload_status;
                    }
                    /* Every 5s real time: diagnostic dump */
                    DWORD e_s = e / 1000;
                    if (e_s >= last_diag_s + 5) {
                        last_diag_s = e_s;
                        ULONG fault = gfxhub_rreg(dev,
                            regGCVM_L2_PROTECTION_FAULT_STATUS_LO32);
                        if (fault != last_fault) {
                            if (fault != 0) {
                                ULONG fa_lo = gfxhub_rreg(dev,
                                    regGCVM_L2_PROTECTION_FAULT_ADDR_LO32);
                                ULONG fa_hi = gfxhub_rreg(dev,
                                    regGCVM_L2_PROTECTION_FAULT_ADDR_HI32);
                                pr_warn("psp_autoload: GCVM FAULT at %us: "
                                        "STATUS=0x%08x ADDR=0x%08x_%08x "
                                        "(firmware VA page fault?)\n",
                                        (unsigned)e_s, fault, fa_hi, fa_lo);
                            }
                            last_fault = fault;
                        }
                        ULONG rlc_s = gpu_smn_rreg(dev,
                            dev->hw.ip.gc_base1 + 0x4c04);
                        ULONG id_s1 = gpu_smn_rreg(dev,
                            dev->hw.ip.gc_base1 + 0x4ec3);
                        ULONG id_s2 = gpu_smn_rreg(dev,
                            dev->hw.ip.gc_base1 + 0x4ec4);
                        ULONG mec_cntl = gpu_smn_rreg(dev,
                            dev->hw.ip.gc_base1 + 0x0802);
                        ULONG cpc_psp = gpu_smn_rreg(dev,
                            dev->hw.ip.gc_base1 + 0x5c11);
                        pr_info("psp_autoload: [%us] bootload=0x%08x "
                                "RLC_STAT=0x%08x GCVM_FAULT=0x%08x "
                                "ID_STATUS=0x%08x_0x%08x MEC_CNTL=0x%08x "
                                "CPC_PSP_DEBUG=0x%08x\n",
                                (unsigned)e_s, bootload_status, rlc_s, fault,
                                id_s2, id_s1, mec_cntl, cpc_psp);
                    }
                    Sleep(200);  /* Poll every 200 ms */
                }
            }
            if (!bootload_done) {
                pr_warn("psp_autoload: RLC bootload still not "
                        "complete after 120s "
                        "(status=0x%08x)\n", bootload_status);
            }
        }
    }

    /* Check CP_STAT */
    if (dev->hw.ip.gc_base != 0) {
        ULONG cp_stat = gpu_smn_rreg(dev, dev->hw.ip.gc_base + 0x0F40);
        pr_info("psp_autoload: CP_STAT = 0x%08x\n", cp_stat);
    }

    /* If bootload didn't complete, try force-writing bit 31.
     * RLC checks this register — force-setting it might unblock
     * subsequent initialization steps that set up page tables. */
    {
    ULONG bl_check = gpu_smn_rreg(dev,
        dev->hw.ip.gc_base1 + 0x4e7c);
    if (!(bl_check & 0x80000000)) {
        ULONG bl_now = gpu_smn_rreg(dev,
            dev->hw.ip.gc_base1 + 0x4e7c);
        pr_info("psp_autoload: attempting force-write bootload "
                "status 0x%08x → 0x%08x\n",
                bl_now, bl_now | 0x80000000);
        gpu_smn_wreg(dev, dev->hw.ip.gc_base1 + 0x4e7c,
                     bl_now | 0x80000000);
        ULONG bl_after = gpu_smn_rreg(dev,
            dev->hw.ip.gc_base1 + 0x4e7c);
        pr_info("psp_autoload: bootload after force-write = 0x%08x "
                "(writable=%s)\n", bl_after,
                (bl_after & 0x80000000) ? "YES" : "NO");
        if (bl_after & 0x80000000) {
            pr_info("psp_autoload: force-set BOOTLOAD_COMPLETE!\n");
        }
    }
    }

    /* Dump GFXHUB state after AUTOLOAD to check if firmware VA
     * page table entries were created */
    {
        ULONG ctx0 = gfxhub_rreg(dev, regGCVM_CONTEXT0_CNTL);
        ULONG pt_lo = gfxhub_rreg(dev,
            regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32);
        ULONG pt_hi = gfxhub_rreg(dev,
            regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32);
        ULONG sys_lo = gfxhub_rreg(dev, 0x1688);
        ULONG sys_hi = gfxhub_rreg(dev, 0x1689);
        pr_info("psp_autoload: post-AUTOLOAD GFXHUB: CNTL=0x%08x "
                "PT_BASE=0x%08x_%08x SYS_APER=0x%08x-0x%08x\n",
                ctx0, pt_hi, pt_lo, sys_lo, sys_hi);
    }

    psp_ring_destroy(&ctx);
    return 0;
}

/* Legacy API wrapper — calls gpu_psp_load_all_fw */
int gpu_psp_load_smu_fw(struct WddmLiteDevice *dev, const char *fw_dir)
{
    return gpu_psp_load_all_fw(dev, fw_dir);
}

/* Non-blocking AUTOLOAD_RLC: just send the PSP command, don't wait.
 * amdgpu sends AUTOLOAD_RLC during psp_load_non_psp_fw and does NOT wait
 * for BOOTLOAD_STATUS. AUTOLOAD completes asynchronously after
 * EnableAllSmuFeatures powers up GFX. */
int gpu_psp_send_autoload_cmd(struct WddmLiteDevice *dev)
{
    struct PspRingContext ctx;
    int ret = psp_ring_init(&ctx, dev);
    if (ret != 0) {
        pr_warn("psp_autoload_cmd: ring init failed\n");
        return ret;
    }

    pr_info("psp_autoload_cmd: sending AUTOLOAD_RLC (non-blocking)...\n");
    ret = psp_ring_autoload_rlc(&ctx);
    if (ret == 0)
        pr_info("psp_autoload_cmd: AUTOLOAD_RLC sent OK\n");
    else
        pr_warn("psp_autoload_cmd: AUTOLOAD_RLC failed\n");

    psp_ring_destroy(&ctx);
    return ret;
}


/* ======================================================================
 * GMC Initialization
 * ====================================================================== */

/* Build a GFX12 GART PTE for a system memory page */
static ULONGLONG build_gart_pte(ULONGLONG bus_addr)
{
    ULONGLONG pte = AMDGPU_PTE_VALID
                  | AMDGPU_PTE_READABLE
                  | AMDGPU_PTE_WRITEABLE
                  | AMDGPU_PTE_EXECUTABLE
                  | AMDGPU_PTE_IS_PTE
                  | AMDGPU_PTE_SYSTEM
                  | AMDGPU_PTE_SNOOPED
                  | GFX12_PTE_MTYPE(MTYPE_UC);
    pte |= (bus_addr & 0x0000FFFFFFFFF000ULL);  /* bits [47:12] = page frame */
    return pte;
}

/* ---- MMHUB init sequence (mmhub_v4_1_0.c) ---- */

static void mmhub_init_gart_aperture(struct WddmLiteDevice *dev)
{
    struct GpuGmcConfig *gmc = &dev->hw.gmc;
    ULONGLONG pt_base = gmc->gart_table_bus_addr | AMDGPU_PTE_VALID;

    mmhub_wreg(dev, regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32,
               (ULONG)(pt_base & 0xFFFFFFFF));
    mmhub_wreg(dev, regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32,
               (ULONG)((pt_base >> 32) & 0xFFFFFFFF));

    mmhub_wreg(dev, regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32,
               (ULONG)((gmc->gart_start >> 12) & 0xFFFFFFFF));
    mmhub_wreg(dev, regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32,
               (ULONG)((gmc->gart_start >> 44) & 0xFFFFFFFF));
    mmhub_wreg(dev, regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32,
               (ULONG)((gmc->gart_end >> 12) & 0xFFFFFFFF));
    mmhub_wreg(dev, regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32,
               (ULONG)((gmc->gart_end >> 44) & 0xFFFFFFFF));
}

static void mmhub_init_system_aperture(struct WddmLiteDevice *dev)
{
    struct GpuGmcConfig *gmc = &dev->hw.gmc;

    /* Disable AGP (bot > top) */
    mmhub_wreg(dev, regMMMC_VM_AGP_BASE, 0);
    mmhub_wreg(dev, regMMMC_VM_AGP_BOT, 0xFFFFFF);
    mmhub_wreg(dev, regMMMC_VM_AGP_TOP, 0);

    /* System aperture bounds */
    mmhub_wreg(dev, regMMMC_VM_SYSTEM_APERTURE_LOW_ADDR,
               (ULONG)(gmc->vram_start >> 18));
    mmhub_wreg(dev, regMMMC_VM_SYSTEM_APERTURE_HIGH_ADDR,
               (ULONG)(gmc->vram_end >> 18));

    /* Default address for unmapped access → dummy page */
    mmhub_wreg(dev, regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB,
               (ULONG)(gmc->dummy_page_bus_addr >> 12));
    mmhub_wreg(dev, regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB,
               (ULONG)(gmc->dummy_page_bus_addr >> 44));

    /* Fault default address → dummy page */
    mmhub_wreg(dev, regMMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32,
               (ULONG)(gmc->dummy_page_bus_addr >> 12));
    mmhub_wreg(dev, regMMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32,
               (ULONG)(gmc->dummy_page_bus_addr >> 44));

    /* MMVM_L2_PROTECTION_FAULT_CNTL: enable all protection fault types.
     * Linux mmhub_v4_1_0_set_fault_enable_default(true) sets all 11 fault
     * enable bits. Without this, the L2 cache may NACK DMA reads silently,
     * causing TransferTableDram2Smu to fail with 0xFF. */
    {
        ULONG cntl = mmhub_rreg(dev, regMMVM_L2_PROTECTION_FAULT_CNTL);
        pr_info("mmhub: PROT_FAULT_CNTL before = 0x%08x\n", cntl);
        /* Enable all protection fault types (bits 0-10):
         * bit 0: RANGE_PROTECTION_FAULT_ENABLE_DEFAULT
         * bit 1: PDE0_PROTECTION_FAULT_ENABLE_DEFAULT
         * bit 2: PDE1_PROTECTION_FAULT_ENABLE_DEFAULT
         * bit 3: PDE2_PROTECTION_FAULT_ENABLE_DEFAULT
         * bit 4: TRANSLATE_FURTHER_PROTECTION_FAULT_ENABLE_DEFAULT
         * bit 5: NACK_PROTECTION_FAULT_ENABLE_DEFAULT
         * bit 6: DUMMY_PAGE_PROTECTION_FAULT_ENABLE_DEFAULT
         * bit 7: VALID_PROTECTION_FAULT_ENABLE_DEFAULT
         * bit 8: READ_PROTECTION_FAULT_ENABLE_DEFAULT
         * bit 9: WRITE_PROTECTION_FAULT_ENABLE_DEFAULT
         * bit 10: EXECUTE_PROTECTION_FAULT_ENABLE_DEFAULT */
        cntl |= 0x7FF;  /* bits 0-10 all set */
        mmhub_wreg(dev, regMMVM_L2_PROTECTION_FAULT_CNTL, cntl);
        pr_info("mmhub: PROT_FAULT_CNTL after  = 0x%08x\n", cntl);
    }

    /* MMVM_L2_PROTECTION_FAULT_CNTL2: enable PTE read retry
     * bit 18: active_page_migration_pte_read_retry=1 (tinygrad) */
    {
        ULONG cntl2 = mmhub_rreg(dev, regMMVM_L2_PROTECTION_FAULT_CNTL2);
        cntl2 |= (1 << 18);
        mmhub_wreg(dev, regMMVM_L2_PROTECTION_FAULT_CNTL2, cntl2);
    }
}

static void mmhub_init_tlb(struct WddmLiteDevice *dev)
{
    ULONG val = mmhub_rreg(dev, regMMMC_VM_MX_L1_TLB_CNTL);

    val |= L1_TLB_ENABLE;
    val = (val & ~L1_TLB_SYSTEM_ACCESS_MODE_MASK) | (3 << 3);
    val |= L1_TLB_ENABLE_ADV_DRIVER_MODEL;
    val &= ~L1_TLB_SYSTEM_APERTURE_UNMAPPED_ACCESS;
    /* Set MTYPE to Uncacheable (0) — matches Linux & tinygrad.
     * MTYPE field is bits [12:9] in the register. Without this, default
     * MTYPE may be cached, causing SMU DMA reads to get stale data
     * while writes (posted/fire-and-forget) succeed. */
    val &= ~(0xF << 9);   /* Clear MTYPE bits [12:9] → UC = 0 */
    val &= ~(0x7 << 13);  /* Clear ECO_BITS [15:13] → 0 (match Linux) */

    mmhub_wreg(dev, regMMMC_VM_MX_L1_TLB_CNTL, val);
    pr_info("mmhub: L1_TLB_CNTL = 0x%08x (MTYPE=UC)\n", val);
}

static void mmhub_init_cache(struct WddmLiteDevice *dev)
{
    ULONG val;

    /* MMVM_L2_CNTL: match tinygrad init_hub for MM */
    val = mmhub_rreg(dev, regMMVM_L2_CNTL);
    val |= (1 << 0);    /* enable_l2_cache */
    val &= ~(1 << 1);   /* disable fragment processing (GFX12) */
    val &= ~(1 << 6);   /* pde_fault_classification=0 */
    val |= (1 << 8);    /* enable_default_page_out_to_system_memory */
    val &= ~(1 << 9);   /* cache_tag_generation_mode=0 */
    val = (val & ~(0x3 << 17)) | (1 << 17);  /* context1_identity_access_mode=1 */
    mmhub_wreg(dev, regMMVM_L2_CNTL, val);

    mmhub_wreg(dev, regMMVM_L2_CNTL2,
               (1 << 0) |   /* INVALIDATE_ALL_L1_TLBS */
               (1 << 1));   /* INVALIDATE_L2_CACHE */

    /* MMVM_L2_CNTL3: Linux starts from regMMVM_L2_CNTL3_DEFAULT=0x80100007.
     * We must preserve the default bits (bit 31, bit 20, bits 0-2) that may
     * contain important hardware configuration. Read-modify-write. */
    {
        ULONG cntl3 = mmhub_rreg(dev, regMMVM_L2_CNTL3);
        pr_info("mmhub: L2_CNTL3 before = 0x%08x\n", cntl3);
        cntl3 |= (1 << 0);    /* l2_cache_4k_associativity */
        cntl3 |= (1 << 12);   /* l2_cache_bigk_associativity */
        cntl3 = (cntl3 & ~(0xF << 15)) | (9 << 15);  /* bank_select=9 */
        cntl3 = (cntl3 & ~(0xF << 20)) | (6 << 20);  /* bigk_fragment_size=6 */
        mmhub_wreg(dev, regMMVM_L2_CNTL3, cntl3);
        pr_info("mmhub: L2_CNTL3 after  = 0x%08x\n", cntl3);
    }

    /* MMVM_L2_CNTL4: Linux starts from regMMVM_L2_CNTL4_DEFAULT=0xc1.
     * Read-modify-write to preserve default bits 6-7. */
    {
        ULONG cntl4 = mmhub_rreg(dev, regMMVM_L2_CNTL4);
        pr_info("mmhub: L2_CNTL4 before = 0x%08x\n", cntl4);
        cntl4 |= (1 << 0);  /* partition_count */
        /* Clear VMC_TAP_PDE/PTE_REQUEST_PHYSICAL (Linux does this) */
        cntl4 &= ~(1 << 6);  /* PDE */
        cntl4 &= ~(1 << 7);  /* PTE */
        mmhub_wreg(dev, regMMVM_L2_CNTL4, cntl4);
        pr_info("mmhub: L2_CNTL4 after  = 0x%08x\n", cntl4);
    }

    /* MMVM_L2_CNTL5: Read-modify-write (Linux starts from 0x3fe0) */
    {
        ULONG cntl5 = mmhub_rreg(dev, regMMVM_L2_CNTL5);
        cntl5 = (cntl5 & ~(0x3FF << 6)) | (0x1FF << 6);  /* walker_priority */
        cntl5 &= ~0x3F;  /* clear smallk_fragment_size (bits 5:0) */
        mmhub_wreg(dev, regMMVM_L2_CNTL5, cntl5);
    }
}

static void mmhub_enable_system_domain(struct WddmLiteDevice *dev)
{
    mmhub_wreg(dev, regMMVM_CONTEXT0_CNTL, VM_CONTEXT_ENABLE_CONTEXT);
}

static void mmhub_disable_identity_aperture(struct WddmLiteDevice *dev)
{
    mmhub_wreg(dev, regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_LO32, 0xFFFFFFFF);
    mmhub_wreg(dev, regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_HI32, 0x0000000F);
    mmhub_wreg(dev, regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_LO32, 0);
    mmhub_wreg(dev, regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_HI32, 0);
    mmhub_wreg(dev, regMMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_LO32, 0);
    mmhub_wreg(dev, regMMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_HI32, 0);
}

static void mmhub_setup_vmid_config(struct WddmLiteDevice *dev)
{
    ULONG max_pfn = (1 << 24) - 1;  /* 48-bit addr >> 12, top 12 bits */

    for (int vmid = 1; vmid <= 15; vmid++) {
        ULONG val = VM_CONTEXT_ENABLE_CONTEXT;
        val |= (3 & 0x3) << 1;      /* PAGE_TABLE_DEPTH = 3 (4-level) */
        val |= (1 << 7);            /* RANGE_PROTECTION_FAULT_ENABLE */
        val |= (1 << 8);            /* DUMMY_PAGE_PROTECTION_FAULT_ENABLE */
        val |= (1 << 9);            /* PDE0_PROTECTION_FAULT_ENABLE */
        val |= (1 << 10);           /* VALID_PROTECTION_FAULT_ENABLE */
        val |= (1 << 11);           /* READ_PROTECTION_FAULT_ENABLE */
        val |= (1 << 12);           /* WRITE_PROTECTION_FAULT_ENABLE */
        val |= (1 << 13);           /* EXECUTE_PROTECTION_FAULT_ENABLE */

        ULONG ctx_reg = regMMVM_CONTEXT1_CNTL + (vmid - 1) * MMHUB_CTX_DISTANCE;
        mmhub_wreg(dev, ctx_reg, val);

        ULONG start_lo = regMMVM_CONTEXT1_PAGE_TABLE_START_ADDR_LO32 +
                          (vmid - 1) * MMHUB_CTX_ADDR_DISTANCE;
        mmhub_wreg(dev, start_lo, 0);
        mmhub_wreg(dev, start_lo + 1, 0);

        ULONG end_lo = regMMVM_CONTEXT1_PAGE_TABLE_END_ADDR_LO32 +
                        (vmid - 1) * MMHUB_CTX_ADDR_DISTANCE;
        mmhub_wreg(dev, end_lo, max_pfn & 0xFFFFFFFF);
        mmhub_wreg(dev, end_lo + 1, 0x0000000F);
    }
}

static void mmhub_program_invalidation(struct WddmLiteDevice *dev)
{
    for (int eng = 0; eng < 18; eng++) {
        ULONG lo_reg = regMMVM_INVALIDATE_ENG0_ADDR_RANGE_LO32 +
                        eng * MMHUB_ENG_ADDR_DISTANCE;
        mmhub_wreg(dev, lo_reg, 0xFFFFFFFF);
        mmhub_wreg(dev, lo_reg + 1, 0x1F);
    }
}

static void mmhub_gart_enable(struct WddmLiteDevice *dev)
{
    mmhub_init_gart_aperture(dev);
    mmhub_init_system_aperture(dev);
    mmhub_init_tlb(dev);
    mmhub_init_cache(dev);
    mmhub_enable_system_domain(dev);
    mmhub_disable_identity_aperture(dev);
    mmhub_setup_vmid_config(dev);
    mmhub_program_invalidation(dev);
}


/* ---- GFXHUB init sequence (gfxhub_v12_0.c) ---- */

static void gfxhub_init_gart_aperture(struct WddmLiteDevice *dev)
{
    struct GpuGmcConfig *gmc = &dev->hw.gmc;
    ULONGLONG pt_base = gmc->gart_table_bus_addr | AMDGPU_PTE_VALID;

    gfxhub_wreg(dev, regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32,
                (ULONG)(pt_base & 0xFFFFFFFF));
    gfxhub_wreg(dev, regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32,
                (ULONG)((pt_base >> 32) & 0xFFFFFFFF));

    gfxhub_wreg(dev, regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32,
                (ULONG)((gmc->gart_start >> 12) & 0xFFFFFFFF));
    gfxhub_wreg(dev, regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32,
                (ULONG)((gmc->gart_start >> 44) & 0xFFFFFFFF));
    gfxhub_wreg(dev, regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32,
                (ULONG)((gmc->gart_end >> 12) & 0xFFFFFFFF));
    gfxhub_wreg(dev, regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32,
                (ULONG)((gmc->gart_end >> 44) & 0xFFFFFFFF));
}

/*
 * Configure GFXHUB VMID 1 (CONTEXT1) with our flat GART page table.
 * Must be called AFTER gfxhub_setup_vmid_config, which sets CONTEXT1_CNTL
 * and full-range START/END. We override START/END to exactly [gart_start,
 * gart_end] so the hardware only translates GART VAs with this PT, and
 * firmware VA (0x7000000003000) correctly faults rather than hitting
 * an out-of-bounds PTE index.
 *
 * CONTEXT0 (VMID 0) is never touched here — AUTOLOAD/RLC configures it
 * with firmware VA mappings that MEC needs for instruction fetch.
 */
static void gfxhub_init_compute_vmid1(struct WddmLiteDevice *dev)
{
    struct GpuGmcConfig *gmc = &dev->hw.gmc;
    ULONGLONG pt_base = gmc->gart_table_bus_addr | AMDGPU_PTE_VALID;

    gfxhub_wreg(dev, regGCVM_CONTEXT1_PAGE_TABLE_BASE_ADDR_LO32,
                (ULONG)(pt_base & 0xFFFFFFFF));
    gfxhub_wreg(dev, regGCVM_CONTEXT1_PAGE_TABLE_BASE_ADDR_HI32,
                (ULONG)((pt_base >> 32) & 0xFFFFFFFF));

    gfxhub_wreg(dev, regGCVM_CONTEXT1_PAGE_TABLE_START_ADDR_LO32,
                (ULONG)((gmc->gart_start >> 12) & 0xFFFFFFFF));
    gfxhub_wreg(dev, regGCVM_CONTEXT1_PAGE_TABLE_START_ADDR_HI32,
                (ULONG)((gmc->gart_start >> 44) & 0xFFFFFFFF));
    gfxhub_wreg(dev, regGCVM_CONTEXT1_PAGE_TABLE_END_ADDR_LO32,
                (ULONG)((gmc->gart_end >> 12) & 0xFFFFFFFF));
    gfxhub_wreg(dev, regGCVM_CONTEXT1_PAGE_TABLE_END_ADDR_HI32,
                (ULONG)((gmc->gart_end >> 44) & 0xFFFFFFFF));
}

static void gfxhub_init_system_aperture(struct WddmLiteDevice *dev)
{
    struct GpuGmcConfig *gmc = &dev->hw.gmc;

    gfxhub_wreg(dev, regGCMC_VM_AGP_BASE, 0);
    gfxhub_wreg(dev, regGCMC_VM_AGP_BOT, 0xFFFFFF);
    gfxhub_wreg(dev, regGCMC_VM_AGP_TOP, 0);

    /* Extend system aperture to include TMR region below VRAM.
     * After AUTOLOAD, MES IC_BASE points to firmware in TMR at addresses
     * like 0x7_f469d000, which is ~192MB below VRAM start (0x8000000000).
     * If the system aperture only covers [vram_start, vram_end], MES
     * instruction fetches from IC_BASE fail because the address falls
     * outside the aperture and VMID 0 has no page table for it.
     *
     * Solution: extend LOW to cover 256MB below VRAM start for TMR.
     * amdgpu does something similar in gmc_v12_0_gart_enable. */
    {
        ULONGLONG aperture_low = gmc->vram_start;
        /* TMR can be several GB below VRAM start (e.g. 0x7f469d000
         * with VRAM at 0x8000000000 = 2.9GB gap). Extend by 4GB. */
        if (aperture_low >= (4ULL * 1024 * 1024 * 1024))
            aperture_low -= (4ULL * 1024 * 1024 * 1024);
        else
            aperture_low = 0;
        gfxhub_wreg(dev, regGCMC_VM_SYSTEM_APERTURE_LOW_ADDR,
                    (ULONG)(aperture_low >> 18));
    }
    /* System aperture HIGH must cover GART range, not just VRAM.
     * VRAM ends at vram_end but GART sits right above it.
     * MEC with VMID=0 uses the system aperture for all accesses,
     * so queue ring buffers in GART must be within the aperture. */
    {
        ULONGLONG aperture_high = gmc->vram_end;
        if (gmc->gart_start + gmc->gart_size > aperture_high)
            aperture_high = gmc->gart_start + gmc->gart_size;
        gfxhub_wreg(dev, regGCMC_VM_SYSTEM_APERTURE_HIGH_ADDR,
                    (ULONG)(aperture_high >> 18));
        pr_info("gpu_gmc: GFXHUB sys aperture HIGH extended: "
                "vram_end=0x%llx gart_end=0x%llx → HIGH=0x%08x (covers to 0x%llx)\n",
                (unsigned long long)gmc->vram_end,
                (unsigned long long)(gmc->gart_start + gmc->gart_size),
                (ULONG)(aperture_high >> 18),
                (unsigned long long)((((ULONGLONG)(ULONG)(aperture_high >> 18)) + 1) << 18));
    }

    gfxhub_wreg(dev, regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB,
                (ULONG)(gmc->dummy_page_bus_addr >> 12));
    gfxhub_wreg(dev, regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB,
                (ULONG)(gmc->dummy_page_bus_addr >> 44));

    gfxhub_wreg(dev, regGCVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32,
                (ULONG)(gmc->dummy_page_bus_addr >> 12));
    gfxhub_wreg(dev, regGCVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32,
                (ULONG)(gmc->dummy_page_bus_addr >> 44));

    /* GCVM_L2_PROTECTION_FAULT_CNTL2: enable PTE read retry for page migration
     * bit 18: active_page_migration_pte_read_retry=1 (tinygrad) */
    {
        ULONG cntl2 = gfxhub_rreg(dev, regGCVM_L2_PROTECTION_FAULT_CNTL2);
        cntl2 |= (1 << 18);  /* ACTIVE_PAGE_MIGRATION_PTE_READ_RETRY */
        gfxhub_wreg(dev, regGCVM_L2_PROTECTION_FAULT_CNTL2, cntl2);
    }
}

static void gfxhub_init_tlb(struct WddmLiteDevice *dev)
{
    ULONG val = gfxhub_rreg(dev, regGCMC_VM_MX_L1_TLB_CNTL);

    val |= L1_TLB_ENABLE;
    val = (val & ~L1_TLB_SYSTEM_ACCESS_MODE_MASK) | (3 << 3);
    val |= L1_TLB_ENABLE_ADV_DRIVER_MODEL;
    val &= ~L1_TLB_SYSTEM_APERTURE_UNMAPPED_ACCESS;

    gfxhub_wreg(dev, regGCMC_VM_MX_L1_TLB_CNTL, val);
}

static void gfxhub_init_cache(struct WddmLiteDevice *dev)
{
    ULONG val;

    /* GCVM_L2_CNTL: match tinygrad init_hub for GC
     * bit 0: enable_l2_cache=1
     * bit 1: enable_l2_fragment_processing=0 (GFX >= 10)
     * bit 6: pde_fault_classification=0
     * bit 8: enable_default_page_out_to_system_memory=1
     * bit 9: l2_pde0_cache_tag_generation_mode=0
     * bits[18:17]: context1_identity_access_mode=1 (shift 17)
     */
    val = gfxhub_rreg(dev, regGCVM_L2_CNTL);
    val |= (1 << 0);    /* enable_l2_cache */
    val &= ~(1 << 1);   /* disable fragment processing (GFX12) */
    val &= ~(1 << 6);   /* pde_fault_classification=0 */
    val |= (1 << 8);    /* enable_default_page_out_to_system_memory */
    val &= ~(1 << 9);   /* cache_tag_generation_mode=0 */
    val = (val & ~(0x3 << 17)) | (1 << 17);  /* context1_identity_access_mode=1 */
    gfxhub_wreg(dev, regGCVM_L2_CNTL, val);

    /* GCVM_L2_CNTL2: invalidate all TLBs and L2 cache */
    gfxhub_wreg(dev, regGCVM_L2_CNTL2, (1 << 0) | (1 << 1));

    /* GCVM_L2_CNTL3: write full value (matching tinygrad)
     * bit 0: l2_cache_4k_associativity=1
     * bit 12: l2_cache_bigk_associativity=1
     * bits[18:15]: bank_select=9
     * bits[23:20]: l2_cache_bigk_fragment_size=6 */
    gfxhub_wreg(dev, regGCVM_L2_CNTL3,
                (1 << 0) | (1 << 12) | (9 << 15) | (6 << 20));

    /* GCVM_L2_CNTL4: partition count (tinygrad sets l2_cache_4k_partition_count=1) */
    gfxhub_wreg(dev, regGCVM_L2_CNTL4, (1 << 0));

    /* GCVM_L2_CNTL5: walker priority (tinygrad: walker_priority_client_id=0x1ff)
     * bits [14:6] = walker_priority_client_id */
    gfxhub_wreg(dev, regGCVM_L2_CNTL5, (0x1FF << 6));
}

static void gfxhub_enable_system_domain(struct WddmLiteDevice *dev)
{
    gfxhub_wreg(dev, regGCVM_CONTEXT0_CNTL, VM_CONTEXT_ENABLE_CONTEXT);
}

static void gfxhub_disable_identity_aperture(struct WddmLiteDevice *dev)
{
    gfxhub_wreg(dev, regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_LO32, 0xFFFFFFFF);
    gfxhub_wreg(dev, regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_HI32, 0x0000000F);
    gfxhub_wreg(dev, regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_LO32, 0);
    gfxhub_wreg(dev, regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_HI32, 0);
    gfxhub_wreg(dev, regGCVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_LO32, 0);
    gfxhub_wreg(dev, regGCVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_HI32, 0);
}

static void gfxhub_setup_vmid_config(struct WddmLiteDevice *dev)
{
    ULONG max_pfn = (1 << 24) - 1;

    for (int vmid = 1; vmid <= 15; vmid++) {
        ULONG val = VM_CONTEXT_ENABLE_CONTEXT;
        val |= (3 & 0x3) << 1;
        val |= (1 << 7) | (1 << 8) | (1 << 9) | (1 << 10);
        val |= (1 << 11) | (1 << 12) | (1 << 13);

        ULONG ctx_reg = regGCVM_CONTEXT1_CNTL + (vmid - 1) * GFXHUB_CTX_DISTANCE;
        gfxhub_wreg(dev, ctx_reg, val);

        ULONG start_lo = regGCVM_CONTEXT1_PAGE_TABLE_START_ADDR_LO32 +
                          (vmid - 1) * GFXHUB_CTX_ADDR_DISTANCE;
        gfxhub_wreg(dev, start_lo, 0);
        gfxhub_wreg(dev, start_lo + 1, 0);

        ULONG end_lo = regGCVM_CONTEXT1_PAGE_TABLE_END_ADDR_LO32 +
                        (vmid - 1) * GFXHUB_CTX_ADDR_DISTANCE;
        gfxhub_wreg(dev, end_lo, max_pfn & 0xFFFFFFFF);
        gfxhub_wreg(dev, end_lo + 1, 0x0000000F);
    }
}

static void gfxhub_program_invalidation(struct WddmLiteDevice *dev)
{
    for (int eng = 0; eng < 18; eng++) {
        ULONG lo_reg = regGCVM_INVALIDATE_ENG0_ADDR_RANGE_LO32 +
                        eng * GFXHUB_ENG_ADDR_DISTANCE;
        gfxhub_wreg(dev, lo_reg, 0xFFFFFFFF);
        gfxhub_wreg(dev, lo_reg + 1, 0x1F);
    }
}

static void gfxhub_gart_enable(struct WddmLiteDevice *dev)
{
    gfxhub_init_gart_aperture(dev);
    gfxhub_init_system_aperture(dev);
    gfxhub_init_tlb(dev);
    gfxhub_init_cache(dev);
    gfxhub_enable_system_domain(dev);
    gfxhub_disable_identity_aperture(dev);
    gfxhub_setup_vmid_config(dev);
    gfxhub_program_invalidation(dev);

    /* Disable CP UTCL1 error halt */
    ULONG val = gfxhub_rreg(dev, regCP_DEBUG);
    val |= (1 << 15);
    gfxhub_wreg(dev, regCP_DEBUG, val);
}


/*
 * Post-AUTOLOAD GFXHUB compute context setup.
 *
 * Unlike gfxhub_gart_enable(), this does NOT touch CONTEXT0 (VMID 0).
 * After AUTOLOAD, RLC has configured CONTEXT0 with firmware VA mappings
 * (0x7000000003000 → TMR) that MEC requires for instruction fetch.
 * Overwriting CONTEXT0 here causes MEC to fetch from the wrong address.
 *
 * Instead we configure CONTEXT1 (VMID 1) with our flat GART page table
 * for compute queue data access. HQDs must use CP_HQD_VMID=1.
 */
static void gfxhub_enable_compute_context(struct WddmLiteDevice *dev)
{
    gfxhub_init_system_aperture(dev);
    gfxhub_init_tlb(dev);
    gfxhub_init_cache(dev);
    gfxhub_disable_identity_aperture(dev);
    gfxhub_setup_vmid_config(dev);   /* VMID 1-15: CNTL + full-range START/END */
    gfxhub_init_compute_vmid1(dev);  /* VMID 1: PAGE_TABLE_BASE + exact GART START/END */
    gfxhub_program_invalidation(dev);

    /* Disable CP UTCL1 error halt */
    ULONG dbg_val = gfxhub_rreg(dev, regCP_DEBUG);
    dbg_val |= (1 << 15);
    gfxhub_wreg(dev, regCP_DEBUG, dbg_val);
}


/*
 * Minimal VMID 1 setup that does NOT invalidate L2 or TLBs.
 *
 * Used when VBIOS AUTOLOAD has already completed and MEC is running.
 * In this state, MEC has active TLB entries for firmware VA
 * (0x7000000003000) in GFXHUB CONTEXT0. Calling gfxhub_init_cache
 * (which writes GCVM_L2_CNTL2 = invalidate_all) would evict those
 * entries. After eviction, GFXHUB walks CONTEXT0 (our flat GART,
 * only covers [gart_start, gart_end]) and firmware VA is out of range
 * → MEC faults and falls through to VBIOS code at 0x4044.
 *
 * This function only writes CONTEXT1 registers and the VMID 1 CNTL
 * bit, which are safe to write without disturbing MEC execution.
 */
static void gfxhub_setup_vmid1_no_invalidate(struct WddmLiteDevice *dev)
{
    /* Enable CONTEXT1 (VMID 1) with same permission bits as gfxhub_setup_vmid_config. */
    ULONG val = VM_CONTEXT_ENABLE_CONTEXT;
    val |= (3 & 0x3) << 1;
    val |= (1 << 7) | (1 << 8) | (1 << 9) | (1 << 10);
    val |= (1 << 11) | (1 << 12) | (1 << 13);
    gfxhub_wreg(dev, regGCVM_CONTEXT1_CNTL, val);

    /* Set VMID 1 PAGE_TABLE_BASE/START/END without global TLB invalidation. */
    gfxhub_init_compute_vmid1(dev);
}


static void flush_gpu_tlb(struct WddmLiteDevice *dev, int vmid,
                          int is_gfxhub);

/* Public wrapper: Initialize GFXHUB with our GART before AUTOLOAD.
 * Called from openclose.cpp after firmware staging wakes GC.
 * AUTOLOAD/RLC may add firmware VA entries to this page table. */
void gpu_gfxhub_init_for_autoload(struct WddmLiteDevice *dev)
{
    pr_info("gpu_gfxhub: initializing GFXHUB GART before AUTOLOAD\n");
    gfxhub_gart_enable(dev);
    flush_gpu_tlb(dev, 0, 1);

    /* Verify */
    ULONG ctx0 = gfxhub_rreg(dev, regGCVM_CONTEXT0_CNTL);
    ULONG pt_lo = gfxhub_rreg(dev,
        regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32);
    ULONG pt_hi = gfxhub_rreg(dev,
        regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32);
    pr_info("gpu_gfxhub: CONTEXT0_CNTL=0x%08x PT_BASE=0x%08x_%08x\n",
            ctx0, pt_hi, pt_lo);
}

/* ---- TLB flush ---- */

static void flush_gpu_tlb(struct WddmLiteDevice *dev, int vmid, int is_gfxhub)
{
    ULONG sem_reg, req_reg, ack_reg;

    /* Use invalidation engine 17 (matches Linux & tinygrad).
     * Engine 0 may conflict with hardware-reserved engines. */
    const ULONG eng_offset = 17;  /* ENG17 */

    /* HDP flush before TLB invalidation (required by Linux & tinygrad) */
    gpu_hdp_flush(dev);

    if (is_gfxhub) {
        sem_reg = regGCVM_INVALIDATE_ENG0_SEM + eng_offset;
        req_reg = regGCVM_INVALIDATE_ENG0_REQ + eng_offset;
        ack_reg = regGCVM_INVALIDATE_ENG0_ACK + eng_offset;
    } else {
        sem_reg = regMMVM_INVALIDATE_ENG0_SEM + eng_offset;
        req_reg = regMMVM_INVALIDATE_ENG0_REQ + eng_offset;
        ack_reg = regMMVM_INVALIDATE_ENG0_ACK + eng_offset;
    }

    /* Acquire semaphore */
    for (int i = 0; i < 10; i++) {
        ULONG val;
        if (is_gfxhub) {
            val = gfxhub_rreg(dev, sem_reg);
            if (val & 0x1) break;
            gfxhub_wreg(dev, sem_reg, 1);
        } else {
            val = mmhub_rreg(dev, sem_reg);
            if (val & 0x1) break;
            mmhub_wreg(dev, sem_reg, 1);
        }
    }

    /* Request invalidation */
    ULONG req = (1 << 0) | ((vmid & 0xF) << 16);
    if (is_gfxhub)
        gfxhub_wreg(dev, req_reg, req);
    else
        mmhub_wreg(dev, req_reg, req);

    /* Poll for completion */
    for (int i = 0; i < 100; i++) {
        ULONG ack;
        if (is_gfxhub)
            ack = gfxhub_rreg(dev, ack_reg);
        else
            ack = mmhub_rreg(dev, ack_reg);
        if (ack & (1 << vmid))
            break;
    }

    /* Release semaphore */
    if (is_gfxhub)
        gfxhub_wreg(dev, sem_reg, 0);
    else
        mmhub_wreg(dev, sem_reg, 0);
}


/* ---- Top-level GMC init ---- */

int gpu_gmc_init(struct WddmLiteDevice *dev)
{
    struct GpuGmcConfig *gmc = &dev->hw.gmc;
    ULONGLONG gart_table_size;

    if (gmc->initialized)
        return 0;

    if (!dev->hw.ip_discovery_done) {
        pr_err("gpu_gmc: IP discovery must be run first\n");
        return -1;
    }

    if (dev->hw.ip.mmhub_base == 0) {
        pr_err("gpu_gmc: MMHUB base not found in IP discovery\n");
        return -1;
    }

    memset(gmc, 0, sizeof(*gmc));
    gmc->gart_size = DEFAULT_GART_SIZE;
    gart_table_size = GART_TABLE_SIZE(gmc->gart_size);

    /* Read FB location from MMHUB */
    ULONG fb_base_reg = mmhub_rreg(dev, regMMMC_VM_FB_LOCATION_BASE);
    ULONG fb_top_reg = mmhub_rreg(dev, regMMMC_VM_FB_LOCATION_TOP);

    gmc->vram_start = (ULONGLONG)fb_base_reg << 24;
    gmc->vram_end = ((ULONGLONG)fb_top_reg << 24) | 0xFFFFFF;

    if (gmc->vram_start == 0 && gmc->vram_end == 0xFFFFFF) {
        /* VBIOS didn't set it — use default */
        gmc->vram_start = 0;
        gmc->vram_end = dev->vram_size - 1;
    }

    /* GART sits after VRAM in MC address space */
    gmc->gart_start = gmc->vram_end + 1;
    gmc->gart_end = gmc->gart_start + gmc->gart_size - 1;

    pr_info("gpu_gmc: VRAM 0x%012llx - 0x%012llx (%llu MB)\n",
            (unsigned long long)gmc->vram_start,
            (unsigned long long)gmc->vram_end,
            (unsigned long long)dev->vram_size / (1024 * 1024));
    pr_info("gpu_gmc: GART 0x%012llx - 0x%012llx (%llu MB)\n",
            (unsigned long long)gmc->gart_start,
            (unsigned long long)gmc->gart_end,
            (unsigned long long)gmc->gart_size / (1024 * 1024));

    /* Allocate GART page table via DMA */
    AMDGPU_ESCAPE_ALLOC_DMA_DATA dma_gart;
    memset(&dma_gart, 0, sizeof(dma_gart));
    dma_gart.Header.Command = AMDGPU_ESCAPE_ALLOC_DMA;
    dma_gart.Header.Size = sizeof(dma_gart);
    dma_gart.Size = gart_table_size;

    if (wddm_lite_escape(dev, &dma_gart, sizeof(dma_gart)) != 0 ||
        dma_gart.CpuAddress == NULL) {
        pr_err("gpu_gmc: failed to allocate GART table (%llu bytes)\n",
               (unsigned long long)gart_table_size);
        return -1;
    }

    gmc->gart_table_bus_addr = dma_gart.BusAddress;
    gmc->gart_table_cpu_addr = dma_gart.CpuAddress;
    gmc->gart_table_handle = dma_gart.AllocationHandle;

    pr_info("gpu_gmc: GART table at bus 0x%012llx, cpu %p (%llu bytes)\n",
            (unsigned long long)gmc->gart_table_bus_addr,
            gmc->gart_table_cpu_addr,
            (unsigned long long)gart_table_size);

    /* Allocate dummy page for fault handling */
    AMDGPU_ESCAPE_ALLOC_DMA_DATA dma_dummy;
    memset(&dma_dummy, 0, sizeof(dma_dummy));
    dma_dummy.Header.Command = AMDGPU_ESCAPE_ALLOC_DMA;
    dma_dummy.Header.Size = sizeof(dma_dummy);
    dma_dummy.Size = 4096;

    if (wddm_lite_escape(dev, &dma_dummy, sizeof(dma_dummy)) != 0 ||
        dma_dummy.CpuAddress == NULL) {
        pr_err("gpu_gmc: failed to allocate dummy page\n");
        goto fail_free_gart;
    }

    gmc->dummy_page_bus_addr = dma_dummy.BusAddress;
    gmc->dummy_page_cpu_addr = dma_dummy.CpuAddress;
    gmc->dummy_page_handle = dma_dummy.AllocationHandle;

    /* Zero out the dummy page */
    memset(gmc->dummy_page_cpu_addr, 0, 4096);

    /* Initialize GART slot allocator */
    gmc->gart_total_slots = (ULONG)(gmc->gart_size / 4096);
    gmc->gart_next_slot = 0;

    /* Fill GART table: all entries point to dummy page */
    {
        ULONGLONG dummy_pte = build_gart_pte(gmc->dummy_page_bus_addr);
        ULONGLONG *table = (ULONGLONG *)gmc->gart_table_cpu_addr;

        for (ULONG i = 0; i < gmc->gart_total_slots; i++)
            table[i] = dummy_pte;
    }

    /* Enable MMHUB GART */
    mmhub_gart_enable(dev);

    /* Flush MMHUB TLB for VMID 0 */
    flush_gpu_tlb(dev, 0, 0);

    pr_info("gpu_gmc: MMHUB GART enabled\n");

    /* Check if GC block is accessible before touching GFXHUB.
     * After mode1_reset, GC is powered off and all GC regs read 0.
     * Writing to dead GFXHUB regs through SMN can cause issues.
     * Tinygrad only inits MM hub before PSP; GC hub is deferred
     * to after DisallowGfxOff powers up GC. */
    {
        ULONG gc_test = gfxhub_rreg(dev, regGCVM_CONTEXT0_CNTL);
        ULONG gc_test2 = gfxhub_rreg(dev, 0x0F40); /* CP_STAT */

        /* Always dump VBIOS GFXHUB page table base (diagnostic) */
        if (gc_test != 0 || gc_test2 != 0) {
            ULONG vbios_pt_lo = gfxhub_rreg(dev,
                regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32);
            ULONG vbios_pt_hi = gfxhub_rreg(dev,
                regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32);
            ULONG vbios_pt_start_lo = gfxhub_rreg(dev,
                regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32);
            ULONG vbios_pt_start_hi = gfxhub_rreg(dev,
                regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32);
            ULONG vbios_pt_end_lo = gfxhub_rreg(dev,
                regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32);
            ULONG vbios_pt_end_hi = gfxhub_rreg(dev,
                regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32);
            pr_info("gpu_gmc: VBIOS GFXHUB page table base = 0x%08x_%08x\n",
                    vbios_pt_hi, vbios_pt_lo);
            pr_info("gpu_gmc: VBIOS GFXHUB PT range: start=0x%08x_%08x "
                    "end=0x%08x_%08x\n",
                    vbios_pt_start_hi, vbios_pt_start_lo,
                    vbios_pt_end_hi, vbios_pt_end_lo);

            /* Dump VBIOS system aperture regs */
            ULONG sys_ap_lo = gfxhub_rreg(dev,
                regGCMC_VM_SYSTEM_APERTURE_LOW_ADDR);
            ULONG sys_ap_hi = gfxhub_rreg(dev,
                regGCMC_VM_SYSTEM_APERTURE_HIGH_ADDR);
            pr_info("gpu_gmc: VBIOS GFXHUB system aperture: "
                    "lo=0x%08x hi=0x%08x\n", sys_ap_lo, sys_ap_hi);
        }

        /* Check HSAKMT_KEEP_VBIOS_GFXHUB — if set, skip GFXHUB reinit
         * to preserve VBIOS/AUTOLOAD page table entries (firmware VA mappings) */
        char keep_gfxhub[32] = {};
        GetEnvironmentVariableA("HSAKMT_KEEP_VBIOS_GFXHUB",
            keep_gfxhub, sizeof(keep_gfxhub));

        if (gc_test == 0 && gc_test2 == 0) {
            pr_info("gpu_gmc: GC block not accessible (regs read 0), "
                    "deferring GFXHUB init\n");
            /* Skip GFXHUB - will be initialized later when GC is powered */
        } else if (keep_gfxhub[0] == '1') {
            pr_info("gpu_gmc: KEEP_VBIOS_GFXHUB=1 — preserving VBIOS GFXHUB "
                    "page tables (CONTEXT0_CNTL=0x%08x)\n", gc_test);
            /* Do NOT call gfxhub_gart_enable — VBIOS page tables may contain
             * firmware VA (0x7000000003000) mappings from AUTOLOAD_RLC */
        } else {
            pr_info("gpu_gmc: GC block accessible (CONTEXT0=0x%08x), "
                    "initializing GFXHUB\n", gc_test);

            /* Clear any stale GFXHUB VM fault status */
            ULONG old_fault = gfxhub_rreg(dev,
                regGCVM_L2_PROTECTION_FAULT_STATUS_LO32);
            if (old_fault != 0) {
                pr_info("gpu_gmc: clearing stale GFXHUB fault "
                        "status=0x%08x\n", old_fault);
                ULONG cntl = gfxhub_rreg(dev,
                    regGCVM_L2_PROTECTION_FAULT_CNTL);
                gfxhub_wreg(dev, regGCVM_L2_PROTECTION_FAULT_CNTL,
                            cntl | (1 << 0));
                gfxhub_wreg(dev, regGCVM_L2_PROTECTION_FAULT_CNTL,
                            (cntl & ~(1 << 0)) | (1 << 1));
            }

            /* Check if CONTEXT0 has RLC/VBIOS page directory config.
             * After AUTOLOAD, RLC sets CONTEXT0_CNTL=0x03fffc0x with page
             * directory bits that map firmware VA (0x7000000003000) → TMR.
             * gfxhub_gart_enable() overwrites this to 0x1, destroying the
             * firmware mappings. MES needs these mappings for IC_BASE access.
             *
             * If CONTEXT0 already has page dir config (upper bits non-zero),
             * use gfxhub_enable_compute_context which only touches VMID 1+
             * and preserves CONTEXT0. */
            if (gc_test & 0xFFFFFFFE) {
                pr_info("gpu_gmc: CONTEXT0 has page dir config (0x%08x), "
                        "preserving for MES firmware access\n", gc_test);
                gfxhub_enable_compute_context(dev);
                flush_gpu_tlb(dev, 1, 1);
            } else {
                /* No page dir config — fresh init or VBIOS minimal state */
                gfxhub_gart_enable(dev);
                flush_gpu_tlb(dev, 0, 1);
            }
            pr_info("gpu_gmc: GFXHUB GART enabled\n");

            /* Check for new faults */
            ULONG new_fault = gfxhub_rreg(dev,
                regGCVM_L2_PROTECTION_FAULT_STATUS_LO32);
            if (new_fault != 0) {
                ULONG faddr_lo = gfxhub_rreg(dev,
                    regGCVM_L2_PROTECTION_FAULT_ADDR_LO32);
                ULONG faddr_hi = gfxhub_rreg(dev,
                    regGCVM_L2_PROTECTION_FAULT_ADDR_HI32);
                pr_warn("gpu_gmc: GFXHUB fault! status=0x%08x "
                        "addr=0x%08x_%08x\n",
                        new_fault, faddr_hi, faddr_lo);
            }
        }
    }

    /* Dump first few GART PTEs for verification */
    {
        ULONGLONG *table = (ULONGLONG *)gmc->gart_table_cpu_addr;
        for (ULONG i = 0; i < 3; i++) {
            pr_info("gpu_gmc: GART PTE[%u] = 0x%016llx\n",
                    i, (unsigned long long)table[i]);
        }
    }

    gmc->initialized = TRUE;
    dev->hw.gmc_initialized = TRUE;

    /* Verify MMHUB registers */
    {
        ULONG ctx0 = mmhub_rreg(dev, regMMVM_CONTEXT0_CNTL);
        ULONG l1_tlb = mmhub_rreg(dev, regMMMC_VM_MX_L1_TLB_CNTL);
        pr_info("gpu_gmc: verify MMHUB CONTEXT0_CNTL=0x%08x "
                "L1_TLB=0x%08x\n", ctx0, l1_tlb);
    }

    return 0;

fail_free_gart:
    {
        AMDGPU_ESCAPE_ALLOC_DMA_DATA free_gart;
        memset(&free_gart, 0, sizeof(free_gart));
        free_gart.Header.Command = AMDGPU_ESCAPE_FREE_DMA;
        free_gart.Header.Size = sizeof(free_gart);
        free_gart.AllocationHandle = gmc->gart_table_handle;
        wddm_lite_escape(dev, &free_gart, sizeof(free_gart));
    }
    return -1;
}

/* ======================================================================
 * GART Mapping API
 * ====================================================================== */

ULONGLONG gpu_gart_map(struct WddmLiteDevice *dev,
                       const ULONGLONG *bus_addrs, ULONG num_pages)
{
    struct GpuGmcConfig *gmc = &dev->hw.gmc;

    if (!gmc->initialized || !gmc->gart_table_cpu_addr) {
        pr_err("gpu_gart_map: GMC not initialized\n");
        return 0;
    }

    if (num_pages == 0)
        return 0;

    /* Bump allocator: grab next N contiguous slots */
    ULONG start_slot = gmc->gart_next_slot;
    if (start_slot + num_pages > gmc->gart_total_slots) {
        pr_err("gpu_gart_map: out of GART slots (need %u, have %u free)\n",
               num_pages, gmc->gart_total_slots - start_slot);
        return 0;
    }

    gmc->gart_next_slot = start_slot + num_pages;

    /* Write PTEs */
    ULONGLONG *table = (ULONGLONG *)gmc->gart_table_cpu_addr;
    for (ULONG i = 0; i < num_pages; i++) {
        table[start_slot + i] = build_gart_pte(bus_addrs[i]);
    }

    /* GPU address = GART aperture start + slot * page_size */
    ULONGLONG gpu_addr = gmc->gart_start + (ULONGLONG)start_slot * 4096;

    /* Flush TLBs so GPU picks up new PTEs */
    flush_gpu_tlb(dev, 0, 0);  /* MMHUB */
    flush_gpu_tlb(dev, 0, 1);  /* GFXHUB */

    pr_info("gpu_gart_map: mapped %u pages at slots %u-%u, GPU addr 0x%012llx\n",
            num_pages, start_slot, start_slot + num_pages - 1,
            (unsigned long long)gpu_addr);

    return gpu_addr;
}

ULONGLONG gpu_gart_map_contig(struct WddmLiteDevice *dev,
                              ULONGLONG bus_addr, ULONGLONG size)
{
    ULONG num_pages = (ULONG)((size + 4095) / 4096);

    /* Build per-page bus address array on stack (max ~256 for 1MB) */
    ULONGLONG bus_addrs[256];
    if (num_pages > 256) {
        pr_err("gpu_gart_map_contig: buffer too large (%u pages)\n", num_pages);
        return 0;
    }

    for (ULONG i = 0; i < num_pages; i++)
        bus_addrs[i] = bus_addr + (ULONGLONG)i * 4096;

    return gpu_gart_map(dev, bus_addrs, num_pages);
}

void gpu_gart_unmap(struct WddmLiteDevice *dev,
                    ULONGLONG gpu_addr, ULONG num_pages)
{
    struct GpuGmcConfig *gmc = &dev->hw.gmc;

    if (!gmc->initialized || !gmc->gart_table_cpu_addr)
        return;

    if (gpu_addr < gmc->gart_start || gpu_addr >= gmc->gart_end)
        return;

    ULONG start_slot = (ULONG)((gpu_addr - gmc->gart_start) / 4096);
    ULONGLONG *table = (ULONGLONG *)gmc->gart_table_cpu_addr;
    ULONGLONG dummy_pte = build_gart_pte(gmc->dummy_page_bus_addr);

    for (ULONG i = 0; i < num_pages; i++) {
        if (start_slot + i < gmc->gart_total_slots)
            table[start_slot + i] = dummy_pte;
    }

    flush_gpu_tlb(dev, 0, 0);
    flush_gpu_tlb(dev, 0, 1);

    pr_info("gpu_gart_unmap: unmapped %u pages at GPU addr 0x%012llx\n",
            num_pages, (unsigned long long)gpu_addr);
}

/* ======================================================================
 * HDP (Host Data Path) flush
 * ====================================================================== */

void gpu_hdp_flush(struct WddmLiteDevice *dev)
{
    /*
     * HDP flush ensures CPU writes to system memory are visible to the GPU.
     * On GFX12/NBIF, the flush register address is obtained from
     * regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL which contains a remapped
     * DWORD offset. Writing 0 to that offset triggers the flush.
     *
     * Fallback: write to HDP_MEM_COHERENCY_FLUSH at NBIO base_index 2
     * offset 0x00F7 (from the Python reference).
     */
    /* regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL uses BASE_IDX=2 (nbio_base2).
     * Previous code used nbio_base (BASE_IDX=0) which gave DWORD 0x0393 → wrong!
     * Correct: nbio_base2 (0x0d20) + 0x0393 = DWORD 0x10B3.
     * Tinygrad: self.adev.reg("regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL").read() // 4 */
    if (dev->hw.ip.nbio_base2 != 0) {
        ULONG remap_offset = (dev->hw.ip.nbio_base2 + 0x0393) * 4;
        ULONG flush_reg = wddm_lite_read_reg32(dev, remap_offset);

        if (flush_reg != 0 && flush_reg != 0xFFFFFFFF) {
            wddm_lite_write_reg32(dev, (flush_reg / 4) * 4, 0);
            return;
        }
        /* First call: log the remap value for debugging */
        static BOOLEAN hdp_logged = FALSE;
        if (!hdp_logged) {
            pr_info("gpu_hdp: remap reg at DWORD 0x%x (base2=0x%x+0x393) = 0x%08x\n",
                    dev->hw.ip.nbio_base2 + 0x0393, dev->hw.ip.nbio_base2, flush_reg);
            /* Also try reading via SMN indirect */
            ULONG smn_val = gpu_smn_rreg(dev, dev->hw.ip.nbio_base2 + 0x0393);
            pr_info("gpu_hdp: remap via SMN = 0x%08x\n", smn_val);
            if (smn_val != 0 && smn_val != 0xFFFFFFFF) {
                pr_info("gpu_hdp: SMN remap works! Using 0x%08x for flush\n", smn_val);
                dev->hw.hdp_flush_addr = smn_val / 4;
            }
            hdp_logged = TRUE;
        }
        /* Try SMN-based flush if direct MMIO remap failed */
        if (dev->hw.hdp_flush_addr != 0) {
            wddm_lite_write_reg32(dev, dev->hw.hdp_flush_addr * 4, 0);
            return;
        }
    }

    /* Fallback: try with nbio_base (BASE_IDX=0) — wrong but try anyway */
    if (dev->hw.ip.nbio_base != 0) {
        ULONG remap_offset = (dev->hw.ip.nbio_base + 0x0393) * 4;
        ULONG flush_reg = wddm_lite_read_reg32(dev, remap_offset);
        if (flush_reg != 0 && flush_reg != 0xFFFFFFFF) {
            wddm_lite_write_reg32(dev, (flush_reg / 4) * 4, 0);
            return;
        }
    }

    pr_debug("gpu_hdp: using fallback flush path\n");
}


/* ======================================================================
 * SMU (System Management Unit) messaging via MP1 C2PMSG mailbox
 *
 * The SMU controls power management features like GFXOFF. We send
 * messages through the MP1 C2PMSG registers (mailbox interface).
 *
 * Protocol (from tinygrad AM_SMU._smu_cmn_send_msg and Linux smu_cmn.c):
 *   1. Write 0 to response register (C2PMSG_90)
 *   2. Write param to param register (C2PMSG_82)
 *   3. Write msg to message register (C2PMSG_66)
 *   4. Poll response register for non-zero (1 = success)
 * ====================================================================== */

/* (MP1 register defs and helpers moved to top of file, after MP0 block) */

/* ======================================================================
 * SMN Indirect Register Access
 *
 * Uses NBIF RSMU INDEX/DATA registers to read/write any register on
 * the System Management Network. This bypasses the direct MMIO BAR
 * path and can reach GC registers even when GFXOFF is active (since
 * SMN is a separate internal bus that doesn't depend on the GC power
 * domain).
 *
 * Register layout (from nbif_6_3_1_offset.h):
 *   regBIF_BX_PF0_RSMU_INDEX    = 0x0000, BASE_IDX=1
 *   regBIF_BX_PF0_RSMU_DATA     = 0x0001, BASE_IDX=1
 *   regBIF_BX_PF0_RSMU_INDEX_HI = 0x0002, BASE_IDX=1
 *
 * For GFX12 (NBIF 6.3.1): nbio_base1 = 0x14
 * So RSMU_INDEX is at DWORD 0x14 (byte 0x50)
 *    RSMU_DATA  is at DWORD 0x15 (byte 0x54)
 *
 * The INDEX register takes a BYTE address (reg * 4).
 * Tinygrad does: self.reg("regBIF_BX_PF0_RSMU_INDEX").write(reg * 4)
 * ====================================================================== */

#define regBIF_BX_PF0_RSMU_INDEX       0x0000  /* BASE_IDX=1 */
#define regBIF_BX_PF0_RSMU_DATA        0x0001  /* BASE_IDX=1 */
#define regBIF_BX_PF0_RSMU_INDEX_HI    0x0002  /* BASE_IDX=1 */

static ULONG nbif1_rreg(struct WddmLiteDevice *dev, ULONG reg)
{
    ULONG offset = (dev->hw.ip.nbio_base1 + reg) * 4;
    return wddm_lite_read_reg32(dev, offset);
}

static void nbif1_wreg(struct WddmLiteDevice *dev, ULONG reg, ULONG val)
{
    ULONG offset = (dev->hw.ip.nbio_base1 + reg) * 4;
    wddm_lite_write_reg32(dev, offset, val);
}

ULONG gpu_smn_rreg(struct WddmLiteDevice *dev, ULONG reg)
{
    if (dev->hw.ip.nbio_base1 == 0) {
        pr_err("gpu_smn: NBIO base1 not found (IP discovery required)\n");
        return 0xFFFFFFFF;
    }

    /* Write byte address to INDEX register */
    nbif1_wreg(dev, regBIF_BX_PF0_RSMU_INDEX, reg * 4);
    /* Read from DATA register */
    return nbif1_rreg(dev, regBIF_BX_PF0_RSMU_DATA);
}

void gpu_smn_wreg(struct WddmLiteDevice *dev, ULONG reg, ULONG val)
{
    if (dev->hw.ip.nbio_base1 == 0) {
        pr_err("gpu_smn: NBIO base1 not found (IP discovery required)\n");
        return;
    }

    /* Write byte address to INDEX register */
    nbif1_wreg(dev, regBIF_BX_PF0_RSMU_INDEX, reg * 4);
    /* Write value to DATA register */
    nbif1_wreg(dev, regBIF_BX_PF0_RSMU_DATA, val);
}

/* ======================================================================
 * Debug SMU Mailbox (for mode1 reset)
 *
 * MP1 C2PMSG registers for debug SMU communication:
 *   C2PMSG_54 = 0x0276 (response, BASE_IDX=0)
 *   C2PMSG_53 = 0x0275 (parameter, BASE_IDX=0)
 *   C2PMSG_75 = 0x028B (message, BASE_IDX=0)
 *
 * __DEBUGSMC_MSG_Mode1Reset = 2 (for MP0 >= v14.0.0)
 * ====================================================================== */

#define regMP1_SMN_C2PMSG_53    0x0275  /* Debug param register */
#define regMP1_SMN_C2PMSG_54    0x0276  /* Debug response register */
#define regMP1_SMN_C2PMSG_75    0x028B  /* Debug message register */

#define DEBUGSMC_MSG_Mode1Reset         2

static int gpu_smu_send_debug_msg(struct WddmLiteDevice *dev, ULONG msg, ULONG param)
{
    if (dev->hw.ip.mp1_base == 0) {
        pr_err("gpu_smu: MP1 base not found\n");
        return -1;
    }

    /* Clear debug response */
    mp1_wreg(dev, regMP1_SMN_C2PMSG_54, 0);
    /* Write debug param */
    mp1_wreg(dev, regMP1_SMN_C2PMSG_53, param);
    /* Write debug message (triggers) */
    mp1_wreg(dev, regMP1_SMN_C2PMSG_75, msg);

    /* Poll debug response (up to 2 seconds wall clock) */
    {
        DWORD start = GetTickCount();
        for (;;) {
            ULONG resp = mp1_rreg(dev, regMP1_SMN_C2PMSG_54);
            if (resp != 0) {
                if (resp == 1) {
                    pr_info("gpu_smu: debug msg %u succeeded\n", msg);
                    return 0;
                } else {
                    pr_warn("gpu_smu: debug msg %u failed (resp=0x%x)\n",
                            msg, resp);
                    return -1;
                }
            }
            if (GetTickCount() - start >= 2000)
                break;
            Sleep(10);
        }
    }

    pr_err("gpu_smu: debug msg %u timed out\n", msg);
    return -1;
}

int gpu_smu_send_msg(struct WddmLiteDevice *dev, ULONG msg, ULONG param)
{
    if (dev->hw.ip.mp1_base == 0) {
        pr_err("gpu_smu: MP1 base not found (IP discovery required)\n");
        return -1;
    }

    /* Step 0: Read current message register — if it already holds our
     * message value (e.g. stale from VBIOS after FLR), the SMU won't
     * detect a write-edge. Clear it to 0 first to guarantee a transition. */
    {
        ULONG cur_msg = mp1_rreg(dev, regMP1_SMN_C2PMSG_66);
        if (cur_msg == msg) {
            pr_info("gpu_smu: C2PMSG_66 already 0x%02x, clearing for edge detect\n", msg);
            mp1_wreg(dev, regMP1_SMN_C2PMSG_66, 0);
            Sleep(1);  /* Give SMU time to see the zero */
        }
    }

    /* Step 1: Clear response register */
    mp1_wreg_direct(dev, regMP1_SMN_C2PMSG_90, 0);

    /* Step 2: Write parameter */
    mp1_wreg_direct(dev, regMP1_SMN_C2PMSG_82, param);

    /* Step 3: Write message via direct MMIO (triggers SMU interrupt).
     * SMN indirect writes update the register but may not trigger the
     * SMU's internal interrupt. Direct BAR0 MMIO does. */
    mp1_wreg_direct(dev, regMP1_SMN_C2PMSG_66, msg);

    /* Step 4: Poll for response (up to 2 seconds wall clock) */
    {
        DWORD start = GetTickCount();
        for (;;) {
            ULONG resp = mp1_rreg(dev, regMP1_SMN_C2PMSG_90);
            if (resp != 0) {
                if (resp == 1) {
                    pr_debug("gpu_smu: msg 0x%02x param 0x%x succeeded\n",
                             msg, param);
                    return 0;
                } else {
                    pr_warn("gpu_smu: msg 0x%02x param 0x%x failed "
                            "(resp=0x%x)\n", msg, param, resp);
                    return -1;
                }
            }
            if (GetTickCount() - start >= 2000)
                break;
            Sleep(10);
        }
    }

    pr_err("gpu_smu: msg 0x%02x param 0x%x timed out\n", msg, param);
    return -1;
}

/*
 * Initialize SMU after firmware loading: send EnableAllSmuFeatures.
 * This is the step tinygrad does in SMU.init_hw() after PSP loads firmware.
 * Returns 0 on success, -1 if SMU is not alive.
 */
int gpu_smu_enable_features(struct WddmLiteDevice *dev)
{
    /* ===================================================================
     * EXACT amdgpu re-init SMU sequence (captured via MMIO sniffer).
     *
     * amdgpu's sequence after Mode1 reset + firmware reload:
     *   1. GetDriverIfVersion (0x03)
     *   2. GetSmuVersion (0x02)
     *   3. SetDriverDramAddrHigh/Low (0x0E/0x0F)
     *   4. SetToolsDramAddrHigh/Low (0x10/0x11)
     *   5. TransferTableSmu2Dram(WATERMARKS=1) (0x12)
     *   6. RunDcBtc (0x36)
     *   7. OverridePcieParameters (0x20)
     *   8. EnableAllSmuFeatures(0) (0x06) → responds in 19ms!
     *
     * Critical: NO DisallowGfxOff, NO IMU unhalt, NO FGCG override,
     * NO Dram2Smu, NO SetAllowedFeaturesMask, NO individual features.
     * The pptable is preserved in TMR from the first amdgpu init.
     * Any extra messages before EnableAllSmuFeatures may corrupt state.
     * =================================================================== */

    if (dev->hw.ip.mp1_base == 0) {
        pr_err("gpu_smu: MP1 base not found\n");
        return -1;
    }

    pr_info("gpu_smu: === amdgpu-style SMU init (no extra messages) ===\n");

    /* Step 1: GetDriverIfVersion + GetSmuVersion */
    {
        int ret = gpu_smu_send_msg(dev, PPSMC_MSG_GetDriverIfVersion, 0);
        if (ret == 0) {
            ULONG if_ver = mp1_rreg(dev, regMP1_SMN_C2PMSG_82);
            pr_info("gpu_smu: GetDriverIfVersion = 0x%08x\n", if_ver);
        } else {
            pr_warn("gpu_smu: GetDriverIfVersion failed — SMU not responding\n");
        }
        ret = gpu_smu_send_msg(dev, PPSMC_MSG_GetSmuVersion, 0);
        if (ret == 0) {
            ULONG ver = mp1_rreg(dev, regMP1_SMN_C2PMSG_82);
            pr_info("gpu_smu: GetSmuVersion = 0x%08x\n", ver);
        } else {
            pr_warn("gpu_smu: GetSmuVersion failed — SMU not alive\n");
            return -1;
        }
    }

    /* Step 2: SetDriverDramAddr (allocate at end of VRAM like amdgpu) */
    {
        ULONG fb_base_reg = mmhub_rreg(dev, regMMMC_VM_FB_LOCATION_BASE);
        ULONG fb_top_reg = mmhub_rreg(dev, regMMMC_VM_FB_LOCATION_TOP);
        ULONGLONG vram_end = ((ULONGLONG)fb_top_reg << 24) + 0xFFFFFF;
        /* Place driver table 64KB before end of VRAM (like amdgpu BO alloc) */
        ULONGLONG smu_table_mc = (vram_end - 0x10000) & ~0xFFFULL;
        ULONG addr_hi = (ULONG)(smu_table_mc >> 32);
        ULONG addr_lo = (ULONG)(smu_table_mc & 0xFFFFFFFF);
        pr_info("gpu_smu: SetDriverDramAddr = 0x%08x_%08x (near VRAM end)\n",
                addr_hi, addr_lo);
        gpu_smu_send_msg(dev, PPSMC_MSG_SetDriverDramAddrHigh, addr_hi);
        gpu_smu_send_msg(dev, PPSMC_MSG_SetDriverDramAddrLow, addr_lo);
    }

    /* Step 3: SetToolsDramAddr */
    {
        ULONG fb_base_reg = mmhub_rreg(dev, regMMMC_VM_FB_LOCATION_BASE);
        ULONG fb_top_reg = mmhub_rreg(dev, regMMMC_VM_FB_LOCATION_TOP);
        ULONGLONG vram_end = ((ULONGLONG)fb_top_reg << 24) + 0xFFFFFF;
        ULONGLONG tools_mc = (vram_end - 0x20000) & ~0xFFFULL;
        gpu_smu_send_msg(dev, PPSMC_MSG_SetToolsDramAddrHigh,
                         (ULONG)(tools_mc >> 32));
        gpu_smu_send_msg(dev, PPSMC_MSG_SetToolsDramAddrLow,
                         (ULONG)(tools_mc & 0xFFFFFFFF));
    }

    /* Step 4: Smu2Dram(WATERMARKS=1) — amdgpu reads watermarks before EnableAll */
    gpu_smu_send_msg(dev, PPSMC_MSG_TransferTableSmu2Dram, 1);

    /* Step 5: RunDcBtc */
    {
        int ret = gpu_smu_send_msg(dev, PPSMC_MSG_RunDcBtc, 0);
        pr_info("gpu_smu: RunDcBtc %s\n", ret == 0 ? "OK" : "failed");
    }

    /* Step 6: OverridePcieParameters (amdgpu sends 0x00020306) */
    gpu_smu_send_msg(dev, 0x20, 0x00020306);

    /* Step 7: EnableAllSmuFeatures(0) — the critical message */
    pr_info("gpu_smu: EnableAllSmuFeatures(param=0)...\n");

    /* SKIP_ENABLE check */
    {
        char skip_enable[32] = {};
        GetEnvironmentVariableA("HSAKMT_SKIP_ENABLE_SMU_FEATURES",
            skip_enable, sizeof(skip_enable));
        if (skip_enable[0] == '1') {
            pr_info("gpu_smu: EnableAllSmuFeatures skipped\n");
            return 0;
        }
    }

    /* NOTE: removed all the old diagnostic code:
     * - DisallowGfxOff probes (corrupt SMU state)
     * - IMU unhalt (not needed before EnableAll — amdgpu doesn't do this)
     * - FGCG override (not needed)
     * - CPC_PSP_DEBUG checks (not needed)
     * - Individual EnableSmuFeatures (partial enables corrupt state)
     * - SetAllowedFeaturesMask (rejected without pptable)
     * - TransferTableDram2Smu attempts (locked after first write)
     * - GetRunningSmuFeatures (not needed before EnableAll)
     * - Debug mailbox probes
     * This matches the EXACT amdgpu message sequence captured via sniffer.
     *
     * IMPORTANT: The old code was OBSOLETED because:
     * 1. amdgpu NEVER sends DisallowGfxOff before EnableAllSmuFeatures
     * 2. amdgpu NEVER sends SetAllowedFeaturesMask on re-init
     * 3. amdgpu NEVER sends Dram2Smu on re-init (pptable in TMR)
     * 4. Individual EnableSmuFeaturesLow/High corrupt the SMU state
     * 5. Extra messages cause EnableAllSmuFeatures to hang */

    {
        /* Try sending via gpu_smu_send_msg first (2s timeout).
         * If it fails, fall back to manual send with long poll. */
        pr_info("gpu_smu: trying EnableAllSmuFeatures via gpu_smu_send_msg...\n");
        int msg_ret = gpu_smu_send_msg(dev, PPSMC_MSG_EnableAllSmuFeatures, 0);
        if (msg_ret == 0) {
            pr_info("gpu_smu: *** EnableAllSmuFeatures SUCCEEDED via send_msg! ***\n");
            /* CRITICAL: EnableAllSmuFeatures ENABLES GFXOFF. We must send
             * DisallowGfxOff immediately to prevent GC power-gating.
             * Without this, GFXOFF kicks in ~200ms later and kills MEC. */
            pr_info("gpu_smu: sending DisallowGfxOff after EnableAllSmuFeatures...\n");
            int gfxoff_ret = gpu_smu_send_msg(dev, PPSMC_MSG_DisallowGfxOff, 0);
            if (gfxoff_ret == 0) {
                pr_info("gpu_smu: DisallowGfxOff SUCCEEDED — GFXOFF disabled\n");
            } else {
                pr_warn("gpu_smu: DisallowGfxOff FAILED (ret=%d) — "
                        "GFXOFF may power-gate GC!\n", gfxoff_ret);
            }
            dev->hw.gfxoff_disabled = TRUE;
            return 0;
        }
        ULONG resp_after = mp1_rreg(dev, regMP1_SMN_C2PMSG_90);
        pr_info("gpu_smu: send_msg returned %d, C2PMSG_90=0x%08x — "
                "falling back to long poll\n", msg_ret, resp_after);

        /* Long poll — clear and resend */
        mp1_wreg_direct(dev, regMP1_SMN_C2PMSG_90, 0);
        mp1_wreg_direct(dev, regMP1_SMN_C2PMSG_82, 0);
        mp1_wreg_direct(dev, regMP1_SMN_C2PMSG_66, PPSMC_MSG_EnableAllSmuFeatures);

        BOOLEAN success = FALSE;
        DWORD timeout_ms = 600000;  /* 10 minutes! Previous test showed response came eventually */
        DWORD start_tick = GetTickCount();
        DWORD last_print_s = 0;
        for (;;) {
            ULONG resp = mp1_rreg(dev, regMP1_SMN_C2PMSG_90);
            if (resp != 0) {
                DWORD elapsed = GetTickCount() - start_tick;
                if (resp == 1) {
                    pr_info("gpu_smu: EnableAllSmuFeatures succeeded! (%u ms)\n",
                            (unsigned)elapsed);
                    success = TRUE;
                } else {
                    pr_warn("gpu_smu: EnableAllSmuFeatures failed "
                            "(resp=0x%x, %u ms)\n", resp, (unsigned)elapsed);
                }
                break;
            }
            DWORD elapsed = GetTickCount() - start_tick;
            if (elapsed >= timeout_ms)
                break;
            DWORD elapsed_s = elapsed / 1000;
            if (elapsed_s > last_print_s && (elapsed_s % 5) == 0) {
                pr_info("gpu_smu: still waiting at %us...\n", (unsigned)elapsed_s);
                last_print_s = elapsed_s;
            }
            Sleep(100);  /* poll every 100 ms — escape overhead makes 1ms wasteful */
        }
        if (!success) {
            DWORD elapsed = GetTickCount() - start_tick;
            pr_info("gpu_smu: EnableAllSmuFeatures timed out (%u ms)\n",
                    (unsigned)elapsed);
            dev->hw.gfxoff_disabled = TRUE;
        }
    }
    return 0;
}

/* Track SMU availability across device open/close cycles */
static BOOLEAN g_smu_unavailable = FALSE;

/* CP_STAT offset (BASE_IDX=0) — used to check GC accessibility */
#define CP_STAT_OFFSET  0x0F40

static BOOLEAN gc_regs_accessible(struct WddmLiteDevice *dev)
{
    /* Read CP_STAT via direct MMIO — if GFXOFF active, returns 0 */
    ULONG cp_stat = gfxhub_rreg(dev, CP_STAT_OFFSET);
    /* Also try reading via SMN indirect */
    ULONG cp_stat_smn = gpu_smn_rreg(dev, dev->hw.ip.gc_base + CP_STAT_OFFSET);

    pr_info("gpu_gfxoff: CP_STAT direct=0x%08x, SMN=0x%08x\n",
            cp_stat, cp_stat_smn);

    /* If either path returns non-zero, GC is alive */
    return (cp_stat != 0 || cp_stat_smn != 0);
}

/*
 * Perform mode1 reset via debug SMU mailbox.
 * On MP0 >= v14, debug mailbox msg=2 is __DEBUGSMC_MSG_Mode1Reset.
 * This fully resets the GPU — SOS dies, bootloader restarts.
 * Caller must reload SOS afterwards.
 */
int gpu_smu_mode1_reset(struct WddmLiteDevice *dev)
{
    if (dev->hw.ip.mp1_base == 0) {
        pr_err("gpu_smu: MP1 base not found for mode1 reset\n");
        return -1;
    }

    pr_info("gpu_smu: === Performing mode1 reset via debug mailbox ===\n");

    /* Send debug mailbox msg=2 (__DEBUGSMC_MSG_Mode1Reset) */
    mp1_wreg(dev, regMP1_SMN_C2PMSG_54, 0);  /* Clear debug resp */
    mp1_wreg(dev, regMP1_SMN_C2PMSG_53, 0);  /* Param = 0 */
    mp1_wreg(dev, regMP1_SMN_C2PMSG_75, 2);  /* msg=2 = Mode1Reset */

    /* Brief wait for command acceptance (don't poll too long —
     * the GPU is about to reset, responses may not come back) */
    for (int i = 0; i < 100; i++) {
        ULONG resp = mp1_rreg(dev, regMP1_SMN_C2PMSG_54);
        if (resp != 0) {
            pr_info("gpu_smu: mode1 reset accepted (resp=0x%x, %d ms)\n",
                    resp, i);
            break;
        }
        Sleep(1);
    }

    /* Wait 500ms for reset to complete (tinygrad does 500ms) */
    pr_info("gpu_smu: waiting 500ms for GPU reset...\n");
    Sleep(500);

    /* Wait for bootloader to come back (C2PMSG_35 = 0x80000000) */
    pr_info("gpu_smu: waiting for bootloader ready...\n");
    for (int i = 0; i < 30000; i++) {  /* Up to 30 seconds */
        ULONG bl = mp0_rreg(dev, regMPASP_SMN_C2PMSG_35);
        if (bl & 0x80000000) {  /* bit 31 = ready (not exact match) */
            pr_info("gpu_smu: bootloader ready after mode1 reset (%d ms, "
                    "C2PMSG_35=0x%08x)\n", 500 + i, bl);
            /* Mark SOS as dead so it gets reloaded */
            dev->hw.psp_sos_alive = FALSE;
            return 0;
        }
        Sleep(1);
    }

    pr_warn("gpu_smu: bootloader not ready after mode1 reset "
            "(C2PMSG_35=0x%08x)\n",
            mp0_rreg(dev, regMPASP_SMN_C2PMSG_35));
    dev->hw.psp_sos_alive = FALSE;
    return -1;
}

int gpu_disable_gfxoff(struct WddmLiteDevice *dev)
{
    if (dev->hw.gfxoff_disabled)
        return 0;

    if (g_smu_unavailable)
        return -1;

    if (!dev->hw.ip_discovery_done) {
        pr_err("gpu_smu: IP discovery must be run first\n");
        return -1;
    }

    /* ---- Step 0: Diagnostic — check PSP SOS and GC register state ---- */

    /* Check PSP SOS alive status */
    if (dev->hw.ip.mp0_base != 0) {
        ULONG sos_status = mp0_rreg(dev, regMPASP_SMN_C2PMSG_81);
        ULONG bl_status = mp0_rreg(dev, regMPASP_SMN_C2PMSG_35);
        pr_info("gpu_psp: MP0 base=0x%x, SOS status=0x%08x (alive=%s), "
                "BL status=0x%08x\n",
                dev->hw.ip.mp0_base, sos_status,
                sos_status ? "YES" : "NO", bl_status);
    } else {
        pr_warn("gpu_psp: MP0 base not found — cannot check PSP status\n");
    }

    pr_info("gpu_gfxoff: checking if GC registers are already accessible...\n");

    /* SMN indirect diagnostic: read a few key registers via SMN */
    if (dev->hw.ip.nbio_base1 != 0) {
        pr_info("gpu_gfxoff: SMN indirect available (RSMU at DWORD 0x%x)\n",
                dev->hw.ip.nbio_base1);

        /* Read CP_STAT via SMN */
        ULONG cp_stat_smn = gpu_smn_rreg(dev, dev->hw.ip.gc_base + CP_STAT_OFFSET);
        pr_info("gpu_gfxoff: CP_STAT via SMN = 0x%08x (addr=0x%x)\n",
                cp_stat_smn, dev->hw.ip.gc_base + CP_STAT_OFFSET);

        /* Read SCRATCH_REG0 (BASE_IDX=1 offset 0x2040) via SMN */
        ULONG scratch_smn = gpu_smn_rreg(dev, dev->hw.ip.gc_base1 + 0x2040);
        pr_info("gpu_gfxoff: SCRATCH_REG0 via SMN = 0x%08x (addr=0x%x)\n",
                scratch_smn, dev->hw.ip.gc_base1 + 0x2040);

        /* Read MMHUB via SMN (control test — should always work) */
        ULONG mmhub_smn = gpu_smn_rreg(dev, dev->hw.ip.mmhub_base + regMMMC_VM_FB_LOCATION_BASE);
        pr_info("gpu_gfxoff: MMHUB FB_LOCATION_BASE via SMN = 0x%08x\n", mmhub_smn);

        /* Write-readback test on SMN: write to GRBM_SCRATCH_REG0 (BASE_IDX=0, offset 0x0DE0) */
        ULONG smn_scratch_addr = dev->hw.ip.gc_base + 0x0DE0;
        ULONG old_val = gpu_smn_rreg(dev, smn_scratch_addr);
        gpu_smn_wreg(dev, smn_scratch_addr, 0xDEADBEEF);
        ULONG new_val = gpu_smn_rreg(dev, smn_scratch_addr);
        pr_info("gpu_gfxoff: SMN GRBM_SCRATCH_REG0 write-readback: "
                "old=0x%08x, wrote 0xDEADBEEF, read=0x%08x\n",
                old_val, new_val);
        /* Restore */
        gpu_smn_wreg(dev, smn_scratch_addr, old_val);
    } else {
        pr_warn("gpu_gfxoff: SMN indirect not available (nbio_base1=0)\n");
    }

    /* Direct MMIO check */
    ULONG cp_stat_direct = gfxhub_rreg(dev, CP_STAT_OFFSET);
    pr_info("gpu_gfxoff: CP_STAT direct MMIO = 0x%08x\n", cp_stat_direct);

    /* ---- Step 0b: Check SMU firmware flags via SMN ---- */
    if (dev->hw.ip.nbio_base1 != 0) {
        /* smnMP1_FIRMWARE_FLAGS = 0x3010024 (byte addr) → DWORD = 0xC04009 */
        ULONG fw_flags = gpu_smn_rreg(dev, 0xC04009);
        pr_info("gpu_smu: MP1_FIRMWARE_FLAGS (SMN 0x3010024) = 0x%08x "
                "(interrupts_enabled=%d)\n",
                fw_flags, (fw_flags >> 0) & 1);

        /* Also try reading C2PMSG_90 via SMN (mp1_base + 0x029A) */
        ULONG c2p90_smn = gpu_smn_rreg(dev,
            dev->hw.ip.mp1_base + regMP1_SMN_C2PMSG_90);
        ULONG c2p90_direct = mp1_rreg(dev, regMP1_SMN_C2PMSG_90);
        pr_info("gpu_smu: C2PMSG_90: direct=0x%08x SMN=0x%08x\n",
                c2p90_direct, c2p90_smn);

        /* Dump a few more MP1 registers via SMN for diagnostics */
        ULONG c2p66_smn = gpu_smn_rreg(dev,
            dev->hw.ip.mp1_base + regMP1_SMN_C2PMSG_66);
        pr_info("gpu_smu: C2PMSG_66 via SMN = 0x%08x\n", c2p66_smn);
    }

    /* ---- Step 1: Try normal SMU mailbox (DisallowGfxOff) ---- */
    {
        ULONG resp_before = mp1_rreg(dev, regMP1_SMN_C2PMSG_90);
        pr_info("gpu_smu: C2PMSG_90 before = 0x%08x\n", resp_before);

        /* Quick SMU alive check (500ms timeout — SMU may need time after POST) */
        mp1_wreg(dev, regMP1_SMN_C2PMSG_90, 0);
        mp1_wreg(dev, regMP1_SMN_C2PMSG_82, 0);
        mp1_wreg(dev, regMP1_SMN_C2PMSG_66, PPSMC_MSG_GetSmuVersion);

        BOOLEAN alive = FALSE;
        for (int i = 0; i < 500; i++) {
            ULONG resp = mp1_rreg(dev, regMP1_SMN_C2PMSG_90);
            if (resp != 0) {
                alive = TRUE;
                ULONG version = mp1_rreg(dev, regMP1_SMN_C2PMSG_82);
                pr_info("gpu_smu: SMU alive, version = 0x%08x (%d ms)\n",
                        version, i);
                break;
            }
            Sleep(1);
        }

        if (alive) {
            pr_info("gpu_smu: sending DisallowGfxOff via normal mailbox...\n");
            int ret = gpu_smu_send_msg(dev, PPSMC_MSG_DisallowGfxOff, 0);
            if (ret == 0) {
                dev->hw.gfxoff_disabled = TRUE;
                pr_info("gpu_smu: GFXOFF disabled via normal mailbox\n");
                Sleep(100);  /* Give GC time to power up */

                /* Verify GC registers are now accessible */
                if (dev->hw.ip.gc_base) {
                    ULONG test_reg = dev->hw.ip.gc_base + 0x0DE0; /* GRBM_SCRATCH_REG0 */
                    ULONG old = wddm_lite_read_reg32(dev, test_reg * 4);
                    wddm_lite_write_reg32(dev, test_reg * 4, 0xCAFEBABE);
                    Sleep(1);
                    ULONG readback = wddm_lite_read_reg32(dev, test_reg * 4);
                    wddm_lite_write_reg32(dev, test_reg * 4, old);
                    pr_info("gpu_gfxoff: post-disable GC test: wrote 0xCAFEBABE, "
                            "read 0x%08x (%s)\n",
                            readback,
                            readback == 0xCAFEBABE ? "GC ACCESSIBLE" : "GC STILL OFF");
                    /* Also try SMN */
                    ULONG smn_val = gpu_smn_rreg(dev, test_reg);
                    pr_info("gpu_gfxoff: post-disable GC via SMN: 0x%08x\n", smn_val);
                }
                return 0;
            }
        } else {
            pr_warn("gpu_smu: GetSmuVersion not responding — "
                    "trying DisallowGfxOff directly\n");
            /* Try sending DisallowGfxOff even without version check */
            int ret = gpu_smu_send_msg(dev, PPSMC_MSG_DisallowGfxOff, 0);
            if (ret == 0) {
                dev->hw.gfxoff_disabled = TRUE;
                pr_info("gpu_smu: GFXOFF disabled (blind send)\n");
                Sleep(50);
                return 0;
            }
            pr_warn("gpu_smu: blind DisallowGfxOff also failed\n");
        }
    }

    /* ---- Step 2: Debug SMU DisallowGfxOff DISABLED ---- */
    /* NOTE: Do NOT send DisallowGfxOff via debug mailbox — it triggers
     * mode1 reset on MP0 >= v14, killing SOS and all GPU state.
     * The debug mailbox only works safely for read-only queries. */
    pr_warn("gpu_smu: skipping debug mailbox DisallowGfxOff (causes reset)\n");

    /* ---- Step 3: Check if SMN can reach GC despite GFXOFF ---- */
    if (dev->hw.ip.nbio_base1 != 0) {
        ULONG cp_stat_smn = gpu_smn_rreg(dev, dev->hw.ip.gc_base + CP_STAT_OFFSET);
        ULONG scratch_smn = gpu_smn_rreg(dev, dev->hw.ip.gc_base + 0x0DE0);

        if (cp_stat_smn != 0 || scratch_smn != 0) {
            pr_info("gpu_gfxoff: GC registers accessible via SMN despite GFXOFF!\n");
            pr_info("gpu_gfxoff: CP_STAT_SMN=0x%08x, SCRATCH_SMN=0x%08x\n",
                    cp_stat_smn, scratch_smn);
            /* GC is actually powered on — mark as accessible */
            dev->hw.gfxoff_disabled = TRUE;
            return 0;
        }

        /* Try writing via SMN to see if writes go through */
        ULONG smn_addr = dev->hw.ip.gc_base + 0x0DE0;  /* GRBM_SCRATCH_REG0 */
        gpu_smn_wreg(dev, smn_addr, 0xCAFE0001);
        ULONG wb = gpu_smn_rreg(dev, smn_addr);
        pr_info("gpu_gfxoff: SMN GRBM_SCRATCH write=0xCAFE0001, read=0x%08x\n", wb);

        if (wb == 0xCAFE0001) {
            pr_info("gpu_gfxoff: GC registers writable via SMN — GFXOFF may not be active\n");
            dev->hw.gfxoff_disabled = TRUE;
            return 0;
        }
    }

    pr_err("gpu_smu: all GFXOFF disable methods failed\n");
    g_smu_unavailable = TRUE;
    return -1;
}

void gpu_enable_gfxoff(struct WddmLiteDevice *dev)
{
    if (!dev->hw.gfxoff_disabled)
        return;

    pr_info("gpu_smu: sending AllowGfxOff...\n");
    gpu_smu_send_msg(dev, PPSMC_MSG_AllowGfxOff, 0);
    dev->hw.gfxoff_disabled = FALSE;
}


/* ======================================================================
 * GFX Engine Initialization
 *
 * Based on tinygrad AM driver (amdev.py, ip.py).
 * For a VBIOS-POST passthrough GPU, PSP has already loaded firmware
 * (RLC, MEC, IMU etc.). We need to:
 * 0. Disable GFXOFF (so GC registers are accessible)
 * 1. Verify RLC autoload completed
 * 2. Init GFXHUB (already done in gpu_gmc_init)
 * 3. Configure SH_MEM for all VMIDs
 * 4. Configure MEC doorbell range
 * 5. Enable MEC
 * ====================================================================== */

/*
 * GC register access via SMN indirect (RSMU_INDEX/DATA).
 * Direct MMIO reads return 0 for GC registers on VFIO passthrough
 * (GFXOFF or BAR mapping issue), but SMN indirect always works.
 */

/* GC base_index 1 register access (for GRBM, RLC, MEC, SH_MEM).
 * Uses SMN indirect by default. For per-pipe registers (MES IC_BASE etc.),
 * use gc1_mmio_wreg/rreg which go through direct MMIO + grbm_select. */
ULONG gc1_rreg(struct WddmLiteDevice *dev, ULONG reg)
{
    return gpu_smn_rreg(dev, dev->hw.ip.gc_base1 + reg);
}

static void gc1_wreg(struct WddmLiteDevice *dev, ULONG reg, ULONG val)
{
    gpu_smn_wreg(dev, dev->hw.ip.gc_base1 + reg, val);
}

/* Direct MMIO access for GC base1 registers.
 * Required for per-pipe registers where grbm_select context matters.
 * SMN indirect bypasses grbm_select, so writes to per-pipe registers
 * don't target the selected pipe. */
static ULONG gc1_mmio_rreg(struct WddmLiteDevice *dev, ULONG reg)
{
    ULONG offset = (dev->hw.ip.gc_base1 + reg) * 4;
    return wddm_lite_read_reg32(dev, offset);
}

static void gc1_mmio_wreg(struct WddmLiteDevice *dev, ULONG reg, ULONG val)
{
    ULONG offset = (dev->hw.ip.gc_base1 + reg) * 4;
    wddm_lite_write_reg32(dev, offset, val);
}

/* Direct MMIO for GC base0 registers (CP_HQD_*, CP_STAT, etc.) */
static ULONG gc0_mmio_rreg(struct WddmLiteDevice *dev, ULONG reg)
{
    ULONG offset = (dev->hw.ip.gc_base + reg) * 4;
    return wddm_lite_read_reg32(dev, offset);
}

static void gc0_mmio_wreg(struct WddmLiteDevice *dev, ULONG reg, ULONG val)
{
    ULONG offset = (dev->hw.ip.gc_base + reg) * 4;
    wddm_lite_write_reg32(dev, offset, val);
}

/* GC base_index 0 register access (for CP_HQD, CP_STAT, GRBM_SOFT_RESET) */
ULONG gc0_rreg(struct WddmLiteDevice *dev, ULONG reg)
{
    return gpu_smn_rreg(dev, dev->hw.ip.gc_base + reg);
}

void gc0_wreg(struct WddmLiteDevice *dev, ULONG reg, ULONG val)
{
    gpu_smn_wreg(dev, dev->hw.ip.gc_base + reg, val);
}


/* ---- GC register offsets (GFX12 / gc_12_0_0) ---- */

/* BASE_IDX=0 registers */
#define regCP_STAT                          0x0f40
#define regGRBM_CNTL                        0x0da0
#define regGRBM_SOFT_RESET                  0x0da8
#define GRBM_SOFT_RESET__SOFT_RESET_CP      (1 << 0)   /* bit 0 */
#define GRBM_SOFT_RESET__SOFT_RESET_CPC     (1 << 18)  /* bit 18 */
#define regGRBM_SCRATCH_REG0                0x0de0
#define regGRBM_SCRATCH_REG6                0x0de6
#define regGRBM_SCRATCH_REG7                0x0de7

/* CP HQD registers (BASE_IDX=0) - for direct MMIO queue programming */
#define regCP_MQD_BASE_ADDR                 0x1fa9
#define regCP_MQD_BASE_ADDR_HI              0x1faa
#define regCP_HQD_ACTIVE                    0x1fab
#define regCP_HQD_VMID                      0x1fac
#define regCP_HQD_PERSISTENT_STATE          0x1fad
#define regCP_HQD_PIPE_PRIORITY             0x1fae
#define regCP_HQD_QUEUE_PRIORITY            0x1faf
#define regCP_HQD_QUANTUM                   0x1fb0
#define regCP_HQD_PQ_BASE                   0x1fb1
#define regCP_HQD_PQ_BASE_HI               0x1fb2
#define regCP_HQD_PQ_RPTR                   0x1fb3
#define regCP_HQD_PQ_RPTR_REPORT_ADDR       0x1fb4
#define regCP_HQD_PQ_RPTR_REPORT_ADDR_HI    0x1fb5
#define regCP_HQD_PQ_WPTR_POLL_ADDR         0x1fb6
#define regCP_HQD_PQ_WPTR_POLL_ADDR_HI      0x1fb7
#define regCP_HQD_PQ_DOORBELL_CONTROL       0x1fb8
#define regCP_HQD_PQ_CONTROL                0x1fba
#define regCP_HQD_IB_CONTROL                0x1fbe
#define regCP_HQD_DEQUEUE_REQUEST           0x1fc1
#define regCP_HQD_HQ_STATUS0                0x1fc9
#define regCP_MQD_CONTROL                   0x1fcb
#define regCP_HQD_EOP_BASE_ADDR             0x1fce
#define regCP_HQD_EOP_BASE_ADDR_HI          0x1fcf
#define regCP_HQD_EOP_CONTROL               0x1fd0
#define regCP_HQD_AQL_CONTROL               0x1fde
#define regCP_HQD_PQ_WPTR_LO                0x1fdf
#define regCP_HQD_PQ_WPTR_HI               0x1fe0
#define regCP_MEC_DOORBELL_RANGE_LOWER      0x1dfc
#define regCP_MEC_DOORBELL_RANGE_UPPER      0x1dfd

/* BASE_IDX=1 registers */
#define regGRBM_GFX_CNTL                    0x0900
#define regSH_MEM_BASES                     0x09e3
#define regSH_MEM_CONFIG                    0x09e4
#define regRLC_SAFE_MODE                    0x0980
#define regRLC_CNTL_GFX12                   0x4c00
#define regRLC_RLCS_BOOTLOAD_STATUS         0x4e7c
#define regCP_MEC_RS64_CNTL                 0x2904
#define regCP_MEC_RS64_PRGRM_CNTR_START     0x2900
#define regCP_MEC_RS64_PRGRM_CNTR_START_HI  0x2938
#define regCP_MEC_RS64_INSTR_PNTR           0x2908
#define regCP_MEC_DC_BASE_LO                0x5870  /* Data segment base (low) */
#define regCP_MEC_DC_BASE_HI                0x5871  /* Data segment base (high) */
#define regCP_MEC_DC_BASE_CNTL              0x290b  /* Data cache base control */
#define regCP_MEC_DC_OP_CNTL                0x290c  /* Data cache operation control */
#define regCP_ME_CNTL                       0x0803

/* BASE_IDX=0 PFP/ME program counter registers */
#define regCP_PFP_PRGRM_CNTR_START          0x1e44
#define regCP_PFP_PRGRM_CNTR_START_HI       0x1e59
#define regCP_ME_PRGRM_CNTR_START           0x1e45
#define regCP_ME_PRGRM_CNTR_START_HI        0x1e79

/* SH_MEM_CONFIG field values (from amd_shared.h) */
#define SH_MEM_ADDRESS_MODE_64              1
#define SH_MEM_ALIGNMENT_MODE_UNALIGNED     3

/* CP_MEC_RS64_CNTL bit definitions */
#define CP_MEC_RS64_CNTL__MEC_INVALIDATE_ICACHE  (1 << 4)
#define CP_MEC_RS64_CNTL__MEC_PIPE0_RESET       (1 << 16)
#define CP_MEC_RS64_CNTL__MEC_PIPE1_RESET       (1 << 17)
#define CP_MEC_RS64_CNTL__MEC_PIPE2_RESET       (1 << 18)
#define CP_MEC_RS64_CNTL__MEC_PIPE3_RESET       (1 << 19)
#define CP_MEC_RS64_CNTL__MEC_PIPE0_ACTIVE      (1 << 26)
#define CP_MEC_RS64_CNTL__MEC_PIPE1_ACTIVE      (1 << 27)
#define CP_MEC_RS64_CNTL__MEC_PIPE2_ACTIVE      (1 << 28)
#define CP_MEC_RS64_CNTL__MEC_PIPE3_ACTIVE      (1 << 29)
#define CP_MEC_RS64_CNTL__MEC_HALT              (1 << 30)

/* Combined masks for all pipes */
#define CP_MEC_RS64_CNTL__ALL_PIPE_RESET  \
    (CP_MEC_RS64_CNTL__MEC_PIPE0_RESET | CP_MEC_RS64_CNTL__MEC_PIPE1_RESET | \
     CP_MEC_RS64_CNTL__MEC_PIPE2_RESET | CP_MEC_RS64_CNTL__MEC_PIPE3_RESET)
#define CP_MEC_RS64_CNTL__ALL_PIPE_ACTIVE \
    (CP_MEC_RS64_CNTL__MEC_PIPE0_ACTIVE | CP_MEC_RS64_CNTL__MEC_PIPE1_ACTIVE | \
     CP_MEC_RS64_CNTL__MEC_PIPE2_ACTIVE | CP_MEC_RS64_CNTL__MEC_PIPE3_ACTIVE)

/* CP_ME_CNTL bit definitions (GFX12) */
#define CP_ME_CNTL__PFP_PIPE0_RESET         (1 << 18)
#define CP_ME_CNTL__ME_PIPE0_RESET          (1 << 20)

/* CP_HQD_PQ_CONTROL bits */
#define PQ_CONTROL_RPTR_BLOCK_SIZE_SHIFT    8
#define PQ_CONTROL_UNORD_DISPATCH           (1 << 28)
#define PQ_CONTROL_PRIV_STATE               (1 << 30)
#define PQ_CONTROL_KMD_QUEUE                (1 << 31)

/* CP_HQD_PQ_DOORBELL_CONTROL bits */
#define DOORBELL_EN                         (1 << 30)
#define DOORBELL_OFFSET_SHIFT               2

/* CP_MQD_CONTROL bits */
#define MQD_CONTROL_PRIV_STATE              (1 << 8)

/* CP_HQD_PERSISTENT_STATE bits */
#define PERSISTENT_STATE_PRELOAD_REQ        (1 << 0)
#define PERSISTENT_STATE_PRELOAD_SIZE_SHIFT 8

/* CP_HQD_IB_CONTROL bits */
#define IB_CONTROL_MIN_IB_AVAIL_SIZE_SHIFT  20

/* Doorbell offsets (from amdgpu_doorbell.h) */
#define AMDGPU_NAVI10_DOORBELL_MEC_RING0    0x3


/* ---- GRBM select (target specific ME/pipe/queue for HQD writes) ---- */

void grbm_select(struct WddmLiteDevice *dev,
                         ULONG me, ULONG pipe, ULONG queue, ULONG vmid)
{
    ULONG val = 0;
    val |= (pipe & 0x3);           /* PIPEID [1:0] */
    val |= (me & 0x3) << 2;       /* MEID [3:2] */
    val |= (vmid & 0xF) << 4;     /* VMID [7:4] */
    val |= (queue & 0x7) << 8;    /* QUEUEID [10:8] */

    /* Use direct MMIO for GRBM_GFX_CNTL — SMN indirect doesn't affect
     * the per-pipe register routing for subsequent MMIO accesses. */
    gc1_mmio_wreg(dev, regGRBM_GFX_CNTL, val);
}

void grbm_select_reset(struct WddmLiteDevice *dev)
{
    gc1_mmio_wreg(dev, regGRBM_GFX_CNTL, 0);
}


/* ---- RLC status check ---- */

static int check_rlc_ready(struct WddmLiteDevice *dev)
{
    struct GpuGfxState *gfx = &dev->hw.gfx;

    /* Read CP_STAT — should be 0 when CP is idle */
    gfx->cp_stat = gc0_rreg(dev, regCP_STAT);
    ULONG cp_stat_smn = gpu_smn_rreg(dev, dev->hw.ip.gc_base + regCP_STAT);

    /* Read RLC bootload status */
    gfx->rlc_bootload_status = gc1_rreg(dev, regRLC_RLCS_BOOTLOAD_STATUS);
    ULONG bl_smn = gpu_smn_rreg(dev,
        dev->hw.ip.gc_base1 + regRLC_RLCS_BOOTLOAD_STATUS);

    pr_info("gpu_gfx: CP_STAT direct=0x%08x, SMN=0x%08x\n",
            gfx->cp_stat, cp_stat_smn);
    pr_info("gpu_gfx: RLC_BOOTLOAD_STATUS direct=0x%08x, SMN=0x%08x\n",
            gfx->rlc_bootload_status, bl_smn);

    /* Prefer SMN value if direct MMIO returns suspicious values */
    if (gfx->rlc_bootload_status == 0 && bl_smn != 0) {
        pr_info("gpu_gfx: using SMN value for bootload status\n");
        gfx->rlc_bootload_status = bl_smn;
    }
    if (gfx->cp_stat == 0xFFFFFFFF && cp_stat_smn != 0xFFFFFFFF) {
        pr_info("gpu_gfx: using SMN value for CP_STAT\n");
        gfx->cp_stat = cp_stat_smn;
    }

    /*
     * bootload_complete is bit 31 of RLC_RLCS_BOOTLOAD_STATUS.
     * Other bits: fuse_dist(0), init_done(1), security_policy(2,3),
     *             iram_loaded(4), iram_done(5)
     */
    BOOLEAN bootload_complete = (gfx->rlc_bootload_status & 0x80000000) != 0;

    pr_info("gpu_gfx: bootload bits: complete=%d, fuse_dist=%d, "
            "init_done=%d, iram_loaded=%d, iram_done=%d\n",
            (gfx->rlc_bootload_status >> 31) & 1,
            (gfx->rlc_bootload_status >> 0) & 1,
            (gfx->rlc_bootload_status >> 1) & 1,
            (gfx->rlc_bootload_status >> 4) & 1,
            (gfx->rlc_bootload_status >> 5) & 1);

    if (bootload_complete) {
        pr_info("gpu_gfx: RLC autoload complete\n");
        gfx->rlc_ready = TRUE;
    } else {
        /* On passthrough, this is expected if VBIOS POST already completed
         * and the status register was cleared. Check CP_STAT instead. */
        if (gfx->cp_stat == 0) {
            pr_info("gpu_gfx: CP idle (CP_STAT=0), assuming RLC ready\n");
            gfx->rlc_ready = TRUE;
        } else {
            pr_warn("gpu_gfx: RLC bootload not complete, CP_STAT=0x%08x\n",
                    gfx->cp_stat);
            gfx->rlc_ready = FALSE;
        }
    }

    /* Also read RLC_CNTL to see if RLC is enabled */
    ULONG rlc_cntl = gc1_rreg(dev, regRLC_CNTL_GFX12);
    pr_info("gpu_gfx: RLC_CNTL = 0x%08x (bit0=RLC_ENABLE)\n", rlc_cntl);

    return gfx->rlc_ready ? 0 : -1;
}


/* ---- SH_MEM configuration for all VMIDs ---- */

static void configure_sh_mem(struct WddmLiteDevice *dev)
{
    /*
     * Configure SH_MEM_CONFIG for each VMID.
     * This sets the addressing mode and memory apertures
     * for shader memory access.
     *
     * Matches tinygrad AM_GFX.init_hw() and Linux amdgpu gfx_v12_0.c.
     */
    for (int vmid = 0; vmid < 16; vmid++) {
        grbm_select(dev, 0, 0, 0, vmid);

        /* SH_MEM_CONFIG:
         *   ADDRESS_MODE = 64-bit (bit 0)
         *   ALIGNMENT_MODE = UNALIGNED (bits 4:3)
         *   INITIAL_INST_PREFETCH = 3 (bits 16:14) for GFX >= 10
         */
        ULONG sh_mem_cfg = 0;
        sh_mem_cfg |= SH_MEM_ADDRESS_MODE_64;      /* bit 0 */
        sh_mem_cfg |= (SH_MEM_ALIGNMENT_MODE_UNALIGNED << 3);  /* bits 4:3 */
        sh_mem_cfg |= (3 << 14);  /* INITIAL_INST_PREFETCH = 3 */
        gc1_wreg(dev, regSH_MEM_CONFIG, sh_mem_cfg);

        /* SH_MEM_BASES:
         *   SHARED_BASE = 0x1 (LDS at 0x10000000'00000000)
         *   PRIVATE_BASE = 0x2 (Scratch at 0x20000000'00000000)
         */
        ULONG sh_mem_bases = 0;
        sh_mem_bases |= (0x1);       /* SHARED_BASE [15:0] */
        sh_mem_bases |= (0x2 << 16); /* PRIVATE_BASE [31:16] */
        gc1_wreg(dev, regSH_MEM_BASES, sh_mem_bases);
    }

    grbm_select_reset(dev);

    pr_info("gpu_gfx: SH_MEM configured for VMIDs 0-15\n");
    dev->hw.gfx.sh_mem_configured = TRUE;
}


/* ---- MEC (Micro Engine Compute) enable ---- */

static void configure_mec(struct WddmLiteDevice *dev)
{
    /* Set GRBM read timeout to 0xFF for robustness */
    ULONG grbm_cntl = gc0_rreg(dev, regGRBM_CNTL);
    grbm_cntl = (grbm_cntl & ~0xFF) | 0xFF;  /* READ_TIMEOUT [7:0] */
    gc0_wreg(dev, regGRBM_CNTL, grbm_cntl);

    /* Configure MEC doorbell range */
    gc0_wreg(dev, regCP_MEC_DOORBELL_RANGE_LOWER, 0x0);
    gc0_wreg(dev, regCP_MEC_DOORBELL_RANGE_UPPER, 0xF8);

    pr_info("gpu_gfx: MEC doorbell range set [0x000, 0x0F8]\n");

    /*
     * GFX12 MEC configuration — matching Linux amdgpu AUTOLOAD path.
     *
     * Key insight from kernel gfx_v12_0.c:
     *   - After AUTOLOAD completes, RLC has set LOCAL_INSTR_BASE, loaded
     *     instruction caches, configured data sections, etc.
     *   - The kernel NEVER writes LOCAL_INSTR_BASE, MTVEC, LOCAL_BASE0,
     *     GP0, or DC_BASE — all set by RLC during AUTOLOAD.
     *   - The kernel NEVER does GRBM_SOFT_RESET in the AUTOLOAD path.
     *   - For AUTOLOAD path: just set PRGRM_CNTR_START per pipe, then enable.
     *   - For PSP path: set PRGRM_CNTR_START, do reset pulse, then enable.
     *
     * Previous approach (tinygrad-style pipe0_reset + GRBM_SOFT_RESET)
     * was destroying the AUTOLOAD state (LOCAL_INSTR_BASE, icache, etc.)
     * that only RLC can set. This is why MEC went to PC=0 after reset.
     */

    if (dev->hw.gfx.mec_ucode_start != 0) {
        ULONG mec_lo = (ULONG)(dev->hw.gfx.mec_ucode_start >> 2);
        ULONG mec_hi = (ULONG)((dev->hw.gfx.mec_ucode_start >> 2) >> 32);

        /*
         * Step 1: Configure MEC using tinygrad-style reset pulse.
         * Tinygrad: set PC_START, then pulse reset (set then clear),
         * then later enable. Uses RMW (read-modify-write) not full writes.
         */
        ULONG cntl_before = gc1_rreg(dev, regCP_MEC_RS64_CNTL);
        pr_info("gpu_gfx: MEC CNTL before config = 0x%08x "
                "(active=%d, halt=%d, reset=%d)\n",
                cntl_before,
                (cntl_before >> 26) & 1, (cntl_before >> 30) & 1,
                (cntl_before >> 16) & 1);

        /*
         * Step 2: Dump RS64 state AFTER disable but BEFORE we touch anything.
         * This shows what AUTOLOAD set up — if LOCAL_INSTR_BASE survived
         * the pipe-level reset, we're in business.
         */
        grbm_select(dev, 1, 0, 0, 0);
        {
            ULONG ip = gc1_rreg(dev, regCP_MEC_RS64_INSTR_PNTR);
            ULONG li_lo = gc1_rreg(dev, 0x292c);  /* LOCAL_INSTR_BASE_LO */
            ULONG li_hi = gc1_rreg(dev, 0x292d);  /* LOCAL_INSTR_BASE_HI */
            ULONG lb0_lo = gc1_rreg(dev, 0x2927); /* LOCAL_BASE0_LO */
            ULONG lb0_hi = gc1_rreg(dev, 0x2928); /* LOCAL_BASE0_HI */
            ULONG mtvec_lo = gc1_rreg(dev, 0x2901);
            ULONG mtvec_hi = gc1_rreg(dev, 0x2902);
            ULONG gp0_lo = gc1_rreg(dev, 0x2910);
            ULONG gp0_hi = gc1_rreg(dev, 0x2911);
            ULONG dc_lo = gc1_rreg(dev, regCP_MEC_DC_BASE_LO);
            ULONG dc_hi = gc1_rreg(dev, regCP_MEC_DC_BASE_HI);

            pr_info("gpu_gfx: RS64 state after disable (AUTOLOAD preserved?):\n");
            pr_info("gpu_gfx:   INSTR_PNTR     = 0x%08x\n", ip);
            pr_info("gpu_gfx:   LOCAL_INSTR    = 0x%08x_%08x\n", li_hi, li_lo);
            pr_info("gpu_gfx:   LOCAL_BASE0    = 0x%08x_%08x\n", lb0_hi, lb0_lo);
            pr_info("gpu_gfx:   MTVEC          = 0x%08x_%08x\n", mtvec_hi, mtvec_lo);
            pr_info("gpu_gfx:   GP0            = 0x%08x_%08x\n", gp0_hi, gp0_lo);
            pr_info("gpu_gfx:   DC_BASE        = 0x%08x_%08x\n", dc_hi, dc_lo);
        }

        /*
         * Step 3: Set PRGRM_CNTR_START for all MEC pipes.
         * Kernel does this for all 4 pipes (num_pipe_per_mec).
         * The register format from kernel:
         *   LO = (ucode_start_addr_lo >> 2) | (ucode_start_addr_hi << 30)
         *   HI = ucode_start_addr_hi >> 2
         * We use the simpler tinygrad encoding since we have the full address.
         */
        /* Set PC_START for pipe 0 (matching tinygrad: only pipe 0) */
        grbm_select(dev, 1, 0, 0, 0);
        gc1_wreg(dev, regCP_MEC_RS64_PRGRM_CNTR_START, mec_lo);
        gc1_wreg(dev, regCP_MEC_RS64_PRGRM_CNTR_START_HI, mec_hi);
        grbm_select_reset(dev);

        pr_info("gpu_gfx: MEC PC_START set to 0x%08x_%08x (pipe 0)\n",
                mec_hi, mec_lo);

        /* Reset pulse: assert then deassert mec_pipe0_reset (tinygrad style RMW) */
        {
            ULONG cntl = gc1_rreg(dev, regCP_MEC_RS64_CNTL);
            cntl |= (1 << 16);  /* mec_pipe0_reset = 1 */
            gc1_wreg(dev, regCP_MEC_RS64_CNTL, cntl);

            cntl = gc1_rreg(dev, regCP_MEC_RS64_CNTL);
            cntl &= ~(1 << 16); /* mec_pipe0_reset = 0 */
            gc1_wreg(dev, regCP_MEC_RS64_CNTL, cntl);

            pr_info("gpu_gfx: MEC reset pulse done (pipe 0)\n");
        }
    } else {
        pr_warn("gpu_gfx: MEC ucode_start not set — cannot configure MEC\n");
    }

    /* PFP and ME configuration (tinygrad-style reset pulse — GFX12 only) */
    if (dev->hw.gfx.pfp_ucode_start != 0) {
        ULONG pfp_lo = (ULONG)(dev->hw.gfx.pfp_ucode_start >> 2);
        ULONG pfp_hi = (ULONG)((dev->hw.gfx.pfp_ucode_start >> 2) >> 32);
        grbm_select(dev, 0, 0, 0, 0);
        gc1_wreg(dev, regCP_PFP_PRGRM_CNTR_START, pfp_lo);
        gc1_wreg(dev, regCP_PFP_PRGRM_CNTR_START_HI, pfp_hi);
        grbm_select_reset(dev);

        ULONG me_cntl = gc1_rreg(dev, regCP_ME_CNTL);
        gc1_wreg(dev, regCP_ME_CNTL, me_cntl | CP_ME_CNTL__PFP_PIPE0_RESET);
        gc1_wreg(dev, regCP_ME_CNTL, me_cntl & ~CP_ME_CNTL__PFP_PIPE0_RESET);
        pr_info("gpu_gfx: PFP PC set to 0x%llx\n",
                (unsigned long long)dev->hw.gfx.pfp_ucode_start);
    }
    if (dev->hw.gfx.me_ucode_start != 0) {
        ULONG me_lo = (ULONG)(dev->hw.gfx.me_ucode_start >> 2);
        ULONG me_hi = (ULONG)((dev->hw.gfx.me_ucode_start >> 2) >> 32);
        grbm_select(dev, 0, 0, 0, 0);
        gc1_wreg(dev, regCP_ME_PRGRM_CNTR_START, me_lo);
        gc1_wreg(dev, regCP_ME_PRGRM_CNTR_START_HI, me_hi);
        grbm_select_reset(dev);

        ULONG me_cntl = gc1_rreg(dev, regCP_ME_CNTL);
        gc1_wreg(dev, regCP_ME_CNTL, me_cntl | CP_ME_CNTL__ME_PIPE0_RESET);
        gc1_wreg(dev, regCP_ME_CNTL, me_cntl & ~CP_ME_CNTL__ME_PIPE0_RESET);
        pr_info("gpu_gfx: ME PC set to 0x%llx\n",
                (unsigned long long)dev->hw.gfx.me_ucode_start);
    }
}

static int enable_mec(struct WddmLiteDevice *dev)
{
    /*
     * Matching kernel gfx_v12_0_cp_compute_enable(adev, true):
     * Single write with all resets cleared, all pipes active, halt cleared.
     * The MEC loads PC from PRGRM_CNTR_START when transitioning from
     * reset/halted state to active.
     *
     * Previous approach (read-modify-write, only setting pipe0) was wrong:
     * - Only activated pipe0, kernel activates all 4 pipes
     * - Left stale bits from the disable write
     */
    ULONG val_before = gc1_rreg(dev, regCP_MEC_RS64_CNTL);
    pr_info("gpu_gfx: CP_MEC_RS64_CNTL before enable = 0x%08x "
            "(active=%d, halt=%d, reset=%d)\n",
            val_before,
            (val_before >> 26) & 1, (val_before >> 30) & 1,
            (val_before >> 16) & 1);

    /* Tinygrad-style RMW: set pipe0_active=1, clear halt, clear reset */
    ULONG enable_val = gc1_rreg(dev, regCP_MEC_RS64_CNTL);
    enable_val |= (1 << 26);   /* mec_pipe0_active = 1 */
    enable_val &= ~(1 << 30);  /* mec_halt = 0 */
    enable_val &= ~(1 << 16);  /* mec_pipe0_reset = 0 */

    pr_info("gpu_gfx: writing MEC_RS64_CNTL = 0x%08x "
            "(pipe0 active, halt=0, reset=0)\n", enable_val);
    gc1_wreg(dev, regCP_MEC_RS64_CNTL, enable_val);

    Sleep(50);  /* kernel waits 50us, tinygrad waits 50ms — use 50ms for safety */

    ULONG val_after = gc1_rreg(dev, regCP_MEC_RS64_CNTL);
    pr_info("gpu_gfx: CP_MEC_RS64_CNTL after enable = 0x%08x "
            "(active=%d, halt=%d, reset=%d)\n",
            val_after,
            (val_after >> 26) & 1, (val_after >> 30) & 1,
            (val_after >> 16) & 1);

    /* Read MEC INSTR_PNTR to see if firmware started executing */
    grbm_select(dev, 1, 0, 0, 0);
    ULONG mec_ip = gc1_rreg(dev, regCP_MEC_RS64_INSTR_PNTR);
    grbm_select_reset(dev);
    pr_info("gpu_gfx: MEC INSTR_PNTR after enable = 0x%08x\n", mec_ip);

    /* Poll a few times if MEC hasn't started yet */
    if (mec_ip == 0) {
        for (int poll = 0; poll < 10; poll++) {
            Sleep(50);
            grbm_select(dev, 1, 0, 0, 0);
            mec_ip = gc1_rreg(dev, regCP_MEC_RS64_INSTR_PNTR);
            grbm_select_reset(dev);
            if (mec_ip != 0) {
                pr_info("gpu_gfx: MEC started after %dms: INSTR_PNTR=0x%08x\n",
                        (poll + 1) * 50, mec_ip);
                break;
            }
        }
    }

    /* Check for GFXHUB fault (instruction fetch from firmware VA may fail) */
    {
        ULONG gc_fault = gfxhub_rreg(dev, regGCVM_L2_PROTECTION_FAULT_STATUS_LO32);
        if (gc_fault != 0) {
            ULONG fa_lo = gfxhub_rreg(dev, regGCVM_L2_PROTECTION_FAULT_ADDR_LO32);
            ULONG fa_hi = gfxhub_rreg(dev, regGCVM_L2_PROTECTION_FAULT_ADDR_HI32);
            pr_warn("gpu_gfx: GFXHUB FAULT during MEC start! "
                    "status=0x%08x addr=0x%08x_%08x\n",
                    gc_fault, fa_hi, fa_lo);
        }
    }

    /* Diagnostic dump */
    if (mec_ip == 0) {
        pr_warn("gpu_gfx: MEC INSTR_PNTR still 0 — dumping diagnostics\n");

        grbm_select(dev, 1, 0, 0, 0);
        {
            ULONG li_lo = gc1_rreg(dev, 0x292c);
            ULONG li_hi = gc1_rreg(dev, 0x292d);
            ULONG pc_lo = gc1_rreg(dev, regCP_MEC_RS64_PRGRM_CNTR_START);
            ULONG pc_hi = gc1_rreg(dev, regCP_MEC_RS64_PRGRM_CNTR_START_HI);
            ULONG dc_lo = gc1_rreg(dev, regCP_MEC_DC_BASE_LO);
            ULONG dc_hi = gc1_rreg(dev, regCP_MEC_DC_BASE_HI);
            ULONG mtvec_lo = gc1_rreg(dev, 0x2901);
            ULONG mtvec_hi = gc1_rreg(dev, 0x2902);

            pr_info("gpu_gfx:   LOCAL_INSTR = 0x%08x_%08x\n", li_hi, li_lo);
            pr_info("gpu_gfx:   PC_START    = 0x%08x_%08x\n", pc_hi, pc_lo);
            pr_info("gpu_gfx:   DC_BASE     = 0x%08x_%08x\n", dc_hi, dc_lo);
            pr_info("gpu_gfx:   MTVEC       = 0x%08x_%08x\n", mtvec_hi, mtvec_lo);
        }
        grbm_select_reset(dev);

        ULONG sys_lo = gfxhub_rreg(dev, regGCMC_VM_SYSTEM_APERTURE_LOW_ADDR);
        ULONG sys_hi = gfxhub_rreg(dev, regGCMC_VM_SYSTEM_APERTURE_HIGH_ADDR);
        pr_info("gpu_gfx:   GFXHUB SYS_APERTURE = 0x%08x - 0x%08x\n",
                sys_lo, sys_hi);
    }

    if (val_after & CP_MEC_RS64_CNTL__MEC_PIPE0_ACTIVE) {
        pr_info("gpu_gfx: MEC pipe0 active (INSTR_PNTR=%s)\n",
                mec_ip != 0 ? "running" : "0=no firmware");
        dev->hw.gfx.mec_enabled = TRUE;
        return (mec_ip != 0) ? 0 : -1;
    }

    pr_err("gpu_gfx: MEC pipe0 NOT active after enable write — "
           "compute queues won't work\n");
    dev->hw.gfx.mec_enabled = TRUE;
    return -1;
}


/* ---- Compute queue: dequeue active HQDs ---- */

static void dequeue_hqd(struct WddmLiteDevice *dev, ULONG pipe, ULONG queue)
{
    grbm_select(dev, 1, pipe, queue, 0);

    ULONG active = gc0_rreg(dev, regCP_HQD_ACTIVE);
    if (active & 1) {
        pr_info("gpu_gfx: dequeuing HQD pipe=%u queue=%u\n", pipe, queue);
        /* DEQUEUE_REQUEST: 2 = RESET_WAVES */
        gc0_wreg(dev, regCP_HQD_DEQUEUE_REQUEST, 0x2);

        /* Poll for HQD to become inactive */
        for (int i = 0; i < 100; i++) {
            if (!(gc0_rreg(dev, regCP_HQD_ACTIVE) & 1))
                break;
            Sleep(1);
        }

        if (gc0_rreg(dev, regCP_HQD_ACTIVE) & 1) {
            pr_warn("gpu_gfx: HQD dequeue timeout pipe=%u queue=%u\n",
                    pipe, queue);
        }
    }

    grbm_select_reset(dev);
}

static void dequeue_all_hqds(struct WddmLiteDevice *dev)
{
    /* Dequeue up to 2 queues (pipe=0, queue=0 and queue=1) */
    for (ULONG q = 0; q < 2; q++) {
        dequeue_hqd(dev, 0, q);
    }
}


/* ---- Top-level GFX init ---- */

int gpu_gfx_init(struct WddmLiteDevice *dev)
{
    if (dev->hw.gfx_initialized)
        return 0;

    if (!dev->hw.ip_discovery_done) {
        pr_err("gpu_gfx: IP discovery must be run first\n");
        return -1;
    }

    if (dev->hw.ip.gc_base1 == 0) {
        pr_err("gpu_gfx: GC base_index 1 not found\n");
        return -1;
    }

    /* Preserve firmware-related values from loading phase, zero the rest */
    {
        ULONGLONG pfp = dev->hw.gfx.pfp_ucode_start;
        ULONGLONG me  = dev->hw.gfx.me_ucode_start;
        ULONGLONG mec = dev->hw.gfx.mec_ucode_start;
        UCHAR *mec_code = dev->hw.gfx.mec_fw_code;
        ULONG  mec_code_sz = dev->hw.gfx.mec_fw_code_size;
        UCHAR *mec_data = dev->hw.gfx.mec_fw_data;
        ULONG  mec_data_sz = dev->hw.gfx.mec_fw_data_size;
        /* Preserve AUTOLOAD status — needed by Step 0b to take TLB-safe
         * VMID1 path when MEC is already executing from firmware VA. */
        ULONG early_bl = dev->hw.gfx.rlc_bootload_status;
        memset(&dev->hw.gfx, 0, sizeof(dev->hw.gfx));
        dev->hw.gfx.pfp_ucode_start = pfp;
        dev->hw.gfx.me_ucode_start  = me;
        dev->hw.gfx.mec_ucode_start = mec;
        dev->hw.gfx.mec_fw_code = mec_code;
        dev->hw.gfx.mec_fw_code_size = mec_code_sz;
        dev->hw.gfx.mec_fw_data = mec_data;
        dev->hw.gfx.mec_fw_data_size = mec_data_sz;
        dev->hw.gfx.rlc_bootload_status = early_bl;
    }

    /* Step 0: Disable GFXOFF so GC registers are accessible.
     * If GFXOFF was already disabled (e.g. EnableAllSmuFeatures brought
     * GC alive and set the flag), skip — the SMU mailbox may be wedged
     * and sending DisallowGfxOff would hang. */
    if (!dev->hw.gfxoff_disabled) {
        if (gpu_disable_gfxoff(dev) != 0) {
            pr_warn("gpu_gfx: GFXOFF disable failed (GC registers may be inaccessible)\n");
        }
    } else {
        pr_info("gpu_gfx: GFXOFF already disabled, skipping\n");
    }

    /* Step 0b: Initialize GFXHUB if it was deferred during gpu_gmc_init.
     * This happens when GC was powered off at GMC init time but came
     * alive later (e.g. after EnableAllSmuFeatures). */
    if (dev->hw.gmc_initialized) {
        char keep_gfxhub[32] = {};
        GetEnvironmentVariableA("HSAKMT_KEEP_VBIOS_GFXHUB",
            keep_gfxhub, sizeof(keep_gfxhub));

        if (keep_gfxhub[0] == '1') {
            /* Dump AUTOLOAD-configured GFXHUB CONTEXT0 state. */
            ULONG gc_test = gfxhub_rreg(dev, regGCVM_CONTEXT0_CNTL);
            ULONG pt_lo = gfxhub_rreg(dev,
                regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32);
            ULONG pt_hi = gfxhub_rreg(dev,
                regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32);
            pr_info("gpu_gfx: KEEP_VBIOS_GFXHUB=1 — CONTEXT0 after AUTOLOAD: "
                    "CNTL=0x%08x PT_BASE=0x%08x_%08x\n",
                    gc_test, pt_hi, pt_lo);

            if (dev->hw.gfx.rlc_bootload_status & 0x80000000) {
                /*
                 * VBIOS AUTOLOAD was already complete when our driver started.
                 * MEC is (or was) running with TLB entries for firmware VA
                 * cached from VBIOS GFXHUB setup. gfxhub_enable_compute_context
                 * calls gfxhub_init_cache which writes GCVM_L2_CNTL2 to
                 * invalidate ALL TLBs — evicting the firmware VA entry. After
                 * eviction, GFXHUB walks CONTEXT0 (flat GART, only covers
                 * [gart_start, gart_end]) and firmware VA is out of range →
                 * MEC faults and executes VBIOS code at 0x4044.
                 *
                 * Use the minimal path: only set CONTEXT1 registers without
                 * any global TLB/L2 invalidation. VMID 1 TLB flush is still
                 * needed (to activate the new CONTEXT1 mapping), but VMID 0
                 * TLBs must remain intact.
                 */
                gfxhub_setup_vmid1_no_invalidate(dev);
                flush_gpu_tlb(dev, 1, 1);
                pr_info("gpu_gfx: GFXHUB CONTEXT1 (VMID 1) set without global "
                        "TLB invalidation — MEC firmware VA TLBs preserved\n");
            } else {
                /*
                 * Fresh AUTOLOAD path: MEC wasn't running before, so global
                 * TLB invalidation is safe. Full compute context setup.
                 */
                gfxhub_enable_compute_context(dev);
                flush_gpu_tlb(dev, 1, 1);
                pr_info("gpu_gfx: GFXHUB CONTEXT1 (VMID 1) configured with GART "
                        "PT — CONTEXT0 firmware VA mapping preserved\n");
            }
        } else {
            /*
             * After AUTOLOAD, CONTEXT0 holds the firmware PT set by RLC.
             * Same rule applies: leave CONTEXT0 alone, use CONTEXT1 for
             * compute queue data.
             */
            ULONG gc_test = gfxhub_rreg(dev, regGCVM_CONTEXT0_CNTL);
            pr_info("gpu_gfx: configuring GFXHUB CONTEXT1 for compute "
                    "(CONTEXT0_CNTL=0x%08x preserved)\n", gc_test);

            /* Dump AUTOLOAD-configured system aperture BEFORE we overwrite */
            {
                ULONG sa_lo_pre = gfxhub_rreg(dev, 0x1619);
                ULONG sa_hi_pre = gfxhub_rreg(dev, 0x161A);
                pr_info("gpu_gfx: GFXHUB sys aperture BEFORE: LOW=0x%08x HIGH=0x%08x\n",
                        sa_lo_pre, sa_hi_pre);
            }

            /* Check env var to skip GFXHUB compute context (for MES debugging) */
            char skip_gfxhub_cc[32] = {};
            GetEnvironmentVariableA("HSAKMT_SKIP_GFXHUB_CC",
                skip_gfxhub_cc, sizeof(skip_gfxhub_cc));
            if (skip_gfxhub_cc[0] == '1') {
                pr_info("gpu_gfx: SKIPPING gfxhub_enable_compute_context "
                        "(HSAKMT_SKIP_GFXHUB_CC=1)\n");
            } else {
                gfxhub_enable_compute_context(dev);
                flush_gpu_tlb(dev, 1, 1);
            }

            /* CRITICAL: Enable CONTEXT0 (VMID 0 page table walks).
             * AUTOLOAD leaves CONTEXT0_CNTL with enable bit CLEARED (0x03fffc00).
             * amdgpu sets bit 0 = 1 in gfxhub_gart_enable. Without this, MEC/MES
             * firmware can't walk the page table for firmware VA → TMR access,
             * causing them to get stuck at instruction 0x4044 during init. */
            {
                ULONG ctx0 = gfxhub_rreg(dev, regGCVM_CONTEXT0_CNTL);
                if (!(ctx0 & 1)) {
                    ctx0 |= 1;  /* VM_CONTEXT_ENABLE_CONTEXT */
                    gfxhub_wreg(dev, regGCVM_CONTEXT0_CNTL, ctx0);
                    pr_info("gpu_gfx: CONTEXT0 ENABLED (was 0x%08x → 0x%08x)\n",
                            ctx0 & ~1, ctx0);
                } else {
                    pr_info("gpu_gfx: CONTEXT0 already enabled (0x%08x)\n", ctx0);
                }
            }

            /* Read system aperture AFTER */
            {
                ULONG sa_lo_post = gfxhub_rreg(dev, 0x1619);
                ULONG sa_hi_post = gfxhub_rreg(dev, 0x161A);
                pr_info("gpu_gfx: GFXHUB sys aperture AFTER: LOW=0x%08x HIGH=0x%08x\n",
                        sa_lo_post, sa_hi_post);
            }
            pr_info("gpu_gfx: GFXHUB CONTEXT1 (VMID 1) enabled for compute\n");
        }
    }

    /* Step 1: Check RLC readiness */
    if (check_rlc_ready(dev) != 0) {
        pr_warn("gpu_gfx: RLC not ready, continuing anyway\n");
    }

    /* Step 2: HDP flush before GPU programming */
    gpu_hdp_flush(dev);

    /* Step 2a: TCP_CNTL golden register (from tinygrad, for GFX >= 11) */
    {
        ULONG tcp_cntl = gc1_rreg(dev, 0x19a2);  /* regTCP_CNTL */
        pr_info("gpu_gfx: TCP_CNTL (before) = 0x%08x\n", tcp_cntl);
        tcp_cntl |= 0x20000000;
        gc1_wreg(dev, 0x19a2, tcp_cntl);
    }

    /* Step 2b: Enable RLC */
    {
        ULONG rlc_cntl = gc1_rreg(dev, regRLC_CNTL_GFX12);
        pr_info("gpu_gfx: RLC_CNTL (before) = 0x%08x\n", rlc_cntl);
        gc1_wreg(dev, regRLC_CNTL_GFX12, 0x1);  /* RLC_ENABLE = 1 */
        rlc_cntl = gc1_rreg(dev, regRLC_CNTL_GFX12);
        pr_info("gpu_gfx: RLC_CNTL (after)  = 0x%08x\n", rlc_cntl);
    }

    /* Step 2c: Enable RLC SRM (save/restore manager) */
    {
        ULONG srm = gc1_rreg(dev, 0x4c80);  /* regRLC_SRM_CNTL */
        pr_info("gpu_gfx: RLC_SRM_CNTL (before) = 0x%08x\n", srm);
        /* srm_enable(bit0)=1, auto_incr_addr(bit1)=1 */
        srm |= 0x3;
        gc1_wreg(dev, 0x4c80, srm);
    }

    /* Step 2d: RLC SPM MC CNTL */
    gc1_wreg(dev, 0x0982, 0xf);  /* regRLC_SPM_MC_CNTL */

    /* Step 2e: NBIO doorbell routing (from tinygrad soc.doorbell_enable)
     *
     * Without this, doorbell BAR2 writes are silently dropped by the
     * NBIO S2A bridge and never reach the GPU engines (MEC, SDMA).
     *
     * All registers are at NBIF base_idx 2 (nbio_base2 from IP discovery).
     * Register offsets from nbif_6_3_1_offset.h.
     */
    {
        ULONG nbio2 = dev->hw.ip.nbio_base2;
        if (nbio2 == 0) {
            pr_warn("gpu_gfx: NBIO base2 not found, skipping doorbell routing\n");
        } else {
            /* regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN (offset 0x00c0, base_idx 2)
             * Enable the doorbell aperture (PCIe BAR2) */
            ULONG aper_en = gpu_smn_rreg(dev, nbio2 + 0x00c0);
            pr_info("gpu_gfx: DOORBELL_APER_EN (before) = 0x%08x\n", aper_en);
            gpu_smn_wreg(dev, nbio2 + 0x00c0, 0x1);
            aper_en = gpu_smn_rreg(dev, nbio2 + 0x00c0);
            pr_info("gpu_gfx: DOORBELL_APER_EN (after)  = 0x%08x\n", aper_en);

            /* regGDC_S2A0_S2A_DOORBELL_ENTRY_0_CTRL (offset 0x01cb, base_idx 2)
             * Port 0: Compute/GFX doorbell routing
             *   enable=1 (bit 0), awid=0x3 (bits [5:1]), awaddr_31_28=0x3 (bits [31:28])
             *   Encoded: 0x1 | (0x3 << 1) | (0x3 << 28) = 0x30000007 */
            /* Dump all S2A doorbell entries */
            for (int si = 0; si < 8; si++) {
                ULONG sval = gpu_smn_rreg(dev, nbio2 + 0x01cb + si);
                pr_info("gpu_gfx: S2A_DOORBELL_ENTRY_%d = 0x%08x "
                        "(en=%d, awid=%d, awaddr=%d)\n",
                        si, sval, sval & 1, (sval >> 1) & 0x1F,
                        (sval >> 28) & 0xF);
            }

            ULONG entry0 = gpu_smn_rreg(dev, nbio2 + 0x01cb);
            pr_info("gpu_gfx: S2A_DOORBELL_ENTRY_0 (before) = 0x%08x\n", entry0);
            gpu_smn_wreg(dev, nbio2 + 0x01cb, 0x30000007);
            entry0 = gpu_smn_rreg(dev, nbio2 + 0x01cb);
            pr_info("gpu_gfx: S2A_DOORBELL_ENTRY_0 (after)  = 0x%08x\n", entry0);

            /* regGDC_S2A0_S2A_DOORBELL_ENTRY_3_CTRL (offset 0x01ce, base_idx 2)
             * Port 3: SDMA doorbell routing
             *   enable=1 (bit 0), awid=0x6 (bits [5:1]), awaddr_31_28=0x3 (bits [31:28])
             *   Encoded: 0x1 | (0x6 << 1) | (0x3 << 28) = 0x3000000D */
            ULONG entry3 = gpu_smn_rreg(dev, nbio2 + 0x01ce);
            pr_info("gpu_gfx: S2A_DOORBELL_ENTRY_3 (before) = 0x%08x\n", entry3);
            gpu_smn_wreg(dev, nbio2 + 0x01ce, 0x3000000D);
            entry3 = gpu_smn_rreg(dev, nbio2 + 0x01ce);
            pr_info("gpu_gfx: S2A_DOORBELL_ENTRY_3 (after)  = 0x%08x\n", entry3);
        }
    }

    /* Step 3: Configure SH_MEM for all VMIDs */
    configure_sh_mem(dev);

    /*
     * Step 3.5a: If we have saved MEC firmware bytes and AUTOLOAD didn't
     * complete (RLC bootload bit 31 not set), copy firmware to VRAM at
     * a known offset. The VRAM system aperture is identity-mapped through
     * GFXHUB, so MEC can fetch instructions from there without page tables.
     *
     * This handles the VBIOS-state case where AUTOLOAD_RLC doesn't complete
     * (the firmware VA 0x7000000003000 has no page table mapping).
     */
    pr_info("gpu_gfx: step 3.5a: mec_fw_code=%p size=%u ucode_start=0x%llx\n",
            dev->hw.gfx.mec_fw_code,
            dev->hw.gfx.mec_fw_code_size,
            (unsigned long long)dev->hw.gfx.mec_ucode_start);
    if (dev->hw.gfx.mec_fw_code && dev->hw.gfx.mec_fw_code_size > 0 &&
        dev->hw.gfx.mec_ucode_start != 0) {
        /* Check if AUTOLOAD completed (bit 31 of bootload status) */
        ULONG bl_status = gc1_rreg(dev, 0x4e7c); /* RLC_RLCS_BOOTLOAD_STATUS */
        pr_info("gpu_gfx: bootload_status for VRAM copy check = 0x%08x\n",
                bl_status);
        /* Check KEEP_VBIOS_GFXHUB — if set, trust VBIOS page tables and
         * use firmware VA as PC_START (don't copy to VRAM) */
        char keep_gfxhub_fw[32] = {};
        GetEnvironmentVariableA("HSAKMT_KEEP_VBIOS_GFXHUB",
            keep_gfxhub_fw, sizeof(keep_gfxhub_fw));

        if (keep_gfxhub_fw[0] == '1') {
            pr_info("gpu_gfx: KEEP_VBIOS_GFXHUB=1 — using firmware VA 0x%llx "
                    "as PC_START (CONTEXT0 maps it to TMR)\n",
                    (unsigned long long)dev->hw.gfx.mec_ucode_start);
            /* Don't override mec_ucode_start — firmware VA from PSP header.
             * AUTOLOAD/RLC configured GFXHUB CONTEXT0 (VMID 0) to map this
             * VA → TMR. MEC uses CONTEXT0 for instruction fetch. */
        } else if (!(bl_status & 0x80000000)) {
            /* AUTOLOAD did NOT complete — firmware VA has no mapping.
             * Scan top of VRAM for TMR (decrypted firmware).
             * TMR is typically at VRAM end minus a few MB. */
            {
                char scan_tmr[32] = {};
                GetEnvironmentVariableA("HSAKMT_SCAN_TMR",
                    scan_tmr, sizeof(scan_tmr));
                if (scan_tmr[0] == '1') {
                    pr_info("gpu_gfx: === TMR SCAN: looking for decrypted "
                            "firmware in VRAM ===\n");
                    /* Scan last 32MB of VRAM in 1MB steps */
                    ULONGLONG vram_end = dev->hw.gmc.vram_end;
                    for (int mb = 1; mb <= 32; mb++) {
                        ULONGLONG off = (vram_end - dev->hw.gmc.vram_start)
                                        - (ULONGLONG)mb * 1024 * 1024;
                        PVOID handle = NULL;
                        PVOID cpu = psp_ring_map_vram(dev, off, 4096,
                                                       &handle);
                        if (cpu) {
                            ULONG *dw = (ULONG *)cpu;
                            /* Check for RISC-V patterns:
                             * Valid 32-bit instructions have low 2 bits = 11
                             * NOP = 0x00000013 (addi x0,x0,0)
                             * JAL = xxxxxxx_1101111 (opcode 0x6F)
                             * AUIPC = xxxxxxx_0010111 (opcode 0x17) */
                            BOOLEAN looks_like_code = FALSE;
                            int valid = 0;
                            for (int i = 0; i < 16; i++) {
                                ULONG insn = dw[i];
                                if ((insn & 0x3) == 0x3 && insn != 0 &&
                                    insn != 0xFFFFFFFF)
                                    valid++;
                            }
                            if (valid >= 12) looks_like_code = TRUE;
                            if (looks_like_code || (dw[0] != 0 &&
                                dw[0] != 0xFFFFFFFF)) {
                                pr_info("gpu_gfx: TMR scan VRAM-%dMB "
                                        "(off=0x%llx): [0]=0x%08x "
                                        "[1]=0x%08x [2]=0x%08x "
                                        "[3]=0x%08x %s\n",
                                        mb, (unsigned long long)off,
                                        dw[0], dw[1], dw[2], dw[3],
                                        looks_like_code ? "<<< CODE?"
                                                        : "");
                            }
                            /* Don't unmap — transient */
                        }
                    }
                }
            }

            /* Copy MEC code to VRAM and use system aperture address. */
            pr_info("gpu_gfx: AUTOLOAD incomplete (status=0x%08x), "
                    "copying MEC fw to VRAM\n", bl_status);

            /* Use VRAM+10MB for MEC code, VRAM+11MB for MEC data */
            ULONGLONG mec_code_vram_off = 10 * 1024 * 1024;  /* 0xA00000 */
            ULONGLONG mec_data_vram_off = 11 * 1024 * 1024;  /* 0xB00000 */

            /* Map VRAM for MEC code */
            PVOID code_handle = NULL;
            PVOID code_cpu = psp_ring_map_vram(dev, mec_code_vram_off,
                                                dev->hw.gfx.mec_fw_code_size,
                                                &code_handle);
            if (code_cpu) {
                /*
                 * DIAGNOSTIC: Fill with NOP sled instead of firmware.
                 * RS64 is RISC-V based. 32-bit NOP = 0x00000013 (addi x0,x0,0).
                 * If MEC advances past 1 instruction with NOPs, the issue is
                 * firmware content (encrypted/signed), not hardware setup.
                 */
                char use_nop[32] = {};
                GetEnvironmentVariableA("HSAKMT_MEC_NOP_TEST",
                    use_nop, sizeof(use_nop));

                if (use_nop[0] == '1') {
                    pr_info("gpu_gfx: === NOP SLED TEST === filling code with NOPs\n");
                    ULONG *code_dw = (ULONG *)code_cpu;
                    ULONG nop_count = dev->hw.gfx.mec_fw_code_size / 4;
                    for (ULONG i = 0; i < nop_count; i++) {
                        code_dw[i] = 0x00000013;  /* RISC-V 32-bit NOP */
                    }
                } else {
                    memcpy(code_cpu, dev->hw.gfx.mec_fw_code,
                           dev->hw.gfx.mec_fw_code_size);
                }

                ULONGLONG code_mc = dev->hw.gmc.vram_start + mec_code_vram_off;
                pr_info("gpu_gfx: MEC code at VRAM MC 0x%llx (%u bytes)\n",
                        (unsigned long long)code_mc,
                        dev->hw.gfx.mec_fw_code_size);

                /* Override ucode_start with VRAM MC address */
                dev->hw.gfx.mec_ucode_start = code_mc;

                /* Flush HDP to ensure CPU writes are visible to GPU */
                gpu_hdp_flush(dev);

                /* Verify content in VRAM by reading back first DWORDs */
                {
                    ULONG *code_dw = (ULONG *)code_cpu;
                    pr_info("gpu_gfx: MEC code verify: [0]=0x%08x [1]=0x%08x "
                            "[2]=0x%08x [3]=0x%08x\n",
                            code_dw[0], code_dw[1], code_dw[2], code_dw[3]);
                }

                /* Don't unmap — firmware must remain accessible */
            } else {
                pr_warn("gpu_gfx: failed to map VRAM for MEC code\n");
            }

            /* Copy MEC data/stack to VRAM */
            if (dev->hw.gfx.mec_fw_data && dev->hw.gfx.mec_fw_data_size > 0) {
                PVOID data_handle = NULL;
                PVOID data_cpu = psp_ring_map_vram(dev, mec_data_vram_off,
                                                    dev->hw.gfx.mec_fw_data_size,
                                                    &data_handle);
                if (data_cpu) {
                    memcpy(data_cpu, dev->hw.gfx.mec_fw_data,
                           dev->hw.gfx.mec_fw_data_size);

                    ULONGLONG data_mc = dev->hw.gmc.vram_start +
                                        mec_data_vram_off;
                    pr_info("gpu_gfx: MEC data at VRAM MC 0x%llx (%u bytes)\n",
                            (unsigned long long)data_mc,
                            dev->hw.gfx.mec_fw_data_size);
                    gpu_hdp_flush(dev);
                } else {
                    pr_warn("gpu_gfx: failed to map VRAM for MEC data\n");
                }
            }
        } else {
            pr_info("gpu_gfx: AUTOLOAD completed (status=0x%08x), "
                    "using firmware VA\n", bl_status);
        }
    }

    /*
     * Step 3.5: MEC initialization.
     *
     * Key insight from Linux amdgpu gfx_v12_0.c:
     * In the AUTOLOAD path, the kernel NEVER does GRBM_SOFT_RESET.
     * GRBM_SOFT_RESET destroys LOCAL_INSTR_BASE and other RLC-programmed
     * state that only RLC can set (during AUTOLOAD). This was the root
     * cause of MEC going to PC=0 — the instruction aperture was wiped.
     *
     * The correct sequence (matching kernel AUTOLOAD path):
     * 1. Dequeue any active HQDs
     * 2. Disable MEC via CP_MEC_RS64_CNTL (halt+reset+deactivate)
     *    This preserves RLC state (LOCAL_INSTR_BASE, page tables, etc.)
     * 3. Set PRGRM_CNTR_START for all pipes
     * 4. Enable MEC (clear halt/reset, set active)
     */

    /* Check if VBIOS AUTOLOAD completed and we should use firmware VA */
    if (dev->hw.gfx.mec_ucode_start == 0 &&
        (dev->hw.gfx.rlc_bootload_status & 0x80000000)) {
        pr_info("gpu_gfx: VBIOS AUTOLOAD complete (status=0x%08x), "
                "using firmware VA 0x7000000003000 as PC_START\n",
                dev->hw.gfx.rlc_bootload_status);
        dev->hw.gfx.mec_ucode_start = 0x7000000003000ULL;
    }

    if (dev->hw.gfx.mec_ucode_start != 0) {
        /*
         * If VBIOS AUTOLOAD was already complete when our driver started,
         * MEC may already be running the correct compute firmware. Disabling
         * and restarting MEC clears the GFXHUB VM TLB entries that map
         * firmware VA (0x7000000003000) → TMR. After restart, GFXHUB
         * CONTEXT0 (our flat GART) doesn't cover firmware VA, so MEC faults
         * and falls through to VBIOS code at 0x4044.
         *
         * When MEC is already active+running (not halted, not in reset),
         * skip the disable/restart. Just configure the doorbell range and
         * dequeue any stale HQDs. MEC retains its firmware VA context.
         */
        BOOLEAN mec_already_running = FALSE;
        if (dev->hw.gfx.rlc_bootload_status & 0x80000000) {
            ULONG mec_cntl = gc1_rreg(dev, regCP_MEC_RS64_CNTL);
            BOOLEAN active = (mec_cntl >> 26) & 1;
            BOOLEAN halted = (mec_cntl >> 30) & 1;
            BOOLEAN in_reset = (mec_cntl >> 16) & 1;
            if (active && !halted && !in_reset) {
                pr_info("gpu_gfx: MEC already running (CNTL=0x%08x) — "
                        "skipping disable/restart to preserve firmware VA "
                        "GFXHUB mapping\n", mec_cntl);
                mec_already_running = TRUE;
            }
        }

        if (mec_already_running) {
            /* Doorbell range and GRBM timeout still needed. */
            ULONG grbm_cntl = gc0_rreg(dev, regGRBM_CNTL);
            grbm_cntl = (grbm_cntl & ~0xFF) | 0xFF;
            gc0_wreg(dev, regGRBM_CNTL, grbm_cntl);
            gc0_wreg(dev, regCP_MEC_DOORBELL_RANGE_LOWER, 0x0);
            gc0_wreg(dev, regCP_MEC_DOORBELL_RANGE_UPPER, 0xF8);
            pr_info("gpu_gfx: MEC doorbell range set [0x000, 0x0F8] (passthrough)\n");
        } else {
            /* Dequeue any active HQDs left from VBIOS */
            dequeue_all_hqds(dev);
            /* Configure MEC (disable → set PC_START → doorbell range) */
            configure_mec(dev);
            /* Enable MEC */
            if (enable_mec(dev) != 0) {
                pr_warn("gpu_gfx: MEC enable failed (continuing)\n");
            }
        }
    } else {
        pr_warn("gpu_gfx: No MEC firmware address — "
                "skipping MEC init (BOOTLOAD=0x%08x)\n",
                dev->hw.gfx.rlc_bootload_status);
    }

    /* Step 6: Read diagnostic registers */
    {
        /* GRBM_SCRATCH_REG7 (BASE_IDX=0, offset 0x0de7) */
        ULONG grbm_scratch7 = gc0_rreg(dev, regGRBM_SCRATCH_REG7);
        ULONG grbm_scratch6 = gc0_rreg(dev, regGRBM_SCRATCH_REG6);
        pr_info("gpu_gfx: GRBM_SCRATCH_REG7 = 0x%08x\n", grbm_scratch7);
        pr_info("gpu_gfx: GRBM_SCRATCH_REG6 = 0x%08x\n", grbm_scratch6);

        /* SCRATCH_REG7 (BASE_IDX=1, offset 0x2047) — tinygrad AM version marker */
        ULONG scratch7 = gc1_rreg(dev, 0x2047);
        ULONG scratch6 = gc1_rreg(dev, 0x2046);
        pr_info("gpu_gfx: SCRATCH_REG7 (AM marker) = 0x%08x\n", scratch7);
        pr_info("gpu_gfx: SCRATCH_REG6 (AM state)  = 0x%08x\n", scratch6);
    }

    /* Verify SH_MEM_CONFIG readback for VMID 0 */
    {
        grbm_select(dev, 0, 0, 0, 0);
        ULONG sh_cfg = gc1_rreg(dev, regSH_MEM_CONFIG);
        ULONG sh_bases = gc1_rreg(dev, regSH_MEM_BASES);
        grbm_select_reset(dev);

        pr_info("gpu_gfx: SH_MEM_CONFIG (VMID 0)  = 0x%08x\n", sh_cfg);
        pr_info("gpu_gfx: SH_MEM_BASES  (VMID 0)  = 0x%08x\n", sh_bases);
    }

    /* Read MEC program counter to verify firmware is executing */
    {
        grbm_select(dev, 1, 0, 0, 0);  /* ME=1 (MEC), pipe=0 */
        ULONG mec_pc_lo = gc1_rreg(dev, regCP_MEC_RS64_PRGRM_CNTR_START);
        ULONG mec_pc_hi = gc1_rreg(dev, regCP_MEC_RS64_PRGRM_CNTR_START_HI);
        grbm_select_reset(dev);
        pr_info("gpu_gfx: MEC PC_START = 0x%08x_%08x\n", mec_pc_hi, mec_pc_lo);

        /* Read current MEC instruction address (0x2901 = regCP_MEC_RS64_INSTR_PNTR) */
        grbm_select(dev, 1, 0, 0, 0);
        ULONG mec_ip = gc1_rreg(dev, regCP_MEC_RS64_INSTR_PNTR);  /* CP_MEC_RS64_INSTR_PNTR */
        grbm_select_reset(dev);
        pr_info("gpu_gfx: MEC INSTR_PNTR = 0x%08x (0=stalled/no firmware)\n", mec_ip);
    }

    /* Check GFXHUB VM fault status (should be 0 after clean init) */
    {
        ULONG fault_status = gfxhub_rreg(dev, regGCVM_L2_PROTECTION_FAULT_STATUS_LO32);
        if (fault_status != 0) {
            ULONG fault_addr_lo = gfxhub_rreg(dev, regGCVM_L2_PROTECTION_FAULT_ADDR_LO32);
            ULONG fault_addr_hi = gfxhub_rreg(dev, regGCVM_L2_PROTECTION_FAULT_ADDR_HI32);
            pr_err("gpu_gfx: GFXHUB VM FAULT at init! status=0x%08x addr=0x%08x_%08x\n",
                   fault_status, fault_addr_hi, fault_addr_lo);
        } else {
            pr_info("gpu_gfx: GFXHUB VM fault status = 0 (clean)\n");
        }

        /* Also check MMHUB */
        ULONG mm_fault = mmhub_rreg(dev, regMMVM_L2_PROTECTION_FAULT_STATUS_LO32);
        if (mm_fault != 0) {
            pr_err("gpu_gfx: MMHUB VM FAULT at init! status=0x%08x\n", mm_fault);
        }
    }

    /* Step 7: IMU and RLC diagnostics — understand where autoload is stuck */
    {
        /* IMU status registers (GC base_idx 2, various offsets) */
        ULONG gc2_base = 0;
        for (ULONG i = 0; i < dev->hw.ip.num_blocks; i++) {
            if (dev->hw.ip.blocks[i].hw_id == GPU_HWID_GC &&
                dev->hw.ip.blocks[i].num_base_addr > 2) {
                gc2_base = dev->hw.ip.blocks[i].base_addr[2];
                break;
            }
        }
        if (gc2_base != 0) {
            /* IMU registers at GC base_idx 2:
             * regGFX_IMU_C2PMSG_0..15 at offsets from gc_12_0_0_offset.h
             * These show IMU firmware status messages */
            pr_info("gpu_gfx: GC base2 = 0x%05x (IMU registers)\n", gc2_base);

            /* Read first 4 IMU C2PMSG registers for status */
            for (int i = 0; i < 4; i++) {
                /* GFX_IMU_C2PMSG_x at base2 + 0x4001 + i (GFX12) */
                ULONG imu_msg = gpu_smn_rreg(dev, gc2_base + 0x4001 + i);
                pr_info("gpu_gfx: IMU_C2PMSG_%d = 0x%08x\n", i, imu_msg);
            }

            /* IMU_STATUS (base2 + 0x4160) — IMU execution status */
            ULONG imu_status = gpu_smn_rreg(dev, gc2_base + 0x4160);
            pr_info("gpu_gfx: IMU_STATUS = 0x%08x\n", imu_status);
        }

        /* RLC additional status registers */
        ULONG rlc_stat = gc1_rreg(dev, 0x4c04);  /* regRLC_STAT */
        ULONG rlc_cntl = gc1_rreg(dev, regRLC_CNTL_GFX12);
        ULONG rlc_bl = gc1_rreg(dev, regRLC_RLCS_BOOTLOAD_STATUS);
        pr_info("gpu_gfx: RLC_STAT=0x%08x RLC_CNTL=0x%08x BOOTLOAD=0x%08x\n",
                rlc_stat, rlc_cntl, rlc_bl);

        /* CP_ME_CNTL — check PFP/ME state */
        ULONG me_cntl = gc1_rreg(dev, regCP_ME_CNTL);
        pr_info("gpu_gfx: CP_ME_CNTL final = 0x%08x\n", me_cntl);

        /* PFP instruction pointer (to see if PFP firmware is running) */
        grbm_select(dev, 0, 0, 0, 0);  /* ME=0 (GFX), pipe=0 */
        ULONG pfp_ip = gc1_rreg(dev, 0x2101);  /* CP_PFP_INSTR_PNTR */
        grbm_select_reset(dev);
        pr_info("gpu_gfx: PFP INSTR_PNTR = 0x%08x (0=stalled)\n", pfp_ip);
    }

    /* ================================================================
     * MES (Micro Engine Scheduler) enable.
     * After AUTOLOAD, MES firmware is in TMR. We need to:
     *   1. Set CP_MES_PRGRM_CNTR_START for each pipe
     *   2. Reset + activate pipes via CP_MES_CNTL
     * MES handles doorbell routing to MEC on GFX12.
     *
     * Register offsets (BASE_IDX=1 → gc_base1):
     *   CP_MES_PRGRM_CNTR_START    = 0x2800
     *   CP_MES_PRGRM_CNTR_START_HI = 0x289d
     *   CP_MES_CNTL                = 0x2807
     *   RLC_CP_SCHEDULERS          = 0x098a
     *
     * CP_MES_CNTL bits:
     *   MES_INVALIDATE_ICACHE = bit 4
     *   MES_PIPE0_RESET = bit 16, MES_PIPE1_RESET = bit 17
     *   MES_PIPE0_ACTIVE = bit 26, MES_PIPE1_ACTIVE = bit 27
     *   MES_HALT = bit 30
     * ================================================================ */
    if (dev->hw.gfx.rlc_bootload_status & 0x80000000) {
        /* MES PC_START from firmware headers (per-pipe).
         * Pipe 0 = scheduler (uni_mes), Pipe 1 = KIQ (mes1).
         * If not cached (e.g. PSP load was skipped), read directly from files. */
        ULONGLONG mes_start[2] = {
            dev->hw.gfx.mes_ucode_start,      /* pipe 0: uni_mes */
            dev->hw.gfx.mes_kiq_ucode_start,   /* pipe 1: mes1 (KIQ) */
        };

        /* Read MES firmware headers if not cached */
        if (mes_start[0] == 0 || mes_start[1] == 0) {
            char fw_dir[260] = {};
            GetEnvironmentVariableA("HSAKMT_FW_DIR", fw_dir, sizeof(fw_dir));
            /* Strip trailing whitespace — Windows cmd "set VAR=val &&" leaves
             * a trailing space before the && separator. */
            for (int i = (int)strlen(fw_dir) - 1; i >= 0 && (fw_dir[i] == ' ' || fw_dir[i] == '\t'); i--)
                fw_dir[i] = '\0';
            if (fw_dir[0]) {
                const char *mes_files[2] = {
                    "gc_12_0_1_uni_mes.bin",  /* pipe 0 */
                    "gc_12_0_1_mes1.bin",     /* pipe 1 */
                };
                for (int mi = 0; mi < 2; mi++) {
                    if (mes_start[mi] != 0) continue;
                    char path[520];
                    snprintf(path, sizeof(path), "%s\\%s", fw_dir, mes_files[mi]);
                    FILE *f = fopen(path, "rb");
                    if (f) {
                        ULONG hdr[16];
                        if (fread(hdr, 4, 16, f) == 16) {
                            /* v1 MES header: ucode_start at DWORDs 14/15
                             * (offsets 0x38/0x3C). NOT 13/14 like v2. */
                            USHORT ver_major = (USHORT)(hdr[2] & 0xFFFF);
                            ULONGLONG start;
                            if (ver_major >= 2) {
                                start = ((ULONGLONG)hdr[14] << 32) | hdr[13];
                            } else {
                                start = ((ULONGLONG)hdr[15] << 32) | hdr[14];
                            }
                            if (start != 0) {
                                mes_start[mi] = start;
                                pr_info("gpu_gfx: MES pipe%d start from %s: 0x%08x_%08x\n",
                                        mi, mes_files[mi],
                                        (ULONG)(start >> 32), (ULONG)(start & 0xFFFFFFFF));
                            }
                        }
                        fclose(f);
                    }
                }
            }
        }
        pr_info("gpu_gfx: MES ucode_start: pipe0=0x%llx pipe1=0x%llx\n",
                (unsigned long long)mes_start[0],
                (unsigned long long)mes_start[1]);

#define regCP_MES_PRGRM_CNTR_START_MES    0x2800
#define regCP_MES_PRGRM_CNTR_START_HI_MES 0x289d
#define regCP_MES_CNTL_MES                0x2807
#define regRLC_CP_SCHEDULERS              0x098a

#define MES_INVALIDATE_ICACHE  (1 << 4)
#define MES_PIPE0_RESET        (1 << 16)
#define MES_PIPE1_RESET        (1 << 17)
#define MES_PIPE0_ACTIVE       (1 << 26)
#define MES_PIPE1_ACTIVE       (1 << 27)
#define MES_HALT               (1 << 30)

/* MES IC/MD register offsets (BASE_IDX=1) — from gc_12_0_0_offset.h */
#define regCP_MES_IC_BASE_LO     0x5850
#define regCP_MES_IC_BASE_HI     0x5851
#define regCP_MES_IC_BASE_CNTL   0x5852
#define regCP_MES_MDBASE_LO_REG  0x5854
#define regCP_MES_MDBASE_HI_REG  0x5855
#define regCP_MES_MIBOUND_LO     0x585b
#define regCP_MES_MDBOUND_LO     0x585d

        /* Enable MES — matching amdgpu mes_v12_0_enable() sequence.
         * amdgpu loops over both pipes: pipe 0 first, then pipe 1.
         * For each pipe:
         *   1. grbm_select(3, pipe, 0, 0)
         *   2. Assert pipe reset
         *   3. Set PC_START
         *   4. Write CNTL with active bits (deassert reset + activate)
         *
         * Before enable, mes_v12_0_load_microcode sets IC_BASE_CNTL=0,
         * IC_BASE, MIBOUND, MDBASE, MDBOUND per pipe.
         * With AUTOLOAD, RLC sets IC_BASE/MDBASE but may not set
         * MIBOUND/MDBOUND. We set them explicitly here.
         *
         * CRITICAL: amdgpu's set_ucode_start_addr calls enable(false) first,
         * which does HALT + INVALIDATE_ICACHE + RESET on ALL pipes. Without
         * icache invalidation, MES reads stale cache data and never executes. */

        /* Read MES CNTL state from AUTOLOAD */
        {
            grbm_select(dev, 3, 0, 0, 0);
            ULONG cntl = gc1_mmio_rreg(dev, regCP_MES_CNTL_MES);
            pr_info("gpu_gfx: MES CNTL from AUTOLOAD = 0x%08x "
                    "(halt=%d, p0_active=%d, p1_active=%d)\n",
                    cntl, (cntl >> 30) & 1, (cntl >> 26) & 1, (cntl >> 27) & 1);
            grbm_select_reset(dev);
        }

        /* Step 2: Load MES firmware to GART buffers.
         * After AUTOLOAD, MES firmware is in TMR (Trusted Memory Region).
         * But TMR contents may not survive VFIO FLR. amdgpu loads MES
         * firmware to GART buffers (mes_v12_0_load_microcode) so MES
         * fetches from accessible memory. We do the same.
         *
         * Layout: allocate DMA pages for each pipe's firmware,
         * GART-map them, copy firmware code, set IC_BASE to GART addr. */
        ULONGLONG mes_ic_base[2] = {0, 0};
        {
            char fw_dir[260] = {};
            GetEnvironmentVariableA("HSAKMT_FW_DIR", fw_dir, sizeof(fw_dir));
            /* Strip trailing whitespace (Windows cmd "set" quirk) */
            for (int i = (int)strlen(fw_dir) - 1; i >= 0 && (fw_dir[i] == ' ' || fw_dir[i] == '\t'); i--)
                fw_dir[i] = '\0';
            const char *mes_files[2] = {
                "gc_12_0_1_uni_mes.bin",  /* pipe 0: scheduler */
                "gc_12_0_1_mes1.bin",     /* pipe 1: KIQ */
            };

            for (int mi = 0; mi < 2; mi++) {
                char path[520];
                snprintf(path, sizeof(path), "%s\\%s", fw_dir, mes_files[mi]);
                FILE *f = fopen(path, "rb");
                if (!f) {
                    pr_warn("gpu_gfx: cannot open %s for GART load\n", path);
                    continue;
                }
                fseek(f, 0, SEEK_END);
                long fsize = ftell(f);
                fseek(f, 0, SEEK_SET);

                /* Read header to get ucode_array_offset and ucode_size */
                ULONG hdr[20];
                if (fread(hdr, 4, 20, f) != 20) {
                    fclose(f);
                    continue;
                }
                /* v1 header: ucode_array_offset at DWORD 6 (0x18),
                 * common_ucode_size at DWORD 5 (0x14) */
                ULONG fw_offset = hdr[6];   /* ucode_array_offset_bytes */
                ULONG fw_size = hdr[5];     /* common_ucode_size_bytes */

                if (fw_offset == 0 || fw_size == 0 ||
                    fw_offset + fw_size > (ULONG)fsize) {
                    pr_warn("gpu_gfx: bad MES fw header: off=0x%x size=%u fsize=%ld\n",
                            fw_offset, fw_size, fsize);
                    fclose(f);
                    continue;
                }

                /* Read firmware blob */
                UCHAR *fw_buf = (UCHAR *)malloc(fw_size);
                if (!fw_buf) { fclose(f); continue; }
                fseek(f, fw_offset, SEEK_SET);
                if (fread(fw_buf, 1, fw_size, f) != fw_size) {
                    free(fw_buf); fclose(f); continue;
                }
                fclose(f);

                /* Allocate DMA pages for firmware (contiguous, 4KB aligned) */
                ULONG num_pages = (fw_size + 4095) / 4096;
                AMDGPU_ESCAPE_ALLOC_MEMORY_DATA alloc;
                memset(&alloc, 0, sizeof(alloc));
                alloc.Header.Command = AMDGPU_ESCAPE_ALLOC_MEMORY;
                alloc.Header.Size = sizeof(alloc);
                alloc.SizeInBytes = num_pages * 4096;
                alloc.Flags = AMDGPU_MEM_TYPE_SYSTEM | AMDGPU_MEM_FLAG_HOST_ACCESS |
                              AMDGPU_MEM_FLAG_UNCACHED;
                if (wddm_lite_escape(dev, &alloc, sizeof(alloc)) != 0 ||
                    !alloc.CpuAddress) {
                    pr_warn("gpu_gfx: MES pipe%d DMA alloc failed (%u pages)\n",
                            mi, num_pages);
                    free(fw_buf);
                    continue;
                }

                /* Copy firmware to DMA buffer */
                memcpy(alloc.CpuAddress, fw_buf, fw_size);
                /* Zero remaining bytes in last page */
                if (fw_size < num_pages * 4096)
                    memset((UCHAR *)alloc.CpuAddress + fw_size, 0,
                           num_pages * 4096 - fw_size);
                free(fw_buf);

                /* Get physical addresses for GART mapping */
                ULONGLONG *phys_addrs = (ULONGLONG *)malloc(num_pages * sizeof(ULONGLONG));
                if (!phys_addrs) continue;

                AMDGPU_ESCAPE_GET_PHYS_PAGES_DATA phys;
                for (ULONG pg = 0; pg < num_pages; pg++) {
                    memset(&phys, 0, sizeof(phys));
                    phys.Header.Command = AMDGPU_ESCAPE_GET_PHYS_PAGES;
                    phys.Header.Size = sizeof(phys);
                    phys.Handle = alloc.Handle;
                    phys.PageOffset = pg;
                    if (wddm_lite_escape(dev, &phys, sizeof(phys)) != 0 ||
                        phys.NumPages == 0) {
                        pr_warn("gpu_gfx: MES pipe%d GET_PHYS_PAGES failed pg=%u\n",
                                mi, pg);
                        phys_addrs[pg] = 0;
                    } else {
                        phys_addrs[pg] = phys.PhysAddrs[0];
                    }
                }

                /* GART-map all pages */
                ULONGLONG gpu_addr = gpu_gart_map(dev, phys_addrs, num_pages);
                free(phys_addrs);

                if (gpu_addr == 0) {
                    pr_warn("gpu_gfx: MES pipe%d GART map failed\n", mi);
                    continue;
                }

                mes_ic_base[mi] = gpu_addr;
                pr_info("gpu_gfx: MES pipe%d firmware loaded to GART: "
                        "gpu=0x%llx size=%u (%u pages)\n",
                        mi, (unsigned long long)gpu_addr, fw_size, num_pages);
            }
        }

        /* Step 3: Enable MES — matching kernel mes_v12_0 sequence:
         *   a) mes_v12_0_load_microcode: set IC_BASE_CNTL=0 per pipe
         *   b) mes_v12_0_enable(false): halt all pipes
         *   c) mes_v12_0_set_ucode_start_addr: set PC_START per pipe
         *   d) mes_v12_0_enable(true): activate all pipes
         */
#define regCP_MES_RS64_INSTR_PNTR_MES  0x2803
#define regCP_MES_RS64_GP0_LO_MES      0x2808
#define regCP_MES_RS64_GP0_HI_MES      0x2809
#define regCP_MES_RS64_GP4_MES         0x2810

        /* Step 3a: Set IC_BASE_CNTL=0 and dump AUTOLOAD state per pipe.
         * IC_BASE_CNTL=0 means: use VMID 0 for instruction cache access
         * (passthrough, no translation). Without this, MES can't fetch code. */
        for (ULONG mes_pipe = 0; mes_pipe < 2; mes_pipe++) {
            grbm_select(dev, 3, mes_pipe, 0, 0);

            /* Read AUTOLOAD state */
            ULONG ic_lo = gc1_mmio_rreg(dev, regCP_MES_IC_BASE_LO);
            ULONG ic_hi = gc1_mmio_rreg(dev, regCP_MES_IC_BASE_HI);
            ULONG ic_cntl = gc1_mmio_rreg(dev, regCP_MES_IC_BASE_CNTL);
            ULONG md_lo = gc1_mmio_rreg(dev, regCP_MES_MDBASE_LO_REG);
            ULONG md_hi = gc1_mmio_rreg(dev, regCP_MES_MDBASE_HI_REG);
            ULONG mibound = gc1_mmio_rreg(dev, regCP_MES_MIBOUND_LO);
            ULONG mdbound = gc1_mmio_rreg(dev, regCP_MES_MDBOUND_LO);
            ULONG pc_lo = gc1_mmio_rreg(dev, regCP_MES_PRGRM_CNTR_START_MES);
            ULONG pc_hi = gc1_mmio_rreg(dev, regCP_MES_PRGRM_CNTR_START_HI_MES);
            pr_info("gpu_gfx: MES pipe%u AUTOLOAD: IC=0x%08x_%08x IC_CNTL=0x%x "
                    "MD=0x%08x_%08x MIBOUND=0x%x MDBOUND=0x%x PC=0x%08x_%08x\n",
                    mes_pipe, ic_hi, ic_lo, ic_cntl,
                    md_hi, md_lo, mibound, mdbound, pc_hi, pc_lo);

            /* Set IC_BASE_CNTL=0 — critical for instruction fetch via VMID 0 */
            gc1_mmio_wreg(dev, regCP_MES_IC_BASE_CNTL, 0);
            if (ic_cntl != 0) {
                pr_info("gpu_gfx: MES pipe%u IC_BASE_CNTL cleared: 0x%x -> 0\n",
                        mes_pipe, ic_cntl);
            }

            /* Invalidate MES instruction cache (kernel mes_v12_0_load_microcode).
             * Without this, MES reads stale cache from before AUTOLOAD. */
#define regCP_MES_IC_OP_CNTL  0x2820
            {
                gc1_mmio_wreg(dev, regCP_MES_IC_OP_CNTL, 0x1); /* INVALIDATE_CACHE */
                for (int iv = 0; iv < 100; iv++) {
                    ULONG op = gc1_mmio_rreg(dev, regCP_MES_IC_OP_CNTL);
                    if ((op & 0x1) == 0) {
                        pr_info("gpu_gfx: MES pipe%u icache invalidated (%d polls)\n",
                                mes_pipe, iv);
                        break;
                    }
                    Sleep(1);
                }
                /* Prime the icache after invalidation */
                gc1_mmio_wreg(dev, regCP_MES_IC_OP_CNTL, 0x10); /* PRIME_ICACHE */
                for (int iv = 0; iv < 100; iv++) {
                    ULONG op = gc1_mmio_rreg(dev, regCP_MES_IC_OP_CNTL);
                    if (op & 0x20) { /* ICACHE_PRIMED */
                        pr_info("gpu_gfx: MES pipe%u icache primed (%d polls)\n",
                                mes_pipe, iv);
                        break;
                    }
                    Sleep(1);
                }
            }

            /* Try to redirect IC_BASE to GART (unencrypted firmware copy).
             * On VFIO, TMR firmware may be encrypted/inaccessible.
             * The kernel always sets IC_BASE to its own VRAM/GART copy. */
            if (mes_ic_base[mes_pipe] != 0) {
                /* Code sizes: pipe0=197280, pipe1=87376 */
                ULONG code_sz = (mes_pipe == 0) ? 197280 : 87376;
                ULONG data_sz = 524288;  /* Same for both pipes */

                ULONG new_ic_lo = (ULONG)(mes_ic_base[mes_pipe] & 0xFFFFFFFF);
                ULONG new_ic_hi = (ULONG)(mes_ic_base[mes_pipe] >> 32);
                gc1_mmio_wreg(dev, regCP_MES_IC_BASE_LO, new_ic_lo);
                gc1_mmio_wreg(dev, regCP_MES_IC_BASE_HI, new_ic_hi);
                gc1_mmio_wreg(dev, regCP_MES_MIBOUND_LO, code_sz - 1);

                /* Data section follows code contiguously in GART blob */
                ULONGLONG md_addr = mes_ic_base[mes_pipe] + code_sz;
                gc1_mmio_wreg(dev, regCP_MES_MDBASE_LO_REG,
                              (ULONG)(md_addr & 0xFFFFFFFF));
                gc1_mmio_wreg(dev, regCP_MES_MDBASE_HI_REG,
                              (ULONG)(md_addr >> 32));
                gc1_mmio_wreg(dev, regCP_MES_MDBOUND_LO, data_sz - 1);

                /* Readback to check if writes took effect */
                ULONG rb_ic_lo = gc1_mmio_rreg(dev, regCP_MES_IC_BASE_LO);
                ULONG rb_ic_hi = gc1_mmio_rreg(dev, regCP_MES_IC_BASE_HI);
                ULONG rb_md_lo = gc1_mmio_rreg(dev, regCP_MES_MDBASE_LO_REG);
                ULONG rb_md_hi = gc1_mmio_rreg(dev, regCP_MES_MDBASE_HI_REG);
                if (rb_ic_lo == new_ic_lo && rb_ic_hi == new_ic_hi) {
                    pr_info("gpu_gfx: MES pipe%u IC_BASE -> GART: 0x%08x_%08x "
                            "MD -> 0x%08x_%08x\n",
                            mes_pipe, rb_ic_hi, rb_ic_lo, rb_md_hi, rb_md_lo);
                } else {
                    pr_info("gpu_gfx: MES pipe%u IC_BASE WRITE-LOCKED "
                            "(wrote 0x%08x_%08x, got 0x%08x_%08x)\n",
                            mes_pipe, new_ic_hi, new_ic_lo, rb_ic_hi, rb_ic_lo);
                }
            }

            grbm_select_reset(dev);
        }

        /* Step 3b: Disable all MES pipes — matching mes_v12_0_enable(false).
         * The full disable (halt+reset+icache_inval) is needed because
         * PSP restores firmware header's PC_START during enable, which
         * is the correct virtual entry point for TMR firmware. The RS64
         * processor has internal virtual memory that maps these addresses
         * to TMR. Without this sequence, MES doesn't start. */
        {
            ULONG cntl = gc1_mmio_rreg(dev, regCP_MES_CNTL_MES);
            cntl &= ~(MES_PIPE0_ACTIVE | MES_PIPE1_ACTIVE);
            cntl |= MES_INVALIDATE_ICACHE | MES_PIPE0_RESET | MES_PIPE1_RESET | MES_HALT;
            gc1_mmio_wreg(dev, regCP_MES_CNTL_MES, cntl);
            pr_info("gpu_gfx: MES disabled: CNTL=0x%08x (halt+icache_inval+reset)\n", cntl);
        }

        /* Step 3c: Write PC_START per pipe.
         *
         * Use the firmware header's ucode_start >> 2. Despite IC_BASE
         * being locked to TMR, this value causes the HQD to start
         * consuming (tested empirically). The hardware relationship
         * between IC_BASE, PC_START and instruction fetch is complex
         * (RS64 has internal virtual memory). */
        for (ULONG mes_pipe = 0; mes_pipe < 2; mes_pipe++) {
            grbm_select(dev, 3, mes_pipe, 0, 0);

            ULONGLONG ucode_start = mes_start[mes_pipe];
            if (ucode_start == 0)
                ucode_start = 0xFFFFFFFFF0005000ULL;
            ULONGLONG pc_addr = ucode_start >> 2;
            ULONG pc_lo = (ULONG)(pc_addr & 0xFFFFFFFF);
            ULONG pc_hi = (ULONG)(pc_addr >> 32);

            gc1_mmio_wreg(dev, regCP_MES_PRGRM_CNTR_START_MES, pc_lo);
            gc1_mmio_wreg(dev, regCP_MES_PRGRM_CNTR_START_HI_MES, pc_hi);

            pr_info("gpu_gfx: MES pipe%u PC_START = 0x%08x_%08x "
                    "(ucode_start=0x%llx)\n",
                    mes_pipe, pc_hi, pc_lo,
                    (unsigned long long)ucode_start);

            grbm_select_reset(dev);
        }

        /* Step 3d: Enable all MES pipes — matching mes_v12_0_enable(true).
         * For each pipe: assert pipe reset, then activate (clear halt/reset). */
        for (ULONG mes_pipe = 0; mes_pipe < 2; mes_pipe++) {
            grbm_select(dev, 3, mes_pipe, 0, 0);

            /* Assert pipe reset (matches kernel: set reset bit per pipe) */
            ULONG cntl = gc1_mmio_rreg(dev, regCP_MES_CNTL_MES);
            if (mes_pipe == 0)
                cntl |= MES_PIPE0_RESET;
            else
                cntl |= MES_PIPE1_RESET;
            gc1_mmio_wreg(dev, regCP_MES_CNTL_MES, cntl);

            /* Activate: write CNTL starting from 0, set only ACTIVE bits.
             * This clears HALT, PIPE_RESET, INVALIDATE_ICACHE simultaneously. */
            ULONG new_cntl = MES_PIPE0_ACTIVE;
            if (mes_pipe == 1)
                new_cntl |= MES_PIPE1_ACTIVE;
            gc1_mmio_wreg(dev, regCP_MES_CNTL_MES, new_cntl);

            /* Wait and check IP */
            Sleep(200);
            ULONG ip_after = gc1_mmio_rreg(dev, regCP_MES_RS64_INSTR_PNTR_MES);
            ULONG cntl_after = gc1_mmio_rreg(dev, regCP_MES_CNTL_MES);
            ULONG pc_after_lo = gc1_mmio_rreg(dev, regCP_MES_PRGRM_CNTR_START_MES);
            ULONG pc_after_hi = gc1_mmio_rreg(dev, regCP_MES_PRGRM_CNTR_START_HI_MES);
            pr_info("gpu_gfx: MES pipe%u enabled: PC=0x%08x_%08x CNTL=0x%08x IP=0x%08x\n",
                    mes_pipe, pc_after_hi, pc_after_lo, cntl_after, ip_after);

            grbm_select_reset(dev);
        }

        /* Wait for MES KIQ to initialize — poll every 100ms for 5 seconds */
        pr_info("gpu_gfx: waiting for MES to initialize...\n");
        for (int mw = 0; mw < 50; mw++) {
            Sleep(100);
            grbm_select(dev, 3, 1, 0, 0);
            ULONG mes_ip = gc1_mmio_rreg(dev, regCP_MES_RS64_INSTR_PNTR_MES);
            grbm_select_reset(dev);
            if (mw % 10 == 0) {
                pr_info("gpu_gfx: MES KIQ t+%dms: IP=0x%08x\n",
                        (mw + 1) * 100, mes_ip);
            }
            /* Check if MES advanced past 0x800 */
            if (mes_ip != 0 && mes_ip != 0x800) {
                pr_info("gpu_gfx: MES KIQ ACTIVE at IP=0x%08x after %dms!\n",
                        mes_ip, (mw + 1) * 100);
                break;
            }
        }

        /* Check GFXHUB fault after MES enable */
        {
            ULONG fault = gfxhub_rreg(dev, regGCVM_L2_PROTECTION_FAULT_STATUS_LO32);
            if (fault != 0) {
                ULONG fa_lo = gfxhub_rreg(dev, regGCVM_L2_PROTECTION_FAULT_ADDR_LO32);
                ULONG fa_hi = gfxhub_rreg(dev, regGCVM_L2_PROTECTION_FAULT_ADDR_HI32);
                pr_err("gpu_gfx: GFXHUB FAULT after MES enable! "
                       "status=0x%08x addr=0x%08x_%08x\n",
                       fault, fa_hi, fa_lo);
            } else {
                pr_info("gpu_gfx: GFXHUB clean after MES enable\n");
            }
        }

        /* Tell RLC which is KIQ: me=3, pipe=1, queue=0 → (3<<5)|(1<<3)|0|0x80 */
        {
            ULONG sched = gc1_rreg(dev, regRLC_CP_SCHEDULERS);
            sched &= 0xFFFFFF00;
            sched |= (3 << 5) | (1 << 3) | 0 | 0x80; /* me=3, pipe=1, queue=0 */
            gc1_wreg(dev, regRLC_CP_SCHEDULERS, sched);
            pr_info("gpu_gfx: RLC_CP_SCHEDULERS = 0x%08x (KIQ=me3,pipe1,q0)\n", sched);
        }

        /* Read back MES state */
        {
            grbm_select(dev, 3, 0, 0, 0);
            ULONG mes_cntl = gc1_rreg(dev, regCP_MES_CNTL_MES);
            grbm_select_reset(dev);
            pr_info("gpu_gfx: MES CNTL readback = 0x%08x "
                    "(pipe0_active=%d, pipe1_active=%d, halt=%d)\n",
                    mes_cntl,
                    (mes_cntl >> 26) & 1, (mes_cntl >> 27) & 1,
                    (mes_cntl >> 30) & 1);
        }
        /* Diagnostic: read GFXHUB system aperture after setup */
        {
            ULONG sa_lo = gfxhub_rreg(dev, 0x1619); /* GCMC_VM_SYSTEM_APERTURE_LOW_ADDR */
            ULONG sa_hi = gfxhub_rreg(dev, 0x161A); /* GCMC_VM_SYSTEM_APERTURE_HIGH_ADDR */
            pr_info("gpu_gfx: GFXHUB system aperture: LOW=0x%08x HIGH=0x%08x "
                    "(covers 0x%llx - 0x%llx)\n",
                    sa_lo, sa_hi,
                    (unsigned long long)sa_lo << 18,
                    (unsigned long long)(sa_hi + 1) << 18);
        }

        /* ============================================================
         * MES KIQ Queue Setup + SET_HW_RESOURCES
         *
         * MES needs a KIQ ring to receive configuration messages.
         * We allocate a small ring in the existing VRAM mapping,
         * program the KIQ HQD via direct register writes, then
         * submit a SET_HW_RESOURCES packet telling MES about
         * available VMIDs, HQDs, and doorbells.
         * ============================================================ */
        {
            /* Use a 4KB region in our GART for the KIQ ring and control buffers.
             * Layout: ring=slot 250, rptr=slot 251, wptr=slot 252, mqd=slot 253 */
            struct GpuGmcConfig *gmc = &dev->hw.gmc;
            if (!gmc->gart_table_cpu_addr) {
                pr_warn("gpu_gfx: no GART — skipping MES KIQ setup\n");
                goto mes_done;
            }

            /* Allocate DMA buffers for KIQ ring and control structures */
            /* We'll use a simple VRAM region at a fixed offset for MES KIQ.
             * VRAM offset 5MB (after PSP ring at 4MB) */
            ULONGLONG mes_vram_offset = 5 * 1024 * 1024;
            ULONGLONG mes_vram_mc = dev->hw.gmc.vram_start + mes_vram_offset;
            UCHAR *mes_vram_cpu = NULL;

            /* Map VRAM for MES KIQ buffers (8 pages = 32KB) */
            {
                AMDGPU_ESCAPE_MAP_BAR_DATA map;
                memset(&map, 0, sizeof(map));
                map.Header.Command = AMDGPU_ESCAPE_MAP_BAR;
                map.Header.Size = sizeof(map);
                map.BarIndex = 0;  /* BAR0 = VRAM */
                map.Offset = mes_vram_offset;
                map.Length = 32768;
                if (wddm_lite_escape(dev, &map, sizeof(map)) == 0 && map.MappedAddress) {
                    mes_vram_cpu = (UCHAR *)map.MappedAddress;
                } else {
                    pr_warn("gpu_gfx: MES VRAM mapping failed — skipping KIQ setup\n");
                    goto mes_done;
                }
            }

            memset(mes_vram_cpu, 0, 32768);

            /* KIQ buffer layout in VRAM:
             * +0x0000: ring buffer (4KB)
             * +0x1000: RPTR writeback (4B)
             * +0x2000: WPTR poll (4B)
             * +0x3000: MQD (4KB)
             * +0x4000: EOP buffer (4KB)
             * +0x5000: fence/status (4B)
             * +0x6000: scheduler context (4KB) */
            ULONGLONG kiq_ring_mc = mes_vram_mc + 0x0000;
            ULONGLONG kiq_rptr_mc = mes_vram_mc + 0x1000;
            ULONGLONG kiq_wptr_mc = mes_vram_mc + 0x2000;
            ULONGLONG kiq_mqd_mc  = mes_vram_mc + 0x3000;
            ULONGLONG kiq_eop_mc  = mes_vram_mc + 0x4000;
            ULONGLONG kiq_fence_mc = mes_vram_mc + 0x5000;
            ULONGLONG kiq_schctx_mc = mes_vram_mc + 0x6000;

            volatile ULONG *kiq_ring = (volatile ULONG *)(mes_vram_cpu + 0x0000);
            volatile ULONG *kiq_rptr = (volatile ULONG *)(mes_vram_cpu + 0x1000);
            volatile ULONG *kiq_wptr = (volatile ULONG *)(mes_vram_cpu + 0x2000);
            volatile ULONG *kiq_fence = (volatile ULONG *)(mes_vram_cpu + 0x5000);

            /* Try KIQ on pipe 0 with MES_RING0 doorbell (0x00B).
             * Unified MES might use pipe 0 for KIQ instead of pipe 1.
             * Check env var to select pipe. */
            ULONG kiq_pipe = 1;
            ULONG kiq_doorbell_qword = 0x00C;  /* MES_RING1 for pipe 1 */
            char mes_pipe0[32] = {};
            GetEnvironmentVariableA("HSAKMT_MES_KIQ_PIPE0", mes_pipe0, sizeof(mes_pipe0));
            if (mes_pipe0[0] == '1') {
                kiq_pipe = 0;
                kiq_doorbell_qword = 0x00B;  /* MES_RING0 for pipe 0 */
                pr_info("gpu_gfx: using MES KIQ on pipe 0 (HSAKMT_MES_KIQ_PIPE0=1)\n");
            }
            ULONG kiq_doorbell_dword = kiq_doorbell_qword << 1;

            /* Build MQD in memory first — MES reads this for queue details.
             * MQD layout matches v12_compute_mqd / our GfxMqd struct.
             * Key fields are at the same offsets as HQD register indices. */
            volatile ULONG *mqd = (volatile ULONG *)(mes_vram_cpu + 0x3000);
            /* Clear MQD */
            memset((void *)mqd, 0, 4096);
            /* cp_mqd_base_addr (MQD offsets 134-135) */
            mqd[134] = (ULONG)(kiq_mqd_mc & 0xFFFFFFFF);
            mqd[135] = (ULONG)(kiq_mqd_mc >> 32);
            /* cp_hqd_vmid (131) */
            mqd[131] = 0;
            /* cp_hqd_pq_base (136-137) = ring_addr >> 8 */
            mqd[136] = (ULONG)((kiq_ring_mc >> 8) & 0xFFFFFFFF);
            mqd[137] = (ULONG)(kiq_ring_mc >> 40);
            /* cp_hqd_pq_rptr (138) */
            mqd[138] = 0;
            /* cp_hqd_pq_rptr_report_addr (139-140) */
            mqd[139] = (ULONG)(kiq_rptr_mc & 0xFFFFFFFF);
            mqd[140] = (ULONG)(kiq_rptr_mc >> 32);
            /* cp_hqd_pq_wptr_poll_addr (141-142) */
            mqd[141] = (ULONG)(kiq_wptr_mc & 0xFFFFFFFF);
            mqd[142] = (ULONG)((kiq_wptr_mc >> 32) & 0xFFFF);
            /* cp_hqd_pq_doorbell_control (143) */
            mqd[143] = (1 << 30) | (kiq_doorbell_dword << DOORBELL_OFFSET_SHIFT);
            /* cp_hqd_pq_control (145) */
            mqd[145] = (9 << 0) | (9 << 8) | (1 << 13) | (1 << 30) | (1u << 31) | (1 << 28);
            /* cp_hqd_persistent_state (150 in some layouts, using amdgpu's offset) */
            mqd[150] = (0x55 << 8) | 1;
            /* cp_hqd_eop_base_addr (152-153) */
            mqd[152] = (ULONG)((kiq_eop_mc >> 8) & 0xFFFFFFFF);
            mqd[153] = (ULONG)(kiq_eop_mc >> 40);
            /* cp_hqd_eop_control (154) */
            mqd[154] = 9;

            /* Program KIQ HQD via direct register writes (me=3, pipe=1, queue=0) */
            pr_info("gpu_gfx: programming MES KIQ HQD (ring=0x%llx)\n",
                    (unsigned long long)kiq_ring_mc);

            /* Use MMIO for ALL HQD register writes — grbm_select is MMIO,
             * so per-pipe context only works with MMIO register access. */
            grbm_select(dev, 3, kiq_pipe, 0, 0);
            {
                /* Disable doorbell first */
                gc0_mmio_wreg(dev, regCP_HQD_PQ_DOORBELL_CONTROL, 0);

                /* MQD base */
                gc0_mmio_wreg(dev, 0x1fa7 /* CP_MQD_BASE_ADDR */, (ULONG)(kiq_mqd_mc & 0xFFFFFFFF));
                gc0_mmio_wreg(dev, 0x1fa7 + 1, (ULONG)(kiq_mqd_mc >> 32));

                /* MQD control: VMID=0 */
                gc0_mmio_wreg(dev, 0x1fac + 1 /* CP_MQD_CONTROL */, 0);

                /* PQ base (ring buffer address >> 8) */
                gc0_mmio_wreg(dev, regCP_HQD_PQ_BASE, (ULONG)((kiq_ring_mc >> 8) & 0xFFFFFFFF));
                gc0_mmio_wreg(dev, regCP_HQD_PQ_BASE + 1, (ULONG)(kiq_ring_mc >> 40));

                /* RPTR report addr */
                gc0_mmio_wreg(dev, regCP_HQD_PQ_RPTR_REPORT_ADDR, (ULONG)(kiq_rptr_mc & 0xFFFFFFFF));
                gc0_mmio_wreg(dev, regCP_HQD_PQ_RPTR_REPORT_ADDR_HI, (ULONG)(kiq_rptr_mc >> 32));

                /* WPTR poll addr */
                gc0_mmio_wreg(dev, regCP_HQD_PQ_WPTR_POLL_ADDR, (ULONG)(kiq_wptr_mc & 0xFFFFFFFF));
                gc0_mmio_wreg(dev, regCP_HQD_PQ_WPTR_POLL_ADDR_HI, (ULONG)((kiq_wptr_mc >> 32) & 0xFFFF));

                /* PQ control */
                ULONG pq_ctrl = (9 << 0)     /* QUEUE_SIZE */
                              | (9 << 8)      /* RPTR_BLOCK_SIZE */
                              | (1 << 13)     /* UNORD_DISPATCH */
                              | (1 << 30)     /* PRIV_STATE */
                              | (1u << 31)    /* KMD_QUEUE */
                              | (1 << 28);    /* NO_UPDATE_RPTR */
                gc0_mmio_wreg(dev, regCP_HQD_PQ_CONTROL, pq_ctrl);

                /* Doorbell */
                ULONG db_ctrl = (1 << 30) | (kiq_doorbell_dword << DOORBELL_OFFSET_SHIFT);
                gc0_mmio_wreg(dev, regCP_HQD_PQ_DOORBELL_CONTROL, db_ctrl);

                /* VMID = 0 */
                gc0_mmio_wreg(dev, 0x1fac /* CP_HQD_VMID */, 0);

                /* Persistent state */
                gc0_mmio_wreg(dev, 0x1fab /* CP_HQD_PERSISTENT_STATE */, 0x55 << 8 | 1);

                /* EOP buffer */
                gc0_mmio_wreg(dev, 0x1fc3 /* CP_HQD_EOP_BASE_ADDR */, (ULONG)((kiq_eop_mc >> 8) & 0xFFFFFFFF));
                gc0_mmio_wreg(dev, 0x1fc3 + 1, (ULONG)(kiq_eop_mc >> 40));
                gc0_mmio_wreg(dev, 0x1fc5 /* CP_HQD_EOP_CONTROL */, 9);

                /* Activate */
                gc0_mmio_wreg(dev, regCP_HQD_ACTIVE, 1);
            }
            grbm_select_reset(dev);

            /* Verify KIQ HQD activation */
            grbm_select(dev, 3, kiq_pipe, 0, 0);
            ULONG kiq_active = gc0_mmio_rreg(dev, regCP_HQD_ACTIVE);
            grbm_select_reset(dev);
            /* Also verify PQ_BASE and doorbell via MMIO readback */
            grbm_select(dev, 3, kiq_pipe, 0, 0);
            ULONG kiq_pq_lo = gc0_mmio_rreg(dev, regCP_HQD_PQ_BASE);
            ULONG kiq_pq_hi = gc0_mmio_rreg(dev, regCP_HQD_PQ_BASE + 1);
            ULONG kiq_db = gc0_mmio_rreg(dev, regCP_HQD_PQ_DOORBELL_CONTROL);
            ULONG kiq_wptr_poll = gc0_mmio_rreg(dev, regCP_HQD_PQ_WPTR_POLL_ADDR);
            ULONG kiq_wptr_poll_hi = gc0_mmio_rreg(dev, regCP_HQD_PQ_WPTR_POLL_ADDR_HI);
            ULONG kiq_pq_ctl = gc0_mmio_rreg(dev, regCP_HQD_PQ_CONTROL);
            grbm_select_reset(dev);
            pr_info("gpu_gfx: MES KIQ HQD ACTIVE=%u PQ_BASE=0x%08x_%08x "
                    "DB_CTL=0x%08x WPTR_POLL=0x%04x_%08x PQ_CTL=0x%08x\n",
                    kiq_active & 1, kiq_pq_hi, kiq_pq_lo,
                    kiq_db, kiq_wptr_poll_hi, kiq_wptr_poll,
                    kiq_pq_ctl);

            /* Test: try writing 0xDEADBEEF to first DWORD of ring and see
             * if it survives (VRAM access verification) */
            kiq_ring[0] = 0xDEADBEEF;
            MemoryBarrier();
            ULONG readback = kiq_ring[0];
            pr_info("gpu_gfx: VRAM ring test: wrote 0xDEADBEEF, read 0x%08x (%s)\n",
                    readback, readback == 0xDEADBEEF ? "OK" : "MISMATCH!");

            /* Build SET_HW_RESOURCES packet (256 bytes = 64 DWORDs) */
            ULONG hw_res[64];
            memset(hw_res, 0, sizeof(hw_res));

            /* DW0: header — type=1 (SCHEDULER), opcode=0 (SET_HW_RSRC), dwsize=64
             * MES_API_HEADER bitfield: type [3:0], opcode [11:4], dwsize [19:12] */
            hw_res[0] = (1 << 0)   /* type = MES_API_TYPE_SCHEDULER */
                      | (0 << 4)   /* opcode = MES_SCH_API_SET_HW_RSRC */
                      | (64 << 12); /* dwsize = API_FRAME_SIZE_IN_DWORDS */

            /* DW1: vmid_mask_mmhub — VMIDs 8-15 for compute */
            hw_res[1] = 0xFF00;

            /* DW2: vmid_mask_gfxhub — VMIDs 4-15 for compute */
            hw_res[2] = 0xFFF0;

            /* DW3: gds_size */
            hw_res[3] = 0;

            /* DW4: paging_vmid */
            hw_res[4] = 0;

            /* DW5-12: compute_hqd_mask[8] — all queues available per pipe */
            for (int i = 0; i < 8; i++)
                hw_res[5 + i] = 0x7E; /* queues 1-6 available (reserve 0 for KIQ) */

            /* DW13-14: gfx_hqd_mask[2] */
            hw_res[13] = 0;
            hw_res[14] = 0;

            /* DW15-16: sdma_hqd_mask[2] */
            hw_res[15] = 0xFC; /* SDMA queues 2-7 */
            hw_res[16] = 0xFC;

            /* DW17-21: aggregated_doorbells[5] — unused for now */

            /* DW22-23: g_sch_ctx_gpu_mc_ptr (64-bit) */
            hw_res[22] = (ULONG)(kiq_schctx_mc & 0xFFFFFFFF);
            hw_res[23] = (ULONG)(kiq_schctx_mc >> 32);

            /* DW24-25: query_status_fence_gpu_mc_ptr */
            hw_res[24] = (ULONG)(kiq_fence_mc & 0xFFFFFFFF);
            hw_res[25] = (ULONG)(kiq_fence_mc >> 32);

            /* DW26-33: gc_base[8] */
            hw_res[26] = dev->hw.ip.gc_base;     /* BASE_IDX=0 */
            hw_res[27] = dev->hw.ip.gc_base1;    /* BASE_IDX=1 */

            /* DW34-41: mmhub_base[8] */
            hw_res[34] = dev->hw.ip.mmhub_base;

            /* DW42-49: osssys_base[8] — IH base */
            hw_res[42] = dev->hw.ip.ih_base;

            /* DW50-53: struct MES_API_STATUS (16 bytes = 4 DWORDs)
             * The kernel sets api_completion_fence_addr and _value here.
             * When MES completes the command, it writes fence_value to
             * fence_addr. Driver polls fence_addr for the expected value. */
            hw_res[50] = (ULONG)(kiq_fence_mc & 0xFFFFFFFF);  /* fence_addr_lo */
            hw_res[51] = (ULONG)(kiq_fence_mc >> 32);          /* fence_addr_hi */
            hw_res[52] = 1;  /* fence_value_lo (completion marker) */
            hw_res[53] = 0;  /* fence_value_hi */

            /* DW54: flags bitfield (was incorrectly at DW51!) */
            hw_res[54] = (1 << 0)   /* disable_reset */
                       | (1 << 1)   /* use_different_vmid_compute */
                       | (1 << 2)   /* disable_mes_log */
                       | (1 << 6)   /* enable_level_process_quantum_check */
                       | (1 << 10)  /* enable_reg_active_poll */
                       | (1 << 19); /* unmapped_doorbell_handling (2 bits, value=1) */

            /* DW55: oversubscription_timer (was incorrectly at DW52!) */
            hw_res[55] = 50;

            /* Write SET_HW_RESOURCES to KIQ ring */
            memcpy((void *)kiq_ring, hw_res, sizeof(hw_res));
            MemoryBarrier();

            /* Update WPTR in memory — use same value as doorbell (byte count).
             * HW polls WPTR_POLL_ADDR and compares to RPTR to detect work.
             * Both should be in same units for HW to match them correctly. */
            kiq_wptr[0] = 64 * 4;  /* 256 bytes = one 64-dword packet */
            kiq_wptr[1] = 0;
            MemoryBarrier();

            /* Notify MES of new WPTR — write doorbell AND MMIO WPTR register */
            /* Map doorbell BAR if not already mapped.
             * Try BAR indices 1, 2, 3 to find the 256MB doorbell BAR. */
            if (!dev->doorbell_base) {
                /* Dump all BARs for diagnostics */
                pr_info("gpu_gfx: BAR layout: NumBars=%u VramBar=%u MmioBar=%u\n",
                        dev->info.NumBars, dev->info.VramBarIndex, dev->info.MmioBarIndex);
                for (ULONG bi = 0; bi < dev->info.NumBars && bi < 8; bi++) {
                    pr_info("gpu_gfx:   BAR%u: phys=0x%llx len=0x%llx isMem=%d\n",
                            bi,
                            (unsigned long long)dev->info.Bars[bi].PhysicalAddress.QuadPart,
                            (unsigned long long)dev->info.Bars[bi].Length,
                            dev->info.Bars[bi].IsMemory);
                }

                /* Find the doorbell BAR — look for 256MB memory BAR that isn't VRAM */
                ULONG db_bar = 1;
                AMDGPU_ESCAPE_MAP_BAR_DATA dbmap;
                memset(&dbmap, 0, sizeof(dbmap));
                dbmap.Header.Command = AMDGPU_ESCAPE_MAP_BAR;
                dbmap.Header.Size = sizeof(dbmap);
                dbmap.BarIndex = db_bar;
                dbmap.Offset = 0;
                dbmap.Length = 4096;
                if (wddm_lite_escape(dev, &dbmap, sizeof(dbmap)) == 0 && dbmap.MappedAddress) {
                    dev->doorbell_base = dbmap.MappedAddress;
                    dev->doorbell_size = 4096;
                    pr_info("gpu_gfx: doorbell BAR%u mapped at %p for MES KIQ\n",
                            db_bar, dev->doorbell_base);
                }
            }

            if (dev->doorbell_base) {
                volatile ULONGLONG *kiq_db = (volatile ULONGLONG *)
                    ((UCHAR *)dev->doorbell_base + kiq_doorbell_qword * 8);
                pr_info("gpu_gfx: MES KIQ doorbell at BAR+0x%x, writing WPTR=%u\n",
                        (ULONG)(kiq_doorbell_qword * 8), 64 * 4);
                *kiq_db = 64 * 4;  /* byte-unit WPTR */
                MemoryBarrier();
            } else {
                pr_warn("gpu_gfx: no doorbell BAR — MES KIQ doorbell not sent\n");
            }
            /* Also write WPTR via MMIO register (dword count, matching memory) */
            grbm_select(dev, 3, kiq_pipe, 0, 0);
            gc0_wreg(dev, regCP_HQD_PQ_WPTR_LO, 64);
            gc0_wreg(dev, regCP_HQD_PQ_WPTR_HI, 0);
            grbm_select_reset(dev);
            pr_info("gpu_gfx: MES KIQ WPTR also written via MMIO\n");

            /* Wait for MES to process SET_HW_RESOURCES.
             * Poll the fence address for the expected value (1). */
            for (int fw = 0; fw < 50; fw++) {
                Sleep(100);
                MemoryBarrier();
                if (kiq_fence[0] == 1) {
                    pr_info("gpu_gfx: SET_HW_RESOURCES completed after %d ms\n",
                            (fw + 1) * 100);
                    break;
                }
                if (fw == 49)
                    pr_warn("gpu_gfx: SET_HW_RESOURCES fence timeout (5s)\n");
            }

            /* Check MES execution state + additional diagnostic registers */
            grbm_select(dev, 3, kiq_pipe, 0, 0);
            {
                /* MES scratch registers / GP registers (RS64 status indicators) */
                ULONG mes_gp0_lo = gc1_rreg(dev, 0x2910); /* GP0_LO */
                ULONG mes_gp0_hi = gc1_rreg(dev, 0x2911); /* GP0_HI */
                ULONG mes_gp4_lo = gc1_rreg(dev, 0x2918); /* GP4_LO */
                ULONG mes_instr = gc1_rreg(dev, 0x2908); /* INSTR_PNTR */
                pr_info("gpu_gfx: MES pipe1 RS64: INSTR_PNTR=0x%08x "
                        "GP0=0x%08x_%08x GP4=0x%08x\n",
                        mes_instr, mes_gp0_hi, mes_gp0_lo, mes_gp4_lo);
            }
            {
                ULONG mes_ic_lo = gc1_rreg(dev, 0x5850); /* CP_MES_IC_BASE_LO */
                ULONG mes_ic_hi = gc1_rreg(dev, 0x5851); /* CP_MES_IC_BASE_HI */
                ULONG mes_md_lo = gc1_rreg(dev, 0x5854); /* CP_MES_MDBASE_LO */
                ULONG mes_md_hi = gc1_rreg(dev, 0x5855); /* CP_MES_MDBASE_HI */
                ULONG mes_pc_lo = gc1_rreg(dev, regCP_MES_PRGRM_CNTR_START_MES);
                ULONG mes_pc_hi = gc1_rreg(dev, regCP_MES_PRGRM_CNTR_START_HI_MES);
                ULONG mes_cntl = gc1_rreg(dev, regCP_MES_CNTL_MES);
                pr_info("gpu_gfx: MES pipe1 state: IC_BASE=0x%08x_%08x "
                        "MDBASE=0x%08x_%08x PC_START=0x%08x_%08x CNTL=0x%08x\n",
                        mes_ic_hi, mes_ic_lo, mes_md_hi, mes_md_lo,
                        mes_pc_hi, mes_pc_lo, mes_cntl);
            }
            grbm_select_reset(dev);

            /* Also check pipe 0 */
            grbm_select(dev, 3, 0, 0, 0);
            {
                ULONG mes0_ic_lo = gc1_rreg(dev, 0x5850);
                ULONG mes0_ic_hi = gc1_rreg(dev, 0x5851);
                ULONG mes0_cntl = gc1_rreg(dev, regCP_MES_CNTL_MES);
                pr_info("gpu_gfx: MES pipe0 state: IC_BASE=0x%08x_%08x CNTL=0x%08x\n",
                        mes0_ic_hi, mes0_ic_lo, mes0_cntl);
            }
            grbm_select_reset(dev);

            /* Check fence/api_status */
            ULONG api_status = kiq_fence[0];
            pr_info("gpu_gfx: MES SET_HW_RESOURCES api_status = 0x%08x "
                    "(0=pending, 1=success)\n", api_status);

            /* Check KIQ RPTR — both memory and register */
            ULONG kiq_rptr_val = kiq_rptr[0];
            grbm_select(dev, 3, kiq_pipe, 0, 0);
            ULONG kiq_rptr_reg = gc0_rreg(dev, regCP_HQD_PQ_RPTR);
            ULONG kiq_wptr_reg = gc0_rreg(dev, regCP_HQD_PQ_WPTR_LO);
            ULONG kiq_active_reg = gc0_rreg(dev, regCP_HQD_ACTIVE);
            ULONG kiq_status_reg = gc0_rreg(dev, regCP_HQD_HQ_STATUS0);
            grbm_select_reset(dev);
            pr_info("gpu_gfx: MES KIQ: mem_RPTR=%u reg_RPTR=%u reg_WPTR=%u "
                    "ACTIVE=%u STATUS=0x%08x\n",
                    kiq_rptr_val, kiq_rptr_reg, kiq_wptr_reg,
                    kiq_active_reg & 1, kiq_status_reg);

            /* Also check ring content (first DWORD should be header=0x00400010) */
            pr_info("gpu_gfx: MES KIQ ring[0]=0x%08x ring[1]=0x%08x\n",
                    kiq_ring[0], kiq_ring[1]);
        }
mes_done:
        (void)0; /* label must be followed by a statement */
    } else {
        pr_info("gpu_gfx: skipping MES enable (AUTOLOAD not complete)\n");
    }

    /* Final MEC health check before returning */
    {
        grbm_select(dev, 1, 0, 0, 0);
        ULONG mec_ip_final = gc1_rreg(dev, regCP_MEC_RS64_INSTR_PNTR);
        ULONG mec_cntl_final = gc1_rreg(dev, regCP_MEC_RS64_CNTL);
        grbm_select_reset(dev);
        pr_info("gpu_gfx: FINAL MEC state: IP=0x%08x CNTL=0x%08x\n",
                mec_ip_final, mec_cntl_final);
    }

    /* ================================================================
     * Direct MEC Dispatch Test
     * MEC is running with RS64 firmware. Test if it processes a
     * directly-programmed HQD + AQL barrier packet without MES.
     * Uses GART-mapped ring buffer, RPTR, WPTR, EOP buffers.
     * ================================================================ */
    {
        grbm_select(dev, 1, 0, 0, 0);
        ULONG mec_ip_test = gc1_rreg(dev, regCP_MEC_RS64_INSTR_PNTR);
        grbm_select_reset(dev);

        /* Forward declaration — defined later in this file */
        extern ULONG gpu_check_gfxhub_fault(struct WddmLiteDevice *dev);

        if (mec_ip_test != 0 && mec_ip_test != 0x800) {
            pr_info("gpu_gfx: === MEC DIRECT DISPATCH TEST (IP=0x%08x) ===\n",
                    mec_ip_test);

            /* Allocate 5 DMA pages:
             *   page 0-1: ring buffer (8KB)
             *   page 2:   RPTR writeback
             *   page 3:   WPTR writeback
             *   page 4:   EOP buffer */
            #define TEST_NUM_PAGES 5
            ULONGLONG test_phys[TEST_NUM_PAGES] = {};
            void *test_cpu[TEST_NUM_PAGES] = {};

            int alloc_ok = 1;
            for (int pg = 0; pg < TEST_NUM_PAGES; pg++) {
                AMDGPU_ESCAPE_ALLOC_MEMORY_DATA alloc;
                memset(&alloc, 0, sizeof(alloc));
                alloc.Header.Command = AMDGPU_ESCAPE_ALLOC_MEMORY;
                alloc.Header.Size = sizeof(alloc);
                alloc.SizeInBytes = 4096;
                alloc.Flags = AMDGPU_MEM_TYPE_SYSTEM |
                              AMDGPU_MEM_FLAG_HOST_ACCESS |
                              AMDGPU_MEM_FLAG_UNCACHED;
                if (wddm_lite_escape(dev, &alloc, sizeof(alloc)) != 0 ||
                    !alloc.CpuAddress) {
                    alloc_ok = 0; break;
                }
                test_cpu[pg] = alloc.CpuAddress;
                memset(test_cpu[pg], 0, 4096);

                AMDGPU_ESCAPE_GET_PHYS_PAGES_DATA phys;
                memset(&phys, 0, sizeof(phys));
                phys.Header.Command = AMDGPU_ESCAPE_GET_PHYS_PAGES;
                phys.Header.Size = sizeof(phys);
                phys.Handle = alloc.Handle;
                phys.PageOffset = 0;
                if (wddm_lite_escape(dev, &phys, sizeof(phys)) != 0 ||
                    phys.NumPages == 0) {
                    alloc_ok = 0; break;
                }
                test_phys[pg] = phys.PhysAddrs[0];
            }

            if (alloc_ok) {
                /* GART-map all pages */
                ULONGLONG test_gpu = gpu_gart_map(dev, test_phys, TEST_NUM_PAGES);
                if (test_gpu) {
                    ULONGLONG ring_gpu = test_gpu;           /* pages 0-1 */
                    ULONGLONG rptr_gpu = test_gpu + 0x2000;  /* page 2 */
                    ULONGLONG wptr_gpu = test_gpu + 0x3000;  /* page 3 */
                    ULONGLONG eop_gpu  = test_gpu + 0x4000;  /* page 4 */

                    volatile ULONG *ring = (volatile ULONG *)test_cpu[0];
                    volatile ULONG *rptr_mem = (volatile ULONG *)test_cpu[2];
                    volatile ULONG *wptr_mem = (volatile ULONG *)test_cpu[3];

                    /* Build MQD */
                    ULONG mqd[256];
                    memset(mqd, 0, sizeof(mqd));
                    mqd[0] = 0xC0310800;  /* header */
                    mqd[11] = 1;          /* pipeline_stat_enable */
                    mqd[23] = 0xFFFFFFFF; mqd[24] = 0xFFFFFFFF; /* SE0/1 */
                    mqd[26] = 0xFFFFFFFF; mqd[27] = 0xFFFFFFFF; /* SE2/3 */
                    mqd[32] = 0x00000007; /* misc_reserved */

                    ULONGLONG mqd_gpu = test_gpu; /* reuse ring page as MQD area... */
                    /* Actually, MQD needs its own page. Use the ring as both. */
                    /* For test, use a simple non-AQL PM4 queue (simpler encoding). */

                    /* HQD registers at MQD offset 128 */
                    mqd[128] = (ULONG)(test_gpu) & 0xFFFFFFFC;       /* MQD base lo */
                    mqd[129] = (ULONG)(test_gpu >> 32);               /* MQD base hi */
                    mqd[130] = 1;       /* ACTIVE=1 (will be set separately) */
                    mqd[131] = 0;       /* VMID=0 */
                    mqd[132] = (0x55 << 8) | 1;  /* persistent_state */
                    mqd[133] = 0x2;     /* pipe_priority */
                    mqd[134] = 0xF;     /* queue_priority */
                    mqd[135] = 0x111;   /* quantum */

                    /* Ring buffer */
                    mqd[136] = (ULONG)((ring_gpu >> 8) & 0xFFFFFFFF);
                    mqd[137] = (ULONG)((ring_gpu >> 40) & 0xFFFFFFFF);
                    mqd[138] = 0;  /* RPTR = 0 */
                    mqd[139] = (ULONG)(rptr_gpu & 0xFFFFFFFC);
                    mqd[140] = (ULONG)((rptr_gpu >> 32) & 0xFFFF);
                    mqd[141] = (ULONG)(wptr_gpu & 0xFFFFFFF8);
                    mqd[142] = (ULONG)((wptr_gpu >> 32) & 0xFFFF);

                    /* Doorbell: use MEC ring 0 doorbell index.
                     * GFX12: DOORBELL_EN at bit 30, DOORBELL_OFFSET at bits [25:2] */
                    ULONG test_db_idx = 0x03; /* AMDGPU_NAVI10_DOORBELL_MEC_RING0 */
                    mqd[143] = ((test_db_idx * 2) << 2) | (1 << 30);

                    /* PQ control: 8KB ring = 2048 DWORDs, log2=11, field=11-2=9.
                     * Use PM4 mode (not AQL) for simplest possible test. */
                    mqd[145] = (9 & 0x3F) | (5 << 8);  /* queue_size=9, rptr_block_size=5 */
                    /* NO AQL flags — pure PM4 mode */

                    mqd[149] = 0x3 << 20;  /* IB control */
                    mqd[160] = 0x20004000; /* HQ_STATUS0 */
                    mqd[162] = (1 << 8);   /* mqd_control: priv_state=1 */
                    mqd[181] = 0;          /* AQL control = 0 (PM4 mode) */

                    /* EOP buffer */
                    mqd[165] = (ULONG)((eop_gpu >> 8) & 0xFFFFFFFF);
                    mqd[166] = (ULONG)((eop_gpu >> 40) & 0xFFFFFFFF);
                    mqd[167] = 6; /* eop_size for 4KB */

                    /* WPTR = 0 */
                    mqd[182] = 0;
                    mqd[183] = 0;

                    /* Record MEC IP before HQD programming */
                    grbm_select(dev, 1, 0, 0, 0);
                    ULONG ip_before = gc1_rreg(dev, regCP_MEC_RS64_INSTR_PNTR);
                    grbm_select_reset(dev);

                    /* Program HQD via bulk register copy */
                    pr_info("gpu_gfx: TEST: Programming HQD me=1 pipe=0 q=0\n");
                    grbm_select(dev, 1, 0, 0, 0);
                    gc0_wreg(dev, regCP_HQD_ACTIVE, 0);
                    for (ULONG i = 0; i < 56; i++)
                        gc0_wreg(dev, 0x1fa9 + i, mqd[128 + i]);
                    gc0_wreg(dev, regCP_HQD_ACTIVE, 1);
                    grbm_select_reset(dev);
                    gpu_hdp_flush(dev);

                    /* Check MEC health after HQD programming */
                    Sleep(100);
                    grbm_select(dev, 1, 0, 0, 0);
                    ULONG ip_after_hqd = gc1_rreg(dev, regCP_MEC_RS64_INSTR_PNTR);
                    ULONG cntl_after = gc1_rreg(dev, regCP_MEC_RS64_CNTL);
                    ULONG hqd_active_rb = gc0_rreg(dev, regCP_HQD_ACTIVE);
                    grbm_select_reset(dev);

                    pr_info("gpu_gfx: TEST: MEC after HQD: IP=0x%08x (was 0x%08x) "
                            "CNTL=0x%08x HQD_ACTIVE=%u\n",
                            ip_after_hqd, ip_before, cntl_after, hqd_active_rb);

                    if (ip_after_hqd == 0) {
                        pr_err("gpu_gfx: TEST: MEC CRASHED after HQD programming!\n");
                    } else {
                        /* Write PM4 NOP packet (8 bytes = 2 DWORDs).
                         * Type 3, opcode 0x10 (NOP), count=0 (1 DWORD body). */
                        memset((void *)ring, 0, 4096);
                        ring[0] = 0xC0001000; /* type3, opcode=NOP(0x10), count=0 */
                        ring[1] = 0x00000000; /* NOP body */
                        MemoryBarrier();

                        /* Set WPTR = 8 (8 bytes = 2 DWORDs of PM4 data) */
                        wptr_mem[0] = 8;  /* byte offset for PM4 WPTR */
                        wptr_mem[1] = 0;
                        MemoryBarrier();

                        /* HDP flush + doorbell */
                        gpu_hdp_flush(dev);
                        ULONG db_off = test_db_idx * 8;
                        volatile ULONGLONG *db = (volatile ULONGLONG *)
                            ((UCHAR *)dev->doorbell_base + db_off);
                        pr_info("gpu_gfx: TEST: Writing PM4 NOP doorbell at offset 0x%x, WPTR=8\n",
                                db_off);
                        *db = 8;
                        MemoryBarrier();

                        /* Wait for RPTR to advance (expect 8 for 2-DWORD PM4 NOP) */
                        for (int poll = 0; poll < 50; poll++) {
                            Sleep(20);
                            ULONG mem_rptr = rptr_mem[0];
                            if (mem_rptr != 0) {
                                pr_info("gpu_gfx: TEST: *** DISPATCH SUCCESS! *** "
                                        "RPTR=%u after %d ms\n",
                                        mem_rptr, (poll + 1) * 20);
                                break;
                            }
                            if (poll % 10 == 9) {
                                grbm_select(dev, 1, 0, 0, 0);
                                ULONG mec_ip_poll = gc1_rreg(dev,
                                    regCP_MEC_RS64_INSTR_PNTR);
                                ULONG reg_rptr = gc0_rreg(dev, regCP_HQD_PQ_RPTR);
                                grbm_select_reset(dev);
                                pr_info("gpu_gfx: TEST: t+%dms RPTR=%u "
                                        "reg_RPTR=%u MEC_IP=0x%08x\n",
                                        (poll + 1) * 20, rptr_mem[0],
                                        reg_rptr, mec_ip_poll);
                            }
                        }

                        /* Final state */
                        grbm_select(dev, 1, 0, 0, 0);
                        ULONG final_ip = gc1_rreg(dev, regCP_MEC_RS64_INSTR_PNTR);
                        ULONG final_rptr = gc0_rreg(dev, regCP_HQD_PQ_RPTR);
                        ULONG final_wptr = gc0_rreg(dev, regCP_HQD_PQ_WPTR_LO);
                        grbm_select_reset(dev);
                        ULONG fault = gpu_check_gfxhub_fault(dev);
                        pr_info("gpu_gfx: TEST: FINAL MEC_IP=0x%08x reg_RPTR=%u "
                                "reg_WPTR=%u mem_RPTR=%u FAULT=0x%08x\n",
                                final_ip, final_rptr, final_wptr,
                                rptr_mem[0], fault);
                    }
                } else {
                    pr_warn("gpu_gfx: TEST: GART map failed\n");
                }
            } else {
                pr_warn("gpu_gfx: TEST: DMA alloc failed\n");
            }
            pr_info("gpu_gfx: === MEC DIRECT DISPATCH TEST END ===\n");
        }
    }

    dev->hw.gfx_initialized = TRUE;
    pr_info("gpu_gfx: GFX engine initialization complete\n");
    return 0;
}

/* Program HQD from MQD via bulk register copy (exported for queues.cpp NOP test) */
void gpu_program_hqd_from_mqd(struct WddmLiteDevice *dev,
    ULONG me, ULONG pipe, ULONG queue, volatile ULONG *mqd)
{
    grbm_select(dev, me, pipe, queue, 0);
    gc0_wreg(dev, regCP_HQD_ACTIVE, 0); /* deactivate first */
    /* Bulk copy MQD[128..183] → regs [0x1fa9..0x1fe0] */
    for (ULONG i = 0; i < 56; i++)
        gc0_wreg(dev, 0x1fa9 + i, mqd[128 + i]);
    gc0_wreg(dev, regCP_HQD_ACTIVE, 1); /* activate */
    grbm_select_reset(dev);
    gpu_hdp_flush(dev);
    pr_info("gpu_program_hqd_from_mqd: me=%u pipe=%u q=%u activated\n", me, pipe, queue);
}

/* Check GFXHUB fault and return status (exported for queues.cpp) */
ULONG gpu_check_gfxhub_fault(struct WddmLiteDevice *dev)
{
    ULONG fault = gfxhub_rreg(dev, regGCVM_L2_PROTECTION_FAULT_STATUS_LO32);
    if (fault != 0) {
        ULONG fa_lo = gfxhub_rreg(dev, regGCVM_L2_PROTECTION_FAULT_ADDR_LO32);
        ULONG fa_hi = gfxhub_rreg(dev, regGCVM_L2_PROTECTION_FAULT_ADDR_HI32);
        pr_err("gpu_check_gfxhub_fault: FAULT! status=0x%08x addr=0x%08x_%08x\n",
               fault, fa_hi, fa_lo);
    }
    return fault;
}

/* Read HQD WPTR register for a specific me/pipe/queue (exported for queues.cpp) */
ULONG gpu_read_hqd_wptr(struct WddmLiteDevice *dev,
    ULONG me, ULONG pipe, ULONG queue)
{
    grbm_select(dev, me, pipe, queue, 0);
    ULONG wptr = gc0_rreg(dev, regCP_HQD_PQ_WPTR_LO);
    ULONG rptr = gc0_rreg(dev, regCP_HQD_PQ_RPTR);
    ULONG active = gc0_rreg(dev, regCP_HQD_ACTIVE);
    grbm_select_reset(dev);
    pr_info("gpu_read_hqd_wptr: me=%u pipe=%u q=%u: WPTR=%u RPTR=%u ACTIVE=%u\n",
            me, pipe, queue, wptr, rptr, active & 1);
    return wptr;
}

void gpu_gfx_cleanup(struct WddmLiteDevice *dev)
{
    if (!dev->hw.gfx_initialized)
        return;

    /* Dequeue any active HQDs */
    dequeue_all_hqds(dev);

    /* Re-enable GFXOFF power saving */
    gpu_enable_gfxoff(dev);

    memset(&dev->hw.gfx, 0, sizeof(dev->hw.gfx));
    dev->hw.gfx_initialized = FALSE;

    pr_info("gpu_gfx: cleanup done\n");
}


/* ======================================================================
 * Compute Queue Setup (Direct HQD Programming)
 * ====================================================================== */

/*
 * v12_compute_mqd: MQD structure for GFX12 compute queues.
 * 512 DWORDs (2048 bytes). HQD registers start at offset 0x80 (128).
 * Matches the hardware definition from amd_shared.h / v12_structs.h.
 * We only define the fields we need; the full struct is zero-initialized.
 */
struct v12_compute_mqd {
    ULONG header;                           /* 0: 0xC0310800 */
    ULONG reserved_1_22[22];                /* 1-22 */
    ULONG compute_static_thread_mgmt_se0;   /* 23 */
    ULONG compute_static_thread_mgmt_se1;   /* 24 */
    ULONG reserved_25;                       /* 25 */
    ULONG compute_static_thread_mgmt_se2;   /* 26 */
    ULONG compute_static_thread_mgmt_se3;   /* 27 */
    ULONG reserved_28_43[16];                /* 28-43 */
    ULONG compute_static_thread_mgmt_se4;   /* 44 */
    ULONG compute_static_thread_mgmt_se5;   /* 45 */
    ULONG compute_static_thread_mgmt_se6;   /* 46 */
    ULONG compute_static_thread_mgmt_se7;   /* 47 */
    ULONG reserved_48_127[80];               /* 48-127 */
    /* --- HQD registers start here (offset 0x80 = 128) --- */
    ULONG cp_mqd_base_addr_lo;              /* 128 → regCP_MQD_BASE_ADDR (0x1fa9) */
    ULONG cp_mqd_base_addr_hi;              /* 129 → regCP_MQD_BASE_ADDR_HI */
    ULONG cp_hqd_active;                    /* 130 → regCP_HQD_ACTIVE */
    ULONG cp_hqd_vmid;                      /* 131 → regCP_HQD_VMID */
    ULONG cp_hqd_persistent_state;          /* 132 → regCP_HQD_PERSISTENT_STATE */
    ULONG cp_hqd_pipe_priority;             /* 133 → regCP_HQD_PIPE_PRIORITY */
    ULONG cp_hqd_queue_priority;            /* 134 → regCP_HQD_QUEUE_PRIORITY */
    ULONG cp_hqd_quantum;                   /* 135 → regCP_HQD_QUANTUM */
    ULONG cp_hqd_pq_base_lo;               /* 136 → regCP_HQD_PQ_BASE */
    ULONG cp_hqd_pq_base_hi;               /* 137 → regCP_HQD_PQ_BASE_HI */
    ULONG cp_hqd_pq_rptr;                  /* 138 → regCP_HQD_PQ_RPTR */
    ULONG cp_hqd_pq_rptr_report_addr_lo;   /* 139 → regCP_HQD_PQ_RPTR_REPORT_ADDR */
    ULONG cp_hqd_pq_rptr_report_addr_hi;   /* 140 → regCP_HQD_PQ_RPTR_REPORT_ADDR_HI */
    ULONG cp_hqd_pq_wptr_poll_addr_lo;     /* 141 → regCP_HQD_PQ_WPTR_POLL_ADDR */
    ULONG cp_hqd_pq_wptr_poll_addr_hi;     /* 142 → regCP_HQD_PQ_WPTR_POLL_ADDR_HI */
    ULONG cp_hqd_pq_doorbell_control;       /* 143 → regCP_HQD_PQ_DOORBELL_CONTROL */
    ULONG reserved_144;                      /* 144 (gap) */
    ULONG cp_hqd_pq_control;               /* 145 → regCP_HQD_PQ_CONTROL */
    ULONG cp_hqd_ib_base_addr_lo;          /* 146 */
    ULONG cp_hqd_ib_base_addr_hi;          /* 147 */
    ULONG cp_hqd_ib_rptr;                  /* 148 */
    ULONG cp_hqd_ib_control;               /* 149 → regCP_HQD_IB_CONTROL */
    ULONG cp_hqd_iq_timer;                 /* 150 */
    ULONG cp_hqd_iq_rptr;                  /* 151 */
    ULONG cp_hqd_dequeue_request;           /* 152 */
    ULONG cp_hqd_dma_offload;              /* 153 */
    ULONG cp_hqd_sema_cmd;                 /* 154 */
    ULONG cp_hqd_msg_type;                 /* 155 */
    ULONG cp_hqd_atomic0_preop_lo;         /* 156 */
    ULONG cp_hqd_atomic0_preop_hi;         /* 157 */
    ULONG cp_hqd_atomic1_preop_lo;         /* 158 */
    ULONG cp_hqd_atomic1_preop_hi;         /* 159 */
    ULONG cp_hqd_hq_status0;               /* 160 → regCP_HQD_HQ_STATUS0 */
    ULONG cp_hqd_hq_control0;              /* 161 */
    ULONG cp_mqd_control;                   /* 162 → regCP_MQD_CONTROL */
    ULONG cp_hqd_hq_status1;               /* 163 */
    ULONG cp_hqd_hq_control1;              /* 164 */
    ULONG cp_hqd_eop_base_addr_lo;         /* 165 → regCP_HQD_EOP_BASE_ADDR */
    ULONG cp_hqd_eop_base_addr_hi;         /* 166 → regCP_HQD_EOP_BASE_ADDR_HI */
    ULONG cp_hqd_eop_control;              /* 167 → regCP_HQD_EOP_CONTROL */
    ULONG cp_hqd_eop_rptr;                 /* 168 */
    ULONG cp_hqd_eop_wptr;                 /* 169 */
    ULONG cp_hqd_eop_done_events;          /* 170 */
    ULONG cp_hqd_ctx_save_base_addr_lo;    /* 171 */
    ULONG cp_hqd_ctx_save_base_addr_hi;    /* 172 */
    ULONG cp_hqd_ctx_save_control;         /* 173 */
    ULONG cp_hqd_cntl_stack_offset;        /* 174 */
    ULONG cp_hqd_cntl_stack_size;          /* 175 */
    ULONG cp_hqd_wg_state_offset;          /* 176 */
    ULONG cp_hqd_ctx_save_size;            /* 177 */
    ULONG reserved_178;                      /* 178 */
    ULONG cp_hqd_error;                     /* 179 */
    ULONG cp_hqd_eop_wptr_mem;             /* 180 */
    ULONG cp_hqd_aql_control;              /* 181 → regCP_HQD_AQL_CONTROL */
    ULONG cp_hqd_pq_wptr_lo;              /* 182 → regCP_HQD_PQ_WPTR_LO */
    ULONG cp_hqd_pq_wptr_hi;              /* 183 → regCP_HQD_PQ_WPTR_HI */
    ULONG reserved_184_511[328];             /* 184-511 */
};

/*
 * PQ_CONTROL bit definitions (from v12_structs.h)
 */
#define PQ_CONTROL__QUEUE_SIZE__SHIFT       0
#define PQ_CONTROL__RPTR_BLOCK_SIZE__SHIFT  8
#define PQ_CONTROL__NO_UPDATE_RPTR          (1 << 27)
#define PQ_CONTROL__UNORD_DISPATCH          (1 << 28)
#define PQ_CONTROL__PRIV_STATE              (1 << 30)
#define PQ_CONTROL__KMD_QUEUE               (1 << 31)

/* Doorbell encoding */
#define DOORBELL_OFFSET__SHIFT              2
#define DOORBELL_EN                         (1 << 30)

/*
 * Doorbell index for MEC compute queues.
 * From amdgpu_doorbell.h: AMDGPU_NAVI10_DOORBELL_MEC_RING0 = 0x003.
 * Tinygrad uses this same value for all queues (all share doorbell index 3).
 * BAR byte offset = doorbell_index * 8 = 0x18.
 * DOORBELL_OFFSET register field = doorbell_index * 2 = 6.
 */
#define DOORBELL_MEC_RING_START             0x003

/*
 * Set up a compute queue by programming HQD registers directly.
 *
 * Follows the Python reference _activate_compute_queue_mmio():
 * 1. Allocate DMA buffer for MQD, GART-map it
 * 2. Fill MQD struct in DMA buffer
 * 3. Deactivate HQD, disable doorbell
 * 4. Write MQD base addr, then all HQD registers
 * 5. Enable doorbell, activate queue
 *
 * Returns 0 on success, -1 on failure.
 */
int gpu_setup_compute_queue(struct WddmLiteDevice *dev,
                            ULONG queue_idx,
                            ULONGLONG ring_addr, ULONG ring_size,
                            ULONGLONG rptr_addr, ULONGLONG wptr_addr,
                            ULONGLONG eop_addr, ULONG eop_size,
                            BOOLEAN aql)
{
    if (!dev->hw.gfx_initialized) {
        pr_err("gpu_queue: GFX engine not initialized\n");
        return -1;
    }

    if (queue_idx >= GPU_MAX_COMPUTE_QUEUES) {
        pr_err("gpu_queue: queue_idx %u out of range\n", queue_idx);
        return -1;
    }

    ULONG pipe = queue_idx / 4;
    ULONG queue = queue_idx % 4;
    ULONG doorbell_index = DOORBELL_MEC_RING_START + queue_idx;

    /* Check MEC health at queue setup entry */
    {
        grbm_select(dev, 1, 0, 0, 0);
        ULONG mec_ip_entry = gc1_rreg(dev, regCP_MEC_RS64_INSTR_PNTR);
        ULONG mec_cntl_entry = gc1_rreg(dev, regCP_MEC_RS64_CNTL);
        grbm_select_reset(dev);
        pr_info("gpu_queue: MEC at queue setup entry: IP=0x%08x CNTL=0x%08x\n",
                mec_ip_entry, mec_cntl_entry);
    }

    pr_info("gpu_queue: setting up queue %u (pipe=%u, queue=%u, aql=%d, doorbell=0x%x)\n",
            queue_idx, pipe, queue, aql, doorbell_index);
    pr_info("gpu_queue: ring=0x%llx size=0x%x rptr=0x%llx wptr=0x%llx\n",
            (unsigned long long)ring_addr, ring_size,
            (unsigned long long)rptr_addr, (unsigned long long)wptr_addr);
    pr_info("gpu_queue: eop=0x%llx eop_size=0x%x\n",
            (unsigned long long)eop_addr, eop_size);

    struct GpuComputeQueue *q = &dev->hw.queues[queue_idx];

    /*
     * Step 1: Allocate DMA buffer for MQD (4KB, page-aligned).
     * The MEC firmware reads the MQD from this GPU-accessible address.
     */
    if (!q->mqd_cpu_addr) {
        AMDGPU_ESCAPE_ALLOC_MEMORY_DATA alloc;
        memset(&alloc, 0, sizeof(alloc));
        alloc.Header.Command = AMDGPU_ESCAPE_ALLOC_MEMORY;
        alloc.Header.Size = sizeof(alloc);
        alloc.SizeInBytes = 4096;
        alloc.Flags = AMDGPU_MEM_TYPE_SYSTEM | AMDGPU_MEM_FLAG_HOST_ACCESS |
                      AMDGPU_MEM_FLAG_UNCACHED;

        if (wddm_lite_escape(dev, &alloc, sizeof(alloc)) != 0 ||
            alloc.Header.Status != 0 || !alloc.CpuAddress) {
            pr_err("gpu_queue: MQD alloc failed\n");
            return -1;
        }
        q->mqd_cpu_addr = alloc.CpuAddress;
        q->mqd_alloc_handle = alloc.Handle;

        /* Get physical address via GET_PHYS_PAGES */
        AMDGPU_ESCAPE_GET_PHYS_PAGES_DATA phys;
        memset(&phys, 0, sizeof(phys));
        phys.Header.Command = AMDGPU_ESCAPE_GET_PHYS_PAGES;
        phys.Header.Size = sizeof(phys);
        phys.Handle = alloc.Handle;
        phys.PageOffset = 0;

        if (wddm_lite_escape(dev, &phys, sizeof(phys)) != 0 ||
            phys.Header.Status != 0 || phys.NumPages == 0) {
            pr_err("gpu_queue: MQD GET_PHYS_PAGES failed\n");
            return -1;
        }
        q->mqd_bus_addr = phys.PhysAddrs[0];

        /* GART-map the MQD so GPU can access it */
        q->mqd_gpu_addr = gpu_gart_map(dev, &q->mqd_bus_addr, 1);
        if (q->mqd_gpu_addr == 0) {
            pr_err("gpu_queue: MQD GART map failed\n");
            return -1;
        }

        pr_info("gpu_queue: MQD cpu=%p bus=0x%llx gpu=0x%llx\n",
                q->mqd_cpu_addr,
                (unsigned long long)q->mqd_bus_addr,
                (unsigned long long)q->mqd_gpu_addr);
    }

    /*
     * Step 2: Fill MQD struct in DMA buffer.
     * Matches Python _init_compute_mqd().
     */
    ULONG *mqd = (ULONG *)q->mqd_cpu_addr;
    memset(mqd, 0, 4096);

    mqd[0] = 0xC0310800;  /* header */
    mqd[11] = 1;          /* compute_pipelinestat_enable */

    /* Thread management: enable all SEs */
    mqd[23] = 0xFFFFFFFF; mqd[24] = 0xFFFFFFFF;  /* SE0, SE1 */
    mqd[26] = 0xFFFFFFFF; mqd[27] = 0xFFFFFFFF;  /* SE2, SE3 */
    mqd[44] = 0xFFFFFFFF; mqd[45] = 0xFFFFFFFF;  /* SE4, SE5 */
    mqd[46] = 0xFFFFFFFF; mqd[47] = 0xFFFFFFFF;  /* SE6, SE7 */

    mqd[32] = 0x00000007; /* compute_misc_reserved */

    /* HQD registers start at offset 128 */
    mqd[128] = (ULONG)(q->mqd_gpu_addr) & 0xFFFFFFFC;          /* cp_mqd_base_addr */
    mqd[129] = (ULONG)(q->mqd_gpu_addr >> 32) & 0xFFFFFFFF;    /* cp_mqd_base_addr_hi */
    mqd[130] = 1;      /* cp_hqd_active */
    mqd[131] = 0;      /* cp_hqd_vmid = 0 (system aperture, matching tinygrad) */
    mqd[132] = (0x55 << 8) | 1;  /* cp_hqd_persistent_state: preload_size=0x55 (bits [17:8]), preload_req=1 (bit 0) */
    mqd[133] = 0x2;    /* cp_hqd_pipe_priority (tinygrad: 2) */
    mqd[134] = 0xF;    /* cp_hqd_queue_priority (tinygrad: 0xf) */
    mqd[135] = 0x111;  /* cp_hqd_quantum (tinygrad: 0x111) */

    /* Ring buffer base (address >> 8) */
    ULONGLONG pq_base = ring_addr >> 8;
    mqd[136] = (ULONG)(pq_base & 0xFFFFFFFF);
    mqd[137] = (ULONG)((pq_base >> 32) & 0xFFFFFFFF);

    mqd[138] = 0;    /* cp_hqd_pq_rptr = 0 */

    /* RPTR report address */
    mqd[139] = (ULONG)(rptr_addr & 0xFFFFFFFC);
    mqd[140] = (ULONG)((rptr_addr >> 32) & 0xFFFF);

    /* WPTR poll address */
    mqd[141] = (ULONG)(wptr_addr & 0xFFFFFFF8);
    mqd[142] = (ULONG)((wptr_addr >> 32) & 0xFFFF);

    /* Doorbell control: doorbell_offset = doorbell_index * 2, then shifted into register field.
     * Tinygrad: encode(doorbell_offset=doorbell*2, doorbell_en=1)
     * The DOORBELL_OFFSET field is a DWORD-granularity offset into the doorbell page.
     * BAR byte offset = doorbell_index * 8, DWORD offset = doorbell_index * 2. */
    mqd[143] = ((doorbell_index * 2) << DOORBELL_OFFSET__SHIFT) | DOORBELL_EN;

    /* PQ control: ring size + flags (matching tinygrad) */
    {
        /*
         * queue_size field = log2(ring_size_in_dwords) - 2
         * = log2(ring_size/4) - 2
         * Tinygrad: (ring_size//4).bit_length() - 2
         */
        ULONG queue_size_log2 = 0;
        ULONG rs = ring_size / 4;
        while (rs > 1) { rs >>= 1; queue_size_log2++; }
        queue_size_log2 -= 1;

        /* Match tinygrad exactly: unord_dispatch=0, no KMD_QUEUE */
        mqd[145] = (queue_size_log2 & 0x3F) |
                   (5 << PQ_CONTROL__RPTR_BLOCK_SIZE__SHIFT);

        /* For AQL queues, tinygrad adds: queue_full_en=1, slot_based_wptr=2, no_update_rptr.
         * GFX12 CP_HQD_PQ_CONTROL bit layout (from gc_12_0_0_sh_mask.h):
         *   QUEUE_FULL_EN  = bit 14 (0x0e)
         *   SLOT_BASED_WPTR = bits [19:18] (0x12)
         *   NO_UPDATE_RPTR = bit 27 (0x1b)
         */
        if (aql) {
            mqd[145] |= (1 << 27) |   /* no_update_rptr (bit 27, single XCC) */
                        (1 << 14) |   /* queue_full_en (bit 14) */
                        (2 << 18);    /* slot_based_wptr (bits [19:18]) */
        }
    }

    /* IB control: min_ib_avail_size=3 (tinygrad) */
    mqd[149] = 0x3 << 20;  /* CP_HQD_IB_CONTROL: min_ib_avail_size at bits [22:20] */

    /* HQ_STATUS0: tinygrad initializes to 0x20004000 */
    mqd[160] = 0x20004000;

    /* cp_mqd_control: priv_state=1 (CRITICAL — tinygrad sets this) */
    mqd[162] = (1 << 8);  /* priv_state bit */

    /* AQL control */
    if (aql) {
        mqd[181] = 1;  /* cp_hqd_aql_control = 1 for AQL queues */
    }

    /* EOP buffer */
    ULONGLONG eop_base = eop_addr >> 8;
    mqd[165] = (ULONG)(eop_base & 0xFFFFFFFF);
    mqd[166] = (ULONG)((eop_base >> 32) & 0xFFFFFFFF);
    /* cp_hqd_eop_control: eop_size = (eop_size_bytes/4).bit_length() - 2
     * Tinygrad: encode(eop_size=(eop_size//4).bit_length()-2) */
    {
        ULONG eop_dwords = eop_size / 4;
        ULONG eop_bits = 0;
        ULONG tmp = eop_dwords;
        while (tmp > 0) { eop_bits++; tmp >>= 1; }
        mqd[167] = (eop_bits >= 2) ? (eop_bits - 2) : 0;
    }

    /* WPTR lo/hi = 0 */
    mqd[182] = 0;
    mqd[183] = 0;

    /*
     * Step 3: Program HQD registers via bulk MQD copy (matching tinygrad).
     *
     * Tinygrad does:
     *   for i, reg in enumerate(range(regCP_MQD_BASE_ADDR, regCP_HQD_PQ_WPTR_HI+1)):
     *       wreg(reg, mqd[0x80 + i])
     *   regCP_HQD_ACTIVE.write(1)
     *
     * This copies MQD DWORDs [128..183] to registers [0x1fa9..0x1fe0],
     * then activates the queue. MQD[130] (HQD_ACTIVE) is written as 0
     * during the bulk copy, and set to 1 separately at the end.
     */
    grbm_select(dev, 1, pipe, queue, 0);

    /* Deactivate queue first */
    gc0_wreg(dev, regCP_HQD_ACTIVE, 0);

    /* Bulk copy MQD to HQD registers: MQD[128..183] → regs [0x1fa9..0x1fe0] */
    {
        const ULONG mqd_start = 128;     /* MQD DWORD offset (0x80) */
        const ULONG reg_start = 0x1fa9;  /* regCP_MQD_BASE_ADDR */
        const ULONG reg_end   = 0x1fe0;  /* regCP_HQD_PQ_WPTR_HI */
        const ULONG num_regs  = reg_end - reg_start + 1;  /* 56 registers */

        for (ULONG i = 0; i < num_regs; i++) {
            gc0_wreg(dev, reg_start + i, mqd[mqd_start + i]);
        }
    }

    /* Activate the queue (must be after bulk copy, tinygrad does this separately) */
    gc0_wreg(dev, regCP_HQD_ACTIVE, 1);

    /* Check MEC right after HQD activation (still in grbm_select me=1) */
    {
        ULONG mec_ip_post_hqd = gc1_rreg(dev, regCP_MEC_RS64_INSTR_PNTR);
        ULONG mec_cntl_post_hqd = gc1_rreg(dev, regCP_MEC_RS64_CNTL);
        pr_info("gpu_queue: MEC after HQD activate: IP=0x%08x CNTL=0x%08x\n",
                mec_ip_post_hqd, mec_cntl_post_hqd);
    }

    /* Deselect GRBM */
    grbm_select_reset(dev);

    /* Flush HDP */
    gpu_hdp_flush(dev);

    /* Verify activation */
    grbm_select(dev, 1, pipe, queue, 0);
    ULONG hqd_active = gc0_rreg(dev, regCP_HQD_ACTIVE);
    grbm_select_reset(dev);

    pr_info("gpu_queue: CP_HQD_ACTIVE readback = 0x%08x\n", hqd_active);

    if (hqd_active & 1) {
        pr_info("gpu_queue: queue %u activated successfully\n", queue_idx);

        /* Diagnostic: read back key HQD registers (using correct defines) */
        grbm_select(dev, 1, pipe, queue, 0);
        {
            ULONG pq_rptr     = gc0_rreg(dev, regCP_HQD_PQ_RPTR);
            ULONG pq_wptr_lo  = gc0_rreg(dev, regCP_HQD_PQ_WPTR_LO);
            ULONG pq_wptr_hi  = gc0_rreg(dev, regCP_HQD_PQ_WPTR_HI);
            ULONG pq_ctrl     = gc0_rreg(dev, regCP_HQD_PQ_CONTROL);
            ULONG db_ctrl     = gc0_rreg(dev, regCP_HQD_PQ_DOORBELL_CONTROL);
            ULONG hq_status   = gc0_rreg(dev, regCP_HQD_HQ_STATUS0);
            ULONG mqd_ctrl    = gc0_rreg(dev, regCP_MQD_CONTROL);
            ULONG persist     = gc0_rreg(dev, regCP_HQD_PERSISTENT_STATE);
            ULONG pq_base_lo  = gc0_rreg(dev, regCP_HQD_PQ_BASE);
            ULONG pq_base_hi  = gc0_rreg(dev, regCP_HQD_PQ_BASE + 1);
            ULONG rptr_rpt_lo = gc0_rreg(dev, regCP_HQD_PQ_RPTR_REPORT_ADDR);
            ULONG rptr_rpt_hi = gc0_rreg(dev, regCP_HQD_PQ_RPTR_REPORT_ADDR_HI);
            ULONG wptr_poll_lo= gc0_rreg(dev, regCP_HQD_PQ_WPTR_POLL_ADDR);
            ULONG wptr_poll_hi= gc0_rreg(dev, regCP_HQD_PQ_WPTR_POLL_ADDR_HI);
            ULONG eop_base_lo = gc0_rreg(dev, regCP_HQD_EOP_BASE_ADDR);
            ULONG eop_base_hi = gc0_rreg(dev, regCP_HQD_EOP_BASE_ADDR_HI);
            ULONG eop_ctrl    = gc0_rreg(dev, regCP_HQD_EOP_CONTROL);
            ULONG hqd_active  = gc0_rreg(dev, regCP_HQD_ACTIVE);

            pr_info("gpu_queue: DIAG q%u: ACTIVE=0x%x RPTR=0x%x WPTR=0x%x:%x\n",
                    queue_idx, hqd_active, pq_rptr, pq_wptr_hi, pq_wptr_lo);
            pr_info("gpu_queue: DIAG q%u: PQ_CTRL=0x%08x DB_CTRL=0x%08x PERSIST=0x%08x\n",
                    queue_idx, pq_ctrl, db_ctrl, persist);
            pr_info("gpu_queue: DIAG q%u: PQ_BASE=0x%x:%08x (expect ring>>8)\n",
                    queue_idx, pq_base_hi, pq_base_lo);
            pr_info("gpu_queue: DIAG q%u: RPTR_RPT=0x%x:%08x WPTR_POLL=0x%x:%08x\n",
                    queue_idx, rptr_rpt_hi, rptr_rpt_lo, wptr_poll_hi, wptr_poll_lo);
            pr_info("gpu_queue: DIAG q%u: EOP=0x%x:%08x CTRL=0x%08x HQ_STATUS=0x%08x\n",
                    queue_idx, eop_base_hi, eop_base_lo, eop_ctrl, hq_status);
            pr_info("gpu_queue: DIAG q%u: MQD_CTRL=0x%08x\n",
                    queue_idx, mqd_ctrl);
        }
        /* Also read MEC control */
        {
            ULONG mec_cntl = gc1_rreg(dev, regCP_MEC_RS64_CNTL);
            grbm_select_reset(dev);
            pr_info("gpu_queue: DIAG MEC: CNTL=0x%08x\n", mec_cntl);
        }
        grbm_select_reset(dev);

        /*
         * Quick doorbell + RPTR check:
         * Read HQD registers before and after a short delay.
         * If MEC is alive, HQ_STATUS0 or other fields may change.
         */
        {
            grbm_select(dev, 1, pipe, queue, 0);
            ULONG rptr_before = gc0_rreg(dev, regCP_HQD_PQ_RPTR);
            ULONG wptr_lo_before = gc0_rreg(dev, regCP_HQD_PQ_WPTR_LO);
            ULONG status_before = gc0_rreg(dev, regCP_HQD_HQ_STATUS0);
            grbm_select_reset(dev);

            pr_info("gpu_queue: q%u pre-use: RPTR=%u WPTR_LO=%u STATUS=0x%08x\n",
                    queue_idx, rptr_before, wptr_lo_before, status_before);
        }

        /* Diagnostic: granular MEC state polling after HQD activation */
        if (queue_idx == 0) {
            ULONG poll_times[] = {1, 5, 10, 20, 50, 100, 200};
            ULONG prev_time = 0;
            for (int pi = 0; pi < 7; pi++) {
                Sleep(poll_times[pi] - prev_time);
                prev_time = poll_times[pi];
                grbm_select(dev, 1, 0, 0, 0);
                ULONG ip = gc1_rreg(dev, regCP_MEC_RS64_INSTR_PNTR);
                ULONG cntl = gc1_rreg(dev, regCP_MEC_RS64_CNTL);
                ULONG rptr2 = gc0_rreg(dev, regCP_HQD_PQ_RPTR);
                grbm_select_reset(dev);
                pr_info("gpu_queue: q0 t+%ums: MEC_IP=0x%08x CNTL=0x%08x RPTR=%u\n",
                        poll_times[pi], ip, cntl, rptr2);
                if (ip == 0 && pi > 0) break;  /* MEC died, stop polling */
            }
        }

        /* Check for VM faults after queue activation */
        {
            ULONG fault = gfxhub_rreg(dev, regGCVM_L2_PROTECTION_FAULT_STATUS_LO32);
            if (fault != 0) {
                ULONG faddr_lo = gfxhub_rreg(dev, regGCVM_L2_PROTECTION_FAULT_ADDR_LO32);
                ULONG faddr_hi = gfxhub_rreg(dev, regGCVM_L2_PROTECTION_FAULT_ADDR_HI32);
                pr_err("gpu_queue: VM FAULT after q%u activate! "
                       "status=0x%08x addr=0x%08x_%08x\n",
                       queue_idx, fault, faddr_hi, faddr_lo);
            }
        }

        q->active = TRUE;
        q->me = 1;
        q->pipe = pipe;
        q->queue = queue;
        q->ring_addr = ring_addr;
        q->ring_size = ring_size;
        q->rptr_addr = rptr_addr;
        q->wptr_addr = wptr_addr;
        q->eop_addr = eop_addr;
        q->eop_size = eop_size;
        /* DOORBELL_OFFSET field value for register = doorbell_index * 2, shifted by 2 */
        q->doorbell_offset = (doorbell_index * 2) << DOORBELL_OFFSET__SHIFT;
        q->doorbell_index = doorbell_index;
        dev->hw.num_active_queues++;
        return 0;
    } else {
        pr_err("gpu_queue: queue %u activation failed\n", queue_idx);
        return -1;
    }
}

/*
 * Deactivate a compute queue's HQD.
 * Must be called before freeing queue DMA buffers so the GPU
 * stops referencing them.
 */
void gpu_deactivate_compute_queue(struct WddmLiteDevice *dev, ULONG queue_idx)
{
    if (!dev->hw.gfx_initialized || queue_idx >= GPU_MAX_COMPUTE_QUEUES)
        return;

    struct GpuComputeQueue *q = &dev->hw.queues[queue_idx];
    if (!q->active)
        return;

    ULONG pipe = queue_idx / 4;
    ULONG queue = queue_idx % 4;

    grbm_select(dev, 1, pipe, queue, 0);
    gc0_wreg(dev, regCP_HQD_ACTIVE, 0);
    gc0_wreg(dev, regCP_HQD_PQ_DOORBELL_CONTROL, 0);
    grbm_select_reset(dev);

    q->active = FALSE;
    if (dev->hw.num_active_queues > 0)
        dev->hw.num_active_queues--;

    pr_info("gpu_deactivate: queue %u (pipe=%u, queue=%u) deactivated\n",
            queue_idx, pipe, queue);
}

void gpu_read_hqd_diag(struct WddmLiteDevice *dev, ULONG queue_idx,
                        ULONG *out_rptr, ULONG *out_wptr_lo,
                        ULONG *out_status, ULONG *out_active)
{
    ULONG pipe = queue_idx / 4;
    ULONG queue = queue_idx % 4;

    grbm_select(dev, 1, pipe, queue, 0);
    if (out_rptr)    *out_rptr    = gc0_rreg(dev, regCP_HQD_PQ_RPTR);
    if (out_wptr_lo) *out_wptr_lo = gc0_rreg(dev, regCP_HQD_PQ_WPTR_LO);
    if (out_status)  *out_status  = gc0_rreg(dev, regCP_HQD_HQ_STATUS0);
    if (out_active)  *out_active  = gc0_rreg(dev, regCP_HQD_ACTIVE);
    grbm_select_reset(dev);

    /* Check for VM protection faults on GFXHUB */
    ULONG fault_status = gfxhub_rreg(dev, regGCVM_L2_PROTECTION_FAULT_STATUS_LO32);
    if (fault_status != 0) {
        ULONG fault_addr_lo = gfxhub_rreg(dev, regGCVM_L2_PROTECTION_FAULT_ADDR_LO32);
        ULONG fault_addr_hi = gfxhub_rreg(dev, regGCVM_L2_PROTECTION_FAULT_ADDR_HI32);
        pr_err("gpu_diag: GFXHUB VM FAULT! status=0x%08x addr=0x%08x_%08x\n",
               fault_status, fault_addr_hi, fault_addr_lo);
        pr_err("gpu_diag:   CID=%u VMID=%u RW=%u mapping_err=%u perm_faults=0x%x walker_err=%u\n",
               (fault_status >> 9) & 0x1FF,   /* CID */
               (fault_status >> 20) & 0xF,     /* VMID */
               (fault_status >> 18) & 1,        /* RW (0=read, 1=write) */
               (fault_status >> 8) & 1,         /* MAPPING_ERROR */
               (fault_status >> 4) & 0xF,       /* PERMISSION_FAULTS */
               (fault_status >> 1) & 0x7);      /* WALKER_ERROR */
    }
}


/* ======================================================================
 * GMC Cleanup
 * ====================================================================== */

void gpu_gmc_cleanup(struct WddmLiteDevice *dev)
{
    struct GpuGmcConfig *gmc = &dev->hw.gmc;

    if (!gmc->initialized)
        return;

    /* Disable VMID 0 contexts */
    if (dev->hw.ip.mmhub_base != 0)
        mmhub_wreg(dev, regMMVM_CONTEXT0_CNTL, 0);
    if (dev->hw.ip.gc_base != 0)
        gfxhub_wreg(dev, regGCVM_CONTEXT0_CNTL, 0);

    /* Free dummy page */
    if (gmc->dummy_page_handle) {
        AMDGPU_ESCAPE_ALLOC_DMA_DATA free_dma;
        memset(&free_dma, 0, sizeof(free_dma));
        free_dma.Header.Command = AMDGPU_ESCAPE_FREE_DMA;
        free_dma.Header.Size = sizeof(free_dma);
        free_dma.AllocationHandle = gmc->dummy_page_handle;
        wddm_lite_escape(dev, &free_dma, sizeof(free_dma));
    }

    /* Free GART table */
    if (gmc->gart_table_handle) {
        AMDGPU_ESCAPE_ALLOC_DMA_DATA free_dma;
        memset(&free_dma, 0, sizeof(free_dma));
        free_dma.Header.Command = AMDGPU_ESCAPE_FREE_DMA;
        free_dma.Header.Size = sizeof(free_dma);
        free_dma.AllocationHandle = gmc->gart_table_handle;
        wddm_lite_escape(dev, &free_dma, sizeof(free_dma));
    }

    memset(gmc, 0, sizeof(*gmc));
    dev->hw.gmc_initialized = FALSE;

    pr_info("gpu_gmc: cleanup done\n");
}

/* ======================================================================
 * IH (Interrupt Handler) Initialization
 *
 * Sets up interrupt rings BEFORE PSP firmware loading.
 * Tinygrad's init order requires IH to be ready before PSP so that
 * PSP/SMU completion interrupts have somewhere to land.
 * ====================================================================== */

/* IH register offsets from ih_base (OSSSYS, base_idx 0) */
#define regIH_RB_CNTL               0x0080
#define regIH_RB_BASE               0x0081
#define regIH_RB_BASE_HI            0x0082
#define regIH_RB_RPTR               0x0083
#define regIH_RB_WPTR               0x0084
#define regIH_RB_WPTR_ADDR_HI       0x0085
#define regIH_RB_WPTR_ADDR_LO       0x0086
#define regIH_DOORBELL_RPTR         0x0087

#define regIH_RB_CNTL_RING1         0x008C
#define regIH_RB_BASE_RING1         0x008D
#define regIH_RB_BASE_HI_RING1      0x008E
#define regIH_RB_RPTR_RING1         0x008F
#define regIH_RB_WPTR_RING1         0x0090
#define regIH_DOORBELL_RPTR_RING1   0x0093

#define regIH_CNTL                  0x00C0
#define regIH_CNTL2                 0x00C1
#define regIH_INT_FLOOD_CNTL        0x00D5
#define regIH_STORM_CLIENT_LIST_CNTL 0x00DA

#define IH_RING_SIZE        (256 * 1024)  /* 256 KB per ring */

/* IH_RB_CNTL bit fields */
#define IH_RB_CNTL_RB_ENABLE           (1 << 0)
#define IH_RB_CNTL_ENABLE_INTR         (1 << 0)  /* combined with RB_ENABLE */
#define IH_RB_CNTL_MC_SNOOP            (1 << 4)  /* actually bit 14 in some versions */
#define IH_RB_CNTL_WPTR_OVERFLOW_EN    (1 << 8)
#define IH_RB_CNTL_RPTR_REARM          (1 << 15)

static ULONG ih_rreg(struct WddmLiteDevice *dev, ULONG reg)
{
    ULONG offset = (dev->hw.ip.ih_base + reg) * 4;
    return wddm_lite_read_reg32(dev, offset);
}

static void ih_wreg(struct WddmLiteDevice *dev, ULONG reg, ULONG val)
{
    ULONG offset = (dev->hw.ip.ih_base + reg) * 4;
    wddm_lite_write_reg32(dev, offset, val);
}

int gpu_ih_init(struct WddmLiteDevice *dev)
{
    if (!dev->hw.ip_discovery_done) {
        pr_err("gpu_ih: IP discovery must be run first\n");
        return -1;
    }

    if (dev->hw.ip.ih_base == 0) {
        pr_warn("gpu_ih: IH base not found in IP discovery, skipping\n");
        return 0;
    }

    pr_info("gpu_ih: initializing IH (base=0x%04x)\n", dev->hw.ip.ih_base);

    /* Allocate ring 0 buffer (256 KB) via DMA */
    AMDGPU_ESCAPE_ALLOC_DMA_DATA dma_ring0;
    memset(&dma_ring0, 0, sizeof(dma_ring0));
    dma_ring0.Header.Command = AMDGPU_ESCAPE_ALLOC_DMA;
    dma_ring0.Header.Size = sizeof(dma_ring0);
    dma_ring0.Size = IH_RING_SIZE;

    if (wddm_lite_escape(dev, &dma_ring0, sizeof(dma_ring0)) != 0 ||
        dma_ring0.CpuAddress == NULL) {
        pr_err("gpu_ih: failed to allocate ring 0 buffer\n");
        return -1;
    }

    /* Zero the ring */
    memset(dma_ring0.CpuAddress, 0, IH_RING_SIZE);

    /* Allocate WPTR writeback page (4 KB) */
    AMDGPU_ESCAPE_ALLOC_DMA_DATA dma_wptr;
    memset(&dma_wptr, 0, sizeof(dma_wptr));
    dma_wptr.Header.Command = AMDGPU_ESCAPE_ALLOC_DMA;
    dma_wptr.Header.Size = sizeof(dma_wptr);
    dma_wptr.Size = 4096;

    if (wddm_lite_escape(dev, &dma_wptr, sizeof(dma_wptr)) != 0 ||
        dma_wptr.CpuAddress == NULL) {
        pr_err("gpu_ih: failed to allocate WPTR writeback page\n");
        return -1;
    }

    memset(dma_wptr.CpuAddress, 0, 4096);

    ULONGLONG ring_bus = dma_ring0.BusAddress;
    ULONGLONG wptr_bus = dma_wptr.BusAddress;

    pr_info("gpu_ih: ring0 at bus 0x%012llx, wptr_wb at bus 0x%012llx\n",
            (unsigned long long)ring_bus, (unsigned long long)wptr_bus);

    /*
     * Configure ring 0 (primary interrupt ring).
     * Matches tinygrad's IH init_hw().
     */

    /* Disable ring first */
    ih_wreg(dev, regIH_RB_CNTL, 0);

    /* Set ring base address (bus addr >> 8) */
    ih_wreg(dev, regIH_RB_BASE, (ULONG)(ring_bus >> 8));
    ih_wreg(dev, regIH_RB_BASE_HI, (ULONG)(ring_bus >> 40));

    /* rb_size = log2(ring_size_in_dwords) = log2(256K/4) = log2(65536) = 16 */
    ULONG rb_size_log2 = 16;

    /* Build RB_CNTL:
     * mc_space=4 (bits [7:4]) — use bus/physical addressing
     * rb_size (bits [6:1] in some encodings, or separate field)
     * wptr_overflow_enable, rptr_rearm, mc_snoop
     *
     * tinygrad sets: mc_space=4, rb_size=16, wptr_overflow_clear=1,
     *                wptr_overflow_enable=1, rptr_rearm=1, mc_snoop=1
     */
    ULONG rb_cntl = 0;
    rb_cntl |= (4 << 4);       /* mc_space = 4 (bus address) */
    rb_cntl |= (rb_size_log2 << 1); /* rb_size */
    rb_cntl |= (1 << 14);      /* mc_snoop */
    rb_cntl |= (1 << 8);       /* wptr_overflow_enable */
    rb_cntl |= (1 << 15);      /* rptr_rearm */
    rb_cntl |= (1 << 31);      /* wptr_overflow_clear */

    ih_wreg(dev, regIH_RB_CNTL, rb_cntl);

    /* Set WPTR writeback address */
    ih_wreg(dev, regIH_RB_WPTR_ADDR_LO, (ULONG)(wptr_bus & 0xFFFFFFFF));
    ih_wreg(dev, regIH_RB_WPTR_ADDR_HI, (ULONG)(wptr_bus >> 32));

    /* Reset read/write pointers */
    ih_wreg(dev, regIH_RB_RPTR, 0);
    ih_wreg(dev, regIH_RB_WPTR, 0);

    /* Configure doorbell for ring 0 (disabled — we poll) */
    ih_wreg(dev, regIH_DOORBELL_RPTR, 0);

    /* Global IH settings */
    /* Storm client list: enable client 18 as storm client */
    ULONG storm = ih_rreg(dev, regIH_STORM_CLIENT_LIST_CNTL);
    storm |= (1 << 18);  /* client18_is_storm_client */
    ih_wreg(dev, regIH_STORM_CLIENT_LIST_CNTL, storm);

    /* Flood control: enable */
    ih_wreg(dev, regIH_INT_FLOOD_CNTL, 1);

    /* Enable ring 0 with interrupts */
    rb_cntl = ih_rreg(dev, regIH_RB_CNTL);
    rb_cntl |= (1 << 0);       /* rb_enable */
    rb_cntl |= (1 << 20);      /* enable_intr (bit 20 in some encodings) */
    ih_wreg(dev, regIH_RB_CNTL, rb_cntl);

    /* Verify */
    ULONG cntl_read = ih_rreg(dev, regIH_RB_CNTL);
    ULONG rptr_read = ih_rreg(dev, regIH_RB_RPTR);
    ULONG wptr_read = ih_rreg(dev, regIH_RB_WPTR);
    pr_info("gpu_ih: ring0 enabled: RB_CNTL=0x%08x RPTR=0x%x WPTR=0x%x\n",
            cntl_read, rptr_read, wptr_read);

    return 0;
}
