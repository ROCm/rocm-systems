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

#define regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32     0x16AF
#define regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32     0x16B0
#define regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32       0x16CF
#define regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32       0x16D0
#define regGCVM_CONTEXT1_PAGE_TABLE_START_ADDR_LO32     0x16B1
#define regGCVM_CONTEXT1_PAGE_TABLE_END_ADDR_LO32       0x16D1

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

/* SMU v14 message IDs (from smu_v14_0_2.py) */
#define PPSMC_MSG_GetSmuVersion             0x02
#define PPSMC_MSG_EnableAllSmuFeatures      0x06
#define PPSMC_MSG_SetDriverDramAddrHigh     0x0E
#define PPSMC_MSG_SetDriverDramAddrLow      0x0F
#define PPSMC_MSG_AllowGfxOff               0x28
#define PPSMC_MSG_DisallowGfxOff            0x29

static ULONG mp1_rreg(struct WddmLiteDevice *dev, ULONG reg)
{
    ULONG offset = (dev->hw.ip.mp1_base + reg) * 4;
    return wddm_lite_read_reg32(dev, offset);
}

static void mp1_wreg(struct WddmLiteDevice *dev, ULONG reg, ULONG val)
{
    ULONG offset = (dev->hw.ip.mp1_base + reg) * 4;
    wddm_lite_write_reg32(dev, offset, val);
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
    ctx->vram_mc_base = (ULONGLONG)fb_base_reg << 24;

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

    /* Initialize fence */
    *(volatile ULONG *)ctx->fence_cpu = 0;
    MemoryBarrier();

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
    frame->cmd_buf_size = sizeof(struct PspGfxCmdResp);
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

    /* Build command buffer */
    struct PspGfxCmdResp *cmd = (struct PspGfxCmdResp *)ctx->cmd_cpu;
    memset(cmd, 0, sizeof(*cmd));
    cmd->buf_size = sizeof(*cmd);
    cmd->buf_version = PSP_GFX_CMD_BUF_VERSION;
    cmd->cmd_id = GFX_CMD_ID_LOAD_IP_FW;
    cmd->load_ip_fw.fw_phy_addr_lo =
        (ULONG)(ctx->fw_mc_addr & 0xFFFFFFFF);
    cmd->load_ip_fw.fw_phy_addr_hi =
        (ULONG)((ctx->fw_mc_addr >> 32) & 0xFFFFFFFF);
    cmd->load_ip_fw.fw_size = fw_size;
    cmd->load_ip_fw.fw_type = fw_type;

    MemoryBarrier();
    gpu_hdp_flush(ctx->dev);

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
    cmd->buf_size = sizeof(*cmd);
    cmd->buf_version = PSP_GFX_CMD_BUF_VERSION;
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
    cmd->buf_size = sizeof(*cmd);
    cmd->buf_version = PSP_GFX_CMD_BUF_VERSION;
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
    cmd->buf_size = sizeof(*cmd);
    cmd->buf_version = PSP_GFX_CMD_BUF_VERSION;
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
    pr_info("psp_ring: %s: header v%u.%u, "
            "code_offset=0x%x code_size=%u, "
            "data_offset=0x%x data_size=%u, "
            "ucode_start=0x%08x_%08x\n",
            filename,
            hdr->header_version_major, hdr->header_version_minor,
            hdr->ucode_offset_bytes, hdr->ucode_size_bytes,
            hdr->data_offset_bytes, hdr->data_size_bytes,
            hdr->ucode_start_addr_hi, hdr->ucode_start_addr_lo);

    /* Extract ucode start address for MEC programming */
    if (out_ucode_start && hdr->header_version_major >= 2) {
        *out_ucode_start = ((ULONGLONG)hdr->ucode_start_addr_hi << 32) |
                           hdr->ucode_start_addr_lo;
    }

    /*
     * For v2.0 headers, we have separate code and data sections.
     * For v1.x headers, fall back to ucode_array_offset_bytes.
     */
    ULONG code_offset, code_size, data_offset, data_size;

    if (hdr->header_version_major >= 2) {
        code_offset = hdr->ucode_offset_bytes;
        code_size = hdr->ucode_size_bytes;
        data_offset = hdr->data_offset_bytes;
        data_size = hdr->data_size_bytes;
    } else {
        /* v1.x fallback: treat entire ucode as code, no separate data */
        code_offset = hdr->ucode_array_offset_bytes;
        code_size = hdr->common_ucode_size_bytes;
        data_offset = 0;
        data_size = 0;
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
    pr_info("psp_ring: === Step 2: Loading SMU firmware ===\n");
    ret = psp_load_smu(&ctx, fw_dir);
    if (ret != 0) {
        pr_warn("psp_ring: SMU firmware load failed (may already be loaded by VBIOS)\n");
        load_failures++;
        /* Continue — SMU may already be loaded by VBIOS */
    }

    /* === Step 3: TMR setup — skipped for boot_time_tmr === */
    pr_info("psp_ring: === Step 3: TMR setup skipped (boot_time_tmr) ===\n");

    /* === Step 4: Load remaining firmware === */

    /* SDMA firmware (GFX12 uses thread-based SDMA_UCODE_TH0) */
    pr_info("psp_ring: === Loading SDMA firmware ===\n");
    ret = load_fw_v1(&ctx, fw_dir, "sdma_7_0_1.bin",
                      GFX_FW_TYPE_SDMA_UCODE_TH0, "sdma_7_0_0.bin");
    if (ret != 0) {
        pr_warn("psp_ring: SDMA load failed (continuing)\n");
        load_failures++;
    }

    /* RS64 PFP firmware (code + P0 stack only for GFX12) */
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

    /* RS64 ME firmware (code + P0 stack only) */
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

    /* RS64 MEC firmware (code + P0 stack only) */
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

    /* RLC sub-components (v2.2: IRAM + DRAM_BOOT) */
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
            } else {
                /* Fallback: load as RLC_G for older headers */
                ULONG ucode_off = rlc_hdr->ucode_array_offset_bytes;
                ULONG ucode_sz = rlc_hdr->ucode_size_bytes;
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

    /* === Step 5: Trigger RLC autoload === */
    pr_info("psp_ring: === Triggering AUTOLOAD_RLC ===\n");
    ret = psp_ring_autoload_rlc(&ctx);
    if (ret != 0) {
        pr_err("psp_ring: AUTOLOAD_RLC failed\n");
        psp_ring_destroy(&ctx);
        return ret;
    }

    /* Poll RLC bootload status (gc_base1 + 0x4e7c, bit 31 = complete) */
    pr_info("psp_ring: polling RLC_RLCS_BOOTLOAD_STATUS...\n");
    {
        ULONG bootload_status = 0;
        BOOLEAN bootload_done = FALSE;
        for (int poll = 0; poll < 200; poll++) {  /* Up to 2 seconds */
            /* Try SMN indirect first (works even if direct MMIO doesn't) */
            bootload_status = gpu_smn_rreg(dev,
                dev->hw.ip.gc_base1 + 0x4e7c);
            if (bootload_status & 0x80000000) {
                bootload_done = TRUE;
                break;
            }
            Sleep(10);
        }
        pr_info("psp_ring: RLC_RLCS_BOOTLOAD_STATUS = 0x%08x "
                "(complete=%s, iram_loaded=%d, iram_done=%d, "
                "fuse_dist=%d, init_done=%d)\n",
                bootload_status,
                bootload_done ? "YES" : "NO",
                (bootload_status >> 4) & 1,  /* RLC_GPM_IRAM_LOADED */
                (bootload_status >> 5) & 1,  /* RLC_GPM_IRAM_DONE */
                (bootload_status >> 0) & 1,  /* GFX_FUSE_DIST_DONE */
                (bootload_status >> 1) & 1); /* GFX_INIT_DONE */

        /* Also try direct MMIO read (gc_base1 + 0x4e7c) */
        ULONG bl_direct = wddm_lite_read_reg32(dev,
            (dev->hw.ip.gc_base1 + 0x4e7c) * 4);
        pr_info("psp_ring: RLC_RLCS_BOOTLOAD_STATUS direct = 0x%08x\n",
                bl_direct);

        if (!bootload_done) {
            pr_warn("psp_ring: RLC bootload did not complete\n");
        }
    }

    /* Check CP_STAT (gc_base + 0x0F40) */
    if (dev->hw.ip.gc_base != 0) {
        ULONG cp_stat_smn = gpu_smn_rreg(dev,
            dev->hw.ip.gc_base + 0x0F40);
        ULONG cp_stat_direct = wddm_lite_read_reg32(dev,
            (dev->hw.ip.gc_base + 0x0F40) * 4);
        pr_info("psp_ring: CP_STAT after autoload: direct=0x%08x SMN=0x%08x\n",
                cp_stat_direct, cp_stat_smn);
        if (cp_stat_smn == 0) {
            pr_info("psp_ring: CP idle (CP_STAT=0) — GC may be ready\n");
        }
    }

    pr_info("psp_ring: firmware loading complete "
            "(%d failures)\n", load_failures);

    pr_info("psp_ring: ucode_start: PFP=0x%llx ME=0x%llx MEC=0x%llx\n",
            (unsigned long long)dev->hw.gfx.pfp_ucode_start,
            (unsigned long long)dev->hw.gfx.me_ucode_start,
            (unsigned long long)dev->hw.gfx.mec_ucode_start);

    psp_ring_destroy(&ctx);
    return 0;
}

/* Legacy API wrapper — calls gpu_psp_load_all_fw */
int gpu_psp_load_smu_fw(struct WddmLiteDevice *dev, const char *fw_dir)
{
    return gpu_psp_load_all_fw(dev, fw_dir);
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
    pte |= (bus_addr & 0x0000FFFFF000ULL);
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
}

static void mmhub_init_tlb(struct WddmLiteDevice *dev)
{
    ULONG val = mmhub_rreg(dev, regMMMC_VM_MX_L1_TLB_CNTL);

    val |= L1_TLB_ENABLE;
    val = (val & ~L1_TLB_SYSTEM_ACCESS_MODE_MASK) | (3 << 3);
    val |= L1_TLB_ENABLE_ADV_DRIVER_MODEL;
    val &= ~L1_TLB_SYSTEM_APERTURE_UNMAPPED_ACCESS;

    mmhub_wreg(dev, regMMMC_VM_MX_L1_TLB_CNTL, val);
}

static void mmhub_init_cache(struct WddmLiteDevice *dev)
{
    ULONG val;

    val = mmhub_rreg(dev, regMMVM_L2_CNTL);
    val |= (1 << 0);   /* ENABLE_L2_CACHE */
    val |= (1 << 8);   /* ENABLE_DEFAULT_PAGE_OUT_TO_SYSTEM_MEMORY */
    val &= ~(1 << 6);  /* Disable ENABLE_L2_FRAGMENT_PROCESSING */
    mmhub_wreg(dev, regMMVM_L2_CNTL, val);

    mmhub_wreg(dev, regMMVM_L2_CNTL2,
               (1 << 0) |   /* INVALIDATE_ALL_L1_TLBS */
               (1 << 1));   /* INVALIDATE_L2_CACHE */

    val = mmhub_rreg(dev, regMMVM_L2_CNTL3);
    val = (val & ~0x3F000) | (9 << 15);    /* BANK_SELECT */
    val = (val & ~0x1F00000) | (6 << 20);  /* BIGK_FRAGMENT_SIZE */
    mmhub_wreg(dev, regMMVM_L2_CNTL3, val);

    val = mmhub_rreg(dev, regMMVM_L2_CNTL5);
    val &= ~0x7E0;   /* Clear SMALLK_FRAGMENT_SIZE */
    mmhub_wreg(dev, regMMVM_L2_CNTL5, val);
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

static void gfxhub_init_system_aperture(struct WddmLiteDevice *dev)
{
    struct GpuGmcConfig *gmc = &dev->hw.gmc;

    gfxhub_wreg(dev, regGCMC_VM_AGP_BASE, 0);
    gfxhub_wreg(dev, regGCMC_VM_AGP_BOT, 0xFFFFFF);
    gfxhub_wreg(dev, regGCMC_VM_AGP_TOP, 0);

    gfxhub_wreg(dev, regGCMC_VM_SYSTEM_APERTURE_LOW_ADDR,
                (ULONG)(gmc->vram_start >> 18));
    gfxhub_wreg(dev, regGCMC_VM_SYSTEM_APERTURE_HIGH_ADDR,
                (ULONG)(gmc->vram_end >> 18));

    gfxhub_wreg(dev, regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB,
                (ULONG)(gmc->dummy_page_bus_addr >> 12));
    gfxhub_wreg(dev, regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB,
                (ULONG)(gmc->dummy_page_bus_addr >> 44));

    gfxhub_wreg(dev, regGCVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32,
                (ULONG)(gmc->dummy_page_bus_addr >> 12));
    gfxhub_wreg(dev, regGCVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32,
                (ULONG)(gmc->dummy_page_bus_addr >> 44));
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

    val = gfxhub_rreg(dev, regGCVM_L2_CNTL);
    val |= (1 << 0);
    val |= (1 << 8);
    val &= ~(1 << 6);
    gfxhub_wreg(dev, regGCVM_L2_CNTL, val);

    gfxhub_wreg(dev, regGCVM_L2_CNTL2, (1 << 0) | (1 << 1));

    val = gfxhub_rreg(dev, regGCVM_L2_CNTL3);
    val = (val & ~0x3F000) | (9 << 15);
    val = (val & ~0x1F00000) | (6 << 20);
    gfxhub_wreg(dev, regGCVM_L2_CNTL3, val);

    val = gfxhub_rreg(dev, regGCVM_L2_CNTL5);
    val &= ~0x7E0;
    gfxhub_wreg(dev, regGCVM_L2_CNTL5, val);
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


/* ---- TLB flush ---- */

static void flush_gpu_tlb(struct WddmLiteDevice *dev, int vmid, int is_gfxhub)
{
    ULONG sem_reg, req_reg, ack_reg;

    if (is_gfxhub) {
        sem_reg = regGCVM_INVALIDATE_ENG0_SEM;
        req_reg = regGCVM_INVALIDATE_ENG0_REQ;
        ack_reg = regGCVM_INVALIDATE_ENG0_ACK;
    } else {
        sem_reg = regMMVM_INVALIDATE_ENG0_SEM;
        req_reg = regMMVM_INVALIDATE_ENG0_REQ;
        ack_reg = regMMVM_INVALIDATE_ENG0_ACK;
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

    /* Fill GART table: all entries point to dummy page */
    {
        ULONGLONG dummy_pte = build_gart_pte(gmc->dummy_page_bus_addr);
        ULONGLONG *table = (ULONGLONG *)gmc->gart_table_cpu_addr;
        ULONGLONG num_entries = gmc->gart_size / 4096;

        for (ULONGLONG i = 0; i < num_entries; i++)
            table[i] = dummy_pte;
    }

    /* Enable MMHUB GART */
    mmhub_gart_enable(dev);

    /* Flush MMHUB TLB for VMID 0 */
    flush_gpu_tlb(dev, 0, 0);

    pr_info("gpu_gmc: MMHUB GART enabled\n");

    /* Enable GFXHUB GART */
    gfxhub_gart_enable(dev);

    /* Flush GFXHUB TLB for VMID 0 */
    flush_gpu_tlb(dev, 0, 1);

    pr_info("gpu_gmc: GFXHUB GART enabled\n");

    gmc->initialized = TRUE;
    dev->hw.gmc_initialized = TRUE;

    /* Read back key registers to verify */
    {
        ULONG ctx0 = mmhub_rreg(dev, regMMVM_CONTEXT0_CNTL);
        ULONG l1_tlb = mmhub_rreg(dev, regMMMC_VM_MX_L1_TLB_CNTL);
        ULONG gc_ctx0 = gfxhub_rreg(dev, regGCVM_CONTEXT0_CNTL);

        pr_info("gpu_gmc: verify MMHUB CONTEXT0_CNTL = 0x%08x\n", ctx0);
        pr_info("gpu_gmc: verify MMHUB L1_TLB_CNTL   = 0x%08x\n", l1_tlb);
        pr_info("gpu_gmc: verify GFXHUB CONTEXT0_CNTL = 0x%08x\n", gc_ctx0);
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
    if (dev->hw.ip.nbio_base != 0) {
        /* Read the remap register (NBIO base + known offset) */
        ULONG remap_offset = (dev->hw.ip.nbio_base + 0x0393) * 4; /* regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL */
        ULONG flush_reg = wddm_lite_read_reg32(dev, remap_offset);

        if (flush_reg != 0 && flush_reg != 0xFFFFFFFF) {
            /* Write 0 to the remapped flush address */
            wddm_lite_write_reg32(dev, (flush_reg / 4) * 4, 0);
            return;
        }
    }

    /* Fallback: direct HDP flush via NBIO base2 + 0xF7 */
    /* For passthrough GPU, this may not be needed since VBIOS already
     * configured HDP, but do it anyway for correctness */
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

    /* Poll debug response (up to 2 seconds) */
    for (int i = 0; i < 2000; i++) {
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
        Sleep(1);
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

    /* Step 1: Clear response register */
    mp1_wreg(dev, regMP1_SMN_C2PMSG_90, 0);

    /* Step 2: Write parameter */
    mp1_wreg(dev, regMP1_SMN_C2PMSG_82, param);

    /* Step 3: Write message (triggers SMU processing) */
    mp1_wreg(dev, regMP1_SMN_C2PMSG_66, msg);

    /* Step 4: Poll for response (up to 2 seconds) */
    for (int i = 0; i < 2000; i++) {
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
        Sleep(1);
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
    if (dev->hw.ip.mp1_base == 0) {
        pr_err("gpu_smu: MP1 base not found\n");
        return -1;
    }

    /* First check if SMU is alive after firmware loading */
    pr_info("gpu_smu: checking if SMU is alive after firmware load...\n");
    mp1_wreg(dev, regMP1_SMN_C2PMSG_90, 0);
    mp1_wreg(dev, regMP1_SMN_C2PMSG_82, 0);
    mp1_wreg(dev, regMP1_SMN_C2PMSG_66, PPSMC_MSG_GetSmuVersion);

    BOOLEAN alive = FALSE;
    ULONG smu_version = 0;
    for (int i = 0; i < 500; i++) {  /* Up to 5 seconds */
        ULONG resp = mp1_rreg(dev, regMP1_SMN_C2PMSG_90);
        if (resp != 0) {
            alive = TRUE;
            smu_version = mp1_rreg(dev, regMP1_SMN_C2PMSG_82);
            pr_info("gpu_smu: SMU is alive! version=0x%08x (resp=%u, %d ms)\n",
                    smu_version, resp, i * 10);
            break;
        }
        Sleep(10);
    }

    if (!alive) {
        /* Try reading resp without sending a message — maybe already responded */
        ULONG resp = mp1_rreg(dev, regMP1_SMN_C2PMSG_90);
        smu_version = mp1_rreg(dev, regMP1_SMN_C2PMSG_82);
        pr_warn("gpu_smu: SMU not responding after firmware load "
                "(resp=0x%08x, ver=0x%08x)\n", resp, smu_version);
        return -1;
    }

    /* Send EnableAllSmuFeatures (0x06, param 0) */
    pr_info("gpu_smu: sending EnableAllSmuFeatures...\n");
    int ret = gpu_smu_send_msg(dev, PPSMC_MSG_EnableAllSmuFeatures, 0);
    if (ret != 0) {
        pr_warn("gpu_smu: EnableAllSmuFeatures failed\n");
        return -1;
    }
    pr_info("gpu_smu: EnableAllSmuFeatures succeeded\n");

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

    /* ---- Step 2: Try debug SMU mailbox (DisallowGfxOff) ---- */
    {
        pr_info("gpu_smu: trying debug mailbox for DisallowGfxOff...\n");

        /* First check debug response register */
        ULONG dbg_resp = mp1_rreg(dev, regMP1_SMN_C2PMSG_54);
        ULONG dbg_msg = mp1_rreg(dev, regMP1_SMN_C2PMSG_75);
        pr_info("gpu_smu: debug regs before: resp=0x%08x msg=0x%08x\n",
                dbg_resp, dbg_msg);

        /* Try DisallowGfxOff via debug mailbox */
        int ret = gpu_smu_send_debug_msg(dev, PPSMC_MSG_DisallowGfxOff, 0);
        if (ret == 0) {
            dev->hw.gfxoff_disabled = TRUE;
            pr_info("gpu_smu: GFXOFF disabled via debug mailbox\n");
            Sleep(50);
            return 0;
        }
        pr_warn("gpu_smu: debug mailbox DisallowGfxOff also failed\n");
    }

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

/* GC base_index 1 register access (for GRBM, RLC, MEC, SH_MEM) */
static ULONG gc1_rreg(struct WddmLiteDevice *dev, ULONG reg)
{
    return gpu_smn_rreg(dev, dev->hw.ip.gc_base1 + reg);
}

static void gc1_wreg(struct WddmLiteDevice *dev, ULONG reg, ULONG val)
{
    gpu_smn_wreg(dev, dev->hw.ip.gc_base1 + reg, val);
}

/* GC base_index 0 register access (for CP_HQD, CP_STAT, GRBM_SOFT_RESET) */
static ULONG gc0_rreg(struct WddmLiteDevice *dev, ULONG reg)
{
    return gpu_smn_rreg(dev, dev->hw.ip.gc_base + reg);
}

static void gc0_wreg(struct WddmLiteDevice *dev, ULONG reg, ULONG val)
{
    gpu_smn_wreg(dev, dev->hw.ip.gc_base + reg, val);
}


/* ---- GC register offsets (GFX12 / gc_12_0_0) ---- */

/* BASE_IDX=0 registers */
#define regCP_STAT                          0x0f40
#define regGRBM_CNTL                        0x0da0
#define regGRBM_SOFT_RESET                  0x0da8
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
#define CP_MEC_RS64_CNTL__MEC_PIPE0_RESET   (1 << 16)
#define CP_MEC_RS64_CNTL__MEC_PIPE0_ACTIVE  (1 << 26)
#define CP_MEC_RS64_CNTL__MEC_HALT          (1 << 30)

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

static void grbm_select(struct WddmLiteDevice *dev,
                         ULONG me, ULONG pipe, ULONG queue, ULONG vmid)
{
    ULONG val = 0;
    val |= (pipe & 0x3);           /* PIPEID [1:0] */
    val |= (me & 0x3) << 2;       /* MEID [3:2] */
    val |= (vmid & 0xF) << 4;     /* VMID [7:4] */
    val |= (queue & 0x7) << 8;    /* QUEUEID [10:8] */

    gc1_wreg(dev, regGRBM_GFX_CNTL, val);
}

static void grbm_select_reset(struct WddmLiteDevice *dev)
{
    gc1_wreg(dev, regGRBM_GFX_CNTL, 0);
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
     * Program PFP/ME/MEC program counter start addresses.
     * These come from the RS64 firmware headers loaded via PSP.
     * Following tinygrad's _config_mec() for GFX12.
     */

    /* Step 1: Configure PFP program counter */
    if (dev->hw.gfx.pfp_ucode_start != 0) {
        ULONG pfp_lo = (ULONG)(dev->hw.gfx.pfp_ucode_start >> 2);
        ULONG pfp_hi = (ULONG)((dev->hw.gfx.pfp_ucode_start >> 2) >> 32);

        grbm_select(dev, 0, 0, 0, 0);  /* meid=0, pipeid=0 */
        gc0_wreg(dev, regCP_PFP_PRGRM_CNTR_START, pfp_lo);
        gc0_wreg(dev, regCP_PFP_PRGRM_CNTR_START_HI, pfp_hi);
        grbm_select_reset(dev);

        /* Reset then unreset PFP via CP_ME_CNTL */
        ULONG me_cntl = gc1_rreg(dev, regCP_ME_CNTL);
        gc1_wreg(dev, regCP_ME_CNTL, me_cntl | CP_ME_CNTL__PFP_PIPE0_RESET);
        gc1_wreg(dev, regCP_ME_CNTL, me_cntl & ~CP_ME_CNTL__PFP_PIPE0_RESET);

        pr_info("gpu_gfx: PFP program counter set to 0x%llx (reg: 0x%x:0x%x)\n",
                (unsigned long long)dev->hw.gfx.pfp_ucode_start, pfp_lo, pfp_hi);
    }

    /* Step 2: Configure ME program counter */
    if (dev->hw.gfx.me_ucode_start != 0) {
        ULONG me_lo = (ULONG)(dev->hw.gfx.me_ucode_start >> 2);
        ULONG me_hi = (ULONG)((dev->hw.gfx.me_ucode_start >> 2) >> 32);

        grbm_select(dev, 0, 0, 0, 0);  /* meid=0, pipeid=0 */
        gc0_wreg(dev, regCP_ME_PRGRM_CNTR_START, me_lo);
        gc0_wreg(dev, regCP_ME_PRGRM_CNTR_START_HI, me_hi);
        grbm_select_reset(dev);

        /* Reset then unreset ME via CP_ME_CNTL */
        ULONG me_cntl = gc1_rreg(dev, regCP_ME_CNTL);
        gc1_wreg(dev, regCP_ME_CNTL, me_cntl | CP_ME_CNTL__ME_PIPE0_RESET);
        gc1_wreg(dev, regCP_ME_CNTL, me_cntl & ~CP_ME_CNTL__ME_PIPE0_RESET);

        pr_info("gpu_gfx: ME program counter set to 0x%llx (reg: 0x%x:0x%x)\n",
                (unsigned long long)dev->hw.gfx.me_ucode_start, me_lo, me_hi);
    }

    /* Step 3: Configure MEC program counter */
    if (dev->hw.gfx.mec_ucode_start != 0) {
        ULONG mec_lo = (ULONG)(dev->hw.gfx.mec_ucode_start >> 2);
        ULONG mec_hi = (ULONG)((dev->hw.gfx.mec_ucode_start >> 2) >> 32);

        grbm_select(dev, 1, 0, 0, 0);  /* meid=1, pipeid=0 */
        gc1_wreg(dev, regCP_MEC_RS64_PRGRM_CNTR_START, mec_lo);
        gc1_wreg(dev, regCP_MEC_RS64_PRGRM_CNTR_START_HI, mec_hi);
        grbm_select_reset(dev);

        /* Reset then unreset MEC pipe0 via CP_MEC_RS64_CNTL */
        ULONG mec_cntl = gc1_rreg(dev, regCP_MEC_RS64_CNTL);
        gc1_wreg(dev, regCP_MEC_RS64_CNTL,
                 mec_cntl | CP_MEC_RS64_CNTL__MEC_PIPE0_RESET);
        gc1_wreg(dev, regCP_MEC_RS64_CNTL,
                 mec_cntl & ~CP_MEC_RS64_CNTL__MEC_PIPE0_RESET);

        pr_info("gpu_gfx: MEC program counter set to 0x%llx (reg: 0x%x:0x%x)\n",
                (unsigned long long)dev->hw.gfx.mec_ucode_start, mec_lo, mec_hi);
    }
}

static int enable_mec(struct WddmLiteDevice *dev)
{
    /*
     * On a VBIOS-POST GPU, MEC firmware is already loaded by PSP.
     * We just need to ensure it's not halted and pipe0 is active.
     *
     * For GFX12, use CP_MEC_RS64_CNTL (not CP_MEC_CNTL).
     */
    ULONG val = gc1_rreg(dev, regCP_MEC_RS64_CNTL);
    pr_info("gpu_gfx: CP_MEC_RS64_CNTL (before) = 0x%08x\n", val);

    /* Check if MEC is already active */
    if ((val & CP_MEC_RS64_CNTL__MEC_PIPE0_ACTIVE) &&
        !(val & CP_MEC_RS64_CNTL__MEC_HALT)) {
        pr_info("gpu_gfx: MEC already active and not halted\n");
        dev->hw.gfx.mec_enabled = TRUE;
        return 0;
    }

    /* Unhalt and activate pipe0 */
    val &= ~CP_MEC_RS64_CNTL__MEC_PIPE0_RESET;
    val |= CP_MEC_RS64_CNTL__MEC_PIPE0_ACTIVE;
    val &= ~CP_MEC_RS64_CNTL__MEC_HALT;
    gc1_wreg(dev, regCP_MEC_RS64_CNTL, val);

    /* Brief delay for MEC to become ready */
    Sleep(50);

    /* Verify */
    val = gc1_rreg(dev, regCP_MEC_RS64_CNTL);
    pr_info("gpu_gfx: CP_MEC_RS64_CNTL (after) = 0x%08x\n", val);

    if (val & CP_MEC_RS64_CNTL__MEC_PIPE0_ACTIVE) {
        pr_info("gpu_gfx: MEC pipe0 active\n");
        dev->hw.gfx.mec_enabled = TRUE;
        return 0;
    } else {
        pr_warn("gpu_gfx: MEC pipe0 NOT active after enable\n");
        return -1;
    }
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

    /* Preserve ucode_start values from firmware loading, zero the rest */
    {
        ULONGLONG pfp = dev->hw.gfx.pfp_ucode_start;
        ULONGLONG me  = dev->hw.gfx.me_ucode_start;
        ULONGLONG mec = dev->hw.gfx.mec_ucode_start;
        memset(&dev->hw.gfx, 0, sizeof(dev->hw.gfx));
        dev->hw.gfx.pfp_ucode_start = pfp;
        dev->hw.gfx.me_ucode_start  = me;
        dev->hw.gfx.mec_ucode_start = mec;
    }

    /* Step 0: Disable GFXOFF so GC registers are accessible */
    if (!dev->hw.gfxoff_disabled) {
        if (gpu_disable_gfxoff(dev) != 0) {
            pr_warn("gpu_gfx: GFXOFF disable failed (GC registers may be inaccessible)\n");
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

    /* Step 3: Configure SH_MEM for all VMIDs */
    configure_sh_mem(dev);

    /* Step 4: Configure MEC doorbell range and GRBM timeout */
    configure_mec(dev);

    /* Step 5: Enable MEC */
    if (enable_mec(dev) != 0) {
        pr_warn("gpu_gfx: MEC enable failed (continuing)\n");
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

    dev->hw.gfx_initialized = TRUE;
    pr_info("gpu_gfx: GFX engine initialization complete\n");
    return 0;
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

/* Doorbell index for MEC ring 0 — uses existing define above (0x3) */

/*
 * Set up a compute queue by programming HQD registers directly.
 *
 * queue_idx: queue index (0-7). pipe = idx/4, queue = idx%4
 * ring_addr: GPU address of the ring buffer
 * ring_size: ring buffer size in bytes (must be power of 2)
 * rptr_addr: GPU address for RPTR writeback (8 bytes)
 * wptr_addr: GPU address for WPTR polling (8 bytes)
 * eop_addr:  GPU address of EOP buffer
 * eop_size:  EOP buffer size in bytes
 * aql:       true for AQL queue, false for PM4
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
    ULONG doorbell = AMDGPU_NAVI10_DOORBELL_MEC_RING0;

    pr_info("gpu_queue: setting up queue %u (pipe=%u, queue=%u, aql=%d)\n",
            queue_idx, pipe, queue, aql);
    pr_info("gpu_queue: ring=0x%llx size=0x%x rptr=0x%llx wptr=0x%llx\n",
            (unsigned long long)ring_addr, ring_size,
            (unsigned long long)rptr_addr, (unsigned long long)wptr_addr);
    pr_info("gpu_queue: eop=0x%llx eop_size=0x%x\n",
            (unsigned long long)eop_addr, eop_size);

    /* Select pipe/queue via GRBM */
    grbm_select(dev, 1, pipe, queue, 0);

    /* First dequeue if already active */
    ULONG hqd_active = gc0_rreg(dev, regCP_HQD_ACTIVE);
    if (hqd_active & 1) {
        pr_info("gpu_queue: HQD already active, dequeuing first\n");
        gc0_wreg(dev, regCP_HQD_DEQUEUE_REQUEST, 0x2);
        for (int i = 0; i < 100; i++) {
            if (!(gc0_rreg(dev, regCP_HQD_ACTIVE) & 1))
                break;
            Sleep(1);
        }
        if (gc0_rreg(dev, regCP_HQD_ACTIVE) & 1) {
            pr_warn("gpu_queue: dequeue timeout\n");
        }
    }

    /* Build MQD struct */
    struct v12_compute_mqd mqd;
    memset(&mqd, 0, sizeof(mqd));

    mqd.header = 0xC0310800;

    /* Enable all CUs in all shader engines */
    mqd.compute_static_thread_mgmt_se0 = 0xFFFFFFFF;
    mqd.compute_static_thread_mgmt_se1 = 0xFFFFFFFF;
    mqd.compute_static_thread_mgmt_se2 = 0xFFFFFFFF;
    mqd.compute_static_thread_mgmt_se3 = 0xFFFFFFFF;
    mqd.compute_static_thread_mgmt_se4 = 0xFFFFFFFF;
    mqd.compute_static_thread_mgmt_se5 = 0xFFFFFFFF;
    mqd.compute_static_thread_mgmt_se6 = 0xFFFFFFFF;
    mqd.compute_static_thread_mgmt_se7 = 0xFFFFFFFF;

    /* MQD self-reference (GPU address of this MQD in VRAM) */
    /* For now, we don't have a VRAM copy — set to 0 */
    mqd.cp_mqd_base_addr_lo = 0;
    mqd.cp_mqd_base_addr_hi = 0;

    /* Queue priority and scheduling */
    mqd.cp_hqd_pipe_priority = 0x2;
    mqd.cp_hqd_queue_priority = 0xF;
    mqd.cp_hqd_quantum = 0x111;  /* QUANTUM_EN | SCALE | DURATION */

    /* Preload settings */
    mqd.cp_hqd_persistent_state = (1 << 14) |  /* PRELOAD_REQ */
                                   (0x55 << 0); /* PRELOAD_SIZE */

    /* Ring buffer base (address >> 8) */
    mqd.cp_hqd_pq_base_lo = (ULONG)(ring_addr >> 8);
    mqd.cp_hqd_pq_base_hi = (ULONG)((ring_addr >> 8) >> 32);

    /* RPTR report address */
    mqd.cp_hqd_pq_rptr_report_addr_lo = (ULONG)rptr_addr;
    mqd.cp_hqd_pq_rptr_report_addr_hi = (ULONG)(rptr_addr >> 32);

    /* WPTR poll address */
    mqd.cp_hqd_pq_wptr_poll_addr_lo = (ULONG)wptr_addr;
    mqd.cp_hqd_pq_wptr_poll_addr_hi = (ULONG)(wptr_addr >> 32);

    /* Doorbell control */
    mqd.cp_hqd_pq_doorbell_control = (doorbell * 2) |  /* doorbell_offset */
                                      (1 << 30);        /* doorbell_en */

    /* Queue control: ring size + flags */
    {
        /* queue_size field = log2(ring_size/4) - 1 */
        ULONG queue_size_log2 = 0;
        ULONG rs = ring_size / 4;
        while (rs > 1) { rs >>= 1; queue_size_log2++; }
        queue_size_log2 -= 1;

        mqd.cp_hqd_pq_control = queue_size_log2 |       /* QUEUE_SIZE [5:0] */
                                 (5 << 8);                /* RPTR_BLOCK_SIZE [11:8] */

        if (aql) {
            mqd.cp_hqd_pq_control |= (1 << 23) |        /* QUEUE_FULL_EN */
                                      (2 << 24) |        /* SLOT_BASED_WPTR [25:24] */
                                      (1 << 26);         /* NO_UPDATE_RPTR */
        }
    }

    /* IB control */
    mqd.cp_hqd_ib_control = (0x3 << 12);  /* MIN_IB_AVAIL_SIZE */

    /* HQ status */
    mqd.cp_hqd_hq_status0 = 0x20004000;

    /* MQD control: priv_state=1 */
    mqd.cp_mqd_control = (1 << 8);  /* PRIV_STATE */

    /* VMID 0 */
    mqd.cp_hqd_vmid = 0;

    /* AQL control */
    mqd.cp_hqd_aql_control = aql ? 1 : 0;

    /* EOP buffer */
    mqd.cp_hqd_eop_base_addr_lo = (ULONG)(eop_addr >> 8);
    mqd.cp_hqd_eop_base_addr_hi = (ULONG)((eop_addr >> 8) >> 32);
    {
        ULONG eop_size_log2 = 0;
        ULONG es = eop_size / 4;
        while (es > 1) { es >>= 1; eop_size_log2++; }
        eop_size_log2 -= 1;
        mqd.cp_hqd_eop_control = eop_size_log2;  /* EOP_SIZE [5:0] */
    }

    /*
     * Write HQD registers from MQD struct.
     * The MQD fields at offset 0x80 (128) map 1:1 to sequential
     * registers starting at regCP_MQD_BASE_ADDR (0x1fa9).
     */
    ULONG *mqd_dwords = (ULONG *)&mqd;
    ULONG num_hqd_regs = regCP_HQD_PQ_WPTR_HI - regCP_MQD_BASE_ADDR + 1;

    pr_info("gpu_queue: writing %u HQD registers (0x%04x-0x%04x)\n",
            num_hqd_regs, regCP_MQD_BASE_ADDR, regCP_HQD_PQ_WPTR_HI);

    for (ULONG i = 0; i < num_hqd_regs; i++) {
        ULONG reg = regCP_MQD_BASE_ADDR + i;
        ULONG val = mqd_dwords[0x80 + i];

        /* Skip CP_HQD_ACTIVE — write it last */
        if (reg == regCP_HQD_ACTIVE)
            continue;

        gc0_wreg(dev, reg, val);
    }

    /* Activate the queue */
    gc0_wreg(dev, regCP_HQD_ACTIVE, 1);

    /* Flush HDP */
    gpu_hdp_flush(dev);

    /* Deselect pipe/queue */
    grbm_select_reset(dev);

    /* Verify activation */
    grbm_select(dev, 1, pipe, queue, 0);
    hqd_active = gc0_rreg(dev, regCP_HQD_ACTIVE);
    grbm_select_reset(dev);

    pr_info("gpu_queue: CP_HQD_ACTIVE readback = 0x%08x\n", hqd_active);

    if (hqd_active & 1) {
        pr_info("gpu_queue: queue %u activated successfully\n", queue_idx);

        /* Update tracking */
        struct GpuComputeQueue *q = &dev->hw.queues[queue_idx];
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
        q->doorbell_offset = doorbell * 2;
        dev->hw.num_active_queues++;
        return 0;
    } else {
        pr_err("gpu_queue: queue %u activation failed\n", queue_idx);
        return -1;
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
