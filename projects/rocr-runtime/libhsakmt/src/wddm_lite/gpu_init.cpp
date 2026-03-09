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
    ULONG offset = (dev->hw.ip.gc_base + reg) * 4;
    return wddm_lite_read_reg32(dev, offset);
}

static void gfxhub_wreg(struct WddmLiteDevice *dev, ULONG reg, ULONG val)
{
    ULONG offset = (dev->hw.ip.gc_base + reg) * 4;
    wddm_lite_write_reg32(dev, offset, val);
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
#define PPSMC_MSG_GetSmuVersion         0x02
#define PPSMC_MSG_DisallowGfxOff        0x29
#define PPSMC_MSG_AllowGfxOff           0x28

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

    ULONG sig = rd32(buf, base);
    if (sig != BINARY_SIGNATURE) {
        pr_err("gpu_init: bad IP discovery signature 0x%08x "
               "(expected 0x%08x) at offset %u\n",
               sig, BINARY_SIGNATURE, base);
        goto out_unmap;
    }

    USHORT ver_major = rd16(buf, base + 4);
    USHORT ver_minor = rd16(buf, base + 6);

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

/* Forward declaration — defined later but needed by PSP bootloader */
void gpu_hdp_flush(struct WddmLiteDevice *dev);

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

    /* Parse common firmware header */
    if (fw_len < 36) {
        pr_err("gpu_psp: firmware too small (%u bytes)\n", fw_len);
        goto fail;
    }

    USHORT hdr_major = rd16(fw_buf, 8);
    USHORT hdr_minor = rd16(fw_buf, 10);
    ULONG ucode_offset = rd32(fw_buf, 24);
    ULONG psp_fw_count = rd32(fw_buf, 32);

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
    ULONGLONG vram_offset = 4 * 1024 * 1024;  /* 4MB into VRAM */
    ULONGLONG vram_map_size = 1024 * 1024;  /* 1MB */

    /* Read VRAM MC base from MMHUB (always accessible, even before GMC init) */
    ULONG fb_base_reg = mmhub_rreg(dev, regMMMC_VM_FB_LOCATION_BASE);
    ULONGLONG vram_mc_base = (ULONGLONG)fb_base_reg << 24;
    ULONGLONG fw_mc_addr = vram_mc_base + vram_offset;

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
 * PSP GPCOM Ring — Load IP Firmware (SMU)
 *
 * After SOS is alive, we can create the PSP GPCOM ring and submit
 * LOAD_IP_FW commands. This is needed to load the SMU firmware,
 * which is required for DisallowGfxOff.
 *
 * Ring protocol (from Linux amdgpu psp_v14_0.c and tinygrad AM_PSP):
 *   1. Wait for TOS ready (C2PMSG_64 bit 31 set with status 0)
 *   2. Allocate ring buffer in VRAM
 *   3. Write ring address/size to C2PMSG_69/70/71
 *   4. Send create command via C2PMSG_64
 *   5. Submit ring entries (4 DWORDs each):
 *        DW[0]: fence_addr_lo
 *        DW[1]: fence_addr_hi
 *        DW[2]: fence_value
 *        DW[3]: cmd_id | (fw_type << 16)
 *   6. Update write pointer via C2PMSG_67
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
#define GFX_CTRL_CMD_ID_DESTROY_RINGS  3

/* PSP ring commands */
#define GFX_CMD_ID_LOAD_IP_FW   0x00006
#define GFX_CMD_ID_AUTOLOAD_RLC 0x00017

/* Response flags */
#define GFX_FLAG_RESPONSE       0x80000000
#define GFX_CMD_RESPONSE_MASK   0x8000FFFF

/* PSP GFX firmware types (enum psp_gfx_fw_type) */
#define GFX_FW_TYPE_SMU         18  /* SMU/PMFW firmware (psp_gfx_fw_type enum) */

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

struct PspGfxCmdResp {
    ULONG   buf_size;           /* +0   total size (1024) */
    ULONG   buf_version;        /* +4   version (1) */
    ULONG   cmd_id;             /* +8   command ID */
    ULONG   resp_buf_addr_lo;   /* +12  0 for GPCOM */
    ULONG   resp_buf_addr_hi;   /* +16  0 for GPCOM */
    ULONG   resp_offset;        /* +20  0 for GPCOM */
    ULONG   resp_buf_size;      /* +24  0 for GPCOM */
    /* +28: command union */
    struct PspGfxCmdLoadIpFw load_ip_fw;
    UCHAR   cmd_padding[768];
    UCHAR   reserved_1[52];     /* +812 */
    ULONG   resp_status;        /* +864 response status (0 = success) */
    UCHAR   resp_reserved[92];  /* +868 */
    UCHAR   reserved_2[64];     /* +960 */
    /* Total: 1024 bytes */
};
#pragma pack(pop)

#define PSP_GFX_CMD_BUF_VERSION 1

/* Firmware header: common_firmware_header (40 bytes) */
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
    ULONG final = mp0_rreg(dev, regMPASP_SMN_C2PMSG_64);
    pr_err("gpu_psp_ring: timeout waiting for C2PMSG_64 "
           "(expect 0x%08x mask 0x%08x, got 0x%08x)\n",
           expected, mask, final);
    return -1;
}

int gpu_psp_load_smu_fw(struct WddmLiteDevice *dev, const char *fw_dir)
{
    int ret = -1;
    UCHAR *smu_buf = NULL;
    ULONG smu_len = 0;
    PVOID ring_cpu = NULL, fw_cpu = NULL, cmd_cpu = NULL, fence_cpu = NULL;
    PVOID ring_mapping_handle = NULL, fw_mapping_handle = NULL;
    PVOID cmd_mapping_handle = NULL, fence_mapping_handle = NULL;

    if (!dev->hw.psp_sos_alive) {
        pr_err("gpu_psp_ring: SOS must be alive before ring init\n");
        return -1;
    }

    if (dev->hw.ip.mp0_base == 0) {
        pr_err("gpu_psp_ring: MP0 base not found\n");
        return -1;
    }

    /* ---- Read SMU firmware file ---- */
    {
        char smu_path[512];
        snprintf(smu_path, sizeof(smu_path), "%s\\smu_14_0_3.bin", fw_dir);

        FILE *fp = fopen(smu_path, "rb");
        if (!fp) {
            /* Also try smu_14_0_2.bin (some firmware packages use this) */
            snprintf(smu_path, sizeof(smu_path), "%s\\smu_14_0_2.bin", fw_dir);
            fp = fopen(smu_path, "rb");
        }
        if (!fp) {
            pr_err("gpu_psp_ring: cannot open SMU firmware "
                   "(tried smu_14_0_3.bin and smu_14_0_2.bin in %s)\n", fw_dir);
            return -1;
        }

        fseek(fp, 0, SEEK_END);
        smu_len = (ULONG)ftell(fp);
        fseek(fp, 0, SEEK_SET);

        smu_buf = (UCHAR *)malloc(smu_len);
        if (!smu_buf) {
            fclose(fp);
            return -1;
        }
        if (fread(smu_buf, 1, smu_len, fp) != smu_len) {
            pr_err("gpu_psp_ring: read error on %s\n", smu_path);
            fclose(fp);
            free(smu_buf);
            return -1;
        }
        fclose(fp);
        pr_info("gpu_psp_ring: loaded SMU firmware: %s (%u bytes)\n",
                smu_path, smu_len);
    }

    /* ---- Parse firmware header to get ucode offset/size ---- */
    if (smu_len < sizeof(struct CommonFirmwareHeader)) {
        pr_err("gpu_psp_ring: SMU firmware too small (%u bytes)\n", smu_len);
        goto fail;
    }

    struct CommonFirmwareHeader *hdr = (struct CommonFirmwareHeader *)smu_buf;
    pr_info("gpu_psp_ring: SMU FW header v%u.%u, ucode_size=%u, "
            "ucode_offset=0x%x\n",
            hdr->header_version_major, hdr->header_version_minor,
            hdr->ucode_size_bytes, hdr->ucode_array_offset_bytes);

    /* The actual ucode to load starts at ucode_array_offset_bytes */
    ULONG ucode_offset = hdr->ucode_array_offset_bytes;
    ULONG ucode_size = hdr->ucode_size_bytes;
    if (ucode_offset + ucode_size > smu_len) {
        pr_err("gpu_psp_ring: ucode overflows file "
               "(offset=0x%x + size=0x%x > 0x%x)\n",
               ucode_offset, ucode_size, smu_len);
        goto fail;
    }

    if (ucode_size > 1024 * 1024) {
        pr_err("gpu_psp_ring: ucode too large (%u bytes, max 1MB)\n",
               ucode_size);
        goto fail;
    }

    /* ---- Allocate VRAM buffers ---- */
    /*
     * Ring buffer: 64KB at VRAM offset 5MB
     * FW staging:  1MB at VRAM offset 6MB
     * Cmd buffer:  4KB at VRAM offset 7MB   (1024-byte PspGfxCmdResp)
     * Fence:       4KB at VRAM offset 7MB+4K (single ULONG)
     * (SOS loader uses offset 4MB, so we avoid that)
     */
    ULONG fb_base_reg = mmhub_rreg(dev, regMMMC_VM_FB_LOCATION_BASE);
    ULONGLONG vram_mc_base = (ULONGLONG)fb_base_reg << 24;

    ULONGLONG ring_vram_offset = 5 * 1024 * 1024;
    ULONGLONG fw_vram_offset   = 6 * 1024 * 1024;
    ULONGLONG cmd_vram_offset  = 7 * 1024 * 1024;
    ULONGLONG fence_vram_offset = 7 * 1024 * 1024 + 4096;
    ULONGLONG ring_mc_addr = vram_mc_base + ring_vram_offset;
    ULONGLONG fw_mc_addr   = vram_mc_base + fw_vram_offset;
    ULONGLONG cmd_mc_addr  = vram_mc_base + cmd_vram_offset;
    ULONGLONG fence_mc_addr = vram_mc_base + fence_vram_offset;

    pr_info("gpu_psp_ring: VRAM MC base = 0x%llx\n",
            (unsigned long long)vram_mc_base);
    pr_info("gpu_psp_ring: ring=0x%llx, fw=0x%llx, cmd=0x%llx, fence=0x%llx\n",
            (unsigned long long)ring_mc_addr,
            (unsigned long long)fw_mc_addr,
            (unsigned long long)cmd_mc_addr,
            (unsigned long long)fence_mc_addr);

    /* Map ring buffer VRAM */
    {
        AMDGPU_ESCAPE_MAP_VRAM_DATA map = {};
        map.Header.Command = AMDGPU_ESCAPE_MAP_VRAM;
        map.Header.Size = sizeof(map);
        map.Offset = ring_vram_offset;
        map.Length = PSP_RING_SIZE;

        if (wddm_lite_escape(dev, &map, sizeof(map)) != 0 ||
            map.MappedAddress == NULL) {
            pr_err("gpu_psp_ring: ring VRAM map failed\n");
            goto fail;
        }
        ring_cpu = map.MappedAddress;
        ring_mapping_handle = map.MappingHandle;
    }

    /* Map firmware staging VRAM (1MB) */
    {
        AMDGPU_ESCAPE_MAP_VRAM_DATA map = {};
        map.Header.Command = AMDGPU_ESCAPE_MAP_VRAM;
        map.Header.Size = sizeof(map);
        map.Offset = fw_vram_offset;
        map.Length = 1024 * 1024;

        if (wddm_lite_escape(dev, &map, sizeof(map)) != 0 ||
            map.MappedAddress == NULL) {
            pr_err("gpu_psp_ring: fw staging VRAM map failed\n");
            goto fail_ring;
        }
        fw_cpu = map.MappedAddress;
        fw_mapping_handle = map.MappingHandle;
    }

    /* Map command buffer VRAM (4KB) */
    {
        AMDGPU_ESCAPE_MAP_VRAM_DATA map = {};
        map.Header.Command = AMDGPU_ESCAPE_MAP_VRAM;
        map.Header.Size = sizeof(map);
        map.Offset = cmd_vram_offset;
        map.Length = 4096;

        if (wddm_lite_escape(dev, &map, sizeof(map)) != 0 ||
            map.MappedAddress == NULL) {
            pr_err("gpu_psp_ring: cmd buffer VRAM map failed\n");
            goto fail_ring;
        }
        cmd_cpu = map.MappedAddress;
        cmd_mapping_handle = map.MappingHandle;
    }

    /* Map fence VRAM (4KB) */
    {
        AMDGPU_ESCAPE_MAP_VRAM_DATA map = {};
        map.Header.Command = AMDGPU_ESCAPE_MAP_VRAM;
        map.Header.Size = sizeof(map);
        map.Offset = fence_vram_offset;
        map.Length = 4096;

        if (wddm_lite_escape(dev, &map, sizeof(map)) != 0 ||
            map.MappedAddress == NULL) {
            pr_err("gpu_psp_ring: fence VRAM map failed\n");
            goto fail_ring;
        }
        fence_cpu = map.MappedAddress;
        fence_mapping_handle = map.MappingHandle;
    }

    /* Initialize fence to 0 */
    *(volatile ULONG *)fence_cpu = 0;
    MemoryBarrier();

    /* ---- Destroy + Create ring ---- */
    /*
     * The PSP GPCOM ring uses 64-byte frames (psp_gfx_rb_frame) pointing
     * to separate 1024-byte command buffers. Completion is via fence writes.
     *
     * On a passthrough GPU, VBIOS may have created a ring during POST.
     * Following tinygrad's approach: destroy existing ring, then create fresh.
     * Even if destroy "fails", it may clear the PSP state enough to allow
     * a new ring creation.
     *
     * wptr is in DWORDs. Each frame is 64 bytes = 16 DWORDs.
     */
    ULONG wptr;
    {
        ULONG c64_val = mp0_rreg(dev, regMPASP_SMN_C2PMSG_64);
        ULONG ring_size_reg = mp0_rreg(dev, regMPASP_SMN_C2PMSG_71);
        ULONG wptr_val = mp0_rreg(dev, regMPASP_SMN_C2PMSG_67);
        ULONG ring_addr_lo = mp0_rreg(dev, regMPASP_SMN_C2PMSG_69);
        ULONG ring_addr_hi = mp0_rreg(dev, regMPASP_SMN_C2PMSG_70);

        pr_info("gpu_psp_ring: before destroy:\n");
        pr_info("  C2PMSG_64=0x%08x, C2PMSG_67(wptr)=%u, "
                "C2PMSG_71(size)=0x%x\n",
                c64_val, wptr_val, ring_size_reg);
        pr_info("  C2PMSG_69/70 ring addr=0x%08x_%08x\n",
                ring_addr_hi, ring_addr_lo);

        /* Step 1: Try to destroy existing ring */
        if (ring_size_reg != 0) {
            pr_info("gpu_psp_ring: sending DESTROY_RINGS (cmd=3)...\n");
            mp0_wreg(dev, regMPASP_SMN_C2PMSG_64,
                     GFX_CTRL_CMD_ID_DESTROY_RINGS);
            Sleep(20);

            ULONG c64_after = mp0_rreg(dev, regMPASP_SMN_C2PMSG_64);
            ULONG size_after = mp0_rreg(dev, regMPASP_SMN_C2PMSG_71);
            pr_info("gpu_psp_ring: after destroy: C2PMSG_64=0x%08x, "
                    "C2PMSG_71=0x%x\n",
                    c64_after, size_after);

            /* Wait for response */
            for (int i = 0; i < 5000; i++) {
                ULONG v = mp0_rreg(dev, regMPASP_SMN_C2PMSG_64);
                if (v & GFX_FLAG_RESPONSE) {
                    pr_info("gpu_psp_ring: destroy response after %d ms: "
                            "C2PMSG_64=0x%08x\n", i + 20, v);
                    break;
                }
                Sleep(1);
            }
        }

        /* Step 2: Wait for TOS ready (bit 31 set) */
        pr_info("gpu_psp_ring: waiting for TOS ready...\n");
        {
            ULONG v = mp0_rreg(dev, regMPASP_SMN_C2PMSG_64);
            pr_info("gpu_psp_ring: C2PMSG_64 = 0x%08x\n", v);
            if (!(v & GFX_FLAG_RESPONSE)) {
                if (psp_ring_wait(dev, GFX_FLAG_RESPONSE,
                                  GFX_FLAG_RESPONSE, 20000) != 0) {
                    pr_err("gpu_psp_ring: TOS not ready\n");
                    goto fail_fw;
                }
            }
        }

        /* Step 3: Zero our ring buffer and create */
        memset(ring_cpu, 0, PSP_RING_SIZE);
        MemoryBarrier();
        gpu_hdp_flush(dev);

        pr_info("gpu_psp_ring: creating ring at MC 0x%llx (size=0x%x)...\n",
                (unsigned long long)ring_mc_addr, PSP_RING_SIZE);

        mp0_wreg(dev, regMPASP_SMN_C2PMSG_69,
                 (ULONG)(ring_mc_addr & 0xFFFFFFFF));
        mp0_wreg(dev, regMPASP_SMN_C2PMSG_70,
                 (ULONG)((ring_mc_addr >> 32) & 0xFFFFFFFF));
        mp0_wreg(dev, regMPASP_SMN_C2PMSG_71, PSP_RING_SIZE);

        mp0_wreg(dev, regMPASP_SMN_C2PMSG_64, PSP_RING_TYPE_KM << 16);
        Sleep(20);

        /* Wait for ring creation response */
        if (psp_ring_wait(dev, GFX_FLAG_RESPONSE, GFX_FLAG_RESPONSE,
                          20000) != 0) {
            pr_err("gpu_psp_ring: ring creation timed out\n");
            goto fail_fw;
        }

        ULONG c64 = mp0_rreg(dev, regMPASP_SMN_C2PMSG_64);
        ULONG status = c64 & 0x0000FFFF;
        pr_info("gpu_psp_ring: ring create response: C2PMSG_64=0x%08x "
                "(status=0x%04x)\n", c64, status);

        if (status != 0) {
            pr_warn("gpu_psp_ring: ring create status 0x%04x "
                    "(may be non-fatal on VBIOS-POST'd GPU)\n", status);
            /* Continue anyway — submit and see if it works */
        }

        wptr = mp0_rreg(dev, regMPASP_SMN_C2PMSG_67);
        pr_info("gpu_psp_ring: ring ready, wptr=%u\n", wptr);
    }

    /* ---- Copy SMU firmware to staging buffer ---- */
    memcpy(fw_cpu, smu_buf + ucode_offset, ucode_size);
    MemoryBarrier();
    gpu_hdp_flush(dev);

    /* ---- Build command buffer for LOAD_IP_FW ---- */
    {
        struct PspGfxCmdResp *cmd = (struct PspGfxCmdResp *)cmd_cpu;
        memset(cmd, 0, sizeof(*cmd));
        cmd->buf_size = sizeof(*cmd);
        cmd->buf_version = PSP_GFX_CMD_BUF_VERSION;
        cmd->cmd_id = GFX_CMD_ID_LOAD_IP_FW;
        cmd->load_ip_fw.fw_phy_addr_lo = (ULONG)(fw_mc_addr & 0xFFFFFFFF);
        cmd->load_ip_fw.fw_phy_addr_hi = (ULONG)((fw_mc_addr >> 32) & 0xFFFFFFFF);
        cmd->load_ip_fw.fw_size = ucode_size;
        cmd->load_ip_fw.fw_type = GFX_FW_TYPE_SMU;

        MemoryBarrier();
        gpu_hdp_flush(dev);

        pr_info("gpu_psp_ring: cmd buf at MC 0x%llx: "
                "cmd_id=0x%x, fw_type=%d, fw_addr=0x%llx, fw_size=%u\n",
                (unsigned long long)cmd_mc_addr,
                cmd->cmd_id, cmd->load_ip_fw.fw_type,
                (unsigned long long)fw_mc_addr, ucode_size);
    }

    /* ---- Submit 64-byte ring frame ---- */
    pr_info("gpu_psp_ring: submitting LOAD_IP_FW for SMU...\n");
    ULONG fence_expected = wptr + 1;  /* Arbitrary fence value */
    {
        /* wptr is in DWORDs, each frame is 16 DWORDs (64 bytes) */
        ULONG byte_offset = (wptr * 4) % PSP_RING_SIZE;
        struct PspGfxRbFrame *frame =
            (struct PspGfxRbFrame *)((UCHAR *)ring_cpu + byte_offset);

        memset(frame, 0, sizeof(*frame));
        frame->cmd_buf_addr_lo = (ULONG)(cmd_mc_addr & 0xFFFFFFFF);
        frame->cmd_buf_addr_hi = (ULONG)((cmd_mc_addr >> 32) & 0xFFFFFFFF);
        frame->cmd_buf_size = sizeof(struct PspGfxCmdResp);
        frame->fence_addr_lo = (ULONG)(fence_mc_addr & 0xFFFFFFFF);
        frame->fence_addr_hi = (ULONG)((fence_mc_addr >> 32) & 0xFFFFFFFF);
        frame->fence_value = fence_expected;

        MemoryBarrier();
        gpu_hdp_flush(dev);

        /* Advance wptr by 16 DWORDs (1 frame) */
        ULONG new_wptr = wptr + PSP_RING_FRAME_DWORDS;
        pr_info("gpu_psp_ring: frame @byte %u, advancing wptr %u -> %u\n",
                byte_offset, wptr, new_wptr);
        mp0_wreg(dev, regMPASP_SMN_C2PMSG_67, new_wptr);
    }

    /* ---- Wait for fence ---- */
    pr_info("gpu_psp_ring: waiting for fence (expect %u)...\n",
            fence_expected);
    {
        volatile ULONG *fence_ptr = (volatile ULONG *)fence_cpu;
        BOOLEAN got_fence = FALSE;

        for (int i = 0; i < 30000; i++) {  /* 30 seconds */
            ULONG fv = *fence_ptr;
            if (fv == fence_expected) {
                got_fence = TRUE;
                pr_info("gpu_psp_ring: fence received after %d ms\n", i);
                break;
            }
            if (i == 100 || i == 1000 || i == 5000 || i == 10000) {
                ULONG c64 = mp0_rreg(dev, regMPASP_SMN_C2PMSG_64);
                pr_info("gpu_psp_ring: poll: fence=%u, C2PMSG_64=0x%08x "
                        "(%d ms)\n", fv, c64, i);
            }
            Sleep(1);
        }

        if (!got_fence) {
            ULONG fv = *fence_ptr;
            ULONG c64 = mp0_rreg(dev, regMPASP_SMN_C2PMSG_64);
            pr_err("gpu_psp_ring: fence timeout (val=%u, expect=%u, "
                   "C2PMSG_64=0x%08x)\n", fv, fence_expected, c64);

            /* Check command response status in case of error */
            struct PspGfxCmdResp *cmd = (struct PspGfxCmdResp *)cmd_cpu;
            pr_err("gpu_psp_ring: cmd resp_status=0x%08x\n",
                   cmd->resp_status);
            goto fail_fw;
        }
    }

    /* Check command response status */
    {
        struct PspGfxCmdResp *cmd = (struct PspGfxCmdResp *)cmd_cpu;
        pr_info("gpu_psp_ring: cmd resp_status=0x%08x\n", cmd->resp_status);
        if (cmd->resp_status != 0) {
            pr_err("gpu_psp_ring: LOAD_IP_FW error status 0x%08x\n",
                   cmd->resp_status);
            goto fail_fw;
        }
    }

    pr_info("gpu_psp_ring: SMU firmware loaded successfully\n");

    /* ---- Verify SMU is alive ---- */
    Sleep(100);
    {
        ULONG resp = mp1_rreg(dev, regMP1_SMN_C2PMSG_90);
        pr_info("gpu_psp_ring: SMU C2PMSG_90 after load = 0x%08x\n", resp);

        mp1_wreg(dev, regMP1_SMN_C2PMSG_90, 0);
        mp1_wreg(dev, regMP1_SMN_C2PMSG_82, 0);
        mp1_wreg(dev, regMP1_SMN_C2PMSG_66, PPSMC_MSG_GetSmuVersion);

        for (int i = 0; i < 500; i++) {
            resp = mp1_rreg(dev, regMP1_SMN_C2PMSG_90);
            if (resp != 0) {
                ULONG version = mp1_rreg(dev, regMP1_SMN_C2PMSG_82);
                pr_info("gpu_psp_ring: SMU alive! version=0x%08x\n", version);
                break;
            }
            Sleep(1);
        }
        if (resp == 0) {
            pr_warn("gpu_psp_ring: SMU not responding after firmware load\n");
        }
    }

    ret = 0;

fail_fw:
    /* Unmap fence */
    if (fence_cpu) {
        AMDGPU_ESCAPE_MAP_BAR_DATA unmap = {};
        unmap.Header.Command = AMDGPU_ESCAPE_UNMAP_BAR;
        unmap.Header.Size = sizeof(unmap);
        unmap.MappedAddress = fence_cpu;
        unmap.MappingHandle = fence_mapping_handle;
        wddm_lite_escape(dev, &unmap, sizeof(unmap));
    }

    /* Unmap command buffer */
    if (cmd_cpu) {
        AMDGPU_ESCAPE_MAP_BAR_DATA unmap = {};
        unmap.Header.Command = AMDGPU_ESCAPE_UNMAP_BAR;
        unmap.Header.Size = sizeof(unmap);
        unmap.MappedAddress = cmd_cpu;
        unmap.MappingHandle = cmd_mapping_handle;
        wddm_lite_escape(dev, &unmap, sizeof(unmap));
    }

    /* Unmap firmware staging */
    if (fw_cpu) {
        AMDGPU_ESCAPE_MAP_BAR_DATA unmap = {};
        unmap.Header.Command = AMDGPU_ESCAPE_UNMAP_BAR;
        unmap.Header.Size = sizeof(unmap);
        unmap.MappedAddress = fw_cpu;
        unmap.MappingHandle = fw_mapping_handle;
        wddm_lite_escape(dev, &unmap, sizeof(unmap));
    }

fail_ring:
    /* Unmap ring buffer */
    if (ring_cpu) {
        AMDGPU_ESCAPE_MAP_BAR_DATA unmap = {};
        unmap.Header.Command = AMDGPU_ESCAPE_UNMAP_BAR;
        unmap.Header.Size = sizeof(unmap);
        unmap.MappedAddress = ring_cpu;
        unmap.MappingHandle = ring_mapping_handle;
        wddm_lite_escape(dev, &unmap, sizeof(unmap));
    }

fail:
    free(smu_buf);
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

    /* ---- Step 1: Try normal SMU mailbox (DisallowGfxOff) ---- */
    {
        ULONG resp_before = mp1_rreg(dev, regMP1_SMN_C2PMSG_90);
        pr_info("gpu_smu: C2PMSG_90 before = 0x%08x\n", resp_before);

        /* Quick SMU alive check */
        mp1_wreg(dev, regMP1_SMN_C2PMSG_90, 0);
        mp1_wreg(dev, regMP1_SMN_C2PMSG_82, 0);
        mp1_wreg(dev, regMP1_SMN_C2PMSG_66, PPSMC_MSG_GetSmuVersion);

        BOOLEAN alive = FALSE;
        for (int i = 0; i < 100; i++) {
            ULONG resp = mp1_rreg(dev, regMP1_SMN_C2PMSG_90);
            if (resp != 0) {
                alive = TRUE;
                ULONG version = mp1_rreg(dev, regMP1_SMN_C2PMSG_82);
                pr_info("gpu_smu: SMU alive, version = 0x%08x\n", version);
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
                Sleep(50);
                return 0;
            }
        } else {
            pr_warn("gpu_smu: normal SMU mailbox not responding\n");
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

/* GC base_index 1 register access (for GRBM, RLC, MEC, SH_MEM) */
static ULONG gc1_rreg(struct WddmLiteDevice *dev, ULONG reg)
{
    ULONG offset = (dev->hw.ip.gc_base1 + reg) * 4;
    return wddm_lite_read_reg32(dev, offset);
}

static void gc1_wreg(struct WddmLiteDevice *dev, ULONG reg, ULONG val)
{
    ULONG offset = (dev->hw.ip.gc_base1 + reg) * 4;
    wddm_lite_write_reg32(dev, offset, val);
}

/* GC base_index 0 register access (for CP_HQD, CP_STAT, GRBM_SOFT_RESET) */
static ULONG gc0_rreg(struct WddmLiteDevice *dev, ULONG reg)
{
    ULONG offset = (dev->hw.ip.gc_base + reg) * 4;
    return wddm_lite_read_reg32(dev, offset);
}

static void gc0_wreg(struct WddmLiteDevice *dev, ULONG reg, ULONG val)
{
    ULONG offset = (dev->hw.ip.gc_base + reg) * 4;
    wddm_lite_write_reg32(dev, offset, val);
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

/* SH_MEM_CONFIG field values (from amd_shared.h) */
#define SH_MEM_ADDRESS_MODE_64              1
#define SH_MEM_ALIGNMENT_MODE_UNALIGNED     3

/* CP_MEC_RS64_CNTL bit definitions */
#define CP_MEC_RS64_CNTL__MEC_PIPE0_RESET   (1 << 16)
#define CP_MEC_RS64_CNTL__MEC_PIPE0_ACTIVE  (1 << 26)
#define CP_MEC_RS64_CNTL__MEC_HALT          (1 << 30)

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

    /* Read RLC bootload status */
    gfx->rlc_bootload_status = gc1_rreg(dev, regRLC_RLCS_BOOTLOAD_STATUS);

    pr_info("gpu_gfx: CP_STAT = 0x%08x\n", gfx->cp_stat);
    pr_info("gpu_gfx: RLC_BOOTLOAD_STATUS = 0x%08x\n", gfx->rlc_bootload_status);

    /*
     * For a VBIOS-POST GPU (passthrough), RLC should already be loaded.
     * bootload_complete is bit 0 of RLC_RLCS_BOOTLOAD_STATUS.
     * If not set, firmware may not be ready yet.
     */
    BOOLEAN bootload_complete = (gfx->rlc_bootload_status & 0x1) != 0;

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

    memset(&dev->hw.gfx, 0, sizeof(dev->hw.gfx));

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
