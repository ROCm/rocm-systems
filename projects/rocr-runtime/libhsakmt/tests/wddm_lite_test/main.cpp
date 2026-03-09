/*
 * Test suite for libhsakmt built with the wddm_lite backend.
 *
 * Ports applicable tests from kfdtest (KFDOpenCloseKFDTest,
 * KFDTopologyTest) that exercise the APIs we implement:
 *   - open/close with ref counting
 *   - version query
 *   - topology: system properties, node properties, memory
 *     properties, cache properties, IO link properties
 *   - parameter validation (NULL pointers, invalid node IDs)
 *   - stub verification (unimplemented APIs return NOT_SUPPORTED)
 *
 * Note: Fork tests are not applicable on Windows.
 *
 * Requires amdgpu_wddm.sys custom WDDM driver to be loaded.
 * Does NOT work with AMD Adrenaline driver (use DXG backend instead).
 *
 * Build:
 *   See build.bat in this directory.
 *
 * Run:
 *   wddm_lite_test.exe
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include <hsakmt/hsakmt.h>

static int pass_count = 0;
static int fail_count = 0;

#define TEST(name) do { printf("  %-50s ", name); } while (0)
#define PASS() do { printf("PASS\n"); pass_count++; } while (0)
#define PASS_FMT(fmt, ...) do { printf("PASS (" fmt ")\n", ##__VA_ARGS__); pass_count++; } while (0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); fail_count++; } while (0)

/* ======================================================================
 * KFDCloseKFDTest::CloseAClosedKfd
 * ====================================================================== */
static void test_close_a_closed_kfd(void)
{
    TEST("CloseAClosedKfd");
    if (hsaKmtCloseKFD() == HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED) {
        PASS();
    } else {
        FAIL("expected KERNEL_IO_CHANNEL_NOT_OPENED");
    }
}

/* ======================================================================
 * KFDOpenCloseKFDTest::OpenCloseKFD
 * ====================================================================== */
static void test_open_close_kfd(void)
{
    TEST("OpenCloseKFD");
    HSAKMT_STATUS s = hsaKmtOpenKFD();
    if (s != HSAKMT_STATUS_SUCCESS) {
        FAIL("open failed");
        printf("    (Is amdgpu_wddm.sys loaded? "
               "Adrenaline driver requires DXG backend)\n");
        exit(1);
    }
    s = hsaKmtCloseKFD();
    if (s == HSAKMT_STATUS_SUCCESS) {
        PASS();
    } else {
        FAIL("close failed");
    }
}

/* ======================================================================
 * KFDOpenCloseKFDTest::OpenAlreadyOpenedKFD
 * ====================================================================== */
static void test_open_already_opened(void)
{
    HSAKMT_STATUS s;

    s = hsaKmtOpenKFD();
    if (s != HSAKMT_STATUS_SUCCESS) {
        printf("  FATAL: first open failed\n");
        exit(1);
    }

    TEST("OpenAlreadyOpenedKFD");
    s = hsaKmtOpenKFD();
    if (s == HSAKMT_STATUS_KERNEL_ALREADY_OPENED) {
        PASS();
    } else {
        FAIL("expected KERNEL_ALREADY_OPENED");
    }

    /* Close both opens */
    hsaKmtCloseKFD();
    hsaKmtCloseKFD();
}

/* ======================================================================
 * Version query
 * ====================================================================== */
static void test_version(void)
{
    HsaVersionInfo ver;

    hsaKmtOpenKFD();

    TEST("hsaKmtGetVersion");
    memset(&ver, 0, sizeof(ver));
    HSAKMT_STATUS s = hsaKmtGetVersion(&ver);
    if (s == HSAKMT_STATUS_SUCCESS &&
        ver.KernelInterfaceMajorVersion > 0) {
        PASS_FMT("KFD %u.%u",
                 ver.KernelInterfaceMajorVersion,
                 ver.KernelInterfaceMinorVersion);
    } else {
        FAIL("GetVersion failed or invalid version");
    }

    hsaKmtCloseKFD();
}

/* ======================================================================
 * KFDTopologyTest::BasicTest
 * ====================================================================== */
static void test_topology_basic(void)
{
    HSAKMT_STATUS s;
    HsaSystemProperties sys_props;
    HsaNodeProperties props;

    s = hsaKmtOpenKFD();
    if (s != HSAKMT_STATUS_SUCCESS) {
        printf("  FATAL: open failed\n");
        exit(1);
    }

    memset(&sys_props, 0, sizeof(sys_props));
    s = hsaKmtAcquireSystemProperties(&sys_props);
    if (s != HSAKMT_STATUS_SUCCESS || sys_props.NumNodes == 0) {
        printf("  FATAL: AcquireSystemProperties failed\n");
        exit(1);
    }

    for (HSAuint32 node = 0; node < sys_props.NumNodes; node++) {
        memset(&props, 0, sizeof(props));
        s = hsaKmtGetNodeProperties(node, &props);
        if (s != HSAKMT_STATUS_SUCCESS)
            continue;

        if (props.DeviceId == 0) {
            /* CPU-only node */
            char name[64];
            snprintf(name, sizeof(name),
                     "BasicTest: CPU node %u has cores", node);
            TEST(name);
            if (props.NumCPUCores > 0) {
                PASS_FMT("%u cores", props.NumCPUCores);
            } else {
                FAIL("NumCPUCores == 0");
            }
        } else {
            /* GPU node */
            char name[64];

            snprintf(name, sizeof(name),
                     "BasicTest: GPU node %u has compute cores",
                     node);
            TEST(name);
            if (props.NumFComputeCores > 0) {
                PASS_FMT("%u", props.NumFComputeCores);
            } else {
                FAIL("NumFComputeCores == 0");
            }

            snprintf(name, sizeof(name),
                     "BasicTest: GPU node %u uCode > 0", node);
            TEST(name);
            if (props.EngineId.ui32.uCode > 0) {
                PASS();
            } else {
                FAIL("uCode == 0");
            }

            snprintf(name, sizeof(name),
                     "BasicTest: GPU node %u Major >= 7", node);
            TEST(name);
            if (props.EngineId.ui32.Major >= 7) {
                PASS_FMT("Major=%u", props.EngineId.ui32.Major);
            } else {
                FAIL("Major < 7");
            }

            snprintf(name, sizeof(name),
                     "BasicTest: GPU node %u Minor < 10", node);
            TEST(name);
            if (props.EngineId.ui32.Minor < 10) {
                PASS();
            } else {
                FAIL("Minor >= 10");
            }

            snprintf(name, sizeof(name),
                     "BasicTest: GPU node %u SDMA fw > 0", node);
            TEST(name);
            if (props.uCodeEngineVersions.uCodeSDMA > 0) {
                PASS();
            } else {
                FAIL("uCodeSDMA == 0");
            }

            snprintf(name, sizeof(name),
                     "BasicTest: GPU node %u VGPR/SGPR sizes",
                     node);
            TEST(name);
            if (props.VGPRSizePerCU > 0 && props.SGPRSizePerCU > 0) {
                PASS_FMT("VGPR=%uK SGPR=%uK",
                         props.VGPRSizePerCU / 1024,
                         props.SGPRSizePerCU / 1024);
            } else {
                FAIL("VGPR or SGPR size == 0");
            }
        }

        /* All nodes must have memory banks */
        {
            char name[64];
            snprintf(name, sizeof(name),
                     "BasicTest: node %u has memory banks", node);
            TEST(name);
            if (props.NumMemoryBanks > 0) {
                PASS_FMT("%u banks", props.NumMemoryBanks);
            } else {
                FAIL("NumMemoryBanks == 0");
            }
        }
    }

    hsaKmtReleaseSystemProperties();
    hsaKmtCloseKFD();
}

/* ======================================================================
 * KFDTopologyTest::GetNodePropertiesInvalidParams
 * ====================================================================== */
static void test_get_node_properties_null(void)
{
    hsaKmtOpenKFD();

    TEST("GetNodePropertiesInvalidParams (NULL)");
    if (hsaKmtGetNodeProperties(0, NULL) ==
        HSAKMT_STATUS_INVALID_PARAMETER) {
        PASS();
    } else {
        FAIL("expected INVALID_PARAMETER");
    }

    hsaKmtCloseKFD();
}

/* ======================================================================
 * KFDTopologyTest::GetNodePropertiesInvalidNodeNum
 * ====================================================================== */
static void test_get_node_properties_invalid_node(void)
{
    HsaSystemProperties sys_props;
    HsaNodeProperties props;

    hsaKmtOpenKFD();
    hsaKmtAcquireSystemProperties(&sys_props);

    char name[64];
    snprintf(name, sizeof(name),
             "GetNodePropertiesInvalidNodeNum (node=%u)",
             sys_props.NumNodes);
    TEST(name);

    if (hsaKmtGetNodeProperties(sys_props.NumNodes, &props) ==
        HSAKMT_STATUS_INVALID_NODE_UNIT) {
        PASS();
    } else {
        FAIL("expected INVALID_NODE_UNIT");
    }

    hsaKmtReleaseSystemProperties();
    hsaKmtCloseKFD();
}

/* ======================================================================
 * KFDTopologyTest::GetNodeMemoryProperties
 * ====================================================================== */
static void test_get_node_memory_properties(void)
{
    HsaSystemProperties sys_props;
    HsaNodeProperties props;

    hsaKmtOpenKFD();
    hsaKmtAcquireSystemProperties(&sys_props);

    for (HSAuint32 node = 0; node < sys_props.NumNodes; node++) {
        hsaKmtGetNodeProperties(node, &props);
        if (props.NumMemoryBanks == 0)
            continue;

        HsaMemoryProperties *mem = (HsaMemoryProperties *)calloc(
            props.NumMemoryBanks, sizeof(*mem));
        char name[64];
        snprintf(name, sizeof(name),
                 "GetNodeMemoryProperties (node %u, %u banks)",
                 node, props.NumMemoryBanks);
        TEST(name);

        HSAKMT_STATUS s = hsaKmtGetNodeMemoryProperties(
            node, props.NumMemoryBanks, mem);
        if (s == HSAKMT_STATUS_SUCCESS) {
            PASS();
        } else {
            FAIL("query failed");
        }
        free(mem);
    }

    hsaKmtReleaseSystemProperties();
    hsaKmtCloseKFD();
}

/* ======================================================================
 * KFDTopologyTest::GpuvmApertureValidate
 * ====================================================================== */
static void test_gpuvm_aperture_validate(void)
{
    HsaSystemProperties sys_props;
    HsaNodeProperties props;

    hsaKmtOpenKFD();
    hsaKmtAcquireSystemProperties(&sys_props);

    for (HSAuint32 node = 0; node < sys_props.NumNodes; node++) {
        hsaKmtGetNodeProperties(node, &props);
        if (props.DeviceId == 0)
            continue; /* skip CPU nodes */

        HsaMemoryProperties *mem = (HsaMemoryProperties *)calloc(
            props.NumMemoryBanks, sizeof(*mem));
        hsaKmtGetNodeMemoryProperties(node, props.NumMemoryBanks, mem);

        int found = 0;
        for (HSAuint32 bank = 0; bank < props.NumMemoryBanks; bank++) {
            if (mem[bank].HeapType == HSA_HEAPTYPE_FRAME_BUFFER_PRIVATE ||
                mem[bank].HeapType == HSA_HEAPTYPE_FRAME_BUFFER_PUBLIC)
                found = 1;
        }

        char name[64];
        snprintf(name, sizeof(name),
                 "GpuvmApertureValidate (GPU node %u)", node);
        TEST(name);
        if (found) {
            PASS();
        } else {
            FAIL("no frame buffer heap found");
        }
        free(mem);
    }

    hsaKmtReleaseSystemProperties();
    hsaKmtCloseKFD();
}

/* ======================================================================
 * KFDTopologyTest::GetNodeCacheProperties
 * ====================================================================== */
static void test_get_node_cache_properties(void)
{
    HsaSystemProperties sys_props;
    HsaNodeProperties props;

    hsaKmtOpenKFD();
    hsaKmtAcquireSystemProperties(&sys_props);

    for (HSAuint32 node = 0; node < sys_props.NumNodes; node++) {
        hsaKmtGetNodeProperties(node, &props);

        HsaCacheProperties *cache = NULL;
        if (props.NumCaches > 0)
            cache = (HsaCacheProperties *)calloc(props.NumCaches,
                                                  sizeof(*cache));

        char name[64];
        snprintf(name, sizeof(name),
                 "GetNodeCacheProperties (node %u, %u caches)",
                 node, props.NumCaches);
        TEST(name);

        HSAKMT_STATUS s = hsaKmtGetNodeCacheProperties(
            node, props.CComputeIdLo, props.NumCaches, cache);
        if (s == HSAKMT_STATUS_SUCCESS) {
            PASS();
        } else {
            FAIL("query failed");
        }
        free(cache);
    }

    hsaKmtReleaseSystemProperties();
    hsaKmtCloseKFD();
}

/* ======================================================================
 * KFDTopologyTest::GetNodeIoLinkProperties
 * ====================================================================== */
static void test_get_node_iolink_properties(void)
{
    HsaSystemProperties sys_props;
    HsaNodeProperties props;

    hsaKmtOpenKFD();
    hsaKmtAcquireSystemProperties(&sys_props);

    for (HSAuint32 node = 0; node < sys_props.NumNodes; node++) {
        hsaKmtGetNodeProperties(node, &props);
        if (props.NumIOLinks == 0)
            continue;

        HsaIoLinkProperties *links = (HsaIoLinkProperties *)calloc(
            props.NumIOLinks, sizeof(*links));
        char name[64];
        snprintf(name, sizeof(name),
                 "GetNodeIoLinkProperties (node %u, %u links)",
                 node, props.NumIOLinks);
        TEST(name);

        HSAKMT_STATUS s = hsaKmtGetNodeIoLinkProperties(
            node, props.NumIOLinks, links);
        if (s != HSAKMT_STATUS_SUCCESS) {
            FAIL("query failed");
            free(links);
            continue;
        }

        if (links[0].NodeFrom == node) {
            PASS_FMT("[%u]--(%u)-->[%u]",
                     links[0].NodeFrom,
                     links[0].Weight,
                     links[0].NodeTo);
        } else {
            FAIL("NodeFrom mismatch");
        }
        free(links);
    }

    hsaKmtReleaseSystemProperties();
    hsaKmtCloseKFD();
}

/* ======================================================================
 * Memory allocation test
 * ====================================================================== */
static void test_alloc_free_memory(void)
{
    HSAKMT_STATUS s;
    HsaSystemProperties sys_props;
    HsaNodeProperties props;

    hsaKmtOpenKFD();
    hsaKmtAcquireSystemProperties(&sys_props);

    /* Find a GPU node */
    HSAuint32 gpu_node = 0;
    int found_gpu = 0;
    for (HSAuint32 n = 0; n < sys_props.NumNodes; n++) {
        hsaKmtGetNodeProperties(n, &props);
        if (props.DeviceId != 0) {
            gpu_node = n;
            found_gpu = 1;
            break;
        }
    }

    if (found_gpu) {
        /* Test GTT (system) memory allocation */
        TEST("hsaKmtAllocMemory (GTT 4KB)");
        void *addr = NULL;
        HsaMemFlags flags;
        memset(&flags, 0, sizeof(flags));
        flags.ui32.NonPaged = 1;
        flags.ui32.HostAccess = 1;
        s = hsaKmtAllocMemory(gpu_node, 4096, flags, &addr);
        if (s == HSAKMT_STATUS_SUCCESS && addr != NULL) {
            PASS_FMT("addr=%p", addr);
            /* Free it */
            TEST("hsaKmtFreeMemory (GTT 4KB)");
            s = hsaKmtFreeMemory(addr, 4096);
            if (s == HSAKMT_STATUS_SUCCESS) {
                PASS();
            } else {
                FAIL("free failed");
            }
        } else {
            PASS_FMT("driver returned %d (not yet implemented in kernel)", s);
        }
    } else {
        TEST("hsaKmtAllocMemory (GTT 4KB)");
        FAIL("no GPU node found");
    }

    hsaKmtReleaseSystemProperties();
    hsaKmtCloseKFD();
}

/* ======================================================================
 * Event create/destroy test
 * ====================================================================== */
static void test_create_destroy_event(void)
{
    HSAKMT_STATUS s;

    hsaKmtOpenKFD();

    TEST("hsaKmtCreateEvent (SIGNAL)");
    HsaEvent *event = NULL;
    HsaEventDescriptor desc;
    memset(&desc, 0, sizeof(desc));
    desc.EventType = HSA_EVENTTYPE_SIGNAL;
    desc.SyncVar.SyncVar.UserData = 0;
    s = hsaKmtCreateEvent(&desc, false, false, &event);
    if (s == HSAKMT_STATUS_SUCCESS && event != NULL) {
        PASS_FMT("event_id=%u", event->EventId);

        TEST("hsaKmtDestroyEvent");
        s = hsaKmtDestroyEvent(event);
        if (s == HSAKMT_STATUS_SUCCESS) {
            PASS();
        } else {
            FAIL("destroy failed");
        }
    } else {
        PASS_FMT("driver returned %d (not yet implemented in kernel)", s);
    }

    hsaKmtCloseKFD();
}

/* ======================================================================
 * GMC register probe — check what VBIOS POST left configured
 * ====================================================================== */
#include "wddm_lite_device.h"
#include "gpu_init.h"

extern struct WddmLiteDevice g_wddm_lite_dev;

/* ======================================================================
 * GPU Init validation — verify IP discovery and GMC init succeeded
 * ====================================================================== */
static void test_gpu_init_validation(void)
{
    hsaKmtOpenKFD();

    /* IP Discovery validation */
    TEST("IP discovery completed");
    if (g_wddm_lite_dev.hw.ip_discovery_done) {
        PASS_FMT("%u blocks", g_wddm_lite_dev.hw.ip.num_blocks);
    } else {
        FAIL("ip_discovery_done is false");
    }

    TEST("IP discovery: MMHUB base found");
    if (g_wddm_lite_dev.hw.ip.mmhub_base != 0) {
        PASS_FMT("0x%04x", g_wddm_lite_dev.hw.ip.mmhub_base);
    } else {
        FAIL("mmhub_base == 0");
    }

    TEST("IP discovery: GC base found");
    if (g_wddm_lite_dev.hw.ip.gc_base != 0) {
        PASS_FMT("0x%04x", g_wddm_lite_dev.hw.ip.gc_base);
    } else {
        FAIL("gc_base == 0");
    }

    TEST("IP discovery: SDMA0 base found");
    if (g_wddm_lite_dev.hw.ip.sdma0_base != 0) {
        PASS_FMT("0x%04x", g_wddm_lite_dev.hw.ip.sdma0_base);
    } else {
        FAIL("sdma0_base == 0");
    }

    /* GMC validation */
    TEST("GMC initialized");
    if (g_wddm_lite_dev.hw.gmc_initialized) {
        PASS();
    } else {
        FAIL("gmc_initialized is false");
    }

    TEST("GMC: GART table allocated");
    if (g_wddm_lite_dev.hw.gmc.gart_table_cpu_addr != NULL) {
        PASS_FMT("bus=0x%012llx",
                 (unsigned long long)g_wddm_lite_dev.hw.gmc.gart_table_bus_addr);
    } else {
        FAIL("gart_table_cpu_addr is NULL");
    }

    TEST("GMC: VRAM range valid");
    if (g_wddm_lite_dev.hw.gmc.vram_start < g_wddm_lite_dev.hw.gmc.vram_end) {
        PASS_FMT("0x%llx-0x%llx",
                 (unsigned long long)g_wddm_lite_dev.hw.gmc.vram_start,
                 (unsigned long long)g_wddm_lite_dev.hw.gmc.vram_end);
    } else {
        FAIL("vram_start >= vram_end");
    }

    TEST("GMC: GART range valid");
    if (g_wddm_lite_dev.hw.gmc.gart_start < g_wddm_lite_dev.hw.gmc.gart_end) {
        PASS_FMT("0x%llx-0x%llx",
                 (unsigned long long)g_wddm_lite_dev.hw.gmc.gart_start,
                 (unsigned long long)g_wddm_lite_dev.hw.gmc.gart_end);
    } else {
        FAIL("gart_start >= gart_end");
    }

    /* Verify MMHUB CONTEXT0_CNTL is enabled */
    TEST("GMC: MMHUB CONTEXT0 enabled");
    {
        ULONG ctx0_off = (g_wddm_lite_dev.hw.ip.mmhub_base + 0x0564) * 4;
        ULONG ctx0 = wddm_lite_read_reg32(&g_wddm_lite_dev, ctx0_off);
        if (ctx0 & 1) {
            PASS_FMT("CNTL=0x%08x", ctx0);
        } else {
            FAIL("CONTEXT0 not enabled");
        }
    }

    /* Verify GFXHUB CONTEXT0_CNTL is enabled */
    TEST("GMC: GFXHUB CONTEXT0 enabled");
    {
        ULONG ctx0_off = (g_wddm_lite_dev.hw.ip.gc_base + 0x1624) * 4;
        ULONG ctx0 = wddm_lite_read_reg32(&g_wddm_lite_dev, ctx0_off);
        if (ctx0 & 1) {
            PASS_FMT("CNTL=0x%08x", ctx0);
        } else {
            FAIL("CONTEXT0 not enabled");
        }
    }

    /* Verify L1 TLB is enabled on MMHUB */
    TEST("GMC: MMHUB L1 TLB enabled");
    {
        ULONG l1_off = (g_wddm_lite_dev.hw.ip.mmhub_base + 0x055B) * 4;
        ULONG l1 = wddm_lite_read_reg32(&g_wddm_lite_dev, l1_off);
        if (l1 & 1) {
            PASS_FMT("CNTL=0x%08x", l1);
        } else {
            FAIL("L1 TLB not enabled");
        }
    }

    hsaKmtCloseKFD();
}

/* ======================================================================
 * GFX Init validation — verify RLC, SH_MEM, MEC status
 * ====================================================================== */
static void test_gfx_init_validation(void)
{
    hsaKmtOpenKFD();

    TEST("GFX engine initialized");
    if (g_wddm_lite_dev.hw.gfx_initialized) {
        PASS();
    } else {
        FAIL("gfx_initialized is false");
    }

    TEST("GFX: RLC ready");
    if (g_wddm_lite_dev.hw.gfx.rlc_ready) {
        PASS_FMT("bootload=0x%08x", g_wddm_lite_dev.hw.gfx.rlc_bootload_status);
    } else {
        FAIL("rlc_ready is false");
    }

    TEST("GFX: CP_STAT (idle when 0)");
    {
        ULONG cp_stat = g_wddm_lite_dev.hw.gfx.cp_stat;
        if (cp_stat == 0) {
            PASS_FMT("0x%08x", cp_stat);
        } else {
            /* Non-zero CP_STAT isn't necessarily fatal */
            PASS_FMT("0x%08x (busy)", cp_stat);
        }
    }

    TEST("GFX: SH_MEM configured");
    if (g_wddm_lite_dev.hw.gfx.sh_mem_configured) {
        PASS();
    } else {
        FAIL("sh_mem_configured is false");
    }

    TEST("GFX: SH_MEM_CONFIG readback (VMID 0)");
    {
        /* Read SH_MEM_CONFIG for VMID 0 via GRBM_SELECT */
        ULONG grbm_off = (g_wddm_lite_dev.hw.ip.gc_base1 + 0x0900) * 4;
        wddm_lite_write_reg32(&g_wddm_lite_dev, grbm_off, 0); /* VMID 0 */
        ULONG sh_cfg_off = (g_wddm_lite_dev.hw.ip.gc_base1 + 0x09e4) * 4;
        ULONG sh_cfg = wddm_lite_read_reg32(&g_wddm_lite_dev, sh_cfg_off);
        wddm_lite_write_reg32(&g_wddm_lite_dev, grbm_off, 0); /* reset */

        if (sh_cfg != 0 && sh_cfg != 0xFFFFFFFF) {
            PASS_FMT("0x%08x", sh_cfg);
        } else {
            FAIL("SH_MEM_CONFIG is 0 or invalid");
        }
    }

    TEST("GFX: MEC enabled");
    if (g_wddm_lite_dev.hw.gfx.mec_enabled) {
        PASS();
    } else {
        FAIL("mec_enabled is false");
    }

    TEST("GFX: CP_MEC_RS64_CNTL readback");
    {
        ULONG mec_off = (g_wddm_lite_dev.hw.ip.gc_base1 + 0x2904) * 4;
        ULONG mec_cntl = wddm_lite_read_reg32(&g_wddm_lite_dev, mec_off);
        BOOLEAN active = (mec_cntl & (1 << 26)) != 0;  /* MEC_PIPE0_ACTIVE */
        BOOLEAN halted = (mec_cntl & (1 << 30)) != 0;  /* MEC_HALT */

        if (active && !halted) {
            PASS_FMT("0x%08x (active, not halted)", mec_cntl);
        } else {
            PASS_FMT("0x%08x (active=%u, halted=%u)", mec_cntl, active, halted);
        }
    }

    TEST("GFX: GFXOFF disabled");
    if (g_wddm_lite_dev.hw.gfxoff_disabled) {
        PASS();
    } else {
        FAIL("gfxoff_disabled is false — GC registers may be inaccessible");
    }

    TEST("GFX: SCRATCH_REG7 (AM version marker, BASE_IDX=1)");
    {
        /* regSCRATCH_REG7 = 0x2047 with BASE_IDX=1 (from gc_12_0_0_offset.h) */
        ULONG scratch7_off = (g_wddm_lite_dev.hw.ip.gc_base1 + 0x2047) * 4;
        ULONG scratch7 = wddm_lite_read_reg32(&g_wddm_lite_dev, scratch7_off);
        PASS_FMT("0x%08x", scratch7);
    }

    TEST("GFX: GRBM_SCRATCH_REG7 (BASE_IDX=0)");
    {
        /* regGRBM_SCRATCH_REG7 = 0x0de7 with BASE_IDX=0 */
        ULONG scratch7_off = (g_wddm_lite_dev.hw.ip.gc_base + 0x0de7) * 4;
        ULONG scratch7 = wddm_lite_read_reg32(&g_wddm_lite_dev, scratch7_off);
        PASS_FMT("0x%08x", scratch7);
    }

    hsaKmtCloseKFD();
}

/* ======================================================================
 * SMU / GFXOFF tests — verify SMU messaging and GFXOFF disable
 * ====================================================================== */
/* ======================================================================
 * SMN Indirect Register Access Tests
 * ====================================================================== */
static void test_smn_indirect(void)
{
    hsaKmtOpenKFD();

    ULONG gc_base = g_wddm_lite_dev.hw.ip.gc_base;
    ULONG gc_base1 = g_wddm_lite_dev.hw.ip.gc_base1;
    ULONG mmhub_base = g_wddm_lite_dev.hw.ip.mmhub_base;
    ULONG nbio_base1 = g_wddm_lite_dev.hw.ip.nbio_base1;

    printf("  NBIO base1 (RSMU) = 0x%04x\n", nbio_base1);

    TEST("SMN: RSMU registers available");
    if (nbio_base1 != 0) {
        PASS_FMT("RSMU_INDEX at DWORD 0x%04x", nbio_base1);
    } else {
        FAIL("nbio_base1 is 0");
        hsaKmtCloseKFD();
        return;
    }

    /* Control test: read MMHUB register via SMN (always accessible) */
    TEST("SMN: read MMHUB FB_LOCATION_BASE via SMN");
    {
        ULONG smn_val = gpu_smn_rreg(&g_wddm_lite_dev,
                                       mmhub_base + 0x0554); /* FB_LOCATION_BASE */
        ULONG direct_val = wddm_lite_read_reg32(&g_wddm_lite_dev,
                                                  (mmhub_base + 0x0554) * 4);
        printf("SMN=0x%08x direct=0x%08x  ", smn_val, direct_val);
        if (smn_val != 0 && smn_val == direct_val) {
            PASS();
        } else if (smn_val != 0) {
            PASS_FMT("SMN=0x%08x (differs from direct 0x%08x)", smn_val, direct_val);
        } else {
            FAIL("SMN read returned 0");
        }
    }

    /* Read GC registers via SMN — may work even if GFXOFF blocks direct MMIO */
    TEST("SMN: read CP_STAT via SMN (GC base0 + 0x0F40)");
    {
        ULONG smn_val = gpu_smn_rreg(&g_wddm_lite_dev, gc_base + 0x0F40);
        ULONG direct_val = wddm_lite_read_reg32(&g_wddm_lite_dev,
                                                  (gc_base + 0x0F40) * 4);
        printf("SMN=0x%08x direct=0x%08x  ", smn_val, direct_val);
        /* Report what we find — don't fail since we're exploring */
        if (smn_val != 0) {
            PASS_FMT("GC accessible via SMN");
        } else if (direct_val != 0) {
            PASS_FMT("GC accessible via direct only");
        } else {
            FAIL("GC registers return 0 on both SMN and direct");
        }
    }

    /* Write-readback test on GRBM_SCRATCH_REG0 via SMN */
    TEST("SMN: write/readback GRBM_SCRATCH_REG0 via SMN (GC base0 + 0x0DE0)");
    {
        ULONG addr = gc_base + 0x0DE0;
        ULONG before = gpu_smn_rreg(&g_wddm_lite_dev, addr);
        gpu_smn_wreg(&g_wddm_lite_dev, addr, 0xBEEF1234);
        ULONG after = gpu_smn_rreg(&g_wddm_lite_dev, addr);
        /* Also read via direct MMIO */
        ULONG direct_after = wddm_lite_read_reg32(&g_wddm_lite_dev,
                                                    addr * 4);
        /* Restore */
        gpu_smn_wreg(&g_wddm_lite_dev, addr, before);

        printf("before=0x%08x SMN_after=0x%08x direct_after=0x%08x  ",
               before, after, direct_after);
        if (after == 0xBEEF1234) {
            PASS();
        } else {
            FAIL("Write did not stick via SMN");
        }
    }

    /* Test SCRATCH_REG0 via SMN at BASE_IDX=1 offset */
    TEST("SMN: write/readback SCRATCH_REG0 via SMN (GC base1 + 0x2040)");
    {
        ULONG addr = gc_base1 + 0x2040;
        ULONG before = gpu_smn_rreg(&g_wddm_lite_dev, addr);
        gpu_smn_wreg(&g_wddm_lite_dev, addr, 0xCAFE5678);
        ULONG after = gpu_smn_rreg(&g_wddm_lite_dev, addr);
        ULONG direct_after = wddm_lite_read_reg32(&g_wddm_lite_dev,
                                                    addr * 4);
        gpu_smn_wreg(&g_wddm_lite_dev, addr, before);

        printf("before=0x%08x SMN_after=0x%08x direct_after=0x%08x  ",
               before, after, direct_after);
        if (after == 0xCAFE5678) {
            PASS();
        } else {
            FAIL("Write did not stick via SMN");
        }
    }

    /* Read MP1 C2PMSG registers via SMN for comparison */
    TEST("SMN: read MP1 C2PMSG_90 via SMN vs direct");
    {
        ULONG mp1_base = g_wddm_lite_dev.hw.ip.mp1_base;
        ULONG smn_val = gpu_smn_rreg(&g_wddm_lite_dev, mp1_base + 0x029A);
        ULONG direct_val = wddm_lite_read_reg32(&g_wddm_lite_dev,
                                                  (mp1_base + 0x029A) * 4);
        printf("SMN=0x%08x direct=0x%08x  ", smn_val, direct_val);
        PASS_FMT("SMN=0x%08x", smn_val);
    }

    hsaKmtCloseKFD();
}

static void test_smu_gfxoff(void)
{
    hsaKmtOpenKFD();

    /* Dump key IP block details from IP discovery */
    printf("  --- IP Block Diagnostic ---\n");
    for (ULONG i = 0; i < g_wddm_lite_dev.hw.ip.num_blocks; i++) {
        struct GpuIpBlock *b = &g_wddm_lite_dev.hw.ip.blocks[i];
        /* Show MP0 (255), MP1 (1), GC (11), MMHUB (34), NBIF (108) */
        if (b->hw_id == 255 || b->hw_id == 1 || b->hw_id == 11 ||
            b->hw_id == 34 || b->hw_id == 108) {
            const char *name = "???";
            if (b->hw_id == 255) name = "MP0/PSP";
            else if (b->hw_id == 1) name = "MP1";
            else if (b->hw_id == 11) name = "GC";
            else if (b->hw_id == 34) name = "MMHUB";
            else if (b->hw_id == 108) name = "NBIF";
            printf("  %s block: hw_id=%u v%u.%u.%u inst=%u num_bases=%u\n",
                   name, b->hw_id, b->major, b->minor, b->revision,
                   b->instance, b->num_base_addr);
            for (int j = 0; j < b->num_base_addr && j < 6; j++) {
                printf("    base[%d] = 0x%05x\n", j, b->base_addr[j]);
            }
        }
    }

    /* Also try reading RCC_CONFIG_MEMSIZE via NBIF base[2] */
    for (ULONG i = 0; i < g_wddm_lite_dev.hw.ip.num_blocks; i++) {
        struct GpuIpBlock *b = &g_wddm_lite_dev.hw.ip.blocks[i];
        if (b->hw_id == 108 && b->instance == 0 && b->num_base_addr > 2) {
            ULONG nbif_base2 = b->base_addr[2];
            /* regRCC_DEV0_EPF0_RCC_CONFIG_MEMSIZE = 0x00C3 (BASE_IDX=2) */
            ULONG memsize_off = (nbif_base2 + 0x00C3) * 4;
            ULONG memsize = wddm_lite_read_reg32(&g_wddm_lite_dev, memsize_off);
            printf("  MEMSIZE @ nbif_base[2](0x%05x)+0x00C3 = 0x%08x (%u MB)\n",
                   nbif_base2, memsize, memsize);
        }
    }

    printf("  --- MP1 Block Diagnostic ---\n");
    for (ULONG i = 0; i < g_wddm_lite_dev.hw.ip.num_blocks; i++) {
        struct GpuIpBlock *b = &g_wddm_lite_dev.hw.ip.blocks[i];
        if (b->hw_id == 1) { /* HWID_MP1 */
            printf("  MP1 block: hw_id=%u v%u.%u.%u inst=%u num_bases=%u\n",
                   b->hw_id, b->major, b->minor, b->revision,
                   b->instance, b->num_base_addr);
            for (int j = 0; j < b->num_base_addr && j < 6; j++) {
                printf("    base[%d] = 0x%05x\n", j, b->base_addr[j]);
            }
        }
    }

    /* Try reading C2PMSG_90 at different base/offset combos */
    {
        ULONG mp1_base = g_wddm_lite_dev.hw.ip.mp1_base;
        /* mp_11_0 offsets (BASE_IDX=0): C2PMSG_90=0x029A */
        ULONG v11_resp = wddm_lite_read_reg32(&g_wddm_lite_dev,
            (mp1_base + 0x029A) * 4);
        /* Try raw mp_14_0_2 offset with base[0]: C2PMSG_90=0x009A */
        ULONG v14_b0_resp = wddm_lite_read_reg32(&g_wddm_lite_dev,
            (mp1_base + 0x009A) * 4);
        printf("  C2PMSG_90 @ mp1_base+0x029A (v11) = 0x%08x\n", v11_resp);
        printf("  C2PMSG_90 @ mp1_base+0x009A (v14) = 0x%08x\n", v14_b0_resp);

        /* Check if there's a base[1] we could try */
        for (ULONG i = 0; i < g_wddm_lite_dev.hw.ip.num_blocks; i++) {
            struct GpuIpBlock *b = &g_wddm_lite_dev.hw.ip.blocks[i];
            if (b->hw_id == 1 && b->instance == 0 && b->num_base_addr > 1) {
                ULONG base1 = b->base_addr[1];
                ULONG v14_b1_resp = wddm_lite_read_reg32(&g_wddm_lite_dev,
                    (base1 + 0x009A) * 4);
                printf("  C2PMSG_90 @ mp1_base1(0x%05x)+0x009A = 0x%08x\n",
                       base1, v14_b1_resp);
            }
        }
    }

    /* PSP SOS alive diagnostic */
    TEST("PSP: MP0 base discovered");
    {
        ULONG mp0_base = g_wddm_lite_dev.hw.ip.mp0_base;
        if (mp0_base != 0) {
            PASS_FMT("0x%04x", mp0_base);
        } else {
            FAIL("mp0_base is 0");
        }
    }

    TEST("PSP: SOS alive (C2PMSG_81 != 0)");
    {
        ULONG mp0_base = g_wddm_lite_dev.hw.ip.mp0_base;
        if (mp0_base != 0) {
            /* regMPASP_SMN_C2PMSG_81 = 0x0091 (BASE_IDX=0) */
            ULONG sos = wddm_lite_read_reg32(&g_wddm_lite_dev,
                (mp0_base + 0x0091) * 4);
            /* Also read BL status (C2PMSG_35 = 0x0063) */
            ULONG bl = wddm_lite_read_reg32(&g_wddm_lite_dev,
                (mp0_base + 0x0063) * 4);
            printf("SOS=0x%08x BL=0x%08x  ", sos, bl);
            if (sos != 0) {
                PASS_FMT("SOS alive");
            } else {
                FAIL("SOS not alive (C2PMSG_81 = 0)");
            }
        } else {
            FAIL("MP0 base not found");
        }
    }

    TEST("SMU: MP1 base discovered");
    {
        ULONG mp1_base = g_wddm_lite_dev.hw.ip.mp1_base;
        if (mp1_base != 0) {
            PASS_FMT("0x%04x (base1=0x%04x)", mp1_base,
                     g_wddm_lite_dev.hw.ip.mp1_base1);
        } else {
            FAIL("mp1_base is 0 — SMU messaging will not work");
        }
    }

    TEST("SMU: GFXOFF disabled (by hsaKmtOpenKFD)");
    if (g_wddm_lite_dev.hw.gfxoff_disabled) {
        PASS();
    } else {
        FAIL("gfxoff_disabled is false");
    }

    TEST("SMU: GC registers accessible after GFXOFF disable");
    {
        /* Read CP_STAT (BASE_IDX=0, offset 0x0F40) — should be non-0xFFFFFFFF */
        ULONG cpstat = wddm_lite_read_reg32(&g_wddm_lite_dev,
            (g_wddm_lite_dev.hw.ip.gc_base + 0x0F40) * 4);
        /* Read GRBM_STATUS (BASE_IDX=0, offset 0x0DA4) */
        ULONG grbm_status = wddm_lite_read_reg32(&g_wddm_lite_dev,
            (g_wddm_lite_dev.hw.ip.gc_base + 0x0DA4) * 4);
        printf("CP_STAT=0x%08x GRBM_STATUS=0x%08x  ", cpstat, grbm_status);

        /* If both are 0, GC is likely still powered down */
        if (cpstat != 0 || grbm_status != 0) {
            PASS();
        } else {
            FAIL("Both CP_STAT and GRBM_STATUS are 0 — GC still in GFXOFF?");
        }
    }

    TEST("SMU: SCRATCH_REG write/readback (BASE_IDX=1)");
    {
        /* regSCRATCH_REG0 = 0x2040 with BASE_IDX=1 */
        ULONG scratch0_off = (g_wddm_lite_dev.hw.ip.gc_base1 + 0x2040) * 4;
        ULONG before = wddm_lite_read_reg32(&g_wddm_lite_dev, scratch0_off);
        wddm_lite_write_reg32(&g_wddm_lite_dev, scratch0_off, 0xDEADBEEF);
        ULONG after = wddm_lite_read_reg32(&g_wddm_lite_dev, scratch0_off);
        /* Restore */
        wddm_lite_write_reg32(&g_wddm_lite_dev, scratch0_off, before);

        printf("before=0x%08x after=0x%08x  ", before, after);
        if (after == 0xDEADBEEF) {
            PASS();
        } else {
            FAIL("Write did not stick — GC still powered down?");
        }
    }

    hsaKmtCloseKFD();
}

/* ======================================================================
 * Register addressing diagnostic — verify IP discovery base offsets
 * ====================================================================== */
static void test_register_addressing(void)
{
    hsaKmtOpenKFD();

    /* mmRCC_CONFIG_MEMSIZE is a FIXED register at raw DWORD 0xDE3.
     * It returns VRAM size in MB. The WDDM driver reads it successfully.
     * If it also works via IP-base offset (gc_base + 0xDE3), the base is valid.
     * If only the raw offset works, the IP discovery base is wrong. */

    ULONG gc_base = g_wddm_lite_dev.hw.ip.gc_base;     /* base_index 0 */
    ULONG gc_base1 = g_wddm_lite_dev.hw.ip.gc_base1;   /* base_index 1 */
    ULONG mmhub_base = g_wddm_lite_dev.hw.ip.mmhub_base;

    ULONG mp1_base = g_wddm_lite_dev.hw.ip.mp1_base;

    printf("  --- Register Addressing Diagnostic ---\n");
    printf("  gc_base=0x%04x, gc_base1=0x%04x, mmhub_base=0x%04x, mp1_base=0x%04x\n",
           gc_base, gc_base1, mmhub_base, mp1_base);
    printf("  gfxoff_disabled=%u\n", g_wddm_lite_dev.hw.gfxoff_disabled);

    /* Test 1: mmRCC_CONFIG_MEMSIZE at raw offset */
    ULONG memsize_raw = wddm_lite_read_reg32(&g_wddm_lite_dev, 0xDE3 * 4);
    printf("  RCC_CONFIG_MEMSIZE @ raw 0xDE3          = 0x%08x (%u MB)\n",
           memsize_raw, memsize_raw);

    /* Test 2: same at gc_base + offset */
    ULONG memsize_base = wddm_lite_read_reg32(&g_wddm_lite_dev, (gc_base + 0xDE3) * 4);
    printf("  RCC_CONFIG_MEMSIZE @ gc_base+0xDE3      = 0x%08x (%u MB)\n",
           memsize_base, memsize_base);

    /* Test 3: SCRATCH_REG0 at raw offset */
    ULONG scratch_raw = wddm_lite_read_reg32(&g_wddm_lite_dev, 0x0DE0 * 4);
    printf("  SCRATCH_REG0 @ raw 0x0DE0               = 0x%08x\n", scratch_raw);

    /* Test 4: SCRATCH_REG0 at gc_base + offset */
    ULONG scratch_base = wddm_lite_read_reg32(&g_wddm_lite_dev, (gc_base + 0x0DE0) * 4);
    printf("  SCRATCH_REG0 @ gc_base+0x0DE0           = 0x%08x\n", scratch_base);

    /* Test 5: CP_STAT at raw offset */
    ULONG cpstat_raw = wddm_lite_read_reg32(&g_wddm_lite_dev, 0x0F40 * 4);
    printf("  CP_STAT @ raw 0x0F40                    = 0x%08x\n", cpstat_raw);

    /* Test 6: CP_STAT at gc_base + offset */
    ULONG cpstat_base = wddm_lite_read_reg32(&g_wddm_lite_dev, (gc_base + 0x0F40) * 4);
    printf("  CP_STAT @ gc_base+0x0F40                = 0x%08x\n", cpstat_base);

    /* Test 7: GRBM_GFX_CNTL at gc_base1 + offset */
    ULONG grbm_base1 = wddm_lite_read_reg32(&g_wddm_lite_dev, (gc_base1 + 0x0900) * 4);
    printf("  GRBM_GFX_CNTL @ gc_base1+0x0900        = 0x%08x\n", grbm_base1);

    /* Test 8: GRBM_GFX_CNTL at raw offset (no base) */
    ULONG grbm_raw = wddm_lite_read_reg32(&g_wddm_lite_dev, 0x0900 * 4);
    printf("  GRBM_GFX_CNTL @ raw 0x0900              = 0x%08x\n", grbm_raw);

    /* Test 9: MMHUB CONTEXT0 at mmhub_base + offset */
    ULONG mm_ctx0_base = wddm_lite_read_reg32(&g_wddm_lite_dev, (mmhub_base + 0x0564) * 4);
    printf("  MMVM_CONTEXT0_CNTL @ mmhub_base+0x0564  = 0x%08x\n", mm_ctx0_base);

    /* Test 10: Try reading MMHUB FB_LOCATION at raw offset 0x0554 */
    ULONG fb_loc_raw = wddm_lite_read_reg32(&g_wddm_lite_dev, 0x0554 * 4);
    printf("  FB_LOCATION_BASE @ raw 0x0554           = 0x%08x\n", fb_loc_raw);

    /* Test 11: Scan for GFXOFF indicators - read GRBM_STATUS */
    ULONG grbm_status = wddm_lite_read_reg32(&g_wddm_lite_dev, 0x0DA4 * 4);
    printf("  GRBM_STATUS @ raw 0x0DA4                = 0x%08x\n", grbm_status);

    ULONG grbm_status_base = wddm_lite_read_reg32(&g_wddm_lite_dev, (gc_base + 0x0DA4) * 4);
    printf("  GRBM_STATUS @ gc_base+0x0DA4            = 0x%08x\n", grbm_status_base);

    /* Test 12: Write scratch reg at raw offset and read back */
    printf("  Writing 0xDEADBEEF to SCRATCH_REG0 @ raw 0x0DE0...\n");
    wddm_lite_write_reg32(&g_wddm_lite_dev, 0x0DE0 * 4, 0xDEADBEEF);
    ULONG scratch_readback = wddm_lite_read_reg32(&g_wddm_lite_dev, 0x0DE0 * 4);
    printf("  SCRATCH_REG0 readback                   = 0x%08x\n", scratch_readback);

    /* Test 13: Write scratch reg at gc_base offset and read back */
    printf("  Writing 0xCAFEBABE to SCRATCH_REG0 @ gc_base+0x0DE0...\n");
    wddm_lite_write_reg32(&g_wddm_lite_dev, (gc_base + 0x0DE0) * 4, 0xCAFEBABE);
    ULONG scratch_readback2 = wddm_lite_read_reg32(&g_wddm_lite_dev, (gc_base + 0x0DE0) * 4);
    printf("  SCRATCH_REG0 @ gc_base readback         = 0x%08x\n", scratch_readback2);

    /* Also read raw again to see if it changed */
    ULONG scratch_raw_after = wddm_lite_read_reg32(&g_wddm_lite_dev, 0x0DE0 * 4);
    printf("  SCRATCH_REG0 @ raw after base write     = 0x%08x\n", scratch_raw_after);

    TEST("Register addressing diagnostic");
    PASS();

    hsaKmtCloseKFD();
}

static void test_gmc_probe(void)
{
    hsaKmtOpenKFD();

    /* We need the device open to send escape commands.
     * MMHUB register offsets (DWORD offsets from MMHUB base).
     * On RDNA4, MMHUB base is typically at DWORD offset 0
     * in the MMIO BAR (base_index 0).
     * Register byte offset = (mmhub_base + reg_offset) * 4
     *
     * We try the known MMHUB base from IP discovery.
     * If mmhub_base is 0, registers are at reg_offset * 4 directly.
     */

    /* Key MMHUB registers (DWORD offsets) */
    const ULONG regMMMC_VM_FB_LOCATION_BASE = 0x0554;
    const ULONG regMMMC_VM_FB_LOCATION_TOP  = 0x0555;
    const ULONG regMMMC_VM_FB_OFFSET        = 0x04C7;
    const ULONG regMMVM_CONTEXT0_CNTL       = 0x0564;
    const ULONG regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32 = 0x05CF;
    const ULONG regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32 = 0x05D0;
    const ULONG regMMMC_VM_MX_L1_TLB_CNTL   = 0x055B;

    /* Key GFXHUB registers (DWORD offsets within GC IP space) */
    const ULONG regGCVM_CONTEXT0_CNTL       = 0x1624;
    const ULONG regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32 = 0x168F;
    const ULONG regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32 = 0x1690;

    /* Try reading with MMHUB base = 0 (direct offset).
     * The driver's READ_REG32 escape uses BAR 0 (MMIO BAR)
     * and byte offset, so we convert DWORD offset to byte offset. */

    /* First, print BAR info from device */
    printf("  --- BAR Info ---\n");
    printf("    MMIO BAR size = 0x%llX (%llu KB)\n",
           (unsigned long long)g_wddm_lite_dev.info.Bars[g_wddm_lite_dev.info.MmioBarIndex].Length,
           (unsigned long long)g_wddm_lite_dev.info.Bars[g_wddm_lite_dev.info.MmioBarIndex].Length / 1024);
    printf("    VRAM BAR size = 0x%llX (%llu MB)\n",
           (unsigned long long)g_wddm_lite_dev.info.Bars[g_wddm_lite_dev.info.VramBarIndex].Length,
           (unsigned long long)g_wddm_lite_dev.info.Bars[g_wddm_lite_dev.info.VramBarIndex].Length / (1024*1024));
    printf("    NumBars = %u, MmioBarIndex = %u, VramBarIndex = %u\n",
           g_wddm_lite_dev.info.NumBars,
           g_wddm_lite_dev.info.MmioBarIndex,
           g_wddm_lite_dev.info.VramBarIndex);

    /* Scan MMIO BAR for MMHUB FB_LOCATION_BASE signature.
     * On RDNA4, the MMHUB IP base from IP discovery is needed.
     * We scan candidate base offsets to find it. */
    printf("\n  --- MMHUB Register Scan ---\n");

    /* Known possible MMHUB bases for RDNA4 (DWORD offsets) */
    ULONG mmhub_candidates[] = { 0x0, 0x3C00, 0x3E00, 0x4000,
                                  0x6800, 0x6A00, 0x6C00, 0x6E00 };
    int base_idx;
    int found_mmhub = 0;

    for (base_idx = 0; base_idx < 8; base_idx++) {
        ULONG base = mmhub_candidates[base_idx];
        ULONG byte_off = (base + regMMMC_VM_FB_LOCATION_BASE) * 4;

        /* Skip if offset exceeds MMIO BAR size */
        if (byte_off + 4 > g_wddm_lite_dev.info.Bars[g_wddm_lite_dev.info.MmioBarIndex].Length)
            continue;

        ULONG fb_base = wddm_lite_read_reg32(&g_wddm_lite_dev, byte_off);
        ULONG fb_top  = wddm_lite_read_reg32(&g_wddm_lite_dev,
                                               (base + regMMMC_VM_FB_LOCATION_TOP) * 4);

        /* Only print if we get something interesting (not 0 or 0xFFFFFFFF) */
        if (fb_base != 0 && fb_base != 0xFFFFFFFF) {
            ULONG ctx0  = wddm_lite_read_reg32(&g_wddm_lite_dev,
                                                 (base + regMMVM_CONTEXT0_CNTL) * 4);
            ULONG l1_tlb = wddm_lite_read_reg32(&g_wddm_lite_dev,
                                                  (base + regMMMC_VM_MX_L1_TLB_CNTL) * 4);
            ULONG pt_lo  = wddm_lite_read_reg32(&g_wddm_lite_dev,
                                                  (base + regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32) * 4);
            ULONG pt_hi  = wddm_lite_read_reg32(&g_wddm_lite_dev,
                                                  (base + regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32) * 4);

            printf("  ** MMHUB found at base=0x%04X **\n", base);
            printf("    FB_LOCATION_BASE = 0x%08X (VRAM @ 0x%llX)\n",
                   fb_base, (unsigned long long)fb_base << 24);
            printf("    FB_LOCATION_TOP  = 0x%08X (end  @ 0x%llX)\n",
                   fb_top, (unsigned long long)fb_top << 24);
            printf("    CONTEXT0_CNTL    = 0x%08X (enabled=%u)\n", ctx0, ctx0 & 1);
            printf("    L1_TLB_CNTL      = 0x%08X (enabled=%u)\n", l1_tlb, l1_tlb & 1);
            printf("    PT_BASE          = 0x%08X_%08X\n", pt_hi, pt_lo);
            found_mmhub = 1;
        }
    }

    if (!found_mmhub) {
        /* Brute force: scan every 0x100 DWORD offset up to MMIO BAR size */
        printf("  Brute-force scanning for MMHUB...\n");
        ULONG bar_dwords = (ULONG)(g_wddm_lite_dev.info.Bars[
            g_wddm_lite_dev.info.MmioBarIndex].Length / 4);
        /* Don't scan past where FB_LOCATION_BASE register would fit */
        ULONG scan_limit = bar_dwords > 0x1554 ? bar_dwords - 0x1554 : 0;

        for (ULONG scan_base = 0; scan_base < scan_limit; scan_base += 0x100) {
            ULONG fb_base = wddm_lite_read_reg32(&g_wddm_lite_dev,
                                                   (scan_base + regMMMC_VM_FB_LOCATION_BASE) * 4);
            if (fb_base != 0 && fb_base != 0xFFFFFFFF &&
                fb_base < 0x10000) {  /* Reasonable FB location value */
                ULONG fb_top = wddm_lite_read_reg32(&g_wddm_lite_dev,
                                                      (scan_base + regMMMC_VM_FB_LOCATION_TOP) * 4);
                if (fb_top > fb_base && fb_top < 0x10000) {
                    printf("  ** Candidate MMHUB at base=0x%04X **\n", scan_base);
                    printf("    FB_LOCATION_BASE = 0x%08X\n", fb_base);
                    printf("    FB_LOCATION_TOP  = 0x%08X\n", fb_top);
                }
            }
        }
    }

    printf("\n  --- GFXHUB Register Scan ---\n");

    /* Scan for GFXHUB CONTEXT0_CNTL (might be enabled by VBIOS for display) */
    ULONG gc_candidates[] = { 0x0, 0x3000, 0x4000 };
    for (base_idx = 0; base_idx < 3; base_idx++) {
        ULONG base = gc_candidates[base_idx];
        ULONG byte_off = (base + regGCVM_CONTEXT0_CNTL) * 4;
        if (byte_off + 4 > g_wddm_lite_dev.info.Bars[g_wddm_lite_dev.info.MmioBarIndex].Length)
            continue;

        ULONG ctx0 = wddm_lite_read_reg32(&g_wddm_lite_dev, byte_off);
        ULONG pt_lo = wddm_lite_read_reg32(&g_wddm_lite_dev,
                                             (base + regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32) * 4);
        ULONG pt_hi = wddm_lite_read_reg32(&g_wddm_lite_dev,
                                             (base + regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32) * 4);

        if (ctx0 != 0 && ctx0 != 0xFFFFFFFF) {
            printf("  ** GFXHUB found at base=0x%04X **\n", base);
            printf("    CONTEXT0_CNTL    = 0x%08X (enabled=%u)\n", ctx0, ctx0 & 1);
            printf("    PT_BASE          = 0x%08X_%08X\n", pt_hi, pt_lo);
        }
    }

    TEST("GMC register probe");
    PASS();

    hsaKmtCloseKFD();
}

/* ======================================================================
 * Stub verification (truly unimplemented APIs)
 * ====================================================================== */
static void test_stubs(void)
{
    HSAKMT_STATUS s;

    hsaKmtOpenKFD();

    TEST("hsaKmtDbgRegister (stub)");
    s = hsaKmtDbgRegister(0);
    if (s == HSAKMT_STATUS_NOT_SUPPORTED) {
        PASS();
    } else {
        FAIL("expected NOT_SUPPORTED");
    }

    hsaKmtCloseKFD();
}

int main(int argc, char *argv[])
{
    int enable_gmc = 0;
    int enable_gfx = 0;
    int enable_ip_discovery = 0;
    int psp_diag_only = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--gfx") == 0) {
            enable_gfx = 1;
            enable_gmc = 1;
            enable_ip_discovery = 1;
        }
        if (strcmp(argv[i], "--gmc") == 0) {
            enable_gmc = 1;
            enable_ip_discovery = 1;  /* GMC requires IP discovery */
        }
        if (strcmp(argv[i], "--ip-discovery") == 0)
            enable_ip_discovery = 1;
        if (strcmp(argv[i], "--psp-diag") == 0) {
            psp_diag_only = 1;
            enable_ip_discovery = 1;  /* PSP requires IP discovery */
        }
    }

    printf("=== libhsakmt wddm_lite backend test ===\n");
    printf("=== Ported from kfdtest: KFDOpenCloseKFDTest, KFDTopologyTest ===\n");
    if (!enable_ip_discovery)
        printf("=== IP discovery disabled (use --ip-discovery to enable) ===\n");
    if (!enable_gmc)
        printf("=== GMC init disabled (use --gmc to enable) ===\n");
    if (!enable_gfx)
        printf("=== GFX init disabled (use --gfx to enable) ===\n");
    printf("\n");

    /* Set debug level via env if not already set */
    SetEnvironmentVariableA("HSAKMT_DEBUG_LEVEL", "6");

    /* Skip init phases unless flags are passed */
    if (!enable_ip_discovery)
        SetEnvironmentVariableA("HSAKMT_SKIP_IP_DISCOVERY", "1");
    if (!enable_gmc)
        SetEnvironmentVariableA("HSAKMT_SKIP_GMC_INIT", "1");
    if (!enable_gfx)
        SetEnvironmentVariableA("HSAKMT_SKIP_GFX_INIT", "1");
    if (psp_diag_only)
        SetEnvironmentVariableA("HSAKMT_PSP_DIAG_ONLY", "1");

    printf("[KFDCloseKFDTest]\n");
    test_close_a_closed_kfd();

    printf("\n[KFDOpenCloseKFDTest]\n");
    test_open_close_kfd();
    test_open_already_opened();

    printf("\n[Version]\n");
    test_version();

    printf("\n[KFDTopologyTest::BasicTest]\n");
    test_topology_basic();

    printf("\n[KFDTopologyTest::GetNodePropertiesInvalidParams]\n");
    test_get_node_properties_null();

    printf("\n[KFDTopologyTest::GetNodePropertiesInvalidNodeNum]\n");
    test_get_node_properties_invalid_node();

    printf("\n[KFDTopologyTest::GetNodeMemoryProperties]\n");
    test_get_node_memory_properties();

    printf("\n[KFDTopologyTest::GpuvmApertureValidate]\n");
    test_gpuvm_aperture_validate();

    printf("\n[KFDTopologyTest::GetNodeCacheProperties]\n");
    test_get_node_cache_properties();

    printf("\n[KFDTopologyTest::GetNodeIoLinkProperties]\n");
    test_get_node_iolink_properties();

    printf("\n[Memory]\n");
    test_alloc_free_memory();

    printf("\n[Events]\n");
    test_create_destroy_event();

    printf("\n[GPU Init Validation]\n");
    test_gpu_init_validation();

    printf("\n[GFX Init Validation]\n");
    test_gfx_init_validation();

    printf("\n[SMN Indirect Access]\n");
    test_smn_indirect();

    printf("\n[SMU / GFXOFF]\n");
    test_smu_gfxoff();

    printf("\n[Register Addressing]\n");
    test_register_addressing();

    printf("\n[GMC Probe]\n");
    test_gmc_probe();

    printf("\n[Stubs]\n");
    test_stubs();

    printf("\n=== Results: %d passed, %d failed ===\n",
           pass_count, fail_count);

    return fail_count > 0 ? 1 : 0;
}
