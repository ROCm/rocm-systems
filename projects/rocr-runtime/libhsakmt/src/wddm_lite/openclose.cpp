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

#include "wddm_lite_internal.h"
#include "wddm_lite_device.h"
#include "gpu_init.h"

#include <stdlib.h>
#include <string.h>

extern struct WddmLiteDevice g_wddm_lite_dev;
extern void wddm_lite_init_mutex(void);

/*
 * Derive the GFX version from the device ID.
 * Returns gfxv in packed format: (major << 16) | (minor << 8) | stepping.
 */
static UINT device_id_to_gfx_version(USHORT device_id)
{
    switch (device_id) {
    case 0x7551: /* RX 9070 XT */
    case 0x7550: /* RX 9070 */
        return 0x0C0001; /* GFX12.0.1 */
    default:
        return 0;
    }
}

/*
 * Try to find our custom WDDM adapter by sending GET_INFO escape
 * to each enumerated adapter. Adapters running the Adrenaline driver
 * will reject the escape (STATUS_INVALID_PARAMETER or similar),
 * so this probe is safe and doesn't interfere with other drivers.
 */
int wddm_lite_open(struct WddmLiteDevice *dev)
{
    D3DKMT_ADAPTERINFO adapters[16];
    D3DKMT_ENUMADAPTERS2 enum_args;
    NTSTATUS status;

    if (dev->is_open)
        return 0;

    memset(dev, 0, sizeof(*dev));

    /* Load D3DKMT functions */
    if (!wddm_lite_d3dkmt_init(&dev->d3d)) {
        pr_err("wddm_lite: failed to load gdi32.dll D3DKMT functions\n");
        return -1;
    }

    /* Enumerate adapters */
    memset(&enum_args, 0, sizeof(enum_args));
    memset(adapters, 0, sizeof(adapters));
    enum_args.NumAdapters = _countof(adapters);
    enum_args.pAdapters = adapters;

    status = dev->d3d.pfnEnumAdapters2(&enum_args);
    if (status < 0) {
        pr_err("wddm_lite: D3DKMTEnumAdapters2 failed: 0x%08lx\n",
               (unsigned long)status);
        goto fail_d3d;
    }

    pr_info("wddm_lite: found %u adapters\n", enum_args.NumAdapters);

    /* Probe each adapter with our custom GET_INFO escape */
    for (ULONG i = 0; i < enum_args.NumAdapters; i++) {
        AMDGPU_ESCAPE_GET_INFO_DATA info;
        memset(&info, 0, sizeof(info));
        info.Header.Command = AMDGPU_ESCAPE_GET_INFO;
        info.Header.Size = sizeof(info);

        D3DKMT_ESCAPE esc;
        memset(&esc, 0, sizeof(esc));
        esc.hAdapter = adapters[i].hAdapter;
        esc.Type = D3DKMT_ESCAPE_DRIVERPRIVATE;
        esc.Flags.Value = 0;
        esc.pPrivateDriverData = &info;
        esc.PrivateDriverDataSize = sizeof(info);

        status = dev->d3d.pfnEscape(&esc);
        if (status >= 0 && info.Header.Status == STATUS_SUCCESS &&
            info.VendorId == 0x1002) {
            /* Found our custom WDDM driver on an AMD GPU */
            dev->adapter_handle = adapters[i].hAdapter;
            dev->info = info;
            dev->device_id = info.DeviceId;
            dev->vram_size = info.VramSizeBytes;
            dev->gfx_version = device_id_to_gfx_version(info.DeviceId);

            pr_info("wddm_lite: adapter %u: device %04x:%04x, "
                    "VRAM %llu MB, gfxv 0x%06x\n",
                    i, info.VendorId, info.DeviceId,
                    (unsigned long long)info.VramSizeBytes / (1024 * 1024),
                    dev->gfx_version);

            /* Close handles for adapters we don't need */
            for (ULONG j = 0; j < enum_args.NumAdapters; j++) {
                if (j != i) {
                    D3DKMT_CLOSEADAPTER ca;
                    ca.hAdapter = adapters[j].hAdapter;
                    dev->d3d.pfnCloseAdapter(&ca);
                }
            }

            dev->is_open = true;
            return 0;
        }

        pr_info("wddm_lite: adapter %u: not our driver (status 0x%08lx)\n",
                i, (unsigned long)status);
    }

    pr_err("wddm_lite: no adapter with custom WDDM driver found\n");
    pr_err("wddm_lite: (is amdgpu_wddm.sys loaded? "
           "Adrenaline requires the DXG backend instead)\n");

    /* Close all adapter handles */
    for (ULONG i = 0; i < enum_args.NumAdapters; i++) {
        D3DKMT_CLOSEADAPTER ca;
        ca.hAdapter = adapters[i].hAdapter;
        dev->d3d.pfnCloseAdapter(&ca);
    }

fail_d3d:
    wddm_lite_d3dkmt_fini(&dev->d3d);
    return -1;
}

void wddm_lite_close(struct WddmLiteDevice *dev)
{
    if (!dev->is_open)
        return;

    if (dev->adapter_handle) {
        D3DKMT_CLOSEADAPTER ca;
        ca.hAdapter = dev->adapter_handle;
        dev->d3d.pfnCloseAdapter(&ca);
        dev->adapter_handle = 0;
    }

    wddm_lite_d3dkmt_fini(&dev->d3d);
    dev->is_open = false;
}

/*
 * Initialize debug level from environment.
 */
static void init_debug_level(void)
{
    char buf[32];
    DWORD len;

    hsakmt_debug_level = HSAKMT_DEBUG_LEVEL_DEFAULT;
    len = GetEnvironmentVariableA("HSAKMT_DEBUG_LEVEL", buf, sizeof(buf));
    if (len > 0 && len < sizeof(buf)) {
        int level = atoi(buf);
        if (level >= HSAKMT_DEBUG_LEVEL_ERR &&
            level <= HSAKMT_DEBUG_LEVEL_DEBUG)
            hsakmt_debug_level = level;
    }
}

static void init_page_size(void)
{
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    hsakmt_page_size = (int)si.dwPageSize;

    /* Compute page shift: log2(page_size) */
    hsakmt_page_shift = 0;
    int ps = hsakmt_page_size;
    while (ps > 1) {
        ps >>= 1;
        hsakmt_page_shift++;
    }
}

HSAKMT_STATUS HSAKMTAPI hsaKmtOpenKFD(void)
{
    HSAKMT_STATUS result;

    wddm_lite_init_mutex();
    EnterCriticalSection(&hsakmt_mutex);

    if (hsakmt_kfd_open_count == 0) {
        init_debug_level();
        init_page_size();

        if (wddm_lite_open(&g_wddm_lite_dev) != 0) {
            result = HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED;
            goto open_failed;
        }

        /* Run IP discovery to find all IP block base addresses */
        {
            char skip_ip[32] = {};
            GetEnvironmentVariableA("HSAKMT_SKIP_IP_DISCOVERY", skip_ip, sizeof(skip_ip));
            if (skip_ip[0] == '1') {
                pr_info("wddm_lite: IP discovery skipped (HSAKMT_SKIP_IP_DISCOVERY=1)\n");
            } else if (gpu_ip_discovery(&g_wddm_lite_dev) != 0) {
                pr_warn("wddm_lite: IP discovery failed (continuing without)\n");
            }
        }

        /* Load PSP SOS firmware if not already alive */
        if (g_wddm_lite_dev.hw.ip_discovery_done &&
            !g_wddm_lite_dev.hw.psp_sos_alive) {
            char fw_path[256] = {};
            char skip_psp[32] = {};
            GetEnvironmentVariableA("HSAKMT_SKIP_PSP_INIT", skip_psp, sizeof(skip_psp));
            if (skip_psp[0] == '1') {
                pr_info("wddm_lite: PSP init skipped (HSAKMT_SKIP_PSP_INIT=1)\n");
            } else {
                /* Look for firmware path in environment, or use default */
                DWORD len = GetEnvironmentVariableA("HSAKMT_PSP_FW_PATH",
                                                     fw_path, sizeof(fw_path));
                if (len == 0 || len >= sizeof(fw_path)) {
                    /* Default: look in current directory */
                    strncpy(fw_path, "psp_14_0_3_sos.bin", sizeof(fw_path) - 1);
                }
                if (gpu_psp_load_sos(&g_wddm_lite_dev, fw_path) != 0) {
                    pr_warn("wddm_lite: PSP SOS load failed (continuing without)\n");
                }
            }
        }

        /*
         * Try GFXOFF disable with VBIOS SMU first (before firmware loading).
         * On a VBIOS-POST'd passthrough GPU, the VBIOS loads a minimal SMU
         * that may support DisallowGfxOff. This avoids the need for
         * full firmware loading just to disable GFXOFF.
         */
        if (g_wddm_lite_dev.hw.ip_discovery_done &&
            !g_wddm_lite_dev.hw.gfxoff_disabled) {
            pr_info("wddm_lite: trying GFXOFF disable with VBIOS SMU...\n");
            if (gpu_disable_gfxoff(&g_wddm_lite_dev) == 0) {
                pr_info("wddm_lite: GFXOFF disabled via VBIOS SMU\n");
            } else {
                pr_warn("wddm_lite: VBIOS SMU GFXOFF disable failed\n");
            }
        }

        /* Load GPU firmware via PSP ring if needed */
        if (g_wddm_lite_dev.hw.psp_sos_alive) {
            char skip_fw[32] = {};
            GetEnvironmentVariableA("HSAKMT_SKIP_FW_LOAD", skip_fw, sizeof(skip_fw));
            if (skip_fw[0] == '1') {
                pr_info("wddm_lite: firmware loading skipped (HSAKMT_SKIP_FW_LOAD=1)\n");
            } else {
                char fw_dir[256] = ".";
                GetEnvironmentVariableA("HSAKMT_FW_DIR", fw_dir, sizeof(fw_dir));
                if (gpu_psp_load_all_fw(&g_wddm_lite_dev, fw_dir) != 0) {
                    pr_warn("wddm_lite: firmware loading failed (continuing without)\n");
                } else {
                    /* After successful firmware loading, try SMU init + GFXOFF */
                    if (gpu_smu_enable_features(&g_wddm_lite_dev) == 0) {
                        pr_info("wddm_lite: SMU features enabled\n");
                    }
                    if (!g_wddm_lite_dev.hw.gfxoff_disabled) {
                        if (gpu_disable_gfxoff(&g_wddm_lite_dev) != 0) {
                            pr_warn("wddm_lite: GFXOFF disable failed after firmware load\n");
                        }
                    }
                }
            }
        }

        /* Initialize GMC (GART, system aperture, TLB, L2 cache) */
        {
            char skip_gmc[32] = {};
            GetEnvironmentVariableA("HSAKMT_SKIP_GMC_INIT", skip_gmc, sizeof(skip_gmc));
            if (skip_gmc[0] == '1') {
                pr_info("wddm_lite: GMC init skipped (HSAKMT_SKIP_GMC_INIT=1)\n");
            } else if (g_wddm_lite_dev.hw.ip_discovery_done) {
                if (gpu_gmc_init(&g_wddm_lite_dev) != 0) {
                    pr_warn("wddm_lite: GMC initialization failed (continuing without)\n");
                }
            }
        }

        /* Initialize GFX engine (SH_MEM, MEC, compute queues) */
        {
            char skip_gfx[32] = {};
            GetEnvironmentVariableA("HSAKMT_SKIP_GFX_INIT", skip_gfx, sizeof(skip_gfx));
            if (skip_gfx[0] == '1') {
                pr_info("wddm_lite: GFX init skipped (HSAKMT_SKIP_GFX_INIT=1)\n");
            } else if (g_wddm_lite_dev.hw.ip_discovery_done) {
                if (gpu_gfx_init(&g_wddm_lite_dev) != 0) {
                    pr_warn("wddm_lite: GFX initialization failed (continuing without)\n");
                }
            }
        }

        result = hsakmt_init_kfd_version();
        if (result != HSAKMT_STATUS_SUCCESS) {
            wddm_lite_close(&g_wddm_lite_dev);
            goto open_failed;
        }

        hsakmt_is_dgpu = true;
        hsakmt_kfd_open_count = 1;
    } else {
        hsakmt_kfd_open_count++;
        result = HSAKMT_STATUS_KERNEL_ALREADY_OPENED;
    }

    LeaveCriticalSection(&hsakmt_mutex);
    return result;

open_failed:
    LeaveCriticalSection(&hsakmt_mutex);
    return result;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtCloseKFD(void)
{
    HSAKMT_STATUS result;

    wddm_lite_init_mutex();
    EnterCriticalSection(&hsakmt_mutex);

    if (hsakmt_kfd_open_count > 0) {
        if (--hsakmt_kfd_open_count == 0) {
            gpu_gfx_cleanup(&g_wddm_lite_dev);
            gpu_gmc_cleanup(&g_wddm_lite_dev);
            gpu_enable_gfxoff(&g_wddm_lite_dev);
            wddm_lite_close(&g_wddm_lite_dev);
        }
        result = HSAKMT_STATUS_SUCCESS;
    } else {
        result = HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED;
    }

    LeaveCriticalSection(&hsakmt_mutex);
    return result;
}
