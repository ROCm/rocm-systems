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
            wddm_lite_close(&g_wddm_lite_dev);
        }
        result = HSAKMT_STATUS_SUCCESS;
    } else {
        result = HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED;
    }

    LeaveCriticalSection(&hsakmt_mutex);
    return result;
}
