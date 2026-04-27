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

/* Trim trailing whitespace in-place */
static void trim_trailing(char *s)
{
    size_t len = strlen(s);
    while (len > 0 && (s[len - 1] == ' ' || s[len - 1] == '\t' ||
                       s[len - 1] == '\r' || s[len - 1] == '\n'))
        s[--len] = '\0';
}

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

    /* Unmap doorbell BAR if mapped */
    if (dev->doorbell_base && dev->doorbell_mapping_handle) {
        AMDGPU_ESCAPE_MAP_BAR_DATA unmap;
        memset(&unmap, 0, sizeof(unmap));
        unmap.Header.Command = AMDGPU_ESCAPE_UNMAP_BAR;
        unmap.Header.Size = sizeof(unmap);
        unmap.MappedAddress = dev->doorbell_base;
        unmap.MappingHandle = dev->doorbell_mapping_handle;
        wddm_lite_escape(dev, &unmap, sizeof(unmap));
        dev->doorbell_base = NULL;
        dev->doorbell_mapping_handle = NULL;
    }

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

        /* Reset compute state to free leaked resources from previous sessions.
         * Events/allocs/queues from crashed processes accumulate in the driver
         * since compute state is per-adapter, not per-process. */
        {
            AMDGPU_ESCAPE_HEADER reset_hdr;
            memset(&reset_hdr, 0, sizeof(reset_hdr));
            reset_hdr.Command = AMDGPU_ESCAPE_RESET_COMPUTE;
            reset_hdr.Size = sizeof(reset_hdr);
            if (wddm_lite_escape(&g_wddm_lite_dev, &reset_hdr, sizeof(reset_hdr)) == 0) {
                pr_info("wddm_lite: compute state reset OK\n");
            } else {
                pr_warn("wddm_lite: compute state reset failed (old driver?)\n");
            }
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

        /*
         * GPU Init Sequence — two paths:
         *
         * Path A (VBIOS state — default after fresh VFIO FLR):
         *   VBIOS already loaded SOS, firmware, enabled SMU features.
         *   Just need: GMC → DisallowGfxOff → GFX init
         *   Set HSAKMT_SKIP_MODE1_RESET=1 to use this path.
         *
         * Path B (full reinit — after mode1 reset):
         *   1. Mode1 reset → 2. SOS → 3. GMC → 4. IH →
         *   5. Firmware + AUTOLOAD → 6. SMU init → 7. GFXOFF → 8. GFX
         *
         * After VFIO FLR, VBIOS state survives: SOS alive, SMU features
         * enabled, GC firmware loaded. Previous sessions accidentally
         * triggered mode1_reset via debug mailbox msg=2 (thought to be
         * GetSmuVersion), which killed this state.
         */
        if (g_wddm_lite_dev.hw.ip_discovery_done) {
            /* === EARLY DIAGNOSTIC: Read bootload status before ANY PSP ops ===
             * This tells us if VBIOS AUTOLOAD completed before we touch anything.
             * PSP ring creation (psp_ring_init) destroys VBIOS ring and may
             * clear bootload_status. Reading it here captures the true VBIOS state. */
            if (g_wddm_lite_dev.hw.ip.gc_base1 != 0 &&
                g_wddm_lite_dev.hw.ip.nbio_base1 != 0) {
                ULONG early_bl = gpu_smn_rreg(&g_wddm_lite_dev,
                    g_wddm_lite_dev.hw.ip.gc_base1 + 0x4e7c);
                ULONG early_rlc_cntl = gpu_smn_rreg(&g_wddm_lite_dev,
                    g_wddm_lite_dev.hw.ip.gc_base1 + 0x4c00);
                pr_info("wddm_lite: EARLY DIAG: RLC_BOOTLOAD_STATUS = 0x%08x "
                        "(bit31=%d) RLC_CNTL = 0x%08x\n",
                        early_bl, (early_bl >> 31) & 1, early_rlc_cntl);
                g_wddm_lite_dev.hw.gfx.rlc_bootload_status = early_bl;

                /* If VBIOS AUTOLOAD already completed, offer passthrough mode */
                if (early_bl & 0x80000000) {
                    pr_info("wddm_lite: VBIOS AUTOLOAD complete! "
                            "Firmware is decrypted and mapped.\n");
                }

                /* Also read GFXHUB state before we touch it.
                 * GFXHUB regs are at gc_base (BASE_IDX=0), not gc_base1. */
                ULONG early_gfxhub_cntl = gpu_smn_rreg(&g_wddm_lite_dev,
                    g_wddm_lite_dev.hw.ip.gc_base + 0x1624);
                ULONG early_pt_lo = gpu_smn_rreg(&g_wddm_lite_dev,
                    g_wddm_lite_dev.hw.ip.gc_base + 0x168F);
                ULONG early_pt_hi = gpu_smn_rreg(&g_wddm_lite_dev,
                    g_wddm_lite_dev.hw.ip.gc_base + 0x1690);
                pr_info("wddm_lite: EARLY DIAG: GFXHUB CONTEXT0_CNTL = 0x%08x "
                        "(enabled=%d) PT_BASE = 0x%08x_%08x\n",
                        early_gfxhub_cntl, early_gfxhub_cntl & 1,
                        early_pt_hi, early_pt_lo);
            }

            char skip_psp[32] = {};
            GetEnvironmentVariableA("HSAKMT_SKIP_PSP_INIT", skip_psp, sizeof(skip_psp));

            if (skip_psp[0] == '1') {
                pr_info("wddm_lite: PSP init skipped (HSAKMT_SKIP_PSP_INIT=1)\n");
            } else {
                /* Check SOS status to determine GPU state */
                ULONG sos_status = 0;
                BOOLEAN did_mode1_reset = FALSE;
                BOOLEAN sos_was_alive_at_boot = FALSE;
                if (g_wddm_lite_dev.hw.ip.mp0_base != 0 &&
                    g_wddm_lite_dev.hw.ip.nbio_base1 != 0) {
                    sos_status = gpu_smn_rreg(&g_wddm_lite_dev,
                        g_wddm_lite_dev.hw.ip.mp0_base + 0x0091);
                    sos_was_alive_at_boot = (sos_status != 0);
                }

                /* Step 1: Mode1 reset (optional) */
                {
                    char skip_reset[32] = {};
                    GetEnvironmentVariableA("HSAKMT_SKIP_MODE1_RESET",
                        skip_reset, sizeof(skip_reset));
                    if (skip_reset[0] == '1') {
                        pr_info("wddm_lite: mode1 reset skipped "
                                "(HSAKMT_SKIP_MODE1_RESET=1)\n");
                        /* If SOS is alive, mark it so we skip SOS loading */
                        if (sos_status != 0) {
                            pr_info("wddm_lite: SOS already alive (0x%08x), "
                                    "using VBIOS state\n", sos_status);
                            g_wddm_lite_dev.hw.psp_sos_alive = TRUE;
                        }
                    } else if (sos_status != 0) {
                        pr_info("wddm_lite: SOS alive (0x%08x), "
                                "performing mode1 reset...\n", sos_status);
                        gpu_smu_mode1_reset(&g_wddm_lite_dev);
                        did_mode1_reset = TRUE;
                        /* Mode1 clears AUTOLOAD — update saved state so
                         * firmware loading isn't skipped later. */
                        g_wddm_lite_dev.hw.gfx.rlc_bootload_status = 0;
                    } else {
                        pr_info("wddm_lite: SOS not alive, "
                                "skipping mode1 reset\n");
                    }
                }

                /* Step 2: Load SOS if needed */
                if (!g_wddm_lite_dev.hw.psp_sos_alive) {
                    char fw_path[256] = {};
                    DWORD len = GetEnvironmentVariableA("HSAKMT_PSP_FW_PATH",
                        fw_path, sizeof(fw_path));
                    if (len > 0 && len < sizeof(fw_path))
                        trim_trailing(fw_path);
                    if (len == 0 || len >= sizeof(fw_path)) {
                        char fw_dir_buf[256] = {};
                        DWORD flen = GetEnvironmentVariableA("HSAKMT_FW_DIR",
                            fw_dir_buf, sizeof(fw_dir_buf));
                        if (flen > 0 && flen < sizeof(fw_dir_buf)) {
                            trim_trailing(fw_dir_buf);
                            snprintf(fw_path, sizeof(fw_path),
                                     "%s\\psp_14_0_3_sos.bin", fw_dir_buf);
                        } else {
                            strncpy(fw_path, "psp_14_0_3_sos.bin",
                                    sizeof(fw_path) - 1);
                        }
                    }
                    if (gpu_psp_load_sos(&g_wddm_lite_dev, fw_path) != 0) {
                        pr_warn("wddm_lite: PSP SOS load failed\n");
                    }
                }

                /* Step 2b: Mode1 reset (only if VBIOS state was present).
                 * Mode1 clears VBIOS hardware state that conflicts with fresh
                 * firmware (EnableAllSmuFeatures hangs without it).
                 * Only needed when SOS was alive at boot (VBIOS ran). On cold
                 * boot (SOS not alive), VFIO FLR already provides clean state
                 * and mode1 after fresh SOS load kills PSP without recovery. */
                if (g_wddm_lite_dev.hw.psp_sos_alive && !did_mode1_reset
                    && sos_was_alive_at_boot) {
                    char skip_reset[32] = {};
                    GetEnvironmentVariableA("HSAKMT_SKIP_MODE1_RESET",
                        skip_reset, sizeof(skip_reset));
                    if (skip_reset[0] != '1') {
                        pr_info("wddm_lite: SOS now alive — performing Mode1 "
                                "hardware reset (matching amdgpu)\n");
                        gpu_smu_mode1_reset(&g_wddm_lite_dev);
                        did_mode1_reset = TRUE;
                        /* Mode1 clears AUTOLOAD — update saved state so
                         * firmware loading isn't skipped later. */
                        g_wddm_lite_dev.hw.gfx.rlc_bootload_status = 0;

                        /* After Mode1 reset, SOS was killed. Reload it.
                         * Mode1 → bootloader runs → we load SOS again. */
                        pr_info("wddm_lite: reloading SOS after Mode1 reset\n");
                        g_wddm_lite_dev.hw.psp_sos_alive = FALSE;
                        {
                            char fw_dir_buf[256] = {};
                            GetEnvironmentVariableA("HSAKMT_FW_DIR",
                                fw_dir_buf, sizeof(fw_dir_buf));
                            char sos_path[256];
                            snprintf(sos_path, sizeof(sos_path),
                                     "%s\\psp_14_0_3_sos.bin",
                                     fw_dir_buf[0] ? fw_dir_buf : ".");
                            if (gpu_psp_load_sos(&g_wddm_lite_dev, sos_path) != 0)
                                pr_warn("wddm_lite: SOS reload after Mode1 failed\n");
                        }
                    }
                }

                /* Step 3: GMC init */
                {
                    char skip_gmc[32] = {};
                    GetEnvironmentVariableA("HSAKMT_SKIP_GMC_INIT",
                        skip_gmc, sizeof(skip_gmc));
                    if (skip_gmc[0] == '1') {
                        pr_info("wddm_lite: GMC init skipped\n");
                    } else {
                        if (gpu_gmc_init(&g_wddm_lite_dev) != 0) {
                            pr_warn("wddm_lite: GMC init failed "
                                    "(continuing without)\n");
                        }
                    }
                }

                /* Step 4: IH init */
                {
                    if (gpu_ih_init(&g_wddm_lite_dev) != 0) {
                        pr_warn("wddm_lite: IH init failed "
                                "(continuing without)\n");
                    }
                }

                /* Step 5: Firmware staging only — DO NOT trigger AUTOLOAD yet.
                 * AUTOLOAD requires GFX power (EnableAllSmuFeatures in Step 6)
                 * to complete (bit 31). Linux order: PSP loads firmware →
                 * SMU hw_init enables GFX features → GFX hw_init triggers
                 * AUTOLOAD_RLC. We follow the same order here. */
                BOOLEAN fw_staged = FALSE;
                char skip_fw[32] = {};
                GetEnvironmentVariableA("HSAKMT_SKIP_FW_LOAD",
                    skip_fw, sizeof(skip_fw));

                /* Auto-skip firmware loading if VBIOS AUTOLOAD already completed.
                 * In this state PSP has already loaded/decrypted all firmware and
                 * reloading via PSP ring hangs (PSP rejects duplicate loads).
                 * We still call EnableAllSmuFeatures in case GFX features need
                 * re-enabling after power state changes. */
                BOOLEAN vbios_autoload_done =
                    (g_wddm_lite_dev.hw.gfx.rlc_bootload_status & 0x80000000) != 0;

                if (g_wddm_lite_dev.hw.psp_sos_alive) {
                    if (skip_fw[0] == '1') {
                        pr_info("wddm_lite: firmware loading skipped "
                                "(HSAKMT_SKIP_FW_LOAD=1)\n");
                    } else if (vbios_autoload_done) {
                        pr_info("wddm_lite: firmware loading skipped — "
                                "VBIOS AUTOLOAD already complete (0x%08x)\n",
                                g_wddm_lite_dev.hw.gfx.rlc_bootload_status);
                        /* Firmware already loaded by VBIOS; call EnableAllSmuFeatures
                         * to ensure GFX power is up before gfx_init. */
                        gpu_smu_enable_features(&g_wddm_lite_dev);
                    } else {
                        char fw_dir[256] = ".";
                        GetEnvironmentVariableA("HSAKMT_FW_DIR",
                            fw_dir, sizeof(fw_dir));
                        trim_trailing(fw_dir);
                        if (gpu_psp_load_all_fw(&g_wddm_lite_dev,
                                                fw_dir) != 0) {
                            pr_warn("wddm_lite: firmware staging failed\n");
                        } else {
                            fw_staged = TRUE;
                            /* SMU init immediately after firmware load.
                             * AUTOLOAD_RLC is now sent inside gpu_psp_load_all_fw
                             * as the last PSP command (matching amdgpu). */
                            pr_info("wddm_lite: SMU init after FW+AUTOLOAD\n");
                            gpu_smu_enable_features(&g_wddm_lite_dev);
                        }
                    }
                }

                /* AUTOLOAD_RLC is now sent inside gpu_psp_load_all_fw as the
                 * last PSP ring command (matching amdgpu's psp_load_non_psp_fw). */

                /* Step 7: SMU features — already called immediately after FW load
                 * or in the vbios_autoload_done branch above. Skip here. */
                if (skip_fw[0] == '1') {
                    pr_info("wddm_lite: SMU enable_features skipped (SKIP_FW_LOAD)\n");
                }

                /* Step 7: DisallowGfxOff */
                if (!g_wddm_lite_dev.hw.gfxoff_disabled) {
                    char skip_gfxoff[32] = {};
                    GetEnvironmentVariableA("HSAKMT_SKIP_GFXOFF",
                        skip_gfxoff, sizeof(skip_gfxoff));
                    if (skip_gfxoff[0] == '1') {
                        pr_info("wddm_lite: GFXOFF disable skipped\n");
                        g_wddm_lite_dev.hw.gfxoff_disabled = TRUE;
                    } else if (gpu_disable_gfxoff(&g_wddm_lite_dev) != 0) {
                        pr_warn("wddm_lite: GFXOFF disable failed\n");
                    }
                }
            }
        }

        /* Step 8: Initialize GFX engine (SH_MEM, MEC, compute queues) */
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
