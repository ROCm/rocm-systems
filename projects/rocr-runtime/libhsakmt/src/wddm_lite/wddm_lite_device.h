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

#ifndef WDDM_LITE_DEVICE_H_INCLUDED
#define WDDM_LITE_DEVICE_H_INCLUDED

#include "wddm_lite_d3dkmt.h"
#include "wddm_lite_escape.h"
#include "gpu_init.h"

struct WddmLiteDevice {
    /* D3DKMT state */
    struct WddmLiteD3dkmt   d3d;
    D3DKMT_HANDLE           adapter_handle;

    /* Device info from GET_INFO escape */
    AMDGPU_ESCAPE_GET_INFO_DATA info;

    /* Derived fields */
    USHORT      device_id;
    ULONGLONG   vram_size;
    UINT        gfx_version;    /* (major << 16) | (minor << 8) | stepping */

    /* Hardware state (IP discovery + GMC) */
    struct GpuHwState       hw;

    bool        is_open;
};

/* Open/close the wddm_lite device */
int wddm_lite_open(struct WddmLiteDevice *dev);
void wddm_lite_close(struct WddmLiteDevice *dev);

/*
 * Send an escape command to the custom WDDM driver.
 * Returns 0 on success, -1 on failure.
 */
static inline int wddm_lite_escape(struct WddmLiteDevice *dev,
                                   void *data, UINT size)
{
    D3DKMT_ESCAPE esc;
    NTSTATUS status;

    memset(&esc, 0, sizeof(esc));
    esc.hAdapter = dev->adapter_handle;
    esc.hDevice = 0;
    esc.Type = D3DKMT_ESCAPE_DRIVERPRIVATE;
    esc.Flags.Value = 0;
    esc.pPrivateDriverData = data;
    esc.PrivateDriverDataSize = size;
    esc.hContext = 0;

    status = dev->d3d.pfnEscape(&esc);
    return (status >= 0) ? 0 : -1;  /* NTSTATUS >= 0 means success */
}

/*
 * Read a 32-bit MMIO register via escape.
 */
static inline ULONG wddm_lite_read_reg32(struct WddmLiteDevice *dev,
                                          ULONG offset)
{
    AMDGPU_ESCAPE_REG32_DATA reg;
    memset(&reg, 0, sizeof(reg));
    reg.Header.Command = AMDGPU_ESCAPE_READ_REG32;
    reg.Header.Size = sizeof(reg);
    reg.BarIndex = 0;  /* 0 = MMIO BAR */
    reg.Offset = offset;

    if (wddm_lite_escape(dev, &reg, sizeof(reg)) != 0)
        return 0xFFFFFFFF;

    return reg.Value;
}

/*
 * Write a 32-bit MMIO register via escape.
 */
static inline void wddm_lite_write_reg32(struct WddmLiteDevice *dev,
                                          ULONG offset, ULONG value)
{
    AMDGPU_ESCAPE_REG32_DATA reg;
    memset(&reg, 0, sizeof(reg));
    reg.Header.Command = AMDGPU_ESCAPE_WRITE_REG32;
    reg.Header.Size = sizeof(reg);
    reg.BarIndex = 0;
    reg.Offset = offset;
    reg.Value = value;

    wddm_lite_escape(dev, &reg, sizeof(reg));
}

#endif /* WDDM_LITE_DEVICE_H_INCLUDED */
