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
 * Minimal D3DKMT type definitions and dynamic function loader.
 *
 * We define only the structures and functions we need rather than
 * pulling in the full d3dkmthk.h header chain from WDK. This makes
 * the build self-contained — no WDK install required for compilation.
 *
 * Functions are loaded dynamically from gdi32.dll at runtime.
 */

#ifndef WDDM_LITE_D3DKMT_H_INCLUDED
#define WDDM_LITE_D3DKMT_H_INCLUDED

#include <windows.h>

/* D3DKMT_HANDLE is just a UINT */
typedef UINT D3DKMT_HANDLE;

/* Adapter info returned by D3DKMTEnumAdapters2 */
typedef struct _D3DKMT_ADAPTERINFO {
    D3DKMT_HANDLE   hAdapter;
    LUID            AdapterLuid;
    ULONG           NumOfSources;
    BOOL            bPrecisePresentRegionsPreferred;
} D3DKMT_ADAPTERINFO;

/* Input/output for D3DKMTEnumAdapters2 */
typedef struct _D3DKMT_ENUMADAPTERS2 {
    ULONG               NumAdapters;
    D3DKMT_ADAPTERINFO *pAdapters;
} D3DKMT_ENUMADAPTERS2;

/* Escape type enum — we only need DRIVERPRIVATE */
typedef enum _D3DKMT_ESCAPETYPE {
    D3DKMT_ESCAPE_DRIVERPRIVATE     = 0,
    D3DKMT_ESCAPE_VIDMM            = 1,
    D3DKMT_ESCAPE_TDRDBGCTRL        = 2,
    D3DKMT_ESCAPE_VIDSCH            = 3,
} D3DKMT_ESCAPETYPE;

/* Escape flags */
typedef union _D3DDDI_ESCAPEFLAGS {
    struct {
        UINT HardwareAccess             : 1;
        UINT DeviceStatusQuery          : 1;
        UINT ChangeFrameLatency         : 1;
        UINT NoAdapterSynchronization   : 1;
        UINT Reserved                   : 24;
        UINT DriverKnownEscape          : 1;
        UINT DriverCommonEscape         : 1;
        UINT Reserved2                  : 2;
    };
    UINT Value;
} D3DDDI_ESCAPEFLAGS;

/* Input for D3DKMTEscape */
typedef struct _D3DKMT_ESCAPE {
    D3DKMT_HANDLE       hAdapter;
    D3DKMT_HANDLE       hDevice;
    D3DKMT_ESCAPETYPE   Type;
    D3DDDI_ESCAPEFLAGS  Flags;
    VOID               *pPrivateDriverData;
    UINT                PrivateDriverDataSize;
    D3DKMT_HANDLE       hContext;
} D3DKMT_ESCAPE;

/* Input for D3DKMTCloseAdapter */
typedef struct _D3DKMT_CLOSEADAPTER {
    D3DKMT_HANDLE hAdapter;
} D3DKMT_CLOSEADAPTER;

/* Function pointer types */
typedef NTSTATUS (WINAPI *PFN_D3DKMTEnumAdapters2)(D3DKMT_ENUMADAPTERS2 *);
typedef NTSTATUS (WINAPI *PFN_D3DKMTEscape)(const D3DKMT_ESCAPE *);
typedef NTSTATUS (WINAPI *PFN_D3DKMTCloseAdapter)(const D3DKMT_CLOSEADAPTER *);

/*
 * Dynamic loader for D3DKMT functions from gdi32.dll.
 * Call wddm_lite_d3dkmt_init() once at startup.
 */
struct WddmLiteD3dkmt {
    HMODULE                     hGdi32;
    PFN_D3DKMTEnumAdapters2     pfnEnumAdapters2;
    PFN_D3DKMTEscape            pfnEscape;
    PFN_D3DKMTCloseAdapter      pfnCloseAdapter;
};

static inline bool wddm_lite_d3dkmt_init(struct WddmLiteD3dkmt *d3d)
{
    memset(d3d, 0, sizeof(*d3d));

    d3d->hGdi32 = LoadLibraryA("gdi32.dll");
    if (!d3d->hGdi32)
        return false;

    d3d->pfnEnumAdapters2 = (PFN_D3DKMTEnumAdapters2)
        GetProcAddress(d3d->hGdi32, "D3DKMTEnumAdapters2");
    d3d->pfnEscape = (PFN_D3DKMTEscape)
        GetProcAddress(d3d->hGdi32, "D3DKMTEscape");
    d3d->pfnCloseAdapter = (PFN_D3DKMTCloseAdapter)
        GetProcAddress(d3d->hGdi32, "D3DKMTCloseAdapter");

    if (!d3d->pfnEnumAdapters2 || !d3d->pfnEscape || !d3d->pfnCloseAdapter) {
        FreeLibrary(d3d->hGdi32);
        memset(d3d, 0, sizeof(*d3d));
        return false;
    }

    return true;
}

static inline void wddm_lite_d3dkmt_fini(struct WddmLiteD3dkmt *d3d)
{
    if (d3d->hGdi32) {
        FreeLibrary(d3d->hGdi32);
        memset(d3d, 0, sizeof(*d3d));
    }
}

#endif /* WDDM_LITE_D3DKMT_H_INCLUDED */
