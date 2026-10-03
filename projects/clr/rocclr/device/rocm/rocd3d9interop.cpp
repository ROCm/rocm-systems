/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "rocdevice.hpp"
#include "rocmemory.hpp"
#include "rocd3d9interop.hpp"

#ifdef _WIN32

#include <d3d9.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

#include "platform/context.hpp"
#include "platform/DxxInteropExt.h"
#include "platform/interop_d3d9.hpp"

namespace amd {
namespace roc {
namespace D3D9Interop {

static PFNAmdDxExtCreate9 gAmdDxExtCreate9 = nullptr;
static std::mutex gAmdDxExtCreate9Lock;

// The entry point is cached for the life of the process, but the D3D9 runtime owns the
// user-mode driver's lifetime and unloads it once the last D3D9 device goes away. Pin the
// module so the cached pointer cannot outlive its code. Without this, an application that
// destroys every D3D9 device and then creates a new one crashes inside the stale pointer.
static void PinModule(HMODULE hDLL) {
  CHAR modulePath[MAX_PATH] = {};
  if (GetModuleFileName(hDLL, modulePath, MAX_PATH) == 0) {
    return;
  }
  HMODULE pinned = nullptr;
  GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_PIN, modulePath, &pinned);
}

// Only a successful lookup is cached. The D3D9 user-mode driver is loaded by the D3D9
// runtime, so a probe that runs before the application has a device would otherwise latch
// a permanent failure.
static PFNAmdDxExtCreate9 GetAmdDxExtCreate9() {
  std::lock_guard<std::mutex> lock(gAmdDxExtCreate9Lock);
  if (gAmdDxExtCreate9 != nullptr) {
    return gAmdDxExtCreate9;
  }
  [] {
    // AmdDxExtCreate9 lives in the D3D9 user-mode driver loader, not in the D3D10/11 DXX
    // module that rocd3d11interop.cpp uses. The D3D9 runtime has already loaded it by the
    // time an application has a device, so a handle lookup is enough. The DXX module is
    // probed second only in case a driver branch co-locates the entry point there.
#if defined _WIN64
    static constexpr const CHAR* kModuleNames[] = {"atidx9loader64.dll", "atidxx64.dll"};
#else
    static constexpr const CHAR* kModuleNames[] = {"atidx9loader32.dll", "atidxx32.dll"};
#endif
    for (const CHAR* moduleName : kModuleNames) {
      HMODULE hDLL = GetModuleHandle(moduleName);
      if (hDLL == nullptr) {
        continue;
      }
      auto func = reinterpret_cast<PFNAmdDxExtCreate9>(GetProcAddress(hDLL, "AmdDxExtCreate9"));
      if (func != nullptr) {
        PinModule(hDLL);
        gAmdDxExtCreate9 = func;
        CHAR modulePath[MAX_PATH] = {};
        GetModuleFileName(hDLL, modulePath, MAX_PATH);
        ClPrint(amd::LOG_INFO, amd::LOG_INIT, "D3D9 interop: AmdDxExtCreate9 from %s", modulePath);
        return;
      }
    }

    // GetModuleHandle resolves a base name to whichever copy loaded first. On a machine with
    // two AMD GPUs on different driver packages, that can be the loader for the other adapter,
    // which may predate the OpenCL D3D9 extension. Fall back to scanning every loaded module
    // for the export instead of trusting the name.
    HMODULE hKernel = GetModuleHandle("kernel32.dll");
    if (hKernel == nullptr) {
      return;
    }
    using PFNEnumProcessModules = BOOL(WINAPI*)(HANDLE, HMODULE*, DWORD, LPDWORD);
    auto enumModules =
        reinterpret_cast<PFNEnumProcessModules>(GetProcAddress(hKernel, "K32EnumProcessModules"));
    if (enumModules == nullptr) {
      return;
    }

    std::vector<HMODULE> modules(512);
    DWORD bytesNeeded = 0;
    if (!enumModules(GetCurrentProcess(), modules.data(),
                     static_cast<DWORD>(modules.size() * sizeof(HMODULE)), &bytesNeeded)) {
      return;
    }
    const size_t count =
        std::min(modules.size(), static_cast<size_t>(bytesNeeded) / sizeof(HMODULE));
    for (size_t i = 0; i < count; ++i) {
      auto func = reinterpret_cast<PFNAmdDxExtCreate9>(
          GetProcAddress(modules[i], "AmdDxExtCreate9"));
      if (func != nullptr) {
        PinModule(modules[i]);
        gAmdDxExtCreate9 = func;
        CHAR modulePath[MAX_PATH] = {};
        GetModuleFileName(modules[i], modulePath, MAX_PATH);
        ClPrint(amd::LOG_INFO, amd::LOG_INIT, "D3D9 interop: AmdDxExtCreate9 from %s", modulePath);
        return;
      }
    }
  }();
  return gAmdDxExtCreate9;
}

//! Stable identity for a COM object. QueryInterface(IID_IUnknown) is the only
//! identity guarantee COM makes, and we need one because the same D3D9 device may be
//! reached either as IDirect3DDevice9 or IDirect3DDevice9Ex.
static void* ComIdentity(IUnknown* obj) {
  if (obj == nullptr) {
    return nullptr;
  }
  IUnknown* unk = nullptr;
  if (FAILED(obj->QueryInterface(IID_IUnknown, reinterpret_cast<void**>(&unk))) || !unk) {
    return nullptr;
  }
  unk->Release();
  // Released on purpose: the pointer is only used as a map key, and the device
  // outlives the OpenCL context that references it.
  return static_cast<void*>(unk);
}

//! Creates (or returns the already cached) IAmdDxExtCLInterop9 for a D3D9 device.
static IAmdDxExtCLInterop9* GetOrCreateCLExt(amd::Context& ctx, IDirect3DDevice9* pDev9) {
  void* key = ComIdentity(pDev9);
  if (key == nullptr) {
    return nullptr;
  }

  if (auto* cached = static_cast<IAmdDxExtCLInterop9*>(ctx.getD3D9Ext(key))) {
    return cached;
  }

  PFNAmdDxExtCreate9 AmdDxExtCreate9 = GetAmdDxExtCreate9();
  if (AmdDxExtCreate9 == nullptr) {
    // Logged once: every surface in a failing run would otherwise repeat it.
    static std::once_flag reported;
    std::call_once(reported, []() {
      LogError("No loaded D3D9 driver module exports AmdDxExtCreate9; D3D9 image interop "
               "unavailable. A driver providing IAmdDxExtCLInterop9::CLQueryResource is required.");
    });
    return nullptr;
  }

  // AmdDxExtCreate9 only accepts a D3D9Ex device; a plain IDirect3DDevice9 has no Ex interface
  // and so cannot back the extension at all.
  IDirect3DDevice9Ex* pDev9Ex = nullptr;
  if (FAILED(pDev9->QueryInterface(IID_IDirect3DDevice9Ex,
                                   reinterpret_cast<void**>(&pDev9Ex))) ||
      (pDev9Ex == nullptr)) {
    return nullptr;
  }

  IAmdDxExt* pExt = nullptr;
  HRESULT hr = AmdDxExtCreate9(pDev9Ex, &pExt);
  pDev9Ex->Release();
  if (FAILED(hr) || !pExt) {
    LogPrintfError("AmdDxExtCreate9 failed: 0x%xh", hr);
    return nullptr;
  }

  // Version gate, matching what the PAL backend's D3D10/D3D11 paths already do. A major
  // version below 1 predates the OpenCL interop extension, so GetExtInterface would hand back
  // an interface whose vtable does not match IAmdDxExtCLInterop9.
  AmdDxExtVersion extVer = {};
  if (FAILED(pExt->GetVersion(&extVer)) || extVer.majorVersion < 1) {
    LogPrintfError("D3D9 DXX extension version %u.%u does not support OpenCL interop",
                   extVer.majorVersion, extVer.minorVersion);
    pExt->Release();
    return nullptr;
  }

  auto* pCLExt = static_cast<IAmdDxExtCLInterop9*>(pExt->GetExtInterface(AmdDxExtCLInteropID));
  if (pCLExt == nullptr) {
    LogError("D3D9 extension object has no OpenCL interop interface");
    pExt->Release();
    return nullptr;
  }

  ctx.setD3D9Ext(key, pExt, pCLExt);
  return pCLExt;
}

//! Reports a failed SRD query. Losing the SRD is not fatal - the caller falls back to the WDDM
//! surface-metadata path - but that path cannot describe every surface, so a failure here is
//! usually the first symptom of a wrong-geometry bug that only shows up much later as corrupt
//! pixels or a bogus image height. Printed to stderr unconditionally because rocclr's
//! LogPrintfError does not reach the console in a normal application or CTS run.
static void ReportSrdQueryFailure(IDirect3DSurface9* surface, UINT plane, const char* reason,
                                  HRESULT hr) {
  D3DSURFACE_DESC desc = {};
  surface->GetDesc(&desc);  // Best effort: the fields stay zero if this fails.

  // Video formats are FourCC codes (NV12, YV12, ...); everything else is a small enumerator.
  const uint32_t format = static_cast<uint32_t>(desc.Format);
  char formatText[16];
  if (format > 0xFFFF) {
    snprintf(formatText, sizeof(formatText), "%c%c%c%c", static_cast<char>(format),
             static_cast<char>(format >> 8), static_cast<char>(format >> 16),
             static_cast<char>(format >> 24));
  } else {
    snprintf(formatText, sizeof(formatText), "0x%x", format);
  }

  fprintf(stderr,
          "[ROCr D3D9 interop] SRD query failed: %s (hr=0x%08lX), surface %ux%u fmt=%s plane=%u"
          " - falling back to the WDDM surface-metadata path\n",
          reason, static_cast<unsigned long>(hr), desc.Width, desc.Height, formatText, plane);
  fflush(stderr);
}

//! Fills `srd` with the driver SRD for one plane of a D3D9 surface. `*srdSize` is the buffer
//! capacity on entry and the number of bytes written on return; it is zeroed when no usable SRD
//! could be obtained, which the caller treats as "fall back to the surface-metadata path"
//! rather than as an error.
static bool QueryImageSrd(IAmdDxExtCLInterop9* pCLExt, IDirect3DSurface9* surface, UINT plane,
                          void* srd, UINT* srdSize) {
  const UINT capacity = *srdSize;
  *srdSize = 0;
  if (capacity == 0) {
    return false;
  }

  memset(srd, 0, capacity);
  UINT size = capacity;
  HRESULT hr = pCLExt->CLQueryResource(surface, plane, srd, &size);

  // A too-small buffer fails, reports the required size in *pSrdSize and leaves the buffer
  // untouched. Retry once at exactly that size instead of treating the first failure as fatal:
  // a driver branch that validates the incoming size rather than treating it as a capacity
  // rejects our 256-byte buffer on the first pass and succeeds on the second.
  if (FAILED(hr) && size != 0 && size != capacity && size <= capacity) {
    memset(srd, 0, capacity);
    hr = pCLExt->CLQueryResource(surface, plane, srd, &size);
  }

  if (FAILED(hr)) {
    // A surface that is a level of a texture can be refused where its container is accepted -
    // the driver describes textures, and the reference sample queries the texture rather than
    // the level. Retry through the container before giving up.
    IDirect3DTexture9* container = nullptr;
    if (SUCCEEDED(surface->GetContainer(IID_IDirect3DTexture9,
                                        reinterpret_cast<void**>(&container))) &&
        container != nullptr) {
      memset(srd, 0, capacity);
      size = capacity;
      hr = pCLExt->CLQueryResource(container, plane, srd, &size);
      container->Release();
    }
  }

  if (FAILED(hr)) {
    LogPrintfError("CLQueryResource failed for D3D9 surface plane %u: 0x%xh", plane, hr);
    ReportSrdQueryFailure(surface, plane, "CLQueryResource rejected the resource", hr);
    return false;
  }

  // An SRD is a whole number of dwords and is never all zero; accepting either would build a
  // degenerate image that fails much later, far from the cause.
  if (size == 0 || size > capacity || (size % sizeof(uint32_t)) != 0) {
    LogPrintfError("CLQueryResource reported an unusable SRD size %u (capacity %u)", size,
                   capacity);
    char reason[96];
    snprintf(reason, sizeof(reason), "unusable SRD size %u (capacity %u)", size, capacity);
    ReportSrdQueryFailure(surface, plane, reason, hr);
    return false;
  }

  const auto* bytes = static_cast<const unsigned char*>(srd);
  if (std::all_of(bytes, bytes + size, [](unsigned char b) { return b == 0; })) {
    LogPrintfError("CLQueryResource returned an all-zero SRD for plane %u", plane);
    ReportSrdQueryFailure(surface, plane, "driver returned an all-zero SRD", hr);
    return false;
  }

  *srdSize = size;
  return true;
}

//! Reads the adapter LUID behind a D3D9 device. Unlike D3D10/D3D11 there is no DXGI
//! path, so this goes through IDirect3D9Ex::GetAdapterLUID the way the PAL backend
//! does (see device/pal/paldeviced3d9.cpp).
static bool GetD3D9AdapterLuid(IDirect3DDevice9* pDev9, LUID* luid) {
  D3DDEVICE_CREATION_PARAMETERS params = {};
  if (FAILED(pDev9->GetCreationParameters(&params))) {
    LogError("Failed to get creation parameters from D3D9 device");
    return false;
  }

  IDirect3D9* pD3D9 = nullptr;
  if (FAILED(pDev9->GetDirect3D(&pD3D9)) || !pD3D9) {
    LogError("Failed to get IDirect3D9 from D3D9 device");
    return false;
  }

  IDirect3D9Ex* pD3D9Ex = nullptr;
  HRESULT hr = pD3D9->QueryInterface(IID_IDirect3D9Ex, reinterpret_cast<void**>(&pD3D9Ex));
  pD3D9->Release();

  if (FAILED(hr) || !pD3D9Ex) {
    // A plain Direct3DCreate9 object has no Ex interface, so it cannot report a LUID.
    // Adapter ordinals are enumerated identically by both factories, so create a throwaway
    // IDirect3D9Ex and look the ordinal up there instead. Without this, every adapter would
    // be accepted and a multi-GPU box could bind the wrong one.
    HMODULE hD3D9 = GetModuleHandle("d3d9.dll");
    if (hD3D9 == nullptr) {
      LogInfo("d3d9.dll not loaded; skipping D3D9 LUID validation");
      return false;
    }
    using PFNDirect3DCreate9Ex = HRESULT(WINAPI*)(UINT, IDirect3D9Ex**);
    auto createEx =
        reinterpret_cast<PFNDirect3DCreate9Ex>(GetProcAddress(hD3D9, "Direct3DCreate9Ex"));
    if (createEx == nullptr || FAILED(createEx(D3D_SDK_VERSION, &pD3D9Ex)) || !pD3D9Ex) {
      LogInfo("Direct3DCreate9Ex unavailable; skipping D3D9 LUID validation");
      return false;
    }
  }

  hr = pD3D9Ex->GetAdapterLUID(params.AdapterOrdinal, luid);
  pD3D9Ex->Release();
  return SUCCEEDED(hr);
}

bool associateD3D9Device(const Device* device, void* gfxDevice, AdapterKind kind,
                         void* gfxContext, bool validateOnly) {
  if (!device || !gfxDevice) {
    return false;
  }

  if (!device->hasValidLUID()) {
    LogError("ROCr device does not have valid LUID for D3D9 interop");
    return false;
  }

  // A DXVA adapter is an IDXVAHD_Device, which exposes no adapter identity. Accept it;
  // the surfaces it produces carry their own owning D3D9 device, which is what the
  // export path keys off. The PAL backend likewise leaves this slot unvalidated.
  if (kind == AdapterKind::DXVA) {
    return true;
  }

  auto* pDev9 = static_cast<IDirect3DDevice9*>(gfxDevice);

  LUID adapterLuid = {};
  if (!GetD3D9AdapterLuid(pDev9, &adapterLuid)) {
    // Without a LUID we cannot prove a mismatch. Returning false here would fail
    // clCreateContext outright for plain D3D9 adapters, so accept and let the export
    // path fail later if the surface really is on another adapter.
    return true;
  }

  bool canInteroperate = (device->getDeviceLUID().HighPart == adapterLuid.HighPart) &&
                         (device->getDeviceLUID().LowPart == adapterLuid.LowPart);

  if (!canInteroperate) {
    // Expected during multi-GPU probing; only log for a real bind.
    if (!validateOnly) {
      LogError("D3D9 device and ROCr device cannot interoperate (LUID mismatch)");
    }
    return false;
  }

  // Probe: compatibility only, skip DXX ext caching.
  if (validateOnly) {
    return true;
  }

  // Only a D3D9Ex device can back AmdDxExtCreate9 (GetOrCreateCLExt checks for the Ex
  // interface itself). When the application handed us a plain IDirect3DDevice9 the extension
  // is created later, from the surface's own device, in Export().
  if (kind == AdapterKind::D3D9Ex && gfxContext != nullptr) {
    GetOrCreateCLExt(*static_cast<amd::Context*>(gfxContext), pDev9);
  }

  return true;
}

void dissociateD3D9Device(const Device* device, void* const gfxDevice[], void* gfxContext) {
  if (!gfxContext) return;

  // Entries are populated lazily and keyed by the surface's owning device rather than
  // by a context slot, so release the whole cache instead of looking up by handle.
  auto* ctx = static_cast<amd::Context*>(gfxContext);
  for (const auto& entry : ctx->drainD3D9Ext()) {
    if (entry.pCLExt) static_cast<IAmdDxExtCLInterop9*>(entry.pCLExt)->Release();
    if (entry.pExt)   static_cast<IAmdDxExt*>(entry.pExt)->Release();
  }
}

bool Export(const Memory* memory, D3D9Object* d3d9Obj, hsa_handle_t* handle, int* offset,
            void* srd, UINT* srdSize, hsa_interop_map_flag_t* mapFlags) {
  if (!memory || !d3d9Obj || !handle || !offset || !mapFlags) {
    return false;
  }

  IDirect3DSurface9* d3d9Surface = d3d9Obj->getD3D9Resource();
  if (d3d9Surface == nullptr) {
    LogError("D3D9 interop object has no shared surface");
    return false;
  }

  // Unlike D3D10/D3D11 there is no DXGI round trip: initD3D9Object already obtained a
  // legacy KMT share handle, either from the application or by creating the staging
  // surface with CreateOffscreenPlainSurface.
  HANDLE hShared = d3d9Obj->getD3D9SharedHandle();
  if (hShared == nullptr) {
    LogError("D3D9 surface has no KMT shared handle");
    return false;
  }

  // misc: 0 = non-planar, 1 = NV12/P010, 2 = YV12, 3 = YUY2
  const cl_uint misc = d3d9Obj->getMiscFlag();
  const cl_uint plane = d3d9Obj->getPlane();

  if (srd && srdSize) {
    const UINT srdCapacity = *srdSize;
    // The DXX extension only exists on a D3D9Ex device. Resolve it from the surface
    // itself (IDirect3DResource9::GetDevice), which works for every adapter type -
    // the DXVA slot never carries a D3D9 device at all. The driver also rejects the query
    // unless the resource belongs to the very device the extension was created from
    // (DxnExtCLInterop::CLQueryResource), so the surface is the only correct source.
    IAmdDxExtCLInterop9* pCLExt = nullptr;
    IDirect3DDevice9* pOwnerDevice = nullptr;
    if (SUCCEEDED(d3d9Surface->GetDevice(&pOwnerDevice)) && pOwnerDevice) {
      pCLExt = GetOrCreateCLExt(memory->owner()->getContext(), pOwnerDevice);
      pOwnerDevice->Release();
    }

    // Not having an SRD is not fatal: the caller unstamps the image descriptor when *srdSize
    // comes back zero and the surface-metadata path takes over. That is the only thing that
    // works for a plain IDirect3DDevice9, which has no Ex interface and so cannot back
    // AmdDxExtCreate9 at all, and for a driver branch that declines the surface.
    if (pCLExt == nullptr) {
      // Never reached the query at all - report it with the same shape so the two causes are
      // distinguishable in a log.
      ReportSrdQueryFailure(d3d9Surface, plane,
                            "no DXX OpenCL interop extension for this surface's device", S_OK);
    }

    if (pCLExt == nullptr || !QueryImageSrd(pCLExt, d3d9Surface, plane, srd, srdSize)) {
      // Leave no partial SRD behind. A rejected query can still have written into the buffer,
      // and the fallback path below copies the surface-metadata blob into that same memory
      // writing only as many bytes as the blob needs - stale dwords past its end would be read
      // back as part of the reconstructed descriptor.
      memset(srd, 0, srdCapacity);
      *srdSize = 0;
    }
  }

  *handle = reinterpret_cast<hsa_handle_t>(hShared);
  *mapFlags = HSA_INTEROP_MAP_FLAG_KMT_HANDLE;
  *offset = 0;

  if (misc == 0 || misc == 3) {
    // Non-planar, and YUY2 which is a single interleaved plane.
    return true;
  }

  // Sub-plane views start partway into the allocation. setObjDesc() halves both
  // dimensions for the chroma planes of NV12/P010 and YV12, so scale back up to
  // recover the luma plane geometry. Mirrors the PAL backend's handling in
  // device/pal/palresource.cpp.
  const uint32_t pitchAlign = memory->dev().info().imagePitchAlignment_;
  const uint32_t lumaWidth = d3d9Obj->getWidth() * 2u;
  const uint32_t lumaHeight = d3d9Obj->getHeight() * 2u;
  const uint32_t lumaElemBytes =
      static_cast<uint32_t>(d3d9Obj->getElementBytes(d3d9Obj->getObjDesc()->d3dFormat_, 0));
  const uint32_t lumaPitch = amd::alignUp(lumaWidth * lumaElemBytes, pitchAlign);
  const uint32_t lumaSize = lumaPitch * lumaHeight;

  const uint32_t chromaPitch = amd::alignUp((lumaWidth / 2u) * lumaElemBytes, pitchAlign);
  const uint32_t chromaSize = chromaPitch * (lumaHeight / 2u);

  switch (misc) {
    case 1:  // NV12 / P010: a single interleaved UV plane follows Y.
      switch (plane) {
        case 0:
          break;
        case 1:
          *offset = static_cast<int>(lumaSize);
          break;
        default:
          LogPrintfError("Unexpected plane %u for NV12/P010 surface", plane);
          return false;
      }
      break;
    case 2:  // YV12: Y, then V and U, each a quarter of Y.
      switch (plane) {
        case 0:
          break;
        case 1:
          *offset = static_cast<int>(lumaSize);
          break;
        case 2:
          *offset = static_cast<int>(lumaSize + chromaSize);
          break;
        default:
          LogPrintfError("Unexpected plane %u for YV12 surface", plane);
          return false;
      }
      break;
    default:
      LogPrintfError("Unknown D3D9 interop view type %u", misc);
      return false;
  }

  return true;
}

}  // namespace D3D9Interop
}  // namespace roc
}  // namespace amd

#endif  // _WIN32
