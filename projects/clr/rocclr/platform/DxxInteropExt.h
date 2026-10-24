/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef __DXXINTEROPEXT_H__
#define __DXXINTEROPEXT_H__

// Abstract extension interface class
// Each extension interface (e.g. OpenCL Interop extension) will derive from this class
class IAmdDxExtInterface {
 public:
  virtual unsigned int AddRef(void) = 0;
  virtual unsigned int Release(void) = 0;

 protected:
  IAmdDxExtInterface() {};
  virtual ~IAmdDxExtInterface() = 0 {};
};

// forward declaration for d3d specific interfaces
interface ID3D10Device;
interface ID3D11Device;
interface IDirect3DDevice9Ex;
interface ID3D10Resource;
interface ID3D11Resource;
interface IDirect3DSurface9;

// forward declaration of extended primitive topology enumeration
enum AmdDxExtPrimitiveTopology;

// Extension version information
struct AmdDxExtVersion {
  unsigned int majorVersion;
  unsigned int minorVersion;
};

// This class serves as the main extension interface.
// AmdDxExtCreate returns a pointer to an instantiation of this interface.
// This object is used to retrieve extension version information
// and to get specific extension interfaces desired.
class IAmdDxExt : public IAmdDxExtInterface {
 public:
  virtual HRESULT GetVersion(AmdDxExtVersion* pExtVer) = 0;
  virtual IAmdDxExtInterface* GetExtInterface(unsigned int iface) = 0;

  // General extensions
  virtual HRESULT IaSetPrimitiveTopology(unsigned int topology) = 0;
  virtual HRESULT IaGetPrimitiveTopology(AmdDxExtPrimitiveTopology* pExtTopology) = 0;
  virtual HRESULT SetSingleSampleRead(ID3D10Resource* pResource, BOOL singleSample) = 0;
  virtual HRESULT SetSingleSampleRead11(ID3D11Resource* pResource, BOOL singleSample) = 0;
  virtual HRESULT SetSingleSampleRead9(IDirect3DSurface9* pResource, BOOL singleSample) = 0;

 protected:
  IAmdDxExt() {};
  virtual ~IAmdDxExt() = 0 {};
};

// OpenCL Interop extension ID passed to IAmdDxExt::GetExtInterface()
const unsigned int AmdDxExtCLInteropID = 7;

// Abstract OpenCL Interop extension interface class
class IAmdDxExtCLInterop : public IAmdDxExtInterface {
 public:
  virtual HRESULT QueryInteropGpuMask(UINT* gpuIdBitmask) = 0;

  virtual HRESULT CLAcquireResource(ID3D10Resource* pResource, UINT* gpuIdBitmask) = 0;
  virtual HRESULT CLReleaseResource(ID3D10Resource* pResource, UINT* gpuIdBitmask) = 0;
  virtual HRESULT CLQueryResource(ID3D10Resource* pResource, UINT planeIndex,
                                  VOID* pSrd, UINT* pSrdSize) = 0;

  virtual HRESULT CLAcquireResource11(ID3D11Resource* pResource, UINT* gpuIdBitmask) = 0;
  virtual HRESULT CLReleaseResource11(ID3D11Resource* pResource, UINT* gpuIdBitmask) = 0;
  virtual HRESULT CLQueryResource11(ID3D11Resource* pResource, UINT planeIndex,
                                    VOID* pSrd, UINT* pSrdSize) = 0;

  virtual HRESULT CLAcquireResource9(IDirect3DSurface9* pResource, UINT* gpuIdBitmask) = 0;
  virtual HRESULT CLReleaseResource9(IDirect3DSurface9* pResource, UINT* gpuIdBitmask) = 0;
  virtual HRESULT CLQueryResource9(IDirect3DSurface9* pResource, UINT planeIndex,
                                   VOID* pSrd, UINT* pSrdSize) = 0;
};

// The D3D9 user-mode driver exposes its own, much smaller OpenCL interop interface under the
// same extension ID. It is NOT the combined interface above: the vtable holds four methods,
// the resource type is IDirect3DResource9, and the gpu mask is passed by value. Obtained from
// the IAmdDxExt returned by AmdDxExtCreate9. Mirrors AmdDxExtOpenCLInteropApi.h in the dx9p
// driver - the layout must stay in lockstep with it.
interface IDirect3DResource9;

class IAmdDxExtCLInterop9 : public IAmdDxExtInterface {
 public:
  virtual HRESULT QueryInteropGpuMask(UINT* pGpuIdBitmask) = 0;

  virtual HRESULT CLAcquireResource(IDirect3DResource9* pResource, UINT gpuIdBitmask) = 0;
  virtual HRESULT CLReleaseResource(IDirect3DResource9* pResource, UINT gpuIdBitmask) = 0;
  // planeIndex selects one plane of a planar YUV image; ignored for other images and buffers.
  // *pSrdSize is the output buffer capacity on input and the required size on return.
  virtual HRESULT CLQueryResource(IDirect3DResource9* pResource, UINT planeIndex,
                                  VOID* pSrd, UINT* pSrdSize) = 0;
};

// Use GetProcAddress, etc. to retrieve exported functions
// The associated typedef provides a convenient way to define the function pointer
HRESULT __cdecl AmdDxExtCreate(ID3D10Device* pDevice, IAmdDxExt** ppExt);
typedef HRESULT(__cdecl* PFNAmdDxExtCreate)(ID3D10Device* pDevice, IAmdDxExt** ppExt);

HRESULT __cdecl AmdDxExtCreate11(ID3D11Device* pDevice, IAmdDxExt** ppExt);
typedef HRESULT(__cdecl* PFNAmdDxExtCreate11)(ID3D11Device* pDevice, IAmdDxExt** ppExt);

HRESULT __cdecl AmdDxExtCreate9(IDirect3DDevice9Ex* pDevice, IAmdDxExt** ppExt);
typedef HRESULT(__cdecl* PFNAmdDxExtCreate9)(IDirect3DDevice9Ex* pDevice, IAmdDxExt** ppExt);

struct CachedExt {
  IAmdDxExt* pExt = nullptr;
  IAmdDxExtCLInterop* pCLExt = nullptr;
};

#endif
