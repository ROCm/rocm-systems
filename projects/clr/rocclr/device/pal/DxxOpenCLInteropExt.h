/* Copyright (c) 2008 - 2021 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE. */

#ifndef __DXXOPENCLINTEROPEXT_H__
#define __DXXOPENCLINTEROPEXT_H__

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

  virtual HRESULT CLAcquireResource11(ID3D11Resource* pResource, UINT* gpuIdBitmask) = 0;
  virtual HRESULT CLReleaseResource11(ID3D11Resource* pResource, UINT* gpuIdBitmask) = 0;

  virtual HRESULT CLAcquireResource9(IDirect3DSurface9* pResource, UINT* gpuIdBitmask) = 0;
  virtual HRESULT CLReleaseResource9(IDirect3DSurface9* pResource, UINT* gpuIdBitmask) = 0;
};

// Use GetProcAddress, etc. to retrieve exported functions
// The associated typedef provides a convenient way to define the function pointer
HRESULT __cdecl AmdDxExtCreate(ID3D10Device* pDevice, IAmdDxExt** ppExt);
typedef HRESULT(__cdecl* PFNAmdDxExtCreate)(ID3D10Device* pDevice, IAmdDxExt** ppExt);

HRESULT __cdecl AmdDxExtCreate11(ID3D11Device* pDevice, IAmdDxExt** ppExt);
typedef HRESULT(__cdecl* PFNAmdDxExtCreate11)(ID3D11Device* pDevice, IAmdDxExt** ppExt);

HRESULT __cdecl AmdDxExtCreate9(IDirect3DDevice9Ex* pDevice, IAmdDxExt** ppExt);
typedef HRESULT(__cdecl* PFNAmdDxExtCreate9)(IDirect3DDevice9Ex* pDevice, IAmdDxExt** ppExt);

#endif
