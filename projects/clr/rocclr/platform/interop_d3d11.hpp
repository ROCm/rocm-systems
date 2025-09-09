/* Copyright (c) 2008 - 2023 Advanced Micro Devices, Inc.

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

#pragma once

#include "interop_d3d10.hpp"
#include "platform/context.hpp"
#include "platform/memory.hpp"

#include <utility>

namespace amd {

typedef struct {
  union {
    UINT ByteWidth;
    UINT Width;
  };
  UINT Height;
  UINT Depth;
} D3D11ObjSize_t;

typedef struct {
  D3D11_USAGE d3d11Usage_;
  UINT bindFlags_;
  UINT cpuAccessFlags_;
  UINT miscFlags_;
  UINT structureByteStride_;
} D3D11Flags_t;

typedef struct {
  D3D11_RESOURCE_DIMENSION objDim_;
  D3D11ObjSize_t objSize_;
  D3D11Flags_t objFlags_;
  UINT mipLevels_;
  UINT arraySize_;
  DXGI_FORMAT dxgiFormat_;
  DXGI_SAMPLE_DESC dxgiSampleDesc_;
} D3D11ObjDesc_t;

//! Class D3D11Object keeps all the info about the D3D11 object
//! from which the CL object is created
class D3D11Object : public InteropObject {
 private:
  ID3D11Resource* pD3D11Aux_;

  bool releaseResources_;

  static std::vector<std::pair<void*, std::pair<UINT, UINT>>> resources_;

 protected:
  //! Global lock.
  static Monitor resLock_;

  ID3D11Resource* pD3D11Res_;
  ID3D11Resource* pD3D11ResOrig_;
  ID3D11Query* pQuery_;
  D3D11ObjDesc_t objDesc_;
  UINT subRes_;
  INT plane_;

 public:
  // Default constructor
  D3D11Object()
      : pD3D11Aux_(nullptr),
        releaseResources_(false),
        pD3D11Res_(nullptr),
        pD3D11ResOrig_(nullptr),
        pQuery_(nullptr),
        subRes_(NULL),
        plane_(NULL) {
    memset(&objDesc_, 0, sizeof(objDesc_));
  }
  // Copy constructor
  D3D11Object(D3D11Object& d3d11obj) : pQuery_(NULL) {
    *this = d3d11obj;
    this->releaseResources_ = true;
    // Add reference to the D3D11 resource to prevent its disappearance
    if (pD3D11ResOrig_) {
      pD3D11ResOrig_->AddRef();
    } else if (pD3D11Res_) {
      pD3D11Res_->AddRef();
    }
    assert(pD3D11Res_ != pD3D11ResOrig_);
  }

  //! Virtual destructor
  virtual ~D3D11Object() {
    ScopedLock sl(resLock_);
    if (releaseResources_) {
      // Decrement reference to the D3D11 objects
      if (pD3D11Res_) pD3D11Res_->Release();
      if (pD3D11Aux_) pD3D11Aux_->Release();
      if (pD3D11ResOrig_) pD3D11ResOrig_->Release();
      if (pQuery_) pQuery_->Release();
      // Check if this resource has already been used for interop
      if (resources_.size()) {
        for (auto it = resources_.cbegin(); it != resources_.cend(); it++) {
          if (((pD3D11ResOrig_ && (*it).first == (void*)pD3D11ResOrig_) ||
               ((*it).first == (void*)pD3D11Res_)) &&
              (*it).second.first == subRes_ && (*it).second.second == plane_) {
            resources_.erase(it);
            break;
          }
        }
      }
    }
  }

  static int initD3D11Object(const Context& amdContext, ID3D11Resource* pRes, UINT subresource,
                             D3D11Object& obj, INT plane = -1);

  D3D11Object* asD3D11Object() { return this; }

  //! D3D11Object query functions to get D3D11 info from member variables
  ID3D11Resource* getD3D11Resource() const { return pD3D11Res_; }
  ID3D11Resource* getD3D11ResOrig() const { return pD3D11ResOrig_; }
  ID3D11Query* getQuery() const { return pQuery_; }
  Monitor& getResLock() { return resLock_; }
  UINT getWidth() const { return objDesc_.objSize_.Width; }
  UINT getHeight() const { return objDesc_.objSize_.Height; }
  UINT getDepth() const { return objDesc_.objSize_.Depth; }
  size_t getElementBytes(DXGI_FORMAT dxgiFomat, cl_uint plane);
  size_t getElementBytes() { return getElementBytes(objDesc_.dxgiFormat_, plane_); }
  DXGI_FORMAT getDxgiFormat() { return objDesc_.dxgiFormat_; }
  UINT getSubresource() const { return subRes_; }
  INT getPlane() const { return plane_; }
  const D3D11ObjDesc_t* getObjDesc() const { return &objDesc_; }

  cl_uint getMiscFlag(void);
  //! Returns bytes per pixel > 0 if conversion successful, 0 otherwise;
  //! if formats are not compatible, cl format channel
  //! order and type are set to 0
  cl_image_format getCLFormatFromDXGI(DXGI_FORMAT dxgiFmt, cl_uint plane);
  cl_image_format getCLFormatFromDXGI() {
    return getCLFormatFromDXGI(objDesc_.dxgiFormat_, plane_);
  }
  size_t getResourceByteSize();

  // On acquire copy data from original resource to shared resource
  virtual bool copyOrigToShared();
  // On release copy data from shared copy to the original resource
  virtual bool copySharedToOrig();
};

//! Class BufferD3D11 is derived from classes Buffer and D3D11Object
//! where the former keeps all data for CL object and
//! the latter keeps all data for D3D11 object
class BufferD3D11 : public D3D11Object, public Buffer {
 protected:
  //! Initializes the device memory array which is nested
  // after'BufferD3D11' object in memory layout.
  virtual void initDeviceMemory();

 public:
  //! BufferD3D11 constructor just calls constructors of base classes
  //! to pass down the parameters
  BufferD3D11(Context& amdContext, cl_mem_flags clFlags,
              D3D11Object& d3d11obj)
      :  // Call base classes constructors
        D3D11Object(d3d11obj),
        Buffer(amdContext, clFlags, d3d11obj.getResourceByteSize()) {
    setInteropObj(this);
  }
  virtual ~BufferD3D11() {}
};

//! Class Image1DD3D11 is derived from classes Image1D and D3D11Object
//! where the former keeps all data for CL object and
//! the latter keeps all data for D3D11 object
class Image1DD3D11 : public D3D11Object, public Image {
 protected:
  //! Initializes the device memory array which is nested
  // after'Image1DD3D11' object in memory layout.
  virtual void initDeviceMemory();

 public:
  //! Image1DD3D11 constructor just calls constructors of base classes
  //! to pass down the parameters
  Image1DD3D11(Context& amdContext, cl_mem_flags clFlags,
               D3D11Object& d3d11obj)
      :  // Call base classes constructors
        D3D11Object(d3d11obj),
        Image(amdContext, CL_MEM_OBJECT_IMAGE1D, clFlags,
              getCLFormatFromDXGI(d3d11obj.getDxgiFormat(), d3d11obj.getPlane()),  // format,
              d3d11obj.getWidth(), 1, 1,
              d3d11obj.getWidth() * d3d11obj.getElementBytes(),  // rowPitch),
              0) {
    setInteropObj(this);
  }
  virtual ~Image1DD3D11() {}
};

//! Class Image2DD3D11 is derived from classes Image2D and D3D11Object
//! where the former keeps all data for CL object and
//! the latter keeps all data for D3D11 object
class Image2DD3D11 : public Image, public D3D11Object {
 protected:
  //! Initializes the device memory array which is nested
  // after'Image2DD3D11' object in memory layout.
  virtual void initDeviceMemory();

 public:
  //! Image2DD3D11 constructor just calls constructors of base classes
  //! to pass down the parameters
  Image2DD3D11(Context& amdContext, cl_mem_flags clFlags,
               D3D11Object& d3d11obj)
      :  // Call base classes constructors
        D3D11Object(d3d11obj),
        Image(amdContext, CL_MEM_OBJECT_IMAGE2D, clFlags,
              getCLFormatFromDXGI(d3d11obj.getDxgiFormat(), d3d11obj.getPlane()),  // format,
              d3d11obj.getWidth(), d3d11obj.getHeight(), 1,
              d3d11obj.getWidth() * d3d11obj.getElementBytes(),  // rowPitch),
              0) {
    setInteropObj(this);
  }
  virtual ~Image2DD3D11() {}
};

//! Class Image3DD3D11 is derived from classes Image3D and D3D11Object
//! where the former keeps all data for CL object and
//! the latter keeps all data for D3D11 object
class Image3DD3D11 : public D3D11Object, public Image {
 protected:
  //! Initializes the device memory array which is nested
  // after'Image3DD3D11' object in memory layout.
  virtual void initDeviceMemory();

 public:
  //! Image2DD3D11 constructor just calls constructors of base classes
  //! to pass down the parameters
  Image3DD3D11(Context& amdContext, cl_mem_flags clFlags,
               D3D11Object& d3d11obj)
      :  // Call base classes constructors
        D3D11Object(d3d11obj),
        Image(amdContext, CL_MEM_OBJECT_IMAGE3D, clFlags,
              getCLFormatFromDXGI(d3d11obj.getDxgiFormat(), d3d11obj.getPlane()),  // format,
              d3d11obj.getWidth(), d3d11obj.getHeight(), d3d11obj.getDepth(),
              d3d11obj.getWidth() * d3d11obj.getElementBytes(),  // rowPitch),
              d3d11obj.getWidth() * d3d11obj.getHeight() * d3d11obj.getElementBytes()) {
    setInteropObj(this);
  }
  virtual ~Image3DD3D11() {}
};

}  // namespace amd
