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

#include "cl_common.hpp"

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
} D3D10ObjSize_t;

typedef struct {
  D3D10_USAGE d3d10Usage_;
  UINT bindFlags_;
  UINT cpuAccessFlags_;
  UINT miscFlags_;
} D3D10Flags_t;

typedef struct {
  D3D10_RESOURCE_DIMENSION objDim_;
  D3D10ObjSize_t objSize_;
  D3D10Flags_t objFlags_;
  UINT mipLevels_;
  UINT arraySize_;
  DXGI_FORMAT dxgiFormat_;
  DXGI_SAMPLE_DESC dxgiSampleDesc_;
} D3D10ObjDesc_t;

const DXGI_SAMPLE_DESC dxgiSampleDescDefault = {1, 0};

//! Class D3D10Object keeps all the info about the D3D10 object
//! from which the CL object is created
class D3D10Object : public InteropObject {
 private:
  ID3D10Resource* pD3D10Aux_;

  bool releaseResources_;

  static std::vector<std::pair<void*, UINT>> resources_;
  //! Global lock.
  static Monitor resLock_;

 protected:
  ID3D10Resource* pD3D10Res_;
  ID3D10Resource* pD3D10ResOrig_;
  ID3D10Query* pQuery_;
  D3D10ObjDesc_t objDesc_;
  D3D10ObjDesc_t objDescOrig_;
  UINT subRes_;

 public:
  // Default constructor
  D3D10Object()
      : pD3D10Aux_(nullptr),
        releaseResources_(false),
        pD3D10Res_(nullptr),
        pD3D10ResOrig_(nullptr),
        pQuery_(nullptr),
        subRes_(0) {
    memset(&objDesc_, 0, sizeof(objDesc_));
    memset(&objDescOrig_, 0, sizeof(objDescOrig_));
  }
  // Copy constructor
  D3D10Object(D3D10Object& d3d10obj) : pQuery_(nullptr) {
    *this = d3d10obj;
    this->releaseResources_ = true;
    // Add reference to the D3D10 resource to prevent its disappearance
    if (pD3D10ResOrig_) {
      pD3D10ResOrig_->AddRef();
    } else if (pD3D10Res_) {
      pD3D10Res_->AddRef();
    }
  }

  //! Virtual destructor
  virtual ~D3D10Object() {
    ScopedLock sl(resLock_);
    if (releaseResources_) {
      // Decrement reference to the D3D10 objects
      if (pD3D10Res_) pD3D10Res_->Release();
      if (pD3D10Aux_) pD3D10Aux_->Release();
      if (pD3D10ResOrig_) pD3D10ResOrig_->Release();
      if (pQuery_) pQuery_->Release();
      // Check if this resource has already been used for interop
      if (resources_.size()) {
        for (auto it = resources_.cbegin(); it != resources_.cend(); it++) {
          if (((pD3D10ResOrig_ && (*it).first == (void*)pD3D10ResOrig_) ||
               ((*it).first == (void*)pD3D10Res_)) &&
              (*it).second == subRes_) {
            resources_.erase(it);
            break;
          }
        }
      }
    }
  }

  static int initD3D10Object(const Context& amdContext, ID3D10Resource* pRes, UINT subresource,
                             D3D10Object& obj);

  D3D10Object* asD3D10Object() { return this; }

  //! D3D10Object query functions to get D3D10 info from member variables
  ID3D10Resource* getD3D10Resource() const { return pD3D10Res_; }
  ID3D10Resource* getD3D10ResOrig() const { return pD3D10ResOrig_; }
  ID3D10Query* getQuery() const { return pQuery_; }

  UINT getWidth() const { return objDesc_.objSize_.Width; }
  UINT getHeight() const { return objDesc_.objSize_.Height; }
  UINT getDepth() const { return objDesc_.objSize_.Depth; }
  size_t getElementBytes(DXGI_FORMAT dxgiFomat);
  size_t getElementBytes() { return getElementBytes(objDesc_.dxgiFormat_); }
  DXGI_FORMAT getDxgiFormat() { return objDesc_.dxgiFormat_; }
  UINT getSubresource() const { return subRes_; }
  const D3D10ObjDesc_t* getObjDesc() const { return &objDesc_; }

  //! Returns bytes per pixel > 0 if conversion successful, 0 otherwise;
  //! if formats are not compatible, cl format channel
  //! order and type are set to 0
  cl_image_format getCLFormatFromDXGI(DXGI_FORMAT dxgiFmt);
  cl_image_format getCLFormatFromDXGI() { return getCLFormatFromDXGI(objDesc_.dxgiFormat_); }
  size_t getResourceByteSize();

  // On acquire copy data from original resource to shared resource
  virtual bool copyOrigToShared();
  // On release copy data from shared copy to the original resource
  virtual bool copySharedToOrig();
};

//! Class BufferD3D10 is derived from classes Buffer and D3D10Object
//! where the former keeps all data for CL object and
//! the latter keeps all data for D3D10 object
class BufferD3D10 : public D3D10Object, public Buffer {
 protected:
  //! Initializes the device memory array which is nested
  // after 'BufferD3D10' object in memory layout.
  virtual void initDeviceMemory();

 public:
  //! BufferD3D10 constructor just calls constructors of base classes
  //! to pass down the parameters
  BufferD3D10(Context& amdContext, cl_mem_flags clFlags,
              D3D10Object& d3d10obj)
      :  // Call base classes constructors
        D3D10Object(d3d10obj),
        Buffer(amdContext, clFlags, d3d10obj.getResourceByteSize()) {
    setInteropObj(this);
  }
  virtual ~BufferD3D10() {}
};

//! Class Image1DD3D10 is derived from classes Image1D and D3D10Object
//! where the former keeps all data for CL object and
//! the latter keeps all data for D3D10 object
class Image1DD3D10 : public D3D10Object, public Image {
 protected:
  //! Initializes the device memory array which is nested
  // after'Image1DD3D10' object in memory layout.
  virtual void initDeviceMemory();

 public:
  //! Image1DD3D10 constructor just calls constructors of base classes
  //! to pass down the parameters
  Image1DD3D10(Context& amdContext, cl_mem_flags clFlags,
               D3D10Object& d3d10obj)
      :  // Call base classes constructors
        D3D10Object(d3d10obj),
        Image(amdContext, CL_MEM_OBJECT_IMAGE1D, clFlags,
              getCLFormatFromDXGI(d3d10obj.getDxgiFormat()),  // format,
              d3d10obj.getWidth(), 1, 1,
              d3d10obj.getWidth() * d3d10obj.getElementBytes(),  // rowPitch),
              0) {
    setInteropObj(this);
  }
  virtual ~Image1DD3D10() {}
};

//! Class Image2DD3D10 is derived from classes Image2D and D3D10Object
//! where the former keeps all data for CL object and
//! the latter keeps all data for D3D10 object
class Image2DD3D10 : public D3D10Object, public Image {
 protected:
  //! Initializes the device memory array which is nested
  // after'Image2DD3D10' object in memory layout.
  virtual void initDeviceMemory();

 public:
  //! Image2DD3D10 constructor just calls constructors of base classes
  //! to pass down the parameters
  Image2DD3D10(Context& amdContext, cl_mem_flags clFlags,
               D3D10Object& d3d10obj)
      :  // Call base classes constructors
        D3D10Object(d3d10obj),
        Image(amdContext, CL_MEM_OBJECT_IMAGE2D, clFlags,
              getCLFormatFromDXGI(d3d10obj.getDxgiFormat()),  // format,
              d3d10obj.getWidth(), d3d10obj.getHeight(), 1,
              d3d10obj.getWidth() * d3d10obj.getElementBytes(),  // rowPitch),
              0) {
    setInteropObj(this);
  }
  virtual ~Image2DD3D10() {}
};

//! Class Image3DD3D10 is derived from classes Image3D and D3D10Object
//! where the former keeps all data for CL object and
//! the latter keeps all data for D3D10 object
class Image3DD3D10 : public D3D10Object, public Image {
 protected:
  //! Initializes the device memory array which is nested
  // after'Image3DD3D10' object in memory layout.
  virtual void initDeviceMemory();

 public:
  //! Image2DD3D10 constructor just calls constructors of base classes
  //! to pass down the parameters
  Image3DD3D10(Context& amdContext, cl_mem_flags clFlags,
               D3D10Object& d3d10obj)
      :  // Call base classes constructors
        D3D10Object(d3d10obj),
        Image(amdContext, CL_MEM_OBJECT_IMAGE3D, clFlags,
              getCLFormatFromDXGI(d3d10obj.getDxgiFormat()),  // format,
              d3d10obj.getWidth(), d3d10obj.getHeight(), d3d10obj.getDepth(),
              d3d10obj.getWidth() * d3d10obj.getElementBytes(),  // rowPitch),
              d3d10obj.getWidth() * d3d10obj.getHeight() * d3d10obj.getElementBytes()) {
    setInteropObj(this);
  }
  virtual ~Image3DD3D10() {}
};

}  // namespace amd
