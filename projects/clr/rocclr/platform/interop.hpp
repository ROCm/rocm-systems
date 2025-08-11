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

#ifndef INTEROP_H_
#define INTEROP_H_

namespace amd {

//! Forward declarations of interop classes
class GLObject;
class BufferGL;
class ExternalMemory;


#ifdef _WIN32
class D3D10Object;
class D3D11Object;
class D3D9Object;
#endif  //_WIN32

//! Base object providing common map/unmap interface for interop objects
class InteropObject {
 public:
  //! Virtual destructor to get rid of linux warning
  virtual ~InteropObject() {}

  // Static cast functions for interop objects
  virtual GLObject* asGLObject() { return nullptr; }
  virtual BufferGL* asBufferGL() { return nullptr; }
  virtual ExternalMemory* asExternalMemory() { return nullptr; }

#ifdef _WIN32
  virtual D3D10Object* asD3D10Object() { return nullptr; }
  virtual D3D11Object* asD3D11Object() { return nullptr; }
  virtual D3D9Object* asD3D9Object() { return nullptr; }
#endif  //_WIN32

  // On acquire copy data from original resource to shared resource
  virtual bool copyOrigToShared() { return true; }
  // On release copy data from shared copy to the original resource
  virtual bool copySharedToOrig() { return true; }
};

}  // namespace amd

#endif  //! INTEROP_H_
