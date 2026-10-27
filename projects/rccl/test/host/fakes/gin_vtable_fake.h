/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// One ncclGin_t vtable for the host microtests that drive a GIN plugin.
// Every slot has a call counter and a result knob. The extra fields record
// what gin-plugin-init-test.cc and gin-host-finalize-test.cc assert.

#ifndef RCCL_TEST_HOST_GIN_VTABLE_FAKE_H_
#define RCCL_TEST_HOST_GIN_VTABLE_FAKE_H_

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "nccl_gin.h"

struct FakeGinVtable {
  int initCalls = 0;
  int devicesCalls = 0;
  int getGinPropertiesCalls = 0;
  int getPropertiesCalls = 0;
  int listenCalls = 0;
  int connectCalls = 0;
  int createContextCalls = 0;
  int regMrSymCalls = 0;
  int regMrSymDmaBufCalls = 0;
  int deregMrSymCalls = 0;
  int destroyContextCalls = 0;
  int closeCollCalls = 0;
  int closeListenCalls = 0;
  int ginProgressCalls = 0;
  int queryLastErrorCalls = 0;
  int finalizeCalls = 0;

  ncclResult_t initResult = ncclSuccess;
  ncclResult_t devicesResult = ncclSuccess;
  ncclResult_t getGinPropertiesResult = ncclSuccess;
  ncclResult_t getPropertiesResult = ncclSuccess;
  ncclResult_t listenResult = ncclSuccess;
  ncclResult_t connectResult = ncclSuccess;
  ncclResult_t createContextResult = ncclSuccess;
  ncclResult_t regMrSymResult = ncclSuccess;
  ncclResult_t regMrSymDmaBufResult = ncclSuccess;
  ncclResult_t deregMrSymResult = ncclSuccess;
  ncclResult_t destroyContextResult = ncclSuccess;
  ncclResult_t closeCollResult = ncclSuccess;
  ncclResult_t closeListenResult = ncclSuccess;
  ncclResult_t ginProgressResult = ncclSuccess;
  ncclResult_t queryLastErrorResult = ncclSuccess;
  ncclResult_t finalizeResult = ncclSuccess;

  // init() allocates this on success, the way the IB plugin calloc'd a config.
  // finalize() frees the ctx it is handed. Leave init null in the vtable when
  // initNullFn is set.
  int ndev = 0;
  void* lastCtx = nullptr;
  bool initNullFn = false;
  bool lastError = false;

  std::vector<void*> closedColls;
  std::vector<void*> deregColls;
  // backend->closed as seen from inside each closeColl, when closedFlag is set.
  std::vector<bool> closedFlagAtClose;
  const bool* closedFlag = nullptr;
  // Fail closeColl only for this handle. Null means use closeCollResult.
  void* closeCollFailOn = nullptr;
  int connectFailOnCall = 0;  // 1-based; 0 never fails
  uintptr_t nextCollComm = 0x5000;
  int listenToken = 0;
  int regToken = 0;
  int ginCtxToken = 0;
  ncclNetDeviceHandle_t ginDevHandle{};

  static FakeGinVtable*& currentPtr() {
    static FakeGinVtable* p = nullptr;
    return p;
  }
  static FakeGinVtable& current() { return *currentPtr(); }
  static void setCurrent(FakeGinVtable* p) { currentPtr() = p; }

  static ncclResult_t Init(void** ctx, uint64_t /*commId*/, ncclDebugLogger_t) {
    FakeGinVtable& self = current();
    ++self.initCalls;
    if (self.initResult != ncclSuccess) return self.initResult;
    self.lastCtx = std::malloc(8);
    if (ctx) *ctx = self.lastCtx;
    return ncclSuccess;
  }

  static ncclResult_t Devices(int* ndev) {
    FakeGinVtable& self = current();
    ++self.devicesCalls;
    if (self.devicesResult != ncclSuccess) return self.devicesResult;
    if (ndev) *ndev = self.ndev;
    return ncclSuccess;
  }

  static ncclResult_t GetGinProperties(ncclGinProperties_t* props) {
    FakeGinVtable& self = current();
    ++self.getGinPropertiesCalls;
    if (self.getGinPropertiesResult == ncclSuccess && props) std::memset(props, 0, sizeof(*props));
    return self.getGinPropertiesResult;
  }

  static ncclResult_t GetProperties(int /*dev*/, ncclNetProperties_t* props) {
    FakeGinVtable& self = current();
    ++self.getPropertiesCalls;
    if (self.getPropertiesResult == ncclSuccess && props) std::memset(props, 0, sizeof(*props));
    return self.getPropertiesResult;
  }

  static ncclResult_t Listen(void* /*ctx*/, int /*dev*/, void* /*handle*/, void** listenComm) {
    FakeGinVtable& self = current();
    ++self.listenCalls;
    if (self.listenResult != ncclSuccess) return self.listenResult;
    if (listenComm) *listenComm = &self.listenToken;
    return ncclSuccess;
  }

  static ncclResult_t Connect(void* /*ctx*/, void* /*handles*/[], int /*nranks*/, int /*rank*/,
                              void* /*listenComm*/, void** collComm) {
    FakeGinVtable& self = current();
    if (++self.connectCalls == self.connectFailOnCall) return ncclSystemError;
    if (self.connectResult != ncclSuccess) return self.connectResult;
    if (collComm) *collComm = reinterpret_cast<void*>(self.nextCollComm++);
    return ncclSuccess;
  }

  // On success write both out-params the way the in-tree GIN fakes do; gin_host
  // rejects a null ginCtx or a null / empty-handle devHandle.
  static ncclResult_t CreateContext(void* /*collComm*/, ncclGinConfig_t* /*config*/, void** ginCtx,
                                    ncclNetDeviceHandle_t** devHandle) {
    FakeGinVtable& self = current();
    ++self.createContextCalls;
    if (self.createContextResult != ncclSuccess) return self.createContextResult;
    self.ginDevHandle.handle = &self.ginCtxToken;
    self.ginDevHandle.needsProxyProgress = 0;
    if (ginCtx) *ginCtx = &self.ginCtxToken;
    if (devHandle) *devHandle = &self.ginDevHandle;
    return ncclSuccess;
  }

  static ncclResult_t RegMrSym(void* /*collComm*/, void* /*data*/, size_t /*size*/, int /*type*/,
                               uint64_t /*mrFlags*/, void** mhandle, void** ginHandle) {
    FakeGinVtable& self = current();
    ++self.regMrSymCalls;
    if (self.regMrSymResult != ncclSuccess) return self.regMrSymResult;
    if (mhandle) *mhandle = &self.regToken;
    if (ginHandle) *ginHandle = &self.regToken;
    return ncclSuccess;
  }

  static ncclResult_t RegMrSymDmaBuf(void* /*collComm*/, void* /*data*/, size_t /*size*/, int /*type*/,
                                     uint64_t /*offset*/, int /*fd*/, uint64_t /*mrFlags*/, void** mhandle,
                                     void** ginHandle) {
    FakeGinVtable& self = current();
    ++self.regMrSymDmaBufCalls;
    if (self.regMrSymDmaBufResult != ncclSuccess) return self.regMrSymDmaBufResult;
    if (mhandle) *mhandle = &self.regToken;
    if (ginHandle) *ginHandle = &self.regToken;
    return ncclSuccess;
  }

  static ncclResult_t DeregMrSym(void* collComm, void* /*mhandle*/) {
    FakeGinVtable& self = current();
    ++self.deregMrSymCalls;
    self.deregColls.push_back(collComm);
    return self.deregMrSymResult;
  }

  static ncclResult_t DestroyContext(void* /*ginCtx*/) {
    ++current().destroyContextCalls;
    return current().destroyContextResult;
  }

  static ncclResult_t CloseColl(void* collComm) {
    FakeGinVtable& self = current();
    ++self.closeCollCalls;
    self.closedColls.push_back(collComm);
    if (self.closedFlag) self.closedFlagAtClose.push_back(*self.closedFlag);
    if (self.closeCollFailOn != nullptr && collComm == self.closeCollFailOn) return ncclSystemError;
    return self.closeCollResult;
  }

  static ncclResult_t CloseListen(void* /*listenComm*/) {
    FakeGinVtable& self = current();
    ++self.closeListenCalls;
    return self.closeListenResult;
  }

  static ncclResult_t GinProgress(void* /*ginCtx*/) {
    ++current().ginProgressCalls;
    return current().ginProgressResult;
  }

  static ncclResult_t QueryLastError(void* /*ginCtx*/, bool* hasError) {
    FakeGinVtable& self = current();
    ++self.queryLastErrorCalls;
    if (hasError) *hasError = self.lastError;
    return self.queryLastErrorResult;
  }

  static ncclResult_t Finalize(void* ctx) {
    FakeGinVtable& self = current();
    ++self.finalizeCalls;
    std::free(ctx);
    if (self.lastCtx == ctx) self.lastCtx = nullptr;
    return self.finalizeResult;
  }

  ncclGin_t vtable() const {
    ncclGin_t gin{};
    gin.name = "FakeGinVtable";
    gin.init = initNullFn ? nullptr : &Init;
    gin.devices = &Devices;
    gin.getGinProperties = &GetGinProperties;
    gin.getProperties = &GetProperties;
    gin.listen = &Listen;
    gin.connect = &Connect;
    gin.createContext = &CreateContext;
    gin.regMrSym = &RegMrSym;
    gin.regMrSymDmaBuf = &RegMrSymDmaBuf;
    gin.deregMrSym = &DeregMrSym;
    gin.destroyContext = &DestroyContext;
    gin.closeColl = &CloseColl;
    gin.closeListen = &CloseListen;
    gin.ginProgress = &GinProgress;
    gin.queryLastError = &QueryLastError;
    gin.finalize = &Finalize;
    return gin;
  }
};

#endif  // RCCL_TEST_HOST_GIN_VTABLE_FAKE_H_
