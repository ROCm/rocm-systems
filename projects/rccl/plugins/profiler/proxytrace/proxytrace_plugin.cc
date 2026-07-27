/*
 * RCCL proxy-trace profiler plugin: implements the same in-memory proxy
 * diagnostics as the legacy RCCL_ENABLE_PROXY_TRACE path, driven only via
 * the NCCL profiler API (ncclProfileProxyDiag + proxy op/step events).
 */

#include "proxytrace_plugin_shim.h"
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/types.h>

#include "proxy_trace/proxy_trace.h"
#include "nccl_profiler.h"

struct CommCtx {
  uint64_t commHash;
  int rank;
  facebook_rccl::ProxyTrace* trace;
};

struct ProxyOpProfEvent {
  uint64_t tag; /* ncclProfileProxyOp */
  ncclProfilerEventDescr_v6_t descr;
  facebook_rccl::ProxyTraceRecordKey traceKey;
  bool haveTraceOp;
  CommCtx* comm;
};

struct StepProfEvent {
  uint64_t tag; /* ncclProfileProxyStep */
};

static ncclDebugLogger_t gLog;

static ncclResult_t pluginInit(void** context, uint64_t commId, int* eActivationMask, const char* /*commName*/,
                               int /*nNodes*/, int /*nRanks*/, int rank, ncclDebugLogger_t logfn) {
  gLog = logfn;
  *eActivationMask = ncclProfileProxyOp | ncclProfileProxyStep | ncclProfileProxyDiag;
  CommCtx* c = new CommCtx();
  c->commHash = commId;
  c->rank = rank;
  c->trace = new facebook_rccl::ProxyTrace(rank);
  *context = c;
  return ncclSuccess;
}

static ncclResult_t pluginStartEvent(void* context, void** eHandle, ncclProfilerEventDescr_v6_t* eDescr) {
  if (eDescr->type == ncclProfileProxyOp) {
    auto* ev = new ProxyOpProfEvent();
    ev->tag = ncclProfileProxyOp;
    memcpy(&ev->descr, eDescr, sizeof(ev->descr));
    ev->traceKey.commHash = eDescr->proxyOp.commHash;
    ev->traceKey.opCount = eDescr->proxyOp.opCount;
    ev->traceKey.proxyOpId = -1;
    ev->haveTraceOp = false;
    ev->comm = (CommCtx*)context;
    *eHandle = ev;
    return ncclSuccess;
  }
  if (eDescr->type == ncclProfileProxyStep) {
    auto* se = new StepProfEvent();
    se->tag = ncclProfileProxyStep;
    *eHandle = se;
    return ncclSuccess;
  }
  *eHandle = nullptr;
  return ncclSuccess;
}

static ncclResult_t pluginStopEvent(void* eHandle) {
  if (!eHandle) return ncclSuccess;
  uint64_t tag = *(uint64_t*)eHandle;
  if (tag == ncclProfileProxyOp) {
    delete (ProxyOpProfEvent*)eHandle;
  } else if (tag == ncclProfileProxyStep) {
    delete (StepProfEvent*)eHandle;
  }
  return ncclSuccess;
}

static ncclResult_t pluginRecordEventState(void* eHandle, ncclProfilerEventState_v6_t eState,
                                           ncclProfilerEventStateArgs_v6_t* eStateArgs) {
  if (!eHandle) return ncclSuccess;
  uint64_t tag = *(uint64_t*)eHandle;
  if (tag != ncclProfileProxyOp) return ncclSuccess;

  auto* ev = (ProxyOpProfEvent*)eHandle;
  CommCtx* ctx = ev->comm;
  if (!ctx || !ctx->trace) return ncclSuccess;

  if (eState == ncclProfilerProxyOpInProgress_v4) {
    facebook_rccl::ProxyTraceExtraInfo ex;
    ex.funcIdx = ev->descr.proxyOp.traceFuncIdx;
    ex.protocol = ev->descr.proxyOp.traceProtocol;
    ex.pattern = ev->descr.proxyOp.tracePattern;
    ex.totalBytes = ev->descr.proxyOp.traceTotalBytes;
    ex.chunkSize = (uint32_t)ev->descr.proxyOp.chunkSize;
    auto opType = ev->descr.proxyOp.isSend ? facebook_rccl::ProxyOpType::SEND : facebook_rccl::ProxyOpType::RECV;
    ctx->trace->addNewProxyOp(ev->traceKey, ex, opType, ev->descr.proxyOp.channelId, ev->descr.proxyOp.rawProxyNsteps,
                              ev->descr.proxyOp.proxyNbytes, ev->descr.proxyOp.peer);
    ev->haveTraceOp = true;
    return ncclSuccess;
  }

  if (eState == ncclProfilerProxyDiagUpdate && eStateArgs && ev->haveTraceOp) {
    auto kind = (facebook_rccl::ProxyCounterTypes)eStateArgs->proxyDiag.counterKind;
    if ((eStateArgs->proxyDiag.flags & 2u) == 0) {
      ctx->trace->updateProxyOpCounter(ev->traceKey, kind, eStateArgs->proxyDiag.value);
    }
    if (eStateArgs->proxyDiag.flags & 1u) {
      ctx->trace->setProxyOpTimestamp(ev->traceKey, kind);
    }
    return ncclSuccess;
  }

  return ncclSuccess;
}

static ncclResult_t pluginFinalize(void* context) {
  CommCtx* c = (CommCtx*)context;
  if (!c) return ncclSuccess;
  delete c->trace;
  delete c;
  return ncclSuccess;
}

extern "C" {

void ncclProfilerProxyTraceDump(void* profilerContext, ncclDebugLogger_t logfn) {
  CommCtx* c = (CommCtx*)profilerContext;
  if (!c || !c->trace || !logfn) return;
  std::string s = c->trace->dump();
  logfn(NCCL_LOG_WARN, NCCL_ALL, __func__, __LINE__, "%s", s.c_str());
}

ncclProfiler_v6_t ncclProfiler_v6 = {"ProxyTrace", pluginInit, pluginStartEvent, pluginStopEvent, pluginRecordEventState,
                                     pluginFinalize};
}
