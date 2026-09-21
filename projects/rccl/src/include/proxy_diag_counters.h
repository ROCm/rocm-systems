/*
 * Counter kinds for ncclProfilerRecordProxyDiagState; values must match
 * facebook_rccl::ProxyCounterTypes in proxy_trace/proxy_trace.h (excluding UNINITIALIZED).
 */
#pragma once

#include <stdint.h>

enum ncclProxyDiagCounter : uint8_t {
  ncclProxyDiagPosted = 0,
  ncclProxyDiagKernelCopyReady = 1,
  ncclProxyDiagRtrRecv = 2,
  ncclProxyDiagRtsSend = 3,
  ncclProxyDiagReceived = 4,
  ncclProxyDiagTransmitted = 5,
  ncclProxyDiagFlushed = 6,
  ncclProxyDiagDone = 7,
  ncclProxyDiagRecvTail = 8,
  ncclProxyDiagTailOrHead = 9,
  ncclProxyDiagFifoSzOrHeadCache = 10,
};
