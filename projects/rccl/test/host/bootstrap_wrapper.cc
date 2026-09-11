// Compiles the real bootstrap.cc (hipified) with hipcc --offload-host-only.
// Only bootstrapBidirEnabled() is exercised by BootstrapBidirTests;
// the rest of bootstrap.cc is dead code for our purposes but compiles
// correctly. Link stubs below satisfy external symbols.

#include "bootstrap.cc"

// --- Link stubs for bootstrap.cc dependencies ---
// These symbols are defined in other RCCL .cc files that we do not compile.
// They are never called during bootstrapBidirEnabled() execution.

#include "net.h"
#include "socket.h"
#include "proxy.h"

// Needed even though bootstrap.cc never names it: it is the default argument for
// ncclSocketInit()'s `magic`, so every call that omits it emits a reference here.
uint64_t ncclSocketDefaultMagic(void) { return 0; }

ncclResult_t ncclSocketInit(ncclSocket*, ncclSocketAddress const*, unsigned long,
                            ncclSocketType, unsigned int volatile*, int, int) { return ncclSuccess; }
ncclResult_t ncclSocketListen(ncclSocket*) { return ncclSuccess; }
ncclResult_t ncclSocketConnect(ncclSocket*) { return ncclSuccess; }
ncclResult_t ncclSocketAccept(ncclSocket*, ncclSocket*, bool) { return ncclSuccess; }
ncclResult_t ncclSocketSend(ncclSocket*, void*, int) { return ncclSuccess; }
ncclResult_t ncclSocketRecv(ncclSocket*, void*, int) { return ncclSuccess; }
ncclResult_t ncclSocketSendRecv(ncclSocket*, void*, int, ncclSocket*, void*, int) { return ncclSuccess; }
ncclResult_t ncclSocketClose(ncclSocket*, bool) { return ncclSuccess; }
ncclResult_t ncclSocketReady(ncclSocket*, int*) { return ncclSuccess; }
ncclResult_t ncclSocketGetAddr(ncclSocket*, ncclSocketAddress*) { return ncclSuccess; }
ncclResult_t ncclSocketGetAddrFromString(ncclSocketAddress*, char const*) { return ncclSuccess; }
const char* ncclSocketToString(ncclSocketAddress const*, char* buf, int) { return buf; }
ncclResult_t ncclSocketMultiOp(ncclSocketOp*, int) { return ncclSuccess; }

ncclResult_t ncclFindInterfaces(char*, ncclSocketAddress*, int, int, int*) { return ncclSuccess; }
ncclResult_t ncclFindInterfaceMatchSubnet(char*, ncclSocketAddress*, ncclSocketAddress*, int, int*) { return ncclSuccess; }
int parseStringList(char const*, netIf*, int) { return 0; }
bool matchIfList(char const*, int, netIf*, int, bool, int*) { return false; }

ncclResult_t ncclProxyInit(ncclComm*, ncclSocket*, ncclSocketAddress*, unsigned long*) { return ncclSuccess; }
ncclResult_t ncclRasAddRanks(rasRankInit*, int) { return ncclSuccess; }
ncclResult_t ncclRasCommInit(ncclComm*, rasRankInit*) { return ncclSuccess; }

ncclResult_t ncclOsSetFilesLimit() { return ncclSuccess; }
uint64_t ncclOsGetPid() { return 0; }
unsigned long getHostHash() { return 0; }
unsigned long getPidHash() { return 0; }
unsigned long hashCombine(unsigned long a, unsigned long) { return a; }
void RegisterSignalHandlers() {}
void ncclSetThreadName(std::thread&, char const*, ...) {}
