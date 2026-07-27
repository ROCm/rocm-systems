// Common preamble for compiling real RCCL host-side .cc files with plain g++.
//
// Pre-defines include guards for all GPU device-code headers (making their
// #include directives no-ops) and provides forward declarations for the types
// those blocked headers would have defined. This lets comm.h, core.h, topo.h,
// and other real RCCL infrastructure headers compile on a CPU-only toolchain.
//
// Usage: #include this file BEFORE any RCCL header or .cc source.
// Requires: -isystem stubs/ for HIP/CUDA/HSA system header interception.
#pragma once

// ===== Block ALL nccl_device/ headers =====
// GPU kernel implementation headers. Pre-defining their guards makes
// #include directives no-ops. The wrapped .cc files are pure host code.

// nccl_device/ top-level
#define _NCCL_DEVICE_CORE_H_
#define _NCCL_DEVICE_COMM_H_
#define _NCCL_DEVICE_COOP_H_
#define _NCCL_DEVICE_UTILITY_H_
#define _NCCL_DEVICE_BARRIER_H_
#define _NCCL_DEVICE_MEM_BARRIER_H_
#define _NCCL_DEVICE_GIN_BARRIER_H_
#define _NCCL_DEVICE_GIN_SESSION_H_
#define _NCCL_DEVICE_HIP_COMPAT_H_
#define _NCCL_DEVICE_LL_A2A_H_
#define _NCCL_DEVICE_PTR_H_
#define _NCCL_DEVICE_REDUCE_COPY_H_
// NCCL_NET_DEVICE_H_ intentionally NOT blocked — it's pure C types

// nccl_device/impl/
#define _NCCL_DEVICE_BARRIER__FUNCS_H_
#define _NCCL_DEVICE_BARRIER__TYPES_H_
#define _NCCL_DEVICE_COMM__FUNCS_H_
#define _NCCL_DEVICE_COMM__TYPES_H_
#define _NCCL_DEVICE_CORE__FUNCS_H_
#define _NCCL_DEVICE_CORE__TYPES_H_
#define _NCCL_DEVICE_GIN_BARRIER__FUNCS_H_
#define _NCCL_DEVICE_GIN_BARRIER__TYPES_H_
#define _NCCL_DEVICE_GIN_SESSION__FUNCS_H_
#define _NCCL_DEVICE_GIN_SESSION__TYPES_H_
#define _NCCL_DEVICE_LL_A2A__FUNCS_H_
#define _NCCL_DEVICE_LL_A2A__TYPES_H_
#define _NCCL_DEVICE_MEM_BARRIER__FUNCS_H_
#define _NCCL_DEVICE_MEM_BARRIER__TYPES_H_
#define _NCCL_DEVICE_MULTIMEM__FUNCS_H_
#define _NCCL_DEVICE_PTR__FUNCS_H_
#define _NCCL_DEVICE_PTR__TYPES_H_
#define _NCCL_DEVICE_REDUCE_COPY__FUNCS_H_
#define _NCCL_DEVICE_REDUCE_COPY__IMPL_H_
#define _NCCL_DEVICE_REDUCE_COPY__TYPES_H_
#define _NCCL_DEVICE_VECTOR__FUNCS_H_
#define _NCCL_DEVICE_VECTOR__TYPES_H_

// nccl_device/gin/ and generated _tmp.h
#define _NCCL_DEVICE_GIN_TMP_H_
#define _NCCL_DEVICE_COMM_TMP_H_
#define _NCCL_GIN_DEVICE_COMMON_H_
#define _NCCL_DEVICE_GIN_PROXY_H_
#define _NCCL_DEVICE_GIN__TYPES_H_
#define _NCCL_DEVICE_GIN__FUNCS_H_

// Other device/GPU headers that bleed through comm.h
#define _NCCL_GIN_HOST_H_
#define NCCL_RCCL_PTR_H_
#define NCCL_SYM_KERNELS_H_
#define NCCL_DEVICE_RUNTIME_H_
#define _NCCL_RMA_PROXY_H_
#define NCCL_COLLTRACE_EVENT_H_

// ===== Forward declarations that blocked headers would provide =====
#include <cstdint>
#include <ostream>

struct ncclDevComm;
typedef struct ncclDevComm ncclDevComm_t;
struct ncclTeam { int type; int localInner; int localOuter; };
typedef struct ncclTeam ncclTeam_t;
struct ncclWindow_vidmem;
typedef struct ncclWindow_vidmem ncclWindow_vidmem_t;
struct ncclMultimemHandle { int dummy; };
typedef struct ncclMultimemHandle ncclMultimemHandle_t;
typedef uint32_t ncclDevResourceHandle;
typedef ncclDevResourceHandle ncclDevResourceHandle_t;
typedef uint32_t ncclGinSignal_t;
typedef uint32_t ncclGinCounter_t;
struct ncclLsaBarrierHandle { int dummy; };
typedef struct ncclLsaBarrierHandle ncclLsaBarrierHandle_t;
struct ncclGinBarrierHandle { int dummy; };
typedef struct ncclGinBarrierHandle ncclGinBarrierHandle_t;
struct ncclLLA2AHandle { int dummy; };
typedef struct ncclLLA2AHandle ncclLLA2AHandle_t;
template<typename T=void> struct ncclSymPtr { int dummy; };
struct ncclTeamTagWorld {};
struct ncclTeamTagLsa {};
struct ncclTeamTagRail {};
struct ncclDevCommRequirements { int dummy; };
typedef struct ncclDevCommRequirements ncclDevCommRequirements_t;
struct ncclDevResourceRequirements { int dummy; };
typedef struct ncclDevResourceRequirements ncclDevResourceRequirements_t;
struct ncclTeamRequirements { int dummy; };
typedef struct ncclTeamRequirements ncclTeamRequirements_t;
struct ncclCommProperties { int dummy; };
typedef struct ncclCommProperties ncclCommProperties_t;
struct ncclGin { int dummy; };
struct ncclResourceWindow_vidmem { int dummy; };
typedef struct ncclResourceWindow_vidmem ncclResourceWindow_vidmem_t;
#define NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER {}
#define NCCL_COMM_PROPERTIES_INITIALIZER {}
#define MAXCHANNELS 128
#define NCCL_EXTERN_C extern "C"
#define NCCL_IR_EXTERN_C extern "C"
#define NCCL_HOST_DEVICE_INLINE inline
#define NCCL_DEVICE_INLINE inline

// GIN types
enum ncclGinType_t { NCCL_GIN_NONE = 0 };
enum ncclGinConnectionType_t { NCCL_GIN_CONN_NONE = 0 };
#define NCCL_GIN_MAX_CONNECTIONS 8
struct ncclGinState { int dummy; };
typedef void* ncclGin_t;
typedef void* ncclGinWindow_t;

// Types from blocked headers (sym_kernels.h, dev_runtime.h, rma_proxy.h)
typedef int ncclSymRegType_t;
struct ncclDevrState { int dummy; };
struct ncclSymkState { int dummy; };
struct ncclRmaProxyState { int dummy; };
