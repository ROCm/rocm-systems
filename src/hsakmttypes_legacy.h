// Legacy struct definitions for backward compatibility with older rocr-runtime
#ifndef _HSAKMTTYPES_LEGACY_H_
#define _HSAKMTTYPES_LEGACY_H_

#pragma pack(push, 4)

// Old HsaExternalHandleDesc layout (rocr-runtime 7.12 and earlier).
// The rocr patch "rocr/hsakmt/clr: Enable IPC feature for wsl and windows"
// changed fd from HSAint32 to HSAint64 and added a void *mem field, breaking
// the struct layout. This legacy definition is kept so that librocdxg can
// still parse the old layout when paired with an older rocr-runtime.
typedef struct _HsaExternalHandleDesc_712 {
    HsaAMDGPUDeviceHandle device_handle;
    HSAint32 fd;
    HsaExternalHandleType type;
    HSAuint32 metadata;
} HsaExternalHandleDesc_712;

#pragma pack(pop)

#endif // _HSAKMTTYPES_LEGACY_H_
