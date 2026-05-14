/****************************************************************************\
 *
 *  Module Name    kmddbg.h
 *
 *  Description    Escape function support for shader debugging.
 *
 *  Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 \****************************************************************************/

#ifndef _KMDDBG_H_
#define _KMDDBG_H_

#if defined(_cplusplus)
extern "C" {
#endif   //  _cplusplusp

#pragma pack(push, 4)

//
// Escape header structure version info
//

#ifndef _LHESCAPE_H_

#define ATI_LHESCAPE_HEADER_MINOR_VER                 0x0000
#define ATI_LHESCAPE_HEADER_MAJOR_VER                 0x0001

// used for ulHeaderVersion
#define ATI_LHESCAPE_HEADER_VER \
    ((ATI_LHESCAPE_HEADER_MAJOR_VER << 16) | ATI_LHESCAPE_HEADER_MINOR_VER)

#define LHESCAPE_OK                                0x00000000
#define LHESCAPE_ERROR                             0x00000001
#define LHESCAPE_NOTSUPPORTED                      0x00000002

typedef struct _ATI_LHESCAPE_HDR {
    ULONG   ulSize;         // size of the header
    ULONG   ulHeaderVersion;// Major/Minor version
    ULONG   ulEscapeCode;   //
    ULONG   ulClientID;     //
    ULONG   ulAdapterID[4]; // 16bytes == sizeof (GUID)
    ULONG   ulMACRsrvd[10]; // will be used for buffer authentication
    ULONG   ulEscapeReturnCode; // return code for ATI escape code.
    ULONG   ulSignature;    // signature to identify ATI escape header
    ULONG   ulRsrvd[12];    // round the structure to a full 16 ULONGs
} ATI_LHESCAPE_HDR, *PATI_LHESCAPE_HDR;

typedef enum _ATI_LH_ESCAPE_SERVICE_COMPONENT
{
  LHESCAPE_SC_KMD = 3,
} ATI_LH_ESCAPE_SERVICE_COMPONENT;

typedef enum _KMD_ESCAPE_SUBFUNCTION
{
  KMD_ESUBFUNC_SHADER_DBGR = 229
} KMD_ESCAPE_SUBFUNCTION;

// macro to be used to make the subcomponent field
#define LHESCAPE_MAKE_SERVICECOMPONENT(n)    (n<<24)

// macro to be used to make the escape sub function field
#define LHESCAPE_MAKE_ESCAPESUBFUNCTION(n)   (n<<8)

#define ATI_ESCAPE_CLIENT_ID_UMD 0x00000003

#define LHESCAPE_UMDKMDIF_SHADER_DBGR \
    (LHESCAPE_MAKE_SERVICECOMPONENT(LHESCAPE_SC_KMD) | \
     LHESCAPE_MAKE_ESCAPESUBFUNCTION(KMD_ESUBFUNC_SHADER_DBGR) | \
     ATI_ESCAPE_CLIENT_ID_UMD)

#endif // _LHESCAPE_H_

//
// Debugger escape structure info
//

#define KMD_DBGR_HEADER_MAJOR_VER                      0x0001
#define KMD_DBGR_HEADER_MINOR_VER                      0x0001

#define MAX_USER_QUEUES_VISIBLE_TO_DEBUGGER 32

// Capability bits in device info
#define KMD_DBGR_CAP_TRAP_DEBUG_SUPPORT                             0x00000001
#define KMD_DBGR_CAP_TRAP_DEBUG_PRECISE_MEMORY_OPERATIONS_SUPPORTED 0x00000002
#define KMD_DBGR_CAP_TRAP_DEBUG_FIRMWARE_SUPPORTED                  0x00000004
#define KMD_DBGR_CAP_WATCH_POINTS_SUPPORTED                         0x00000008
#define KMD_DBGR_CAP_WATCH_POINTS_TOTALBITS_MASK                    0x000000f0
#define KMD_DBGR_CAP_WATCH_POINTS_TOTALBITS_SHIFT                   4

// DebugProp bits in device info
#define KMD_DBGR_DBG_WATCH_ADDR_MASK_LO_BIT_MASK     0x0000000f
#define KMD_DBGR_DBG_WATCH_ADDR_MASK_LO_BIT_SHIFT    0
#define KMD_DBGR_DBG_WATCH_ADDR_MASK_HI_BIT_MASK     0x000003f0
#define KMD_DBGR_DBG_WATCH_ADDR_MASK_HI_BIT_SHIFT    4
#define KMD_DBGR_DBG_DISPATCH_INFO_ALWAYS_VALID      0x00000400

enum KMD_DBGR_CMDS_OP
{
    KMD_DBGR_CMD_OP_INVALID = 0,
    KMD_DBGR_CMD_OP_SET_RUNTIME_INFO = 1,
    KMD_DBGR_CMD_OP_GET_RUNTIME_INFO = 2,
    KMD_DBGR_CMD_OP_ENABLE_GLOBAL_TRAP = 3,
    KMD_DBGR_CMD_OP_ENABLE_EXCEPTIONS = 4,
    KMD_DBGR_CMD_OP_ENABLE_TRAPS_FOR_EXCEPTIONS = 5,
    KMD_DBGR_CMD_OP_SET_WAVE_LAUNCH_MODE = 6,
    KMD_DBGR_CMD_OP_SUSPEND_QUEUE = 7,
    KMD_DBGR_CMD_OP_RESUME_QUEUE = 8,
    KMD_DBGR_CMD_OP_GET_EXCEPTIONS = 9,
    KMD_DBGR_CMD_OP_GET_EXCEPTION_INFO = 10,
    KMD_DBGR_CMD_OP_GET_QUEUE_INFO = 11,
    KMD_DBGR_CMD_OP_GET_DEVICE_INFO = 12,
    KMD_DBGR_CMD_OP_GET_TRAP_VERSION = 13,
    KMD_DBGR_CMD_OP_SET_ADDR_WATCH = 14,
    KMD_DBGR_CMD_OP_CLEAR_ADDR_WATCH = 15,
    KMD_DBGR_CMD_OP_SET_PRECISE_MEMOPS = 16,
    KMD_DBGR_CMD_OP_SEND_EVENT_TO_INFERIOR = 17,
    KMD_DBGR_CMD_OP_READ_BUFFER = 18,
    KMD_DBGR_CMD_OP_WRITE_BUFFER = 19,
    KMD_DBGR_CMD_OP_SET_USER_TRAP = 20,
    KMD_DBGR_CMD_OP_MAX
};

enum KMD_DBGR_RUNTIME_STATE 
{
    KMD_DBGR_RUNTIME_STATE_DISABLED = 0,
    KMD_DBGR_RUNTIME_STATE_ENABLED = 1,
    KMD_DBGR_RUNTIME_STATE_ENABLED_ERROR = 2
};

enum KMD_DBGR_TRAP_MASK 
{
    KMD_DBGR_TRAP_MASK_FP_INVALID = 1,
    KMD_DBGR_TRAP_MASK_FP_INPUT_DENORMAL = 2,
    KMD_DBGR_TRAP_MASK_FP_DIVIDE_BY_ZERO = 4,
    KMD_DBGR_TRAP_MASK_FP_OVERFLOW = 8,
    KMD_DBGR_TRAP_MASK_FP_UNDERFLOW = 16,
    KMD_DBGR_TRAP_MASK_FP_INEXACT = 32,
    KMD_DBGR_TRAP_MASK_INT_DIVIDE_BY_ZERO = 64,
    KMD_DBGR_TRAP_MASK_DBG_ADDRESS_WATCH = 128,
    KMD_DBGR_TRAP_MASK_DBG_MEMORY_VIOLATION = 256,
};

enum KMD_DBGR_EXCEPTIONS
{
    KMD_DBGR_EXCP_QUEUE_WAVE_NONE = 0,
    KMD_DBGR_EXCP_QUEUE_WAVE_ABORT = 1,
    KMD_DBGR_EXCP_QUEUE_WAVE_TRAP = 2,
    KMD_DBGR_EXCP_QUEUE_WAVE_MATH_ERROR = 3,
    KMD_DBGR_EXCP_QUEUE_WAVE_ILLEGAL_INSTRUCTION = 4,
    KMD_DBGR_EXCP_QUEUE_WAVE_MEMORY_VIOLATION = 5,
    KMD_DBGR_EXCP_QUEUE_WAVE_APERTURE_VIOLATION = 6,
    KMD_DBGR_EXCP_QUEUE_PACKET_DISPATCH_DIM_INVALID = 16,
    KMD_DBGR_EXCP_QUEUE_PACKET_DISPATCH_GROUP_SEGMENT_SIZE_INVALID = 17,
    KMD_DBGR_EXCP_QUEUE_PACKET_DISPATCH_CODE_INVALID = 18,
    KMD_DBGR_EXCP_QUEUE_PACKET_RESERVED = 19,
    KMD_DBGR_EXCP_QUEUE_PACKET_UNSUPPORTED = 20,
    KMD_DBGR_EXCP_QUEUE_PACKET_DISPATCH_WORK_GROUP_SIZE_INVALID = 21,
    KMD_DBGR_EXCP_QUEUE_PACKET_DISPATCH_REGISTER_INVALID = 22,
    KMD_DBGR_EXCP_QUEUE_PACKET_VENDOR_UNSUPPORTED = 23,
    KMD_DBGR_EXCP_QUEUE_PREEMPTION_ERROR = 30,
    KMD_DBGR_EXCP_QUEUE_NEW = 31,
    /* per device */
    KMD_DBGR_EXCP_DEVICE_QUEUE_DELETE = 32,
    KMD_DBGR_EXCP_DEVICE_MEMORY_VIOLATION = 33,
    KMD_DBGR_EXCP_DEVICE_RAS_ERROR = 34,
    KMD_DBGR_EXCP_DEVICE_FATAL_HALT = 35,
    KMD_DBGR_EXCP_DEVICE_NEW = 36,
    /* per process */
    KMD_DBGR_EXCP_PROCESS_RUNTIME = 48,
    KMD_DBGR_EXCP_PROCESS_DEVICE_REMOVE = 49,
};

enum KMD_DBGR_WAVE_LAUNCH_MODE
{
    KMD_DBGR_WAVE_LAUNCH_MODE_NORMAL = 0,
    KMD_DBGR_WAVE_LAUNCH_MODE_HALT = 1,
    KMD_DBGR_WAVE_LAUNCH_MODE_KILL = 2,
    KMD_DBGR_WAVE_LAUNCH_MODE_TRAP_AFTER_INST = 3,
    KMD_DBGR_WAVE_LAUNCH_MODE_DISABLE = 4,
    KMD_DBGR_WAVE_LAUNCH_MODE_MAX
};

enum KMD_DBGR_ENABLE_TRAPS_FOR_EXCEPTIONS_MODE
{
    KMD_DBGR_ENABLE_TRAPS_FOR_EXCEPTIONS_MODE_OR = 0,
    KMD_DBGR_ENABLE_TRAPS_FOR_EXCEPTIONS_MODE_REPLACE = 1,
};

enum KMD_DBGR_USER_QUEUE_TYPE
{
    KMD_DBGR_USER_QUEUE_TYPE_GFX = 0,
    KMD_DBGR_USER_QUEUE_TYPE_COMPUTE = 1,
    KMD_DBGR_USER_QUEUE_TYPE_DMA = 2,
};

enum KMD_DBGR_USER_QUEUE_STATUS
{
    KMD_DBGR_USER_QUEUE_STATUS_SUCCESS = 0,
    KMD_DBGR_USER_QUEUE_STATUS_ERROR = 1,
    KMD_DBGR_USER_QUEUE_STATUS_INVALID = 2
};

typedef struct _KMDDBGRIF_SET_RUNTIME_INFO_INPUT
{
    union
    {
        struct
        {
            D3DKMT_ALIGN64 ULONG64                     rDebug;
            ULONG                                      runtimeState;
            ULONG                                      ttmpSetup;
        } rocRuntimeInfo;

        ULONG clientData[16];
    };
} KMDDBGRIF_SET_RUNTIME_INFO_INPUT, * PKMDDBGRIF_SET_RUNTIME_INFO_INPUT;

typedef struct _KMDDBGRIF_SET_RUNTIME_INFO_OUTPUT
{
    ULONG                                              Reserved[16];
} KMDDBGRIF_SET_RUNTIME_INFO_OUTPUT, * PKMDDBGRIF_SET_RUNTIME_INFO_OUTPUT;

typedef struct _KMDDBGRIF_GET_RUNTIME_INFO_INPUT
{
    ULONG                                              Reserved[16];
} KMDDBGRIF_GET_RUNTIME_INFO_INPUT, * PKMDDBGRIF_GET_RUNTIME_INFO_INPUT;

typedef struct _KMDDBGRIF_GET_RUNTIME_INFO_OUTPUT
{
    union
    {
        struct
        {
            D3DKMT_ALIGN64 ULONG64                     rDebug;
            ULONG                                      runtimeState;
            ULONG                                      ttmpSetup;
        } rocRuntimeInfo;

        ULONG clientData[16];
    };
} KMDDBGRIF_GET_RUNTIME_INFO_OUTPUT, * PKMDDBGRIF_GET_RUNTIME_INFO_OUTPUT;

typedef struct _KMDDBGRIF_ENABLE_GLOBAL_TRAP_INPUT
{
    BOOL                                               enable;
    ULONG64                                            exceptionMask; // Exception(s) to raise to the debugger
    D3DKMT_ALIGN64 ULONG64                             umdEventHandle64; // Event handle used by KMD to signal exceptions
    ULONG                                              Reserved[11];
} KMDDBGRIF_ENABLE_GLOBAL_TRAP_INPUT, * PKMDDBGRIF_ENABLE_GLOBAL_TRAP_INPUT;

typedef struct _KMDDBGRIF_ENABLE_GLOBAL_TRAP_OUTPUT
{
    ULONG                                              Reserved[16];
} KMDDBGRIF_ENABLE_GLOBAL_TRAP_OUTPUT, * PKMDDBGRIF_ENABLE_GLOBAL_TRAP_OUTPUT;

typedef struct _KMDDBGRIF_ENABLE_EXCEPTIONS_INPUT
{
    ULONG64                                            exceptionMask; // New exception(s) to be sent to the debugger
    ULONG                                              Reserved[14];
} KMDDBGRIF_ENABLE_EXCEPTIONS_INPUT, * PKMDDBGRIF_ENABLE_EXCEPTIONS_INPUT;

typedef struct _KMDDBGRIF_ENABLE_EXCEPTIONS_OUTPUT
{
    ULONG                                              Reserved[16];
} KMDDBGRIF_ENABLE_EXCEPTIONS_OUTPUT, * PKMDDBGRIF_ENABLE_EXCEPTIONS_OUTPUT;

typedef struct _KMDDBGRIF_ENABLE_TRAPS_FOR_EXCEPTIONS_INPUT
{
    KMD_DBGR_ENABLE_TRAPS_FOR_EXCEPTIONS_MODE          mode;
    ULONG                                              trapEnableMask;
    ULONG                                              trapRequestedMask;
    ULONG                                              Reserved[13];
} KMDDBGRIF_ENABLE_TRAPS_FOR_EXCEPTIONS_INPUT, * PKMDDBGRIF_ENABLE_TRAPS_FOR_EXCEPTIONS_INPUT;

typedef struct _KMDDBGRIF_ENABLE_TRAPS_FOR_EXCEPTIONS_OUTPUT
{
    ULONG                                              trapSupportMask;
    ULONG                                              Reserved[15];
} KMDDBGRIF_ENABLE_TRAPS_FOR_EXCEPTIONS_OUTPUT, * PKMDDBGRIF_ENABLE_TRAPS_FOR_EXCEPTIONS_OUTPUT;

typedef struct _KMDDBGRIF_SET_WAVE_LAUNCH_MODE_INPUT
{
    KMD_DBGR_WAVE_LAUNCH_MODE                          mode;
    ULONG                                              Reserved[15];
} KMDDBGRIF_SET_WAVE_LAUNCH_MODE_INPUT, * PKMDDBGRIF_SET_WAVE_LAUNCH_MODE_INPUT;

typedef struct _KMDDBGRIF_SET_WAVE_LAUNCH_MODE_OUTPUT
{
    ULONG                                              Reserved[16];
} KMDDBGRIF_SET_WAVE_LAUNCH_MODE_OUTPUT, * PKMDDBGRIF_SET_WAVE_LAUNCH_MODE_OUTPUT;

typedef struct _KMDDBGRIF_SUSPEND_QUEUE_INPUT
{
    ULONG64                                            exceptionMaskToClear; // Raised exception(s) that needs to be cleared
    ULONG                                              gracePeriodIn100us; // Time provided for queue to drain before preemption
    ULONG                                              numQueues;
    ULONG                                              queueIds[MAX_USER_QUEUES_VISIBLE_TO_DEBUGGER];
    ULONG                                              Reserved[12];
} KMDDBGRIF_SUSPEND_QUEUE_INPUT, * PKMDDBGRIF_SUSPEND_QUEUE_INPUT;

typedef struct _KMDDBGRIF_SUSPEND_QUEUE_OUTPUT
{
    ULONG                                              queueStatus[MAX_USER_QUEUES_VISIBLE_TO_DEBUGGER];
    ULONG                                              Reserved[16];
} KMDDBGRIF_SUSPEND_QUEUE_OUTPUT, * PKMDDBGRIF_SUSPEND_QUEUE_OUTPUT;

typedef struct _KMDDBGRIF_RESUME_QUEUE_INPUT
{
    ULONG                                              numQueues;
    ULONG                                              queueIds[MAX_USER_QUEUES_VISIBLE_TO_DEBUGGER];
    ULONG                                              Reserved[15];
} KMDDBGRIF_RESUME_QUEUE_INPUT, * PKMDDBGRIF_RESUME_QUEUE_INPUT;

typedef struct _KMDDBGRIF_RESUME_QUEUE_OUTPUT
{
    ULONG                                              queueStatus[MAX_USER_QUEUES_VISIBLE_TO_DEBUGGER];
    ULONG                                              Reserved[16];
} KMDDBGRIF_RESUME_QUEUE_OUTPUT, * PKMDDBGRIF_RESUME_QUEUE_OUTPUT;

typedef struct _KMDDBGRIF_GET_EXCEPTIONS_INPUT
{
    ULONG64                                            exceptionMaskToClear;
    ULONG                                              Reserved[14];
} KMDDBGRIF_GET_EXCEPTIONS_INPUT, * PKMDDBGRIF_GET_EXCEPTIONS_INPUT;

typedef struct _KMDDBGRIF_GET_EXCEPTIONS_OUTPUT
{
    ULONG64                                            exceptionMask;
    ULONG                                              queueId; // Non-zero for queue exception(s)
    BOOL                                               noExceptions;
    ULONG                                              Reserved[12];
} KMDDBGRIF_GET_EXCEPTIONS_OUTPUT, * PKMDDBGRIF_GET_EXCEPTIONS_OUTPUT;

typedef struct _KMDDBGRIF_GET_EXCEPTION_INFO_INPUT
{
    ULONG64                                            exceptionId;
    ULONG                                              queueId; // Non-zero for queue exception(s)
    BOOL                                               clearException;
    ULONG                                              Reserved[12];
} KMDDBGRIF_GET_EXCEPTION_INFO_INPUT, * PKMDDBGRIF_GET_EXCEPTION_INFO_INPUT;

typedef struct _KMDDBGRIF_GET_EXCEPTION_INFO_OUTPUT
{
    ULONG64                                            exceptionId;

    union
    {
        struct
        {
            D3DKMT_ALIGN64 ULONG64                     rDebug;
            ULONG                                      runtimeState;
            ULONG                                      ttmpSetup;
        } rocRuntimeInfo;

        struct
        {
            D3DKMT_ALIGN64 ULONG64                     faultingAddr;
        } pageFault;

        ULONG                                          exceptionInfoData[14];
    };
} KMDDBGRIF_GET_EXCEPTION_INFO_OUTPUT, * PKMDDBGRIF_GET_EXCEPTION_INFO_OUTPUT;

struct KMDDBGR_QUEUE_INFO {
    ULONG64                                            queueExceptionMask;
    D3DKMT_ALIGN64 ULONG64                             ringBufferAddress;
    ULONG                                              ringSize;
    D3DKMT_ALIGN64 ULONG64                             writePtrAddress;
    D3DKMT_ALIGN64 ULONG64                             readPtrAddress;
    D3DKMT_ALIGN64 ULONG64                             ctxSaveRestoreAddress;
    ULONG                                              ctxSaveRestoreSize;
    ULONG                                              queueId; // Non-zero for queue exception(s)
    KMD_DBGR_USER_QUEUE_TYPE                           queueType;
    D3DKMT_ALIGN64 ULONG64                             aqlPacketList;
    ULONG                                              computeTmpRingSize;
    ULONG                                              Reserved[15];
};

typedef struct _KMDDBGRIF_GET_QUEUE_INFO_INPUT
{
    ULONG64                                            exceptionMaskToClear;
    ULONG                                              Reserved[14];
} KMDDBGRIF_GET_QUEUE_INFO_INPUT, * PKMDDBGRIF_GET_QUEUE_INFO_INPUT;

typedef struct _KMDDBGRIF_GET_QUEUE_INFO_OUTPUT
{
    ULONG                                              numQueues;
    KMDDBGR_QUEUE_INFO                                 queueInfo[MAX_USER_QUEUES_VISIBLE_TO_DEBUGGER];
    ULONG                                              Reserved[15];
} KMDDBGRIF_GET_QUEUE_INFO_OUTPUT, * PKMDDBGRIF_GET_QUEUE_INFO_OUTPUT;

typedef struct _KMDDBGRIF_GET_TRAP_VERSION_INPUT
{
    ULONG                                              Reserved[16];
} KMDDBGRIF_GET_TRAP_VERSION_INPUT, * PKMDDBGRIF_GET_TRAP_VERSION_INPUT;

typedef struct _KMDDBGRIF_GET_TRAP_VERSION_OUTPUT
{
    ULONG                                              majorVersion;
    ULONG                                              minorVersion;
    ULONG                                              Reserved[14];
} KMDDBGRIF_GET_TRAP_VERSION_OUTPUT, * PKMDDBGRIF_GET_TRAP_VERSION_OUTPUT;

enum KMDDBGRIF_ADDR_WATCH_ID
{
    KMDDBGRIF_ADDR_WATCH_ID_0 = 0,
    KMDDBGRIF_ADDR_WATCH_ID_1 = 1,
    KMDDBGRIF_ADDR_WATCH_ID_2 = 2,
    KMDDBGRIF_ADDR_WATCH_ID_3 = 3,
    KMDDBGRIF_ADDR_WATCH_ID_MAX = 0xFFFF
};

enum KMDDBGRIF_ADDR_WATCH_MODE
{
    KMDDBGRIF_ADDR_WATCH_MODE_READ = 0,
    KMDDBGRIF_ADDR_WATCH_MODE_NONREAD = 1,
    KMDDBGRIF_ADDR_WATCH_MODE_ATOMIC = 2,
    KMDDBGRIF_ADDR_WATCH_MODE_ALL = 3,
    KMDDBGRIF_ADDR_WATCH_MODE_MAX = 0xFFFF
};

typedef struct _KMDDBGRIF_SET_ADDR_WATCH_INPUT
{
    KMDDBGRIF_ADDR_WATCH_ID                            watchId;
    KMDDBGRIF_ADDR_WATCH_MODE                          mode;
    D3DKMT_ALIGN64 ULONG64                             watchAddr;
    ULONG                                              watchAddrMask;
    ULONG                                              Reserved[11];
} KMDDBGRIF_SET_ADDR_WATCH_INPUT, * PKMDDBGRIF_SET_ADDR_WATCH_INPUT;

typedef struct _KMDDBGRIF_SET_ADDR_WATCH_OUTPUT
{
    ULONG                                              Reserved[16];
} KMDDBGRIF_SET_ADDR_WATCH_OUTPUT, * PKMDDBGRIF_SET_ADDR_WATCH_OUTPUT;

typedef struct _KMDDBGRIF_CLEAR_ADDR_WATCH_INPUT
{
    KMDDBGRIF_ADDR_WATCH_ID                            watchId;
    ULONG                                              Reserved[15];
} KMDDBGRIF_CLEAR_ADDR_WATCH_INPUT, * PKMDDBGRIF_CLEAR_ADDR_WATCH_INPUT;

typedef struct _KMDDBGRIF_CLEAR_ADDR_WATCH_OUTPUT
{
    ULONG                                              Reserved[16];
} KMDDBGRIF_CLEAR_ADDR_WATCH_OUTPUT, * PKMDDBGRIF_CLEAR_ADDR_WATCH_OUTPUT;

typedef struct _KMDDBGRIF_SET_PRECISE_MEMOPS_INPUT
{
    BOOL                                               enable;
    ULONG                                              Reserved[15];
} KMDDBGRIF_SET_PRECISE_MEMOPS_INPUT, * PKMDDBGRIF_SET_PRECISE_MEMOPS_INPUT;

typedef struct _KMDDBGRIF_SET_PRECISE_MEMOPS_OUTPUT
{
    ULONG                                              Reserved[16];
} KMDDBGRIF_SET_PRECISE_MEMOPS_OUTPUT, * PKMDDBGRIF_SET_PRECISE_MEMOPS_OUTPUT;

struct KMDDBGR_DEVICE_INFO
{
    ULONG64                                            deviceExceptionMask;
    D3DKMT_ALIGN64 ULONG64                             ldsBase;
    D3DKMT_ALIGN64 ULONG64                             ldsLimit;
    D3DKMT_ALIGN64 ULONG64                             scratchBase;
    D3DKMT_ALIGN64 ULONG64                             scratchLimit;
    D3DKMT_ALIGN64 ULONG64                             gpuvmBase;
    D3DKMT_ALIGN64 ULONG64                             gpuvmLimit;
    ULONG                                              locationId;
    ULONG                                              pciSegment;
    ULONG                                              vendorId;
    ULONG                                              deviceId;
    ULONG                                              subVendorId;
    ULONG                                              subDeviceId;
    ULONG                                              revisionId;
    ULONG                                              fwVersion;
    ULONG                                              gfxTargetVersion;
    ULONG                                              simdCount;
    ULONG                                              maxWavesPerSimd;
    ULONG                                              arrayCount;
    ULONG                                              simdArraysPerEngine;
    ULONG                                              capability;
    ULONG                                              debugProp;
    ULONG                                              Reserved[3];
};

typedef struct _KMDDBGRIF_GET_DEVICE_INFO_INPUT
{
    ULONG64                                            exceptionMaskToClear; // Raised exception(s) to clear
    ULONG                                              Reserved[14];
} KMDDBGRIF_GET_DEVICE_INFO_INPUT, * PKMDDBGRIF_GET_DEVICE_INFO_INPUT;

typedef struct _KMDDBGRIF_GET_DEVICE_INFO_OUTPUT
{
    KMDDBGR_DEVICE_INFO                                deviceInfo;
    ULONG                                              Reserved[16];
} KMDDBGRIF_GET_DEVICE_INFO_OUTPUT, * PKMDDBGRIF_GET_DEVICE_INFO_OUTPUT;

typedef struct _KMDDBGRIF_SEND_EVENT_TO_INFERIOR_INPUT
{
    ULONG64                                            exceptionMask; // Exception(s) to send to the inferior's UMD
    ULONG                                              queueId; // Non-zero for queue exception(s)
    ULONG                                              Reserved[13];
} KMDDBGRIF_SEND_EVENT_TO_INFERIOR_INPUT, * PKMDDBGRIF_SEND_EVENT_TO_INFERIOR_INPUT;

typedef struct _KMDDBGRIF_SEND_EVENT_TO_INFERIOR_OUTPUT
{
    ULONG                                              Reserved[16];
} KMDDBGRIF_SEND_EVENT_TO_INFERIOR_OUTPUT, * PKMDDBGRIF_SEND_EVENT_TO_INFERIOR_OUTPUT;

typedef struct _KMDDBGRIF_READ_BUFFER_INPUT
{                  
    D3DKMT_ALIGN64 ULONG64                             srcGpuVA; // Gpu VA in the inferior's GPU VA space
    D3DKMT_ALIGN64 ULONG64                             dstCpuVA; // Host VA in debugger's VA space
    ULONG                                              size;
    ULONG                                              Reserved[11];
} KMDDBGRIF_READ_BUFFER_INPUT, * PKMDDBGRIF_READ_BUFFER_INPUT;

typedef struct _KMDDBGRIF_READ_BUFFER_OUTPUT
{
    ULONG                                              bytesRead;
    ULONG                                              Reserved[15];
} KMDDBGRIF_READ_BUFFER_OUTPUT, * PKMDDBGRIF_READ_BUFFER_OUTPUT;

typedef struct _KMDDBGRIF_WRITE_BUFFER_INPUT
{
    D3DKMT_ALIGN64 ULONG64                             srcCpuVA; // Host VA in debugger's VA space
    D3DKMT_ALIGN64 ULONG64                             dstGpuVA; // Gpu VA in the inferior's GPU VA space
    ULONG                                              size;
    ULONG                                              Reserved[11];
} KMDDBGRIF_WRITE_BUFFER_INPUT, * PKMDDBGRIF_WRITE_BUFFER_INPUT;

typedef struct _KMDDBGRIF_WRITE_BUFFER_OUTPUT
{ 
    ULONG                                              bytesWritten;
    ULONG                                              Reserved[15];
} KMDDBGRIF_WRITE_BUFFER_OUTPUT, * PKMDDBGRIF_WRITE_BUFFER_OUTPUT;

typedef struct _KMDDBGRIF_SET_USER_TRAP_INPUT
{
    D3DKMT_ALIGN64 ULONG64                             userTrapHandlerVA; // GPU VA of the second level trap handler
    D3DKMT_ALIGN64 ULONG64                             userTrapMemoryVA; // GPU VA of the second level trap memory
    unsigned int                                       Reserved[12];
} KMDDBGRIF_SET_USER_TRAP_INPUT, * PKMDDBGRIF_SET_USER_TRAP_INPUT;

typedef struct _KMDDBGRIF_SET_USER_TRAP_OUTPUT
{
    unsigned int                                       Reserved[16];
} KMDDBGRIF_SET_USER_TRAP_OUTPUT, * PKMDDBGRIF_SET_USER_TRAP_OUTPUT;

typedef struct _KMDDBGRIF_DBGR_CMDS_INPUT
{
    ULONG                                              processId; // Process id of the process being debugged
    KMD_DBGR_CMDS_OP                                   cmd; // Debugger cmd
    ULONG                                              Reserved[14];

    union
    {
        KMDDBGRIF_SET_RUNTIME_INFO_INPUT               setRuntimeInfoIn;
        KMDDBGRIF_GET_RUNTIME_INFO_INPUT               getRuntimeInfoIn;
        KMDDBGRIF_ENABLE_GLOBAL_TRAP_INPUT             enableGlobalTrapIn;
        KMDDBGRIF_ENABLE_EXCEPTIONS_INPUT              enableExceptionsIn;
        KMDDBGRIF_ENABLE_TRAPS_FOR_EXCEPTIONS_INPUT    enableTrapsForExceptionsIn;
        KMDDBGRIF_SET_WAVE_LAUNCH_MODE_INPUT           setWaveLaunchModeIn;
        KMDDBGRIF_SUSPEND_QUEUE_INPUT                  suspendQueueIn;
        KMDDBGRIF_RESUME_QUEUE_INPUT                   resumeQueueIn;
        KMDDBGRIF_GET_EXCEPTIONS_INPUT                 getExceptionsIn;
        KMDDBGRIF_GET_EXCEPTION_INFO_INPUT             getExceptionInfoIn;
        KMDDBGRIF_GET_QUEUE_INFO_INPUT                 getQueueInfoIn;
        KMDDBGRIF_GET_DEVICE_INFO_INPUT                getDeviceInfoIn;
        KMDDBGRIF_GET_TRAP_VERSION_INPUT               getTrapVersionIn;
        KMDDBGRIF_SET_ADDR_WATCH_INPUT                 setAddrWatchIn;
        KMDDBGRIF_CLEAR_ADDR_WATCH_INPUT               clearAddrWatchIn;
        KMDDBGRIF_SET_PRECISE_MEMOPS_INPUT             setPreciseMemOpsIn;
        KMDDBGRIF_SEND_EVENT_TO_INFERIOR_INPUT         sendEventToInferiorIn;
        KMDDBGRIF_READ_BUFFER_INPUT                    readBufferIn;
        KMDDBGRIF_WRITE_BUFFER_INPUT                   writeBufferIn;
        KMDDBGRIF_SET_USER_TRAP_INPUT                  setUserTrapIn;
    };
} KMDDBGRIF_DBGR_CMDS_INPUT, * PKMDDBGRIF_DBGR_CMDS_INPUT;

typedef struct _KMDDBGRIF_DBGR_CMDS_OUTPUT
{
    KMD_DBGR_CMDS_OP                                   cmd; // Debugger cmd
    ULONG                                              Reserved[15];

    union
    {
        KMDDBGRIF_SET_RUNTIME_INFO_OUTPUT              setRuntimeInfoOut;
        KMDDBGRIF_GET_RUNTIME_INFO_OUTPUT              getRuntimeInfoOut;
        KMDDBGRIF_ENABLE_GLOBAL_TRAP_OUTPUT            enableGlobalTrap;
        KMDDBGRIF_ENABLE_EXCEPTIONS_OUTPUT             enableExceptionsOut;
        KMDDBGRIF_ENABLE_TRAPS_FOR_EXCEPTIONS_OUTPUT   enableTrapsForExceptionsOut;
        KMDDBGRIF_SET_WAVE_LAUNCH_MODE_OUTPUT          setWaveLaunchModeOut;
        KMDDBGRIF_SUSPEND_QUEUE_OUTPUT                 suspendQueueOut;
        KMDDBGRIF_RESUME_QUEUE_OUTPUT                  resumeQueueOut;
        KMDDBGRIF_GET_DEVICE_INFO_OUTPUT               getDeviceInfoOut;
        KMDDBGRIF_GET_EXCEPTIONS_OUTPUT                getExceptionsOut;
        KMDDBGRIF_GET_EXCEPTION_INFO_OUTPUT            getExceptionInfoOut;
        KMDDBGRIF_GET_QUEUE_INFO_OUTPUT                getQueueInfoOut;
        KMDDBGRIF_GET_TRAP_VERSION_OUTPUT              getTrapVersionOut;
        KMDDBGRIF_SET_ADDR_WATCH_OUTPUT                setAddrWatchOut;
        KMDDBGRIF_CLEAR_ADDR_WATCH_OUTPUT              clearAddrWatchOut;
        KMDDBGRIF_SET_PRECISE_MEMOPS_OUTPUT            setPreciseMemOpsOut;
        KMDDBGRIF_SEND_EVENT_TO_INFERIOR_OUTPUT        sendEventToInferiorOut;
        KMDDBGRIF_READ_BUFFER_OUTPUT                   readBufferOut;
        KMDDBGRIF_WRITE_BUFFER_OUTPUT                  writeBufferOut;
        KMDDBGRIF_SET_USER_TRAP_OUTPUT                 setUserTrapOut;
    };
} KMDDBGRIF_DBGR_CMDS_OUTPUT, * PKMDDBGRIF_DBGR_CMDS_OUTPUT;

typedef struct _KMDDBGRIF_DBGR_CMDS
{
    ATI_LHESCAPE_HDR                                   Header;
    ULONG                                              InputSize;
    KMDDBGRIF_DBGR_CMDS_INPUT                          Input;
    ULONG                                              OutputSize;
    KMDDBGRIF_DBGR_CMDS_OUTPUT                         Output;
} KMDDBGRIF_DBGR_CMDS, * PKMDDBGRIF_DBGR_CMDS;

//
// Proxy QueryAdapterInfo/Escape header structure version info
//

#ifndef _PROXY_PROXYUMKMIF_H_

#define PXDRV_HEADER_VERSION_MAJOR     0x0001
#define PXDRV_HEADER_VERSION_MINOR     0x0002
#define PXDRV_HEADER_VERSION           ((PXDRV_HEADER_VERSION_MAJOR<<16) | PXDRV_HEADER_VERSION_MINOR)

//
// Enumerate the various drivers loadable by the proxy
//

typedef enum {
    AdapterDiscreteDriver = 0,
    AdapterIntegratedDriver,
    AdapterProxyDriver,
    AdapterMaxDriverId,
    AdapterForceUINT = 0xffffffff,
    AdapterInvalid = AdapterForceUINT
} AdapterDriverId;

// To support linked adapter, we define here the maximum number 
// of GPUs in a link chain supported by the proxy drivers. 
#define PROXY_MAX_GPUS_LDA    4

// The following enum defines the possible operating modes of the proxy driver

typedef enum {
    ProxyModePowerXpress            = 0,
    ProxyModeMultiAdapter           = 1,
    ProxyModeCrossfire              = 2,
    ProxyModePowerXpress_Dynamic    = 3,
    ProxyModePowerXpress_Dynamic_CF = 4,
    ProxyModePowerXpress_Fixed      = ProxyModePowerXpress,
    ProxyModeWsMgpuSls              = 5,
    ProxyModeForceUINT              = 0xffffffff
} ProxyMode;

// The following enum defines the possible crossfire compatibilities

typedef enum
{
    ProxyCFUnknown                   = 0,
    ProxyCFSymmetric                 = 1,
    ProxyCFHybrid                    = 2,
    ProxyCFIncompatibleAsymmetricAFR = 3,
    ProxyCFCompatibleAsymmetricAFR   = 4,
    ProxyCFForceUINT                 = 0xffffffff
} ProxyCFCompatibility;

// "Dynamic Power States" to be reported to CFX 
typedef enum
{
    DynamicPowerState_Invalid             = 0,
    DynamicPowerState_PowerSaving         = 1,
    DynamicPowerState_Balanced            = 2,
    DynamicPowerState_HighPerformance     = 3,
    DynamicPowerState_ExtremePerformance  = 4
} DynamicPowerStates;

// WDDM version. We should use the same value as DXGK_WDDMVERSION
typedef enum 
{
    WDDM_INVALID    = 0,
    WDDM_V1         = 0x1000,
    WDDM_V1_2       = 0x1200,
    WDDM_V1_3       = 0x1300,
    WDDM_V2         = 0x2000,
    WDDM_V2_1       = 0x2100,
    WDDM_V2_1_5     = 0x2105,
    WDDM_V2_1_6     = 0x2106,
    WDDM_V2_2       = 0x2200,
    WDDM_V2_3       = 0x2300,
    WDDM_V2_4       = 0x2400,
    WDDM_V2_5       = 0x2500,
    WDDM_V2_6       = 0x2600,
    WDDM_V2_7       = 0x2700,
    WDDM_V2_8       = 0x2800,
    WDDM_V2_9       = 0x2900,
    WDDM_V3_0       = 0x3000,
    WDDM_V3_1       = 0x3100
} WddmVersion;

//
// Wrapper header structures for QueryAdapterInfoCb and EscapeCb calls
// to allow targeting either vendor or proxy drivers.
//

typedef struct _PROXY_ADAPTER_INFO
{
    UINT                   version;                  // version of the interface
    ProxyMode              proxyMode;                // operating mode of the proxy drivers
    UINT                   numChainedGpus;           // number of GPU's per LDA chain, one (=1) for PX, multi-adapter

    USHORT                 VendorID[PROXY_MAX_GPUS_LDA]; // PCI ID parameters
    USHORT                 DeviceID[PROXY_MAX_GPUS_LDA];                 
    USHORT                 SubSystemID[PROXY_MAX_GPUS_LDA];
    USHORT                 SubVendorID[PROXY_MAX_GPUS_LDA];
    AdapterDriverId        adapterDriverId[PROXY_MAX_GPUS_LDA];      // Marks the driver to be targeted 

    USHORT                 vendorPath[PROXY_MAX_GPUS_LDA][MAX_PATH]; // registry path to the driver instances' entries

    union
    {
        struct
        {
            DWORD p2pInvHeapSupported                        : 1;    // P2P blts to the invisible heap are supported
            DWORD highPerfGpuAvail                           : 1;    // whether highPerfGpu is valid
            DWORD powerSaveGpuAvail                          : 1;    // whether powerSaveGpu is valid
            DWORD forceVideoPlaybackToIntegrated             : 1;    // force video playback to integrated Gpu
            DWORD extendedBatteryMode                        : 1;    // PX 5.0: whether Extended Battery Mode is enabled
            DWORD suppressPerformanceGPUCaps                 : 1;    // PX 5.0: expose only Power Saving GPU DX capabilities if different from capabilities of the High Performance GPU
            DWORD primarySurfaceLocalDisplaySupported        : 1;    // Primary surface can be displayed by local GPU
            DWORD mgpuSlsSupported                           : 1;    // MGPU SLS is supported
			DWORD primarySurfaceLocalDisplayActive           : 1;    // Local Display is currently active.      
            DWORD wddmVersionAvail                           : 1;    // WDDM version is available
            DWORD detachableGfx                              : 1;    // detachable gfx config, if this bit is set
            DWORD p2pSupportDisplay                          : 1;    // whether display surfaces can be transferred over P2P
            DWORD p2pSupportEncode                           : 1;    // whether encoding HW can access FB memory of remote GPU in chain
            DWORD p2pSupportDecode                           : 1;    // whether decoding HW can access FB memory of remote GPU in chain
            DWORD p2pSupportTmz                              : 1;    // whether protected content can be transferred over P2P
            DWORD p2pCrossGPUCoherency                       : 1;    // whether remote FB memory can be accessed without need for cache flush
            DWORD reserved : 16;
        };
        DWORD u32All;
    } flags;

    UINT	               familyId[PROXY_MAX_GPUS_LDA];
    BYTE                   p2pWriteInfo[PROXY_MAX_GPUS_LDA];    // Specifies GPUs that have P2P write support
    UINT                   chipRevision[PROXY_MAX_GPUS_LDA];
    UINT                   PXDefaultGpu;    // The index of the GPU where unknown application should start
    UINT                   defaultFrameRatio[PROXY_MAX_GPUS_LDA]; // For asymmetric xfire, number
                                                                  // of frames that should be 
                                                                  // rendered on each GPU.

    ProxyCFCompatibility   crossfireCompatibility;    // symmetric, hybrid, asymmetric

    UINT                   highPerfGpu;               // The index of faster GPU
    UINT                   powerSaveGpu;              // The index of GPU which uses the least power

    union SharedMiniSegmentOptions
    {
        struct
        {
            DWORD DX9Supported        : 1;    // DX9 utilizes A+I shared mini-segment
            DWORD DX10PlusSupported   : 1;    // DX10+ utilizes A+I shared mini-segment
            DWORD OGLSupported        : 1;    // OGL utilizes A+I shared mini-segment
            DWORD DisableAutoShrink   : 1;    // Disables Auto Shrink if not enough memory is available
            DWORD reserved            : 28;
        };
        DWORD u32All;
    } APlusISharedSegmentOptions;

    DynamicPowerStates     SDGPowerState;
    UINT                   checkSum;                 // checksum for entire PPROXY_ADAPTER_INFO

    WddmVersion            wddmVer;

    BYTE                   p2pReadInfo[PROXY_MAX_GPUS_LDA];     // Specifies GPUs that have P2P read support
    BYTE                   p2pTopology[PROXY_MAX_GPUS_LDA][PROXY_MAX_GPUS_LDA];  // 2D adjacency matrix denoting number of hops between adapters

    UINT                   Reserved[37]; 

    UINT                   privateDataLengthBytes;   // Length of the vendor driver buffer

} PROXY_ADAPTER_INFO, *PPROXY_ADAPTER_INFO;

//
// Header for DxEscape() calls. A checksum is introduced to detect modifications done to the buffer by test tools
// e.g. DTM CRASH  
//

typedef struct _PROXY_ESCAPE_INFO
{
    AdapterDriverId        adapterDriverId;          // Marks the driver to be targeted 
    UINT                   version;
    UINT                   checksum;                 // checksum for entire escape private data buffer
    UINT                   gpuOrdinal;               // vendor driver GPU ID that needs to process the command
    UINT                   Reserved[14];             // for future extension
    UINT                   privateDataLengthBytes;   // Length of the vendor driver buffer
} PROXY_ESCAPE_INFO, *PPROXY_ESCAPE_INFO;

#endif // _PROXY_PROXYUMKMIF_H_

#pragma pack(pop)

#if defined(_cplusplus)
}
#endif //  _cplusplus

#endif  //  _KMDDBG_H_
