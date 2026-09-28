// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//go:build linux && cgo

// Package amdsmi provides read-only Linux GPU queries through AMD SMI 27.1 and CGO.
// Calls are serialized. Balance Init with ShutDown and rediscover handles after final shutdown.
package amdsmi

/*
#cgo LDFLAGS: -lamd_smi
#include <stddef.h>
#include <amd_smi/amdsmi.h>
// CGO reads high-bit enum constants as signed unless their expression has an unsigned type.
#define AMDSMI_GPU_BLOCK_RESERVED ((unsigned long long)AMDSMI_GPU_BLOCK_RESERVED)

static size_t go_amdsmi_string_length(const char *p, size_t capacity) {
    size_t n = 0;
    if (p == NULL) return 0;
    while (n < capacity && p[n] != '\0') ++n;
    return n;
}

static amdsmi_status_t go_amdsmi_get_bdf(amdsmi_processor_handle h,
    uint64_t *domain, uint8_t *bus, uint8_t *device, uint8_t *function) {
    amdsmi_bdf_t out = {0};
    amdsmi_status_t status = amdsmi_get_gpu_device_bdf(h, &out);
    if (status == AMDSMI_STATUS_SUCCESS) {
        *domain = out.bdf.domain_number;
        *bus = out.bdf.bus_number;
        *device = out.bdf.device_number;
        *function = out.bdf.function_number;
    }
    return status;
}

static amdsmi_status_t go_amdsmi_lookup_bdf(uint64_t domain, uint8_t bus,
    uint8_t device, uint8_t function, amdsmi_processor_handle *out) {
    amdsmi_bdf_t bdf = {0};
    bdf.bdf.domain_number = domain;
    bdf.bdf.bus_number = bus;
    bdf.bdf.device_number = device;
    bdf.bdf.function_number = function;
    return amdsmi_get_processor_handle_from_bdf(bdf, out);
}

static uint32_t go_amdsmi_nps_mask(amdsmi_nps_caps_t value) {
	return value.nps_cap_mask;
}
*/
import "C"

import (
	"fmt"
	"sync"
	"unsafe"
)

const nativeStringCapacity int = C.AMDSMI_MAX_STRING_LENGTH

func boundedString(p *C.char, capacity int) string {
	if p == nil || capacity <= 0 {
		return ""
	}
	n := C.go_amdsmi_string_length(p, C.size_t(capacity))
	return C.GoStringN(p, C.int(n))
}

type StatusCode uint32

type Error struct {
	Op      string
	Code    StatusCode
	Message string
}

func (s StatusCode) Error() string {
	return fmt.Sprintf("AMD SMI status %d (0x%08x)", uint32(s), uint32(s))
}

func (e *Error) Error() string {
	return fmt.Sprintf("%s: %s: %s", e.Op, e.Code.Error(), e.Message)
}

func (e *Error) Unwrap() error { return e.Code }

func statusStringLocked(code StatusCode) string {
	var text *C.char
	status := C.amdsmi_status_code_to_string(C.amdsmi_status_t(code), &text)
	if status != C.AMDSMI_STATUS_SUCCESS || text == nil {
		return code.Error()
	}
	return boundedString(text, int(C.AMDSMI_MAX_STRING_LENGTH))
}

func StatusString(code StatusCode) string {
	nativeState.mu.Lock()
	defer nativeState.mu.Unlock()
	return statusStringLocked(code)
}

func nativeErrorLocked(op string, status C.amdsmi_status_t) error {
	if status == C.AMDSMI_STATUS_SUCCESS {
		return nil
	}
	code := StatusCode(status)
	return &Error{Op: op, Code: code, Message: statusStringLocked(code)}
}

func checkedCount(op string, count uint64, capacity int) (int, error) {
	if count > uint64(capacity) {
		return 0, &Error{
			Op: op, Code: AMDSMI_STATUS_UNEXPECTED_SIZE,
			Message: fmt.Sprintf("count %d exceeds capacity %d", count, capacity),
		}
	}
	return int(count), nil
}

const (
	AMDSMI_STATUS_SUCCESS             StatusCode = C.AMDSMI_STATUS_SUCCESS
	AMDSMI_STATUS_INVAL               StatusCode = C.AMDSMI_STATUS_INVAL
	AMDSMI_STATUS_NOT_SUPPORTED       StatusCode = C.AMDSMI_STATUS_NOT_SUPPORTED
	AMDSMI_STATUS_NOT_YET_IMPLEMENTED StatusCode = C.AMDSMI_STATUS_NOT_YET_IMPLEMENTED
	AMDSMI_STATUS_FAIL_LOAD_MODULE    StatusCode = C.AMDSMI_STATUS_FAIL_LOAD_MODULE
	AMDSMI_STATUS_FAIL_LOAD_SYMBOL    StatusCode = C.AMDSMI_STATUS_FAIL_LOAD_SYMBOL
	AMDSMI_STATUS_DRM_ERROR           StatusCode = C.AMDSMI_STATUS_DRM_ERROR
	AMDSMI_STATUS_API_FAILED          StatusCode = C.AMDSMI_STATUS_API_FAILED
	AMDSMI_STATUS_TIMEOUT             StatusCode = C.AMDSMI_STATUS_TIMEOUT
	AMDSMI_STATUS_RETRY               StatusCode = C.AMDSMI_STATUS_RETRY
	AMDSMI_STATUS_NO_PERM             StatusCode = C.AMDSMI_STATUS_NO_PERM
	AMDSMI_STATUS_INTERRUPT           StatusCode = C.AMDSMI_STATUS_INTERRUPT
	AMDSMI_STATUS_IO                  StatusCode = C.AMDSMI_STATUS_IO
	AMDSMI_STATUS_ADDRESS_FAULT       StatusCode = C.AMDSMI_STATUS_ADDRESS_FAULT
	AMDSMI_STATUS_FILE_ERROR          StatusCode = C.AMDSMI_STATUS_FILE_ERROR
	AMDSMI_STATUS_OUT_OF_RESOURCES    StatusCode = C.AMDSMI_STATUS_OUT_OF_RESOURCES
	AMDSMI_STATUS_INTERNAL_EXCEPTION  StatusCode = C.AMDSMI_STATUS_INTERNAL_EXCEPTION
	AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS StatusCode = C.AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS
	AMDSMI_STATUS_INIT_ERROR          StatusCode = C.AMDSMI_STATUS_INIT_ERROR
	AMDSMI_STATUS_REFCOUNT_OVERFLOW   StatusCode = C.AMDSMI_STATUS_REFCOUNT_OVERFLOW
	AMDSMI_STATUS_DIRECTORY_NOT_FOUND StatusCode = C.AMDSMI_STATUS_DIRECTORY_NOT_FOUND
	AMDSMI_STATUS_IPC_ERROR           StatusCode = C.AMDSMI_STATUS_IPC_ERROR
	AMDSMI_STATUS_BUSY                StatusCode = C.AMDSMI_STATUS_BUSY
	AMDSMI_STATUS_NOT_FOUND           StatusCode = C.AMDSMI_STATUS_NOT_FOUND
	AMDSMI_STATUS_NOT_INIT            StatusCode = C.AMDSMI_STATUS_NOT_INIT
	AMDSMI_STATUS_NO_SLOT             StatusCode = C.AMDSMI_STATUS_NO_SLOT
	AMDSMI_STATUS_DRIVER_NOT_LOADED   StatusCode = C.AMDSMI_STATUS_DRIVER_NOT_LOADED
	AMDSMI_STATUS_MORE_DATA           StatusCode = C.AMDSMI_STATUS_MORE_DATA
	AMDSMI_STATUS_NO_DATA             StatusCode = C.AMDSMI_STATUS_NO_DATA
	AMDSMI_STATUS_INSUFFICIENT_SIZE   StatusCode = C.AMDSMI_STATUS_INSUFFICIENT_SIZE
	AMDSMI_STATUS_UNEXPECTED_SIZE     StatusCode = C.AMDSMI_STATUS_UNEXPECTED_SIZE
	AMDSMI_STATUS_UNEXPECTED_DATA     StatusCode = C.AMDSMI_STATUS_UNEXPECTED_DATA
	AMDSMI_STATUS_NON_AMD_CPU         StatusCode = C.AMDSMI_STATUS_NON_AMD_CPU
	AMDSMI_STATUS_NO_ENERGY_DRV       StatusCode = C.AMDSMI_STATUS_NO_ENERGY_DRV
	AMDSMI_STATUS_NO_MSR_DRV          StatusCode = C.AMDSMI_STATUS_NO_MSR_DRV
	AMDSMI_STATUS_NO_HSMP_DRV         StatusCode = C.AMDSMI_STATUS_NO_HSMP_DRV
	AMDSMI_STATUS_NO_HSMP_SUP         StatusCode = C.AMDSMI_STATUS_NO_HSMP_SUP
	AMDSMI_STATUS_NO_HSMP_MSG_SUP     StatusCode = C.AMDSMI_STATUS_NO_HSMP_MSG_SUP
	AMDSMI_STATUS_HSMP_TIMEOUT        StatusCode = C.AMDSMI_STATUS_HSMP_TIMEOUT
	AMDSMI_STATUS_NO_DRV              StatusCode = C.AMDSMI_STATUS_NO_DRV
	AMDSMI_STATUS_FILE_NOT_FOUND      StatusCode = C.AMDSMI_STATUS_FILE_NOT_FOUND
	AMDSMI_STATUS_ARG_PTR_NULL        StatusCode = C.AMDSMI_STATUS_ARG_PTR_NULL
	AMDSMI_STATUS_AMDGPU_RESTART_ERR  StatusCode = C.AMDSMI_STATUS_AMDGPU_RESTART_ERR
	AMDSMI_STATUS_SETTING_UNAVAILABLE StatusCode = C.AMDSMI_STATUS_SETTING_UNAVAILABLE
	AMDSMI_STATUS_CORRUPTED_EEPROM    StatusCode = C.AMDSMI_STATUS_CORRUPTED_EEPROM
	AMDSMI_STATUS_MAP_ERROR           StatusCode = C.AMDSMI_STATUS_MAP_ERROR
	AMDSMI_STATUS_UNKNOWN_ERROR       StatusCode = C.AMDSMI_STATUS_UNKNOWN_ERROR
)

// ProcessorHandle is C-owned and expires after this package's final ShutDown.
type ProcessorHandle struct {
	ptr        unsafe.Pointer
	generation uint64
}

var nativeState = struct {
	mu         sync.Mutex
	refs       uint32
	generation uint64
}{generation: 1}

type InitFlags uint64

const InitAMDGPUs InitFlags = C.AMDSMI_INIT_AMD_GPUS

// Init accepts only InitAMDGPUs. Balance each successful call with ShutDown.
func Init(flags InitFlags) error {
	nativeState.mu.Lock()
	defer nativeState.mu.Unlock()
	if flags != InitAMDGPUs {
		return &Error{Op: "amdsmi_init", Code: AMDSMI_STATUS_INVAL,
			Message: "only InitAMDGPUs is supported"}
	}
	if nativeState.refs == 1<<31-1 {
		return &Error{Op: "amdsmi_init", Code: AMDSMI_STATUS_REFCOUNT_OVERFLOW,
			Message: "initialization reference limit reached"}
	}
	status := C.amdsmi_init(C.uint64_t(flags))
	if err := nativeErrorLocked("amdsmi_init", status); err != nil {
		return err
	}
	nativeState.refs++
	return nil
}

func ShutDown() error {
	nativeState.mu.Lock()
	defer nativeState.mu.Unlock()
	if nativeState.refs == 0 {
		return &Error{Op: "amdsmi_shut_down", Code: AMDSMI_STATUS_NOT_INIT,
			Message: "no initialization reference is held by this package"}
	}
	status := C.amdsmi_shut_down()
	nativeState.refs--
	if nativeState.refs == 0 {
		nativeState.generation++
	}
	return nativeErrorLocked("amdsmi_shut_down", status)
}

func withLibrary[T any](op string, fn func() (T, error)) (T, error) {
	nativeState.mu.Lock()
	defer nativeState.mu.Unlock()
	if nativeState.refs == 0 {
		var zero T
		return zero, &Error{Op: op, Code: AMDSMI_STATUS_NOT_INIT,
			Message: "call Init before querying processors"}
	}
	return fn()
}

func withProcessor[T any](h ProcessorHandle, op string,
	fn func(C.amdsmi_processor_handle) (T, error)) (T, error) {
	return withLibrary(op, func() (T, error) {
		if h.ptr == nil || h.generation != nativeState.generation {
			var zero T
			return zero, &Error{Op: op, Code: AMDSMI_STATUS_INVAL,
				Message: "zero or expired processor handle"}
		}
		return fn(C.amdsmi_processor_handle(h.ptr))
	})
}

const (
	compiledMajor   uint32 = C.AMDSMI_LIB_VERSION_MAJOR
	compiledMinor   uint32 = C.AMDSMI_LIB_VERSION_MINOR
	compiledRelease uint32 = C.AMDSMI_LIB_VERSION_RELEASE
)

type Version struct {
	Major   uint32
	Minor   uint32
	Release uint32
	Build   string
}

// GetLibraryVersion does not require initialization or access any processors.
func GetLibraryVersion() (Version, error) {
	nativeState.mu.Lock()
	defer nativeState.mu.Unlock()
	var out C.amdsmi_version_t
	if err := nativeErrorLocked("amdsmi_get_lib_version", C.amdsmi_get_lib_version(&out)); err != nil {
		return Version{}, err
	}
	return Version{Major: uint32(out.major), Minor: uint32(out.minor),
		Release: uint32(out.release), Build: boundedString(out.build, nativeStringCapacity)}, nil
}

func fetchHandlesLocked(op string,
	fetch func(*C.uint32_t, *unsafe.Pointer) C.amdsmi_status_t) ([]unsafe.Pointer, error) {
	var count C.uint32_t
	if err := nativeErrorLocked(op, fetch(&count, nil)); err != nil {
		return nil, err
	}
	if count == 0 {
		return []unsafe.Pointer{}, nil
	}
	maxInt := int(^uint(0) >> 1)
	capacity, err := checkedCount(op, uint64(count),
		maxInt/int(unsafe.Sizeof(unsafe.Pointer(nil))))
	if err != nil {
		return nil, err
	}
	pointers := make([]unsafe.Pointer, capacity)
	if err := nativeErrorLocked(op, fetch(&count, &pointers[0])); err != nil {
		return nil, err
	}
	n, err := checkedCount(op, uint64(count), len(pointers))
	if err != nil {
		return nil, err
	}
	return pointers[:n], nil
}

func GetProcessorHandles() ([]ProcessorHandle, error) {
	return withLibrary("amdsmi_get_socket_handles", getProcessorHandlesLocked)
}

// GetProcessorHandleFromIndex uses the current filtered GPU discovery order.
func GetProcessorHandleFromIndex(index uint32) (ProcessorHandle, error) {
	return withLibrary("amdsmi_get_socket_handles", func() (ProcessorHandle, error) {
		handles, err := getProcessorHandlesLocked()
		if err != nil {
			return ProcessorHandle{}, err
		}
		if uint64(index) >= uint64(len(handles)) {
			return ProcessorHandle{}, &Error{Op: "GetProcessorHandleFromIndex",
				Code: AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS, Message: "processor index is out of bounds"}
		}
		return handles[index], nil
	})
}

func getProcessorHandlesLocked() ([]ProcessorHandle, error) {
	sockets, err := fetchHandlesLocked("amdsmi_get_socket_handles",
		func(n *C.uint32_t, p *unsafe.Pointer) C.amdsmi_status_t {
			return C.amdsmi_get_socket_handles(n,
				(*C.amdsmi_socket_handle)(unsafe.Pointer(p)))
		})
	if err != nil {
		return nil, err
	}
	handles := make([]ProcessorHandle, 0)
	for _, socket := range sockets {
		if socket == nil {
			return nil, &Error{Op: "amdsmi_get_socket_handles",
				Code: AMDSMI_STATUS_UNEXPECTED_DATA, Message: "native null socket handle"}
		}
		processors, err := fetchHandlesLocked("amdsmi_get_processor_handles",
			func(n *C.uint32_t, p *unsafe.Pointer) C.amdsmi_status_t {
				return C.amdsmi_get_processor_handles(C.amdsmi_socket_handle(socket), n,
					(*C.amdsmi_processor_handle)(unsafe.Pointer(p)))
			})
		if err != nil {
			return nil, err
		}
		for _, processor := range processors {
			if processor == nil {
				return nil, &Error{Op: "amdsmi_get_processor_handles",
					Code: AMDSMI_STATUS_UNEXPECTED_DATA, Message: "native null handle"}
			}
			var kind C.amdsmi_processor_type_t
			status := C.amdsmi_get_processor_type(C.amdsmi_processor_handle(processor), &kind)
			if err := nativeErrorLocked("amdsmi_get_processor_type", status); err != nil {
				return nil, err
			}
			if kind == C.AMDSMI_PROCESSOR_TYPE_AMD_GPU {
				handles = append(handles, ProcessorHandle{
					ptr: processor, generation: nativeState.generation,
				})
			}
		}
	}
	return handles, nil
}

type BDF struct {
	Domain   uint64
	Bus      uint8
	Device   uint8
	Function uint8
}

func (b BDF) String() string {
	return fmt.Sprintf("%04x:%02x:%02x.%x", b.Domain, b.Bus, b.Device, b.Function)
}

func GetGPUDeviceBDF(h ProcessorHandle) (BDF, error) {
	const op = "amdsmi_get_gpu_device_bdf"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (BDF, error) {
		var domain C.uint64_t
		var bus, device, function C.uint8_t
		status := C.go_amdsmi_get_bdf(p, &domain, &bus, &device, &function)
		if err := nativeErrorLocked(op, status); err != nil {
			return BDF{}, err
		}
		return BDF{Domain: uint64(domain), Bus: uint8(bus),
			Device: uint8(device), Function: uint8(function)}, nil
	})
}

func GetProcessorHandleFromBDF(b BDF) (ProcessorHandle, error) {
	const op = "amdsmi_get_processor_handle_from_bdf"
	return withLibrary(op, func() (ProcessorHandle, error) {
		if b.Domain > (1<<48)-1 || b.Device > 31 || b.Function > 7 {
			return ProcessorHandle{}, &Error{Op: op, Code: AMDSMI_STATUS_INVAL,
				Message: "BDF field exceeds its native width"}
		}
		var p C.amdsmi_processor_handle
		status := C.go_amdsmi_lookup_bdf(C.uint64_t(b.Domain), C.uint8_t(b.Bus),
			C.uint8_t(b.Device), C.uint8_t(b.Function), &p)
		if err := nativeErrorLocked(op, status); err != nil {
			return ProcessorHandle{}, err
		}
		if p == nil {
			return ProcessorHandle{}, &Error{Op: op, Code: AMDSMI_STATUS_UNEXPECTED_DATA,
				Message: "native lookup returned a null handle"}
		}
		var kind C.amdsmi_processor_type_t
		if err := nativeErrorLocked("amdsmi_get_processor_type",
			C.amdsmi_get_processor_type(p, &kind)); err != nil {
			return ProcessorHandle{}, err
		}
		if kind != C.AMDSMI_PROCESSOR_TYPE_AMD_GPU {
			return ProcessorHandle{}, &Error{Op: op, Code: AMDSMI_STATUS_NOT_SUPPORTED,
				Message: "BDF does not identify an AMD GPU"}
		}
		return ProcessorHandle{ptr: unsafe.Pointer(p), generation: nativeState.generation}, nil
	})
}

const gpuUUIDSize int = C.AMDSMI_GPU_UUID_SIZE

type AsicInfo struct {
	MarketName            string
	VendorID              uint32
	VendorName            string
	SubvendorID           uint32
	DeviceID              uint64
	RevID                 uint32
	AsicSerial            string
	OamID                 uint32
	NumComputeUnits       uint32
	TargetGraphicsVersion uint64
	SubsystemID           uint32
	Flags                 uint64
	PhysicalAcceleratorID uint32
	ChipRevisionID        uint32
	ExternalRevisionID    uint32
}

type DriverInfo struct {
	Version string
	Date    string
	Name    string
}

type BoardInfo struct {
	ModelNumber      string
	ProductSerial    string
	FRUID            string
	ProductName      string
	ManufacturerName string
}

type FirmwareBlock uint32

type FirmwareInfo struct {
	ID      FirmwareBlock
	Version uint64
}

type VBIOSInfo struct {
	Name         string
	BuildDate    string
	PartNumber   string
	Version      string
	BootFirmware string
}

const maxFirmwareEntries int = C.AMDSMI_FW_ID__MAX

const (
	AMDSMI_FW_ID_SMU                      FirmwareBlock = C.AMDSMI_FW_ID_SMU
	AMDSMI_FW_ID_FIRST                    FirmwareBlock = C.AMDSMI_FW_ID_FIRST
	AMDSMI_FW_ID_CP_CE                    FirmwareBlock = C.AMDSMI_FW_ID_CP_CE
	AMDSMI_FW_ID_CP_PFP                   FirmwareBlock = C.AMDSMI_FW_ID_CP_PFP
	AMDSMI_FW_ID_CP_ME                    FirmwareBlock = C.AMDSMI_FW_ID_CP_ME
	AMDSMI_FW_ID_CP_MEC_JT1               FirmwareBlock = C.AMDSMI_FW_ID_CP_MEC_JT1
	AMDSMI_FW_ID_CP_MEC_JT2               FirmwareBlock = C.AMDSMI_FW_ID_CP_MEC_JT2
	AMDSMI_FW_ID_CP_MEC1                  FirmwareBlock = C.AMDSMI_FW_ID_CP_MEC1
	AMDSMI_FW_ID_CP_MEC2                  FirmwareBlock = C.AMDSMI_FW_ID_CP_MEC2
	AMDSMI_FW_ID_RLC                      FirmwareBlock = C.AMDSMI_FW_ID_RLC
	AMDSMI_FW_ID_SDMA0                    FirmwareBlock = C.AMDSMI_FW_ID_SDMA0
	AMDSMI_FW_ID_SDMA1                    FirmwareBlock = C.AMDSMI_FW_ID_SDMA1
	AMDSMI_FW_ID_SDMA2                    FirmwareBlock = C.AMDSMI_FW_ID_SDMA2
	AMDSMI_FW_ID_SDMA3                    FirmwareBlock = C.AMDSMI_FW_ID_SDMA3
	AMDSMI_FW_ID_SDMA4                    FirmwareBlock = C.AMDSMI_FW_ID_SDMA4
	AMDSMI_FW_ID_SDMA5                    FirmwareBlock = C.AMDSMI_FW_ID_SDMA5
	AMDSMI_FW_ID_SDMA6                    FirmwareBlock = C.AMDSMI_FW_ID_SDMA6
	AMDSMI_FW_ID_SDMA7                    FirmwareBlock = C.AMDSMI_FW_ID_SDMA7
	AMDSMI_FW_ID_VCN                      FirmwareBlock = C.AMDSMI_FW_ID_VCN
	AMDSMI_FW_ID_UVD                      FirmwareBlock = C.AMDSMI_FW_ID_UVD
	AMDSMI_FW_ID_VCE                      FirmwareBlock = C.AMDSMI_FW_ID_VCE
	AMDSMI_FW_ID_ISP                      FirmwareBlock = C.AMDSMI_FW_ID_ISP
	AMDSMI_FW_ID_DMCU_ERAM                FirmwareBlock = C.AMDSMI_FW_ID_DMCU_ERAM
	AMDSMI_FW_ID_DMCU_ISR                 FirmwareBlock = C.AMDSMI_FW_ID_DMCU_ISR
	AMDSMI_FW_ID_RLC_RESTORE_LIST_GPM_MEM FirmwareBlock = C.AMDSMI_FW_ID_RLC_RESTORE_LIST_GPM_MEM
	AMDSMI_FW_ID_RLC_RESTORE_LIST_SRM_MEM FirmwareBlock = C.AMDSMI_FW_ID_RLC_RESTORE_LIST_SRM_MEM
	AMDSMI_FW_ID_RLC_RESTORE_LIST_CNTL    FirmwareBlock = C.AMDSMI_FW_ID_RLC_RESTORE_LIST_CNTL
	AMDSMI_FW_ID_RLC_V                    FirmwareBlock = C.AMDSMI_FW_ID_RLC_V
	AMDSMI_FW_ID_MMSCH                    FirmwareBlock = C.AMDSMI_FW_ID_MMSCH
	AMDSMI_FW_ID_PSP_SYSDRV               FirmwareBlock = C.AMDSMI_FW_ID_PSP_SYSDRV
	AMDSMI_FW_ID_PSP_SOSDRV               FirmwareBlock = C.AMDSMI_FW_ID_PSP_SOSDRV
	AMDSMI_FW_ID_PSP_TOC                  FirmwareBlock = C.AMDSMI_FW_ID_PSP_TOC
	AMDSMI_FW_ID_PSP_KEYDB                FirmwareBlock = C.AMDSMI_FW_ID_PSP_KEYDB
	AMDSMI_FW_ID_DFC                      FirmwareBlock = C.AMDSMI_FW_ID_DFC
	AMDSMI_FW_ID_PSP_SPL                  FirmwareBlock = C.AMDSMI_FW_ID_PSP_SPL
	AMDSMI_FW_ID_DRV_CAP                  FirmwareBlock = C.AMDSMI_FW_ID_DRV_CAP
	AMDSMI_FW_ID_MC                       FirmwareBlock = C.AMDSMI_FW_ID_MC
	AMDSMI_FW_ID_PSP_BL                   FirmwareBlock = C.AMDSMI_FW_ID_PSP_BL
	AMDSMI_FW_ID_CP_PM4                   FirmwareBlock = C.AMDSMI_FW_ID_CP_PM4
	AMDSMI_FW_ID_RLC_P                    FirmwareBlock = C.AMDSMI_FW_ID_RLC_P
	AMDSMI_FW_ID_SEC_POLICY_STAGE2        FirmwareBlock = C.AMDSMI_FW_ID_SEC_POLICY_STAGE2
	AMDSMI_FW_ID_REG_ACCESS_WHITELIST     FirmwareBlock = C.AMDSMI_FW_ID_REG_ACCESS_WHITELIST
	AMDSMI_FW_ID_IMU_DRAM                 FirmwareBlock = C.AMDSMI_FW_ID_IMU_DRAM
	AMDSMI_FW_ID_IMU_IRAM                 FirmwareBlock = C.AMDSMI_FW_ID_IMU_IRAM
	AMDSMI_FW_ID_SDMA_TH0                 FirmwareBlock = C.AMDSMI_FW_ID_SDMA_TH0
	AMDSMI_FW_ID_SDMA_TH1                 FirmwareBlock = C.AMDSMI_FW_ID_SDMA_TH1
	AMDSMI_FW_ID_CP_MES                   FirmwareBlock = C.AMDSMI_FW_ID_CP_MES
	AMDSMI_FW_ID_MES_KIQ                  FirmwareBlock = C.AMDSMI_FW_ID_MES_KIQ
	AMDSMI_FW_ID_MES_STACK                FirmwareBlock = C.AMDSMI_FW_ID_MES_STACK
	AMDSMI_FW_ID_MES_THREAD1              FirmwareBlock = C.AMDSMI_FW_ID_MES_THREAD1
	AMDSMI_FW_ID_MES_THREAD1_STACK        FirmwareBlock = C.AMDSMI_FW_ID_MES_THREAD1_STACK
	AMDSMI_FW_ID_RLX6                     FirmwareBlock = C.AMDSMI_FW_ID_RLX6
	AMDSMI_FW_ID_RLX6_DRAM_BOOT           FirmwareBlock = C.AMDSMI_FW_ID_RLX6_DRAM_BOOT
	AMDSMI_FW_ID_RS64_ME                  FirmwareBlock = C.AMDSMI_FW_ID_RS64_ME
	AMDSMI_FW_ID_RS64_ME_P0_DATA          FirmwareBlock = C.AMDSMI_FW_ID_RS64_ME_P0_DATA
	AMDSMI_FW_ID_RS64_ME_P1_DATA          FirmwareBlock = C.AMDSMI_FW_ID_RS64_ME_P1_DATA
	AMDSMI_FW_ID_RS64_PFP                 FirmwareBlock = C.AMDSMI_FW_ID_RS64_PFP
	AMDSMI_FW_ID_RS64_PFP_P0_DATA         FirmwareBlock = C.AMDSMI_FW_ID_RS64_PFP_P0_DATA
	AMDSMI_FW_ID_RS64_PFP_P1_DATA         FirmwareBlock = C.AMDSMI_FW_ID_RS64_PFP_P1_DATA
	AMDSMI_FW_ID_RS64_MEC                 FirmwareBlock = C.AMDSMI_FW_ID_RS64_MEC
	AMDSMI_FW_ID_RS64_MEC_P0_DATA         FirmwareBlock = C.AMDSMI_FW_ID_RS64_MEC_P0_DATA
	AMDSMI_FW_ID_RS64_MEC_P1_DATA         FirmwareBlock = C.AMDSMI_FW_ID_RS64_MEC_P1_DATA
	AMDSMI_FW_ID_RS64_MEC_P2_DATA         FirmwareBlock = C.AMDSMI_FW_ID_RS64_MEC_P2_DATA
	AMDSMI_FW_ID_RS64_MEC_P3_DATA         FirmwareBlock = C.AMDSMI_FW_ID_RS64_MEC_P3_DATA
	AMDSMI_FW_ID_PPTABLE                  FirmwareBlock = C.AMDSMI_FW_ID_PPTABLE
	AMDSMI_FW_ID_PSP_SOC                  FirmwareBlock = C.AMDSMI_FW_ID_PSP_SOC
	AMDSMI_FW_ID_PSP_DBG                  FirmwareBlock = C.AMDSMI_FW_ID_PSP_DBG
	AMDSMI_FW_ID_PSP_INTF                 FirmwareBlock = C.AMDSMI_FW_ID_PSP_INTF
	AMDSMI_FW_ID_RLX6_CORE1               FirmwareBlock = C.AMDSMI_FW_ID_RLX6_CORE1
	AMDSMI_FW_ID_RLX6_DRAM_BOOT_CORE1     FirmwareBlock = C.AMDSMI_FW_ID_RLX6_DRAM_BOOT_CORE1
	AMDSMI_FW_ID_RLCV_LX7                 FirmwareBlock = C.AMDSMI_FW_ID_RLCV_LX7
	AMDSMI_FW_ID_RLC_SAVE_RESTORE_LIST    FirmwareBlock = C.AMDSMI_FW_ID_RLC_SAVE_RESTORE_LIST
	AMDSMI_FW_ID_ASD                      FirmwareBlock = C.AMDSMI_FW_ID_ASD
	AMDSMI_FW_ID_TA_RAS                   FirmwareBlock = C.AMDSMI_FW_ID_TA_RAS
	AMDSMI_FW_ID_TA_XGMI                  FirmwareBlock = C.AMDSMI_FW_ID_TA_XGMI
	AMDSMI_FW_ID_RLC_SRLG                 FirmwareBlock = C.AMDSMI_FW_ID_RLC_SRLG
	AMDSMI_FW_ID_RLC_SRLS                 FirmwareBlock = C.AMDSMI_FW_ID_RLC_SRLS
	AMDSMI_FW_ID_PM                       FirmwareBlock = C.AMDSMI_FW_ID_PM
	AMDSMI_FW_ID_DMCU                     FirmwareBlock = C.AMDSMI_FW_ID_DMCU
	AMDSMI_FW_ID_PLDM_BUNDLE              FirmwareBlock = C.AMDSMI_FW_ID_PLDM_BUNDLE
	AMDSMI_FW_ID__MAX                     FirmwareBlock = C.AMDSMI_FW_ID__MAX
)

func GetUUID(h ProcessorHandle) (string, error) {
	const op = "amdsmi_get_gpu_device_uuid"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (string, error) {
		var out [C.AMDSMI_GPU_UUID_SIZE]C.char
		size := C.uint(len(out))
		if err := nativeErrorLocked(op, C.amdsmi_get_gpu_device_uuid(p, &size, &out[0])); err != nil {
			return "", err
		}
		n, err := checkedCount(op, uint64(size), len(out))
		if err != nil {
			return "", err
		}
		return boundedString(&out[0], n), nil
	})
}

func GetGPUAsicInfo(h ProcessorHandle) (AsicInfo, error) {
	const op = "amdsmi_get_gpu_asic_info"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (AsicInfo, error) {
		var out C.amdsmi_asic_info_t
		if err := nativeErrorLocked(op, C.amdsmi_get_gpu_asic_info(p, &out)); err != nil {
			return AsicInfo{}, err
		}
		return AsicInfo{
			MarketName:  boundedString(&out.market_name[0], len(out.market_name)),
			VendorID:    uint32(out.vendor_id),
			VendorName:  boundedString(&out.vendor_name[0], len(out.vendor_name)),
			SubvendorID: uint32(out.subvendor_id), DeviceID: uint64(out.device_id),
			RevID:      uint32(out.rev_id),
			AsicSerial: boundedString(&out.asic_serial[0], len(out.asic_serial)),
			OamID:      uint32(out.oam_id), NumComputeUnits: uint32(out.num_of_compute_units),
			TargetGraphicsVersion: uint64(out.target_graphics_version),
			SubsystemID:           uint32(out.subsystem_id), Flags: uint64(out.flags),
			PhysicalAcceleratorID: uint32(out.physical_acc_id),
			ChipRevisionID:        uint32(out.chip_rev_id), ExternalRevisionID: uint32(out.external_rev_id),
		}, nil
	})
}

func GetDriverInfo(h ProcessorHandle) (DriverInfo, error) {
	const op = "amdsmi_get_gpu_driver_info"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (DriverInfo, error) {
		var out C.amdsmi_driver_info_t
		if err := nativeErrorLocked(op, C.amdsmi_get_gpu_driver_info(p, &out)); err != nil {
			return DriverInfo{}, err
		}
		return DriverInfo{
			Version: boundedString(&out.driver_version[0], len(out.driver_version)),
			Date:    boundedString(&out.driver_date[0], len(out.driver_date)),
			Name:    boundedString(&out.driver_name[0], len(out.driver_name)),
		}, nil
	})
}

func GetBoardInfo(h ProcessorHandle) (BoardInfo, error) {
	const op = "amdsmi_get_gpu_board_info"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (BoardInfo, error) {
		var out C.amdsmi_board_info_t
		if err := nativeErrorLocked(op, C.amdsmi_get_gpu_board_info(p, &out)); err != nil {
			return BoardInfo{}, err
		}
		return BoardInfo{
			ModelNumber:      boundedString(&out.model_number[0], len(out.model_number)),
			ProductSerial:    boundedString(&out.product_serial[0], len(out.product_serial)),
			FRUID:            boundedString(&out.fru_id[0], len(out.fru_id)),
			ProductName:      boundedString(&out.product_name[0], len(out.product_name)),
			ManufacturerName: boundedString(&out.manufacturer_name[0], len(out.manufacturer_name)),
		}, nil
	})
}

func GetFirmwareInfo(h ProcessorHandle) ([]FirmwareInfo, error) {
	const op = "amdsmi_get_fw_info"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) ([]FirmwareInfo, error) {
		var out C.amdsmi_fw_info_t
		if err := nativeErrorLocked(op, C.amdsmi_get_fw_info(p, &out)); err != nil {
			return nil, err
		}
		n, err := checkedCount(op, uint64(out.num_fw_info), len(out.fw_info_list))
		if err != nil {
			return nil, err
		}
		result := make([]FirmwareInfo, n)
		for i := range result {
			result[i] = FirmwareInfo{ID: FirmwareBlock(out.fw_info_list[i].fw_id),
				Version: uint64(out.fw_info_list[i].fw_version)}
		}
		return result, nil
	})
}

func GetVBIOSInfo(h ProcessorHandle) (VBIOSInfo, error) {
	const op = "amdsmi_get_gpu_vbios_info"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (VBIOSInfo, error) {
		var out C.amdsmi_vbios_info_t
		if err := nativeErrorLocked(op, C.amdsmi_get_gpu_vbios_info(p, &out)); err != nil {
			return VBIOSInfo{}, err
		}
		return VBIOSInfo{
			Name:         boundedString(&out.name[0], len(out.name)),
			BuildDate:    boundedString(&out.build_date[0], len(out.build_date)),
			PartNumber:   boundedString(&out.part_number[0], len(out.part_number)),
			Version:      boundedString(&out.version[0], len(out.version)),
			BootFirmware: boundedString(&out.boot_firmware[0], len(out.boot_firmware)),
		}, nil
	})
}

type TemperatureType uint32
type TemperatureMetric uint32
type ClockType uint32
type MemoryType uint32

const maxFrequencies int = C.AMDSMI_MAX_NUM_FREQUENCIES

const (
	AMDSMI_TEMPERATURE_TYPE_EDGE                             TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_EDGE
	AMDSMI_TEMPERATURE_TYPE_FIRST                            TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_FIRST
	AMDSMI_TEMPERATURE_TYPE_HOTSPOT                          TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_HOTSPOT
	AMDSMI_TEMPERATURE_TYPE_JUNCTION                         TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_JUNCTION
	AMDSMI_TEMPERATURE_TYPE_VRAM                             TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_VRAM
	AMDSMI_TEMPERATURE_TYPE_HBM_0                            TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_HBM_0
	AMDSMI_TEMPERATURE_TYPE_HBM_1                            TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_HBM_1
	AMDSMI_TEMPERATURE_TYPE_HBM_2                            TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_HBM_2
	AMDSMI_TEMPERATURE_TYPE_HBM_3                            TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_HBM_3
	AMDSMI_TEMPERATURE_TYPE_PLX                              TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_PLX
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_FIRST              TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_FIRST
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_RETIMER_X          TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_RETIMER_X
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_OAM_X_IBC          TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_OAM_X_IBC
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_OAM_X_IBC_2        TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_OAM_X_IBC_2
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_OAM_X_VDD18_VR     TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_OAM_X_VDD18_VR
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_OAM_X_04_HBM_B_VR  TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_OAM_X_04_HBM_B_VR
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_OAM_X_04_HBM_D_VR  TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_OAM_X_04_HBM_D_VR
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_LAST               TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_LAST
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VR_FIRST                TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VR_FIRST
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_VDD0              TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_VDD0
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_VDD1              TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_VDD1
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_VDD2              TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_VDD2
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_VDD3              TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_VDD3
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_SOC_A             TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_SOC_A
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_SOC_C             TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_SOC_C
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_SOCIO_A           TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_SOCIO_A
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_SOCIO_C           TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_SOCIO_C
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDD_085_HBM             TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDD_085_HBM
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_11_HBM_B          TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_11_HBM_B
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_11_HBM_D          TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_11_HBM_D
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDD_USR                 TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDD_USR
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDIO_11_E32            TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDIO_11_E32
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDIO_04_HBM_B          TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDIO_04_HBM_B
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDIO_04_HBM_D          TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDIO_04_HBM_D
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_075_HBM_B         TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_075_HBM_B
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_075_HBM_D         TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_075_HBM_D
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDIO_11_GTA_A          TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDIO_11_GTA_A
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDIO_11_GTA_C          TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDIO_11_GTA_C
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDAN_075_GTA_A         TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDAN_075_GTA_A
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDAN_075_GTA_C         TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDAN_075_GTA_C
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_075_UCIE          TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_075_UCIE
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDIO_065_UCIEAA        TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDIO_065_UCIEAA
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDIO_065_UCIEAM_A      TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDIO_065_UCIEAM_A
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDIO_065_UCIEAM_C      TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDIO_065_UCIEAM_C
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDAN_075               TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDAN_075
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VR_LAST                 TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VR_LAST
	AMDSMI_TEMPERATURE_TYPE_GPUBOARD_LAST                    TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_GPUBOARD_LAST
	AMDSMI_TEMPERATURE_TYPE_BASEBOARD_FIRST                  TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_BASEBOARD_FIRST
	AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_FPGA               TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_FPGA
	AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_FRONT              TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_FRONT
	AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_BACK               TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_BACK
	AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_OAM7               TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_OAM7
	AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_IBC                TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_IBC
	AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_UFPGA              TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_UFPGA
	AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_OAM1               TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_OAM1
	AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_0_1_HSC            TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_0_1_HSC
	AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_2_3_HSC            TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_2_3_HSC
	AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_4_5_HSC            TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_4_5_HSC
	AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_6_7_HSC            TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_6_7_HSC
	AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_FPGA_0V72_VR       TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_FPGA_0V72_VR
	AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_FPGA_3V3_VR        TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_FPGA_3V3_VR
	AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_0_1_2_3_1V2_VR TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_0_1_2_3_1V2_VR
	AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_4_5_6_7_1V2_VR TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_4_5_6_7_1V2_VR
	AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_0_1_0V9_VR     TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_0_1_0V9_VR
	AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_4_5_0V9_VR     TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_4_5_0V9_VR
	AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_2_3_0V9_VR     TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_2_3_0V9_VR
	AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_6_7_0V9_VR     TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_6_7_0V9_VR
	AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_0_1_2_3_3V3_VR     TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_0_1_2_3_3V3_VR
	AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_4_5_6_7_3V3_VR     TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_4_5_6_7_3V3_VR
	AMDSMI_TEMPERATURE_TYPE_BASEBOARD_IBC_HSC                TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_BASEBOARD_IBC_HSC
	AMDSMI_TEMPERATURE_TYPE_BASEBOARD_IBC                    TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_BASEBOARD_IBC
	AMDSMI_TEMPERATURE_TYPE_BASEBOARD_LAST                   TemperatureType = C.AMDSMI_TEMPERATURE_TYPE_BASEBOARD_LAST
	AMDSMI_TEMPERATURE_TYPE__MAX                             TemperatureType = C.AMDSMI_TEMPERATURE_TYPE__MAX
)

const (
	AMDSMI_TEMP_CURRENT        TemperatureMetric = C.AMDSMI_TEMP_CURRENT
	AMDSMI_TEMP_FIRST          TemperatureMetric = C.AMDSMI_TEMP_FIRST
	AMDSMI_TEMP_MAX            TemperatureMetric = C.AMDSMI_TEMP_MAX
	AMDSMI_TEMP_MIN            TemperatureMetric = C.AMDSMI_TEMP_MIN
	AMDSMI_TEMP_MAX_HYST       TemperatureMetric = C.AMDSMI_TEMP_MAX_HYST
	AMDSMI_TEMP_MIN_HYST       TemperatureMetric = C.AMDSMI_TEMP_MIN_HYST
	AMDSMI_TEMP_CRITICAL       TemperatureMetric = C.AMDSMI_TEMP_CRITICAL
	AMDSMI_TEMP_CRITICAL_HYST  TemperatureMetric = C.AMDSMI_TEMP_CRITICAL_HYST
	AMDSMI_TEMP_EMERGENCY      TemperatureMetric = C.AMDSMI_TEMP_EMERGENCY
	AMDSMI_TEMP_EMERGENCY_HYST TemperatureMetric = C.AMDSMI_TEMP_EMERGENCY_HYST
	AMDSMI_TEMP_CRIT_MIN       TemperatureMetric = C.AMDSMI_TEMP_CRIT_MIN
	AMDSMI_TEMP_CRIT_MIN_HYST  TemperatureMetric = C.AMDSMI_TEMP_CRIT_MIN_HYST
	AMDSMI_TEMP_OFFSET         TemperatureMetric = C.AMDSMI_TEMP_OFFSET
	AMDSMI_TEMP_LOWEST         TemperatureMetric = C.AMDSMI_TEMP_LOWEST
	AMDSMI_TEMP_HIGHEST        TemperatureMetric = C.AMDSMI_TEMP_HIGHEST
	AMDSMI_TEMP_SHUTDOWN       TemperatureMetric = C.AMDSMI_TEMP_SHUTDOWN
	AMDSMI_TEMP_LAST           TemperatureMetric = C.AMDSMI_TEMP_LAST
)

const (
	AMDSMI_CLK_TYPE_SYS   ClockType = C.AMDSMI_CLK_TYPE_SYS
	AMDSMI_CLK_TYPE_FIRST ClockType = C.AMDSMI_CLK_TYPE_FIRST
	AMDSMI_CLK_TYPE_GFX   ClockType = C.AMDSMI_CLK_TYPE_GFX
	AMDSMI_CLK_TYPE_DF    ClockType = C.AMDSMI_CLK_TYPE_DF
	AMDSMI_CLK_TYPE_DCEF  ClockType = C.AMDSMI_CLK_TYPE_DCEF
	AMDSMI_CLK_TYPE_SOC   ClockType = C.AMDSMI_CLK_TYPE_SOC
	AMDSMI_CLK_TYPE_MEM   ClockType = C.AMDSMI_CLK_TYPE_MEM
	AMDSMI_CLK_TYPE_PCIE  ClockType = C.AMDSMI_CLK_TYPE_PCIE
	AMDSMI_CLK_TYPE_VCLK0 ClockType = C.AMDSMI_CLK_TYPE_VCLK0
	AMDSMI_CLK_TYPE_VCLK1 ClockType = C.AMDSMI_CLK_TYPE_VCLK1
	AMDSMI_CLK_TYPE_DCLK0 ClockType = C.AMDSMI_CLK_TYPE_DCLK0
	AMDSMI_CLK_TYPE_DCLK1 ClockType = C.AMDSMI_CLK_TYPE_DCLK1
	AMDSMI_CLK_TYPE__MAX  ClockType = C.AMDSMI_CLK_TYPE__MAX
)

const (
	AMDSMI_MEM_TYPE_FIRST    MemoryType = C.AMDSMI_MEM_TYPE_FIRST
	AMDSMI_MEM_TYPE_VRAM     MemoryType = C.AMDSMI_MEM_TYPE_VRAM
	AMDSMI_MEM_TYPE_VIS_VRAM MemoryType = C.AMDSMI_MEM_TYPE_VIS_VRAM
	AMDSMI_MEM_TYPE_GTT      MemoryType = C.AMDSMI_MEM_TYPE_GTT
	AMDSMI_MEM_TYPE_LAST     MemoryType = C.AMDSMI_MEM_TYPE_LAST
)

// PowerInfo preserves native unavailable values at each field's width.
type PowerInfo struct {
	SocketPowerWatts        uint64
	CurrentSocketPowerWatts uint32
	AverageSocketPowerWatts uint32
	GFXVoltageMillivolts    uint64
	SOCVoltageMillivolts    uint64
	MemoryVoltageMillivolts uint64
	PowerLimitMicrowatts    uint32
	UBBPowerWatts           uint32
}

// PowerCapInfo may contain zero auxiliary fields after partial native success.
type PowerCapInfo struct {
	PowerCapMicrowatts        uint64
	DefaultPowerCapMicrowatts uint64
	DPMLevel                  uint64 // Level index, not MHz.
	MinPowerCapMicrowatts     uint64
	MaxPowerCapMicrowatts     uint64
}

type ClockInfo struct {
	ClockMHz     uint32
	MinClockMHz  uint32
	MaxClockMHz  uint32
	LockedRaw    uint8 // Not populated by the current bare-metal implementation.
	DeepSleepRaw uint8 // Native narrowed sleep-frequency byte.
}

type Frequencies struct {
	HasDeepSleep bool
	CurrentIndex uint32 // May be UINT32_MAX or outside Hertz.
	Hertz        []uint64
}

type Activity struct {
	GFXPercent uint32 // 65535 can indicate unavailable data.
	UMCPercent uint32
	MMPercent  uint32
}

type VRAMInfo struct {
	Type                    VRAMType
	Vendor                  string
	SizeMB                  uint64
	BitWidth                uint32
	MaxBandwidthGBPerSecond uint64
}

type VRAMType uint32

const (
	AMDSMI_VRAM_TYPE_UNKNOWN VRAMType = C.AMDSMI_VRAM_TYPE_UNKNOWN
	AMDSMI_VRAM_TYPE_HBM     VRAMType = C.AMDSMI_VRAM_TYPE_HBM
	AMDSMI_VRAM_TYPE_HBM2    VRAMType = C.AMDSMI_VRAM_TYPE_HBM2
	AMDSMI_VRAM_TYPE_HBM2E   VRAMType = C.AMDSMI_VRAM_TYPE_HBM2E
	AMDSMI_VRAM_TYPE_HBM3    VRAMType = C.AMDSMI_VRAM_TYPE_HBM3
	AMDSMI_VRAM_TYPE_HBM3E   VRAMType = C.AMDSMI_VRAM_TYPE_HBM3E
	AMDSMI_VRAM_TYPE_DDR2    VRAMType = C.AMDSMI_VRAM_TYPE_DDR2
	AMDSMI_VRAM_TYPE_DDR3    VRAMType = C.AMDSMI_VRAM_TYPE_DDR3
	AMDSMI_VRAM_TYPE_DDR4    VRAMType = C.AMDSMI_VRAM_TYPE_DDR4
	AMDSMI_VRAM_TYPE_DDR5    VRAMType = C.AMDSMI_VRAM_TYPE_DDR5
	AMDSMI_VRAM_TYPE_GDDR1   VRAMType = C.AMDSMI_VRAM_TYPE_GDDR1
	AMDSMI_VRAM_TYPE_GDDR2   VRAMType = C.AMDSMI_VRAM_TYPE_GDDR2
	AMDSMI_VRAM_TYPE_GDDR3   VRAMType = C.AMDSMI_VRAM_TYPE_GDDR3
	AMDSMI_VRAM_TYPE_GDDR4   VRAMType = C.AMDSMI_VRAM_TYPE_GDDR4
	AMDSMI_VRAM_TYPE_GDDR5   VRAMType = C.AMDSMI_VRAM_TYPE_GDDR5
	AMDSMI_VRAM_TYPE_GDDR6   VRAMType = C.AMDSMI_VRAM_TYPE_GDDR6
	AMDSMI_VRAM_TYPE_GDDR7   VRAMType = C.AMDSMI_VRAM_TYPE_GDDR7
	AMDSMI_VRAM_TYPE_LPDDR4  VRAMType = C.AMDSMI_VRAM_TYPE_LPDDR4
	AMDSMI_VRAM_TYPE_LPDDR5  VRAMType = C.AMDSMI_VRAM_TYPE_LPDDR5
	AMDSMI_VRAM_TYPE__MAX    VRAMType = C.AMDSMI_VRAM_TYPE__MAX
)

// GetTemperature returns whole degrees Celsius, preserving negative values.
func GetTemperature(h ProcessorHandle, sensor TemperatureType, metric TemperatureMetric) (int64, error) {
	const op = "amdsmi_get_temp_metric"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (int64, error) {
		var out C.int64_t
		status := C.amdsmi_get_temp_metric(p, C.amdsmi_temperature_type_t(sensor),
			C.amdsmi_temperature_metric_t(metric), &out)
		if err := nativeErrorLocked(op, status); err != nil {
			return 0, err
		}
		return int64(out), nil
	})
}

func GetPowerInfo(h ProcessorHandle) (PowerInfo, error) {
	const op = "amdsmi_get_power_info"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (PowerInfo, error) {
		var out C.amdsmi_power_info_t
		if err := nativeErrorLocked(op, C.amdsmi_get_power_info(p, &out)); err != nil {
			return PowerInfo{}, err
		}
		return PowerInfo{
			SocketPowerWatts:        uint64(out.socket_power),
			CurrentSocketPowerWatts: uint32(out.current_socket_power),
			AverageSocketPowerWatts: uint32(out.average_socket_power),
			GFXVoltageMillivolts:    uint64(out.gfx_voltage),
			SOCVoltageMillivolts:    uint64(out.soc_voltage),
			MemoryVoltageMillivolts: uint64(out.mem_voltage),
			PowerLimitMicrowatts:    uint32(out.power_limit),
			UBBPowerWatts:           uint32(out.ubb_power),
		}, nil
	})
}

func GetPowerCapInfo(h ProcessorHandle, sensorIndex uint32) (PowerCapInfo, error) {
	const op = "amdsmi_get_power_cap_info"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (PowerCapInfo, error) {
		var out C.amdsmi_power_cap_info_t
		status := C.amdsmi_get_power_cap_info(p, C.uint32_t(sensorIndex), &out)
		if err := nativeErrorLocked(op, status); err != nil {
			return PowerCapInfo{}, err
		}
		return PowerCapInfo{
			PowerCapMicrowatts:        uint64(out.power_cap),
			DefaultPowerCapMicrowatts: uint64(out.default_power_cap),
			DPMLevel:                  uint64(out.dpm_cap),
			MinPowerCapMicrowatts:     uint64(out.min_power_cap),
			MaxPowerCapMicrowatts:     uint64(out.max_power_cap),
		}, nil
	})
}

func GetClockInfo(h ProcessorHandle, clock ClockType) (ClockInfo, error) {
	const op = "amdsmi_get_clock_info"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (ClockInfo, error) {
		var out C.amdsmi_clk_info_t
		status := C.amdsmi_get_clock_info(p, C.amdsmi_clk_type_t(clock), &out)
		if err := nativeErrorLocked(op, status); err != nil {
			return ClockInfo{}, err
		}
		return ClockInfo{
			ClockMHz:     uint32(out.clk),
			MinClockMHz:  uint32(out.min_clk),
			MaxClockMHz:  uint32(out.max_clk),
			LockedRaw:    uint8(out.clk_locked),
			DeepSleepRaw: uint8(out.clk_deep_sleep),
		}, nil
	})
}

func GetClockFrequencies(h ProcessorHandle, clock ClockType) (Frequencies, error) {
	const op = "amdsmi_get_clk_freq"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (Frequencies, error) {
		var out C.amdsmi_frequencies_t
		status := C.amdsmi_get_clk_freq(p, C.amdsmi_clk_type_t(clock), &out)
		if err := nativeErrorLocked(op, status); err != nil {
			return Frequencies{}, err
		}
		n, err := checkedCount(op, uint64(out.num_supported), len(out.frequency))
		if err != nil {
			return Frequencies{}, err
		}
		result := Frequencies{
			HasDeepSleep: bool(out.has_deep_sleep),
			CurrentIndex: uint32(out.current),
			Hertz:        make([]uint64, n),
		}
		for i := range result.Hertz {
			result.Hertz[i] = uint64(out.frequency[i])
		}
		return result, nil
	})
}

func GetActivity(h ProcessorHandle) (Activity, error) {
	const op = "amdsmi_get_gpu_activity"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (Activity, error) {
		var out C.amdsmi_engine_usage_t
		if err := nativeErrorLocked(op, C.amdsmi_get_gpu_activity(p, &out)); err != nil {
			return Activity{}, err
		}
		return Activity{
			GFXPercent: uint32(out.gfx_activity),
			UMCPercent: uint32(out.umc_activity),
			MMPercent:  uint32(out.mm_activity),
		}, nil
	})
}

// GetMemoryTotal returns bytes.
func GetMemoryTotal(h ProcessorHandle, memory MemoryType) (uint64, error) {
	const op = "amdsmi_get_gpu_memory_total"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (uint64, error) {
		var out C.uint64_t
		status := C.amdsmi_get_gpu_memory_total(p, C.amdsmi_memory_type_t(memory), &out)
		if err := nativeErrorLocked(op, status); err != nil {
			return 0, err
		}
		return uint64(out), nil
	})
}

// GetMemoryUsage returns bytes.
func GetMemoryUsage(h ProcessorHandle, memory MemoryType) (uint64, error) {
	const op = "amdsmi_get_gpu_memory_usage"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (uint64, error) {
		var out C.uint64_t
		status := C.amdsmi_get_gpu_memory_usage(p, C.amdsmi_memory_type_t(memory), &out)
		if err := nativeErrorLocked(op, status); err != nil {
			return 0, err
		}
		return uint64(out), nil
	})
}

func GetVRAMInfo(h ProcessorHandle) (VRAMInfo, error) {
	const op = "amdsmi_get_gpu_vram_info"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (VRAMInfo, error) {
		var out C.amdsmi_vram_info_t
		if err := nativeErrorLocked(op, C.amdsmi_get_gpu_vram_info(p, &out)); err != nil {
			return VRAMInfo{}, err
		}
		return VRAMInfo{
			Type:                    VRAMType(out.vram_type),
			Vendor:                  boundedString(&out.vram_vendor[0], len(out.vram_vendor)),
			SizeMB:                  uint64(out.vram_size),
			BitWidth:                uint32(out.vram_bit_width),
			MaxBandwidthGBPerSecond: uint64(out.vram_max_bandwidth),
		}, nil
	})
}

type KFDInfo struct {
	KFDID              uint64
	NodeID             uint32
	CurrentPartitionID uint32
}

type MemoryPartitionType uint32
type AcceleratorPartitionType uint32

// MemoryCapabilities preserves the native NPS capability mask, not NPS enum values.
type MemoryCapabilities uint32

type NUMARange struct {
	MemoryType VRAMType
	Start      uint64
	End        uint64
}

type MemoryPartitionConfig struct {
	Capabilities MemoryCapabilities
	Mode         MemoryPartitionType
	// Empty ranges can mean unpopulated native metadata.
	NUMARanges []NUMARange
}

type AcceleratorPartitionProfile struct {
	Type               AcceleratorPartitionType
	NumPartitions      uint32
	MemoryCapabilities MemoryCapabilities
	ProfileIndex       uint32
	NumResources       uint32
	// Current native profiles leave resource metadata empty.
	Resources   [][]uint32
	PartitionID uint32
}

const (
	maxNUMARanges            int = C.AMDSMI_MAX_NUM_NUMA_NODES
	maxAcceleratorPartitions int = C.AMDSMI_MAX_ACCELERATOR_PARTITIONS
	maxProfileResources      int = C.AMDSMI_MAX_CP_PROFILE_RESOURCES
)

const (
	AMDSMI_MEMORY_PARTITION_UNKNOWN MemoryPartitionType = C.AMDSMI_MEMORY_PARTITION_UNKNOWN
	AMDSMI_MEMORY_PARTITION_NPS1    MemoryPartitionType = C.AMDSMI_MEMORY_PARTITION_NPS1
	AMDSMI_MEMORY_PARTITION_NPS2    MemoryPartitionType = C.AMDSMI_MEMORY_PARTITION_NPS2
	AMDSMI_MEMORY_PARTITION_NPS4    MemoryPartitionType = C.AMDSMI_MEMORY_PARTITION_NPS4
	AMDSMI_MEMORY_PARTITION_NPS8    MemoryPartitionType = C.AMDSMI_MEMORY_PARTITION_NPS8
)

const (
	AMDSMI_ACCELERATOR_PARTITION_INVALID AcceleratorPartitionType = C.AMDSMI_ACCELERATOR_PARTITION_INVALID
	AMDSMI_ACCELERATOR_PARTITION_SPX     AcceleratorPartitionType = C.AMDSMI_ACCELERATOR_PARTITION_SPX
	AMDSMI_ACCELERATOR_PARTITION_DPX     AcceleratorPartitionType = C.AMDSMI_ACCELERATOR_PARTITION_DPX
	AMDSMI_ACCELERATOR_PARTITION_TPX     AcceleratorPartitionType = C.AMDSMI_ACCELERATOR_PARTITION_TPX
	AMDSMI_ACCELERATOR_PARTITION_QPX     AcceleratorPartitionType = C.AMDSMI_ACCELERATOR_PARTITION_QPX
	AMDSMI_ACCELERATOR_PARTITION_CPX     AcceleratorPartitionType = C.AMDSMI_ACCELERATOR_PARTITION_CPX
	AMDSMI_ACCELERATOR_PARTITION_MAX     AcceleratorPartitionType = C.AMDSMI_ACCELERATOR_PARTITION_MAX
)

func GetKFDInfo(h ProcessorHandle) (KFDInfo, error) {
	const op = "amdsmi_get_gpu_kfd_info"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (KFDInfo, error) {
		var out C.amdsmi_kfd_info_t
		if err := nativeErrorLocked(op, C.amdsmi_get_gpu_kfd_info(p, &out)); err != nil {
			return KFDInfo{}, err
		}
		return KFDInfo{KFDID: uint64(out.kfd_id), NodeID: uint32(out.node_id),
			CurrentPartitionID: uint32(out.current_partition_id)}, nil
	})
}

func GetMemoryPartitionConfig(h ProcessorHandle) (MemoryPartitionConfig, error) {
	const op = "amdsmi_get_gpu_memory_partition_config"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (MemoryPartitionConfig, error) {
		var out C.amdsmi_memory_partition_config_t
		if err := nativeErrorLocked(op, C.amdsmi_get_gpu_memory_partition_config(p, &out)); err != nil {
			return MemoryPartitionConfig{}, err
		}
		n, err := checkedCount(op, uint64(out.num_numa_ranges), len(out.numa_range))
		if err != nil {
			return MemoryPartitionConfig{}, err
		}
		result := MemoryPartitionConfig{
			Capabilities: MemoryCapabilities(C.go_amdsmi_nps_mask(out.partition_caps)),
			Mode:         MemoryPartitionType(out.mp_mode),
			NUMARanges:   make([]NUMARange, n),
		}
		for i := range result.NUMARanges {
			value := out.numa_range[i]
			result.NUMARanges[i] = NUMARange{MemoryType: VRAMType(value.memory_type),
				Start: uint64(value.start), End: uint64(value.end)}
		}
		return result, nil
	})
}

func GetAcceleratorPartitionProfile(h ProcessorHandle) (AcceleratorPartitionProfile, error) {
	const op = "amdsmi_get_gpu_accelerator_partition_profile"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (AcceleratorPartitionProfile, error) {
		var out C.amdsmi_accelerator_partition_profile_t
		var partitionIDs [C.AMDSMI_MAX_ACCELERATOR_PARTITIONS]C.uint32_t
		status := C.amdsmi_get_gpu_accelerator_partition_profile(p, &out, &partitionIDs[0])
		if err := nativeErrorLocked(op, status); err != nil {
			return AcceleratorPartitionProfile{}, err
		}
		result := AcceleratorPartitionProfile{
			Type:               AcceleratorPartitionType(out.profile_type),
			NumPartitions:      uint32(out.num_partitions),
			MemoryCapabilities: MemoryCapabilities(C.go_amdsmi_nps_mask(out.memory_caps)),
			ProfileIndex:       uint32(out.profile_index),
			NumResources:       uint32(out.num_resources),
			PartitionID:        uint32(partitionIDs[0]),
		}
		if out.num_resources == 0 {
			return result, nil
		}
		rows, err := checkedCount(op, uint64(out.num_partitions), len(out.resources))
		if err != nil {
			return AcceleratorPartitionProfile{}, err
		}
		columns, err := checkedCount(op, uint64(out.num_resources), len(out.resources[0]))
		if err != nil {
			return AcceleratorPartitionProfile{}, err
		}
		result.Resources = make([][]uint32, rows)
		for i := range result.Resources {
			result.Resources[i] = make([]uint32, columns)
			for j := range result.Resources[i] {
				result.Resources[i][j] = uint32(out.resources[i][j])
			}
		}
		return result, nil
	})
}

type GPUBlock uint64
type RASState uint32

type ECCCounts struct {
	Correctable   uint64
	Uncorrectable uint64
	Deferred      uint64
}

// RASFeatureInfo contains populated native metadata, not an overall health assessment.
type RASFeatureInfo struct {
	EEPROMVersion       uint32
	ECCCorrectionSchema uint32
}

const (
	AMDSMI_GPU_BLOCK_INVALID    GPUBlock = C.AMDSMI_GPU_BLOCK_INVALID
	AMDSMI_GPU_BLOCK_FIRST      GPUBlock = C.AMDSMI_GPU_BLOCK_FIRST
	AMDSMI_GPU_BLOCK_UMC        GPUBlock = C.AMDSMI_GPU_BLOCK_UMC
	AMDSMI_GPU_BLOCK_SDMA       GPUBlock = C.AMDSMI_GPU_BLOCK_SDMA
	AMDSMI_GPU_BLOCK_GFX        GPUBlock = C.AMDSMI_GPU_BLOCK_GFX
	AMDSMI_GPU_BLOCK_MMHUB      GPUBlock = C.AMDSMI_GPU_BLOCK_MMHUB
	AMDSMI_GPU_BLOCK_ATHUB      GPUBlock = C.AMDSMI_GPU_BLOCK_ATHUB
	AMDSMI_GPU_BLOCK_PCIE_BIF   GPUBlock = C.AMDSMI_GPU_BLOCK_PCIE_BIF
	AMDSMI_GPU_BLOCK_HDP        GPUBlock = C.AMDSMI_GPU_BLOCK_HDP
	AMDSMI_GPU_BLOCK_XGMI_WAFL  GPUBlock = C.AMDSMI_GPU_BLOCK_XGMI_WAFL
	AMDSMI_GPU_BLOCK_DF         GPUBlock = C.AMDSMI_GPU_BLOCK_DF
	AMDSMI_GPU_BLOCK_SMN        GPUBlock = C.AMDSMI_GPU_BLOCK_SMN
	AMDSMI_GPU_BLOCK_SEM        GPUBlock = C.AMDSMI_GPU_BLOCK_SEM
	AMDSMI_GPU_BLOCK_MP0        GPUBlock = C.AMDSMI_GPU_BLOCK_MP0
	AMDSMI_GPU_BLOCK_MP1        GPUBlock = C.AMDSMI_GPU_BLOCK_MP1
	AMDSMI_GPU_BLOCK_FUSE       GPUBlock = C.AMDSMI_GPU_BLOCK_FUSE
	AMDSMI_GPU_BLOCK_MCA        GPUBlock = C.AMDSMI_GPU_BLOCK_MCA
	AMDSMI_GPU_BLOCK_VCN        GPUBlock = C.AMDSMI_GPU_BLOCK_VCN
	AMDSMI_GPU_BLOCK_JPEG       GPUBlock = C.AMDSMI_GPU_BLOCK_JPEG
	AMDSMI_GPU_BLOCK_IH         GPUBlock = C.AMDSMI_GPU_BLOCK_IH
	AMDSMI_GPU_BLOCK_MPIO       GPUBlock = C.AMDSMI_GPU_BLOCK_MPIO
	AMDSMI_GPU_BLOCK_MMSCH      GPUBlock = C.AMDSMI_GPU_BLOCK_MMSCH
	AMDSMI_GPU_BLOCK_MP5        GPUBlock = C.AMDSMI_GPU_BLOCK_MP5
	AMDSMI_GPU_BLOCK_ATU        GPUBlock = C.AMDSMI_GPU_BLOCK_ATU
	AMDSMI_GPU_BLOCK_DACC_BE    GPUBlock = C.AMDSMI_GPU_BLOCK_DACC_BE
	AMDSMI_GPU_BLOCK_ECLR       GPUBlock = C.AMDSMI_GPU_BLOCK_ECLR
	AMDSMI_GPU_BLOCK_KPX_SERDES GPUBlock = C.AMDSMI_GPU_BLOCK_KPX_SERDES
	AMDSMI_GPU_BLOCK_LSDMA      GPUBlock = C.AMDSMI_GPU_BLOCK_LSDMA
	AMDSMI_GPU_BLOCK_MPART      GPUBlock = C.AMDSMI_GPU_BLOCK_MPART
	AMDSMI_GPU_BLOCK_MPIFOE     GPUBlock = C.AMDSMI_GPU_BLOCK_MPIFOE
	AMDSMI_GPU_BLOCK_MPRAS      GPUBlock = C.AMDSMI_GPU_BLOCK_MPRAS
	AMDSMI_GPU_BLOCK_NBIF       GPUBlock = C.AMDSMI_GPU_BLOCK_NBIF
	AMDSMI_GPU_BLOCK_NBIO       GPUBlock = C.AMDSMI_GPU_BLOCK_NBIO
	AMDSMI_GPU_BLOCK_OXRP       GPUBlock = C.AMDSMI_GPU_BLOCK_OXRP
	AMDSMI_GPU_BLOCK_PCIE_PL    GPUBlock = C.AMDSMI_GPU_BLOCK_PCIE_PL
	AMDSMI_GPU_BLOCK_PCS_XGMI   GPUBlock = C.AMDSMI_GPU_BLOCK_PCS_XGMI
	AMDSMI_GPU_BLOCK_PIE        GPUBlock = C.AMDSMI_GPU_BLOCK_PIE
	AMDSMI_GPU_BLOCK_CS         GPUBlock = C.AMDSMI_GPU_BLOCK_CS
	AMDSMI_GPU_BLOCK_SHUB       GPUBlock = C.AMDSMI_GPU_BLOCK_SHUB
	AMDSMI_GPU_BLOCK_SSBDCI     GPUBlock = C.AMDSMI_GPU_BLOCK_SSBDCI
	AMDSMI_GPU_BLOCK_UCIE_PCS   GPUBlock = C.AMDSMI_GPU_BLOCK_UCIE_PCS
	AMDSMI_GPU_BLOCK_LAST       GPUBlock = C.AMDSMI_GPU_BLOCK_LAST
	AMDSMI_GPU_BLOCK_RESERVED   GPUBlock = C.AMDSMI_GPU_BLOCK_RESERVED
)

const (
	AMDSMI_RAS_ERR_STATE_NONE     RASState = C.AMDSMI_RAS_ERR_STATE_NONE
	AMDSMI_RAS_ERR_STATE_DISABLED RASState = C.AMDSMI_RAS_ERR_STATE_DISABLED
	AMDSMI_RAS_ERR_STATE_PARITY   RASState = C.AMDSMI_RAS_ERR_STATE_PARITY
	AMDSMI_RAS_ERR_STATE_SING_C   RASState = C.AMDSMI_RAS_ERR_STATE_SING_C
	AMDSMI_RAS_ERR_STATE_MULT_UC  RASState = C.AMDSMI_RAS_ERR_STATE_MULT_UC
	AMDSMI_RAS_ERR_STATE_POISON   RASState = C.AMDSMI_RAS_ERR_STATE_POISON
	AMDSMI_RAS_ERR_STATE_ENABLED  RASState = C.AMDSMI_RAS_ERR_STATE_ENABLED
	AMDSMI_RAS_ERR_STATE_LAST     RASState = C.AMDSMI_RAS_ERR_STATE_LAST
	AMDSMI_RAS_ERR_STATE_INVALID  RASState = C.AMDSMI_RAS_ERR_STATE_INVALID
)

func GetECCEnabled(h ProcessorHandle) (GPUBlock, error) {
	const op = "amdsmi_get_gpu_ecc_enabled"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (GPUBlock, error) {
		var out C.uint64_t
		if err := nativeErrorLocked(op, C.amdsmi_get_gpu_ecc_enabled(p, &out)); err != nil {
			return 0, err
		}
		return GPUBlock(out), nil
	})
}

func GetECCCount(h ProcessorHandle, block GPUBlock) (ECCCounts, error) {
	const op = "amdsmi_get_gpu_ecc_count"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (ECCCounts, error) {
		var out C.amdsmi_error_count_t
		status := C.amdsmi_get_gpu_ecc_count(p, C.amdsmi_gpu_block_t(block), &out)
		if err := nativeErrorLocked(op, status); err != nil {
			return ECCCounts{}, err
		}
		return ECCCounts{Correctable: uint64(out.correctable_count),
			Uncorrectable: uint64(out.uncorrectable_count), Deferred: uint64(out.deferred_count)}, nil
	})
}

// GetTotalECCCount preserves native totals, which may omit unavailable blocks.
func GetTotalECCCount(h ProcessorHandle) (ECCCounts, error) {
	const op = "amdsmi_get_gpu_total_ecc_count"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (ECCCounts, error) {
		var out C.amdsmi_error_count_t
		if err := nativeErrorLocked(op, C.amdsmi_get_gpu_total_ecc_count(p, &out)); err != nil {
			return ECCCounts{}, err
		}
		return ECCCounts{Correctable: uint64(out.correctable_count),
			Uncorrectable: uint64(out.uncorrectable_count), Deferred: uint64(out.deferred_count)}, nil
	})
}

func GetRASBlockState(h ProcessorHandle, block GPUBlock) (RASState, error) {
	const op = "amdsmi_get_gpu_ras_block_features_enabled"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (RASState, error) {
		var out C.amdsmi_ras_err_state_t
		status := C.amdsmi_get_gpu_ras_block_features_enabled(p, C.amdsmi_gpu_block_t(block), &out)
		if err := nativeErrorLocked(op, status); err != nil {
			return 0, err
		}
		return RASState(out), nil
	})
}

func GetRASFeatureInfo(h ProcessorHandle) (RASFeatureInfo, error) {
	const op = "amdsmi_get_gpu_ras_feature_info"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (RASFeatureInfo, error) {
		var out C.amdsmi_ras_feature_t
		if err := nativeErrorLocked(op, C.amdsmi_get_gpu_ras_feature_info(p, &out)); err != nil {
			return RASFeatureInfo{}, err
		}
		return RASFeatureInfo{EEPROMVersion: uint32(out.ras_eeprom_version),
			ECCCorrectionSchema: uint32(out.ecc_correction_schema_flag)}, nil
	})
}
