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

static int go_amdsmi_nps_cap(amdsmi_nps_caps_t value, unsigned int index) {
	switch (index) {
		case 0: return value.nps_flags.nps1_cap;
		case 1: return value.nps_flags.nps2_cap;
		case 2: return value.nps_flags.nps4_cap;
		case 3: return value.nps_flags.nps8_cap;
		default: return 0;
	}
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

type Status uint32

type StatusError struct {
	Op      string
	Code    Status
	Name    string
	Message string
}

func (s Status) Error() string {
	return fmt.Sprintf("AMD SMI status %d (0x%08x)", uint32(s), uint32(s))
}

func (e *StatusError) Error() string {
	return fmt.Sprintf("%s: %s: %s: %s", e.Op, e.Code.Error(), e.Name, e.Message)
}

func (e *StatusError) Unwrap() error { return e.Code }

func statusStringLocked(code Status) (string, error) {
	const op = "amdsmi_status_code_to_string"
	var text *C.char
	status := C.amdsmi_status_code_to_string(C.amdsmi_status_t(code), &text)
	if status != C.AMDSMI_STATUS_SUCCESS {
		return "", &StatusError{Op: op, Code: Status(status), Name: statusName(Status(status)),
			Message: "native status lookup failed"}
	}
	if text == nil {
		return "", &StatusError{Op: op, Code: AMDSMI_STATUS_UNEXPECTED_DATA,
			Name: statusName(AMDSMI_STATUS_UNEXPECTED_DATA), Message: "native status lookup returned null"}
	}
	return boundedString(text, nativeStringCapacity), nil
}

func StatusCodeToString(code Status) (string, error) {
	nativeState.mu.Lock()
	defer nativeState.mu.Unlock()
	return statusStringLocked(code)
}

func nativeErrorLocked(op string, status C.amdsmi_status_t) error {
	if status == C.AMDSMI_STATUS_SUCCESS {
		return nil
	}
	code := Status(status)
	message, err := statusStringLocked(code)
	if err != nil {
		message = code.Error()
	}
	return &StatusError{Op: op, Code: code, Name: statusName(code), Message: message}
}

func checkedCount(op string, count uint64, capacity int) (int, error) {
	if count > uint64(capacity) {
		return 0, &StatusError{
			Op: op, Code: AMDSMI_STATUS_UNEXPECTED_SIZE,
			Name:    statusName(AMDSMI_STATUS_UNEXPECTED_SIZE),
			Message: fmt.Sprintf("count %d exceeds capacity %d", count, capacity),
		}
	}
	return int(count), nil
}

const (
	AMDSMI_STATUS_SUCCESS             Status = C.AMDSMI_STATUS_SUCCESS
	AMDSMI_STATUS_INVAL               Status = C.AMDSMI_STATUS_INVAL
	AMDSMI_STATUS_NOT_SUPPORTED       Status = C.AMDSMI_STATUS_NOT_SUPPORTED
	AMDSMI_STATUS_NOT_YET_IMPLEMENTED Status = C.AMDSMI_STATUS_NOT_YET_IMPLEMENTED
	AMDSMI_STATUS_FAIL_LOAD_MODULE    Status = C.AMDSMI_STATUS_FAIL_LOAD_MODULE
	AMDSMI_STATUS_FAIL_LOAD_SYMBOL    Status = C.AMDSMI_STATUS_FAIL_LOAD_SYMBOL
	AMDSMI_STATUS_DRM_ERROR           Status = C.AMDSMI_STATUS_DRM_ERROR
	AMDSMI_STATUS_API_FAILED          Status = C.AMDSMI_STATUS_API_FAILED
	AMDSMI_STATUS_TIMEOUT             Status = C.AMDSMI_STATUS_TIMEOUT
	AMDSMI_STATUS_RETRY               Status = C.AMDSMI_STATUS_RETRY
	AMDSMI_STATUS_NO_PERM             Status = C.AMDSMI_STATUS_NO_PERM
	AMDSMI_STATUS_INTERRUPT           Status = C.AMDSMI_STATUS_INTERRUPT
	AMDSMI_STATUS_IO                  Status = C.AMDSMI_STATUS_IO
	AMDSMI_STATUS_ADDRESS_FAULT       Status = C.AMDSMI_STATUS_ADDRESS_FAULT
	AMDSMI_STATUS_FILE_ERROR          Status = C.AMDSMI_STATUS_FILE_ERROR
	AMDSMI_STATUS_OUT_OF_RESOURCES    Status = C.AMDSMI_STATUS_OUT_OF_RESOURCES
	AMDSMI_STATUS_INTERNAL_EXCEPTION  Status = C.AMDSMI_STATUS_INTERNAL_EXCEPTION
	AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS Status = C.AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS
	AMDSMI_STATUS_INIT_ERROR          Status = C.AMDSMI_STATUS_INIT_ERROR
	AMDSMI_STATUS_REFCOUNT_OVERFLOW   Status = C.AMDSMI_STATUS_REFCOUNT_OVERFLOW
	AMDSMI_STATUS_DIRECTORY_NOT_FOUND Status = C.AMDSMI_STATUS_DIRECTORY_NOT_FOUND
	AMDSMI_STATUS_IPC_ERROR           Status = C.AMDSMI_STATUS_IPC_ERROR
	AMDSMI_STATUS_BUSY                Status = C.AMDSMI_STATUS_BUSY
	AMDSMI_STATUS_NOT_FOUND           Status = C.AMDSMI_STATUS_NOT_FOUND
	AMDSMI_STATUS_NOT_INIT            Status = C.AMDSMI_STATUS_NOT_INIT
	AMDSMI_STATUS_NO_SLOT             Status = C.AMDSMI_STATUS_NO_SLOT
	AMDSMI_STATUS_DRIVER_NOT_LOADED   Status = C.AMDSMI_STATUS_DRIVER_NOT_LOADED
	AMDSMI_STATUS_MORE_DATA           Status = C.AMDSMI_STATUS_MORE_DATA
	AMDSMI_STATUS_NO_DATA             Status = C.AMDSMI_STATUS_NO_DATA
	AMDSMI_STATUS_INSUFFICIENT_SIZE   Status = C.AMDSMI_STATUS_INSUFFICIENT_SIZE
	AMDSMI_STATUS_UNEXPECTED_SIZE     Status = C.AMDSMI_STATUS_UNEXPECTED_SIZE
	AMDSMI_STATUS_UNEXPECTED_DATA     Status = C.AMDSMI_STATUS_UNEXPECTED_DATA
	AMDSMI_STATUS_NON_AMD_CPU         Status = C.AMDSMI_STATUS_NON_AMD_CPU
	AMDSMI_STATUS_NO_ENERGY_DRV       Status = C.AMDSMI_STATUS_NO_ENERGY_DRV
	AMDSMI_STATUS_NO_MSR_DRV          Status = C.AMDSMI_STATUS_NO_MSR_DRV
	AMDSMI_STATUS_NO_HSMP_DRV         Status = C.AMDSMI_STATUS_NO_HSMP_DRV
	AMDSMI_STATUS_NO_HSMP_SUP         Status = C.AMDSMI_STATUS_NO_HSMP_SUP
	AMDSMI_STATUS_NO_HSMP_MSG_SUP     Status = C.AMDSMI_STATUS_NO_HSMP_MSG_SUP
	AMDSMI_STATUS_HSMP_TIMEOUT        Status = C.AMDSMI_STATUS_HSMP_TIMEOUT
	AMDSMI_STATUS_NO_DRV              Status = C.AMDSMI_STATUS_NO_DRV
	AMDSMI_STATUS_FILE_NOT_FOUND      Status = C.AMDSMI_STATUS_FILE_NOT_FOUND
	AMDSMI_STATUS_ARG_PTR_NULL        Status = C.AMDSMI_STATUS_ARG_PTR_NULL
	AMDSMI_STATUS_AMDGPU_RESTART_ERR  Status = C.AMDSMI_STATUS_AMDGPU_RESTART_ERR
	AMDSMI_STATUS_SETTING_UNAVAILABLE Status = C.AMDSMI_STATUS_SETTING_UNAVAILABLE
	AMDSMI_STATUS_CORRUPTED_EEPROM    Status = C.AMDSMI_STATUS_CORRUPTED_EEPROM
	AMDSMI_STATUS_MAP_ERROR           Status = C.AMDSMI_STATUS_MAP_ERROR
	AMDSMI_STATUS_UNKNOWN_ERROR       Status = C.AMDSMI_STATUS_UNKNOWN_ERROR
)

var statusNames = map[Status]string{
	AMDSMI_STATUS_SUCCESS:             "AMDSMI_STATUS_SUCCESS",
	AMDSMI_STATUS_INVAL:               "AMDSMI_STATUS_INVAL",
	AMDSMI_STATUS_NOT_SUPPORTED:       "AMDSMI_STATUS_NOT_SUPPORTED",
	AMDSMI_STATUS_NOT_YET_IMPLEMENTED: "AMDSMI_STATUS_NOT_YET_IMPLEMENTED",
	AMDSMI_STATUS_FAIL_LOAD_MODULE:    "AMDSMI_STATUS_FAIL_LOAD_MODULE",
	AMDSMI_STATUS_FAIL_LOAD_SYMBOL:    "AMDSMI_STATUS_FAIL_LOAD_SYMBOL",
	AMDSMI_STATUS_DRM_ERROR:           "AMDSMI_STATUS_DRM_ERROR",
	AMDSMI_STATUS_API_FAILED:          "AMDSMI_STATUS_API_FAILED",
	AMDSMI_STATUS_TIMEOUT:             "AMDSMI_STATUS_TIMEOUT",
	AMDSMI_STATUS_RETRY:               "AMDSMI_STATUS_RETRY",
	AMDSMI_STATUS_NO_PERM:             "AMDSMI_STATUS_NO_PERM",
	AMDSMI_STATUS_INTERRUPT:           "AMDSMI_STATUS_INTERRUPT",
	AMDSMI_STATUS_IO:                  "AMDSMI_STATUS_IO",
	AMDSMI_STATUS_ADDRESS_FAULT:       "AMDSMI_STATUS_ADDRESS_FAULT",
	AMDSMI_STATUS_FILE_ERROR:          "AMDSMI_STATUS_FILE_ERROR",
	AMDSMI_STATUS_OUT_OF_RESOURCES:    "AMDSMI_STATUS_OUT_OF_RESOURCES",
	AMDSMI_STATUS_INTERNAL_EXCEPTION:  "AMDSMI_STATUS_INTERNAL_EXCEPTION",
	AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS: "AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS",
	AMDSMI_STATUS_INIT_ERROR:          "AMDSMI_STATUS_INIT_ERROR",
	AMDSMI_STATUS_REFCOUNT_OVERFLOW:   "AMDSMI_STATUS_REFCOUNT_OVERFLOW",
	AMDSMI_STATUS_DIRECTORY_NOT_FOUND: "AMDSMI_STATUS_DIRECTORY_NOT_FOUND",
	AMDSMI_STATUS_IPC_ERROR:           "AMDSMI_STATUS_IPC_ERROR",
	AMDSMI_STATUS_BUSY:                "AMDSMI_STATUS_BUSY",
	AMDSMI_STATUS_NOT_FOUND:           "AMDSMI_STATUS_NOT_FOUND",
	AMDSMI_STATUS_NOT_INIT:            "AMDSMI_STATUS_NOT_INIT",
	AMDSMI_STATUS_NO_SLOT:             "AMDSMI_STATUS_NO_SLOT",
	AMDSMI_STATUS_DRIVER_NOT_LOADED:   "AMDSMI_STATUS_DRIVER_NOT_LOADED",
	AMDSMI_STATUS_MORE_DATA:           "AMDSMI_STATUS_MORE_DATA",
	AMDSMI_STATUS_NO_DATA:             "AMDSMI_STATUS_NO_DATA",
	AMDSMI_STATUS_INSUFFICIENT_SIZE:   "AMDSMI_STATUS_INSUFFICIENT_SIZE",
	AMDSMI_STATUS_UNEXPECTED_SIZE:     "AMDSMI_STATUS_UNEXPECTED_SIZE",
	AMDSMI_STATUS_UNEXPECTED_DATA:     "AMDSMI_STATUS_UNEXPECTED_DATA",
	AMDSMI_STATUS_NON_AMD_CPU:         "AMDSMI_STATUS_NON_AMD_CPU",
	AMDSMI_STATUS_NO_ENERGY_DRV:       "AMDSMI_STATUS_NO_ENERGY_DRV",
	AMDSMI_STATUS_NO_MSR_DRV:          "AMDSMI_STATUS_NO_MSR_DRV",
	AMDSMI_STATUS_NO_HSMP_DRV:         "AMDSMI_STATUS_NO_HSMP_DRV",
	AMDSMI_STATUS_NO_HSMP_SUP:         "AMDSMI_STATUS_NO_HSMP_SUP",
	AMDSMI_STATUS_NO_HSMP_MSG_SUP:     "AMDSMI_STATUS_NO_HSMP_MSG_SUP",
	AMDSMI_STATUS_HSMP_TIMEOUT:        "AMDSMI_STATUS_HSMP_TIMEOUT",
	AMDSMI_STATUS_NO_DRV:              "AMDSMI_STATUS_NO_DRV",
	AMDSMI_STATUS_FILE_NOT_FOUND:      "AMDSMI_STATUS_FILE_NOT_FOUND",
	AMDSMI_STATUS_ARG_PTR_NULL:        "AMDSMI_STATUS_ARG_PTR_NULL",
	AMDSMI_STATUS_AMDGPU_RESTART_ERR:  "AMDSMI_STATUS_AMDGPU_RESTART_ERR",
	AMDSMI_STATUS_SETTING_UNAVAILABLE: "AMDSMI_STATUS_SETTING_UNAVAILABLE",
	AMDSMI_STATUS_CORRUPTED_EEPROM:    "AMDSMI_STATUS_CORRUPTED_EEPROM",
	AMDSMI_STATUS_MAP_ERROR:           "AMDSMI_STATUS_MAP_ERROR",
	AMDSMI_STATUS_UNKNOWN_ERROR:       "AMDSMI_STATUS_UNKNOWN_ERROR",
}

func statusName(code Status) string {
	if name, ok := statusNames[code]; ok {
		return name
	}
	return code.Error()
}

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

const AMDSMI_INIT_AMD_GPUS InitFlags = C.AMDSMI_INIT_AMD_GPUS

// Init accepts only AMDSMI_INIT_AMD_GPUS. Balance each successful call with ShutDown.
func Init(flags InitFlags) error {
	nativeState.mu.Lock()
	defer nativeState.mu.Unlock()
	if flags != AMDSMI_INIT_AMD_GPUS {
		return &StatusError{Op: "amdsmi_init", Code: AMDSMI_STATUS_INVAL, Name: statusName(AMDSMI_STATUS_INVAL),
			Message: "only AMDSMI_INIT_AMD_GPUS is supported"}
	}
	if nativeState.refs == 1<<31-1 {
		return &StatusError{Op: "amdsmi_init", Code: AMDSMI_STATUS_REFCOUNT_OVERFLOW, Name: statusName(AMDSMI_STATUS_REFCOUNT_OVERFLOW),
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
		return &StatusError{Op: "amdsmi_shut_down", Code: AMDSMI_STATUS_NOT_INIT, Name: statusName(AMDSMI_STATUS_NOT_INIT),
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
		return zero, &StatusError{Op: op, Code: AMDSMI_STATUS_NOT_INIT, Name: statusName(AMDSMI_STATUS_NOT_INIT),
			Message: "call Init before querying processors"}
	}
	return fn()
}

func withProcessor[T any](h ProcessorHandle, op string,
	fn func(C.amdsmi_processor_handle) (T, error)) (T, error) {
	return withLibrary(op, func() (T, error) {
		if h.ptr == nil || h.generation != nativeState.generation {
			var zero T
			return zero, &StatusError{Op: op, Code: AMDSMI_STATUS_INVAL, Name: statusName(AMDSMI_STATUS_INVAL),
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

// GetLibVersion does not require initialization or access any processors.
func GetLibVersion() (Version, error) {
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
			return ProcessorHandle{}, &StatusError{Op: "GetProcessorHandleFromIndex", Name: statusName(AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS),
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
			return nil, &StatusError{Op: "amdsmi_get_socket_handles", Name: statusName(AMDSMI_STATUS_UNEXPECTED_DATA),
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
				return nil, &StatusError{Op: "amdsmi_get_processor_handles", Name: statusName(AMDSMI_STATUS_UNEXPECTED_DATA),
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

type Bdf uint64

func (b Bdf) Function() uint8 { return uint8(b & 7) }
func (b Bdf) Device() uint8   { return uint8(b >> 3 & 31) }
func (b Bdf) Bus() uint8      { return uint8(b >> 8 & 255) }
func (b Bdf) Domain() uint64  { return uint64(b >> 16) }

func (b Bdf) String() string {
	return fmt.Sprintf("%04x:%02x:%02x.%x", b.Domain(), b.Bus(), b.Device(), b.Function())
}

func GetGpuDeviceBdf(h ProcessorHandle) (Bdf, error) {
	const op = "amdsmi_get_gpu_device_bdf"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (Bdf, error) {
		var domain C.uint64_t
		var bus, device, function C.uint8_t
		status := C.go_amdsmi_get_bdf(p, &domain, &bus, &device, &function)
		if err := nativeErrorLocked(op, status); err != nil {
			return 0, err
		}
		return Bdf(uint64(domain)<<16 | uint64(bus)<<8 | uint64(device)<<3 | uint64(function)), nil
	})
}

func GetProcessorHandleFromBdf(b Bdf) (ProcessorHandle, error) {
	const op = "amdsmi_get_processor_handle_from_bdf"
	return withLibrary(op, func() (ProcessorHandle, error) {
		var p C.amdsmi_processor_handle
		status := C.go_amdsmi_lookup_bdf(C.uint64_t(b.Domain()), C.uint8_t(b.Bus()),
			C.uint8_t(b.Device()), C.uint8_t(b.Function()), &p)
		if err := nativeErrorLocked(op, status); err != nil {
			return ProcessorHandle{}, err
		}
		if p == nil {
			return ProcessorHandle{}, &StatusError{Op: op, Code: AMDSMI_STATUS_UNEXPECTED_DATA, Name: statusName(AMDSMI_STATUS_UNEXPECTED_DATA),
				Message: "native lookup returned a null handle"}
		}
		var kind C.amdsmi_processor_type_t
		if err := nativeErrorLocked("amdsmi_get_processor_type",
			C.amdsmi_get_processor_type(p, &kind)); err != nil {
			return ProcessorHandle{}, err
		}
		if kind != C.AMDSMI_PROCESSOR_TYPE_AMD_GPU {
			return ProcessorHandle{}, &StatusError{Op: op, Code: AMDSMI_STATUS_NOT_SUPPORTED, Name: statusName(AMDSMI_STATUS_NOT_SUPPORTED),
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
	PhysicalAccId         uint32
	ChipRevId             uint32
	ExternalRevId         uint32
}

type DriverInfo struct {
	DriverVersion string
	DriverDate    string
	DriverName    string
}

type BoardInfo struct {
	ModelNumber      string
	ProductSerial    string
	FruID            string
	ProductName      string
	ManufacturerName string
}

type FwBlock int32

var fwBlockNames = map[FwBlock]string{
	AMDSMI_FW_ID_SMU:                      "FW_ID_SMU",
	AMDSMI_FW_ID_CP_CE:                    "FW_ID_CP_CE",
	AMDSMI_FW_ID_CP_PFP:                   "FW_ID_CP_PFP",
	AMDSMI_FW_ID_CP_ME:                    "FW_ID_CP_ME",
	AMDSMI_FW_ID_CP_MEC_JT1:               "FW_ID_CP_MEC_JT1",
	AMDSMI_FW_ID_CP_MEC_JT2:               "FW_ID_CP_MEC_JT2",
	AMDSMI_FW_ID_CP_MEC1:                  "FW_ID_CP_MEC1",
	AMDSMI_FW_ID_CP_MEC2:                  "FW_ID_CP_MEC2",
	AMDSMI_FW_ID_RLC:                      "FW_ID_RLC",
	AMDSMI_FW_ID_SDMA0:                    "FW_ID_SDMA0",
	AMDSMI_FW_ID_SDMA1:                    "FW_ID_SDMA1",
	AMDSMI_FW_ID_SDMA2:                    "FW_ID_SDMA2",
	AMDSMI_FW_ID_SDMA3:                    "FW_ID_SDMA3",
	AMDSMI_FW_ID_SDMA4:                    "FW_ID_SDMA4",
	AMDSMI_FW_ID_SDMA5:                    "FW_ID_SDMA5",
	AMDSMI_FW_ID_SDMA6:                    "FW_ID_SDMA6",
	AMDSMI_FW_ID_SDMA7:                    "FW_ID_SDMA7",
	AMDSMI_FW_ID_VCN:                      "FW_ID_VCN",
	AMDSMI_FW_ID_UVD:                      "FW_ID_UVD",
	AMDSMI_FW_ID_VCE:                      "FW_ID_VCE",
	AMDSMI_FW_ID_ISP:                      "FW_ID_ISP",
	AMDSMI_FW_ID_DMCU_ERAM:                "FW_ID_DMCU_ERAM",
	AMDSMI_FW_ID_DMCU_ISR:                 "FW_ID_DMCU_ISR",
	AMDSMI_FW_ID_RLC_RESTORE_LIST_GPM_MEM: "FW_ID_RLC_RESTORE_LIST_GPM_MEM",
	AMDSMI_FW_ID_RLC_RESTORE_LIST_SRM_MEM: "FW_ID_RLC_RESTORE_LIST_SRM_MEM",
	AMDSMI_FW_ID_RLC_RESTORE_LIST_CNTL:    "FW_ID_RLC_RESTORE_LIST_CNTL",
	AMDSMI_FW_ID_RLC_V:                    "FW_ID_RLC_V",
	AMDSMI_FW_ID_MMSCH:                    "FW_ID_MMSCH",
	AMDSMI_FW_ID_PSP_SYSDRV:               "FW_ID_PSP_SYSDRV",
	AMDSMI_FW_ID_PSP_SOSDRV:               "FW_ID_PSP_SOSDRV",
	AMDSMI_FW_ID_PSP_TOC:                  "FW_ID_PSP_TOC",
	AMDSMI_FW_ID_PSP_KEYDB:                "FW_ID_PSP_KEYDB",
	AMDSMI_FW_ID_DFC:                      "FW_ID_DFC",
	AMDSMI_FW_ID_PSP_SPL:                  "FW_ID_PSP_SPL",
	AMDSMI_FW_ID_DRV_CAP:                  "FW_ID_DRV_CAP",
	AMDSMI_FW_ID_MC:                       "FW_ID_MC",
	AMDSMI_FW_ID_PSP_BL:                   "FW_ID_PSP_BL",
	AMDSMI_FW_ID_CP_PM4:                   "FW_ID_CP_PM4",
	AMDSMI_FW_ID_RLC_P:                    "FW_ID_RLC_P",
	AMDSMI_FW_ID_SEC_POLICY_STAGE2:        "FW_ID_SEC_POLICY_STAGE2",
	AMDSMI_FW_ID_REG_ACCESS_WHITELIST:     "FW_ID_REG_ACCESS_WHITELIST",
	AMDSMI_FW_ID_IMU_DRAM:                 "FW_ID_IMU_DRAM",
	AMDSMI_FW_ID_IMU_IRAM:                 "FW_ID_IMU_IRAM",
	AMDSMI_FW_ID_SDMA_TH0:                 "FW_ID_SDMA_TH0",
	AMDSMI_FW_ID_SDMA_TH1:                 "FW_ID_SDMA_TH1",
	AMDSMI_FW_ID_CP_MES:                   "FW_ID_CP_MES",
	AMDSMI_FW_ID_MES_KIQ:                  "FW_ID_MES_KIQ",
	AMDSMI_FW_ID_MES_STACK:                "FW_ID_MES_STACK",
	AMDSMI_FW_ID_MES_THREAD1:              "FW_ID_MES_THREAD1",
	AMDSMI_FW_ID_MES_THREAD1_STACK:        "FW_ID_MES_THREAD1_STACK",
	AMDSMI_FW_ID_RLX6:                     "FW_ID_RLX6",
	AMDSMI_FW_ID_RLX6_DRAM_BOOT:           "FW_ID_RLX6_DRAM_BOOT",
	AMDSMI_FW_ID_RS64_ME:                  "FW_ID_RS64_ME",
	AMDSMI_FW_ID_RS64_ME_P0_DATA:          "FW_ID_RS64_ME_P0_DATA",
	AMDSMI_FW_ID_RS64_ME_P1_DATA:          "FW_ID_RS64_ME_P1_DATA",
	AMDSMI_FW_ID_RS64_PFP:                 "FW_ID_RS64_PFP",
	AMDSMI_FW_ID_RS64_PFP_P0_DATA:         "FW_ID_RS64_PFP_P0_DATA",
	AMDSMI_FW_ID_RS64_PFP_P1_DATA:         "FW_ID_RS64_PFP_P1_DATA",
	AMDSMI_FW_ID_RS64_MEC:                 "FW_ID_RS64_MEC",
	AMDSMI_FW_ID_RS64_MEC_P0_DATA:         "FW_ID_RS64_MEC_P0_DATA",
	AMDSMI_FW_ID_RS64_MEC_P1_DATA:         "FW_ID_RS64_MEC_P1_DATA",
	AMDSMI_FW_ID_RS64_MEC_P2_DATA:         "FW_ID_RS64_MEC_P2_DATA",
	AMDSMI_FW_ID_RS64_MEC_P3_DATA:         "FW_ID_RS64_MEC_P3_DATA",
	AMDSMI_FW_ID_PPTABLE:                  "FW_ID_PPTABLE",
	AMDSMI_FW_ID_PSP_SOC:                  "FW_ID_PSP_SOC",
	AMDSMI_FW_ID_PSP_DBG:                  "FW_ID_PSP_DBG",
	AMDSMI_FW_ID_PSP_INTF:                 "FW_ID_PSP_INTF",
	AMDSMI_FW_ID_RLX6_CORE1:               "FW_ID_RLX6_CORE1",
	AMDSMI_FW_ID_RLX6_DRAM_BOOT_CORE1:     "FW_ID_RLX6_DRAM_BOOT_CORE1",
	AMDSMI_FW_ID_RLCV_LX7:                 "FW_ID_RLCV_LX7",
	AMDSMI_FW_ID_RLC_SAVE_RESTORE_LIST:    "FW_ID_RLC_SAVE_RESTORE_LIST",
	AMDSMI_FW_ID_ASD:                      "FW_ID_ASD",
	AMDSMI_FW_ID_TA_RAS:                   "FW_ID_TA_RAS",
	AMDSMI_FW_ID_TA_XGMI:                  "FW_ID_TA_XGMI",
	AMDSMI_FW_ID_RLC_SRLG:                 "FW_ID_RLC_SRLG",
	AMDSMI_FW_ID_RLC_SRLS:                 "FW_ID_RLC_SRLS",
	AMDSMI_FW_ID_PM:                       "FW_ID_PM",
	AMDSMI_FW_ID_DMCU:                     "FW_ID_DMCU",
	AMDSMI_FW_ID_PLDM_BUNDLE:              "FW_ID_PLDM_BUNDLE",
	AMDSMI_FW_ID__MAX:                     "FW_ID__MAX",
}

func (b FwBlock) String() string {
	if name, ok := fwBlockNames[b]; ok {
		return name
	}
	return fmt.Sprintf("UNKNOWN(%d)", int32(b))
}

type FwInfoList struct {
	FwID      FwBlock
	FwVersion uint64
}

type FwInfo struct {
	NumFwInfo uint8
	FwList    [AMDSMI_FW_ID__MAX]FwInfoList
}

type VbiosInfo struct {
	Name         string
	BuildDate    string
	PartNumber   string
	Version      string
	BootFirmware string
}

const maxFirmwareEntries int = C.AMDSMI_FW_ID__MAX

const (
	AMDSMI_FW_ID_SMU                      FwBlock = C.AMDSMI_FW_ID_SMU
	AMDSMI_FW_ID_FIRST                    FwBlock = C.AMDSMI_FW_ID_FIRST
	AMDSMI_FW_ID_CP_CE                    FwBlock = C.AMDSMI_FW_ID_CP_CE
	AMDSMI_FW_ID_CP_PFP                   FwBlock = C.AMDSMI_FW_ID_CP_PFP
	AMDSMI_FW_ID_CP_ME                    FwBlock = C.AMDSMI_FW_ID_CP_ME
	AMDSMI_FW_ID_CP_MEC_JT1               FwBlock = C.AMDSMI_FW_ID_CP_MEC_JT1
	AMDSMI_FW_ID_CP_MEC_JT2               FwBlock = C.AMDSMI_FW_ID_CP_MEC_JT2
	AMDSMI_FW_ID_CP_MEC1                  FwBlock = C.AMDSMI_FW_ID_CP_MEC1
	AMDSMI_FW_ID_CP_MEC2                  FwBlock = C.AMDSMI_FW_ID_CP_MEC2
	AMDSMI_FW_ID_RLC                      FwBlock = C.AMDSMI_FW_ID_RLC
	AMDSMI_FW_ID_SDMA0                    FwBlock = C.AMDSMI_FW_ID_SDMA0
	AMDSMI_FW_ID_SDMA1                    FwBlock = C.AMDSMI_FW_ID_SDMA1
	AMDSMI_FW_ID_SDMA2                    FwBlock = C.AMDSMI_FW_ID_SDMA2
	AMDSMI_FW_ID_SDMA3                    FwBlock = C.AMDSMI_FW_ID_SDMA3
	AMDSMI_FW_ID_SDMA4                    FwBlock = C.AMDSMI_FW_ID_SDMA4
	AMDSMI_FW_ID_SDMA5                    FwBlock = C.AMDSMI_FW_ID_SDMA5
	AMDSMI_FW_ID_SDMA6                    FwBlock = C.AMDSMI_FW_ID_SDMA6
	AMDSMI_FW_ID_SDMA7                    FwBlock = C.AMDSMI_FW_ID_SDMA7
	AMDSMI_FW_ID_VCN                      FwBlock = C.AMDSMI_FW_ID_VCN
	AMDSMI_FW_ID_UVD                      FwBlock = C.AMDSMI_FW_ID_UVD
	AMDSMI_FW_ID_VCE                      FwBlock = C.AMDSMI_FW_ID_VCE
	AMDSMI_FW_ID_ISP                      FwBlock = C.AMDSMI_FW_ID_ISP
	AMDSMI_FW_ID_DMCU_ERAM                FwBlock = C.AMDSMI_FW_ID_DMCU_ERAM
	AMDSMI_FW_ID_DMCU_ISR                 FwBlock = C.AMDSMI_FW_ID_DMCU_ISR
	AMDSMI_FW_ID_RLC_RESTORE_LIST_GPM_MEM FwBlock = C.AMDSMI_FW_ID_RLC_RESTORE_LIST_GPM_MEM
	AMDSMI_FW_ID_RLC_RESTORE_LIST_SRM_MEM FwBlock = C.AMDSMI_FW_ID_RLC_RESTORE_LIST_SRM_MEM
	AMDSMI_FW_ID_RLC_RESTORE_LIST_CNTL    FwBlock = C.AMDSMI_FW_ID_RLC_RESTORE_LIST_CNTL
	AMDSMI_FW_ID_RLC_V                    FwBlock = C.AMDSMI_FW_ID_RLC_V
	AMDSMI_FW_ID_MMSCH                    FwBlock = C.AMDSMI_FW_ID_MMSCH
	AMDSMI_FW_ID_PSP_SYSDRV               FwBlock = C.AMDSMI_FW_ID_PSP_SYSDRV
	AMDSMI_FW_ID_PSP_SOSDRV               FwBlock = C.AMDSMI_FW_ID_PSP_SOSDRV
	AMDSMI_FW_ID_PSP_TOC                  FwBlock = C.AMDSMI_FW_ID_PSP_TOC
	AMDSMI_FW_ID_PSP_KEYDB                FwBlock = C.AMDSMI_FW_ID_PSP_KEYDB
	AMDSMI_FW_ID_DFC                      FwBlock = C.AMDSMI_FW_ID_DFC
	AMDSMI_FW_ID_PSP_SPL                  FwBlock = C.AMDSMI_FW_ID_PSP_SPL
	AMDSMI_FW_ID_DRV_CAP                  FwBlock = C.AMDSMI_FW_ID_DRV_CAP
	AMDSMI_FW_ID_MC                       FwBlock = C.AMDSMI_FW_ID_MC
	AMDSMI_FW_ID_PSP_BL                   FwBlock = C.AMDSMI_FW_ID_PSP_BL
	AMDSMI_FW_ID_CP_PM4                   FwBlock = C.AMDSMI_FW_ID_CP_PM4
	AMDSMI_FW_ID_RLC_P                    FwBlock = C.AMDSMI_FW_ID_RLC_P
	AMDSMI_FW_ID_SEC_POLICY_STAGE2        FwBlock = C.AMDSMI_FW_ID_SEC_POLICY_STAGE2
	AMDSMI_FW_ID_REG_ACCESS_WHITELIST     FwBlock = C.AMDSMI_FW_ID_REG_ACCESS_WHITELIST
	AMDSMI_FW_ID_IMU_DRAM                 FwBlock = C.AMDSMI_FW_ID_IMU_DRAM
	AMDSMI_FW_ID_IMU_IRAM                 FwBlock = C.AMDSMI_FW_ID_IMU_IRAM
	AMDSMI_FW_ID_SDMA_TH0                 FwBlock = C.AMDSMI_FW_ID_SDMA_TH0
	AMDSMI_FW_ID_SDMA_TH1                 FwBlock = C.AMDSMI_FW_ID_SDMA_TH1
	AMDSMI_FW_ID_CP_MES                   FwBlock = C.AMDSMI_FW_ID_CP_MES
	AMDSMI_FW_ID_MES_KIQ                  FwBlock = C.AMDSMI_FW_ID_MES_KIQ
	AMDSMI_FW_ID_MES_STACK                FwBlock = C.AMDSMI_FW_ID_MES_STACK
	AMDSMI_FW_ID_MES_THREAD1              FwBlock = C.AMDSMI_FW_ID_MES_THREAD1
	AMDSMI_FW_ID_MES_THREAD1_STACK        FwBlock = C.AMDSMI_FW_ID_MES_THREAD1_STACK
	AMDSMI_FW_ID_RLX6                     FwBlock = C.AMDSMI_FW_ID_RLX6
	AMDSMI_FW_ID_RLX6_DRAM_BOOT           FwBlock = C.AMDSMI_FW_ID_RLX6_DRAM_BOOT
	AMDSMI_FW_ID_RS64_ME                  FwBlock = C.AMDSMI_FW_ID_RS64_ME
	AMDSMI_FW_ID_RS64_ME_P0_DATA          FwBlock = C.AMDSMI_FW_ID_RS64_ME_P0_DATA
	AMDSMI_FW_ID_RS64_ME_P1_DATA          FwBlock = C.AMDSMI_FW_ID_RS64_ME_P1_DATA
	AMDSMI_FW_ID_RS64_PFP                 FwBlock = C.AMDSMI_FW_ID_RS64_PFP
	AMDSMI_FW_ID_RS64_PFP_P0_DATA         FwBlock = C.AMDSMI_FW_ID_RS64_PFP_P0_DATA
	AMDSMI_FW_ID_RS64_PFP_P1_DATA         FwBlock = C.AMDSMI_FW_ID_RS64_PFP_P1_DATA
	AMDSMI_FW_ID_RS64_MEC                 FwBlock = C.AMDSMI_FW_ID_RS64_MEC
	AMDSMI_FW_ID_RS64_MEC_P0_DATA         FwBlock = C.AMDSMI_FW_ID_RS64_MEC_P0_DATA
	AMDSMI_FW_ID_RS64_MEC_P1_DATA         FwBlock = C.AMDSMI_FW_ID_RS64_MEC_P1_DATA
	AMDSMI_FW_ID_RS64_MEC_P2_DATA         FwBlock = C.AMDSMI_FW_ID_RS64_MEC_P2_DATA
	AMDSMI_FW_ID_RS64_MEC_P3_DATA         FwBlock = C.AMDSMI_FW_ID_RS64_MEC_P3_DATA
	AMDSMI_FW_ID_PPTABLE                  FwBlock = C.AMDSMI_FW_ID_PPTABLE
	AMDSMI_FW_ID_PSP_SOC                  FwBlock = C.AMDSMI_FW_ID_PSP_SOC
	AMDSMI_FW_ID_PSP_DBG                  FwBlock = C.AMDSMI_FW_ID_PSP_DBG
	AMDSMI_FW_ID_PSP_INTF                 FwBlock = C.AMDSMI_FW_ID_PSP_INTF
	AMDSMI_FW_ID_RLX6_CORE1               FwBlock = C.AMDSMI_FW_ID_RLX6_CORE1
	AMDSMI_FW_ID_RLX6_DRAM_BOOT_CORE1     FwBlock = C.AMDSMI_FW_ID_RLX6_DRAM_BOOT_CORE1
	AMDSMI_FW_ID_RLCV_LX7                 FwBlock = C.AMDSMI_FW_ID_RLCV_LX7
	AMDSMI_FW_ID_RLC_SAVE_RESTORE_LIST    FwBlock = C.AMDSMI_FW_ID_RLC_SAVE_RESTORE_LIST
	AMDSMI_FW_ID_ASD                      FwBlock = C.AMDSMI_FW_ID_ASD
	AMDSMI_FW_ID_TA_RAS                   FwBlock = C.AMDSMI_FW_ID_TA_RAS
	AMDSMI_FW_ID_TA_XGMI                  FwBlock = C.AMDSMI_FW_ID_TA_XGMI
	AMDSMI_FW_ID_RLC_SRLG                 FwBlock = C.AMDSMI_FW_ID_RLC_SRLG
	AMDSMI_FW_ID_RLC_SRLS                 FwBlock = C.AMDSMI_FW_ID_RLC_SRLS
	AMDSMI_FW_ID_PM                       FwBlock = C.AMDSMI_FW_ID_PM
	AMDSMI_FW_ID_DMCU                     FwBlock = C.AMDSMI_FW_ID_DMCU
	AMDSMI_FW_ID_PLDM_BUNDLE              FwBlock = C.AMDSMI_FW_ID_PLDM_BUNDLE
	AMDSMI_FW_ID__MAX                     FwBlock = C.AMDSMI_FW_ID__MAX
)

func GetGpuDeviceUuid(h ProcessorHandle) (string, error) {
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

func GetGpuAsicInfo(h ProcessorHandle) (AsicInfo, error) {
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
			PhysicalAccId: uint32(out.physical_acc_id),
			ChipRevId:     uint32(out.chip_rev_id), ExternalRevId: uint32(out.external_rev_id),
		}, nil
	})
}

func GetGpuDriverInfo(h ProcessorHandle) (DriverInfo, error) {
	const op = "amdsmi_get_gpu_driver_info"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (DriverInfo, error) {
		var out C.amdsmi_driver_info_t
		if err := nativeErrorLocked(op, C.amdsmi_get_gpu_driver_info(p, &out)); err != nil {
			return DriverInfo{}, err
		}
		return DriverInfo{
			DriverVersion: boundedString(&out.driver_version[0], len(out.driver_version)),
			DriverDate:    boundedString(&out.driver_date[0], len(out.driver_date)),
			DriverName:    boundedString(&out.driver_name[0], len(out.driver_name)),
		}, nil
	})
}

func GetGpuBoardInfo(h ProcessorHandle) (BoardInfo, error) {
	const op = "amdsmi_get_gpu_board_info"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (BoardInfo, error) {
		var out C.amdsmi_board_info_t
		if err := nativeErrorLocked(op, C.amdsmi_get_gpu_board_info(p, &out)); err != nil {
			return BoardInfo{}, err
		}
		return BoardInfo{
			ModelNumber:      boundedString(&out.model_number[0], len(out.model_number)),
			ProductSerial:    boundedString(&out.product_serial[0], len(out.product_serial)),
			FruID:            boundedString(&out.fru_id[0], len(out.fru_id)),
			ProductName:      boundedString(&out.product_name[0], len(out.product_name)),
			ManufacturerName: boundedString(&out.manufacturer_name[0], len(out.manufacturer_name)),
		}, nil
	})
}

func GetFwInfo(h ProcessorHandle) (FwInfo, error) {
	const op = "amdsmi_get_fw_info"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (FwInfo, error) {
		var out C.amdsmi_fw_info_t
		if err := nativeErrorLocked(op, C.amdsmi_get_fw_info(p, &out)); err != nil {
			return FwInfo{}, err
		}
		n, err := checkedCount(op, uint64(out.num_fw_info), len(out.fw_info_list))
		if err != nil {
			return FwInfo{}, err
		}
		if _, err := checkedCount(op, uint64(n), int(^uint8(0))); err != nil {
			return FwInfo{}, err
		}
		result := FwInfo{NumFwInfo: uint8(n)}
		for i := 0; i < n; i++ {
			result.FwList[i] = FwInfoList{FwID: FwBlock(out.fw_info_list[i].fw_id),
				FwVersion: uint64(out.fw_info_list[i].fw_version)}
		}
		return result, nil
	})
}

func GetGpuVbiosInfo(h ProcessorHandle) (VbiosInfo, error) {
	const op = "amdsmi_get_gpu_vbios_info"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (VbiosInfo, error) {
		var out C.amdsmi_vbios_info_t
		if err := nativeErrorLocked(op, C.amdsmi_get_gpu_vbios_info(p, &out)); err != nil {
			return VbiosInfo{}, err
		}
		return VbiosInfo{
			Name:         boundedString(&out.name[0], len(out.name)),
			BuildDate:    boundedString(&out.build_date[0], len(out.build_date)),
			PartNumber:   boundedString(&out.part_number[0], len(out.part_number)),
			Version:      boundedString(&out.version[0], len(out.version)),
			BootFirmware: boundedString(&out.boot_firmware[0], len(out.boot_firmware)),
		}, nil
	})
}

type TemperatureType int32
type TemperatureMetric int32
type ClkType int32
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
	AMDSMI_CLK_TYPE_SYS   ClkType = C.AMDSMI_CLK_TYPE_SYS
	AMDSMI_CLK_TYPE_FIRST ClkType = C.AMDSMI_CLK_TYPE_FIRST
	AMDSMI_CLK_TYPE_GFX   ClkType = C.AMDSMI_CLK_TYPE_GFX
	AMDSMI_CLK_TYPE_DF    ClkType = C.AMDSMI_CLK_TYPE_DF
	AMDSMI_CLK_TYPE_DCEF  ClkType = C.AMDSMI_CLK_TYPE_DCEF
	AMDSMI_CLK_TYPE_SOC   ClkType = C.AMDSMI_CLK_TYPE_SOC
	AMDSMI_CLK_TYPE_MEM   ClkType = C.AMDSMI_CLK_TYPE_MEM
	AMDSMI_CLK_TYPE_PCIE  ClkType = C.AMDSMI_CLK_TYPE_PCIE
	AMDSMI_CLK_TYPE_VCLK0 ClkType = C.AMDSMI_CLK_TYPE_VCLK0
	AMDSMI_CLK_TYPE_VCLK1 ClkType = C.AMDSMI_CLK_TYPE_VCLK1
	AMDSMI_CLK_TYPE_DCLK0 ClkType = C.AMDSMI_CLK_TYPE_DCLK0
	AMDSMI_CLK_TYPE_DCLK1 ClkType = C.AMDSMI_CLK_TYPE_DCLK1
	AMDSMI_CLK_TYPE__MAX  ClkType = C.AMDSMI_CLK_TYPE__MAX
)

const (
	AMDSMI_MEM_TYPE_FIRST    MemoryType = C.AMDSMI_MEM_TYPE_FIRST
	AMDSMI_MEM_TYPE_VRAM     MemoryType = C.AMDSMI_MEM_TYPE_VRAM
	AMDSMI_MEM_TYPE_VIS_VRAM MemoryType = C.AMDSMI_MEM_TYPE_VIS_VRAM
	AMDSMI_MEM_TYPE_GTT      MemoryType = C.AMDSMI_MEM_TYPE_GTT
	AMDSMI_MEM_TYPE_LAST     MemoryType = C.AMDSMI_MEM_TYPE_LAST
)

// PowerInfo retains BM W, mV, and PowerLimit uW without conversion to Host units.
// Unavailable values retain their native widths.
type PowerInfo struct {
	SocketPower        uint64
	CurrentSocketPower uint32
	AverageSocketPower uint32
	GfxVoltage         uint64
	SocVoltage         uint64
	MemVoltage         uint64
	PowerLimit         uint32
	UbbPower           uint32
}

// PowerCapInfo uses uW except DpmCap, a BM level index, not MHz.
// Auxiliary fields can be zero after partial native success.
type PowerCapInfo struct {
	PowerCap        uint64
	DefaultPowerCap uint64
	DpmCap          uint64
	MinPowerCap     uint64
	MaxPowerCap     uint64
}

// ClkInfo clocks are MHz. Booleans are false/unavailable on BM, not measured false.
// BM raw bytes are preserved; they do not reliably encode Boolean states.
type ClkInfo struct {
	Clk             uint32
	MinClk          uint32
	MaxClk          uint32
	ClkLocked       bool
	ClkDeepSleep    bool
	ClkLockedRaw    uint8
	ClkDeepSleepRaw uint8
}

type Frequencies struct {
	HasDeepSleep bool
	NumSupported uint32
	Current      uint32 // May be UINT32_MAX or outside Values.
	Values       []uint64
}

type EngineUsage struct {
	GfxActivity uint32 // 65535 can indicate unavailable data.
	UmcActivity uint32
	MmActivity  uint32
}

// VramInfo retains native MB and GB/s units.
type VramInfo struct {
	VramType         VramType
	VramVendor       string
	VramSize         uint64
	VramBitWidth     uint32
	VramMaxBandwidth uint64
}

type VramType int32

var vramTypeNames = map[VramType]string{
	AMDSMI_VRAM_TYPE_UNKNOWN: "UNKNOWN",
	AMDSMI_VRAM_TYPE_HBM:     "HBM",
	AMDSMI_VRAM_TYPE_HBM2:    "HBM2",
	AMDSMI_VRAM_TYPE_HBM2E:   "HBM2E",
	AMDSMI_VRAM_TYPE_HBM3:    "HBM3",
	AMDSMI_VRAM_TYPE_HBM3E:   "HBM3E",
	AMDSMI_VRAM_TYPE_HBM4:    "HBM4",
	AMDSMI_VRAM_TYPE_DDR2:    "DDR2",
	AMDSMI_VRAM_TYPE_DDR3:    "DDR3",
	AMDSMI_VRAM_TYPE_DDR4:    "DDR4",
	AMDSMI_VRAM_TYPE_DDR5:    "DDR5",
	AMDSMI_VRAM_TYPE_GDDR1:   "GDDR1",
	AMDSMI_VRAM_TYPE_GDDR2:   "GDDR2",
	AMDSMI_VRAM_TYPE_GDDR3:   "GDDR3",
	AMDSMI_VRAM_TYPE_GDDR4:   "GDDR4",
	AMDSMI_VRAM_TYPE_GDDR5:   "GDDR5",
	AMDSMI_VRAM_TYPE_GDDR6:   "GDDR6",
	AMDSMI_VRAM_TYPE_GDDR7:   "GDDR7",
	AMDSMI_VRAM_TYPE_LPDDR4:  "LPDDR4",
	AMDSMI_VRAM_TYPE_LPDDR5:  "LPDDR5",
}

func (v VramType) String() string {
	if name, ok := vramTypeNames[v]; ok {
		return name
	}
	return fmt.Sprintf("UNKNOWN(%d)", int32(v))
}

const (
	AMDSMI_VRAM_TYPE_UNKNOWN VramType = C.AMDSMI_VRAM_TYPE_UNKNOWN
	AMDSMI_VRAM_TYPE_HBM     VramType = C.AMDSMI_VRAM_TYPE_HBM
	AMDSMI_VRAM_TYPE_HBM2    VramType = C.AMDSMI_VRAM_TYPE_HBM2
	AMDSMI_VRAM_TYPE_HBM2E   VramType = C.AMDSMI_VRAM_TYPE_HBM2E
	AMDSMI_VRAM_TYPE_HBM3    VramType = C.AMDSMI_VRAM_TYPE_HBM3
	AMDSMI_VRAM_TYPE_HBM3E   VramType = C.AMDSMI_VRAM_TYPE_HBM3E
	AMDSMI_VRAM_TYPE_HBM4    VramType = C.AMDSMI_VRAM_TYPE_HBM4
	AMDSMI_VRAM_TYPE_DDR2    VramType = C.AMDSMI_VRAM_TYPE_DDR2
	AMDSMI_VRAM_TYPE_DDR3    VramType = C.AMDSMI_VRAM_TYPE_DDR3
	AMDSMI_VRAM_TYPE_DDR4    VramType = C.AMDSMI_VRAM_TYPE_DDR4
	AMDSMI_VRAM_TYPE_DDR5    VramType = C.AMDSMI_VRAM_TYPE_DDR5
	AMDSMI_VRAM_TYPE_GDDR1   VramType = C.AMDSMI_VRAM_TYPE_GDDR1
	AMDSMI_VRAM_TYPE_GDDR2   VramType = C.AMDSMI_VRAM_TYPE_GDDR2
	AMDSMI_VRAM_TYPE_GDDR3   VramType = C.AMDSMI_VRAM_TYPE_GDDR3
	AMDSMI_VRAM_TYPE_GDDR4   VramType = C.AMDSMI_VRAM_TYPE_GDDR4
	AMDSMI_VRAM_TYPE_GDDR5   VramType = C.AMDSMI_VRAM_TYPE_GDDR5
	AMDSMI_VRAM_TYPE_GDDR6   VramType = C.AMDSMI_VRAM_TYPE_GDDR6
	AMDSMI_VRAM_TYPE_GDDR7   VramType = C.AMDSMI_VRAM_TYPE_GDDR7
	AMDSMI_VRAM_TYPE_LPDDR4  VramType = C.AMDSMI_VRAM_TYPE_LPDDR4
	AMDSMI_VRAM_TYPE_LPDDR5  VramType = C.AMDSMI_VRAM_TYPE_LPDDR5
	AMDSMI_VRAM_TYPE__MAX    VramType = C.AMDSMI_VRAM_TYPE__MAX
)

// GetTempMetric returns whole degrees Celsius, preserving negative values.
func GetTempMetric(h ProcessorHandle, sensor TemperatureType, metric TemperatureMetric) (int64, error) {
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
			SocketPower:        uint64(out.socket_power),
			CurrentSocketPower: uint32(out.current_socket_power),
			AverageSocketPower: uint32(out.average_socket_power),
			GfxVoltage:         uint64(out.gfx_voltage),
			SocVoltage:         uint64(out.soc_voltage),
			MemVoltage:         uint64(out.mem_voltage),
			PowerLimit:         uint32(out.power_limit),
			UbbPower:           uint32(out.ubb_power),
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
			PowerCap:        uint64(out.power_cap),
			DefaultPowerCap: uint64(out.default_power_cap),
			DpmCap:          uint64(out.dpm_cap),
			MinPowerCap:     uint64(out.min_power_cap),
			MaxPowerCap:     uint64(out.max_power_cap),
		}, nil
	})
}

func GetClockInfo(h ProcessorHandle, clock ClkType) (ClkInfo, error) {
	const op = "amdsmi_get_clock_info"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (ClkInfo, error) {
		var out C.amdsmi_clk_info_t
		status := C.amdsmi_get_clock_info(p, C.amdsmi_clk_type_t(clock), &out)
		if err := nativeErrorLocked(op, status); err != nil {
			return ClkInfo{}, err
		}
		return ClkInfo{
			Clk:             uint32(out.clk),
			MinClk:          uint32(out.min_clk),
			MaxClk:          uint32(out.max_clk),
			ClkLockedRaw:    uint8(out.clk_locked),
			ClkDeepSleepRaw: uint8(out.clk_deep_sleep),
		}, nil
	})
}

// GetClockFrequencies returns BM frequencies in Hz, not GetClockInfo's MHz.
func GetClockFrequencies(h ProcessorHandle, clock ClkType) (Frequencies, error) {
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
			NumSupported: uint32(n),
			Current:      uint32(out.current),
			Values:       make([]uint64, n),
		}
		for i := range result.Values {
			result.Values[i] = uint64(out.frequency[i])
		}
		return result, nil
	})
}

func GetGpuActivity(h ProcessorHandle) (EngineUsage, error) {
	const op = "amdsmi_get_gpu_activity"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (EngineUsage, error) {
		var out C.amdsmi_engine_usage_t
		if err := nativeErrorLocked(op, C.amdsmi_get_gpu_activity(p, &out)); err != nil {
			return EngineUsage{}, err
		}
		return EngineUsage{
			GfxActivity: uint32(out.gfx_activity),
			UmcActivity: uint32(out.umc_activity),
			MmActivity:  uint32(out.mm_activity),
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

func GetGpuVramInfo(h ProcessorHandle) (VramInfo, error) {
	const op = "amdsmi_get_gpu_vram_info"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (VramInfo, error) {
		var out C.amdsmi_vram_info_t
		if err := nativeErrorLocked(op, C.amdsmi_get_gpu_vram_info(p, &out)); err != nil {
			return VramInfo{}, err
		}
		return VramInfo{
			VramType:         VramType(out.vram_type),
			VramVendor:       boundedString(&out.vram_vendor[0], len(out.vram_vendor)),
			VramSize:         uint64(out.vram_size),
			VramBitWidth:     uint32(out.vram_bit_width),
			VramMaxBandwidth: uint64(out.vram_max_bandwidth),
		}, nil
	})
}

type KFDInfo struct {
	KFDID              uint64
	NodeID             uint32
	CurrentPartitionID uint32
}

type MemoryPartitionType int32
type AcceleratorPartitionType int32

var acceleratorPartitionTypeNames = map[AcceleratorPartitionType]string{
	AMDSMI_ACCELERATOR_PARTITION_INVALID: "INVALID",
	AMDSMI_ACCELERATOR_PARTITION_SPX:     "SPX",
	AMDSMI_ACCELERATOR_PARTITION_DPX:     "DPX",
	AMDSMI_ACCELERATOR_PARTITION_TPX:     "TPX",
	AMDSMI_ACCELERATOR_PARTITION_QPX:     "QPX",
	AMDSMI_ACCELERATOR_PARTITION_CPX:     "CPX",
}

func (a AcceleratorPartitionType) String() string {
	if name, ok := acceleratorPartitionTypeNames[a]; ok {
		return name
	}
	return fmt.Sprintf("UNKNOWN(%d)", int32(a))
}

type NpsCaps struct {
	Nps1Cap bool
	Nps2Cap bool
	Nps4Cap bool
	Nps8Cap bool
	RawMask uint32 // BM extension preserves unknown native capability bits.
}

func npsCaps(value C.amdsmi_nps_caps_t) NpsCaps {
	return NpsCaps{
		Nps1Cap: C.go_amdsmi_nps_cap(value, 0) != 0,
		Nps2Cap: C.go_amdsmi_nps_cap(value, 1) != 0,
		Nps4Cap: C.go_amdsmi_nps_cap(value, 2) != 0,
		Nps8Cap: C.go_amdsmi_nps_cap(value, 3) != 0,
		RawMask: uint32(C.go_amdsmi_nps_mask(value)),
	}
}

func (n NpsCaps) Supported() []MemoryPartitionType {
	var result []MemoryPartitionType
	for _, cap := range []struct {
		enabled bool
		mode    MemoryPartitionType
	}{
		{n.Nps1Cap, AMDSMI_MEMORY_PARTITION_NPS1},
		{n.Nps2Cap, AMDSMI_MEMORY_PARTITION_NPS2},
		{n.Nps4Cap, AMDSMI_MEMORY_PARTITION_NPS4},
		{n.Nps8Cap, AMDSMI_MEMORY_PARTITION_NPS8},
	} {
		if cap.enabled {
			result = append(result, cap.mode)
		}
	}
	return result
}

func (n NpsCaps) String() string { return fmt.Sprint(n.Supported()) }

func (m MemoryPartitionType) String() string {
	for _, mode := range []struct {
		value MemoryPartitionType
		name  string
	}{
		{AMDSMI_MEMORY_PARTITION_NPS1, "NPS1"},
		{AMDSMI_MEMORY_PARTITION_NPS2, "NPS2"},
		{AMDSMI_MEMORY_PARTITION_NPS4, "NPS4"},
		{AMDSMI_MEMORY_PARTITION_NPS8, "NPS8"},
	} {
		if m == mode.value {
			return mode.name
		}
	}
	return fmt.Sprintf("UNKNOWN(%d)", int32(m))
}

type NumaRange struct {
	MemoryType VramType
	Start      uint64
	End        uint64
}

type MemoryPartitionConfig struct {
	PartitionCaps NpsCaps
	Mode          MemoryPartitionType
	NumNumaRanges uint32
	NumaRanges    [AMDSMI_MAX_NUM_NUMA_NODES]NumaRange
}

type AcceleratorPartitionProfile struct {
	ProfileType   AcceleratorPartitionType
	NumPartitions uint32
	MemoryCaps    NpsCaps
	ProfileIndex  uint32
	NumResources  uint32
	// Current native profiles leave resource metadata empty.
	Resources [][]uint32
}

const (
	AMDSMI_MAX_NUM_NUMA_NODES     = C.AMDSMI_MAX_NUM_NUMA_NODES
	maxNUMARanges             int = C.AMDSMI_MAX_NUM_NUMA_NODES
	maxAcceleratorPartitions  int = C.AMDSMI_MAX_ACCELERATOR_PARTITIONS
	maxProfileResources       int = C.AMDSMI_MAX_CP_PROFILE_RESOURCES
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

func GetGpuMemoryPartitionConfig(h ProcessorHandle) (MemoryPartitionConfig, error) {
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
			PartitionCaps: npsCaps(out.partition_caps),
			Mode:          MemoryPartitionType(out.mp_mode),
			NumNumaRanges: uint32(n),
		}
		for i := 0; i < n; i++ {
			value := out.numa_range[i]
			result.NumaRanges[i] = NumaRange{MemoryType: VramType(value.memory_type),
				Start: uint64(value.start), End: uint64(value.end)}
		}
		return result, nil
	})
}

// GetGpuAcceleratorPartitionProfile returns only the current BM ID in its slice.
// The native buffer is full-sized even though BM populates only slot zero.
func GetGpuAcceleratorPartitionProfile(h ProcessorHandle) (AcceleratorPartitionProfile, []uint32, error) {
	const op = "amdsmi_get_gpu_accelerator_partition_profile"
	var ids []uint32
	profile, err := withProcessor(h, op, func(p C.amdsmi_processor_handle) (AcceleratorPartitionProfile, error) {
		var out C.amdsmi_accelerator_partition_profile_t
		var partitionIDs [C.AMDSMI_MAX_ACCELERATOR_PARTITIONS]C.uint32_t
		status := C.amdsmi_get_gpu_accelerator_partition_profile(p, &out, &partitionIDs[0])
		if err := nativeErrorLocked(op, status); err != nil {
			return AcceleratorPartitionProfile{}, err
		}
		result := AcceleratorPartitionProfile{
			ProfileType:   AcceleratorPartitionType(out.profile_type),
			NumPartitions: uint32(out.num_partitions),
			MemoryCaps:    npsCaps(out.memory_caps),
			ProfileIndex:  uint32(out.profile_index),
			NumResources:  uint32(out.num_resources),
		}
		if uint32(out.num_partitions) == ^uint32(0) && out.num_resources == 0 {
			ids = []uint32{uint32(partitionIDs[0])}
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
		ids = []uint32{uint32(partitionIDs[0])}
		if columns == 0 {
			return result, nil
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
	return profile, ids, err
}

type GpuBlock uint64
type RASState uint32

type ErrorCount struct {
	CorrectableCount   uint64
	UncorrectableCount uint64
	DeferredCount      uint64
}

// RasFeatureInfo contains populated native metadata, not an overall health assessment.
type RasFeatureInfo struct {
	RasEepromVersion        uint32
	EccCorrectionSchemaFlag uint32
}

const (
	AMDSMI_GPU_BLOCK_INVALID    GpuBlock = C.AMDSMI_GPU_BLOCK_INVALID
	AMDSMI_GPU_BLOCK_FIRST      GpuBlock = C.AMDSMI_GPU_BLOCK_FIRST
	AMDSMI_GPU_BLOCK_UMC        GpuBlock = C.AMDSMI_GPU_BLOCK_UMC
	AMDSMI_GPU_BLOCK_SDMA       GpuBlock = C.AMDSMI_GPU_BLOCK_SDMA
	AMDSMI_GPU_BLOCK_GFX        GpuBlock = C.AMDSMI_GPU_BLOCK_GFX
	AMDSMI_GPU_BLOCK_MMHUB      GpuBlock = C.AMDSMI_GPU_BLOCK_MMHUB
	AMDSMI_GPU_BLOCK_ATHUB      GpuBlock = C.AMDSMI_GPU_BLOCK_ATHUB
	AMDSMI_GPU_BLOCK_PCIE_BIF   GpuBlock = C.AMDSMI_GPU_BLOCK_PCIE_BIF
	AMDSMI_GPU_BLOCK_HDP        GpuBlock = C.AMDSMI_GPU_BLOCK_HDP
	AMDSMI_GPU_BLOCK_XGMI_WAFL  GpuBlock = C.AMDSMI_GPU_BLOCK_XGMI_WAFL
	AMDSMI_GPU_BLOCK_DF         GpuBlock = C.AMDSMI_GPU_BLOCK_DF
	AMDSMI_GPU_BLOCK_SMN        GpuBlock = C.AMDSMI_GPU_BLOCK_SMN
	AMDSMI_GPU_BLOCK_SEM        GpuBlock = C.AMDSMI_GPU_BLOCK_SEM
	AMDSMI_GPU_BLOCK_MP0        GpuBlock = C.AMDSMI_GPU_BLOCK_MP0
	AMDSMI_GPU_BLOCK_MP1        GpuBlock = C.AMDSMI_GPU_BLOCK_MP1
	AMDSMI_GPU_BLOCK_FUSE       GpuBlock = C.AMDSMI_GPU_BLOCK_FUSE
	AMDSMI_GPU_BLOCK_MCA        GpuBlock = C.AMDSMI_GPU_BLOCK_MCA
	AMDSMI_GPU_BLOCK_VCN        GpuBlock = C.AMDSMI_GPU_BLOCK_VCN
	AMDSMI_GPU_BLOCK_JPEG       GpuBlock = C.AMDSMI_GPU_BLOCK_JPEG
	AMDSMI_GPU_BLOCK_IH         GpuBlock = C.AMDSMI_GPU_BLOCK_IH
	AMDSMI_GPU_BLOCK_MPIO       GpuBlock = C.AMDSMI_GPU_BLOCK_MPIO
	AMDSMI_GPU_BLOCK_MMSCH      GpuBlock = C.AMDSMI_GPU_BLOCK_MMSCH
	AMDSMI_GPU_BLOCK_MP5        GpuBlock = C.AMDSMI_GPU_BLOCK_MP5
	AMDSMI_GPU_BLOCK_ATU        GpuBlock = C.AMDSMI_GPU_BLOCK_ATU
	AMDSMI_GPU_BLOCK_DACC_BE    GpuBlock = C.AMDSMI_GPU_BLOCK_DACC_BE
	AMDSMI_GPU_BLOCK_ECLR       GpuBlock = C.AMDSMI_GPU_BLOCK_ECLR
	AMDSMI_GPU_BLOCK_KPX_SERDES GpuBlock = C.AMDSMI_GPU_BLOCK_KPX_SERDES
	AMDSMI_GPU_BLOCK_LSDMA      GpuBlock = C.AMDSMI_GPU_BLOCK_LSDMA
	AMDSMI_GPU_BLOCK_MPART      GpuBlock = C.AMDSMI_GPU_BLOCK_MPART
	AMDSMI_GPU_BLOCK_MPIFOE     GpuBlock = C.AMDSMI_GPU_BLOCK_MPIFOE
	AMDSMI_GPU_BLOCK_MPRAS      GpuBlock = C.AMDSMI_GPU_BLOCK_MPRAS
	AMDSMI_GPU_BLOCK_NBIF       GpuBlock = C.AMDSMI_GPU_BLOCK_NBIF
	AMDSMI_GPU_BLOCK_NBIO       GpuBlock = C.AMDSMI_GPU_BLOCK_NBIO
	AMDSMI_GPU_BLOCK_OXRP       GpuBlock = C.AMDSMI_GPU_BLOCK_OXRP
	AMDSMI_GPU_BLOCK_PCIE_PL    GpuBlock = C.AMDSMI_GPU_BLOCK_PCIE_PL
	AMDSMI_GPU_BLOCK_PCS_XGMI   GpuBlock = C.AMDSMI_GPU_BLOCK_PCS_XGMI
	AMDSMI_GPU_BLOCK_PIE        GpuBlock = C.AMDSMI_GPU_BLOCK_PIE
	AMDSMI_GPU_BLOCK_CS         GpuBlock = C.AMDSMI_GPU_BLOCK_CS
	AMDSMI_GPU_BLOCK_SHUB       GpuBlock = C.AMDSMI_GPU_BLOCK_SHUB
	AMDSMI_GPU_BLOCK_SSBDCI     GpuBlock = C.AMDSMI_GPU_BLOCK_SSBDCI
	AMDSMI_GPU_BLOCK_UCIE_PCS   GpuBlock = C.AMDSMI_GPU_BLOCK_UCIE_PCS
	AMDSMI_GPU_BLOCK_LAST       GpuBlock = C.AMDSMI_GPU_BLOCK_LAST
	AMDSMI_GPU_BLOCK_RESERVED   GpuBlock = C.AMDSMI_GPU_BLOCK_RESERVED
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

// GetGpuEccEnabled includes known blocks and any unknown enabled native bits.
func GetGpuEccEnabled(h ProcessorHandle) (map[GpuBlock]bool, error) {
	const op = "amdsmi_get_gpu_ecc_enabled"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (map[GpuBlock]bool, error) {
		var out C.uint64_t
		if err := nativeErrorLocked(op, C.amdsmi_get_gpu_ecc_enabled(p, &out)); err != nil {
			return nil, err
		}
		result := make(map[GpuBlock]bool)
		for bit := GpuBlock(1); bit != 0; bit <<= 1 {
			enabled := GpuBlock(out)&bit != 0
			if enabled || (bit >= AMDSMI_GPU_BLOCK_FIRST && bit <= AMDSMI_GPU_BLOCK_LAST) {
				result[bit] = enabled
			}
		}
		return result, nil
	})
}

func GetGpuEccCount(h ProcessorHandle, block GpuBlock) (ErrorCount, error) {
	const op = "amdsmi_get_gpu_ecc_count"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (ErrorCount, error) {
		var out C.amdsmi_error_count_t
		status := C.amdsmi_get_gpu_ecc_count(p, C.amdsmi_gpu_block_t(block), &out)
		if err := nativeErrorLocked(op, status); err != nil {
			return ErrorCount{}, err
		}
		return ErrorCount{CorrectableCount: uint64(out.correctable_count),
			UncorrectableCount: uint64(out.uncorrectable_count), DeferredCount: uint64(out.deferred_count)}, nil
	})
}

// GetGpuTotalEccCount preserves native totals, which may omit unavailable blocks.
func GetGpuTotalEccCount(h ProcessorHandle) (ErrorCount, error) {
	const op = "amdsmi_get_gpu_total_ecc_count"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (ErrorCount, error) {
		var out C.amdsmi_error_count_t
		if err := nativeErrorLocked(op, C.amdsmi_get_gpu_total_ecc_count(p, &out)); err != nil {
			return ErrorCount{}, err
		}
		return ErrorCount{CorrectableCount: uint64(out.correctable_count),
			UncorrectableCount: uint64(out.uncorrectable_count), DeferredCount: uint64(out.deferred_count)}, nil
	})
}

func GetRASBlockState(h ProcessorHandle, block GpuBlock) (RASState, error) {
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

func GetGpuRasFeatureInfo(h ProcessorHandle) (RasFeatureInfo, error) {
	const op = "amdsmi_get_gpu_ras_feature_info"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (RasFeatureInfo, error) {
		var out C.amdsmi_ras_feature_t
		if err := nativeErrorLocked(op, C.amdsmi_get_gpu_ras_feature_info(p, &out)); err != nil {
			return RasFeatureInfo{}, err
		}
		return RasFeatureInfo{RasEepromVersion: uint32(out.ras_eeprom_version),
			EccCorrectionSchemaFlag: uint32(out.ecc_correction_schema_flag)}, nil
	})
}
