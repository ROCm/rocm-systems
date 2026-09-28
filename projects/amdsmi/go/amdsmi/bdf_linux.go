// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//go:build linux && cgo

package amdsmi

/*
#include <stddef.h>
#include <amd_smi/amdsmi.h>
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
*/
import "C"

import (
	"fmt"
	"unsafe"
)

type BDF struct {
	Domain   uint64
	Bus      uint8
	Device   uint8
	Function uint8
}

func (b BDF) String() string {
	return fmt.Sprintf("%04x:%02x:%02x.%x", b.Domain, b.Bus, b.Device, b.Function)
}

func GetBDF(h ProcessorHandle) (BDF, error) {
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
