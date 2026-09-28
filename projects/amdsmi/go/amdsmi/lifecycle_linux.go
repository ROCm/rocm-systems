// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//go:build linux && cgo

package amdsmi

/*
#include <stddef.h>
#include <amd_smi/amdsmi.h>
*/
import "C"

import (
	"sync"
	"unsafe"
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

func Init() error {
	nativeState.mu.Lock()
	defer nativeState.mu.Unlock()
	if nativeState.refs == 1<<31-1 {
		return &Error{Op: "amdsmi_init", Code: AMDSMI_STATUS_REFCOUNT_OVERFLOW,
			Message: "initialization reference limit reached"}
	}
	status := C.amdsmi_init(C.uint64_t(C.AMDSMI_INIT_AMD_GPUS))
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
