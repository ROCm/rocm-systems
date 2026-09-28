// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//go:build linux && cgo && amdsmi_mock

package amdsmi

/*
#include <stdlib.h>
#include "testdata/mock.h"
static amdsmi_processor_handle go_amdsmi_processor_token(void) {
	static unsigned char token;
	return &token;
}
*/
import "C"

import "unsafe"

func mockReset() {
	nativeState.mu.Lock()
	defer nativeState.mu.Unlock()
	if nativeState.refs != 0 {
		panic("fixture reset with active Go references")
	}
	C.mock_reset()
}

func mockConfigure(op string, code StatusCode, mode uint32) {
	nativeState.mu.Lock()
	defer nativeState.mu.Unlock()
	name := C.CString(op)
	defer C.free(unsafe.Pointer(name))
	C.mock_configure(name, C.uint32_t(code), C.uint32_t(mode))
}

func mockTopology(sockets, processorsPerSocket, nonGPUStride uint32) {
	nativeState.mu.Lock()
	defer nativeState.mu.Unlock()
	C.mock_topology(C.uint32_t(sockets), C.uint32_t(processorsPerSocket), C.uint32_t(nonGPUStride))
}

func mockCalls(op string) uint64 {
	nativeState.mu.Lock()
	defer nativeState.mu.Unlock()
	name := C.CString(op)
	defer C.free(unsafe.Pointer(name))
	return uint64(C.mock_calls(name))
}

func mockMaxActive() uint32 {
	nativeState.mu.Lock()
	defer nativeState.mu.Unlock()
	return uint32(C.mock_max_active())
}

func mockNativeRefs() uint32 {
	nativeState.mu.Lock()
	defer nativeState.mu.Unlock()
	return uint32(C.mock_native_refs())
}

func mockBoundedString(value string, capacity int) string {
	nativeState.mu.Lock()
	defer nativeState.mu.Unlock()
	text := C.CString(value)
	defer C.free(unsafe.Pointer(text))
	return boundedString(text, capacity)
}

func mockProcessorToken() ProcessorHandle {
	nativeState.mu.Lock()
	defer nativeState.mu.Unlock()
	return ProcessorHandle{ptr: unsafe.Pointer(C.go_amdsmi_processor_token()),
		generation: nativeState.generation}
}

func mockWithProcessor(h ProcessorHandle, op string, fn func(unsafe.Pointer) (int, error)) (int, error) {
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (int, error) {
		return fn(unsafe.Pointer(p))
	})
}
