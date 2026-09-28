// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//go:build linux && cgo

package amdsmi

/*
#include <stddef.h>
#include <amd_smi/amdsmi.h>
*/
import "C"

import "unsafe"

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
	return withLibrary("amdsmi_get_socket_handles", func() ([]ProcessorHandle, error) {
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
	})
}
