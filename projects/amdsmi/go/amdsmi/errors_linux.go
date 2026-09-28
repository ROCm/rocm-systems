// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//go:build linux && cgo

package amdsmi

/*
#include <stddef.h>
#include <amd_smi/amdsmi.h>
*/
import "C"

import "fmt"

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
