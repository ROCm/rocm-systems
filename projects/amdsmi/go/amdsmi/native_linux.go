// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//go:build linux && cgo

package amdsmi

/*
#cgo LDFLAGS: -lamd_smi
#include <stddef.h>
#include <amd_smi/amdsmi.h>
static size_t go_amdsmi_string_length(const char *p, size_t capacity) {
    size_t n = 0;
    if (p == NULL) return 0;
    while (n < capacity && p[n] != '\0') ++n;
    return n;
}
*/
import "C"

const nativeStringCapacity int = C.AMDSMI_MAX_STRING_LENGTH

func boundedString(p *C.char, capacity int) string {
	if p == nil || capacity <= 0 {
		return ""
	}
	n := C.go_amdsmi_string_length(p, C.size_t(capacity))
	return C.GoStringN(p, C.int(n))
}
