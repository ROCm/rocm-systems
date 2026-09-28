// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//go:build linux && cgo

package amdsmi

/*
#include <stddef.h>
#include <amd_smi/amdsmi.h>
*/
import "C"

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
