// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//go:build linux && cgo

package amdsmi

import "testing"

func TestNativeVersion(t *testing.T) {
	got, err := GetLibraryVersion()
	if err != nil {
		t.Fatal(err)
	}
	if got.Major != compiledMajor || got.Minor != compiledMinor || got.Release != compiledRelease {
		t.Fatalf("native/header version mismatch: %+v", got)
	}
	if got.Build == "" {
		t.Fatal("empty native build string")
	}
}
