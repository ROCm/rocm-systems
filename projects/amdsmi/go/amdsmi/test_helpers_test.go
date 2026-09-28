// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//go:build linux && cgo && amdsmi_mock

package amdsmi

import (
	"errors"
	"reflect"
	"testing"
)

func resetCoreFixture(t *testing.T) {
	t.Helper()
	mockReset()
	t.Cleanup(func() {
		mockConfigure("amdsmi_shut_down", AMDSMI_STATUS_SUCCESS, 0)
		for {
			nativeState.mu.Lock()
			refs := nativeState.refs
			nativeState.mu.Unlock()
			if refs == 0 {
				break
			}
			if err := ShutDown(); err != nil {
				t.Errorf("cleanup: %v", err)
			}
		}
		mockReset()
	})
}

func fixtureHandle(t *testing.T) ProcessorHandle {
	t.Helper()
	resetCoreFixture(t)
	if err := Init(); err != nil {
		t.Fatal(err)
	}
	handles, err := GetProcessorHandles()
	if err != nil {
		t.Fatal(err)
	}
	if len(handles) == 0 {
		t.Fatal("fixture did not return an AMD GPU")
	}
	return handles[0]
}

func assertNativeError(t *testing.T, err error, op string, code StatusCode) {
	t.Helper()
	var native *Error
	if !errors.Is(err, code) || !errors.As(err, &native) {
		t.Fatalf("want native code %d, got %v", code, err)
	}
	if native.Op != op || native.Code != code {
		t.Fatalf("want %s/%d, got %+v", op, code, native)
	}
}

func assertZero[T any](t *testing.T, got T) {
	t.Helper()
	var zero T
	if !reflect.DeepEqual(got, zero) {
		t.Fatalf("want zero result on error, got %#v", got)
	}
}

func checkQuery[T any](t *testing.T, op string,
	query func(ProcessorHandle) (T, error), want T) {
	t.Helper()
	h := fixtureHandle(t)
	got, err := query(h)
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("want %#v, got %#v, err=%v", want, got, err)
	}
	for _, code := range []StatusCode{AMDSMI_STATUS_NOT_SUPPORTED,
		AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_TIMEOUT, AMDSMI_STATUS_MORE_DATA,
		AMDSMI_STATUS_UNKNOWN_ERROR, StatusCode(0x12345678)} {
		mockConfigure(op, code, 0)
		got, err = query(h)
		assertNativeError(t, err, op, code)
		assertZero(t, got)
	}
	mockConfigure(op, AMDSMI_STATUS_SUCCESS, 0)
	calls := mockCalls(op)
	got, err = query(ProcessorHandle{})
	assertNativeError(t, err, op, AMDSMI_STATUS_INVAL)
	assertZero(t, got)
	if err := ShutDown(); err != nil {
		t.Fatal(err)
	}
	got, err = query(h)
	assertNativeError(t, err, op, AMDSMI_STATUS_NOT_INIT)
	assertZero(t, got)
	if err := Init(); err != nil {
		t.Fatal(err)
	}
	got, err = query(h)
	assertNativeError(t, err, op, AMDSMI_STATUS_INVAL)
	assertZero(t, got)
	if mockCalls(op) != calls {
		t.Fatal("a rejected handle reached the native query")
	}
}
