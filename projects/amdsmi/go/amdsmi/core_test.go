// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//go:build linux && cgo && amdsmi_mock

package amdsmi

import (
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"unsafe"
)

func TestCoreStatuses(t *testing.T) {
	mockReset()
	t.Cleanup(mockReset)
	codes := []StatusCode{AMDSMI_STATUS_MAP_ERROR, AMDSMI_STATUS_UNKNOWN_ERROR, StatusCode(0x12345678)}
	for code := AMDSMI_STATUS_SUCCESS; code <= AMDSMI_STATUS_CORRUPTED_EEPROM; code++ {
		codes = append(codes, code)
	}
	for _, code := range codes {
		t.Run(fmt.Sprintf("%08x", uint32(code)), func(t *testing.T) {
			mockConfigure("amdsmi_get_lib_version", code, 0)
			got, err := GetLibraryVersion()
			if code == AMDSMI_STATUS_SUCCESS {
				if err != nil || got.Major != compiledMajor {
					t.Fatalf("success: %+v, %v", got, err)
				}
				return
			}
			assertNativeError(t, err, "amdsmi_get_lib_version", code)
			assertZero(t, got)
			var native *Error
			if !errors.As(err, &native) {
				t.Fatal(err)
			}
			calls := mockCalls("amdsmi_status_code_to_string")
			want := fmt.Sprintf("AMD SMI status %d (0x%08x)", uint32(code), uint32(code))
			if code.Error() != want || !strings.Contains(err.Error(), want) ||
				!strings.Contains(err.Error(), "amdsmi_get_lib_version") || native.Message == "" {
				t.Fatalf("missing diagnostic: %v", err)
			}
			if mockCalls("amdsmi_status_code_to_string") != calls {
				t.Fatal("error formatting called native code")
			}
			if code == AMDSMI_STATUS_TIMEOUT || code == AMDSMI_STATUS_MORE_DATA || code == StatusCode(0x12345678) {
				if native.Message != want {
					t.Fatalf("failed lookup lost original status: %q", native.Message)
				}
			}
		})
	}
}

func TestCoreStatusesStrings(t *testing.T) {
	mockReset()
	t.Cleanup(mockReset)
	for _, code := range []StatusCode{AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NO_PERM} {
		want := "AMDSMI_STATUS_SUCCESS"
		if code == AMDSMI_STATUS_NO_PERM {
			want = "AMDSMI_STATUS_NO_PERM"
		}
		if got := StatusString(code); got != want {
			t.Fatalf("status %d: want %q, got %q", uint32(code), want, got)
		}
	}
	for _, code := range []StatusCode{AMDSMI_STATUS_TIMEOUT, AMDSMI_STATUS_MORE_DATA, StatusCode(0x12345678)} {
		if got := StatusString(code); got != code.Error() {
			t.Fatalf("failed lookup: %q", got)
		}
	}
	mockConfigure("amdsmi_status_code_to_string", AMDSMI_STATUS_IO, 0)
	if got := StatusString(AMDSMI_STATUS_NO_PERM); got != AMDSMI_STATUS_NO_PERM.Error() {
		t.Fatalf("injected lookup failure: %q", got)
	}
}

func TestCoreStatusesNullMessage(t *testing.T) {
	mockReset()
	t.Cleanup(mockReset)
	mockConfigure("amdsmi_status_code_to_string", AMDSMI_STATUS_SUCCESS, 1)
	if got := StatusString(AMDSMI_STATUS_NO_PERM); got != AMDSMI_STATUS_NO_PERM.Error() {
		t.Fatalf("null-message fallback: %q", got)
	}
	mockConfigure("amdsmi_get_lib_version", AMDSMI_STATUS_NO_PERM, 0)
	got, err := GetLibraryVersion()
	assertNativeError(t, err, "amdsmi_get_lib_version", AMDSMI_STATUS_NO_PERM)
	assertZero(t, got)
	var native *Error
	if !errors.As(err, &native) || native.Message != AMDSMI_STATUS_NO_PERM.Error() {
		t.Fatalf("null error message: %v", err)
	}
}

func TestCoreStatusesCheckedCount(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	for _, test := range []struct {
		name     string
		count    uint64
		capacity int
		fails    bool
	}{
		{"Empty", 0, 0, false},
		{"Within", 2, 3, false},
		{"Exact", 3, 3, false},
		{"MaxInt", uint64(maxInt), maxInt, false},
		{"OverCapacity", 4, 3, true},
		{"OverInt", uint64(maxInt) + 1, maxInt, true},
		{"MaxUint", ^uint64(0), maxInt, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := checkedCount("count", test.count, test.capacity)
			if test.fails {
				assertNativeError(t, err, "count", AMDSMI_STATUS_UNEXPECTED_SIZE)
				assertZero(t, got)
				want := fmt.Sprintf("count %d exceeds capacity %d", test.count, test.capacity)
				if !strings.Contains(err.Error(), want) {
					t.Fatalf("count diagnostic: %v", err)
				}
			} else if err != nil || uint64(got) != test.count {
				t.Fatalf("want %d, got %d, err=%v", test.count, got, err)
			}
		})
	}
}

func TestCoreStatusesBoundedString(t *testing.T) {
	t.Run("Nil", func(t *testing.T) {
		if got := boundedString(nil, nativeStringCapacity); got != "" {
			t.Fatalf("null string: %q", got)
		}
	})
	for _, test := range []struct {
		name     string
		value    string
		capacity int
		want     string
	}{
		{"NegativeCapacity", "abc", -1, ""},
		{"ZeroCapacity", "abc", 0, ""},
		{"Bounded", "abc", 2, "ab"},
		{"Terminated", "abc\x00tail", 8, "abc"},
		{"ShortLiteral", "abc", nativeStringCapacity, "abc"},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := mockBoundedString(test.value, test.capacity); got != test.want {
				t.Fatalf("want %q, got %q", test.want, got)
			}
		})
	}
	t.Run("NullVersionBuild", func(t *testing.T) {
		mockReset()
		t.Cleanup(mockReset)
		mockConfigure("amdsmi_get_lib_version", AMDSMI_STATUS_SUCCESS, 1)
		got, err := GetLibraryVersion()
		if err != nil || got.Build != "" || got.Major != compiledMajor {
			t.Fatalf("null build string: %+v, %v", got, err)
		}
	})
}

func TestCoreLifecycleInitFlags(t *testing.T) {
	resetCoreFixture(t)
	if uint64(InitAMDGPUs) != 2 {
		t.Fatalf("unexpected GPU flag: %d", InitAMDGPUs)
	}
	for held := uint32(0); held < 2; held++ {
		nativeState.mu.Lock()
		generation := nativeState.generation
		nativeState.mu.Unlock()
		calls, statusCalls := mockCalls("amdsmi_init"), mockCalls("amdsmi_status_code_to_string")
		for _, flags := range []InitFlags{0, 1, 3, 1 << 63, InitFlags(^uint64(0))} {
			assertNativeError(t, Init(flags), "amdsmi_init", AMDSMI_STATUS_INVAL)
			if mockCalls("amdsmi_init") != calls || mockNativeRefs() != held ||
				mockCalls("amdsmi_status_code_to_string") != statusCalls {
				t.Fatalf("invalid flags %d reached native code", flags)
			}
			nativeState.mu.Lock()
			refs, current := nativeState.refs, nativeState.generation
			nativeState.mu.Unlock()
			if refs != held || current != generation {
				t.Fatalf("invalid flags changed Go state: refs=%d generation=%d", refs, current)
			}
		}
		if err := Init(InitAMDGPUs); err != nil {
			t.Fatal(err)
		}
		if mockNativeRefs() != held+1 || mockCalls("amdsmi_init") != calls+1 {
			t.Fatal("valid flags did not acquire exactly one reference")
		}
	}
	for want := uint32(2); want > 0; want-- {
		if err := ShutDown(); err != nil {
			t.Fatal(err)
		}
		if mockNativeRefs() != want-1 {
			t.Fatal("valid flags left unbalanced references")
		}
	}
}

func TestCoreLifecycleReferences(t *testing.T) {
	resetCoreFixture(t)
	nativeState.mu.Lock()
	generation := nativeState.generation
	nativeState.mu.Unlock()
	for want := uint32(1); want <= 2; want++ {
		if err := Init(InitAMDGPUs); err != nil {
			t.Fatal(err)
		}
		if refs := mockNativeRefs(); refs != want {
			t.Fatalf("native Init references: want %d, got %d", want, refs)
		}
		nativeState.mu.Lock()
		refs, current := nativeState.refs, nativeState.generation
		nativeState.mu.Unlock()
		if refs != want || current != generation {
			t.Fatalf("Go Init state: refs=%d generation=%d", refs, current)
		}
	}
	for _, want := range []uint32{1, 0} {
		if err := ShutDown(); err != nil {
			t.Fatal(err)
		}
		if refs := mockNativeRefs(); refs != want {
			t.Fatalf("native shutdown references: want %d, got %d", want, refs)
		}
		if want == 0 {
			generation++
		}
		nativeState.mu.Lock()
		refs, current := nativeState.refs, nativeState.generation
		nativeState.mu.Unlock()
		if refs != want || current != generation {
			t.Fatalf("Go shutdown state: refs=%d generation=%d", refs, current)
		}
	}
	assertNativeError(t, ShutDown(), "amdsmi_shut_down", AMDSMI_STATUS_NOT_INIT)
	if mockCalls("amdsmi_init") != 2 || mockCalls("amdsmi_shut_down") != 2 {
		t.Fatal("unbalanced native calls")
	}
	nativeState.mu.Lock()
	current := nativeState.generation
	nativeState.mu.Unlock()
	if current != generation {
		t.Fatal("unmatched shutdown changed generation")
	}
}

func TestCoreLifecycleInitFailure(t *testing.T) {
	resetCoreFixture(t)
	nativeState.mu.Lock()
	generation := nativeState.generation
	nativeState.mu.Unlock()
	for held := uint32(0); held < 2; held++ {
		mockConfigure("amdsmi_init", AMDSMI_STATUS_INIT_ERROR, 0)
		assertNativeError(t, Init(InitAMDGPUs), "amdsmi_init", AMDSMI_STATUS_INIT_ERROR)
		if refs := mockNativeRefs(); refs != held {
			t.Fatalf("failed Init acquired native reference: %d", refs)
		}
		nativeState.mu.Lock()
		refs, current := nativeState.refs, nativeState.generation
		nativeState.mu.Unlock()
		if refs != held || current != generation {
			t.Fatalf("failed Init changed Go state: refs=%d generation=%d", refs, current)
		}
		mockConfigure("amdsmi_init", AMDSMI_STATUS_SUCCESS, 0)
		if err := Init(InitAMDGPUs); err != nil {
			t.Fatal(err)
		}
	}
	if refs := mockNativeRefs(); refs != 2 {
		t.Fatalf("recovery references: %d", refs)
	}
}

func TestCoreLifecycleOverflow(t *testing.T) {
	resetCoreFixture(t)
	if err := Init(InitAMDGPUs); err != nil {
		t.Fatal(err)
	}
	nativeState.mu.Lock()
	refs, generation := nativeState.refs, nativeState.generation
	nativeState.refs = 1<<31 - 1
	nativeState.mu.Unlock()
	t.Cleanup(func() {
		nativeState.mu.Lock()
		nativeState.refs = refs
		nativeState.mu.Unlock()
	})
	calls := mockCalls("amdsmi_init")
	err := Init(InitAMDGPUs)
	if err == nil {
		if cleanupErr := ShutDown(); cleanupErr != nil {
			t.Fatal(cleanupErr)
		}
	}
	assertNativeError(t, err, "amdsmi_init", AMDSMI_STATUS_REFCOUNT_OVERFLOW)
	if mockCalls("amdsmi_init") != calls || mockNativeRefs() != refs {
		t.Fatal("reference overflow reached native Init")
	}
	nativeState.mu.Lock()
	currentRefs, currentGeneration := nativeState.refs, nativeState.generation
	nativeState.mu.Unlock()
	if currentRefs != 1<<31-1 || currentGeneration != generation {
		t.Fatal("reference overflow changed Go state")
	}
}

func TestCoreLifecycleCleanupFailure(t *testing.T) {
	resetCoreFixture(t)
	if err := Init(InitAMDGPUs); err != nil {
		t.Fatal(err)
	}
	if err := Init(InitAMDGPUs); err != nil {
		t.Fatal(err)
	}
	nativeState.mu.Lock()
	generation := nativeState.generation
	nativeState.mu.Unlock()
	mockConfigure("amdsmi_shut_down", AMDSMI_STATUS_IO, 0)
	for _, want := range []uint32{1, 0} {
		assertNativeError(t, ShutDown(), "amdsmi_shut_down", AMDSMI_STATUS_IO)
		if refs := mockNativeRefs(); refs != want {
			t.Fatalf("failed cleanup native references: %d", refs)
		}
		if want == 0 {
			generation++
		}
		nativeState.mu.Lock()
		refs, current := nativeState.refs, nativeState.generation
		nativeState.mu.Unlock()
		if refs != want || current != generation {
			t.Fatalf("failed cleanup Go state: refs=%d generation=%d", refs, current)
		}
	}
	calls := mockCalls("amdsmi_shut_down")
	assertNativeError(t, ShutDown(), "amdsmi_shut_down", AMDSMI_STATUS_NOT_INIT)
	if mockCalls("amdsmi_shut_down") != calls {
		t.Fatal("cleanup error left a shutdown reference")
	}
	if err := Init(InitAMDGPUs); err != nil {
		t.Fatal(err)
	}
	if refs := mockNativeRefs(); refs != 1 {
		t.Fatalf("reinitialization references: %d", refs)
	}
}

func TestCoreLifecycleWithLibrary(t *testing.T) {
	resetCoreFixture(t)
	calls := 0
	query := func() (int, error) {
		calls++
		if nativeState.mu.TryLock() {
			nativeState.mu.Unlock()
			t.Fatal("library callback ran without the mutex")
		}
		return 7, nil
	}
	got, err := withLibrary("library", query)
	assertNativeError(t, err, "library", AMDSMI_STATUS_NOT_INIT)
	assertZero(t, got)
	if calls != 0 {
		t.Fatal("uninitialized library called the callback")
	}
	if err := Init(InitAMDGPUs); err != nil {
		t.Fatal(err)
	}
	got, err = withLibrary("library", query)
	if err != nil || got != 7 || calls != 1 {
		t.Fatalf("library callback: %d, %v, calls=%d", got, err, calls)
	}
	wantErr := errors.New("callback failed")
	got, err = withLibrary("library", func() (int, error) { return 0, wantErr })
	if !errors.Is(err, wantErr) {
		t.Fatalf("callback error changed: %v", err)
	}
	assertZero(t, got)
	if err := ShutDown(); err != nil {
		t.Fatal(err)
	}
	got, err = withLibrary("library", query)
	assertNativeError(t, err, "library", AMDSMI_STATUS_NOT_INIT)
	assertZero(t, got)
	if calls != 1 {
		t.Fatal("shut-down library called the callback")
	}
}

func TestCoreLifecycleWithProcessor(t *testing.T) {
	for _, code := range []StatusCode{AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_IO} {
		t.Run(fmt.Sprintf("Shutdown%d", uint32(code)), func(t *testing.T) {
			resetCoreFixture(t)
			calls := 0
			var want unsafe.Pointer
			query := func(p unsafe.Pointer) (int, error) {
				calls++
				if p != want {
					t.Fatal("callback received a different handle")
				}
				if nativeState.mu.TryLock() {
					nativeState.mu.Unlock()
					t.Fatal("processor callback ran without the mutex")
				}
				return 7, nil
			}
			got, err := mockWithProcessor(ProcessorHandle{}, "processor", query)
			assertNativeError(t, err, "processor", AMDSMI_STATUS_NOT_INIT)
			assertZero(t, got)
			if err := Init(InitAMDGPUs); err != nil {
				t.Fatal(err)
			}
			h := mockProcessorToken()
			want = h.ptr
			got, err = mockWithProcessor(ProcessorHandle{generation: h.generation}, "processor", query)
			assertNativeError(t, err, "processor", AMDSMI_STATUS_INVAL)
			assertZero(t, got)
			got, err = mockWithProcessor(h, "processor", query)
			if err != nil || got != 7 || calls != 1 {
				t.Fatalf("processor callback: %d, %v, calls=%d", got, err, calls)
			}
			mockConfigure("amdsmi_shut_down", code, 0)
			err = ShutDown()
			if code != AMDSMI_STATUS_SUCCESS {
				assertNativeError(t, err, "amdsmi_shut_down", code)
			} else if err != nil {
				t.Fatal(err)
			}
			got, err = mockWithProcessor(h, "processor", query)
			assertNativeError(t, err, "processor", AMDSMI_STATUS_NOT_INIT)
			assertZero(t, got)
			if err := Init(InitAMDGPUs); err != nil {
				t.Fatal(err)
			}
			got, err = mockWithProcessor(h, "processor", query)
			assertNativeError(t, err, "processor", AMDSMI_STATUS_INVAL)
			assertZero(t, got)
			current := mockProcessorToken()
			if current.ptr != h.ptr || current.generation == h.generation {
				t.Fatal("expected the same C address with a new generation")
			}
			got, err = mockWithProcessor(current, "processor", query)
			if err != nil || got != 7 || calls != 2 {
				t.Fatalf("renewed processor callback: %d, %v, calls=%d", got, err, calls)
			}
		})
	}
}

func TestCoreLifecycleSerialization(t *testing.T) {
	resetCoreFixture(t)
	if err := Init(InitAMDGPUs); err != nil {
		t.Fatal(err)
	}
	start := make(chan struct{})
	failures := make(chan error, 32)
	var workers sync.WaitGroup
	for worker := 0; worker < 32; worker++ {
		workers.Add(1)
		go func() {
			defer workers.Done()
			<-start
			for iteration := 0; iteration < 64; iteration++ {
				if err := Init(InitAMDGPUs); err != nil {
					failures <- err
					return
				}
				_, queryErr := GetLibraryVersion()
				shutdownErr := ShutDown()
				if queryErr != nil {
					failures <- queryErr
					return
				}
				if shutdownErr != nil {
					failures <- shutdownErr
					return
				}
				if got := StatusString(AMDSMI_STATUS_TIMEOUT); got != AMDSMI_STATUS_TIMEOUT.Error() {
					failures <- fmt.Errorf("concurrent status: %q", got)
					return
				}
			}
		}()
	}
	close(start)
	workers.Wait()
	close(failures)
	for err := range failures {
		t.Error(err)
	}
	if active := mockMaxActive(); active != 1 {
		t.Fatalf("overlapping native calls: %d", active)
	}
	nativeState.mu.Lock()
	refs := nativeState.refs
	nativeState.mu.Unlock()
	if refs != 1 || mockNativeRefs() != 1 {
		t.Fatalf("unbalanced references: Go=%d native=%d", refs, mockNativeRefs())
	}
}
