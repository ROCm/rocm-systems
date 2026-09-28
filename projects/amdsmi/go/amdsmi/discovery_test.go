// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//go:build linux && cgo && amdsmi_mock

package amdsmi

import "testing"

func TestCoreDiscovery(t *testing.T) {
	for _, test := range []struct {
		name                        string
		sockets, processors, stride uint32
		want                        int
	}{
		{"Empty", 0, 0, 0, 0},
		{"EmptySockets", 2, 0, 0, 0},
		{"MultiSocket", 2, 40, 0, 80},
		{"Filter", 2, 40, 10, 72},
		{"NoGPUs", 2, 40, 1, 0},
		{"FixtureCapacity", 3, 64, 0, 192},
	} {
		t.Run(test.name, func(t *testing.T) {
			resetCoreFixture(t)
			mockTopology(test.sockets, test.processors, test.stride)
			if err := Init(); err != nil {
				t.Fatal(err)
			}
			got, err := GetProcessorHandles()
			if err != nil || got == nil || len(got) != test.want {
				t.Fatalf("want %d GPUs, got %v, err=%v", test.want, got, err)
			}
			if test.sockets == 0 && (mockCalls("amdsmi_get_socket_handles") != 1 ||
				mockCalls("amdsmi_get_processor_handles") != 0) {
				t.Fatal("empty discovery called a native fill")
			}
			seen := make(map[ProcessorHandle]bool)
			for _, h := range got {
				if h.ptr == nil || h.generation == 0 || seen[h] {
					t.Fatalf("null, unversioned, or duplicate processor handle: %+v", h)
				}
				seen[h] = true
			}
		})
	}
}

func TestCoreDiscoveryBounds(t *testing.T) {
	for _, op := range []string{"amdsmi_get_socket_handles", "amdsmi_get_processor_handles"} {
		t.Run(op, func(t *testing.T) {
			resetCoreFixture(t)
			if err := Init(); err != nil {
				t.Fatal(err)
			}
			mockConfigure(op, AMDSMI_STATUS_SUCCESS, 1)
			got, err := GetProcessorHandles()
			assertNativeError(t, err, op, AMDSMI_STATUS_UNEXPECTED_SIZE)
			assertZero(t, got)
			if mockCalls(op) != 2 || mockCalls("amdsmi_get_processor_type") != 0 {
				t.Fatal("invalid fill count was retried or used")
			}
		})
	}
}

func TestCoreDiscoveryShortFill(t *testing.T) {
	for _, test := range []struct {
		op   string
		want int
	}{
		{"amdsmi_get_socket_handles", 3},
		{"amdsmi_get_processor_handles", 4},
	} {
		t.Run(test.op, func(t *testing.T) {
			resetCoreFixture(t)
			mockTopology(2, 3, 0)
			if err := Init(); err != nil {
				t.Fatal(err)
			}
			mockConfigure(test.op, AMDSMI_STATUS_SUCCESS, 3)
			got, err := GetProcessorHandles()
			if err != nil || len(got) != test.want {
				t.Fatalf("want %d GPUs after shorter fill, got %d, err=%v", test.want, len(got), err)
			}
		})
	}
}

func TestCoreDiscoveryNullHandles(t *testing.T) {
	for _, test := range []struct {
		op, next string
	}{
		{"amdsmi_get_socket_handles", "amdsmi_get_processor_handles"},
		{"amdsmi_get_processor_handles", "amdsmi_get_processor_type"},
	} {
		t.Run(test.op, func(t *testing.T) {
			resetCoreFixture(t)
			if err := Init(); err != nil {
				t.Fatal(err)
			}
			mockConfigure(test.op, AMDSMI_STATUS_SUCCESS, 2)
			got, err := GetProcessorHandles()
			assertNativeError(t, err, test.op, AMDSMI_STATUS_UNEXPECTED_DATA)
			assertZero(t, got)
			if mockCalls(test.next) != 0 {
				t.Fatal("null handle reached the next native call")
			}
		})
	}
}

func TestCoreDiscoveryFailures(t *testing.T) {
	for _, test := range []struct {
		name, op string
		code     StatusCode
		mode     uint32
		calls    uint64
	}{
		{"SocketCount", "amdsmi_get_socket_handles", AMDSMI_STATUS_NO_PERM, 0, 1},
		{"SocketFill", "amdsmi_get_socket_handles", AMDSMI_STATUS_IO, 5, 2},
		{"ProcessorCount", "amdsmi_get_processor_handles", AMDSMI_STATUS_MORE_DATA, 0, 1},
		{"ProcessorFill", "amdsmi_get_processor_handles", AMDSMI_STATUS_IO, 5, 2},
		{"ProcessorType", "amdsmi_get_processor_type", AMDSMI_STATUS_NOT_SUPPORTED, 0, 1},
	} {
		t.Run(test.name, func(t *testing.T) {
			resetCoreFixture(t)
			if err := Init(); err != nil {
				t.Fatal(err)
			}
			code := test.code
			if test.mode != 0 {
				code = AMDSMI_STATUS_SUCCESS
			}
			mockConfigure(test.op, code, test.mode)
			got, err := GetProcessorHandles()
			assertNativeError(t, err, test.op, test.code)
			assertZero(t, got)
			if mockCalls(test.op) != test.calls {
				t.Fatalf("native calls: want %d, got %d", test.calls, mockCalls(test.op))
			}
		})
	}
}

func TestCoreDiscoveryEmptySocket(t *testing.T) {
	resetCoreFixture(t)
	mockTopology(2, 3, 0)
	if err := Init(); err != nil {
		t.Fatal(err)
	}
	mockConfigure("amdsmi_get_processor_handles", AMDSMI_STATUS_SUCCESS, 4)
	got, err := GetProcessorHandles()
	if err != nil || len(got) != 3 {
		t.Fatalf("want 3 GPUs from the populated socket, got %d, err=%v", len(got), err)
	}
	if mockCalls("amdsmi_get_processor_handles") != 3 || mockCalls("amdsmi_get_processor_type") != 3 {
		t.Fatal("empty socket was filled or populated socket was skipped")
	}
}

func TestCoreHandlesFixture(t *testing.T) {
	t.Run("Default", func(t *testing.T) {
		h := fixtureHandle(t)
		if h.ptr == nil || h.generation == 0 || mockNativeRefs() != 1 {
			t.Fatalf("invalid default fixture: %+v", h)
		}
		if err := Init(); err != nil {
			t.Fatal(err)
		}
		mockConfigure("amdsmi_shut_down", AMDSMI_STATUS_IO, 0)
	})
	if mockNativeRefs() != 0 || mockCalls("amdsmi_init") != 0 || mockCalls("amdsmi_shut_down") != 0 {
		t.Fatal("fixture cleanup did not release references and reset controls")
	}
}

func TestCoreBDF48Bit(t *testing.T) {
	h := fixtureHandle(t)
	bdf, err := GetBDF(h)
	if err != nil {
		t.Fatal(err)
	}
	if bdf.Domain != 0xabcde1234567 || bdf.Bus != 0 || bdf.Device != 0 || bdf.Function != 0 ||
		bdf.String() != "abcde1234567:00:00.0" {
		t.Fatalf("BDF truncated or altered: %+v", bdf)
	}
	found, err := GetProcessorHandleFromBDF(bdf)
	if err != nil || found != h {
		t.Fatalf("BDF lookup mismatch: %+v, err=%v", found, err)
	}
}

func TestCoreBDFBounds(t *testing.T) {
	fixtureHandle(t)
	const op = "amdsmi_get_processor_handle_from_bdf"
	calls := mockCalls(op)
	for _, bdf := range []BDF{{Domain: 1 << 48}, {Device: 32}, {Function: 8}} {
		got, err := GetProcessorHandleFromBDF(bdf)
		assertNativeError(t, err, op, AMDSMI_STATUS_INVAL)
		assertZero(t, got)
	}
	if mockCalls(op) != calls {
		t.Fatal("invalid BDF reached C")
	}
}

func TestCoreBDFLookupResults(t *testing.T) {
	for _, test := range []struct {
		mode uint32
		code StatusCode
	}{
		{1, AMDSMI_STATUS_UNEXPECTED_DATA},
		{2, AMDSMI_STATUS_NOT_SUPPORTED},
	} {
		t.Run(test.code.Error(), func(t *testing.T) {
			resetCoreFixture(t)
			if test.mode == 2 {
				mockTopology(1, 1, 1)
			}
			if err := Init(); err != nil {
				t.Fatal(err)
			}
			const op = "amdsmi_get_processor_handle_from_bdf"
			mockConfigure(op, AMDSMI_STATUS_SUCCESS, test.mode)
			got, err := GetProcessorHandleFromBDF(BDF{Domain: 0xabcde1234567})
			assertNativeError(t, err, op, test.code)
			assertZero(t, got)
		})
	}
}

func TestCoreBDFQuery(t *testing.T) {
	checkQuery(t, "amdsmi_get_gpu_device_bdf", GetBDF, BDF{Domain: 0xabcde1234567})
}

func TestCoreBDFFormatting(t *testing.T) {
	bdf := BDF{Bus: 0xab, Device: 0x1f, Function: 7}
	if got := bdf.String(); got != "0000:ab:1f.7" {
		t.Fatalf("unexpected BDF: %s", got)
	}
}
