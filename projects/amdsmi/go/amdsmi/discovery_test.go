// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//go:build linux && cgo && amdsmi_mock

package amdsmi

import "testing"

func TestCoreTopologyNearest(t *testing.T) {
	checkQuery(t, "amdsmi_get_link_topology_nearest", func(handle ProcessorHandle) (uint32, error) {
		got, err := GetLinkTopologyNearest(handle, AMDSMI_LINK_TYPE_XGMI)
		if err != nil {
			assertZero(t, got)
			return 0, err
		}
		if len(got.ProcessorList) != 1 || got.ProcessorList[0] != handle {
			t.Fatalf("want current-generation handle %+v, got %#v", handle, got)
		}
		bdf, err := GetGpuDeviceBdf(got.ProcessorList[0])
		if err != nil || bdf != Bdf(0xabcde12345670000) {
			t.Fatalf("returned handle query: %v, err=%v", bdf, err)
		}
		return got.Count, nil
	}, uint32(1))
}

func TestCoreTopologyNearestNullResult(t *testing.T) {
	handle := fixtureHandle(t)
	mockConfigure("amdsmi_get_link_topology_nearest", AMDSMI_STATUS_SUCCESS, 1)
	got, err := GetLinkTopologyNearest(handle, AMDSMI_LINK_TYPE_XGMI)
	assertNativeError(t, err, "amdsmi_get_link_topology_nearest", AMDSMI_STATUS_UNEXPECTED_DATA)
	assertZero(t, got)
}

func TestCoreTopologyNearestNonGPUResult(t *testing.T) {
	resetCoreFixture(t)
	mockTopology(1, 2, 2)
	if err := Init(AMDSMI_INIT_AMD_GPUS); err != nil {
		t.Fatal(err)
	}
	handle, err := GetProcessorHandleFromIndex(0)
	if err != nil {
		t.Fatal(err)
	}
	mockConfigure("amdsmi_get_link_topology_nearest", AMDSMI_STATUS_SUCCESS, 2)
	got, err := GetLinkTopologyNearest(handle, AMDSMI_LINK_TYPE_XGMI)
	assertNativeError(t, err, "amdsmi_get_link_topology_nearest", AMDSMI_STATUS_NOT_SUPPORTED)
	assertZero(t, got)
}

func TestCoreTopologyNearestCounts(t *testing.T) {
	for _, test := range []struct {
		name  string
		mode  uint32
		count int
	}{
		{"Empty", 3, 0}, {"ExactCapacity", 4, 256},
		{"Oversized", 5, -1}, {"Maximum", 6, -1},
	} {
		t.Run(test.name, func(t *testing.T) {
			handle := fixtureHandle(t)
			calls := mockCalls("amdsmi_get_processor_type")
			mockConfigure("amdsmi_get_link_topology_nearest", AMDSMI_STATUS_SUCCESS, test.mode)
			got, err := GetLinkTopologyNearest(handle, AMDSMI_LINK_TYPE_XGMI)
			if test.count < 0 {
				assertNativeError(t, err, "amdsmi_get_link_topology_nearest", AMDSMI_STATUS_UNEXPECTED_SIZE)
				assertZero(t, got)
			} else {
				if err != nil || got.Count != uint32(test.count) || len(got.ProcessorList) != test.count {
					t.Fatalf("want %d handles, got %#v, err=%v", test.count, got, err)
				}
				for index, returned := range got.ProcessorList {
					if returned != handle {
						t.Fatalf("handle %d: want %+v, got %+v", index, handle, returned)
					}
				}
			}
			if test.count <= 0 && mockCalls("amdsmi_get_processor_type") != calls {
				t.Fatal("empty or oversized output reached processor-type validation")
			}
		})
	}
}

func TestCoreTopologyNearestLinkType(t *testing.T) {
	for _, test := range []struct {
		name     string
		mode     uint32
		linkType LinkType
	}{
		{"PCIe", 7, AMDSMI_LINK_TYPE_PCIE},
		{"Unknown", 8, LinkType(0x12345678)},
	} {
		t.Run(test.name, func(t *testing.T) {
			handle := fixtureHandle(t)
			mockConfigure("amdsmi_get_link_topology_nearest", AMDSMI_STATUS_SUCCESS, test.mode)
			got, err := GetLinkTopologyNearest(handle, test.linkType)
			if err != nil || got.Count != 1 || len(got.ProcessorList) != 1 || got.ProcessorList[0] != handle {
				t.Fatalf("link type %d: got %#v, err=%v", test.linkType, got, err)
			}
			if mockCalls("amdsmi_get_link_topology_nearest") != 1 {
				t.Fatal("requested link type did not reach the native query exactly once")
			}
		})
	}
}

func TestCoreTopologyNearestPartialFailure(t *testing.T) {
	handle := fixtureHandle(t)
	calls := mockCalls("amdsmi_get_processor_type")
	mockConfigure("amdsmi_get_link_topology_nearest", AMDSMI_STATUS_IO, 0)
	got, err := GetLinkTopologyNearest(handle, AMDSMI_LINK_TYPE_XGMI)
	assertNativeError(t, err, "amdsmi_get_link_topology_nearest", AMDSMI_STATUS_IO)
	assertZero(t, got)
	if mockCalls("amdsmi_get_processor_type") != calls {
		t.Fatal("partial output reached processor-type validation")
	}
}

func TestCoreTopologyNearestTypeFailure(t *testing.T) {
	resetCoreFixture(t)
	mockTopology(1, 2, 0)
	if err := Init(AMDSMI_INIT_AMD_GPUS); err != nil {
		t.Fatal(err)
	}
	handle, err := GetProcessorHandleFromIndex(0)
	if err != nil {
		t.Fatal(err)
	}
	calls := mockCalls("amdsmi_get_processor_type")
	mockConfigure("amdsmi_get_link_topology_nearest", AMDSMI_STATUS_SUCCESS, 2)
	mockConfigure("amdsmi_get_processor_type", AMDSMI_STATUS_SUCCESS, 1)
	got, err := GetLinkTopologyNearest(handle, AMDSMI_LINK_TYPE_XGMI)
	assertNativeError(t, err, "amdsmi_get_processor_type", AMDSMI_STATUS_IO)
	assertZero(t, got)
	if mockCalls("amdsmi_get_processor_type") != calls+2 {
		t.Fatal("expected the second returned handle's type query to fail")
	}
}

func TestCoreTopologyNearestCopiedList(t *testing.T) {
	handle := fixtureHandle(t)
	first, err := GetLinkTopologyNearest(handle, AMDSMI_LINK_TYPE_XGMI)
	if err != nil || len(first.ProcessorList) != 1 {
		t.Fatalf("first query: %#v, err=%v", first, err)
	}
	second, err := GetLinkTopologyNearest(handle, AMDSMI_LINK_TYPE_XGMI)
	if err != nil || len(second.ProcessorList) != 1 {
		t.Fatalf("second query: %#v, err=%v", second, err)
	}
	second.ProcessorList[0] = ProcessorHandle{}
	if first.ProcessorList[0] != handle {
		t.Fatal("returned topology lists share storage")
	}
	third, err := GetLinkTopologyNearest(handle, AMDSMI_LINK_TYPE_XGMI)
	if err != nil || len(third.ProcessorList) != 1 || third.ProcessorList[0] != handle {
		t.Fatalf("caller mutation changed native output: %#v, err=%v", third, err)
	}
}

func TestCoreTopologyNearestReturnedHandleLifecycle(t *testing.T) {
	handle := fixtureHandle(t)
	got, err := GetLinkTopologyNearest(handle, AMDSMI_LINK_TYPE_XGMI)
	if err != nil || len(got.ProcessorList) != 1 {
		t.Fatalf("topology query: %#v, err=%v", got, err)
	}
	returned := got.ProcessorList[0]
	calls := mockCalls("amdsmi_get_gpu_device_bdf")
	if err := ShutDown(); err != nil {
		t.Fatal(err)
	}
	bdf, err := GetGpuDeviceBdf(returned)
	assertNativeError(t, err, "amdsmi_get_gpu_device_bdf", AMDSMI_STATUS_NOT_INIT)
	assertZero(t, bdf)
	if err := Init(AMDSMI_INIT_AMD_GPUS); err != nil {
		t.Fatal(err)
	}
	bdf, err = GetGpuDeviceBdf(returned)
	assertNativeError(t, err, "amdsmi_get_gpu_device_bdf", AMDSMI_STATUS_INVAL)
	assertZero(t, bdf)
	if mockCalls("amdsmi_get_gpu_device_bdf") != calls {
		t.Fatal("stale topology handle reached the native query")
	}
	fresh, err := GetProcessorHandleFromIndex(0)
	if err != nil {
		t.Fatal(err)
	}
	got, err = GetLinkTopologyNearest(fresh, AMDSMI_LINK_TYPE_XGMI)
	if err != nil || len(got.ProcessorList) != 1 || got.ProcessorList[0] != fresh || fresh == returned {
		t.Fatalf("reinitialized topology: %#v, err=%v", got, err)
	}
	bdf, err = GetGpuDeviceBdf(got.ProcessorList[0])
	if err != nil || bdf != Bdf(0xabcde12345670000) {
		t.Fatalf("reinitialized handle query: %v, err=%v", bdf, err)
	}
}

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
			if err := Init(AMDSMI_INIT_AMD_GPUS); err != nil {
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
			if err := Init(AMDSMI_INIT_AMD_GPUS); err != nil {
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
			if err := Init(AMDSMI_INIT_AMD_GPUS); err != nil {
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
			if err := Init(AMDSMI_INIT_AMD_GPUS); err != nil {
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
		code     Status
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
			if err := Init(AMDSMI_INIT_AMD_GPUS); err != nil {
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
	if err := Init(AMDSMI_INIT_AMD_GPUS); err != nil {
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
		if err := Init(AMDSMI_INIT_AMD_GPUS); err != nil {
			t.Fatal(err)
		}
		mockConfigure("amdsmi_shut_down", AMDSMI_STATUS_IO, 0)
	})
	if mockNativeRefs() != 0 || mockCalls("amdsmi_init") != 0 || mockCalls("amdsmi_shut_down") != 0 {
		t.Fatal("fixture cleanup did not release references and reset controls")
	}
}

func TestCoreIndex(t *testing.T) {
	for _, test := range []struct {
		name                        string
		sockets, processors, stride uint32
		want                        []uint32
	}{
		{"Empty", 0, 0, 0, nil},
		{"EmptySockets", 2, 0, 0, nil},
		{"NoGPUs", 2, 3, 1, nil},
		{"Single", 1, 1, 0, []uint32{0}},
		{"MultiSocket", 2, 3, 0, []uint32{0, 1, 2, 3, 4, 5}},
		{"Filtered", 2, 3, 2, []uint32{0, 2, 4}},
	} {
		t.Run(test.name, func(t *testing.T) {
			resetCoreFixture(t)
			mockTopology(test.sockets, test.processors, test.stride)
			if err := Init(AMDSMI_INIT_AMD_GPUS); err != nil {
				t.Fatal(err)
			}
			handles, err := GetProcessorHandles()
			if err != nil || len(handles) != len(test.want) {
				t.Fatalf("discovery: %v, err=%v", handles, err)
			}
			for index, physical := range test.want {
				got, err := GetProcessorHandleFromIndex(uint32(index))
				if err != nil || got != handles[index] {
					t.Fatalf("index %d: want %+v, got %+v, err=%v", index, handles[index], got, err)
				}
				bdf, err := GetGpuDeviceBdf(got)
				want := Bdf(0xabcde12345670000 | uint64(physical/32)<<8 | uint64(physical%32))
				if err != nil || bdf != want {
					t.Fatalf("index %d: want BDF %v, got %v, err=%v", index, want, bdf, err)
				}
			}
			for _, index := range []uint32{uint32(len(test.want)), ^uint32(0)} {
				got, err := GetProcessorHandleFromIndex(index)
				assertNativeError(t, err, "GetProcessorHandleFromIndex", AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS)
				assertZero(t, got)
			}
		})
	}
}

func TestCoreIndexFailures(t *testing.T) {
	for _, test := range []struct {
		name, op string
		code     Status
		mode     uint32
		want     Status
		calls    uint64
	}{
		{"SocketCount", "amdsmi_get_socket_handles", AMDSMI_STATUS_NO_PERM, 0, AMDSMI_STATUS_NO_PERM, 1},
		{"SocketFill", "amdsmi_get_socket_handles", AMDSMI_STATUS_SUCCESS, 5, AMDSMI_STATUS_IO, 2},
		{"ProcessorCount", "amdsmi_get_processor_handles", AMDSMI_STATUS_MORE_DATA, 0, AMDSMI_STATUS_MORE_DATA, 1},
		{"ProcessorFill", "amdsmi_get_processor_handles", AMDSMI_STATUS_SUCCESS, 5, AMDSMI_STATUS_IO, 2},
		{"ProcessorType", "amdsmi_get_processor_type", AMDSMI_STATUS_NOT_SUPPORTED, 0, AMDSMI_STATUS_NOT_SUPPORTED, 1},
		{"SocketBounds", "amdsmi_get_socket_handles", AMDSMI_STATUS_SUCCESS, 1, AMDSMI_STATUS_UNEXPECTED_SIZE, 2},
		{"ProcessorBounds", "amdsmi_get_processor_handles", AMDSMI_STATUS_SUCCESS, 1, AMDSMI_STATUS_UNEXPECTED_SIZE, 2},
		{"NullSocket", "amdsmi_get_socket_handles", AMDSMI_STATUS_SUCCESS, 2, AMDSMI_STATUS_UNEXPECTED_DATA, 2},
		{"NullProcessor", "amdsmi_get_processor_handles", AMDSMI_STATUS_SUCCESS, 2, AMDSMI_STATUS_UNEXPECTED_DATA, 2},
	} {
		t.Run(test.name, func(t *testing.T) {
			resetCoreFixture(t)
			if err := Init(AMDSMI_INIT_AMD_GPUS); err != nil {
				t.Fatal(err)
			}
			mockConfigure(test.op, test.code, test.mode)
			got, err := GetProcessorHandleFromIndex(0)
			assertNativeError(t, err, test.op, test.want)
			assertZero(t, got)
			if mockCalls(test.op) != test.calls {
				t.Fatalf("native calls: want %d, got %d", test.calls, mockCalls(test.op))
			}
		})
	}
}

func TestCoreIndexLifetime(t *testing.T) {
	resetCoreFixture(t)
	for _, index := range []uint32{0, ^uint32(0)} {
		got, err := GetProcessorHandleFromIndex(index)
		assertNativeError(t, err, "amdsmi_get_socket_handles", AMDSMI_STATUS_NOT_INIT)
		assertZero(t, got)
	}
	if mockCalls("amdsmi_get_socket_handles") != 0 {
		t.Fatal("uninitialized index lookup reached native code")
	}
	if err := Init(AMDSMI_INIT_AMD_GPUS); err != nil {
		t.Fatal(err)
	}
	old, err := GetProcessorHandleFromIndex(0)
	if err != nil {
		t.Fatal(err)
	}
	if err := ShutDown(); err != nil {
		t.Fatal(err)
	}
	calls := mockCalls("amdsmi_get_socket_handles")
	got, err := GetProcessorHandleFromIndex(0)
	assertNativeError(t, err, "amdsmi_get_socket_handles", AMDSMI_STATUS_NOT_INIT)
	assertZero(t, got)
	if mockCalls("amdsmi_get_socket_handles") != calls {
		t.Fatal("shut-down index lookup reached native code")
	}
	if err := Init(AMDSMI_INIT_AMD_GPUS); err != nil {
		t.Fatal(err)
	}
	queryCalls := mockCalls("amdsmi_get_gpu_device_bdf")
	bdf, err := GetGpuDeviceBdf(old)
	assertNativeError(t, err, "amdsmi_get_gpu_device_bdf", AMDSMI_STATUS_INVAL)
	assertZero(t, bdf)
	if mockCalls("amdsmi_get_gpu_device_bdf") != queryCalls {
		t.Fatal("expired index handle reached native code")
	}
	current, err := GetProcessorHandleFromIndex(0)
	if err != nil || current.ptr != old.ptr || current.generation == old.generation {
		t.Fatalf("index did not return a renewed handle: %+v, err=%v", current, err)
	}
	if _, err := GetGpuDeviceBdf(current); err != nil {
		t.Fatal(err)
	}
}

func TestCoreBDF48Bit(t *testing.T) {
	h := fixtureHandle(t)
	bdf, err := GetGpuDeviceBdf(h)
	if err != nil {
		t.Fatal(err)
	}
	if bdf.Domain() != 0xabcde1234567 || bdf.Bus() != 0 || bdf.Device() != 0 || bdf.Function() != 0 ||
		bdf.String() != "abcde1234567:00:00.0" {
		t.Fatalf("BDF truncated or altered: %+v", bdf)
	}
	found, err := GetProcessorHandleFromBdf(bdf)
	if err != nil || found != h {
		t.Fatalf("BDF lookup mismatch: %+v, err=%v", found, err)
	}
}

func TestCoreBDFBounds(t *testing.T) {
	fixtureHandle(t)
	const op = "amdsmi_get_processor_handle_from_bdf"
	calls := mockCalls(op)
	for _, bdf := range []Bdf{0, 1 << 63, ^Bdf(0)} {
		got, err := GetProcessorHandleFromBdf(bdf)
		assertNativeError(t, err, op, AMDSMI_STATUS_NOT_FOUND)
		assertZero(t, got)
	}
	if mockCalls(op) != calls+3 {
		t.Fatal("representable BDF did not reach C")
	}
}

func TestCoreBDFLookupResults(t *testing.T) {
	for _, test := range []struct {
		mode uint32
		code Status
	}{
		{1, AMDSMI_STATUS_UNEXPECTED_DATA},
		{2, AMDSMI_STATUS_NOT_SUPPORTED},
	} {
		t.Run(test.code.Error(), func(t *testing.T) {
			resetCoreFixture(t)
			if test.mode == 2 {
				mockTopology(1, 1, 1)
			}
			if err := Init(AMDSMI_INIT_AMD_GPUS); err != nil {
				t.Fatal(err)
			}
			const op = "amdsmi_get_processor_handle_from_bdf"
			mockConfigure(op, AMDSMI_STATUS_SUCCESS, test.mode)
			got, err := GetProcessorHandleFromBdf(Bdf(0xabcde12345670000))
			assertNativeError(t, err, op, test.code)
			assertZero(t, got)
		})
	}
}

func TestCoreBDFQuery(t *testing.T) {
	checkQuery(t, "amdsmi_get_gpu_device_bdf", GetGpuDeviceBdf, Bdf(0xabcde12345670000))
}

func TestCoreBDFFormatting(t *testing.T) {
	bdf := Bdf(0xabff)
	if got := bdf.String(); got != "0000:ab:1f.7" {
		t.Fatalf("unexpected BDF: %s", got)
	}
}

func TestCorePackedBdf(t *testing.T) {
	for _, raw := range []uint64{0, 0xabcde1234567abff, ^uint64(0)} {
		bdf := Bdf(raw)
		packed := bdf.Domain()<<16 | uint64(bdf.Bus())<<8 | uint64(bdf.Device())<<3 | uint64(bdf.Function())
		if packed != raw {
			t.Fatalf("BDF roundtrip: %x != %x", packed, raw)
		}
	}
}
