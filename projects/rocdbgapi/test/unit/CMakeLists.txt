## Copyright (c) 2026 Advanced Micro Devices, Inc.
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to
## deal in the Software without restriction, including without limitation the
## rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
## sell copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
## FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
## IN THE SOFTWARE.

# Unit tests for rocdbgapi.
#
# Tests added here must NOT depend on /dev/kfd, dxg/libdxg, or a real
# GPU.  Anything that touches the kernel driver belongs under
# ../feature/.

# Toolchain smoke test.  Confirms that GoogleTest is wired up on the
# host platform.  Uses the add_dbgapi_unit_test() helper which applies
# the standard labels ("dbgapi", "unit", OS label), C++17 settings,
# GTest/GMock linkage, and CTest working directory.
add_dbgapi_unit_test(dbgapi_smoke_test SOURCES smoke_test.cpp)

# Tests that reach library internals link against the amd-dbgapi-internal
# test seam (see cmake/AmdDbgapiInternal.cmake).  The seam exposes the
# library's source headers and is built without -fno-rtti / hidden
# visibility so test TUs can use RTTI and instantiate internal classes.

# OS-agnostic utilities (utils.h template helpers, scope guards,
# doubly-linked list, monotonic counter, flag operators) and the
# exception hierarchy.
add_dbgapi_unit_test(dbgapi_utils_test
  SOURCES utils_test.cpp
  LIBS amd-dbgapi-internal)

add_dbgapi_unit_test(dbgapi_exception_test
  SOURCES exception_test.cpp
  LIBS amd-dbgapi-internal)

# handle_object<>: the CRTP base + handle_object_set_t<> bookkeeping used
# by process/agent/queue/wave/...  Tested with a synthetic handle/object
# pair so the test doesn't drag in agent_t / queue_t or any architecture
# state.
add_dbgapi_unit_test(dbgapi_handle_object_test
  SOURCES handle_object_test.cpp
  LIBS amd-dbgapi-internal)

# logging.h to_string<> specializations + log_level global + vlog
# suppression.  Architecture/code_object/register/address_* handle
# to_string conversions are exercised only on the NONE sentinel here -
# the populated path goes through find() which needs process/architecture
# state that isn't set up in a pure unit-test TU.
add_dbgapi_unit_test(dbgapi_logging_test
  SOURCES logging_test.cpp
  LIBS amd-dbgapi-internal)

# architecture.{h,cpp}: the s_architecture_map registry is populated
# at static-init time, so the concrete gfx9/10/11/12 architectures are
# available from a unit-test TU without any amd_dbgapi_initialize()
# call or process state.  Tests here exercise the find() lookups, the
# regnum<->register_id bit packing, and a tiny smoke test of the
# per-architecture virtual register descriptors (PC on gfx942).
add_dbgapi_unit_test(dbgapi_architecture_test
  SOURCES architecture_test.cpp
  LIBS amd-dbgapi-internal)

# code_object.{h,cpp}: small value-object holding (process&, uri,
# load_address) + a mark counter.  We bind code_object_t's process_t&
# member to an aligned-storage buffer so the reference can be
# initialized, then carefully avoid any code path that would actually
# dereference it.  See the header comment in code_object_test.cpp for
# the full contract.  Tests that need a real process_t are deferred
# to the process-lifecycle commit (MockOsDriver).
add_dbgapi_unit_test(dbgapi_code_object_test
  SOURCES code_object_test.cpp
  LIBS amd-dbgapi-internal)

# register.{h,cpp}: amdgpu_regnum_t enum invariants + operators, and
# register_class_t bookkeeping (add/remove/contains/register_set).
# The register_class_t tests build instances against a real
# architecture from architecture_t::s_architecture_map (gfx942), which
# is populated at static-init time so no amd_dbgapi_initialize() is
# needed.  Wave-bound surfaces (read_register, is_register_available)
# and the anonymous-namespace DWARF mapping are deferred to feature
# tests.
add_dbgapi_unit_test(dbgapi_register_test
  SOURCES register_test.cpp
  LIBS amd-dbgapi-internal)

# MockOsDriver smoke test.  Exercises the Commit 11 friend-key seam
# (process_t::test_access_key_t) end-to-end by constructing a
# GMock-generated MockOsDriver (from support/mock_os_driver.h),
# injecting it into a process_t via the process_test_access trampoline
# (support/process_test_access.h), and asserting the seam wiring.
# Per-method MockOsDriver behavior is exercised in later
# process/agent/queue lifecycle tests.
add_dbgapi_unit_test(dbgapi_mock_os_driver_test
  SOURCES mock_os_driver_test.cpp
  LIBS amd-dbgapi-internal)

# process_t lifecycle surfaces that do not require agent_t / queue_t
# (flag bookkeeping, from_core(), record-only paths before runtime load,
# xfer_*_memory delegation + status translation).  Anything that walks
# range<agent_t> / range<queue_t> is deferred to Commits 9/14.
add_dbgapi_unit_test(dbgapi_process_lifecycle_test
  SOURCES process_lifecycle_test.cpp
  LIBS amd-dbgapi-internal)

# agent_t surfaces that do NOT require queue_t / wave_t / dbgapi callback
# allocator: accessors, supports_debugging, spi_ttmps_setup_enabled,
# watchpoint_share_kind, exception bookkeeping, mark, apertures, and POD
# get_info() queries.  Agents are built directly via
# process.create<agent_t>() using architecture_t::find("gfx942").
# insert_watchpoint / remove_watchpoint are deferred to Commit 9 where a
# watchpoint_t can also be constructed.
add_dbgapi_unit_test(dbgapi_agent_test
  SOURCES agent_test.cpp
  LIBS amd-dbgapi-internal)

# watchpoint.{h,cpp} + agent_t::insert_watchpoint / remove_watchpoint.
# Exercises the mask-bits algorithm in watchpoint_t's ctor (with and
# without a debugging-capable agent in the process), the three POD
# get_info() queries, and the agent-side bookkeeping that calls
# os_driver_t::set_address_watch / clear_address_watch through the
# MockOsDriver seam.  displaced_stepping_t is deferred until queue_t
# has a test seam.
add_dbgapi_unit_test(dbgapi_watchpoint_test
  SOURCES watchpoint_test.cpp
  LIBS amd-dbgapi-internal)

# event.{h,cpp}: the four event_t ctors (one per std::variant arm), the
# kind-discriminated get_info queries, and the cheap state transitions
# (created -> queued -> reported, plus the inert "processed" branches
# that don't walk queues or call send_exceptions).  Heavier processed
# paths (RUNTIME without runtime_enable_during_attach, and
# CODE_OBJECT_LIST_UPDATED with a real breakpoint_resume_event_id) are
# deferred until the queue_t test seam exists.
add_dbgapi_unit_test(dbgapi_event_test
  SOURCES event_test.cpp
  LIBS amd-dbgapi-internal)

# memory.{h,cpp}: address-type wrappers (base_address_t<T> arithmetic +
# global -> agent/host conversions), std::numeric_limits / std::hash
# specializations for the three address types, address_space_t::global()
# / ::host() singletons + non-NAME get_info queries, address_class_t
# value object, and memory_cache_t<agent_address_t> exercised end-to-end
# through a stub backing-store delegate (prefetch / read / write +
# write_back / discard).  Concrete address_space_t::lower / convert paths
# need a wave_t/agent_t and are deferred to feature tests; NAME get_info
# queries route through the client allocate_memory callback (unset in a
# unit-test TU) and are also deferred.
add_dbgapi_unit_test(dbgapi_memory_test
  SOURCES memory_test.cpp
  LIBS amd-dbgapi-internal)

# amd_dbgapi_read_memory() and amd_dbgapi_write_memory() C API entry
# points at the argument-validation layer: NOT_INITIALIZED before
# initialize, INVALID_ARGUMENT for null pointers, INVALID_PROCESS_ID /
# INVALID_WAVE_ID / INVALID_LANE_ID / INVALID_ADDRESS_SPACE_ID for bad
# handles, and PROCESS_FROZEN for writes to a frozen process.  Success
# paths delegate to process xfer_segment_memory for process-dependent
# address spaces (global/host) with wave_id == WAVE_NONE.  Wave-bound
# transfers and workgroup/lane-specific memory are deferred to feature
# tests.
add_dbgapi_unit_test(dbgapi_memory_io_test
  SOURCES memory_io_test.cpp
  LIBS amd-dbgapi-internal)

# Per-OS notifier_t implementation.  The factory notifier_t::create()
# lives in src/utils_linux.cpp on Linux and src/utils_windows.cpp on
# Windows; each implementation has different semantics (pipe vs Event)
# so the tests live in OS-specific files rather than being #ifdef'd
# inside a single TU.
# versioning.{h,cpp}: amd_dbgapi_get_version null-arg tolerance + macro
# round-trip, amd_dbgapi_get_build_name non-empty, and
# amd_dbgapi_get_status_string per-enum mapping driven by a table of
# every documented AMD_DBGAPI_STATUS_* value (regression net for new
# status codes added to the public header without a switch update).
add_dbgapi_unit_test(dbgapi_versioning_test
  SOURCES versioning_test.cpp
  LIBS amd-dbgapi-internal)

# callbacks.{h,cpp}: breakpoint_t ctor/dtor/get_info exercised through
# the global insert_breakpoint / remove_breakpoint client callbacks
# installed by amd_dbgapi_initialize().  Stub callbacks capture the
# library's calls and decide success/failure so both the m_inserted=true
# (dtor calls remove) and m_inserted=false (dtor skips) branches are
# covered.
add_dbgapi_unit_test(dbgapi_breakpoint_test
  SOURCES breakpoint_test.cpp
  LIBS amd-dbgapi-internal)

# displaced_stepping.cpp C API entry points at the argument-validation
# layer that does not need a real stopped wave: NOT_INITIALIZED before
# initialize, INVALID_ARGUMENT for null pointers, INVALID_WAVE_ID /
# INVALID_DISPLACED_STEPPING_ID for arbitrary handles.  The happy paths
# need a real process/queue/wave and live in feature tests.
add_dbgapi_unit_test(dbgapi_displaced_stepping_api_test
  SOURCES displaced_stepping_api_test.cpp
  LIBS amd-dbgapi-internal)

# displaced_stepping.{h,cpp} at the object level (no real wave): construction
# with std::nullopt destination, retain/release refcounting through the process
# handle set, get_info(PROCESS) round-trip, and get_info(unknown) throw.  Uses
# the test_compute_queue_t seam (support/test_queue.h) so set_state(suspended)
# doesn't run the heavy queue_state_changed callout.
add_dbgapi_unit_test(dbgapi_displaced_stepping_test
  SOURCES displaced_stepping_test.cpp
  LIBS amd-dbgapi-internal)

# aql_dispatch_t (private nested in queue.cpp's detail::aql_queue_t).
# Accessed via the test::make_aql_queue_and_dispatch_for_test factory
# (defined in queue.cpp, declared in support/aql_dispatch_test_access.h)
# which uses dual friend declarations on aql_queue_t (to name the private
# nested type) and aql_dispatch_t (to access its ctor).  MockOsDriver
# serves a crafted hsa_kernel_dispatch_packet_t for xfer_host_memory_partial
# at the queue ring base and a 64-byte kernel_descriptor blob for
# xfer_global_memory_partial at the packet's kernel_object.  Tests exercise
# every get_info query (backlinks, BARRIER/ACQUIRE/RELEASE fence bits in
# header, GRID_DIMENSIONS from setup, grid/workgroup sizes, segment sizes
# and addresses, kernel descriptor/entry/completion addresses, and the
# unknown-query INVALID_ARGUMENT throw).
add_dbgapi_unit_test(dbgapi_aql_dispatch_test
  SOURCES aql_dispatch_test.cpp
  LIBS amd-dbgapi-internal)

# workgroup.{h,cpp}: xfer_local_memory bounds + overflow regression for the
# fix in workgroup.cpp (offset near UINT64_MAX must not wrap when computing
# offset+size > limit).  Uses the test_compute_queue_t / test_dispatch_t
# seams (support/test_queue.h) to stand up a queue/dispatch chain without
# wave-save infra, and the workgroup_test_access friend trampoline
# (support/workgroup_test_access.h) to bypass xfer_segment_memory's
# local_address_space lower() that zero-extends offsets to 32 bits.
add_dbgapi_unit_test(dbgapi_workgroup_test
  SOURCES workgroup_test.cpp
  LIBS amd-dbgapi-internal)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_dbgapi_unit_test(dbgapi_utils_linux_test
    SOURCES utils_linux_test.cpp
    LIBS amd-dbgapi-internal)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  add_dbgapi_unit_test(dbgapi_utils_windows_test
    SOURCES utils_windows_test.cpp
    LIBS amd-dbgapi-internal)
endif()
