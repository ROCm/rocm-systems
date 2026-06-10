/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "hip_capture.h"
#include "hrr_api_args.h"

#include <cstddef>
#include <cstdint>

namespace hrr_cap {
namespace writer {

// Open events.bin for writing in output_dir. Creates directory tree.
// Must be called before any write_* functions.
bool open(const char* output_dir);

// Returns true if open() has been called successfully.
bool is_open();

// Flush events.bin, append the clean-shutdown trailer (hrr_eof_record), fsync,
// and write manifest.json with "complete": true. Safe to call multiple times
// (the trailer is written only once). Call on normal shutdown.
void flush(const char* output_dir);

// Force any buffered events to disk and fsync. Bounds how much capture data a
// crash can lose. Cheap relative to capture; called periodically by the writer
// itself and may be called by the host. Thread-safe.
void checkpoint();

// Best-effort, async-signal-safe finalize for use from a fatal-signal handler.
// Flushes the in-memory event buffer with raw write()+fsync (only when the
// writer lock can be taken without blocking, so no torn record is emitted) and
// writes a minimal manifest.json with "complete": false using raw open/write.
// It deliberately does NOT append the clean trailer — the trailer's absence is
// how the reader detects a crash-truncated archive. Never allocates, never uses
// stdio. Safe to call from SIGSEGV/SIGABRT/etc.
void emergency_finalize();

// Close and free resources.
void close();

// Write an event. hdr must point to the hrr_event_header at the front of an
// hrr_args_* struct. payload_len is sizeof the full hrr_args_* struct.
// Fills all header fields (magic, version, event_type, sequence_id, timestamp_ns,
// thread_id, payload_length) then does a single fwrite of the whole struct.
// Thread-safe.
void write_event_raw(uint16_t api_id, hrr_event_header* hdr, uint16_t payload_len);

// Write a buffer as a content-addressed blob. Returns hash.
// Thread-safe. Skips write if blob already exists on disk.
Hash128 write_blob(const void* data, size_t len);

// Write a code object (.hsaco) blob. Returns hash.
Hash128 write_code_object(const void* image, size_t image_size);

// Number of events written so far.
uint64_t event_count();

// Number of blobs written so far.
uint64_t blob_count();

}  // namespace writer
}  // namespace hrr_cap
