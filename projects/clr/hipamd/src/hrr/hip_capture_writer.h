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

// Best-effort finalize for use from CLR's crash callback.
// Flushes the in-memory event buffer with raw write()+fsync only when no writer
// thread is mutating it, so no torn record is emitted. Never allocates and never
// uses stdio.
//
// clean_shutdown distinguishes normal shutdown from a crash callback:
//   - crash (false): writes manifest "complete": false and does NOT append the
//     trailer — its absence is how the reader detects a crash-truncated archive.
//   - clean (true): appends the clean-shutdown trailer (raw write of a fixed
//     hrr_eof_record) and writes manifest "complete": true. The trailer is only
//     written when the writer lock was free (no torn record in flight);
//     otherwise it degrades to the crash path.
void emergency_finalize(bool clean_shutdown = false);

// Close and free resources.
void close();

// Write an event. hdr must point to the hrr_event_header at the front of an
// hrr_args_* struct. payload_len is sizeof the full hrr_args_* struct.
// Fills all header fields (magic, version, event_type, sequence_id, timestamp_ns,
// thread_id, payload_length) then does a single fwrite of the whole struct.
// Thread-safe.
void write_event_raw(uint16_t api_id, hrr_event_header* hdr, uint32_t payload_len);

// Mark the capture as incomplete. After this is called, flush() will NOT append
// the clean-shutdown trailer and the manifest is written with "complete": false,
// so the reader/replayer cannot mistake the archive for a faithful, whole
// capture. Used when an event cannot be serialized losslessly (e.g. a kernel
// launch whose payload exceeds the wire-format limits) and is therefore dropped.
// Thread-safe; idempotent.
void mark_incomplete(const char* reason);

// Returns true if mark_incomplete() has been called.
bool is_incomplete();

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
