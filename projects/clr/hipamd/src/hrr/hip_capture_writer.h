/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "hip_capture.h"
#include "hrr/hrr_api_args.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace hrr_cap {
namespace writer {

// Open events.bin for writing in output_dir. Creates directory tree.
bool open(const char* output_dir);

// True while events.bin is open: from a successful open() until close(), or
// until capture stops because the file system fell to the reserve or the archive
bool is_open();

// Cache process-level metadata JSON for manifest writers. The emergency
// finalizer only raw-writes this cached byte buffer; it never collects metadata.
void set_capture_metadata_json(const std::string& metadata_json);

// Flush events.bin, append the clean-shutdown trailer (hrr_eof_record), fsync,
// and write manifest.json with "complete": true. Call on normal shutdown: events,
void flush(const char* output_dir);

// Best-effort finalize for use from CLR's crash callback.
void emergency_finalize(bool clean_shutdown = false);

// Close and free resources.
void close();

// Write an event. hdr must point to the hrr_event_header at the front of an
// hrr_args_* struct. payload_len is sizeof the full hrr_args_* struct.
void write_event_raw(uint16_t api_id, hrr_event_header* hdr, uint32_t payload_len);

// Mark the capture as incomplete. After this is called, flush() will NOT append
// the clean-shutdown trailer and the manifest is written with "complete": false,
void mark_incomplete(const char* reason);

// Returns true if mark_incomplete() has been called.
bool is_incomplete();

// Record that `api` was captured but cannot be replayed faithfully — the call
// is in the archive, its effect is not reproducible (a host callback belonging
void note_unreplayable(const char* api, const char* reason);

// Write a buffer as a content-addressed blob. Returns hash, or {} when the
// writer is not open, flush() has written the trailer, or capture has stopped
Hash128 write_blob(const void* data, size_t len, bool capped = true);

// Write a code object (.hsaco) blob. Returns hash, or {} as write_blob() does.
Hash128 write_code_object(const void* image, size_t image_size);

// Number of events written so far.
uint64_t event_count();

// Number of blobs written so far.
uint64_t blob_count();

}  // namespace writer
}  // namespace hrr_cap
