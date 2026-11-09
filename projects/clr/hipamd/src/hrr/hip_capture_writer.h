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

bool open(const char* output_dir);

// True while events.bin is open: from open() until close(), or until capture stops
// for lack of space or at the size limit.
bool is_open();

// Caches the metadata JSON for manifest writers. The emergency finalizer only raw-writes this buffer; it never collects metadata.
void set_capture_metadata_json(const std::string& metadata_json);

// Flushes events.bin, appends the clean-shutdown trailer (hrr_eof_record), fsyncs and writes
// manifest.json with "complete": true. Call on normal shutdown.
void flush(const char* output_dir);

// Best-effort finalize for use from CLR's crash callback.
void emergency_finalize(bool clean_shutdown = false);

void close();

// hdr points to the hrr_event_header at the front of an hrr_args_* struct; payload_len is that struct's full size.
void write_event_raw(uint16_t api_id, hrr_event_header* hdr, uint32_t payload_len);

// After this call flush() omits the clean-shutdown trailer and writes "complete": false.
void mark_incomplete(const char* reason);

bool is_incomplete();

// Records that `api` is in the archive but replay cannot reproduce its effect, such as
// a host callback. The note is listed in the manifest under `reason`.
void note_unreplayable(const char* api, const char* reason);

// Writes a content-addressed blob. Returns its hash, or {} when the writer is not open,
// flush() has written the trailer, capture has stopped, or the blob was left out.
Hash128 write_blob(const void* data, size_t len, bool capped = true);

// Write a code object (.hsaco) blob. Returns hash, or {} as write_blob() does.
Hash128 write_code_object(const void* image, size_t image_size);

uint64_t event_count();

uint64_t blob_count();

}  // namespace writer
}  // namespace hrr_cap
