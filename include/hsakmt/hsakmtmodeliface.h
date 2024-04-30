/*
 * Copyright © 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including
 * the next paragraph) shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef _HSAKMTMODELIFACE_H_
#define _HSAKMTMODELIFACE_H_

#include <inttypes.h>

// Changelog:
//  0.2: Add set_set_event function to hsakmt_model_functions
#define HSAKMT_MODEL_INTERFACE_VERSION_MAJOR 0
#define HSAKMT_MODEL_INTERFACE_VERSION_MINOR 4

typedef struct hsakmt_model hsakmt_model_t;
typedef struct hsakmt_model_queue hsakmt_model_queue_t;

// Description of a queue to be registered with the model.
//
// Addresses are relative to the global aperture.
struct hsakmt_model_queue_info {
	uint64_t ring_base_address;
	uint64_t write_pointer_address;
	uint64_t read_pointer_address;

	uint64_t *doorbell;

	uint32_t ring_size; // in bytes
	uint32_t queue_type;
};

// Pointer to a "set event" function.
//
// data is a user-provided opaque pointer.
// event_id is the ID of the event to set (as in amd_signal_s::event_id).
typedef void (*hsakmt_model_set_event_fn)(void *data, unsigned event_id);

// Interface provided by the software model implementation.
//
// Queried from a shared library by calling an export called
// `get_hsakmt_model_functions`
//
// Interface versioning follows the semantic versioning model: clients that
// know about interface version X.Y can use any implementation that provides
// version X.Z with Z >= Y.
//
// The model is designed to support only one VMID space.
struct hsakmt_model_functions {
	uint32_t version_major; // HSAKMT_MODEL_INTERFACE_VERSION_MAJOR
	uint32_t version_minor; // HSAKMT_MODEL_INTERFACE_VERSION_MINOR

	// Create a GPU device model.
	hsakmt_model_t *(*create)(void);

	// Destroy a GPU device model.
	void (*destroy)(hsakmt_model_t *model);

	// Set the global aperture. GPU virtual address 0 is at CPU address `base`.
	void (*set_global_aperture)(hsakmt_model_t *model, void *base, uint64_t size);
	void (*alloced_memory)(hsakmt_model_t *model, void *base, uint64_t size, uint32_t flags);
	void (*freed_memory)(hsakmt_model_t *model, void *base, uint64_t size);
	// Register a callback that the model should call when an event is signaled.
	// `data` is client data that is opaque to the model.
	//
	// TODO: Deprecated -- remove this!
	void (*set_notify_event)(hsakmt_model_t *model, void (*callback)(void *data), void *data);

	// Register a callback that the model should call in order to wait for an
	// event to be signaled.
	// `data` is client data that is opaque to the model.
	void (*set_wait_event)(hsakmt_model_t *model, void (*callback)(void *data, uint64_t address, uint64_t age), void *data);

	// Register a queue with the model. The model will immediately begin
	// asynchronous processing of the queue (but by default, the model need not
	// provide forward progress guarantees between multiple queues).
	hsakmt_model_queue_t *(*register_queue)(hsakmt_model_t *model, struct hsakmt_model_queue_info *info);

	// Register a callback that allows the model to set an event.
	void (*set_set_event)(hsakmt_model_t *model, hsakmt_model_set_event_fn fn, void *data);

	// Destroy a queue that was returned by register_queue.
	void (*destroy_queue)(hsakmt_model_t *model, hsakmt_model_queue_t *queue);
};

// Type of a shared library export called `get_hsakmt_model_functions`.
typedef const struct hsakmt_model_functions *(*get_hsakmt_model_functions_t)(void);

#endif // _HSAKMTMODELIFACE_H_