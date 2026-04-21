// htl_callback.hpp — single TracerCallback function entry point.
#pragma once

#include "htl_writer.hpp"
#include <atomic>

namespace htl {

// Set by the loader on init; read by the callback. Never freed during process
// lifetime — destructor only stops the writer thread.
extern Writer* g_writer;

// Tracks which domains/ops the loader wants captured.
extern std::atomic<bool> g_capture_hip_ops;
extern std::atomic<bool> g_capture_hip_api;

// CLR-facing callback.
extern "C" int htl_tracer_callback(uint32_t domain, uint32_t op, void* data);

}  // namespace htl
