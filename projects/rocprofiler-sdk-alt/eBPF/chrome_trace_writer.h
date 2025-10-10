#ifndef CHROME_TRACE_WRITER_H
#define CHROME_TRACE_WRITER_H

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// Chrome tracing JSON format writer
// This creates a JSON file that can be loaded by Perfetto UI

typedef struct {
    FILE *file;
    int first_event;
} perfetto_writer_t;

// Create a new Perfetto writer
perfetto_writer_t* perfetto_writer_create(const char *filename);

// Destroy the Perfetto writer
void perfetto_writer_destroy(perfetto_writer_t *writer);

// Add a track for a process
int perfetto_writer_add_track(perfetto_writer_t *writer, uint32_t pid, uint32_t tid, const char *process_name);

// Add a slice begin event
int perfetto_writer_add_slice_begin(perfetto_writer_t *writer, uint64_t timestamp,
                                   const char *name, uint32_t pid, uint32_t tid);

// Add a slice end event
int perfetto_writer_add_slice_end(perfetto_writer_t *writer, uint64_t timestamp,
                                 const char *name, uint32_t pid, uint32_t tid);

// Add a slice event with arguments (metadata)
int perfetto_writer_add_slice_with_args(perfetto_writer_t *writer,
                                        uint64_t start_ts, uint64_t end_ts,
                                        const char *name, uint32_t pid, uint32_t tid,
                                        const char *args_json);

// Add a slice event with arguments and custom category
int perfetto_writer_add_slice_with_args_and_category(perfetto_writer_t *writer,
                                        uint64_t start_ts, uint64_t end_ts,
                                        const char *name, uint32_t pid, uint32_t tid,
                                        const char *args_json, const char *category);

// Add an instant event (for kernel dispatch events)
int perfetto_writer_add_instant_event(perfetto_writer_t *writer, uint64_t timestamp,
                                     const char *name, uint32_t pid, uint32_t tid);

// Add a flow event to connect two events (for API -> Kernel correlation)
// Use ph='s' for flow start and ph='f' for flow finish
int perfetto_writer_add_flow_event(perfetto_writer_t *writer, uint64_t timestamp,
                                   const char *name, uint32_t pid, uint32_t tid,
                                   uint64_t flow_id, const char *phase);

// Finalize the trace file
int perfetto_writer_finalize(perfetto_writer_t *writer);

#ifdef __cplusplus
}
#endif

#endif // CHROME_TRACE_WRITER_H
