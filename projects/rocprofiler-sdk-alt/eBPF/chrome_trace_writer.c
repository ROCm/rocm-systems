#include "chrome_trace_writer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

// Chrome tracing JSON format writer
// This creates a JSON file that can be loaded by Perfetto UI

typedef struct {
    FILE *file;
    int first_event;
} chrome_trace_writer_t;

static void escape_json_string(FILE *file, const char *str) {
    fputc('"', file);
    while (*str) {
        switch (*str) {
            case '"': fputs("\\\"", file); break;
            case '\\': fputs("\\\\", file); break;
            case '\b': fputs("\\b", file); break;
            case '\f': fputs("\\f", file); break;
            case '\n': fputs("\\n", file); break;
            case '\r': fputs("\\r", file); break;
            case '\t': fputs("\\t", file); break;
            default: fputc(*str, file); break;
        }
        str++;
    }
    fputc('"', file);
}

perfetto_writer_t* perfetto_writer_create(const char *filename) {
    chrome_trace_writer_t *writer = malloc(sizeof(chrome_trace_writer_t));
    if (!writer) return NULL;

    writer->file = fopen(filename, "w");
    if (!writer->file) {
        free(writer);
        return NULL;
    }

    writer->first_event = 1;

    // Write Chrome tracing JSON header
    fprintf(writer->file, "{\n  \"traceEvents\": [\n");
    fflush(writer->file);

    return (perfetto_writer_t*)writer;
}

void perfetto_writer_destroy(perfetto_writer_t *writer) {
    if (writer) {
        chrome_trace_writer_t *chrome_writer = (chrome_trace_writer_t*)writer;
        if (chrome_writer->file) {
            // Close the JSON array and object
            fprintf(chrome_writer->file, "\n  ],\n  \"displayTimeUnit\": \"ns\"\n}\n");
            fclose(chrome_writer->file);
        }
        free(writer);
    }
}

int perfetto_writer_add_track(perfetto_writer_t *writer, uint32_t pid, uint32_t tid, const char *process_name) {
    if (!writer) return -1;

    chrome_trace_writer_t *chrome_writer = (chrome_trace_writer_t*)writer;

    // Add process metadata event
    if (!chrome_writer->first_event) {
        fprintf(chrome_writer->file, ",\n");
    }

    fprintf(chrome_writer->file, "    {\n");
    fprintf(chrome_writer->file, "      \"name\": \"process_name\",\n");
    fprintf(chrome_writer->file, "      \"ph\": \"M\",\n");
    fprintf(chrome_writer->file, "      \"pid\": %u,\n", pid);
    fprintf(chrome_writer->file, "      \"tid\": %u,\n", tid);
    fprintf(chrome_writer->file, "      \"args\": {\n");
    fprintf(chrome_writer->file, "        \"name\": ");
    escape_json_string(chrome_writer->file, process_name);
    fprintf(chrome_writer->file, "\n      }\n");
    fprintf(chrome_writer->file, "    }");

    chrome_writer->first_event = 0;
    fflush(chrome_writer->file);

    return 0;
}

int perfetto_writer_add_slice_begin(perfetto_writer_t *writer, uint64_t timestamp, const char *name, uint32_t pid, uint32_t tid) {
    if (!writer) return -1;

    chrome_trace_writer_t *chrome_writer = (chrome_trace_writer_t*)writer;

    if (!chrome_writer->first_event) {
        fprintf(chrome_writer->file, ",\n");
    }

    fprintf(chrome_writer->file, "    {\n");
    fprintf(chrome_writer->file, "      \"name\": ");
    escape_json_string(chrome_writer->file, name);
    fprintf(chrome_writer->file, ",\n");
    fprintf(chrome_writer->file, "      \"cat\": \"HIP\",\n");
    fprintf(chrome_writer->file, "      \"ph\": \"B\",\n");
    fprintf(chrome_writer->file, "      \"ts\": %llu,\n", (unsigned long long)timestamp);
    fprintf(chrome_writer->file, "      \"pid\": %u,\n", pid);
    fprintf(chrome_writer->file, "      \"tid\": %u\n", tid);
    fprintf(chrome_writer->file, "    }");

    chrome_writer->first_event = 0;
    fflush(chrome_writer->file);

    return 0;
}

int perfetto_writer_add_slice_end(perfetto_writer_t *writer, uint64_t timestamp, const char *name, uint32_t pid, uint32_t tid) {
    if (!writer) return -1;

    chrome_trace_writer_t *chrome_writer = (chrome_trace_writer_t*)writer;

    if (!chrome_writer->first_event) {
        fprintf(chrome_writer->file, ",\n");
    }

    fprintf(chrome_writer->file, "    {\n");
    fprintf(chrome_writer->file, "      \"name\": ");
    escape_json_string(chrome_writer->file, name);
    fprintf(chrome_writer->file, ",\n");
    fprintf(chrome_writer->file, "      \"cat\": \"HIP\",\n");
    fprintf(chrome_writer->file, "      \"ph\": \"E\",\n");
    fprintf(chrome_writer->file, "      \"ts\": %llu,\n", (unsigned long long)timestamp);
    fprintf(chrome_writer->file, "      \"pid\": %u,\n", pid);
    fprintf(chrome_writer->file, "      \"tid\": %u\n", tid);
    fprintf(chrome_writer->file, "    }");

    chrome_writer->first_event = 0;
    fflush(chrome_writer->file);

    return 0;
}

int perfetto_writer_add_slice_with_args(perfetto_writer_t *writer,
                                        uint64_t start_ts, uint64_t end_ts,
                                        const char *name, uint32_t pid, uint32_t tid,
                                        const char *args_json) {
    if (!writer) return -1;

    chrome_trace_writer_t *chrome_writer = (chrome_trace_writer_t*)writer;

    /* Write complete event with duration and args */
    if (!chrome_writer->first_event) {
        fprintf(chrome_writer->file, ",\n");
    }

    fprintf(chrome_writer->file, "    {\n");
    fprintf(chrome_writer->file, "      \"name\": ");
    escape_json_string(chrome_writer->file, name);
    fprintf(chrome_writer->file, ",\n");
    fprintf(chrome_writer->file, "      \"cat\": \"HIP\",\n");
    fprintf(chrome_writer->file, "      \"ph\": \"X\",\n");  // Complete event
    fprintf(chrome_writer->file, "      \"ts\": %llu,\n", (unsigned long long)start_ts);
    fprintf(chrome_writer->file, "      \"dur\": %llu,\n", (unsigned long long)(end_ts - start_ts));
    fprintf(chrome_writer->file, "      \"pid\": %u,\n", pid);
    fprintf(chrome_writer->file, "      \"tid\": %u", tid);

    /* Add args if provided */
    if (args_json && args_json[0] != '\0') {
        fprintf(chrome_writer->file, ",\n      \"args\": %s\n", args_json);
    } else {
        fprintf(chrome_writer->file, "\n");
    }

    fprintf(chrome_writer->file, "    }");

    chrome_writer->first_event = 0;
    fflush(chrome_writer->file);

    return 0;
}

int perfetto_writer_add_instant_event(perfetto_writer_t *writer, uint64_t timestamp, const char *name, uint32_t pid, uint32_t tid) {
    if (!writer) return -1;

    chrome_trace_writer_t *chrome_writer = (chrome_trace_writer_t*)writer;

    if (!chrome_writer->first_event) {
        fprintf(chrome_writer->file, ",\n");
    }

    fprintf(chrome_writer->file, "    {\n");
    fprintf(chrome_writer->file, "      \"name\": ");
    escape_json_string(chrome_writer->file, name);
    fprintf(chrome_writer->file, ",\n");
    fprintf(chrome_writer->file, "      \"cat\": \"Kernel\",\n");
    fprintf(chrome_writer->file, "      \"ph\": \"i\",\n");
    fprintf(chrome_writer->file, "      \"ts\": %llu,\n", (unsigned long long)timestamp);
    fprintf(chrome_writer->file, "      \"pid\": %u,\n", pid);
    fprintf(chrome_writer->file, "      \"tid\": %u,\n", tid);
    fprintf(chrome_writer->file, "      \"s\": \"g\"\n");  // Global scope
    fprintf(chrome_writer->file, "    }");

    chrome_writer->first_event = 0;
    fflush(chrome_writer->file);
    return 0;
}

int perfetto_writer_finalize(perfetto_writer_t *writer) {
    if (!writer) return -1;

    chrome_trace_writer_t *chrome_writer = (chrome_trace_writer_t*)writer;

    if (chrome_writer->file) {
        fflush(chrome_writer->file);
    }

    return 0;
}
