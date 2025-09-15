/**
 * @file aql_cmd_buffer.c
 * @brief Command buffer management implementation
 *
 * This file implements the command buffer interface for accumulating PM4
 * commands. It provides memory management, bounds checking, and debug
 * support for command generation.
 */

#include "aql_cmd_buffer.h"

#ifdef __KERNEL__
#include <linux/kernel.h>
#include <linux/string.h>
#define AQL_PRINT(fmt, ...) printk(KERN_DEBUG "aql: " fmt, ##__VA_ARGS__)
#else
#include <stdio.h>
#include <string.h>
#define AQL_PRINT(fmt, ...) printf("aql: " fmt, ##__VA_ARGS__)
#endif

/*
 * Command Buffer Management Functions
 */

aql_result_t aql_cmd_buffer_init(aql_cmd_buffer_t* buf, uint32_t* data, size_t capacity) {
    if (!buf || !data || capacity == 0) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    buf->data = data;
    buf->capacity = capacity;
    buf->used = 0;
    buf->is_external = true;

    return AQL_SUCCESS;
}

aql_result_t aql_cmd_buffer_init_alloc(aql_cmd_buffer_t* buf, size_t capacity,
                                      aql_memory_alloc_cb_t alloc_cb, void* userdata) {
    void* data;
    aql_result_t result;

    if (!buf || !alloc_cb || capacity == 0) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    /* Allocate buffer using callback */
    result = alloc_cb(&data, capacity * sizeof(uint32_t),
                     AQL_MEM_FLAG_HOST_ACCESS | AQL_MEM_FLAG_DEVICE_ACCESS, userdata);
    if (result != AQL_SUCCESS) {
        return result;
    }

    buf->data = (uint32_t*)data;
    buf->capacity = capacity;
    buf->used = 0;
    buf->is_external = false;

    return AQL_SUCCESS;
}

void aql_cmd_buffer_clear(aql_cmd_buffer_t* buf) {
    if (buf) {
        buf->used = 0;
    }
}

void aql_cmd_buffer_cleanup(aql_cmd_buffer_t* buf,
                           aql_memory_dealloc_cb_t dealloc_cb, void* userdata) {
    if (!buf) return;

    if (!buf->is_external && buf->data && dealloc_cb) {
        dealloc_cb(buf->data, userdata);
    }

    buf->data = NULL;
    buf->capacity = 0;
    buf->used = 0;
    buf->is_external = true;
}

/*
 * Command Appending Functions
 */

aql_result_t aql_cmd_buffer_append_dwords(aql_cmd_buffer_t* buf,
                                         const uint32_t* data, size_t dword_count) {
    if (!buf || !data || dword_count == 0) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    if (!aql_cmd_buffer_is_valid(buf)) {
        return AQL_ERROR_INVALID_STATE;
    }

    if (aql_cmd_buffer_available(buf) < dword_count) {
        return AQL_ERROR_BUFFER_TOO_SMALL;
    }

    /* Copy data to buffer */
#ifdef __KERNEL__
    memcpy(&buf->data[buf->used], data, dword_count * sizeof(uint32_t));
#else
    memcpy(&buf->data[buf->used], data, dword_count * sizeof(uint32_t));
#endif

    buf->used += dword_count;
    return AQL_SUCCESS;
}

aql_result_t aql_cmd_buffer_reserve(aql_cmd_buffer_t* buf, size_t dword_count,
                                   uint32_t** reserved_ptr) {
    if (!buf || !reserved_ptr || dword_count == 0) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    if (!aql_cmd_buffer_is_valid(buf)) {
        return AQL_ERROR_INVALID_STATE;
    }

    if (aql_cmd_buffer_available(buf) < dword_count) {
        return AQL_ERROR_BUFFER_TOO_SMALL;
    }

    *reserved_ptr = &buf->data[buf->used];
    return AQL_SUCCESS;
}

aql_result_t aql_cmd_buffer_commit(aql_cmd_buffer_t* buf, size_t dword_count) {
    if (!buf) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    if (!aql_cmd_buffer_is_valid(buf)) {
        return AQL_ERROR_INVALID_STATE;
    }

    if (buf->used + dword_count > buf->capacity) {
        return AQL_ERROR_BUFFER_TOO_SMALL;
    }

    buf->used += dword_count;
    return AQL_SUCCESS;
}

aql_result_t aql_cmd_buffer_set_dword(aql_cmd_buffer_t* buf,
                                     size_t index, uint32_t value) {
    if (!buf) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    if (!aql_cmd_buffer_is_valid(buf)) {
        return AQL_ERROR_INVALID_STATE;
    }

    if (index >= buf->used) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    buf->data[index] = value;
    return AQL_SUCCESS;
}

/*
 * Buffer Alignment and Padding
 */

aql_result_t aql_cmd_buffer_align(aql_cmd_buffer_t* buf, size_t alignment) {
    size_t current_size, aligned_size, padding;

    if (!buf || alignment == 0 || (alignment & (alignment - 1)) != 0) {
        return AQL_ERROR_INVALID_ARGUMENT; /* alignment must be power of 2 */
    }

    if (!aql_cmd_buffer_is_valid(buf)) {
        return AQL_ERROR_INVALID_STATE;
    }

    current_size = buf->used;
    aligned_size = (current_size + alignment - 1) & ~(alignment - 1);
    padding = aligned_size - current_size;

    if (padding == 0) {
        return AQL_SUCCESS; /* Already aligned */
    }

    /* Add NOP padding */
    return aql_cmd_buffer_pad_to_size(buf, aligned_size);
}

aql_result_t aql_cmd_buffer_pad_to_size(aql_cmd_buffer_t* buf, size_t target_size) {
    size_t padding;
    uint32_t nop_header;

    if (!buf) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    if (!aql_cmd_buffer_is_valid(buf)) {
        return AQL_ERROR_INVALID_STATE;
    }

    if (target_size < buf->used) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    padding = target_size - buf->used;
    if (padding == 0) {
        return AQL_SUCCESS;
    }

    if (aql_cmd_buffer_available(buf) < padding) {
        return AQL_ERROR_BUFFER_TOO_SMALL;
    }

    /* Create NOP packet header */
    nop_header = 0xC0000000 | (((padding - 2) & 0x3FFF) << 16) | (0x10 & 0xFF); /* PACKET3_NOP */

    /* Add NOP packet */
    aql_cmd_buffer_append_dword(buf, nop_header);

    /* Pad remaining with zeros */
    for (size_t i = 1; i < padding; i++) {
        aql_cmd_buffer_append_dword(buf, 0);
    }

    return AQL_SUCCESS;
}

/*
 * Buffer Validation and Debug
 */

aql_result_t aql_cmd_buffer_validate(const aql_cmd_buffer_t* buf) {
    if (!buf) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    if (!buf->data) {
        return AQL_ERROR_INVALID_STATE;
    }

    if (buf->capacity == 0) {
        return AQL_ERROR_INVALID_STATE;
    }

    if (buf->used > buf->capacity) {
        return AQL_ERROR_INVALID_STATE;
    }

    /* Validate that used size contains complete packets */
    /* This is a simplified validation - could be enhanced to parse actual PM4 packets */
    if (buf->used > 0) {
        /* Check that we have at least one complete packet (minimum 1 dword header) */
        if (buf->used < 1) {
            return AQL_ERROR_INVALID_STATE;
        }
    }

    return AQL_SUCCESS;
}

aql_result_t aql_cmd_buffer_copy(const aql_cmd_buffer_t* src, aql_cmd_buffer_t* dst) {
    if (!src || !dst) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    if (!aql_cmd_buffer_is_valid(src) || !aql_cmd_buffer_is_valid(dst)) {
        return AQL_ERROR_INVALID_STATE;
    }

    if (dst->capacity < src->used) {
        return AQL_ERROR_BUFFER_TOO_SMALL;
    }

    /* Copy data */
#ifdef __KERNEL__
    memcpy(dst->data, src->data, src->used * sizeof(uint32_t));
#else
    memcpy(dst->data, src->data, src->used * sizeof(uint32_t));
#endif

    dst->used = src->used;
    return AQL_SUCCESS;
}

bool aql_cmd_buffer_equals(const aql_cmd_buffer_t* buf1, const aql_cmd_buffer_t* buf2) {
    if (!buf1 || !buf2) {
        return false;
    }

    if (!aql_cmd_buffer_is_valid(buf1) || !aql_cmd_buffer_is_valid(buf2)) {
        return false;
    }

    if (buf1->used != buf2->used) {
        return false;
    }

    if (buf1->used == 0) {
        return true; /* Both empty */
    }

    return memcmp(buf1->data, buf2->data, buf1->used * sizeof(uint32_t)) == 0;
}

/*
 * Debug and Tracing Support
 */

#ifdef AQL_DEBUG_TRACE

void aql_cmd_buffer_debug_print(const aql_cmd_buffer_t* buf, const char* prefix) {
    if (!buf || !prefix) return;

    AQL_PRINT("%s: buffer %p, capacity=%zu, used=%zu\n",
              prefix, (void*)buf, buf->capacity, buf->used);

    if (!aql_cmd_buffer_is_valid(buf)) {
        AQL_PRINT("%s: INVALID BUFFER STATE\n", prefix);
        return;
    }

    /* Print buffer contents in hex */
    for (size_t i = 0; i < buf->used; i += 8) {
        AQL_PRINT("%s: [%04zx]", prefix, i);
        for (size_t j = 0; j < 8 && (i + j) < buf->used; j++) {
            AQL_PRINT(" %08x", buf->data[i + j]);
        }
        AQL_PRINT("\n");
    }
}

void aql_cmd_buffer_debug_print_recent(const aql_cmd_buffer_t* buf,
                                      size_t count, const char* command_name) {
    size_t start_idx;

    if (!buf || !command_name || !aql_cmd_buffer_is_valid(buf)) return;

    if (count > buf->used) {
        start_idx = 0;
        count = buf->used;
    } else {
        start_idx = buf->used - count;
    }

    AQL_PRINT("CMD '%s' (%zu dwords):", command_name, count);
    for (size_t i = 0; i < count; i++) {
        AQL_PRINT(" %08x", buf->data[start_idx + i]);
    }
    AQL_PRINT("\n");
}

aql_result_t aql_cmd_buffer_debug_validate_packet(const aql_cmd_buffer_t* buf,
                                                 size_t packet_start,
                                                 const char* packet_name) {
    uint32_t header, packet_type, packet_size;

    if (!buf || !packet_name || !aql_cmd_buffer_is_valid(buf)) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    if (packet_start >= buf->used) {
        AQL_PRINT("PACKET '%s': Invalid start index %zu (buffer size %zu)\n",
                  packet_name, packet_start, buf->used);
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    header = buf->data[packet_start];
    packet_type = (header >> 30) & 0x3;

    if (packet_type == 3) { /* Type 3 packet */
        packet_size = ((header >> 16) & 0x3FFF) + 2; /* +2 for header + count encoding */

        if (packet_start + packet_size > buf->used) {
            AQL_PRINT("PACKET '%s': Incomplete packet (need %u dwords, have %zu)\n",
                      packet_name, packet_size, buf->used - packet_start);
            return AQL_ERROR_INVALID_STATE;
        }

        AQL_PRINT("PACKET '%s': Type 3, opcode=0x%02x, size=%u dwords\n",
                  packet_name, header & 0xFF, packet_size);
    } else {
        AQL_PRINT("PACKET '%s': Non-Type3 packet (type=%u)\n", packet_name, packet_type);
    }

    return AQL_SUCCESS;
}

#endif /* AQL_DEBUG_TRACE */