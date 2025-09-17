/**
 * @file aql_cmd_buffer.h
 * @brief Command buffer management for PM4 command generation
 *
 * This file provides the command buffer interface that replaces the C++
 * CmdBuffer class. It manages accumulation of PM4 commands in a buffer
 * and provides utilities for buffer management, validation, and debugging.
 *
 * Design principles:
 * - Support both static and dynamic buffer allocation
 * - Provide bounds checking for kernel safety
 * - Enable debug tracing of command generation
 * - Support buffer reuse and cleanup
 */

#ifndef AQL_CMD_BUFFER_H
#define AQL_CMD_BUFFER_H

#include "aql_types.h"

/*
 * Command Buffer Management Functions
 */

/**
 * @brief Initialize command buffer with external storage
 * @param buf Command buffer to initialize
 * @param data Pre-allocated buffer storage
 * @param capacity Buffer capacity in dwords
 * @return AQL_SUCCESS or error code
 */
aql_result_t aql_cmd_buffer_init(aql_cmd_buffer_t* buf, uint32_t* data, size_t capacity);

/**
 * @brief Initialize command buffer with internal allocation
 * @param buf Command buffer to initialize
 * @param capacity Desired capacity in dwords
 * @param alloc_cb Memory allocation callback
 * @param userdata User data for allocation callback
 * @return AQL_SUCCESS or error code
 */
aql_result_t aql_cmd_buffer_init_alloc(aql_cmd_buffer_t* buf, size_t capacity,
                                      aql_memory_alloc_cb_t alloc_cb, void* userdata);

/**
 * @brief Clear command buffer contents
 * @param buf Command buffer to clear
 */
void aql_cmd_buffer_clear(aql_cmd_buffer_t* buf);

/**
 * @brief Cleanup command buffer and free resources
 * @param buf Command buffer to cleanup
 * @param dealloc_cb Memory deallocation callback (if allocated)
 * @param userdata User data for deallocation callback
 */
void aql_cmd_buffer_cleanup(aql_cmd_buffer_t* buf,
                           aql_memory_dealloc_cb_t dealloc_cb, void* userdata);

/**
 * @brief Get available space in command buffer
 * @param buf Command buffer
 * @return Available space in dwords
 */
static inline size_t aql_cmd_buffer_available(const aql_cmd_buffer_t* buf) {
    return buf ? (buf->capacity - buf->used) : 0;
}

/**
 * @brief Get used space in command buffer
 * @param buf Command buffer
 * @return Used space in dwords
 */
static inline size_t aql_cmd_buffer_used(const aql_cmd_buffer_t* buf) {
    return buf ? buf->used : 0;
}

/**
 * @brief Get total capacity of command buffer
 * @param buf Command buffer
 * @return Total capacity in dwords
 */
static inline size_t aql_cmd_buffer_capacity(const aql_cmd_buffer_t* buf) {
    return buf ? buf->capacity : 0;
}

/**
 * @brief Get used size in bytes
 * @param buf Command buffer
 * @return Used size in bytes
 */
static inline size_t aql_cmd_buffer_size_bytes(const aql_cmd_buffer_t* buf) {
    return buf ? (buf->used * sizeof(uint32_t)) : 0;
}

/**
 * @brief Get pointer to buffer data
 * @param buf Command buffer
 * @return Pointer to buffer data, or NULL if invalid
 */
static inline const void* aql_cmd_buffer_data(const aql_cmd_buffer_t* buf) {
    return (buf && buf->data) ? buf->data : NULL;
}

/**
 * @brief Check if buffer is valid and usable
 * @param buf Command buffer to check
 * @return True if buffer is valid
 */
static inline bool aql_cmd_buffer_is_valid(const aql_cmd_buffer_t* buf) {
    return buf && buf->data && buf->capacity > 0 && buf->used <= buf->capacity;
}

/*
 * Command Appending Functions
 */

/**
 * @brief Append raw dwords to command buffer
 * @param buf Command buffer
 * @param data Pointer to data to append
 * @param dword_count Number of dwords to append
 * @return AQL_SUCCESS or error code
 */
aql_result_t aql_cmd_buffer_append_dwords(aql_cmd_buffer_t* buf,
                                         const uint32_t* data, size_t dword_count);

/**
 * @brief Append a single dword to command buffer
 * @param buf Command buffer
 * @param value Dword value to append
 * @return AQL_SUCCESS or error code
 */
static inline aql_result_t aql_cmd_buffer_append_dword(aql_cmd_buffer_t* buf,
                                                      uint32_t value) {
    return aql_cmd_buffer_append_dwords(buf, &value, 1);
}

/**
 * @brief Reserve space in command buffer without writing
 * @param buf Command buffer
 * @param dword_count Number of dwords to reserve
 * @param reserved_ptr Output: pointer to reserved space
 * @return AQL_SUCCESS or error code
 */
aql_result_t aql_cmd_buffer_reserve(aql_cmd_buffer_t* buf, size_t dword_count,
                                   uint32_t** reserved_ptr);

/**
 * @brief Commit previously reserved space
 * @param buf Command buffer
 * @param dword_count Number of dwords actually used
 * @return AQL_SUCCESS or error code
 */
aql_result_t aql_cmd_buffer_commit(aql_cmd_buffer_t* buf, size_t dword_count);

/**
 * @brief Set a specific dword in the buffer (for patching)
 * @param buf Command buffer
 * @param index Index of dword to set
 * @param value New value for the dword
 * @return AQL_SUCCESS or error code
 */
aql_result_t aql_cmd_buffer_set_dword(aql_cmd_buffer_t* buf,
                                     size_t index, uint32_t value);

/*
 * Buffer Alignment and Padding
 */

/**
 * @brief Align buffer to specified boundary with NOP padding
 * @param buf Command buffer
 * @param alignment Alignment boundary in dwords
 * @return AQL_SUCCESS or error code
 */
aql_result_t aql_cmd_buffer_align(aql_cmd_buffer_t* buf, size_t alignment);

/**
 * @brief Add NOP padding to reach specific size
 * @param buf Command buffer
 * @param target_size Target size in dwords
 * @return AQL_SUCCESS or error code
 */
aql_result_t aql_cmd_buffer_pad_to_size(aql_cmd_buffer_t* buf, size_t target_size);

/*
 * Buffer Validation and Debug
 */

/**
 * @brief Validate buffer contents for correctness
 * @param buf Command buffer to validate
 * @return AQL_SUCCESS if valid, error code if validation fails
 */
aql_result_t aql_cmd_buffer_validate(const aql_cmd_buffer_t* buf);

/**
 * @brief Copy buffer contents to another buffer
 * @param src Source buffer
 * @param dst Destination buffer
 * @return AQL_SUCCESS or error code
 */
aql_result_t aql_cmd_buffer_copy(const aql_cmd_buffer_t* src, aql_cmd_buffer_t* dst);

/**
 * @brief Compare two buffers for equality
 * @param buf1 First buffer
 * @param buf2 Second buffer
 * @return True if buffers contain identical data
 */
bool aql_cmd_buffer_equals(const aql_cmd_buffer_t* buf1, const aql_cmd_buffer_t* buf2);

/*
 * Debug and Tracing Support
 */

#ifdef AQL_DEBUG_TRACE

/**
 * @brief Print buffer contents in hex format
 * @param buf Command buffer to print
 * @param prefix Prefix string for debug output
 */
void aql_cmd_buffer_debug_print(const aql_cmd_buffer_t* buf, const char* prefix);

/**
 * @brief Print last N dwords added to buffer
 * @param buf Command buffer
 * @param count Number of recent dwords to print
 * @param command_name Name of command for debug output
 */
void aql_cmd_buffer_debug_print_recent(const aql_cmd_buffer_t* buf,
                                      size_t count, const char* command_name);

/**
 * @brief Validate and print a PM4 packet
 * @param buf Command buffer
 * @param packet_start Index of packet start in buffer
 * @param packet_name Name of packet for debug output
 * @return AQL_SUCCESS if packet is valid
 */
aql_result_t aql_cmd_buffer_debug_validate_packet(const aql_cmd_buffer_t* buf,
                                                 size_t packet_start,
                                                 const char* packet_name);

#else

/* Debug functions become no-ops when debugging disabled */
#define aql_cmd_buffer_debug_print(buf, prefix) do {} while(0)
#define aql_cmd_buffer_debug_print_recent(buf, count, name) do {} while(0)
#define aql_cmd_buffer_debug_validate_packet(buf, start, name) AQL_SUCCESS

#endif /* AQL_DEBUG_TRACE */

/*
 * Utility Macros for Command Building
 */

/**
 * @brief Macro to safely append a command to buffer with debug tracing
 * @param buf Command buffer
 * @param cmd_data Pointer to command data
 * @param cmd_size Size of command in dwords
 * @param cmd_name Name of command for debugging
 */
#define AQL_APPEND_COMMAND(buf, cmd_data, cmd_size, cmd_name) \
    do { \
        aql_result_t _result = aql_cmd_buffer_append_dwords(buf, cmd_data, cmd_size); \
        if (_result != AQL_SUCCESS) return _result; \
        aql_cmd_buffer_debug_print_recent(buf, cmd_size, cmd_name); \
    } while(0)

/**
 * @brief Macro to append a single dword with debug tracing
 * @param buf Command buffer
 * @param value Dword value
 * @param desc Description for debugging
 */
#define AQL_APPEND_DWORD(buf, value, desc) \
    do { \
        aql_result_t _result = aql_cmd_buffer_append_dword(buf, value); \
        if (_result != AQL_SUCCESS) return _result; \
        aql_cmd_buffer_debug_print_recent(buf, 1, desc); \
    } while(0)

/**
 * @brief Check buffer space before operation
 * @param buf Command buffer
 * @param required_dwords Number of dwords needed
 */
#define AQL_CHECK_BUFFER_SPACE(buf, required_dwords) \
    do { \
        if (!aql_cmd_buffer_is_valid(buf)) return AQL_ERROR_INVALID_ARGUMENT; \
        if (aql_cmd_buffer_available(buf) < (required_dwords)) return AQL_ERROR_BUFFER_TOO_SMALL; \
    } while(0)

/*
 * High-Level Command Buffer Operations
 */

/**
 * @brief Create a command buffer with default size for typical operations
 * @param buf Command buffer to initialize
 * @param alloc_cb Memory allocation callback
 * @param userdata User data for allocation
 * @return AQL_SUCCESS or error code
 */
static inline aql_result_t aql_cmd_buffer_create_default(aql_cmd_buffer_t* buf,
                                                        aql_memory_alloc_cb_t alloc_cb,
                                                        void* userdata) {
    return aql_cmd_buffer_init_alloc(buf, 1024, alloc_cb, userdata); /* 4KB default */
}

/**
 * @brief Create a small command buffer for simple operations
 * @param buf Command buffer to initialize
 * @param alloc_cb Memory allocation callback
 * @param userdata User data for allocation
 * @return AQL_SUCCESS or error code
 */
static inline aql_result_t aql_cmd_buffer_create_small(aql_cmd_buffer_t* buf,
                                                      aql_memory_alloc_cb_t alloc_cb,
                                                      void* userdata) {
    return aql_cmd_buffer_init_alloc(buf, 256, alloc_cb, userdata); /* 1KB */
}

/**
 * @brief Reset buffer for reuse
 * @param buf Command buffer to reset
 */
static inline void aql_cmd_buffer_reset(aql_cmd_buffer_t* buf) {
    if (buf) {
        buf->used = 0;
    }
}

#endif /* AQL_CMD_BUFFER_H */