/**
 * @file utils.h
 * @brief Common utility functions
 */
#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>

/**
 * @brief Convert a 64-bit value from little endian to big endian
 * @param[in] value Value to convert
 * @return Converted value in big endian
 */
static inline uint64_t le64_to_be64(uint64_t value) {
    return ((value & 0xFF00000000000000ULL) >> 56) |
           ((value & 0x00FF000000000000ULL) >> 40) |
           ((value & 0x0000FF0000000000ULL) >> 24) |
           ((value & 0x000000FF00000000ULL) >> 8)  |
           ((value & 0x00000000FF000000ULL) << 8)  |
           ((value & 0x0000000000FF0000ULL) << 24) |
           ((value & 0x000000000000FF00ULL) << 40) |
           ((value & 0x00000000000000FFULL) << 56);
}

/**
 * @brief Convert an array of 64-bit values from little endian to big endian
 * @param[in,out] array Array to convert
 * @param[in] len Length of the array
 */
static inline void convert_array_le_to_be(uint64_t *array, size_t len) {
    for (size_t i = 0; i < len; i++) {
        array[i] = le64_to_be64(array[i]);
    }
}

#endif /* UTILS_H */ 