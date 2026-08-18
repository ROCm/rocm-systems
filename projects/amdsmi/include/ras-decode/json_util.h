// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef JSON_UTIL_H
#define JSON_UTIL_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief JSON value types enumeration
 */
typedef enum { JSON_NULL, JSON_BOOL, JSON_NUMBER, JSON_STRING, JSON_OBJECT, JSON_ARRAY } JsonType;

typedef struct JsonValue JsonValue;
typedef struct JsonPair JsonPair;

/**
 * @brief JSON key-value pair structure for objects
 */
struct JsonPair {
  char* key;
  JsonValue* value;
  JsonPair* next;
};

/**
 * @brief JSON value structure
 */
struct JsonValue {
  JsonType type;
  union {
    bool boolean;
    double number;
    char* string;
    JsonPair* object;  // Linked list of key-value pairs
    struct {
      JsonValue** items;
      size_t count;
      size_t capacity;
    } array;
  } data;
};

/**
 * @brief Create a null JSON value
 * @return Pointer to new JsonValue or NULL on failure
 */
JsonValue* json_create_null(void);

/**
 * @brief Create a boolean JSON value
 * @param b Boolean value
 * @return Pointer to new JsonValue or NULL on failure
 */
JsonValue* json_create_bool(bool b);

/**
 * @brief Create a number JSON value
 * @param num Numeric value
 * @return Pointer to new JsonValue or NULL on failure
 */
JsonValue* json_create_number(double num);

/**
 * @brief Create a string JSON value
 * @param str String value (will be copied)
 * @return Pointer to new JsonValue or NULL on failure
 */
JsonValue* json_create_string(const char* str);

/**
 * @brief Create an empty JSON object
 * @return Pointer to new JsonValue or NULL on failure
 */
JsonValue* json_create_object(void);

/**
 * @brief Create an empty JSON array
 * @return Pointer to new JsonValue or NULL on failure
 */
JsonValue* json_create_array(void);

/**
 * @brief Add a key-value pair to a JSON object
 * @param obj JSON object to modify
 * @param key Key string (will be copied)
 * @param value Value to add
 */
void json_object_set(JsonValue* obj, const char* key, JsonValue* value);

/**
 * @brief Get a value by key from a JSON object
 * @param obj JSON object to search
 * @param key Key to search for
 * @return Pointer to JsonValue or NULL if not found
 */
JsonValue* json_object_get(JsonValue* obj, const char* key);

/**
 * @brief Check if a key exists in a JSON object
 * @param obj JSON object to check
 * @param key Key to check for
 * @return true if key exists, false otherwise
 */
bool json_object_has_key(JsonValue* obj, const char* key);

/**
 * @brief Add a value to a JSON array
 * @param arr JSON array to modify
 * @param value Value to add
 * @return true on success, false on failure
 */
bool json_array_push(JsonValue* arr, JsonValue* value);

/**
 * @brief Get a value by index from a JSON array
 * @param arr JSON array to access
 * @param index Array index
 * @return Pointer to JsonValue or NULL if index out of bounds
 */
JsonValue* json_array_get(JsonValue* arr, size_t index);

/**
 * @brief Get the size of a JSON array
 * @param arr JSON array
 * @return Number of elements in array, or 0 if not an array
 */
size_t json_array_size(JsonValue* arr);

/**
 * @brief Free a JSON value and all its contents
 * @param val JSON value to free
 */
void json_free(JsonValue* val);

#ifdef __cplusplus
}
#endif

#endif /* JSON_UTIL_H */
