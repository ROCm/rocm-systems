/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "json_util.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define JSON_ARRAY_INITIAL_CAPACITY 16

JsonValue* json_create_null(void) {
    JsonValue *val = calloc(1, sizeof(JsonValue));
    if (!val) return NULL;
    val->type = JSON_NULL;
    return val;
}

JsonValue* json_create_bool(bool b) {
    JsonValue *val = calloc(1, sizeof(JsonValue));
    if (!val) return NULL;
    val->type = JSON_BOOL;
    val->data.boolean = b;
    return val;
}

JsonValue* json_create_number(double num) {
    JsonValue *val = calloc(1, sizeof(JsonValue));
    if (!val) return NULL;
    val->type = JSON_NUMBER;
    val->data.number = num;
    return val;
}

JsonValue* json_create_string(const char *str) {
    if (!str) return NULL;
    
    JsonValue *val = calloc(1, sizeof(JsonValue));
    if (!val) return NULL;
    
    val->type = JSON_STRING;
    val->data.string = strdup(str);
    if (!val->data.string) {
        free(val);
        return NULL;
    }
    return val;
}

JsonValue* json_create_object(void) {
    JsonValue *val = calloc(1, sizeof(JsonValue));
    if (!val) return NULL;
    val->type = JSON_OBJECT;
    val->data.object = NULL;
    return val;
}

JsonValue* json_create_array(void) {
    JsonValue *val = calloc(1, sizeof(JsonValue));
    if (!val) return NULL;
    
    val->type = JSON_ARRAY;
    val->data.array.items = malloc(sizeof(JsonValue*) * JSON_ARRAY_INITIAL_CAPACITY);
    if (!val->data.array.items) {
        free(val);
        return NULL;
    }
    val->data.array.count = 0;
    val->data.array.capacity = JSON_ARRAY_INITIAL_CAPACITY;
    return val;
}

void json_object_set(JsonValue *obj, const char *key, JsonValue *value) {
    if (!obj || obj->type != JSON_OBJECT || !key || !value) return;
    
    // Check if key already exists and update it
    JsonPair *current = obj->data.object;
    while (current) {
        if (strcmp(current->key, key) == 0) {
            json_free(current->value);
            current->value = value;
            return;
        }
        current = current->next;
    }
    
    // Key doesn't exist, create new pair
    JsonPair *pair = malloc(sizeof(JsonPair));
    if (!pair) return;
    
    pair->key = strdup(key);
    if (!pair->key) {
        free(pair);
        return;
    }
    
    pair->value = value;
    pair->next = NULL;
    
    if (!obj->data.object) {
        obj->data.object = pair;
    } else {
        JsonPair *last = obj->data.object;
        while (last->next) {
            last = last->next;
        }
        last->next = pair;
    }
}

JsonValue* json_object_get(JsonValue *obj, const char *key) {
    if (!obj || obj->type != JSON_OBJECT || !key) return NULL;
    
    JsonPair *current = obj->data.object;
    while (current) {
        if (strcmp(current->key, key) == 0) {
            return current->value;
        }
        current = current->next;
    }
    return NULL;
}

bool json_object_has_key(JsonValue *obj, const char *key) {
    return json_object_get(obj, key) != NULL;
}

bool json_array_push(JsonValue *arr, JsonValue *value) {
    if (!arr || arr->type != JSON_ARRAY || !value) return false;
    
    // Resize array if needed
    if (arr->data.array.count >= arr->data.array.capacity) {
        size_t new_capacity = arr->data.array.capacity * 2;
        JsonValue **new_items = realloc(arr->data.array.items, 
                                       sizeof(JsonValue*) * new_capacity);
        if (!new_items) return false;
        
        arr->data.array.items = new_items;
        arr->data.array.capacity = new_capacity;
    }
    
    arr->data.array.items[arr->data.array.count] = value;
    arr->data.array.count++;
    return true;
}

JsonValue* json_array_get(JsonValue *arr, size_t index) {
    if (!arr || arr->type != JSON_ARRAY || index >= arr->data.array.count) {
        return NULL;
    }
    return arr->data.array.items[index];
}

size_t json_array_size(JsonValue *arr) {
    if (!arr || arr->type != JSON_ARRAY) return 0;
    return arr->data.array.count;
}

void json_free(JsonValue *val) {
    if (!val) return;
    
    switch (val->type) {
        case JSON_STRING:
            free(val->data.string);
            break;
        case JSON_OBJECT: {
            JsonPair *current = val->data.object;
            while (current) {
                JsonPair *next = current->next;
                free(current->key);
                json_free(current->value);
                free(current);
                current = next;
            }
            break;
        }
        case JSON_ARRAY:
            for (size_t i = 0; i < val->data.array.count; i++) {
                json_free(val->data.array.items[i]);
            }
            free(val->data.array.items);
            break;
        default:
            break;
    }
    free(val);
}
