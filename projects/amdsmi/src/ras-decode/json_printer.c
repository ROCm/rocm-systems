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

#include "json_printer.h"
#include <stdio.h>

static void print_json_value_internal(JsonValue *value, int indent) {
    if (!value) return;
    
    switch (value->type) {
        case JSON_NULL:
            printf("null");
            break;
        case JSON_BOOL:
            printf("%s", value->data.boolean ? "true" : "false");
            break;
        case JSON_NUMBER:
            printf("%.0f", value->data.number);
            break;
        case JSON_STRING:
            printf("\"%s\"", value->data.string ? value->data.string : "");
            break;
        case JSON_OBJECT: {
            printf("{\n");
            JsonPair *pair = value->data.object;
            bool first = true;
            while (pair) {
                if (!first) printf(",\n");
                for (int i = 0; i < indent + 3; i++) printf(" ");
                printf("\"%s\": ", pair->key);
                print_json_value_internal(pair->value, indent + 3);
                pair = pair->next;
                first = false;
            }
            printf("\n");
            for (int i = 0; i < indent; i++) printf(" ");
            printf("}");
            break;
        }
        case JSON_ARRAY: {
            printf("[");
            for (size_t i = 0; i < value->data.array.count; i++) {
                if (i > 0) printf(", ");
                print_json_value_internal(value->data.array.items[i], indent);
            }
            printf("]");
            break;
        }
    }
}

void print_json_value(JsonValue *value) {
    print_json_value_internal(value, 0);
    printf("\n");
}
