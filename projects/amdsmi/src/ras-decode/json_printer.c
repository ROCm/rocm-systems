// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "json_printer.h"

#include <stdio.h>

static void print_json_value_internal(JsonValue* value, int indent) {
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
      JsonPair* pair = value->data.object;
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

void print_json_value(JsonValue* value) {
  print_json_value_internal(value, 0);
  printf("\n");
}
