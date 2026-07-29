#
# Native unit tests for NCCL inspector collInfo lifecycle (NCCL issue #2000).
#

CC := gcc
CFLAGS := -Wall -Wextra -g -std=c11 -pthread
TARGET := test_coll_info_lifecycle
SOURCES := test_coll_info_lifecycle.c
ASAN ?= 0

ifneq ($(ASAN), 0)
CFLAGS += -fsanitize=address -fno-omit-frame-pointer
LDFLAGS += -fsanitize=address -static-libasan
endif

.PHONY: all test clean

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(TARGET) $(SOURCES)

test: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
