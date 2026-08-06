#
# Native unit tests for NCCL inspector collInfo lifecycle (NCCL issue #2000).
#

CXX := g++
CXXFLAGS := -Wall -Wextra -g -std=c++17 -pthread
TARGET := test_coll_info_lifecycle
SOURCES := test_coll_info_lifecycle.cpp
ASAN ?= 0

GTEST_CFLAGS ?= $(shell pkg-config --cflags gtest 2>/dev/null)
GTEST_LIBS ?= $(shell pkg-config --libs gtest 2>/dev/null)

ifneq ($(GTEST_CFLAGS),)
CXXFLAGS += $(GTEST_CFLAGS)
endif

ifneq ($(GTEST_LIBS),)
LDFLAGS += $(GTEST_LIBS)
else
LDFLAGS += -lgtest
endif
LDFLAGS += -lgtest_main -pthread

ifneq ($(ASAN), 0)
CXXFLAGS += -fsanitize=address -fno-omit-frame-pointer
LDFLAGS += -fsanitize=address -static-libasan
endif

.PHONY: all test clean

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

test: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
