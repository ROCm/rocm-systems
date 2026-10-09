#
# Native unit tests for NCCL inspector collInfo lifecycle (NCCL issue #2000).
#

CXX := g++
CXXFLAGS := -Wall -Wextra -g -std=c++17 -pthread
TARGET := test_coll_info_lifecycle
SOURCES := test_coll_info_lifecycle.cpp
ASAN ?= 0

GTEST_VERSION ?= 1.12.0
GTEST_PREFIX ?= $(CURDIR)/.deps/gtest
GTEST_SRC := $(CURDIR)/.deps/googletest-release-$(GTEST_VERSION)
GTEST_LIB := $(GTEST_PREFIX)/lib/libgtest.a

# 1) pkg-config, 2) system install, 3) local .deps build
GTEST_CXXFLAGS :=
GTEST_LDFLAGS :=
GTEST_DEPS :=

ifeq ($(shell pkg-config --exists gtest 2>/dev/null && echo yes),yes)
GTEST_CXXFLAGS := $(shell pkg-config --cflags gtest)
GTEST_LDFLAGS := $(shell pkg-config --libs gtest) -lgtest_main
else ifneq ($(wildcard /usr/include/gtest/gtest.h),)
GTEST_CXXFLAGS := -DGTEST_HAS_PTHREAD=1
ifneq ($(wildcard /usr/lib/x86_64-linux-gnu/libgtest.a),)
GTEST_LDFLAGS := -L/usr/lib/x86_64-linux-gnu -lgtest -lgtest_main
else ifneq ($(wildcard /usr/lib/libgtest.a),)
GTEST_LDFLAGS := -L/usr/lib -lgtest -lgtest_main
else ifneq ($(wildcard /usr/local/lib/libgtest.a),)
GTEST_LDFLAGS := -L/usr/local/lib -lgtest -lgtest_main
else
GTEST_CXXFLAGS := -I$(GTEST_PREFIX)/include -DGTEST_HAS_PTHREAD=1
GTEST_LDFLAGS := -L$(GTEST_PREFIX)/lib -lgtest -lgtest_main
GTEST_DEPS := $(GTEST_LIB)
endif
else
GTEST_CXXFLAGS := -I$(GTEST_PREFIX)/include -DGTEST_HAS_PTHREAD=1
GTEST_LDFLAGS := -L$(GTEST_PREFIX)/lib -lgtest -lgtest_main
GTEST_DEPS := $(GTEST_LIB)
endif

GTEST_LDFLAGS += -pthread

ifneq ($(GTEST_CXXFLAGS),)
CXXFLAGS += $(GTEST_CXXFLAGS)
endif
LDFLAGS += $(GTEST_LDFLAGS)

ifneq ($(ASAN), 0)
CXXFLAGS += -fsanitize=address -fno-omit-frame-pointer
LDFLAGS += -fsanitize=address -static-libasan
endif

.PHONY: all test clean distclean

all: $(TARGET)

$(TARGET): $(SOURCES) $(GTEST_DEPS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

test: $(TARGET)
	./$(TARGET)

$(GTEST_LIB):
	@mkdir -p $(CURDIR)/.deps
	@if [ -f /usr/src/googletest/CMakeLists.txt ]; then \
		echo "Building gtest from /usr/src/googletest into $(GTEST_PREFIX)"; \
		cmake -S /usr/src/googletest -B $(GTEST_SRC)/build \
			-DCMAKE_INSTALL_PREFIX=$(GTEST_PREFIX) \
			-DBUILD_GMOCK=OFF \
			-DCMAKE_CXX_FLAGS=-DGTEST_HAS_PTHREAD=1; \
		cmake --build $(GTEST_SRC)/build --target install; \
	elif [ ! -d $(GTEST_SRC) ]; then \
		echo "Downloading googletest $(GTEST_VERSION) into $(GTEST_SRC)"; \
		curl -fsSL \
			"https://github.com/google/googletest/archive/refs/tags/release-$(GTEST_VERSION).tar.gz" \
			| tar xz -C $(CURDIR)/.deps; \
		cmake -S $(GTEST_SRC) -B $(GTEST_SRC)/build \
			-DCMAKE_INSTALL_PREFIX=$(GTEST_PREFIX) \
			-DBUILD_GMOCK=OFF \
			-DCMAKE_CXX_FLAGS=-DGTEST_HAS_PTHREAD=1; \
		cmake --build $(GTEST_SRC)/build --target install; \
	else \
		cmake --build $(GTEST_SRC)/build --target install; \
	fi

clean:
	rm -f $(TARGET)

distclean: clean
	rm -rf $(CURDIR)/.deps
