# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

BUILD_DIR := build

all: $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. && $(MAKE)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)
