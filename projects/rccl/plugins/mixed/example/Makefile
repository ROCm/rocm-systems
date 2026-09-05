#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for more license information
#
.PHONY=test

INC:= -I$(CURDIR)/nccl

ifeq ($(shell uname -s), Linux)
  CFLAGS += -DNCCL_OS_LINUX
else ifeq ($(OS), Windows_NT)
  CFLAGS += -DNCCL_OS_WINDOWS
endif

PLUGIN_SO:=librccl-mixed.so

default: $(PLUGIN_SO)

$(PLUGIN_SO): plugin.c
	$(CC) $(CFLAGS) $(INC) -fPIC -shared -o $@ -Wl,-soname,$(PLUGIN_SO) $^

test: $(PLUGIN_SO)
	@bash test.sh

clean:
	rm -f $(PLUGIN_SO)
