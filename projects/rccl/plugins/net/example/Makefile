#
# Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
#
.DEFAULT_GOAL: build
include ../../../makefiles/common.mk
SRCDIR   ?= $(abspath ../../..)
BUILDDIR ?= .
NCCLDIR  := $(BUILDDIR)

SRC_FILES := $(wildcard *.c)

build: ${BUILDDIR}/librccl-net-example.so

${BUILDDIR}/librccl-net-example.so: ${SRC_FILES}
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${BUILDDIR}
	$(CC) -Inccl -fPIC -shared -o $@ $^

clean:
	rm -f ${BUILDDIR}/librccl-net-example.so
