#
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#
.DEFAULT_GOAL: build
include ../../makefiles/common.mk
SRCDIR   ?= $(abspath ../..)
BUILDDIR ?= .
NCCLDIR  := $(BUILDDIR)

SRC_FILES := $(wildcard *.c)

build: ${BUILDDIR}/librccl-profiler.so

${BUILDDIR}/librccl-profiler.so: ${SRC_FILES}
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${BUILDDIR}
	$(CC) -Inccl -fPIC -shared -o $@ $^

clean:
	rm -f ${BUILDDIR}/librccl-profiler.so
