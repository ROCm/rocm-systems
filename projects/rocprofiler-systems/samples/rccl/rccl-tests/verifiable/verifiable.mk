# Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
# Modifications Copyright (c) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information


# We requires both of the following paths to be set upon including this makefile
# TEST_VERIFIABLE_SRCDIR = <points to this directory>
# TEST_VERIFIABLE_BUILDDIR = <points to destination of .o file>

TEST_VERIFIABLE_HDRS      = $(TEST_VERIFIABLE_SRCDIR)/verifiable.h
TEST_VERIFIABLE_OBJS      = $(TEST_VERIFIABLE_BUILDDIR)/verifiable.o

${HIPIFY_DIR}/verifiable.cpp: $(TEST_VERIFIABLE_SRCDIR)/verifiable.cpp
	@printf "Hipifying  %-35s > %s\n" $< $@
	@mkdir -p ${HIPIFY_DIR}
	hipify-perl -quiet-warnings $< > $@

${HIPIFY_DIR}/verifiable.h: $(TEST_VERIFIABLE_SRCDIR)/verifiable.h
	@printf "Hipifying  %-35s > %s\n" $< $@
	@mkdir -p ${HIPIFY_DIR}
	hipify-perl -quiet-warnings $< > $@

${HIPIFY_DIR}/rccl_float8.h: $(TEST_VERIFIABLE_SRCDIR)/../src/rccl_float8.h
	@printf "Hipifying  %-35s > %s\n" $< $@
	@mkdir -p ${HIPIFY_DIR}
	hipify-perl -quiet-warnings $< > $@

$(TEST_VERIFIABLE_BUILDDIR)/verifiable.o: $(HIPIFY_DIR)/verifiable.cpp $(HIPIFY_DIR)/verifiable.h $(HIPIFY_DIR)/rccl_float8.h
	@printf "Compiling %s\n" $@
	@mkdir -p $(TEST_VERIFIABLE_BUILDDIR)
	echo " $(HIPCC) -o $@ $(HIPCUFLAGS) -c $<"
	$(HIPCC) -o $@ $(HIPCUFLAGS) -c $<
