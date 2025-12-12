# Copyright (c) 2016-2025, NVIDIA CORPORATION. All rights reserved.
# Modifications Copyright (c) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information


# We require both of the following paths to be set upon including this makefile
# TEST_VERIFIABLE_SRCDIR = <points to this directory>
# TEST_VERIFIABLE_BUILDDIR = <points to destination of .so file>

TEST_VERIFIABLE_HDRS      = $(TEST_VERIFIABLE_SRCDIR)/verifiable.h
TEST_VERIFIABLE_OBJS      = $(TEST_VERIFIABLE_BUILDDIR)/verifiable.o
TEST_VERIFIABLE_LIBS      = $(TEST_VERIFIABLE_BUILDDIR)/libverifiable.so

${HIPIFY_DIR}/verifiable.cu.cpp: $(TEST_VERIFIABLE_SRCDIR)/verifiable.cu
	@printf "Hipifying  %-35s > %s\n" $< $@
	@mkdir -p ${HIPIFY_DIR}
	${HIPIFY_PL_EXE} ${HIPIFY_PL_FLAGS} $< > $@

${HIPIFY_DIR}/verifiable.h: $(TEST_VERIFIABLE_SRCDIR)/verifiable.h
	@printf "Hipifying  %-35s > %s\n" $< $@
	@mkdir -p ${HIPIFY_DIR}
	${HIPIFY_PL_EXE} ${HIPIFY_PL_FLAGS} $< > $@

${HIPIFY_DIR}/rccl_float8.h: $(TEST_VERIFIABLE_SRCDIR)/../src/rccl_float8.h
	@printf "Hipifying  %-35s > %s\n" $< $@
	@mkdir -p ${HIPIFY_DIR}
	${HIPIFY_PL_EXE} ${HIPIFY_PL_FLAGS} $< > $@

$(TEST_VERIFIABLE_BUILDDIR)/verifiable.o: $(HIPIFY_DIR)/verifiable.cu.cpp $(HIPIFY_DIR)/verifiable.h $(HIPIFY_DIR)/rccl_float8.h
	@printf "Compiling %s\n" $@
	@mkdir -p $(TEST_VERIFIABLE_BUILDDIR)
	echo " $(HIPCC) -o $@ $(HIPCUFLAGS) -c $<"
	$(HIPCC) -o $@ $(HIPCUFLAGS) -c $<

$(TEST_VERIFIABLE_BUILDDIR)/libverifiable.so: $(TEST_VERIFIABLE_OBJS)
	@printf "Creating DSO %s\n" $@
	@mkdir -p $(TEST_VERIFIABLE_BUILDDIR)
	$(CC) -shared -o $@.0 $^ -Wl,-soname,$(notdir $@).0
	ln -sf $(notdir $@).0 $@
