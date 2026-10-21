#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Parallel *_diff binaries. Included from src/Makefile. Stock *_perf link
# lines are unchanged. New objects live under $(BUILDDIR)/diff.
#

DIFF_BINS := all_reduce
DIFF_TARGETS := $(DIFF_BINS:%=${DST_DIR}/%_diff)
DIFF_OBJDIR := ${DST_DIR}/diff
DIFF_HIPIFY := ${HIPIFY_DIR}/diff

DIFF_LDFLAGS := $(filter-out -lrccl,$(HIPLDFLAGS))

.PHONY: diff
diff: $(DIFF_TARGETS)
build: $(DIFF_TARGETS)

${DIFF_HIPIFY}/%.cu.cpp: diff/%.cu
	@printf "Hipifying  %-35s > %s\n" $< $@
	@mkdir -p ${DIFF_HIPIFY}
	${HIPIFY_PL_EXE} ${HIPIFY_PL_FLAGS} $< > $@

${DIFF_OBJDIR}/common.o: ${HIPIFY_DIR}/common.cu.cpp ${HIPIFY_DIR}/common.h ${HIPIFY_DIR}/util.h
	@printf "Compiling  %-35s > %s\n" common.cu $@
	@mkdir -p ${DIFF_OBJDIR}
	$(HIPCC) $(HIPCUFLAGS) -DRCCL_TESTS_DIFF -I. -I${HIPIFY_DIR} -c -o $@ ${HIPIFY_DIR}/common.cu.cpp

${DIFF_OBJDIR}/util.o: ${HIPIFY_DIR}/util.cu.cpp ${HIPIFY_DIR}/util.h ${HIPIFY_DIR}/common.h
	@printf "Compiling  %-35s > %s\n" util.cu $@
	@mkdir -p ${DIFF_OBJDIR}
	$(HIPCC) $(HIPCUFLAGS) -DRCCL_TESTS_DIFF -I. -I${HIPIFY_DIR} -c -o $@ ${HIPIFY_DIR}/util.cu.cpp

${DIFF_OBJDIR}/diff.o: ${DIFF_HIPIFY}/diff.cu.cpp ${HIPIFY_DIR}/util.h ${HIPIFY_DIR}/common.h diff/diff_api.h diff/diff_ctrl.h
	@printf "Compiling  %-35s > %s\n" diff/diff.cu $@
	@mkdir -p ${DIFF_OBJDIR}
	$(HIPCC) $(HIPCUFLAGS) -DRCCL_TESTS_DIFF -I${HIPIFY_DIR} -I. -c -o $@ ${DIFF_HIPIFY}/diff.cu.cpp

${DIFF_OBJDIR}/diff_main.o: ${DIFF_HIPIFY}/diff_main.cu.cpp diff/diff_ctrl.h
	@printf "Compiling  %-35s > %s\n" diff/diff_main.cu $@
	@mkdir -p ${DIFF_OBJDIR}
	$(HIPCC) $(HIPCUFLAGS) -I${HIPIFY_DIR} -I. -c -o $@ ${DIFF_HIPIFY}/diff_main.cu.cpp

${DIFF_OBJDIR}/shim.o: diff/rccl_diff_shim.cpp diff/diff_ctrl.h
	@printf "Compiling  %-35s > %s\n" diff/rccl_diff_shim.cpp $@
	@mkdir -p ${DIFF_OBJDIR}
	$(HIPCC) $(HIPCUFLAGS) -fvisibility=hidden -I. -c -o $@ $<

${DST_DIR}/all_reduce_diff: ${DST_DIR}/all_reduce.o ${DIFF_OBJDIR}/common.o ${DIFF_OBJDIR}/util.o ${DIFF_OBJDIR}/diff.o ${DIFF_OBJDIR}/diff_main.o ${DIFF_OBJDIR}/shim.o ${DST_DIR}/collector.o ${DST_DIR}/timer.o $(TEST_VERIFIABLE_OBJS) $(DST_DIR)/src/git_version.cpp $(TEST_OS_LIB)
	@printf "Linking    %-35s > %s\n" all_reduce_diff $@
	@mkdir -p ${DST_DIR}
	$(HIPCC) -o $@ $^ $(DIFF_LDFLAGS)
