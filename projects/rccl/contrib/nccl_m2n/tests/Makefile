# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# See LICENSE.txt for more license information.

# Makefile for the basic_api correctness suite.
#
# Mirrors an external pytest reference matrix at the C level. Two binaries
# share a single descriptor table in basic_api_test_core.h:
#
# TU split (matches MR !34's host-cc-refactor convention):
#   test_helpers.cu    — kernels + launchers (the only file that needs nvcc)
#   *.cc / *.h         — host-only test driver code; nvcc forwards .cc to g++
#
#   basic_api_test_mpi   — MPI bootstrap (multi-host)
#   basic_api_test_local — single-process pthreads via ncclCommInitAll
#
# This Makefile is fully self-contained: it can be built standalone (`make`
# from inside tests/) or invoked from the project root via the delegation
# rule there (`make tests` at the project root).
#
# Both Makefiles include ../Makefile.common to pick up the same NCCL/CUDA/MPI
# toolchain config.
#
# Usage (standalone):
#   make                        - Build all public test binaries
#   make basic_api_test_mpi
#   make basic_api_test_local
#   make clean                  - Remove only the test binaries
#   make help                   - Show this help
#
# Usage (from project root):
#   make tests                  - Equivalent to `make -C tests`
#   make basic_api_test_mpi     - Delegates to `make -C tests basic_api_test_mpi`
#   make tests-clean            - Delegates to `make -C tests clean`
#
# Required env: NCCL_HOME (see ../Makefile.common).

.DEFAULT_GOAL := all

include ../Makefile.common
include gtest.mk

# Test sources / headers (paths absolute via $(TESTSDIR) from common.mk).
# test_helpers.cu is the only nvcc-required TU; both binaries link it in.
TEST_HDRS := $(TESTSDIR)/test_helpers.h $(TESTSDIR)/basic_api_test_core.h
TEST_HELPERS_SRC := $(TESTSDIR)/test_helpers.cu
TEST_MPI_SRCS    := $(TESTSDIR)/basic_api_test_mpi.cc   \
                    $(TESTSDIR)/pipe_graph_path_test.cc \
                    $(TEST_HELPERS_SRC)
TEST_LOCAL_SRCS  := $(TESTSDIR)/nonblocking_comm_test.cc \
                    $(TESTSDIR)/basic_api_test_local.cc \
                    $(TESTSDIR)/pack_staging_pool_test.cc \
                    $(TESTSDIR)/devcomm_cache_test.cc \
                    $(TEST_HELPERS_SRC)

.PHONY: all tests basic_api_test_mpi basic_api_test_local gtest gtest-clean clean help

# Multi-dimensional planning and MPI checks do not depend on GTest.
.PHONY: multi_dim_geometry_test multi_dim_test_mpi
multi_dim_geometry_test: $(BINDIR)/multi_dim_geometry_test
multi_dim_test_mpi: $(BINDIR)/multi_dim_test_mpi
$(BINDIR)/multi_dim_geometry_test: $(TESTSDIR)/multi_dim_geometry_test.cc $(LIB_SRCS) $(LIB_HDRS)
	@mkdir -p $(BINDIR)
	$(NVCC) -o $@ $(NVCUFLAGS) -I$(NCCL_RESHARD_ROOT)/src $< $(LIB_SRCS) $(LIB_LDFLAGS) -lpthread -ldl
$(BINDIR)/multi_dim_test_mpi: $(TESTSDIR)/multi_dim_test.cc $(M2N_LIBTARGET) $(LIB_HDRS)
	@mkdir -p $(BINDIR)
	$(NVCC) -o $@ $(NVCUFLAGS) $(MPI_CFLAGS) -DOMPI_SKIP_MPICXX=1 $< $(M2N_CLIENT_LDFLAGS) $(BENCH_LDFLAGS) -lpthread

all: tests
tests: multi_dim_geometry_test multi_dim_test_mpi
.PHONY: pipe_commb_resources_test
pipe_commb_resources_test: $(BINDIR)/pipe_commb_resources_test
	$<

$(BINDIR)/pipe_commb_resources_test: $(TESTSDIR)/pipe_commb_resources_test.cc $(SRCDIR)/reshard_pipe_resources.h
	@mkdir -p $(BINDIR)
	$(CXX) -std=c++17 -Wall -Wextra -I$(SRCDIR) $< -o $@

tests: pipe_commb_resources_test

ifeq ($(GTEST_AVAILABLE),0)

# No gtest source available — emit a friendly no-op for every gtest-
# dependent target. Library and benchmarks build fine without these;
# point GTEST_DIR at a googletest source tree to build the tests.
tests basic_api_test_mpi basic_api_test_local gtest:
	@echo "tests/Makefile: googletest not found — skipping basic_api test build."
	@echo "  Looked for $(GTEST_SUBMODULE) and \$$(TESTSDIR)/googletest."
	@echo "  Fetch the submodule:"
	@echo "    git submodule update --init third_party/googletest"
	@echo "  Or point GTEST_DIR at a googletest source tree:"
	@echo "    make tests GTEST_DIR=/path/to/googletest"

gtest-clean:
	rm -rf $(GTEST_OBJDIR)

clean: gtest-clean
	@printf "Cleaning test binaries...\n"
	rm -f $(BINDIR)/basic_api_test_mpi $(BINDIR)/basic_api_test_local $(BINDIR)/multi_dim_geometry_test $(BINDIR)/multi_dim_test_mpi

else

tests: $(BINDIR)/basic_api_test_mpi $(BINDIR)/basic_api_test_local

basic_api_test_mpi:   $(BINDIR)/basic_api_test_mpi   ; @:
basic_api_test_local: $(BINDIR)/basic_api_test_local ; @:

$(BINDIR)/basic_api_test_mpi: $(TEST_MPI_SRCS) $(GTEST_A) \
                              $(M2N_LIBTARGET) $(LIB_HDRS) $(TEST_HDRS)
	@printf "Building basic_api_test_mpi (gtest, MPI, GTEST_DIR=$(GTEST_DIR))...\n"
	@mkdir -p $(BINDIR)
	$(NVCC) -o $@ $(NVCUFLAGS) $(MPI_CFLAGS) -I$(TESTSDIR) -I$(GTEST_INC) \
	    $(TEST_MPI_SRCS) $(GTEST_A) $(M2N_CLIENT_LDFLAGS) $(BENCH_LDFLAGS) -lpthread
	@printf "Done! Test binary at: $@\n"

# basic_api_test_local exercises library-internal symbols, including the
# NCCL_M2N_TESTING-only reshardTestBroadcastMaxInt, so link the library sources
# directly (same pattern as tests/ci/unit_tests) instead of -lnccl_m2n.
$(BINDIR)/basic_api_test_local: $(TEST_LOCAL_SRCS) $(LIB_SRCS) $(GTEST_A) \
                                $(LIB_HDRS) $(TEST_HDRS)
	@printf "Building basic_api_test_local (gtest, no MPI, GTEST_DIR=$(GTEST_DIR))...\n"
	@mkdir -p $(BINDIR)
	$(NVCC) -o $@ $(NVCUFLAGS) -DNCCL_M2N_TESTING -I$(TESTSDIR) -I$(GTEST_INC) \
	    $(TEST_LOCAL_SRCS) $(LIB_SRCS) $(GTEST_A) $(LIB_LDFLAGS) -lpthread -ldl
	@printf "Done! Test binary at: $@\n"

clean:
	@printf "Cleaning test binaries...\n"
	rm -f $(BINDIR)/basic_api_test_mpi $(BINDIR)/basic_api_test_local $(BINDIR)/multi_dim_geometry_test $(BINDIR)/multi_dim_test_mpi
	rm -rf $(GTEST_OBJDIR)

endif

# Convenience: building the lib triggers the parent Makefile, so the test
# rules above resolve $(M2N_LIBTARGET) via the implicit chain.
$(M2N_LIBTARGET):
	@$(MAKE) --no-print-directory -C $(NCCL_RESHARD_ROOT) lib

help:
	@echo "tests/Makefile — public basic_api correctness suite"
	@echo ""
	@echo "Targets:"
	@echo "  all (default)         - Build placement tests and available GTest binaries"
	@echo "  tests                 - Same as all"
	@echo "  basic_api_test_mpi    - Build only the MPI gtest matrix binary"
	@echo "  basic_api_test_local  - Build only the local pthread gtest matrix binary"
	@echo "  multi_dim_geometry_test - Build the host placement geometry test"
	@echo "  multi_dim_test_mpi    - Build the MPI placement test (requires MPI)"
	@echo "  gtest                 - Build gtest.a / gtest_main.a (from \$$(GTEST_DIR))"
	@echo "  clean                 - Remove all test binaries + gtest objects"
	@echo "  help                  - Show this help"
	@echo ""
	@echo "Configuration is shared with the parent Makefile via ../Makefile.common"
	@echo "(NCCL_HOME, CUDA_HOME, MPI_HOME, NVCC_GENCODE, DEBUG, BUILDDIR)."
	@echo ""
	@echo "From the project root, equivalent invocations are:"
	@echo "  make tests"
	@echo "  make basic_api_test_mpi"
	@echo "  make basic_api_test_local"
	@echo "  make tests-clean"
