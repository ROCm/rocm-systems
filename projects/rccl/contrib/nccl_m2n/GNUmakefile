# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# See LICENSE.txt for more license information.

# ============================================================================
# Front-end that picks the build backend automatically (GNU make reads
# GNUmakefile before Makefile), so `make [target]` works unchanged on ROCm:
#   * hipcc present -> Makefile.rocm  (ROCm build)
#   * otherwise     -> Makefile       (upstream build)
#
# Options:
#   make M2N_BACKEND=Makefile.rocm   # force ROCm backend
#   make M2N_BACKEND=Makefile        # force upstream backend
# ============================================================================

ROCM_PATH ?= /opt/rocm

ifeq ($(origin M2N_BACKEND),undefined)
  ifneq ($(wildcard $(ROCM_PATH)/bin/hipcc),)
    M2N_BACKEND := Makefile.rocm
  else ifneq ($(shell command -v hipcc 2>/dev/null),)
    M2N_BACKEND := Makefile.rocm
  else
    M2N_BACKEND := Makefile
  endif
endif

# Forward every requested goal (and the bare `make` default) to the selected
# backend in a SINGLE sub-make. The goals depend on one shared forwarding rule
# so multiple goals (`make tests bench`) still run exactly one sub-make, and
# the goals themselves become no-ops.
GOALS := $(or $(MAKECMDGOALS),all)

.PHONY: $(GOALS) __m2n_dispatch
$(GOALS): __m2n_dispatch ; @:

__m2n_dispatch:
	@$(MAKE) --no-print-directory -f $(M2N_BACKEND) $(MAKECMDGOALS)
