# ---------------------------------------------------------------------------
# src/scheduler/: one binary per test file, each linking only the fakes its unit needs.
# ---------------------------------------------------------------------------
set(_MICRO_SCHEDULER_DEFS
  # Hipify renames this to *_tmp.cc: src/enqueue/task_sched/allgatherv_sched.cc keeps the basename instead.
  ALLGATHERV_SCHED_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/scheduler/allgatherv_sched_tmp.cc"
  SYMMETRIC_SCHED_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/scheduler/symmetric_sched.cc")

# One src/scheduler/ test file per binary; see rccl_add_unit_micro_binary().
macro(rccl_add_scheduler_micro_binary _name _test)
  rccl_add_unit_micro_binary(${_name} scheduler/${_test} ${ARGN}
    OBJECT_PREFIX rccl-scheduler-obj
    COMMON_DEFS ${_MICRO_SCHEDULER_DEFS}
    CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_scheduler.yaml")
endmacro()

rccl_add_scheduler_micro_binary(rccl-UnitTestsMicroScheduler scheduler-test.cc
  SOURCES
    fakes/enqueue_fakes.cc
    fakes/bootstrap_stubs.cc
    fakes/nccl_stubs.cc
    fakes/nccl_fakes.cc
    fakes/collectives_fakes.cc
    fakes/dev_runtime_fakes.cc
    fakes/profiler_fakes.cc
    fakes/rccl_wrap_fakes.cc
    fakes/sym_kernels_fakes.cc
    fakes/sym_kernels_index_fakes.cc
    fakes/tuning_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
  EXTRA_DEFS
    # Binary-wide: every TU must agree on struct ncclComm's layout.
    ENABLE_WARP_SPEED
    # Not inherited via RCCL_COMMON_COMPILE_DEFS (unlike ENABLE_FAULT_INJECTION); the default-ON, shipping config.
    GENERATE_SYM_KERNELS)
