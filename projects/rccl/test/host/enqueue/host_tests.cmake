# ---------------------------------------------------------------------------
# src/enqueue/: one binary per test file, each linking only the fakes its unit needs.
# ---------------------------------------------------------------------------
# rccl_mark_oracle_tus_generated() only covers misc/; these hipified siblings live elsewhere.
set_source_files_properties(
  ${RCCL_HOST_HIPIFY_DIR}/src/config/algorithm_parser.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/config/algorithm_registry.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/config/collconfig.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/enqueue/task_prep/task_classify.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/enqueue/task_prep/task_posttuning.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/enqueue/task_prep/task_pretuning.cc
  PROPERTIES GENERATED TRUE)
set(_MICRO_ENQUEUE_DEFS
  ${_MICRO_AMDSMI_DEF}
  ENQUEUE_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/enqueue/enqueue.cc"
  MGMT_TASK_ENQ_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/enqueue/mgmt_task_enq.cc"
  # Hipify renames none of task_prep/ or task_sched/: the allgatherv_sched _tmp.cc rename lands on
  # src/scheduler/allgatherv_sched.cc instead.
  TASK_PREP_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/enqueue/task_prep/task_prep.cc"
  TASK_CLASSIFY_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/enqueue/task_prep/task_classify.cc"
  TASK_PRETUNING_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/enqueue/task_prep/task_pretuning.cc"
  TASK_POSTTUNING_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/enqueue/task_prep/task_posttuning.cc"
  TASK_SCHED_DISPATCH_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/enqueue/task_sched/task_sched.cc"
  TASK_SCHED_SYM_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/enqueue/task_sched/sym_sched.cc"
  TASK_SCHED_LEGACY_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/enqueue/task_sched/legacy_sched.cc"
  TASK_SCHED_ALLGATHERV_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/enqueue/task_sched/allgatherv_sched.cc"
  TASK_SCHED_P2P_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/enqueue/task_sched/p2p_sched.cc"
  TASK_SCHED_RMA_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/enqueue/task_sched/rma_sched.cc"
  TASK_SCHED_CE_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/enqueue/task_sched/ce_sched.cc")

# One src/enqueue/ test file per binary; see rccl_add_unit_micro_binary().
macro(rccl_add_enqueue_micro_binary _name _test)
  rccl_add_unit_micro_binary(${_name} enqueue/${_test} ${ARGN}
    OBJECT_PREFIX rccl-enqueue-obj
    COMMON_INCLUDES
      ${RCCL_HOST_HIPIFY_DIR}/src/device               # device/common.h (neutered)
      ${RCCL_HOST_HIPIFY_DIR}/src/include/nccl_device
    COMMON_DEFS ${_MICRO_ENQUEUE_DEFS}
    CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_enqueue.yaml")
endmacro()

set(_enq_sources
  fakes/allocator_fakes.cc
  fakes/bootstrap_stubs.cc
  fakes/ce_fakes.cc
  fakes/channel_fakes.cc
  fakes/collconfig_fakes.cc
  fakes/collectives_fakes.cc
  fakes/comm_fakes.cc
  fakes/dev_runtime_fakes.cc
  fakes/enqueue_test_deps.cc
  fakes/env_fakes.cc
  fakes/group_fakes.cc
  fakes/hip_fakes.cc
  fakes/mgmt_task_enq_fakes.cc
  fakes/nccl_fakes.cc
  fakes/nccl_stubs.cc
  fakes/os_fakes.cc
  fakes/profiler_fakes.cc
  fakes/proxy_fakes.cc
  fakes/rccl_arch_thresholds_fakes.cc
  fakes/rccl_wrap_fakes.cc
  fakes/recorder_fakes.cc
  fakes/register_stubs.cc
  fakes/rma_fakes.cc
  fakes/sched_stubs.cc
  fakes/strongstream_stubs.cc
  fakes/sym_kernels_fakes.cc
  fakes/transport_stubs.cc
  fakes/tuning_fakes.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/misc/argcheck.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc)
set(_enq_defs
  RCCL_TEST_ENQUEUE_HOOKS
  # Per-symbol omissions from the shared nccl_stubs.cc; enqueue.cc itself
  # supplies both, so neither needs a replacement definition.
  RCCL_STUBS_OMIT_ncclInitKernelsForDevice
  RCCL_STUBS_OMIT_ncclParamGraphStreamOrdering
  # enqueue-test.cc provides these launch-prepare floors directly.
  RCCL_ENQUEUE_TEST_DEPS_OMIT_scheduleRmaTasksToPlan
  RCCL_HIP_FAKES_OMIT_hipLaunchHostFunc
  # Production librccl always builds this arm (CMakeLists.txt LL128_ENABLED=ON).
  # Standalone host CI otherwise compiles enqueue.cc without it, so addP2pToPlan's
  # gfx1250 ENABLE=1 window is unreachable there.
  ENABLE_LL128)
rccl_add_enqueue_micro_binary(rccl-UnitTestsMicroEnqueue enqueue-test.cc
  SOURCES ${_enq_sources}
  REMOVE_DEFS ENABLE_WARP_SPEED
  EXTRA_DEFS ${_enq_defs})
rccl_add_enqueue_micro_binary(rccl-UnitTestsMicroEnqueue-warpspeed enqueue-test.cc
  SOURCES ${_enq_sources}
  REMOVE_DEFS ENABLE_WARP_SPEED
  EXTRA_DEFS ${_enq_defs})
target_compile_definitions(rccl-UnitTestsMicroEnqueue-warpspeed PRIVATE ENABLE_WARP_SPEED)
if(_ENQ_BUILD_DEVLINKER)
  rccl_add_enqueue_micro_binary(rccl-UnitTestsMicroEnqueue-devlinker enqueue-test.cc
    SOURCES ${_enq_sources}
    REMOVE_DEFS ENABLE_WARP_SPEED
    EXTRA_DEFS ${_enq_defs} RCCL_DEVICE_LINKER)
endif()

# Own binary: it defines ncclMgmtTaskEnqueue, which fakes/mgmt_task_enq_fakes.cc fakes for every other target.
rccl_add_enqueue_micro_binary(rccl-UnitTestsMicroEnqueueMgmtTaskEnq mgmt-task-enq-test.cc
  SOURCES
    fakes/group_fakes.cc
    fakes/nccl_fakes.cc
    fakes/utils_fakes.cc)

# TaskPrepScene.h resets every fake it drives, so each task_prep/ and task_sched/ binary links them all.
rccl_add_enqueue_micro_binary(rccl-UnitTestsMicroEnqueueTaskPrep task_prep/task-prep-test.cc
  SOURCES
    fakes/allocator_fakes.cc
    fakes/bootstrap_stubs.cc
    fakes/ce_fakes.cc
    fakes/channel_fakes.cc
    fakes/collectives_fakes.cc
    fakes/comm_fakes.cc
    fakes/dev_runtime_fakes.cc
    fakes/enqueue_fakes.cc
    fakes/env_fakes.cc
    fakes/group_fakes.cc
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc
    fakes/nccl_stubs.cc
    fakes/os_fakes.cc
    fakes/proxy_fakes.cc
    fakes/rccl_wrap_fakes.cc
    fakes/recorder_fakes.cc
    fakes/register_stubs.cc
    fakes/rma_fakes.cc
    fakes/sched_stubs.cc
    fakes/sym_kernels_fakes.cc
    fakes/transport_stubs.cc
    fakes/tuning_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/config/algorithm_parser.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/config/algorithm_registry.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/config/collconfig.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/enqueue/task_prep/task_classify.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/enqueue/task_prep/task_posttuning.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/enqueue/task_prep/task_pretuning.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/argcheck.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc)

rccl_add_enqueue_micro_binary(rccl-UnitTestsMicroEnqueueTaskPrepTaskClassify task_prep/task-classify-test.cc
  SOURCES
    fakes/allocator_fakes.cc
    fakes/ce_fakes.cc
    fakes/collectives_fakes.cc
    fakes/comm_fakes.cc
    fakes/dev_runtime_fakes.cc
    fakes/enqueue_fakes.cc
    fakes/env_fakes.cc
    fakes/group_fakes.cc
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc
    fakes/nccl_stubs.cc
    fakes/os_fakes.cc
    fakes/proxy_fakes.cc
    fakes/rccl_wrap_fakes.cc
    fakes/recorder_fakes.cc
    fakes/register_stubs.cc
    fakes/rma_fakes.cc
    fakes/sched_stubs.cc
    fakes/sym_kernels_fakes.cc
    fakes/transport_stubs.cc
    fakes/tuning_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/config/algorithm_parser.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/config/algorithm_registry.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/config/collconfig.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc)

rccl_add_enqueue_micro_binary(rccl-UnitTestsMicroEnqueueTaskPrepTaskPretuning task_prep/task-pretuning-test.cc
  SOURCES
    fakes/allocator_fakes.cc
    fakes/ce_fakes.cc
    fakes/comm_fakes.cc
    fakes/dev_runtime_fakes.cc
    fakes/enqueue_fakes.cc
    fakes/env_fakes.cc
    fakes/group_fakes.cc
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc
    fakes/nccl_stubs.cc
    fakes/os_fakes.cc
    fakes/proxy_fakes.cc
    fakes/rccl_wrap_fakes.cc
    fakes/recorder_fakes.cc
    fakes/register_stubs.cc
    fakes/rma_fakes.cc
    fakes/sched_stubs.cc
    fakes/sym_kernels_fakes.cc
    fakes/transport_stubs.cc
    fakes/tuning_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc)

rccl_add_enqueue_micro_binary(rccl-UnitTestsMicroEnqueueTaskPrepTaskPosttuning task_prep/task-posttuning-test.cc
  SOURCES
    fakes/allocator_fakes.cc
    fakes/bootstrap_stubs.cc
    fakes/ce_fakes.cc
    fakes/channel_fakes.cc
    fakes/collectives_fakes.cc
    fakes/comm_fakes.cc
    fakes/dev_runtime_fakes.cc
    fakes/enqueue_fakes.cc
    fakes/env_fakes.cc
    fakes/group_fakes.cc
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc
    fakes/nccl_stubs.cc
    fakes/os_fakes.cc
    fakes/proxy_fakes.cc
    fakes/rccl_wrap_fakes.cc
    fakes/recorder_fakes.cc
    fakes/register_stubs.cc
    fakes/rma_fakes.cc
    fakes/sched_stubs.cc
    fakes/sym_kernels_fakes.cc
    fakes/transport_stubs.cc
    fakes/tuning_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/config/algorithm_parser.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/config/algorithm_registry.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/config/collconfig.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/argcheck.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc)

# The seven task_sched/ files share one test file: none references another.
rccl_add_enqueue_micro_binary(rccl-UnitTestsMicroEnqueueTaskSched task_sched/task-sched-test.cc
  SOURCES
    fakes/allocator_fakes.cc
    fakes/ce_fakes.cc
    fakes/comm_fakes.cc
    fakes/dev_runtime_fakes.cc
    fakes/enqueue_fakes.cc
    fakes/env_fakes.cc
    fakes/group_fakes.cc
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc
    fakes/nccl_stubs.cc
    fakes/os_fakes.cc
    fakes/proxy_fakes.cc
    fakes/rccl_wrap_fakes.cc
    fakes/recorder_fakes.cc
    fakes/register_stubs.cc
    fakes/rma_fakes.cc
    fakes/sched_stubs.cc
    fakes/sym_kernels_fakes.cc
    fakes/transport_stubs.cc
    fakes/tuning_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc)
