# ---------------------------------------------------------------------------
# src/register/: one binary per test file, each linking only the fakes its unit needs.
# ---------------------------------------------------------------------------
# register_fakes.cc mirrors both arms of register.cc's cuMem split.
set(_rccl_local_register_default [=[
#if NCCL_CUMEM_VERSION_SUPPORTED(HIP_VERSION)
NCCL_PARAM(LocalRegister, "LOCAL_REGISTER", 1);
#else
NCCL_PARAM(LocalRegister, "LOCAL_REGISTER", 0);
#endif]=])
rccl_assert_source_text_count("${CMAKE_CURRENT_LIST_DIR}/../../../src/register/register.cc"
  "${_rccl_local_register_default}" 1
  "fakes/register_fakes.cc: ncclParamLocalRegister()")
unset(_rccl_local_register_default)
set(_MICRO_REGISTER_DEFS
  REGISTER_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/register/register.cc"
  COLL_REG_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/register/coll_reg.cc"
  SENDRECV_REG_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/register/sendrecv_reg.cc")

# One src/register/ test file per binary; see rccl_add_unit_micro_binary().
macro(rccl_add_register_micro_binary _name _test)
  rccl_add_unit_micro_binary(${_name} register/${_test} ${ARGN}
    OBJECT_PREFIX rccl-register-obj
    COMMON_DEFS ${_MICRO_REGISTER_DEFS}
    CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_register.yaml")
endmacro()

rccl_add_register_micro_binary(rccl-UnitTestsMicroRegister register-test.cc
  SOURCES
    fakes/argcheck_fakes.cc
    fakes/comm_fakes.cc
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc
    fakes/recorder_fakes.cc
    fakes/transport_p2p_reg_fakes.cc
    fakes/transport_stubs.cc
    fakes/utils_fakes.cc)

rccl_add_register_micro_binary(rccl-UnitTestsMicroRegisterCollReg coll-reg-test.cc
  SOURCES
    fakes/enqueue_fakes.cc
    fakes/nccl_fakes.cc
    fakes/register_fakes.cc
    fakes/transport_p2p_reg_fakes.cc
    fakes/transport_stubs.cc)

rccl_add_register_micro_binary(rccl-UnitTestsMicroRegisterSendrecvReg sendrecv-reg-test.cc
  SOURCES
    fakes/enqueue_fakes.cc
    fakes/nccl_fakes.cc
    fakes/register_fakes.cc
    fakes/transport_p2p_reg_fakes.cc
    fakes/transport_stubs.cc)
