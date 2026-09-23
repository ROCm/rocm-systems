# Tuning sources
set(TUNING_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/ce_model.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/collective_execution_policy.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/collective_execution_policy_rules.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/policy_adapter.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/collnet.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cost_model.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/nvls.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/pat.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ring.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/sym_model.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/tree.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/tuning_general.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/tuning.cc
)

# Add tuning sources to parent scope
set(TUNING_SOURCES ${TUNING_SOURCES} PARENT_SCOPE)
