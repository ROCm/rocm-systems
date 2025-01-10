# -------------------------------------------------------------------------------------- #
#
# video decode tests
#
# -------------------------------------------------------------------------------------- #

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_RUNTIME SKIP_REWRITE
    NAME videodecode
    TARGET videodecode
    GPU ON
    RUN_ARGS -i ${ROCmVersion_DIR}/share/rocdecode/video/ -t 2
    LABELS "videodecode")

rocprofiler_systems_add_validation_test(
    NAME videodecode-sampling
    PERFETTO_METRIC "host"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "videodecode"
    ARGS -l videodecode -c 1 -d 0 --counter-names "GPU VCN Activity")
