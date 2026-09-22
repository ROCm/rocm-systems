-- Analysis views used for benchmarking
CREATE VIEW IF NOT EXISTS
    `benchmark_analysis_{{metric}}` AS
WITH
    baseline AS (
        SELECT
            *
        FROM
            benchmark_statistics BL
        WHERE
            BL.sdk_id IS NULL
            AND BL.metric_name = "{{metric}}"
    )
SELECT
    ST.id,
    ST.app_id,
    ST.cfg_id,
    ST.sdk_id,
    BS.git_revision,
    BA.command,
    ST.metric_name,
    ST.metric_unit,
    ST.count,
    ST.mean,
    ST.std_dev AS `+/-`,
    BL.mean AS baseline_mean,
    BL.std_dev AS `+/- (baseline)`,
    ((ST.mean - BL.mean) / BL.mean) * 100 AS `overhead (%)`,
    BC.benchmark_mode,
    BC.label AS benchmark_label
FROM
    benchmark_statistics ST
    JOIN benchmark_config BC ON BC.id = ST.cfg_id
    JOIN benchmarked_sdk BS ON BS.id = ST.sdk_id
    JOIN benchmarked_app BA ON BA.id = ST.app_id
    JOIN baseline BL ON (
        BL.app_id = ST.app_id
        AND BL.metric_name = ST.metric_name
    )
WHERE
    ST.metric_name = "{{metric}}"
    AND ST.sdk_id IS NOT NULL
ORDER BY
    `overhead (%)` DESC;

-- benchmarked_app without environment info
CREATE VIEW IF NOT EXISTS
    `benchmarked_app_without_env` AS
SELECT
    id,
    hash_id,
    md5sum,
    revision,
    command,
    compiler_id,
    compiler_version,
    library_arch,
    system_name,
    system_processor,
    system_version,
    threads,
    hip_compiler_api,
    hip_runtime_api,
    hsa_api,
    kernel_dispatch,
    marker_api,
    memory_allocation,
    memory_copy,
    ompt,
    rccl_api,
    rocdecode_api,
    rocjpeg_api,
    scratch_memory
FROM
    benchmarked_app;
