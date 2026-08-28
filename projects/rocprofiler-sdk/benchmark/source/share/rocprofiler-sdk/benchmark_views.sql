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

-- Cost of collecting every requested counter group, for each way of collecting
-- them.
--
-- Application replay collects G groups by running the whole application G
-- times, so its cost is G times the cost of a single-group run. Kernel replay
-- collects the same G groups in one application run by replaying each dispatch
-- once per group, so its cost is whatever that run measured. Both produce every
-- group on every dispatch, which makes them comparable; multiplexed collection
-- is listed too but rotates groups across dispatches rather than collecting
-- them all everywhere, so it is cheaper for a reason.
--
-- `application_replay_projected` is only meaningful for a metric that
-- accumulates across runs, i.e. wall_time and cpu_time. For a residency metric
-- such as peak_rss the projection is left to the reader.
CREATE VIEW IF NOT EXISTS
    `benchmark_replay_{{metric}}` AS
WITH
    single_group AS (
        SELECT
            ST.app_id AS app_id,
            ST.sdk_id AS sdk_id,
            MIN(ST.mean) AS mean
        FROM
            benchmark_statistics ST
            JOIN benchmark_config BC ON BC.id = ST.cfg_id
        WHERE
            ST.metric_name = "{{metric}}"
            AND BC.counter_collection_mode = "single-pass"
        GROUP BY
            ST.app_id,
            ST.sdk_id
    )
SELECT
    ST.id,
    ST.app_id,
    ST.cfg_id,
    ST.sdk_id,
    BS.git_revision,
    BA.command,
    BC.label AS benchmark_label,
    BC.benchmark_mode,
    BC.counter_collection_mode,
    BC.counter_group_count AS counter_groups,
    BC.kernel_replay,
    ST.metric_name,
    ST.metric_unit,
    ST.count,
    ST.mean AS measured,
    ST.std_dev AS `+/-`,
    SG.mean AS single_group_measured,
    SG.mean * BC.counter_group_count AS application_replay_projected,
    CASE
        WHEN ST.mean > 0 THEN (SG.mean * BC.counter_group_count) / ST.mean
    END AS `speedup vs application replay`
FROM
    benchmark_statistics ST
    JOIN benchmark_config BC ON BC.id = ST.cfg_id
    JOIN benchmarked_sdk BS ON BS.id = ST.sdk_id
    JOIN benchmarked_app BA ON BA.id = ST.app_id
    LEFT JOIN single_group SG ON (
        SG.app_id = ST.app_id
        AND SG.sdk_id = ST.sdk_id
    )
WHERE
    ST.metric_name = "{{metric}}"
    AND BC.counter_group_count IS NOT NULL
    AND BC.counter_group_count > 0;
