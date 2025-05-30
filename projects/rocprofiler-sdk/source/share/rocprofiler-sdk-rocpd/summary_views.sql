--
-- Useful summary views
--
--
-- Sorted list of kernels which consume the most overall time
CREATE VIEW IF NOT EXISTS
    `top_kernels` AS
SELECT
    S.display_name AS name,
    COUNT(K.kernel_id) AS total_calls,
    SUM(K.end - K.start) / 1000 AS total_duration,
    (SUM(K.end - K.start) / COUNT(K.kernel_id)) / 1000 AS average,
    SUM(K.end - K.start) * 100.0 / (
        SELECT
            SUM(A.end - A.start)
        FROM
            `rocpd_kernel_dispatch` A
    ) AS percentage
FROM
    `rocpd_kernel_dispatch` K
    INNER JOIN `rocpd_info_kernel_symbol` S ON S.id = K.kernel_id
    AND S.guid = K.guid
GROUP BY
    name
ORDER BY
    total_duration DESC;

--
-- GPU utilization metrics including kernels and memory copy operations
CREATE VIEW IF NOT EXISTS
    `busy` AS
SELECT
    A.agent_id,
    AG.type,
    GpuTime,
    WallTime,
    GpuTime * 1.0 / WallTime AS Busy
FROM
    (
        SELECT
            agent_id,
            SUM(END - start) AS GpuTime
        FROM
            (
                SELECT
                    agent_id,
                    END,
                    start
                FROM
                    `rocpd_kernel_dispatch`
                UNION ALL
                SELECT
                    dst_agent_id AS agent_id,
                    END,
                    start
                FROM
                    `rocpd_memory_copy`
            )
        GROUP BY
            agent_id
    ) A
    INNER JOIN (
        SELECT
            MAX(END) - MIN(start) AS WallTime
        FROM
            (
                SELECT
                    END,
                    start
                FROM
                    `rocpd_kernel_dispatch`
                UNION ALL
                SELECT
                    END,
                    start
                FROM
                    `rocpd_memory_copy`
            )
    ) W ON 1 = 1
    INNER JOIN `rocpd_info_agent` AG ON AG.id = A.agent_id
    AND AG.guid = A.guid;

--
-- Overall performance summary including kernels and memory copy operations
CREATE VIEW
    `top` AS
SELECT
    name,
    COUNT(*) AS total_calls,
    SUM(duration) / 1000.0 AS total_duration,
    (SUM(duration) / COUNT(*)) / 1000.0 AS average,
    SUM(duration) * 100.0 / total_time AS percentage
FROM
    (
        -- Kernel operations
        SELECT
            ks.display_name AS name,
            (kd.end - kd.start) AS duration
        FROM
            `rocpd_kernel_dispatch` kd
            INNER JOIN `rocpd_info_kernel_symbol` ks ON kd.kernel_id = ks.id
            AND kd.guid = ks.guid
        UNION ALL
        -- Memory operations
        SELECT
            rs.string AS name,
            (END - start) AS duration
        FROM
            `rocpd_memory_copy` mc
            INNER JOIN `rocpd_string` rs ON rs.id = mc.name_id
            AND rs.guid = mc.guid
        UNION ALL
        -- Regions
        SELECT
            rs.string AS name,
            (END - start) AS duration
        FROM
            `rocpd_region` rr
            INNER JOIN `rocpd_string` rs ON rs.id = rr.name_id
            AND rs.guid = rr.guid
    ) operations
    CROSS JOIN (
        SELECT
            SUM(END - start) AS total_time
        FROM
            (
                SELECT
                    END,
                    start
                FROM
                    `rocpd_kernel_dispatch`
                UNION ALL
                SELECT
                    END,
                    start
                FROM
                    `rocpd_memory_copy`
                UNION ALL
                SELECT
                    END,
                    start
                FROM
                    `rocpd_region`
            )
    ) TOTAL
GROUP BY
    name
ORDER BY
    total_duration DESC;

--
-- summaries
--
--
-- Kernel summary by name
CREATE VIEW
    `kernel_summary` AS
WITH
    avg_data AS (
        SELECT
            name,
            AVG(duration) AS avg_duration
        FROM
            `kernels`
        GROUP BY
            name
    ),
    aggregated_data AS (
        SELECT
            K.name,
            COUNT(*) AS calls,
            SUM(K.duration) AS total_duration,
            CAST(SUM(K.duration * K.duration) AS REAL) AS sqr_duration,
            A.avg_duration AS average_duration,
            MIN(K.duration) AS min_duration,
            MAX(K.duration) AS max_duration,
            SUM((K.duration - A.avg_duration) * (K.duration - A.avg_duration)) / (COUNT(*) - 1) AS variance_duration,
            SQRT(SUM((K.duration - A.avg_duration) * (K.duration - A.avg_duration)) / (COUNT(*) - 1)) AS std_dev_duration
        FROM
            `kernels` K
            JOIN avg_data A ON K.name = A.name
        GROUP BY
            K.name
    ),
    total_duration AS (
        SELECT
            SUM(total_duration) AS grand_total_duration
        FROM
            aggregated_data
    )
SELECT
    AD.name AS name,
    AD.calls,
    AD.total_duration AS "DURATION (nsec)",
    AD.sqr_duration AS "SQR (nsec)",
    AD.average_duration AS "AVERAGE (nsec)",
    (CAST(AD.total_duration AS REAL) / TD.grand_total_duration) * 100 AS "PERCENT (INC)",
    AD.min_duration AS "MIN (nsec)",
    AD.max_duration AS "MAX (nsec)",
    AD.variance_duration AS "VARIANCE",
    AD.std_dev_duration AS "STD_DEV"
FROM
    aggregated_data AD
    CROSS JOIN total_duration TD;

--
-- Kernel summary by region name
CREATE VIEW
    `kernel_summary_region` AS
WITH
    avg_data AS (
        SELECT
            region,
            AVG(duration) AS avg_duration
        FROM
            `kernels`
        GROUP BY
            region
    ),
    aggregated_data AS (
        SELECT
            K.region AS name,
            COUNT(*) AS calls,
            SUM(K.duration) AS total_duration,
            CAST(SUM(K.duration * K.duration) AS REAL) AS sqr_duration,
            A.avg_duration AS average_duration,
            MIN(K.duration) AS min_duration,
            MAX(K.duration) AS max_duration,
            SUM((K.duration - A.avg_duration) * (K.duration - A.avg_duration)) / (COUNT(*) - 1) AS variance_duration,
            SQRT(SUM((K.duration - A.avg_duration) * (K.duration - A.avg_duration)) / (COUNT(*) - 1)) AS std_dev_duration
        FROM
            `kernels` K
            JOIN avg_data A ON K.region = A.region
        GROUP BY
            K.region
    ),
    total_duration AS (
        SELECT
            SUM(total_duration) AS grand_total_duration
        FROM
            aggregated_data
    )
SELECT
    AD.name AS name,
    AD.calls,
    AD.total_duration AS "DURATION (nsec)",
    AD.sqr_duration AS "SQR (nsec)",
    AD.average_duration AS "AVERAGE (nsec)",
    (CAST(AD.total_duration AS REAL) / TD.grand_total_duration) * 100 AS "PERCENT (INC)",
    AD.min_duration AS "MIN (nsec)",
    AD.max_duration AS "MAX (nsec)",
    AD.variance_duration AS "VARIANCE",
    AD.std_dev_duration AS "STD_DEV"
FROM
    aggregated_data AD
    CROSS JOIN total_duration TD;

--
-- Memory copy summary
CREATE VIEW
    `memory_copy_summary` AS
WITH
    avg_data AS (
        SELECT
            name,
            AVG(duration) AS avg_duration
        FROM
            `memory_copies`
        GROUP BY
            name
    ),
    aggregated_data AS (
        SELECT
            MC.name,
            COUNT(*) AS calls,
            SUM(MC.duration) AS total_duration,
            CAST(SUM(MC.duration * MC.duration) AS REAL) AS sqr_duration,
            A.avg_duration AS average_duration,
            MIN(MC.duration) AS min_duration,
            MAX(MC.duration) AS max_duration,
            SUM((MC.duration - A.avg_duration) * (MC.duration - A.avg_duration)) / (COUNT(*) - 1) AS variance_duration,
            SQRT(SUM((MC.duration - A.avg_duration) * (MC.duration - A.avg_duration)) / (COUNT(*) - 1)) AS std_dev_duration
        FROM
            `memory_copies` MC
            JOIN avg_data A ON MC.name = A.name
        GROUP BY
            MC.name
    ),
    total_duration AS (
        SELECT
            SUM(total_duration) AS grand_total_duration
        FROM
            aggregated_data
    )
SELECT
    AD.name AS name,
    AD.calls,
    AD.total_duration AS "DURATION (nsec)",
    AD.sqr_duration AS "SQR (nsec)",
    AD.average_duration AS "AVERAGE (nsec)",
    (CAST(AD.total_duration AS REAL) / TD.grand_total_duration) * 100 AS "PERCENT (INC)",
    AD.min_duration AS "MIN (nsec)",
    AD.max_duration AS "MAX (nsec)",
    AD.variance_duration AS "VARIANCE",
    AD.std_dev_duration AS "STD_DEV"
FROM
    aggregated_data AD
    CROSS JOIN total_duration TD;

--
-- Memory allocation summary
CREATE VIEW
    `memory_allocation_summary` AS
WITH
    avg_data AS (
        SELECT
            type AS name,
            AVG(duration) AS avg_duration
        FROM
            `memory_allocations`
        GROUP BY
            type
    ),
    aggregated_data AS (
        SELECT
            MA.type AS name,
            COUNT(*) AS calls,
            SUM(MA.duration) AS total_duration,
            CAST(SUM(MA.duration * MA.duration) AS REAL) AS sqr_duration,
            A.avg_duration AS average_duration,
            MIN(MA.duration) AS min_duration,
            MAX(MA.duration) AS max_duration,
            SUM((MA.duration - A.avg_duration) * (MA.duration - A.avg_duration)) / (COUNT(*) - 1) AS variance_duration,
            SQRT(SUM((MA.duration - A.avg_duration) * (MA.duration - A.avg_duration)) / (COUNT(*) - 1)) AS std_dev_duration
        FROM
            `memory_allocations` MA
            JOIN avg_data A ON MA.type = A.name
        GROUP BY
            MA.type
    ),
    total_duration AS (
        SELECT
            SUM(total_duration) AS grand_total_duration
        FROM
            aggregated_data
    )
SELECT
    'MEMORY_ALLOCATION_' || AD.name AS name,
    AD.calls,
    AD.total_duration AS "DURATION (nsec)",
    AD.sqr_duration AS "SQR (nsec)",
    AD.average_duration AS "AVERAGE (nsec)",
    (CAST(AD.total_duration AS REAL) / TD.grand_total_duration) * 100 AS "PERCENT (INC)",
    AD.min_duration AS "MIN (nsec)",
    AD.max_duration AS "MAX (nsec)",
    AD.variance_duration AS "VARIANCE",
    AD.std_dev_duration AS "STD_DEV"
FROM
    aggregated_data AD
    CROSS JOIN total_duration TD;

--
-- Scratch Memory summary
CREATE VIEW
    `scratch_memory_summary` AS
WITH
    avg_data AS (
        SELECT
            operation AS name,
            AVG(END - start) AS avg_duration
        FROM
            `scratch_memory`
        GROUP BY
            operation
    ),
    aggregated_data AS (
        SELECT
            SM.operation AS name,
            COUNT(*) AS calls,
            SUM(SM.end - SM.start) AS total_duration,
            CAST(SUM((SM.end - SM.start) * (SM.end - SM.start)) AS REAL) AS sqr_duration,
            A.avg_duration AS average_duration,
            MIN(SM.end - SM.start) AS min_duration,
            MAX(SM.end - SM.start) AS max_duration,
            SUM((SM.end - SM.start - A.avg_duration) * (SM.end - SM.start - A.avg_duration)) / (COUNT(*) - 1) AS variance_duration,
            SQRT(
                SUM((SM.end - SM.start - A.avg_duration) * (SM.end - SM.start - A.avg_duration)) / (COUNT(*) - 1)
            ) AS std_dev_duration
        FROM
            `scratch_memory` SM
            JOIN avg_data A ON SM.operation = A.name
        GROUP BY
            SM.operation
    ),
    total_duration AS (
        SELECT
            SUM(total_duration) AS grand_total_duration
        FROM
            aggregated_data
    )
SELECT
    AD.name AS name,
    AD.calls,
    AD.total_duration AS "DURATION (nsec)",
    AD.sqr_duration AS "SQR (nsec)",
    AD.average_duration AS "AVERAGE (nsec)",
    (CAST(AD.total_duration AS REAL) / TD.grand_total_duration) * 100 AS "PERCENT (INC)",
    AD.min_duration AS "MIN (nsec)",
    AD.max_duration AS "MAX (nsec)",
    AD.variance_duration AS "VARIANCE",
    AD.std_dev_duration AS "STD_DEV"
FROM
    aggregated_data AD
    CROSS JOIN total_duration TD
ORDER BY
    AD.total_duration DESC;

--
-- HIP summary
CREATE VIEW
    `hip_api_summary` AS
WITH
    avg_data AS (
        SELECT
            name,
            AVG(duration) AS avg_duration
        FROM
            `regions`
        WHERE
            category LIKE 'HIP_%'
        GROUP BY
            name
    ),
    aggregated_data AS (
        SELECT
            R.name,
            COUNT(*) AS calls,
            SUM(R.duration) AS total_duration,
            CAST(SUM(R.duration * R.duration) AS REAL) AS sqr_duration,
            A.avg_duration AS average_duration,
            MIN(R.duration) AS min_duration,
            MAX(R.duration) AS max_duration,
            SUM((R.duration - A.avg_duration) * (R.duration - A.avg_duration)) / (COUNT(*) - 1) AS variance_duration,
            SQRT(SUM((R.duration - A.avg_duration) * (R.duration - A.avg_duration)) / (COUNT(*) - 1)) AS std_dev_duration
        FROM
            `regions` R
            JOIN avg_data A ON R.name = A.name
        WHERE
            R.category LIKE 'HIP_%'
        GROUP BY
            R.name
    ),
    total_duration AS (
        SELECT
            SUM(total_duration) AS grand_total_duration
        FROM
            aggregated_data
    )
SELECT
    AD.name AS name,
    AD.calls,
    AD.total_duration AS "DURATION (nsec)",
    AD.sqr_duration AS "SQR (nsec)",
    AD.average_duration AS "AVERAGE (nsec)",
    (CAST(AD.total_duration AS REAL) / TD.grand_total_duration) * 100 AS "PERCENT (INC)",
    AD.min_duration AS "MIN (nsec)",
    AD.max_duration AS "MAX (nsec)",
    AD.variance_duration AS "VARIANCE",
    AD.std_dev_duration AS "STD_DEV"
FROM
    aggregated_data AD
    CROSS JOIN total_duration TD;

--
-- HSA summary
CREATE VIEW
    `hsa_api_summary` AS
WITH
    avg_data AS (
        SELECT
            name,
            AVG(duration) AS avg_duration
        FROM
            `regions`
        WHERE
            category LIKE 'HSA_%'
        GROUP BY
            name
    ),
    aggregated_data AS (
        SELECT
            R.name,
            COUNT(*) AS calls,
            SUM(R.duration) AS total_duration,
            CAST(SUM(R.duration * R.duration) AS REAL) AS sqr_duration,
            A.avg_duration AS average_duration,
            MIN(R.duration) AS min_duration,
            MAX(R.duration) AS max_duration,
            SUM((R.duration - A.avg_duration) * (R.duration - A.avg_duration)) / (COUNT(*) - 1) AS variance_duration,
            SQRT(SUM((R.duration - A.avg_duration) * (R.duration - A.avg_duration)) / (COUNT(*) - 1)) AS std_dev_duration
        FROM
            `regions` R
            JOIN avg_data A ON R.name = A.name
        WHERE
            R.category LIKE 'HSA_%'
        GROUP BY
            R.name
    ),
    total_duration AS (
        SELECT
            SUM(total_duration) AS grand_total_duration
        FROM
            aggregated_data
    )
SELECT
    AD.name AS name,
    AD.calls,
    AD.total_duration AS "DURATION (nsec)",
    AD.sqr_duration AS "SQR (nsec)",
    AD.average_duration AS "AVERAGE (nsec)",
    (CAST(AD.total_duration AS REAL) / TD.grand_total_duration) * 100 AS "PERCENT (INC)",
    AD.min_duration AS "MIN (nsec)",
    AD.max_duration AS "MAX (nsec)",
    AD.variance_duration AS "VARIANCE",
    AD.std_dev_duration AS "STD_DEV"
FROM
    aggregated_data AD
    CROSS JOIN total_duration TD;

--
-- RCCL API summary
CREATE VIEW
    `rccl_summary` AS
WITH
    avg_data AS (
        SELECT
            name,
            AVG(duration) AS avg_duration
        FROM
            `regions`
        WHERE
            category LIKE 'RCCL_%'
        GROUP BY
            name
    ),
    aggregated_data AS (
        SELECT
            R.name,
            COUNT(*) AS calls,
            SUM(R.duration) AS total_duration,
            CAST(SUM(R.duration * R.duration) AS REAL) AS sqr_duration,
            A.avg_duration AS average_duration,
            MIN(R.duration) AS min_duration,
            MAX(R.duration) AS max_duration,
            SUM((R.duration - A.avg_duration) * (R.duration - A.avg_duration)) / (COUNT(*) - 1) AS variance_duration,
            SQRT(SUM((R.duration - A.avg_duration) * (R.duration - A.avg_duration)) / (COUNT(*) - 1)) AS std_dev_duration
        FROM
            `regions` R
            JOIN avg_data A ON R.name = A.name
        WHERE
            R.category LIKE 'RCCL_%'
        GROUP BY
            R.name
    ),
    total_duration AS (
        SELECT
            SUM(total_duration) AS grand_total_duration
        FROM
            aggregated_data
    )
SELECT
    AD.name AS name,
    AD.calls,
    AD.total_duration AS "DURATION (nsec)",
    AD.sqr_duration AS "SQR (nsec)",
    AD.average_duration AS "AVERAGE (nsec)",
    (CAST(AD.total_duration AS REAL) / TD.grand_total_duration) * 100 AS "PERCENT (INC)",
    AD.min_duration AS "MIN (nsec)",
    AD.max_duration AS "MAX (nsec)",
    AD.variance_duration AS "VARIANCE",
    AD.std_dev_duration AS "STD_DEV"
FROM
    aggregated_data AD
    CROSS JOIN total_duration TD
ORDER BY
    AD.total_duration DESC;

--
-- ROCJPEG API summary
CREATE VIEW
    `rocjpeg_summary` AS
WITH
    avg_data AS (
        SELECT
            name,
            AVG(duration) AS avg_duration
        FROM
            `regions`
        WHERE
            category LIKE 'ROCJPEG_%'
        GROUP BY
            name
    ),
    aggregated_data AS (
        SELECT
            R.name,
            COUNT(*) AS calls,
            SUM(R.duration) AS total_duration,
            CAST(SUM(R.duration * R.duration) AS REAL) AS sqr_duration,
            A.avg_duration AS average_duration,
            MIN(R.duration) AS min_duration,
            MAX(R.duration) AS max_duration,
            SUM((R.duration - A.avg_duration) * (R.duration - A.avg_duration)) / (COUNT(*) - 1) AS variance_duration,
            SQRT(SUM((R.duration - A.avg_duration) * (R.duration - A.avg_duration)) / (COUNT(*) - 1)) AS std_dev_duration
        FROM
            `regions` R
            JOIN avg_data A ON R.name = A.name
        WHERE
            R.category LIKE 'ROCJPEG_%'
        GROUP BY
            R.name
    ),
    total_duration AS (
        SELECT
            SUM(total_duration) AS grand_total_duration
        FROM
            aggregated_data
    )
SELECT
    AD.name AS name,
    AD.calls,
    AD.total_duration AS "DURATION (nsec)",
    AD.sqr_duration AS "SQR (nsec)",
    AD.average_duration AS "AVERAGE (nsec)",
    (CAST(AD.total_duration AS REAL) / TD.grand_total_duration) * 100 AS "PERCENT (INC)",
    AD.min_duration AS "MIN (nsec)",
    AD.max_duration AS "MAX (nsec)",
    AD.variance_duration AS "VARIANCE",
    AD.std_dev_duration AS "STD_DEV"
FROM
    aggregated_data AD
    CROSS JOIN total_duration TD
ORDER BY
    AD.total_duration DESC;

--
-- ROCDECODE API summary
CREATE VIEW
    `rocdecode_summary` AS
WITH
    avg_data AS (
        SELECT
            name,
            AVG(duration) AS avg_duration
        FROM
            `regions`
        WHERE
            category LIKE 'ROCDECODE_%'
        GROUP BY
            name
    ),
    aggregated_data AS (
        SELECT
            R.name,
            COUNT(*) AS calls,
            SUM(R.duration) AS total_duration,
            CAST(SUM(R.duration * R.duration) AS REAL) AS sqr_duration,
            A.avg_duration AS average_duration,
            MIN(R.duration) AS min_duration,
            MAX(R.duration) AS max_duration,
            SUM((R.duration - A.avg_duration) * (R.duration - A.avg_duration)) / (COUNT(*) - 1) AS variance_duration,
            SQRT(SUM((R.duration - A.avg_duration) * (R.duration - A.avg_duration)) / (COUNT(*) - 1)) AS std_dev_duration
        FROM
            `regions` R
            JOIN avg_data A ON R.name = A.name
        WHERE
            R.category LIKE 'ROCDECODE_%'
        GROUP BY
            R.name
    ),
    total_duration AS (
        SELECT
            SUM(total_duration) AS grand_total_duration
        FROM
            aggregated_data
    )
SELECT
    AD.name AS name,
    AD.calls,
    AD.total_duration AS "DURATION (nsec)",
    AD.sqr_duration AS "SQR (nsec)",
    AD.average_duration AS "AVERAGE (nsec)",
    (CAST(AD.total_duration AS REAL) / TD.grand_total_duration) * 100 AS "PERCENT (INC)",
    AD.min_duration AS "MIN (nsec)",
    AD.max_duration AS "MAX (nsec)",
    AD.variance_duration AS "VARIANCE",
    AD.std_dev_duration AS "STD_DEV"
FROM
    aggregated_data AD
    CROSS JOIN total_duration TD
ORDER BY
    AD.total_duration DESC;

--
-- Domain summary
CREATE VIEW
    `domain_summary` AS
WITH
    kernel_times AS (
        WITH
            kernel_avg AS (
                SELECT
                    AVG(duration) AS avg_duration
                FROM
                    `kernels`
            )
        SELECT
            'KERNEL_DISPATCH' AS domain,
            COUNT(*) AS calls,
            SUM(duration) AS total_duration,
            CAST(SUM(duration * duration) AS REAL) AS sqr_duration,
            (
                SELECT
                    avg_duration
                FROM
                    kernel_avg
            ) AS avg_duration,
            MIN(duration) AS min_duration,
            MAX(duration) AS max_duration,
            SUM(
                (
                    duration - (
                        SELECT
                            avg_duration
                        FROM
                            kernel_avg
                    )
                ) * (
                    duration - (
                        SELECT
                            avg_duration
                        FROM
                            kernel_avg
                    )
                )
            ) / (COUNT(*) - 1) AS variance_duration,
            SQRT(
                SUM(
                    (
                        duration - (
                            SELECT
                                avg_duration
                            FROM
                                kernel_avg
                        )
                    ) * (
                        duration - (
                            SELECT
                                avg_duration
                            FROM
                                kernel_avg
                        )
                    )
                ) / (COUNT(*) - 1)
            ) AS std_dev_duration
        FROM
            `kernels`
    ),
    mcopy_times AS (
        WITH
            mcopy_avg AS (
                SELECT
                    AVG(duration) AS avg_duration
                FROM
                    `memory_copies`
            )
        SELECT
            'MEMORY_COPY' AS domain,
            COUNT(*) AS calls,
            SUM(duration) AS total_duration,
            CAST(SUM(duration * duration) AS REAL) AS sqr_duration,
            (
                SELECT
                    avg_duration
                FROM
                    mcopy_avg
            ) AS avg_duration,
            MIN(duration) AS min_duration,
            MAX(duration) AS max_duration,
            SUM(
                (
                    duration - (
                        SELECT
                            avg_duration
                        FROM
                            mcopy_avg
                    )
                ) * (
                    duration - (
                        SELECT
                            avg_duration
                        FROM
                            mcopy_avg
                    )
                )
            ) / (COUNT(*) - 1) AS variance_duration,
            SQRT(
                SUM(
                    (
                        duration - (
                            SELECT
                                avg_duration
                            FROM
                                mcopy_avg
                        )
                    ) * (
                        duration - (
                            SELECT
                                avg_duration
                            FROM
                                mcopy_avg
                        )
                    )
                ) / (COUNT(*) - 1)
            ) AS std_dev_duration
        FROM
            `memory_copies`
    ),
    malloc_times AS (
        WITH
            malloc_avg AS (
                SELECT
                    AVG(duration) AS avg_duration
                FROM
                    `memory_allocations`
            )
        SELECT
            'MEMORY_ALLOCATION' AS domain,
            COUNT(*) AS calls,
            SUM(duration) AS total_duration,
            CAST(SUM(duration * duration) AS REAL) AS sqr_duration,
            (
                SELECT
                    avg_duration
                FROM
                    malloc_avg
            ) AS avg_duration,
            MIN(duration) AS min_duration,
            MAX(duration) AS max_duration,
            SUM(
                (
                    duration - (
                        SELECT
                            avg_duration
                        FROM
                            malloc_avg
                    )
                ) * (
                    duration - (
                        SELECT
                            avg_duration
                        FROM
                            malloc_avg
                    )
                )
            ) / (COUNT(*) - 1) AS variance_duration,
            SQRT(
                SUM(
                    (
                        duration - (
                            SELECT
                                avg_duration
                            FROM
                                malloc_avg
                        )
                    ) * (
                        duration - (
                            SELECT
                                avg_duration
                            FROM
                                malloc_avg
                        )
                    )
                ) / (COUNT(*) - 1)
            ) AS std_dev_duration
        FROM
            `memory_allocations`
    ),
    scratch_times AS (
        WITH
            scratch_avg AS (
                SELECT
                    AVG(END - start) AS avg_duration
                FROM
                    `scratch_memory`
            )
        SELECT
            'SCRATCH_MEMORY' AS domain,
            COUNT(*) AS calls,
            SUM(END - start) AS total_duration,
            CAST(SUM((END - start) * (END - start)) AS REAL) AS sqr_duration,
            (
                SELECT
                    avg_duration
                FROM
                    scratch_avg
            ) AS avg_duration,
            MIN(END - start) AS min_duration,
            MAX(END - start) AS max_duration,
            SUM(
                (
                    END - start - (
                        SELECT
                            avg_duration
                        FROM
                            scratch_avg
                    )
                ) * (
                    END - start - (
                        SELECT
                            avg_duration
                        FROM
                            scratch_avg
                    )
                )
            ) / (COUNT(*) - 1) AS variance_duration,
            SQRT(
                SUM(
                    (
                        END - start - (
                            SELECT
                                avg_duration
                            FROM
                                scratch_avg
                        )
                    ) * (
                        END - start - (
                            SELECT
                                avg_duration
                            FROM
                                scratch_avg
                        )
                    )
                ) / (COUNT(*) - 1)
            ) AS std_dev_duration
        FROM
            `scratch_memory`
    ),
    hip_api_times AS (
        WITH
            hip_avg AS (
                SELECT
                    AVG(duration) AS avg_duration
                FROM
                    `regions`
                WHERE
                    category LIKE 'HIP_%'
            )
        SELECT
            'HIP_API' AS domain,
            COUNT(*) AS calls,
            SUM(duration) AS total_duration,
            CAST(SUM(duration * duration) AS REAL) AS sqr_duration,
            (
                SELECT
                    avg_duration
                FROM
                    hip_avg
            ) AS avg_duration,
            MIN(duration) AS min_duration,
            MAX(duration) AS max_duration,
            SUM(
                (
                    duration - (
                        SELECT
                            avg_duration
                        FROM
                            hip_avg
                    )
                ) * (
                    duration - (
                        SELECT
                            avg_duration
                        FROM
                            hip_avg
                    )
                )
            ) / (COUNT(*) - 1) AS variance_duration,
            SQRT(
                SUM(
                    (
                        duration - (
                            SELECT
                                avg_duration
                            FROM
                                hip_avg
                        )
                    ) * (
                        duration - (
                            SELECT
                                avg_duration
                            FROM
                                hip_avg
                        )
                    )
                ) / (COUNT(*) - 1)
            ) AS std_dev_duration
        FROM
            `regions`
        WHERE
            category LIKE 'HIP_%'
    ),
    hsa_api_times AS (
        WITH
            hsa_avg AS (
                SELECT
                    AVG(duration) AS avg_duration
                FROM
                    `regions`
                WHERE
                    category LIKE 'HSA_%'
            )
        SELECT
            'HSA_API' AS domain,
            COUNT(*) AS calls,
            SUM(duration) AS total_duration,
            CAST(SUM(duration * duration) AS REAL) AS sqr_duration,
            (
                SELECT
                    avg_duration
                FROM
                    hsa_avg
            ) AS avg_duration,
            MIN(duration) AS min_duration,
            MAX(duration) AS max_duration,
            SUM(
                (
                    duration - (
                        SELECT
                            avg_duration
                        FROM
                            hsa_avg
                    )
                ) * (
                    duration - (
                        SELECT
                            avg_duration
                        FROM
                            hsa_avg
                    )
                )
            ) / (COUNT(*) - 1) AS variance_duration,
            SQRT(
                SUM(
                    (
                        duration - (
                            SELECT
                                avg_duration
                            FROM
                                hsa_avg
                        )
                    ) * (
                        duration - (
                            SELECT
                                avg_duration
                            FROM
                                hsa_avg
                        )
                    )
                ) / (COUNT(*) - 1)
            ) AS std_dev_duration
        FROM
            `regions`
        WHERE
            category LIKE 'HSA_%'
    ),
    marker_times AS (
        WITH
            marker_avg AS (
                SELECT
                    AVG(duration) AS avg_duration
                FROM
                    `markers`
            )
        SELECT
            'MARKER_API' AS domain,
            COUNT(*) AS calls,
            SUM(duration) AS total_duration,
            CAST(SUM(duration * duration) AS REAL) AS sqr_duration,
            (
                SELECT
                    avg_duration
                FROM
                    marker_avg
            ) AS avg_duration,
            MIN(duration) AS min_duration,
            MAX(duration) AS max_duration,
            SUM(
                (
                    duration - (
                        SELECT
                            avg_duration
                        FROM
                            marker_avg
                    )
                ) * (
                    duration - (
                        SELECT
                            avg_duration
                        FROM
                            marker_avg
                    )
                )
            ) / (COUNT(*) - 1) AS variance_duration,
            SQRT(
                SUM(
                    (
                        duration - (
                            SELECT
                                avg_duration
                            FROM
                                marker_avg
                        )
                    ) * (
                        duration - (
                            SELECT
                                avg_duration
                            FROM
                                marker_avg
                        )
                    )
                ) / (COUNT(*) - 1)
            ) AS std_dev_duration
        FROM
            `markers`
    ),
    rccl_times AS (
        WITH
            rccl_avg AS (
                SELECT
                    AVG(duration) AS avg_duration
                FROM
                    `rccl`
            )
        SELECT
            'RCCL_API' AS domain,
            COUNT(*) AS calls,
            SUM(duration) AS total_duration,
            CAST(SUM(duration * duration) AS REAL) AS sqr_duration,
            (
                SELECT
                    avg_duration
                FROM
                    rccl_avg
            ) AS avg_duration,
            MIN(duration) AS min_duration,
            MAX(duration) AS max_duration,
            SUM(
                (
                    duration - (
                        SELECT
                            avg_duration
                        FROM
                            rccl_avg
                    )
                ) * (
                    duration - (
                        SELECT
                            avg_duration
                        FROM
                            rccl_avg
                    )
                )
            ) / (COUNT(*) - 1) AS variance_duration,
            SQRT(
                SUM(
                    (
                        duration - (
                            SELECT
                                avg_duration
                            FROM
                                rccl_avg
                        )
                    ) * (
                        duration - (
                            SELECT
                                avg_duration
                            FROM
                                rccl_avg
                        )
                    )
                ) / (COUNT(*) - 1)
            ) AS std_dev_duration
        FROM
            `rccl`
    ),
    rocdecode_times AS (
        WITH
            rocdecode_avg AS (
                SELECT
                    AVG(duration) AS avg_duration
                FROM
                    `rocdecode`
            )
        SELECT
            'ROCDECODE_API' AS domain,
            COUNT(*) AS calls,
            SUM(duration) AS total_duration,
            CAST(SUM(duration * duration) AS REAL) AS sqr_duration,
            (
                SELECT
                    avg_duration
                FROM
                    rocdecode_avg
            ) AS avg_duration,
            MIN(duration) AS min_duration,
            MAX(duration) AS max_duration,
            SUM(
                (
                    duration - (
                        SELECT
                            avg_duration
                        FROM
                            rocdecode_avg
                    )
                ) * (
                    duration - (
                        SELECT
                            avg_duration
                        FROM
                            rocdecode_avg
                    )
                )
            ) / (COUNT(*) - 1) AS variance_duration,
            SQRT(
                SUM(
                    (
                        duration - (
                            SELECT
                                avg_duration
                            FROM
                                rocdecode_avg
                        )
                    ) * (
                        duration - (
                            SELECT
                                avg_duration
                            FROM
                                rocdecode_avg
                        )
                    )
                ) / (COUNT(*) - 1)
            ) AS std_dev_duration
        FROM
            `rocdecode`
    ),
    rocjpeg_times AS (
        WITH
            rocjpeg_avg AS (
                SELECT
                    AVG(duration) AS avg_duration
                FROM
                    `rocjpeg`
            )
        SELECT
            'ROCJPEG_API' AS domain,
            COUNT(*) AS calls,
            SUM(duration) AS total_duration,
            CAST(SUM(duration * duration) AS REAL) AS sqr_duration,
            (
                SELECT
                    avg_duration
                FROM
                    rocjpeg_avg
            ) AS avg_duration,
            MIN(duration) AS min_duration,
            MAX(duration) AS max_duration,
            SUM(
                (
                    duration - (
                        SELECT
                            avg_duration
                        FROM
                            rocjpeg_avg
                    )
                ) * (
                    duration - (
                        SELECT
                            avg_duration
                        FROM
                            rocjpeg_avg
                    )
                )
            ) / (COUNT(*) - 1) AS variance_duration,
            SQRT(
                SUM(
                    (
                        duration - (
                            SELECT
                                avg_duration
                            FROM
                                rocjpeg_avg
                        )
                    ) * (
                        duration - (
                            SELECT
                                avg_duration
                            FROM
                                rocjpeg_avg
                        )
                    )
                ) / (COUNT(*) - 1)
            ) AS std_dev_duration
        FROM
            `rocjpeg`
    ),
    all_domains AS (
        SELECT
            *
        FROM
            kernel_times
        UNION ALL
        SELECT
            *
        FROM
            mcopy_times
        UNION ALL
        SELECT
            *
        FROM
            malloc_times
        UNION ALL
        SELECT
            *
        FROM
            scratch_times
        UNION ALL
        SELECT
            *
        FROM
            hip_api_times
        UNION ALL
        SELECT
            *
        FROM
            hsa_api_times
        UNION ALL
        SELECT
            *
        FROM
            marker_times
        UNION ALL
        SELECT
            *
        FROM
            rccl_times
        UNION ALL
        SELECT
            *
        FROM
            rocdecode_times
        UNION ALL
        SELECT
            *
        FROM
            rocjpeg_times
    ),
    total_duration AS (
        SELECT
            SUM(total_duration) AS grand_total_duration
        FROM
            all_domains
    )
SELECT
    AD.domain AS name,
    AD.calls,
    AD.total_duration AS "DURATION (nsec)",
    AD.sqr_duration AS "SQR (nsec)",
    AD.avg_duration AS "AVERAGE (nsec)",
    (CAST(AD.total_duration AS REAL) / TD.grand_total_duration) * 100 AS "PERCENT (INC)",
    AD.min_duration AS "MIN (nsec)",
    AD.max_duration AS "MAX (nsec)",
    AD.variance_duration AS "VARIANCE",
    AD.std_dev_duration AS "STD_DEV"
FROM
    all_domains AD
    CROSS JOIN total_duration TD
ORDER BY
    AD.total_duration DESC;
