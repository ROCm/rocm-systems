--
-- Useful summary views
--
--
-- Sorted list of kernels which consume the most overall time
CREATE VIEW IF NOT EXISTS
    `top_kernels` AS
SELECT
    K.name,
    COUNT(K.kernel_id) AS total_calls,
    SUM(K.end - K.start) / 1000.0 AS total_duration,
    (SUM(K.end - K.start) / COUNT(K.kernel_id)) / 1000.0 AS average,
    SUM(K.end - K.start) * 100.0 / (
        SELECT
            SUM(A.end - A.start)
        FROM
            `kernels` A
    ) AS percentage
FROM
    `kernels` K
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
            `guid`,
            SUM(`end` - `start`) AS GpuTime
        FROM
            (
                SELECT
                    agent_id,
                    `guid`,
                    `end`,
                    `start`
                FROM
                    `kernels`
                UNION ALL
                SELECT
                    dst_agent_id AS agent_id,
                    `guid`,
                    `end`,
                    `start`
                FROM
                    `memory_copies`
            )
        GROUP BY
            agent_id,
            `guid`
    ) A
    INNER JOIN (
        SELECT
            MAX(`end`) - MIN(`start`) AS WallTime
        FROM
            (
                SELECT
                    `end`,
                    `start`
                FROM
                    `kernels`
                UNION ALL
                SELECT
                    `end`,
                    `start`
                FROM
                    `memory_copies`
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
            K.name,
            K.duration
        FROM
            `kernels` K
        UNION ALL
        -- Memory operations
        SELECT
            MC.name,
            MC.duration
        FROM
            `memory_copies` MC
        UNION ALL
        -- Regions
        SELECT
            R.name,
            R.duration
        FROM
            `regions` R
    ) operations
    CROSS JOIN (
        SELECT
            SUM(`end` - `start`) AS total_time
        FROM
            (
                SELECT
                    `end`,
                    `start`
                FROM
                    `kernels`
                UNION ALL
                SELECT
                    `end`,
                    `start`
                FROM
                    `memory_copies`
                UNION ALL
                SELECT
                    `end`,
                    `start`
                FROM
                    `regions`
            )
    ) TOTAL
GROUP BY
    name
ORDER BY
    total_duration DESC;
