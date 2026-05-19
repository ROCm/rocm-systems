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
    SUM(K.end - K.start) / 1000.0 AS total_duration,
    (SUM(K.end - K.start) / COUNT(K.kernel_id)) / 1000.0 AS average,
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
            guid,
            SUM(END - start) AS GpuTime
        FROM
            (
                SELECT
                    agent_id,
                    guid,
                    END,
                    start
                FROM
                    `rocpd_kernel_dispatch`
                UNION ALL
                SELECT
                    dst_agent_id AS agent_id,
                    guid,
                    END,
                    start
                FROM
                    `rocpd_memory_copy`
            )
        GROUP BY
            agent_id,
            guid
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
