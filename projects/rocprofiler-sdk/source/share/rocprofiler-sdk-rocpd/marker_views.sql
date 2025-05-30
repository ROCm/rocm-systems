--
-- Views related to markers
--
--
--
CREATE VIEW
    `range_markers` AS
SELECT
    E.id,
    E.guid,
    ST.string AS category,
    JSON_EXTRACT(E.extdata, '$.message') AS name,
    R.start,
    R.end,
    (R.end - R.start) AS duration,
    E.stack_id,
    E.parent_stack_id,
    E.correlation_id,
    E.call_stack,
    E.line_info,
    R.pid,
    R.tid,
    R.nid,
    ST2.string AS operation
FROM
    `rocpd_event` E
    INNER JOIN `rocpd_string` ST ON ST.id = E.category_id
    AND ST.guid = E.guid
    INNER JOIN `rocpd_region` R ON R.event_id = E.id
    AND R.guid = E.guid
    INNER JOIN `rocpd_string` ST2 ON ST2.id = R.name_id
    AND ST2.guid = R.guid
WHERE
    ST.string LIKE '%MARKER%'
    AND JSON_VALID(E.extdata);

--
--
CREATE VIEW
    `single_markers` AS
SELECT
    E.id,
    E.guid,
    ST.string AS category,
    JSON_EXTRACT(E.extdata, '$.message') AS name,
    S.timestamp AS timestamp,
    E.stack_id,
    E.parent_stack_id,
    E.correlation_id,
    E.call_stack,
    E.line_info,
    T.pid AS pid,
    T.tid AS tid,
    T.nid
FROM
    `rocpd_event` E
    INNER JOIN `rocpd_string` ST ON ST.id = E.category_id
    AND ST.guid = E.guid
    INNER JOIN `rocpd_sample` S ON S.event_id = E.id
    AND S.guid = E.guid
    INNER JOIN `rocpd_track` T ON T.id = S.track_id
    AND S.guid = T.guid
WHERE
    ST.string LIKE '%MARKER%';

--
--
CREATE VIEW
    `markers` AS
SELECT
    *
FROM
    `range_markers`
UNION
SELECT
    SM.id,
    SM.guid,
    SM.category,
    SM.name,
    SM.timestamp AS start,
    SM.timestamp AS end,
    0 AS duration,
    SM.stack_id,
    SM.parent_stack_id,
    SM.correlation_id,
    SM.call_stack,
    SM.line_info,
    SM.pid,
    SM.tid,
    SM.nid,
    'roctxMarkA' AS operation
FROM
    `single_markers` SM;

--
--
CREATE VIEW
    `range_marker_summary` AS
WITH
    avg_data AS (
        SELECT
            name,
            AVG(duration) AS avg_duration
        FROM
            `range_markers`
        GROUP BY
            name
    ),
    aggregated_data AS (
        SELECT
            RM.name,
            COUNT(*) AS calls,
            SUM(RM.duration) AS total_duration,
            A.avg_duration AS average_duration,
            MIN(RM.duration) AS min_duration,
            MAX(RM.duration) AS max_duration,
            SQRT(AVG((RM.duration - A.avg_duration) * (RM.duration - A.avg_duration))) AS std_dev_duration
        FROM
            `range_markers` RM
            JOIN avg_data A ON RM.name = A.name
        GROUP BY
            RM.name
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
    AD.total_duration AS `DURATION (nsec)`,
    AD.average_duration AS `AVERAGE (nsec)`,
    (CAST(AD.total_duration AS REAL) / TD.grand_total_duration) * 100 AS `PERCENT (INC)`,
    AD.min_duration AS `MIN (nsec)`,
    AD.max_duration AS `MAX (nsec)`,
    AD.std_dev_duration AS `STD_DEV`
FROM
    aggregated_data AD
    CROSS JOIN total_duration TD;

--
--
CREATE VIEW
    `single_marker_summary` AS
SELECT
    SM.name,
    COUNT(*) AS calls
FROM
    `single_markers` SM
GROUP BY
    SM.name;

--
-- Markers summary
CREATE VIEW
    `marker_summary` AS
WITH
    all_markers AS (
        SELECT
            name,
            duration
        FROM
            `range_markers`
        UNION ALL
        SELECT
            name,
            0 AS duration
        FROM
            `single_markers`
    ),
    avg_data AS (
        SELECT
            name,
            AVG(duration) AS avg_duration
        FROM
            all_markers
        GROUP BY
            name
    ),
    aggregated_data AS (
        SELECT
            M.name,
            COUNT(*) AS calls,
            SUM(M.duration) AS total_duration,
            CAST(SUM(M.duration * M.duration) AS REAL) AS sqr_duration,
            A.avg_duration AS average_duration,
            MIN(M.duration) AS min_duration,
            MAX(M.duration) AS max_duration,
            AVG((M.duration - A.avg_duration) * (M.duration - A.avg_duration)) AS variance_duration,
            SQRT(AVG((M.duration - A.avg_duration) * (M.duration - A.avg_duration))) AS std_dev_duration
        FROM
            all_markers M
            JOIN avg_data A ON M.name = A.name
        GROUP BY
            M.name
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
    AD.total_duration AS `DURATION (nsec)`,
    AD.sqr_duration AS `SQR (nsec)`,
    AD.average_duration AS `AVERAGE (nsec)`,
    (CAST(AD.total_duration AS REAL) / NULLIF(TD.grand_total_duration, 0)) * 100 AS `PERCENT (INC)`,
    AD.min_duration AS `MIN (nsec)`,
    AD.max_duration AS `MAX (nsec)`,
    AD.variance_duration AS `VARIANCE`,
    AD.std_dev_duration AS `STD_DEV`
FROM
    aggregated_data AD
    CROSS JOIN total_duration TD;
