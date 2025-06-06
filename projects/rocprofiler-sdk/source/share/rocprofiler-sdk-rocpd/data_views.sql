--
-- Useful views
--
-- Code objects
CREATE VIEW IF NOT EXISTS
    `code_objects` AS
SELECT
    CO.id,
    CO.guid,
    CO.nid,
    P.pid,
    A.absolute_index AS agent_abs_index,
    CO.uri,
    CO.load_base,
    CO.load_size,
    CO.load_delta,
    CO.storage_type AS storage_type_str,
    JSON_EXTRACT(CO.extdata, '$.size') AS code_object_size,
    JSON_EXTRACT(CO.extdata, '$.storage_type') AS storage_type,
    JSON_EXTRACT(CO.extdata, '$.memory_base') AS memory_base,
    JSON_EXTRACT(CO.extdata, '$.memory_size') AS memory_size
FROM
    `rocpd_info_code_object` CO
    INNER JOIN `rocpd_info_agent` A ON CO.agent_id = A.id
    AND CO.guid = A.guid
    INNER JOIN `rocpd_info_process` P ON CO.pid = P.id
    AND CO.guid = P.guid;

CREATE VIEW IF NOT EXISTS
    `kernel_symbols` AS
SELECT
    KS.id,
    KS.guid,
    KS.nid,
    P.pid,
    KS.code_object_id,
    KS.kernel_name,
    KS.display_name,
    KS.kernel_object,
    KS.kernarg_segment_size,
    KS.kernarg_segment_alignment,
    KS.group_segment_size,
    KS.private_segment_size,
    KS.sgpr_count,
    KS.arch_vgpr_count,
    KS.accum_vgpr_count,
    JSON_EXTRACT(KS.extdata, '$.size') AS kernel_symbol_size,
    JSON_EXTRACT(KS.extdata, '$.kernel_id') AS kernel_id,
    JSON_EXTRACT(KS.extdata, '$.kernel_code_entry_byte_offset') AS kernel_code_entry_byte_offset,
    JSON_EXTRACT(KS.extdata, '$.formatted_kernel_name') AS formatted_kernel_name,
    JSON_EXTRACT(KS.extdata, '$.demangled_kernel_name') AS demangled_kernel_name,
    JSON_EXTRACT(KS.extdata, '$.truncated_kernel_name') AS truncated_kernel_name,
    JSON_EXTRACT(KS.extdata, '$.kernel_address.handle') AS kernel_address
FROM
    `rocpd_info_kernel_symbol` KS
    INNER JOIN `rocpd_info_process` P ON KS.pid = P.id
    AND KS.guid = P.guid;

-- Processes
CREATE VIEW IF NOT EXISTS
    `processes` AS
SELECT
    N.id AS nid,
    N.machine_id,
    N.system_name,
    N.hostname,
    N.release AS system_release,
    N.version AS system_version,
    P.guid,
    P.ppid,
    P.pid,
    P.init,
    P.start,
    P.end,
    P.fini,
    P.command
FROM
    `rocpd_info_process` P
    INNER JOIN `rocpd_info_node` N ON N.id = P.nid
    AND N.guid = P.guid;

-- Threads
CREATE VIEW IF NOT EXISTS
    `threads` AS
SELECT
    N.id AS nid,
    N.machine_id,
    N.system_name,
    N.hostname,
    N.release AS system_release,
    N.version AS system_version,
    P.guid,
    P.ppid,
    P.pid,
    T.tid,
    T.start,
    T.end,
    T.name
FROM
    `rocpd_info_thread` T
    INNER JOIN `rocpd_info_process` P ON P.id = T.pid
    AND N.guid = T.guid
    INNER JOIN `rocpd_info_node` N ON N.id = T.nid
    AND N.guid = T.guid;

-- CPU regions
CREATE VIEW IF NOT EXISTS
    `regions` AS
SELECT
    R.id,
    R.guid,
    (
        SELECT
            string
        FROM
            `rocpd_string` RS
        WHERE
            RS.id = E.category_id
            AND RS.guid = E.guid
    ) AS category,
    S.string AS name,
    R.nid,
    P.pid,
    T.tid,
    R.start,
    R.end,
    (R.end - R.start) AS duration,
    R.event_id,
    E.stack_id,
    E.parent_stack_id,
    E.correlation_id AS corr_id,
    E.extdata,
    E.call_stack,
    E.line_info
FROM
    `rocpd_region` R
    INNER JOIN `rocpd_event` E ON E.id = R.event_id
    AND E.guid = R.guid
    INNER JOIN `rocpd_string` S ON S.id = R.name_id
    AND S.guid = R.guid
    INNER JOIN `rocpd_info_process` P ON P.id = R.pid
    AND P.guid = R.guid
    INNER JOIN `rocpd_info_thread` T ON T.id = R.tid
    AND T.guid = R.guid;

CREATE VIEW IF NOT EXISTS
    `region_args` AS
SELECT
    R.id,
    R.guid,
    R.nid,
    P.pid,
    A.type,
    A.name,
    A.value
FROM
    `rocpd_region` R
    INNER JOIN `rocpd_event` E ON E.id = R.event_id
    AND E.guid = R.guid
    INNER JOIN `rocpd_arg` A ON A.event_id = E.id
    AND A.guid = R.guid
    INNER JOIN `rocpd_info_process` P ON P.id = R.pid
    AND P.guid = R.guid;

--
-- Samples
CREATE VIEW IF NOT EXISTS
    `samples` AS
SELECT
    R.id,
    R.guid,
    (
        SELECT
            string
        FROM
            `rocpd_string` RS
        WHERE
            RS.id = E.category_id
            AND RS.guid = E.guid
    ) AS category,
    (
        SELECT
            string
        FROM
            `rocpd_string` RS
        WHERE
            RS.id = T.name_id
            AND RS.guid = T.guid
    ) AS name,
    T.nid,
    P.pid,
    TH.tid,
    R.timestamp,
    R.event_id,
    E.stack_id AS stack_id,
    E.parent_stack_id AS parent_stack_id,
    E.correlation_id AS corr_id,
    E.extdata AS extdata,
    E.call_stack AS call_stack,
    E.line_info AS line_info
FROM
    `rocpd_sample` R
    INNER JOIN `rocpd_track` T ON T.id = R.track_id
    AND T.guid = R.guid
    INNER JOIN `rocpd_event` E ON E.id = R.event_id
    AND E.guid = R.guid
    INNER JOIN `rocpd_info_process` P ON P.id = T.pid
    AND P.guid = T.guid
    INNER JOIN `rocpd_info_thread` TH ON TH.id = T.tid
    AND TH.guid = T.guid;

--
-- Provides samples view with the same columns as regions view
CREATE VIEW IF NOT EXISTS
    `sample_regions` AS
SELECT
    R.id,
    R.guid,
    (
        SELECT
            string
        FROM
            `rocpd_string` RS
        WHERE
            RS.id = E.category_id
            AND RS.guid = E.guid
    ) AS category,
    (
        SELECT
            string
        FROM
            `rocpd_string` RS
        WHERE
            RS.id = T.name_id
            AND RS.guid = T.guid
    ) AS name,
    T.nid,
    P.pid,
    TH.tid,
    R.timestamp AS start,
    R.timestamp AS END,
    (R.timestamp - R.timestamp) AS duration,
    R.event_id,
    E.stack_id AS stack_id,
    E.parent_stack_id AS parent_stack_id,
    E.correlation_id AS corr_id,
    E.extdata AS extdata,
    E.call_stack AS call_stack,
    E.line_info AS line_info
FROM
    `rocpd_sample` R
    INNER JOIN `rocpd_track` T ON T.id = R.track_id
    AND T.guid = R.guid
    INNER JOIN `rocpd_event` E ON E.id = R.event_id
    AND E.guid = R.guid
    INNER JOIN `rocpd_info_process` P ON P.id = T.pid
    AND P.guid = T.guid
    INNER JOIN `rocpd_info_thread` TH ON TH.id = T.tid
    AND TH.guid = T.guid;

--
-- Provides a unified view of the regions and samples
CREATE VIEW IF NOT EXISTS
    `regions_and_samples` AS
SELECT
    *
FROM
    `regions`
UNION ALL
SELECT
    *
FROM
    `sample_regions`;

--
-- Kernel information
CREATE VIEW
    `kernels` AS
SELECT
    K.id,
    K.guid,
    T.tid,
    (
        SELECT
            string
        FROM
            `rocpd_string` RS
        WHERE
            RS.id = E.category_id
            AND RS.guid = E.guid
    ) AS category,
    R.string AS region,
    S.display_name AS name,
    K.nid,
    P.pid,
    A.absolute_index AS agent_abs_index,
    A.logical_index AS agent_log_index,
    A.type_index AS agent_type_index,
    A.type AS agent_type,
    S.code_object_id AS code_object_id,
    K.kernel_id,
    K.dispatch_id,
    K.stream_id,
    K.queue_id,
    Q.name AS queue,
    ST.name AS stream,
    K.start,
    K.end,
    (K.end - K.start) AS duration,
    K.grid_size_x AS grid_x,
    K.grid_size_y AS grid_y,
    K.grid_size_z AS grid_z,
    K.workgroup_size_x AS workgroup_x,
    K.workgroup_size_y AS workgroup_y,
    K.workgroup_size_z AS workgroup_z,
    K.group_segment_size AS lds_size,
    K.private_segment_size AS scratch_size,
    S.group_segment_size AS static_lds_size,
    S.private_segment_size AS static_scratch_size,
    E.stack_id,
    E.parent_stack_id,
    E.correlation_id AS corr_id
FROM
    `rocpd_kernel_dispatch` K
    INNER JOIN `rocpd_info_agent` A ON A.id = K.agent_id
    AND A.guid = K.guid
    INNER JOIN `rocpd_event` E ON E.id = K.event_id
    AND E.guid = K.guid
    INNER JOIN `rocpd_string` R ON R.id = K.region_name_id
    AND R.guid = K.guid
    INNER JOIN `rocpd_info_kernel_symbol` S ON S.id = K.kernel_id
    AND S.guid = K.guid
    LEFT JOIN `rocpd_info_stream` ST ON ST.id = K.stream_id
    AND ST.guid = K.guid
    LEFT JOIN `rocpd_info_queue` Q ON Q.id = K.queue_id
    AND Q.guid = K.guid
    INNER JOIN `rocpd_info_process` P ON P.id = Q.pid
    AND P.guid = Q.guid
    INNER JOIN `rocpd_info_thread` T ON T.id = K.tid
    AND T.guid = K.guid;

--
-- Performance Monitoring Counters (PMC)
CREATE VIEW IF NOT EXISTS
    `pmc_info` AS
SELECT
    PMC_I.id,
    PMC_I.guid,
    PMC_I.nid,
    P.pid,
    A.absolute_index AS agent_abs_index,
    PMC_I.is_constant,
    PMC_I.is_derived,
    PMC_I.name,
    PMC_I.description,
    PMC_I.block,
    PMC_I.expression
FROM
    `rocpd_info_pmc` PMC_I
    INNER JOIN `rocpd_info_agent` A ON PMC_I.agent_id = A.id
    AND PMC_I.guid = A.guid
    INNER JOIN `rocpd_info_process` P ON P.id = PMC_I.pid
    AND PMC_I.guid = P.guid;

CREATE VIEW IF NOT EXISTS
    `pmc_events` AS
SELECT
    PMC_E.id,
    PMC_E.guid,
    PMC_E.pmc_id,
    E.id AS event_id,
    (
        SELECT
            string
        FROM
            `rocpd_string` RS
        WHERE
            RS.id = E.category_id
            AND RS.guid = E.guid
    ) AS category,
    (
        SELECT
            display_name
        FROM
            `rocpd_info_kernel_symbol` KS
        WHERE
            KS.id = K.kernel_id
            AND KS.guid = K.guid
    ) AS name,
    K.nid,
    P.pid,
    K.dispatch_id,
    K.start,
    K.end,
    (K.end - K.start) AS duration,
    PMC_I.name AS counter_name,
    PMC_E.value AS counter_value
FROM
    `rocpd_pmc_event` PMC_E
    INNER JOIN `rocpd_info_pmc` PMC_I ON PMC_I.id = PMC_E.pmc_id
    AND PMC_I.guid = PMC_E.guid
    INNER JOIN `rocpd_event` E ON E.id = PMC_E.event_id
    AND E.guid = PMC_E.guid
    INNER JOIN `rocpd_kernel_dispatch` K ON K.event_id = PMC_E.event_id
    AND K.guid = PMC_E.guid
    INNER JOIN `rocpd_info_process` P ON P.id = K.pid
    AND P.guid = K.guid;

-- events with arguments ---
CREATE VIEW IF NOT EXISTS
    `events_args` AS
SELECT
    E.id AS event_id,
    (
        SELECT
            string
        FROM
            `rocpd_string` RS
        WHERE
            RS.id = E.category_id
            AND RS.guid = E.guid
    ) AS category,
    E.stack_id,
    E.parent_stack_id,
    E.correlation_id,
    A.position AS arg_position,
    A.type AS arg_type,
    A.name AS arg_name,
    A.value AS arg_value,
    E.call_stack,
    E.line_info,
    A.extdata
FROM
    `rocpd_event` E
    INNER JOIN `rocpd_arg` A ON A.event_id = E.id
    AND A.guid = E.guid;

-- list of astream arguments enriched by the corresponding stream descriptions
CREATE VIEW IF NOT EXISTS
    `stream_args` AS
SELECT
    A.id AS argument_id,
    A.event_id AS event_id,
    A.position AS arg_position,
    A.type AS arg_type,
    A.value AS arg_value,
    JSON_EXTRACT(A.extdata, '$.stream_id') AS stream_id,
    S.nid,
    P.pid,
    S.name AS stream_name,
    S.extdata AS extdata
FROM
    `rocpd_arg` A
    INNER JOIN `rocpd_info_stream` S ON JSON_EXTRACT(A.extdata, '$.stream_id') = S.id
    AND A.guid = S.guid
    INNER JOIN `rocpd_info_process` P ON P.id = S.pid
    AND P.guid = S.guid
WHERE
    A.name = 'stream';

--
--
CREATE VIEW IF NOT EXISTS
    `memory_copies` AS
SELECT
    M.id,
    M.guid,
    (
        SELECT
            string
        FROM
            `rocpd_string` RS
        WHERE
            RS.id = E.category_id
            AND RS.guid = E.guid
    ) AS category,
    M.nid,
    P.pid,
    T.tid,
    M.start,
    M.end,
    (M.end - M.start) AS duration,
    S.string AS name,
    R.string AS region_name,
    M.stream_id,
    M.queue_id,
    ST.name AS stream_name,
    Q.name AS queue_name,
    M.size,
    dst_agent.name AS dst_device,
    dst_agent.absolute_index AS dst_agent_abs_index,
    dst_agent.logical_index AS dst_agent_log_index,
    dst_agent.type_index AS dst_agent_type_index,
    dst_agent.type AS dst_agent_type,
    M.dst_address,
    src_agent.name AS src_device,
    src_agent.absolute_index AS src_agent_abs_index,
    src_agent.logical_index AS src_agent_log_index,
    src_agent.type_index AS src_agent_type_index,
    src_agent.type AS src_agent_type,
    M.src_address,
    E.stack_id,
    E.parent_stack_id,
    E.correlation_id AS corr_id
FROM
    `rocpd_memory_copy` M
    INNER JOIN `rocpd_string` S ON S.id = M.name_id
    AND S.guid = M.guid
    LEFT JOIN `rocpd_string` R ON R.id = M.region_name_id
    AND R.guid = M.guid
    INNER JOIN `rocpd_info_agent` dst_agent ON dst_agent.id = M.dst_agent_id
    AND dst_agent.guid = M.guid
    INNER JOIN `rocpd_info_agent` src_agent ON src_agent.id = M.src_agent_id
    AND src_agent.guid = M.guid
    LEFT JOIN `rocpd_info_queue` Q ON Q.id = M.queue_id
    AND Q.guid = M.guid
    LEFT JOIN `rocpd_info_stream` ST ON ST.id = M.stream_id
    AND ST.guid = M.guid
    INNER JOIN `rocpd_event` E ON E.id = M.event_id
    AND E.guid = M.guid
    INNER JOIN `rocpd_info_process` P ON P.id = M.pid
    AND P.guid = M.guid
    INNER JOIN `rocpd_info_thread` T ON T.id = M.tid
    AND T.guid = M.guid;

--
--
CREATE VIEW IF NOT EXISTS
    `memory_allocations` AS
SELECT
    M.id,
    M.guid,
    (
        SELECT
            string
        FROM
            `rocpd_string` RS
        WHERE
            RS.id = E.category_id
            AND RS.guid = E.guid
    ) AS category,
    M.nid,
    P.pid,
    T.tid,
    M.start,
    M.end,
    (M.end - M.start) AS duration,
    M.type,
    M.level,
    A.name AS agent_name,
    A.absolute_index AS agent_abs_index,
    A.logical_index AS agent_log_index,
    A.type_index AS agent_type_index,
    A.type AS agent_type,
    M.address,
    M.size,
    M.queue_id,
    Q.name AS queue_name,
    M.stream_id,
    ST.name AS stream_name,
    E.stack_id,
    E.parent_stack_id,
    E.correlation_id AS corr_id
FROM
    `rocpd_memory_allocate` M
    LEFT JOIN `rocpd_info_agent` A ON M.agent_id = A.id
    AND M.guid = A.guid
    LEFT JOIN `rocpd_info_queue` Q ON Q.id = M.queue_id
    AND Q.guid = M.guid
    LEFT JOIN `rocpd_info_stream` ST ON ST.id = M.stream_id
    AND ST.guid = M.guid
    INNER JOIN `rocpd_event` E ON E.id = M.event_id
    AND E.guid = M.guid
    INNER JOIN `rocpd_info_process` P ON P.id = M.pid
    AND P.guid = M.guid
    INNER JOIN `rocpd_info_thread` T ON T.id = M.tid
    AND P.guid = M.guid;

--
--
CREATE VIEW IF NOT EXISTS
    `scratch_memory` AS
SELECT
    M.id,
    M.guid,
    M.nid,
    P.pid,
    M.type AS operation,
    A.name AS agent_name,
    A.absolute_index AS agent_abs_index,
    A.logical_index AS agent_log_index,
    A.type_index AS agent_type_index,
    A.type AS agent_type,
    M.queue_id,
    T.tid,
    JSON_EXTRACT(M.extdata, '$.flags') AS alloc_flags,
    M.start,
    M.end,
    M.size,
    M.address,
    E.correlation_id,
    E.stack_id,
    E.parent_stack_id,
    E.correlation_id AS corr_id,
    (
        SELECT
            string
        FROM
            `rocpd_string` RS
        WHERE
            RS.id = E.category_id
            AND RS.guid = E.guid
    ) AS category,
    E.extdata AS event_extdata
FROM
    `rocpd_memory_allocate` M
    LEFT JOIN `rocpd_info_agent` A ON M.agent_id = A.id
    AND M.guid = A.guid
    LEFT JOIN `rocpd_info_queue` Q ON Q.id = M.queue_id
    AND Q.guid = M.guid
    INNER JOIN `rocpd_event` E ON E.id = M.event_id
    AND E.guid = M.guid
    INNER JOIN `rocpd_info_process` P ON P.id = M.pid
    AND P.guid = M.guid
    INNER JOIN `rocpd_info_thread` T ON T.id = M.tid
    AND T.guid = M.guid
WHERE
    M.level = 'SCRATCH'
ORDER BY
    M.start ASC;

--
--
CREATE VIEW IF NOT EXISTS
    `counters_collection` AS
SELECT
    MIN(PMC_E.id) AS id,
    PMC_E.guid,
    K.dispatch_id,
    K.kernel_id,
    E.id AS event_id,
    E.correlation_id,
    E.stack_id,
    E.parent_stack_id,
    P.pid,
    T.tid,
    K.agent_id,
    A.absolute_index AS agent_abs_index,
    A.logical_index AS agent_log_index,
    A.type_index AS agent_type_index,
    A.type AS agent_type,
    K.queue_id,
    k.grid_size_x AS grid_size_x,
    k.grid_size_y AS grid_size_y,
    k.grid_size_z AS grid_size_z,
    (K.grid_size_x * K.grid_size_y * K.grid_size_z) AS grid_size,
    S.display_name AS kernel_name,
    (
        SELECT
            string
        FROM
            `rocpd_string` RS
        WHERE
            RS.id = K.region_name_id
            AND RS.guid = K.guid
    ) AS kernel_region,
    K.workgroup_size_x AS workgroup_size_x,
    K.workgroup_size_y AS workgroup_size_y,
    K.workgroup_size_z AS workgroup_size_z,
    (K.workgroup_size_x * K.workgroup_size_y * K.workgroup_size_z) AS workgroup_size,
    K.group_segment_size AS lds_block_size,
    K.private_segment_size AS scratch_size,
    S.arch_vgpr_count AS vgpr_count,
    S.accum_vgpr_count,
    S.sgpr_count,
    PMC_I.name AS counter_name,
    PMC_I.symbol AS counter_symbol,
    PMC_I.component,
    PMC_I.description,
    PMC_I.block,
    PMC_I.expression,
    PMC_I.value_type,
    PMC_I.id AS counter_id,
    SUM(PMC_E.value) AS value,
    K.start,
    K.end,
    PMC_I.is_constant,
    PMC_I.is_derived,
    (K.end - K.start) AS duration,
    (
        SELECT
            string
        FROM
            `rocpd_string` RS
        WHERE
            RS.id = E.category_id
            AND RS.guid = E.guid
    ) AS category,
    K.nid,
    E.extdata,
    S.code_object_id
FROM
    `rocpd_pmc_event` PMC_E
    INNER JOIN `rocpd_info_pmc` PMC_I ON PMC_I.id = PMC_E.pmc_id
    AND PMC_I.guid = PMC_E.guid
    INNER JOIN `rocpd_event` E ON E.id = PMC_E.event_id
    AND E.guid = PMC_E.guid
    INNER JOIN `rocpd_kernel_dispatch` K ON K.event_id = PMC_E.event_id
    AND K.guid = PMC_E.guid
    INNER JOIN `rocpd_info_agent` A ON A.id = K.agent_id
    AND A.guid = K.guid
    INNER JOIN `rocpd_info_kernel_symbol` S ON S.id = K.kernel_id
    AND S.guid = K.guid
    INNER JOIN `rocpd_info_process` P ON P.id = K.pid
    AND P.guid = K.guid
    INNER JOIN `rocpd_info_thread` T ON T.id = K.tid
    AND T.guid = K.guid
GROUP BY
    PMC_E.guid,
    K.dispatch_id,
    PMC_I.name,
    K.agent_id;
