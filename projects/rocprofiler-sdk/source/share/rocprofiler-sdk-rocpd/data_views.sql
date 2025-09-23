--
-- Useful views
--
-- This is a view of all the joined track information
CREATE VIEW IF NOT EXISTS
    `tracks` AS
SELECT
    T.id,
    T.guid,
    T.name_id AS track_name_id,
    ST.string AS track_name,
    T.nid,
    N.name AS node_name,
    N.hash AS node_hash,
    N.machine_id AS node_machine_id,
    N.system_name AS node_system_name,
    N.hostname AS node_hostname,
    N.release AS node_release,
    N.version AS node_version,
    N.hardware_name AS node_hardware_version,
    T.pid,
    P.name AS process_name,
    P.ppid,
    P.init AS process_init,
    P.fini AS process_fini,
    P.start AS process_start,
    P.end AS process_end,
    P.command AS process_command,
    T.tid,
    TH.name AS thread_name,
    TH.start AS thread_start,
    TH.end AS thread_end,
    T.agent_id,
    A.name AS agent_name,
    A.type AS agent_type,
    A.absolute_index AS agent_absolute_index,
    A.logical_index AS agent_logical_index,
    A.type_index AS agent_type_index,
    A.uuid AS agent_uuid,
    A.generic_name AS agent_generic_name,
    A.model_name AS agent_model_name,
    A.vendor_name AS agent_vendor_name,
    A.product_name AS agent_product_name,
    T.queue_id,
    Q.name AS queue_name,
    T.stream_id,
    S.name AS stream_name
FROM
    `rocpd_track` T
    LEFT JOIN `rocpd_info_node` N ON N.id = T.nid
    AND N.guid = T.guid
    LEFT JOIN `rocpd_info_process` P ON P.pid = T.pid
    AND P.guid = T.guid
    LEFT JOIN `rocpd_info_thread` TH ON TH.tid = T.tid
    AND TH.guid = T.guid
    LEFT JOIN `rocpd_info_agent` A ON A.id = T.agent_id
    AND A.guid = T.guid
    LEFT JOIN `rocpd_info_queue` Q ON Q.id = T.queue_id
    AND Q.guid = T.guid
    LEFT JOIN `rocpd_info_stream` S ON S.id = T.stream_id
    AND S.guid = T.guid
    LEFT JOIN `rocpd_string` ST ON ST.id = T.name_id
    AND ST.guid = T.guid;

--
-- This is a view of all the joined event information
CREATE VIEW IF NOT EXISTS
    `events` AS
SELECT
    E.id,
    E.guid,
    E.category_id,
    (
        SELECT
            name
        FROM
            `rocpd_info_category` C
        WHERE
            C.id = E.category_id
            AND C.guid = E.guid
    ) AS category,
    E.stack_id,
    E.parent_stack_id,
    E.correlation_id,
    E.extdata
FROM
    `rocpd_event` E;

--
-- Code objects
CREATE VIEW IF NOT EXISTS
    `code_objects` AS
SELECT
    CO.id,
    CO.guid,
    CO.nid,
    P.pid,
    A.absolute_index AS agent_absolute_index,
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
    INNER JOIN `rocpd_info_process` P ON CO.pid = P.pid
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
    INNER JOIN `rocpd_info_process` P ON KS.pid = P.pid
    AND KS.guid = P.guid;

-- Processes
CREATE VIEW IF NOT EXISTS
    `processes` AS
SELECT
    P.id,
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
    T.id,
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
    INNER JOIN `rocpd_info_process` P ON P.pid = T.pid
    AND N.guid = T.guid
    INNER JOIN `rocpd_info_node` N ON N.id = T.nid
    AND N.guid = T.guid;

-- CPU regions
CREATE VIEW IF NOT EXISTS
    `regions` AS
SELECT
    R.id,
    R.guid,
    E.category,
    NS.string AS name,
    T.nid,
    T.pid,
    T.tid,
    DS.value AS `start`,
    DE.value AS `end`,
    (DE.value - DS.value) AS `duration`,
    R.event_id,
    R.track_id,
    E.stack_id,
    E.parent_stack_id,
    E.correlation_id,
    E.extdata
FROM
    `rocpd_region` R
    INNER JOIN `events` E ON E.id = R.event_id
    AND E.guid = R.guid
    INNER JOIN `tracks` T ON T.id = R.track_id
    AND T.guid = R.guid
    INNER JOIN `rocpd_string` NS ON NS.id = R.name_id
    AND NS.guid = R.guid
    INNER JOIN `rocpd_timestamp` DS ON DS.id = R.start_id
    AND DS.guid = R.guid
    INNER JOIN `rocpd_timestamp` DE ON DE.id = R.end_id
    AND DE.guid = R.guid;

--
-- Samples
CREATE VIEW IF NOT EXISTS
    `samples` AS
SELECT
    S.id,
    S.guid,
    E.category,
    NS.string AS `name`,
    T.nid,
    T.pid,
    T.tid,
    DI.value AS `timestamp`,
    S.event_id,
    S.track_id,
    E.stack_id AS stack_id,
    E.parent_stack_id AS parent_stack_id,
    E.correlation_id,
    E.extdata AS extdata
FROM
    `rocpd_sample` S
    INNER JOIN `tracks` T ON T.id = S.track_id
    AND T.guid = S.guid
    INNER JOIN `events` E ON E.id = S.event_id
    AND E.guid = S.guid
    INNER JOIN `rocpd_string` NS ON NS.id = S.name_id
    AND NS.guid = S.guid
    INNER JOIN `rocpd_timestamp` DI ON DI.id = S.timestamp_id
    AND DI.guid = S.guid;

--
-- Provides samples view with the same columns as regions view
CREATE VIEW IF NOT EXISTS
    `sample_regions` AS
SELECT
    S.id,
    S.guid,
    S.category,
    S.name,
    S.nid,
    S.pid,
    S.tid,
    S.timestamp AS `start`,
    S.timestamp AS `end`,
    (S.timestamp - S.timestamp) AS `duration`,
    S.event_id,
    S.track_id,
    S.stack_id,
    S.parent_stack_id,
    S.correlation_id,
    S.extdata
FROM
    `samples` S;

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
    T.nid,
    T.pid,
    T.tid,
    E.category,
    R.string AS region,
    S.display_name AS name,
    T.agent_id,
    T.agent_absolute_index,
    T.agent_logical_index,
    T.agent_type_index,
    T.agent_type,
    S.code_object_id,
    K.kernel_id,
    K.dispatch_id,
    T.queue_id,
    T.queue_name AS `queue`,
    T.stream_id,
    T.stream_name AS `stream`,
    DS.value AS `start`,
    DE.value AS `end`,
    (DE.value - DS.value) AS `duration`,
    K.event_id,
    K.track_id,
    -- OpenCL uses "grid" to mean number of work-items in a dimension
    K.grid_size_x AS grid_x,
    K.grid_size_y AS grid_y,
    K.grid_size_z AS grid_z,
    (K.grid_size_x * K.grid_size_y * K.grid_size_z) AS ocl_grid_size,
    -- OpenCL uses "work-group" to mean a group of work-items that execute together
    K.workgroup_size_x AS workgroup_x,
    K.workgroup_size_y AS workgroup_y,
    K.workgroup_size_z AS workgroup_z,
    (K.workgroup_size_x * K.workgroup_size_y * K.workgroup_size_z) AS ocl_workgroup_size,
    -- HIP uses "block" to mean number of threads in a workgroup
    K.workgroup_size_x AS block_size_x,
    K.workgroup_size_y AS block_size_y,
    K.workgroup_size_z AS block_size_z,
    (K.workgroup_size_x * K.workgroup_size_y * K.workgroup_size_z) AS block_size,
    -- HIP uses "grid" to mean number of blocks in a grid
    (K.grid_size_x / K.workgroup_size_x) AS grid_size_x,
    (K.grid_size_y / K.workgroup_size_y) AS grid_size_y,
    (K.grid_size_z / K.workgroup_size_z) AS grid_size_z,
    (K.grid_size_x / K.workgroup_size_x) * (K.grid_size_y / K.workgroup_size_y) * (K.grid_size_z / K.workgroup_size_z) AS grid_size,
    -- lds_block_size is the group segment size aligned to 512 bytes
    ((K.group_segment_size + 511) / 512) * 512 AS lds_block_size,
    K.private_segment_size AS scratch_size,
    S.group_segment_size AS static_lds_block_size,
    S.private_segment_size AS static_scratch_size,
    S.sgpr_count,
    S.arch_vgpr_count,
    S.accum_vgpr_count,
    E.stack_id,
    E.parent_stack_id,
    E.correlation_id,
    E.extdata
FROM
    `rocpd_kernel_dispatch` K
    INNER JOIN `tracks` T ON T.id = K.track_id
    AND T.guid = K.guid
    INNER JOIN `events` E ON E.id = K.event_id
    AND E.guid = K.guid
    INNER JOIN `rocpd_string` R ON R.id = K.region_name_id
    AND R.guid = K.guid
    INNER JOIN `rocpd_info_kernel_symbol` S ON S.id = K.kernel_id
    AND S.guid = K.guid
    INNER JOIN `rocpd_timestamp` DS ON DS.id = K.start_id
    AND DS.guid = K.guid
    INNER JOIN `rocpd_timestamp` DE ON DE.id = K.end_id
    AND DE.guid = K.guid;

--
-- Performance Monitoring Counters (PMC)
CREATE VIEW IF NOT EXISTS
    `pmc_info` AS
SELECT
    PMC_I.id,
    PMC_I.guid,
    PMC_I.nid,
    P.pid,
    A.absolute_index AS agent_absolute_index,
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
    INNER JOIN `rocpd_info_process` P ON P.pid = PMC_I.pid
    AND PMC_I.guid = P.guid;

--
-- Join PMC records with PMC info and event info
CREATE VIEW IF NOT EXISTS
    `pmc_events` AS
SELECT
    PMC_E.id,
    PMC_E.guid,
    PMC_E.pmc_id,
    PMC_E.event_id,
    E.category_id,
    E.category,
    PMC_I.name,
    PMC_I.symbol,
    PMC_E.value,
    PMC_I.agent_id,
    PMC_I.target_arch,
    PMC_I.event_code,
    PMC_I.instance_id,
    PMC_I.component,
    PMC_I.units,
    PMC_I.value_type,
    PMC_I.block,
    PMC_I.expression,
    PMC_I.is_constant,
    PMC_I.is_derived,
    PMC_I.description,
    PMC_I.long_description,
    PMC_I.extdata AS pmc_info_extdata,
    PMC_E.extdata AS pmc_event_extdata
FROM
    `rocpd_pmc_event` PMC_E
    INNER JOIN `rocpd_info_pmc` PMC_I ON PMC_I.id = PMC_E.pmc_id
    AND PMC_I.guid = PMC_E.guid
    INNER JOIN `events` E ON E.id = PMC_E.event_id
    AND E.guid = PMC_E.guid;

--
--
CREATE VIEW IF NOT EXISTS
    `memory_copies` AS
SELECT
    M.id,
    M.guid,
    T.nid,
    T.pid,
    T.tid,
    E.id AS event_id,
    E.category,
    NS.string AS name,
    R.string AS region_name,
    DS.value AS `start`,
    DE.value AS `end`,
    (DE.value - DS.value) AS `duration`,
    T.queue_id,
    T.queue_name,
    T.stream_id,
    T.stream_name,
    M.size,
    dst_agent.name AS dst_device,
    dst_agent.id AS dst_agent_id,
    dst_agent.absolute_index AS dst_agent_absolute_index,
    dst_agent.logical_index AS dst_agent_logical_index,
    dst_agent.type_index AS dst_agent_type_index,
    dst_agent.type AS dst_agent_type,
    M.dst_address,
    src_agent.name AS src_device,
    src_agent.id AS src_agent_id,
    src_agent.absolute_index AS src_agent_absolute_index,
    src_agent.logical_index AS src_agent_logical_index,
    src_agent.type_index AS src_agent_type_index,
    src_agent.type AS src_agent_type,
    M.src_address,
    E.stack_id,
    E.parent_stack_id,
    E.correlation_id,
    E.extdata
FROM
    `rocpd_memory_copy` M
    INNER JOIN `events` E ON E.id = M.event_id
    AND E.guid = M.guid
    INNER JOIN `tracks` T ON T.id = M.track_id
    AND T.guid = M.guid
    INNER JOIN `rocpd_string` NS ON NS.id = M.name_id
    AND NS.guid = M.guid
    LEFT JOIN `rocpd_string` R ON R.id = M.region_name_id
    AND R.guid = M.guid
    INNER JOIN `rocpd_info_agent` dst_agent ON dst_agent.id = M.dst_agent_id
    AND dst_agent.guid = M.guid
    INNER JOIN `rocpd_info_agent` src_agent ON src_agent.id = M.src_agent_id
    AND src_agent.guid = M.guid
    INNER JOIN `rocpd_timestamp` DS ON DS.id = M.start_id
    AND DS.guid = M.guid
    INNER JOIN `rocpd_timestamp` DE ON DE.id = M.end_id
    AND DE.guid = M.guid;

--
--
CREATE VIEW IF NOT EXISTS
    `memory_allocations` AS
SELECT
    M.id,
    M.guid,
    T.nid,
    T.pid,
    T.tid,
    E.id AS event_id,
    E.category,
    NS.string AS name,
    R.string AS region_name,
    DS.value AS `start`,
    DE.value AS `end`,
    (DE.value - DS.value) AS `duration`,
    T.queue_id,
    T.queue_name,
    T.stream_id,
    T.stream_name,
    M.size,
    M.type,
    M.level,
    T.agent_name,
    T.agent_absolute_index,
    T.agent_logical_index,
    T.agent_type_index,
    T.agent_type,
    M.address,
    E.stack_id,
    E.parent_stack_id,
    E.correlation_id,
    E.extdata
FROM
    `rocpd_memory_allocate` M
    INNER JOIN `events` E ON E.id = M.event_id
    AND E.guid = M.guid
    INNER JOIN `tracks` T ON T.id = M.track_id
    AND E.guid = M.guid
    INNER JOIN `rocpd_string` NS ON NS.id = M.name_id
    AND NS.guid = M.guid
    LEFT JOIN `rocpd_string` R ON R.id = M.region_name_id
    AND R.guid = M.guid
    INNER JOIN `rocpd_timestamp` DS ON DS.id = M.start_id
    AND DS.guid = M.guid
    INNER JOIN `rocpd_timestamp` DE ON DE.id = M.end_id
    AND DE.guid = M.guid;

--
-- PMC events specific to kernels
CREATE VIEW IF NOT EXISTS
    `kernel_pmc_events` AS
SELECT
    K.id,
    K.guid,
    K.nid,
    K.pid,
    K.tid,
    K.category,
    K.region,
    K.name,
    K.agent_id,
    K.agent_absolute_index,
    K.agent_logical_index,
    K.agent_type_index,
    K.agent_type,
    K.code_object_id,
    K.kernel_id,
    K.dispatch_id,
    K.queue_id,
    K.queue,
    K.stream_id,
    K.stream,
    K.start,
    K.end,
    K.duration,
    K.event_id,
    K.track_id,
    K.stack_id,
    K.parent_stack_id,
    K.correlation_id,
    K.grid_x,
    K.grid_y,
    K.grid_z,
    K.workgroup_x,
    K.workgroup_y,
    K.workgroup_z,
    K.lds_block_size,
    K.scratch_size,
    K.static_lds_block_size,
    K.static_scratch_size,
    K.sgpr_count,
    K.arch_vgpr_count,
    K.accum_vgpr_count,
    E.pmc_id,
    E.name AS `pmc_name`,
    E.symbol AS `pmc_symbol`,
    E.value AS `pmc_value`,
    E.agent_id AS `pmc_agent_id`,
    E.target_arch AS `pmc_target_arch`,
    E.event_code AS `pmc_event_code`,
    E.instance_id AS `pmc_instance_id`,
    E.component AS `pmc_component`,
    E.units AS `pmc_units`,
    E.value_type AS `pmc_value_type`,
    E.block AS `pmc_block`,
    E.expression AS `pmc_expression`,
    E.is_constant AS `pmc_is_constant`,
    E.is_derived AS `pmc_is_derived`,
    E.description AS `pmc_description`,
    E.long_description AS `pmc_long_description`
FROM
    `kernels` K
    INNER JOIN `pmc_events` E ON E.event_id = K.event_id;
