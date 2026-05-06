CREATE VIEW IF NOT EXISTS
    `rocpd_metadata` AS
SELECT
    "id",
    "tag",
    "value"
FROM
    `rocpd_metadata{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_string` AS
SELECT
    "id",
    "guid",
    "string"
FROM
    `rocpd_string{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_info_node` AS
SELECT
    "id",
    "guid",
    "hash",
    "machine_id",
    "system_name",
    "hostname",
    "release",
    "version",
    "hardware_name",
    "domain_name"
FROM
    `rocpd_info_node{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_info_process` AS
SELECT
    "id",
    "guid",
    "nid",
    "ppid",
    "pid",
    "init",
    "fini",
    "start",
    "end",
    "command",
    "environment",
    CAST(lz4_decompress("extdata") AS TEXT) AS "extdata"
FROM
    `rocpd_info_process{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_info_thread` AS
SELECT
    "id",
    "guid",
    "nid",
    "ppid",
    "pid",
    "tid",
    "name",
    "start",
    "end",
    CAST(lz4_decompress("extdata") AS TEXT) AS "extdata"
FROM
    `rocpd_info_thread{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_info_agent` AS
SELECT
    "id",
    "guid",
    "nid",
    "pid",
    "type",
    "absolute_index",
    "logical_index",
    "type_index",
    "uuid",
    "name",
    "model_name",
    "vendor_name",
    "product_name",
    "user_name",
    CAST(lz4_decompress("extdata") AS TEXT) AS "extdata"
FROM
    `rocpd_info_agent{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_info_queue` AS
SELECT
    "id",
    "guid",
    "nid",
    "pid",
    "name",
    CAST(lz4_decompress("extdata") AS TEXT) AS "extdata"
FROM
    `rocpd_info_queue{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_info_stream` AS
SELECT
    "id",
    "guid",
    "nid",
    "pid",
    "name",
    CAST(lz4_decompress("extdata") AS TEXT) AS "extdata"
FROM
    `rocpd_info_stream{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_info_pmc` AS
SELECT
    "id",
    "guid",
    "nid",
    "pid",
    "agent_id",
    "target_arch",
    "event_code",
    "instance_id",
    "name",
    "symbol",
    "description",
    "long_description",
    "component",
    "units",
    "value_type",
    "block",
    "expression",
    "is_constant",
    "is_derived",
    CAST(lz4_decompress("extdata") AS TEXT) AS "extdata"
FROM
    `rocpd_info_pmc{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_info_code_object` AS
SELECT
    "id",
    "guid",
    "nid",
    "pid",
    "agent_id",
    "uri",
    "load_base",
    "load_size",
    "load_delta",
    "storage_type",
    CAST(lz4_decompress("extdata") AS TEXT) AS "extdata"
FROM
    `rocpd_info_code_object{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_info_kernel_symbol` AS
SELECT
    "id",
    "guid",
    "nid",
    "pid",
    "code_object_id",
    "kernel_name",
    "display_name",
    "kernel_object",
    "kernarg_segment_size",
    "kernarg_segment_alignment",
    "group_segment_size",
    "private_segment_size",
    "sgpr_count",
    "arch_vgpr_count",
    "accum_vgpr_count",
    CAST(lz4_decompress("extdata") AS TEXT) AS "extdata"
FROM
    `rocpd_info_kernel_symbol{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_track` AS
SELECT
    "id",
    "guid",
    "nid",
    "pid",
    "tid",
    "name_id",
    CAST(lz4_decompress("extdata") AS TEXT) AS "extdata"
FROM
    `rocpd_track{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_event` AS
SELECT
    "id",
    "guid",
    "category_id",
    "stack_id",
    "parent_stack_id",
    "correlation_id",
    CAST(lz4_decompress("call_stack") AS TEXT) AS "call_stack",
    CAST(lz4_decompress("line_info") AS TEXT) AS "line_info",
    CAST(lz4_decompress("extdata") AS TEXT) AS "extdata"
FROM
    `rocpd_event{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_arg` AS
SELECT
    "id",
    "guid",
    "event_id",
    "position",
    "type",
    "name",
    "value",
    CAST(lz4_decompress("extdata") AS TEXT) AS "extdata"
FROM
    `rocpd_arg{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_pmc_event` AS
SELECT
    "id",
    "guid",
    "event_id",
    "pmc_id",
    "value",
    CAST(lz4_decompress("extdata") AS TEXT) AS "extdata"
FROM
    `rocpd_pmc_event{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_region` AS
SELECT
    "id",
    "guid",
    "nid",
    "pid",
    "tid",
    "start",
    "end",
    "name_id",
    "event_id",
    CAST(lz4_decompress("extdata") AS TEXT) AS "extdata"
FROM
    `rocpd_region{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_sample` AS
SELECT
    "id",
    "guid",
    "track_id",
    "timestamp",
    "event_id",
    CAST(lz4_decompress("extdata") AS TEXT) AS "extdata"
FROM
    `rocpd_sample{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_kernel_dispatch` AS
SELECT
    "id",
    "guid",
    "nid",
    "pid",
    "tid",
    "agent_id",
    "kernel_id",
    "dispatch_id",
    "queue_id",
    "stream_id",
    "start",
    "end",
    "private_segment_size",
    "group_segment_size",
    "workgroup_size_x",
    "workgroup_size_y",
    "workgroup_size_z",
    "grid_size_x",
    "grid_size_y",
    "grid_size_z",
    "region_name_id",
    "event_id",
    CAST(lz4_decompress("extdata") AS TEXT) AS "extdata"
FROM
    `rocpd_kernel_dispatch{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_memory_copy` AS
SELECT
    "id",
    "guid",
    "nid",
    "pid",
    "tid",
    "start",
    "end",
    "name_id",
    "dst_agent_id",
    "dst_address",
    "src_agent_id",
    "src_address",
    "size",
    "queue_id",
    "stream_id",
    "region_name_id",
    "event_id",
    CAST(lz4_decompress("extdata") AS TEXT) AS "extdata"
FROM
    `rocpd_memory_copy{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_memory_allocate` AS
SELECT
    "id",
    "guid",
    "nid",
    "pid",
    "tid",
    "agent_id",
    "type",
    "level",
    "start",
    "end",
    "address",
    "size",
    "queue_id",
    "stream_id",
    "event_id",
    CAST(lz4_decompress("extdata") AS TEXT) AS "extdata"
FROM
    `rocpd_memory_allocate{{uuid}}`;
