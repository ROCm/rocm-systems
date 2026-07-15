CREATE VIEW IF NOT EXISTS
    `rocpd_metadata` AS
SELECT
    *
FROM
    `rocpd_metadata{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_string` AS
SELECT
    *
FROM
    `rocpd_string{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_info_node` AS
SELECT
    *
FROM
    `rocpd_info_node{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_info_process` AS
SELECT
    *
FROM
    `rocpd_info_process{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_info_thread` AS
SELECT
    *
FROM
    `rocpd_info_thread{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_info_agent` AS
SELECT
    *
FROM
    `rocpd_info_agent{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_info_queue` AS
SELECT
    *
FROM
    `rocpd_info_queue{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_info_stream` AS
SELECT
    *
FROM
    `rocpd_info_stream{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_info_pmc` AS
SELECT
    *
FROM
    `rocpd_info_pmc{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_info_blob_schema` AS
SELECT
    *
FROM
    `rocpd_info_blob_schema{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_info_blob_field` AS
SELECT
    *
FROM
    `rocpd_info_blob_field{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_info_code_object` AS
SELECT
    *
FROM
    `rocpd_info_code_object{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_info_kernel_symbol` AS
SELECT
    *
FROM
    `rocpd_info_kernel_symbol{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_track` AS
SELECT
    *
FROM
    `rocpd_track{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_event` AS
SELECT
    *
FROM
    `rocpd_event{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_arg` AS
SELECT
    *
FROM
    `rocpd_arg{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_pmc_event` AS
SELECT
    *
FROM
    `rocpd_pmc_event{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_region` AS
SELECT
    *
FROM
    `rocpd_region{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_sample` AS
SELECT
    *
FROM
    `rocpd_sample{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_kernel_dispatch` AS
SELECT
    *
FROM
    `rocpd_kernel_dispatch{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_memory_copy` AS
SELECT
    *
FROM
    `rocpd_memory_copy{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_memory_allocate` AS
SELECT
    *
FROM
    `rocpd_memory_allocate{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_graph_launch` AS
SELECT
    *
FROM
    `rocpd_graph_launch{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_gpu_pc_sample` AS
SELECT
    *
FROM
    `rocpd_gpu_pc_sample{{uuid}}`;
-- Human-readable name expansion for instruction-type and stall-reason integers.
-- Joins against nothing: the CASE expressions are pure look-ups derived from
-- the rocprofiler_pc_sampling_instruction_type_t and
-- rocprofiler_pc_sampling_instruction_not_issued_reason_t enumerations so that
-- the per-row table stays lean while names remain available at query time.
CREATE VIEW IF NOT EXISTS
    `rocpd_gpu_pc_sample_named` AS
SELECT
    *,
    CASE "inst_type"
        WHEN 0  THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_NONE'
        WHEN 1  THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_VALU'
        WHEN 2  THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_MATRIX'
        WHEN 3  THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_SCALAR'
        WHEN 4  THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_TEX'
        WHEN 5  THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_LDS'
        WHEN 6  THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_LDS_DIRECT'
        WHEN 7  THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_FLAT'
        WHEN 8  THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_EXPORT'
        WHEN 9  THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_MESSAGE'
        WHEN 10 THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BARRIER'
        WHEN 11 THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BRANCH_NOT_TAKEN'
        WHEN 12 THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BRANCH_TAKEN'
        WHEN 13 THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_JUMP'
        WHEN 14 THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_OTHER'
        WHEN 15 THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_NO_INST'
        WHEN 16 THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_DUAL_VALU'
        ELSE NULL
    END AS "inst_type_name",
    CASE "stall_reason"
        WHEN 0 THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NONE'
        WHEN 1 THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE'
        WHEN 2 THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ALU_DEPENDENCY'
        WHEN 3 THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_WAITCNT'
        WHEN 4 THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_INTERNAL_INSTRUCTION'
        WHEN 5 THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_BARRIER_WAIT'
        WHEN 6 THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN'
        WHEN 7 THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_WIN_EX_STALL'
        WHEN 8 THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_OTHER_WAIT'
        WHEN 9 THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_SLEEP_WAIT'
        ELSE NULL
    END AS "stall_reason_name"
FROM
    `rocpd_gpu_pc_sample`;


CREATE VIEW IF NOT EXISTS
    `rocpd_blob_event` AS
SELECT
    *
FROM
    `rocpd_blob_event{{uuid}}`;

CREATE VIEW IF NOT EXISTS
    `rocpd_disassembly_data` AS
SELECT
    *
FROM
    `rocpd_disassembly_data{{uuid}}`;
