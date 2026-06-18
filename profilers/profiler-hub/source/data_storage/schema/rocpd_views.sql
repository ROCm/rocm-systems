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
