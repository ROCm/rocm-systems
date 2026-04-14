--
-- Indexes for the various fields
--

-- these have been verified to improve performance in perfetto
CREATE INDEX `rocpd_arg{{uuid}}_event_id_idx` ON `rocpd_arg{{uuid}}` ("event_id");
CREATE INDEX `rocpd_pmc_event{{uuid}}_event_id_idx` ON `rocpd_pmc_event{{uuid}}` ("event_id");
CREATE INDEX `rocpd_arg{{uuid}}_guid_event_id_idx` ON `rocpd_arg{{uuid}}` ("guid", "event_id");
CREATE INDEX `rocpd_pmc_event{{uuid}}_guid_event_id_idx` ON `rocpd_pmc_event{{uuid}}` ("guid", "event_id");
CREATE INDEX `rocpd_event{{uuid}}_category_id_idx` ON `rocpd_event{{uuid}}` ("category_id");

-- these are speculative
CREATE INDEX `rocpd_info_process{{uuid}}_pid_idx` ON `rocpd_info_process{{uuid}}` ("pid");
CREATE INDEX `rocpd_info_thread{{uuid}}_tid_idx` ON `rocpd_info_thread{{uuid}}` ("tid");
CREATE INDEX `rocpd_info_process{{uuid}}_guid_pid_idx` ON `rocpd_info_process{{uuid}}` ("guid", "pid");
CREATE INDEX `rocpd_info_thread{{uuid}}_guid_tid_idx` ON `rocpd_info_thread{{uuid}}` ("guid", "tid");
CREATE INDEX `rocpd_timestamp{{uuid}}_value_idx` ON `rocpd_timestamp{{uuid}}` ("value");
CREATE INDEX `rocpd_timestamp{{uuid}}_track_id_idx` ON `rocpd_timestamp{{uuid}}` ("track_id");

-- CREATE INDEX `rocpd_kernel_dispatch{{uuid}}_guid_pid_tid_idx` ON `rocpd_kernel_dispatch{{uuid}}` ("guid", "pid", "tid");
CREATE INDEX `rocpd_memory_copy{{uuid}}_guid_pid_tid_idx` ON `rocpd_memory_copy{{uuid}}` ("guid", "pid", "tid");
-- CREATE INDEX `rocpd_memory_allocate{{uuid}}_guid_pid_tid_idx` ON `rocpd_memory_allocate{{uuid}}` ("guid", "pid", "tid");
