--
-- Indexes for the various fields
--

-- these have been verified to improve performance in perfetto
CREATE INDEX `rocpd_arg{{uuid}}_event_id_idx` ON `rocpd_arg{{uuid}}` ("event_id");
CREATE INDEX `rocpd_pmc_event{{uuid}}_event_id_idx` ON `rocpd_pmc_event{{uuid}}` ("event_id");
CREATE INDEX `rocpd_arg{{uuid}}_guid_event_id_idx` ON `rocpd_arg{{uuid}}` ("guid", "event_id");
CREATE INDEX `rocpd_pmc_event{{uuid}}_guid_event_id_idx` ON `rocpd_pmc_event{{uuid}}` ("guid", "event_id");

-- these are speculative
CREATE INDEX `rocpd_info_process{{uuid}}_pid_idx` ON `rocpd_info_process{{uuid}}` ("pid");
CREATE INDEX `rocpd_info_thread{{uuid}}_tid_idx` ON `rocpd_info_thread{{uuid}}` ("tid");
CREATE INDEX `rocpd_info_process{{uuid}}_guid_pid_idx` ON `rocpd_info_process{{uuid}}` ("guid", "pid");
CREATE INDEX `rocpd_info_thread{{uuid}}_guid_tid_idx` ON `rocpd_info_thread{{uuid}}` ("guid", "tid");
