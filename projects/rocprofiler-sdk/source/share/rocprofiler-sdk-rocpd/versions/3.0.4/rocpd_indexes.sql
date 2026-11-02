--
-- Indexes for the various fields
--

-- string field
-- CREATE INDEX `rocpd_string{{uuid}}_string_idx` ON `rocpd_string{{uuid}}` ("string");

-- guid field
-- CREATE INDEX `rocpd_string{{uuid}}_guid_idx` ON `rocpd_string{{uuid}}` ("id", "guid");
-- CREATE INDEX `rocpd_info_node{{uuid}}_guid_idx` ON `rocpd_info_node{{uuid}}` ("id", "guid");
-- CREATE INDEX `rocpd_info_process{{uuid}}_guid_idx` ON `rocpd_info_process{{uuid}}` ("id", "guid");
-- CREATE INDEX `rocpd_info_thread{{uuid}}_guid_idx` ON `rocpd_info_thread{{uuid}}` ("id", "guid");
-- CREATE INDEX `rocpd_info_agent{{uuid}}_guid_idx` ON `rocpd_info_agent{{uuid}}` ("id", "guid");
-- CREATE INDEX `rocpd_info_queue{{uuid}}_guid_idx` ON `rocpd_info_queue{{uuid}}` ("id", "guid");
-- CREATE INDEX `rocpd_info_stream{{uuid}}_guid_idx` ON `rocpd_info_stream{{uuid}}` ("id", "guid");
-- CREATE INDEX `rocpd_info_pmc{{uuid}}_guid_idx` ON `rocpd_info_pmc{{uuid}}` ("id", "guid");
-- CREATE INDEX `rocpd_info_code_object{{uuid}}_guid_idx` ON `rocpd_info_code_object{{uuid}}` ("id", "guid");
-- CREATE INDEX `rocpd_info_kernel_symbol{{uuid}}_guid_idx` ON `rocpd_info_kernel_symbol{{uuid}}` ("id", "guid");
-- CREATE INDEX `rocpd_track{{uuid}}_guid_idx` ON `rocpd_track{{uuid}}` ("id", "guid");
-- CREATE INDEX `rocpd_event{{uuid}}_guid_idx` ON `rocpd_event{{uuid}}` ("id", "guid");
-- CREATE INDEX `rocpd_arg{{uuid}}_guid_idx` ON `rocpd_arg{{uuid}}` ("id", "guid");
-- CREATE INDEX `rocpd_pmc_event{{uuid}}_guid_idx` ON `rocpd_pmc_event{{uuid}}` ("id", "guid");
-- CREATE INDEX `rocpd_region{{uuid}}_guid_idx` ON `rocpd_region{{uuid}}` ("id", "guid");
-- CREATE INDEX `rocpd_sample{{uuid}}_guid_idx` ON `rocpd_sample{{uuid}}` ("id", "guid");
-- CREATE INDEX `rocpd_kernel_dispatch{{uuid}}_guid_idx` ON `rocpd_kernel_dispatch{{uuid}}` ("id", "guid");
-- CREATE INDEX `rocpd_memory_copy{{uuid}}_guid_idx` ON `rocpd_memory_copy{{uuid}}` ("id", "guid");
-- CREATE INDEX `rocpd_memory_allocate{{uuid}}_guid_idx` ON `rocpd_memory_allocate{{uuid}}` ("id", "guid");

-- CREATE INDEX `rocpd_event{{uuid}}_category_idx` ON `rocpd_event{{uuid}}` ("id", "guid", "category_id");
-- CREATE INDEX `rocpd_region{{uuid}}_event_idx` ON `rocpd_region{{uuid}}` ("id", "guid", "event_id");
-- CREATE INDEX `rocpd_region{{uuid}}_name_idx` ON `rocpd_region{{uuid}}` ("id", "guid", "name_id");
-- CREATE INDEX `rocpd_sample{{uuid}}_event_idx` ON `rocpd_sample{{uuid}}` ("id", "guid", "event_id");
-- CREATE INDEX `rocpd_sample{{uuid}}_track_idx` ON `rocpd_sample{{uuid}}` ("id", "guid", "track_id");
-- CREATE INDEX `rocpd_track{{uuid}}_name_idx` ON `rocpd_track{{uuid}}` ("id", "guid", "name_id");

-- CREATE INDEX `rocpd_memory_copy{{uuid}}_guid_nid_pid_idx` ON `rocpd_memory_copy{{uuid}}` ("guid", "nid", "pid");
-- CREATE INDEX `rocpd_kernel_dispatch{{uuid}}_guid_nid_pid_idx` ON `rocpd_kernel_dispatch{{uuid}}` ("guid", "nid", "pid");
-- CREATE INDEX `rocpd_region{{uuid}}_guid_idx` ON `rocpd_region{{uuid}}` ("guid", "nid", "pid");
-- CREATE INDEX `rocpd_sample{{uuid}}_guid_nid_pid_idx` ON `rocpd_sample{{uuid}}` ("guid", "nid", "pid");

-- CREATE INDEX `rocpd_region{{uuid}}_guid_idx` ON `rocpd_region{{uuid}}` ("guid");
-- CREATE INDEX `rocpd_region{{uuid}}_nid_idx` ON `rocpd_region{{uuid}}` ("nid");
-- CREATE INDEX `rocpd_region{{uuid}}_pid_idx` ON `rocpd_region{{uuid}}` ("pid");
-- CREATE INDEX `rocpd_region{{uuid}}_start_idx` ON `rocpd_region{{uuid}}` ("start");
-- CREATE INDEX `rocpd_region{{uuid}}_end_idx` ON `rocpd_region{{uuid}}` ("end");
