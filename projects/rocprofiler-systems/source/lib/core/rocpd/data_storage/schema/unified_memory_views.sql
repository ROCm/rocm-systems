-- ===================================================================
-- Unified Memory Profiling SQL Views
-- ===================================================================
-- These views aggregate KFD event data for unified memory profiling
-- Features: migration statistics, page fault counting, bandwidth calculation
-- ===================================================================

-- ===================================================================
-- View 1: Migration Summary by Direction
-- ===================================================================
-- Aggregates page migration events by direction (Host→Device, Device→Host, Device→Device)
-- Groups by destination agent for per-device statistics
--
CREATE VIEW IF NOT EXISTS unified_memory_migration_summary{{uuid}} AS
SELECT
    -- Classify migration direction based on src/dst agent types
    CASE
        -- Host → Device (CPU to GPU)
        WHEN src_agent.type = 'CPU' AND dst_agent.type = 'GPU'
        THEN 'Host To Device'

        -- Device → Host (GPU to CPU)
        WHEN src_agent.type = 'GPU' AND dst_agent.type = 'CPU'
        THEN 'Device To Host'

        -- Device → Device (GPU to GPU)
        WHEN src_agent.type = 'GPU' AND dst_agent.type = 'GPU'
        THEN 'Device To Device'

        -- Fallback for unexpected combinations
        ELSE 'Unknown'
    END AS direction,

    -- Destination agent information
    dst_agent.id AS dst_agent_id,
    dst_agent.name AS dst_agent_name,
    dst_agent.type AS dst_agent_type,

    -- Source agent information (for reference)
    src_agent.id AS src_agent_id,
    src_agent.name AS src_agent_name,

    -- Migration statistics
    COUNT(*) AS count,
    AVG(pmc.value) AS avg_size_bytes,
    MIN(pmc.value) AS min_size_bytes,
    MAX(pmc.value) AS max_size_bytes,
    SUM(pmc.value) AS total_size_bytes,

    -- Time statistics (nanoseconds)
    SUM(r.end - r.start) AS total_time_ns,
    AVG(r.end - r.start) AS avg_time_ns,
    MIN(r.end - r.start) AS min_time_ns,
    MAX(r.end - r.start) AS max_time_ns,

    -- Bandwidth calculation (GB/s)
    -- Formula: (total_bytes / total_time_ns) * 1e9 / 1e9 = total_bytes / total_time_ns
    CASE
        WHEN SUM(r.end - r.start) > 0
        THEN (SUM(pmc.value) * 1.0) / SUM(r.end - r.start)
        ELSE 0
    END AS bandwidth_gbps

FROM rocpd_event{{uuid}} e

-- Join with category to filter kfd_page_migrate events
JOIN rocpd_string{{uuid}} category ON category.id = e.category_id

-- Join with region for timing information
JOIN rocpd_region{{uuid}} r ON r.event_id = e.id

-- Join with PMC events for migration size
JOIN rocpd_pmc_event{{uuid}} pmc ON pmc.event_id = e.id

-- Join with arguments to extract src_agent and dst_agent
JOIN rocpd_arg{{uuid}} arg_src ON arg_src.event_id = e.id AND arg_src.name = 'src_agent'
JOIN rocpd_arg{{uuid}} arg_dst ON arg_dst.event_id = e.id AND arg_dst.name = 'dst_agent'

-- Join with agent info to get agent types (CPU/GPU)
JOIN rocpd_info_agent{{uuid}} src_agent ON src_agent.name = arg_src.value
JOIN rocpd_info_agent{{uuid}} dst_agent ON dst_agent.name = arg_dst.value

WHERE category.string = 'kfd_page_migrate'

GROUP BY direction, dst_agent.id, dst_agent.name, dst_agent.type, src_agent.id, src_agent.name
ORDER BY direction, dst_agent.name;

-- ===================================================================
-- View 2: Page Fault Summary
-- ===================================================================
-- Aggregates page fault events by agent (CPU/GPU) and fault type (read/write)
--
CREATE VIEW IF NOT EXISTS unified_memory_page_faults{{uuid}} AS
SELECT
    -- Agent information
    agent.id AS agent_id,
    agent.name AS agent_name,
    agent.type AS agent_type,

    -- Total fault count
    COUNT(*) AS total_faults,

    -- Breakdown by fault type (based on event name)
    SUM(CASE WHEN name_str.string LIKE '%READ_FAULT%' THEN 1 ELSE 0 END) AS read_faults,
    SUM(CASE WHEN name_str.string LIKE '%WRITE_FAULT%' THEN 1 ELSE 0 END) AS write_faults,

    -- Breakdown by resolution type
    SUM(CASE WHEN name_str.string LIKE '%MIGRATED%' THEN 1 ELSE 0 END) AS faults_migrated,
    SUM(CASE WHEN name_str.string LIKE '%UPDATED%' THEN 1 ELSE 0 END) AS faults_updated,

    -- Timing statistics
    AVG(r.end - r.start) AS avg_fault_time_ns,
    MIN(r.end - r.start) AS min_fault_time_ns,
    MAX(r.end - r.start) AS max_fault_time_ns

FROM rocpd_event{{uuid}} e

-- Join with category to filter kfd_page_fault events
JOIN rocpd_string{{uuid}} category ON category.id = e.category_id

-- Join with region for timing
JOIN rocpd_region{{uuid}} r ON r.event_id = e.id

-- Join with event name for fault type classification
JOIN rocpd_string{{uuid}} name_str ON name_str.id = r.name_id

-- Join with arguments to extract agent
JOIN rocpd_arg{{uuid}} arg_agent ON arg_agent.event_id = e.id AND arg_agent.name = 'agent'

-- Join with agent info
JOIN rocpd_info_agent{{uuid}} agent ON agent.name = arg_agent.value

WHERE category.string = 'kfd_page_fault'

GROUP BY agent.id, agent.name, agent.type
ORDER BY agent.type, agent.name;

-- ===================================================================
-- View 3: Unified Memory Summary (Overall Statistics)
-- ===================================================================
-- Provides overall unified memory statistics across all agents
--
CREATE VIEW IF NOT EXISTS unified_memory_summary{{uuid}} AS
SELECT
    -- Total migration counts by direction
    SUM(CASE WHEN direction = 'Host To Device' THEN count ELSE 0 END) AS host_to_device_count,
    SUM(CASE WHEN direction = 'Device To Host' THEN count ELSE 0 END) AS device_to_host_count,
    SUM(CASE WHEN direction = 'Device To Device' THEN count ELSE 0 END) AS device_to_device_count,

    -- Total migration bytes by direction
    SUM(CASE WHEN direction = 'Host To Device' THEN total_size_bytes ELSE 0 END) AS host_to_device_bytes,
    SUM(CASE WHEN direction = 'Device To Host' THEN total_size_bytes ELSE 0 END) AS device_to_host_bytes,
    SUM(CASE WHEN direction = 'Device To Device' THEN total_size_bytes ELSE 0 END) AS device_to_device_bytes,

    -- Overall totals
    SUM(count) AS total_migrations,
    SUM(total_size_bytes) AS total_bytes_migrated,
    SUM(total_time_ns) AS total_migration_time_ns,

    -- Overall bandwidth
    CASE
        WHEN SUM(total_time_ns) > 0
        THEN (SUM(total_size_bytes) * 1.0) / SUM(total_time_ns)
        ELSE 0
    END AS overall_bandwidth_gbps

FROM unified_memory_migration_summary{{uuid}};

-- ===================================================================
-- View 4: Page Fault Summary by Type (CPU vs GPU)
-- ===================================================================
-- Provides aggregated page fault statistics separated by agent type
--
CREATE VIEW IF NOT EXISTS unified_memory_page_faults_by_type{{uuid}} AS
SELECT
    agent_type,
    SUM(total_faults) AS total_faults,
    SUM(read_faults) AS total_read_faults,
    SUM(write_faults) AS total_write_faults,
    SUM(faults_migrated) AS total_faults_migrated,
    SUM(faults_updated) AS total_faults_updated,
    AVG(avg_fault_time_ns) AS avg_fault_time_ns
FROM unified_memory_page_faults{{uuid}}
GROUP BY agent_type
ORDER BY agent_type;

-- ===================================================================
-- View 5: Migration Bandwidth Timeline
-- ===================================================================
-- Provides time-series data for migration bandwidth (useful for Perfetto counters)
--
CREATE VIEW IF NOT EXISTS unified_memory_migration_timeline{{uuid}} AS
SELECT
    r.start AS timestamp_ns,
    r.end - r.start AS duration_ns,
    pmc.value AS size_bytes,

    -- Calculate instantaneous bandwidth (GB/s)
    CASE
        WHEN (r.end - r.start) > 0
        THEN (pmc.value * 1.0) / (r.end - r.start)
        ELSE 0
    END AS bandwidth_gbps,

    -- Migration direction
    CASE
        WHEN src_agent.type = 'CPU' AND dst_agent.type = 'GPU'
        THEN 'Host To Device'
        WHEN src_agent.type = 'GPU' AND dst_agent.type = 'CPU'
        THEN 'Device To Host'
        WHEN src_agent.type = 'GPU' AND dst_agent.type = 'GPU'
        THEN 'Device To Device'
        ELSE 'Unknown'
    END AS direction,

    -- Agent information
    dst_agent.id AS dst_agent_id,
    dst_agent.name AS dst_agent_name,
    src_agent.name AS src_agent_name

FROM rocpd_event{{uuid}} e

JOIN rocpd_string{{uuid}} category ON category.id = e.category_id
JOIN rocpd_region{{uuid}} r ON r.event_id = e.id
JOIN rocpd_pmc_event{{uuid}} pmc ON pmc.event_id = e.id

JOIN rocpd_arg{{uuid}} arg_src ON arg_src.event_id = e.id AND arg_src.name = 'src_agent'
JOIN rocpd_arg{{uuid}} arg_dst ON arg_dst.event_id = e.id AND arg_dst.name = 'dst_agent'

JOIN rocpd_info_agent{{uuid}} src_agent ON src_agent.name = arg_src.value
JOIN rocpd_info_agent{{uuid}} dst_agent ON dst_agent.name = arg_dst.value

WHERE category.string = 'kfd_page_migrate'

ORDER BY r.start;

-- ===================================================================
-- View 6: Page Fault Timeline
-- ===================================================================
-- Provides time-series data for page faults (useful for Perfetto counters)
--
CREATE VIEW IF NOT EXISTS unified_memory_page_fault_timeline{{uuid}} AS
SELECT
    r.start AS timestamp_ns,
    r.end - r.start AS duration_ns,
    agent.id AS agent_id,
    agent.name AS agent_name,
    agent.type AS agent_type,

    -- Fault type
    CASE
        WHEN name_str.string LIKE '%READ_FAULT%' THEN 'Read'
        WHEN name_str.string LIKE '%WRITE_FAULT%' THEN 'Write'
        ELSE 'Unknown'
    END AS fault_type,

    -- Resolution type
    CASE
        WHEN name_str.string LIKE '%MIGRATED%' THEN 'Migrated'
        WHEN name_str.string LIKE '%UPDATED%' THEN 'Updated'
        ELSE 'Unknown'
    END AS resolution_type

FROM rocpd_event{{uuid}} e

JOIN rocpd_string{{uuid}} category ON category.id = e.category_id
JOIN rocpd_region{{uuid}} r ON r.event_id = e.id
JOIN rocpd_string{{uuid}} name_str ON name_str.id = r.name_id

JOIN rocpd_arg{{uuid}} arg_agent ON arg_agent.event_id = e.id AND arg_agent.name = 'agent'
JOIN rocpd_info_agent{{uuid}} agent ON agent.name = arg_agent.value

WHERE category.string = 'kfd_page_fault'

ORDER BY r.start;
