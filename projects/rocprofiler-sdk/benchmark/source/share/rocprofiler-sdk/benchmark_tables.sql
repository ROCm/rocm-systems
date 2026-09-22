-- Application used for benchmarking
-- Columns such "hip_compiler_api", ..., "scratch_memory" are
-- the number of events in the given category, e.g. kernel_dispatch
-- represents the number of kernel dispatches in the app. These
-- can be approximate since for a given application, the exact
-- count may vary.
CREATE TABLE IF NOT EXISTS
    `benchmarked_app` (
        id INT PRIMARY KEY AUTO_INCREMENT UNIQUE,
        hash_id TEXT NOT NULL,
        md5sum TEXT NOT NULL,
        revision INT DEFAULT 0,
        command JSON NOT NULL,
        compiler_id TEXT,
        compiler_version TEXT,
        library_arch TEXT,
        system_name TEXT,
        system_processor TEXT,
        system_version TEXT,
        threads INT,
        hip_compiler_api INT,
        hip_runtime_api INT,
        hsa_api INT,
        kernel_dispatch INT,
        marker_api INT,
        memory_allocation INT,
        memory_copy INT,
        ompt INT,
        rccl_api INT,
        rocdecode_api INT,
        rocjpeg_api INT,
        scratch_memory INT,
        environment JSON DEFAULT ("{}")
    );

-- rocprofiler-sdk used for benchmarking
CREATE TABLE IF NOT EXISTS
    `benchmarked_sdk` (
        id INT PRIMARY KEY AUTO_INCREMENT UNIQUE,
        hash_id TEXT NOT NULL,
        version_major INT NOT NULL,
        version_minor INT NOT NULL,
        version_patch INT NOT NULL,
        soversion INT NOT NULL,
        compiler_id TEXT NOT NULL,
        compiler_version TEXT NOT NULL,
        git_revision TEXT NOT NULL,
        library_arch TEXT NOT NULL,
        system_name TEXT NOT NULL,
        system_processor TEXT NOT NULL,
        system_version TEXT NOT NULL
    );

-- rocprofiler-sdk used for benchmarking
CREATE TABLE IF NOT EXISTS
    `benchmark_config` (
        id INT PRIMARY KEY AUTO_INCREMENT UNIQUE,
        hash_id TEXT NOT NULL,
        sdk_id INT,
        label TEXT, -- name identifier
        benchmark_mode TEXT CHECK (
            benchmark_mode IN (
                "baseline",
                "disabled-sdk-contexts",
                "sdk-buffer-overhead",
                "sdk-callback-overhead",
                "tool-runtime-overhead"
            )
        ) NOT NULL,
        kernel_rename INT,
        group_by_queue INT,
        kernel_trace INT,
        hsa_trace INT,
        hip_runtime_trace INT,
        hip_compiler_trace INT,
        marker_trace INT,
        memory_copy_trace INT,
        memory_allocation_trace INT,
        scratch_memory_trace INT,
        dispatch_counter_collection INT,
        rccl_trace INT,
        rocdecode_trace INT,
        rocjpeg_trace INT,
        pmc_counters JSON DEFAULT ("[]"),
        pc_sampling_host_trap INT,
        pc_sampling_stocastic INT,
        advanced_thread_trace INT,
        --
        -- Eventually, we will create tables for storing the subconfigurations for pc sampling and ATT
        --
        -- pc_sampling_host_trap_config_id INT,
        -- pc_sampling_stocastic_config_id INT,
        -- advanced_thread_trace_config_id INT,
        -- FOREIGN KEY (pc_sampling_host_trap_config_id) REFERENCES benchmark_pc_sampling_host_trap_config (id) ON UPDATE CASCADE,
        -- FOREIGN KEY (pc_sampling_stocastic_config_id) REFERENCES benchmark_pc_sampling_stocastic_config (id) ON UPDATE CASCADE,
        -- FOREIGN KEY (advanced_thread_trace_config_id) REFERENCES benchmark_advanced_thread_trace_config (id) ON UPDATE CASCADE
        FOREIGN KEY (sdk_id) REFERENCES benchmarked_sdk (id) ON UPDATE CASCADE
    );

-- metrics for the benchmark
CREATE TABLE IF NOT EXISTS
    `benchmark_metrics` (
        id INT PRIMARY KEY AUTO_INCREMENT UNIQUE,
        app_id INT NOT NULL,
        cfg_id INT NOT NULL,
        sdk_id INT,
        executed_at TIMESTAMP NOT NULL,
        wall_time DOUBLE NOT NULL,
        cpu_time DOUBLE NOT NULL,
        cpu_util DOUBLE NOT NULL,
        peak_rss DOUBLE NOT NULL,
        page_rss DOUBLE NOT NULL,
        virtual_memory DOUBLE NOT NULL,
        major_page_faults BIGINT NOT NULL,
        minor_page_faults BIGINT NOT NULL,
        priority_context_switches BIGINT NOT NULL,
        voluntary_context_switches BIGINT NOT NULL,
        FOREIGN KEY (app_id) REFERENCES benchmarked_app (id) ON UPDATE CASCADE,
        FOREIGN KEY (cfg_id) REFERENCES benchmark_config (id) ON UPDATE CASCADE,
        FOREIGN KEY (sdk_id) REFERENCES benchmarked_sdk (id) ON UPDATE CASCADE
    );

CREATE TABLE IF NOT EXISTS
    `benchmark_statistics` (
        id INT PRIMARY KEY AUTO_INCREMENT UNIQUE,
        app_id INT NOT NULL,
        cfg_id INT NOT NULL,
        sdk_id INT,
        metric_name TEXT NOT NULL,
        metric_unit TEXT NOT NULL,
        count INT NOT NULL,
        sum DOUBLE NOT NULL,
        mean DOUBLE NOT NULL,
        min DOUBLE NOT NULL,
        max DOUBLE NOT NULL,
        std_dev DOUBLE
    );
