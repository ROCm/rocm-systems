-- RCCL Test Runner -> results dashboard schema
--
-- Shared by two writers:
--   1. test_runner.py --db-push  (best-effort direct push from the test host)
--   2. the dashboard-side ingest job (reads scraped *.tar.gz -> upserts here)
--
-- Both writers are idempotent on runs.run_id, so re-ingesting a tarball that a
-- direct push already landed is a no-op. run_id is generated once per test-runner
-- invocation (see lib/results_emitter.py), so it is stable across both paths.
--
-- Apply with:  psql "$RCCL_RESULTS_DSN" -f schema.sql

CREATE TABLE IF NOT EXISTS runs (
    run_id           TEXT PRIMARY KEY,
    run_number       BIGINT,             -- immutable, monotonic, DB-assigned on first insert
    official         BOOLEAN     NOT NULL DEFAULT false,  -- CI/trusted source -> gets CI badge on timelines
    schema_version   INTEGER     NOT NULL DEFAULT 1,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at       TIMESTAMPTZ,
    finished_at      TIMESTAMPTZ,
    rccl_sha         TEXT,
    rccl_branch      TEXT,
    rocm_version     TEXT,
    host             TEXT,               -- controller hostname
    hosts            JSONB,              -- MPI hosts (host_list / hostfile contents)
    gpu_arch         TEXT,               -- e.g. gfx942, gfx90a, gfx1100
    node_names       JSONB,              -- specific compute node hostnames used by the job
    num_nodes        INTEGER,
    gpus_per_node    INTEGER,
    config_name      TEXT,               -- system_configurations.name
    config_description TEXT,
    runner_version   TEXT,
    label            TEXT,               -- optional --run-label
    tags             TEXT[],             -- --tag/--tags at emit time; mutable only by dashboard admins afterwards
    env              JSONB,              -- global env fingerprint
    metadata         JSONB,              -- rich host/telemetry snapshot (versions, GPU BDFs/CUs, firmware, UALink station mask, ...)
    summary          JSONB,              -- {total,passed,failed,timeout,skipped,duration_s}
    ingested_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS test_results (
    id            BIGSERIAL PRIMARY KEY,
    run_id        TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    suite         TEXT,
    test_name     TEXT NOT NULL,
    test_binary   TEXT,
    is_gtest      BOOLEAN,
    num_nodes     INTEGER,
    gpus_per_node INTEGER,
    num_ranks     INTEGER,
    dtype         TEXT,
    exec_mode     TEXT,                  -- mpi | threaded | single
    nthreads      INTEGER,
    status        TEXT NOT NULL,         -- PASSED | FAILED | TIMEOUT | SKIPPED
    duration_s    DOUBLE PRECISION,
    exit_code     INTEGER,
    error         TEXT,
    log_relpath   TEXT,                  -- path inside the run tarball
    UNIQUE (run_id, suite, test_name)
);

CREATE TABLE IF NOT EXISTS perf_rows (
    id             BIGSERIAL PRIMARY KEY,
    run_id         TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    test_result_id BIGINT REFERENCES test_results(id) ON DELETE CASCADE,
    collective     TEXT,                 -- derived from binary (e.g. all_reduce)
    size_bytes     BIGINT,
    count_elements BIGINT,
    dtype          TEXT,
    redop          TEXT,
    root           TEXT,
    place          TEXT,                 -- 'out_of_place' | 'in_place'
    time_us        DOUBLE PRECISION,
    algbw_gbs      DOUBLE PRECISION,
    busbw_gbs      DOUBLE PRECISION,
    wrong          TEXT,
    avg_busbw_gbs  DOUBLE PRECISION      -- run/test-level "Avg bus bandwidth" if present
);

CREATE TABLE IF NOT EXISTS coverage (
    run_id         TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
    regions_pct    DOUBLE PRECISION,
    functions_pct  DOUBLE PRECISION,
    lines_pct      DOUBLE PRECISION,
    branches_pct   DOUBLE PRECISION,
    raw            JSONB                  -- full parsed TOTAL line / extra metrics
);

CREATE TABLE IF NOT EXISTS file_coverage (
    id             BIGSERIAL PRIMARY KEY,
    run_id         TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    filename       TEXT NOT NULL,        -- repo-relative RCCL source path
    regions_pct    DOUBLE PRECISION,
    lines_pct      DOUBLE PRECISION,
    branches_pct   DOUBLE PRECISION,
    functions_pct  DOUBLE PRECISION,
    lines_covered  INTEGER,
    lines_total    INTEGER,
    regions_covered INTEGER,
    regions_total  INTEGER,
    branches_covered INTEGER,
    branches_total INTEGER,
    UNIQUE (run_id, filename)
);

CREATE INDEX IF NOT EXISTS idx_file_coverage_run      ON file_coverage(run_id);
CREATE INDEX IF NOT EXISTS idx_file_coverage_file     ON file_coverage(filename);
CREATE INDEX IF NOT EXISTS idx_test_results_run       ON test_results(run_id);
CREATE INDEX IF NOT EXISTS idx_perf_rows_run          ON perf_rows(run_id);
CREATE INDEX IF NOT EXISTS idx_perf_rows_collective   ON perf_rows(collective, size_bytes);
CREATE INDEX IF NOT EXISTS idx_runs_created_at        ON runs(created_at);
CREATE INDEX IF NOT EXISTS idx_runs_sha               ON runs(rccl_sha);
CREATE INDEX IF NOT EXISTS idx_runs_host              ON runs(host);
CREATE INDEX IF NOT EXISTS idx_runs_gpu_arch          ON runs(gpu_arch);
CREATE INDEX IF NOT EXISTS idx_runs_tags              ON runs USING GIN (tags);

-- Immutable, monotonic run number: DB-assigned on first insert (writers never
-- set it), preserved across idempotent upserts, and protected from any change.
CREATE SEQUENCE IF NOT EXISTS runs_run_number_seq;
ALTER TABLE runs ALTER COLUMN run_number SET DEFAULT nextval('runs_run_number_seq');
CREATE UNIQUE INDEX IF NOT EXISTS idx_runs_run_number ON runs(run_number);

CREATE OR REPLACE FUNCTION runs_run_number_immutable() RETURNS trigger AS $$
BEGIN
    IF NEW.run_number IS DISTINCT FROM OLD.run_number THEN
        RAISE EXCEPTION 'run_number is immutable once assigned (run_id=%)', OLD.run_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_runs_run_number_immutable ON runs;
CREATE TRIGGER trg_runs_run_number_immutable
    BEFORE UPDATE ON runs FOR EACH ROW
    EXECUTE FUNCTION runs_run_number_immutable();
