#!/usr/bin/env python3
"""085 fixture property checker — verifies a candidate replacement for tests/unit/rocpd_v4.db.

Usage:  python3 checker.py /path/to/candidate.db
Exit 0 = every HARD predicate passed. Exit 1 = at least one failed.

Every predicate below is derived from a specific assertion in a dependent test.
The suffix is discovered, not hardcoded, EXCEPT in H2 where the literal value is
itself the thing under test (database_test asserts the string).
"""
import sqlite3
import sys

DB = sys.argv[1] if len(sys.argv) > 1 else '/tmp/t085/rocpd_v4.db'
con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
cur = con.cursor()

results = []


def check(cid, tests, desc, sql, expected, params=()):
    """Run one predicate. `expected` is compared against the single returned row
    (tuple) or, for 1-column results, the scalar."""
    try:
        cur.execute(sql, params)
        row = cur.fetchone()
        got = row[0] if (row is not None and len(row) == 1) else row
    except Exception as exc:                      # missing table/view is a failure, not a crash
        got = f'ERROR: {type(exc).__name__}: {exc}'
    ok = (got == expected)
    results.append((ok, cid, tests, desc, expected, got))
    return ok


# --- suffix discovery (mirrors database_backend::discover_uuids) ---------------
cur.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='rocpd_info_node'")
has_view = cur.fetchone()[0] > 0
if has_view:
    cur.execute("SELECT DISTINCT replace(guid,'-','_') FROM rocpd_info_node")
else:
    cur.execute("SELECT DISTINCT substr(name, length('rocpd_info_node_')+1) "
                "FROM sqlite_master WHERE type='table' "
                "AND name LIKE 'rocpd\\_info\\_node\\_%' ESCAPE '\\'")
uuids = [r[0] for r in cur.fetchall()]
U = uuids[0] if len(uuids) == 1 else None
S = f'_{U}' if U else '_UNRESOLVED'

results.append((len(uuids) == 1, 'H1',
                'all 20',
                'discover_uuids() yields exactly ONE uuid (create() only adopts it when '
                'uuids.size()==1; otherwise every suffixed table name is wrong)',
                1, len(uuids)))

results.append((U == '00001eca_d4de_74de_b70e_c34ecf8c3a87', 'H2',
                'database_test.discover_uuids_recovers_full_hyphenated_v4_uuid',
                'the discovered uuid is the full underscore-joined hyphenated GUID '
                '(and is NOT the truncated last segment "c34ecf8c3a87")',
                '00001eca_d4de_74de_b70e_c34ecf8c3a87', U))

# --- H3: backend selection -----------------------------------------------------
check('H3a', 'all 19 reader tests',
      'a rocpd_metadata object is reachable (unsuffixed view or suffixed table); '
      'reader_impl.cpp:116-119 probes in that order',
      "SELECT (SELECT COUNT(*) FROM sqlite_master WHERE name='rocpd_metadata') "
      f"+ (SELECT COUNT(*) FROM sqlite_master WHERE name='rocpd_metadata{S}') > 0", 1)

check('H3b', 'all 19 reader tests',
      'rocpd_metadata is a key/value table and tag=\'schema_version\' has major version 4 '
      '-> reader selects the v4 read backend (m_is_v4)',
      f"SELECT CAST(substr(value,1,instr(value,'.')-1) AS INTEGER) "
      f"FROM rocpd_metadata{S} WHERE tag='schema_version'", 4)

# --- H4: every suffixed table the v4 backend prepares against must exist -------
V4_TABLES = [
    'rocpd_arg', 'rocpd_call_stack', 'rocpd_event', 'rocpd_info_address_range',
    'rocpd_info_agent', 'rocpd_info_category', 'rocpd_info_code_object',
    'rocpd_info_kernel_symbol', 'rocpd_info_node', 'rocpd_info_pc', 'rocpd_info_pmc',
    'rocpd_info_process', 'rocpd_info_queue', 'rocpd_info_source_code',
    'rocpd_info_stream', 'rocpd_info_thread', 'rocpd_kernel_dispatch',
    'rocpd_line_info', 'rocpd_memory_allocate', 'rocpd_memory_copy',
    'rocpd_pmc_event', 'rocpd_region', 'rocpd_sample', 'rocpd_string',
    'rocpd_timestamp', 'rocpd_track',
]
names = ','.join("'" + t + S + "'" for t in V4_TABLES)
check('H4', 'all 19 reader tests',
      'all 26 suffixed tables the v4 read backend prepares statements against exist '
      '(preparing against a missing table throws in the reader ctor -> every test fails)',
      f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN ({names})",
      len(V4_TABLES))

# --- G2: track shape -----------------------------------------------------------
check('H5a', 'v4_track_classification_and_identity',
      'exactly 4 rocpd_track rows',
      f'SELECT COUNT(*) FROM rocpd_track{S}', 4)

check('H5b', 'v4_track_classification_and_identity / all cpu_thread tests',
      'exactly 1 rocpd_track row with queue_id IS NULL AND stream_id IS NULL '
      '-> exactly one cpu_thread track (build_v4_tracks final else-branch)',
      f'SELECT COUNT(*) FROM rocpd_track{S} '
      'WHERE queue_id IS NULL AND stream_id IS NULL', 1)

check('H5c', 'v4_track_classification_and_identity / all gpu_queue tests',
      'exactly 1 rocpd_track row with queue_id NOT NULL -> exactly one gpu_queue track',
      f'SELECT COUNT(*) FROM rocpd_track{S} WHERE queue_id IS NOT NULL', 1)

check('H5d', 'v4_track_classification_and_identity / all dma tests',
      'exactly 2 rocpd_track rows with queue_id IS NULL AND stream_id NOT NULL '
      '-> exactly two dma tracks',
      f'SELECT COUNT(*) FROM rocpd_track{S} '
      'WHERE queue_id IS NULL AND stream_id IS NOT NULL', 2)

check('H6', 'v4_track_classification_and_identity, stream tests',
      'exactly 1 DISTINCT (nid,pid,stream_id) with stream_id NOT NULL '
      '-> exactly one synthesized stream track (tracks.size()==5 total)',
      f'SELECT COUNT(*) FROM (SELECT DISTINCT nid,pid,stream_id FROM rocpd_track{S} '
      'WHERE stream_id IS NOT NULL)', 1)

check('H7a', 'v4_track_classification_and_identity, v4_get_scalar_track_on_interval_track'
             '_returns_empty, v4_non_ambiguous_track_not_flagged',
      'no counter tracks: zero rocpd_sample rows join rocpd_pmc_event on event_id '
      '(a counter classification would change the track census and the type mix)',
      f'SELECT COUNT(DISTINCT s.track_id) FROM rocpd_sample{S} s '
      f'JOIN rocpd_pmc_event{S} pe ON pe.event_id = s.event_id', 0)

check('H7b', 'v4_track_classification_and_identity, v4_non_ambiguous_track_not_flagged',
      'no memory tracks: zero rocpd_track ids referenced by rocpd_memory_allocate',
      f'SELECT COUNT(DISTINCT track_id) FROM rocpd_memory_allocate{S}', 0)

check('H7c', 'v4_non_ambiguous_track_not_flagged',
      'no track_id appears in BOTH the counter and the memory-allocate discovery set '
      '-> ambiguous_classification stays false on every track',
      f'SELECT COUNT(*) FROM (SELECT DISTINCT s.track_id FROM rocpd_sample{S} s '
      f'JOIN rocpd_pmc_event{S} pe ON pe.event_id=s.event_id INTERSECT '
      f'SELECT DISTINCT track_id FROM rocpd_memory_allocate{S})', 0)

check('H8a', 'selects_v4_backend_on_underscore_joined_hyphenated_suffix, '
             'v4_track_classification_and_identity, v4_gpu_queue_track_carries_agent_id',
      "the gpu_queue track's agent_id resolves in rocpd_info_agent to id=6, "
      "type='GPU', name='AMD Instinct MI300X'",
      f'SELECT a.id, a.type, a.name FROM rocpd_track{S} t '
      f'JOIN rocpd_info_agent{S} a ON a.id = t.agent_id '
      'WHERE t.queue_id IS NOT NULL', (6, 'GPU', 'AMD Instinct MI300X'))

check('H8b', 'v4_track_classification_and_identity',
      "the gpu_queue track's queue_id resolves in rocpd_info_queue to name='Queue 0'",
      f'SELECT q.name FROM rocpd_track{S} t '
      f'JOIN rocpd_info_queue{S} q ON q.id = t.queue_id '
      'WHERE t.queue_id IS NOT NULL', 'Queue 0')

check('H9', 'v4_track_classification_and_identity',
      'the 2 dma tracks resolve to agents of BOTH types: one GPU-side, one CPU-side',
      f"SELECT SUM(a.type='GPU'), SUM(a.type='CPU') FROM rocpd_track{S} t "
      f'JOIN rocpd_info_agent{S} a ON a.id = t.agent_id '
      'WHERE t.queue_id IS NULL AND t.stream_id IS NOT NULL', (1, 1))

# --- G3: cpu_thread interval spine --------------------------------------------
CPU = (f'SELECT r.id, ts_s.value AS s, ts_e.value AS e, r.name_id, IC.name AS cat '
       f'FROM rocpd_region{S} r '
       f'JOIN rocpd_timestamp{S} ts_s ON ts_s.id = r.start_id '
       f'JOIN rocpd_timestamp{S} ts_e ON ts_e.id = r.end_id '
       f'LEFT JOIN rocpd_event{S} E ON E.id = r.event_id '
       f'LEFT JOIN rocpd_info_category{S} IC ON IC.id = E.category_id '
       f'WHERE r.track_id = (SELECT id FROM rocpd_track{S} '
       f'                    WHERE queue_id IS NULL AND stream_id IS NULL)')

check('H10', 'v4_get_interval_track_cpu_thread_regions, _carries_category, '
             'selects_v4_backend..., v4_get_track_stats_matches_slices_for_interval_tracks',
      'the cpu_thread track carries exactly 384 rocpd_region rows whose start_id AND '
      'end_id both resolve through the rocpd_timestamp spine (INNER JOIN: an unresolvable '
      'FK silently drops the row)',
      f'SELECT COUNT(*) FROM ({CPU})', 384)

check('H10b', 'v4_get_interval_track_cpu_thread_regions',
      'no rocpd_region row is lost to the spine join (spine-resolved count == raw count)',
      f'SELECT (SELECT COUNT(*) FROM ({CPU})) = (SELECT COUNT(*) FROM rocpd_region{S})', 1)

check('H11a', 'selects_v4_backend..., v4_get_interval_track_cpu_thread_regions, '
              'v4_get_event_info_region_header, v4_get_track_stats_matches_slices...',
      'MIN(start) on the cpu_thread spine == 516609802359041',
      f'SELECT MIN(s) FROM ({CPU})', 516609802359041)

check('H11b', 'v4_get_interval_track_cpu_thread_regions (front() determinism)',
      'the minimum start is UNIQUE — ORDER BY ts_s.value would otherwise leave '
      'intervals.front() nondeterministic',
      f'SELECT COUNT(*) FROM ({CPU}) WHERE s = (SELECT MIN(s) FROM ({CPU}))', 1)

check('H11c', 'v4_get_interval_track_cpu_thread_regions',
      'the earliest-start cpu_thread region ends at 516609802359341',
      f'SELECT e FROM ({CPU}) ORDER BY s LIMIT 1', 516609802359341)

check('H12', 'v4_get_interval_track_cpu_thread_carries_category',
      "every cpu_thread region resolves category 'hsa_api' through "
      'rocpd_event -> rocpd_info_category (both the interval arm and the detail arm)',
      f"SELECT COUNT(*) FROM ({CPU}) WHERE cat IS NOT 'hsa_api'", 0)

check('H13', 'v4_get_event_info_region_header',
      "the earliest cpu_thread region's name_id resolves in rocpd_string to "
      "'hsa_system_get_major_extension_table'",
      f'SELECT st.string FROM ({CPU}) x JOIN rocpd_string{S} st ON st.id = x.name_id '
      'ORDER BY x.s LIMIT 1', 'hsa_system_get_major_extension_table')

check('H14', 'v4_get_interval_track_cpu_thread_regions, v4_get_track_stats_matches_slices',
      'no cpu_thread region has end < start',
      f'SELECT COUNT(*) FROM ({CPU}) WHERE e < s', 0)

check('H15', 'v4_get_interval_track_cpu_thread_regions (row_id_of(first.id) > 0)',
      'every rocpd_region primary key is > 0 (opaque handles encode the row id)',
      f'SELECT COUNT(*) FROM rocpd_region{S} WHERE id <= 0', 0)

# --- G4: gpu_queue intervals ---------------------------------------------------
GPU = (f'SELECT k.id, ts_s.value AS s, ts_e.value AS e, k.kernel_id, IC.name AS cat, '
       f'k.dispatch_id, k.workgroup_size_x, k.grid_size_x '
       f'FROM rocpd_kernel_dispatch{S} k '
       f'JOIN rocpd_timestamp{S} ts_s ON ts_s.id = k.start_id '
       f'JOIN rocpd_timestamp{S} ts_e ON ts_e.id = k.end_id '
       f'LEFT JOIN rocpd_event{S} E ON E.id = k.event_id '
       f'LEFT JOIN rocpd_info_category{S} IC ON IC.id = E.category_id '
       f'WHERE k.track_id = (SELECT id FROM rocpd_track{S} WHERE queue_id IS NOT NULL)')

check('H16', 'v4_get_interval_track_gpu_queue_dispatches, _carries_category, '
             'v4_get_track_stats_matches_slices_for_interval_tracks',
      'the gpu_queue track carries exactly 20 spine-resolvable rocpd_kernel_dispatch rows',
      f'SELECT COUNT(*) FROM ({GPU})', 20)

check('H17a', 'v4_get_interval_track_gpu_queue_dispatches, v4_get_track_stats_matches_slices',
      'MIN(start) on the gpu_queue spine == 516609921772013',
      f'SELECT MIN(s) FROM ({GPU})', 516609921772013)

check('H17b', 'v4_get_interval_track_gpu_queue_dispatches (front() determinism)',
      'the gpu_queue minimum start is UNIQUE',
      f'SELECT COUNT(*) FROM ({GPU}) WHERE s = (SELECT MIN(s) FROM ({GPU}))', 1)

check('H17c', 'v4_get_interval_track_gpu_queue_dispatches',
      'the earliest gpu_queue dispatch ends at 516609921781427',
      f'SELECT e FROM ({GPU}) ORDER BY s LIMIT 1', 516609921781427)

check('H18', 'v4_get_interval_track_gpu_queue_carries_category',
      "every gpu_queue dispatch resolves category 'kernel_dispatch'",
      f"SELECT COUNT(*) FROM ({GPU}) WHERE cat IS NOT 'kernel_dispatch'", 0)

check('H19a', 'v4_get_event_info_kernel_dispatch_properties',
      'the earliest dispatch carries dispatch_id=1, workgroup_size_x=16, grid_size_x=1024',
      f'SELECT dispatch_id, workgroup_size_x, grid_size_x FROM ({GPU}) ORDER BY s LIMIT 1',
      (1, 16, 1024))

check('H19b', 'v4_get_event_info_kernel_dispatch_properties (kernel_symbol_id present)',
      "the earliest dispatch's kernel_id resolves in rocpd_info_kernel_symbol "
      '(an unresolved link omits the property under the omit-absent policy)',
      f'SELECT COUNT(*) FROM ({GPU}) x '
      f'JOIN rocpd_info_kernel_symbol{S} ks ON ks.id = x.kernel_id '
      f'WHERE x.s = (SELECT MIN(s) FROM ({GPU}))', 1)

check('H19c', 'v4_get_interval_track_gpu_queue_dispatches, v4_get_track_stats_matches_slices',
      'no gpu_queue dispatch has end < start',
      f'SELECT COUNT(*) FROM ({GPU}) WHERE e < s', 0)

# --- G5: dma intervals ---------------------------------------------------------
DMA = (f'SELECT mc.id, mc.track_id, ts_s.value AS s, ts_e.value AS e, mc.name_id, '
       f'IC.name AS cat, mc.size, mc.src_agent_id, mc.dst_agent_id '
       f'FROM rocpd_memory_copy{S} mc '
       f'JOIN rocpd_timestamp{S} ts_s ON ts_s.id = mc.start_id '
       f'JOIN rocpd_timestamp{S} ts_e ON ts_e.id = mc.end_id '
       f'LEFT JOIN rocpd_event{S} E ON E.id = mc.event_id '
       f'LEFT JOIN rocpd_info_category{S} IC ON IC.id = E.category_id '
       f'WHERE mc.track_id IN (SELECT id FROM rocpd_track{S} '
       f'                      WHERE queue_id IS NULL AND stream_id IS NOT NULL)')

check('H20a', 'v4_get_interval_track_dma_memory_copies, _carries_category, '
              'v4_get_event_info_memory_copy_properties, v4_get_track_stats_matches_slices',
      'EACH of the 2 dma tracks carries EXACTLY 1 spine-resolvable rocpd_memory_copy row',
      f'SELECT COUNT(*) FROM (SELECT track_id, COUNT(*) c FROM ({DMA}) '
      'GROUP BY track_id HAVING c = 1)', 2)

check('H20b', 'v4_get_interval_track_dma_memory_copies',
      'no dma memory_copy has end < start',
      f'SELECT COUNT(*) FROM ({DMA}) WHERE e < s', 0)

check('H20c', 'v4_get_interval_track_dma_carries_category',
      "both dma memory copies resolve category 'memory_copy'",
      f"SELECT COUNT(*) FROM ({DMA}) WHERE cat IS NOT 'memory_copy'", 0)

check('H21a', 'v4_get_event_info_memory_copy_properties',
      "at least one dma memory_copy's name_id resolves in rocpd_string to "
      "'MEMORY_COPY_HOST_TO_DEVICE' (the test selects the H2D copy by that literal name)",
      f'SELECT COUNT(*) FROM ({DMA}) x JOIN rocpd_string{S} st ON st.id = x.name_id '
      "WHERE st.string = 'MEMORY_COPY_HOST_TO_DEVICE'", 1)

check('H21b', 'v4_get_event_info_memory_copy_properties',
      'the H2D copy has size = 4194304',
      f'SELECT x.size FROM ({DMA}) x JOIN rocpd_string{S} st ON st.id = x.name_id '
      "WHERE st.string = 'MEMORY_COPY_HOST_TO_DEVICE'", 4194304)

check('H21c', 'v4_get_event_info_memory_copy_properties',
      'the H2D copy\'s src_agent_id AND dst_agent_id both resolve in rocpd_info_agent '
      '(unresolved links are omitted from the property bag)',
      f'SELECT COUNT(*) FROM ({DMA}) x JOIN rocpd_string{S} st ON st.id = x.name_id '
      f'JOIN rocpd_info_agent{S} sa ON sa.id = x.src_agent_id '
      f'JOIN rocpd_info_agent{S} da ON da.id = x.dst_agent_id '
      "WHERE st.string = 'MEMORY_COPY_HOST_TO_DEVICE'", 1)

# --- G6: stream track ----------------------------------------------------------
STREAM = (
    f'SELECT ts_s.value AS s, 1 AS op FROM rocpd_kernel_dispatch{S} k '
    f'JOIN rocpd_track{S} T ON T.id = k.track_id '
    f'JOIN rocpd_timestamp{S} ts_s ON ts_s.id = k.start_id '
    f'JOIN rocpd_timestamp{S} ts_e ON ts_e.id = k.end_id WHERE T.stream_id = 0 '
    'UNION ALL '
    f'SELECT ts_s.value, 2 FROM rocpd_memory_copy{S} mc '
    f'JOIN rocpd_track{S} T ON T.id = mc.track_id '
    f'JOIN rocpd_timestamp{S} ts_s ON ts_s.id = mc.start_id '
    f'JOIN rocpd_timestamp{S} ts_e ON ts_e.id = mc.end_id WHERE T.stream_id = 0 '
    'UNION ALL '
    f'SELECT ts_s.value, 3 FROM rocpd_memory_allocate{S} ma '
    f'JOIN rocpd_track{S} T ON T.id = ma.track_id '
    f'JOIN rocpd_timestamp{S} ts_s ON ts_s.id = ma.start_id '
    f'JOIN rocpd_timestamp{S} ts_e ON ts_e.id = ma.end_id WHERE T.stream_id = 0')

check('H22', 'v4_get_interval_track_stream_aggregates_ops_with_op_kind, '
             'v4_get_track_stats_stream_matches_interval_slice',
      'the sole stream (stream_id=0) unions exactly 20 kernel_dispatch + 2 memory_copy '
      '+ 0 memory_allocate = 22 spine-resolvable rows',
      f'SELECT COUNT(*), SUM(op=1), SUM(op=2), SUM(op=3) FROM ({STREAM})',
      (22, 20, 2, 0))

check('H23a', 'v4_get_interval_track_stream_aggregates_ops_with_op_kind',
      'the stream MIN(start) == 516609915990946',
      f'SELECT MIN(s) FROM ({STREAM})', 516609915990946)

check('H23b', 'v4_get_interval_track_stream_aggregates_ops_with_op_kind (front() '
              'determinism under ORDER BY 2)',
      'the stream minimum start is UNIQUE',
      f'SELECT COUNT(*) FROM ({STREAM}) WHERE s = (SELECT MIN(s) FROM ({STREAM}))', 1)

check('H23c', 'v4_get_interval_track_stream_aggregates_ops_with_op_kind',
      'the stream aggregates ACROSS ops: its earliest start is a memory_copy and strictly '
      'PRECEDES the gpu_queue first dispatch — proof the stream is not the queue relabeled',
      f'SELECT (SELECT op FROM ({STREAM}) ORDER BY s LIMIT 1) = 2 '
      f'AND (SELECT MIN(s) FROM ({STREAM})) < (SELECT MIN(s) FROM ({GPU}))', 1)

# --- G7: flows -----------------------------------------------------------------
def clique(st, sa, dt, da):
    return (f'SELECT {sa}.id AS sid, {da}.id AS did, '
            f'(SELECT value FROM rocpd_timestamp{S} WHERE id={sa}.start_id) AS s_start, '
            f'(SELECT value FROM rocpd_timestamp{S} WHERE id={da}.start_id) AS d_start, '
            f'E{sa}.stack_id AS stack, E{sa}.parent_stack_id AS s_par, '
            f'E{da}.parent_stack_id AS d_par '
            f'FROM {st}{S} {sa} '
            f'JOIN rocpd_event{S} E{sa} ON {sa}.event_id = E{sa}.id '
            f'JOIN rocpd_event{S} E{da} ON E{da}.stack_id = E{sa}.stack_id '
            f'  AND E{da}.id != E{sa}.id '
            f'JOIN {dt}{S} {da} ON {da}.event_id = E{da}.id '
            f'WHERE E{sa}.stack_id IS NOT NULL AND E{sa}.stack_id != 0')


R2K = clique('rocpd_region', 'R', 'rocpd_kernel_dispatch', 'K')
R2M = clique('rocpd_region', 'R', 'rocpd_memory_copy', 'MC')
FLOWSETS = {
    'region->kernel_dispatch': (R2K, 20),
    'region->memory_copy': (R2M, 2),
    'region->memory_allocate': (clique('rocpd_region', 'R', 'rocpd_memory_allocate', 'MA'), 0),
    'region->region': (clique('rocpd_region', 'R', 'rocpd_region', 'R2'), 0),
    'kd sibling': (clique('rocpd_kernel_dispatch', 'K', 'rocpd_kernel_dispatch', 'K2'), 0),
    'mc sibling': (clique('rocpd_memory_copy', 'MC', 'rocpd_memory_copy', 'MC2'), 0),
    'ma sibling': (clique('rocpd_memory_allocate', 'MA', 'rocpd_memory_allocate', 'MA2'), 0),
}
for label, (sql, exp) in FLOWSETS.items():
    check(f'H24[{label}]', 'v4_get_flows_links_regions_to_gpu_events',
          f'stack_id-clique flow set {label} yields exactly {exp} row(s) '
          '(20 + 2 + 0*5 = 22 edges after pair de-duplication)',
          f'SELECT COUNT(*) FROM ({sql})', exp)

# The reader's orientation rule (reader_impl.cpp:2981-3003), encoded verbatim.
ORIENT = ('CASE WHEN d_par IS NOT NULL AND d_par = stack THEN 1 '
          '     WHEN s_par IS NOT NULL AND s_par = stack THEN 0 '
          '     WHEN s_start != d_start THEN (s_start < d_start) '
          '     ELSE 1 END')
for label, sql in (('region->kernel_dispatch', R2K), ('region->memory_copy', R2M)):
    check(f'H25[{label}]', 'v4_get_flows_links_regions_to_gpu_events',
          f'for every {label} pair the reader\'s orientation rule (parent lineage, else '
          'earlier start) makes the REGION the source and the GPU-side event the dest',
          f'SELECT COUNT(*) FROM ({sql}) WHERE NOT ({ORIENT})', 0)

check('H26', 'v4_get_flows_links_regions_to_gpu_events (flow_id_value(f.flow_id) > 0)',
      'every flow-participating stack_id is STRICTLY POSITIVE (the SQL only excludes '
      '0/NULL; a negative stack_id would pass the query and fail the assertion)',
      f'SELECT COUNT(*) FROM ({R2K}) WHERE stack <= 0', 0)

check('H26b', 'v4_get_flows_links_regions_to_gpu_events',
      'same for the region->memory_copy leg',
      f'SELECT COUNT(*) FROM ({R2M}) WHERE stack <= 0', 0)

check('H27', 'v4_get_scalar_track_on_interval_track_returns_empty',
      'the gpu_queue track carries ZERO rocpd_sample rows, so get_scalar_track() on it '
      'returns an empty series',
      f'SELECT COUNT(*) FROM rocpd_sample{S} s '
      f'WHERE s.track_id = (SELECT id FROM rocpd_track{S} WHERE queue_id IS NOT NULL)', 0)

# --- G8: timeline / counts / window -------------------------------------------
EVENTS = (
    f'SELECT R.id AS id, ts_s.value AS s, ts_e.value AS e, 1 AS ty '
    f'FROM rocpd_region{S} R '
    f'JOIN rocpd_timestamp{S} ts_s ON ts_s.id=R.start_id '
    f'JOIN rocpd_timestamp{S} ts_e ON ts_e.id=R.end_id '
    f'JOIN rocpd_track{S} TR ON TR.id=R.track_id '
    'UNION ALL '
    f'SELECT K.id, ts_s.value, ts_e.value, 2 FROM rocpd_kernel_dispatch{S} K '
    f'JOIN rocpd_timestamp{S} ts_s ON ts_s.id=K.start_id '
    f'JOIN rocpd_timestamp{S} ts_e ON ts_e.id=K.end_id '
    f'JOIN rocpd_track{S} TR ON TR.id=K.track_id '
    'UNION ALL '
    f'SELECT MC.id, ts_s.value, ts_e.value, 3 FROM rocpd_memory_copy{S} MC '
    f'JOIN rocpd_timestamp{S} ts_s ON ts_s.id=MC.start_id '
    f'JOIN rocpd_timestamp{S} ts_e ON ts_e.id=MC.end_id '
    f'JOIN rocpd_track{S} TR ON TR.id=MC.track_id '
    'UNION ALL '
    f'SELECT MA.id, ts_s.value, ts_e.value, 4 FROM rocpd_memory_allocate{S} MA '
    f'JOIN rocpd_timestamp{S} ts_s ON ts_s.id=MA.start_id '
    f'JOIN rocpd_timestamp{S} ts_e ON ts_e.id=MA.end_id '
    f'JOIN rocpd_track{S} TR ON TR.id=MA.track_id')

check('H29a', 'v4_get_event_counts_time_window_filters',
      'get_events() returns at least 2 rows',
      f'SELECT COUNT(*) >= 2 FROM ({EVENTS})', 1)

check('H29b', 'v4_get_event_counts_time_window_filters',
      'MIN(start) < MAX(start) across all timeline events (the test builds its window '
      'from that span and asserts the strict inequality)',
      f'SELECT MIN(s) < MAX(s) FROM ({EVENTS})', 1)

check('H30', 'v4_get_event_counts_time_window_filters',
      'EVERY event row\'s track_id resolves to a rocpd_track row. get_event_counts() '
      'counts the raw table; get_events() INNER JOINs rocpd_track — a dangling track_id '
      'makes windowed.at(t) != per_type[t]',
      f'SELECT (SELECT COUNT(*) FROM ({EVENTS})) = '
      f'(SELECT (SELECT COUNT(*) FROM rocpd_region{S}) '
      f'      + (SELECT COUNT(*) FROM rocpd_kernel_dispatch{S}) '
      f'      + (SELECT COUNT(*) FROM rocpd_memory_copy{S}) '
      f'      + (SELECT COUNT(*) FROM rocpd_memory_allocate{S}))', 1)

check('H31', 'v4_get_event_counts_time_window_filters',
      'at least one event lies strictly OUTSIDE the half-window '
      '[min_start, min_start + (max_start-min_start)/2] under the reader\'s overlap '
      'filter (ts_s <= hi AND ts_e >= lo) — otherwise windowed_total == unwindowed_total '
      'and the final ASSERT_LT fails',
      f'SELECT COUNT(*) > 0 FROM ({EVENTS}) WHERE s > '
      f'(SELECT MIN(s) + (MAX(s)-MIN(s))/2 FROM ({EVENTS}))', 1)

# --- report --------------------------------------------------------------------
width = max(len(r[1]) for r in results)
failed = 0
for ok, cid, tests, desc, exp, got in results:
    tag = 'PASS' if ok else 'FAIL'
    if not ok:
        failed += 1
    print(f'[{tag}] {cid:<{width}}  expected={exp!r:<40} got={got!r}')
    if not ok:
        print(f'        {desc}')
        print(f'        tests: {tests}')
print(f'\n{len(results) - failed}/{len(results)} predicates passed; uuid={U}')
sys.exit(1 if failed else 0)
