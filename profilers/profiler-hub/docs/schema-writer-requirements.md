# Requirements: RocPD schema writers 3.0.0 through 3.0.5

## 1. Purpose

profiler-hub writes RocPD databases. RocPD changes between schema versions: columns are
added, a column moves to another table, a new table appears, and the allowed values of a
column widen. This document states what the writer must do for each kind of change, and
how a caller learns what the selected schema stores.

Requirements marked `ADR` (ADR2 to ADR11) are taken from the AMD Confluence page
[Requirements: library API versioning on SQL schema change](https://amd.atlassian.net/wiki/spaces/AGSRCIT/pages/2143793913/Requirements+library+API+versioning+on+SQL+schema+change).
Each keeps the number of its source requirement, with `ADR` in front: ADR5 is the source's
R5. Each is placed next to the requirement it extends. The source's R1 (one unversioned data
type and one writer) is already R5 here, so it is not repeated.

The writer must let one caller program write any schema from 3.0.0 through 3.0.5 with the
same input structs. The caller chooses the schema once. Every insert after that follows
that schema.

## 2. Terms

| Term | Meaning |
|---|---|
| Schema | A RocPD version such as 3.0.5, defined by `rocpd_tables.sql` for that version. |
| Selected schema | The version passed to `storage_t`. `version_t{}` means the newest writable schema. |
| Input struct | A public struct in `include/writer_types.hpp`, for example `pmc_info_t`. |
| Superset field | A field in an input struct that only some schemas store. |
| Owner schema | The first schema that stores a superset field, or the range of schemas that store it. |
| Unset | A `std::optional` field with no value. |
| Schema version | The layout of tables and columns in a database file. It belongs to the file. |
| API version | The write contract the library offers: which fields and tables a caller may set. In this library the selected schema is that contract. The two numbers are the same value today. A later design may let several API versions write one schema. |

## 3. Schema changes in scope

| Schema | Change | Kind of change |
|---|---|---|
| 3.0.1 | `rocpd_info_agent.type` and `rocpd_info_pmc.target_arch` accept `NIC`. | Allowed values widened |
| 3.0.2 | `graph_exec_id`, `graph_node_id` on `rocpd_kernel_dispatch` and `rocpd_memory_copy`, `NOT NULL DEFAULT 0`. | Column added |
| 3.0.2 | `rocpd_graph_launch` | Table added |
| 3.0.3 | `rocpd_info_pmc.spm_support` | Column added |
| 3.0.3 | `rocpd_pmc_event.sample_id`, `xcc`, `shader_engine`, `instance` | Column added |
| 3.0.4 | `rocpd_hip_event` | Table added |
| 3.0.5 | `xcc`, `shader_engine`, `instance` leave `rocpd_pmc_event`. `rocpd_info_pmc` gains `xcc`, `shader_engine`, `shader_array`, `wgp`, `instance`. `sample_id` stays on the event. | Column moved |

Tables with no change across these versions: node, process, thread, queue, stream, code
object, kernel symbol, track, string, event, arg, sample, region, memory allocate.

These schema changes are not in 3.0.0 through 3.0.5, and this document does not say what
the writer does for them:

- a required column added, with no default
- a column type change
- a meaning or unit change, where the column name stays and the stored number means
  something else
- a rename
- a constraint change other than the 3.0.1 `NIC` value (R19)
- a table split or a table merge

## 4. Functional requirements

### 4.1 Schema selection

- **R1.** The caller selects the schema on `storage_t`. `writer_t` reads that version once,
  at construction, and uses it for every insert.
- **R2.** `version_t{}` and the two-argument `storage_t` constructor select the newest
  writable schema. Today that is 3.0.5.

  `version_t` has three fields: `major`, `minor`, and `patch`. `version_t{}` leaves all
  three at `0`, so the value is `0.0.0`. The writer does not treat `0.0.0` as a schema.
  It reads it as "no version was chosen, so use the newest writable schema."

  ```cpp
  profiler_hub::version_t{}          // 0.0.0, means the newest schema: 3.0.5
  profiler_hub::version_t{ 3, 0, 1 } // an explicit choice: 3.0.1
  ```

  The two-argument `storage_t` constructor takes only the database path and the UUID. It
  calls the three-argument constructor with `version_t{}`. Leaving the version out and
  passing `version_t{}` select the same schema. Which schema that should be is open
  question 2.
- **ADR6.** A new file gets a defined default schema, and the caller can request any
  writable schema explicitly. R2 is the default. R1 is the explicit request.

  ```cpp
  profiler_hub::storage_t by_default{ path, uuid };                  // the default
  profiler_hub::storage_t explicit_one{ path, uuid, profiler_hub::version_t{ 3, 0, 0 } };
  ```

  Today the default is the newest writable schema, 3.0.5. Which value the default should
  be, including "stay on the previous schema until a major release" from ADR4, is open
  question 2.
- **R3.** A version outside the writable list throws `std::runtime_error` at `storage_t`
  construction. The message names the requested version and lists the writable ones.
- **ADR7.** A file the library cannot use is rejected with an error that says why, not with
  a raw SQL failure. The three cases are a foreign file, a schema older than the writable
  list, and a schema newer than the writable list.

  ```cpp
  profiler_hub::storage_t foreign{ "notes.db", uuid, profiler_hub::version_t{ 3, 0, 0 } };
  // throws: notes.db is not a RocPD database

  profiler_hub::storage_t too_old{ "trace-2.0.db", uuid, profiler_hub::version_t{ 3, 0, 0 } };
  // throws: file schema 2.0.0 is older than this library; writable schemas are 3.0.0 through 3.0.5

  profiler_hub::storage_t too_new{ "trace-3.0.6.db", uuid, profiler_hub::version_t{ 3, 0, 5 } };
  // throws: file schema 3.0.6 is newer than this library; writable schemas are 3.0.0 through 3.0.5
  ```

  R3 covers a requested version that is not writable, checked before a file is created.
  ADR7 covers an existing file. Today an existing file skips table creation and the
  mismatch can surface as a SQLite error on the first insert.
- **R4.** `storage_t::get_storage_version()` returns the resolved version, so a caller can
  log or branch on it.
- **ADR5 (not yet implemented).** An existing file keeps the schema it was created with.
  The library does not migrate it on open.

  A file created as 3.0.1, opened again:

  ```cpp
  profiler_hub::storage_t again{ path, uuid, profiler_hub::version_t{ 3, 0, 5 } };
  ```

  The 3.0.1 tables stay 3.0.1. The library does not add `graph_exec_id`, `rocpd_hip_event`,
  or the 3.0.5 PMC location. What the writer should do with the requested 3.0.5 is open
  question 3: follow the file, or reject the mismatch. Following the request and writing
  3.0.5 rows into 3.0.1 tables is the behavior this requirement rules out.
- **ADR9.** Removing a schema from the writable list happens only in a library major
  release, and the release documents how to move existing files.

  Dropping 3.0.0 is a major-release change. After it, opening a 3.0.0 file throws a
  migrate error that names the oldest schema this library still writes, for example
  `file schema 3.0.0 is no longer writable; migrate it to 3.0.1 or newer`. The message is
  not a SQLite error (ADR7).

### 4.2 One input type per table

- **R5.** Each table has one public input struct, shared by every schema. There is no
  `v3_0_0::pmc_info_t` or similar.
- **R6.** A field stored by every schema keeps its current type.

  A field that 3.0.0, 3.0.1, 3.0.2, 3.0.3, 3.0.4, and 3.0.5 all store stays the type it
  already has. Adding a later schema does not turn it into `std::optional` and does not
  rename it. The caller fills it the same way for every selected schema.

  ```cpp
  region.name            = "gemm";   // std::string_view on every schema
  region.start_timestamp = 1000;     // size_t on every schema
  kernel.dispatch_id     = 1;        // size_t on every schema
  ```

  A field that only some schemas store is covered by R7. That field is `std::optional`,
  so leaving it unset is the older-schema case. `kernel.graph_exec_id` is one of those:
  3.0.0 and 3.0.1 have no such column, and 3.0.2 and newer do.
- **R7.** A superset field is `std::optional`, so unset is the older-schema case.
- **R8.** Each superset field carries a comment naming the schemas that store it, for
  example `///< Stored by schema 3.0.5 and newer.` The caller reads the struct to learn
  what the selected schema stores.
- **ADR8.** The same client code runs against every writable schema when it sets only
  fields that every schema stores.

  ```cpp
  profiler_hub::writer_types::kernel_dispatch_data_t kernel;
  kernel.dispatch_id = 1;  // no superset field set

  writer_300.insert_kernel_dispatch_data(kernel, env);  // 3.0.0: written
  writer_305.insert_kernel_dispatch_data(kernel, env);  // 3.0.5: written, graph ids 0
  ```

  Setting a field that the selected schema does not store is R12, not this requirement.
  That call throws today (open question 1).

### 4.3 Behavior for each kind of change

**Column added (3.0.2 graph ids, 3.0.3 `spm_support` and `sample_id`).**

- **R9.** On a schema that stores the column, a set field is written.
- **R10.** On a schema that stores the column, an unset field is written as the column
  default: `0` for the graph ids, `NULL` for the others.
- **R11.** On a schema that has no such column, an unset field writes the row normally.
- **ADR2.** Adding an optional column or table does not require a client change. A caller
  that leaves the new field unset keeps writing, and the new column is stored as its
  default (R10, R11).

  A kernel written the 3.0.0 way, on a 3.0.2 writer. The caller never mentions
  `graph_exec_id`:

  ```cpp
  profiler_hub::storage_t storage{ path, uuid, profiler_hub::version_t{ 3, 0, 2 } };
  profiler_hub::writer_t  writer{ storage };

  profiler_hub::writer_types::kernel_dispatch_data_t kernel;
  kernel.dispatch_id = 1;  // graph_exec_id and graph_node_id left unset

  writer.insert_kernel_dispatch_data(kernel, env);
  // graph_exec_id = 0, graph_node_id = 0
  ```

  The API versioning note also keeps an older API version writing the same file beside a
  newer one. profiler-hub opens one schema per `storage_t`, so two selected schemas do not
  share one writer. Whether two writers may append to one file is open question 3.
- **R12.** On a schema that has no such column, a set field throws `std::runtime_error`
  before any row is written.

**Column moved (3.0.5 PMC location).**

- **R13.** The struct keeps a field at each place the column has lived:
  `pmc_event_data_t::xcc`, `shader_engine`, `instance` for 3.0.3 and 3.0.4, and
  `pmc_info_unique_id_t::xcc`, `shader_engine`, `shader_array`, `wgp`, `instance` for 3.0.5.

  When a column moves from one table to another, the public input keeps a field in both
  places. There is still one struct per table. The caller fills the place that matches the
  selected schema. R14 and R15 say which place that schema accepts.

  The PMC location is this case. In 3.0.3 and 3.0.4 the location sits on the event row. In
  3.0.5 it leaves that row and sits on the counter, `pmc_info`. The counter also gains
  `shader_array` and `wgp`, which the event never had.

  ```cpp
  // 3.0.3 and 3.0.4: fill the event.
  profiler_hub::writer_types::pmc_event_data_t sample;
  sample.value         = 9.5;
  sample.xcc           = 1;
  sample.shader_engine = 2;
  sample.instance      = 3;

  // 3.0.5: fill the counter. shader_array and wgp exist only here.
  profiler_hub::writer_types::pmc_info_t counter;
  counter.unique_id.name          = "SQ_WAVES";
  counter.unique_id.xcc           = 1;
  counter.unique_id.shader_engine = 2;
  counter.unique_id.shader_array  = 0;
  counter.unique_id.wgp           = 4;
  counter.unique_id.instance      = 3;
  ```

  Keeping both fields lets an older call site and a newer call site compile against the same
  headers. On 3.0.5 a location set on the event throws and the message points at `pmc_info`.
  On 3.0.0 through 3.0.4 a location set on the counter throws and the message points at 3.0.5.
- **R14.** On 3.0.5, a location set on `pmc_event_data_t` throws. The message names the new
  place: `pmc_event.xcc is not stored by schema 3.0.5. It is stored by pmc_info.`

  Schema 3.0.5 no longer stores `xcc`, `shader_engine`, or `instance` on the event. Setting
  any of them throws `std::runtime_error` before a row is written. `shader_engine` and
  `instance` use the same sentence with their own field name. Leaving all three unset writes
  the event. `sample_id` still belongs on the event, so setting it is valid.

  ```cpp
  profiler_hub::writer_types::pmc_event_data_t sample;
  sample.value = 9.5;
  sample.xcc   = 1;  // schema 3.0.5: throws, no row written
  ```
- **R15.** On 3.0.0 through 3.0.4, a location set on `pmc_info_unique_id_t` throws and names
  3.0.5 as the schema that stores it.

  Those schemas have no location columns on `rocpd_info_pmc`. Setting `xcc`,
  `shader_engine`, `shader_array`, `wgp`, or `instance` throws. The version in the message
  is the schema that was selected, for example
  `pmc_info.xcc is not stored by schema 3.0.0. It is stored by 3.0.5.` Leaving the five
  fields unset registers the counter normally.

  ```cpp
  profiler_hub::writer_types::pmc_info_t counter;
  counter.unique_id.name = "SQ_WAVES";
  counter.unique_id.xcc  = 1;  // schema 3.0.0 through 3.0.4: throws
  ```
- **R16.** On 3.0.5, the location is part of the PMC identity. Two `pmc_info_t` rows with the
  same name and different locations are two counters.

  The location is part of the counter's identity, together with the name. Each registration
  gets its own row and its own primary key. A counter with the location left unset is the
  base counter, separate from a located one.

  ```cpp
  profiler_hub::writer_types::pmc_info_t base;
  base.unique_id.name = "SQ_WAVES";  // one counter

  profiler_hub::writer_types::pmc_info_t located;
  located.unique_id.name = "SQ_WAVES";
  located.unique_id.xcc  = 1;  // a second counter
  ```

**Table added (3.0.2 `rocpd_graph_launch`, 3.0.4 `rocpd_hip_event`).**

- **R17.** The writer has one insert method per table, for example
  `insert_graph_launch_data` and `insert_hip_event_data`, available on every schema.
- **R18.** On a schema with no such table, the insert throws and names the schema that adds
  the table.

**Allowed values widened (3.0.1 `NIC`).**

- **R19.** On 3.0.0, `NIC` on `agent_info_t` or `pmc_info_t::target_arch` throws
  `std::runtime_error` and names 3.0.1.
- **R20.** A value that no schema allows, for example `TPU`, throws `std::invalid_argument`
  on every schema.

**Column removed with no new place.** This has not happened in 3.0.0 through 3.0.5.

- **R21.** When it happens, the field stays in the struct and is marked
  `[[deprecated("<schema> removed <column>; <what to use instead>")]]`. Schemas that still
  store it write it. Schemas without it reject it as in R12. The field is deleted from the
  struct only in a library major release.
- **ADR3.** A field or table removed from the file is never ignored. Writing it throws,
  with the R22 text. R12 and R21 say this for a column. This adds a removed table.

  No table is removed between 3.0.0 and 3.0.5. If a later schema dropped
  `rocpd_hip_event`, the call below would throw and write nothing:

  ```cpp
  profiler_hub::storage_t storage{ path, uuid, profiler_hub::version_t{ 3, 0, 6 } };
  profiler_hub::writer_t  writer{ storage };

  writer.insert_hip_event_data(hip_event, env);
  // throws: hip_event is not stored by schema 3.0.6. It is stored by 3.0.4.
  ```
- **ADR4.** A removed field stays in the public struct, marked deprecated, for at least one
  release before it is deleted, and it is deleted only in a library major release. R21
  states the attribute and the major release. This adds the one-release wait and the
  proposed transition. The transition is not agreed.

  ```cpp
  // One release after the column leaves the file. The warning names the replacement.
  [[deprecated("3.0.6 removed kernel_dispatch.name; use the kernel symbol name")]]
  std::string_view name;
  ```

  Proposed steps, for a column removed in schema 3.0.6:

  1. Minor library release: the field is deprecated. The default schema for a new file
     stays 3.0.5. A caller opts in to 3.0.6 by passing `version_t{ 3, 0, 6 }`.
  2. Major library release: the field is removed from the struct. The default for a new
     file becomes 3.0.6.
  3. A later major release: 3.0.5 drops out of the writable list. Opening a 3.0.5 file
     throws a migrate error (ADR9), not a raw SQL error.

  There is no automatic fallback. A caller who wants the new columns opens a file with the
  new schema. An old file is migrated by a documented step, not rewritten on open (ADR5).
- **ADR11.** The library version and the schema version are different numbers, and the
  release notes map each kind of schema change to a library release.

  | Schema change | Library release |
  |---|---|
  | Optional column or table added (3.0.2 graph ids, 3.0.2 `rocpd_graph_launch`) | Minor. Old callers keep compiling and leave the new field unset (ADR2). |
  | Allowed value added (`NIC` in 3.0.1) | Minor. |
  | Column moved (3.0.5 PMC location) | Minor. Both fields stay in the struct (R13). |
  | Column or table removed | Major, after the deprecation in ADR4. |
  | Schema dropped from the writable list | Major, with the migrate note in ADR9. |

  `version_t{ 3, 0, 5 }` is the RocPD schema. It is not the profiler-hub package version.

### 4.4 Error message contract

- **R22.** Every rejection for an unsupported field or table uses this text:

  ```text
  <object>.<field> is not stored by schema <selected>. It is stored by <owner>.
  ```

  For a table: `<table> is not stored by schema <selected>. It is stored by <owner>.`
- **R23.** The writer rejects before it writes. A rejected call leaves no partial row, and
  the next valid call succeeds.
- **R24.** A missing required value throws. Examples: unregistered node or process, empty
  region name, kernel dispatch without agent, queue, stream, or a registered kernel symbol.

## 5. Caller scenarios

These show what happens when a caller fills structs for one schema and selects another.

| Selected | Input | Result | Why |
|---|---|---|---|
| 3.0.5 | Filled the 3.0.0 way: no superset field set. | Written. New columns get their defaults: graph ids `0`, `spm_support` `NULL`, location `NULL`, `sample_id` `NULL`. | No error. An unset field means the caller did not ask for a column this schema added. 3.0.5 stores those columns, so R10 writes the column default. |
| 3.0.5 | PMC event with `xcc` set, the 3.0.3 or 3.0.4 way. | Throws. The message says the location is stored by `pmc_info`. | In 3.0.5 `xcc`, `shader_engine`, and `instance` left `rocpd_pmc_event`. Setting them on the event asks for a column that table no longer has (R14). Put the location on `pmc_info` instead. |
| 3.0.5 | `pmc_info_t` with `unique_id.xcc` set. | Written on `rocpd_info_pmc`. | No error. 3.0.5 is the schema that stores the location on the counter (R13, R16). |
| 3.0.0 | `kernel_dispatch_data_t` with `graph_exec_id` set. | Throws. The message names 3.0.2. | `graph_exec_id` was added in 3.0.2. Schema 3.0.0 has no such column. A set field is rejected (R12) so the value is not dropped. |
| 3.0.0 | `pmc_info_t` with `unique_id.xcc` set. | Throws. The message names 3.0.5. | Location columns on `rocpd_info_pmc` exist only from 3.0.5. Schema 3.0.0 cannot store them (R15). |
| 3.0.1 | `insert_hip_event_data(...)` | Throws. The message names 3.0.4. | `rocpd_hip_event` was added in 3.0.4. Schema 3.0.1 has no such table (R18). |
| 3.0.0 | `agent_info_t` with type `NIC`. | Throws. The message names 3.0.1. | `NIC` became an allowed agent type in 3.0.1. Schema 3.0.0 does not allow it (R19). |
| any | `agent_info_t` with type `TPU`. | Throws `std::invalid_argument`. | `TPU` is not a value any schema from 3.0.0 through 3.0.5 allows. This is an unknown value, not a version mismatch (R20). |

Example. A caller that used to write 3.0.4 PMC events and now writes 3.0.5:

```cpp
// 3.0.4: location on the event.
profiler_hub::writer_types::pmc_event_data_t sample;
sample.value = 9.5;
sample.xcc   = 1;
writer.insert_pmc_event_data(sample, counter.unique_id);

// 3.0.5: location on the counter, one counter per location.
counter.unique_id.xcc = 1;
writer.register_pmc_info(counter);

profiler_hub::writer_types::pmc_event_data_t sample_305;
sample_305.value = 9.5;
writer.insert_pmc_event_data(sample_305, counter.unique_id);
```

## 6. Design requirements

- **D1.** The 3.0.0 writer is the common base. Shared code does validation, key lookup,
  event and sample inserts.
- **D2.** Each later schema has a folder `source/writers/schema_v3/schema_3_0_X/` that holds
  only the writers whose table changed in that schema, plus `writer_policy.hpp`.
- **D3.** Each policy inherits the previous policy and replaces only the writer types that
  changed.
- **D4.** `writer_t` holds a `std::variant` with one core per schema and picks one in its
  constructor from the storage version. Later calls visit that one core.

  ```cpp
  using writer_variant_t = std::variant<writer_impl_core<writer_policy_3_0_0>,
                                        writer_impl_core<writer_policy_3_0_1>,
                                        writer_impl_core<writer_policy_3_0_2>,
                                        writer_impl_core<writer_policy_3_0_3>,
                                        writer_impl_core<writer_policy_3_0_4>,
                                        writer_impl_core<writer_policy_3_0_5>>;

  // storage version 3.0.5 picks this alternative once.
  if(version == version_t{ 3, 0, 5 })
  {
      return writer_impl_core<writer_policy_3_0_5>(std::move(ctx));
  }

  // a later insert reaches only the core that was picked.
  std::visit([&](auto& core) { core.insert_kernel_dispatch_data(data); }, m_core);
  ```
- **D5.** SQL for a column or table that only a later schema has is prepared inside that
  schema's writer. Opening a 3.0.0 database never prepares a 3.0.2 statement.

  Preparing means SQLite compiles the `INSERT` when the writer is constructed, before any
  row is written. The statement names its columns. A 3.0.0 database has no `graph_exec_id`
  column and no `rocpd_graph_launch` table. Preparing an `INSERT` that names either of them
  would fail while the database is opening, even if the caller never uses that field.

  The statement that names a new column lives in the writer for the schema that added it.
  D4 already constructs only the core for the selected version, so a 3.0.0 storage never
  runs the 3.0.2 constructor.

  ```cpp
  // 3.0.0 kernel writer: INSERT names the columns that exist in 3.0.0.
  // graph_exec_id is not in this statement.

  // 3.0.2 kernel writer, also used by 3.0.3, 3.0.4, and 3.0.5:
  query.set_table_name("rocpd_kernel_dispatch_<uuid>")
      .set_columns("id", /* ...columns from 3.0.0... */
                   "graph_exec_id",
                   "graph_node_id");

  // 3.0.0 graph_launch writer prepares no INSERT. The call throws (R18).
  // 3.0.2 graph_launch writer prepares INSERT INTO rocpd_graph_launch_<uuid>.
  ```
- **D6.** Adding a schema means: one tag, one folder with the changed writers and a policy,
  one variant entry, one branch in `create_core`, and one entry in the writable list.
  Schema 3.0.5 is the example of all five.

  **One tag.** An empty type, `schema_3_0_5_tag`, in `source/data_storage/schema_version.hpp`.
  The writers are templates on that tag, so the 3.0.5 info writer is a different type from
  the 3.0.4 one.

  **One folder.** `source/writers/schema_v3/schema_3_0_5/` holds only what changed in that
  schema, plus `writer_policy.hpp`. For 3.0.5 that is the info writer and the PMC event
  writer, because the location moved. Region, kernel dispatch, and the other unchanged
  tables stay in the older folders. The policy inherits the previous one and replaces only
  those two:

  ```cpp
  struct writer_policy_3_0_5 : writer_policy_3_0_4
  {
      using info_writer_t      = info_registration_writer<data_storage::schema_3_0_5_tag>;
      using pmc_event_writer_t = pmc_event_writer<data_storage::schema_3_0_5_tag>;
  };
  ```

  **One variant entry.** `writer_variant_t` in `source/writer_impl.hpp` gains
  `writer_impl_core<writer_policy_3_0_5>`. That is the slot D4 picks.

  **One branch in `create_core`.** `source/writer_impl.cpp` compares the storage version and
  returns that core. A version with no branch throws `No writer for schema ...` before a
  database is created.

  ```cpp
  if(version == version_t{ 3, 0, 5 })
  {
      return writer_impl_core<writer_policy_3_0_5>(std::move(ctx));
  }
  ```

  **One entry in the writable list.** `k_writer_supported_versions` in
  `sqlite_backend_impl.hpp` gains `{ 3, 0, 5 }`. `storage_t` accepts a version only when it
  is in this list. `version_t{}` selects the last entry, so adding 3.0.5 at the end makes an
  omitted version resolve to 3.0.5.
- **ADR10.** The public headers do not expose SQLite. `include/writer.hpp`,
  `include/writer_types.hpp`, and `include/storage.hpp` do not mention `sqlite3` or include
  a SQLite header. Callers pass structs and a `storage_t`. The prepared statements stay in
  `source/`.

  ```cpp
  profiler_hub::storage_t storage{ path, uuid, profiler_hub::version_t{ 3, 0, 5 } };
  profiler_hub::writer_t  writer{ storage };
  writer.insert_kernel_dispatch_data(kernel, env);  // no SQL in the caller
  ```

  The public headers already follow this.

## 7. Test requirements

- **T1.** For every schema 3.0.0 through 3.0.5, a database holds a minimum row of every
  table that schema stores, and the reader returns it after flush.
- **T2.** For every schema, a full row sets every field that schema stores.
- **T3.** For every superset field and every schema, the test checks R9 through R19: written
  on its owner schemas, rejected elsewhere with the R22 text.
- **T4.** For every required value, leaving it out throws, and a later valid row still
  writes.
- **T5.** For a moved column, a test reads the value back from the new table. A test does
  the same for an added column on its owner schema.
- **T6.** Tests for one schema live in `tests/unit/schema_3_0_X/`. Cross-schema tests live
  in `tests/unit/schema_writers/`.

## 8. Current status

| Requirement | Status |
|---|---|
| R1 to R20, R22 to R24, D1 to D6 | Implemented. |
| ADR2 to ADR11 | Placed next to the requirement each one extends. Rule only, except ADR10, which the public headers already follow. |
| R21 | No removed column exists yet. Rule only. |
| T1, T3, T6 | Implemented in `schema_writers/version_matrix_test.cpp` and `schema_3_0_5/`. |
| T2 | Full row written for 3.0.5 only. |
| T4 | Most required values covered. Region start/end and memory allocation size not yet. |
| T5 | Not yet. The public reader has no location, graph id, or `spm_support` fields. |

## 9. Open questions

1. **Reject or drop.** A set field the selected schema cannot store throws today. The
   [kdlib experiment](https://github.com/adjordje-amd/libray-tests) drops it instead.
   Should profiler-hub keep throwing, drop it with a warning, or return a list of dropped
   fields?

   The same call, a 3.0.0 writer and a kernel that sets `graph_exec_id` (a 3.0.2 column):

   ```cpp
   profiler_hub::storage_t storage{ path, uuid, profiler_hub::version_t{ 3, 0, 0 } };
   profiler_hub::writer_t  writer{ storage };

   profiler_hub::writer_types::kernel_dispatch_data_t kernel;
   kernel.dispatch_id   = 1;
   kernel.graph_exec_id = 7;

   writer.insert_kernel_dispatch_data(kernel, env);
   ```

   Today this throws before any row is written:

   ```text
   kernel_dispatch.graph_exec_id is not stored by schema 3.0.0. It is stored by 3.0.2.
   ```

   Dropping would write the kernel row and ignore `graph_exec_id`. The three choices are:
   keep this throw, write the row and warn that `graph_exec_id` was dropped, or write the
   row and return the dropped field names to the caller.
2. **Default schema when none is passed.** Both of these select a schema without naming one:

   ```cpp
   profiler_hub::version_t{}                 // 0.0.0, "no version chosen"
   storage_t{ path, uuid }                   // calls the constructor above with version_t{}
   ```

   Today both select the newest writable schema, 3.0.5. Before this work the two-argument
   constructor wrote 3.0.1, so existing callers now get a newer database without changing
   code. Which writer should be the default?

   - the newest writable schema, which moves forward each time a schema is added
   - a fixed schema, such as 3.0.0 or 3.0.1, until the caller passes a version
   - the version already stored in the file, when the file exists
3. **Existing database.** When a caller opens an existing file, should the writer follow the
   version stored in that file, as kdlib does, instead of the requested version?

   Example: a file created with 3.0.1. Later the caller opens it and requests 3.0.5:

   ```cpp
   profiler_hub::storage_t storage{ path, uuid, profiler_hub::version_t{ 3, 0, 5 } };
   ```

   - **Follow the file (kdlib).** The writer reads the version stored in the file, 3.0.1,
     and writes 3.0.1 rows. The request for 3.0.5 is ignored, or the call fails. The file
     keeps its shape.
   - **Follow the request (profiler-hub today).** The writer uses 3.0.5 and writes 3.0.5
     rows into tables created for 3.0.1. Columns such as `rocpd_pmc_event.xcc` or the 3.0.5
     `rocpd_info_pmc` location may be missing, so an insert can fail or land in a table with
     a different shape.

   Today profiler-hub follows the request. Opening an existing file skips table creation,
   and `storage_t` stores the requested version without reading the file. The source does
   not read a `user_version` value. kdlib uses SQLite's `PRAGMA user_version` for this, so
   that is the mechanism the first option would need.
4. **Capability query.** Should the writer offer something like
   `writer.stores("pmc_info.xcc")` or `writer.supported_fields()`, so a caller can check
   before filling a field instead of catching an exception?
5. **Moved column.** On 3.0.5, should the writer translate an event-level location into the
   matching `pmc_info` location counter automatically, or keep rejecting it?

   A caller that still fills the location the 3.0.3 and 3.0.4 way, while the selected
   schema is 3.0.5:

   ```cpp
   profiler_hub::storage_t storage{ path, uuid, profiler_hub::version_t{ 3, 0, 5 } };
   profiler_hub::writer_t  writer{ storage };

   profiler_hub::writer_types::pmc_event_data_t sample;
   sample.value         = 9.5;
   sample.xcc           = 1;
   sample.shader_engine = 2;
   sample.instance      = 3;

   writer.insert_pmc_event_data(sample, counter.unique_id);
   ```

   Today this throws before any row is written (R14):

   ```text
   pmc_event.xcc is not stored by schema 3.0.5. It is stored by pmc_info.
   ```

   Translating would take `xcc`, `shader_engine`, and `instance` off the event, find or
   register the `pmc_info` counter with that location, and write the event against that
   counter. The event row itself would have no location columns. The question is whether
   the writer should do that move, or keep the throw and leave the caller to set
   `pmc_info_unique_id_t` as in R13.
6. **Location identity.** On 3.0.5 a location is a separate counter. Is that the intended
   meaning, and how should a caller pick the counter for a given event?

   Example: one counter name, `SQ_WAVES`, registered twice, once with no location and once
   at `xcc = 1`:

   ```cpp
   profiler_hub::storage_t storage{ path, uuid, profiler_hub::version_t{ 3, 0, 5 } };
   profiler_hub::writer_t  writer{ storage };

   profiler_hub::writer_types::pmc_info_t base;
   base.unique_id.name = "SQ_WAVES";

   profiler_hub::writer_types::pmc_info_t located;
   located.unique_id.name = "SQ_WAVES";
   located.unique_id.xcc  = 1;

   writer.register_pmc_info(base);     // counter A
   writer.register_pmc_info(located);  // counter B, a different row

   profiler_hub::writer_types::pmc_event_data_t sample;
   sample.value = 9.5;

   writer.insert_pmc_event_data(sample, located.unique_id);  // stored against counter B
   writer.insert_pmc_event_data(sample, base.unique_id);     // stored against counter A
   ```

   The caller picks the counter by passing its `pmc_info_unique_id_t` to
   `insert_pmc_event_data`. If the caller passes `base.unique_id` for a sample that was
   measured at `xcc = 1`, the row is attached to the wrong counter and no error is raised.
   The question is whether the writer should check that the event's location matches the
   counter it is attached to, or whether the caller owns that match.
7. **`sample_id`.** Setting it means "insert `sample` and store that row's id". The number
   the caller assigns is not written. Should the field be a `bool` or a separate method
   instead, to avoid confusion?

   Example: two events, one with `sample_id` set and one without. The value `42` is the
   caller's number, not the stored one:

   ```cpp
   profiler_hub::writer_types::pmc_event_data_t with_sample;
   with_sample.value     = 9.5;
   with_sample.sample_id = 42;   // the writer inserts `sample` and stores that row's id

   profiler_hub::writer_types::pmc_event_data_t without_sample;
   without_sample.value = 9.5;   // sample_id unset: stored as NULL

   writer.insert_pmc_event_data(with_sample, counter.unique_id);     // sample_id = new sample row id
   writer.insert_pmc_event_data(without_sample, counter.unique_id);  // sample_id = NULL
   ```

   The `42` is not in `rocpd_pmc_event` after the insert. The stored id is the one SQLite
   gives the new `rocpd_sample` row. The caller cannot read that id back, because the public
   reader has no `sample_id` field (open question 12).

   A `bool` would make the meaning clear: `with_sample.store_sample = true`. A separate method
   would make it clear too: `writer.insert_pmc_event_with_sample(...)`. The question is
   which one profiler-hub should use.
8. **Graph ids.** Unset graph ids are written as `0` on 3.0.2 and newer. Is `0` an
   acceptable "no graph" value, or should the field be required on those schemas?

   The same kernel on a 3.0.2 writer, once with the graph ids left unset and once with them
   set. `graph_exec_id` and `graph_node_id` were added in 3.0.2 on kernel dispatch and
   memory copy. The writer stores an unset field with `value_or(0)`.

   ```cpp
   profiler_hub::storage_t storage{ path, uuid, profiler_hub::version_t{ 3, 0, 2 } };
   profiler_hub::writer_t  writer{ storage };

   profiler_hub::writer_types::kernel_dispatch_data_t plain;
   plain.dispatch_id = 1;  // graph ids unset

   profiler_hub::writer_types::kernel_dispatch_data_t graphed;
   graphed.dispatch_id   = 2;
   graphed.graph_exec_id = 7;
   graphed.graph_node_id = 3;

   writer.insert_kernel_dispatch_data(plain, env);    // graph_exec_id = 0, graph_node_id = 0
   writer.insert_kernel_dispatch_data(graphed, env);  // graph_exec_id = 7, graph_node_id = 3
   ```

   A reader cannot tell a kernel that had no graph from a kernel whose graph id really is
   `0`. Requiring the fields on 3.0.2 and newer would throw when they are unset, the same
   way a missing agent throws today. The question is which of those two `0` should mean.
9. **Exception types.** Unsupported fields throw `std::runtime_error`. Unknown values throw
   `std::invalid_argument`. Should profiler-hub define its own exception type with the
   field, the selected schema, and the owner schema as members?
10. **Deprecation.** When a column is removed with no new place, which release adds
    `[[deprecated]]` and which release deletes the field?
11. **ABI.** Adding a field to a public struct changes its layout. Is rebuilding callers on
    every minor release acceptable, or does the struct need reserved space?
12. **Reader parity.** Should the public reader expose the new columns (location, graph ids,
    `spm_support`, graph launch, HIP event) so T5 can read values back?
13. **A 3.0.5 struct passed to a 3.0.0 writer.** The public struct is the 3.0.0 fields plus
    the fields later schemas added. A caller can fill every field and still select the
    3.0.0 writer. Should the writer warn that the extra fields will be dropped and insert
    the row, or error out with a message?

    The same kernel, filled with a 3.0.2 field, on a 3.0.0 writer:

    ```cpp
    profiler_hub::storage_t storage{ path, uuid, profiler_hub::version_t{ 3, 0, 0 } };
    profiler_hub::writer_t  writer{ storage };

    profiler_hub::writer_types::kernel_dispatch_data_t kernel;
    kernel.dispatch_id   = 1;  // stored by 3.0.0
    kernel.graph_exec_id = 7;  // added in 3.0.2; 3.0.0 has no such column

    writer.insert_kernel_dispatch_data(kernel, env);
    ```

    Today this errors out before any row is written (R12):

    ```text
    kernel_dispatch.graph_exec_id is not stored by schema 3.0.0. It is stored by 3.0.2.
    ```

    The other choice is to warn and continue. The warning would name each extra field, for
    example `kernel_dispatch.graph_exec_id is not stored by schema 3.0.0 and will be
    dropped`, and the insert would write `dispatch_id` and the other 3.0.0 columns. The
    same choice applies to every extra field: `graph_node_id`, `spm_support`, `sample_id`,
    the PMC location, `NIC`, `graph_launch`, and `hip_event`. This is the concrete case of
    question 1.

