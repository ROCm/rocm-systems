-- Copyright (c) Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: MIT
--
-- Synthetic multi-node fixture (task 063, coverage cell A1/A2 x multi-node).
--
-- discover_uuids() enumerates one UUID per DISTINCT rocpd_info_node.guid. Every
-- other committed fixture has exactly ONE node, so the create() factory always
-- takes its uuids.size()==1 branch and overwrites the caller-supplied UUID. This
-- fixture has TWO distinct hyphenated GUIDs in the unsuffixed rocpd_info_node
-- (the authoritative primary path added by task 061), so discover_uuids() returns
-- two UUIDs and the factory's uuids.size()==1 branch is NOT taken -- the >1 node
-- enumeration path that no other fixture reaches.
--
-- This is deliberately a minimal, self-contained schema (only the rocpd_info_node
-- object discover_uuids() reads) rather than the canonical rocpd_tables.sql: the
-- factory only runs discover_uuids(), which touches rocpd_info_node and
-- sqlite_master alone, so no other table is needed to exercise the path. The
-- unsuffixed table (not a suffixed rocpd_info_node_<uuid>) is what a pristine SDK
-- capture presents, so the primary replace(guid,'-','_') query runs, matching real
-- multi-node captures. The two GUIDs are hyphenated to also cover the 061 '-'->'_'
-- normalization on the multi-node path.
CREATE TABLE rocpd_info_node (
    id       INTEGER PRIMARY KEY,
    guid     TEXT NOT NULL,
    hostname TEXT
);

INSERT INTO rocpd_info_node (id, guid, hostname) VALUES
    (1, '00001eca-d4de-74de-b70e-c34ecf8c3a87', 'node-alpha'),
    (2, 'ffffdead-beef-4bad-9a0f-0123456789ab', 'node-beta');
