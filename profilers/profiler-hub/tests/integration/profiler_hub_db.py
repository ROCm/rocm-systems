# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SQLite helpers for validating profiler-hub writer integration fixtures."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class ProfilerHubDb:
    """Small wrapper around the schema_v3 SQLite DB produced by examples."""

    DEFAULT_NODE_ID = 1
    DEFAULT_PID = 1000
    DEFAULT_TID = 100

    def __init__(
        self,
        db_path: Path,
        uuid: str = "integration",
        node_id: int = DEFAULT_NODE_ID,
        pid: int = DEFAULT_PID,
        tid: int = DEFAULT_TID,
    ) -> None:
        self.db_path = db_path
        self.uuid = uuid
        self.node_id = node_id
        self.pid = pid
        self.tid = tid
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row

    @classmethod
    def open(
        cls,
        db_path: Path,
        uuid: str = "integration",
        node_id: int = DEFAULT_NODE_ID,
        pid: int = DEFAULT_PID,
        tid: int = DEFAULT_TID,
    ) -> "ProfilerHubDb":
        return cls(db_path, uuid, node_id=node_id, pid=pid, tid=tid)

    def close(self) -> None:
        self._conn.close()

    def delete(self) -> None:
        self.db_path.unlink(missing_ok=True)

    def table(self, base_name: str) -> str:
        return f"{base_name}_{self.uuid}"

    def row(self, query: str, params: tuple[Any, ...] = ()) -> sqlite3.Row:
        rows = self.rows(query, params)
        assert len(rows) == 1, f"expected one row, got {len(rows)}: {query}"
        return rows[0]

    def rows(self, query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        return list(self._conn.execute(query, params))

    def scalar(self, query: str, params: tuple[Any, ...] = ()) -> Any:
        row = self.row(query, params)
        return row[0]

    def read_memory_alloc_info(self) -> dict[str, Any]:
        row = self.row(f"""
            select
                MA.*,
                P.pid as process_id,
                T.tid as thread_id,
                T.name as thread_name,
                A.type as agent_type,
                A.absolute_index as agent_absolute_index,
                A.logical_index as agent_logical_index,
                A.type_index as agent_type_index,
                A.uuid as agent_uuid,
                A.name as agent_name,
                A.model_name as agent_model_name,
                A.vendor_name as agent_vendor_name,
                A.product_name as agent_product_name,
                A.user_name as agent_user_name,
                AP.pid as agent_process_id,
                Q.id as queue_id,
                Q.name as queue_name,
                QP.pid as queue_process_id,
                S.id as stream_id,
                S.name as stream_name,
                SP.pid as stream_process_id
            from {self.table('rocpd_memory_allocate')} MA
            join {self.table('rocpd_info_process')} P on MA.pid = P.id
            join {self.table('rocpd_info_thread')} T on MA.tid = T.id
            join {self.table('rocpd_info_agent')} A on MA.agent_id = A.id
            join {self.table('rocpd_info_process')} AP on A.pid = AP.id
            join {self.table('rocpd_info_queue')} Q on MA.queue_id = Q.id
            join {self.table('rocpd_info_process')} QP on Q.pid = QP.id
            join {self.table('rocpd_info_stream')} S on MA.stream_id = S.id
            join {self.table('rocpd_info_process')} SP on S.pid = SP.id
            """)
        read_info = {
            "type": row["type"],
            "level": row["level"],
            "start_timestamp": row["start"],
            "end_timestamp": row["end"],
            "address": row["address"],
            "size": row["size"],
            "extdata": row["extdata"],
            "thread_info.thread_id": row["thread_id"],
            "thread_info.name": row["thread_name"],
            "agent_info.agent_type": row["agent_type"],
            "agent_info.absolute_index": row["agent_absolute_index"],
            "agent_info.logical_index": row["agent_logical_index"],
            "agent_info.type_index": row["agent_type_index"],
            "agent_info.uuid": row["agent_uuid"],
            "agent_info.name": row["agent_name"],
            "agent_info.model_name": row["agent_model_name"],
            "agent_info.vendor_name": row["agent_vendor_name"],
            "agent_info.product_name": row["agent_product_name"],
            "agent_info.user_name": row["agent_user_name"],
            "agent_info.node_info.node_id": row["nid"],
            "agent_info.process_info.pid": row["agent_process_id"],
            "queue_info.db_id": row["queue_id"],
            "queue_info.name": row["queue_name"],
            "queue_info.node_info.node_id": row["nid"],
            "queue_info.process_info.pid": row["queue_process_id"],
            "stream_info.db_id": row["stream_id"],
            "stream_info.name": row["stream_name"],
            "stream_info.node_info.node_id": row["nid"],
            "stream_info.process_info.pid": row["stream_process_id"],
        }
        read_info.update(self._read_event_info(row["event_id"]))
        read_info.update(self._read_common_metadata_info())
        return read_info

    def read_memory_copy_info(self) -> dict[str, Any]:
        row = self.row(f"""
            select
                MC.*,
                P.pid as process_id,
                T.tid as thread_id,
                T.name as thread_name,
                N.string as copy_name,
                RN.string as region_name,
                SRC.type as src_agent_type,
                SRC.absolute_index as src_agent_absolute_index,
                SRC.logical_index as src_agent_logical_index,
                SRC.type_index as src_agent_type_index,
                SRC.uuid as src_agent_uuid,
                SRC.name as src_agent_name,
                SRC.model_name as src_agent_model_name,
                SRC.vendor_name as src_agent_vendor_name,
                SRC.product_name as src_agent_product_name,
                SRC.user_name as src_agent_user_name,
                DST.type as dst_agent_type,
                DST.absolute_index as dst_agent_absolute_index,
                DST.logical_index as dst_agent_logical_index,
                DST.type_index as dst_agent_type_index,
                DST.uuid as dst_agent_uuid,
                DST.name as dst_agent_name,
                DST.model_name as dst_agent_model_name,
                DST.vendor_name as dst_agent_vendor_name,
                DST.product_name as dst_agent_product_name,
                DST.user_name as dst_agent_user_name,
                Q.id as queue_id,
                Q.name as queue_name,
                QP.pid as queue_process_id,
                S.id as stream_id,
                S.name as stream_name,
                SP.pid as stream_process_id
            from {self.table('rocpd_memory_copy')} MC
            join {self.table('rocpd_info_process')} P on MC.pid = P.id
            join {self.table('rocpd_info_thread')} T on MC.tid = T.id
            join {self.table('rocpd_string')} N on MC.name_id = N.id
            join {self.table('rocpd_string')} RN on MC.region_name_id = RN.id
            join {self.table('rocpd_info_agent')} SRC on MC.src_agent_id = SRC.id
            join {self.table('rocpd_info_agent')} DST on MC.dst_agent_id = DST.id
            join {self.table('rocpd_info_queue')} Q on MC.queue_id = Q.id
            join {self.table('rocpd_info_process')} QP on Q.pid = QP.id
            join {self.table('rocpd_info_stream')} S on MC.stream_id = S.id
            join {self.table('rocpd_info_process')} SP on S.pid = SP.id
            """)
        read_info = {
            "start_timestamp": row["start"],
            "end_timestamp": row["end"],
            "dst_address": row["dst_address"],
            "src_address": row["src_address"],
            "size": row["size"],
            "name": row["copy_name"],
            "region_name": row["region_name"],
            "extdata": row["extdata"],
            "thread_info.thread_id": row["thread_id"],
            "thread_info.name": row["thread_name"],
            "src_agent_info.agent_type": row["src_agent_type"],
            "src_agent_info.absolute_index": row["src_agent_absolute_index"],
            "src_agent_info.logical_index": row["src_agent_logical_index"],
            "src_agent_info.type_index": row["src_agent_type_index"],
            "src_agent_info.uuid": row["src_agent_uuid"],
            "src_agent_info.name": row["src_agent_name"],
            "src_agent_info.model_name": row["src_agent_model_name"],
            "src_agent_info.vendor_name": row["src_agent_vendor_name"],
            "src_agent_info.product_name": row["src_agent_product_name"],
            "src_agent_info.user_name": row["src_agent_user_name"],
            "src_agent_info.node_info.node_id": row["nid"],
            "src_agent_info.process_info.pid": row["process_id"],
            "dst_agent_info.agent_type": row["dst_agent_type"],
            "dst_agent_info.absolute_index": row["dst_agent_absolute_index"],
            "dst_agent_info.logical_index": row["dst_agent_logical_index"],
            "dst_agent_info.type_index": row["dst_agent_type_index"],
            "dst_agent_info.uuid": row["dst_agent_uuid"],
            "dst_agent_info.name": row["dst_agent_name"],
            "dst_agent_info.model_name": row["dst_agent_model_name"],
            "dst_agent_info.vendor_name": row["dst_agent_vendor_name"],
            "dst_agent_info.product_name": row["dst_agent_product_name"],
            "dst_agent_info.user_name": row["dst_agent_user_name"],
            "dst_agent_info.node_info.node_id": row["nid"],
            "dst_agent_info.process_info.pid": row["process_id"],
            "queue_info.db_id": row["queue_id"],
            "queue_info.name": row["queue_name"],
            "queue_info.node_info.node_id": row["nid"],
            "queue_info.process_info.pid": row["queue_process_id"],
            "stream_info.db_id": row["stream_id"],
            "stream_info.name": row["stream_name"],
            "stream_info.node_info.node_id": row["nid"],
            "stream_info.process_info.pid": row["stream_process_id"],
        }
        read_info.update(self._read_event_info(row["event_id"]))
        read_info.update(self._read_common_metadata_info())
        return read_info

    def read_region_info(self) -> dict[str, Any]:
        row = self.row(f"""
            select
                R.*,
                P.pid as process_id,
                T.tid as thread_id,
                T.name as thread_name,
                N.string as region_name
            from {self.table('rocpd_region')} R
            join {self.table('rocpd_info_process')} P on R.pid = P.id
            join {self.table('rocpd_info_thread')} T on R.tid = T.id
            join {self.table('rocpd_string')} N on R.name_id = N.id
            """)
        read_info = {
            "start_timestamp": row["start"],
            "end_timestamp": row["end"],
            "name": row["region_name"],
            "extdata": row["extdata"],
            "thread_info.thread_id": row["thread_id"],
            "thread_info.name": row["thread_name"],
        }
        read_info.update(self._read_event_info(row["event_id"]))
        read_info.update(self._read_region_arg_info(row["event_id"]))
        read_info.update(self._read_region_sample_info(row["event_id"]))
        read_info.update(self._read_common_metadata_info())
        return read_info

    def read_kernel_dispatch_info(self) -> dict[str, Any]:
        row = self.row(f"""
            select
                KD.*,
                P.pid as process_id,
                T.tid as thread_id,
                T.name as thread_name,
                RN.string as region_name,
                A.type as agent_type,
                A.absolute_index as agent_absolute_index,
                A.logical_index as agent_logical_index,
                A.type_index as agent_type_index,
                A.uuid as agent_uuid,
                A.name as agent_name,
                A.model_name as agent_model_name,
                A.vendor_name as agent_vendor_name,
                A.product_name as agent_product_name,
                A.user_name as agent_user_name,
                AP.pid as agent_process_id,
                Q.id as queue_id,
                Q.name as queue_name,
                QP.pid as queue_process_id,
                S.id as stream_id,
                S.name as stream_name,
                SP.pid as stream_process_id,
                CO.id as code_object_original_id,
                CO.uri as code_object_uri,
                CO.load_base as code_object_load_base,
                CO.load_size as code_object_load_size,
                CO.load_delta as code_object_load_delta,
                CO.storage_type as code_object_storage_type,
                KS.id as kernel_symbol_original_id,
                KS.kernel_name as kernel_symbol_name,
                KS.display_name as kernel_symbol_display_name,
                KS.kernel_object as kernel_symbol_kernel_object,
                KS.kernarg_segment_size as kernel_symbol_kernarg_segment_size,
                KS.kernarg_segment_alignment as kernel_symbol_kernarg_segment_alignment,
                KS.group_segment_size as kernel_symbol_group_segment_size,
                KS.private_segment_size as kernel_symbol_private_segment_size,
                KS.sgpr_count as kernel_symbol_sgpr_count,
                KS.arch_vgpr_count as kernel_symbol_arch_vgpr_count,
                KS.accum_vgpr_count as kernel_symbol_accum_vgpr_count
            from {self.table('rocpd_kernel_dispatch')} KD
            join {self.table('rocpd_info_process')} P on KD.pid = P.id
            join {self.table('rocpd_info_thread')} T on KD.tid = T.id
            join {self.table('rocpd_string')} RN on KD.region_name_id = RN.id
            join {self.table('rocpd_info_agent')} A on KD.agent_id = A.id
            join {self.table('rocpd_info_process')} AP on A.pid = AP.id
            join {self.table('rocpd_info_queue')} Q on KD.queue_id = Q.id
            join {self.table('rocpd_info_process')} QP on Q.pid = QP.id
            join {self.table('rocpd_info_stream')} S on KD.stream_id = S.id
            join {self.table('rocpd_info_process')} SP on S.pid = SP.id
            join {self.table('rocpd_info_kernel_symbol')} KS on KD.kernel_id = KS.id
            join {self.table('rocpd_info_code_object')} CO on KS.code_object_id = CO.id
            """)
        read_info = {
            "dispatch_id": row["dispatch_id"],
            "start_timestamp": row["start"],
            "end_timestamp": row["end"],
            "private_segment_size": row["private_segment_size"],
            "group_segment_size": row["group_segment_size"],
            "workgroup_size_x": row["workgroup_size_x"],
            "workgroup_size_y": row["workgroup_size_y"],
            "workgroup_size_z": row["workgroup_size_z"],
            "grid_size_x": row["grid_size_x"],
            "grid_size_y": row["grid_size_y"],
            "grid_size_z": row["grid_size_z"],
            "name": row["region_name"],
            "extdata": row["extdata"],
            "thread_info.thread_id": row["thread_id"],
            "thread_info.name": row["thread_name"],
            "agent_info.agent_type": row["agent_type"],
            "agent_info.absolute_index": row["agent_absolute_index"],
            "agent_info.logical_index": row["agent_logical_index"],
            "agent_info.type_index": row["agent_type_index"],
            "agent_info.uuid": row["agent_uuid"],
            "agent_info.name": row["agent_name"],
            "agent_info.model_name": row["agent_model_name"],
            "agent_info.vendor_name": row["agent_vendor_name"],
            "agent_info.product_name": row["agent_product_name"],
            "agent_info.user_name": row["agent_user_name"],
            "agent_info.node_info.node_id": row["nid"],
            "agent_info.process_info.pid": row["agent_process_id"],
            "queue_info.db_id": row["queue_id"],
            "queue_info.name": row["queue_name"],
            "queue_info.node_info.node_id": row["nid"],
            "queue_info.process_info.pid": row["queue_process_id"],
            "stream_info.db_id": row["stream_id"],
            "stream_info.name": row["stream_name"],
            "stream_info.node_info.node_id": row["nid"],
            "stream_info.process_info.pid": row["stream_process_id"],
            "code_object_info.id": row["code_object_original_id"],
            "code_object_info.uri": row["code_object_uri"],
            "code_object_info.load_base": row["code_object_load_base"],
            "code_object_info.load_size": row["code_object_load_size"],
            "code_object_info.load_delta": row["code_object_load_delta"],
            "code_object_info.storage_type": row["code_object_storage_type"],
            "code_object_info.node_info.node_id": row["nid"],
            "code_object_info.process_info.pid": row["process_id"],
            "code_object_info.agent_info.agent_type": row["agent_type"],
            "code_object_info.agent_info.type_index": row["agent_type_index"],
            "kernel_symbol_info.id": row["kernel_symbol_original_id"],
            "kernel_symbol_info.name": row["kernel_symbol_name"],
            "kernel_symbol_info.display_name": row["kernel_symbol_display_name"],
            "kernel_symbol_info.kernel_object": row["kernel_symbol_kernel_object"],
            "kernel_symbol_info.kernarg_segment_size": row[
                "kernel_symbol_kernarg_segment_size"
            ],
            "kernel_symbol_info.kernarg_segment_alignment": row[
                "kernel_symbol_kernarg_segment_alignment"
            ],
            "kernel_symbol_info.group_segment_size": row[
                "kernel_symbol_group_segment_size"
            ],
            "kernel_symbol_info.private_segment_size": row[
                "kernel_symbol_private_segment_size"
            ],
            "kernel_symbol_info.sgpr_count": row["kernel_symbol_sgpr_count"],
            "kernel_symbol_info.arch_vgpr_count": row["kernel_symbol_arch_vgpr_count"],
            "kernel_symbol_info.accum_vgpr_count": row[
                "kernel_symbol_accum_vgpr_count"
            ],
            "kernel_symbol_info.node_info.node_id": row["nid"],
            "kernel_symbol_info.process_info.pid": row["process_id"],
            "kernel_symbol_info.code_object_info.id": row["code_object_original_id"],
        }
        read_info.update(self._read_event_info(row["event_id"]))
        read_info.update(self._read_common_metadata_info())
        return read_info

    def read_pmc_event_info(self) -> dict[str, Any]:
        row = self.row(f"""
            select
                PE.*,
                PMC.target_arch as pmc_target_arch,
                PMC.event_code as pmc_event_code,
                PMC.instance_id as pmc_instance_id,
                PMC.name as pmc_name,
                PMC.symbol as pmc_symbol,
                PMC.description as pmc_description,
                PMC.long_description as pmc_long_description,
                PMC.component as pmc_component,
                PMC.units as pmc_units,
                PMC.value_type as pmc_value_type,
                PMC.block as pmc_block,
                PMC.expression as pmc_expression,
                PMC.is_constant as pmc_is_constant,
                PMC.is_derived as pmc_is_derived,
                PMC.extdata as pmc_extdata,
                PMC.nid as pmc_node_id,
                PP.pid as pmc_process_id,
                A.type as agent_type,
                A.absolute_index as agent_absolute_index,
                A.logical_index as agent_logical_index,
                A.type_index as agent_type_index,
                A.uuid as agent_uuid,
                A.name as agent_name,
                A.model_name as agent_model_name,
                A.vendor_name as agent_vendor_name,
                A.product_name as agent_product_name,
                A.user_name as agent_user_name,
                AP.pid as agent_process_id
            from {self.table('rocpd_pmc_event')} PE
            join {self.table('rocpd_info_pmc')} PMC on PE.pmc_id = PMC.id
            join {self.table('rocpd_info_process')} PP on PMC.pid = PP.id
            join {self.table('rocpd_info_agent')} A on PMC.agent_id = A.id
            join {self.table('rocpd_info_process')} AP on A.pid = AP.id
            """)
        read_info = {
            "value": row["value"],
            "extdata": row["extdata"],
            "pmc_info.target_arch": row["pmc_target_arch"],
            "pmc_info.event_code": row["pmc_event_code"],
            "pmc_info.instance_id": row["pmc_instance_id"],
            "pmc_info.name": row["pmc_name"],
            "pmc_info.symbol": row["pmc_symbol"],
            "pmc_info.description": row["pmc_description"],
            "pmc_info.long_description": row["pmc_long_description"],
            "pmc_info.component": row["pmc_component"],
            "pmc_info.units": row["pmc_units"],
            "pmc_info.value_type": row["pmc_value_type"],
            "pmc_info.block": row["pmc_block"],
            "pmc_info.expression": row["pmc_expression"],
            "pmc_info.is_constant": row["pmc_is_constant"],
            "pmc_info.is_derived": row["pmc_is_derived"],
            "pmc_info.extdata": row["pmc_extdata"],
            "pmc_info.node_info.node_id": row["pmc_node_id"],
            "pmc_info.process_info.pid": row["pmc_process_id"],
            "pmc_info.agent_info.agent_type": row["agent_type"],
            "pmc_info.agent_info.type_index": row["agent_type_index"],
            "agent_info.agent_type": row["agent_type"],
            "agent_info.absolute_index": row["agent_absolute_index"],
            "agent_info.logical_index": row["agent_logical_index"],
            "agent_info.type_index": row["agent_type_index"],
            "agent_info.uuid": row["agent_uuid"],
            "agent_info.name": row["agent_name"],
            "agent_info.model_name": row["agent_model_name"],
            "agent_info.vendor_name": row["agent_vendor_name"],
            "agent_info.product_name": row["agent_product_name"],
            "agent_info.user_name": row["agent_user_name"],
            "agent_info.node_info.node_id": row["pmc_node_id"],
            "agent_info.process_info.pid": row["agent_process_id"],
        }
        read_info.update(self._read_event_info(row["event_id"]))
        read_info.update(self._read_region_sample_info(row["event_id"]))
        read_info.update(self._read_common_metadata_info())
        return read_info

    def _read_event_info(self, event_id: int) -> dict[str, Any]:
        row = self.row(
            f"""
            select E.*, C.string as category
            from {self.table('rocpd_event')} E
            left join {self.table('rocpd_string')} C on E.category_id = C.id
            where E.id = ?
            """,
            (event_id,),
        )
        return {
            "event.stack_id": row["stack_id"],
            "event.parent_stack_id": row["parent_stack_id"],
            "event.correlation_id": row["correlation_id"],
            "event.event_category": row["category"],
            "event.extdata": row["extdata"],
        }

    def _read_common_metadata_info(self) -> dict[str, Any]:
        node = self.row(
            f"select * from {self.table('rocpd_info_node')} where id = ?",
            (self.node_id,),
        )
        process = self.row(
            f"""
            select P.*, N.id as node_id
            from {self.table('rocpd_info_process')} P
            join {self.table('rocpd_info_node')} N on P.nid = N.id
            where P.pid = ?
            """,
            (self.pid,),
        )
        thread = self.row(
            f"""
            select T.*, P.pid as process_id
            from {self.table('rocpd_info_thread')} T
            join {self.table('rocpd_info_process')} P on T.pid = P.id
            where T.tid = ?
            """,
            (self.tid,),
        )
        return {
            "node_info.node_id": node["id"],
            "node_info.hash": node["hash"],
            "node_info.machine_id": node["machine_id"],
            "node_info.system_name": node["system_name"],
            "node_info.hostname": node["hostname"],
            "node_info.release": node["release"],
            "node_info.version": node["version"],
            "node_info.hardware_name": node["hardware_name"],
            "node_info.domain_name": node["domain_name"],
            "process_info.pid": process["pid"],
            "process_info.ppid": process["ppid"],
            "process_info.node_info.node_id": process["node_id"],
            "thread_info.parent_process_id": thread["ppid"],
            "thread_info.start": thread["start"],
            "thread_info.end": thread["end"],
            "thread_info.node_info.node_id": thread["nid"],
            "thread_info.process_info.pid": thread["process_id"],
        }

    def _read_region_arg_info(self, event_id: int) -> dict[str, Any]:
        args = self.rows(
            f"""
            select position, type, name, value, extdata
            from {self.table('rocpd_arg')}
            where event_id = ?
            order by position
            """,
            (event_id,),
        )
        read_info: dict[str, Any] = {"arg_count": len(args)}
        for index, arg in enumerate(args):
            read_info[f"arg.{index}.position"] = arg["position"]
            read_info[f"arg.{index}.type"] = arg["type"]
            read_info[f"arg.{index}.name"] = arg["name"]
            read_info[f"arg.{index}.value"] = arg["value"]
            read_info[f"arg.{index}.extdata"] = arg["extdata"]
        return read_info

    def _read_region_sample_info(self, event_id: int) -> dict[str, Any]:
        row = self.row(
            f"""
            select
                S.timestamp,
                S.extdata as sample_extdata,
                TR.nid as track_node_id,
                TR.extdata as track_extdata,
                T.tid as track_thread_id,
                P.pid as track_process_id,
                N.string as track_name
            from {self.table('rocpd_sample')} S
            join {self.table('rocpd_track')} TR on S.track_id = TR.id
            join {self.table('rocpd_info_thread')} T on TR.tid = T.id
            join {self.table('rocpd_info_process')} P on TR.pid = P.id
            join {self.table('rocpd_string')} N on TR.name_id = N.id
            where S.event_id = ?
            """,
            (event_id,),
        )
        return {
            "sample.timestamp": row["timestamp"],
            "sample.extdata": row["sample_extdata"],
            "sample.track.name": row["track_name"],
            "sample.track.extdata": row["track_extdata"],
            "sample.track.node_info.node_id": row["track_node_id"],
            "sample.track.process_info.pid": row["track_process_id"],
            "sample.track.thread_id": row["track_thread_id"],
        }
