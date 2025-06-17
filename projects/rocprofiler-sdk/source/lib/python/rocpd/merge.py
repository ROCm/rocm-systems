import argparse
import sqlite3
import uuid
import os
from typing import Any

from .importer import RocpdImportData, execute_statement
from .schema import RocpdSchema
from .time_window import get_column_names


def create_empty_db(output_file, new_uuid, new_guid):
    """
    Create an empty database with the schema.

    Returns:
        Connection to the database
    """
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(output_file):
        os.remove(output_file)

    conn = sqlite3.connect(output_file)

    schema = RocpdSchema(uuid=new_uuid, guid=new_guid)
    schema.write_schema(conn)

    conn.commit()
    return RocpdImportData(output_file)


def get_all_db_uuids(import_data):
    result = execute_statement(import_data, "PRAGMA database_list").fetchall()
    all_db_uuids = []
    for db in result:
        if db[1] in ["main", "temp"]:
            continue
        for itr in execute_statement(
            import_data,
            f"SELECT value FROM {db[1]}.rocpd_metadata WHERE tag='uuid'",
        ).fetchall():
            all_db_uuids.append((db[1], itr[0]))
    return all_db_uuids


def update_table_ids(import_data, alias, table, uid, max_id):
    if max_id == 0:
        return

    stmt = f"SELECT id FROM {alias}.{table}{uid} ORDER BY id DESC"
    ids = execute_statement(import_data, stmt).fetchall()

    for (old_id,) in ids:
        update_stmt = f"""  
            UPDATE {alias}.{table}{uid}   
            SET id = {old_id + max_id}   
            WHERE id = {old_id}  
        """
        execute_statement(import_data, update_stmt)

    import_data.commit()


def undo_update_table_ids(db_conn, alias, table, uid):
    ids = execute_statement(
        db_conn, f"SELECT id FROM {alias}.{table}{uid} ORDER BY id ASC"
    ).fetchall()
    for idx, (old_id,) in enumerate(ids):
        update_stmt = f"""  
            UPDATE {alias}.{table}{uid}   
            SET id = {idx}
            WHERE id = {old_id}  
        """
        execute_statement(db_conn, update_stmt)
    db_conn.commit()


def insert_rocpd_info_node(connection, all_db_uuids, new_connection, new_uuid) -> None:
    updates_needed = {}
    unique_nodes = {}

    for alias, _uuid in all_db_uuids:
        updates_needed[alias] = []  # TODO alias -> alias+ _uuid ?
        rows = execute_statement(
            connection, f"SELECT * FROM {alias}.rocpd_info_node{_uuid}"
        ).fetchall()
        for row in rows:
            node_hash = row[2]  # Hash value
            node_id = row[0]
            if node_hash not in unique_nodes:
                unique_nodes[node_hash] = (node_id, row)  # TODO hash -> hash+machine_id ?

            elif node_id != unique_nodes[node_hash][0]:
                updates_needed[alias].append((node_id, unique_nodes[node_hash][0]))

    for alias, _uuid in all_db_uuids:
        if updates_needed[alias]:
            for old_id, new_id in updates_needed[alias]:
                execute_statement(
                    connection,
                    f"""  
                    UPDATE {alias}.rocpd_info_node{_uuid}   
                    SET id = ?  
                    WHERE id = ?  
                """,
                    (new_id, old_id),
                )
            connection.commit()

    cur = new_connection.cursor()
    for _, node in unique_nodes.values():
        # TODO
        cur.execute(
            f"INSERT OR IGNORE INTO rocpd_info_node{new_uuid}  (id, guid, hash, machine_id, system_name, hostname, release, version, hardware_name, domain_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            node,
        )
    new_connection.commit()


def insert_rocpd_string(connection, all_db_uuids, new_connection, new_uuid):
    strings_mapping = {}
    ids = []
    updates_needed = {}

    for alias, _uuid in all_db_uuids:
        updates_needed[alias] = []  # TODO alias -> alias + _uuid ?
        rows = connection.execute(
            f"SELECT id, string FROM {alias}.rocpd_string{_uuid}"
        ).fetchall()
        for r in rows:
            row_id = r[0]
            row_string = r[1]

            if row_string not in strings_mapping.keys():
                if row_id not in ids:
                    strings_mapping[row_string] = row_id
                    ids.append(row_id)
                else:
                    ids.sort()
                    new_id = ids[-1] + 1
                    strings_mapping[row_string] = new_id
                    ids.append(new_id)
                    updates_needed[alias].append((new_id, row_id, row_string))

            elif row_id != strings_mapping[row_string]:
                updates_needed[alias].append(
                    (row_id, strings_mapping[row_string], row_string)
                )

    for alias, _uuid in all_db_uuids:
        if updates_needed[alias]:
            for old_id, new_id, string in updates_needed[alias]:
                connection.execute(
                    f"""  
                    UPDATE {alias}.rocpd_string{_uuid}   
                    SET id = ?  
                    WHERE id = ?  
                """,
                    (new_id, old_id),
                )

            connection.commit()

    cur = new_connection.cursor()
    for string, id_value in strings_mapping.items():
        cur.execute(
            f"INSERT INTO rocpd_string{new_uuid} (id, string) VALUES (?, ?)",
            (id_value, string),
        )
    new_connection.commit()


def insert_table(table, alias, uid, new_uuid, import_data, import_data_merge):
    rows = execute_statement(
        import_data, f"SELECT * FROM {alias}.{table}{uid}"
    ).fetchall()

    column_names = get_column_names(import_data, table)
    placeholders = ",".join(["?"] * len(column_names))
    insert_sql = f"INSERT INTO {table}{new_uuid} ({','.join(column_names)}) VALUES ({placeholders})"

    dest_cur = import_data_merge.cursor()
    dest_cur.executemany(insert_sql, rows)
    import_data_merge.commit()


def update_tables_new_guid(new_db_conn, new_guid):
    cursor = new_db_conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = [t[0] for t in cursor.fetchall() if not t[0].startswith("sqlite_")]

    # Update all guids in all tables with the new GUID
    for t in table_names:
        cursor.execute(f"PRAGMA table_info({t})")
        cols = [col[1] for col in cursor.fetchall()]
        if "guid" in cols:
            new_db_conn.execute(f"UPDATE {t} SET guid = ?", (new_guid,))

    new_db_conn.commit()


def post_process(new_import_data, new_uuid):
    # Update agents absolute_index
    agent_types = [
        row[0]
        for row in execute_statement(
            new_import_data, f"SELECT DISTINCT type FROM rocpd_info_agent{new_uuid}"
        ).fetchall()
    ]
    ids = [
        row[0]
        for row in execute_statement(
            new_import_data, f"SELECT id FROM rocpd_info_agent{new_uuid} ORDER BY id"
        ).fetchall()
    ]
    for id in ids:
        execute_statement(
            new_import_data,
            f"UPDATE rocpd_info_agent{new_uuid} SET absolute_index = {id} WHERE id = {id}",
        )

    # Update agents Type index
    for agent_type in agent_types:
        ids = [
            row[0]
            for row in execute_statement(
                new_import_data,
                f"SELECT id FROM rocpd_info_agent{new_uuid} WHERE type ='{agent_type}' ORDER BY id",
            ).fetchall()
        ]

        for new_type_index, agent_id in enumerate(ids):
            execute_statement(
                new_import_data,
                f"UPDATE rocpd_info_agent{new_uuid} SET type_index = {new_type_index} WHERE id = {agent_id}",
            )
    new_import_data.commit()


def merge(import_data: RocpdImportData, **kwargs: Any) -> None:
    import time

    start_time = time.time()

    new_guid = str(uuid.uuid1())
    new_uuid = f"_{new_guid}".replace("-", "_")

    # Create an empty db in output_merge_path
    output = kwargs.get("output_merge_path")
    new_import_data = create_empty_db(output, new_uuid, new_guid)

    # List all dbs and their uuids
    all_db_uuids = get_all_db_uuids(import_data)

    special_table_cases = ["rocpd_metadata", "rocpd_string", "rocpd_info_node"]
    table_names = [
        t for t in import_data.table_info.keys() if t not in special_table_cases
    ]

    # Update ids in orig connection
    print("Updating ids in original database (this may take a while)...")
    for table in table_names:
        max_id = 0
        for alias, _uuid in all_db_uuids:
            update_table_ids(import_data, alias, table, _uuid, max_id)

            new_max = execute_statement(
                import_data, f"SELECT max(id) FROM {alias}.{table}{_uuid}"
            ).fetchall()[0][0]

            if new_max:
                max_id += new_max + 1

    # Insert special cases rocpd_info_node + rocpd_string
    print("Inserting special cases rocpd_info_node + rocpd_string...")
    insert_rocpd_info_node(import_data, all_db_uuids, new_import_data, new_uuid)
    insert_rocpd_string(import_data, all_db_uuids, new_import_data, new_uuid)

    # Insert rest of the data
    print("Inserting data from all tables...")
    for table in table_names:
        for alias, _uuid in all_db_uuids:
            insert_table(table, alias, _uuid, new_uuid, import_data, new_import_data)

    # Revert changes in original db
    print("Reverting changes in original database (this may take a while)...")
    for table in import_data.table_info.keys():
        for alias, _uuid in all_db_uuids:
            undo_update_table_ids(import_data, alias, table, _uuid)

    # Update new guid
    print("Updating GUID in new database...")
    update_tables_new_guid(new_import_data, new_guid)

    # Post-process agents
    post_process(new_import_data, new_uuid)

    elapsed_time = time.time() - start_time
    print(f"Merge completed successfully! Output saved to: {output}")
    print(f"Time: {elapsed_time:.2f} sec")


#
# Command-line interface functions
#
def add_args(parser):
    """Add arguments for merger."""
    merge_options = parser.add_argument_group("Merge options")
    merge_options.add_argument(
        "--output-merge-path",
        help="Sets the output path where the output merge files will be saved (default path: `./rocpd-output-data/db_merged.db`)",
        default=os.environ.get("ROCPD_OUTPUT_PATH", "./rocpd-output-data/db_merged.db"),
        type=str,
        required=False,
    )
    return ["output_merge_path"]


def process_args(args, valid_args):
    ret = {}
    for itr in valid_args:
        if hasattr(args, itr):
            val = getattr(args, itr)
            if val is not None:
                ret[itr] = val
    return ret


def execute(input_rpd: str, **kwargs: Any) -> RocpdImportData:

    importData = RocpdImportData(input_rpd)

    merge(importData, **kwargs)

    return importData


def main(argv=None) -> int:
    """Main entry point for command line execution."""

    parser = argparse.ArgumentParser(description="Merge ROCpd databases")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the input ROCpd database files",
    )

    valid_args = add_args(parser)

    args = parser.parse_args(argv)

    merged_args = process_args(args, valid_args)

    execute(args.input, **merged_args)


if __name__ == "__main__":
    main()
